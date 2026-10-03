//! Chat inside the TUI. The model lives on its own thread (Metal objects stay
//! where they were created); the UI talks to it over channels and streams
//! tokens as they arrive.
use crate::agent::{self, Sandbox, ToolMode};
use crate::attach::{self, Attachment, Clip};
use crate::brain::{self, Ponytail};
use crate::coding::{self, SharedRef};
use crate::*;
use ferrum::hybrid::{
    plan::PlanOptions,
    runtime::{ChatHooks, ChatRequest, Delta, Runtime, SpecOptions},
};
use std::cell::Cell;
use std::collections::VecDeque;
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
    mpsc::{Receiver, Sender, channel},
};
use std::thread::JoinHandle;

pub struct ChatOpts {
    pub model: PathBuf,
    pub context: Option<usize>,
    pub system: Option<String>,
    pub max_tokens: usize,
    pub think: bool,
    pub effort: Option<String>,
    pub show_thinking: bool,
    pub budget: Option<usize>,
    pub overrides: Vec<(&'static str, f32)>,
    pub spec: Option<SpecOptions>,
    pub tools: ToolMode,
    pub workspace: String,
    pub network: bool,
    pub search: crate::search::Config,
    pub agent: Option<AgentOpts>,
}

pub struct AgentOpts {
    pub project: PathBuf,
    pub ponytail: Ponytail,
    pub compact: bool,
}

/// A fresh folder for one chat session: `<root>/<date-time>-<model>`, never one that exists.
pub fn session_dir(root: &str, model: &str) -> PathBuf {
    let root = expand(if root.trim().is_empty() {
        "~/ferrum-workspace"
    } else {
        root.trim()
    });
    let stamp = Command::new("/bin/date")
        .arg("+%Y%m%d-%H%M%S")
        .output()
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .unwrap_or_default();
    let slug: String = model
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() {
                c.to_ascii_lowercase()
            } else {
                '-'
            }
        })
        .collect::<String>()
        .split('-')
        .filter(|p| !p.is_empty())
        .collect::<Vec<_>>()
        .join("-")
        .chars()
        .take(28)
        .collect();
    let base = format!("{stamp}-{slug}");
    let mut dir = root.join(&base);
    let mut n = 2;
    while dir.exists() {
        dir = root.join(format!("{base}-{n}"));
        n += 1;
    }
    dir
}

/// Where an agent keeps its scratch home, temp files and attachments: outside the project.
fn agent_state_dir(project: &Path) -> PathBuf {
    let name = project
        .file_name()
        .map(|n| n.to_string_lossy().to_string())
        .unwrap_or_else(|| "project".into());
    let mut h: u64 = 0xcbf29ce484222325;
    for b in project.display().to_string().bytes() {
        h = (h ^ b as u64).wrapping_mul(0x100000001b3);
    }
    home().join(format!(".ferrum/agents/{name}-{:08x}", h as u32))
}

impl ChatOpts {
    fn from_app(app: &App) -> Result<Self, String> {
        let model = app.selected().ok_or("no model selected")?.path.clone();
        let count = |k: &str| app.value(k).parse::<usize>().ok();
        let mut overrides = Vec::new();
        for (key, flag) in [
            ("temperature", "--temperature"),
            ("top_p", "--top-p"),
            ("top_k", "--top-k"),
            ("min_p", "--min-p"),
            ("presence", "--presence-penalty"),
            ("frequency", "--frequency-penalty"),
            ("repeat", "--repetition-penalty"),
        ] {
            if let Ok(v) = app.value(key).parse::<f32>() {
                overrides.push((flag, v));
            }
        }
        let draft = app.value("draft");
        let spec = if draft.is_empty() || draft == "off" {
            None
        } else {
            let mut spec = None;
            let mut apply = |k: &str, v: &str| {
                SpecOptions::apply_flag(&mut spec, k, v).map_err(|e| e.to_string())
            };
            apply("--draft", draft)?;
            if let Some(n) = count("draft_max") {
                apply("--draft-max", &n.to_string())?;
            }
            let quant = app.value("draft_quant");
            if !quant.is_empty() {
                apply("--draft-quant", quant)?;
            }
            spec
        };
        Ok(Self {
            model,
            context: count("context"),
            system: Some(app.value("system").to_string()).filter(|s| !s.is_empty()),
            max_tokens: count("max_tokens").unwrap_or(32768),
            think: app.value("think") != "off",
            effort: Some(app.value("effort").to_string())
                .filter(|s| !s.is_empty() && s != "default"),
            show_thinking: app.value("show") != "hide",
            budget: count("budget"),
            overrides,
            spec,
            tools: if app.mode == Mode::Agent {
                match app.value("agent_tools") {
                    "" => ToolMode::Edits,
                    v => ToolMode::parse(v),
                }
            } else {
                ToolMode::parse(app.value("tools"))
            },
            workspace: app.value("workspace").to_string(),
            network: app.value("network") != "off",
            search: app.search.clone(),
            agent: if app.mode == Mode::Agent {
                let project = match app.value("project") {
                    "" => std::env::current_dir().map_err(|e| e.to_string())?,
                    p => expand(p),
                };
                if !project.is_dir() {
                    return Err(format!("{} is not a folder", project.display()));
                }
                let project = fs::canonicalize(&project).map_err(|e| e.to_string())?;
                Some(AgentOpts {
                    project,
                    ponytail: Ponytail::parse(app.value("ponytail")),
                    compact: app.value("compact") != "off",
                })
            } else {
                None
            },
        })
    }
}

enum ToWorker {
    Turn {
        messages: Vec<Json>,
        think: bool,
        tools: Vec<Json>,
    },
    Reset,
}

#[derive(Clone)]
pub struct LoadInfo {
    pub name: String,
    pub context: usize,
    pub max_context: usize,
    pub weights_gib: f64,
    pub load_secs: f64,
    pub drafter: Option<String>,
    /// File name of the vision projector, when images can be shown to the model.
    pub vision: Option<String>,
    pub sampling: String,
}

pub struct TurnDone {
    content: String,
    reasoning: Option<String>,
    prompt_tokens: usize,
    reused: usize,
    prefill_tps: f64,
    generated: usize,
    decode_tps: f64,
    stop: String,
    acceptance: Option<f64>,
    ctx_used: usize,
    ctx_cap: usize,
    tool_calls: Vec<(String, Map<String, Json>)>,
}

enum FromWorker {
    Stage(String),
    Ready(Box<LoadInfo>),
    Failed(String),
    Begin { prompt: usize, reused: usize },
    Prefill(usize, usize),
    Delta { reasoning: bool, text: String },
    Tokens(usize),
    Done(Box<TurnDone>),
    TurnFailed(String),
}

struct Hooks<'a> {
    tx: &'a Sender<FromWorker>,
    cancel: &'a AtomicBool,
}

impl ChatHooks for Hooks<'_> {
    fn delta(&mut self, delta: Delta) -> ferrum::Result<()> {
        let (reasoning, text) = match delta {
            Delta::Reasoning(t) => (true, t),
            Delta::Content(t) => (false, t),
        };
        let _ = self.tx.send(FromWorker::Delta { reasoning, text });
        Ok(())
    }
    fn begin(&mut self, prompt: usize, reused: usize) {
        let _ = self.tx.send(FromWorker::Begin { prompt, reused });
    }
    fn prefill(&mut self, processed: usize, total: usize) -> bool {
        let _ = self.tx.send(FromWorker::Prefill(processed, total));
        !self.cancel.load(Ordering::SeqCst)
    }
    fn cancelled(&self) -> bool {
        self.cancel.load(Ordering::SeqCst)
    }
    fn generated(&mut self, tokens: usize) {
        let _ = self.tx.send(FromWorker::Tokens(tokens));
    }
}

fn worker(opts: ChatOpts, rx: Receiver<ToWorker>, tx: Sender<FromWorker>, cancel: Arc<AtomicBool>) {
    let send = |m: FromWorker| {
        let _ = tx.send(m);
    };
    send(FromWorker::Stage("loading weights".into()));
    // A projector next to the model lets attached images reach it as pixels.
    let mmproj = ferrum::vision::resolve_projector(&opts.model, None)
        .ok()
        .flatten();
    let mut rt = match Runtime::load_multimodal(
        &opts.model,
        PlanOptions {
            context: opts.context,
            ..Default::default()
        },
        opts.spec.as_ref(),
        mmproj.as_deref(),
        ferrum::vision::DEFAULT_MAX_TOKENS,
    ) {
        Ok(rt) => rt,
        Err(e) => return send(FromWorker::Failed(e.to_string())),
    };
    send(FromWorker::Stage("warming up kernels".into()));
    if let Err(e) = rt.warmup() {
        return send(FromWorker::Failed(e.to_string()));
    }
    let mut sampling = rt.default_sampling.clone();
    for (flag, v) in &opts.overrides {
        match *flag {
            "--temperature" => sampling.temperature = *v,
            "--top-p" => sampling.top_p = *v,
            "--top-k" => sampling.top_k = *v as usize,
            "--min-p" => sampling.min_p = *v,
            "--presence-penalty" => sampling.presence_penalty = *v,
            "--frequency-penalty" => sampling.frequency_penalty = *v,
            "--repetition-penalty" => sampling.repetition_penalty = *v,
            _ => {}
        }
    }
    if let Err(e) = sampling.validate() {
        return send(FromWorker::Failed(e.to_string()));
    }
    let plan = &rt.loaded.plan;
    send(FromWorker::Ready(Box::new(LoadInfo {
        name: rt.model_name.clone(),
        context: plan.context,
        max_context: plan.max_context,
        weights_gib: plan.weights as f64 / (1u64 << 30) as f64,
        load_secs: rt.loaded.load_time.as_secs_f64(),
        drafter: rt.session.drafter().map(|d| {
            format!(
                "{} · {} drafts · {:.1} GiB",
                d.name().split(" (").next().unwrap_or(""),
                d.max_drafts(),
                rt.draft_loaded_bytes as f64 / (1u64 << 30) as f64
            )
        }),
        vision: rt.vision.as_ref().map(|v| {
            v.path
                .file_name()
                .map_or_else(String::new, |n| n.to_string_lossy().into_owned())
        }),
        sampling: format!(
            "t {} · p {} · k {} · min-p {}",
            sampling.temperature, sampling.top_p, sampling.top_k, sampling.min_p
        ),
    })));
    while let Ok(msg) = rx.recv() {
        match msg {
            ToWorker::Reset => rt.session.reset(),
            ToWorker::Turn {
                messages,
                think,
                tools,
            } => {
                cancel.store(false, Ordering::SeqCst);
                let mut vars = Map::new();
                vars.insert("enable_thinking".into(), Json::Bool(think));
                if let Some(effort) = &opts.effort {
                    vars.insert("reasoning_effort".into(), Json::String(effort.clone()));
                }
                let request = ChatRequest {
                    messages,
                    tools,
                    max_tokens: opts.max_tokens,
                    sampling: sampling.clone(),
                    template_vars: vars,
                    stop: Vec::new(),
                    reasoning_budget: opts.budget,
                    raw_reasoning: false,
                };
                let hooks = Hooks {
                    tx: &tx,
                    cancel: &cancel,
                };
                match rt.chat(&request, hooks) {
                    Ok(result) => {
                        let c = &result.completion;
                        let spec = &rt.session.spec_stats;
                        let tool_calls: Vec<_> = result
                            .output
                            .tool_calls
                            .iter()
                            .map(|c| (c.name.clone(), c.arguments.clone()))
                            .collect();
                        send(FromWorker::Done(Box::new(TurnDone {
                            tool_calls,
                            content: result.output.content,
                            reasoning: result.output.reasoning,
                            prompt_tokens: c.prompt_tokens,
                            reused: c.reused_tokens,
                            prefill_tps: (c.prompt_tokens - c.reused_tokens) as f64
                                / c.prefill.as_secs_f64().max(1e-9),
                            generated: c.tokens.len(),
                            decode_tps: c.tokens.len() as f64 / c.decode.as_secs_f64().max(1e-9),
                            stop: format!("{:?}", c.stop),
                            acceptance: (spec.steps > 0).then(|| spec.acceptance_length()),
                            ctx_used: rt.session.len(),
                            ctx_cap: rt.session.capacity(),
                        })));
                    }
                    Err(e) => send(FromWorker::TurnFailed(e.to_string())),
                }
            }
        }
    }
}

#[derive(Clone, Copy, PartialEq)]
pub enum Role {
    User,
    Assistant,
    Tool,
    Note,
}

#[derive(Clone, Copy, PartialEq)]
pub enum ToolState {
    Queued,
    Waiting,
    Running,
    Ok,
    Failed,
    Denied,
}

pub struct ToolView {
    pub id: String,
    pub name: String,
    pub args: Map<String, Json>,
    pub summary: String,
    pub state: ToolState,
    pub output: String,
    pub secs: f32,
    pub started: Option<Instant>,
}

pub struct Turn {
    pub role: Role,
    pub reasoning: String,
    pub content: String,
    pub done: bool,
    pub stats: Option<String>,
    pub error: Option<String>,
    pub tool: Option<ToolView>,
    /// Chips for the files that came with a user message.
    pub attachments: Vec<String>,
}

impl Turn {
    fn new(role: Role, content: String) -> Self {
        Self {
            role,
            reasoning: String::new(),
            content,
            done: role != Role::Assistant,
            stats: None,
            error: None,
            tool: None,
            attachments: Vec::new(),
        }
    }
}

#[derive(Clone, PartialEq)]
pub enum Phase {
    Loading(String),
    Ready,
    Generating,
    /// Running (or waiting to run) the model's tool calls.
    Tools,
    Failed(String),
}

/// Slash commands offered by the palette: (text to insert, description).
pub const COMMANDS: &[(&str, &str)] = &[
    ("/new", "start a new session (this one stays in /resume)"),
    ("/resume", "pick up a past session"),
    ("/think on", "turn reasoning on"),
    ("/think off", "turn reasoning off"),
    ("/thoughts", "show or hide the model's thinking"),
    ("/stats", "speed and context of the last turn"),
    ("/help", "list the commands"),
    ("/tools off", "no tools"),
    ("/tools ask", "approve writes, commands and downloads"),
    ("/tools auto", "run tools without asking"),
    ("/attach ", "attach a file by path"),
    ("/paste", "attach the clipboard (image, files or text)"),
    ("/screenshot", "drag out a screen region to attach"),
    ("/detach", "remove the attachments"),
    ("/workspace", "show this chat's sandbox folder"),
    ("/exit", "leave the chat"),
];

/// Extra state of a coding-agent session.
pub struct Agent {
    pub project: PathBuf,
    pub shared: SharedRef,
    pub ponytail: Ponytail,
    pub plan: bool,
    pub compact: bool,
    map: String,
    notes: Option<String>,
    pub branch: Option<String>,
    pub dirty: usize,
    /// The model asked the user something: stop after this round.
    asked: bool,
}

impl Agent {
    fn refresh_git(&mut self) {
        self.branch = brain::git_branch(&self.project);
        self.dirty = brain::git_dirty(&self.project);
    }
}

/// Commands in an agent session.
pub const AGENT_COMMANDS: &[(&str, &str)] = &[
    (
        "/plan on",
        "read-only: explore and write a plan, change nothing",
    ),
    ("/plan off", "let the agent make changes again"),
    (
        "/review",
        "ponytail: find over-engineering in your uncommitted diff",
    ),
    ("/audit", "ponytail: find code that does not need to exist"),
    (
        "/debt",
        "ponytail: ledger of the shortcuts marked `ponytail:`",
    ),
    ("/ponytail full", "minimal-code guidance: the full ladder"),
    ("/ponytail lite", "minimal-code guidance: light touch"),
    (
        "/ponytail ultra",
        "minimal-code guidance: delete before adding",
    ),
    ("/ponytail off", "no minimal-code guidance"),
    ("/compact", "snapshot the session and clear old history now"),
    ("/ctx", "context use and what was saved"),
    ("/todo", "show the agent's plan"),
    (
        "/tools edits",
        "file edits run freely; commands and downloads ask",
    ),
    ("/tools ask", "confirm every change"),
    ("/tools auto", "never ask"),
    ("/new", "start a new session (this one stays in /resume)"),
    ("/resume", "pick up a past session"),
    ("/think on", "turn reasoning on"),
    ("/think off", "turn reasoning off"),
    ("/thoughts", "show or hide the model's thinking"),
    ("/attach ", "attach a file by path"),
    ("/paste", "attach the clipboard (image, files or text)"),
    ("/screenshot", "drag out a screen region to attach"),
    ("/help", "list the commands"),
    ("/exit", "leave the agent"),
];

/// The `/resume` list.
pub struct ResumeList {
    pub items: Vec<sessions::Meta>,
    pub sel: usize,
}

pub struct Chat {
    tx: Sender<ToWorker>,
    rx: Receiver<FromWorker>,
    cancel: Arc<AtomicBool>,
    handle: Option<JoinHandle<()>>,
    pub label: String,
    pub phase: Phase,
    pub since: Instant,
    pub info: Option<LoadInfo>,
    pub turns: Vec<Turn>,
    history: Vec<Json>,
    pub input: String,
    /// Cursor position in chars.
    pub cursor: usize,
    pub think: bool,
    pub show_thinking: bool,
    gen_started: Option<Instant>,
    first_token: Option<Instant>,
    gen_tokens: usize,
    prefill: Option<(usize, usize)>,
    ctx: Option<(usize, usize)>,
    follow: bool,
    top: Cell<usize>,
    max_top: Cell<usize>,
    page: Cell<usize>,
    last_stats: String,
    palette_sel: usize,
    palette_off: bool,
    system: Option<String>,
    pub tools: ToolMode,
    workspace: String,
    network: bool,
    search: crate::search::Config,
    sandbox: Option<Sandbox>,
    pub attachments: Vec<Attachment>,
    /// Tool rounds since the last user message.
    rounds: usize,
    call_seq: usize,
    /// Calls still to run: (turn index, call id, name, arguments).
    queue: VecDeque<(usize, String, String, Map<String, Json>)>,
    tool_msgs: Vec<Json>,
    /// A call waiting for the user's yes or no.
    pub approval: Option<(usize, String, String, Map<String, Json>)>,
    exec_rx: Option<Receiver<(usize, bool, String, f32)>>,
    tool_cancel: Arc<AtomicBool>,
    aborted: bool,
    flash: Option<(String, bool, Instant)>,
    pub agent: Option<Agent>,
    pub id: String,
    created: u64,
    title: Option<String>,
    pub resume: Option<ResumeList>,
}

/// Most tool rounds the model may chain after one message.
const MAX_ROUNDS: usize = 24;

impl Chat {
    pub fn start(app: &App) -> Result<Chat, String> {
        let opts = ChatOpts::from_app(app)?;
        let (tx, wrx) = channel();
        let (wtx, rx) = channel();
        let cancel = Arc::new(AtomicBool::new(false));
        let label = opts
            .model
            .file_stem()
            .map(|s| s.to_string_lossy().to_string())
            .unwrap_or_default();
        let history = Vec::new();
        let (think, show) = (opts.think, opts.show_thinking);
        let (system, tools, network) = (opts.system.clone(), opts.tools, opts.network);
        let search = opts.search.clone();
        // Each chat gets its own sandbox folder inside the configured one.
        let workspace = session_dir(&opts.workspace, &label).display().to_string();
        let (agent, sandbox) = match &opts.agent {
            Some(a) => {
                let mut sb =
                    Sandbox::with_state(a.project.clone(), agent_state_dir(&a.project), network)?;
                sb.output_limit = coding::AGENT_OUTPUT_LIMIT;
                sb.search = search.clone();
                let mut agent = Agent {
                    project: a.project.clone(),
                    shared: coding::new_shared(),
                    ponytail: a.ponytail,
                    plan: false,
                    compact: a.compact,
                    map: format!("{}{}", brain::repo_map(&a.project), brain::env_notes()),
                    notes: brain::instructions(&a.project),
                    branch: None,
                    dirty: 0,
                    asked: false,
                };
                agent.refresh_git();
                (Some(agent), Some(sb))
            }
            None => (None, None),
        };
        let label = match &agent {
            Some(a) => format!(
                "{} · {}",
                a.project
                    .file_name()
                    .map(|n| n.to_string_lossy().to_string())
                    .unwrap_or_default(),
                label
            ),
            None => label,
        };
        let flag = cancel.clone();
        let handle = std::thread::Builder::new()
            .name("ferrum-chat".into())
            .spawn(move || worker(opts, wrx, wtx, flag))
            .map_err(|e| e.to_string())?;
        Ok(Chat {
            tx,
            rx,
            cancel,
            handle: Some(handle),
            label,
            phase: Phase::Loading("starting".into()),
            since: Instant::now(),
            info: None,
            turns: Vec::new(),
            history,
            input: String::new(),
            cursor: 0,
            think,
            show_thinking: show,
            gen_started: None,
            first_token: None,
            gen_tokens: 0,
            prefill: None,
            ctx: None,
            follow: true,
            top: Cell::new(0),
            max_top: Cell::new(0),
            page: Cell::new(10),
            last_stats: String::new(),
            palette_sel: 0,
            palette_off: false,
            system,
            tools,
            workspace,
            network,
            search,
            sandbox,
            attachments: Vec::new(),
            rounds: 0,
            call_seq: 0,
            queue: VecDeque::new(),
            tool_msgs: Vec::new(),
            approval: None,
            exec_rx: None,
            tool_cancel: Arc::new(AtomicBool::new(false)),
            aborted: false,
            flash: None,
            agent,
            id: sessions::new_id(),
            created: bench::now(),
            title: None,
            resume: None,
        })
    }

    /// Tell the worker to stop; the returned handle completes once the model is unloaded.
    pub fn close(mut self) -> Option<JoinHandle<()>> {
        self.cancel.store(true, Ordering::SeqCst);
        drop(self.tx);
        self.handle.take()
    }

    /// Commands matching what is typed after a leading `/`.
    pub fn palette(&self) -> Vec<(&'static str, &'static str)> {
        if self.phase != Phase::Ready
            || self.palette_off
            || !self.input.starts_with('/')
            || self.input.contains('\n')
        {
            return Vec::new();
        }
        let q = self.input.to_lowercase();
        let word = q.trim_start_matches('/');
        let list: &[(&'static str, &'static str)] = if self.agent.is_some() {
            AGENT_COMMANDS
        } else {
            COMMANDS
        };
        let mut out: Vec<_> = list
            .iter()
            .filter(|(n, _)| n.starts_with(&q))
            .copied()
            .collect();
        out.extend(
            list.iter()
                .filter(|(n, _)| !n.starts_with(&q) && n.contains(word))
                .copied(),
        );
        out
    }

    fn set_input(&mut self, text: &str) {
        self.input = text.to_string();
        self.cursor = self.input.chars().count();
    }

    /// The model is writing or its tools are running.
    pub fn ctx_use(&self) -> Option<(usize, usize)> {
        self.ctx
    }

    pub fn generating(&self) -> bool {
        matches!(self.phase, Phase::Generating | Phase::Tools)
    }

    pub fn accepts_input(&self) -> bool {
        !matches!(self.phase, Phase::Loading(_) | Phase::Failed(_))
    }

    pub fn flash(&mut self, text: impl Into<String>, ok: bool) {
        self.flash = Some((text.into(), ok, Instant::now()));
    }

    pub fn flash_text(&self) -> Option<(&str, bool)> {
        self.flash
            .as_ref()
            .filter(|(_, _, t)| t.elapsed() < Duration::from_secs(5))
            .map(|(s, ok, _)| (s.as_str(), *ok))
    }

    /// Messages for the model: the system prompt (plus the tool briefing) and the history.
    fn messages_for_send(&mut self) -> Vec<Json> {
        let mut system = self.system.clone().unwrap_or_default();
        if let Some(a) = &self.agent {
            if !system.is_empty() {
                system.push_str("\n\n");
            }
            system.push_str(&brain::system_prompt(
                &a.project,
                a.ponytail,
                a.plan,
                self.network,
                &a.map,
                a.notes.as_deref(),
            ));
        } else if self.tools != ToolMode::Off
            && let Ok(sb) = self.sandbox()
        {
            if !system.is_empty() {
                system.push_str("\n\n");
            }
            system.push_str(&agent::system_prompt(&sb.workspace, self.network));
        }
        let mut messages = Vec::new();
        if !system.is_empty() {
            messages.push(json!({"role": "system", "content": system}));
        }
        messages.extend(self.history.iter().cloned());
        messages
    }

    fn tool_defs(&self) -> Vec<Json> {
        if self.tools == ToolMode::Off {
            Vec::new()
        } else if let Some(a) = &self.agent {
            coding::tool_defs(a.plan, self.network)
        } else {
            agent::tool_defs(self.network)
        }
    }

    /// Send the history to the model and show an empty reply being written.
    fn start_turn(&mut self) {
        self.maybe_compact();
        let messages = self.messages_for_send();
        let tools = self.tool_defs();
        self.turns.push(Turn::new(Role::Assistant, String::new()));
        self.phase = Phase::Generating;
        self.gen_started = Some(Instant::now());
        self.first_token = None;
        self.gen_tokens = 0;
        self.follow = true;
        let _ = self.tx.send(ToWorker::Turn {
            messages,
            think: self.think,
            tools,
        });
    }

    pub fn sandbox(&mut self) -> Result<Sandbox, String> {
        if self.sandbox.is_none() {
            let mut sb = Sandbox::new(&self.workspace, self.network)?;
            sb.search = self.search.clone();
            self.sandbox = Some(sb);
        }
        Ok(self.sandbox.clone().expect("just created"))
    }

    pub fn pump(&mut self) {
        let mut results = Vec::new();
        if let Some(rx) = &self.exec_rx {
            while let Ok(r) = rx.try_recv() {
                results.push(r);
            }
        }
        for (idx, ok, out, secs) in results {
            self.tool_done(idx, ok, out, secs);
        }
        while let Ok(msg) = self.rx.try_recv() {
            match msg {
                FromWorker::Stage(s) => self.phase = Phase::Loading(s),
                FromWorker::Ready(info) => {
                    if self.agent.is_none() {
                        self.label = info.name.clone();
                    }
                    self.info = Some(*info);
                    self.phase = Phase::Ready;
                    self.since = Instant::now();
                }
                FromWorker::Failed(e) => self.phase = Phase::Failed(e),
                FromWorker::Begin { prompt, reused } => {
                    self.prefill = Some((reused, prompt));
                }
                FromWorker::Prefill(done, total) => self.prefill = Some((done, total)),
                FromWorker::Delta { reasoning, text } => {
                    self.prefill = None;
                    self.first_token.get_or_insert_with(Instant::now);
                    if let Some(t) = self.turns.last_mut() {
                        if reasoning {
                            t.reasoning.push_str(&text);
                        } else {
                            t.content.push_str(&text);
                        }
                    }
                }
                FromWorker::Tokens(n) => self.gen_tokens = n,
                FromWorker::Done(d) => self.finish(*d),
                FromWorker::TurnFailed(e) => {
                    if let Some(t) = self.turns.last_mut() {
                        t.done = true;
                        t.error = Some(e);
                    }
                    if self.rounds == 0 {
                        self.history.pop();
                    }
                    self.rounds = 0;
                    self.end_generation();
                }
            }
        }
    }

    fn end_generation(&mut self) {
        self.persist();
        self.phase = Phase::Ready;
        self.gen_started = None;
        self.first_token = None;
        self.prefill = None;
    }

    fn finish(&mut self, d: TurnDone) {
        let cancelled = d.stop == "Cancelled";
        let mut stats = format!(
            "{:.1} tok/s · {} tokens · prompt {}{}",
            d.decode_tps,
            d.generated,
            d.prompt_tokens,
            if d.reused > 0 {
                format!(" ({} cached)", d.reused)
            } else {
                String::new()
            }
        );
        if d.prefill_tps.is_finite() && d.prompt_tokens > d.reused {
            stats += &format!(" · prefill {:.0} tok/s", d.prefill_tps);
        }
        if let Some(a) = d.acceptance {
            stats += &format!(" · drafter {a:.2} per step");
        }
        stats += &format!(
            " · {}",
            if cancelled {
                "stopped".to_string()
            } else {
                d.stop.to_lowercase()
            }
        );
        if let Some(t) = self.turns.last_mut() {
            t.content = d.content.clone();
            t.reasoning = d.reasoning.clone().unwrap_or_default();
            t.done = true;
            t.stats = Some(stats.clone());
        }
        let mut reply = json!({"role": "assistant", "content": d.content});
        if let Some(r) = d.reasoning {
            reply["reasoning_content"] = Json::String(r);
        }
        self.ctx = Some((d.ctx_used, d.ctx_cap));
        self.last_stats = stats;
        if !d.tool_calls.is_empty() && self.tools != ToolMode::Off && !cancelled {
            if self.rounds >= MAX_ROUNDS {
                self.history.push(reply);
                self.note(format!("stopped after {MAX_ROUNDS} rounds of tool calls"));
                self.rounds = 0;
                self.end_generation();
                return;
            }
            let mut calls = Vec::new();
            for (name, args) in d.tool_calls {
                self.call_seq += 1;
                let id = format!("call_{}", self.call_seq);
                calls.push(json!({
                    "id": id,
                    "type": "function",
                    "function": {"name": name, "arguments": Json::Object(args.clone()).to_string()}
                }));
                let mut turn = Turn::new(Role::Tool, String::new());
                turn.tool = Some(ToolView {
                    id: id.clone(),
                    args: args.clone(),
                    summary: agent::summary(&name, &args),
                    name: name.clone(),
                    state: ToolState::Queued,
                    output: String::new(),
                    secs: 0.0,
                    started: None,
                });
                self.turns.push(turn);
                self.queue.push_back((self.turns.len() - 1, id, name, args));
            }
            reply["tool_calls"] = Json::Array(calls);
            self.history.push(reply);
            self.rounds += 1;
            self.aborted = false;
            self.phase = Phase::Tools;
            self.next_tool();
            return;
        }
        self.history.push(reply);
        self.rounds = 0;
        self.end_generation();
    }

    fn tool_message(&mut self, idx: usize, text: &str) {
        let (id, name) = self
            .turns
            .get(idx)
            .and_then(|t| t.tool.as_ref())
            .map(|t| (t.id.clone(), t.name.clone()))
            .unwrap_or_default();
        self.tool_msgs.push(json!({
            "role": "tool",
            "tool_call_id": id,
            "name": name,
            "content": text,
        }));
    }

    fn set_tool(&mut self, idx: usize, state: ToolState, output: &str) {
        if let Some(t) = self.turns.get_mut(idx).and_then(|t| t.tool.as_mut()) {
            t.state = state;
            t.output = output.to_string();
        }
    }

    /// Run, ask about, or skip the next queued call; start the next generation when none are left.
    fn next_tool(&mut self) {
        if self.aborted {
            while let Some((idx, _, _, _)) = self.queue.pop_front() {
                self.set_tool(idx, ToolState::Denied, "cancelled");
                self.tool_message(idx, "cancelled by the user");
            }
            return self.finish_round();
        }
        let Some(call) = self.queue.pop_front() else {
            return self.finish_round();
        };
        let needs = match &self.agent {
            Some(_) => coding::needs_approval(&call.2, self.tools),
            None => self.tools == ToolMode::Ask && agent::needs_approval(&call.2),
        };
        if needs {
            self.set_tool(call.0, ToolState::Waiting, "");
            self.approval = Some(call);
            return;
        }
        self.run_tool(call);
    }

    fn run_tool(&mut self, call: (usize, String, String, Map<String, Json>)) {
        let (idx, _id, name, args) = call;
        let sb = match self.sandbox() {
            Ok(sb) => sb,
            Err(e) => {
                self.set_tool(idx, ToolState::Failed, &e);
                self.tool_message(idx, &format!("error: {e}"));
                return self.next_tool();
            }
        };
        if let Some(t) = self.turns.get_mut(idx).and_then(|t| t.tool.as_mut()) {
            t.state = ToolState::Running;
            t.started = Some(Instant::now());
        }
        let (tx, rx) = channel();
        self.exec_rx = Some(rx);
        self.tool_cancel.store(false, Ordering::SeqCst);
        let cancel = self.tool_cancel.clone();
        let coder = self.agent.as_ref().map(|a| (a.shared.clone(), a.plan));
        std::thread::spawn(move || {
            let started = Instant::now();
            let (ok, out) = match coder {
                Some((shared, plan)) => {
                    if plan && !coding::read_only(&name) {
                        (false, "error: plan mode is on, so nothing can be changed. Finish the plan; the user will turn plan mode off to carry it out.".to_string())
                    } else if shared
                        .lock()
                        .map(|mut g| g.is_loop(&name, &args))
                        .unwrap_or(false)
                    {
                        (false, "error: you already made this exact call twice. Do something different, or ask the user.".to_string())
                    } else {
                        coding::execute(&shared, &sb, &name, &args, &cancel)
                    }
                }
                None => sb.execute(&name, &args, &cancel),
            };
            let _ = tx.send((idx, ok, out, started.elapsed().as_secs_f32()));
        });
    }

    fn tool_done(&mut self, idx: usize, ok: bool, out: String, secs: f32) {
        self.exec_rx = None;
        if let Some(t) = self.turns.get_mut(idx).and_then(|t| t.tool.as_mut()) {
            t.state = if ok { ToolState::Ok } else { ToolState::Failed };
            t.output = out.clone();
            t.secs = secs;
        }
        self.tool_message(idx, &out);
        let question = self
            .turns
            .get(idx)
            .and_then(|t| t.tool.as_ref())
            .filter(|t| t.name == "ask_user")
            .and_then(|t| {
                t.args
                    .get("question")
                    .and_then(Json::as_str)
                    .map(str::to_string)
            });
        if let (Some(q), Some(a)) = (question, self.agent.as_mut()) {
            a.asked = true;
            let mut turn = Turn::new(Role::Assistant, format!("**{q}**"));
            turn.done = true;
            self.turns.push(turn);
        }
        self.next_tool();
    }

    pub fn approve(&mut self, always: bool) {
        if let Some(call) = self.approval.take() {
            if always {
                self.tools = ToolMode::Auto;
            }
            self.run_tool(call);
        }
    }

    pub fn deny(&mut self) {
        if let Some((idx, _, _, _)) = self.approval.take() {
            self.set_tool(idx, ToolState::Denied, "denied");
            self.tool_message(idx, "The user denied this tool call.");
            self.next_tool();
        }
    }

    fn finish_round(&mut self) {
        self.history.append(&mut self.tool_msgs);
        if let Some(a) = self.agent.as_mut() {
            a.refresh_git();
            if std::mem::take(&mut a.asked) {
                self.aborted = false;
                self.rounds = 0;
                self.end_generation();
                return;
            }
        }
        if self.aborted {
            self.aborted = false;
            self.rounds = 0;
            self.end_generation();
            self.note("stopped");
            return;
        }
        self.start_turn();
    }

    /// Esc while tools run: kill the running command and skip the rest.
    fn cancel_tools(&mut self) {
        self.aborted = true;
        self.tool_cancel.store(true, Ordering::SeqCst);
        if self.approval.is_some() {
            self.deny();
        }
    }

    pub fn insert_str(&mut self, s: &str) {
        let at = self
            .input
            .char_indices()
            .nth(self.cursor)
            .map_or(self.input.len(), |(i, _)| i);
        self.input.insert_str(at, s);
        self.cursor += s.chars().count();
    }

    fn byte_at(&self, chars: usize) -> usize {
        self.input
            .char_indices()
            .nth(chars)
            .map_or(self.input.len(), |(i, _)| i)
    }

    fn note(&mut self, text: impl Into<String>) {
        self.turns.push(Turn::new(Role::Note, text.into()));
        self.follow = true;
    }

    /// Images can be shown to the model (a vision projector is loaded).
    fn sees_images(&self) -> bool {
        self.info.as_ref().is_some_and(|i| i.vision.is_some())
    }

    fn submit(&mut self, text: String) {
        self.submit_with(text.clone(), text);
    }

    /// Send `body` to the model while the transcript shows the shorter `display`.
    fn submit_with(&mut self, display: String, body: String) {
        let text = body;
        let attachments = std::mem::take(&mut self.attachments);
        let sees = self.sees_images();
        let mut content = String::new();
        let mut pictures = Vec::new();
        for a in &attachments {
            if sees && a.kind == attach::Kind::Image {
                match attach::image_bytes(a) {
                    Ok(bytes) => {
                        pictures.push(ferrum::vision::media::image_part_from_bytes(&bytes))
                    }
                    Err(e) => self.note(e),
                }
            }
            content += &a.block(sees);
            content += "\n\n";
        }
        content += &text;
        let message = if pictures.is_empty() {
            json!({"role": "user", "content": content})
        } else {
            pictures.push(json!({"type": "text", "text": content}));
            json!({"role": "user", "content": pictures})
        };
        self.history.push(message);
        if self.title.is_none() {
            self.title = Some(clip(display.lines().next().unwrap_or(""), 70));
        }
        let mut turn = Turn::new(Role::User, display);
        turn.attachments = attachments.iter().map(|a| a.chip()).collect();
        self.turns.push(turn);
        self.rounds = 0;
        if let Some(a) = &self.agent
            && let Ok(mut g) = a.shared.lock()
        {
            g.new_message();
        }
        self.start_turn();
    }

    /// Free context before a turn when it is filling up. Old tool output is
    /// elided in one batch (so the model's cache is rebuilt once, not every
    /// turn); when that is not enough the session is snapshotted.
    fn maybe_compact(&mut self) {
        let Some((used, cap)) = self.ctx else { return };
        let Some(a) = &self.agent else { return };
        if !a.compact || cap == 0 {
            return;
        }
        let ratio = used as f64 / cap as f64;
        if ratio >= 0.80 {
            self.compact_now("context is 80% full");
        } else if ratio >= 0.55 {
            let Some(a) = &self.agent else { return };
            let Ok(mut g) = a.shared.lock() else { return };
            let (n, saved) = brain::evict_old(&mut self.history, &mut g.store, 6);
            if n > 0 {
                g.evicted += n;
                g.evicted_bytes += saved;
                g.forget_reads();
                drop(g);
                self.ctx = None;
                self.note(format!(
                    "context {:.0}% full: elided {n} old tool outputs ({} KB); ctx_read brings any back",
                    ratio * 100.0,
                    saved / 1024
                ));
            }
        }
    }

    fn compact_now(&mut self, why: &str) {
        let Some(a) = &self.agent else { return };
        let project = a.project.clone();
        let Ok(mut g) = a.shared.lock() else { return };
        // Keep what the old output said reachable, then replace the history.
        brain::evict_old(&mut self.history, &mut g.store, 0);
        let removed = brain::compact(&mut self.history, &g, &project);
        g.compactions += 1;
        g.forget_reads();
        drop(g);
        self.ctx = None;
        if removed > 0 {
            self.note(format!(
                "{why}: compacted the session into a snapshot ({} KB freed)",
                removed / 1024
            ));
        } else {
            self.note("nothing to compact yet");
        }
    }

    fn ctx_report(&self) -> String {
        let mut s = match self.ctx {
            Some((u, c)) => format!(
                "context {u} / {c} tokens ({:.0}%)",
                u as f64 * 100.0 / c.max(1) as f64
            ),
            None => "context use is measured after each reply".to_string(),
        };
        if let Some(a) = &self.agent
            && let Ok(g) = a.shared.lock()
        {
            s += &format!(
                "\n{} long outputs stored, {} KB kept out of the context\n{} old outputs elided ({} KB) · {} compaction(s)",
                g.store.len(),
                g.store.saved / 1024,
                g.evicted,
                g.evicted_bytes / 1024,
                g.compactions
            );
        }
        s
    }

    fn kind(&self) -> &'static str {
        if self.agent.is_some() {
            "agent"
        } else {
            "chat"
        }
    }

    fn project_key(&self) -> String {
        self.agent
            .as_ref()
            .map(|a| a.project.display().to_string())
            .unwrap_or_default()
    }

    /// Write this session to disk (nothing is saved before the first message).
    pub fn persist(&self) {
        if self.history.is_empty() {
            return;
        }
        let state = |s: ToolState| match s {
            ToolState::Ok => "ok",
            ToolState::Failed => "failed",
            ToolState::Denied => "denied",
            _ => "interrupted",
        };
        let turns: Vec<Json> = self
            .turns
            .iter()
            .map(|t| {
                json!({
                    "role": match t.role { Role::User => "user", Role::Assistant => "assistant", Role::Tool => "tool", Role::Note => "note" },
                    "reasoning": t.reasoning, "content": t.content, "stats": t.stats,
                    "error": t.error, "attachments": t.attachments,
                    "tool": t.tool.as_ref().map(|v| json!({
                        "id": v.id, "name": v.name, "args": v.args, "summary": v.summary,
                        "state": state(v.state), "output": v.output, "secs": v.secs,
                    })),
                })
            })
            .collect();
        let mut data = json!({
            "id": self.id, "kind": self.kind(), "title": self.title,
            "model": self.label, "project": self.project_key(), "workspace": self.workspace,
            "created": self.created, "updated": bench::now(),
            "turns": turns, "history": self.history,
        });
        if let Some(a) = &self.agent {
            let mut agent = a.shared.lock().map(|g| g.snapshot()).unwrap_or(Json::Null);
            agent["ponytail"] = json!(a.ponytail.label());
            agent["plan"] = json!(a.plan);
            data["agent"] = agent;
        }
        sessions::save(&self.id, &data);
    }

    /// Replace the conversation with a saved one.
    fn restore(&mut self, j: &Json) {
        self.history = j["history"].as_array().cloned().unwrap_or_default();
        self.turns = j["turns"]
            .as_array()
            .into_iter()
            .flatten()
            .map(|t| {
                let role = match t["role"].as_str() {
                    Some("user") => Role::User,
                    Some("assistant") => Role::Assistant,
                    Some("tool") => Role::Tool,
                    _ => Role::Note,
                };
                let mut turn = Turn::new(role, t["content"].as_str().unwrap_or("").into());
                turn.done = true;
                turn.reasoning = t["reasoning"].as_str().unwrap_or("").into();
                turn.stats = t["stats"].as_str().map(str::to_string);
                turn.error = t["error"].as_str().map(str::to_string);
                turn.attachments = t["attachments"]
                    .as_array()
                    .map(|a| {
                        a.iter()
                            .filter_map(|x| x.as_str().map(str::to_string))
                            .collect()
                    })
                    .unwrap_or_default();
                if let Some(v) = t["tool"].as_object() {
                    turn.tool = Some(ToolView {
                        id: v["id"].as_str().unwrap_or("").into(),
                        name: v["name"].as_str().unwrap_or("").into(),
                        args: v["args"].as_object().cloned().unwrap_or_default(),
                        summary: v["summary"].as_str().unwrap_or("").into(),
                        state: match v["state"].as_str() {
                            Some("ok") => ToolState::Ok,
                            Some("failed") => ToolState::Failed,
                            _ => ToolState::Denied,
                        },
                        output: v["output"].as_str().unwrap_or("").into(),
                        secs: v["secs"].as_f64().unwrap_or(0.0) as f32,
                        started: None,
                    });
                }
                turn
            })
            .collect();
        self.id = j["id"].as_str().unwrap_or(&self.id).to_string();
        self.created = j["created"].as_u64().unwrap_or(self.created);
        self.title = j["title"].as_str().map(str::to_string);
        if self.agent.is_none()
            && let Some(w) = j["workspace"].as_str().filter(|w| !w.is_empty())
        {
            self.workspace = w.to_string();
            self.sandbox = None;
        }
        if let Some(a) = self.agent.as_mut() {
            if let Ok(mut g) = a.shared.lock() {
                g.restore(&j["agent"]);
            }
            if let Some(p) = j["agent"]["ponytail"].as_str() {
                a.ponytail = Ponytail::parse(p);
            }
            a.plan = j["agent"]["plan"].as_bool().unwrap_or(false);
        }
        self.queue.clear();
        self.tool_msgs.clear();
        self.attachments.clear();
        self.approval = None;
        self.rounds = 0;
        self.ctx = None;
        self.last_stats.clear();
        self.follow = true;
        let _ = self.tx.send(ToWorker::Reset);
    }

    /// Keep this session on disk and start a fresh one in the same loaded model.
    fn new_session(&mut self) {
        self.persist();
        self.reset();
        self.id = sessions::new_id();
        self.created = bench::now();
        self.title = None;
        self.note("new session · the previous one is in /resume");
    }

    fn open_resume(&mut self) {
        let project = self.agent.as_ref().map(|a| a.project.display().to_string());
        let items: Vec<sessions::Meta> = sessions::list(self.kind(), project.as_deref())
            .into_iter()
            .filter(|m| m.id != self.id)
            .collect();
        if items.is_empty() {
            self.note("no earlier sessions to resume");
        } else {
            self.resume = Some(ResumeList { items, sel: 0 });
        }
    }

    fn reset(&mut self) {
        self.history.clear();
        self.tool_msgs.clear();
        self.queue.clear();
        self.attachments.clear();
        self.rounds = 0;
        self.turns.clear();
        self.ctx = None;
        self.last_stats.clear();
        if let Some(a) = &self.agent
            && let Ok(mut g) = a.shared.lock()
        {
            g.reset_conversation();
        }
        let _ = self.tx.send(ToWorker::Reset);
    }

    /// A line starting with `/`. Returns true when the chat should close.
    fn command(&mut self, line: &str) -> bool {
        match line.trim() {
            "/exit" | "/quit" => return true,
            "/reset" | "/clear" | "/new" => self.new_session(),
            "/resume" => self.open_resume(),
            "/think on" | "/think off" => {
                self.think = line.trim().ends_with("on");
                self.note(format!(
                    "thinking {}",
                    if self.think { "on" } else { "off" }
                ));
            }
            "/stats" => {
                let s = if self.last_stats.is_empty() {
                    "no turns yet".to_string()
                } else {
                    self.last_stats.clone()
                };
                let ctx = self
                    .ctx
                    .map(|(u, c)| format!(" · context {u} / {c} tokens"))
                    .unwrap_or_default();
                self.note(format!("{s}{ctx}"));
            }
            "/thoughts" => {
                self.show_thinking = !self.show_thinking;
                self.note(format!(
                    "thinking is now {}",
                    if self.show_thinking {
                        "shown"
                    } else {
                        "hidden"
                    }
                ));
            }
            "/help" | "/?" => {
                let mut text = String::from("type / to see the commands as you type\n");
                let list: &[(&str, &str)] = if self.agent.is_some() {
                    AGENT_COMMANDS
                } else {
                    COMMANDS
                };
                for &(name, what) in list {
                    text += &format!("{name:<16} {what}\n");
                }
                text += "ctrl-t       show or hide the model's thinking\n";
                text += "alt+enter or a trailing \\ makes a new line";
                self.note(text);
            }
            "/tools edits" if self.agent.is_some() => {
                self.tools = ToolMode::Edits;
                self.note("approvals: file edits run freely, commands and downloads ask");
            }
            "/plan on" | "/plan off" if self.agent.is_some() => {
                let on = line.trim().ends_with("on");
                if let Some(a) = self.agent.as_mut() {
                    a.plan = on;
                }
                self.note(if on {
                    "plan mode: the agent can only read and plan"
                } else {
                    "plan mode off: the agent can make changes"
                });
            }
            "/review" if self.agent.is_some() => {
                let project = self
                    .agent
                    .as_ref()
                    .map(|a| a.project.clone())
                    .unwrap_or_default();
                match brain::git_diff(&project) {
                    Some(diff) if !diff.trim().is_empty() => {
                        self.submit_with("/review".into(), brain::review_prompt(&diff))
                    }
                    _ => self.note("no uncommitted changes to review"),
                }
            }
            "/audit" if self.agent.is_some() => {
                self.submit_with("/audit".into(), brain::AUDIT_PROMPT.into())
            }
            "/debt" if self.agent.is_some() => {
                self.submit_with("/debt".into(), brain::DEBT_PROMPT.into())
            }
            "/compact" if self.agent.is_some() => self.compact_now("compacted on request"),
            "/ctx" => {
                let report = self.ctx_report();
                self.note(report);
            }
            "/todo" if self.agent.is_some() => {
                let text = self
                    .agent
                    .as_ref()
                    .and_then(|a| a.shared.lock().ok())
                    .map(|g| {
                        if g.todos.is_empty() {
                            "no plan yet".to_string()
                        } else {
                            g.todos
                                .iter()
                                .map(|t| {
                                    let m = match t.state {
                                        coding::TodoState::Done => "[x]",
                                        coding::TodoState::Active => "[>]",
                                        coding::TodoState::Pending => "[ ]",
                                    };
                                    format!("{m} {}", t.text)
                                })
                                .collect::<Vec<_>>()
                                .join("\n")
                        }
                    })
                    .unwrap_or_default();
                self.note(text);
            }
            l if l.starts_with("/ponytail") && self.agent.is_some() => {
                let mode = l.trim_start_matches("/ponytail").trim();
                if matches!(mode, "off" | "lite" | "full" | "ultra") {
                    if let Some(a) = self.agent.as_mut() {
                        a.ponytail = Ponytail::parse(mode);
                    }
                    self.note(format!("ponytail {mode}"));
                } else {
                    let cur = self
                        .agent
                        .as_ref()
                        .map(|a| a.ponytail.label())
                        .unwrap_or("off");
                    self.note(format!("ponytail is {cur} (off, lite, full or ultra)"));
                }
            }
            "/tools off" | "/tools ask" | "/tools auto" => {
                self.tools = ToolMode::parse(line.trim().trim_start_matches("/tools "));
                self.note(match self.tools {
                    ToolMode::Off => "tools are off".to_string(),
                    ToolMode::Ask => {
                        "tools on: I'll ask before writes, commands and downloads".to_string()
                    }
                    ToolMode::Edits => "tools on: file edits run freely, commands ask".to_string(),
                    ToolMode::Auto => "tools on: they run without asking".to_string(),
                });
            }
            "/workspace" => match self.sandbox() {
                Ok(sb) => self.note(format!("workspace: {}", sb.workspace.display())),
                Err(e) => self.note(e),
            },
            "/paste" => self.paste_clipboard(),
            "/screenshot" => self.take_screenshot(),
            "/detach" => {
                let n = self.attachments.len();
                self.attachments.clear();
                self.flash(format!("removed {n} attachment(s)"), true);
            }
            l if l.starts_with("/attach") => {
                let arg = l.trim_start_matches("/attach").trim();
                if arg.is_empty() {
                    self.flash("usage: /attach /path/to/file", false);
                } else {
                    let found = attach::parse_dropped(arg);
                    let paths = if found.is_empty() {
                        vec![expand(arg)]
                    } else {
                        found
                    };
                    self.attach_paths(&paths);
                }
            }
            other => self.note(format!("unknown command {other} (try /help)")),
        }
        false
    }

    pub fn stop_generation(&mut self) {
        if self.phase == Phase::Tools {
            self.cancel_tools();
        } else {
            self.cancel.store(true, Ordering::SeqCst);
        }
    }

    // ---- attachments ----

    pub fn attach_paths(&mut self, paths: &[PathBuf]) {
        let sb = match self.sandbox() {
            Ok(sb) => sb,
            Err(e) => return self.flash(e, false),
        };
        for path in paths {
            match attach::add_file(&sb, path) {
                Ok(a) => {
                    self.flash(format!("attached {}", a.name), true);
                    self.attachments.push(a);
                }
                Err(e) => return self.flash(e, false),
            }
        }
    }

    /// Attach whatever is on the clipboard: files, an image, or (as text in the box) text.
    pub fn paste_clipboard(&mut self) {
        let sb = match self.sandbox() {
            Ok(sb) => sb,
            Err(e) => return self.flash(e, false),
        };
        match attach::clipboard(&sb) {
            Ok(Clip::Files(paths)) => self.attach_paths(&paths),
            Ok(Clip::Image(a)) => {
                let note = match (&a.ocr, a.kind) {
                    _ if self.sees_images() => {
                        format!("attached {} · the model will see it", a.name)
                    }
                    (Some(t), _) => {
                        format!("attached {} · read {} characters of text", a.name, t.len())
                    }
                    _ => format!("attached {} · no text found in it", a.name),
                };
                self.flash(note, true);
                self.attachments.push(a);
            }
            Ok(Clip::Text(text)) => self.insert_str(&text),
            Ok(Clip::Empty) => self.flash("the clipboard is empty", false),
            Err(e) => self.flash(e, false),
        }
    }

    pub fn take_screenshot(&mut self) {
        let sb = match self.sandbox() {
            Ok(sb) => sb,
            Err(e) => return self.flash(e, false),
        };
        match attach::screenshot(&sb) {
            Ok(Some(a)) => {
                self.flash(format!("attached {}", a.name), true);
                self.attachments.push(a);
            }
            Ok(None) => self.flash("screenshot cancelled", false),
            Err(e) => self.flash(e, false),
        }
    }

    fn scroll(&mut self, up: bool, amount: usize) {
        let top = if self.follow {
            self.max_top.get()
        } else {
            self.top.get()
        };
        let next = if up {
            top.saturating_sub(amount)
        } else {
            (top + amount).min(self.max_top.get())
        };
        self.follow = !up && next >= self.max_top.get();
        self.top.set(next);
    }
}

impl App {
    pub fn leave_chat(&mut self) {
        if let Some(chat) = self.chat.take() {
            chat.persist();
            if let Some(h) = chat.close() {
                self.closing.push(h);
            }
        }
        self.screen = Screen::Pick;
    }

    pub fn chat_key(&mut self, key: KeyEvent) {
        let before = self.chat.as_ref().map(|c| c.input.clone());
        self.chat_key_inner(key);
        if let (Some(c), Some(b)) = (self.chat.as_mut(), before)
            && c.input != b
        {
            c.palette_sel = 0;
            c.palette_off = false;
        }
    }

    fn chat_key_inner(&mut self, key: KeyEvent) {
        let ctrl = key.modifiers.contains(KeyModifiers::CONTROL);
        let alt = key.modifiers.contains(KeyModifiers::ALT);
        let shift = key.modifiers.contains(KeyModifiers::SHIFT);
        let Some(chat) = self.chat.as_mut() else {
            self.screen = Screen::Home;
            return;
        };
        let busy = matches!(chat.phase, Phase::Loading(_) | Phase::Failed(_));
        if busy {
            if matches!(key.code, KeyCode::Esc | KeyCode::Char('q'))
                || (ctrl && key.code == KeyCode::Char('c'))
            {
                self.leave_chat();
            }
            return;
        }
        if let Some(list) = chat.resume.as_mut() {
            let last = list.items.len() - 1;
            match key.code {
                KeyCode::Up | KeyCode::Char('k') => list.sel = list.sel.saturating_sub(1),
                KeyCode::Down | KeyCode::Char('j') => list.sel = (list.sel + 1).min(last),
                KeyCode::Esc | KeyCode::Char('q') => chat.resume = None,
                KeyCode::Enter => {
                    let id = list.items[list.sel].id.clone();
                    chat.resume = None;
                    match sessions::load(&id) {
                        Some(data) => {
                            chat.persist();
                            chat.restore(&data);
                            let n = chat.history.iter().filter(|m| m["role"] == "user").count();
                            chat.note(format!(
                                "resumed {:?} ({n} messages) · the model re-reads it with your next message",
                                chat.title.clone().unwrap_or_default()
                            ));
                        }
                        None => chat.flash("that session could not be read", false),
                    }
                }
                KeyCode::Char('x') | KeyCode::Delete => {
                    let id = list.items[list.sel].id.clone();
                    if self.armed.as_deref() == Some(id.as_str()) {
                        sessions::delete(&id);
                        list.items.remove(list.sel);
                        list.sel = list.sel.min(list.items.len().saturating_sub(1));
                        if list.items.is_empty() {
                            chat.resume = None;
                        }
                    } else {
                        self.confirm = Some(id);
                        chat.flash("press x again to delete this session", false);
                    }
                }
                _ => {}
            }
            return;
        }
        if chat.approval.is_some() {
            match key.code {
                KeyCode::Char('y') | KeyCode::Enter => chat.approve(false),
                KeyCode::Char('a') => chat.approve(true),
                KeyCode::Char('n') | KeyCode::Esc => chat.deny(),
                KeyCode::Char('c') if ctrl => chat.stop_generation(),
                _ => {}
            }
            return;
        }
        let palette = chat.palette();
        if !palette.is_empty() && !alt && !shift {
            let n = palette.len();
            let sel = chat.palette_sel.min(n - 1);
            match key.code {
                KeyCode::Up => {
                    chat.palette_sel = (sel + n - 1) % n;
                    return;
                }
                KeyCode::Down => {
                    chat.palette_sel = (sel + 1) % n;
                    return;
                }
                KeyCode::Tab => {
                    chat.set_input(palette[sel].0);
                    return;
                }
                KeyCode::Esc => {
                    chat.palette_off = true;
                    return;
                }
                KeyCode::Enter => {
                    chat.set_input(palette[sel].0);
                    // Commands that take an argument wait for it.
                    if palette[sel].0.ends_with(' ') {
                        return;
                    }
                }
                _ => {}
            }
        }
        match key.code {
            KeyCode::Esc => {
                if chat.generating() {
                    chat.stop_generation();
                } else if !chat.input.is_empty() {
                    chat.input.clear();
                    chat.cursor = 0;
                } else {
                    self.leave_chat();
                }
            }
            KeyCode::Char('c') if ctrl => {
                if chat.generating() {
                    chat.stop_generation();
                } else {
                    self.leave_chat();
                }
            }
            KeyCode::Char('d') if ctrl && chat.input.is_empty() => self.leave_chat(),
            KeyCode::Char('t') if ctrl => chat.show_thinking = !chat.show_thinking,
            KeyCode::PageUp => {
                let page = chat.page.get().max(2) - 1;
                chat.scroll(true, page)
            }
            KeyCode::PageDown => {
                let page = chat.page.get().max(2) - 1;
                chat.scroll(false, page)
            }
            KeyCode::Up if chat.input.is_empty() => chat.scroll(true, 1),
            KeyCode::Down if chat.input.is_empty() => chat.scroll(false, 1),
            KeyCode::Enter if alt || shift => chat.insert_str("\n"),
            KeyCode::Enter => {
                if chat.generating() {
                    return;
                }
                if chat.input.ends_with('\\') {
                    chat.input.pop();
                    chat.cursor = chat.input.chars().count();
                    chat.insert_str("\n");
                    return;
                }
                let mut text = chat.input.trim().to_string();
                chat.input.clear();
                chat.cursor = 0;
                if text.is_empty() && chat.attachments.is_empty() {
                    return;
                }
                if text.is_empty() {
                    text = "(see the attachment)".into();
                }
                if text.starts_with('/') {
                    if chat.command(&text) {
                        self.leave_chat();
                    }
                } else {
                    chat.submit(text);
                }
            }
            KeyCode::Left => chat.cursor = chat.cursor.saturating_sub(1),
            KeyCode::Right => chat.cursor = (chat.cursor + 1).min(chat.input.chars().count()),
            KeyCode::Home => chat.cursor = 0,
            KeyCode::End => chat.cursor = chat.input.chars().count(),
            KeyCode::Char('a') if ctrl => chat.cursor = 0,
            KeyCode::Char('e') if ctrl => chat.cursor = chat.input.chars().count(),
            KeyCode::Char('u') if ctrl => {
                let at = chat.byte_at(chat.cursor);
                chat.input.drain(..at);
                chat.cursor = 0;
            }
            KeyCode::Char('k') if ctrl => {
                let at = chat.byte_at(chat.cursor);
                chat.input.truncate(at);
            }
            KeyCode::Char('w') if ctrl => {
                let chars: Vec<char> = chat.input.chars().collect();
                let mut i = chat.cursor;
                while i > 0 && chars[i - 1].is_whitespace() {
                    i -= 1;
                }
                while i > 0 && !chars[i - 1].is_whitespace() {
                    i -= 1;
                }
                let (a, b) = (chat.byte_at(i), chat.byte_at(chat.cursor));
                chat.input.drain(a..b);
                chat.cursor = i;
            }
            KeyCode::Char('v') if ctrl || key.modifiers.contains(KeyModifiers::SUPER) => {
                chat.paste_clipboard()
            }
            KeyCode::Backspace => {
                if chat.input.is_empty() && !chat.attachments.is_empty() {
                    chat.attachments.pop();
                } else if chat.cursor > 0 {
                    let (a, b) = (chat.byte_at(chat.cursor - 1), chat.byte_at(chat.cursor));
                    chat.input.drain(a..b);
                    chat.cursor -= 1;
                }
            }
            KeyCode::Delete => {
                if chat.cursor < chat.input.chars().count() {
                    let (a, b) = (chat.byte_at(chat.cursor), chat.byte_at(chat.cursor + 1));
                    chat.input.drain(a..b);
                }
            }
            KeyCode::Char(c) if !ctrl && !alt => {
                let mut buf = [0u8; 4];
                chat.insert_str(c.encode_utf8(&mut buf));
            }
            _ => {}
        }
    }

    pub fn draw_chat(&self, f: &mut Frame) {
        let Some(chat) = self.chat.as_ref() else {
            return;
        };
        let full = f.area();
        let (area, side) = if chat.agent.is_some() && full.width >= 112 {
            let [main, side] =
                Layout::horizontal([Constraint::Min(70), Constraint::Length(38)]).areas(full);
            (main, Some(side))
        } else {
            (full, None)
        };
        let content_w = (area.width.saturating_sub(2) as usize).min(110);
        let input_w = content_w.saturating_sub(6).max(8);
        let rows = wrap_rows(&chat.input, input_w);
        let input_rows = rows.len().clamp(1, 6) as u16;
        let chips_h = u16::from(!chat.attachments.is_empty());
        let [head, mid, status, chips, input, footer] = Layout::vertical([
            Constraint::Length(2),
            Constraint::Min(4),
            Constraint::Length(1),
            Constraint::Length(chips_h),
            Constraint::Length(input_rows + 2),
            Constraint::Length(1),
        ])
        .areas(area);

        let t = self.elapsed();
        let (state, color) = match &chat.phase {
            Phase::Loading(_) => ("LOADING", GOLD),
            Phase::Ready => ("READY", GOOD),
            Phase::Generating => ("THINKING", EMBER),
            Phase::Tools => ("WORKING", EMBER),
            Phase::Failed(_) => ("FAILED", BAD),
        };
        let mut sub = vec![chat.label.clone()];
        if let Some(i) = &chat.info {
            sub.push(format!("ctx {}", i.context));
            if let Some(d) = &i.drafter {
                sub.push(format!("⚡ {}", d.split(" · ").next().unwrap_or("")));
            }
            if i.vision.is_some() {
                sub.push("👁 vision".into());
            }
        }
        let crumb = sub.join(" · ");
        let title = if chat.agent.is_some() {
            "Agents"
        } else {
            "Chat"
        };
        header(f, head, &[title, &crumb], vec![pill(state, color)]);
        if let Some(side) = side {
            self.draw_agent_side(f, side, chat);
        }

        let col = |r: Rect| centered(r, content_w as u16 + 2, r.height);
        let mid = col(mid);
        match &chat.phase {
            Phase::Loading(stage) => self.draw_loading(f, mid, chat, stage, t),
            Phase::Failed(e) => {
                let lines = vec![
                    Line::styled(
                        "✗ could not load the model",
                        Style::new().fg(BAD).add_modifier(Modifier::BOLD),
                    ),
                    Line::default(),
                    Line::styled(e.clone(), Style::new().fg(TEXT)),
                    Line::default(),
                    Line::styled("esc to go back", Style::new().fg(DIM)),
                ];
                f.render_widget(
                    Paragraph::new(lines).wrap(Wrap { trim: false }),
                    centered(mid, mid.width.min(90), 8),
                );
            }
            _ => self.draw_transcript(f, mid, chat, t),
        }

        self.draw_chat_status(f, col(status), chat, t);
        if chips_h > 0 {
            let mut spans = vec![Span::raw(" ")];
            for a in &chat.attachments {
                spans.push(Span::styled(
                    format!(" ⊕ {} ", a.chip()),
                    Style::new().fg(GOLD).bg(SELECTED),
                ));
                spans.push(Span::raw(" "));
            }
            if chat.input.is_empty() {
                spans.push(Span::styled("⌫ removes the last", Style::new().fg(FAINT)));
            }
            f.render_widget(Paragraph::new(Line::from(spans)), col(chips));
        }

        let ready = chat.phase == Phase::Ready;
        let input = col(input);
        let block = Block::bordered()
            .border_type(BorderType::Rounded)
            .border_style(Style::new().fg(if ready { EMBER } else { FAINT }))
            .title(Line::from(vec![
                Span::raw(" "),
                Span::styled(
                    if chat.generating() {
                        "generating · esc to stop"
                    } else {
                        "message"
                    },
                    Style::new()
                        .fg(if ready { GOLD } else { DIM })
                        .add_modifier(Modifier::BOLD),
                ),
                Span::raw(" "),
            ]));
        let inner = block.inner(input);
        f.render_widget(block, input);
        let chars: Vec<char> = chat.input.chars().collect();
        let inner_w = inner.width.saturating_sub(3) as usize;
        let rows = wrap_rows(&chat.input, inner_w.max(4));
        let cursor_row = rows
            .iter()
            .rposition(|&(s, _)| s <= chat.cursor)
            .unwrap_or(0);
        let first = (cursor_row + 1).saturating_sub(input_rows as usize);
        let mut lines: Vec<Line> = Vec::new();
        if chat.input.is_empty() {
            lines.push(Line::from(vec![
                Span::styled(" ❯ ", Style::new().fg(EMBER)),
                Span::styled(
                    if matches!(chat.phase, Phase::Loading(_)) {
                        "waiting for the model…"
                    } else {
                        "ask anything · / for commands"
                    },
                    Style::new().fg(FAINT),
                ),
            ]));
        } else {
            for (n, &(s, e)) in rows
                .iter()
                .enumerate()
                .skip(first)
                .take(input_rows as usize)
            {
                let text: String = chars[s..e].iter().collect();
                lines.push(Line::from(vec![
                    Span::styled(if n == 0 { " ❯ " } else { "   " }, Style::new().fg(EMBER)),
                    Span::styled(text, Style::new().fg(Color::White)),
                ]));
            }
        }
        f.render_widget(Paragraph::new(lines), inner);
        if ready || chat.generating() {
            let (s, _) = rows[cursor_row];
            let before: String = chars[s..chat.cursor.min(chars.len())].iter().collect();
            let x = inner.x + 3 + text_width(&before) as u16;
            let y = inner.y + (cursor_row - first) as u16;
            f.set_cursor_position((x.min(inner.x + inner.width.saturating_sub(1)), y));
        }

        let items = chat.palette();
        if !items.is_empty() {
            let n = items.len().min(7);
            let sel = chat.palette_sel.min(items.len() - 1);
            let h = (n as u16 + 2).min(input.y.saturating_sub(area.y));
            let rect = Rect::new(input.x, input.y - h, input.width.min(70), h);
            f.render_widget(Clear, rect);
            let block = panel("commands", true);
            let inner = block.inner(rect);
            f.render_widget(block, rect);
            let typed = chat.input.chars().count();
            let top = (sel + 1).saturating_sub(inner.height as usize);
            let lines: Vec<Line> = items
                .iter()
                .enumerate()
                .skip(top)
                .take(inner.height as usize)
                .map(|(i, (name, what))| {
                    let on = i == sel;
                    let bg = if on { SELECTED } else { Color::Reset };
                    let hit = name.to_lowercase().starts_with(&chat.input.to_lowercase());
                    let split = if hit {
                        typed.min(name.chars().count())
                    } else {
                        0
                    };
                    let (head, tail): (String, String) = (
                        name.chars().take(split).collect(),
                        name.chars().skip(split).collect(),
                    );
                    let pad = " ".repeat(14usize.saturating_sub(name.chars().count()));
                    Line::from(vec![
                        Span::styled(
                            if on { " ▌ " } else { "   " },
                            Style::new().fg(EMBER).bg(bg),
                        ),
                        Span::styled(
                            head,
                            Style::new().fg(GOLD).bg(bg).add_modifier(Modifier::BOLD),
                        ),
                        Span::styled(
                            tail,
                            Style::new().fg(if on { Color::White } else { TEXT }).bg(bg),
                        ),
                        Span::styled(pad, Style::new().bg(bg)),
                        Span::styled(
                            format!("{:<w$}", what, w = inner.width.saturating_sub(18) as usize),
                            Style::new().fg(if on { GOLD } else { DIM }).bg(bg),
                        ),
                    ])
                })
                .collect();
            f.render_widget(Paragraph::new(lines), inner);
        }

        if chat.approval.is_some() {
            self.draw_approval(f, input, chat);
        }
        if let Some(list) = &chat.resume {
            self.draw_resume(f, area, list);
        }
        let keys: &[(&str, &str)] = if chat.resume.is_some() {
            &[
                ("↑↓", "choose"),
                ("enter", "resume"),
                ("x", "delete"),
                ("esc", "close"),
            ]
        } else if chat.approval.is_some() {
            &[
                ("y", "allow once"),
                ("a", "allow all this chat"),
                ("n", "deny"),
            ]
        } else if !items.is_empty() {
            &[
                ("↑↓", "choose"),
                ("tab", "complete"),
                ("enter", "run"),
                ("esc", "close"),
            ]
        } else {
            match chat.phase {
                Phase::Generating => &[
                    ("esc", "stop"),
                    ("pgup/pgdn", "scroll"),
                    ("ctrl-t", "thinking"),
                ],
                Phase::Tools => &[("esc", "cancel the tools"), ("pgup/pgdn", "scroll")],
                Phase::Ready => &[
                    ("enter", "send"),
                    ("alt+enter", "new line"),
                    ("pgup/pgdn", "scroll"),
                    ("ctrl-v", "attach clipboard"),
                    ("ctrl-t", "thinking"),
                    ("/help", "commands"),
                    ("esc", "leave"),
                ],
                _ => &[("esc", "back")],
            }
        };
        keys_footer(f, footer, keys, &self.status);
    }

    fn draw_resume(&self, f: &mut Frame, area: Rect, list: &ResumeList) {
        let w = area.width.saturating_sub(6).min(110);
        let h = (list.items.len() as u16 + 4)
            .min(area.height.saturating_sub(4))
            .max(5);
        let rect = centered(area, w, h);
        f.render_widget(Clear, rect);
        let block = panel("Resume a session", true);
        let inner = block.inner(rect);
        f.render_widget(block, rect);
        let rows = inner.height.saturating_sub(1) as usize;
        let top = (list.sel + 1).saturating_sub(rows);
        let width = inner.width as usize;
        let mut lines: Vec<Line> = list
            .items
            .iter()
            .enumerate()
            .skip(top)
            .take(rows)
            .map(|(i, m)| {
                let on = i == list.sel;
                let bg = if on { SELECTED } else { Color::Reset };
                let tail = format!("{} msgs · {}", m.turns, crate::benchui::ago(m.updated));
                let model = clip(&m.model, 24);
                let title_w =
                    width.saturating_sub(tail.chars().count() + model.chars().count() + 9);
                Line::from(vec![
                    Span::styled(
                        if on { " ▌ " } else { "   " },
                        Style::new().fg(EMBER).bg(bg),
                    ),
                    Span::styled(
                        format!("{:<title_w$}", clip(&m.title, title_w)),
                        Style::new().fg(if on { Color::White } else { TEXT }).bg(bg),
                    ),
                    Span::styled(format!("  {model}"), Style::new().fg(FAINT).bg(bg)),
                    Span::styled(
                        format!("  {tail} "),
                        Style::new().fg(if on { GOLD } else { DIM }).bg(bg),
                    ),
                ])
            })
            .collect();
        lines.push(Line::styled(
            " resuming replaces this conversation; it stays saved",
            Style::new().fg(FAINT),
        ));
        f.render_widget(Paragraph::new(lines), inner);
    }

    fn draw_approval(&self, f: &mut Frame, input: Rect, chat: &Chat) {
        let Some((_, _, name, args)) = &chat.approval else {
            return;
        };
        let s = |k: &str| args.get(k).and_then(Json::as_str).unwrap_or("");
        let width = input.width.min(100);
        let inner_w = width.saturating_sub(4) as usize;
        let mut lines: Vec<Line> = vec![Line::from(vec![
            Span::styled(" ⚙ ", Style::new().fg(EMBER)),
            Span::styled(
                name.clone(),
                Style::new().fg(Color::White).add_modifier(Modifier::BOLD),
            ),
            Span::styled(
                match name.as_str() {
                    "bash" => "  wants to run a command in the sandbox",
                    "write_file" => "  wants to write a file",
                    "edit_file" => "  wants to edit a file",
                    "fetch_url" => "  wants to download",
                    "web_search" => "  wants to search the web",
                    _ => "  wants to run",
                },
                Style::new().fg(DIM),
            ),
        ])];
        let mut add = |text: &str, style: Style, max: usize| {
            let rows = wrap_rows(text, inner_w);
            let chars: Vec<char> = text.chars().collect();
            for (n, &(a, b)) in rows.iter().enumerate() {
                if n == max {
                    lines.push(Line::styled("   …", Style::new().fg(FAINT)));
                    break;
                }
                lines.push(Line::styled(
                    format!("   {}", chars[a..b].iter().collect::<String>()),
                    style,
                ));
            }
        };
        match name.as_str() {
            "bash" => add(s("command"), Style::new().fg(GOLD), 6),
            "fetch_url" => add(s("url"), Style::new().fg(GOLD), 2),
            "web_search" => add(s("query"), Style::new().fg(GOLD), 2),
            "write_file" => {
                add(
                    &format!("{} ({} lines)", s("path"), s("content").lines().count()),
                    Style::new().fg(GOLD),
                    1,
                );
                let head: Vec<&str> = s("content").lines().take(5).collect();
                add(&head.join("\n"), Style::new().fg(DIM), 5);
            }
            "edit_file" => {
                add(s("path"), Style::new().fg(GOLD), 1);
                add(&format!("- {}", s("old_text")), Style::new().fg(BAD), 3);
                add(&format!("+ {}", s("new_text")), Style::new().fg(GOOD), 3);
            }
            _ => add(&agent::summary(name, args), Style::new().fg(GOLD), 2),
        }
        let h = (lines.len() as u16 + 2).min(input.y.saturating_sub(2));
        let rect = Rect::new(input.x, input.y.saturating_sub(h), width, h);
        f.render_widget(Clear, rect);
        let block = panel("allow this tool call?", true);
        let inner = block.inner(rect);
        f.render_widget(block, rect);
        f.render_widget(Paragraph::new(lines), inner);
    }

    fn draw_loading(&self, f: &mut Frame, area: Rect, chat: &Chat, stage: &str, t: f32) {
        let secs = chat.since.elapsed().as_secs_f32();
        let w = 66u16;
        let rect = centered(area, w, 7);
        let pos = ((t * 1.1).fract() * 2.0 - 0.5) * w as f32;
        let mut bar = String::new();
        for i in 0..w {
            let d = (i as f32 - pos).abs();
            bar.push(if d < 5.0 { '━' } else { '─' });
        }
        let mut bar_spans = Vec::new();
        for (i, c) in bar.chars().enumerate() {
            let d = (i as f32 - pos).abs();
            let k = (1.0 - d / 5.0).clamp(0.0, 1.0);
            bar_spans.push(Span::styled(
                c.to_string(),
                Style::new().fg(lerp((60, 40, 32), (255, 150, 70), k)),
            ));
        }
        let lines = vec![
            Line::from(vec![
                Span::styled(format!("{} ", spinner(t)), Style::new().fg(EMBER)),
                Span::styled(
                    stage.to_string(),
                    Style::new().fg(Color::White).add_modifier(Modifier::BOLD),
                ),
            ]),
            Line::default(),
            Line::from(bar_spans),
            Line::default(),
            Line::styled(chat.label.clone(), Style::new().fg(TEXT)),
            Line::styled(
                format!("{secs:.0}s · the first load pages the weights into memory"),
                Style::new().fg(DIM),
            ),
        ];
        f.render_widget(Paragraph::new(lines).alignment(Alignment::Center), rect);
    }

    fn draw_chat_status(&self, f: &mut Frame, area: Rect, chat: &Chat, t: f32) {
        let mut left: Vec<Span> = Vec::new();
        match &chat.phase {
            Phase::Generating => {
                left.push(Span::styled(
                    format!(" {} ", spinner(t)),
                    Style::new().fg(EMBER),
                ));
                if let Some((done, total)) = chat.prefill {
                    left.push(Span::styled(
                        format!("reading prompt {done} / {total}"),
                        Style::new().fg(TEXT),
                    ));
                } else {
                    let tps = chat
                        .first_token
                        .map(|s| chat.gen_tokens as f64 / s.elapsed().as_secs_f64().max(0.05))
                        .unwrap_or(0.0);
                    left.push(Span::styled(
                        format!("{} tokens · {tps:.1} tok/s", chat.gen_tokens),
                        Style::new().fg(TEXT),
                    ));
                }
            }
            Phase::Tools => {
                left.push(Span::styled(
                    format!(" {} ", spinner(t)),
                    Style::new().fg(EMBER),
                ));
                left.push(Span::styled(
                    if chat.approval.is_some() {
                        "waiting for your approval".to_string()
                    } else {
                        "running tools".to_string()
                    },
                    Style::new().fg(TEXT),
                ));
            }
            _ => {}
        }
        if let Some((text, ok)) = chat.flash_text() {
            left.push(Span::styled(
                format!("  {text}"),
                Style::new().fg(if ok { GOOD } else { BAD }),
            ));
        }
        f.render_widget(Paragraph::new(Line::from(left)), area);
        let mut right = vec![Span::styled(
            format!(
                "{}think {} ",
                if chat.tools == ToolMode::Off {
                    String::new()
                } else {
                    format!("tools {} · ", chat.tools.label())
                },
                if chat.think { "on" } else { "off" }
            ),
            Style::new().fg(DIM),
        )];
        if let Some((used, cap)) = chat.ctx {
            right.insert(
                0,
                Span::styled(format!("ctx {used}/{cap} · "), Style::new().fg(DIM)),
            );
        }
        f.render_widget(
            Paragraph::new(Line::from(right)).alignment(Alignment::Right),
            area,
        );
    }

    fn draw_transcript(&self, f: &mut Frame, area: Rect, chat: &Chat, t: f32) {
        if chat.turns.is_empty() {
            let mut lines = vec![
                Line::from(vec![
                    Span::styled("◆ ", Style::new().fg(EMBER)),
                    Span::styled(
                        chat.label.clone(),
                        Style::new().fg(Color::White).add_modifier(Modifier::BOLD),
                    ),
                ]),
                Line::default(),
            ];
            if let Some(i) = &chat.info {
                lines.push(Line::styled(
                    format!(
                        "{:.1} GiB · context {} (up to {}) · loaded in {:.1}s",
                        i.weights_gib, i.context, i.max_context, i.load_secs
                    ),
                    Style::new().fg(DIM),
                ));
                if let Some(d) = &i.drafter {
                    lines.push(Line::styled(format!("⚡ {d}"), Style::new().fg(GOLD)));
                }
                if let Some(v) = &i.vision {
                    lines.push(Line::styled(
                        format!("👁 images are shown to the model · {v}"),
                        Style::new().fg(GOLD),
                    ));
                }
                lines.push(Line::styled(i.sampling.clone(), Style::new().fg(DIM)));
            }
            lines.push(Line::default());
            if chat.agent.is_some() {
                for tip in [
                    "describe a task: the agent explores, edits and tests in your project",
                    "/plan on to plan first · /review to check your diff for bloat",
                ] {
                    lines.push(Line::styled(tip, Style::new().fg(TEXT)));
                }
            } else {
                lines.push(Line::styled(
                    "type a message below to begin",
                    Style::new().fg(TEXT),
                ));
            }
            lines.push(Line::styled(
                "/help for commands · alt+enter for a new line",
                Style::new().fg(FAINT),
            ));
            f.render_widget(
                Paragraph::new(lines).alignment(Alignment::Center),
                centered(area, area.width, 9),
            );
            return;
        }
        let width = area.width.saturating_sub(3) as usize;
        let lines = self.transcript_lines(chat, width, t);
        let h = area.height as usize;
        let max_top = lines.len().saturating_sub(h);
        chat.max_top.set(max_top);
        chat.page.set(h);
        let top = if chat.follow {
            max_top
        } else {
            chat.top.get().min(max_top)
        };
        chat.top.set(top);
        let visible: Vec<Line> = lines.into_iter().skip(top).take(h).collect();
        let text_area = Rect::new(area.x, area.y, area.width.saturating_sub(1), area.height);
        f.render_widget(Paragraph::new(visible), text_area);
        if max_top > 0 {
            let mut state = ratatui::widgets::ScrollbarState::new(max_top + 1)
                .position(top)
                .viewport_content_length(h);
            f.render_stateful_widget(
                ratatui::widgets::Scrollbar::new(
                    ratatui::widgets::ScrollbarOrientation::VerticalRight,
                )
                .track_symbol(Some("│"))
                .thumb_symbol("┃")
                .begin_symbol(None)
                .end_symbol(None)
                .track_style(Style::new().fg(FAINT))
                .thumb_style(Style::new().fg(EMBER)),
                area,
                &mut state,
            );
        }
    }

    fn transcript_lines(&self, chat: &Chat, width: usize, t: f32) -> Vec<Line<'static>> {
        let mut out: Vec<Line<'static>> = Vec::new();
        let indent = || Span::raw("  ");
        for (n, turn) in chat.turns.iter().enumerate() {
            let prev = n.checked_sub(1).map(|p| chat.turns[p].role);
            if n > 0 && !(turn.role == Role::Tool && prev == Some(Role::Tool)) {
                out.push(Line::default());
            }
            match turn.role {
                Role::Tool => {
                    let Some(tool) = &turn.tool else { continue };
                    let (icon, color) = match tool.state {
                        ToolState::Queued => ("·".to_string(), DIM),
                        ToolState::Waiting => ("?".to_string(), GOLD),
                        ToolState::Running => (spinner(t).to_string(), EMBER),
                        ToolState::Ok => ("✓".to_string(), GOOD),
                        ToolState::Failed => ("✗".to_string(), BAD),
                        ToolState::Denied => ("⊘".to_string(), DIM),
                    };
                    let tail = match tool.state {
                        ToolState::Running => tool
                            .started
                            .map(|s| format!("{:.0}s", s.elapsed().as_secs_f32()))
                            .unwrap_or_default(),
                        ToolState::Ok | ToolState::Failed => format!("{:.1}s", tool.secs),
                        ToolState::Waiting => "waiting for you".to_string(),
                        ToolState::Denied => "denied".to_string(),
                        ToolState::Queued => String::new(),
                    };
                    let room =
                        width.saturating_sub(tool.name.chars().count() + tail.chars().count() + 8);
                    out.push(Line::from(vec![
                        indent(),
                        Span::styled(format!("{icon} "), Style::new().fg(color)),
                        Span::styled(
                            tool.name.clone(),
                            Style::new().fg(Color::White).add_modifier(Modifier::BOLD),
                        ),
                        Span::styled(
                            format!("  {}", clip(&tool.summary, room)),
                            Style::new().fg(DIM),
                        ),
                        Span::styled(format!("  {tail}"), Style::new().fg(FAINT)),
                    ]));
                    let diff_card = tool.state == ToolState::Ok
                        && matches!(tool.name.as_str(), "edit_file" | "write_file");
                    if diff_card {
                        let s = |k: &str| tool.args.get(k).and_then(Json::as_str).unwrap_or("");
                        let (minus, plus) = if tool.name == "edit_file" {
                            (s("old_text"), s("new_text"))
                        } else {
                            ("", s("content"))
                        };
                        let bar = |text: &str, sign: char, color: Color| {
                            let style = Style::new().fg(color);
                            let chars: Vec<Sc> = format!("{sign} {text}")
                                .chars()
                                .map(|c| (c, style))
                                .collect();
                            let row = wrap_styled(&chars, width.saturating_sub(6), 0)
                                .into_iter()
                                .next()
                                .unwrap_or_default();
                            to_line(&row, vec![Span::styled("  │ ", Style::new().fg(FAINT))])
                        };
                        for (text, sign, color) in [(minus, '-', BAD), (plus, '+', GOOD)] {
                            let all: Vec<&str> = text.lines().collect();
                            for line in all.iter().take(6) {
                                out.push(bar(line, sign, color));
                            }
                            if all.len() > 6 {
                                out.push(Line::styled(
                                    format!("  │ {sign} … {} more lines", all.len() - 6),
                                    Style::new().fg(FAINT),
                                ));
                            }
                        }
                    } else if matches!(tool.state, ToolState::Ok | ToolState::Failed)
                        && !tool.output.is_empty()
                    {
                        let style = Style::new().fg(if tool.state == ToolState::Failed {
                            BAD
                        } else {
                            DIM
                        });
                        let lines: Vec<&str> = tool.output.lines().collect();
                        for line in lines.iter().take(8) {
                            let line = line.replace('\t', "  ");
                            let chars: Vec<Sc> = line.chars().map(|c| (c, style)).collect();
                            let row = wrap_styled(&chars, width.saturating_sub(6), 0)
                                .into_iter()
                                .next()
                                .unwrap_or_default();
                            out.push(to_line(
                                &row,
                                vec![Span::styled("  │ ", Style::new().fg(FAINT))],
                            ));
                        }
                        if lines.len() > 8 {
                            out.push(Line::styled(
                                format!("  │ … {} more lines", lines.len() - 8),
                                Style::new().fg(FAINT),
                            ));
                        }
                    }
                }
                Role::User => {
                    let style = Style::new().fg(Color::White);
                    for line in turn.content.split('\n') {
                        let chars: Vec<Sc> = line.chars().map(|c| (c, style)).collect();
                        for row in wrap_styled(&chars, width.saturating_sub(2), 0) {
                            out.push(to_line(
                                &row,
                                vec![Span::styled("▌ ", Style::new().fg(GOLD))],
                            ));
                        }
                    }
                    for chip in &turn.attachments {
                        out.push(Line::from(vec![
                            Span::styled("▌ ", Style::new().fg(GOLD)),
                            Span::styled(format!("⊕ {chip}"), Style::new().fg(GOLD)),
                        ]));
                    }
                }
                Role::Note => {
                    let style = Style::new().fg(DIM).add_modifier(Modifier::ITALIC);
                    for line in turn.content.split('\n') {
                        let chars: Vec<Sc> = line.chars().map(|c| (c, style)).collect();
                        for row in wrap_styled(&chars, width.saturating_sub(2), 2) {
                            out.push(to_line(&row, vec![indent()]));
                        }
                    }
                }
                Role::Assistant => {
                    if prev != Some(Role::Tool) {
                        out.push(Line::from(vec![
                            Span::styled("◆ ", Style::new().fg(EMBER)),
                            Span::styled(
                                chat.label.clone(),
                                Style::new().fg(DIM).add_modifier(Modifier::BOLD),
                            ),
                        ]));
                    }
                    let words = turn.reasoning.split_whitespace().count();
                    let thinking_now = !turn.done && turn.content.is_empty();
                    if !turn.reasoning.is_empty() {
                        if chat.show_thinking {
                            let style = Style::new().fg(DIM).add_modifier(Modifier::ITALIC);
                            out.push(Line::styled(
                                format!(
                                    "{}┊ thinking{}",
                                    "  ",
                                    if thinking_now { " …" } else { "" }
                                ),
                                Style::new().fg(FAINT),
                            ));
                            for line in turn.reasoning.trim_end().split('\n') {
                                let chars: Vec<Sc> = line.chars().map(|c| (c, style)).collect();
                                for row in wrap_styled(&chars, width.saturating_sub(4), 0) {
                                    out.push(to_line(
                                        &row,
                                        vec![Span::styled("  ┊ ", Style::new().fg(FAINT))],
                                    ));
                                }
                            }
                        } else if thinking_now {
                            out.push(Line::from(vec![
                                indent(),
                                Span::styled(spinner(t).to_string(), Style::new().fg(EMBER)),
                                Span::styled(
                                    format!(" thinking · {words} words"),
                                    Style::new().fg(DIM).add_modifier(Modifier::ITALIC),
                                ),
                            ]));
                        } else {
                            out.push(Line::from(vec![
                                indent(),
                                Span::styled(
                                    format!("▸ thought for {words} words · ctrl-t to show"),
                                    Style::new().fg(FAINT),
                                ),
                            ]));
                        }
                        if !turn.content.is_empty() {
                            out.push(Line::default());
                        }
                    }
                    if !turn.content.is_empty() {
                        let mut text = turn.content.clone();
                        if !turn.done {
                            text.push('▍');
                        }
                        for row in markdown(&text, width.saturating_sub(2), Style::new().fg(TEXT)) {
                            out.push(to_line(&row, vec![indent()]));
                        }
                    } else if turn.reasoning.is_empty() && !turn.done {
                        let msg = match chat.prefill {
                            Some((d, tot)) => format!("reading prompt {d} / {tot}"),
                            None => "starting".to_string(),
                        };
                        out.push(Line::from(vec![
                            indent(),
                            Span::styled(spinner(t).to_string(), Style::new().fg(EMBER)),
                            Span::styled(format!(" {msg}"), Style::new().fg(DIM)),
                        ]));
                    }
                    if let Some(e) = &turn.error {
                        out.push(Line::from(vec![
                            indent(),
                            Span::styled(format!("✗ {e}"), Style::new().fg(BAD)),
                        ]));
                    }
                    if let Some(s) = &turn.stats {
                        out.push(Line::from(vec![
                            indent(),
                            Span::styled(s.clone(), Style::new().fg(FAINT)),
                        ]));
                    }
                }
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn each_session_gets_its_own_folder() {
        let root = std::env::temp_dir().join(format!("ferrum-sessions-{}", std::process::id()));
        let root_s = root.display().to_string();
        let a = session_dir(&root_s, "Qwen3.8-27B-UD-Q4_K_XL");
        assert!(a.starts_with(&root));
        let name = a.file_name().unwrap().to_string_lossy().to_string();
        assert!(name.ends_with("qwen3-8-27b-ud-q4-k-xl"), "{name}");
        assert!(
            name.chars().next().unwrap().is_ascii_digit(),
            "starts with the date: {name}"
        );
        // A folder that already exists is never reused, even in the same second.
        fs::create_dir_all(&a).unwrap();
        let b = session_dir(&root_s, "Qwen3.8-27B-UD-Q4_K_XL");
        assert_ne!(a, b);
        assert!(!b.exists());
        let _ = fs::remove_dir_all(root);
    }
}
