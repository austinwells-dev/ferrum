//! Agent benchmark: ten real bug-fix pull requests from small, dependency-free
//! Python and Rust repos. For each, the coding agent's own tools (the ones the
//! Agents tab uses, in its Seatbelt sandbox with the network off) work in a
//! fresh checkout of the commit before the fix, given only a description of
//! the bug. Afterwards the PR's tests (or a test written from them) run on the
//! result. Solved tasks and the tokens spent on them are what get compared.
use crate::agent::Sandbox;
use crate::bench::{self, Candidate, Source};
use crate::brain::{self, Ponytail};
use crate::coding;
use crate::evalsets::{InstallEvent, Installing};
use crate::*;
use ferrum::hybrid::{
    runtime::{ChatHooks, ChatRequest, Delta, Runtime},
    session::StopReason,
};
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
    mpsc::{Receiver, Sender, channel},
};
use std::thread::JoinHandle;

/// The tasks, each checked to fail before its fix and pass after it.
pub const TASKS: &str = include_str!("eval/agent_tasks.json");
/// Most model replies (rounds of tool calls) per task.
pub const MAX_TURNS: usize = 50;
/// Most wall-clock time per task.
pub const TASK_LIMIT: Duration = Duration::from_secs(30 * 60);
/// How long the tests may take, building included.
const TEST_LIMIT: Duration = Duration::from_secs(10 * 60);
/// Most tokens per reply, before the thinking budget.
const REPLY_TOKENS: usize = 4096;
/// Context the run asks for, and the fallback when that does not fit.
pub const CONTEXT: usize = 65_536;
const CONTEXT_FALLBACK: usize = 32_768;
/// Assistant replies, counted from the newest, whose thinking stays in the
/// prompt (see `prompt_messages`).
const KEEP_REASONING: usize = 1;

#[derive(Clone, Debug, PartialEq)]
pub struct Task {
    pub id: String,
    pub lang: String,
    pub repo: String,
    pub pr: u64,
    /// The commit the agent starts from (the fix's parent).
    pub base: String,
    pub fix: String,
    /// Test files taken from the fix commit.
    pub tests: Vec<String>,
    /// Test files written for the task, as (path, content).
    pub hidden: Vec<(String, String)>,
    /// Run once in the checkout at install, with the network (`cargo fetch`).
    pub setup: Option<String>,
    /// Decides the task: run in the sandbox after the agent finishes.
    pub test: String,
    pub issue: String,
}

impl Task {
    /// The test command the agent is told about (hidden test files do not
    /// exist yet, so Rust tasks with one get the plain `cargo test`).
    fn hint(&self) -> &str {
        if self.hidden.is_empty() {
            &self.test
        } else {
            "cargo test --offline"
        }
    }

    /// "owner/repo#123"
    pub fn source(&self) -> String {
        format!(
            "{}#{}",
            self.repo.trim_start_matches("https://github.com/"),
            self.pr
        )
    }
}

pub fn parse_tasks(text: &str) -> Result<Vec<Task>, String> {
    let v: Json = serde_json::from_str(text).map_err(|e| e.to_string())?;
    v.as_array()
        .ok_or("tasks: not a list")?
        .iter()
        .map(|t| {
            let s = |k: &str| {
                t[k].as_str()
                    .map(str::to_string)
                    .ok_or_else(|| format!("task without {k}"))
            };
            Ok(Task {
                id: s("id")?,
                lang: s("lang")?,
                repo: s("repo")?,
                pr: t["pr"].as_u64().unwrap_or(0),
                base: s("base")?,
                fix: s("fix")?,
                tests: t["tests"]
                    .as_array()
                    .map(|a| {
                        a.iter()
                            .filter_map(|p| p.as_str().map(str::to_string))
                            .collect()
                    })
                    .unwrap_or_default(),
                hidden: t["hidden"]
                    .as_object()
                    .map(|o| {
                        o.iter()
                            .filter_map(|(p, c)| Some((p.clone(), c.as_str()?.to_string())))
                            .collect()
                    })
                    .unwrap_or_default(),
                setup: t["setup"].as_str().map(str::to_string),
                test: s("test")?,
                issue: s("issue")?,
            })
        })
        .collect()
}

pub fn tasks() -> Vec<Task> {
    parse_tasks(TASKS).unwrap_or_default()
}

/// Changes whenever the bundled tasks do.
pub fn tasks_fingerprint() -> String {
    format!("{:016x}", crate::evalsets::eval_hash(TASKS))
}

// ---- install ----

/// `$FERRUM_AGENT_DIR`, else `~/.config/ferrum/agent`.
pub fn tasks_dir() -> PathBuf {
    std::env::var_os("FERRUM_AGENT_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| home().join(".config/ferrum/agent"))
}

/// How many of `tasks` are installed in `dir`.
pub fn installed(dir: &Path, tasks: &[Task]) -> usize {
    tasks
        .iter()
        .filter(|t| dir.join(&t.id).join("ready").exists())
        .count()
}

fn git(args: &[&str], cwd: &Path) -> Result<Vec<u8>, String> {
    // No background gc or maintenance: it keeps writing into `.git` after
    // the command returns, and the folder may be deleted right away.
    let out = Command::new("/usr/bin/git")
        .args(["-c", "gc.auto=0", "-c", "maintenance.auto=false"])
        .args(args)
        .current_dir(cwd)
        .output()
        .map_err(|e| format!("could not run git: {e}"))?;
    if out.status.success() {
        Ok(out.stdout)
    } else {
        Err(format!(
            "git {}: {}",
            args.first().unwrap_or(&""),
            String::from_utf8_lossy(&out.stderr).trim()
        ))
    }
}

/// Delete a folder, retrying briefly: something (Spotlight, a lingering
/// child process) may still be writing into it.
fn remove_tree(dir: &Path) -> Result<(), String> {
    let mut last = String::new();
    for _ in 0..10 {
        match fs::remove_dir_all(dir) {
            Ok(()) => return Ok(()),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(()),
            Err(e) => last = e.to_string(),
        }
        std::thread::sleep(Duration::from_millis(300));
    }
    Err(format!("could not delete {}: {last}", dir.display()))
}

/// A checkout of `url` at `sha` alone: no other commit is fetched, so the
/// fix cannot be found in its history.
fn checkout(url: &str, sha: &str, dir: &Path) -> Result<(), String> {
    fs::create_dir_all(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
    git(&["init", "-q"], dir)?;
    git(
        &[
            "fetch",
            "-q",
            "--no-auto-maintenance",
            "--depth",
            "1",
            url,
            sha,
        ],
        dir,
    )?;
    git(&["checkout", "-q", "FETCH_HEAD"], dir)?;
    git(&["config", "user.name", "Ferrum Agent"], dir)?;
    git(&["config", "user.email", "agent@ferrum.invalid"], dir)?;
    Ok(())
}

fn install_task(t: &Task, dir: &Path, progress: &impl Fn(String)) -> Result<(), String> {
    remove_tree(dir)?;
    progress(format!("{} · fetching {}", t.id, t.source()));
    checkout(&t.repo, &t.base, &dir.join("repo"))?;
    let hidden = dir.join("hidden");
    if !t.tests.is_empty() {
        progress(format!("{} · fetching the fix's tests", t.id));
        let fix = dir.join(".fix");
        checkout(&t.repo, &t.fix, &fix)?;
        for path in &t.tests {
            let text = git(&["show", &format!("HEAD:{path}")], &fix)?;
            let to = hidden.join(path);
            fs::create_dir_all(to.parent().unwrap_or(&hidden)).map_err(|e| e.to_string())?;
            fs::write(&to, text).map_err(|e| format!("{}: {e}", to.display()))?;
        }
        remove_tree(&fix)?;
    }
    for (path, text) in &t.hidden {
        let to = hidden.join(path);
        fs::create_dir_all(to.parent().unwrap_or(&hidden)).map_err(|e| e.to_string())?;
        fs::write(&to, text).map_err(|e| format!("{}: {e}", to.display()))?;
    }
    if let Some(setup) = &t.setup {
        progress(format!("{} · {setup}", t.id));
        let out = Command::new("/bin/bash")
            .args(["-c", setup])
            .current_dir(dir.join("repo"))
            .output()
            .map_err(|e| format!("{setup}: {e}"))?;
        if !out.status.success() {
            return Err(format!(
                "{}: {setup} failed: {}",
                t.id,
                String::from_utf8_lossy(&out.stderr).trim()
            ));
        }
    }
    fs::write(dir.join("ready"), "").map_err(|e| e.to_string())
}

/// Fetches every task not yet installed; returns how many are ready.
pub fn install(dir: &Path, progress: impl Fn(String), stop: &AtomicBool) -> Result<usize, String> {
    let all = tasks();
    for t in &all {
        if stop.load(Ordering::SeqCst) {
            return Err("stopped".into());
        }
        let td = dir.join(&t.id);
        if !td.join("ready").exists() {
            install_task(t, &td, &progress).map_err(|e| format!("{}: {e}", t.id))?;
        }
    }
    Ok(installed(dir, &all))
}

pub fn remove(dir: &Path) -> Result<(), String> {
    remove_tree(dir)
}

pub fn spawn_install(dir: PathBuf) -> Result<Installing, String> {
    let (tx, rx) = channel();
    let stop = Arc::new(AtomicBool::new(false));
    let flag = stop.clone();
    let handle = std::thread::Builder::new()
        .name("ferrum-agent-install".into())
        .spawn(move || {
            let progress = |s| {
                let _ = tx.send(InstallEvent::Progress(s));
            };
            let result = install(&dir, progress, &flag);
            let _ = tx.send(InstallEvent::Done(result));
        })
        .map_err(|e| e.to_string())?;
    Ok(Installing {
        rx,
        stop,
        handle: Some(handle),
    })
}

// ---- running ----

#[derive(Clone)]
pub struct AgentPlan {
    pub model: PathBuf,
    /// Reasoning budget per reply (None: thinking off).
    pub thinking: Option<usize>,
    pub draft: Option<(Candidate, usize)>,
    pub kv_cache: KvCache,
}

impl AgentPlan {
    pub fn draft_label(&self) -> String {
        match &self.draft {
            Some((c, d)) => format!("{} · depth {d} · {}", c.label(), c.quant_label()),
            None => "plain decoding".into(),
        }
    }
}

/// How one task went.
#[derive(Clone, Debug, PartialEq)]
pub struct TaskResult {
    pub id: String,
    pub solved: bool,
    /// Model replies.
    pub turns: usize,
    /// Tokens the model wrote (thinking included).
    pub generated: usize,
    /// Prompt tokens it had to read (not counting ones reused from the cache).
    pub prompt: usize,
    pub secs: f64,
    /// Why the agent stopped: finished, turn limit, time limit, context full
    /// or an error.
    pub end: String,
    /// passed, failed or timed out.
    pub test: String,
    /// The saved transcript (conversation, changes, test output).
    pub log: Option<String>,
}

impl TaskResult {
    pub fn tokens(&self) -> usize {
        self.generated + self.prompt
    }

    fn to_json(&self) -> Json {
        json!({
            "id": self.id, "solved": self.solved, "turns": self.turns,
            "generated": self.generated, "prompt": self.prompt, "secs": self.secs,
            "end": self.end, "test": self.test, "log": self.log,
        })
    }
}

pub enum AgentEvent {
    Stage(String),
    Loaded {
        context: usize,
        weights_gib: f64,
    },
    /// After each reply of task `task` (an index into the run's tasks).
    Turn {
        task: usize,
        turn: usize,
        generated: usize,
        prompt: usize,
        last: String,
    },
    TaskDone(TaskResult),
    /// Task `task` (an index into the run's tasks) begins.
    TaskStart(usize),
    /// Text of the reply being written, as it streams.
    Delta {
        reasoning: bool,
        text: String,
    },
    /// A tool call ran: what it was, whether it worked, the start of its output.
    Tool {
        summary: String,
        ok: bool,
        output: String,
    },
    Failed(String),
    Done,
}

pub struct AgentRunning {
    pub rx: Receiver<AgentEvent>,
    pub stop: Arc<AtomicBool>,
    pub handle: Option<JoinHandle<()>>,
}

pub fn spawn(plan: AgentPlan, tasks: Vec<Task>) -> Result<AgentRunning, String> {
    let (tx, rx) = channel();
    let stop = Arc::new(AtomicBool::new(false));
    let flag = stop.clone();
    let handle = std::thread::Builder::new()
        .name("ferrum-agentbench".into())
        .spawn(move || run(plan, tasks, tx, flag))
        .map_err(|e| e.to_string())?;
    Ok(AgentRunning {
        rx,
        stop,
        handle: Some(handle),
    })
}

/// Stops a reply when the run is stopped or the task is out of time.
struct Hooks<'a> {
    stop: &'a AtomicBool,
    deadline: Instant,
    /// Streams the reply to the Bench tab's live feed.
    send: &'a dyn Fn(AgentEvent),
}

impl ChatHooks for Hooks<'_> {
    fn delta(&mut self, delta: Delta) -> ferrum::Result<()> {
        let (reasoning, text) = match delta {
            Delta::Reasoning(t) => (true, t),
            Delta::Content(t) => (false, t),
        };
        (self.send)(AgentEvent::Delta { reasoning, text });
        Ok(())
    }
    fn cancelled(&self) -> bool {
        self.stop.load(Ordering::SeqCst) || Instant::now() >= self.deadline
    }
}

fn load(plan: &AgentPlan, context: usize) -> Result<Runtime, String> {
    let bench_plan = bench::Plan {
        model: plan.model.clone(),
        candidates: Vec::new(),
        quick: true,
        tokens: 0,
        chunks: false,
        context,
        kv_cache: plan.kv_cache,
        trees: false,
    };
    let plain = Candidate {
        source: Source::Baseline,
        quant: "q4_0",
        on: true,
    };
    let (cand, depth) = match &plan.draft {
        Some((c, d)) => (c, Some(*d)),
        None => (&plain, None),
    };
    let mut rt = bench::load(&bench_plan, cand, depth, None)?;
    rt.warmup().map_err(|e| e.to_string())?;
    rt.session.depth_cap = depth;
    rt.session.speculate = depth.is_some();
    Ok(rt)
}

/// Where transcripts go: a folder per run, a file per task.
pub fn runs_dir() -> PathBuf {
    home().join(".config/ferrum/agent-runs")
}

fn run(plan: AgentPlan, tasks: Vec<Task>, tx: Sender<AgentEvent>, stop: Arc<AtomicBool>) {
    let send = |e: AgentEvent| {
        let _ = tx.send(e);
    };
    let stem = plan
        .model
        .file_stem()
        .map(|s| s.to_string_lossy().to_string())
        .unwrap_or_default();
    let run_dir = runs_dir().join(format!("{}-{stem}", bench::now()));
    send(AgentEvent::Stage(format!("loading {}", plan.draft_label())));
    let loaded = load(&plan, CONTEXT).or_else(|_| load(&plan, CONTEXT_FALLBACK));
    let mut rt = match loaded {
        Ok(rt) => rt,
        Err(e) => {
            send(AgentEvent::Failed(e));
            send(AgentEvent::Done);
            return;
        }
    };
    send(AgentEvent::Loaded {
        context: rt.loaded.plan.context,
        weights_gib: rt.loaded.plan.weights as f64 / (1u64 << 30) as f64,
    });
    for (i, task) in tasks.iter().enumerate() {
        if stop.load(Ordering::SeqCst) {
            break;
        }
        send(AgentEvent::Stage(format!(
            "task {}/{} · {}",
            i + 1,
            tasks.len(),
            task.id
        )));
        match attempt(&mut rt, &plan, i, task, &run_dir, &send, &stop) {
            Ok(r) => send(AgentEvent::TaskDone(r)),
            Err(e) => {
                if !stop.load(Ordering::SeqCst) {
                    send(AgentEvent::Failed(format!("{}: {e}", task.id)));
                }
                break;
            }
        }
    }
    send(AgentEvent::Done);
}

/// The first user message: the bug, where the code is, how to test.
fn task_message(task: &Task) -> String {
    format!(
        "{}\n\nThe repository is your project folder. Fix the bug there. The tests run with:\n\n    {}\n\nOnly that command decides whether the bug is fixed: other tests may fail for reasons unrelated to it, such as a missing optional dependency. The network is off.",
        task.issue,
        task.hint()
    )
}

/// The conversation as sent to the model: thinking is kept only on the newest
/// `KEEP_REASONING` assistant replies. Qwen-style templates render the
/// thinking of every assistant turn after the last user message, and in a
/// task that is all of them (there is one user message); some also default
/// `preserve_thinking` to true, which keeps it everywhere. With a 4K
/// thinking budget, ten replies of thinking fill the context before the agent
/// edits anything. A template variable cannot fix this (Qwen templates have
/// none for turns after the user message), so the thinking is dropped from
/// the messages. What the model did and saw stays: its replies, tool calls
/// and their output, and the plan of its latest reply, which it is acting on.
/// That is also what these models see between user turns in training.
fn prompt_messages(messages: &[Json]) -> Vec<Json> {
    let mut kept = 0;
    let mut out: Vec<Json> = messages
        .iter()
        .rev()
        .map(|m| {
            let mut m = m.clone();
            if m["role"] == "assistant" && m.get("reasoning_content").is_some() {
                if kept < KEEP_REASONING {
                    kept += 1;
                } else if let Some(o) = m.as_object_mut() {
                    o.remove("reasoning_content");
                }
            }
            m
        })
        .collect();
    out.reverse();
    out
}

/// Python packages find their plugins through entry points, which exist only
/// once the package is installed, and the task checkouts are not: without
/// them Python-Markdown's suite fails about 390 tests (every extension by
/// name), and agents spend their replies proving those errors are not theirs.
/// Writes the `[project.entry-points]` of `pyproject.toml` into a
/// `.dist-info` folder at the top of the checkout, where `importlib.metadata`
/// finds it (the tests run from there), and hides it from git. Best effort:
/// without it, only the full suite is affected.
fn add_entry_points(work: &Path) {
    if !work.join("pyproject.toml").exists() {
        return;
    }
    let _ = Command::new("python3")
        .args(["-I", "-c", ENTRY_POINTS])
        .current_dir(work)
        .output();
}

const ENTRY_POINTS: &str = r#"
import pathlib, re, tomllib
p = tomllib.load(open("pyproject.toml", "rb")).get("project", {})
groups = p.get("entry-points", {})
if p.get("name") and groups:
    name = re.sub(r"[-_.]+", "_", p["name"])
    d = pathlib.Path(name + "-0.dist-info")
    d.mkdir(exist_ok=True)
    (d / "METADATA").write_text("Metadata-Version: 2.1\nName: %s\nVersion: 0\n" % p["name"])
    (d / "entry_points.txt").write_text("".join(
        "[%s]\n%s\n" % (g, "".join("%s = %s\n" % kv for kv in eps.items()))
        for g, eps in groups.items()))
    pathlib.Path(".git/info").mkdir(parents=True, exist_ok=True)
    with open(".git/info/exclude", "a") as f:
        f.write("/%s/\n" % d)
"#;

/// Copy `from` into a new folder `to`.
fn copy_tree(from: &Path, to: &Path) -> Result<(), String> {
    let out = Command::new("/bin/cp")
        .arg("-R")
        .arg(from)
        .arg(to)
        .output()
        .map_err(|e| e.to_string())?;
    if out.status.success() {
        Ok(())
    } else {
        Err(String::from_utf8_lossy(&out.stderr).trim().to_string())
    }
}

/// One task: the agent loop in a fresh copy of the checkout, then the tests.
fn attempt(
    rt: &mut Runtime,
    plan: &AgentPlan,
    index: usize,
    task: &Task,
    run_dir: &Path,
    send: &impl Fn(AgentEvent),
    stop: &AtomicBool,
) -> Result<TaskResult, String> {
    let source = tasks_dir().join(&task.id);
    let tmp = std::env::temp_dir().join(format!("ferrum-agent-{}-{}", std::process::id(), task.id));
    let _ = fs::remove_dir_all(&tmp);
    fs::create_dir_all(&tmp).map_err(|e| e.to_string())?;
    let work = tmp.join("work");
    copy_tree(&source.join("repo"), &work)?;
    let work = fs::canonicalize(&work).map_err(|e| e.to_string())?;
    if task.lang == "python" {
        add_entry_points(&work);
    }
    let sb = Sandbox::with_state(work.clone(), tmp.join("state"), false)?;
    let shared = coding::new_shared();
    let tools: Vec<Json> = coding::tool_defs(false, false)
        .into_iter()
        .filter(|d| d["function"]["name"] != "ask_user")
        .collect();
    let mut messages = vec![
        json!({"role": "system", "content": brain::system_prompt(&work, Ponytail::Full, false, false, "", None)}),
        json!({"role": "user", "content": task_message(task)}),
    ];
    send(AgentEvent::TaskStart(index));
    let started = Instant::now();
    let deadline = started + TASK_LIMIT;
    let (mut turns, mut generated, mut prompt) = (0, 0, 0);
    let mut end = "turn limit".to_string();
    let cancel = AtomicBool::new(false);
    rt.session.reset();
    let mut vars = Map::new();
    vars.insert(
        "enable_thinking".into(),
        Json::Bool(plan.thinking.is_some()),
    );
    while turns < MAX_TURNS {
        let request = ChatRequest {
            // `messages` keeps all the thinking, for the saved transcript.
            messages: prompt_messages(&messages),
            tools: tools.clone(),
            max_tokens: REPLY_TOKENS + plan.thinking.unwrap_or(0),
            sampling: ferrum::hybrid::session::SamplingParams::default(),
            template_vars: vars.clone(),
            stop: Vec::new(),
            reasoning_budget: plan.thinking,
            raw_reasoning: false,
            image_budget: Default::default(),
        };
        let r = match rt.chat(
            &request,
            Hooks {
                stop,
                deadline,
                send,
            },
        ) {
            Ok(r) => r,
            Err(e) => {
                end = format!("error: {e}");
                break;
            }
        };
        turns += 1;
        let c = &r.completion;
        generated += c.tokens.len();
        prompt += c.prompt_tokens.saturating_sub(c.reused_tokens);
        // Paths inside the workspace read better relative to it.
        let root = work.display().to_string();
        let last = match r.output.tool_calls.first() {
            Some(call) => crate::agent::summary(&call.name, &call.arguments)
                .replace(&format!("{root}/"), "")
                .replace(&root, "."),
            None => "answered".into(),
        };
        send(AgentEvent::Turn {
            task: index,
            turn: turns,
            generated,
            prompt,
            last,
        });
        if stop.load(Ordering::SeqCst) {
            let _ = fs::remove_dir_all(&tmp);
            return Err("stopped".into());
        }
        if Instant::now() >= deadline {
            end = "time limit".into();
            break;
        }
        if c.stop == StopReason::ContextFull {
            end = "context full".into();
            break;
        }
        let mut reply = json!({"role": "assistant", "content": r.output.content});
        if let Some(reasoning) = &r.output.reasoning {
            reply["reasoning_content"] = Json::String(reasoning.clone());
        }
        if r.output.tool_calls.is_empty() {
            messages.push(reply);
            end = "finished".into();
            break;
        }
        let ids: Vec<String> = (0..r.output.tool_calls.len())
            .map(|k| format!("call_{turns}_{k}"))
            .collect();
        reply["tool_calls"] = Json::Array(
            r.output
                .tool_calls
                .iter()
                .zip(&ids)
                .map(|(call, id)| {
                    json!({"id": id, "type": "function", "function": {
                        "name": call.name,
                        "arguments": Json::Object(call.arguments.clone()).to_string(),
                    }})
                })
                .collect(),
        );
        messages.push(reply);
        for (call, id) in r.output.tool_calls.iter().zip(&ids) {
            let looping = shared
                .lock()
                .map(|mut g| g.is_loop(&call.name, &call.arguments))
                .unwrap_or(false);
            let (ok, out) = if looping {
                (
                    false,
                    "error: you already made this exact call twice. Do something different."
                        .to_string(),
                )
            } else {
                coding::execute(&shared, &sb, &call.name, &call.arguments, &cancel)
            };
            let root = work.display().to_string();
            send(AgentEvent::Tool {
                summary: format!(
                    "{} {}",
                    call.name,
                    crate::agent::summary(&call.name, &call.arguments)
                        .replace(&format!("{root}/"), "")
                        .replace(&root, ".")
                ),
                ok: ok && !out.starts_with("error"),
                output: out.lines().take(6).collect::<Vec<_>>().join("\n"),
            });
            messages.push(json!({
                "role": "tool", "tool_call_id": id, "name": call.name, "content": out,
            }));
        }
    }
    // Background processes the agent started end with its state.
    drop(shared);
    // What the agent changed, before the task's tests are added.
    let diff = changes(&work);
    let (test, output) = grade(task, &source, &sb, stop);
    let _ = fs::remove_dir_all(&tmp);
    let mut result = TaskResult {
        id: task.id.clone(),
        solved: test == "passed",
        turns,
        generated,
        prompt,
        secs: started.elapsed().as_secs_f64(),
        end,
        test,
        log: None,
    };
    let log = run_dir.join(format!("{}.json", task.id));
    let saved = fs::create_dir_all(run_dir).and_then(|_| {
        fs::write(
            &log,
            serde_json::to_string_pretty(&json!({
                "task": task.id, "source": task.source(), "test_command": task.test,
                "result": result.to_json(), "messages": messages,
                "changes": diff, "test_output": output,
            }))
            .unwrap_or_default(),
        )
    });
    if saved.is_ok() {
        result.log = Some(log.display().to_string());
    }
    Ok(result)
}

/// The agent's changes: `git diff` of the checkout plus the files it added.
/// Git runs confined: the agent could have planted settings that run programs.
fn changes(work: &Path) -> String {
    let w = work.display().to_string();
    let git = |args: &[&str]| {
        let mut argv = vec!["git", "--no-optional-locks", "-C", w.as_str()];
        argv.extend_from_slice(args);
        crate::agent::confined(&argv)
            .output()
            .map(|o| String::from_utf8_lossy(&o.stdout).into_owned())
            .unwrap_or_default()
    };
    let mut out = git(&["diff"]);
    for file in git(&["ls-files", "--others", "--exclude-standard"]).lines() {
        out.push_str(&format!("new file: {file}\n"));
    }
    out
}

/// Puts the task's tests in the workspace (over any the agent changed) and
/// runs them in the sandbox: ("passed", "failed", "timed out" or an error,
/// the end of their output).
fn grade(task: &Task, source: &Path, sb: &Sandbox, stop: &AtomicBool) -> (String, String) {
    let hidden = source.join("hidden");
    if hidden.exists()
        && let Err(e) = copy_tree(&hidden.join("."), &sb.workspace)
    {
        return (format!("could not add the tests: {e}"), String::new());
    }
    let tail = |out: &str| {
        let chars: Vec<char> = out.chars().collect();
        chars[chars.len().saturating_sub(6000)..]
            .iter()
            .collect::<String>()
    };
    match sb.run(&["/bin/bash", "-c", &task.test], TEST_LIMIT, stop) {
        Ok((Some(0), out, _)) => ("passed".into(), tail(&out)),
        Ok((Some(_), out, _)) => ("failed".into(), tail(&out)),
        Ok((None, out, _)) => ("timed out".into(), tail(&out)),
        Err(e) => (format!("could not run the tests: {e}"), String::new()),
    }
}

// ---- saved results ----

#[derive(Clone, Debug, PartialEq)]
pub struct AgentResult {
    pub model: String,
    pub when: u64,
    pub thinking: Option<usize>,
    pub draft: String,
    pub kv_cache: String,
    /// `tasks_fingerprint()` of the run.
    pub tasks: String,
    pub results: Vec<TaskResult>,
}

/// Totals over a run's tasks.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Totals {
    pub solved: usize,
    pub done: usize,
    pub generated: usize,
    pub prompt: usize,
    pub turns: usize,
    pub secs: f64,
}

impl Totals {
    pub fn of(results: &[TaskResult]) -> Self {
        let mut t = Self::default();
        for r in results {
            t.solved += usize::from(r.solved);
            t.done += 1;
            t.generated += r.generated;
            t.prompt += r.prompt;
            t.turns += r.turns;
            t.secs += r.secs;
        }
        t
    }

    /// Generated tokens per solved task: the efficiency figure (None when
    /// nothing was solved).
    pub fn per_solve(&self) -> Option<usize> {
        (self.solved > 0).then(|| self.generated / self.solved)
    }
}

impl AgentResult {
    pub fn totals(&self) -> Totals {
        Totals::of(&self.results)
    }

    fn same_setup(&self, o: &Self) -> bool {
        (
            &self.model,
            self.thinking,
            &self.draft,
            &self.kv_cache,
            &self.tasks,
        ) == (&o.model, o.thinking, &o.draft, &o.kv_cache, &o.tasks)
    }

    fn to_json(&self) -> Json {
        let results: Vec<Json> = self.results.iter().map(TaskResult::to_json).collect();
        json!({
            "model": self.model, "when": self.when, "thinking": self.thinking,
            "draft": self.draft, "kv_cache": self.kv_cache, "tasks": self.tasks,
            "results": results,
        })
    }

    fn from_json(r: &Json) -> Self {
        let n = |v: &Json| v.as_u64().unwrap_or(0) as usize;
        let s = |v: &Json| v.as_str().unwrap_or("").to_string();
        Self {
            model: s(&r["model"]),
            when: r["when"].as_u64().unwrap_or(0),
            thinking: r["thinking"].as_u64().map(|b| b as usize),
            draft: s(&r["draft"]),
            kv_cache: s(&r["kv_cache"]),
            tasks: s(&r["tasks"]),
            results: r["results"]
                .as_array()
                .map(|a| {
                    a.iter()
                        .map(|t| TaskResult {
                            id: s(&t["id"]),
                            solved: t["solved"].as_bool().unwrap_or(false),
                            turns: n(&t["turns"]),
                            generated: n(&t["generated"]),
                            prompt: n(&t["prompt"]),
                            secs: t["secs"].as_f64().unwrap_or(0.0),
                            end: s(&t["end"]),
                            test: s(&t["test"]),
                            log: t["log"].as_str().map(str::to_string),
                        })
                        .collect()
                })
                .unwrap_or_default(),
        }
    }
}

pub fn results_path() -> PathBuf {
    home().join(".config/ferrum/agent.json")
}

pub fn load_results(path: &Path) -> Vec<AgentResult> {
    fs::read_to_string(path)
        .ok()
        .and_then(|t| serde_json::from_str::<Json>(&t).ok())
        .and_then(|j| {
            j["runs"]
                .as_array()
                .map(|a| a.iter().map(AgentResult::from_json).collect())
        })
        .unwrap_or_default()
}

/// Keep the newest result per setup, and at most 60.
pub fn record(results: &mut Vec<AgentResult>, run: AgentResult) {
    results.retain(|r| !r.same_setup(&run));
    results.insert(0, run);
    results.truncate(60);
}

pub fn write_results(path: &Path, results: &[AgentResult]) {
    let runs: Vec<Json> = results.iter().map(AgentResult::to_json).collect();
    if let Some(dir) = path.parent() {
        fs::create_dir_all(dir).ok();
    }
    fs::write(
        path,
        serde_json::to_string_pretty(&json!({"runs": runs})).unwrap_or_default(),
    )
    .ok();
}

/// Results best first: most solved, then fewest generated tokens.
pub fn ranked(results: &[AgentResult]) -> Vec<&AgentResult> {
    let mut rows: Vec<&AgentResult> = results.iter().collect();
    rows.sort_by(|a, b| {
        let (ta, tb) = (a.totals(), b.totals());
        tb.solved
            .cmp(&ta.solved)
            .then(ta.generated.cmp(&tb.generated))
    });
    rows
}

#[cfg(test)]
mod tests {
    use super::*;

    fn result(id: &str, solved: bool, generated: usize) -> TaskResult {
        TaskResult {
            id: id.into(),
            solved,
            turns: 4,
            generated,
            prompt: 1000,
            secs: 10.0,
            end: "finished".into(),
            test: if solved { "passed" } else { "failed" }.into(),
            log: Some(format!("/runs/{id}.json")),
        }
    }

    #[test]
    fn bundled_tasks_are_complete() {
        let tasks = parse_tasks(TASKS).unwrap();
        assert_eq!(tasks.len(), 10);
        let ids: HashSet<&str> = tasks.iter().map(|t| t.id.as_str()).collect();
        assert_eq!(ids.len(), 10, "ids are unique");
        assert_eq!(tasks.iter().filter(|t| t.lang == "python").count(), 7);
        assert_eq!(tasks.iter().filter(|t| t.lang == "rust").count(), 3);
        for t in &tasks {
            assert_eq!(t.base.len(), 40, "{}", t.id);
            assert_ne!(t.base, t.fix);
            assert!(
                !t.tests.is_empty() || !t.hidden.is_empty(),
                "{} has no tests",
                t.id
            );
            assert!(t.issue.len() > 80, "{}", t.id);
            assert!(t.repo.starts_with("https://github.com/"));
        }
        let humantime = tasks
            .iter()
            .find(|t| t.id.starts_with("humantime"))
            .unwrap();
        assert_eq!(humantime.hint(), "cargo test --offline");
        assert!(task_message(humantime).contains("network is off"));
        assert_eq!(humantime.source(), "chronotope/humantime#73");
    }

    /// The whole grading path without a model: every task fails as checked
    /// out and passes with its real fix, with the tests run in the sandbox.
    #[test]
    #[ignore = "downloads the ten repos"]
    fn tasks_fail_before_and_pass_after_their_fix() {
        let dir = std::env::temp_dir().join(format!("ferrum-agent-tasks-{}", std::process::id()));
        let n = install(&dir, |s| eprintln!("{s}"), &AtomicBool::new(false)).unwrap();
        assert_eq!(n, 10);
        for t in tasks() {
            let src = dir.join(&t.id);
            for fixed in [false, true] {
                let work = dir.join(format!("{}-{fixed}", t.id));
                copy_tree(&src.join("repo"), &work).unwrap();
                if fixed {
                    git(
                        &[
                            "fetch",
                            "-q",
                            "--no-auto-maintenance",
                            "--depth",
                            "1",
                            &t.repo,
                            &t.fix,
                        ],
                        &work,
                    )
                    .unwrap();
                    git(&["checkout", "-q", "FETCH_HEAD", "--", "."], &work).unwrap();
                }
                let work = fs::canonicalize(&work).unwrap();
                if t.lang == "python" {
                    add_entry_points(&work);
                }
                let sb = Sandbox::with_state(work, dir.join("state"), false).unwrap();
                let (got, _) = grade(&t, &src, &sb, &AtomicBool::new(false));
                eprintln!("{} fixed={fixed}: {got}", t.id);
                assert_eq!(got == "passed", fixed, "{} fixed={fixed}: {got}", t.id);
            }
        }
        fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn thinking_stays_only_on_the_newest_reply() {
        let messages = vec![
            json!({"role": "system", "content": "s"}),
            json!({"role": "user", "content": "u"}),
            json!({"role": "assistant", "content": "", "reasoning_content": "one", "tool_calls": []}),
            json!({"role": "tool", "content": "out"}),
            json!({"role": "assistant", "content": "", "reasoning_content": "two"}),
            json!({"role": "tool", "content": "out"}),
            json!({"role": "assistant", "content": "", "reasoning_content": "three"}),
            json!({"role": "tool", "content": "out"}),
        ];
        let sent = prompt_messages(&messages);
        assert_eq!(sent.len(), messages.len());
        let thoughts: Vec<_> = sent
            .iter()
            .filter_map(|m| m["reasoning_content"].as_str())
            .collect();
        assert_eq!(thoughts, ["three"]);
        assert!(sent[2].get("tool_calls").is_some(), "only thinking dropped");
        // The transcript keeps everything.
        assert_eq!(messages[2]["reasoning_content"], "one");
    }

    #[test]
    fn python_checkouts_get_their_entry_points() {
        let dir = std::env::temp_dir().join(format!("ferrum-agent-ep-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(dir.join("pkg")).unwrap();
        git(&["init", "-q"], &dir).unwrap();
        fs::write(
            dir.join("pyproject.toml"),
            "[project]\nname = 'My-Pkg'\n\n[project.entry-points.'my.plugins']\nfoo = 'pkg.foo:Foo'\n",
        )
        .unwrap();
        fs::write(dir.join("pkg/__init__.py"), "").unwrap();
        fs::write(dir.join("pkg/foo.py"), "class Foo: pass\n").unwrap();
        add_entry_points(&dir);
        let out = Command::new("python3")
            .args([
                "-c",
                "from importlib.metadata import entry_points as e; print(e(group='my.plugins')['foo'].load().__name__)",
            ])
            .current_dir(&dir)
            .output()
            .unwrap();
        assert_eq!(String::from_utf8_lossy(&out.stdout).trim(), "Foo");
        // Hidden from the agent's `git status` and from the saved diff.
        let status =
            String::from_utf8_lossy(&git(&["status", "--porcelain"], &dir).unwrap()).into_owned();
        assert!(!status.contains("dist-info"), "{status}");
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn totals_and_efficiency() {
        let t = Totals::of(&[
            result("a", true, 3000),
            result("b", false, 9000),
            result("c", true, 1000),
        ]);
        assert_eq!(
            (t.solved, t.done, t.generated, t.prompt),
            (2, 3, 13_000, 3000)
        );
        assert_eq!(t.per_solve(), Some(6500));
        assert_eq!(Totals::of(&[result("a", false, 10)]).per_solve(), None);
    }

    #[test]
    fn saves_and_ranks_results() {
        let dir = std::env::temp_dir().join(format!("ferrum-agentres-{}", std::process::id()));
        let path = dir.join("agent.json");
        let run = |model: &str, solved: usize, generated: usize| AgentResult {
            model: model.into(),
            when: 1,
            thinking: None,
            draft: "plain decoding".into(),
            kv_cache: "F16".into(),
            tasks: "t".into(),
            results: (0..10)
                .map(|i| result(&i.to_string(), i < solved, generated / 10))
                .collect(),
        };
        let mut results = Vec::new();
        record(&mut results, run("/a", 5, 50_000));
        record(&mut results, run("/b", 5, 20_000));
        record(&mut results, run("/c", 7, 90_000));
        record(&mut results, run("/a", 6, 40_000));
        assert_eq!(results.len(), 3, "same setup replaced");
        write_results(&path, &results);
        assert_eq!(load_results(&path), results);
        let order: Vec<&str> = ranked(&results).iter().map(|r| r.model.as_str()).collect();
        // Most solved first; on a tie, fewer tokens.
        assert_eq!(order, vec!["/c", "/a", "/b"]);
        fs::remove_dir_all(dir).ok();
    }
}
