//! Chat inside the TUI. The model lives on its own thread (Metal objects stay
//! where they were created); the UI talks to it over channels and streams
//! tokens as they arrive.
use crate::*;
use ferrum::hybrid::{
    plan::PlanOptions,
    runtime::{ChatHooks, ChatRequest, Delta, Runtime, SpecOptions},
};
use std::cell::Cell;
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
        })
    }
}

enum ToWorker {
    Turn { messages: Vec<Json>, think: bool },
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
    let mut rt = match Runtime::load_speculative(
        &opts.model,
        PlanOptions {
            context: opts.context,
            ..Default::default()
        },
        opts.spec.as_ref(),
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
        sampling: format!(
            "t {} · p {} · k {} · min-p {}",
            sampling.temperature, sampling.top_p, sampling.top_k, sampling.min_p
        ),
    })));
    while let Ok(msg) = rx.recv() {
        match msg {
            ToWorker::Reset => rt.session.reset(),
            ToWorker::Turn { messages, think } => {
                cancel.store(false, Ordering::SeqCst);
                let mut vars = Map::new();
                vars.insert("enable_thinking".into(), Json::Bool(think));
                if let Some(effort) = &opts.effort {
                    vars.insert("reasoning_effort".into(), Json::String(effort.clone()));
                }
                let request = ChatRequest {
                    messages,
                    tools: Vec::new(),
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
                        send(FromWorker::Done(Box::new(TurnDone {
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
    Note,
}

pub struct Turn {
    pub role: Role,
    pub reasoning: String,
    pub content: String,
    pub done: bool,
    pub stats: Option<String>,
    pub error: Option<String>,
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
        }
    }
}

#[derive(Clone, PartialEq)]
pub enum Phase {
    Loading(String),
    Ready,
    Generating,
    Failed(String),
}

/// Slash commands offered by the palette: (text to insert, description).
pub const COMMANDS: [(&str, &str); 7] = [
    ("/reset", "clear the conversation"),
    ("/think on", "turn reasoning on"),
    ("/think off", "turn reasoning off"),
    ("/thoughts", "show or hide the model's thinking"),
    ("/stats", "speed and context of the last turn"),
    ("/help", "list the commands"),
    ("/exit", "leave the chat"),
];

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
}

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
        let mut history = Vec::new();
        if let Some(system) = &opts.system {
            history.push(json!({"role": "system", "content": system}));
        }
        let (think, show) = (opts.think, opts.show_thinking);
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
        let mut out: Vec<_> = COMMANDS
            .iter()
            .filter(|(n, _)| n.starts_with(&q))
            .copied()
            .collect();
        out.extend(
            COMMANDS
                .iter()
                .filter(|(n, _)| !n.starts_with(&q) && n.contains(word))
                .copied(),
        );
        out
    }

    fn set_input(&mut self, text: &str) {
        self.input = text.to_string();
        self.cursor = self.input.chars().count();
    }

    pub fn generating(&self) -> bool {
        self.phase == Phase::Generating
    }

    pub fn pump(&mut self) {
        while let Ok(msg) = self.rx.try_recv() {
            match msg {
                FromWorker::Stage(s) => self.phase = Phase::Loading(s),
                FromWorker::Ready(info) => {
                    self.label = info.name.clone();
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
                    self.history.pop();
                    self.end_generation();
                }
            }
        }
    }

    fn end_generation(&mut self) {
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
        self.history.push(reply);
        self.ctx = Some((d.ctx_used, d.ctx_cap));
        self.last_stats = stats;
        self.end_generation();
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

    fn submit(&mut self, text: String) {
        self.history.push(json!({"role": "user", "content": text}));
        self.turns.push(Turn::new(Role::User, text));
        self.turns.push(Turn::new(Role::Assistant, String::new()));
        self.phase = Phase::Generating;
        self.gen_started = Some(Instant::now());
        self.first_token = None;
        self.gen_tokens = 0;
        self.follow = true;
        let _ = self.tx.send(ToWorker::Turn {
            messages: self.history.clone(),
            think: self.think,
        });
    }

    fn reset(&mut self) {
        self.history.retain(|m| m["role"] == "system");
        self.turns.clear();
        self.ctx = None;
        self.last_stats.clear();
        let _ = self.tx.send(ToWorker::Reset);
    }

    /// A line starting with `/`. Returns true when the chat should close.
    fn command(&mut self, line: &str) -> bool {
        match line.trim() {
            "/exit" | "/quit" => return true,
            "/reset" | "/clear" | "/new" => {
                self.reset();
                self.note("conversation cleared");
            }
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
                for (name, what) in COMMANDS {
                    text += &format!("{name:<12} {what}\n");
                }
                text += "ctrl-t       show or hide the model's thinking\n";
                text += "alt+enter or a trailing \\ makes a new line";
                self.note(text);
            }
            other => self.note(format!("unknown command {other} (try /help)")),
        }
        false
    }

    pub fn stop_generation(&self) {
        self.cancel.store(true, Ordering::SeqCst);
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
            if let Some(h) = chat.close() {
                self.closing.push(h);
            }
        }
        self.screen = Screen::Pick;
    }

    pub fn chat_key(&mut self, key: KeyEvent) {
        let before = self.chat.as_ref().map(|c| c.input.clone());
        self.chat_key_inner(key);
        if let (Some(c), Some(b)) = (self.chat.as_mut(), before) {
            if c.input != b {
                c.palette_sel = 0;
                c.palette_off = false;
            }
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
                KeyCode::Enter => chat.set_input(palette[sel].0),
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
                let text = chat.input.trim().to_string();
                chat.input.clear();
                chat.cursor = 0;
                if text.is_empty() {
                    return;
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
            KeyCode::Backspace => {
                if chat.cursor > 0 {
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
        let area = f.area();
        let content_w = (area.width.saturating_sub(2) as usize).min(110);
        let input_w = content_w.saturating_sub(6).max(8);
        let rows = wrap_rows(&chat.input, input_w);
        let input_rows = rows.len().clamp(1, 6) as u16;
        let [head, mid, status, input, footer] = Layout::vertical([
            Constraint::Length(2),
            Constraint::Min(4),
            Constraint::Length(1),
            Constraint::Length(input_rows + 2),
            Constraint::Length(1),
        ])
        .areas(area);

        let t = self.elapsed();
        let (state, color) = match &chat.phase {
            Phase::Loading(_) => ("LOADING", GOLD),
            Phase::Ready => ("READY", GOOD),
            Phase::Generating => ("THINKING", EMBER),
            Phase::Failed(_) => ("FAILED", BAD),
        };
        let mut sub = vec![chat.label.clone()];
        if let Some(i) = &chat.info {
            sub.push(format!("ctx {}", i.context));
            if let Some(d) = &i.drafter {
                sub.push(format!("⚡ {}", d.split(" · ").next().unwrap_or("")));
            }
        }
        let crumb = sub.join(" · ");
        header(f, head, &["Chat", &crumb], vec![pill(state, color)]);

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

        let keys: &[(&str, &str)] = if !items.is_empty() {
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
                Phase::Ready => &[
                    ("enter", "send"),
                    ("alt+enter", "new line"),
                    ("pgup/pgdn", "scroll"),
                    ("ctrl-t", "thinking"),
                    ("/help", "commands"),
                    ("esc", "leave"),
                ],
                _ => &[("esc", "back")],
            }
        };
        keys_footer(f, footer, keys, &self.status);
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
            _ => {}
        }
        f.render_widget(Paragraph::new(Line::from(left)), area);
        let mut right = vec![Span::styled(
            format!("think {} ", if chat.think { "on" } else { "off" }),
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
                lines.push(Line::styled(i.sampling.clone(), Style::new().fg(DIM)));
            }
            lines.push(Line::default());
            lines.push(Line::styled(
                "type a message below to begin",
                Style::new().fg(TEXT),
            ));
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
            if n > 0 {
                out.push(Line::default());
            }
            match turn.role {
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
                    out.push(Line::from(vec![
                        Span::styled("◆ ", Style::new().fg(EMBER)),
                        Span::styled(
                            chat.label.clone(),
                            Style::new().fg(DIM).add_modifier(Modifier::BOLD),
                        ),
                    ]));
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
