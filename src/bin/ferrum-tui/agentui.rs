//! The agent side of the Benchmark tab: install the ten tasks, run the coding
//! agent on them, and rank saved runs by tasks solved, then tokens spent.
use crate::agentbench::{
    self, AgentEvent, AgentPlan, AgentResult, AgentRunning, Task, TaskResult, Totals,
};
use crate::agentcompare::AgentCompare;
use crate::bench;
use crate::benchui::{BenchPhase, ago};
use crate::benchview::{Chart, Parts, Tile, Widths, form_row};
use crate::evalsets::{InstallEvent, Installing};
use crate::evalui::{Measure, THINKING, bar_ranged, score_color};
use crate::layout::fit_w;
use crate::*;
use std::sync::atomic::Ordering;

pub struct AgentState {
    pub tasks: Vec<Task>,
    /// How many tasks are installed.
    pub ready: usize,
    pub sel: usize,
    /// Index into `THINKING`.
    pub thinking: usize,
    pub tuned: bool,
    run: Option<AgentRunning>,
    pub plan: Option<AgentPlan>,
    /// Finished tasks of the current run.
    pub done: Vec<TaskResult>,
    /// The task being worked on: (index, reply, generated, prompt, last action).
    pub current: Option<(usize, usize, usize, usize, String)>,
    pub task_started: Instant,
    pub failed: Option<String>,
    pub stage: String,
    pub started: Instant,
    pub loaded: Option<(usize, f64)>,
    pub stopping: bool,
    pub results: Vec<AgentResult>,
    pub saved: bool,
    install: Option<Installing>,
    pub install_stage: String,
    pub install_note: Option<(String, bool)>,
    pub confirm_remove: bool,
    /// The finished task the arrows point at.
    pub pick: usize,
    /// An open transcript.
    pub viewer: Option<Viewer>,
    /// The Compare view, when open.
    pub compare: Option<AgentCompare>,
    /// What the agent is doing on the current task, as it happens.
    pub feed: Vec<Feed>,
}

/// One piece of the live feed.
pub enum Feed {
    Thinking(String),
    Text(String),
    Tool {
        summary: String,
        ok: bool,
        output: String,
    },
    /// A reply ended: (reply number, tokens generated in the task so far).
    Reply(usize, usize),
    /// An intelligence question: its heading and (shortened) text.
    Question {
        title: String,
        text: String,
    },
    /// How the answer to it was graded.
    Verdict {
        correct: bool,
        answer: String,
    },
}

/// How much of the current task's feed is kept, in characters.
const FEED_KEEP: usize = 200_000;

impl AgentState {
    /// The newest words of the feed, for the tab bar.
    pub fn feed_tail(&self) -> String {
        feed_tail(&self.feed)
    }
}

/// The newest words of a feed, for the tab bar.
pub fn feed_tail(feed: &[Feed]) -> String {
    match feed.last() {
        Some(Feed::Thinking(t)) => format!(
            "thinking: {}",
            t.split_whitespace().collect::<Vec<_>>().join(" ")
        ),
        Some(Feed::Text(t)) => t.split_whitespace().collect::<Vec<_>>().join(" "),
        Some(Feed::Tool { summary, .. }) => summary.clone(),
        Some(Feed::Reply(n, _)) => format!("reply {n} done"),
        Some(Feed::Question { title, .. }) => title.clone(),
        Some(Feed::Verdict { correct, answer }) => {
            format!("{} · {answer}", if *correct { "right" } else { "wrong" })
        }
        None => String::new(),
    }
}

/// Add streamed text to the feed, joining it to the piece it continues.
pub fn stream(feed: &mut Vec<Feed>, reasoning: bool, text: String) {
    match (feed.last_mut(), reasoning) {
        (Some(Feed::Thinking(t)), true) | (Some(Feed::Text(t)), false) => t.push_str(&text),
        (_, true) => feed.push(Feed::Thinking(text)),
        (_, false) => feed.push(Feed::Text(text)),
    }
    trim_feed(feed);
}

/// Add a piece to the feed.
pub fn feed_push(feed: &mut Vec<Feed>, item: Feed) {
    feed.push(item);
    trim_feed(feed);
}

/// Drop the oldest pieces once the feed holds more than `FEED_KEEP`.
fn trim_feed(feed: &mut Vec<Feed>) {
    let mut size: usize = feed.iter().map(Feed::len).sum();
    while size > FEED_KEEP && feed.len() > 1 {
        size -= feed.remove(0).len();
    }
}

/// A long question cut to its start and end, where the asking usually is.
pub fn shorten(text: &str) -> String {
    let chars: Vec<char> = text.chars().collect();
    if chars.len() <= 1500 {
        return text.to_string();
    }
    let head: String = chars[..700].iter().collect();
    let tail: String = chars[chars.len() - 600..].iter().collect();
    format!(
        "{head}\n… {} characters left out …\n{tail}",
        chars.len() - 1300
    )
}

impl Feed {
    fn len(&self) -> usize {
        match self {
            Feed::Thinking(t) | Feed::Text(t) => t.len(),
            Feed::Tool {
                summary, output, ..
            } => summary.len() + output.len(),
            Feed::Question { title, text } => title.len() + text.len(),
            Feed::Verdict { answer, .. } => answer.len() + 16,
            Feed::Reply(..) => 16,
        }
    }
}

/// The feed as lines `width` wide: thinking dim behind a rule, the reply in
/// full, each tool call with whether it worked and the start of its output.
pub fn feed_lines(feed: &[Feed], width: usize) -> Vec<Line<'static>> {
    let mut out = Vec::new();
    for item in feed {
        match item {
            Feed::Thinking(t) => {
                for mut l in crate::layout::wrapped(
                    t.trim(),
                    3,
                    width,
                    Style::new().fg(DIM).add_modifier(Modifier::ITALIC),
                ) {
                    l.spans[0] = Span::styled(" ┊ ", Style::new().fg(FAINT));
                    out.push(l);
                }
            }
            Feed::Text(t) => {
                for para in t.trim().split('\n') {
                    out.extend(crate::layout::wrapped(
                        para,
                        1,
                        width,
                        Style::new().fg(TEXT),
                    ));
                }
            }
            Feed::Tool {
                summary,
                ok,
                output,
            } => {
                out.push(Line::from(vec![
                    Span::styled(
                        if *ok { " ✓ " } else { " ✗ " },
                        Style::new().fg(if *ok { GOOD } else { BAD }),
                    ),
                    Span::styled(
                        clip(summary, width.saturating_sub(4)),
                        Style::new().fg(GOLD).add_modifier(Modifier::BOLD),
                    ),
                ]));
                for l in output.lines().filter(|l| !l.trim().is_empty()).take(4) {
                    out.push(Line::styled(
                        format!("     {}", clip(l, width.saturating_sub(6))),
                        Style::new().fg(FAINT),
                    ));
                }
            }
            Feed::Question { title, text } => {
                out.push(Line::default());
                out.push(Line::styled(
                    format!(" ▸ {}", clip(title, width.saturating_sub(4))),
                    Style::new().fg(EMBER).add_modifier(Modifier::BOLD),
                ));
                // Long prompts show their first and last lines.
                let mut rows: Vec<Line> = Vec::new();
                for para in text.trim().split('\n') {
                    rows.extend(crate::layout::wrapped(
                        para,
                        3,
                        width,
                        Style::new().fg(TEXT),
                    ));
                }
                if rows.len() > 14 {
                    let left = rows.len() - 10;
                    let tail = rows.split_off(rows.len() - 4);
                    rows.truncate(6);
                    rows.push(Line::styled(
                        format!("   … {left} more lines …"),
                        Style::new().fg(FAINT),
                    ));
                    rows.extend(tail);
                }
                out.extend(rows);
            }
            Feed::Verdict { correct, answer } => out.push(Line::from(vec![
                Span::styled(
                    if *correct { " ✓ right" } else { " ✗ wrong" },
                    Style::new()
                        .fg(if *correct { GOOD } else { BAD })
                        .add_modifier(Modifier::BOLD),
                ),
                Span::styled(
                    format!(
                        " · graded answer: {}",
                        clip(answer, width.saturating_sub(28))
                    ),
                    Style::new().fg(DIM),
                ),
            ])),
            Feed::Reply(n, tokens) => out.push(Line::styled(
                format!(
                    " {} reply {n} · {} tokens so far",
                    "─".repeat(2),
                    kilo(*tokens)
                ),
                Style::new().fg(FAINT),
            )),
        }
    }
    out
}

/// A transcript laid out for reading.
pub struct Viewer {
    pub title: String,
    pub lines: Vec<Line<'static>>,
    pub scroll: usize,
}

impl AgentState {
    pub fn new() -> Self {
        let tasks = agentbench::tasks();
        Self {
            ready: agentbench::installed(&agentbench::tasks_dir(), &tasks),
            tasks,
            sel: 0,
            thinking: 0,
            tuned: false,
            run: None,
            plan: None,
            done: Vec::new(),
            current: None,
            task_started: Instant::now(),
            failed: None,
            stage: String::new(),
            started: Instant::now(),
            loaded: None,
            stopping: false,
            results: agentbench::load_results(&agentbench::results_path()),
            saved: false,
            install: None,
            install_stage: String::new(),
            install_note: None,
            confirm_remove: false,
            pick: 0,
            viewer: None,
            compare: None,
            feed: Vec::new(),
        }
    }

    fn all_ready(&self) -> bool {
        self.ready == self.tasks.len()
    }

    pub fn stop(&mut self) -> Vec<std::thread::JoinHandle<()>> {
        let mut out = Vec::new();
        if let Some(mut run) = self.run.take() {
            run.stop.store(true, Ordering::SeqCst);
            out.extend(run.handle.take());
        }
        if let Some(mut install) = self.install.take() {
            install.stop.store(true, Ordering::SeqCst);
            out.extend(install.handle.take());
        }
        out
    }

    fn poll_install(&mut self) {
        let Some(install) = self.install.as_ref() else {
            return;
        };
        let mut done = None;
        while let Ok(event) = install.rx.try_recv() {
            match event {
                InstallEvent::Progress(s) => self.install_stage = s,
                InstallEvent::Done(r) => done = Some(r),
            }
        }
        let Some(result) = done else { return };
        if let Some(mut install) = self.install.take()
            && let Some(h) = install.handle.take()
        {
            let _ = h.join();
        }
        self.ready = agentbench::installed(&agentbench::tasks_dir(), &self.tasks);
        self.install_note = Some(match result {
            Ok(n) => (format!("{n} tasks installed"), true),
            Err(e) => (
                format!("install failed (finished tasks are kept): {e}"),
                false,
            ),
        });
    }

    /// Enter on the tasks line: install what is missing, or remove after a confirm.
    fn tasks_action(&mut self) {
        if self.install.is_some() {
            return;
        }
        if !self.all_ready() {
            self.confirm_remove = false;
            self.install_note = None;
            self.install_stage = "starting".into();
            match agentbench::spawn_install(agentbench::tasks_dir()) {
                Ok(i) => self.install = Some(i),
                Err(e) => self.install_note = Some((e, false)),
            }
        } else if self.confirm_remove {
            self.confirm_remove = false;
            self.install_note = Some(match agentbench::remove(&agentbench::tasks_dir()) {
                Ok(()) => ("removed the tasks".into(), true),
                Err(e) => (format!("could not remove them: {e}"), false),
            });
            self.ready = agentbench::installed(&agentbench::tasks_dir(), &self.tasks);
        } else {
            self.confirm_remove = true;
        }
    }
}

#[derive(Clone, Copy, PartialEq)]
enum AgLine {
    Model,
    Measure,
    Tasks,
    Thinking,
    Draft,
    Run,
}

/// What a setup row does.
fn ag_help(line: AgLine) -> &'static str {
    match line {
        AgLine::Model => {
            "The model that drives the coding agent. Runs are saved per model and setup."
        }
        AgLine::Measure => {
            "Agentic coding gives the agent real bugs to fix in small projects; a task counts when its tests pass. m switches."
        }
        AgLine::Tasks => {
            "The task projects, fetched once with git. Enter installs them, or (twice) removes them."
        }
        AgLine::Thinking => {
            "Let the model reason before each reply, up to this many tokens. Slower, often more tasks solved."
        }
        AgLine::Draft => {
            "Use the drafter the speed benchmark picked for this Mac, so the tasks finish sooner."
        }
        AgLine::Run => {
            "Start the tasks. Esc while they run stops and shows what finished; enter opens a task's transcript."
        }
    }
}

const LINES: [AgLine; 6] = [
    AgLine::Model,
    AgLine::Measure,
    AgLine::Tasks,
    AgLine::Thinking,
    AgLine::Draft,
    AgLine::Run,
];

/// `text` wrapped to `width`, each row indented by `indent` spaces.
fn wrapped(out: &mut Vec<Line<'static>>, text: &str, indent: usize, width: usize, style: Style) {
    for raw in text.lines() {
        let chars: Vec<char> = raw.chars().collect();
        if chars.is_empty() {
            out.push(Line::default());
            continue;
        }
        for (a, z) in wrap_rows(raw, width.saturating_sub(indent + 1).max(10)) {
            out.push(Line::styled(
                format!(
                    "{}{}",
                    " ".repeat(indent),
                    chars[a..z].iter().collect::<String>()
                ),
                style,
            ));
        }
    }
}

/// A saved task log as lines: the conversation, the changes and the tests.
pub fn transcript(log: &Json, width: usize) -> Vec<Line<'static>> {
    let mut out = Vec::new();
    let head = |s: &str, color: Color| {
        Line::styled(
            s.to_string(),
            Style::new().fg(color).add_modifier(Modifier::BOLD),
        )
    };
    let r = &log["result"];
    out.push(Line::styled(
        format!(
            " {} · {} · {} replies · {} generated · {} · tests {}",
            log["source"].as_str().unwrap_or(""),
            if r["solved"].as_bool().unwrap_or(false) {
                "solved"
            } else {
                "not solved"
            },
            r["turns"],
            kilo(r["generated"].as_u64().unwrap_or(0) as usize),
            r["end"].as_str().unwrap_or(""),
            r["test"].as_str().unwrap_or("")
        ),
        Style::new().fg(TEXT),
    ));
    out.push(Line::default());
    for m in log["messages"].as_array().into_iter().flatten() {
        let content = m["content"].as_str().unwrap_or("");
        match m["role"].as_str().unwrap_or("") {
            "system" => out.push(Line::styled(
                format!(" system prompt · {} characters", content.len()),
                Style::new().fg(FAINT),
            )),
            "user" => {
                out.push(head(" USER", GOLD));
                wrapped(&mut out, content, 3, width, Style::new().fg(TEXT));
            }
            "assistant" => {
                out.push(Line::default());
                out.push(head(" MODEL", EMBER));
                if let Some(thought) = m["reasoning_content"].as_str() {
                    wrapped(&mut out, thought, 3, width, Style::new().fg(DIM));
                }
                wrapped(&mut out, content, 3, width, Style::new().fg(TEXT));
                for call in m["tool_calls"].as_array().into_iter().flatten() {
                    let f = &call["function"];
                    wrapped(
                        &mut out,
                        &format!(
                            "⚙ {} {}",
                            f["name"].as_str().unwrap_or(""),
                            clip(f["arguments"].as_str().unwrap_or(""), 600)
                        ),
                        3,
                        width,
                        Style::new().fg(GOLD),
                    );
                }
            }
            "tool" => {
                let lines: Vec<&str> = content.lines().collect();
                let shown = lines.len().min(15);
                wrapped(
                    &mut out,
                    &lines[..shown].join("\n"),
                    5,
                    width,
                    Style::new().fg(DIM),
                );
                if lines.len() > shown {
                    out.push(Line::styled(
                        format!("     … {} more lines", lines.len() - shown),
                        Style::new().fg(FAINT),
                    ));
                }
            }
            _ => {}
        }
    }
    out.push(Line::default());
    out.push(head(" CHANGES", EMBER));
    let diff = log["changes"].as_str().unwrap_or("");
    if diff.trim().is_empty() {
        out.push(Line::styled("   no changes", Style::new().fg(FAINT)));
    }
    for line in diff.lines() {
        let color = if line.starts_with("+++") || line.starts_with("---") {
            DIM
        } else if line.starts_with('+') || line.starts_with("new file:") {
            GOOD
        } else if line.starts_with('-') {
            BAD
        } else if line.starts_with("@@") {
            GOLD
        } else {
            DIM
        };
        wrapped(&mut out, line, 3, width, Style::new().fg(color));
    }
    out.push(Line::default());
    out.push(head(
        &format!(
            " TESTS · {} · {}",
            r["test"].as_str().unwrap_or(""),
            log["test_command"].as_str().unwrap_or("")
        ),
        if r["solved"].as_bool().unwrap_or(false) {
            GOOD
        } else {
            BAD
        },
    ));
    let output = log["test_output"].as_str().unwrap_or("");
    let lines: Vec<&str> = output.lines().collect();
    let from = lines.len().saturating_sub(80);
    wrapped(
        &mut out,
        &lines[from..].join("\n"),
        3,
        width,
        Style::new().fg(DIM),
    );
    out
}

/// 1234 -> "1.2K", 1234567 -> "1.2M".
pub fn kilo(n: usize) -> String {
    match n {
        0..=999 => n.to_string(),
        1000..=999_999 => format!("{:.1}K", n as f64 / 1000.0),
        _ => format!("{:.1}M", n as f64 / 1_000_000.0),
    }
}

/// 75 -> "1m15s", 3900 -> "1h05m".
pub fn duration(secs: f64) -> String {
    let s = secs.round() as u64;
    if s >= 3600 {
        format!("{}h{:02}m", s / 3600, (s % 3600) / 60)
    } else {
        format!("{}m{:02}s", s / 60, s % 60)
    }
}

fn model_name(path: &str) -> String {
    let name = Path::new(path)
        .file_name()
        .map(|n| n.to_string_lossy().to_string())
        .unwrap_or_else(|| path.to_string());
    name.strip_suffix(".gguf")
        .map(str::to_string)
        .unwrap_or(name)
}

impl App {
    fn agent_plan(&self) -> Option<AgentPlan> {
        let b = self.bench.as_ref()?;
        let model = self.models.get(b.model)?;
        Some(AgentPlan {
            model: model.path.clone(),
            thinking: THINKING[b.ag.thinking],
            draft: if b.ag.tuned { self.tuned_draft() } else { None },
            kv_cache: self.kv_cache().unwrap_or_default(),
        })
    }

    fn start_agent(&mut self) {
        if !self.make_room("the benchmark", false, false) {
            return;
        }
        let Some(plan) = self.agent_plan() else {
            self.status = Some(("no supported model to test".into(), false));
            return;
        };
        let Some(b) = self.bench.as_ref() else { return };
        if b.ag.install.is_some() {
            self.status = Some(("wait for the tasks to finish installing".into(), false));
            return;
        }
        if !b.ag.all_ready() {
            self.status = Some((
                "install the tasks first (enter on the Tasks line)".into(),
                false,
            ));
            return;
        }
        let tasks = b.ag.tasks.clone();
        self.reap();
        let Some(b) = self.bench.as_mut() else { return };
        match agentbench::spawn(plan.clone(), tasks) {
            Ok(run) => {
                let ag = &mut b.ag;
                ag.run = Some(run);
                ag.plan = Some(plan);
                ag.done.clear();
                ag.feed.clear();
                ag.pick = 0;
                ag.viewer = None;
                ag.current = None;
                ag.failed = None;
                ag.loaded = None;
                ag.stopping = false;
                ag.saved = false;
                ag.started = Instant::now();
                ag.task_started = Instant::now();
                ag.stage = "starting".into();
                b.phase = BenchPhase::Running;
            }
            Err(e) => self.status = Some((e, false)),
        }
    }

    pub fn agent_tick(&mut self) {
        let Some(b) = self.bench.as_mut() else { return };
        let ag = &mut b.ag;
        ag.poll_install();
        let Some(run) = ag.run.as_ref() else { return };
        let mut finished = false;
        while let Ok(event) = run.rx.try_recv() {
            match event {
                AgentEvent::Stage(s) => {
                    if s.starts_with("task ") {
                        ag.task_started = Instant::now();
                        ag.current = None;
                    }
                    ag.stage = s;
                }
                AgentEvent::Loaded {
                    context,
                    weights_gib,
                } => ag.loaded = Some((context, weights_gib)),
                AgentEvent::Turn {
                    task,
                    turn,
                    generated,
                    prompt,
                    last,
                } => {
                    ag.feed.push(Feed::Reply(turn, generated));
                    ag.current = Some((task, turn, generated, prompt, last));
                }
                AgentEvent::TaskStart(task) => {
                    ag.feed.clear();
                    ag.task_started = Instant::now();
                    ag.current = Some((task, 0, 0, 0, "reading the task".into()));
                }
                AgentEvent::Delta { reasoning, text } => stream(&mut ag.feed, reasoning, text),
                AgentEvent::Tool {
                    summary,
                    ok,
                    output,
                } => ag.feed.push(Feed::Tool {
                    summary,
                    ok,
                    output,
                }),
                AgentEvent::TaskDone(r) => {
                    ag.current = None;
                    ag.done.push(r);
                }
                AgentEvent::Failed(e) => ag.failed = Some(e),
                AgentEvent::Done => finished = true,
            }
        }
        if !finished {
            return;
        }
        if let Some(mut run) = ag.run.take()
            && let Some(h) = run.handle.take()
        {
            let _ = h.join();
        }
        b.phase = BenchPhase::Results;
        let complete = ag.failed.is_none() && ag.done.len() == ag.tasks.len();
        if let (true, Some(plan)) = (complete, &ag.plan) {
            agentbench::record(
                &mut ag.results,
                AgentResult {
                    model: plan.model.display().to_string(),
                    when: bench::now(),
                    thinking: plan.thinking,
                    draft: plan.draft_label(),
                    kv_cache: format!("{:?}", plan.kv_cache),
                    tasks: agentbench::tasks_fingerprint(),
                    results: ag.done.clone(),
                },
            );
            agentbench::write_results(&agentbench::results_path(), &ag.results);
            ag.saved = true;
        }
    }

    pub fn agent_stop(&mut self) {
        if let Some(b) = self.bench.as_mut()
            && let Some(run) = &b.ag.run
        {
            run.stop.store(true, Ordering::SeqCst);
            b.ag.stopping = true;
        }
    }

    pub fn agent_setup_key(&mut self, key: KeyEvent) {
        let runnable: Vec<usize> = (0..self.models.len())
            .filter(|&i| self.models[i].problem.is_none())
            .collect();
        let has_tuned = self.tuned_draft().is_some();
        let Some(b) = self.bench.as_mut() else { return };
        let ag = &mut b.ag;
        ag.sel = ag.sel.min(LINES.len() - 1);
        let removing = ag.confirm_remove;
        ag.confirm_remove = false;
        let mut model_changed = false;
        let mut start = false;
        match key.code {
            KeyCode::Esc | KeyCode::Char('q') => return self.leave_bench(),
            KeyCode::Up | KeyCode::Char('k') => ag.sel = ag.sel.saturating_sub(1),
            KeyCode::Down | KeyCode::Char('j') | KeyCode::Tab => {
                ag.sel = (ag.sel + 1).min(LINES.len() - 1)
            }
            KeyCode::Char('r') => start = true,
            KeyCode::Char('c') => ag.compare = Some(AgentCompare::default()),
            KeyCode::Left | KeyCode::Right | KeyCode::Enter | KeyCode::Char(' ') => {
                let step: i32 = if key.code == KeyCode::Left { -1 } else { 1 };
                match LINES[ag.sel] {
                    AgLine::Model => {
                        if let Some(at) = runnable.iter().position(|&i| i == b.model) {
                            let n = runnable.len() as i32;
                            b.model = runnable[(at as i32 + step).rem_euclid(n) as usize];
                            model_changed = true;
                        }
                    }
                    AgLine::Measure => {
                        b.measure = Measure::Speed;
                        b.sel = 1;
                    }
                    AgLine::Tasks => {
                        ag.confirm_remove = removing;
                        if key.code != KeyCode::Left {
                            ag.tasks_action();
                        }
                    }
                    AgLine::Thinking => {
                        ag.thinking =
                            (ag.thinking as i32 + step).rem_euclid(THINKING.len() as i32) as usize
                    }
                    AgLine::Draft => ag.tuned = !ag.tuned && has_tuned,
                    AgLine::Run => start = true,
                }
            }
            _ => {}
        }
        if model_changed {
            self.bench_model_changed();
            if let Some(b) = self.bench.as_mut() {
                b.ag.tuned = false;
            }
        }
        if start {
            self.start_agent();
        }
    }

    /// Arrows pick a finished task, Enter opens its transcript. Returns true
    /// when the key was one of those.
    fn agent_pick_key(&mut self, key: KeyEvent) -> bool {
        let Some(b) = self.bench.as_mut() else {
            return false;
        };
        let ag = &mut b.ag;
        let last = ag.done.len().saturating_sub(1);
        match key.code {
            KeyCode::Up | KeyCode::Char('k') => ag.pick = ag.pick.saturating_sub(1).min(last),
            KeyCode::Down | KeyCode::Char('j') => ag.pick = (ag.pick + 1).min(last),
            KeyCode::Enter => {
                let Some(r) = ag.done.get(ag.pick) else {
                    return true;
                };
                let loaded = r
                    .log
                    .as_ref()
                    .ok_or("no transcript was saved for this task".to_string())
                    .and_then(|p| fs::read_to_string(p).map_err(|e| format!("{p}: {e}")))
                    .and_then(|t| serde_json::from_str::<Json>(&t).map_err(|e| e.to_string()));
                match loaded {
                    Ok(log) => {
                        let width = crossterm::terminal::size()
                            .map_or(100, |(w, _)| w.min(104) as usize)
                            .saturating_sub(4);
                        ag.viewer = Some(Viewer {
                            title: format!("Transcript · {}", r.id),
                            lines: transcript(&log, width),
                            scroll: 0,
                        });
                    }
                    Err(e) => self.status = Some((e, false)),
                }
            }
            _ => return false,
        }
        true
    }

    pub fn agent_running_key(&mut self, key: KeyEvent) {
        if self.bench.as_ref().is_some_and(|b| b.ag.viewer.is_some()) {
            return self.agent_viewer_key(key);
        }
        if self.agent_pick_key(key) {
            return;
        }
        if matches!(key.code, KeyCode::Esc | KeyCode::Char('q')) {
            self.agent_stop();
        }
    }

    pub fn agent_results_key(&mut self, key: KeyEvent) {
        if self.bench.as_ref().is_some_and(|b| b.ag.viewer.is_some()) {
            return self.agent_viewer_key(key);
        }
        if self.agent_pick_key(key) {
            return;
        }
        match key.code {
            KeyCode::Char('c') => {
                if let Some(b) = self.bench.as_mut() {
                    b.ag.compare = Some(AgentCompare::default());
                }
            }
            KeyCode::Char('r') => {
                if let Some(b) = self.bench.as_mut() {
                    b.phase = BenchPhase::Setup;
                }
            }
            KeyCode::Esc | KeyCode::Char('q') => self.leave_bench(),
            _ => {}
        }
    }

    fn agent_viewer_key(&mut self, key: KeyEvent) {
        let Some(b) = self.bench.as_mut() else { return };
        let Some(v) = b.ag.viewer.as_mut() else {
            return;
        };
        let page = crossterm::terminal::size()
            .map_or(20, |(_, h)| h as usize)
            .saturating_sub(8);
        let max = v.lines.len().saturating_sub(page);
        match key.code {
            KeyCode::Esc | KeyCode::Char('q') | KeyCode::Enter => b.ag.viewer = None,
            KeyCode::Up | KeyCode::Char('k') => v.scroll = v.scroll.saturating_sub(1),
            KeyCode::Down | KeyCode::Char('j') => v.scroll = (v.scroll + 1).min(max),
            KeyCode::PageUp | KeyCode::Char('b') => v.scroll = v.scroll.saturating_sub(page),
            KeyCode::PageDown | KeyCode::Char(' ') => v.scroll = (v.scroll + page).min(max),
            KeyCode::Home | KeyCode::Char('g') => v.scroll = 0,
            KeyCode::End | KeyCode::Char('G') => v.scroll = max,
            _ => {}
        }
    }

    pub fn draw_agent_viewer(&self, f: &mut Frame, area: Rect, b: &BenchState) {
        let Some(v) = &b.ag.viewer else { return };
        let block = panel(&v.title, true);
        let inner = block.inner(area);
        f.render_widget(block, area);
        let h = inner.height as usize;
        let scroll = v.scroll.min(v.lines.len().saturating_sub(h));
        f.render_widget(
            Paragraph::new(
                v.lines
                    .iter()
                    .skip(scroll)
                    .take(h)
                    .cloned()
                    .collect::<Vec<_>>(),
            ),
            inner,
        );
    }

    // ---- drawing ----

    /// Saved runs of every model, most solved first, then fewest tokens.
    fn agent_board(&self, b: &BenchState, width: usize) -> Vec<Line<'static>> {
        let rows = agentbench::ranked(&b.ag.results);
        if rows.is_empty() {
            return Vec::new();
        }
        let current = agentbench::tasks_fingerprint();
        // Narrow, each run's setup goes on a line of its own under it.
        let narrow = width < 120;
        // Narrower still, the token columns go: solved and time are the result.
        let tight = width < 80;
        let name_w = if tight {
            width.saturating_sub(34).clamp(12, 40)
        } else if narrow {
            width.saturating_sub(52).clamp(16, 40)
        } else {
            width.saturating_sub(84).clamp(16, 40)
        };
        let mut lines = vec![
            Line::styled(
                "  most solved first, then fewest tokens",
                Style::new().fg(FAINT),
            ),
            Line::styled(
                if tight {
                    format!(
                        "   {:<name_w$} {:>6}  {:<10} {:>8}",
                        "model", "solved", "", "time"
                    )
                } else {
                    format!(
                        "   {:<name_w$} {:>6}  {:<10} {:>9} {:>9} {:>8}{}",
                        "model",
                        "solved",
                        "",
                        "tok/solve",
                        "generated",
                        "time",
                        if narrow { "" } else { "  setup" }
                    )
                },
                Style::new().fg(FAINT),
            ),
        ];
        for r in rows.iter().take(12) {
            let t = r.totals();
            let share = t.solved as f64 / t.done.max(1) as f64;
            let stale = r.tasks != current;
            let mut spans = vec![Span::styled(
                format!(
                    "   {:<name_w$} {:>3}/{:<2}  ",
                    clip(&model_name(&r.model), name_w),
                    t.solved,
                    t.done
                ),
                Style::new().fg(TEXT),
            )];
            spans.extend(bar_ranged(share, 0.0, 10, score_color(share)));
            spans.push(Span::styled(
                if tight {
                    format!(" {:>8}", duration(t.secs))
                } else {
                    format!(
                        " {:>9} {:>9} {:>8}  ",
                        t.per_solve().map_or("–".to_string(), kilo),
                        kilo(t.generated),
                        duration(t.secs)
                    )
                },
                Style::new().fg(TEXT),
            ));
            let setup = format!(
                "thinking {} · {} · {}{}",
                r.thinking.map_or("off".to_string(), |b| format!("≤{b}")),
                r.draft,
                ago(r.when),
                if stale { " · older tasks" } else { "" }
            );
            let style = Style::new().fg(if stale { FAINT } else { DIM });
            if narrow {
                lines.push(Line::from(spans));
                lines.push(Line::styled(
                    format!("     {}", clip(&setup, width.saturating_sub(6))),
                    Style::new().fg(FAINT),
                ));
            } else {
                spans.push(Span::styled(setup, style));
                lines.push(Line::from(spans));
            }
        }
        lines
    }

    /// One row per finished task.
    fn agent_task_rows(done: &[TaskResult], pick: usize, width: usize) -> Vec<Line<'static>> {
        let id_w = width.saturating_sub(70).clamp(20, 36);
        let mut lines = vec![Line::styled(
            format!(
                "   {:<id_w$} {:>6} {:>9} {:>9} {:>8}  how it ended",
                "task", "replies", "generated", "read", "time"
            ),
            Style::new().fg(FAINT),
        )];
        for (i, r) in done.iter().enumerate() {
            let on = i == pick.min(done.len().saturating_sub(1));
            lines.push(Line::styled(
                format!(
                    "{}{} {:<id_w$} {:>6} {:>9} {:>9} {:>8}  {} · tests {}",
                    if on { "▌" } else { " " },
                    if r.solved { "✓" } else { "✗" },
                    clip(&r.id, id_w),
                    r.turns,
                    kilo(r.generated),
                    kilo(r.prompt),
                    duration(r.secs),
                    r.end,
                    r.test
                ),
                Style::new()
                    .fg(if r.solved { GOOD } else { BAD })
                    .bg(if on { SELECTED } else { Color::Reset }),
            ));
        }
        lines
    }

    /// The agentic-coding measure's panes.
    pub fn agent_parts(&self, b: &BenchState, w: Widths) -> Parts {
        let ag = &b.ag;
        let setup = b.phase == BenchPhase::Setup;
        let tuned = self.tuned_draft();
        let model = self.models.get(b.model);
        let path = model
            .map(|m| m.path.display().to_string())
            .unwrap_or_default();
        let sel = ag.sel.min(LINES.len() - 1);
        let mut form = Vec::new();
        let mut form_sel = None;
        for (n, &def) in LINES.iter().enumerate() {
            let on = n == sel && setup;
            let (label, value): (&str, String) = match def {
                AgLine::Model => (
                    "Model",
                    model
                        .map(|m| m.name.clone())
                        .unwrap_or_else(|| "none".into()),
                ),
                AgLine::Measure => ("Measure", "agentic coding".into()),
                AgLine::Tasks => (
                    "Tasks",
                    if ag.install.is_some() {
                        "installing…".into()
                    } else if ag.all_ready() && ag.confirm_remove {
                        "enter again removes".into()
                    } else if ag.all_ready() {
                        format!("{} installed", ag.ready)
                    } else if ag.ready > 0 {
                        format!("{} of {} · enter adds", ag.ready, ag.tasks.len())
                    } else {
                        "enter installs".into()
                    },
                ),
                AgLine::Thinking => (
                    "Thinking",
                    match THINKING[ag.thinking] {
                        None => "off".into(),
                        Some(t) => format!("≤ {t} per reply"),
                    },
                ),
                AgLine::Draft => (
                    "Drafter",
                    match (&tuned, ag.tuned) {
                        (Some((c, d)), true) => {
                            format!("tuned · {} d{d}", crate::benchui::short_drafter(&c.label()))
                        }
                        (Some(_), false) => "off · tuned available".into(),
                        (None, _) => "off · none tuned yet".into(),
                    },
                ),
                AgLine::Run => ("▶ Run agent tasks", String::new()),
            };
            if def == AgLine::Run {
                form.push(Line::default());
            }
            if on {
                form_sel = Some(form.len());
            }
            let value = clip(&value, w.form.saturating_sub(label.len() + 8));
            let warn = def == AgLine::Tasks && ag.confirm_remove;
            form.push(form_row(on, true, w.form, label, &value, warn, !setup));
        }
        form.push(Line::styled(
            format!(
                "   ≤{} replies and {} min a task, network off",
                agentbench::MAX_TURNS,
                agentbench::TASK_LIMIT.as_secs() / 60
            ),
            Style::new().fg(FAINT),
        ));

        // The last saved run of this model, for the tiles before a run.
        let last = ag
            .results
            .iter()
            .filter(|r| r.model == path)
            .max_by_key(|r| r.when);
        let now = ag.current.as_ref().map(|c| c.0);
        let result_of = |id: &str| -> Option<&TaskResult> {
            if setup {
                last.and_then(|r| r.results.iter().find(|t| t.id == id))
            } else {
                ag.done.iter().find(|t| t.id == id)
            }
        };
        let state = |i: usize, id: &str| -> (String, Color) {
            match result_of(id) {
                Some(r) if r.solved => ("✓".into(), GOOD),
                Some(_) => ("✗".into(), BAD),
                None if now == Some(i) && !setup => (spinner(self.elapsed()).to_string(), EMBER),
                None => ("·".into(), FAINT),
            }
        };

        // The queue: every task with its state.
        let mut queue = Vec::new();
        if ag.install.is_some() {
            queue.push(Line::styled(
                format!(
                    " {} {}",
                    spinner(self.elapsed()),
                    clip(&ag.install_stage, w.form.saturating_sub(4))
                ),
                Style::new().fg(EMBER),
            ));
        }
        if let Some((note, ok)) = &ag.install_note {
            queue.extend(crate::layout::wrapped(
                &format!("{} {note}", if *ok { "✓" } else { "✗" }),
                1,
                w.form,
                Style::new().fg(if *ok { GOOD } else { BAD }),
            ));
        }
        for (i, t) in ag.tasks.iter().enumerate() {
            let (mark, color) = if setup {
                let ready = agentbench::tasks_dir().join(&t.id).join("ready").exists();
                if ready {
                    ("·".to_string(), DIM)
                } else {
                    (" ".to_string(), FAINT)
                }
            } else {
                state(i, &t.id)
            };
            let right = if setup {
                t.lang.clone()
            } else {
                result_of(&t.id).map_or(t.lang.clone(), |r| duration(r.secs))
            };
            queue.push(Line::from(vec![
                Span::styled(format!(" {mark} "), Style::new().fg(color)),
                Span::styled(
                    fit_w(&t.id, w.form.saturating_sub(text_width(&right) + 6)),
                    Style::new().fg(TEXT),
                ),
                Span::styled(
                    format!(" {right}"),
                    Style::new().fg(if t.lang == "rust" { GOLD } else { DIM }),
                ),
            ]));
        }

        // The chart: a tile per task.
        let tiles: Vec<Tile> = ag
            .tasks
            .iter()
            .enumerate()
            .map(|(i, t)| {
                let (mark, color) = state(i, &t.id);
                let lines = match result_of(&t.id) {
                    Some(r) => vec![
                        format!(
                            "{} · tests {}",
                            if r.solved { "solved" } else { "not solved" },
                            r.test
                        ),
                        format!(
                            "{} · {} tok · {} replies",
                            duration(r.secs),
                            kilo(r.generated),
                            r.turns
                        ),
                    ],
                    None => match &ag.current {
                        Some((task, turn, generated, _, last)) if *task == i && !setup => vec![
                            format!("reply {turn} · {} tok", kilo(*generated)),
                            last.clone(),
                        ],
                        _ => vec![
                            if setup && last.is_some() {
                                "not in that run".into()
                            } else {
                                "waiting".into()
                            },
                            format!("{} · {}", t.lang, t.source()),
                        ],
                    },
                };
                Tile {
                    mark,
                    color,
                    title: t.id.clone(),
                    lines,
                }
            })
            .collect();
        let t = Totals::of(&ag.done);
        let (chart_title, chart_note) = if setup {
            match last {
                Some(r) => {
                    let lt = r.totals();
                    (
                        format!("Last run of this model · {}/{} solved", lt.solved, lt.done),
                        format!("{} · {}", duration(lt.secs), ago(r.when)),
                    )
                }
                None => ("Tasks · not run with this model yet".into(), String::new()),
            }
        } else if b.phase == BenchPhase::Results {
            (
                format!(
                    "★ {}/{} solved · {}",
                    t.solved,
                    ag.tasks.len(),
                    duration(t.secs)
                ),
                format!("{} generated", kilo(t.generated)),
            )
        } else {
            (
                format!("Tasks · {} solved of {} done", t.solved, t.done),
                ag.plan.as_ref().map_or(String::new(), |p| {
                    format!(
                        "thinking {} · {}",
                        p.thinking.map_or("off".into(), |t| format!("≤{t}")),
                        p.draft_label()
                    )
                }),
            )
        };

        // Rows: every saved run before, this run's tasks during and after.
        let running = b.phase == BenchPhase::Running;
        let (rows_title, rows_note, rows, rows_keep) = if setup {
            let board = self.agent_board(b, w.rows);
            let rows = if board.is_empty() {
                vec![Line::styled(" no saved runs yet", Style::new().fg(FAINT))]
            } else {
                board
            };
            (
                "Saved runs · every model".to_string(),
                "most solved first".to_string(),
                rows,
                None,
            )
        } else if running {
            // The live feed: what the agent thinks, writes and runs.
            let (title, note) = match &ag.current {
                Some((task, turn, generated, prompt, _)) => (
                    format!(
                        "Live · {}",
                        ag.tasks.get(*task).map_or("", |t| t.id.as_str())
                    ),
                    format!(
                        "reply {}/{} · {} written · {} read · {}",
                        turn + 1,
                        agentbench::MAX_TURNS,
                        kilo(*generated),
                        kilo(*prompt),
                        duration(ag.task_started.elapsed().as_secs_f64())
                    ),
                ),
                None => ("Live".to_string(), ag.stage.clone()),
            };
            let mut rows = feed_lines(&ag.feed, w.rows);
            if rows.is_empty() {
                rows.push(Line::styled(
                    format!(" {} {}", spinner(self.elapsed()), ag.stage),
                    Style::new().fg(DIM),
                ));
            } else if ag.current.is_some() {
                rows.push(Line::styled(
                    format!(" {}", spinner(self.elapsed())),
                    Style::new().fg(EMBER),
                ));
            }
            (title, note, rows, None)
        } else {
            let mut rows = Vec::new();
            if let Some((task, turn, generated, prompt, last)) = &ag.current {
                let id = ag.tasks.get(*task).map_or("", |t| t.id.as_str());
                rows.push(Line::styled(
                    clip(
                        &format!(
                            " ▸ {id} · reply {turn}/{} · {} generated · {} read · {}",
                            agentbench::MAX_TURNS,
                            kilo(*generated),
                            kilo(*prompt),
                            duration(ag.task_started.elapsed().as_secs_f64()),
                        ),
                        w.rows,
                    ),
                    Style::new().fg(EMBER),
                ));
                rows.push(Line::styled(
                    format!("   last: {}", clip(last, w.rows.saturating_sub(10))),
                    Style::new().fg(DIM),
                ));
                rows.push(Line::default());
            }
            let first = rows.len() + 1;
            rows.extend(Self::agent_task_rows(&ag.done, ag.pick, w.rows));
            let keep = (!ag.done.is_empty()).then(|| first + ag.pick.min(ag.done.len() - 1));
            (
                "Finished tasks".to_string(),
                "↑↓ choose · enter opens the transcript".to_string(),
                rows,
                keep,
            )
        };

        let mut notes: Vec<Line> = Vec::new();
        if let Some(e) = &ag.failed {
            notes.push(Line::styled(
                " ✗ the run failed · not saved",
                Style::new().fg(BAD),
            ));
            notes.extend(crate::layout::wrapped(e, 3, w.notes, Style::new().fg(DIM)));
            notes.push(Line::default());
        } else if b.phase == BenchPhase::Results {
            notes.push(if ag.saved {
                Line::styled(" ✓ saved", Style::new().fg(GOOD))
            } else {
                Line::styled(
                    format!(
                        " stopped after {} of {} · not saved",
                        ag.done.len(),
                        ag.tasks.len()
                    ),
                    Style::new().fg(GOLD),
                )
            });
            notes.push(Line::default());
        }
        for (k, what) in [
            ("solved", "the fix's own tests pass once the agent stops"),
            (
                "limits",
                "replies and minutes per task are capped; the network is off",
            ),
            (
                "ranking",
                "most tasks solved first, then fewest generated tokens",
            ),
            ("needs", "git, python3 and cargo on this Mac"),
            ("compare", "c lines up every saved run, task by task"),
        ] {
            notes.extend(crate::benchview::legend(k, what, TEXT, w.notes));
        }
        Parts {
            form,
            form_sel,
            queue_title: "Tasks".into(),
            queue_note: format!("{} of {}", ag.ready, ag.tasks.len()),
            queue,
            queue_sel: None,
            help: ag_help(LINES[sel]).into(),
            chart_title,
            chart_note,
            chart: Chart::Tiles(tiles),
            rows_title,
            rows_note,
            rows,
            rows_keep,
            rows_tail: running,
            notes,
            running: b.phase == BenchPhase::Running,
            small_chart: false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lays_out_a_transcript() {
        let log = json!({
            "source": "o/r#1", "test_command": "python3 -m unittest t",
            "result": {"solved": false, "turns": 2, "generated": 1500, "end": "finished", "test": "failed"},
            "messages": [
                {"role": "system", "content": "be good"},
                {"role": "user", "content": "Fix the bug."},
                {"role": "assistant", "content": "", "tool_calls": [
                    {"id": "c", "type": "function", "function": {"name": "read_file", "arguments": "{\"path\":\"a.py\"}"}}]},
                {"role": "tool", "content": (0..20).map(|i| format!("line {i}")).collect::<Vec<_>>().join("\n")},
                {"role": "assistant", "content": "Done."},
            ],
            "changes": "--- a/a.py\n+++ b/a.py\n@@ -1 +1 @@\n-x = 1\n+x = 2\nnew file: b.py\n",
            "test_output": "FAILED (failures=1)",
        });
        let lines = transcript(&log, 80);
        let text: Vec<String> = lines
            .iter()
            .map(|l| l.spans.iter().map(|s| s.content.to_string()).collect())
            .collect();
        let has = |needle: &str| text.iter().any(|t| t.contains(needle));
        assert!(has("o/r#1 · not solved · 2 replies · 1.5K generated"));
        assert!(has("system prompt · 7 characters"));
        assert!(has(" USER") && has("Fix the bug.") && has(" MODEL") && has("Done."));
        assert!(has("⚙ read_file {\"path\":\"a.py\"}"));
        assert!(has("… 5 more lines"), "tool output is cut to 15 lines");
        assert!(has(" CHANGES") && has("+x = 2") && has("new file: b.py"));
        assert!(has(" TESTS · failed") && has("FAILED (failures=1)"));
        let added = lines
            .iter()
            .find(|l| {
                l.spans
                    .first()
                    .is_some_and(|s| s.content.contains("+x = 2"))
            })
            .unwrap();
        assert_eq!(added.style.fg, Some(GOOD));
    }

    #[test]
    fn formats_counts_and_times() {
        assert_eq!(kilo(999), "999");
        assert_eq!(kilo(54_120), "54.1K");
        assert_eq!(kilo(2_345_678), "2.3M");
        assert_eq!(duration(75.0), "1m15s");
        assert_eq!(duration(3900.0), "1h05m");
        assert_eq!(model_name("/m/Qwen.gguf"), "Qwen");
    }
}
