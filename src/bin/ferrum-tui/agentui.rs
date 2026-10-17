//! The agent side of the Benchmark tab: install the ten tasks, run the coding
//! agent on them, and rank saved runs by tasks solved, then tokens spent.
use crate::agentbench::{
    self, AgentEvent, AgentPlan, AgentResult, AgentRunning, Task, TaskResult, Totals,
};
use crate::bench;
use crate::benchui::{BenchPhase, ago};
use crate::evalsets::{InstallEvent, Installing};
use crate::evalui::{Measure, THINKING, bar_ranged, score_color};
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

const LINES: [AgLine; 6] = [
    AgLine::Model,
    AgLine::Measure,
    AgLine::Tasks,
    AgLine::Thinking,
    AgLine::Draft,
    AgLine::Run,
];

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
                } => ag.current = Some((task, turn, generated, prompt, last)),
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

    pub fn agent_results_key(&mut self, key: KeyEvent) {
        match key.code {
            KeyCode::Char('r') | KeyCode::Enter => {
                if let Some(b) = self.bench.as_mut() {
                    b.phase = BenchPhase::Setup;
                }
            }
            KeyCode::Esc | KeyCode::Char('q') => self.leave_bench(),
            _ => {}
        }
    }

    // ---- drawing ----

    /// Saved runs of every model, most solved first, then fewest tokens.
    fn agent_board(&self, b: &BenchState, width: usize) -> Vec<Line<'static>> {
        let rows = agentbench::ranked(&b.ag.results);
        if rows.is_empty() {
            return Vec::new();
        }
        let current = agentbench::tasks_fingerprint();
        let name_w = width.saturating_sub(84).clamp(16, 40);
        let mut lines = vec![
            Line::styled(
                "  RESULTS · every model, most solved first, then fewest tokens",
                Style::new().fg(DIM).add_modifier(Modifier::BOLD),
            ),
            Line::styled(
                format!(
                    "   {:<name_w$} {:>6}  {:<10} {:>9} {:>9} {:>8}  setup",
                    "model", "solved", "", "tok/solve", "generated", "time"
                ),
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
                format!(
                    " {:>9} {:>9} {:>8}  ",
                    t.per_solve().map_or("–".to_string(), kilo),
                    kilo(t.generated),
                    duration(t.secs)
                ),
                Style::new().fg(TEXT),
            ));
            spans.push(Span::styled(
                format!(
                    "thinking {} · {} · {}{}",
                    r.thinking.map_or("off".to_string(), |b| format!("≤{b}")),
                    r.draft,
                    ago(r.when),
                    if stale { " · older tasks" } else { "" }
                ),
                Style::new().fg(if stale { FAINT } else { DIM }),
            ));
            lines.push(Line::from(spans));
        }
        lines
    }

    /// One row per finished task.
    fn agent_task_rows(done: &[TaskResult], width: usize) -> Vec<Line<'static>> {
        let id_w = width.saturating_sub(70).clamp(20, 36);
        let mut lines = vec![Line::styled(
            format!(
                "   {:<id_w$} {:>6} {:>9} {:>9} {:>8}  how it ended",
                "task", "replies", "generated", "read", "time"
            ),
            Style::new().fg(FAINT),
        )];
        for r in done {
            lines.push(Line::styled(
                format!(
                    " {} {:<id_w$} {:>6} {:>9} {:>9} {:>8}  {} · tests {}",
                    if r.solved { "✓" } else { "✗" },
                    clip(&r.id, id_w),
                    r.turns,
                    kilo(r.generated),
                    kilo(r.prompt),
                    duration(r.secs),
                    r.end,
                    r.test
                ),
                Style::new().fg(if r.solved { GOOD } else { BAD }),
            ));
        }
        lines
    }

    pub fn draw_agent_setup(&self, f: &mut Frame, area: Rect, b: &BenchState) {
        let block = panel("Measure agentic coding", true);
        let inner = block.inner(area);
        f.render_widget(block, area);
        let width = inner.width as usize;
        let ag = &b.ag;
        let title = |s: &str| {
            Line::styled(
                format!("  {s}"),
                Style::new().fg(DIM).add_modifier(Modifier::BOLD),
            )
        };
        let tuned = self.tuned_draft();
        let model = self.models.get(b.model);
        let mut out: Vec<Line> = Vec::new();
        let mut sel_line = 0;
        for (n, &def) in LINES.iter().enumerate() {
            match def {
                AgLine::Tasks => {
                    out.push(Line::default());
                    out.push(title("TASKS"));
                }
                AgLine::Thinking => {
                    out.push(Line::default());
                    out.push(title("OPTIONS"));
                }
                AgLine::Run => out.push(Line::default()),
                _ => {}
            }
            let on = n == ag.sel;
            let bg = if on { SELECTED } else { Color::Reset };
            if on {
                sel_line = out.len();
            }
            let (label, value): (&str, String) = match def {
                AgLine::Model => (
                    "Model",
                    format!(
                        "‹ {} ›",
                        model
                            .map(|m| clip(&m.name, 44))
                            .unwrap_or_else(|| "none".into())
                    ),
                ),
                AgLine::Measure => ("Measure", "‹ agent ›".into()),
                AgLine::Tasks => (
                    "Tasks",
                    if ag.install.is_some() {
                        "installing…".into()
                    } else if ag.all_ready() && ag.confirm_remove {
                        "‹ press enter again to remove ›".into()
                    } else if ag.all_ready() {
                        format!("‹ {} installed · enter removes ›", ag.ready)
                    } else if ag.ready > 0 {
                        format!(
                            "‹ {} of {} installed · enter installs the rest ›",
                            ag.ready,
                            ag.tasks.len()
                        )
                    } else {
                        "‹ not installed · enter installs ›".into()
                    },
                ),
                AgLine::Thinking => (
                    "Thinking",
                    match THINKING[ag.thinking] {
                        None => "‹ off ›".into(),
                        Some(t) => format!("‹ on, up to {t} tokens per reply ›"),
                    },
                ),
                AgLine::Draft => (
                    "Drafter",
                    match (&tuned, ag.tuned) {
                        (Some((c, d)), true) => {
                            format!("‹ tuned: {} · depth {d} ›", clip(&c.label(), 30))
                        }
                        (Some(_), false) => "‹ off (tuned one available) ›".into(),
                        (None, _) => "off · run the speed benchmark to tune one".into(),
                    },
                ),
                AgLine::Run => ("▶  Run agent tasks", String::new()),
            };
            let run = def == AgLine::Run;
            let label_w = width.saturating_sub(value.chars().count() + 7);
            out.push(Line::from(vec![
                Span::styled(
                    if on { " ▌ " } else { "   " },
                    Style::new().fg(EMBER).bg(bg),
                ),
                Span::styled(
                    format!("{:<label_w$}", clip(label, label_w)),
                    if run {
                        Style::new()
                            .fg(if on { Color::Black } else { EMBER })
                            .bg(if on { EMBER } else { Color::Reset })
                            .add_modifier(Modifier::BOLD)
                    } else {
                        Style::new().fg(if on { Color::White } else { TEXT }).bg(bg)
                    },
                ),
                Span::styled(
                    format!(" {value} "),
                    Style::new()
                        .fg(if def == AgLine::Tasks && ag.confirm_remove {
                            BAD
                        } else {
                            EMBER
                        })
                        .add_modifier(Modifier::BOLD)
                        .bg(bg),
                ),
                Span::styled("  ", Style::new().bg(bg)),
            ]));
            if def == AgLine::Tasks {
                if ag.install.is_some() {
                    out.push(Line::styled(
                        format!(
                            "   {} {}",
                            spinner(self.elapsed()),
                            clip(&ag.install_stage, width.saturating_sub(6))
                        ),
                        Style::new().fg(EMBER),
                    ));
                }
                if let Some((note, ok)) = &ag.install_note {
                    let text = format!("{} {note}", if *ok { "✓" } else { "✗" });
                    let chars: Vec<char> = text.chars().collect();
                    for (a, z) in wrap_rows(&text, width.saturating_sub(5)) {
                        out.push(Line::styled(
                            format!("   {}", chars[a..z].iter().collect::<String>()),
                            Style::new().fg(if *ok { GOOD } else { BAD }),
                        ));
                    }
                }
                for t in &ag.tasks {
                    let ready = agentbench::tasks_dir().join(&t.id).join("ready").exists();
                    out.push(Line::styled(
                        clip(
                            &format!(
                                "   {} {:<6} {:<36} {}",
                                if ready { "·" } else { " " },
                                t.lang,
                                t.id,
                                t.source()
                            ),
                            width,
                        ),
                        Style::new().fg(if ready { DIM } else { FAINT }),
                    ));
                }
            }
        }
        out.push(Line::default());
        out.push(Line::styled(
            clip(
                &format!(
                    "   each task: up to {} replies and {} min, network off · solved = the fix's tests pass · needs git, python3 and cargo",
                    agentbench::MAX_TURNS,
                    agentbench::TASK_LIMIT.as_secs() / 60
                ),
                width,
            ),
            Style::new().fg(FAINT),
        ));
        out.push(Line::default());
        out.extend(self.agent_board(b, width));
        let h = inner.height as usize;
        let top = (sel_line + 4).saturating_sub(h);
        f.render_widget(
            Paragraph::new(out.into_iter().skip(top).take(h).collect::<Vec<_>>()),
            inner,
        );
    }

    pub fn draw_agent_running(&self, f: &mut Frame, area: Rect, b: &BenchState) {
        let block = panel("Running", true);
        let inner = block.inner(area);
        f.render_widget(block, area);
        let width = inner.width as usize;
        let ag = &b.ag;
        let total = ag.tasks.len();
        let secs = ag.started.elapsed().as_secs();
        let bar_w = width.saturating_sub(40).clamp(10, 60);
        let mut lines = vec![
            Line::from(vec![
                Span::styled(
                    format!(" {} ", spinner(self.elapsed())),
                    Style::new().fg(EMBER),
                ),
                Span::styled(
                    clip(&ag.stage, width.saturating_sub(6)),
                    Style::new().fg(Color::White).add_modifier(Modifier::BOLD),
                ),
            ]),
            Line::from(
                [Span::raw(" ")]
                    .into_iter()
                    .chain(bar_ranged(
                        ag.done.len() as f64 / total.max(1) as f64,
                        0.0,
                        bar_w,
                        EMBER,
                    ))
                    .chain([Span::styled(
                        format!(
                            "  {}/{total} tasks  {}",
                            ag.done.len(),
                            duration(secs as f64)
                        ),
                        Style::new().fg(DIM),
                    )])
                    .collect::<Vec<_>>(),
            ),
        ];
        if let Some(plan) = &ag.plan {
            lines.push(Line::styled(
                format!(
                    " thinking {} · {}{}",
                    plan.thinking.map_or("off".to_string(), |t| format!("≤{t}")),
                    plan.draft_label(),
                    ag.loaded.map_or(String::new(), |(c, w)| format!(
                        " · context {c} · weights {w:.1} GiB"
                    ))
                ),
                Style::new().fg(FAINT),
            ));
        }
        if let Some(e) = &ag.failed {
            lines.push(Line::styled(
                format!(" ✗ {}", clip(e, width.saturating_sub(4))),
                Style::new().fg(BAD),
            ));
        }
        lines.push(Line::default());
        if let Some((task, turn, generated, prompt, last)) = &ag.current {
            let id = ag.tasks.get(*task).map_or("", |t| t.id.as_str());
            lines.push(Line::styled(
                format!(
                    " {id} · reply {turn}/{} · {} generated · {} read · {} of {} min",
                    agentbench::MAX_TURNS,
                    kilo(*generated),
                    kilo(*prompt),
                    duration(ag.task_started.elapsed().as_secs_f64()),
                    agentbench::TASK_LIMIT.as_secs() / 60
                ),
                Style::new().fg(TEXT),
            ));
            lines.push(Line::styled(
                format!("   last: {}", clip(last, width.saturating_sub(10))),
                Style::new().fg(DIM),
            ));
            lines.push(Line::default());
        }
        if !ag.done.is_empty() {
            let t = Totals::of(&ag.done);
            lines.push(Line::styled(
                format!(
                    " so far {}/{} solved · {} generated · {} read",
                    t.solved,
                    t.done,
                    kilo(t.generated),
                    kilo(t.prompt)
                ),
                Style::new()
                    .fg(score_color(t.solved as f64 / t.done.max(1) as f64))
                    .add_modifier(Modifier::BOLD),
            ));
            lines.extend(Self::agent_task_rows(&ag.done, width));
        }
        let h = inner.height as usize;
        f.render_widget(
            Paragraph::new(lines.into_iter().take(h).collect::<Vec<_>>()),
            inner,
        );
    }

    pub fn draw_agent_results(&self, f: &mut Frame, area: Rect, b: &BenchState) {
        let block = panel("Results", true);
        let inner = block.inner(area);
        f.render_widget(block, area);
        let width = inner.width as usize;
        let ag = &b.ag;
        let t = Totals::of(&ag.done);
        let mut lines = vec![Line::from(vec![
            Span::styled(" ★ ", Style::new().fg(EMBER)),
            Span::styled(
                format!("{}/{} solved", t.solved, ag.tasks.len()),
                Style::new()
                    .fg(score_color(t.solved as f64 / ag.tasks.len().max(1) as f64))
                    .add_modifier(Modifier::BOLD),
            ),
            Span::styled(
                format!(
                    "  {} generated tokens{} · {} prompt tokens read · {} replies · {}",
                    kilo(t.generated),
                    t.per_solve()
                        .map_or(String::new(), |p| format!(" ({} per solved task)", kilo(p))),
                    kilo(t.prompt),
                    t.turns,
                    duration(t.secs)
                ),
                Style::new().fg(TEXT),
            ),
        ])];
        if let Some(plan) = &ag.plan {
            lines.push(Line::styled(
                format!(
                    "   thinking {} · {}",
                    plan.thinking
                        .map_or("off".to_string(), |t| format!("up to {t} tokens per reply")),
                    plan.draft_label()
                ),
                Style::new().fg(DIM),
            ));
        }
        if ag.saved {
            lines.push(Line::styled(
                format!("   ✓ saved to {}", agentbench::results_path().display()),
                Style::new().fg(GOOD),
            ));
        } else if ag.failed.is_none() {
            lines.push(Line::styled(
                format!(
                    "   stopped after {} of {} tasks · not saved",
                    ag.done.len(),
                    ag.tasks.len()
                ),
                Style::new().fg(GOLD),
            ));
        }
        if let Some(e) = &ag.failed {
            lines.push(Line::styled(
                "   ✗ the run failed · not saved",
                Style::new().fg(BAD),
            ));
            let chars: Vec<char> = e.chars().collect();
            for (a, z) in wrap_rows(e, width.saturating_sub(6)) {
                lines.push(Line::styled(
                    format!("     {}", chars[a..z].iter().collect::<String>()),
                    Style::new().fg(DIM),
                ));
            }
        }
        lines.push(Line::default());
        lines.extend(Self::agent_task_rows(&ag.done, width));
        lines.push(Line::default());
        lines.extend(self.agent_board(b, width));
        f.render_widget(Paragraph::new(lines), inner);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
