//! The intelligence side of the Benchmark tab: score a model on the installed
//! question sets, quick (a fixed subset, predicting the full score) or full,
//! and keep the results per model and setup.
use crate::agentui::{Feed, feed_lines, feed_push, shorten, stream};
use crate::bench::{self, Candidate, Source};
use crate::benchui::{BenchPhase, ago};
use crate::benchview::{Chart, ChartBar, Parts, Widths, form_row};
use crate::eval::{
    self, CATEGORIES, Difficulty, EvalEvent, EvalPlan, EvalResult, EvalRunning, Suite, Tally,
};
use crate::evalcompare::Compare;
use crate::evalsets::{self, InstallEvent, Installing};
use crate::layout::fit_w;
use crate::*;
use std::sync::atomic::Ordering;

/// Reasoning budgets offered per question (None: thinking off).
pub const THINKING: [Option<usize>; 4] = [None, Some(1024), Some(2048), Some(4096)];

/// What the Benchmark tab measures.
#[derive(Clone, Copy, PartialEq)]
pub enum Measure {
    Speed,
    Intelligence,
    Agent,
}

pub struct IqState {
    pub difficulty: Difficulty,
    pub suite: Result<Suite, String>,
    pub quick: bool,
    /// Index into `THINKING`.
    pub thinking: usize,
    /// Use the drafter this Mac's speed benchmark recommended.
    pub tuned: bool,
    pub sel: usize,
    run: Option<EvalRunning>,
    pub plan: Option<EvalPlan>,
    pub tallies: Vec<Tally>,
    /// Newest last: (category, question id, correct, extracted answer).
    pub answers: Vec<(usize, String, bool, Option<String>)>,
    pub expected: usize,
    pub failed: Option<String>,
    pub stage: String,
    pub started: Instant,
    pub loaded: Option<(usize, f64)>,
    pub stopping: bool,
    pub results: Vec<EvalResult>,
    /// The finished run was complete and saved.
    pub saved: bool,
    install: Option<Installing>,
    /// Latest download step while installing.
    pub install_stage: String,
    /// How the last install or removal went (text, succeeded).
    pub install_note: Option<(String, bool)>,
    /// Enter was pressed once on installed sets: the next press removes them.
    pub confirm_remove: bool,
    /// The Compare view, when open.
    pub compare: Option<Compare>,
    /// The questions as they are asked, with the model's replies streaming.
    pub feed: Vec<Feed>,
}

impl IqState {
    pub fn new() -> Self {
        Self {
            difficulty: Difficulty::default(),
            suite: Suite::load(&eval::suite_dir(Difficulty::default())),
            quick: true,
            thinking: 0,
            tuned: false,
            sel: 0,
            run: None,
            plan: None,
            tallies: Vec::new(),
            answers: Vec::new(),
            expected: 0,
            failed: None,
            stage: String::new(),
            started: Instant::now(),
            loaded: None,
            stopping: false,
            results: eval::load_results(&eval::results_path()),
            saved: false,
            install: None,
            install_stage: String::new(),
            install_note: None,
            confirm_remove: false,
            compare: None,
            feed: Vec::new(),
        }
    }

    pub fn installing(&self) -> bool {
        self.install.is_some()
    }

    /// Ask a running evaluation and download to stop; returns their threads
    /// to join later.
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

    /// Pick up download progress; reload the sets when it finishes.
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
        self.install_note = Some(match result {
            Ok(n) => (format!("installed {n} questions"), true),
            Err(e) => (format!("install failed: {e}"), false),
        });
        self.reload();
    }

    fn reload(&mut self) {
        self.suite = Suite::load(&eval::suite_dir(self.difficulty));
    }

    /// Switch between the standard and hard sets (not while installing).
    fn toggle_difficulty(&mut self) {
        if self.installing() {
            return;
        }
        self.difficulty = self.difficulty.other();
        self.install_note = None;
        self.reload();
    }

    /// Enter on the question-sets line: install, or remove after a confirm.
    fn sets_action(&mut self) {
        if self.installing() {
            return;
        }
        if self.suite.is_err() {
            self.confirm_remove = false;
            self.install_note = None;
            self.install_stage = "starting".into();
            match evalsets::spawn_install(eval::suite_dir(self.difficulty), self.difficulty) {
                Ok(i) => self.install = Some(i),
                Err(e) => self.install_note = Some((e, false)),
            }
        } else if self.confirm_remove {
            self.confirm_remove = false;
            self.install_note = Some(match evalsets::remove(&eval::suite_dir(self.difficulty)) {
                Ok(()) => ("removed the question sets".into(), true),
                Err(e) => (format!("could not remove them: {e}"), false),
            });
            self.reload();
        } else {
            self.confirm_remove = true;
        }
    }
}

#[derive(Clone, Copy, PartialEq)]
enum IqLine {
    Model,
    Measure,
    Difficulty,
    Sets,
    Size,
    Thinking,
    Draft,
    Run,
}

/// What a setup row does.
fn iq_help(line: IqLine) -> &'static str {
    match line {
        IqLine::Model => {
            "The model to test. Results are kept per model and setup, so runs compare."
        }
        IqLine::Measure => {
            "Intelligence asks graded questions and checks each answer with a program. m switches to speed or agentic coding."
        }
        IqLine::Difficulty => {
            "Hard sets separate current models; standard ones are mostly solved by them."
        }
        IqLine::Sets => {
            "The question files, downloaded once from Hugging Face. Enter installs them, or (twice) removes them."
        }
        IqLine::Size => {
            "Quick asks a fixed subset and predicts the full score with a 95% range; full asks everything."
        }
        IqLine::Thinking => {
            "Let the model reason before it answers, up to this many tokens per question. Slower, usually better."
        }
        IqLine::Draft => {
            "Use the drafter the speed benchmark picked for this Mac, so the test finishes sooner."
        }
        IqLine::Run => {
            "Start the test. Esc while it runs stops it; a stopped run is shown but not saved."
        }
    }
}

const LINES: [IqLine; 8] = [
    IqLine::Model,
    IqLine::Measure,
    IqLine::Difficulty,
    IqLine::Sets,
    IqLine::Size,
    IqLine::Thinking,
    IqLine::Draft,
    IqLine::Run,
];

pub fn score_color(share: f64) -> Color {
    if share >= 0.75 {
        GOOD
    } else if share >= 0.5 {
        GOLD
    } else {
        BAD
    }
}

pub fn bar_ranged(share: f64, margin: f64, width: usize, color: Color) -> Vec<Span<'static>> {
    let cells = |x: f64| ((x.clamp(0.0, 1.0)) * width as f64).round() as usize;
    let (solid, upper) = if margin > 0.0 {
        (cells(share - margin), cells(share + margin))
    } else {
        let s = cells(share);
        (s, s)
    };
    vec![
        Span::styled("█".repeat(solid), Style::new().fg(color)),
        Span::styled("▒".repeat(upper - solid), Style::new().fg(color)),
        Span::styled("░".repeat(width - upper), Style::new().fg(FAINT)),
    ]
}

/// An answer as shown: IFEval answers are which rules held ("101").
fn shown_answer(cat: usize, answer: Option<&str>) -> String {
    match answer {
        None => "no answer found".into(),
        Some(a) if CATEGORIES[cat].0 == "instructions" => {
            let held = a.chars().filter(|&c| c == '1').count();
            format!("{held}/{} rules held", a.len())
        }
        Some(a) => a.to_string(),
    }
}

impl App {
    /// The drafter and depth this Mac's last speed benchmark of the model
    /// recommended, if it still exists.
    pub fn tuned_draft(&self) -> Option<(Candidate, usize)> {
        let b = self.bench.as_ref()?;
        let model = self.models.get(b.model)?.path.display().to_string();
        let s = b
            .saved
            .iter()
            .find(|s| s.machine == b.machine.label() && s.model == model)?;
        let text = |k: &str| s.values.get(k).and_then(|v| v.as_str());
        let depth: usize = text("draft_max")?.parse().ok()?;
        let quant = if text("draft_quant") == Some("q8_0") {
            "q8_0"
        } else {
            "q4_0"
        };
        let source = match text("draft")? {
            "off" => return None,
            "mtp" => Source::Mtp,
            "lookup" => Source::Lookup,
            path => {
                let path = PathBuf::from(path);
                if !path.exists() {
                    return None;
                }
                let known = self.drafters.iter().find(|d| d.path == path);
                Source::Drafter {
                    label: known.map_or_else(
                        || {
                            path.file_name()
                                .map(|n| n.to_string_lossy().to_string())
                                .unwrap_or_default()
                        },
                        |d| d.label.clone(),
                    ),
                    kind: known.map_or("drafter", |d| d.kind),
                    max_depth: depth,
                    path,
                }
            }
        };
        Some((
            Candidate {
                source,
                quant,
                on: true,
            },
            depth,
        ))
    }

    fn iq_plan(&self) -> Option<EvalPlan> {
        let b = self.bench.as_ref()?;
        let model = self.models.get(b.model)?;
        Some(EvalPlan {
            model: model.path.clone(),
            difficulty: b.iq.difficulty,
            quick: b.iq.quick,
            thinking: THINKING[b.iq.thinking],
            draft: if b.iq.tuned { self.tuned_draft() } else { None },
            context: b.iq.suite.as_ref().map_or(8192, |s| {
                s.context_needed(b.iq.quick, THINKING[b.iq.thinking])
            }),
            kv_cache: self.kv_cache().unwrap_or_default(),
        })
    }

    fn start_iq(&mut self) {
        if !self.make_room("the benchmark", false, false) {
            return;
        }
        let Some(plan) = self.iq_plan() else {
            self.status = Some(("no supported model to test".into(), false));
            return;
        };
        let Some(b) = self.bench.as_mut() else { return };
        if b.iq.installing() {
            self.status = Some((
                "wait for the question sets to finish installing".into(),
                false,
            ));
            return;
        }
        let suite = match &b.iq.suite {
            Ok(s) => s.clone(),
            Err(e) => {
                self.status = Some((e.clone(), false));
                return;
            }
        };
        self.reap();
        let Some(b) = self.bench.as_mut() else { return };
        let iq = &mut b.iq;
        iq.tallies = suite.tallies();
        iq.expected = suite.questions(plan.quick).len();
        match eval::spawn(plan.clone(), suite) {
            Ok(run) => {
                iq.run = Some(run);
                iq.plan = Some(plan);
                iq.answers.clear();
                iq.feed.clear();
                iq.failed = None;
                iq.loaded = None;
                iq.stopping = false;
                iq.saved = false;
                iq.started = Instant::now();
                iq.stage = "starting".into();
                b.phase = BenchPhase::Running;
            }
            Err(e) => self.status = Some((e, false)),
        }
    }

    pub fn iq_tick(&mut self) {
        let Some(b) = self.bench.as_mut() else { return };
        let iq = &mut b.iq;
        iq.poll_install();
        let Some(run) = iq.run.as_ref() else { return };
        let mut done = false;
        while let Ok(event) = run.rx.try_recv() {
            match event {
                EvalEvent::Stage(s) => iq.stage = s,
                EvalEvent::Loaded {
                    context,
                    weights_gib,
                } => iq.loaded = Some((context, weights_gib)),
                EvalEvent::Answered {
                    cat,
                    id,
                    correct,
                    answer,
                } => {
                    if let Some(t) = iq.tallies.get_mut(cat) {
                        t.n += 1;
                        t.right += usize::from(correct);
                    }
                    feed_push(
                        &mut iq.feed,
                        Feed::Verdict {
                            correct,
                            answer: shown_answer(cat, answer.as_deref()),
                        },
                    );
                    iq.answers.push((cat, id, correct, answer));
                }
                EvalEvent::Question { cat, id, text } => feed_push(
                    &mut iq.feed,
                    Feed::Question {
                        title: format!(
                            "{} · {id} · question {} of {}",
                            CATEGORIES[cat].1,
                            iq.answers.len() + 1,
                            iq.expected
                        ),
                        text: shorten(&text),
                    },
                ),
                EvalEvent::Delta { reasoning, text } => stream(&mut iq.feed, reasoning, text),
                EvalEvent::Failed(e) => iq.failed = Some(e),
                EvalEvent::Done => done = true,
            }
        }
        if !done {
            return;
        }
        if let Some(mut run) = iq.run.take()
            && let Some(h) = run.handle.take()
        {
            let _ = h.join();
        }
        b.phase = BenchPhase::Results;
        let complete = iq.failed.is_none() && iq.answers.len() == iq.expected;
        if let (true, Some(plan), Ok(suite)) = (complete, &iq.plan, &iq.suite) {
            let result = EvalResult {
                model: plan.model.display().to_string(),
                difficulty: plan.difficulty,
                when: bench::now(),
                quick: plan.quick,
                thinking: plan.thinking,
                draft: plan.draft_label(),
                kv_cache: format!("{:?}", plan.kv_cache),
                suite: suite.fingerprint.clone(),
                tallies: iq.tallies.clone(),
                wrong: iq
                    .answers
                    .iter()
                    .filter(|a| !a.2)
                    .map(|a| a.1.clone())
                    .collect(),
            };
            eval::record(&mut iq.results, result);
            eval::write_results(&eval::results_path(), &iq.results);
            iq.saved = true;
        }
    }

    /// Stop the running evaluation (it reports what finished).
    pub fn iq_stop(&mut self) {
        if let Some(b) = self.bench.as_mut()
            && let Some(run) = &b.iq.run
        {
            run.stop.store(true, Ordering::SeqCst);
            b.iq.stopping = true;
        }
    }

    pub fn iq_setup_key(&mut self, key: KeyEvent) {
        let runnable: Vec<usize> = (0..self.models.len())
            .filter(|&i| self.models[i].problem.is_none())
            .collect();
        let has_tuned = self.tuned_draft().is_some();
        let Some(b) = self.bench.as_mut() else { return };
        let iq = &mut b.iq;
        iq.sel = iq.sel.min(LINES.len() - 1);
        let mut model_changed = false;
        let mut start = false;
        let removing = iq.confirm_remove;
        iq.confirm_remove = false;
        match key.code {
            KeyCode::Esc | KeyCode::Char('q') => return self.leave_bench(),
            KeyCode::Up | KeyCode::Char('k') => iq.sel = iq.sel.saturating_sub(1),
            KeyCode::Down | KeyCode::Char('j') | KeyCode::Tab => {
                iq.sel = (iq.sel + 1).min(LINES.len() - 1)
            }
            KeyCode::Char('r') => start = true,
            KeyCode::Char('c') => iq.compare = Some(Compare::new(iq.difficulty)),
            KeyCode::Left | KeyCode::Right | KeyCode::Enter | KeyCode::Char(' ') => {
                let step: i32 = if key.code == KeyCode::Left { -1 } else { 1 };
                match LINES[iq.sel] {
                    IqLine::Model => {
                        if let Some(at) = runnable.iter().position(|&i| i == b.model) {
                            let n = runnable.len() as i32;
                            b.model = runnable[(at as i32 + step).rem_euclid(n) as usize];
                            model_changed = true;
                        }
                    }
                    IqLine::Measure => {
                        b.measure = Measure::Agent;
                        b.ag.sel = 1;
                    }
                    IqLine::Difficulty => iq.toggle_difficulty(),
                    IqLine::Sets => {
                        iq.confirm_remove = removing;
                        if key.code != KeyCode::Left {
                            iq.sets_action();
                        }
                    }
                    IqLine::Size => iq.quick = !iq.quick,
                    IqLine::Thinking => {
                        iq.thinking =
                            (iq.thinking as i32 + step).rem_euclid(THINKING.len() as i32) as usize
                    }
                    IqLine::Draft => iq.tuned = !iq.tuned && has_tuned,
                    IqLine::Run => start = true,
                }
            }
            _ => {}
        }
        if model_changed {
            self.bench_model_changed();
            if let Some(b) = self.bench.as_mut() {
                b.iq.tuned = false;
            }
        }
        if start {
            self.start_iq();
        }
    }

    pub fn iq_results_key(&mut self, key: KeyEvent) {
        match key.code {
            KeyCode::Char('c') => {
                if let Some(b) = self.bench.as_mut() {
                    b.iq.compare = Some(Compare::new(b.iq.difficulty));
                }
            }
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

    /// Saved results for the selected model, newest first.
    fn iq_history(&self, b: &BenchState, width: usize, most: usize) -> Vec<Line<'static>> {
        let Some(model) = self.models.get(b.model) else {
            return Vec::new();
        };
        let path = model.path.display().to_string();
        let current = b.iq.suite.as_ref().ok().map(|s| s.fingerprint.as_str());
        let mut lines = Vec::new();
        for r in b.iq.results.iter().filter(|r| r.model == path).take(most) {
            let Some(s) = r.score() else { continue };
            let stale = r.difficulty == b.iq.difficulty && current.is_some_and(|f| f != r.suite);
            lines.push(Line::from(vec![
                Span::styled(
                    format!("   {:>10}  ", s.label()),
                    Style::new()
                        .fg(score_color(s.share))
                        .add_modifier(Modifier::BOLD),
                ),
                Span::styled(
                    clip(&r.setup_label(), width.saturating_sub(40)),
                    Style::new().fg(TEXT),
                ),
                Span::styled(
                    format!(
                        "  {}{}",
                        ago(r.when),
                        if stale { " · older questions" } else { "" }
                    ),
                    Style::new().fg(FAINT),
                ),
            ]));
        }
        if !lines.is_empty() {
            lines.insert(
                0,
                Line::styled(
                    "  PREVIOUS RESULTS",
                    Style::new().fg(DIM).add_modifier(Modifier::BOLD),
                ),
            );
        }
        lines
    }

    /// The intelligence measure's panes.
    pub fn iq_parts(&self, b: &BenchState, w: Widths) -> Parts {
        let iq = &b.iq;
        let setup = b.phase == BenchPhase::Setup;
        let tuned = self.tuned_draft();
        let model = self.models.get(b.model);
        let questions = iq.suite.as_ref().map_or(0, |s| s.questions(iq.quick).len());
        let sel = iq.sel.min(LINES.len() - 1);
        let mut form = Vec::new();
        let mut form_sel = None;
        for (n, &def) in LINES.iter().enumerate() {
            let on = n == sel && setup;
            let (label, value): (&str, String) = match def {
                IqLine::Model => (
                    "Model",
                    model
                        .map(|m| m.name.clone())
                        .unwrap_or_else(|| "none".into()),
                ),
                IqLine::Measure => ("Measure", "intelligence".into()),
                IqLine::Difficulty => (
                    "Difficulty",
                    match iq.difficulty {
                        Difficulty::Hard => "hard".into(),
                        Difficulty::Standard => "standard · easy now".into(),
                    },
                ),
                IqLine::Sets => (
                    "Question sets",
                    match &iq.suite {
                        _ if iq.installing() => "installing…".into(),
                        Ok(_) if iq.confirm_remove => "enter again removes".into(),
                        Ok(s) => {
                            format!("installed · {}", s.sets.iter().map(Vec::len).sum::<usize>())
                        }
                        Err(_) => "enter installs".into(),
                    },
                ),
                IqLine::Size => (
                    "Questions",
                    if iq.suite.is_err() {
                        "install the sets first".into()
                    } else if iq.quick {
                        format!("quick · {questions}")
                    } else {
                        format!("full · {questions}")
                    },
                ),
                IqLine::Thinking => (
                    "Thinking",
                    match THINKING[iq.thinking] {
                        None => "off".into(),
                        Some(t) => format!("≤ {t} tokens"),
                    },
                ),
                IqLine::Draft => (
                    "Drafter",
                    match (&tuned, iq.tuned) {
                        (Some((c, d)), true) => {
                            format!("tuned · {} d{d}", crate::benchui::short_drafter(&c.label()))
                        }
                        (Some(_), false) => "off · tuned available".into(),
                        (None, _) => "off · none tuned yet".into(),
                    },
                ),
                IqLine::Run => ("▶ Run intelligence test", String::new()),
            };
            if def == IqLine::Run {
                form.push(Line::default());
            }
            if on {
                form_sel = Some(form.len());
            }
            let value = clip(&value, w.form.saturating_sub(label.len() + 8));
            let warn = def == IqLine::Sets && iq.confirm_remove;
            form.push(form_row(on, true, w.form, label, &value, warn, !setup));
        }
        if let (Some(plan), Ok(suite)) = (self.iq_plan(), &iq.suite) {
            form.push(Line::styled(
                format!(
                    "   about {} min · checked by a program",
                    plan.estimate_secs(suite).div_ceil(60)
                ),
                Style::new().fg(FAINT),
            ));
        }

        // The queue: the seven question sets, and how each is going.
        let difficulty = iq.plan.as_ref().map_or(iq.difficulty, |p| p.difficulty);
        let mut queue = Vec::new();
        if iq.installing() {
            queue.push(Line::styled(
                format!(
                    " {} {}",
                    spinner(self.elapsed()),
                    clip(&iq.install_stage, w.form.saturating_sub(4))
                ),
                Style::new().fg(EMBER),
            ));
        }
        if let Some((note, ok)) = &iq.install_note {
            queue.extend(crate::layout::wrapped(
                &format!("{} {note}", if *ok { "✓" } else { "✗" }),
                1,
                w.form,
                Style::new().fg(if *ok { GOOD } else { BAD }),
            ));
        }
        for (i, (_, name)) in CATEGORIES.iter().enumerate() {
            let right = match (setup, iq.tallies.get(i), &iq.suite) {
                (false, Some(t), _) if t.n > 0 => format!("{}/{}", t.right, t.n),
                (false, _, _) => "·".into(),
                (true, _, Ok(s)) => format!("{} q", s.sets.get(i).map_or(0, Vec::len)),
                (true, _, Err(_)) => "–".into(),
            };
            let color = match iq.tallies.get(i) {
                Some(t) if !setup && t.n > 0 => score_color(t.share()),
                _ => DIM,
            };
            let mark = if iq.suite.is_ok() { "✓" } else { "·" };
            queue.push(Line::from(vec![
                Span::styled(
                    format!(" {mark} "),
                    Style::new().fg(if iq.suite.is_ok() { GOOD } else { FAINT }),
                ),
                Span::styled(
                    fit_w(name, w.form.saturating_sub(text_width(&right) + 6)),
                    Style::new().fg(TEXT),
                ),
                Span::styled(
                    format!(" {right}"),
                    Style::new().fg(color).add_modifier(Modifier::BOLD),
                ),
            ]));
            queue.push(Line::styled(
                format!("   {}", difficulty.datasets()[i]),
                Style::new().fg(FAINT),
            ));
        }

        // The chart: the leaderboard before a run, the categories after.
        let (chart_title, chart_note, chart) = if setup {
            let path = model.map(|m| m.path.display().to_string());
            let mut best: Vec<(&EvalResult, eval::Score)> = Vec::new();
            for r in iq.results.iter().filter(|r| r.difficulty == iq.difficulty) {
                let Some(sc) = r.score() else { continue };
                match best.iter_mut().find(|(x, _)| x.model == r.model) {
                    Some(slot) if sc.share > slot.1.share => *slot = (r, sc),
                    Some(_) => {}
                    None => best.push((r, sc)),
                }
            }
            best.sort_by(|a, c| c.1.share.total_cmp(&a.1.share));
            let bars: Vec<ChartBar> = best
                .iter()
                .map(|(r, sc)| ChartBar {
                    label: crate::evalcompare::model_name(&r.model),
                    sub: String::new(),
                    share: sc.share,
                    margin: if sc.predicted { sc.margin } else { 0.0 },
                    color: score_color(sc.share),
                    value: format!("{} · {}", sc.label(), ago(r.when)),
                    value_color: score_color(sc.share),
                    star: path.as_deref() == Some(r.model.as_str()),
                    pending: false,
                })
                .collect();
            let chart = if bars.is_empty() {
                Chart::Empty(crate::layout::wrapped(
                    "No scores yet. r runs a quick test on the model on the left: a fixed subset of every set that predicts the full score.",
                    2,
                    w.chart,
                    Style::new().fg(DIM),
                ))
            } else {
                Chart::Bars(bars)
            };
            (
                format!("Best score per model · {}", iq.difficulty.key()),
                "▒ 95% range of a quick run".to_string(),
                chart,
            )
        } else {
            let bars: Vec<ChartBar> = iq
                .tallies
                .iter()
                .enumerate()
                .map(|(i, t)| ChartBar {
                    label: CATEGORIES[i].1.to_string(),
                    sub: String::new(),
                    share: t.share(),
                    margin: 0.0,
                    color: score_color(t.share()),
                    value: if t.n == 0 {
                        "not asked".into()
                    } else {
                        format!("{:>3.0}%  {}/{}", t.share() * 100.0, t.right, t.n)
                    },
                    value_color: if t.n == 0 {
                        FAINT
                    } else {
                        score_color(t.share())
                    },
                    star: false,
                    pending: false,
                })
                .collect();
            let title = match eval::score(&iq.tallies) {
                Some(sc) if b.phase == BenchPhase::Results => format!(
                    "★ {} {}",
                    sc.label(),
                    if sc.predicted {
                        format!(
                            "predicted for all {} questions",
                            iq.tallies.iter().map(|t| t.total).sum::<usize>()
                        )
                    } else {
                        "on every question".into()
                    }
                ),
                Some(sc) => format!("Score by category · so far {}", sc.label()),
                None => "Score by category".into(),
            };
            let note = match &iq.plan {
                Some(p) => format!(
                    "{} · thinking {} · {}",
                    if p.quick { "quick" } else { "full" },
                    p.thinking.map_or("off".to_string(), |t| format!("≤{t}")),
                    p.draft_label()
                ),
                None => String::new(),
            };
            (title, note, Chart::Bars(bars))
        };

        // Rows: this model's earlier results, or the answers as they come.
        let (rows_title, rows_note, rows) = if setup {
            let mut rows = self.iq_history(b, w.rows, 40);
            if rows.is_empty() {
                rows.push(Line::styled(
                    " no results for this model yet",
                    Style::new().fg(FAINT),
                ));
            } else {
                rows.remove(0);
            }
            (
                "This model's results".to_string(),
                "newest first".to_string(),
                rows,
            )
        } else if b.phase == BenchPhase::Running {
            // The live feed: each question, then the reply as it streams.
            let mut rows = feed_lines(&iq.feed, w.rows);
            rows.push(Line::styled(
                format!(" {}", spinner(self.elapsed())),
                Style::new().fg(EMBER),
            ));
            let right = iq.answers.iter().filter(|a| a.2).count();
            (
                format!(
                    "Live · question {} of {}",
                    iq.answers.len() + 1,
                    iq.expected
                ),
                format!("{right} right of {} so far", iq.answers.len()),
                rows,
            )
        } else {
            let answer_w = w.rows.saturating_sub(42).max(12);
            let rows: Vec<Line> = iq
                .answers
                .iter()
                .rev()
                .map(|(cat, id, correct, answer)| {
                    Line::styled(
                        format!(
                            " {} {:<22} {:<13} {}",
                            if *correct { "✓" } else { "✗" },
                            clip(id, 22),
                            CATEGORIES[*cat].0,
                            clip(&shown_answer(*cat, answer.as_deref()), answer_w)
                        ),
                        Style::new().fg(if *correct { DIM } else { BAD }),
                    )
                })
                .collect();
            let right = iq.answers.iter().filter(|a| a.2).count();
            (
                "Answers".to_string(),
                format!("{right} right of {} · newest first", iq.answers.len()),
                rows,
            )
        };

        let mut notes: Vec<Line> = Vec::new();
        if let Some(e) = &iq.failed {
            notes.push(Line::styled(
                " ✗ the run failed · not saved",
                Style::new().fg(BAD),
            ));
            notes.extend(crate::layout::wrapped(e, 3, w.notes, Style::new().fg(DIM)));
            notes.push(Line::default());
        } else if b.phase == BenchPhase::Results {
            notes.push(if iq.saved {
                Line::styled(" ✓ saved", Style::new().fg(GOOD))
            } else {
                Line::styled(
                    format!(
                        " stopped after {} of {} · not saved",
                        iq.answers.len(),
                        iq.expected
                    ),
                    Style::new().fg(GOLD),
                )
            });
            notes.push(Line::default());
        }
        for (k, what) in [
            (
                "quick",
                "a fixed subset of every set; the full score is predicted with a 95% range (the ±)",
            ),
            ("full", "every question; the score is exact"),
            ("hard", "sets that still separate current models"),
            (
                "graded",
                "each answer is checked by a program: exact match, test cases or rule checks",
            ),
            ("compare", "c lines up every saved result, any model"),
        ] {
            notes.extend(crate::benchview::legend(k, what, TEXT, w.notes));
        }
        Parts {
            form,
            form_sel,
            queue_title: "Question sets".into(),
            queue_note: iq.difficulty.key().into(),
            queue,
            queue_sel: None,
            help: iq_help(LINES[sel]).into(),
            chart_title,
            chart_note,
            chart,
            rows_title,
            rows_note,
            rows,
            rows_keep: None,
            rows_tail: b.phase == BenchPhase::Running,
            notes,
            running: b.phase == BenchPhase::Running,
            small_chart: b.phase == BenchPhase::Running,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bars_fill_to_the_share() {
        assert_eq!(score_color(0.8), GOOD);
        let ranged = bar_ranged(0.5, 0.2, 10, GOOD)
            .iter()
            .map(|s| s.content.to_string())
            .collect::<String>();
        assert_eq!(ranged, "███▒▒▒▒░░░");
        assert_eq!(shown_answer(2, Some("101")), "2/3 rules held");
        assert_eq!(shown_answer(2, Some("1")), "1/1 rules held");
        assert_eq!(shown_answer(0, Some("42")), "42");
        assert_eq!(shown_answer(1, None), "no answer found");
        assert_eq!(score_color(0.2), BAD);
    }
}
