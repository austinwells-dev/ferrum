//! The intelligence side of the Benchmark tab: score a model on the installed
//! question sets, quick (a fixed subset, predicting the full score) or full,
//! and keep the results per model and setup.
use crate::bench::{self, Candidate, Source};
use crate::benchui::{BenchPhase, ago};
use crate::eval::{
    self, CATEGORIES, Difficulty, EvalEvent, EvalPlan, EvalResult, EvalRunning, Suite, Tally,
};
use crate::evalcompare::Compare;
use crate::evalsets::{self, InstallEvent, Installing};
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

/// `width`-cell bar filled to `share`.
fn bar(share: f64, width: usize, color: Color) -> Vec<Span<'static>> {
    bar_ranged(share, 0.0, width, color)
}

/// A bar solid up to `share - margin`, shaded (`▒`) across the range
/// `share ± margin`, empty after.
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
                    iq.answers.push((cat, id, correct, answer));
                }
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
    fn iq_history(&self, b: &BenchState, width: usize) -> Vec<Line<'static>> {
        let Some(model) = self.models.get(b.model) else {
            return Vec::new();
        };
        let path = model.path.display().to_string();
        let current = b.iq.suite.as_ref().ok().map(|s| s.fingerprint.as_str());
        let mut lines = Vec::new();
        for r in b.iq.results.iter().filter(|r| r.model == path).take(8) {
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

    /// One row per category: name, right of asked, bar, share.
    pub fn iq_table(tallies: &[Tally], difficulty: Difficulty, width: usize) -> Vec<Line<'static>> {
        let bar_w = width.saturating_sub(66).clamp(10, 30);
        tallies
            .iter()
            .enumerate()
            .map(|(i, t)| {
                let mut spans = vec![Span::styled(
                    format!(
                        "   {:<34} {:>3}/{:<3}  ",
                        difficulty.cat_name(i),
                        t.right,
                        t.n
                    ),
                    Style::new().fg(TEXT),
                )];
                if t.n == 0 {
                    spans.push(Span::styled("not asked", Style::new().fg(FAINT)));
                } else {
                    spans.extend(bar(t.share(), bar_w, score_color(t.share())));
                    spans.push(Span::styled(
                        format!(" {:>4.0}%", t.share() * 100.0),
                        Style::new().fg(score_color(t.share())),
                    ));
                }
                Line::from(spans)
            })
            .collect()
    }

    pub fn draw_iq_setup(&self, f: &mut Frame, area: Rect, b: &BenchState) {
        let block = panel("Measure intelligence", true);
        let inner = block.inner(area);
        f.render_widget(block, area);
        let width = inner.width as usize;
        let iq = &b.iq;
        let title = |s: &str| {
            Line::styled(
                format!("  {s}"),
                Style::new().fg(DIM).add_modifier(Modifier::BOLD),
            )
        };
        // What sits under the question-sets line.
        let mut sets_info: Vec<Line> = Vec::new();
        match &iq.suite {
            _ if iq.installing() => sets_info.push(Line::styled(
                format!(
                    "   {} {}",
                    spinner(self.elapsed()),
                    clip(&iq.install_stage, width.saturating_sub(6))
                ),
                Style::new().fg(EMBER),
            )),
            Ok(s) => {
                let sizes: Vec<String> = s
                    .sets
                    .iter()
                    .zip(CATEGORIES)
                    .map(|(set, (cat, _))| format!("{cat} {}", set.len()))
                    .collect();
                sets_info.push(Line::styled(
                    clip(
                        &format!(
                            "   {} · {}",
                            sizes.join(" · "),
                            eval::suite_dir(iq.difficulty).display()
                        ),
                        width,
                    ),
                    Style::new().fg(FAINT),
                ));
            }
            Err(_) => sets_info.push(Line::styled(
                clip(
                    &format!(
                        "   {} questions each from {}, from Hugging Face",
                        evalsets::PER_SET,
                        iq.difficulty.datasets().join(", ")
                    ),
                    width,
                ),
                Style::new().fg(FAINT),
            )),
        }
        if let Some((note, ok)) = &iq.install_note {
            // Errors carry advice: wrap them instead of cutting them off.
            let text = format!("{} {note}", if *ok { "✓" } else { "✗" });
            let chars: Vec<char> = text.chars().collect();
            for (a, z) in wrap_rows(&text, width.saturating_sub(5)) {
                sets_info.push(Line::styled(
                    format!("   {}", chars[a..z].iter().collect::<String>()),
                    Style::new().fg(if *ok { GOOD } else { BAD }),
                ));
            }
        }
        let mut out: Vec<Line> = Vec::new();
        let tuned = self.tuned_draft();
        let model = self.models.get(b.model);
        let questions = iq.suite.as_ref().map_or(0, |s| s.questions(iq.quick).len());
        let mut sel_line = 0;
        for (n, &def) in LINES.iter().enumerate() {
            if def == IqLine::Difficulty {
                out.push(Line::default());
                out.push(title("QUESTION SETS"));
            }
            if def == IqLine::Size {
                out.push(Line::default());
                out.push(title("OPTIONS"));
            }
            if def == IqLine::Run {
                out.push(Line::default());
            }
            let on = n == iq.sel;
            let bg = if on { SELECTED } else { Color::Reset };
            if on {
                sel_line = out.len();
            }
            let (label, value): (&str, String) = match def {
                IqLine::Model => (
                    "Model",
                    format!(
                        "‹ {} ›",
                        model
                            .map(|m| clip(&m.name, 44))
                            .unwrap_or_else(|| "none".into())
                    ),
                ),
                IqLine::Measure => ("Measure", "‹ intelligence ›".into()),
                IqLine::Difficulty => (
                    "Difficulty",
                    match iq.difficulty {
                        Difficulty::Hard => "‹ hard ›".into(),
                        Difficulty::Standard => "‹ standard (easy for current models) ›".into(),
                    },
                ),
                IqLine::Sets => (
                    "Question sets",
                    match &iq.suite {
                        _ if iq.installing() => "installing…".into(),
                        Ok(_) if iq.confirm_remove => "‹ press enter again to remove ›".into(),
                        Ok(s) => format!(
                            "‹ installed · {} questions · enter removes ›",
                            s.sets.iter().map(Vec::len).sum::<usize>()
                        ),
                        Err(_) => "‹ not installed · enter installs ›".into(),
                    },
                ),
                IqLine::Size => (
                    "Questions",
                    if iq.suite.is_err() {
                        "install the question sets first".into()
                    } else if iq.quick {
                        format!("‹ quick: {questions}, predicts the full score ›")
                    } else {
                        format!("‹ full: all {questions} ›")
                    },
                ),
                IqLine::Thinking => (
                    "Thinking",
                    match THINKING[iq.thinking] {
                        None => "‹ off ›".into(),
                        Some(t) => format!("‹ on, up to {t} tokens ›"),
                    },
                ),
                IqLine::Draft => (
                    "Drafter",
                    match (&tuned, iq.tuned) {
                        (Some((c, d)), true) => {
                            format!("‹ tuned: {} · depth {d} ›", clip(&c.label(), 30))
                        }
                        (Some(_), false) => "‹ off (tuned one available) ›".into(),
                        (None, _) => "off · run the speed benchmark to tune one".into(),
                    },
                ),
                IqLine::Run => ("▶  Run intelligence test", String::new()),
            };
            let run = def == IqLine::Run;
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
                        .fg(if def == IqLine::Sets && iq.confirm_remove {
                            BAD
                        } else {
                            EMBER
                        })
                        .add_modifier(Modifier::BOLD)
                        .bg(bg),
                ),
                Span::styled("  ", Style::new().bg(bg)),
            ]));
            if def == IqLine::Sets {
                out.append(&mut sets_info);
            }
        }
        out.push(Line::default());
        if let (Some(plan), Ok(suite)) = (self.iq_plan(), &iq.suite) {
            out.push(Line::styled(
                format!(
                    "   about {} min · greedy decoding · each answer is checked by a program, not by a model",
                    plan.estimate_secs(suite).div_ceil(60)
                ),
                Style::new().fg(FAINT),
            ));
        }
        out.push(Line::default());
        out.extend(self.iq_history(b, width));
        let h = inner.height as usize;
        let top = (sel_line + 4).saturating_sub(h);
        f.render_widget(
            Paragraph::new(out.into_iter().skip(top).take(h).collect::<Vec<_>>()),
            inner,
        );
    }

    pub fn draw_iq_running(&self, f: &mut Frame, area: Rect, b: &BenchState) {
        let block = panel("Running", true);
        let inner = block.inner(area);
        f.render_widget(block, area);
        let width = inner.width as usize;
        let iq = &b.iq;
        let done = iq.answers.len();
        let secs = iq.started.elapsed().as_secs();
        let eta =
            (done > 0).then(|| secs * (iq.expected.saturating_sub(done)) as u64 / done as u64);
        let frac = done as f64 / iq.expected.max(1) as f64;
        let bar_w = width.saturating_sub(40).clamp(10, 60);
        let mut lines = vec![
            Line::from(vec![
                Span::styled(
                    format!(" {} ", spinner(self.elapsed())),
                    Style::new().fg(EMBER),
                ),
                Span::styled(
                    clip(&iq.stage, width.saturating_sub(6)),
                    Style::new().fg(Color::White).add_modifier(Modifier::BOLD),
                ),
            ]),
            Line::from(
                [Span::styled(" ", Style::new())]
                    .into_iter()
                    .chain(bar(frac, bar_w, EMBER))
                    .chain([Span::styled(
                        format!(
                            "  {done}/{}  {}m{:02}s{}",
                            iq.expected,
                            secs / 60,
                            secs % 60,
                            eta.map_or(String::new(), |e| format!(
                                " · ~{} min left",
                                e.div_ceil(60)
                            ))
                        ),
                        Style::new().fg(DIM),
                    )])
                    .collect::<Vec<_>>(),
            ),
        ];
        if let Some(plan) = &iq.plan {
            lines.push(Line::styled(
                format!(
                    " {} · thinking {} · {}{}",
                    if plan.quick { "quick" } else { "full" },
                    plan.thinking.map_or("off".to_string(), |t| format!("≤{t}")),
                    plan.draft_label(),
                    iq.loaded.map_or(String::new(), |(c, w)| format!(
                        " · context {c} · weights {w:.1} GiB"
                    ))
                ),
                Style::new().fg(FAINT),
            ));
        }
        if let Some(e) = &iq.failed {
            lines.push(Line::styled(
                format!(" ✗ {}", clip(e, width.saturating_sub(4))),
                Style::new().fg(BAD),
            ));
        }
        lines.push(Line::default());
        if let Some(s) = eval::score(&iq.tallies) {
            lines.push(Line::styled(
                format!(" so far {}", s.label()),
                Style::new()
                    .fg(score_color(s.share))
                    .add_modifier(Modifier::BOLD),
            ));
        }
        lines.extend(Self::iq_table(
            &iq.tallies,
            iq.plan.as_ref().map_or(iq.difficulty, |p| p.difficulty),
            width,
        ));
        lines.push(Line::default());
        for (cat, id, correct, answer) in iq.answers.iter().rev().take(8) {
            lines.push(Line::styled(
                format!(
                    "   {} {:<22} {:<12} {}",
                    if *correct { "✓" } else { "✗" },
                    clip(id, 22),
                    CATEGORIES[*cat].0,
                    clip(&shown_answer(*cat, answer.as_deref()), 30)
                ),
                Style::new().fg(if *correct { DIM } else { BAD }),
            ));
        }
        let h = inner.height as usize;
        f.render_widget(
            Paragraph::new(lines.into_iter().take(h).collect::<Vec<_>>()),
            inner,
        );
    }

    pub fn draw_iq_results(&self, f: &mut Frame, area: Rect, b: &BenchState) {
        let block = panel("Results", true);
        let inner = block.inner(area);
        f.render_widget(block, area);
        let width = inner.width as usize;
        let iq = &b.iq;
        let mut lines: Vec<Line> = Vec::new();
        let asked = iq.answers.len();
        match eval::score(&iq.tallies) {
            Some(s) => {
                lines.push(Line::from(vec![
                    Span::styled(" ★ ", Style::new().fg(EMBER)),
                    Span::styled(
                        s.label(),
                        Style::new()
                            .fg(score_color(s.share))
                            .add_modifier(Modifier::BOLD),
                    ),
                    Span::styled(
                        if s.predicted {
                            let total: usize = iq.tallies.iter().map(|t| t.total).sum();
                            format!(
                                "  predicted for all {total} questions from {asked} (95% range)"
                            )
                        } else {
                            format!("  actual, all {asked} questions")
                        },
                        Style::new().fg(TEXT),
                    ),
                ]));
                if let Some(plan) = &iq.plan {
                    lines.push(Line::styled(
                        format!(
                            "   thinking {} · {} · mean of the category scores",
                            plan.thinking
                                .map_or("off".to_string(), |t| format!("up to {t} tokens")),
                            plan.draft_label()
                        ),
                        Style::new().fg(DIM),
                    ));
                }
            }
            None => lines.push(Line::styled(
                " no question was answered",
                Style::new().fg(GOLD),
            )),
        }
        if iq.saved {
            lines.push(Line::styled(
                format!("   ✓ saved to {}", eval::results_path().display()),
                Style::new().fg(GOOD),
            ));
        } else if iq.failed.is_none() {
            lines.push(Line::styled(
                format!(
                    "   stopped after {asked} of {} questions · not saved",
                    iq.expected
                ),
                Style::new().fg(GOLD),
            ));
        }
        if let Some(e) = &iq.failed {
            lines.push(Line::styled(
                "   ✗ the run failed · not saved",
                Style::new().fg(BAD),
            ));
            for (a, z) in wrap_rows(e, width.saturating_sub(6)) {
                let chars: Vec<char> = e.chars().collect();
                lines.push(Line::styled(
                    format!("     {}", chars[a..z].iter().collect::<String>()),
                    Style::new().fg(DIM),
                ));
            }
        }
        lines.push(Line::default());
        lines.extend(Self::iq_table(
            &iq.tallies,
            iq.plan.as_ref().map_or(iq.difficulty, |p| p.difficulty),
            width,
        ));
        lines.push(Line::default());
        lines.extend(self.iq_history(b, width));
        f.render_widget(Paragraph::new(lines), inner);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bars_fill_to_the_share() {
        let text = |share| {
            bar(share, 10, GOOD)
                .iter()
                .map(|s| s.content.to_string())
                .collect::<String>()
        };
        assert_eq!(text(0.5), "█████░░░░░");
        assert_eq!(text(1.2), "██████████");
        assert_eq!(text(-1.0), "░░░░░░░░░░");
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
