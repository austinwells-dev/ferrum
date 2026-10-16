//! The intelligence side of the Benchmark tab: score a model on the installed
//! question sets, quick (a fixed subset, predicting the full score) or full,
//! and keep the results per model and setup.
use crate::bench::{self, Candidate, Source};
use crate::benchui::{BenchPhase, ago};
use crate::eval::{self, CATEGORIES, EvalEvent, EvalPlan, EvalResult, EvalRunning, Suite, Tally};
use crate::*;
use std::sync::atomic::Ordering;

/// Reasoning budgets offered per question (None: thinking off).
pub const THINKING: [Option<usize>; 4] = [None, Some(1024), Some(2048), Some(4096)];

/// What the Benchmark tab measures.
#[derive(Clone, Copy, PartialEq)]
pub enum Measure {
    Speed,
    Intelligence,
}

pub struct IqState {
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
}

impl IqState {
    pub fn new() -> Self {
        Self {
            suite: Suite::load(&eval::suite_dir()),
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
        }
    }

    /// Ask a running evaluation to stop; returns its thread to join later.
    pub fn stop(&mut self) -> Option<std::thread::JoinHandle<()>> {
        let mut run = self.run.take()?;
        run.stop.store(true, Ordering::SeqCst);
        run.handle.take()
    }
}

#[derive(Clone, Copy, PartialEq)]
enum IqLine {
    Model,
    Measure,
    Size,
    Thinking,
    Draft,
    Run,
}

const LINES: [IqLine; 6] = [
    IqLine::Model,
    IqLine::Measure,
    IqLine::Size,
    IqLine::Thinking,
    IqLine::Draft,
    IqLine::Run,
];

fn score_color(share: f64) -> Color {
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
    let filled = ((share.clamp(0.0, 1.0)) * width as f64).round() as usize;
    vec![
        Span::styled("█".repeat(filled), Style::new().fg(color)),
        Span::styled("░".repeat(width - filled), Style::new().fg(FAINT)),
    ]
}

impl App {
    /// The drafter and depth this Mac's last speed benchmark of the model
    /// recommended, if it still exists.
    fn tuned_draft(&self) -> Option<(Candidate, usize)> {
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
            quick: b.iq.quick,
            thinking: THINKING[b.iq.thinking],
            draft: if b.iq.tuned { self.tuned_draft() } else { None },
            context: 8192,
            kv_cache: self.kv_cache().unwrap_or_default(),
        })
    }

    fn start_iq(&mut self) {
        let Some(plan) = self.iq_plan() else {
            self.status = Some(("no supported model to test".into(), false));
            return;
        };
        let Some(b) = self.bench.as_mut() else { return };
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
        match key.code {
            KeyCode::Esc | KeyCode::Char('q') => return self.leave_bench(),
            KeyCode::Up | KeyCode::Char('k') => iq.sel = iq.sel.saturating_sub(1),
            KeyCode::Down | KeyCode::Char('j') | KeyCode::Tab => {
                iq.sel = (iq.sel + 1).min(LINES.len() - 1)
            }
            KeyCode::Char('r') => start = true,
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
                        b.measure = Measure::Speed;
                        b.sel = 1;
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
            let stale = current.is_some_and(|f| f != r.suite);
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
    fn iq_table(tallies: &[Tally], width: usize) -> Vec<Line<'static>> {
        let bar_w = width.saturating_sub(58).clamp(10, 30);
        tallies
            .iter()
            .zip(CATEGORIES)
            .map(|(t, (_, name))| {
                let mut spans = vec![Span::styled(
                    format!("   {name:<26} {:>3}/{:<3}  ", t.right, t.n),
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
        let mut out: Vec<Line> = vec![title("QUESTION SETS")];
        match &iq.suite {
            Ok(s) => {
                let sizes: Vec<String> = s
                    .sets
                    .iter()
                    .zip(CATEGORIES)
                    .map(|(set, (cat, _))| format!("{cat} {}", set.len()))
                    .collect();
                out.push(Line::styled(
                    format!("   {} · {}", sizes.join(" · "), eval::suite_dir().display()),
                    Style::new().fg(TEXT),
                ));
            }
            Err(e) => {
                out.push(Line::styled(format!("   ✗ {e}"), Style::new().fg(BAD)));
                out.push(Line::styled(
                    "   install them with: python3 tools/eval_import.py (downloads GSM8K, MMLU, IFEval and CRUXEval samples)",
                    Style::new().fg(GOLD),
                ));
            }
        }
        out.push(Line::default());
        let tuned = self.tuned_draft();
        let model = self.models.get(b.model);
        let questions = iq.suite.as_ref().map_or(0, |s| s.questions(iq.quick).len());
        let mut sel_line = 0;
        for (n, &def) in LINES.iter().enumerate() {
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
                IqLine::Size => (
                    "Questions",
                    if iq.quick {
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
                    Style::new().fg(EMBER).add_modifier(Modifier::BOLD).bg(bg),
                ),
                Span::styled("  ", Style::new().bg(bg)),
            ]));
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
        lines.extend(Self::iq_table(&iq.tallies, width));
        lines.push(Line::default());
        for (cat, id, correct, answer) in iq.answers.iter().rev().take(8) {
            lines.push(Line::styled(
                format!(
                    "   {} {:<22} {:<12} {}",
                    if *correct { "✓" } else { "✗" },
                    clip(id, 22),
                    CATEGORIES[*cat].0,
                    clip(answer.as_deref().unwrap_or("no answer found"), 30)
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
        lines.extend(Self::iq_table(&iq.tallies, width));
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
        assert_eq!(score_color(0.2), BAD);
    }
}
