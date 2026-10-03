//! The Benchmark tab: choose a model and what to test, watch the sweep, then
//! save the winning settings as a favorite.
use crate::bench::{self, Candidate, ChunkRow, Event, Machine, Plan, Row, Running, Saved, Source};
use crate::*;

const TOKENS: [usize; 3] = [128, 192, 256];

#[derive(Clone, Copy, PartialEq)]
pub enum BenchPhase {
    Setup,
    Running,
    Results,
}

pub struct BenchState {
    pub phase: BenchPhase,
    pub machine: Machine,
    /// Index into `App::models`.
    pub model: usize,
    pub cands: Vec<Candidate>,
    pub quick: bool,
    pub tokens: usize,
    pub q8: bool,
    pub chunks: bool,
    pub sel: usize,
    run: Option<Running>,
    pub rows: Vec<Row>,
    pub chunk_rows: Vec<ChunkRow>,
    pub skipped: Vec<(String, String)>,
    pub stage: String,
    pub started: Instant,
    pub loaded: Option<(usize, f64, f64)>,
    pub expected: usize,
    pub stopping: bool,
    pub saved: Vec<Saved>,
    /// What was saved as a favorite (shown in the results).
    pub note: Option<String>,
}

#[derive(Clone, Copy, PartialEq)]
enum Line_ {
    Model,
    Cand(usize),
    Sweep,
    Tokens,
    Q8,
    Chunks,
    Run,
}

/// Append `text` wrapped to `width` columns, each row indented by `indent` spaces.
fn push_wrapped(
    lines: &mut Vec<Line<'static>>,
    text: &str,
    indent: usize,
    width: usize,
    style: Style,
) {
    let chars: Vec<char> = text.chars().collect();
    for (a, b) in wrap_rows(text, width.saturating_sub(indent + 1)) {
        let part: String = chars[a..b].iter().collect();
        lines.push(Line::styled(format!("{}{part}", " ".repeat(indent)), style));
    }
}

pub fn ago(secs: u64) -> String {
    let d = bench::now().saturating_sub(secs);
    match d {
        0..=89 => "just now".into(),
        90..=5399 => format!("{} min ago", d / 60),
        5400..=129_599 => format!("{} h ago", d / 3600),
        _ => format!("{} days ago", d / 86_400),
    }
}

impl App {
    fn runnable_models(&self) -> Vec<usize> {
        (0..self.models.len())
            .filter(|&i| self.models[i].problem.is_none())
            .collect()
    }

    /// Baseline, MTP (when the GGUF has a working head) and every drafter that fits.
    fn bench_candidates(&self, model: &Model, q8: bool) -> Vec<Candidate> {
        let mut out = vec![Candidate {
            source: Source::Baseline,
            quant: "q4_0",
            on: true,
        }];
        if model.mtp {
            out.push(Candidate {
                source: Source::Mtp,
                quant: "q4_0",
                on: true,
            });
        }
        for d in &self.drafters {
            let Some(config) = &d.config else { continue };
            if drafter_problem(d, model).is_some() {
                continue;
            }
            for quant in if q8 {
                vec!["q4_0", "q8_0"]
            } else {
                vec!["q4_0"]
            } {
                out.push(Candidate {
                    source: Source::Drafter {
                        path: d.path.clone(),
                        label: d.label.clone(),
                        kind: d.kind,
                        max_depth: config.max_drafts(),
                    },
                    quant,
                    on: quant == "q4_0",
                });
            }
        }
        out
    }

    pub fn open_bench(&mut self) {
        self.reap();
        let runnable = self.runnable_models();
        let model = runnable
            .iter()
            .copied()
            .find(|&i| i == self.cursor)
            .or(runnable.first().copied())
            .unwrap_or(0);
        let cands = self
            .models
            .get(model)
            .map(|m| self.bench_candidates(m, false))
            .unwrap_or_default();
        self.bench = Some(BenchState {
            phase: BenchPhase::Setup,
            machine: Machine::detect(),
            model,
            cands,
            quick: true,
            tokens: 1,
            q8: false,
            chunks: false,
            sel: 0,
            run: None,
            rows: Vec::new(),
            chunk_rows: Vec::new(),
            skipped: Vec::new(),
            stage: String::new(),
            started: Instant::now(),
            loaded: None,
            expected: 0,
            stopping: false,
            saved: bench::load_saved(),
            note: None,
        });
        self.screen = Screen::Bench;
    }

    pub fn leave_bench(&mut self) {
        if let Some(mut b) = self.bench.take()
            && let Some(mut run) = b.run.take()
        {
            run.stop.store(true, std::sync::atomic::Ordering::SeqCst);
            if let Some(h) = run.handle.take() {
                self.closing.push(h);
            }
        }
        self.screen = Screen::Home;
    }

    fn bench_lines(&self) -> Vec<Line_> {
        let Some(b) = &self.bench else {
            return Vec::new();
        };
        let mut rows = vec![Line_::Model];
        rows.extend((0..b.cands.len()).map(Line_::Cand));
        rows.extend([
            Line_::Sweep,
            Line_::Tokens,
            Line_::Q8,
            Line_::Chunks,
            Line_::Run,
        ]);
        rows
    }

    fn bench_plan(&self) -> Option<Plan> {
        let b = self.bench.as_ref()?;
        let model = self.models.get(b.model)?;
        Some(Plan {
            model: model.path.clone(),
            candidates: b.cands.clone(),
            quick: b.quick,
            tokens: TOKENS[b.tokens],
            chunks: b.chunks,
            context: 8192,
        })
    }

    /// How many (candidate, depth) rows the run will produce.
    fn expected_rows(plan: &Plan) -> usize {
        plan.candidates
            .iter()
            .filter(|c| c.on)
            .map(|c| match c.source {
                Source::Baseline => 1,
                _ => bench::depth_list(c.max_depth(), plan.quick).len(),
            })
            .sum()
    }

    fn start_bench(&mut self) {
        let Some(plan) = self.bench_plan() else {
            self.status = Some(("no supported model to benchmark".into(), false));
            return;
        };
        self.reap();
        let expected = Self::expected_rows(&plan) + usize::from(plan.chunks) * 4;
        match bench::spawn(plan) {
            Ok(run) => {
                if let Some(b) = self.bench.as_mut() {
                    b.run = Some(run);
                    b.phase = BenchPhase::Running;
                    b.rows.clear();
                    b.chunk_rows.clear();
                    b.skipped.clear();
                    b.loaded = None;
                    b.stopping = false;
                    b.started = Instant::now();
                    b.expected = expected;
                    b.stage = "starting".into();
                    b.note = None;
                }
            }
            Err(e) => self.status = Some((e, false)),
        }
    }

    pub fn bench_tick(&mut self) {
        let Some(b) = self.bench.as_mut() else { return };
        let Some(run) = b.run.as_ref() else { return };
        let mut done = false;
        while let Ok(event) = run.rx.try_recv() {
            match event {
                Event::Stage(s) => b.stage = s,
                Event::Loaded {
                    context,
                    weights_gib,
                    draft_gib,
                } => b.loaded = Some((context, weights_gib, draft_gib)),
                Event::Row(r) => b.rows.push(r),
                Event::Chunk(c) => b.chunk_rows.push(c),
                Event::Skipped(label, why) => b.skipped.push((label, why)),
                Event::Done => done = true,
            }
        }
        if done {
            b.phase = BenchPhase::Results;
            if let Some(mut run) = b.run.take()
                && let Some(h) = run.handle.take()
            {
                let _ = h.join();
            }
            self.finish_bench();
        }
    }

    /// Remember the winner for this Mac and model.
    fn finish_bench(&mut self) {
        let Some(b) = self.bench.as_mut() else { return };
        let (Some(best), Some(base)) = (bench::best(&b.rows), bench::baseline(&b.rows)) else {
            return;
        };
        let chunk = bench::best_chunk(&b.chunk_rows);
        let model = self
            .models
            .get(b.model)
            .map(|m| m.path.display().to_string())
            .unwrap_or_default();
        let best_label = match best.depth {
            Some(d) => format!("{} · depth {d} · {}", best.label, best.quant),
            None => "no drafter".to_string(),
        };
        bench::record(
            &mut b.saved,
            Saved {
                machine: b.machine.label(),
                model,
                when: bench::now(),
                best: best_label,
                speedup: best.tps / base.tps.max(1e-9),
                values: bench::settings_for(best, chunk),
            },
        );
        bench::write_saved(&b.saved);
    }

    fn save_tuned(&mut self, serve: bool) {
        let Some(b) = self.bench.as_mut() else { return };
        let Some(best) = bench::best(&b.rows) else {
            return;
        };
        let Some(model) = self.models.get(b.model) else {
            return;
        };
        let chunk = serve.then(|| bench::best_chunk(&b.chunk_rows)).flatten();
        let name = if serve {
            format!("{} · tuned server", model.name)
        } else {
            format!("{} · tuned", model.name)
        };
        let fav = Fav {
            name: name.clone(),
            model: model.path.display().to_string(),
            mode: if serve { Mode::Serve } else { Mode::Chat },
            values: bench::settings_for(best, chunk),
        };
        match self.favs.iter().position(|f| f.name == name) {
            Some(i) => self.favs[i] = fav,
            None => self.favs.push(fav),
        }
        b.note = Some(format!(
            "saved favorite {name:?} · find it under {}",
            if serve { "Serve" } else { "Chat and Agents" }
        ));
        self.save();
    }

    pub fn bench_key(&mut self, key: KeyEvent) {
        let phase = match &self.bench {
            Some(b) => b.phase,
            None => return self.go_home(),
        };
        match phase {
            BenchPhase::Setup => self.bench_setup_key(key),
            BenchPhase::Running => {
                if matches!(key.code, KeyCode::Esc | KeyCode::Char('q'))
                    && let Some(b) = self.bench.as_mut()
                    && let Some(run) = &b.run
                {
                    run.stop.store(true, std::sync::atomic::Ordering::SeqCst);
                    b.stopping = true;
                }
            }
            BenchPhase::Results => match key.code {
                KeyCode::Char('s') => self.save_tuned(false),
                KeyCode::Char('v') => self.save_tuned(true),
                KeyCode::Char('r') | KeyCode::Enter => {
                    if let Some(b) = self.bench.as_mut() {
                        b.phase = BenchPhase::Setup;
                    }
                }
                KeyCode::Esc | KeyCode::Char('q') => self.leave_bench(),
                _ => {}
            },
        }
    }

    fn bench_setup_key(&mut self, key: KeyEvent) {
        let lines = self.bench_lines();
        let Some(b) = self.bench.as_mut() else { return };
        b.sel = b.sel.min(lines.len() - 1);
        let line = lines[b.sel];
        let runnable: Vec<usize> = (0..self.models.len())
            .filter(|&i| self.models[i].problem.is_none())
            .collect();
        let mut rebuild = false;
        let mut start = false;
        match key.code {
            KeyCode::Esc | KeyCode::Char('q') => return self.leave_bench(),
            KeyCode::Up | KeyCode::Char('k') => b.sel = b.sel.saturating_sub(1),
            KeyCode::Down | KeyCode::Char('j') | KeyCode::Tab => {
                b.sel = (b.sel + 1).min(lines.len() - 1)
            }
            KeyCode::Char('r') => start = true,
            KeyCode::Left | KeyCode::Right | KeyCode::Enter | KeyCode::Char(' ') => {
                let step: i32 = if key.code == KeyCode::Left { -1 } else { 1 };
                match line {
                    Line_::Model => {
                        if let Some(at) = runnable.iter().position(|&i| i == b.model) {
                            let n = runnable.len() as i32;
                            b.model = runnable[(at as i32 + step).rem_euclid(n) as usize];
                            rebuild = true;
                        }
                    }
                    Line_::Cand(i) => {
                        if b.cands[i].source != Source::Baseline {
                            b.cands[i].on = !b.cands[i].on;
                        }
                    }
                    Line_::Sweep => b.quick = !b.quick,
                    Line_::Tokens => {
                        b.tokens = (b.tokens as i32 + step).rem_euclid(TOKENS.len() as i32) as usize
                    }
                    Line_::Q8 => {
                        b.q8 = !b.q8;
                        rebuild = true;
                    }
                    Line_::Chunks => b.chunks = !b.chunks,
                    Line_::Run => start = true,
                }
            }
            _ => {}
        }
        if rebuild {
            let model = b.model;
            let q8 = b.q8;
            let cands = self
                .models
                .get(model)
                .map(|m| self.bench_candidates(m, q8))
                .unwrap_or_default();
            if let Some(b) = self.bench.as_mut() {
                b.cands = cands;
            }
            let last = self.bench_lines().len() - 1;
            if let Some(b) = self.bench.as_mut() {
                b.sel = b.sel.min(last);
            }
        }
        if start {
            self.start_bench();
        }
    }

    // ---- drawing ----

    pub fn draw_bench(&self, f: &mut Frame) {
        let Some(b) = self.bench.as_ref() else { return };
        let [head, body, footer] = Layout::vertical([
            Constraint::Length(2),
            Constraint::Min(8),
            Constraint::Length(1),
        ])
        .areas(f.area());
        let model = self.models.get(b.model);
        let (pill_text, pill_color) = match (b.phase, b.stopping) {
            (BenchPhase::Setup, _) => ("SETUP", GOLD),
            (BenchPhase::Running, true) => ("STOPPING", GOLD),
            (BenchPhase::Running, false) => ("RUNNING", EMBER),
            (BenchPhase::Results, _) => ("RESULTS", GOOD),
        };
        let name = model.map(|m| m.name.as_str()).unwrap_or("");
        header(
            f,
            head,
            &["Benchmark", name],
            vec![pill(pill_text, pill_color)],
        );
        let area = centered(body, 104, body.height);
        match b.phase {
            BenchPhase::Setup => self.draw_bench_setup(f, area, b),
            BenchPhase::Running => self.draw_bench_running(f, area, b),
            BenchPhase::Results => self.draw_bench_results(f, area, b),
        }
        let keys: &[(&str, &str)] = match b.phase {
            BenchPhase::Setup => &[
                ("↑↓", "move"),
                ("←→/space", "change"),
                ("r", "run"),
                ("esc", "back"),
            ],
            BenchPhase::Running => &[("esc", "stop and show what finished")],
            BenchPhase::Results => &[
                ("s", "save as favorite"),
                ("v", "save for Serve"),
                ("r", "run again"),
                ("esc", "home"),
            ],
        };
        keys_footer(f, footer, keys, &self.status);
    }

    fn draw_bench_setup(&self, f: &mut Frame, area: Rect, b: &BenchState) {
        let block = panel("Tune this Mac", true);
        let inner = block.inner(area);
        f.render_widget(block, area);
        let width = inner.width as usize;
        let title = |s: &str| {
            Line::styled(
                format!("  {s}"),
                Style::new().fg(DIM).add_modifier(Modifier::BOLD),
            )
        };
        let lines_def = self.bench_lines();
        let sel = b.sel.min(lines_def.len() - 1);
        let mut out: Vec<Line> = Vec::new();
        let mut sel_line = 0;
        out.push(title("THIS MAC"));
        out.push(Line::styled(
            format!("   {}", b.machine.label()),
            Style::new().fg(TEXT),
        ));
        out.push(Line::default());
        let model = self.models.get(b.model);
        for (n, &def) in lines_def.iter().enumerate() {
            match def {
                Line_::Cand(0) => {
                    out.push(Line::default());
                    out.push(title("WHAT TO TEST"));
                }
                Line_::Sweep => {
                    out.push(Line::default());
                    out.push(title("OPTIONS"));
                }
                Line_::Run => out.push(Line::default()),
                _ => {}
            }
            let on = n == sel;
            let bg = if on { SELECTED } else { Color::Reset };
            if on {
                sel_line = out.len();
            }
            let (label, value, value_style): (String, String, Style) = match def {
                Line_::Model => (
                    "Model".into(),
                    format!(
                        "‹ {} ›",
                        model
                            .map(|m| clip(&m.name, 44))
                            .unwrap_or_else(|| "none".into())
                    ),
                    Style::new().fg(EMBER).add_modifier(Modifier::BOLD),
                ),
                Line_::Cand(i) => {
                    let c = &b.cands[i];
                    let locked = c.source == Source::Baseline;
                    let mark = if locked {
                        "[✓]"
                    } else if c.on {
                        "[x]"
                    } else {
                        "[ ]"
                    };
                    let depth = if c.max_depth() > 0 {
                        format!("up to {} drafts · {}", c.max_depth(), c.quant)
                    } else {
                        "always measured first".to_string()
                    };
                    (
                        format!("{mark} {}", clip(&c.label(), width.saturating_sub(46))),
                        depth,
                        Style::new().fg(if c.on { DIM } else { FAINT }),
                    )
                }
                Line_::Sweep => (
                    "Depth sweep".into(),
                    if b.quick {
                        "‹ quick: a few depths ›"
                    } else {
                        "‹ full: every depth ›"
                    }
                    .into(),
                    Style::new().fg(EMBER).add_modifier(Modifier::BOLD),
                ),
                Line_::Tokens => (
                    "Tokens per prompt".into(),
                    format!("‹ {} ›", TOKENS[b.tokens]),
                    Style::new().fg(EMBER).add_modifier(Modifier::BOLD),
                ),
                Line_::Q8 => (
                    "Also test q8_0 drafters".into(),
                    if b.q8 { "‹ yes ›" } else { "‹ no ›" }.into(),
                    Style::new().fg(EMBER).add_modifier(Modifier::BOLD),
                ),
                Line_::Chunks => (
                    "Tune prefill chunk".into(),
                    if b.chunks {
                        "‹ yes (4 more loads) ›"
                    } else {
                        "‹ no ›"
                    }
                    .into(),
                    Style::new().fg(EMBER).add_modifier(Modifier::BOLD),
                ),
                Line_::Run => ("▶  Run benchmark".into(), String::new(), Style::new()),
            };
            let label_w = width.saturating_sub(value.chars().count() + 7);
            let run = def == Line_::Run;
            out.push(Line::from(vec![
                Span::styled(
                    if on { " ▌ " } else { "   " },
                    Style::new().fg(EMBER).bg(bg),
                ),
                Span::styled(
                    format!("{:<label_w$}", clip(&label, label_w)),
                    if run {
                        Style::new()
                            .fg(if on { Color::Black } else { EMBER })
                            .bg(if on { EMBER } else { Color::Reset })
                            .add_modifier(Modifier::BOLD)
                    } else {
                        Style::new().fg(if on { Color::White } else { TEXT }).bg(bg)
                    },
                ),
                Span::styled(format!(" {value} "), value_style.bg(bg)),
                Span::styled("  ", Style::new().bg(bg)),
            ]));
        }
        out.push(Line::default());
        if let Some(plan) = self.bench_plan() {
            out.push(Line::styled(
                format!(
                    "   about {} min · greedy decoding on {} prompts, results are checked against the no-drafter output",
                    plan.estimate_secs().div_ceil(60),
                    bench::PROMPTS.len()
                ),
                Style::new().fg(FAINT),
            ));
        }
        if let Some(m) = model {
            let path = m.path.display().to_string();
            if let Some(s) = b
                .saved
                .iter()
                .find(|s| s.machine == b.machine.label() && s.model == path)
            {
                out.push(Line::styled(
                    format!(
                        "   last result here ({}): {} → {:.2}×",
                        ago(s.when),
                        s.best,
                        s.speedup
                    ),
                    Style::new().fg(GOOD),
                ));
            }
        }
        let h = inner.height as usize;
        let top = (sel_line + 4).saturating_sub(h);
        f.render_widget(
            Paragraph::new(out.into_iter().skip(top).take(h).collect::<Vec<_>>()),
            inner,
        );
    }

    fn result_table(
        &self,
        rows: &[&Row],
        base: Option<f64>,
        best_row: Option<&Row>,
        width: usize,
    ) -> Vec<Line<'static>> {
        let top = rows.iter().map(|r| r.tps).fold(1.0, f64::max);
        let bar_w = 14usize;
        let label_w = width.saturating_sub(bar_w + 56).clamp(16, 44);
        let mut lines = vec![Line::styled(
            format!(
                "   {:<label_w$} {:>5} {:>5} {:>8}  {:<bar_w$} {:>6} {:>6} {:>5}",
                "config", "depth", "quant", "tok/s", "", "×base", "accept", "same"
            ),
            Style::new().fg(DIM).add_modifier(Modifier::BOLD),
        )];
        for r in rows {
            let is_best = best_row.is_some_and(|b| std::ptr::eq(*r, b));
            let filled = ((r.tps / top) * bar_w as f64).round() as usize;
            let speed = base
                .map(|b| format!("{:.2}×", r.tps / b.max(1e-9)))
                .unwrap_or_default();
            let color = if is_best { EMBER } else { GOLD };
            lines.push(Line::from(vec![
                Span::styled(if is_best { " ★ " } else { "   " }, Style::new().fg(EMBER)),
                Span::styled(
                    format!("{:<label_w$}", clip(&r.label, label_w)),
                    Style::new()
                        .fg(if is_best { Color::White } else { TEXT })
                        .add_modifier(if is_best {
                            Modifier::BOLD
                        } else {
                            Modifier::empty()
                        }),
                ),
                Span::styled(
                    format!(
                        " {:>5} {:>5} ",
                        r.depth.map_or("–".to_string(), |d| d.to_string()),
                        if r.quant.is_empty() { "–" } else { &r.quant }
                    ),
                    Style::new().fg(DIM),
                ),
                Span::styled(
                    format!("{:>8.1}", r.tps),
                    Style::new().fg(color).add_modifier(Modifier::BOLD),
                ),
                Span::styled("  ", Style::new()),
                Span::styled("█".repeat(filled), Style::new().fg(color)),
                Span::styled(
                    "░".repeat(bar_w - filled.min(bar_w)),
                    Style::new().fg(FAINT),
                ),
                Span::styled(
                    format!(" {:>6}", speed),
                    Style::new().fg(if r.depth.is_some() { TEXT } else { FAINT }),
                ),
                Span::styled(
                    format!(
                        " {:>6}",
                        r.accept.map_or("–".to_string(), |a| format!("{a:.2}"))
                    ),
                    Style::new().fg(DIM),
                ),
                Span::styled(
                    match r.same {
                        Some(true) => "     ✓".to_string(),
                        Some(false) => "     ≠".to_string(),
                        None => "      ".to_string(),
                    },
                    Style::new().fg(if r.same == Some(false) { BAD } else { GOOD }),
                ),
            ]));
        }
        lines
    }

    fn draw_bench_running(&self, f: &mut Frame, area: Rect, b: &BenchState) {
        let block = panel("Running", true);
        let inner = block.inner(area);
        f.render_widget(block, area);
        let t = self.elapsed();
        let secs = b.started.elapsed().as_secs();
        let done = b.rows.len() + b.chunk_rows.len();
        let frac = (done as f64 / b.expected.max(1) as f64).clamp(0.0, 1.0);
        let bar_w = (inner.width as usize).saturating_sub(24).clamp(10, 60);
        let filled = (frac * bar_w as f64) as usize;
        let mut lines = vec![
            Line::from(vec![
                Span::styled(format!(" {} ", spinner(t)), Style::new().fg(EMBER)),
                Span::styled(
                    clip(&b.stage, inner.width as usize - 6),
                    Style::new().fg(Color::White).add_modifier(Modifier::BOLD),
                ),
            ]),
            Line::from(vec![
                Span::styled(" ", Style::new()),
                Span::styled("█".repeat(filled), Style::new().fg(EMBER)),
                Span::styled("░".repeat(bar_w - filled), Style::new().fg(FAINT)),
                Span::styled(
                    format!("  {done}/{}  {}m{:02}s", b.expected, secs / 60, secs % 60),
                    Style::new().fg(DIM),
                ),
            ]),
        ];
        if let Some((ctx, w, d)) = b.loaded {
            lines.push(Line::styled(
                format!(
                    " context {ctx} · weights {w:.1} GiB{}",
                    if d > 0.0 {
                        format!(" · drafter {d:.1} GiB")
                    } else {
                        String::new()
                    }
                ),
                Style::new().fg(FAINT),
            ));
        }
        for (label, why) in &b.skipped {
            lines.push(Line::styled(
                format!(" ✗ {} skipped: {}", label, clip(why, 70)),
                Style::new().fg(BAD),
            ));
        }
        lines.push(Line::default());
        let refs: Vec<&Row> = b.rows.iter().collect();
        let base = bench::baseline(&b.rows).map(|r| r.tps);
        lines.extend(self.result_table(&refs, base, None, inner.width as usize));
        let h = inner.height as usize;
        let skip = lines.len().saturating_sub(h);
        f.render_widget(
            Paragraph::new(lines.into_iter().skip(skip).collect::<Vec<_>>()),
            inner,
        );
    }

    fn draw_bench_results(&self, f: &mut Frame, area: Rect, b: &BenchState) {
        let block = panel("Results", true);
        let inner = block.inner(area);
        f.render_widget(block, area);
        let width = inner.width as usize;
        let mut lines: Vec<Line> = Vec::new();
        let best = bench::best(&b.rows);
        let base = bench::baseline(&b.rows).map(|r| r.tps);
        match (best, base) {
            (Some(best), Some(base)) => {
                lines.push(Line::from(vec![
                    Span::styled(" ★ ", Style::new().fg(EMBER)),
                    Span::styled(
                        match best.depth {
                            Some(d) => format!("{} · depth {d} · {}", best.label, best.quant),
                            None => "no drafter".to_string(),
                        },
                        Style::new().fg(Color::White).add_modifier(Modifier::BOLD),
                    ),
                ]));
                lines.push(Line::styled(
                    if best.depth.is_some() {
                        format!(
                            "   {:.1} tok/s, {:.2}× the {:.1} tok/s of plain decoding on {}",
                            best.tps,
                            best.tps / base.max(1e-9),
                            base,
                            b.machine.label()
                        )
                    } else {
                        format!("   no drafter beats plain decoding here ({base:.1} tok/s): keep drafting off")
                    },
                    Style::new().fg(GOOD),
                ));
                if let Some(chunk) = bench::best_chunk(&b.chunk_rows) {
                    let tps = b
                        .chunk_rows
                        .iter()
                        .find(|c| c.chunk == chunk)
                        .map_or(0.0, |c| c.prefill_tps);
                    lines.push(Line::styled(
                        format!("   prefill chunk {chunk} ({tps:.0} tok/s prompt processing)"),
                        Style::new().fg(TEXT),
                    ));
                }
                if b.rows.iter().any(|r| r.same == Some(false)) {
                    push_wrapped(
                        &mut lines,
                        "≠ rows produced different text than plain decoding (greedy), so they are not recommended: batched verification can flip near-ties.",
                        3,
                        width,
                        Style::new().fg(GOLD),
                    );
                }
            }
            _ => lines.push(Line::styled(
                " not enough finished to recommend anything (the baseline needs to complete)",
                Style::new().fg(GOLD),
            )),
        }
        if let Some(note) = &b.note {
            lines.push(Line::styled(format!("   ✓ {note}"), Style::new().fg(GOOD)));
        }
        for (label, why) in &b.skipped {
            lines.push(Line::styled(
                format!("   ✗ {label} was skipped"),
                Style::new().fg(BAD),
            ));
            push_wrapped(&mut lines, why, 5, width, Style::new().fg(DIM));
        }
        lines.push(Line::default());
        let mut sorted: Vec<&Row> = b.rows.iter().collect();
        sorted.sort_by(|a, c| c.tps.total_cmp(&a.tps));
        lines.extend(self.result_table(&sorted, base, best, width));
        if !b.chunk_rows.is_empty() {
            lines.push(Line::default());
            lines.push(Line::styled(
                " PREFILL CHUNK",
                Style::new().fg(DIM).add_modifier(Modifier::BOLD),
            ));
            for c in &b.chunk_rows {
                lines.push(Line::styled(
                    format!(
                        "   {:>5} rows per pass   {:>7.0} tok/s",
                        c.chunk, c.prefill_tps
                    ),
                    Style::new().fg(TEXT),
                ));
            }
        }
        f.render_widget(Paragraph::new(lines), inner);
    }
}
