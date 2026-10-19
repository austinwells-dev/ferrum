//! The Benchmark tab: choose a model and what to test, watch the sweep, then
//! save the winning settings as a favorite. Measuring intelligence instead
//! lives in [`crate::evalui`].
use crate::agentui::AgentState;
use crate::bench::{self, Candidate, ChunkRow, Event, Machine, Plan, Row, Running, Saved, Source};
use crate::benchview::{Chart, ChartBar, Parts, Widths, form_row};
use crate::eval;
use crate::evalui::{IqState, Measure};
use crate::layout::{fit_w, list_row};
use crate::*;

const TOKENS: [usize; 3] = [128, 192, 256];

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum BenchPhase {
    Setup,
    Running,
    Results,
}

pub struct BenchState {
    pub phase: BenchPhase,
    pub measure: Measure,
    pub iq: IqState,
    pub ag: AgentState,
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

#[derive(Clone, Copy, PartialEq, Debug)]
enum Line_ {
    Model,
    Measure,
    Cand(usize),
    Sweep,
    Tokens,
    Q8,
    Chunks,
    Run,
}

/// `fresh` with each candidate ticked as it was in `old` (same drafter and
/// quant), so switching q8_0 or the model does not re-tick what was cleared.
fn keep_ticks(old: &[Candidate], mut fresh: Vec<Candidate>) -> Vec<Candidate> {
    for c in &mut fresh {
        if let Some(o) = old
            .iter()
            .find(|o| o.source == c.source && o.quant == c.quant)
        {
            c.on = o.on;
        }
    }
    fresh
}

/// Cursor order of the speed setup: the Setup form top to bottom, then the
/// candidates in the "What to test" list below it, as `speed_parts` draws them.
fn line_order(cands: usize) -> Vec<Line_> {
    let mut rows = vec![
        Line_::Model,
        Line_::Measure,
        Line_::Sweep,
        Line_::Tokens,
        Line_::Q8,
        Line_::Chunks,
        Line_::Run,
    ];
    rows.extend((0..cands).map(Line_::Cand));
    rows
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
            measure: Measure::Speed,
            iq: IqState::new(),
            ag: AgentState::new(),
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

    /// Home, keeping the tab as it is. A run in progress is asked to stop
    /// instead, and shows what finished.
    pub fn leave_bench(&mut self) {
        let running = self
            .bench
            .as_ref()
            .filter(|b| b.phase == BenchPhase::Running)
            .map(|b| b.measure);
        match running {
            Some(Measure::Speed) => self.speed_stop(),
            Some(Measure::Intelligence) => self.iq_stop(),
            Some(Measure::Agent) => self.agent_stop(),
            None => self.screen = Screen::Home,
        }
    }

    fn bench_lines(&self) -> Vec<Line_> {
        self.bench
            .as_ref()
            .map_or_else(Vec::new, |b| line_order(b.cands.len()))
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
            kv_cache: self.kv_cache().unwrap_or_default(),
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
        if !self.make_room("the benchmark", false, false) {
            return;
        }
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
        match self.bench.as_ref().map(|b| b.measure) {
            Some(Measure::Intelligence) => return self.iq_tick(),
            Some(Measure::Agent) => return self.agent_tick(),
            _ => {}
        }
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
                Event::Checked(i, v) => {
                    if let Some(r) = b.rows.get_mut(i) {
                        r.check = Some(v);
                    }
                }
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
        let (phase, measure) = match &self.bench {
            Some(b) => (b.phase, b.measure),
            None => return self.go_home(),
        };
        let comparing = self
            .bench
            .as_ref()
            .is_some_and(|b| b.iq.compare.is_some() || b.ag.compare.is_some());
        let viewing = self.bench.as_ref().is_some_and(|b| b.ag.viewer.is_some());
        // A narrow window shows one group of panes; tab moves between them.
        if key.code == KeyCode::Tab
            && !comparing
            && !viewing
            && self.size.get().0 < self.prefs.compact
        {
            self.bench_pane = (self.bench_pane + 1) % crate::benchview::GROUPS.len();
            return;
        }
        let before = phase;
        self.bench_key_inner(key, phase, measure, comparing);
        if before != BenchPhase::Running
            && self
                .bench
                .as_ref()
                .is_some_and(|b| b.phase == BenchPhase::Running)
        {
            self.bench_pane = 1;
        }
    }

    fn bench_key_inner(
        &mut self,
        key: KeyEvent,
        phase: BenchPhase,
        measure: Measure,
        comparing: bool,
    ) {
        if phase != BenchPhase::Running
            && !comparing
            && key.code == KeyCode::Char('m')
            && let Some(b) = self.bench.as_mut()
        {
            b.phase = BenchPhase::Setup;
            b.measure = match b.measure {
                Measure::Speed => Measure::Intelligence,
                Measure::Intelligence => Measure::Agent,
                Measure::Agent => Measure::Speed,
            };
            // Land on the Measure row, as switching there would.
            b.sel = 1;
            b.iq.sel = 1;
            b.ag.sel = 1;
            return;
        }
        if measure == Measure::Agent {
            if self.bench.as_ref().is_some_and(|b| b.ag.compare.is_some()) {
                return self.agent_compare_key(key);
            }
            return match phase {
                BenchPhase::Setup => self.agent_setup_key(key),
                BenchPhase::Running => self.agent_running_key(key),
                BenchPhase::Results => self.agent_results_key(key),
            };
        }
        if measure == Measure::Intelligence {
            if self.bench.as_ref().is_some_and(|b| b.iq.compare.is_some()) {
                return self.iq_compare_key(key);
            }
            return match phase {
                BenchPhase::Setup => self.iq_setup_key(key),
                BenchPhase::Running => {
                    if matches!(key.code, KeyCode::Esc | KeyCode::Char('q')) {
                        self.iq_stop();
                    }
                }
                BenchPhase::Results => self.iq_results_key(key),
            };
        }
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
                    Line_::Measure => {
                        b.measure = Measure::Intelligence;
                        b.iq.sel = 1;
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
            self.bench_model_changed();
        }
        if start {
            self.start_bench();
        }
    }

    /// Rebuild the candidate list after the model (or q8 choice) changed.
    pub fn bench_model_changed(&mut self) {
        let Some(b) = self.bench.as_ref() else { return };
        let (model, q8) = (b.model, b.q8);
        let cands = self
            .models
            .get(model)
            .map(|m| self.bench_candidates(m, q8))
            .unwrap_or_default();
        if let Some(b) = self.bench.as_mut() {
            b.cands = keep_ticks(&b.cands, cands);
        }
        let last = self.bench_lines().len() - 1;
        if let Some(b) = self.bench.as_mut() {
            b.sel = b.sel.min(last);
        }
    }

    // ---- drawing ----

    pub fn draw_bench(&self, f: &mut Frame, body: Rect, footer: Rect) {
        let Some(b) = self.bench.as_ref() else { return };
        let comparing = (b.measure == Measure::Intelligence && b.iq.compare.is_some())
            || (b.measure == Measure::Agent && b.ag.compare.is_some());
        let iq = b.measure == Measure::Intelligence;
        let agent = b.measure == Measure::Agent;
        let viewing = agent && b.ag.viewer.is_some();
        let narrow = body.width < self.prefs.compact;
        match () {
            _ if comparing && agent => self.draw_agent_compare(f, body, b),
            _ if comparing => self.draw_iq_compare(f, body, b),
            // Transcripts are laid out 104 wide (`agent_pick_key`).
            _ if viewing => self.draw_agent_viewer(f, centered(body, 108, body.height), b),
            _ => self.draw_bench_view(f, body, b),
        }
        let mut keys: Vec<(&str, &str)> = match b.phase {
            _ if comparing && agent => vec![
                ("←→", "move"),
                ("t", "thinking"),
                ("x", "clear"),
                ("esc", "back"),
            ],
            _ if comparing => vec![
                ("↑↓", "move"),
                ("enter", "details"),
                ("d", "difficulty"),
                ("t", "thinking"),
                ("f", "full only"),
                ("x", "clear"),
                ("esc", "back"),
            ],
            _ if viewing => vec![
                ("↑↓", "scroll"),
                ("pgup/pgdn", "page"),
                ("g/G", "top/end"),
                ("esc", "close"),
            ],
            BenchPhase::Running if agent => vec![("esc", "stop, keep what finished")],
            BenchPhase::Results if agent => vec![
                ("↑↓", "task"),
                ("enter", "transcript"),
                ("c", "compare"),
                ("r", "run again"),
                ("m", "measure"),
            ],
            BenchPhase::Setup if agent || iq => vec![
                ("↑↓", "move"),
                ("←→", "change"),
                ("r", "run"),
                ("m", "measure"),
                ("c", "compare"),
            ],
            BenchPhase::Results if iq => {
                vec![("c", "compare"), ("r", "run again"), ("m", "measure")]
            }
            BenchPhase::Setup => vec![
                ("↑↓", "move"),
                ("←→/space", "change"),
                ("r", "run"),
                ("m", "measure"),
            ],
            BenchPhase::Running => vec![("esc", "stop, keep what finished")],
            BenchPhase::Results => vec![
                ("s", "save as favorite"),
                ("v", "save for Serve"),
                ("r", "run again"),
                ("m", "measure"),
            ],
        };
        if narrow && !comparing && !viewing {
            keys.insert(0, ("tab", "next group"));
        }
        if !comparing && !viewing && b.phase != BenchPhase::Running {
            keys.push(("[ ]", "resize"));
        }
        self.footer(f, footer, &keys);
    }

    /// What the selected setup row does.
    fn bench_help(&self, line: Line_, b: &BenchState) -> String {
        match line {
            Line_::Model => "The model to tune. Only models ferrum can run are offered.".into(),
            Line_::Measure => "Speed finds the fastest drafter and depth for this Mac; intelligence scores answers; agentic coding runs real bug fixes. m switches.".into(),
            Line_::Cand(i) if b.cands.get(i).is_some_and(|c| c.source == Source::Baseline) => {
                "Plain decoding, no drafter: the speed and the answers every drafter is checked against.".into()
            }
            Line_::Cand(_) => "A drafter guesses several tokens ahead and the model checks them in one step. Space includes or skips it.".into(),
            Line_::Sweep => "Quick tries a few draft depths per drafter; full tries every depth up to the drafter's maximum.".into(),
            Line_::Tokens => "Tokens generated per prompt. More gives steadier numbers and takes longer.".into(),
            Line_::Q8 => "Also try each drafter at q8_0: more memory, sometimes more accepted drafts.".into(),
            Line_::Chunks => "Also find the fastest prefill chunk for the server: how many prompt rows go through per pass.".into(),
            Line_::Run => "Start the sweep. Esc while it runs stops it and keeps what finished.".into(),
        }
    }

    /// The speed measure's panes.
    pub fn speed_parts(&self, b: &BenchState, w: Widths) -> Parts {
        let lines_def = self.bench_lines();
        let sel = b.sel.min(lines_def.len() - 1);
        let setup = b.phase == BenchPhase::Setup;
        let model = self.models.get(b.model);
        // The form: every row but the candidates, which are the queue.
        let mut form = Vec::new();
        let mut form_sel = None;
        let mut queue = Vec::new();
        let mut queue_sel = None;
        for (n, &def) in lines_def.iter().enumerate() {
            let on = n == sel && setup;
            if let Line_::Cand(i) = def {
                let c = &b.cands[i];
                if on {
                    queue_sel = Some(queue.len());
                }
                queue.extend(self.candidate_lines(b, i, c, on, w.form));
                continue;
            }
            let (label, value) = match def {
                Line_::Model => (
                    "Model",
                    model
                        .map(|m| m.name.clone())
                        .unwrap_or_else(|| "none".into()),
                ),
                Line_::Measure => ("Measure", "speed".into()),
                Line_::Sweep => (
                    "Depth sweep",
                    if b.quick {
                        "quick · a few depths"
                    } else {
                        "full · every depth"
                    }
                    .into(),
                ),
                Line_::Tokens => ("Tokens per prompt", TOKENS[b.tokens].to_string()),
                Line_::Q8 => ("q8_0 drafters too", if b.q8 { "yes" } else { "no" }.into()),
                Line_::Chunks => (
                    "Tune prefill chunk",
                    if b.chunks {
                        "yes · 4 more loads"
                    } else {
                        "no"
                    }
                    .into(),
                ),
                Line_::Run => ("▶ Run benchmark", String::new()),
                Line_::Cand(_) => continue,
            };
            if def == Line_::Run {
                form.push(Line::default());
            }
            if on {
                form_sel = Some(form.len());
            }
            let value = clip(&value, w.form.saturating_sub(label.len() + 8));
            form.push(form_row(on, true, w.form, label, &value, false, !setup));
        }
        if let Some(plan) = self.bench_plan() {
            form.push(Line::styled(
                format!(
                    "   about {} min · {} prompts, greedy",
                    plan.estimate_secs().div_ceil(60),
                    bench::PROMPTS.len()
                ),
                Style::new().fg(FAINT),
            ));
        }
        let on_count = b.cands.iter().filter(|c| c.on).count();
        let (queue_title, queue_note) = if setup {
            (
                "What to test".to_string(),
                format!("{on_count} of {}", b.cands.len()),
            )
        } else {
            ("Queue".to_string(), format!("{} done", b.rows.len()))
        };

        // The chart: earlier results before a run, this run's rows after.
        let base = bench::baseline(&b.rows).map(|r| r.tps);
        let best = bench::best(&b.rows);
        let (chart_title, chart_note, chart) = if setup {
            let mut saved: Vec<&Saved> = b
                .saved
                .iter()
                .filter(|s| s.machine == b.machine.label())
                .collect();
            saved.sort_by(|a, c| c.speedup.total_cmp(&a.speedup));
            let path = model
                .map(|m| m.path.display().to_string())
                .unwrap_or_default();
            let top = saved.iter().map(|s| s.speedup).fold(2.0, f64::max);
            let bars: Vec<ChartBar> = saved
                .iter()
                .map(|s| ChartBar {
                    label: crate::evalcompare::model_name(&s.model),
                    sub: String::new(),
                    share: s.speedup / top,
                    margin: 0.0,
                    color: if s.model == path { EMBER } else { GOLD },
                    value: format!("{:.2}× · {}", s.speedup, ago(s.when)),
                    value_color: if s.speedup >= 1.2 { GOOD } else { GOLD },
                    star: s.model == path,
                    pending: false,
                })
                .collect();
            let chart = if bars.is_empty() {
                Chart::Empty(crate::layout::wrapped(
                    "Nothing measured on this Mac yet. r runs the sweep for the model on the left: each drafter at a few depths, on the same prompts, checked against plain decoding.",
                    2,
                    w.chart,
                    Style::new().fg(DIM),
                ))
            } else {
                Chart::Bars(bars)
            };
            (
                "Best speedup per model on this Mac".to_string(),
                "from earlier runs".to_string(),
                chart,
            )
        } else {
            let mut rows: Vec<&Row> = b.rows.iter().collect();
            if b.phase == BenchPhase::Results {
                rows.sort_by(|a, c| c.tps.total_cmp(&a.tps));
            }
            let top = rows.iter().map(|r| r.tps).fold(1.0, f64::max);
            let mut bars: Vec<ChartBar> = rows
                .iter()
                .map(|r| {
                    let is_best = best.is_some_and(|x| std::ptr::eq(*r, x));
                    let (mark, mark_color) = speed_mark(r);
                    ChartBar {
                        label: short_drafter(&r.label),
                        sub: r.depth.map_or(String::new(), |d| format!("d{d}")),
                        share: r.tps / top,
                        margin: 0.0,
                        color: if is_best {
                            EMBER
                        } else if r.same == Some(bench::Match::Differs) {
                            FAINT
                        } else {
                            GOLD
                        },
                        value: format!(
                            "{:.1}{}",
                            r.tps,
                            if mark.trim().is_empty() {
                                String::new()
                            } else {
                                format!(" {}", mark.trim())
                            }
                        ),
                        value_color: mark_color,
                        star: is_best && b.phase == BenchPhase::Results,
                        pending: false,
                    }
                })
                .collect();
            if b.phase == BenchPhase::Running {
                let pulse = (self.elapsed() % 2.0) as f64 / 2.0;
                bars.push(ChartBar {
                    label: clip(&b.stage, 28),
                    sub: String::new(),
                    share: pulse,
                    margin: 0.0,
                    color: EMBER,
                    value: "measuring…".into(),
                    value_color: EMBER,
                    star: false,
                    pending: true,
                });
            }
            let title = match (b.phase, best, base) {
                (BenchPhase::Results, Some(best), Some(base)) => match best.depth {
                    Some(d) => format!(
                        "★ {} · depth {d} · {} → {:.2}× plain decoding",
                        short_drafter(&best.label),
                        best.quant,
                        best.tps / base.max(1e-9)
                    ),
                    None => "No drafter beats plain decoding here".to_string(),
                },
                (BenchPhase::Results, _, _) => "Not enough finished to recommend anything".into(),
                _ => "Decode speed · tok/s".into(),
            };
            let note = match b.loaded {
                Some((ctx, wg, d)) => format!(
                    "context {ctx} · {wg:.1} GiB{}",
                    if d > 0.0 {
                        format!(" + {d:.1}")
                    } else {
                        String::new()
                    }
                ),
                None => format!("{} tokens · greedy", TOKENS[b.tokens]),
            };
            (title, note, Chart::Bars(bars))
        };

        // Rows: the plan before a run, the measured table during and after.
        let (rows_title, rows_note, rows, rows_tail) = if setup {
            let mut rows = vec![crate::benchview::table_head(&[
                ("candidate", w.rows.saturating_sub(30).min(48), false),
                ("depths", 16, false),
                ("quant", 6, false),
            ])];
            let mut runs = 0;
            for c in b.cands.iter().filter(|c| c.on) {
                let depths = match c.source {
                    Source::Baseline => vec![],
                    _ => bench::depth_list(c.max_depth(), b.quick),
                };
                runs += depths.len().max(1);
                rows.push(Line::styled(
                    format!(
                        "  {} {} {}",
                        fit_w(
                            &short_drafter(&c.label()),
                            w.rows.saturating_sub(30).min(48)
                        ),
                        fit_w(
                            &if depths.is_empty() {
                                "baseline".to_string()
                            } else {
                                depths
                                    .iter()
                                    .map(|d| d.to_string())
                                    .collect::<Vec<_>>()
                                    .join(", ")
                            },
                            16
                        ),
                        if depths.is_empty() { "–" } else { c.quant }
                    ),
                    Style::new().fg(TEXT),
                ));
            }
            ("Plan".to_string(), format!("{runs} runs"), rows, false)
        } else {
            let mut sorted: Vec<&Row> = b.rows.iter().collect();
            if b.phase == BenchPhase::Results {
                sorted.sort_by(|a, c| c.tps.total_cmp(&a.tps));
            }
            let table = self.result_table(
                &sorted,
                base,
                if b.phase == BenchPhase::Results {
                    best
                } else {
                    None
                },
                w.rows,
            );
            let mut rows = table;
            for c in &b.chunk_rows {
                rows.push(Line::styled(
                    format!(
                        "   prefill chunk {:>5}   {:>7.0} tok/s",
                        c.chunk, c.prefill_tps
                    ),
                    Style::new().fg(DIM),
                ));
            }
            (
                "Rows".to_string(),
                format!("{} measured", b.rows.len()),
                rows,
                b.phase == BenchPhase::Running,
            )
        };
        let notes = self.speed_notes(b, w.notes);
        Parts {
            form,
            form_sel,
            queue_title,
            queue_note,
            queue,
            queue_sel,
            help: self.bench_help(lines_def[sel], b),
            chart_title,
            chart_note,
            chart,
            rows_title,
            rows_note,
            rows,
            rows_keep: None,
            rows_tail,
            notes,
            running: b.phase == BenchPhase::Running,
            small_chart: false,
        }
    }

    /// A candidate in the queue: checkbox before a run, progress during it
    /// (nothing for one left unticked, which the run skips).
    fn candidate_lines(
        &self,
        b: &BenchState,
        i: usize,
        c: &Candidate,
        on: bool,
        w: usize,
    ) -> Vec<Line<'static>> {
        if b.phase != BenchPhase::Setup && !c.on {
            return Vec::new();
        }
        let planned = match c.source {
            Source::Baseline => 1,
            _ => bench::depth_list(c.max_depth(), b.quick).len(),
        };
        let measured: Vec<&Row> = b.rows.iter().filter(|r| r.cand == i).collect();
        let skipped = b.skipped.iter().any(|(l, _)| *l == c.label());
        let running = b.phase == BenchPhase::Running;
        // Candidates run in order: the first unfinished one is running.
        let now = running
            && !skipped
            && measured.len() < planned
            && b.cands[..i].iter().enumerate().all(|(j, cj)| {
                !cj.on
                    || b.rows.iter().filter(|r| r.cand == j).count()
                        >= match cj.source {
                            Source::Baseline => 1,
                            _ => bench::depth_list(cj.max_depth(), b.quick).len(),
                        }
                    || b.skipped.iter().any(|(l, _)| *l == cj.label())
            });
        let (mark, color) = match b.phase {
            BenchPhase::Setup if c.source == Source::Baseline => ("[✓]".to_string(), TEXT),
            BenchPhase::Setup if c.on => ("[x]".to_string(), TEXT),
            BenchPhase::Setup => ("[ ]".to_string(), FAINT),
            _ if skipped => (" ✗ ".to_string(), BAD),
            _ if measured.len() >= planned => (" ✓ ".to_string(), GOOD),
            _ if now => (format!(" {} ", spinner(self.elapsed())), EMBER),
            _ => (" · ".to_string(), FAINT),
        };
        let right = match b.phase {
            BenchPhase::Setup if c.source == Source::Baseline => "baseline".to_string(),
            BenchPhase::Setup => format!("≤{} · {}", c.max_depth(), c.quant),
            _ => format!("{}/{planned}", measured.len()),
        };
        let best = measured.iter().map(|r| r.tps).fold(0.0, f64::max);
        let sub = match b.phase {
            BenchPhase::Setup => String::new(),
            _ if skipped => "skipped".into(),
            _ if best > 0.0 => format!("best {best:.1} tok/s"),
            _ if now => "measuring…".into(),
            _ => "waiting".into(),
        };
        let name_w = w.saturating_sub(text_width(&right) + 7);
        let mut out = vec![list_row(
            on,
            true,
            w,
            vec![
                Span::styled(mark, Style::new().fg(color)),
                Span::styled(
                    format!(" {}", fit_w(&short_drafter(&c.label()), name_w)),
                    Style::new().fg(if c.on || running { TEXT } else { FAINT }),
                ),
                Span::styled(format!(" {right}"), Style::new().fg(DIM)),
            ],
        )];
        if !sub.is_empty() {
            out.push(Line::styled(format!("     {sub}"), Style::new().fg(FAINT)));
        }
        out
    }

    /// The legend, the answer checks, skips and what was saved.
    fn speed_notes(&self, b: &BenchState, w: usize) -> Vec<Line<'static>> {
        let mut out: Vec<Line> = Vec::new();
        if let Some(note) = &b.note {
            out.extend(crate::layout::wrapped(
                &format!("✓ {note}"),
                1,
                w,
                Style::new().fg(GOOD),
            ));
            out.push(Line::default());
        }
        for (label, why) in &b.skipped {
            out.push(Line::styled(
                format!(" ✗ {label} was skipped"),
                Style::new().fg(BAD),
            ));
            out.extend(crate::layout::wrapped(why, 3, w, Style::new().fg(DIM)));
            out.push(Line::default());
        }
        for r in &b.rows {
            let Some(v) = &r.check else { continue };
            out.push(Line::styled(
                format!(
                    " {} {} d{}: {}/{} same, {} lost, {} gained",
                    if v.passed() { "≈✓" } else { "≈✗" },
                    short_drafter(&r.label),
                    r.depth.unwrap_or(0),
                    v.same,
                    v.n,
                    v.lost,
                    v.gained
                ),
                Style::new().fg(if v.passed() { GOOD } else { BAD }),
            ));
        }
        let legend = [
            ("✓", "same as plain decoding, token for token", GOOD),
            (
                "≈✓",
                "a near-tie flipped after 25% matched; kept plain decoding's right answers on the math check",
                GOOD,
            ),
            ("≈✗", "the same, but lost answers: not recommended", BAD),
            ("≈?", "not checked, so not recommended", GOLD),
            ("≠", "went off plain decoding early: never recommended", BAD),
            (
                "accept",
                "tokens taken per verify step; 1.0 is no gain",
                DIM,
            ),
        ];
        out.push(crate::layout::heading("marks"));
        for (k, what, color) in legend {
            out.extend(crate::benchview::legend(k, what, color, w));
        }
        let _ = eval::check_items;
        out
    }

    /// Stop a speed run; it keeps what finished.
    fn speed_stop(&mut self) {
        if let Some(b) = self.bench.as_mut()
            && let Some(run) = &b.run
        {
            run.stop.store(true, std::sync::atomic::Ordering::SeqCst);
            b.stopping = true;
        }
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
                    format!("{:<label_w$}", clip(&short_drafter(&r.label), label_w)),
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
                {
                    let (mark, color) = match r.same {
                        Some(bench::Match::Exact) => ("     ✓", GOOD),
                        Some(bench::Match::Close) => match &r.check {
                            Some(v) if v.passed() => ("    ≈✓", GOOD),
                            Some(_) => ("    ≈✗", BAD),
                            None => ("    ≈?", GOLD),
                        },
                        Some(bench::Match::Differs) => ("     ≠", BAD),
                        None => ("      ", GOOD),
                    };
                    Span::styled(mark, Style::new().fg(color))
                },
            ]));
        }
        lines
    }
}

/// A drafter label without the model's own name in it.
pub fn short_drafter(label: &str) -> String {
    let (org, rest) = label.split_once('/').unwrap_or(("", label));
    let kind = rest.find(" (").map_or("", |i| &rest[i..]);
    let name = rest.trim_end_matches(kind);
    let parts: Vec<&str> = name.split(['-', '.', '_']).collect();
    // Drop the parts that repeat the model (Qwen3.8, 27B…), keep the rest.
    let keep: Vec<&str> = parts
        .iter()
        .copied()
        .filter(|p| {
            let l = p.to_lowercase();
            !(l.starts_with("qwen")
                || l.ends_with('b') && l[..l.len() - 1].parse::<f64>().is_ok()
                || l.starts_with('a') && l.ends_with('b') && l.len() <= 4
                || l.parse::<f64>().is_ok())
        })
        .collect();
    let short = if keep.is_empty() {
        name.to_string()
    } else {
        keep.join("-")
    };
    match org {
        "" => short,
        o => format!("{o} {short}"),
    }
}

/// The match mark of a row and its colour.
fn speed_mark(r: &Row) -> (&'static str, Color) {
    match r.same {
        Some(bench::Match::Exact) => ("✓", GOOD),
        Some(bench::Match::Close) => match &r.check {
            Some(v) if v.passed() => ("≈✓", GOOD),
            Some(_) => ("≈✗", BAD),
            None => ("≈?", GOLD),
        },
        Some(bench::Match::Differs) => ("≠", BAD),
        None => ("base", DIM),
    }
}

impl BenchState {
    /// GiB of weights a running measure holds.
    pub fn loaded_gib(&self) -> Option<f64> {
        match self.measure {
            Measure::Speed => self.loaded.map(|(_, w, d)| w + d),
            Measure::Intelligence => self.iq.loaded.map(|(_, w)| w),
            Measure::Agent => self.ag.loaded.map(|(_, w)| w),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn drafter_labels_drop_the_model_name() {
        assert_eq!(
            short_drafter("z-lab/Qwen3.8-27B-DFlash2 (dflash)"),
            "z-lab DFlash2"
        );
        assert_eq!(
            short_drafter("RedHatAI/Qwen3.8-27B-speculator.dspark (dspark)"),
            "RedHatAI speculator-dspark"
        );
        assert_eq!(short_drafter("MTP head"), "MTP head");
        assert_eq!(short_drafter("no drafter"), "no drafter");
    }

    #[test]
    fn rebuilding_candidates_keeps_ticks() {
        let drafter = |label: &str| Source::Drafter {
            path: PathBuf::from(format!("/m/{label}.gguf")),
            label: label.into(),
            kind: "dflash2",
            max_depth: 7,
        };
        let cand = |source: Source, quant, on| Candidate { source, quant, on };
        // Only the OrcaCyber drafter left ticked; then q8_0 is switched on.
        let old = vec![
            cand(Source::Baseline, "q4_0", true),
            cand(Source::Mtp, "q4_0", false),
            cand(drafter("orca"), "q4_0", true),
            cand(drafter("zlab"), "q4_0", false),
        ];
        let fresh = vec![
            cand(Source::Baseline, "q4_0", true),
            cand(Source::Mtp, "q4_0", true),
            cand(drafter("orca"), "q4_0", true),
            cand(drafter("orca"), "q8_0", false),
            cand(drafter("zlab"), "q4_0", true),
            cand(drafter("zlab"), "q8_0", false),
        ];
        let on: Vec<bool> = keep_ticks(&old, fresh).iter().map(|c| c.on).collect();
        assert_eq!(on, [true, false, true, false, false, false]);
    }

    #[test]
    fn cursor_walks_the_setup_form_before_the_candidates() {
        let rows = line_order(3);
        // Down from Measure reaches the next form row, not the list.
        let measure = rows.iter().position(|&r| r == Line_::Measure).unwrap();
        assert_eq!(rows[measure + 1], Line_::Sweep);
        // Every form row comes before every candidate, in drawing order.
        let first_cand = rows
            .iter()
            .position(|r| matches!(r, Line_::Cand(_)))
            .unwrap();
        assert_eq!(rows[first_cand - 1], Line_::Run);
        assert!(
            rows[first_cand..]
                .iter()
                .all(|r| matches!(r, Line_::Cand(_)))
        );
        assert_eq!(
            rows[first_cand..],
            [Line_::Cand(0), Line_::Cand(1), Line_::Cand(2)]
        );
    }
}
