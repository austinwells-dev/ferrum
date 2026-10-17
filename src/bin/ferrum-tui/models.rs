//! The Models tab: every model file found, what ferrum knows about it and
//! how it measured on this Mac, with the drafters that fit it. Starting a
//! chat, agent, server or benchmark from here preselects the model.
use crate::layout::{self, bar, fit_r, fit_w, heading, kv, list_row, pane, show};
use crate::*;
use crate::{agentbench, bench, benchui, eval, evalui};

/// What the table is ordered by.
#[derive(Clone, Copy, PartialEq)]
pub enum Sort {
    Speed,
    Intelligence,
    Name,
    Size,
}

impl Sort {
    fn next(self) -> Self {
        match self {
            Sort::Speed => Sort::Intelligence,
            Sort::Intelligence => Sort::Name,
            Sort::Name => Sort::Size,
            Sort::Size => Sort::Speed,
        }
    }

    fn label(self) -> &'static str {
        match self {
            Sort::Speed => "speed",
            Sort::Intelligence => "intelligence",
            Sort::Name => "name",
            Sort::Size => "size",
        }
    }
}

/// Saved measurements, read again when a few seconds old.
pub struct Dashboard {
    at: Instant,
    /// This Mac's speed results, newest first.
    pub speed: Vec<bench::Saved>,
    /// The best intelligence score per model, best first.
    pub iq: Vec<eval::EvalResult>,
    /// The best agent run per model, best first.
    pub agent: Vec<agentbench::AgentResult>,
}

impl Dashboard {
    fn load(machine: &str) -> Self {
        let mut speed: Vec<bench::Saved> = bench::load_saved()
            .into_iter()
            .filter(|s| s.machine == machine)
            .collect();
        speed.sort_by_key(|s| std::cmp::Reverse(s.when));
        let mut iq: Vec<eval::EvalResult> = Vec::new();
        for r in eval::load_results(&eval::results_path()) {
            let Some(s) = r.score() else { continue };
            match iq.iter_mut().find(|x| x.model == r.model) {
                Some(x) if x.score().is_none_or(|b| s.share > b.share) => *x = r,
                Some(_) => {}
                None => iq.push(r),
            }
        }
        iq.sort_by(|a, b| {
            let share = |r: &eval::EvalResult| r.score().map_or(0.0, |s| s.share);
            share(b).total_cmp(&share(a))
        });
        let mut agent: Vec<agentbench::AgentResult> = Vec::new();
        for r in agentbench::ranked(&agentbench::load_results(&agentbench::results_path())) {
            if !agent.iter().any(|x| x.model == r.model) {
                agent.push(r.clone());
            }
        }
        Self {
            at: Instant::now(),
            speed,
            iq,
            agent,
        }
    }

    pub fn speed_of(&self, model: &str) -> Option<&bench::Saved> {
        self.speed.iter().find(|s| s.model == model)
    }

    pub fn iq_of(&self, model: &str) -> Option<(&eval::EvalResult, eval::Score)> {
        self.iq
            .iter()
            .find(|r| r.model == model)
            .and_then(|r| Some((r, r.score()?)))
    }

    pub fn agent_of(&self, model: &str) -> Option<&agentbench::AgentResult> {
        self.agent.iter().find(|r| r.model == model)
    }
}

impl App {
    /// The saved measurements, reloaded when a few seconds old.
    pub fn dashboard(&self) -> std::cell::Ref<'_, Dashboard> {
        let stale = self
            .dash
            .borrow()
            .as_ref()
            .is_none_or(|d| d.at.elapsed() > Duration::from_secs(4));
        if stale {
            *self.dash.borrow_mut() = Some(Dashboard::load(&self.machine.label()));
        }
        std::cell::Ref::map(self.dash.borrow(), |d| d.as_ref().expect("just loaded"))
    }

    /// Model indexes in table order: runnable first, then by the sort.
    fn model_order(&self) -> Vec<usize> {
        let d = self.dashboard();
        let key = |i: usize| -> f64 {
            let path = self.models[i].path.display().to_string();
            match self.models_sort {
                Sort::Speed => d.speed_of(&path).map_or(0.0, |s| s.speedup),
                Sort::Intelligence => d.iq_of(&path).map_or(0.0, |(_, s)| s.share),
                Sort::Size => self.models[i].size as f64,
                Sort::Name => 0.0,
            }
        };
        let mut order: Vec<usize> = (0..self.models.len()).collect();
        order.sort_by(|&a, &b| {
            let (ma, mb) = (&self.models[a], &self.models[b]);
            ma.problem
                .is_some()
                .cmp(&mb.problem.is_some())
                .then_with(|| match self.models_sort {
                    Sort::Name => ma.name.to_lowercase().cmp(&mb.name.to_lowercase()),
                    _ => key(b).total_cmp(&key(a)),
                })
        });
        order
    }

    fn picked_model(&self) -> Option<usize> {
        let order = self.model_order();
        order
            .get(self.models_sel.min(order.len().saturating_sub(1)))
            .copied()
    }

    pub fn models_key(&mut self, key: KeyEvent) {
        let n = self.models.len();
        match key.code {
            KeyCode::Up | KeyCode::Char('k') => self.models_sel = self.models_sel.saturating_sub(1),
            KeyCode::Down | KeyCode::Char('j') => {
                self.models_sel = (self.models_sel + 1).min(n.saturating_sub(1))
            }
            KeyCode::Home => self.models_sel = 0,
            KeyCode::End => self.models_sel = n.saturating_sub(1),
            KeyCode::Char('o') => {
                let keep = self.picked_model();
                self.models_sort = self.models_sort.next();
                if let Some(k) = keep {
                    self.models_sel = self.model_order().iter().position(|&i| i == k).unwrap_or(0);
                }
            }
            KeyCode::Char('r') => self.rescan(None),
            KeyCode::Esc | KeyCode::Char('q') => self.go_tab(Tab::Home),
            KeyCode::Char('c' | 'a' | 's' | 'b' | 'f') | KeyCode::Enter
                if self.picked_model().is_some() =>
            {
                let c = match key.code {
                    KeyCode::Enter => 'c',
                    KeyCode::Char(c) => c,
                    _ => return,
                };
                let Some(i) = self.picked_model() else { return };
                if let Some(why) = &self.models[i].problem {
                    let why = why.clone();
                    self.status =
                        Some((format!("{} can't run: {why}", self.models[i].name), false));
                    return;
                }
                match c {
                    'b' => {
                        self.go_tab(Tab::Bench);
                        if let Some(b) = self.bench.as_mut()
                            && b.phase != benchui::BenchPhase::Running
                        {
                            b.model = i;
                            self.bench_model_changed();
                        }
                    }
                    _ => {
                        let (mode, origin) = match c {
                            'a' => (Mode::Agent, Origin::OneTime),
                            's' => (Mode::Serve, Origin::OneTime),
                            'f' => (Mode::Chat, Origin::Custom),
                            _ => (Mode::Chat, Origin::OneTime),
                        };
                        self.open_pick(mode);
                        self.open_editor(mode, origin);
                        self.cursor = i;
                        self.load_info();
                        self.focus = Focus::Settings;
                    }
                }
            }
            _ => {}
        }
    }

    // ---- drawing ----

    pub fn draw_models_tab(&self, f: &mut Frame, body: Rect, footer: Rect) {
        // The table takes the rows it needs, up to its share; the detail the rest.
        let table_h = ((body.height as f32 * self.prefs.models).round() as u16)
            .min(self.models.len() as u16 + 4)
            .max(6)
            .min(body.height.saturating_sub(8));
        let [top, bottom] =
            Layout::vertical([Constraint::Length(table_h), Constraint::Min(0)]).areas(body);
        self.draw_library(f, top);
        match layout::columns(bottom, 0.62, 50, 44, self.prefs.compact.max(120)) {
            (detail, Some(drafters)) => {
                self.draw_model_detail(f, detail, false);
                self.draw_model_drafters(f, drafters);
            }
            (detail, None) => self.draw_model_detail(f, detail, true),
        }
        self.footer(
            f,
            footer,
            &[
                ("↑↓", "choose"),
                ("enter", "chat"),
                ("a", "agent"),
                ("s", "serve"),
                ("b", "benchmark"),
                ("f", "save a setup"),
                ("o", "sort"),
                ("r", "rescan"),
            ],
        );
    }

    fn draw_library(&self, f: &mut Frame, area: Rect) {
        let runnable = self.runnable();
        let title = format!(
            "Library · {} models · {} drafters",
            self.models.len(),
            self.drafters.len()
        );
        let right = format!("sorted by {} · o", self.models_sort.label());
        let inner = pane(f, area, &title, true, &right);
        let w = inner.width as usize;
        if self.models.is_empty() {
            let lines = vec![
                Line::default(),
                Line::styled("  No .gguf files found.", Style::new().fg(TEXT)),
                Line::styled(
                    "  a adds a file or folder; FERRUM_MODELS=dir:dir adds more.",
                    Style::new().fg(DIM),
                ),
            ];
            f.render_widget(Paragraph::new(lines), inner);
            return;
        }
        // (title, width, priority: the highest number goes first when narrow)
        let mut cols: Vec<(&str, usize, u8)> = vec![
            ("quant", 9, 4),
            ("size", 9, 2),
            ("arch", 11, 6),
            ("context", 9, 5),
            ("vision", 7, 7),
            ("drafters", 9, 3),
            ("speed", 8, 1),
            ("intel.", 10, 2),
            ("agent", 7, 3),
        ];
        while !cols.is_empty() && cols.iter().map(|c| c.1).sum::<usize>() + 26 > w {
            let worst = cols.iter().map(|c| c.2).max().unwrap_or(0);
            cols.retain(|c| c.2 != worst);
        }
        let name_w = w.saturating_sub(2 + cols.iter().map(|c| c.1).sum::<usize>());
        let mut head = vec![Span::styled(
            format!("  {}", fit_w("name", name_w)),
            Style::new().fg(FAINT),
        )];
        for (t, cw, _) in &cols {
            head.push(Span::styled(fit_r(t, cw - 1) + " ", Style::new().fg(FAINT)));
        }
        let mut lines = vec![Line::from(head)];
        let d = self.dashboard();
        let order = self.model_order();
        let sel = self.models_sel.min(order.len().saturating_sub(1));
        for (row, &i) in order.iter().enumerate() {
            let m = &self.models[i];
            let on = row == sel;
            let path = m.path.display().to_string();
            if let Some(why) = &m.problem {
                lines.push(list_row(
                    on,
                    true,
                    w,
                    vec![
                        Span::styled("✗ ", Style::new().fg(BAD)),
                        Span::styled(
                            fit_w(&m.name, name_w.saturating_sub(1)),
                            Style::new().fg(DIM),
                        ),
                        Span::styled(
                            clip(&format!("can't run: {why}"), w.saturating_sub(name_w + 3)),
                            Style::new().fg(BAD),
                        ),
                    ],
                ));
                continue;
            }
            let drafters = self
                .drafters
                .iter()
                .filter(|dr| dr.config.is_some() && drafter_problem(dr, m).is_none())
                .count();
            let speed = d.speed_of(&path);
            let iq = d.iq_of(&path);
            let agent = d.agent_of(&path);
            let mut spans = vec![Span::styled(
                format!(" {}", fit_w(&m.name, name_w)),
                Style::new()
                    .fg(if on { Color::White } else { TEXT })
                    .add_modifier(if on {
                        Modifier::BOLD
                    } else {
                        Modifier::empty()
                    }),
            )];
            for (t, cw, _) in &cols {
                let (text, color): (String, Color) = match *t {
                    "quant" => (quant_of(&m.name).unwrap_or_default(), DIM),
                    "size" => (gib(m.size), DIM),
                    "arch" => (m.info.as_ref().map_or("?".into(), |i| i.arch.clone()), DIM),
                    "context" => (
                        m.info
                            .as_ref()
                            .and_then(|i| i.train_ctx)
                            .map_or("–".into(), |c| format!("{}K", c / 1024)),
                        DIM,
                    ),
                    "vision" => match m.vision {
                        Some(_) => ("yes".into(), TEAL),
                        None => ("–".into(), FAINT),
                    },
                    "drafters" => (
                        match (drafters, m.mtp) {
                            (0, false) => "–".into(),
                            (n, true) => format!("{n} + mtp"),
                            (n, false) => n.to_string(),
                        },
                        GOLD,
                    ),
                    "speed" => match speed {
                        Some(s) => (format!("{:.2}×", s.speedup), GOOD),
                        None => ("–".into(), FAINT),
                    },
                    "intel." => match iq {
                        Some((_, s)) => (s.label(), evalui::score_color(s.share)),
                        None => ("–".into(), FAINT),
                    },
                    _ => match agent {
                        Some(r) => {
                            let t = r.totals();
                            (format!("{}/{}", t.solved, t.done), GOLD)
                        }
                        None => ("–".into(), FAINT),
                    },
                };
                spans.push(Span::styled(
                    fit_r(&text, cw - 1) + " ",
                    Style::new().fg(color).add_modifier(if *t == "speed" {
                        Modifier::BOLD
                    } else {
                        Modifier::empty()
                    }),
                ));
            }
            lines.push(list_row(on, true, w, spans));
        }
        let _ = runnable;
        show(f, inner, lines, Some(sel + 1));
    }

    fn draw_model_detail(&self, f: &mut Frame, area: Rect, with_drafters: bool) {
        let Some(i) = self.picked_model() else {
            pane(f, area, "Detail", false, "");
            return;
        };
        let m = &self.models[i];
        let inner = pane(
            f,
            area,
            &m.name,
            false,
            if m.problem.is_some() {
                "unsupported"
            } else {
                "✓ supported"
            },
        );
        let w = inner.width as usize;
        let mut lines: Vec<Line> = Vec::new();
        if let Some(why) = &m.problem {
            lines.extend(layout::wrapped(
                &format!("ferrum can't run this file: {why}. It stays listed so you can see why it is missing from setups."),
                1,
                w,
                Style::new().fg(BAD),
            ));
            lines.push(Line::default());
            lines.push(kv("file", tilde(&m.path), Style::new().fg(DIM), 10));
            f.render_widget(Paragraph::new(lines), inner);
            return;
        }
        let text = Style::new().fg(TEXT);
        lines.push(kv("from", m.origin.clone(), text, 10));
        lines.push(kv(
            "file",
            clip(&tilde(&m.path), w.saturating_sub(12)),
            Style::new().fg(DIM),
            10,
        ));
        if let Some(info) = &m.info {
            let mut arch = info.arch.clone();
            if let Some(l) = info.layers {
                arch += &format!(" · {l} layers");
            }
            if let Some(e) = info.experts {
                arch += &format!(" · {e} experts");
            }
            lines.push(kv("arch", arch, text, 10));
            if let Some(c) = info.train_ctx {
                lines.push(kv("trained", format!("{c} tokens of context"), text, 10));
            }
            lines.push(kv("sampling", info.sampling.clone(), text, 10));
        }
        lines.push(kv(
            "vision",
            match &m.vision {
                Some(p) => format!(
                    "yes · {}",
                    p.file_name()
                        .map(|n| n.to_string_lossy().into_owned())
                        .unwrap_or_default()
                ),
                None => "no projector next to the file".into(),
            },
            Style::new().fg(if m.vision.is_some() { TEAL } else { FAINT }),
            10,
        ));
        lines.push(kv("size", gib(m.size), text, 10));
        lines.push(Line::default());
        lines.extend(self.measured_lines(&m.path.display().to_string(), w));
        if with_drafters {
            lines.push(Line::default());
            lines.push(heading("drafters that fit"));
            lines.extend(self.drafter_lines(m, w));
        }
        f.render_widget(Paragraph::new(lines), inner);
    }

    fn draw_model_drafters(&self, f: &mut Frame, area: Rect) {
        let Some(i) = self.picked_model() else {
            pane(f, area, "Drafters that fit", false, "");
            return;
        };
        let m = &self.models[i];
        let lines = self.drafter_lines(m, inner_w(area));
        let inner = pane(f, area, "Drafters that fit", false, "");
        f.render_widget(Paragraph::new(lines), inner);
    }

    /// The MTP head and every drafter that fits `m`, each with its best
    /// measured result here.
    fn drafter_lines(&self, m: &Model, w: usize) -> Vec<Line<'static>> {
        if m.problem.is_some() {
            return vec![Line::styled(
                " none: the model can't run",
                Style::new().fg(FAINT),
            )];
        }
        let d = self.dashboard();
        let saved = d.speed_of(&m.path.display().to_string());
        let best = saved.map(|s| s.best.clone());
        let speedup = saved.map_or(0.0, |s| s.speedup);
        let mut lines = Vec::new();
        let mut push = |name: String, kind: &str, depth: String, won: bool| {
            let tail = format!("{kind}  ");
            lines.push(Line::from(vec![
                Span::styled(if won { " ★ " } else { "   " }, Style::new().fg(EMBER)),
                Span::styled(
                    fit_w(&name, w.saturating_sub(text_width(&tail) + 4)),
                    Style::new().fg(if won { Color::White } else { TEXT }),
                ),
                Span::styled(
                    tail,
                    Style::new().fg(match kind {
                        "dflash" => EMBER,
                        "mtp" => TEAL,
                        _ => GOLD,
                    }),
                ),
            ]));
            lines.push(Line::styled(
                format!("     {depth}"),
                Style::new().fg(FAINT),
            ));
        };
        if m.mtp {
            let won = best.as_deref().is_some_and(|b| b.starts_with("MTP"));
            push(
                "MTP head in the GGUF".into(),
                "mtp",
                "built in · no extra file".into(),
                won,
            );
        }
        for dr in &self.drafters {
            let Some(config) = &dr.config else { continue };
            if drafter_problem(dr, m).is_some() {
                continue;
            }
            let won = best.as_deref().is_some_and(|b| b.starts_with(&dr.label));
            let depth = match (&best, won) {
                (Some(b), true) => format!(
                    "up to {} drafts · fastest here: {speedup:.2}× at {}",
                    config.max_drafts(),
                    b.split(" · ").skip(1).collect::<Vec<_>>().join(" · ")
                ),
                _ => format!("up to {} drafts", config.max_drafts()),
            };
            push(dr.label.clone(), dr.kind, depth, won);
        }
        if lines.is_empty() {
            lines.push(Line::styled(
                " none found · drafters live next to models or in the Hugging Face cache",
                Style::new().fg(FAINT),
            ));
        }
        lines
    }

    /// This Mac's speed, intelligence and agent results for a model, as bars.
    pub fn measured_lines(&self, model: &str, w: usize) -> Vec<Line<'static>> {
        let d = self.dashboard();
        let bw = w.saturating_sub(48).clamp(6, 24);
        let row = |k: &str, share: Option<f64>, label: String, note: String, color: Color| {
            let mut spans = vec![Span::styled(format!(" {k:<14}"), Style::new().fg(DIM))];
            match share {
                None => spans.push(Span::styled("not measured", Style::new().fg(FAINT))),
                Some(s) => {
                    spans.extend(bar(s, bw, color));
                    spans.push(Span::styled(
                        format!(" {label} "),
                        Style::new().fg(color).add_modifier(Modifier::BOLD),
                    ));
                    let room = w.saturating_sub(16 + bw + text_width(&label) + 2);
                    spans.push(Span::styled(clip(&note, room), Style::new().fg(FAINT)));
                }
            }
            Line::from(spans)
        };
        let mut lines = vec![heading("measured on this mac")];
        lines.push(match d.speed_of(model) {
            Some(s) => row(
                "speed",
                Some(s.speedup / 2.5),
                format!("{:.2}×", s.speedup),
                format!("{} · {}", s.best, benchui::ago(s.when)),
                if s.speedup >= 1.2 { GOOD } else { GOLD },
            ),
            None => row("speed", None, String::new(), String::new(), DIM),
        });
        lines.push(match d.iq_of(model) {
            Some((r, s)) => row(
                "intelligence",
                Some(s.share),
                s.label(),
                format!("{} · {}", r.setup_label(), benchui::ago(r.when)),
                evalui::score_color(s.share),
            ),
            None => row("intelligence", None, String::new(), String::new(), DIM),
        });
        lines.push(match d.agent_of(model) {
            Some(r) => {
                let t = r.totals();
                let share = t.solved as f64 / t.done.max(1) as f64;
                row(
                    "agent tasks",
                    Some(share),
                    format!("{}/{}", t.solved, t.done),
                    format!("{} · {}", agentui::duration(t.secs), benchui::ago(r.when)),
                    evalui::score_color(share),
                )
            }
            None => row("agent tasks", None, String::new(), String::new(), DIM),
        });
        lines
    }
}
