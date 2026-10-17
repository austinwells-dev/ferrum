//! The Bench tab's frame, shared by speed, intelligence and agentic coding:
//! a strip of measures along the top, the setup and its queue in a column on
//! the left, a chart and a table of rows filling the rest, and notes in a
//! column of their own on a wide window. On a narrow one, tab moves between
//! the groups. Each measure only supplies the contents (`Parts`).
use crate::benchui::{BenchPhase, BenchState};
use crate::evalui::Measure;
use crate::layout::{self, fit_r, fit_w, list_row, pane};
use crate::*;

/// One bar of a chart.
pub struct ChartBar {
    pub label: String,
    /// A short column after the label, like a draft depth.
    pub sub: String,
    /// 0..=1 of the chart's scale.
    pub share: f64,
    /// Drawn on a lighter track past the bar: a ± range.
    pub margin: f64,
    pub color: Color,
    /// The number at the end.
    pub value: String,
    pub value_color: Color,
    pub star: bool,
    /// Still being measured: drawn as a pulse.
    pub pending: bool,
}

/// One task tile.
pub struct Tile {
    pub mark: String,
    pub color: Color,
    pub title: String,
    pub lines: Vec<String>,
}

pub enum Chart {
    Bars(Vec<ChartBar>),
    Tiles(Vec<Tile>),
    Empty(Vec<Line<'static>>),
}

/// What a measure shows in each pane.
pub struct Parts {
    pub form: Vec<Line<'static>>,
    /// The selected form line, when the selection is in the form.
    pub form_sel: Option<usize>,
    pub queue_title: String,
    pub queue_note: String,
    pub queue: Vec<Line<'static>>,
    pub queue_sel: Option<usize>,
    /// Help for the selected setup row, pinned under the form.
    pub help: String,
    pub chart_title: String,
    pub chart_note: String,
    pub chart: Chart,
    pub rows_title: String,
    pub rows_note: String,
    pub rows: Vec<Line<'static>>,
    /// Keep this row in view (None: show the top).
    pub rows_keep: Option<usize>,
    /// Show the newest rows.
    pub rows_tail: bool,
    pub notes: Vec<Line<'static>>,
    pub running: bool,
    /// The chart gives up its height to a live feed below it.
    pub small_chart: bool,
}

/// Widths the parts are laid out for.
#[derive(Clone, Copy)]
pub struct Widths {
    pub form: usize,
    pub chart: usize,
    pub rows: usize,
    pub notes: usize,
}

/// The groups a narrow window shows one at a time.
pub const GROUPS: [&str; 4] = ["Setup", "Chart", "Rows", "Notes"];

/// One row of a setup form: label on the left, value on the right, the
/// Run row as a button.
pub fn form_row(
    on: bool,
    focused: bool,
    w: usize,
    label: &str,
    value: &str,
    warn: bool,
    locked: bool,
) -> Line<'static> {
    let run = label.starts_with('▶');
    if run {
        let style = if on && focused {
            Style::new()
                .fg(Color::Black)
                .bg(EMBER)
                .add_modifier(Modifier::BOLD)
        } else {
            Style::new().fg(EMBER).add_modifier(Modifier::BOLD)
        };
        return Line::from(vec![
            Span::raw(" "),
            Span::styled(fit_w(&format!(" {label}"), w.saturating_sub(2)), style),
        ]);
    }
    let shown = if on && focused && !locked {
        format!("‹ {value} ›")
    } else {
        value.to_string()
    };
    let label_w = w.saturating_sub(text_width(&shown) + 4).max(6);
    list_row(
        on && !locked,
        focused,
        w,
        vec![
            Span::styled(
                format!(" {}", fit_w(label, label_w)),
                Style::new().fg(if on && !locked { Color::White } else { TEXT }),
            ),
            Span::styled(
                format!("{shown} "),
                Style::new()
                    .fg(if locked {
                        DIM
                    } else if warn {
                        BAD
                    } else {
                        EMBER
                    })
                    .add_modifier(Modifier::BOLD),
            ),
        ],
    )
}

impl App {
    /// Laid out for a body: (left column, middle column, notes column).
    /// While a run goes, the notes give their column to the live panes.
    fn bench_columns(&self, body: Rect, running: bool) -> Option<(Rect, Rect, Option<Rect>)> {
        let (left, rest) = layout::columns(body, self.prefs.bench, 34, 56, self.prefs.compact);
        let rest = rest?;
        if body.width >= 170 && !running {
            let [mid, notes] =
                Layout::horizontal([Constraint::Min(56), Constraint::Length(44)]).areas(rest);
            Some((left, mid, Some(notes)))
        } else {
            Some((left, rest, None))
        }
    }

    pub fn draw_bench_view(&self, f: &mut Frame, body: Rect, b: &BenchState) {
        let [strip, body] =
            Layout::vertical([Constraint::Length(2), Constraint::Min(4)]).areas(body);
        let cols = self.bench_columns(body, b.phase == BenchPhase::Running);
        let widths = match cols {
            Some((l, m, n)) => Widths {
                form: inner_w(l),
                chart: inner_w(m),
                rows: inner_w(m),
                notes: n.map_or(inner_w(m), inner_w),
            },
            None => {
                let w = inner_w(body);
                Widths {
                    form: w,
                    chart: w,
                    rows: w,
                    notes: w,
                }
            }
        };
        let parts = match b.measure {
            Measure::Speed => self.speed_parts(b, widths),
            Measure::Intelligence => self.iq_parts(b, widths),
            Measure::Agent => self.agent_parts(b, widths),
        };
        self.draw_strip(f, strip, b, cols.is_none());
        let setup_focus = b.phase == BenchPhase::Setup;
        match cols {
            Some((left, mid, notes)) => {
                self.draw_setup_column(f, left, &parts, setup_focus);
                let mut chart_h = ((mid.height as f32) * 0.56).round() as u16;
                // Tiles hold three lines each, or one while a run's live
                // feed wants the room: the grid takes what they need.
                if let Chart::Tiles(tiles) = &parts.chart {
                    let (cell_w, cell_h) = if parts.running {
                        (TILE_LINE_W, 1)
                    } else {
                        (24, TILE_H)
                    };
                    let cols = (mid.width.saturating_sub(2) as usize / cell_w).clamp(1, 5);
                    let want = tiles.len().div_ceil(cols) as u16 * cell_h + 2;
                    chart_h = chart_h.min(want);
                }
                if parts.small_chart
                    && let Chart::Bars(bars) = &parts.chart
                {
                    chart_h = chart_h.min(bars.len() as u16 + 2);
                }
                let [chart, rows] =
                    Layout::vertical([Constraint::Length(chart_h), Constraint::Min(4)]).areas(mid);
                draw_chart_pane(f, chart, &parts, !setup_focus);
                draw_rows_pane(f, rows, &parts, false);
                if let Some(n) = notes {
                    draw_notes_pane(f, n, &parts);
                }
            }
            None => match self.bench_pane.min(GROUPS.len() - 1) {
                0 => self.draw_setup_column(f, body, &parts, true),
                1 => draw_chart_pane(f, body, &parts, true),
                2 => draw_rows_pane(f, body, &parts, true),
                _ => draw_notes_pane(f, body, &parts),
            },
        }
    }

    /// Measures as tabs on the left, the model and progress on the right,
    /// and on a narrow window the pane groups under them.
    fn draw_strip(&self, f: &mut Frame, area: Rect, b: &BenchState, narrow: bool) {
        let [top, under] =
            Layout::vertical([Constraint::Length(1), Constraint::Length(1)]).areas(area);
        let mut spans = vec![Span::raw(" ")];
        for (m, name) in [
            (Measure::Speed, "◷ Speed"),
            (Measure::Intelligence, "✦ Intelligence"),
            (Measure::Agent, "▣ Agentic coding"),
        ] {
            if m == b.measure {
                spans.push(Span::styled(
                    format!(" {name} "),
                    Style::new()
                        .fg(Color::Black)
                        .bg(EMBER)
                        .add_modifier(Modifier::BOLD),
                ));
            } else {
                spans.push(Span::styled(format!(" {name} "), Style::new().fg(DIM)));
            }
            spans.push(Span::raw(" "));
        }
        if b.phase == BenchPhase::Setup {
            spans.push(Span::styled(" m switches", Style::new().fg(FAINT)));
        }
        let left_w: usize = spans.iter().map(|s| text_width(&s.content)).sum();
        f.render_widget(Paragraph::new(Line::from(spans)), top);
        let (text, color) = self.bench_progress(b);
        let model = self.models.get(b.model).map_or("", |m| m.name.as_str());
        let right = format!("{model}  ·  {}  ", b.machine.label());
        let room = (top.width as usize).saturating_sub(left_w + text_width(&text) + 6);
        let right = if room >= 20 {
            clip(&right, room)
        } else {
            String::new()
        };
        f.render_widget(
            Paragraph::new(Line::from(vec![
                Span::styled(right, Style::new().fg(DIM)),
                Span::styled(
                    format!("{text} "),
                    Style::new().fg(color).add_modifier(Modifier::BOLD),
                ),
            ]))
            .alignment(Alignment::Right),
            top,
        );
        if narrow {
            let at = self.bench_pane.min(GROUPS.len() - 1);
            let mut spans = vec![Span::raw(" ")];
            for (i, g) in GROUPS.iter().enumerate() {
                spans.push(Span::styled(
                    format!(" {g} "),
                    if i == at {
                        Style::new()
                            .fg(GOLD)
                            .bg(KEYCAP)
                            .add_modifier(Modifier::BOLD)
                    } else {
                        Style::new().fg(DIM)
                    },
                ));
            }
            spans.push(Span::styled("  tab switches", Style::new().fg(FAINT)));
            f.render_widget(Paragraph::new(Line::from(spans)), under);
        }
    }

    /// The phase, or how far a run has got, for the strip.
    fn bench_progress(&self, b: &BenchState) -> (String, Color) {
        let t = self.elapsed();
        let stopping = match b.measure {
            Measure::Speed => b.stopping,
            Measure::Intelligence => b.iq.stopping,
            Measure::Agent => b.ag.stopping,
        };
        match b.phase {
            BenchPhase::Setup => ("SETUP".into(), GOLD),
            BenchPhase::Results => ("RESULTS".into(), GOOD),
            BenchPhase::Running if stopping => (format!("{} stopping", spinner(t)), GOLD),
            BenchPhase::Running => {
                let (done, total, secs) = match b.measure {
                    Measure::Speed => (
                        b.rows.len() + b.chunk_rows.len(),
                        b.expected,
                        b.started.elapsed().as_secs(),
                    ),
                    Measure::Intelligence => (
                        b.iq.answers.len(),
                        b.iq.expected,
                        b.iq.started.elapsed().as_secs(),
                    ),
                    Measure::Agent => (
                        b.ag.done.len(),
                        b.ag.tasks.len(),
                        b.ag.started.elapsed().as_secs(),
                    ),
                };
                let left = (done > 0 && done < total)
                    .then(|| secs * (total - done) as u64 / done as u64)
                    .map_or(String::new(), |l| {
                        format!(" · ~{} min left", l.div_ceil(60))
                    });
                (
                    format!(
                        "{} {done}/{total} · {}m{:02}s{left}",
                        spinner(t),
                        secs / 60,
                        secs % 60
                    ),
                    EMBER,
                )
            }
        }
    }

    fn draw_setup_column(&self, f: &mut Frame, area: Rect, parts: &Parts, focused: bool) {
        let help_h = if parts.help.is_empty() || !focused {
            0
        } else {
            4
        };
        let form_h = (parts.form.len() as u16 + 2 + help_h)
            .min(area.height.saturating_sub(5).max(area.height / 2));
        let [form, queue] =
            Layout::vertical([Constraint::Length(form_h), Constraint::Min(3)]).areas(area);
        let form_focus = focused && parts.form_sel.is_some();
        let note = match (focused, parts.running) {
            (true, _) => "",
            (false, true) => "locked while running",
            (false, false) => "r changes it",
        };
        let inner = pane(f, form, "Setup", form_focus, note);
        let inner = if parts.help.is_empty() || !focused {
            inner
        } else {
            layout::draw_help(f, inner, &parts.help)
        };
        layout::show(f, inner, parts.form.clone(), parts.form_sel);
        let inner = pane(
            f,
            queue,
            &parts.queue_title,
            focused && parts.queue_sel.is_some(),
            &parts.queue_note,
        );
        layout::show(f, inner, parts.queue.clone(), parts.queue_sel);
    }
}

fn draw_chart_pane(f: &mut Frame, area: Rect, parts: &Parts, focused: bool) {
    let inner = pane(f, area, &parts.chart_title, focused, &parts.chart_note);
    match &parts.chart {
        Chart::Bars(bars) => draw_bars(f, inner, bars),
        Chart::Tiles(tiles) => draw_tiles(f, inner, tiles),
        Chart::Empty(lines) => f.render_widget(Paragraph::new(lines.clone()), inner),
    }
}

fn draw_rows_pane(f: &mut Frame, area: Rect, parts: &Parts, focused: bool) {
    let inner = pane(f, area, &parts.rows_title, focused, &parts.rows_note);
    if parts.rows_tail {
        layout::show_tail(f, inner, parts.rows.clone());
    } else {
        layout::show(f, inner, parts.rows.clone(), parts.rows_keep);
    }
}

fn draw_notes_pane(f: &mut Frame, area: Rect, parts: &Parts) {
    let inner = pane(f, area, "Notes", false, "");
    f.render_widget(Paragraph::new(parts.notes.clone()), inner);
}

/// Horizontal bars that use the pane's height: one to three rows each, with
/// a row between them when there is room.
pub fn draw_bars(f: &mut Frame, inner: Rect, bars: &[ChartBar]) {
    let n = bars.len();
    if n == 0 || inner.height == 0 {
        return;
    }
    let h = inner.height as usize;
    let per = (h / n).max(1);
    let thick = match per {
        0..=2 => 1,
        3 => 2,
        _ => 3,
    };
    let gap = usize::from(per > thick);
    let label_w = bars
        .iter()
        .map(|b| text_width(&b.label))
        .max()
        .unwrap_or(8)
        .clamp(8, (inner.width as usize / 3).max(8));
    let sub_w = bars.iter().map(|b| text_width(&b.sub)).max().unwrap_or(0);
    let sub_w = if sub_w > 0 { sub_w + 1 } else { 0 };
    let value_w = bars
        .iter()
        .map(|b| text_width(&b.value))
        .max()
        .unwrap_or(4)
        .min(24)
        + 1;
    let bar_w = (inner.width as usize)
        .saturating_sub(label_w + sub_w + value_w + 4)
        .max(4);
    let mut lines: Vec<Line> = Vec::new();
    // Too many bars for the pane: show the ones that fit, first first.
    for b in bars.iter().take(h.div_ceil(thick + gap)) {
        for row in 0..thick {
            let mid = row == thick / 2;
            let mut spans = vec![
                Span::styled(
                    if mid && b.star { "★" } else { " " },
                    Style::new().fg(EMBER),
                ),
                Span::styled(
                    if mid {
                        fit_w(&b.label, label_w)
                    } else {
                        " ".repeat(label_w)
                    },
                    Style::new()
                        .fg(if b.star { Color::White } else { TEXT })
                        .add_modifier(if b.star {
                            Modifier::BOLD
                        } else {
                            Modifier::empty()
                        }),
                ),
                Span::raw(" "),
            ];
            if sub_w > 0 {
                spans.push(Span::styled(
                    if mid {
                        fit_w(&b.sub, sub_w)
                    } else {
                        " ".repeat(sub_w)
                    },
                    Style::new().fg(DIM),
                ));
            }
            if b.pending {
                let filled = ((b.share * bar_w as f64).round() as usize).min(bar_w);
                spans.push(Span::styled(
                    "▒".repeat(filled),
                    Style::new().fg(EMBER).bg(TRACK),
                ));
                spans.push(Span::styled(
                    " ".repeat(bar_w - filled),
                    Style::new().bg(TRACK),
                ));
            } else if b.margin > 0.0 {
                spans.extend(crate::evalui::bar_ranged(b.share, b.margin, bar_w, b.color));
            } else {
                spans.extend(layout::bar(b.share, bar_w, b.color));
            }
            spans.push(Span::styled(
                if mid {
                    format!(" {}", fit_w(&b.value, value_w))
                } else {
                    String::new()
                },
                Style::new().fg(b.value_color).add_modifier(Modifier::BOLD),
            ));
            lines.push(Line::from(spans));
        }
        for _ in 0..gap {
            lines.push(Line::default());
        }
    }
    lines.truncate(h);
    f.render_widget(Paragraph::new(lines), inner);
}

/// Rows of one task tile: its border and three lines.
const TILE_H: u16 = 5;

/// Columns of a one-line tile.
const TILE_LINE_W: usize = 40;

/// Task tiles in a grid that fills the pane.
pub fn draw_tiles(f: &mut Frame, inner: Rect, tiles: &[Tile]) {
    let n = tiles.len();
    if n == 0 || inner.width < 10 {
        return;
    }
    // Too short for boxed tiles: one line per task, in columns.
    let box_rows = n.div_ceil((inner.width as usize / 24).clamp(1, 5).min(n));
    if (inner.height as usize) < box_rows * TILE_H as usize {
        let cols = (inner.width as usize / TILE_LINE_W).clamp(1, 5).min(n);
        let cell = inner.width as usize / cols;
        let rows = n.div_ceil(cols);
        let mut lines = Vec::new();
        for r in 0..rows {
            let mut spans = Vec::new();
            for c in 0..cols {
                let Some(t) = tiles.get(r * cols + c) else {
                    break;
                };
                let status = t.lines.first().cloned().unwrap_or_default();
                let text = format!("{} · {status}", t.title);
                spans.push(Span::styled(
                    format!(" {} ", t.mark),
                    Style::new().fg(t.color).add_modifier(Modifier::BOLD),
                ));
                spans.push(Span::styled(
                    fit_w(&text, cell.saturating_sub(4)),
                    Style::new().fg(if t.color == FAINT { DIM } else { TEXT }),
                ));
                spans.push(Span::raw(" "));
            }
            lines.push(Line::from(spans));
        }
        f.render_widget(Paragraph::new(lines), inner);
        return;
    }
    if inner.height < 3 {
        return;
    }
    let cols = (inner.width as usize / 24).clamp(1, 5).min(n);
    let rows = n.div_ceil(cols);
    let tile_w = inner.width / cols as u16;
    let tile_h = (inner.height / rows as u16).max(3);
    for (i, t) in tiles.iter().enumerate() {
        let (cx, cy) = ((i % cols) as u16, (i / cols) as u16);
        let y = inner.y + cy * tile_h;
        if y + 3 > inner.y + inner.height {
            break;
        }
        let w = if cx as usize == cols - 1 {
            inner.width - cx * tile_w
        } else {
            tile_w
        };
        let h = if cy as usize == rows - 1 {
            inner.y + inner.height - y
        } else {
            tile_h
        };
        let rect = Rect::new(inner.x + cx * tile_w, y, w, h);
        let block = Block::bordered()
            .border_type(BorderType::Rounded)
            .border_style(Style::new().fg(t.color));
        let ti = block.inner(rect);
        f.render_widget(block, rect);
        let tw = ti.width as usize;
        let mut lines = vec![Line::from(vec![
            Span::styled(
                format!("{} ", t.mark),
                Style::new().fg(t.color).add_modifier(Modifier::BOLD),
            ),
            Span::styled(
                clip(&t.title, tw.saturating_sub(2)),
                Style::new().fg(TEXT).add_modifier(Modifier::BOLD),
            ),
        ])];
        for l in &t.lines {
            lines.push(Line::styled(
                format!("  {}", clip(l, tw.saturating_sub(2))),
                Style::new().fg(DIM),
            ));
        }
        f.render_widget(Paragraph::new(lines), ti);
    }
}

/// A table header line in faint text.
pub fn table_head(cols: &[(&str, usize, bool)]) -> Line<'static> {
    let text: String = cols
        .iter()
        .map(
            |(t, w, right)| {
                if *right { fit_r(t, *w) } else { fit_w(t, *w) }
            },
        )
        .collect::<Vec<_>>()
        .join(" ");
    Line::styled(format!("  {text}"), Style::new().fg(FAINT))
}

/// A legend entry: the key in bold, its meaning wrapped beside it.
pub fn legend(key: &str, what: &str, color: Color, width: usize) -> Vec<Line<'static>> {
    // A key of up to 7 columns, a space, then the meaning.
    let mut lines = layout::wrapped(what, 9, width, Style::new().fg(DIM));
    if let Some(first) = lines.first_mut() {
        first.spans[0] = Span::styled(
            format!(" {} ", fit_w(key, 7)),
            Style::new().fg(color).add_modifier(Modifier::BOLD),
        );
    }
    lines
}
