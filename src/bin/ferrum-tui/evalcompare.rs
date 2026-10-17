//! The Compare view of the intelligence benchmark: every saved result, from
//! every model, as one bar each, with its setup and this Mac's speedup.
use crate::benchui::ago;
use crate::eval::{self, CATEGORIES, Difficulty, EvalResult};
use crate::evalui::{bar_ranged, score_color};
use crate::*;

/// Which results the Compare view lists.
#[derive(Clone, Copy, Default, PartialEq)]
pub enum ThinkingFilter {
    #[default]
    All,
    Off,
    On,
}

impl ThinkingFilter {
    fn next(self) -> Self {
        match self {
            Self::All => Self::Off,
            Self::Off => Self::On,
            Self::On => Self::All,
        }
    }

    fn keeps(self, r: &EvalResult) -> bool {
        match self {
            Self::All => true,
            Self::Off => r.thinking.is_none(),
            Self::On => r.thinking.is_some(),
        }
    }

    fn label(self) -> &'static str {
        match self {
            Self::All => "thinking on and off",
            Self::Off => "thinking off",
            Self::On => "thinking on",
        }
    }
}

#[derive(Clone, Default)]
pub struct Compare {
    pub sel: usize,
    /// Only results of this difficulty (None: both).
    pub difficulty: Option<Difficulty>,
    pub thinking: ThinkingFilter,
    pub full_only: bool,
    /// Show the selected result's categories.
    pub detail: bool,
    /// `x` was pressed once: the next `x` deletes every saved result.
    pub confirm_clear: bool,
}

impl Compare {
    /// Opens on the difficulty the setup shows.
    pub fn new(difficulty: Difficulty) -> Self {
        Self {
            difficulty: Some(difficulty),
            ..Self::default()
        }
    }

    fn next_difficulty(&mut self) {
        self.difficulty = match self.difficulty {
            Some(Difficulty::Hard) => Some(Difficulty::Standard),
            Some(Difficulty::Standard) => None,
            None => Some(Difficulty::Hard),
        };
    }
}

/// One line of a column's labels.
type Label<'a> = Box<dyn Fn(&EvalResult) -> String + 'a>;

/// Screen width from which Compare uses all of it and draws columns.
pub const WIDE: u16 = 140;

/// Eighths of a cell, for the tops of the columns.
const EIGHTHS: [&str; 8] = [" ", "▁", "▂", "▃", "▄", "▅", "▆", "▇"];

/// What one column cell shows at height `y` (in rows from the bottom): solid
/// up to `solid` eighths, shaded (the predicted range) up to `upper`.
fn column_cell(y: usize, solid: usize, upper: usize) -> &'static str {
    let (lo, hi) = (y * 8, y * 8 + 8);
    if hi <= solid {
        "█"
    } else if lo < solid {
        EIGHTHS[solid - lo]
    } else if lo < upper {
        "▒"
    } else {
        " "
    }
}

/// Text `s` centered in `w` cells (clipped when longer).
fn centered_text(s: &str, w: usize) -> String {
    let s = clip(s, w);
    let n = s.chars().count();
    let left = (w - n) / 2;
    format!("{}{s}{}", " ".repeat(left), " ".repeat(w - n - left))
}

/// The model's file name without `.gguf`.
fn model_name(path: &str) -> String {
    let name = Path::new(path)
        .file_name()
        .map(|n| n.to_string_lossy().to_string())
        .unwrap_or_else(|| path.to_string());
    name.strip_suffix(".gguf")
        .map(str::to_string)
        .unwrap_or(name)
}

/// The results `c` lists, best score first.
pub fn compare_rows<'a>(results: &'a [EvalResult], c: &Compare) -> Vec<&'a EvalResult> {
    let mut rows: Vec<&EvalResult> = results
        .iter()
        .filter(|r| {
            c.difficulty.is_none_or(|d| d == r.difficulty)
                && c.thinking.keeps(r)
                && (!c.full_only || !r.quick)
                && r.score().is_some()
        })
        .collect();
    rows.sort_by(|a, b| {
        let s = |r: &EvalResult| r.score().map_or(0.0, |s| s.share);
        s(b).total_cmp(&s(a))
    });
    rows
}

impl App {
    pub fn iq_compare_key(&mut self, key: KeyEvent) {
        let Some(b) = self.bench.as_mut() else { return };
        let n =
            b.iq.compare
                .as_ref()
                .map_or(0, |c| compare_rows(&b.iq.results, c).len());
        let Some(c) = b.iq.compare.as_mut() else {
            return;
        };
        let clearing = c.confirm_clear;
        c.confirm_clear = false;
        match key.code {
            KeyCode::Char('x') if clearing => {
                b.iq.results.clear();
                eval::write_results(&eval::results_path(), &b.iq.results);
                c.sel = 0;
                c.detail = false;
                self.status = Some(("cleared every intelligence result".into(), true));
            }
            KeyCode::Char('x') => c.confirm_clear = true,
            KeyCode::Char('d') => {
                c.next_difficulty();
                c.sel = 0;
            }
            KeyCode::Esc | KeyCode::Char('q') | KeyCode::Char('c') => b.iq.compare = None,
            KeyCode::Up | KeyCode::Left | KeyCode::Char('k') | KeyCode::Char('h') => {
                c.sel = c.sel.saturating_sub(1)
            }
            KeyCode::Down
            | KeyCode::Right
            | KeyCode::Char('j')
            | KeyCode::Char('l')
            | KeyCode::Tab => c.sel = (c.sel + 1).min(n.saturating_sub(1)),
            KeyCode::Enter | KeyCode::Char(' ') => c.detail = !c.detail,
            KeyCode::Char('t') => {
                c.thinking = c.thinking.next();
                c.sel = 0;
            }
            KeyCode::Char('f') => {
                c.full_only = !c.full_only;
                c.sel = 0;
            }
            _ => {}
        }
    }

    pub fn draw_iq_compare(&self, f: &mut Frame, area: Rect, b: &BenchState) {
        let block = panel("Compare", true);
        let inner = block.inner(area);
        f.render_widget(block, area);
        let width = inner.width as usize;
        let Some(c) = &b.iq.compare else { return };
        let rows = compare_rows(&b.iq.results, c);
        let current = b.iq.suite.as_ref().ok().map(|s| s.fingerprint.as_str());
        let mut head = vec![Line::styled(
            format!(
                " {} · {} · {} · {} result{}",
                c.difficulty.map_or("hard and standard", |d| d.key()),
                c.thinking.label(),
                if c.full_only {
                    "full runs only"
                } else {
                    "quick and full runs"
                },
                rows.len(),
                if rows.len() == 1 { "" } else { "s" }
            ),
            Style::new().fg(DIM),
        )];
        head.push(Line::styled(
            clip(
                " ▒ 95% range of a quick (predicted) score · × this Mac's best speedup for the model",
                width,
            ),
            Style::new().fg(FAINT),
        ));
        if c.confirm_clear {
            head.push(Line::styled(
                format!(
                    " press x again to delete all {} saved results (every model and difficulty) · any other key keeps them",
                    b.iq.results.len()
                ),
                Style::new().fg(BAD).add_modifier(Modifier::BOLD),
            ));
        }
        head.push(Line::default());
        if rows.is_empty() {
            head.push(Line::styled(
                if b.iq.results.is_empty() {
                    "   no results yet: run an intelligence test first"
                } else {
                    "   nothing matches these filters (d: difficulty, t: thinking, f: full runs)"
                },
                Style::new().fg(GOLD),
            ));
            f.render_widget(Paragraph::new(head), inner);
            return;
        }
        if inner.width >= WIDE - 4 {
            head.extend(self.compare_columns(b, c, &rows, inner, head.len()));
            f.render_widget(Paragraph::new(head), inner);
            return;
        }
        // Room left of the marker (3), score (11) and speed (7) columns and
        // the gaps, shared so the model name gets what the bar does not need.
        let setup_w = 28;
        let bar_w = width.saturating_sub(setup_w + 24 + 40).clamp(10, 20);
        let name_w = width.saturating_sub(setup_w + bar_w + 24).clamp(16, 48);
        let sel = c.sel.min(rows.len() - 1);
        let mut list: Vec<Line> = Vec::new();
        for (i, r) in rows.iter().enumerate() {
            let Some(s) = r.score() else { continue };
            let on = i == sel;
            let bg = if on { SELECTED } else { Color::Reset };
            let stale = r.difficulty == b.iq.difficulty && current.is_some_and(|f| f != r.suite);
            let speed = b
                .saved
                .iter()
                .find(|x| x.machine == b.machine.label() && x.model == r.model)
                .map_or("–".to_string(), |x| format!("{:.2}×", x.speedup));
            let setup = format!(
                "{} · {} · {}{}",
                r.difficulty.key(),
                if r.quick { "quick" } else { "full" },
                r.thinking
                    .map_or("no think".to_string(), |t| format!("think ≤{t}")),
                if r.draft == "plain decoding" {
                    ""
                } else {
                    " · draft"
                }
            );
            let mut spans = vec![
                Span::styled(
                    if on { " ▌ " } else { "   " },
                    Style::new().fg(EMBER).bg(bg),
                ),
                Span::styled(
                    format!("{:<name_w$} ", clip(&model_name(&r.model), name_w)),
                    Style::new().fg(if on { Color::White } else { TEXT }).bg(bg),
                ),
                Span::styled(
                    format!("{:<setup_w$} ", clip(&setup, setup_w)),
                    Style::new().fg(if stale { FAINT } else { DIM }).bg(bg),
                ),
            ];
            spans.extend(
                bar_ranged(
                    s.share,
                    if s.predicted { s.margin } else { 0.0 },
                    bar_w,
                    score_color(s.share),
                )
                .into_iter()
                .map(|sp| {
                    let style = sp.style.bg(bg);
                    sp.style(style)
                }),
            );
            spans.push(Span::styled(
                format!(" {:>9} ", s.label()),
                Style::new()
                    .fg(score_color(s.share))
                    .add_modifier(Modifier::BOLD)
                    .bg(bg),
            ));
            spans.push(Span::styled(
                format!("{speed:>6} "),
                Style::new().fg(DIM).bg(bg),
            ));
            list.push(Line::from(spans));
            if on && c.detail {
                list.push(Line::styled(
                    format!(
                        "     {} · K/V {} · {} wrong · {}{}",
                        r.draft,
                        r.kv_cache.to_lowercase(),
                        r.wrong.len(),
                        ago(r.when),
                        if stale {
                            " · older question files"
                        } else {
                            ""
                        }
                    ),
                    Style::new().fg(FAINT),
                ));
                for (i, t) in r.tallies.iter().enumerate() {
                    let mut spans = vec![Span::styled(
                        format!(
                            "     {:<34} {:>3}/{:<3} ",
                            r.difficulty.cat_name(i),
                            t.right,
                            t.n
                        ),
                        Style::new().fg(TEXT),
                    )];
                    if t.n > 0 {
                        spans.extend(bar_ranged(t.share(), 0.0, 16, score_color(t.share())));
                        spans.push(Span::styled(
                            format!(" {:>4.0}%", t.share() * 100.0),
                            Style::new().fg(score_color(t.share())),
                        ));
                    }
                    list.push(Line::from(spans));
                }
            }
        }
        // Keep the selected row (and its details) in view.
        let room = (inner.height as usize).saturating_sub(head.len());
        let sel_line = list
            .iter()
            .position(|l| l.spans.first().is_some_and(|s| s.content.starts_with(" ▌")))
            .unwrap_or(0);
        let tail = if c.detail { CATEGORIES.len() + 2 } else { 1 };
        let top = (sel_line + tail).saturating_sub(room);
        head.extend(list.into_iter().skip(top).take(room));
        f.render_widget(Paragraph::new(head), inner);
    }

    /// The wide layout: one column per result, best first, labels under the
    /// axis, the selected result's details below.
    fn compare_columns(
        &self,
        b: &BenchState,
        c: &Compare,
        rows: &[&EvalResult],
        inner: Rect,
        used: usize,
    ) -> Vec<Line<'static>> {
        let width = inner.width as usize;
        let n = rows.len();
        let col_w = ((width - 2) / n).clamp(9, 18);
        let visible = ((width - 2) / col_w).max(1);
        let sel = c.sel.min(n - 1);
        let first = sel
            .saturating_sub(visible - 1)
            .min(n.saturating_sub(visible));
        let shown = &rows[first..(first + visible).min(n)];
        let details = if c.detail { CATEGORIES.len() + 3 } else { 3 };
        let chart_h = (inner.height as usize)
            .saturating_sub(used + 5 + details)
            .max(6);
        // The top row is kept for the value above a full column.
        let full = (chart_h - 1) * 8;
        let bar_w = col_w - 3;
        let mut lines = Vec::new();
        for row in 0..chart_h {
            let y = chart_h - 1 - row;
            let mut spans = vec![Span::raw(" ")];
            for (k, r) in shown.iter().enumerate() {
                let Some(s) = r.score() else { continue };
                let on = first + k == sel;
                let m = if s.predicted { s.margin } else { 0.0 };
                let solid = ((s.share - m).clamp(0.0, 1.0) * full as f64).round() as usize;
                let upper = ((s.share + m).clamp(0.0, 1.0) * full as f64).round() as usize;
                let label_y = upper.max(solid).div_ceil(8);
                let color = score_color(s.share);
                if y == label_y {
                    spans.push(Span::styled(
                        centered_text(&s.label(), col_w),
                        Style::new()
                            .fg(if on { Color::White } else { color })
                            .add_modifier(Modifier::BOLD),
                    ));
                } else {
                    spans.push(Span::raw(" "));
                    spans.push(Span::styled(
                        column_cell(y, solid, upper).repeat(bar_w),
                        Style::new().fg(color),
                    ));
                    spans.push(Span::raw("  "));
                }
            }
            lines.push(Line::from(spans));
        }
        lines.push(Line::styled(
            format!(" {}", "─".repeat(shown.len() * col_w)),
            Style::new().fg(FAINT),
        ));
        let label_rows: [Label; 4] = [
            Box::new(|r| model_name(&r.model)),
            Box::new(|r| {
                format!(
                    "{} {}",
                    r.difficulty.key(),
                    if r.quick { "quick" } else { "full" }
                )
            }),
            Box::new(|r| {
                format!(
                    "{}{}",
                    r.thinking
                        .map_or("no think".to_string(), |t| format!("think {t}")),
                    if r.draft == "plain decoding" {
                        ""
                    } else {
                        " +d"
                    }
                )
            }),
            Box::new(|r| {
                b.saved
                    .iter()
                    .find(|x| x.machine == b.machine.label() && x.model == r.model)
                    .map_or("–".to_string(), |x| format!("{:.2}×", x.speedup))
            }),
        ];
        for (i, label) in label_rows.iter().enumerate() {
            let mut spans = vec![Span::raw(" ")];
            for (k, r) in shown.iter().enumerate() {
                let on = first + k == sel;
                spans.push(Span::styled(
                    format!(" {:<w$}", clip(&label(r), col_w - 2), w = col_w - 1),
                    Style::new()
                        .fg(if on {
                            Color::White
                        } else if i == 0 {
                            TEXT
                        } else {
                            DIM
                        })
                        .bg(if on { SELECTED } else { Color::Reset }),
                ));
            }
            lines.push(Line::from(spans));
        }
        if n > visible {
            lines.push(Line::styled(
                format!(
                    " results {}-{} of {n} · ←→ scrolls",
                    first + 1,
                    first + shown.len()
                ),
                Style::new().fg(FAINT),
            ));
        } else {
            lines.push(Line::default());
        }
        // The selected result in full.
        let r = rows[sel];
        let stale = r.difficulty == b.iq.difficulty
            && b.iq
                .suite
                .as_ref()
                .ok()
                .is_some_and(|s| s.fingerprint != r.suite);
        lines.push(Line::from(vec![
            Span::styled(
                format!(" ▲ {}", model_name(&r.model)),
                Style::new().fg(Color::White).add_modifier(Modifier::BOLD),
            ),
            Span::styled(
                format!(
                    "  {} · {} · K/V {} · {} wrong · {}{}",
                    r.setup_label(),
                    r.score().map_or(String::new(), |s| s.label()),
                    r.kv_cache.to_lowercase(),
                    r.wrong.len(),
                    ago(r.when),
                    if stale {
                        " · older question files"
                    } else {
                        ""
                    }
                ),
                Style::new().fg(DIM),
            ),
        ]));
        if c.detail {
            for (i, t) in r.tallies.iter().enumerate() {
                let mut spans = vec![Span::styled(
                    format!(
                        "   {:<34} {:>3}/{:<3} ",
                        r.difficulty.cat_name(i),
                        t.right,
                        t.n
                    ),
                    Style::new().fg(TEXT),
                )];
                if t.n > 0 {
                    spans.extend(bar_ranged(t.share(), 0.0, 24, score_color(t.share())));
                    spans.push(Span::styled(
                        format!(" {:>4.0}%", t.share() * 100.0),
                        Style::new().fg(score_color(t.share())),
                    ));
                }
                lines.push(Line::from(spans));
            }
        } else {
            lines.push(Line::styled(
                "   enter shows its categories",
                Style::new().fg(FAINT),
            ));
        }
        lines
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::eval::Tally;

    fn result(model: &str, quick: bool, thinking: Option<usize>, right: usize) -> EvalResult {
        EvalResult {
            model: model.into(),
            when: 1,
            difficulty: crate::eval::Difficulty::Hard,
            quick,
            thinking,
            draft: "plain decoding".into(),
            kv_cache: "F16".into(),
            suite: "s".into(),
            tallies: vec![
                Tally {
                    right,
                    n: 10,
                    total: if quick { 100 } else { 10 },
                };
                4
            ],
            wrong: Vec::new(),
        }
    }

    #[test]
    fn draws_column_cells() {
        // Solid to 12 eighths, shaded range to 20: one full row, half a row,
        // then the range, then nothing.
        assert_eq!(column_cell(0, 12, 20), "█");
        assert_eq!(column_cell(1, 12, 20), "▄");
        assert_eq!(column_cell(2, 12, 20), "▒");
        assert_eq!(column_cell(3, 12, 20), " ");
        assert_eq!(centered_text("85%", 9), "   85%   ");
        assert_eq!(centered_text("a long label", 5).chars().count(), 5);
    }

    #[test]
    fn lists_results_best_first_and_filters() {
        let results = vec![
            result("/a/one.gguf", true, None, 5),
            result("/a/two.gguf", false, Some(1024), 9),
            result("/a/three.gguf", false, None, 7),
        ];
        let mut c = Compare::new(crate::eval::Difficulty::Hard);
        let rows: Vec<&str> = compare_rows(&results, &c)
            .iter()
            .map(|r| r.model.as_str())
            .collect();
        assert_eq!(rows, vec!["/a/two.gguf", "/a/three.gguf", "/a/one.gguf"]);
        c.thinking = ThinkingFilter::Off;
        assert_eq!(compare_rows(&results, &c).len(), 2);
        c.full_only = true;
        assert_eq!(compare_rows(&results, &c)[0].model, "/a/three.gguf");
        c.thinking = c.thinking.next();
        assert_eq!(compare_rows(&results, &c)[0].model, "/a/two.gguf");
        assert_eq!(model_name("/m/Tiel-Coder-35B.gguf"), "Tiel-Coder-35B");
        // Standard results only show once the filter allows them.
        let mut easy = result("/a/four.gguf", false, None, 10);
        easy.difficulty = crate::eval::Difficulty::Standard;
        let both = vec![easy, result("/a/five.gguf", false, None, 6)];
        let mut c = Compare::new(crate::eval::Difficulty::Hard);
        assert_eq!(compare_rows(&both, &c).len(), 1);
        c.next_difficulty();
        assert_eq!(compare_rows(&both, &c)[0].model, "/a/four.gguf");
        c.next_difficulty();
        assert_eq!(compare_rows(&both, &c).len(), 2);
    }
}
