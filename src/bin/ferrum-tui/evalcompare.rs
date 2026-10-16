//! The Compare view of the intelligence benchmark: every saved result, from
//! every model, as one bar each, with its setup and this Mac's speedup.
use crate::benchui::ago;
use crate::eval::{CATEGORIES, EvalResult};
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
    pub thinking: ThinkingFilter,
    pub full_only: bool,
    /// Show the selected result's categories.
    pub detail: bool,
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
        .filter(|r| c.thinking.keeps(r) && (!c.full_only || !r.quick) && r.score().is_some())
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
        match key.code {
            KeyCode::Esc | KeyCode::Char('q') | KeyCode::Char('c') => b.iq.compare = None,
            KeyCode::Up | KeyCode::Char('k') => c.sel = c.sel.saturating_sub(1),
            KeyCode::Down | KeyCode::Char('j') | KeyCode::Tab => {
                c.sel = (c.sel + 1).min(n.saturating_sub(1))
            }
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
                " {} · {} · {} result{}",
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
            " ▒ is the 95% range of a predicted (quick) score · speed is this Mac's best speedup over plain decoding",
            Style::new().fg(FAINT),
        ));
        head.push(Line::default());
        if rows.is_empty() {
            head.push(Line::styled(
                if b.iq.results.is_empty() {
                    "   no results yet: run an intelligence test first"
                } else {
                    "   nothing matches these filters (t: thinking, f: full runs)"
                },
                Style::new().fg(GOLD),
            ));
            f.render_widget(Paragraph::new(head), inner);
            return;
        }
        let name_w = width.saturating_sub(70).clamp(16, 34);
        let setup_w = 26;
        let bar_w = width.saturating_sub(name_w + setup_w + 26).clamp(10, 24);
        let sel = c.sel.min(rows.len() - 1);
        let mut list: Vec<Line> = Vec::new();
        for (i, r) in rows.iter().enumerate() {
            let Some(s) = r.score() else { continue };
            let on = i == sel;
            let bg = if on { SELECTED } else { Color::Reset };
            let stale = current.is_some_and(|f| f != r.suite);
            let speed = b
                .saved
                .iter()
                .find(|x| x.machine == b.machine.label() && x.model == r.model)
                .map_or("–".to_string(), |x| format!("{:.2}×", x.speedup));
            let setup = format!(
                "{} · {}{}",
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
                for (t, (_, name)) in r.tallies.iter().zip(CATEGORIES) {
                    let mut spans = vec![Span::styled(
                        format!("     {name:<26} {:>3}/{:<3} ", t.right, t.n),
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
        let tail = if c.detail { 6 } else { 1 };
        let top = (sel_line + tail).saturating_sub(room);
        head.extend(list.into_iter().skip(top).take(room));
        f.render_widget(Paragraph::new(head), inner);
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
    fn lists_results_best_first_and_filters() {
        let results = vec![
            result("/a/one.gguf", true, None, 5),
            result("/a/two.gguf", false, Some(1024), 9),
            result("/a/three.gguf", false, None, 7),
        ];
        let mut c = Compare::default();
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
    }
}
