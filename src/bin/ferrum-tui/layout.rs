//! How the screens divide the window (Settings › Layout) and the small
//! drawing pieces every pane shares. Every pane fills the space it is given:
//! one pane per split takes what the others leave, so nothing is left blank.
use crate::*;

/// When chat and agent sessions show their side panel.
#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Panel {
    /// From `PANEL_AUTO` columns.
    Auto,
    Always,
    Off,
}

/// The width at which an automatic side panel appears.
pub const PANEL_AUTO: u16 = 110;

impl Panel {
    const ALL: [Panel; 3] = [Panel::Auto, Panel::Always, Panel::Off];

    pub fn key(self) -> &'static str {
        match self {
            Panel::Auto => "auto",
            Panel::Always => "always",
            Panel::Off => "off",
        }
    }

    fn parse(s: &str) -> Self {
        Self::ALL
            .into_iter()
            .find(|p| p.key() == s)
            .unwrap_or(Panel::Auto)
    }

    pub fn cycle(self, step: i32) -> Self {
        let at = Self::ALL.iter().position(|&p| p == self).unwrap_or(0) as i32;
        Self::ALL[(at + step).rem_euclid(3) as usize]
    }
}

/// Split sizes and widths, saved in tui.json under `layout`.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct Prefs {
    /// Share of the width the setup list takes in Chat, Agents and Serve.
    pub setups: f32,
    /// Share of the width the Bench setup column takes.
    pub bench: f32,
    /// Share of the width the Serve endpoint column takes.
    pub serve: f32,
    /// Share of the height the Models table takes.
    pub models: f32,
    pub panel: Panel,
    /// Columns of the chat and agent side panel.
    pub panel_w: u16,
    /// Columns chat prose wraps at.
    pub prose: u16,
    /// Below this many columns a screen shows one group of panes at a time.
    pub compact: u16,
}

impl Default for Prefs {
    fn default() -> Self {
        Self {
            setups: 0.42,
            bench: 0.28,
            serve: 0.34,
            models: 0.5,
            panel: Panel::Auto,
            panel_w: 40,
            prose: 100,
            compact: 100,
        }
    }
}

/// An adjustable size in Settings › Layout.
pub struct Knob {
    pub label: &'static str,
    pub help: &'static str,
}

impl Prefs {
    pub fn load(saved: &Json) -> Self {
        let d = Self::default();
        let share = |k: &str, def: f32, lo: f32, hi: f32| {
            saved[k].as_f64().map_or(def, |v| (v as f32).clamp(lo, hi))
        };
        let cols = |k: &str, def: u16, lo: u16, hi: u16| {
            saved[k]
                .as_u64()
                .map_or(def, |v| (v.min(u16::MAX as u64) as u16).clamp(lo, hi))
        };
        Self {
            setups: share("setups", d.setups, 0.25, 0.7),
            bench: share("bench", d.bench, 0.18, 0.5),
            serve: share("serve", d.serve, 0.2, 0.6),
            models: share("models", d.models, 0.25, 0.8),
            panel: saved["panel"].as_str().map_or(d.panel, Panel::parse),
            panel_w: cols("panel_w", d.panel_w, 28, 64),
            prose: cols("prose", d.prose, 60, 240),
            compact: cols("compact", d.compact, 60, 160),
        }
    }

    pub fn to_json(&self) -> Json {
        let r = |v: f32| (v as f64 * 100.0).round() / 100.0;
        json!({
            "setups": r(self.setups),
            "bench": r(self.bench),
            "serve": r(self.serve),
            "models": r(self.models),
            "panel": self.panel.key(),
            "panel_w": self.panel_w,
            "prose": self.prose,
            "compact": self.compact,
        })
    }

    /// The adjustable sizes, in Settings order.
    pub const KNOBS: [Knob; 8] = [
        Knob {
            label: "Setup list width",
            help: "How much of the width the list of setups takes in Chat, Agents and Serve. [ and ] change it on those tabs too.",
        },
        Knob {
            label: "Bench setup column",
            help: "Width of the setup and queue column on Bench; the chart and rows take the rest. [ and ] change it on Bench too.",
        },
        Knob {
            label: "Serve endpoint column",
            help: "Width of the endpoint column beside the server log.",
        },
        Knob {
            label: "Models table height",
            help: "The most of the height the model table takes when it has that many models; the detail and drafters fill the rest.",
        },
        Knob {
            label: "Chat side panel",
            help: "The model and context panel beside a chat, the plan and files beside an agent. Auto shows it from 110 columns; ctrl-b toggles it in a session.",
        },
        Knob {
            label: "Side panel width",
            help: "Columns the chat and agent side panel takes.",
        },
        Knob {
            label: "Chat text width",
            help: "Prose wraps here so it stays readable; the conversation pane itself always fills the window.",
        },
        Knob {
            label: "One pane group below",
            help: "Narrower than this, Bench shows one group of panes at a time (tab moves between them) and the other screens stack their panes.",
        },
    ];

    pub fn knob_value(&self, i: usize) -> String {
        let pct = |v: f32| format!("{}%", (v * 100.0).round());
        match i {
            0 => pct(self.setups),
            1 => pct(self.bench),
            2 => pct(self.serve),
            3 => pct(self.models),
            4 => self.panel.key().into(),
            5 => format!("{} cols", self.panel_w),
            6 => format!("{} cols", self.prose),
            _ => format!("{} cols", self.compact),
        }
    }

    /// The share a knob shows as a bar (None: not a share).
    pub fn knob_share(&self, i: usize) -> Option<f32> {
        match i {
            0 => Some(self.setups),
            1 => Some(self.bench),
            2 => Some(self.serve),
            3 => Some(self.models),
            _ => None,
        }
    }

    pub fn nudge(&mut self, i: usize, step: i32) {
        let s = step as f32;
        let n =
            |v: u16, by: i32, lo: u16, hi: u16| (v as i32 + by).clamp(lo as i32, hi as i32) as u16;
        match i {
            0 => self.setups = (self.setups + s * 0.04).clamp(0.25, 0.7),
            1 => self.bench = (self.bench + s * 0.03).clamp(0.18, 0.5),
            2 => self.serve = (self.serve + s * 0.03).clamp(0.2, 0.6),
            3 => self.models = (self.models + s * 0.05).clamp(0.25, 0.8),
            4 => self.panel = self.panel.cycle(step),
            5 => self.panel_w = n(self.panel_w, step * 4, 28, 64),
            6 => self.prose = n(self.prose, step * 10, 60, 240),
            _ => self.compact = n(self.compact, step * 10, 60, 160),
        }
    }
}

/// Split `area` left and right, the left taking `share` (at least `min_l`
/// columns, leaving the right at least `min_r`). None for the right when the
/// window is narrower than `compact`: the caller stacks or groups instead.
pub fn columns(
    area: Rect,
    share: f32,
    min_l: u16,
    min_r: u16,
    compact: u16,
) -> (Rect, Option<Rect>) {
    if area.width < compact || area.width < min_l + min_r {
        return (area, None);
    }
    let left = ((area.width as f32 * share).round() as u16)
        .max(min_l)
        .min(area.width - min_r);
    let [l, r] = Layout::horizontal([Constraint::Length(left), Constraint::Min(0)]).areas(area);
    (l, Some(r))
}

/// The width inside a bordered pane.
pub fn inner_w(r: Rect) -> usize {
    r.width.saturating_sub(2) as usize
}

/// A bordered pane with a title on the left and, optionally, a note on the
/// right of its top edge. Returns the inside.
pub fn pane(f: &mut Frame, r: Rect, title: &str, focused: bool, right: &str) -> Rect {
    f.render_widget(Clear, r);
    let mut block = panel(title, focused);
    if !right.is_empty() && r.width as usize > text_width(title) + text_width(right) + 8 {
        block = block
            .title_top(Line::styled(format!(" {right} "), Style::new().fg(DIM)).right_aligned());
    }
    let inner = block.inner(r);
    f.render_widget(block, r);
    inner
}

/// `lines` drawn in `inner`, starting at the row that keeps `keep` (and a
/// few rows after it) in view.
pub fn show(f: &mut Frame, inner: Rect, lines: Vec<Line<'static>>, keep: Option<usize>) {
    let h = inner.height as usize;
    let top = keep.map_or(0, |k| {
        (k + 3).saturating_sub(h).min(lines.len().saturating_sub(h))
    });
    f.render_widget(
        Paragraph::new(lines.into_iter().skip(top).take(h).collect::<Vec<_>>()),
        inner,
    );
}

/// The newest `lines` that fit.
pub fn show_tail(f: &mut Frame, inner: Rect, lines: Vec<Line<'static>>) {
    let skip = lines.len().saturating_sub(inner.height as usize);
    f.render_widget(
        Paragraph::new(lines.into_iter().skip(skip).collect::<Vec<_>>()),
        inner,
    );
}

/// A section heading inside a pane.
pub fn heading(s: &str) -> Line<'static> {
    Line::styled(
        format!(" {}", s.to_uppercase()),
        Style::new().fg(DIM).add_modifier(Modifier::BOLD),
    )
}

/// A heading with a note at the right edge.
pub fn heading_note(s: &str, note: &str, width: usize) -> Line<'static> {
    let left = format!(" {}", s.to_uppercase());
    let gap = width.saturating_sub(text_width(&left) + text_width(note) + 1);
    Line::from(vec![
        Span::styled(left, Style::new().fg(DIM).add_modifier(Modifier::BOLD)),
        Span::raw(" ".repeat(gap)),
        Span::styled(note.to_string(), Style::new().fg(FAINT)),
    ])
}

/// `text` word-wrapped to `width`, each row indented by `indent` spaces.
pub fn wrapped(text: &str, indent: usize, width: usize, style: Style) -> Vec<Line<'static>> {
    let chars: Vec<Sc> = text.chars().map(|c| (c, style)).collect();
    wrap_styled(&chars, width.saturating_sub(indent + 1).max(8), 0)
        .into_iter()
        .map(|row| {
            let skip = row.iter().take_while(|(c, _)| *c == ' ').count();
            let row = if skip < row.len() {
                &row[skip..]
            } else {
                &row[..]
            };
            to_line(row, vec![Span::raw(" ".repeat(indent))])
        })
        .collect()
}

/// Help for the selected row, pinned to the bottom of a pane under a rule.
/// Returns the part of `inner` above it.
pub fn draw_help(f: &mut Frame, inner: Rect, text: &str) -> Rect {
    let mut lines = wrapped(
        &format!("ⓘ {text}"),
        1,
        inner.width as usize,
        Style::new().fg(DIM),
    );
    lines.truncate((inner.height as usize).saturating_sub(4).min(4));
    let h = lines.len() as u16;
    if h == 0 {
        return inner;
    }
    let [rest, rule, help] = Layout::vertical([
        Constraint::Min(0),
        Constraint::Length(1),
        Constraint::Length(h),
    ])
    .areas(inner);
    f.render_widget(
        Paragraph::new(Line::styled(
            "─".repeat(rule.width as usize),
            Style::new().fg(FAINT),
        )),
        rule,
    );
    f.render_widget(Paragraph::new(lines), help);
    rest
}

/// A label and a value on one line.
pub fn kv(label: &str, value: String, style: Style, label_w: usize) -> Line<'static> {
    Line::from(vec![
        Span::styled(format!(" {label:<label_w$}"), Style::new().fg(DIM)),
        Span::styled(value, style),
    ])
}

/// A horizontal bar `width` cells wide on a faint track, drawn in eighths
/// so short bars still show their size.
pub fn bar(share: f64, width: usize, color: Color) -> Vec<Span<'static>> {
    const EIGHTHS: [&str; 8] = ["", "▏", "▎", "▍", "▌", "▋", "▊", "▉"];
    let units = (share.clamp(0.0, 1.0) * width as f64 * 8.0).round() as usize;
    let (whole, rest) = (units / 8, units % 8);
    let mut out = vec![Span::styled(
        "█".repeat(whole),
        Style::new().fg(color).bg(TRACK),
    )];
    let mut used = whole;
    if rest > 0 && used < width {
        out.push(Span::styled(
            EIGHTHS[rest],
            Style::new().fg(color).bg(TRACK),
        ));
        used += 1;
    }
    out.push(Span::styled(
        " ".repeat(width.saturating_sub(used)),
        Style::new().bg(TRACK),
    ));
    out
}

/// A row of a selectable list: a marker, then the spans, on the selection
/// background when selected (dimmer when the pane is not focused).
pub fn list_row(on: bool, focused: bool, width: usize, spans: Vec<Span<'static>>) -> Line<'static> {
    let bg = match (on, focused) {
        (false, _) => Color::Reset,
        (true, true) => SELECTED,
        (true, false) => SELECTED_DIM,
    };
    let mut out = vec![Span::styled(
        if on { "▌" } else { " " },
        Style::new().fg(EMBER).bg(bg),
    )];
    let mut used = 1;
    for s in spans {
        used += text_width(&s.content);
        let style = if on { s.style.bg(bg) } else { s.style };
        out.push(Span::styled(s.content, style));
    }
    if on {
        out.push(Span::styled(
            " ".repeat(width.saturating_sub(used)),
            Style::new().bg(bg),
        ));
    }
    Line::from(out)
}

/// `s` cut or padded to exactly `w` columns.
pub fn fit_w(s: &str, w: usize) -> String {
    let c = clip(s, w);
    let pad = w.saturating_sub(text_width(&c));
    format!("{c}{}", " ".repeat(pad))
}

/// `s` right-aligned in `w` columns.
pub fn fit_r(s: &str, w: usize) -> String {
    let c = clip(s, w);
    let pad = w.saturating_sub(text_width(&c));
    format!("{}{c}", " ".repeat(pad))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn columns_split_or_hand_back_one() {
        let wide = Rect::new(0, 0, 200, 40);
        let (l, r) = columns(wide, 0.3, 30, 60, 100);
        assert_eq!(l.width, 60);
        assert_eq!(r.unwrap().width, 140);
        let narrow = Rect::new(0, 0, 90, 40);
        assert!(columns(narrow, 0.3, 30, 60, 100).1.is_none());
        // The left keeps its minimum, the right keeps its own.
        let (l, r) = columns(Rect::new(0, 0, 110, 40), 0.1, 30, 60, 100);
        assert_eq!((l.width, r.unwrap().width), (30, 80));
    }

    #[test]
    fn prefs_round_trip_and_clamp() {
        let mut p = Prefs::default();
        p.nudge(1, 3);
        p.nudge(4, 1);
        p.nudge(6, -50);
        let back = Prefs::load(&p.to_json());
        assert!((back.bench - 0.37).abs() < 0.011);
        assert_eq!(back.panel, Panel::Always);
        assert_eq!(back.prose, 60);
        assert_eq!(back.setups, p.setups);
        let wild = Prefs::load(&json!({"bench": 9.0, "compact": 3}));
        assert_eq!((wild.bench, wild.compact), (0.5, 60));
    }

    #[test]
    fn bars_fill_in_eighths() {
        let w = |spans: &[Span]| spans.iter().map(|s| text_width(&s.content)).sum::<usize>();
        for share in [0.0, 0.07, 0.5, 0.99, 1.0, 1.4] {
            assert_eq!(w(&bar(share, 13, GOOD)), 13, "share {share}");
        }
        assert_eq!(bar(0.5, 10, GOOD)[0].content, "█████");
    }
}
