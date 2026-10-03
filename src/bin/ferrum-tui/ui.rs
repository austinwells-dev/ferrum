//! Shared drawing helpers: panels, header, footer, popups and the big logo.
use crate::*;
use unicode_width::UnicodeWidthChar;

pub fn cw(c: char) -> usize {
    c.width().unwrap_or(0)
}

pub fn text_width(s: &str) -> usize {
    s.chars().map(cw).sum()
}

pub fn panel(title: &str, focused: bool) -> Block<'static> {
    let color = if focused { EMBER } else { FAINT };
    Block::bordered()
        .border_type(BorderType::Rounded)
        .border_style(Style::new().fg(color))
        .title(Line::from(vec![
            Span::raw(" "),
            Span::styled(
                title.to_string(),
                Style::new()
                    .fg(if focused { GOLD } else { DIM })
                    .add_modifier(Modifier::BOLD),
            ),
            Span::raw(" "),
        ]))
}

pub fn lerp(a: (u8, u8, u8), b: (u8, u8, u8), k: f32) -> Color {
    let k = k.clamp(0.0, 1.0);
    let m = |x: u8, y: u8| (x as f32 + (y as f32 - x as f32) * k) as u8;
    Color::Rgb(m(a.0, b.0), m(a.1, b.1), m(a.2, b.2))
}

/// The ember gradient used for the wordmark, `t` in 0..1 across the letters.
pub fn ember(t: f32) -> Color {
    lerp((255, 205, 115), (255, 105, 45), t)
}

pub fn hash(a: u32, b: u32) -> u32 {
    let mut h = a.wrapping_mul(0x9E37_79B1) ^ b.wrapping_mul(0x85EB_CA6B);
    h ^= h >> 15;
    h = h.wrapping_mul(0x2C1B_3C6D);
    h ^= h >> 12;
    h = h.wrapping_mul(0x297A_2D39);
    h ^ (h >> 15)
}

pub fn spinner(t: f32) -> char {
    const FRAMES: [char; 10] = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏'];
    FRAMES[(t * 12.0) as usize % FRAMES.len()]
}

/// Write text at a cell, ignoring anything outside the screen.
pub fn put(f: &mut Frame, x: u16, y: u16, s: &str, style: Style) {
    let a = f.area();
    if x >= a.x + a.width || y >= a.y + a.height {
        return;
    }
    f.buffer_mut().set_string(x, y, s, style);
}

pub fn centered(area: Rect, w: u16, h: u16) -> Rect {
    let w = w.min(area.width);
    let h = h.min(area.height);
    Rect::new(
        area.x + (area.width - w) / 2,
        area.y + (area.height - h) / 2,
        w,
        h,
    )
}

/// Top bar: wordmark, breadcrumbs on the left, anything on the right.
pub fn header(f: &mut Frame, area: Rect, crumbs: &[&str], right: Vec<Span<'static>>) {
    let block = Block::new()
        .borders(Borders::BOTTOM)
        .border_style(Style::new().fg(FAINT));
    let inner = block.inner(area);
    f.render_widget(block, area);
    let right_w: usize = right.iter().map(|s| s.width()).sum();
    let [left, rest] =
        Layout::horizontal([Constraint::Min(10), Constraint::Length(right_w as u16 + 1)])
            .areas(inner);
    let mut spans = vec![Span::styled(" ◆ ", Style::new().fg(EMBER))];
    let word = "FERRUM";
    for (i, c) in word.chars().enumerate() {
        spans.push(Span::styled(
            format!("{c} "),
            Style::new()
                .fg(ember(i as f32 / 5.0))
                .add_modifier(Modifier::BOLD),
        ));
    }
    for (i, c) in crumbs.iter().enumerate() {
        spans.push(Span::styled(" › ", Style::new().fg(FAINT)));
        let last = i + 1 == crumbs.len();
        spans.push(Span::styled(
            c.to_string(),
            if last {
                Style::new().fg(GOLD).add_modifier(Modifier::BOLD)
            } else {
                Style::new().fg(DIM)
            },
        ));
    }
    f.render_widget(Paragraph::new(Line::from(spans)), left);
    f.render_widget(
        Paragraph::new(Line::from(right)).alignment(Alignment::Right),
        rest,
    );
}

pub fn pill(label: &str, color: Color) -> Span<'static> {
    Span::styled(
        format!(" {label} "),
        Style::new()
            .fg(Color::Black)
            .bg(color)
            .add_modifier(Modifier::BOLD),
    )
}

pub fn keys_footer(
    f: &mut Frame,
    area: Rect,
    keys: &[(&str, &str)],
    status: &Option<(String, bool)>,
) {
    let mut spans: Vec<Span> = Vec::new();
    for (k, w) in keys {
        spans.push(Span::styled(format!(" {k}"), Style::new().fg(GOLD)));
        spans.push(Span::styled(format!(" {w} "), Style::new().fg(DIM)));
    }
    if let Some((msg, ok)) = status {
        spans.push(Span::styled(
            format!(" · {msg}"),
            Style::new().fg(if *ok { GOOD } else { BAD }),
        ));
    }
    f.render_widget(Paragraph::new(Line::from(spans)), area);
}

/// A one-line text prompt in the middle of the screen; the terminal cursor
/// sits at the end of the text.
pub fn input_popup(f: &mut Frame, title: &str, hint: &str, buffer: &str) {
    let area = f.area();
    let w = area.width.saturating_sub(8).min(72);
    let rect = centered(area, w, 5);
    f.render_widget(Clear, rect);
    let block = panel(title, true);
    let inner = block.inner(rect);
    f.render_widget(block, rect);
    f.render_widget(
        Paragraph::new(vec![
            Line::from(vec![
                Span::styled(" ❯ ", Style::new().fg(EMBER)),
                Span::styled(buffer.to_string(), Style::new().fg(Color::White)),
            ]),
            Line::styled(format!(" {hint}"), Style::new().fg(DIM)),
        ]),
        inner,
    );
    let x = (inner.x + 3 + text_width(buffer) as u16).min(inner.x + inner.width.saturating_sub(1));
    f.set_cursor_position((x, inner.y));
}

fn glyph(c: char) -> [&'static str; 5] {
    match c {
        'F' => ["█████", "█    ", "████ ", "█    ", "█    "],
        'E' => ["█████", "█    ", "████ ", "█    ", "█████"],
        'R' => ["████ ", "█   █", "████ ", "█  █ ", "█   █"],
        'U' => ["█   █", "█   █", "█   █", "█   █", " ███ "],
        'M' => ["█   █", "██ ██", "█ █ █", "█   █", "█   █"],
        _ => ["     "; 5],
    }
}

/// Block-letter rows for `word`, one blank column between letters.
pub fn big_word(word: &str) -> Vec<Vec<char>> {
    (0..5)
        .map(|row| {
            let mut line = Vec::new();
            for (i, c) in word.chars().enumerate() {
                if i > 0 {
                    line.push(' ');
                }
                line.extend(glyph(c)[row].chars());
            }
            line
        })
        .collect()
}

/// Rising embers over `area`, `t` seconds in; `fade_in` seconds to ramp up.
pub fn sparks(f: &mut Frame, area: Rect, base_y: u16, t: f32, count: u32, wrap: Option<f32>) {
    const CHARS: [&str; 5] = ["·", "∙", "•", "˙", "✦"];
    for i in 0..count {
        let h = hash(i, 7);
        let born = (h % 1000) as f32 / 1000.0 * wrap.unwrap_or(1.8);
        let life = 1.1 + (hash(i, 11) % 100) as f32 / 120.0;
        let mut age = t - born;
        if let Some(period) = wrap {
            age = age.rem_euclid(period.max(0.1));
        }
        if age < 0.0 || age > life {
            continue;
        }
        let x = area.x + (hash(i, 3) % area.width.max(1) as u32) as u16;
        let rise = age * (4.0 + (hash(i, 5) % 50) as f32 / 12.0);
        let y = base_y as f32 - rise;
        if y < area.y as f32 || y >= (area.y + area.height) as f32 {
            continue;
        }
        let fade = 1.0 - age / life;
        let color = lerp((90, 40, 22), (255, 170, 80), fade);
        put(
            f,
            x,
            y as u16,
            CHARS[(hash(i, 9) % CHARS.len() as u32) as usize],
            Style::new().fg(color),
        );
    }
}

/// The wordmark with an optional left-to-right reveal (`reveal` in columns)
/// and a bright sweep (`sweep` is the column of its centre).
pub fn draw_wordmark(f: &mut Frame, x0: u16, y0: u16, reveal: Option<f32>, sweep: Option<f32>) {
    let word = big_word("FERRUM");
    let w = word[0].len() as f32;
    for (r, row) in word.iter().enumerate() {
        for (c, &ch) in row.iter().enumerate() {
            if ch == ' ' {
                continue;
            }
            let cf = c as f32;
            let mut heat = 0.0;
            if let Some(front) = reveal {
                if cf > front {
                    continue;
                }
                heat = (1.0 - (front - cf) / 9.0).clamp(0.0, 1.0);
            }
            if let Some(center) = sweep {
                heat = heat.max((1.0 - (cf - center).abs() / 4.0).clamp(0.0, 1.0) * 0.8);
            }
            let base = ember(cf / w);
            let color = match base {
                Color::Rgb(r, g, b) => lerp((r, g, b), (255, 250, 235), heat),
                other => other,
            };
            put(f, x0 + c as u16, y0 + r as u16, "█", Style::new().fg(color));
        }
    }
}
