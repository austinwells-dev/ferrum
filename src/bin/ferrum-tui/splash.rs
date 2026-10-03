//! Startup animation: the wordmark ignites left to right, embers rise, a bar fills.
use crate::*;

pub const SPLASH_SECS: f32 = 2.7;
const TAGLINE: &str = "local models · Apple Silicon · Metal";
const WORD_W: u16 = 35;

/// Top-left of the wordmark, a little above centre.
pub fn logo_origin(area: Rect) -> (u16, u16) {
    let x = area.x + area.width.saturating_sub(WORD_W) / 2;
    let y = area.y + (area.height.saturating_sub(5) / 2).saturating_sub(2);
    (x, y)
}

pub fn draw_splash(f: &mut Frame, t: f32) {
    let area = f.area();
    let (x0, y0) = logo_origin(area);
    let embers = Rect::new(
        x0.saturating_sub(8),
        y0.saturating_sub(6),
        (WORD_W + 16).min(area.width),
        12,
    );
    sparks(f, embers, y0 + 4, t, 54, None);

    let k = (t / 1.15).clamp(0.0, 1.0);
    let eased = 1.0 - (1.0 - k).powi(3);
    let reveal = eased * (WORD_W as f32 + 10.0);
    let sweep = (t > 1.4).then_some((t - 1.4) * 38.0 - 6.0);
    draw_wordmark(f, x0, y0, Some(reveal), sweep);

    for c in 0..(reveal as u16).min(WORD_W) {
        let fade = 0.35 + 0.65 * (1.0 - c as f32 / WORD_W as f32);
        let color = lerp((60, 30, 20), (255, 120, 50), fade * (k * 0.9 + 0.1));
        put(f, x0 + c, y0 + 6, "━", Style::new().fg(color));
    }

    let typed = ((t - 0.85) * 32.0).max(0.0) as usize;
    let len = TAGLINE.chars().count();
    if typed > 0 {
        let shown: String = TAGLINE.chars().take(typed.min(len)).collect();
        let tx = area.x + area.width.saturating_sub(len as u16) / 2;
        put(f, tx, y0 + 8, &shown, Style::new().fg(DIM));
        if typed < len {
            put(f, tx + typed as u16, y0 + 8, "▌", Style::new().fg(EMBER));
        }
    }

    if t > 1.2 {
        let p = ((t - 1.2) / 1.25).clamp(0.0, 1.0);
        let bar_w = 30u16;
        let bx = area.x + area.width.saturating_sub(bar_w) / 2;
        let filled = (p * bar_w as f32) as u16;
        for c in 0..bar_w {
            let (glyph, style) = if c < filled {
                ("━", Style::new().fg(ember(c as f32 / bar_w as f32)))
            } else if c == filled && p < 1.0 {
                ("╸", Style::new().fg(Color::White))
            } else {
                ("━", Style::new().fg(FAINT))
            };
            put(f, bx + c, y0 + 10, glyph, style);
        }
    }

    if t > 0.4 {
        let hint = "press any key to skip";
        let hx = area.x + area.width.saturating_sub(hint.len() as u16) / 2;
        put(
            f,
            hx,
            area.y + area.height.saturating_sub(2),
            hint,
            Style::new().fg(FAINT),
        );
    }
}
