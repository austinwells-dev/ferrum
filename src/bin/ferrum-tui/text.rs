//! Word wrapping and a small markdown renderer for the chat transcript.
use crate::ui::cw;
use crate::*;

pub type Sc = (char, Style);

/// Greedy word wrap of styled characters. Continuation lines are indented by
/// `hang` columns. Always returns at least one line.
pub fn wrap_styled(chars: &[Sc], width: usize, hang: usize) -> Vec<Vec<Sc>> {
    let width = width.max(2);
    let hang = hang.min(width - 1);
    let indent = || -> Vec<Sc> { vec![(' ', Style::new()); hang] };
    let mut lines: Vec<Vec<Sc>> = Vec::new();
    let mut cur: Vec<Sc> = Vec::new();
    let mut cur_w = 0usize;
    let mut last_space: Option<usize> = None;
    for &(c, st) in chars {
        let w = cw(c);
        if cur_w + w > width {
            if c == ' ' {
                lines.push(std::mem::take(&mut cur));
                cur = indent();
                cur_w = hang;
                last_space = None;
                continue;
            }
            match last_space {
                Some(i) if i >= hang => {
                    let right = cur.split_off(i + 1);
                    cur.pop();
                    lines.push(std::mem::take(&mut cur));
                    cur = indent();
                    cur.extend(right);
                    cur_w = cur.iter().map(|&(c, _)| cw(c)).sum();
                }
                _ => {
                    lines.push(std::mem::take(&mut cur));
                    cur = indent();
                    cur_w = hang;
                }
            }
            last_space = None;
        }
        if c == ' ' {
            last_space = Some(cur.len());
        }
        cur.push((c, st));
        cur_w += w;
    }
    lines.push(cur);
    lines
}

/// Group runs of equal style into spans, after an optional prefix.
pub fn to_line(chars: &[Sc], mut spans: Vec<Span<'static>>) -> Line<'static> {
    let mut run = String::new();
    let mut style = Style::new();
    for &(c, st) in chars {
        if !run.is_empty() && st != style {
            spans.push(Span::styled(std::mem::take(&mut run), style));
        }
        style = st;
        run.push(c);
    }
    if !run.is_empty() {
        spans.push(Span::styled(run, style));
    }
    Line::from(spans)
}

fn styled(s: &str, style: Style) -> Vec<Sc> {
    s.chars().map(|c| (c, style)).collect()
}

/// `code` and **bold** inside a line of text.
fn inline(s: &str, base: Style) -> Vec<Sc> {
    let chars: Vec<char> = s.chars().collect();
    let (mut out, mut i) = (Vec::new(), 0);
    let (mut code, mut bold, mut italic) = (false, false, false);
    while i < chars.len() {
        let c = chars[i];
        if c == '`' {
            code = !code;
            i += 1;
            continue;
        }
        if !code && c == '*' && chars.get(i + 1) == Some(&'*') {
            bold = !bold;
            i += 2;
            continue;
        }
        if !code && c == '*' {
            let next_word = chars
                .get(i + 1)
                .is_some_and(|n| !n.is_whitespace() && *n != '*');
            let prev_word = i > 0 && !chars[i - 1].is_whitespace();
            if (!italic && next_word) || (italic && prev_word) {
                italic = !italic;
                i += 1;
                continue;
            }
        }
        let style = if code {
            Style::new().fg(GOLD).bg(CODE_BG)
        } else {
            let mut st = base;
            if bold {
                st = st.add_modifier(Modifier::BOLD);
            }
            if italic {
                st = st.add_modifier(Modifier::ITALIC);
            }
            st
        };
        out.push((c, style));
        i += 1;
    }
    out
}

/// Render markdown-ish text to styled display lines no wider than `width`.
pub fn markdown(text: &str, width: usize, base: Style) -> Vec<Vec<Sc>> {
    let width = width.max(8);
    let code = Style::new().fg(TEXT).bg(CODE_BG);
    let mut out: Vec<Vec<Sc>> = Vec::new();
    let mut in_code = false;
    for raw in text.split('\n') {
        let line = raw.trim_end_matches('\r');
        let trimmed = line.trim_start();
        if trimmed.starts_with("```") {
            if !in_code {
                in_code = true;
                let lang = trimmed.trim_start_matches('`').trim();
                let mut row = styled(&format!(" {lang}"), Style::new().fg(DIM).bg(CODE_BG));
                row.resize(width, (' ', code));
                out.push(row);
            } else {
                in_code = false;
            }
            continue;
        }
        if in_code {
            let body: Vec<Sc> = std::iter::once(' ')
                .chain(line.replace('\t', "    ").chars())
                .map(|c| (c, code))
                .collect();
            for chunk in body.chunks(width) {
                let mut row = chunk.to_vec();
                let used: usize = row.iter().map(|&(c, _)| cw(c)).sum();
                row.extend(std::iter::repeat_n((' ', code), width.saturating_sub(used)));
                out.push(row);
            }
            if body.is_empty() {
                out.push(vec![(' ', code); width]);
            }
            continue;
        }
        if let Some(rest) = trimmed.strip_prefix('#') {
            let text = rest.trim_start_matches('#').trim();
            let style = Style::new().fg(GOLD).add_modifier(Modifier::BOLD);
            out.extend(wrap_styled(&inline(text, style), width, 0));
            continue;
        }
        let lead = line.len() - trimmed.len();
        let (marker, body) = if let Some(r) = trimmed
            .strip_prefix("- ")
            .or_else(|| trimmed.strip_prefix("* "))
        {
            (Some("• ".to_string()), r)
        } else if let Some(r) = trimmed.strip_prefix("> ") {
            out.extend(wrap_styled(
                &[
                    styled("│ ", Style::new().fg(FAINT)),
                    inline(r, base.fg(DIM)),
                ]
                .concat(),
                width,
                2,
            ));
            continue;
        } else {
            let digits = trimmed.chars().take_while(|c| c.is_ascii_digit()).count();
            if digits > 0 && trimmed[digits..].starts_with(". ") {
                (
                    Some(trimmed[..digits + 2].to_string()),
                    &trimmed[digits + 2..],
                )
            } else {
                (None, trimmed)
            }
        };
        let mut chars = styled(&" ".repeat(lead.min(width / 2)), base);
        let mut hang = 0;
        if let Some(m) = marker {
            hang = lead.min(width / 2) + m.chars().count();
            chars.extend(styled(&m, Style::new().fg(EMBER)));
        }
        chars.extend(inline(body, base));
        out.extend(wrap_styled(&chars, width, hang));
    }
    out
}

/// Hard-wrap plain text to `width` columns, keeping explicit newlines.
/// Each row is (start, end) in chars of the original string.
pub fn wrap_rows(s: &str, width: usize) -> Vec<(usize, usize)> {
    let width = width.max(1);
    let mut rows = Vec::new();
    let (mut start, mut w, mut i) = (0usize, 0usize, 0usize);
    for c in s.chars() {
        if c == '\n' {
            rows.push((start, i));
            start = i + 1;
            w = 0;
        } else {
            let cwid = cw(c);
            if w + cwid > width {
                rows.push((start, i));
                start = i;
                w = 0;
            }
            w += cwid;
        }
        i += 1;
    }
    rows.push((start, i));
    rows
}
