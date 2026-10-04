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
        if !code
            && c == '['
            && let Some((label, url, end)) = link_at(&chars, i)
        {
            {
                let st = Style::new().fg(BLUE).add_modifier(Modifier::UNDERLINED);
                out.extend(label.chars().map(|c| (c, st)));
                if url != label {
                    let shown = if url.chars().count() > 48 {
                        format!("{}…", url.chars().take(47).collect::<String>())
                    } else {
                        url
                    };
                    out.extend(styled(&format!(" ({shown})"), Style::new().fg(DIM)));
                }
                i = end;
                continue;
            }
        }
        if !code && (c == 'h') && is_url_start(&chars, i) {
            let mut end = i;
            while end < chars.len()
                && !chars[end].is_whitespace()
                && !matches!(chars[end], ')' | '>' | '<' | '"')
            {
                end += 1;
            }
            while end > i && matches!(chars[end - 1], '.' | ',' | ';' | ':' | '!' | '?') {
                end -= 1;
            }
            let st = Style::new().fg(BLUE).add_modifier(Modifier::UNDERLINED);
            out.extend(chars[i..end].iter().map(|&c| (c, st)));
            i = end;
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

fn is_url_start(chars: &[char], i: usize) -> bool {
    let rest: String = chars[i..chars.len().min(i + 8)].iter().collect();
    (rest.starts_with("http://") || rest.starts_with("https://"))
        && (i == 0 || !chars[i - 1].is_alphanumeric())
}

/// `[label](url)` starting at `i`: (label, url, index after it).
fn link_at(chars: &[char], i: usize) -> Option<(String, String, usize)> {
    let close = (i + 1..chars.len()).find(|&j| chars[j] == ']')?;
    if chars.get(close + 1) != Some(&'(') || close == i + 1 {
        return None;
    }
    let end = (close + 2..chars.len()).find(|&j| chars[j] == ')')?;
    let url: String = chars[close + 2..end].iter().collect();
    if url.contains(' ') || url.is_empty() {
        return None;
    }
    Some((chars[i + 1..close].iter().collect(), url, end + 1))
}

fn cells(row: &str) -> Vec<String> {
    let row = row.trim().trim_start_matches('|');
    let row = row.strip_suffix('|').unwrap_or(row);
    row.split('|').map(|c| c.trim().to_string()).collect()
}

fn is_separator(row: &str) -> bool {
    let t = row.trim();
    t.contains('-') && t.contains('|') && t.chars().all(|c| matches!(c, '|' | '-' | ':' | ' '))
}

/// A markdown table as aligned rows: a bold header, a rule, then the body.
/// Columns shrink to fit and long cells wrap.
fn table(rows: &[Vec<String>], width: usize, base: Style) -> Vec<Vec<Sc>> {
    let n = rows.iter().map(Vec::len).max().unwrap_or(0);
    if n == 0 {
        return Vec::new();
    }
    let head = Style::new().fg(GOLD).add_modifier(Modifier::BOLD);
    let styled_rows: Vec<Vec<Vec<Sc>>> = rows
        .iter()
        .enumerate()
        .map(|(r, row)| {
            (0..n)
                .map(|c| {
                    inline(
                        row.get(c).map_or("", String::as_str),
                        if r == 0 { head } else { base },
                    )
                })
                .collect()
        })
        .collect();
    let mut w: Vec<usize> = (0..n)
        .map(|c| {
            styled_rows
                .iter()
                .map(|r| r[c].iter().map(|&(ch, _)| cw(ch)).sum::<usize>())
                .max()
                .unwrap_or(1)
                .max(1)
        })
        .collect();
    let room = width.saturating_sub(3 * (n - 1)).max(n * 4);
    while w.iter().sum::<usize>() > room {
        let widest = (0..n).max_by_key(|&c| w[c]).unwrap_or(0);
        if w[widest] <= 4 {
            break;
        }
        w[widest] -= 1;
    }
    let bar = Style::new().fg(FAINT);
    let mut out = Vec::new();
    for (r, row) in styled_rows.iter().enumerate() {
        let wrapped: Vec<Vec<Vec<Sc>>> = row
            .iter()
            .enumerate()
            .map(|(c, cell)| wrap_styled(cell, w[c], 0))
            .collect();
        let height = wrapped.iter().map(Vec::len).max().unwrap_or(1);
        for k in 0..height {
            let mut line: Vec<Sc> = Vec::new();
            for (c, cell) in wrapped.iter().enumerate() {
                let part = cell.get(k).cloned().unwrap_or_default();
                let used: usize = part.iter().map(|&(ch, _)| cw(ch)).sum();
                line.extend(part);
                if c + 1 < n {
                    line.extend(std::iter::repeat_n((' ', base), w[c].saturating_sub(used)));
                    line.extend(styled(" │ ", bar));
                }
            }
            out.push(line);
        }
        if r == 0 {
            let rule: Vec<String> = w.iter().map(|&x| "─".repeat(x)).collect();
            out.push(styled(&rule.join("─┼─"), bar));
        }
    }
    out
}

/// Render markdown-ish text to styled display lines no wider than `width`.
pub fn markdown(text: &str, width: usize, base: Style) -> Vec<Vec<Sc>> {
    let width = width.max(8);
    let code = Style::new().fg(TEXT).bg(CODE_BG);
    let mut out: Vec<Vec<Sc>> = Vec::new();
    let mut in_code = false;
    let mut hl = highlight::Hl::default();
    let mut lines = text.split('\n').peekable();
    while let Some(raw) = lines.next() {
        let line = raw.trim_end_matches('\r');
        let trimmed = line.trim_start();
        if trimmed.starts_with("```") {
            if !in_code {
                in_code = true;
                let lang = trimmed.trim_start_matches('`').trim();
                hl = highlight::Hl::new(lang);
                let mut row = styled(&format!(" {lang}"), Style::new().fg(DIM).bg(CODE_BG));
                row.resize(width, (' ', code));
                out.push(row);
            } else {
                in_code = false;
            }
            continue;
        }
        if in_code {
            let body: Vec<Sc> = std::iter::once((' ', code))
                .chain(hl.line(&line.replace('\t', "    ")))
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
        if trimmed.starts_with('|') && lines.peek().is_some_and(|n| is_separator(n)) {
            let mut rows = vec![cells(trimmed)];
            lines.next();
            while let Some(next) = lines.peek() {
                if !next.trim_start().starts_with('|') {
                    break;
                }
                rows.push(cells(next.trim()));
                lines.next();
            }
            out.extend(table(&rows, width, base));
            continue;
        }
        if trimmed.len() >= 3
            && ["---", "***", "___"].iter().any(|r| {
                trimmed.starts_with(r)
                    && trimmed
                        .chars()
                        .all(|c| c == r.chars().next().unwrap_or('-'))
            })
        {
            out.push(styled(&"─".repeat(width), Style::new().fg(FAINT)));
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

#[cfg(test)]
mod tests {
    use super::*;

    fn plain(rows: &[Vec<Sc>]) -> Vec<String> {
        rows.iter()
            .map(|r| {
                r.iter()
                    .map(|&(c, _)| c)
                    .collect::<String>()
                    .trim_end()
                    .to_string()
            })
            .collect()
    }

    #[test]
    fn renders_a_table_and_wraps_to_width() {
        let md = "| Name | Notes |\n|---|---|\n| a | short |\n| b | a much longer note that has to wrap |";
        let rows = plain(&markdown(md, 24, Style::new()));
        assert_eq!(rows[0], "Name │ Notes");
        assert!(rows[1].contains('┼'), "{rows:?}");
        assert!(rows.len() > 4, "{rows:?}");
        assert!(rows.iter().all(|r| r.chars().count() <= 24), "{rows:?}");
    }

    #[test]
    fn links_and_bare_urls_are_styled() {
        let rows = markdown(
            "see [the docs](https://x.test/d) and https://y.test.",
            80,
            Style::new(),
        );
        let text = plain(&rows).join("\n");
        assert_eq!(text, "see the docs (https://x.test/d) and https://y.test.");
        let first = &rows[0];
        let docs = first.iter().find(|(c, _)| *c == 'd').unwrap();
        assert_eq!(docs.1.fg, Some(BLUE));
    }

    #[test]
    fn code_blocks_are_highlighted_and_rules_drawn() {
        let rows = markdown("```rust\nlet x = 1;\n```\n---", 20, Style::new());
        let code = &rows[1];
        assert_eq!(
            code.iter().find(|(c, _)| *c == 'l').unwrap().1.fg,
            Some(EMBER)
        );
        assert!(rows.last().unwrap().iter().all(|&(c, _)| c == '─'));
    }
}
