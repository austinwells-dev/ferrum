//! The coding agent's sidebar: project, plan, files, processes and context.
use crate::coding::{TodoState, Touch};
use crate::*;

impl App {
    pub fn draw_agent_side(&self, f: &mut Frame, area: Rect, chat: &Chat) {
        let Some(agent) = chat.agent.as_ref() else {
            return;
        };
        let block = panel("Agent", false);
        let inner = block.inner(area);
        f.render_widget(block, area);
        let w = inner.width as usize;
        let head = |s: &str| {
            Line::styled(
                format!(" {s}"),
                Style::new().fg(DIM).add_modifier(Modifier::BOLD),
            )
        };
        let dim = Style::new().fg(DIM);
        let mut lines: Vec<Line> = Vec::new();

        let name = agent
            .project
            .file_name()
            .map(|n| n.to_string_lossy().to_string())
            .unwrap_or_default();
        lines.push(Line::styled(
            format!(" {}", clip(&name, w.saturating_sub(2))),
            Style::new().fg(Color::White).add_modifier(Modifier::BOLD),
        ));
        lines.push(Line::styled(
            format!(" {}", clip(&tilde(&agent.project), w.saturating_sub(2))),
            Style::new().fg(FAINT),
        ));
        if let Some(branch) = &agent.branch {
            lines.push(Line::from(vec![
                Span::styled(" ⎇ ", Style::new().fg(EMBER)),
                Span::styled(clip(branch, w.saturating_sub(14)), Style::new().fg(TEXT)),
                Span::styled(
                    if agent.dirty > 0 {
                        format!("  {} changed", agent.dirty)
                    } else {
                        "  clean".to_string()
                    },
                    Style::new().fg(if agent.dirty > 0 { GOLD } else { FAINT }),
                ),
            ]));
        }
        lines.push(Line::from(vec![
            Span::styled(format!(" ponytail {}", agent.ponytail.label()), dim),
            Span::styled(format!(" · {}", chat.tools.label()), dim),
            Span::styled(
                if agent.plan { " · PLAN" } else { "" },
                Style::new().fg(GOLD).add_modifier(Modifier::BOLD),
            ),
        ]));
        lines.push(Line::default());

        let shared = agent.shared.try_lock();
        if let Ok(g) = &shared {
            lines.push(head(&if g.todos.is_empty() {
                "PLAN".to_string()
            } else {
                let (done, total) = g.todo_counts();
                format!("PLAN  {done}/{total}")
            }));
            if g.todos.is_empty() {
                lines.push(Line::styled(" steps appear here", Style::new().fg(FAINT)));
            }
            for t in g.todos.iter().take(10) {
                let (mark, style) = match t.state {
                    TodoState::Done => ("✓", Style::new().fg(FAINT)),
                    TodoState::Active => ("▸", Style::new().fg(EMBER).add_modifier(Modifier::BOLD)),
                    TodoState::Pending => ("○", Style::new().fg(TEXT)),
                };
                lines.push(Line::from(vec![
                    Span::styled(format!(" {mark} "), style),
                    Span::styled(clip(&t.text, w.saturating_sub(4)), style),
                ]));
            }
            lines.push(Line::default());

            if !g.touched.is_empty() {
                lines.push(head("FILES"));
                let mut files: Vec<_> = g.touched.iter().collect();
                files.sort_by_key(|(_, t)| **t != Touch::Edited);
                for (path, how) in files.into_iter().take(8) {
                    let (mark, color) = match how {
                        Touch::Edited => ("✎", GOLD),
                        Touch::Read => ("◔", DIM),
                    };
                    lines.push(Line::from(vec![
                        Span::styled(format!(" {mark} "), Style::new().fg(color)),
                        Span::styled(clip(path, w.saturating_sub(4)), Style::new().fg(TEXT)),
                    ]));
                }
                lines.push(Line::default());
            }

            if !g.procs.is_empty() {
                lines.push(head("PROCESSES"));
                for p in &g.procs {
                    lines.push(Line::from(vec![
                        Span::styled(" ● ", Style::new().fg(GOOD)),
                        Span::styled(clip(&p.name, w.saturating_sub(4)), Style::new().fg(TEXT)),
                    ]));
                }
                lines.push(Line::default());
            }
        }

        lines.push(head("CONTEXT"));
        match chat.ctx_use() {
            Some((used, cap)) => {
                let ratio = (used as f64 / cap.max(1) as f64).clamp(0.0, 1.0);
                let bar_w = w.saturating_sub(10).clamp(6, 24);
                let filled = (ratio * bar_w as f64).round() as usize;
                let color = if ratio > 0.8 {
                    BAD
                } else if ratio > 0.55 {
                    GOLD
                } else {
                    GOOD
                };
                lines.push(Line::from(vec![
                    Span::styled(" ", dim),
                    Span::styled("█".repeat(filled), Style::new().fg(color)),
                    Span::styled("░".repeat(bar_w - filled), Style::new().fg(FAINT)),
                    Span::styled(format!(" {:.0}%", ratio * 100.0), Style::new().fg(color)),
                ]));
                lines.push(Line::styled(format!(" {used} / {cap} tokens"), dim));
            }
            None => lines.push(Line::styled(
                " shown after each reply",
                Style::new().fg(FAINT),
            )),
        }
        if let Ok(g) = &shared {
            if g.store.saved > 0 || g.evicted > 0 || g.compactions > 0 {
                lines.push(Line::styled(
                    format!(" kept out: {} KB", (g.store.saved) / 1024),
                    Style::new().fg(GOOD),
                ));
                lines.push(Line::styled(
                    format!(
                        " {} stored · {} elided · {}×compact",
                        g.store.len(),
                        g.evicted,
                        g.compactions
                    ),
                    Style::new().fg(FAINT),
                ));
            }
        }
        let lines: Vec<Line> = lines.into_iter().map(|l| fit(l, w)).collect();
        f.render_widget(Paragraph::new(lines), inner);
    }
}

/// Cut a line to `width` columns, ending in an ellipsis, so nothing is clipped mid-word by the border.
fn fit(line: Line<'static>, width: usize) -> Line<'static> {
    let total: usize = line.spans.iter().map(|s| text_width(&s.content)).sum();
    if total <= width {
        return line;
    }
    let mut left = width.saturating_sub(1);
    let mut spans = Vec::new();
    for span in line.spans {
        if left == 0 {
            break;
        }
        let mut kept = String::new();
        for c in span.content.chars() {
            let cw = cw(c);
            if cw > left {
                left = 0;
                break;
            }
            kept.push(c);
            left -= cw;
        }
        spans.push(Span::styled(kept, span.style));
    }
    spans.push(Span::styled("…", Style::new().fg(FAINT)));
    Line::from(spans)
}
