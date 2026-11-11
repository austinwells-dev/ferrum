//! The panels beside a conversation: the coding agent's (project, plan,
//! files, processes, context) and a chat's (model, settings, last reply).
use crate::chat::Role;
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
        lines.extend(context_lines(chat, w));
        if let Ok(g) = &shared
            && (g.store.saved > 0 || g.evicted > 0 || g.compactions > 0)
        {
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
        let lines: Vec<Line> = lines.into_iter().map(|l| fit(l, w)).collect();
        f.render_widget(Paragraph::new(lines), inner);
    }

    /// A chat's panel: what is loaded, how it is set, how the last reply went.
    pub fn draw_chat_side(&self, f: &mut Frame, area: Rect, chat: &Chat) {
        let block = panel("Session", false);
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
        let text = Style::new().fg(TEXT);
        let row = |k: &str, v: String, style: Style| {
            Line::from(vec![
                Span::styled(format!(" {k:<10}"), dim),
                Span::styled(v, style),
            ])
        };
        let mut lines: Vec<Line> = Vec::new();
        lines.push(head("MODEL"));
        match &chat.info {
            Some(i) => {
                lines.push(Line::styled(
                    format!(" {}", i.name),
                    Style::new().fg(Color::White).add_modifier(Modifier::BOLD),
                ));
                lines.push(row(
                    "weights",
                    format!("{:.1} GiB · loaded in {:.1}s", i.weights_gib, i.load_secs),
                    text,
                ));
                lines.push(row("sampling", i.sampling.clone(), text));
            }
            None => lines.push(Line::styled(format!(" {}", chat.label), text)),
        }
        lines.push(Line::default());
        lines.push(head("SETTINGS"));
        let on_off = |b: bool| if b { "on" } else { "off" };
        lines.push(row(
            "thinking",
            format!(
                "{}{}",
                on_off(chat.think),
                if chat.think && !chat.show_thinking {
                    " · hidden"
                } else {
                    ""
                }
            ),
            text,
        ));
        if let Some(i) = &chat.info {
            lines.push(match &i.drafter {
                Some(d) => row("drafter", format!("⚡ {d}"), Style::new().fg(GOLD)),
                None => row("drafter", "off".into(), Style::new().fg(FAINT)),
            });
            lines.push(match (&i.vision, &i.audio) {
                (Some(_), _) => row("vision", "👁 images supported".into(), text),
                (None, Some(_)) => row("audio", "🔊 WAV clips supported".into(), text),
                (None, None) => row("vision", "text only".into(), Style::new().fg(FAINT)),
            });
        }
        lines.push(row("tools", chat.tools.label().into(), text));
        lines.push(Line::default());

        let stats = chat.turns.iter().rev().find_map(|t| t.stats.as_deref());
        lines.push(head("LAST REPLY"));
        match stats {
            Some(st) => {
                let mut parts = st.split(" · ");
                if let Some(speed) = parts.next() {
                    lines.push(Line::styled(
                        format!(" {speed}"),
                        Style::new().fg(GOOD).add_modifier(Modifier::BOLD),
                    ));
                }
                for p in parts {
                    lines.push(Line::styled(format!(" {p}"), text));
                }
            }
            None => lines.push(Line::styled(
                " speed and tokens appear here",
                Style::new().fg(FAINT),
            )),
        }
        lines.push(Line::default());
        let asked = chat.turns.iter().filter(|t| t.role == Role::User).count();
        lines.push(head(&format!(
            "CONTEXT  {asked} message{}",
            if asked == 1 { "" } else { "s" }
        )));
        lines.extend(context_lines(chat, w));
        let lines: Vec<Line> = lines.into_iter().map(|l| fit(l, w)).collect();
        f.render_widget(Paragraph::new(lines), inner);
    }
}

/// How full the context window is: a bar and the token counts.
fn context_lines(chat: &Chat, w: usize) -> Vec<Line<'static>> {
    let dim = Style::new().fg(DIM);
    let mut lines = Vec::new();
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
    lines
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
