//! Home: the wordmark and one button per tab.
use crate::*;

/// (tab, icon, name, what it is for).
const BUTTONS: [(Tab, &str, &str, &str); 7] = [
    (Tab::Chat, "◆", "Chat", "talk to a model right here"),
    (
        Tab::Agents,
        "▣",
        "Agents",
        "a coding agent for your project",
    ),
    (
        Tab::Serve,
        "◈",
        "Serve",
        "OpenAI & Anthropic compatible API",
    ),
    (Tab::Image, "◐", "Image", "text to image with Qwen-Image"),
    (
        Tab::Bench,
        "◷",
        "Bench",
        "speed, intelligence and agentic coding",
    ),
    (
        Tab::Models,
        "▤",
        "Models",
        "every model and drafter on this Mac",
    ),
    (Tab::Settings, "⚙", "Settings", "folders, layout, favorites"),
];

/// The buttons, Quit last.
const ROWS: usize = BUTTONS.len() + 1;

impl App {
    pub fn home_key(&mut self, key: KeyEvent) {
        match key.code {
            KeyCode::Up | KeyCode::Char('k') => self.home_sel = (self.home_sel + ROWS - 1) % ROWS,
            KeyCode::Down | KeyCode::Char('j') | KeyCode::Tab => {
                self.home_sel = (self.home_sel + 1) % ROWS
            }
            KeyCode::Char('c') => self.go_tab(Tab::Chat),
            KeyCode::Char('a') => self.go_tab(Tab::Agents),
            KeyCode::Char('s') => self.go_tab(Tab::Serve),
            KeyCode::Char('i') => self.go_tab(Tab::Image),
            KeyCode::Char('b') => self.go_tab(Tab::Bench),
            KeyCode::Char('m') => self.go_tab(Tab::Models),
            KeyCode::Char('q') | KeyCode::Esc => self.quit = true,
            KeyCode::Enter | KeyCode::Char(' ') | KeyCode::Right => {
                match BUTTONS.get(self.home_sel) {
                    Some(b) => self.go_tab(b.0),
                    None => self.quit = true,
                }
            }
            _ => {}
        }
    }

    /// A word on what a tab has running, for its button.
    fn tab_note(&self, tab: Tab) -> Option<(String, Color)> {
        let session = self.chat.as_ref().map(|c| c.agent.is_some());
        match tab {
            Tab::Chat if session == Some(false) => Some(("● open".into(), GOOD)),
            Tab::Agents if session == Some(true) => Some(("● open".into(), GOOD)),
            Tab::Serve => self
                .serve
                .as_ref()
                .filter(|s| s.exit.is_none())
                .map(|s| (format!("● {}", s.port_label()), GOOD)),
            Tab::Image => self.image.progress().map(|p| (format!("● {p}"), EMBER)),
            Tab::Bench => self
                .bench
                .as_ref()
                .filter(|b| b.phase == benchui::BenchPhase::Running)
                .map(|_| ("● running".into(), EMBER)),
            _ => None,
        }
    }

    pub fn draw_home(&self, f: &mut Frame, area: Rect, footer: Rect) {
        let t = self.elapsed();
        let boxed = area.height >= 34;
        let (btn_h, gap): (u16, u16) = if boxed { (3, 0) } else { (1, 1) };
        let buttons_h = ROWS as u16 * btn_h + (ROWS as u16 - 1) * gap;
        let total = 5 + 2 + 2 + buttons_h;
        let top = area.y + area.height.saturating_sub(total) / 2;
        let x0 = area.x + area.width.saturating_sub(35) / 2;

        let embers = Rect::new(
            x0.saturating_sub(10),
            top.saturating_sub(3).max(area.y),
            (35 + 20).min(area.width),
            9,
        );
        sparks(f, embers, top + 4, t, 22, Some(4.0));
        draw_wordmark(f, x0, top, None, Some((t * 11.0) % 80.0 - 20.0));

        let tag = "local models · Apple Silicon · Metal";
        let tx = area.x + area.width.saturating_sub(tag.chars().count() as u16) / 2;
        put(f, tx, top + 6, tag, Style::new().fg(DIM));

        let w = 64u16.min(area.width.saturating_sub(4));
        let bx = area.x + (area.width - w) / 2;
        let mut y = top + 9;
        for i in 0..ROWS {
            let (icon, title, desc, badge, note) = match BUTTONS.get(i) {
                Some(&(tab, icon, title, desc)) => {
                    let n = shell::TABS.iter().position(|(t, _)| *t == tab).unwrap_or(0) + 1;
                    (icon, title, desc, n.to_string(), self.tab_note(tab))
                }
                None => ("✕", "Quit", "see you soon", "q".to_string(), None),
            };
            let on = i == self.home_sel;
            let desc = match &note {
                Some((n, _)) => n.as_str(),
                None => desc,
            };
            let desc_color = note.as_ref().map(|(_, c)| *c);
            if y + btn_h > area.y + area.height {
                break;
            }
            if boxed {
                let rect = Rect::new(bx, y, w, 3);
                let block = Block::bordered()
                    .border_type(BorderType::Rounded)
                    .border_style(Style::new().fg(if on { EMBER } else { FAINT }))
                    .style(Style::new().bg(if on { SELECTED } else { Color::Reset }));
                let inner = block.inner(rect);
                f.render_widget(block, rect);
                f.render_widget(
                    Paragraph::new(button_line(
                        icon,
                        title,
                        desc,
                        &badge,
                        on,
                        inner.width as usize,
                        desc_color,
                    )),
                    inner,
                );
            } else {
                f.render_widget(
                    Paragraph::new(button_line(
                        icon, title, desc, &badge, on, w as usize, desc_color,
                    )),
                    Rect::new(bx, y, w, 1),
                );
            }
            y += btn_h + gap;
        }
        self.footer(
            f,
            footer,
            &[
                ("↑↓", "move"),
                ("enter", "open"),
                ("2-7", "jump"),
                ("q", "quit"),
            ],
        );
    }
}

fn button_line(
    icon: &str,
    title: &str,
    desc: &str,
    badge: &str,
    on: bool,
    width: usize,
    desc_color: Option<Color>,
) -> Line<'static> {
    let bg = if on { SELECTED } else { Color::Reset };
    let left = format!(" {} {} {title}", if on { "▸" } else { " " }, icon);
    let tail = format!("  {badge} ");
    // Whatever room the title leaves (keeping a gap of two) is the description's.
    let room = width.saturating_sub(left.chars().count() + tail.chars().count() + 2);
    let desc = clip(desc, room);
    let pad =
        width.saturating_sub(left.chars().count() + desc.chars().count() + tail.chars().count());
    Line::from(vec![
        Span::styled(
            left,
            Style::new()
                .fg(if on { Color::White } else { TEXT })
                .bg(bg)
                .add_modifier(if on {
                    Modifier::BOLD
                } else {
                    Modifier::empty()
                }),
        ),
        Span::styled(" ".repeat(pad), Style::new().bg(bg)),
        Span::styled(
            desc,
            Style::new()
                .fg(desc_color.unwrap_or(if on { GOLD } else { DIM }))
                .bg(bg),
        ),
        Span::styled(tail, Style::new().fg(FAINT).bg(bg)),
    ])
}
