//! Home screen: the wordmark and four big buttons.
use crate::*;

const BUTTONS: [(&str, &str, &str); 5] = [
    ("◆", "Chat", "talk to a model right here"),
    ("▣", "Agents", "a coding agent for your project"),
    ("◈", "Serve", "OpenAI & Anthropic compatible API"),
    ("⚙", "Settings", "folders, favorites, startup"),
    ("✕", "Quit", "see you soon"),
];

impl App {
    pub fn home_key(&mut self, key: KeyEvent) {
        match key.code {
            KeyCode::Up | KeyCode::Char('k') => {
                self.home_sel = (self.home_sel + BUTTONS.len() - 1) % BUTTONS.len()
            }
            KeyCode::Down | KeyCode::Char('j') | KeyCode::Tab => {
                self.home_sel = (self.home_sel + 1) % BUTTONS.len()
            }
            KeyCode::Char(c @ '1'..='5') => {
                self.home_sel = c as usize - '1' as usize;
                self.home_activate();
            }
            KeyCode::Char('c') => {
                self.home_sel = 0;
                self.home_activate();
            }
            KeyCode::Char('a') => {
                self.home_sel = 1;
                self.home_activate();
            }
            KeyCode::Char('s') => {
                self.home_sel = 2;
                self.home_activate();
            }
            KeyCode::Char('q') | KeyCode::Esc => self.quit = true,
            KeyCode::Enter | KeyCode::Char(' ') | KeyCode::Right => self.home_activate(),
            _ => {}
        }
    }

    fn home_activate(&mut self) {
        match self.home_sel {
            0 => self.open_pick(Mode::Chat),
            1 => self.open_pick(Mode::Agent),
            2 => self.open_pick(Mode::Serve),
            3 => {
                self.set_sel = 0;
                self.screen = Screen::Settings;
            }
            _ => self.quit = true,
        }
    }

    pub fn draw_home(&self, f: &mut Frame) {
        let area = f.area();
        let t = self.elapsed();
        let boxed = area.height >= 30;
        let (btn_h, gap): (u16, u16) = if boxed { (3, 0) } else { (1, 1) };
        let buttons_h = 5 * btn_h + 4 * gap;
        let total = 5 + 2 + 2 + buttons_h + 2;
        let top = area.y + area.height.saturating_sub(total + 1) / 2;
        let x0 = area.x + area.width.saturating_sub(35) / 2;

        let embers = Rect::new(
            x0.saturating_sub(10),
            top.saturating_sub(3),
            (35 + 20).min(area.width),
            9,
        );
        sparks(f, embers, top + 4, t, 22, Some(4.0));
        draw_wordmark(f, x0, top, None, Some((t * 11.0) % 80.0 - 20.0));

        let tag = "local models · Apple Silicon · Metal";
        let tx = area.x + area.width.saturating_sub(tag.chars().count() as u16) / 2;
        put(f, tx, top + 6, tag, Style::new().fg(DIM));

        let w = 62u16.min(area.width.saturating_sub(4));
        let bx = area.x + (area.width - w) / 2;
        let mut y = top + 9;
        for (i, (icon, title, desc)) in BUTTONS.iter().enumerate() {
            let on = i == self.home_sel;
            let badge = format!("{}", i + 1);
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
                    )),
                    inner,
                );
            } else {
                let rect = Rect::new(bx, y, w, 1);
                f.render_widget(
                    Paragraph::new(button_line(icon, title, desc, &badge, on, w as usize)),
                    rect,
                );
            }
            y += btn_h + gap;
        }

        let status = format!(
            "{} models ready · {} favorites · {} drafters",
            self.runnable(),
            self.favs.len(),
            self.drafters.len()
        );
        let sx = area.x + area.width.saturating_sub(status.chars().count() as u16) / 2;
        put(f, sx, y + 1, &status, Style::new().fg(FAINT));

        let footer = Rect::new(
            area.x,
            area.y + area.height.saturating_sub(1),
            area.width,
            1,
        );
        keys_footer(
            f,
            footer,
            &[
                ("↑↓", "move"),
                ("enter", "select"),
                ("1-5", "jump"),
                ("q", "quit"),
            ],
            &self.status,
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
        Span::styled(desc, Style::new().fg(if on { GOLD } else { DIM }).bg(bg)),
        Span::styled(tail, Style::new().fg(FAINT).bg(bg)),
    ])
}
