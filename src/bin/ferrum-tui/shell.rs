//! The frame every screen lives in: the tab bar with what is running on the
//! right, the key bar along the bottom, the key list (`?`) and the command
//! palette (ctrl-p). Switching tabs leaves sessions, the server and runs
//! going; only an explicit stop or unload ends them.
use crate::benchui::BenchPhase;
use crate::layout::pane;
use crate::*;

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Tab {
    Home,
    Chat,
    Agents,
    Serve,
    Bench,
    Models,
    Image,
    Settings,
}

pub const TABS: [(Tab, &str); 8] = [
    (Tab::Home, "Home"),
    (Tab::Chat, "Chat"),
    (Tab::Agents, "Agents"),
    (Tab::Serve, "Serve"),
    (Tab::Bench, "Bench"),
    (Tab::Models, "Models"),
    (Tab::Image, "Image"),
    (Tab::Settings, "Settings"),
];

/// What floats over the screen.
#[derive(Clone, PartialEq)]
pub enum Overlay {
    None,
    /// The keys of the screen underneath.
    Keys,
    Palette {
        query: String,
        sel: usize,
    },
}

/// Commands the palette offers.
#[derive(Clone, Copy, PartialEq)]
enum Cmd {
    Go(Tab),
    NewImage,
    RunSpeed,
    RunIq,
    RunAgent,
    NewChat,
    NewServe,
    StopServer,
    Unload,
    TogglePanel,
    Layout,
    Rescan,
    Quit,
}

const COMMANDS: [(Cmd, &str); 20] = [
    (Cmd::Go(Tab::Home), "Go to Home"),
    (Cmd::Go(Tab::Chat), "Go to Chat"),
    (Cmd::Go(Tab::Agents), "Go to Agents"),
    (Cmd::Go(Tab::Serve), "Go to Serve"),
    (Cmd::Go(Tab::Bench), "Go to Bench"),
    (Cmd::Go(Tab::Models), "Go to Models"),
    (Cmd::Go(Tab::Image), "Go to Image"),
    (Cmd::NewImage, "Generate an image"),
    (Cmd::Go(Tab::Settings), "Go to Settings"),
    (Cmd::RunSpeed, "Bench: measure speed"),
    (Cmd::RunIq, "Bench: measure intelligence"),
    (Cmd::RunAgent, "Bench: measure agentic coding"),
    (Cmd::NewChat, "New chat setup"),
    (Cmd::NewServe, "New server setup"),
    (Cmd::StopServer, "Stop the server"),
    (Cmd::Unload, "Unload the chat model"),
    (Cmd::TogglePanel, "Show or hide the chat side panel"),
    (Cmd::Layout, "Settings › Layout"),
    (Cmd::Rescan, "Rescan model folders"),
    (Cmd::Quit, "Quit ferrum"),
];

impl App {
    /// The tab the current screen belongs to.
    pub fn tab(&self) -> Tab {
        match self.screen {
            Screen::Splash | Screen::Home => Tab::Home,
            Screen::Pick | Screen::Editor => match self.mode {
                Mode::Chat => Tab::Chat,
                Mode::Agent => Tab::Agents,
                Mode::Serve => Tab::Serve,
            },
            Screen::Chat => match self.chat.as_ref().is_some_and(|c| c.agent.is_some()) {
                true => Tab::Agents,
                false => Tab::Chat,
            },
            Screen::Serve => Tab::Serve,
            Screen::Bench => Tab::Bench,
            Screen::Models => Tab::Models,
            Screen::Image => Tab::Image,
            Screen::Settings => Tab::Settings,
        }
    }

    /// Show a tab: its running session when there is one, otherwise its
    /// list of setups. Nothing running is stopped.
    pub fn go_tab(&mut self, tab: Tab) {
        self.overlay = Overlay::None;
        if tab == self.tab() && self.screen != Screen::Editor {
            return;
        }
        let session = self.chat.as_ref().map(|c| c.agent.is_some());
        match tab {
            Tab::Home => self.screen = Screen::Home,
            Tab::Chat if session == Some(false) => self.screen = Screen::Chat,
            Tab::Agents if session == Some(true) => self.screen = Screen::Chat,
            Tab::Chat => self.open_pick(Mode::Chat),
            Tab::Agents => self.open_pick(Mode::Agent),
            Tab::Serve if self.serve.is_some() => self.screen = Screen::Serve,
            Tab::Serve => self.open_pick(Mode::Serve),
            Tab::Bench if self.bench.is_some() => self.screen = Screen::Bench,
            Tab::Bench => self.open_bench(),
            Tab::Models => self.screen = Screen::Models,
            Tab::Image => self.screen = Screen::Image,
            Tab::Settings => self.screen = Screen::Settings,
        }
    }

    /// Typing goes to a text box: digits, `?` and brackets are text there.
    pub fn typing(&self) -> bool {
        !matches!(self.editing, Editing::No)
            || (self.screen == Screen::Chat && self.chat_box && self.chat.is_some())
    }

    /// Keys that work on every screen. True when the key was used.
    pub fn shell_key(&mut self, key: KeyEvent) -> bool {
        let ctrl = key.modifiers.contains(KeyModifiers::CONTROL);
        if let Overlay::Palette { query, sel } = &mut self.overlay {
            let hits = palette_hits(query);
            match key.code {
                KeyCode::Esc => self.overlay = Overlay::None,
                KeyCode::Up => *sel = sel.saturating_sub(1),
                KeyCode::Down | KeyCode::Tab => *sel = (*sel + 1).min(hits.len().saturating_sub(1)),
                KeyCode::Backspace => {
                    query.pop();
                    *sel = 0;
                }
                KeyCode::Enter => {
                    let cmd = hits.get(*sel).map(|c| c.0);
                    self.overlay = Overlay::None;
                    if let Some(cmd) = cmd {
                        self.run_cmd(cmd);
                    }
                }
                KeyCode::Char(c) if !ctrl => {
                    query.push(c);
                    *sel = 0;
                }
                _ => {}
            }
            return true;
        }
        if self.overlay == Overlay::Keys {
            self.overlay = Overlay::None;
            return true;
        }
        if ctrl && key.code == KeyCode::Char('p') {
            self.overlay = Overlay::Palette {
                query: String::new(),
                sel: 0,
            };
            return true;
        }
        if ctrl && key.code == KeyCode::Char('b') {
            self.toggle_panel();
            return true;
        }
        if self.typing() || self.screen == Screen::Splash {
            return false;
        }
        match key.code {
            KeyCode::Char(c @ '1'..='8') if !ctrl => {
                self.go_tab(TABS[c as usize - '1' as usize].0);
                true
            }
            KeyCode::Char('?') => {
                self.overlay = Overlay::Keys;
                true
            }
            KeyCode::Char(c @ ('[' | ']')) => {
                self.resize_split(if c == ']' { 1 } else { -1 });
                true
            }
            _ => false,
        }
    }

    fn run_cmd(&mut self, cmd: Cmd) {
        match cmd {
            Cmd::Go(t) => self.go_tab(t),
            Cmd::NewImage => {
                self.go_tab(Tab::Image);
                self.status = Some(("p writes the prompt, g generates".into(), true));
            }
            Cmd::RunSpeed | Cmd::RunIq | Cmd::RunAgent => {
                self.go_tab(Tab::Bench);
                if let Some(b) = self.bench.as_mut()
                    && b.phase != BenchPhase::Running
                {
                    b.measure = match cmd {
                        Cmd::RunSpeed => evalui::Measure::Speed,
                        Cmd::RunIq => evalui::Measure::Intelligence,
                        _ => evalui::Measure::Agent,
                    };
                    b.phase = BenchPhase::Setup;
                    b.iq.compare = None;
                    b.ag.compare = None;
                }
            }
            Cmd::NewChat => {
                self.open_pick(Mode::Chat);
                self.open_editor(Mode::Chat, Origin::Custom);
            }
            Cmd::NewServe => {
                self.open_pick(Mode::Serve);
                self.open_editor(Mode::Serve, Origin::Custom);
            }
            Cmd::StopServer => match self.serve.take() {
                Some(mut s) => {
                    s.stop();
                    self.status = Some(("server stopped".into(), true));
                    if self.screen == Screen::Serve {
                        self.open_pick(Mode::Serve);
                    }
                }
                None => self.status = Some(("no server is running".into(), false)),
            },
            Cmd::Unload => match self.chat.is_some() {
                true => {
                    self.unload_chat();
                    self.status = Some(("model unloaded".into(), true));
                }
                false => self.status = Some(("no model is loaded".into(), false)),
            },
            Cmd::TogglePanel => self.toggle_panel(),
            Cmd::Layout => {
                self.screen = Screen::Settings;
                self.set_sel = self.layout_row();
            }
            Cmd::Rescan => self.rescan(None),
            Cmd::Quit => self.quit = true,
        }
    }

    /// Stop the chat or agent session and free its model.
    pub fn unload_chat(&mut self) {
        if let Some(chat) = self.chat.take() {
            chat.persist();
            if let Some(h) = chat.close() {
                self.closing.push(h);
            }
        }
        if self.screen == Screen::Chat {
            self.screen = Screen::Pick;
        }
    }

    pub fn toggle_panel(&mut self) {
        let shown = !self.panel_shown(self.size.get().0);
        self.panel_override = Some(shown);
        self.status = Some((
            format!("side panel {}", if shown { "shown" } else { "hidden" }),
            true,
        ));
    }

    /// Whether a chat's side panel shows at this width.
    pub fn panel_shown(&self, width: u16) -> bool {
        self.panel_override.unwrap_or(match self.prefs.panel {
            layout::Panel::Always => true,
            layout::Panel::Off => false,
            layout::Panel::Auto => width >= layout::PANEL_AUTO,
        })
    }

    /// `[` and `]` on a tab with a split move it, and the choice is kept.
    fn resize_split(&mut self, step: i32) {
        let knob = match self.screen {
            Screen::Pick => 0,
            Screen::Bench => 1,
            Screen::Serve => 2,
            Screen::Models => 3,
            Screen::Chat => 5,
            _ => {
                self.status = Some(("this screen has no split to move".into(), false));
                return;
            }
        };
        self.prefs.nudge(knob, step);
        self.status = Some((
            format!(
                "{} {} · saved",
                layout::Prefs::KNOBS[knob].label.to_lowercase(),
                self.prefs.knob_value(knob)
            ),
            true,
        ));
        self.save();
    }

    /// Something else holds a model in memory: say so and wait for the same
    /// key again, then free it. True when `what` may load now.
    pub fn make_room(&mut self, what: &str, keep_chat: bool, keep_serve: bool) -> bool {
        let chat = self.chat.is_some() && !keep_chat;
        let serve = self.serve.is_some() && !keep_serve;
        let bench = self
            .bench
            .as_ref()
            .is_some_and(|b| b.phase == BenchPhase::Running);
        if bench {
            self.status = Some((
                "a benchmark is running: stop it on Bench first".into(),
                false,
            ));
            return false;
        }
        if !chat && !serve {
            return true;
        }
        let holder = match (chat, serve) {
            (true, true) => "the chat model and the server",
            (true, false) => "the chat model",
            _ => "the server",
        };
        let token = format!("room:{what}");
        if self.armed.as_deref() == Some(token.as_str()) {
            if chat {
                self.unload_chat();
            }
            if serve && let Some(mut s) = self.serve.take() {
                s.stop();
            }
            self.reap();
            return true;
        }
        self.status = Some((
            format!("{what} needs the memory {holder} uses: press it again to stop {holder} first"),
            false,
        ));
        self.confirm = Some(token);
        false
    }

    // ---- drawing ----

    /// Tab bar and rule on top, the key bar at the bottom; returns the body.
    pub fn draw_frame(&self, f: &mut Frame) -> (Rect, Rect) {
        let area = f.area();
        self.size.set((area.width, area.height));
        let [top, rule, body, footer] = Layout::vertical([
            Constraint::Length(1),
            Constraint::Length(1),
            Constraint::Min(4),
            Constraint::Length(1),
        ])
        .areas(area);
        let current = self.tab();
        let short = area.width < 112;
        let mut spans = vec![
            Span::styled(" ◆ ", Style::new().fg(EMBER)),
            Span::styled("FERRUM", Style::new().fg(GOLD).add_modifier(Modifier::BOLD)),
            Span::raw("  "),
        ];
        for (i, (tab, name)) in TABS.iter().enumerate() {
            let label = if short && name.len() > 5 {
                &name[..if *tab == Tab::Settings { 3 } else { 5 }]
            } else {
                name
            };
            if *tab == current {
                spans.push(Span::styled(
                    format!(" {} {label} ", i + 1),
                    Style::new()
                        .fg(Color::Black)
                        .bg(EMBER)
                        .add_modifier(Modifier::BOLD),
                ));
            } else {
                spans.push(Span::styled(format!(" {}", i + 1), Style::new().fg(FAINT)));
                spans.push(Span::styled(format!(" {label} "), Style::new().fg(DIM)));
            }
            if !short {
                spans.push(Span::raw(" "));
            }
        }
        let left_w: usize = spans.iter().map(|s| text_width(&s.content)).sum();
        f.render_widget(Paragraph::new(Line::from(spans)), top);
        let room = (top.width as usize).saturating_sub(left_w + 3);
        let right = self.running_spans(room);
        f.render_widget(
            Paragraph::new(Line::from(right)).alignment(Alignment::Right),
            top,
        );
        f.render_widget(
            Paragraph::new(Line::styled(
                "─".repeat(rule.width as usize),
                Style::new().fg(FAINT),
            )),
            rule,
        );
        (body, footer)
    }

    /// What is loaded and what it is doing, for the right end of the tab
    /// bar, in at most `room` columns. The live part (the tail of what the
    /// model writes, or the step a run is on) gives way first.
    fn running_spans(&self, room: usize) -> Vec<Span<'static>> {
        let t = self.elapsed();
        // (model, what it is doing, the live detail, colour)
        let (model, doing, detail, color): (String, String, String, Color) =
            if let Some(c) = &self.chat {
                let (doing, detail, color) = c.activity();
                (short_name(&c.label), doing, detail, color)
            } else if let Some(b) = self
                .bench
                .as_ref()
                .filter(|b| b.phase == BenchPhase::Running)
            {
                let name = self
                    .models
                    .get(b.model)
                    .map_or(String::new(), |m| short_name(&m.name));
                let (doing, detail) = match b.measure {
                    evalui::Measure::Speed => (
                        format!(
                            "speed bench {}/{}",
                            b.rows.len() + b.chunk_rows.len(),
                            b.expected
                        ),
                        b.stage.clone(),
                    ),
                    evalui::Measure::Intelligence => (
                        format!("intelligence {}/{}", b.iq.answers.len(), b.iq.expected),
                        match agentui::feed_tail(&b.iq.feed) {
                            t if t.is_empty() => b.iq.stage.clone(),
                            t => t,
                        },
                    ),
                    evalui::Measure::Agent => match &b.ag.current {
                        Some((task, turn, _, _, last)) => (
                            format!(
                                "agent task {}/{} · reply {}",
                                task + 1,
                                b.ag.tasks.len(),
                                turn + 1
                            ),
                            match b.ag.feed_tail() {
                                t if t.is_empty() => last.clone(),
                                t => t,
                            },
                        ),
                        None => (
                            format!("agent tasks {}/{}", b.ag.done.len(), b.ag.tasks.len()),
                            b.ag.stage.clone(),
                        ),
                    },
                };
                (name, doing, detail, EMBER)
            } else if let Some(s) = self.serve.as_ref().filter(|s| s.exit.is_none()) {
                let stats = serve::request_stats(&s.lines);
                let detail = match stats.last() {
                    Some((_, tps)) => format!("{} requests · last {tps:.1} tok/s", stats.len()),
                    None => "no requests yet".into(),
                };
                let doing = if s.ready { "serving" } else { "loading" };
                (
                    short_name(&s.name),
                    doing.into(),
                    detail,
                    if s.ready { GOOD } else { GOLD },
                )
            } else {
                (String::new(), String::new(), String::new(), DIM)
            };
        let busy = color == EMBER || color == GOLD;
        let mut tail: Vec<(String, Color)> = Vec::new();
        if let Some(s) = &self.serve
            && self.chat.is_some()
        {
            tail.push(match (&s.exit, s.ready) {
                (Some(_), _) => ("server stopped".into(), BAD),
                (None, true) => (format!("{} ●", s.port_label()), GOOD),
                _ => (format!("{} {}", s.port_label(), spinner(t)), GOLD),
            });
        }
        let used = self.weights_loaded();
        if used > 0.0 {
            let share = used / self.machine.mem_gb.max(1) as f64;
            tail.push((
                format!("{used:.1}/{} GB", self.machine.mem_gb),
                if share > 0.85 {
                    BAD
                } else if share > 0.65 {
                    GOLD
                } else {
                    DIM
                },
            ));
        }
        let sep = "  ·  ";
        let mut spans: Vec<Span<'static>> = Vec::new();
        let fixed: usize = tail
            .iter()
            .map(|(s, _)| text_width(s) + sep.len() - 2)
            .sum::<usize>()
            + 1;
        if model.is_empty() {
            spans.push(Span::styled("○ no model loaded", Style::new().fg(DIM)));
        } else {
            let mark = if busy {
                spinner(t).to_string()
            } else {
                "●".into()
            };
            // What it is doing matters more than its name: the name goes first.
            let with_name = format!("{mark} {model} · {doing}");
            let bare = format!("{mark} {doing}");
            let show_name = text_width(&with_name) + fixed <= room;
            let head_w = if show_name {
                text_width(&with_name)
            } else {
                text_width(&bare)
            };
            spans.push(Span::styled(format!("{mark} "), Style::new().fg(color)));
            if show_name {
                spans.push(Span::styled(
                    model,
                    Style::new().fg(TEXT).add_modifier(Modifier::BOLD),
                ));
                spans.push(Span::styled(" · ", Style::new().fg(FAINT)));
            }
            spans.push(Span::styled(
                clip(&doing, room.saturating_sub(fixed + 2)),
                Style::new().fg(color),
            ));
            // The newest part of the detail, as much as fits.
            let left = room.saturating_sub(head_w + fixed + 3).min(64);
            if !detail.is_empty() && left >= 12 {
                let chars: Vec<char> = detail.chars().collect();
                let mut start = chars.len();
                let mut w = 0;
                while start > 0 && w + cw(chars[start - 1]) < left {
                    start -= 1;
                    w += cw(chars[start]);
                }
                let shown: String = chars[start..].iter().collect();
                let shown = if start > 0 {
                    format!("…{}", shown.trim_start())
                } else {
                    shown
                };
                spans.push(Span::styled(" · ", Style::new().fg(FAINT)));
                spans.push(Span::styled(shown, Style::new().fg(DIM)));
            }
        }
        for (text, c) in tail {
            spans.push(Span::styled(sep, Style::new().fg(FAINT)));
            spans.push(Span::styled(text, Style::new().fg(c)));
        }
        spans.push(Span::raw(" "));
        spans
    }

    /// GiB of weights held by the chat model, the server and a running bench.
    pub fn weights_loaded(&self) -> f64 {
        let chat = self
            .chat
            .as_ref()
            .and_then(|c| c.info.as_ref())
            .map_or(0.0, |i| i.weights_gib);
        let serve = if self.serve.as_ref().is_some_and(|s| s.exit.is_none()) {
            self.serve_gib
        } else {
            0.0
        };
        let bench = self
            .bench
            .as_ref()
            .filter(|b| b.phase == BenchPhase::Running)
            .and_then(|b| b.loaded_gib())
            .unwrap_or(0.0);
        chat + serve + bench
    }

    /// The key bar, with `?` added; remembered for the key list.
    pub fn footer(&self, f: &mut Frame, area: Rect, keys: &[(&str, &str)]) {
        let mut all: Vec<(&str, &str)> = keys.to_vec();
        if !self.typing() {
            all.push(("?", "keys"));
        }
        *self.keys_shown.borrow_mut() = keys
            .iter()
            .map(|(k, w)| (k.to_string(), w.to_string()))
            .collect();
        keys_footer(f, area, &all, &self.status);
    }

    /// The key list or the palette, over everything.
    pub fn draw_overlay(&self, f: &mut Frame) {
        match &self.overlay {
            Overlay::None => {}
            Overlay::Keys => self.draw_keys(f),
            Overlay::Palette { query, sel } => draw_palette(f, query, *sel),
        }
    }

    fn draw_keys(&self, f: &mut Frame) {
        let here = self.keys_shown.borrow().clone();
        let global: [(&str, &str); 6] = [
            ("1-8", "switch tab"),
            ("ctrl-p", "command palette"),
            ("[ ]", "move the split on this screen"),
            ("ctrl-b", "chat side panel"),
            ("esc", "back"),
            ("?", "this list"),
        ];
        let h = (here.len() + global.len() + 6) as u16;
        let rect = centered(f.area(), 64, h);
        let title = format!("Keys · {:?}", self.tab());
        let inner = pane(f, rect, &title, true, "any key closes");
        let row = |k: &str, w: &str| {
            Line::from(vec![
                Span::styled(
                    format!("  {k:<14}"),
                    Style::new().fg(GOLD).add_modifier(Modifier::BOLD),
                ),
                Span::styled(w.to_string(), Style::new().fg(TEXT)),
            ])
        };
        let mut lines = vec![layout::heading("this screen")];
        lines.extend(here.iter().map(|(k, w)| row(k, w)));
        lines.push(Line::default());
        lines.push(layout::heading("everywhere"));
        lines.extend(global.iter().map(|(k, w)| row(k, w)));
        f.render_widget(Paragraph::new(lines), inner);
    }

    /// The Settings row index of Layout's first knob.
    pub fn layout_row(&self) -> usize {
        4
    }
}

fn palette_hits(query: &str) -> Vec<(Cmd, &'static str)> {
    let q = query.to_lowercase();
    COMMANDS
        .iter()
        .filter(|(_, name)| {
            q.split_whitespace()
                .all(|w| name.to_lowercase().contains(w))
        })
        .copied()
        .collect()
}

fn draw_palette(f: &mut Frame, query: &str, sel: usize) {
    let hits = palette_hits(query);
    let n = hits.len().clamp(1, 12) as u16;
    let area = f.area();
    let mut rect = centered(area, 64, n + 4);
    rect.y = (area.height / 5)
        .max(2)
        .min(area.height.saturating_sub(rect.height));
    let inner = pane(f, rect, "Commands", true, "enter runs · esc closes");
    let w = inner.width as usize;
    let mut lines = vec![
        Line::from(vec![
            Span::styled(" ❯ ", Style::new().fg(EMBER)),
            Span::styled(query.to_string(), Style::new().fg(Color::White)),
            Span::styled("▏", Style::new().fg(GOLD)),
        ]),
        Line::styled("─".repeat(w), Style::new().fg(FAINT)),
    ];
    if hits.is_empty() {
        lines.push(Line::styled("  no command matches", Style::new().fg(FAINT)));
    }
    let sel = sel.min(hits.len().saturating_sub(1));
    let top = (sel + 1).saturating_sub(12);
    for (i, (_, name)) in hits.iter().enumerate().skip(top).take(12) {
        lines.push(layout::list_row(
            i == sel,
            true,
            w,
            vec![Span::styled(
                format!(" {name}"),
                Style::new().fg(if i == sel { Color::White } else { TEXT }),
            )],
        ));
    }
    f.render_widget(Paragraph::new(lines), inner);
}

/// A model label without its folder, quant or "UD" tag, for tight spots:
/// "Qwen3.8-27B-UD-Q4_K_XL" reads "Qwen3.8-27B".
pub fn short_name(label: &str) -> String {
    let name = label.rsplit('/').next().unwrap_or(label);
    let name = name.split(" · ").next().unwrap_or(name);
    let name = name.trim_end_matches(".gguf");
    let quant = quant_of(name);
    let kept: Vec<&str> = name
        .split('-')
        .filter(|p| *p != "UD" && quant.as_deref() != Some(&p.to_ascii_uppercase()))
        .collect();
    // The quant is the tail: "Q4_K_XL" splits on '-' no further, so drop
    // the pieces from the quant on.
    let joined = kept.join("-");
    let short = match quant
        .as_deref()
        .and_then(|q| joined.to_ascii_uppercase().find(q))
    {
        Some(at) => joined[..at].trim_end_matches('-').to_string(),
        None => joined,
    };
    clip(if short.is_empty() { name } else { &short }, 28)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn short_names_drop_the_quant() {
        assert_eq!(short_name("Qwen3.8-27B-UD-Q4_K_XL"), "Qwen3.8-27B");
        assert_eq!(
            short_name("Tiel-Coder-35B-A3B-MTP-UD-IQ4_XS"),
            "Tiel-Coder-35B-A3B-MTP"
        );
        assert_eq!(
            short_name("Swift-1.5-Qwen3.8-27B-Q4_K_M · tuned"),
            "Swift-1.5-Qwen3.8-27B"
        );
        assert_eq!(short_name("plain"), "plain");
    }
}
