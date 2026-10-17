//! "Chat" / "Serve": choose a favorite, build a new setup, or run one once.
use crate::layout;
use crate::*;

#[derive(Clone, Copy, PartialEq)]
pub enum Entry {
    Fav(usize),
    Custom,
    OneTime,
}

impl App {
    pub fn open_pick(&mut self, mode: Mode) {
        self.mode = mode;
        self.pick_sel = 0;
        self.screen = Screen::Pick;
    }

    pub fn pick_entries(&self) -> Vec<Entry> {
        let mode = self.mode;
        self.favs
            .iter()
            .enumerate()
            // Agents can run any chat favorite too: it supplies the model and settings.
            .filter(|(_, f)| match mode {
                Mode::Agent => f.mode != Mode::Serve,
                m => f.mode == m,
            })
            .map(|(i, _)| Entry::Fav(i))
            .chain([Entry::Custom, Entry::OneTime])
            .collect()
    }

    fn fav_model(&self, fav: &Fav) -> Option<&Model> {
        self.models
            .iter()
            .find(|m| m.path.display().to_string() == fav.model)
    }

    pub fn pick_key(&mut self, key: KeyEvent) {
        let entries = self.pick_entries();
        let last = entries.len() - 1;
        self.pick_sel = self.pick_sel.min(last);
        let entry = entries[self.pick_sel];
        match key.code {
            KeyCode::Up | KeyCode::Char('k') => self.pick_sel = self.pick_sel.saturating_sub(1),
            KeyCode::Down | KeyCode::Char('j') => self.pick_sel = (self.pick_sel + 1).min(last),
            KeyCode::Esc | KeyCode::Char('q') | KeyCode::Left => self.go_tab(Tab::Home),
            KeyCode::Char('n') => self.open_editor(self.mode, Origin::Custom),
            KeyCode::Char('o') => self.open_editor(self.mode, Origin::OneTime),
            KeyCode::Enter | KeyCode::Right | KeyCode::Char(' ') => match entry {
                Entry::Fav(i) => {
                    let wanted = self.mode;
                    if self.load_favorite(i) {
                        // A chat favorite opened from Agents runs as an agent.
                        if wanted == Mode::Agent {
                            self.mode = Mode::Agent;
                        }
                        self.start();
                    }
                }
                Entry::Custom => self.open_editor(self.mode, Origin::Custom),
                Entry::OneTime => self.open_editor(self.mode, Origin::OneTime),
            },
            KeyCode::Char('e') => {
                if let Entry::Fav(i) = entry {
                    let name = self.favs[i].name.clone();
                    self.load_favorite(i);
                    self.open_editor_keep(Origin::Edit(name));
                }
            }
            KeyCode::Char('x') | KeyCode::Delete | KeyCode::Backspace => {
                if let Entry::Fav(i) = entry {
                    let name = self.favs[i].name.clone();
                    if self.armed.as_deref() == Some(name.as_str()) {
                        self.favs.remove(i);
                        self.pick_sel = self.pick_sel.saturating_sub(1);
                        self.status = Some((format!("removed {name:?}"), true));
                        self.save();
                    } else {
                        self.status = Some((format!("press x again to delete {name:?}"), false));
                        self.confirm = Some(name);
                    }
                }
            }
            _ => {}
        }
    }

    /// Open the editor on whatever settings are loaded right now.
    pub fn open_editor_keep(&mut self, origin: Origin) {
        self.origin = origin;
        self.screen = Screen::Editor;
        self.focus = Focus::Settings;
        self.sel = 0;
    }

    fn drafter_badge(&self, fav: &Fav) -> String {
        match fav.values.get("draft").and_then(|v| v.as_str()) {
            None | Some("off") | Some("") => String::new(),
            Some("mtp") => "⚡ mtp".into(),
            Some(p) => format!(
                "⚡ {}",
                self.drafters
                    .iter()
                    .find(|d| d.path.display().to_string() == p)
                    .map(|d| d.kind)
                    .unwrap_or("drafter")
            ),
        }
    }

    pub fn draw_pick(&self, f: &mut Frame, body: Rect, footer: Rect) {
        let entries = self.pick_entries();
        let sel = self.pick_sel.min(entries.len() - 1);
        let (list, side) = layout::columns(body, self.prefs.setups, 40, 50, self.prefs.compact);
        let width = inner_w(list);
        let fav_count = entries
            .iter()
            .filter(|e| matches!(e, Entry::Fav(_)))
            .count();
        let mut lines: Vec<Line> = Vec::new();
        let mut sel_line = 0;
        lines.push(layout::heading_note(
            "favorites",
            &fav_count.to_string(),
            width,
        ));
        if fav_count == 0 {
            lines.push(Line::styled(
                "  none yet · n makes one",
                Style::new().fg(FAINT),
            ));
        }
        for (n, entry) in entries.iter().enumerate() {
            if *entry == Entry::Custom {
                lines.push(Line::default());
                lines.push(layout::heading("new"));
            }
            let on = n == sel;
            if on {
                sel_line = lines.len();
            }
            match entry {
                Entry::Fav(i) => {
                    let fav = &self.favs[*i];
                    let model = self.fav_model(fav);
                    let (sub, bad) = match model {
                        Some(m) => match &m.problem {
                            Some(why) => (format!("✗ {why}"), true),
                            None => (
                                format!(
                                    "{} · {}{}",
                                    gib(m.size),
                                    fav.mode.key(),
                                    fav.values
                                        .get("context")
                                        .and_then(|v| v.as_str())
                                        .map(|c| format!(" · context {c}"))
                                        .unwrap_or_default()
                                ),
                                false,
                            ),
                        },
                        None => ("✗ model file not found".to_string(), true),
                    };
                    let speed = model
                        .and_then(|m| {
                            self.dashboard()
                                .speed_of(&m.path.display().to_string())
                                .map(|s| format!("{:.2}×", s.speedup))
                        })
                        .unwrap_or_default();
                    let badge = self.drafter_badge(fav);
                    let tail = format!(
                        "{badge}{}{speed} ",
                        if speed.is_empty() { "" } else { "  " }
                    );
                    let name_w = width.saturating_sub(text_width(&tail) + 4);
                    lines.push(layout::list_row(
                        on,
                        true,
                        width,
                        vec![
                            Span::styled(" ★ ", Style::new().fg(GOLD)),
                            Span::styled(
                                layout::fit_w(&fav.name, name_w),
                                Style::new()
                                    .fg(if on { Color::White } else { TEXT })
                                    .add_modifier(if on {
                                        Modifier::BOLD
                                    } else {
                                        Modifier::empty()
                                    }),
                            ),
                            Span::styled(badge.clone(), Style::new().fg(GOLD)),
                            Span::styled(
                                format!("{}{speed} ", if speed.is_empty() { "" } else { "  " }),
                                Style::new().fg(GOOD).add_modifier(Modifier::BOLD),
                            ),
                        ],
                    ));
                    lines.push(layout::list_row(
                        on,
                        true,
                        width,
                        vec![Span::styled(
                            format!("   {}", clip(&sub, width.saturating_sub(5))),
                            Style::new().fg(if bad { BAD } else { DIM }),
                        )],
                    ));
                }
                Entry::Custom | Entry::OneTime => {
                    let (title, what) = if *entry == Entry::Custom {
                        ("+ New setup", "saved as a favorite when you launch it")
                    } else {
                        ("▷ One-time run", "your last settings, nothing saved")
                    };
                    lines.push(layout::list_row(
                        on,
                        true,
                        width,
                        vec![
                            Span::styled(
                                format!(" {title}  "),
                                Style::new()
                                    .fg(if on { Color::White } else { TEXT })
                                    .add_modifier(if on {
                                        Modifier::BOLD
                                    } else {
                                        Modifier::empty()
                                    }),
                            ),
                            Span::styled(
                                clip(what, width.saturating_sub(text_width(title) + 5)),
                                Style::new().fg(FAINT),
                            ),
                        ],
                    ));
                }
            }
        }
        // Side by side, or stacked: the list takes what it needs (up to
        // half) and the preview the rest.
        let (list, preview) = match side {
            Some(side) => (list, side),
            None => {
                let h = (lines.len() as u16 + 2).clamp(6, list.height / 2);
                let [top, bottom] =
                    Layout::vertical([Constraint::Length(h), Constraint::Min(4)]).areas(list);
                (top, bottom)
            }
        };
        let title = match self.mode {
            Mode::Chat => "Chat setups",
            Mode::Agent => "Agent setups",
            Mode::Serve => "Server setups",
        };
        let inner = layout::pane(f, list, title, true, "");
        layout::show(f, inner, lines, Some(sel_line + 1));
        self.draw_preview(f, preview, entries[sel]);
        let keys: &[(&str, &str)] = match entries[sel] {
            Entry::Fav(_) => &[
                ("↑↓", "choose"),
                ("enter", "launch"),
                ("e", "edit"),
                ("x", "delete"),
                ("n", "new"),
                ("o", "one-time"),
                ("[ ]", "resize"),
            ],
            _ => &[
                ("↑↓", "choose"),
                ("enter", "open"),
                ("n", "new"),
                ("o", "one-time"),
            ],
        };
        self.footer(f, footer, keys);
    }

    fn draw_preview(&self, f: &mut Frame, area: Rect, entry: Entry) {
        let w = inner_w(area);
        let dim = Style::new().fg(DIM);
        let mut lines: Vec<Line> = Vec::new();
        let title = match entry {
            Entry::Fav(i) => self.favs[i].name.clone(),
            Entry::Custom => "New setup".into(),
            Entry::OneTime => "One-time run".into(),
        };
        match entry {
            Entry::Fav(i) => {
                let fav = &self.favs[i];
                let model = self.fav_model(fav);
                match model {
                    Some(m) => {
                        lines.push(layout::kv(
                            "model",
                            format!("{} · {}", m.name, gib(m.size)),
                            Style::new().fg(Color::White),
                            10,
                        ));
                        if let Some(why) = &m.problem {
                            lines.push(layout::kv(
                                "status",
                                format!("✗ can't run: {why}"),
                                Style::new().fg(BAD),
                                10,
                            ));
                        }
                    }
                    None => lines.push(layout::kv(
                        "model",
                        "✗ model file not found".into(),
                        Style::new().fg(BAD),
                        10,
                    )),
                }
                lines.push(Line::default());
                lines.push(layout::heading("settings"));
                let mut any = false;
                for fl in &self.fields {
                    let in_scope = fl.scope.applies(fav.mode);
                    let v = fav
                        .values
                        .get(fl.key)
                        .and_then(|v| v.as_str())
                        .unwrap_or("");
                    if !in_scope || v.is_empty() {
                        continue;
                    }
                    any = true;
                    let shown = if fl.key == "draft" {
                        self.draft_label(v)
                    } else if matches!(fl.kind, Kind::Text { secret: true, .. }) {
                        "••••••".to_string()
                    } else {
                        v.to_string()
                    };
                    lines.push(Line::from(vec![
                        Span::styled(format!(" {:<20}", fl.label), dim),
                        Span::styled(clip(&shown, w.saturating_sub(22)), Style::new().fg(EMBER)),
                    ]));
                }
                if !any {
                    lines.push(Line::styled(
                        " defaults for everything",
                        Style::new().fg(FAINT),
                    ));
                }
                if let Some(m) = model {
                    lines.push(Line::default());
                    lines.extend(self.measured_lines(&m.path.display().to_string(), w));
                }
                lines.push(Line::default());
                lines.push(layout::heading("command"));
                let (bin, args) =
                    self.command_for(fav.mode, model.map(|m| m.path.as_path()), &|k| {
                        fav.values
                            .get(k)
                            .and_then(|v| v.as_str())
                            .unwrap_or("")
                            .to_string()
                    });
                lines.extend(command_lines(bin, &args, w));
            }
            Entry::Custom => {
                lines.extend(layout::wrapped(
                    "Start from the defaults: pick a model, tune sampling and reasoning, add a DSpark, DFlash or MTP drafter. When you launch you name it, and it is kept here as a favorite.",
                    1,
                    w,
                    Style::new().fg(TEXT),
                ));
                lines.push(Line::default());
                lines.extend(layout::wrapped(
                    "Bench › speed finds the fastest drafter for a model and can save it as a favorite for you.",
                    1,
                    w,
                    Style::new().fg(DIM),
                ));
            }
            Entry::OneTime => {
                lines.extend(layout::wrapped(
                    "Starts from the settings you used last, lets you change anything, and launches without saving a favorite.",
                    1,
                    w,
                    Style::new().fg(TEXT),
                ));
            }
        }
        let inner = layout::pane(f, area, &title, false, "");
        f.render_widget(Paragraph::new(lines), inner);
    }
}

/// A command one flag per line, each continued with a backslash, so it reads
/// at a glance and still pastes into a shell. Long values wrap under the flag.
pub fn command_lines(bin: &str, args: &[String], width: usize) -> Vec<Line<'static>> {
    let flag = Style::new().fg(GOLD);
    let value = Style::new().fg(TEXT);
    let mut groups: Vec<Vec<Sc>> = vec![
        format!(" {bin}")
            .chars()
            .map(|c| (c, Style::new().fg(GOOD).add_modifier(Modifier::BOLD)))
            .collect(),
    ];
    for a in args {
        if a.starts_with('-') {
            groups.push(format!("   {a}").chars().map(|c| (c, flag)).collect());
        } else if let Some(g) = groups.last_mut() {
            g.extend(format!(" {}", shell_quote(a)).chars().map(|c| (c, value)));
        }
    }
    let n = groups.len();
    let mut out = Vec::new();
    for (i, mut g) in groups.into_iter().enumerate() {
        if i + 1 < n {
            g.extend(" \\".chars().map(|c| (c, Style::new().fg(FAINT))));
        }
        for row in wrap_styled(&g, width, 5) {
            out.push(to_line(&row, Vec::new()));
        }
    }
    out
}
