//! "Chat" / "Serve": choose a favorite, build a new setup, or run one once.
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
            KeyCode::Down | KeyCode::Char('j') | KeyCode::Tab => {
                self.pick_sel = (self.pick_sel + 1).min(last)
            }
            KeyCode::Esc | KeyCode::Char('q') | KeyCode::Left => self.go_home(),
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

    pub fn draw_pick(&self, f: &mut Frame) {
        let [head, body, footer] = Layout::vertical([
            Constraint::Length(2),
            Constraint::Min(8),
            Constraint::Length(1),
        ])
        .areas(f.area());
        header(
            f,
            head,
            &[self.mode.label(), "choose a setup"],
            vec![pill(self.mode.pill(), EMBER)],
        );
        let [list, preview] =
            Layout::horizontal([Constraint::Percentage(46), Constraint::Percentage(54)])
                .areas(body);

        let entries = self.pick_entries();
        let sel = self.pick_sel.min(entries.len() - 1);
        let block = panel("Setups", true);
        let inner = block.inner(list);
        f.render_widget(block, list);
        let width = inner.width as usize;
        let mut lines: Vec<Line> = Vec::new();
        let mut sel_line = 0;
        let fav_count = entries
            .iter()
            .filter(|e| matches!(e, Entry::Fav(_)))
            .count();
        let heading = |s: &str| {
            Line::styled(
                format!("  {s}"),
                Style::new().fg(DIM).add_modifier(Modifier::BOLD),
            )
        };
        lines.push(heading("FAVORITES"));
        if fav_count == 0 {
            lines.push(Line::styled(
                "  none yet · create one with New setup",
                Style::new().fg(FAINT),
            ));
        }
        for (n, entry) in entries.iter().enumerate() {
            if *entry == Entry::Custom {
                lines.push(Line::default());
                lines.push(heading("OTHER"));
            }
            let on = n == sel;
            let bg = if on { SELECTED } else { Color::Reset };
            let bar = Span::styled(
                if on { " ▌ " } else { "   " },
                Style::new().fg(EMBER).bg(bg),
            );
            let (title, sub, tail, bad) = match entry {
                Entry::Fav(i) => {
                    let fav = &self.favs[*i];
                    let model = self.fav_model(fav);
                    let sub = match model {
                        Some(m) => match &m.problem {
                            Some(why) => (format!("✗ {why}"), true),
                            None => (
                                format!(
                                    "{}{}",
                                    m.name,
                                    fav.values
                                        .get("context")
                                        .and_then(|v| v.as_str())
                                        .map(|c| format!(" · ctx {c}"))
                                        .unwrap_or_default()
                                ),
                                false,
                            ),
                        },
                        None => ("✗ model file not found".to_string(), true),
                    };
                    (
                        format!("★ {}", fav.name),
                        sub.0,
                        self.drafter_badge(fav),
                        sub.1,
                    )
                }
                Entry::Custom => (
                    "＋ New setup".to_string(),
                    "pick a model and settings, saved as a favorite".to_string(),
                    String::new(),
                    false,
                ),
                Entry::OneTime => (
                    "▷ One-time run".to_string(),
                    "configure and launch without saving".to_string(),
                    String::new(),
                    false,
                ),
            };
            if on {
                sel_line = lines.len();
            }
            let name_w = width.saturating_sub(tail.chars().count() + 5);
            lines.push(Line::from(vec![
                bar.clone(),
                Span::styled(
                    format!("{:<name_w$}", clip(&title, name_w)),
                    Style::new()
                        .fg(if on { Color::White } else { TEXT })
                        .bg(bg)
                        .add_modifier(if on {
                            Modifier::BOLD
                        } else {
                            Modifier::empty()
                        }),
                ),
                Span::styled(format!(" {tail} "), Style::new().fg(GOLD).bg(bg)),
            ]));
            lines.push(Line::from(vec![
                bar,
                Span::styled(
                    format!(
                        "{:<w$}",
                        clip(&sub, width.saturating_sub(5)),
                        w = width.saturating_sub(5)
                    ),
                    Style::new().fg(if bad { BAD } else { DIM }).bg(bg),
                ),
                Span::styled("  ", Style::new().bg(bg)),
            ]));
        }
        let h = inner.height as usize;
        let top = (sel_line + 2).saturating_sub(h);
        f.render_widget(
            Paragraph::new(lines.into_iter().skip(top).take(h).collect::<Vec<_>>()),
            inner,
        );

        self.draw_preview(f, preview, entries[sel]);
        let keys: &[(&str, &str)] = match entries[sel] {
            Entry::Fav(_) => &[
                ("↑↓", "move"),
                ("enter", "run"),
                ("e", "edit"),
                ("x", "delete"),
                ("n", "new"),
                ("o", "one-time"),
                ("esc", "back"),
            ],
            _ => &[
                ("↑↓", "move"),
                ("enter", "open"),
                ("n", "new"),
                ("o", "one-time"),
                ("esc", "back"),
            ],
        };
        keys_footer(f, footer, keys, &self.status);
    }

    fn draw_preview(&self, f: &mut Frame, area: Rect, entry: Entry) {
        let block = panel("Preview", false);
        let inner = block.inner(area);
        f.render_widget(block, area);
        let w = inner.width as usize;
        let dim = Style::new().fg(DIM);
        let mut lines: Vec<Line> = Vec::new();
        match entry {
            Entry::Fav(i) => {
                let fav = &self.favs[i];
                lines.push(Line::styled(
                    format!(" ★ {}", fav.name),
                    Style::new().fg(GOLD).add_modifier(Modifier::BOLD),
                ));
                match self.fav_model(fav) {
                    Some(m) => {
                        lines.push(Line::styled(
                            format!(" {}", clip(&m.name, w.saturating_sub(2))),
                            Style::new().fg(TEXT),
                        ));
                        match &m.problem {
                            Some(why) => lines.push(Line::styled(
                                format!(" ✗ can't run: {}", clip(why, w.saturating_sub(14))),
                                Style::new().fg(BAD),
                            )),
                            None => lines.push(Line::styled(
                                format!(" {} · ✓ supported", gib(m.size)),
                                Style::new().fg(GOOD),
                            )),
                        }
                    }
                    None => lines.push(Line::styled(
                        " ✗ model file not found",
                        Style::new().fg(BAD),
                    )),
                }
                lines.push(Line::default());
                lines.push(Line::styled(" SETTINGS", dim.add_modifier(Modifier::BOLD)));
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
                        Span::styled(format!("  {:<20}", fl.label), dim),
                        Span::styled(clip(&shown, w.saturating_sub(24)), Style::new().fg(EMBER)),
                    ]));
                }
                if !any {
                    lines.push(Line::styled(
                        "  defaults for everything",
                        Style::new().fg(FAINT),
                    ));
                }
                lines.push(Line::default());
                lines.push(Line::styled(" COMMAND", dim.add_modifier(Modifier::BOLD)));
                let (bin, args) = self.command_for(
                    fav.mode,
                    self.fav_model(fav).map(|m| m.path.as_path()),
                    &|k| {
                        fav.values
                            .get(k)
                            .and_then(|v| v.as_str())
                            .unwrap_or("")
                            .to_string()
                    },
                );
                let mut spans = vec![Span::styled(format!(" {bin}"), Style::new().fg(GOOD))];
                for a in &args {
                    spans.push(if a.starts_with('-') {
                        Span::styled(format!(" {a}"), Style::new().fg(GOLD))
                    } else {
                        Span::styled(format!(" {}", shell_quote(a)), Style::new().fg(TEXT))
                    });
                }
                lines.push(Line::from(spans));
            }
            Entry::Custom => {
                lines.push(Line::styled(
                    " ＋ New setup",
                    Style::new().fg(GOLD).add_modifier(Modifier::BOLD),
                ));
                lines.push(Line::default());
                for s in [
                    "Start from the defaults: pick a model, tune",
                    "sampling and reasoning, add a DSpark / DFlash",
                    "drafter. When you launch you'll name it and it",
                    "is saved here as a favorite.",
                ] {
                    lines.push(Line::styled(format!(" {s}"), Style::new().fg(TEXT)));
                }
            }
            Entry::OneTime => {
                lines.push(Line::styled(
                    " ▷ One-time run",
                    Style::new().fg(GOLD).add_modifier(Modifier::BOLD),
                ));
                lines.push(Line::default());
                for s in [
                    "Starts from the settings you used last, lets you",
                    "change anything, and launches without saving a",
                    "favorite.",
                ] {
                    lines.push(Line::styled(format!(" {s}"), Style::new().fg(TEXT)));
                }
            }
        }
        f.render_widget(Paragraph::new(lines).wrap(Wrap { trim: false }), inner);
    }
}
