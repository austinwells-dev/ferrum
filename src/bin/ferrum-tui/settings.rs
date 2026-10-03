//! Settings screen: startup animation, model folders and favorites.
use crate::*;

#[derive(Clone, Copy, PartialEq)]
enum Row {
    Splash,
    Search,
    SearchUrl,
    SearchKey,
    Folder(usize),
    AddFolder,
    Fav(usize),
}

impl App {
    fn settings_rows(&self) -> Vec<Row> {
        let mut rows = vec![Row::Splash, Row::Search, Row::SearchUrl, Row::SearchKey];
        rows.extend((0..self.extra.len()).map(Row::Folder));
        rows.push(Row::AddFolder);
        rows.extend((0..self.favs.len()).map(Row::Fav));
        rows
    }

    fn cycle_search(&mut self, forward: bool) {
        let all = search::PROVIDERS;
        let at = all
            .iter()
            .position(|p| *p == self.search.provider)
            .unwrap_or(0);
        let next = if forward {
            (at + 1) % all.len()
        } else {
            (at + all.len() - 1) % all.len()
        };
        self.search.provider = all[next].into();
    }

    pub fn settings_key(&mut self, key: KeyEvent) {
        let rows = self.settings_rows();
        let last = rows.len() - 1;
        self.set_sel = self.set_sel.min(last);
        let row = rows[self.set_sel];
        match key.code {
            KeyCode::Up | KeyCode::Char('k') => self.set_sel = self.set_sel.saturating_sub(1),
            KeyCode::Down | KeyCode::Char('j') | KeyCode::Tab => {
                self.set_sel = (self.set_sel + 1).min(last)
            }
            KeyCode::Esc | KeyCode::Char('q') => {
                self.save();
                self.go_home();
            }
            KeyCode::Char('r') => self.rescan(None),
            KeyCode::Char('a') => {
                self.buffer.clear();
                self.editing = Editing::AddPath;
            }
            KeyCode::Left | KeyCode::Right if row == Row::Splash => self.splash = !self.splash,
            KeyCode::Left | KeyCode::Right if row == Row::Search => {
                self.cycle_search(key.code == KeyCode::Right)
            }
            KeyCode::Enter | KeyCode::Char(' ') => match row {
                Row::Splash => self.splash = !self.splash,
                Row::Search => self.cycle_search(true),
                Row::SearchUrl => {
                    self.buffer = self.search.url.clone();
                    self.fresh = true;
                    self.editing = Editing::SearchUrl;
                }
                Row::SearchKey => {
                    self.buffer = self.search.key.clone();
                    self.fresh = true;
                    self.editing = Editing::SearchKey;
                }
                Row::AddFolder => {
                    self.buffer.clear();
                    self.editing = Editing::AddPath;
                }
                Row::Fav(i) => {
                    self.buffer = self.favs[i].name.clone();
                    self.fresh = true;
                    self.editing = Editing::RenameFav(i);
                }
                Row::Folder(_) => {}
            },
            KeyCode::Char('x') | KeyCode::Delete | KeyCode::Backspace => match row {
                Row::Folder(i) => {
                    let gone = self.extra.remove(i);
                    self.rescan(None);
                    self.status = Some((format!("removed {}", tilde(&expand(&gone))), true));
                }
                Row::Fav(i) => {
                    let name = self.favs[i].name.clone();
                    if self.armed.as_deref() == Some(name.as_str()) {
                        self.favs.remove(i);
                        self.status = Some((format!("removed {name:?}"), true));
                    } else {
                        self.status = Some((format!("press x again to delete {name:?}"), false));
                        self.confirm = Some(name);
                    }
                }
                _ => {}
            },
            _ => {}
        }
        self.save();
    }

    pub fn draw_settings_screen(&self, f: &mut Frame) {
        let [head, body, footer] = Layout::vertical([
            Constraint::Length(2),
            Constraint::Min(8),
            Constraint::Length(1),
        ])
        .areas(f.area());
        header(f, head, &["Settings"], vec![]);
        let area = centered(body, 96, body.height);
        let block = panel("Settings", true);
        let inner = block.inner(area);
        f.render_widget(block, area);
        let width = inner.width as usize;
        let rows = self.settings_rows();
        let sel = self.set_sel.min(rows.len() - 1);
        let title = |s: &str| {
            Line::styled(
                format!("  {s}"),
                Style::new().fg(DIM).add_modifier(Modifier::BOLD),
            )
        };
        let mut lines: Vec<Line> = Vec::new();
        let mut sel_line = 0;
        let mut put_row = |lines: &mut Vec<Line<'static>>,
                           n: usize,
                           label: String,
                           value: String,
                           value_style: Style| {
            let on = rows[n] == rows[sel];
            let bg = if on { SELECTED } else { Color::Reset };
            if on {
                sel_line = lines.len();
            }
            let label_w = width.saturating_sub(value.chars().count() + 7);
            lines.push(Line::from(vec![
                Span::styled(
                    if on { " ▌ " } else { "   " },
                    Style::new().fg(EMBER).bg(bg),
                ),
                Span::styled(
                    format!("{:<label_w$}", clip(&label, label_w)),
                    Style::new().fg(if on { Color::White } else { TEXT }).bg(bg),
                ),
                Span::styled(format!(" {value} "), value_style.bg(bg)),
                Span::styled("  ", Style::new().bg(bg)),
            ]));
        };
        lines.push(title("STARTUP"));
        put_row(
            &mut lines,
            0,
            "Startup animation".into(),
            if self.splash {
                "‹ on ›".into()
            } else {
                "‹ off ›".into()
            },
            Style::new().fg(EMBER).add_modifier(Modifier::BOLD),
        );
        lines.push(Line::default());
        lines.push(title("WEB SEARCH"));
        let bold = Style::new().fg(EMBER).add_modifier(Modifier::BOLD);
        put_row(
            &mut lines,
            1,
            "Provider".into(),
            format!("‹ {} ›", self.search.provider),
            bold,
        );
        put_row(
            &mut lines,
            2,
            "Search URL".into(),
            if self.search.url.is_empty() {
                "not set".into()
            } else {
                clip(&self.search.url, 40)
            },
            Style::new().fg(DIM),
        );
        put_row(
            &mut lines,
            3,
            "API key".into(),
            match self.search.key.as_str() {
                "" => "not set".into(),
                k if k.starts_with('$') => k.to_string(),
                _ => "•••• saved".into(),
            },
            Style::new().fg(DIM),
        );
        lines.push(Line::styled(
            match self.search.provider.as_str() {
                "searxng" => {
                    "     your SearXNG address; its settings.yml must allow the json format"
                }
                "brave" => "     needs an API key; write $NAME to read it from the environment",
                "custom" => "     a URL with {query} (and {key}); the answer must be JSON",
                "instant" => "     DuckDuckGo's answer API: summaries only, often empty",
                _ => "     DuckDuckGo's HTML results, no setup",
            },
            Style::new().fg(FAINT),
        ));
        lines.push(Line::default());
        lines.push(title("MODEL FOLDERS"));
        let mut n = 4;
        for p in &self.extra {
            put_row(
                &mut lines,
                n,
                tilde(&expand(p)),
                "x remove".into(),
                Style::new().fg(FAINT),
            );
            n += 1;
        }
        put_row(
            &mut lines,
            n,
            "＋ Add a folder or model file".into(),
            String::new(),
            Style::new(),
        );
        n += 1;
        let auto: Vec<String> = roots(&[])
            .into_iter()
            .map(|(p, _)| p)
            .filter(|p| p.exists())
            .map(|p| tilde(&p))
            .collect();
        lines.push(Line::styled(
            format!(
                "     also searched: {}",
                clip(&auto.join("  ·  "), width.saturating_sub(24))
            ),
            Style::new().fg(FAINT),
        ));
        lines.push(Line::default());
        lines.push(title("FAVORITES"));
        if self.favs.is_empty() {
            lines.push(Line::styled(
                "     none yet · save one from Chat or Serve",
                Style::new().fg(FAINT),
            ));
        }
        for (i, fav) in self.favs.iter().enumerate() {
            put_row(
                &mut lines,
                n,
                format!("★ {}", fav.name),
                format!("{} · enter rename · x delete", fav.mode.key()),
                Style::new().fg(DIM),
            );
            n += 1;
            let _ = i;
        }
        lines.push(Line::default());
        lines.push(title("ABOUT"));
        lines.push(Line::styled(
            format!("     config   {}", tilde(&config_path())),
            Style::new().fg(DIM),
        ));
        lines.push(Line::styled(
            format!(
                "     found    {} models ({} supported) · {} drafters",
                self.models.len(),
                self.runnable(),
                self.drafters.len()
            ),
            Style::new().fg(DIM),
        ));
        let h = inner.height as usize;
        let top = (sel_line + 3).saturating_sub(h);
        f.render_widget(
            Paragraph::new(lines.into_iter().skip(top).take(h).collect::<Vec<_>>()),
            inner,
        );
        keys_footer(
            f,
            footer,
            &[
                ("↑↓", "move"),
                ("enter", "change"),
                ("a", "add folder"),
                ("x", "remove"),
                ("r", "rescan"),
                ("esc", "back"),
            ],
            &self.status,
        );
    }
}
