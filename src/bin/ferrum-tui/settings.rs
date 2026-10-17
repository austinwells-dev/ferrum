//! Settings: an index of sections on the left, the selected section's
//! settings on the right, and help for the selected one pinned below them.
use crate::layout::{self, Prefs, fit_w, heading, list_row};
use crate::*;

const SECTIONS: [&str; 5] = [
    "Startup",
    "Web search",
    "Layout",
    "Model folders",
    "Favorites",
];

#[derive(Clone, Copy, PartialEq)]
enum Row {
    Splash,
    Search,
    SearchUrl,
    SearchKey,
    Knob(usize),
    Folder(usize),
    AddFolder,
    Fav(usize),
}

impl Row {
    fn section(self) -> usize {
        match self {
            Row::Splash => 0,
            Row::Search | Row::SearchUrl | Row::SearchKey => 1,
            Row::Knob(_) => 2,
            Row::Folder(_) | Row::AddFolder => 3,
            Row::Fav(_) => 4,
        }
    }
}

impl App {
    fn settings_rows(&self) -> Vec<Row> {
        let mut rows = vec![Row::Splash, Row::Search, Row::SearchUrl, Row::SearchKey];
        rows.extend((0..Prefs::KNOBS.len()).map(Row::Knob));
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
        let step = if key.code == KeyCode::Left { -1 } else { 1 };
        match key.code {
            KeyCode::Up | KeyCode::Char('k') => self.set_sel = self.set_sel.saturating_sub(1),
            KeyCode::Down | KeyCode::Char('j') => self.set_sel = (self.set_sel + 1).min(last),
            // Tab jumps between sections.
            KeyCode::Tab | KeyCode::BackTab => {
                let here = row.section();
                let found = if key.code == KeyCode::Tab {
                    rows.iter().position(|r| r.section() > here)
                } else {
                    rows.iter()
                        .position(|r| r.section() + 1 == here.max(1))
                        .filter(|_| here > 0)
                };
                self.set_sel = found.unwrap_or(if key.code == KeyCode::Tab { 0 } else { last });
            }
            KeyCode::Esc | KeyCode::Char('q') => {
                self.save();
                self.go_tab(Tab::Home);
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
            KeyCode::Left | KeyCode::Right | KeyCode::Enter | KeyCode::Char(' ')
                if matches!(row, Row::Knob(_)) =>
            {
                if let Row::Knob(i) = row {
                    self.prefs.nudge(i, step);
                    if i == 4 {
                        self.panel_override = None;
                    }
                }
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
                Row::Folder(_) | Row::Knob(_) => {}
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

    /// What the selected row does.
    fn settings_help(&self, row: Row) -> String {
        match row {
            Row::Splash => "The ember animation when ferrum starts. Any key skips it; FERRUM_NO_SPLASH=1 turns it off for one run.".into(),
            Row::Search => "Where the agent's web_search tool looks things up. ←→ picks a provider.".into(),
            Row::SearchUrl => match self.search.provider.as_str() {
                "searxng" => "Your SearXNG address; its settings.yml must allow the json format. Enter edits it.".into(),
                "custom" => "A URL with {query} (and {key}); the answer must be JSON. Enter edits it.".into(),
                _ => "Only SearXNG and custom search use an address. Enter edits it.".into(),
            },
            Row::SearchKey => "Kept in ~/.config/ferrum/tui.json, readable only by you. Write $NAME to read it from an environment variable instead.".into(),
            Row::Knob(i) => Prefs::KNOBS[i].help.into(),
            Row::Folder(_) => "Searched for .gguf models and drafters, with the folders listed below. x removes it.".into(),
            Row::AddFolder => "Add a folder to search, or a single .gguf file.".into(),
            Row::Fav(_) => "A saved setup: its model and every setting. Enter renames it · x twice deletes it.".into(),
        }
    }

    pub fn draw_settings_screen(&self, f: &mut Frame, body: Rect, footer: Rect) {
        let rows = self.settings_rows();
        let sel = self.set_sel.min(rows.len() - 1);
        let selected = rows[sel];
        let section = selected.section();
        let (index, right) = layout::columns(body, 0.0, 24, 50, self.prefs.compact);
        let fields = match right {
            Some(r) => {
                let inner = layout::pane(f, index, "Settings", false, "");
                let w = inner.width as usize;
                let mut lines = Vec::new();
                for (i, name) in SECTIONS.iter().enumerate() {
                    let on = i == section;
                    lines.push(list_row(
                        on,
                        false,
                        w,
                        vec![Span::styled(
                            format!(" {name}"),
                            Style::new()
                                .fg(if on { Color::White } else { TEXT })
                                .add_modifier(if on {
                                    Modifier::BOLD
                                } else {
                                    Modifier::empty()
                                }),
                        )],
                    ));
                    lines.push(Line::default());
                }
                lines.push(Line::styled(" tab jumps between", Style::new().fg(FAINT)));
                lines.push(Line::styled(" sections", Style::new().fg(FAINT)));
                f.render_widget(Paragraph::new(lines), inner);
                r
            }
            None => index,
        };
        let title = match right {
            Some(_) => SECTIONS[section],
            None => "Settings",
        };
        let note = match section {
            2 => "changes apply as you make them",
            4 => "saved setups",
            _ => "",
        };
        let inner = layout::pane(f, fields, title, true, note);
        let inner = layout::draw_help(f, inner, &self.settings_help(selected));
        let w = inner.width as usize;
        // Side by side, just this section; on one column, all of them.
        let show_all = right.is_none();
        let mut lines: Vec<Line> = Vec::new();
        let mut sel_line = 0;
        let mut last_section = usize::MAX;
        let label_w = 28.min(w / 2);
        for (n, &row) in rows.iter().enumerate() {
            if !show_all && row.section() != section {
                continue;
            }
            if show_all && row.section() != last_section {
                if !lines.is_empty() {
                    lines.push(Line::default());
                }
                lines.push(heading(SECTIONS[row.section()]));
            }
            last_section = row.section();
            let on = n == sel;
            if on {
                sel_line = lines.len();
            }
            let bold = Style::new().fg(EMBER).add_modifier(Modifier::BOLD);
            let (label, value, style, share): (String, String, Style, Option<f32>) = match row {
                Row::Splash => (
                    "Startup animation".into(),
                    if self.splash { "on" } else { "off" }.into(),
                    bold,
                    None,
                ),
                Row::Search => ("Provider".into(), self.search.provider.clone(), bold, None),
                Row::SearchUrl => (
                    "Search URL".into(),
                    if self.search.url.is_empty() {
                        "not set".into()
                    } else {
                        self.search.url.clone()
                    },
                    Style::new().fg(TEXT),
                    None,
                ),
                Row::SearchKey => (
                    "API key".into(),
                    match self.search.key.as_str() {
                        "" => "not set".into(),
                        k if k.starts_with('$') => k.to_string(),
                        _ => "•••• saved".into(),
                    },
                    Style::new().fg(TEXT),
                    None,
                ),
                Row::Knob(i) => (
                    Prefs::KNOBS[i].label.into(),
                    self.prefs.knob_value(i),
                    bold,
                    self.prefs.knob_share(i),
                ),
                Row::Folder(i) => (
                    tilde(&expand(&self.extra[i])),
                    "x removes".into(),
                    Style::new().fg(FAINT),
                    None,
                ),
                Row::AddFolder => (
                    "＋ Add a folder or model file".into(),
                    String::new(),
                    Style::new(),
                    None,
                ),
                Row::Fav(i) => (
                    format!("★ {}", self.favs[i].name),
                    self.favs[i].mode.key().into(),
                    Style::new().fg(DIM),
                    None,
                ),
            };
            let adjustable = matches!(row, Row::Splash | Row::Search | Row::Knob(_));
            let shown = if on && adjustable {
                format!("‹ {value} ›")
            } else {
                format!("  {value}")
            };
            let mut spans = vec![
                Span::styled(
                    format!(" {}", fit_w(&label, label_w)),
                    Style::new().fg(if on { Color::White } else { TEXT }),
                ),
                Span::styled(fit_w(&shown, 22.min(w.saturating_sub(label_w + 3))), style),
            ];
            if let Some(share) = share {
                let bw = w.saturating_sub(label_w + 26).min(30);
                if bw >= 6 {
                    spans.extend(layout::bar(
                        share as f64,
                        bw,
                        if on { EMBER } else { FAINT },
                    ));
                }
            }
            lines.push(list_row(on, true, w, spans));
        }
        if !show_all && section == 3 {
            let auto: Vec<String> = roots(&[])
                .into_iter()
                .map(|(p, _)| p)
                .filter(|p| p.exists())
                .map(|p| tilde(&p))
                .collect();
            lines.push(Line::default());
            lines.extend(layout::wrapped(
                &format!("also searched: {}", auto.join("  ·  ")),
                2,
                w,
                Style::new().fg(FAINT),
            ));
        }
        if !show_all && section == 4 && self.favs.is_empty() {
            lines.push(Line::styled(
                "  none yet · save one from Chat, Agents or Serve",
                Style::new().fg(FAINT),
            ));
        }
        // About has no rows of its own: shown under Favorites' neighbour.
        if show_all || section == 4 {
            lines.push(Line::default());
            lines.push(heading("About"));
            lines.push(layout::kv(
                "config",
                tilde(&config_path()),
                Style::new().fg(DIM),
                10,
            ));
            lines.push(layout::kv(
                "models",
                format!(
                    "{} found · {} supported · {} drafters",
                    self.models.len(),
                    self.runnable(),
                    self.drafters.len()
                ),
                Style::new().fg(DIM),
                10,
            ));
            lines.push(layout::kv(
                "this Mac",
                self.machine.label(),
                Style::new().fg(DIM),
                10,
            ));
        }
        layout::show(f, inner, lines, Some(sel_line));
        let mut keys: Vec<(&str, &str)> = vec![("↑↓", "move"), ("tab", "next section")];
        match selected {
            Row::Splash | Row::Search | Row::Knob(_) => keys.push(("←→", "change")),
            Row::Folder(_) => keys.push(("x", "remove")),
            Row::Fav(_) => keys.extend([("enter", "rename"), ("x", "delete")]),
            _ => keys.push(("enter", "edit")),
        }
        keys.extend([("a", "add folder"), ("r", "rescan"), ("esc", "home")]);
        self.footer(f, footer, &keys);
    }
}
