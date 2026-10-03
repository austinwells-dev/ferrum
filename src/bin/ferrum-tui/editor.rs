use crate::*;

impl App {
    pub fn visible(&self) -> Vec<usize> {
        (0..self.fields.len())
            .filter(|&i| self.fields[i].scope.applies(self.mode))
            .collect()
    }

    pub fn current_field(&self) -> Option<usize> {
        self.visible().get(self.sel).copied()
    }

    pub fn open_picker(&mut self) {
        let model = self.selected();
        let opt = |value: &str, label: &str, note: &str, fit: Fit| Opt {
            value: value.into(),
            label: label.into(),
            note: note.into(),
            fit,
        };
        let mut options = vec![opt("off", "Off", "no speculation", Fit::Plain)];
        options.push(match model {
            Some(m) if m.mtp => opt("mtp", "MTP", "head built into the GGUF", Fit::Plain),
            Some(m) => opt("mtp", "MTP", &m.mtp_why, Fit::Blocked),
            None => opt("mtp", "MTP", "no model selected", Fit::Blocked),
        });
        let (mut fits, mut blocked) = (Vec::new(), Vec::new());
        for d in &self.drafters {
            let why = match model {
                Some(m) => drafter_problem(d, m),
                None => Some("no model selected".into()),
            };
            let path = d.path.display().to_string();
            match why {
                None => fits.push(opt(&path, &d.label, d.kind, Fit::Fits)),
                Some(why) => blocked.push(opt(&path, &d.label, &why, Fit::Blocked)),
            }
        }
        options.extend(fits);
        options.push(opt("", "Custom path…", "type a directory", Fit::Plain));
        options.extend(blocked);
        let current = self.value("draft");
        let cur = options
            .iter()
            .position(|o| !o.value.is_empty() && o.value == current && o.fit != Fit::Blocked)
            .unwrap_or(0);
        self.picker = Some(Picker { options, cur });
    }

    pub fn picker_key(&mut self, code: KeyCode) {
        let Some(p) = &mut self.picker else {
            return;
        };
        let open = |p: &Picker, i: usize| p.options[i].fit != Fit::Blocked;
        match code {
            KeyCode::Up | KeyCode::Char('k') => {
                if let Some(i) = (0..p.cur).rev().find(|&i| open(p, i)) {
                    p.cur = i;
                }
            }
            KeyCode::Down | KeyCode::Char('j') => {
                if let Some(i) = (p.cur + 1..p.options.len()).find(|&i| open(p, i)) {
                    p.cur = i;
                }
            }
            KeyCode::Esc | KeyCode::Char('q') => self.picker = None,
            KeyCode::Enter => {
                let value = p.options[p.cur].value.clone();
                self.picker = None;
                let Some(i) = self.fields.iter().position(|f| f.key == "draft") else {
                    return;
                };
                if value.is_empty() {
                    self.buffer = String::new();
                    self.editing = Editing::Field(i);
                } else {
                    self.fields[i].value = value;
                }
            }
            _ => {}
        }
    }

    pub fn step(&mut self, idx: usize, dir: i32) {
        if self.fields[idx].key == "draft" {
            let opts = self.draft_values();
            let cur = match self.fields[idx].value.as_str() {
                "" => "off",
                v => v,
            };
            let at = opts.iter().position(|o| o == cur).unwrap_or(0) as i32;
            self.fields[idx].value =
                opts[(at + dir).rem_euclid(opts.len() as i32) as usize].clone();
            return;
        }
        let f = &mut self.fields[idx];
        match &f.kind {
            Kind::Cycle(opts) => {
                let at = opts.iter().position(|o| *o == f.value).unwrap_or(0) as i32;
                let next = (at + dir).rem_euclid(opts.len() as i32) as usize;
                f.value = opts[next].to_string();
            }
            Kind::Num {
                step,
                min,
                max,
                base,
                int,
            } => {
                let cur = f.value.parse::<f64>().unwrap_or(*base);
                let next = (cur + step * dir as f64).clamp(*min, *max);
                f.value = fmt_num(next, *int);
            }
            Kind::Text { presets, .. } => {
                if presets.is_empty() {
                    return;
                }
                let at = presets.iter().position(|o| *o == f.value);
                let next = match at {
                    Some(i) => (i as i32 + dir).rem_euclid(presets.len() as i32) as usize,
                    None => 0,
                };
                f.value = presets[next].to_string();
            }
        }
    }

    pub fn commit(&mut self) {
        match std::mem::replace(&mut self.editing, Editing::No) {
            Editing::Field(i) => {
                let raw = self.buffer.trim().to_string();
                let f = &mut self.fields[i];
                match &f.kind {
                    Kind::Num { min, max, int, .. } => {
                        if raw.is_empty() {
                            f.value.clear();
                        } else if let Ok(n) = raw.parse::<f64>() {
                            f.value = fmt_num(n.clamp(*min, *max), *int);
                        } else {
                            self.status = Some((format!("{raw:?} is not a number"), false));
                        }
                    }
                    _ => f.value = raw,
                }
            }
            Editing::AddPath => {
                let raw = self
                    .buffer
                    .trim()
                    .trim_matches('\'')
                    .trim_matches('"')
                    .to_string();
                if raw.is_empty() {
                    return;
                }
                let path = expand(&raw);
                if !path.exists() {
                    self.status = Some((format!("{} does not exist", tilde(&path)), false));
                } else {
                    self.extra.push(path.display().to_string());
                    self.rescan(Some(&path));
                }
            }
            Editing::FavName => {
                let name = self.buffer.trim().to_string();
                let launch = std::mem::take(&mut self.pending_launch);
                if !name.is_empty() && self.save_favorite(&name) && launch {
                    self.origin = Origin::Edit(name);
                    self.buffer.clear();
                    self.start();
                    return;
                }
            }
            Editing::RenameFav(i) => {
                let name = self.buffer.trim().to_string();
                if name.is_empty() {
                    // keep the old name
                } else if self
                    .favs
                    .iter()
                    .enumerate()
                    .any(|(j, f)| j != i && f.name == name)
                {
                    self.status = Some((format!("{name:?} already exists"), false));
                } else if let Some(f) = self.favs.get_mut(i) {
                    f.name = name;
                    self.save();
                }
            }
            Editing::No => {}
        }
        self.buffer.clear();
    }

    /// Start from the editor; Custom setups are named (and saved) first.
    pub fn launch(&mut self) {
        if self.selected().is_none() {
            return self.start();
        }
        match self.origin.clone() {
            Origin::OneTime => self.start(),
            Origin::Edit(name) => {
                if self.save_favorite(&name) {
                    self.start();
                }
            }
            Origin::Custom => {
                self.pending_launch = true;
                self.buffer = self.selected().map(|m| m.name.clone()).unwrap_or_default();
                self.fresh = true;
                self.editing = Editing::FavName;
            }
        }
    }

    pub fn editor_key(&mut self, key: KeyEvent) {
        match key.code {
            KeyCode::Char('q') | KeyCode::Esc => {
                self.save();
                self.screen = Screen::Pick;
            }
            KeyCode::Char('l') | KeyCode::F(5) => self.launch(),
            KeyCode::Char('a') => {
                self.editing = Editing::AddPath;
                self.buffer.clear();
            }
            KeyCode::Char('r') => self.rescan(None),
            KeyCode::Char('f') => {
                self.buffer = match &self.origin {
                    Origin::Edit(name) => name.clone(),
                    _ => self
                        .models
                        .get(self.cursor)
                        .map(|m| m.name.clone())
                        .unwrap_or_default(),
                };
                self.fresh = true;
                self.editing = Editing::FavName;
            }
            KeyCode::Tab | KeyCode::BackTab => {
                self.focus = if self.focus == Focus::Models {
                    Focus::Settings
                } else {
                    Focus::Models
                };
            }
            code => match self.focus {
                Focus::Models => match code {
                    KeyCode::Up | KeyCode::Char('k') => {
                        self.cursor = self.cursor.saturating_sub(1);
                        self.load_info();
                    }
                    KeyCode::Down | KeyCode::Char('j') => {
                        self.cursor = (self.cursor + 1).min(self.models.len().saturating_sub(1));
                        self.load_info();
                    }
                    KeyCode::Enter | KeyCode::Right => self.focus = Focus::Settings,
                    _ => {}
                },
                Focus::Settings => {
                    let rows = self.visible().len();
                    let on_launch = self.sel >= rows;
                    match code {
                        KeyCode::Up | KeyCode::Char('k') => self.sel = self.sel.saturating_sub(1),
                        KeyCode::Down | KeyCode::Char('j') => self.sel = (self.sel + 1).min(rows),
                        KeyCode::Left | KeyCode::Char('h') => match self.current_field() {
                            Some(i) => self.step(i, -1),
                            None => self.focus = Focus::Models,
                        },
                        KeyCode::Right => {
                            if let Some(i) = self.current_field() {
                                self.step(i, 1)
                            }
                        }
                        KeyCode::Backspace | KeyCode::Delete | KeyCode::Char('d') => {
                            if let Some(i) = self.current_field() {
                                self.fields[i].value.clear();
                            }
                        }
                        KeyCode::Enter if on_launch => self.launch(),
                        KeyCode::Enter | KeyCode::Char(' ') => {
                            if let Some(i) = self.current_field() {
                                match self.fields[i].kind {
                                    _ if self.fields[i].key == "draft" => self.open_picker(),
                                    Kind::Cycle(_) => self.step(i, 1),
                                    _ => {
                                        self.buffer = self.fields[i].value.clone();
                                        self.editing = Editing::Field(i);
                                    }
                                }
                            }
                        }
                        _ => {}
                    }
                }
            },
        }
    }

    // ---- drawing ----

    pub fn draw_editor(&self, f: &mut Frame) {
        let [head, body, command, footer] = Layout::vertical([
            Constraint::Length(2),
            Constraint::Min(10),
            Constraint::Length(7),
            Constraint::Length(1),
        ])
        .areas(f.area());
        let what = match &self.origin {
            Origin::Custom => "new setup".to_string(),
            Origin::OneTime => "one-time run".to_string(),
            Origin::Edit(name) => format!("edit {name}"),
        };
        header(
            f,
            head,
            &[self.mode.label(), &what],
            vec![pill(self.mode.pill(), EMBER)],
        );
        let [left, right] =
            Layout::horizontal([Constraint::Percentage(40), Constraint::Percentage(60)])
                .areas(body);
        let [list, details] =
            Layout::vertical([Constraint::Min(6), Constraint::Length(8)]).areas(left);
        self.draw_models(f, list);
        self.draw_details(f, details);
        self.draw_settings(f, right);
        self.draw_command(f, command);
        self.draw_footer(f, footer);
    }

    pub fn panel(&self, title: &str, focused: bool) -> Block<'static> {
        panel(title, focused)
    }

    pub fn draw_models(&self, f: &mut Frame, area: Rect) {
        let focused = self.focus == Focus::Models;
        let unsupported = self.models.len() - self.runnable();
        let title = if unsupported > 0 {
            format!("Models ({} · {unsupported} unsupported)", self.runnable())
        } else {
            format!("Models ({})", self.runnable())
        };
        let block = self.panel(&title, focused);
        let inner = block.inner(area);
        f.render_widget(block, area);
        if self.models.is_empty() {
            f.render_widget(
                Paragraph::new(vec![
                    Line::default(),
                    Line::styled("  No .gguf files found.", Style::new().fg(TEXT)),
                    Line::styled("  Press a to add a file or folder,", Style::new().fg(DIM)),
                    Line::styled("  or set FERRUM_MODELS=dir:dir.", Style::new().fg(DIM)),
                ]),
                inner,
            );
            return;
        }
        let per_page = (inner.height as usize / 2).max(1);
        let top = (self.cursor + 1).saturating_sub(per_page);
        let width = inner.width as usize;
        let mut lines: Vec<Line> = Vec::new();
        for (i, m) in self.models.iter().enumerate().skip(top).take(per_page) {
            let on = i == self.cursor;
            let bad = m.problem.is_some();
            let bg = if on { SELECTED } else { Color::Reset };
            let size = gib(m.size);
            let quant = quant_of(&m.name).unwrap_or_default();
            let tail = if quant.is_empty() {
                size.clone()
            } else {
                format!("{quant}  {size}")
            };
            let name_w = width.saturating_sub(tail.chars().count() + 4);
            lines.push(Line::from(vec![
                Span::styled(if on { " ▌" } else { "  " }, Style::new().fg(EMBER).bg(bg)),
                Span::styled(
                    format!("{:<name_w$}", clip(&m.name, name_w)),
                    Style::new()
                        .fg(if bad {
                            DIM
                        } else if on {
                            Color::White
                        } else {
                            TEXT
                        })
                        .bg(bg)
                        .add_modifier(if on && !bad {
                            Modifier::BOLD
                        } else {
                            Modifier::empty()
                        }),
                ),
                Span::styled(format!(" {tail} "), Style::new().fg(DIM).bg(bg)),
            ]));
            lines.push(Line::from(vec![
                Span::styled(
                    if on { " ▌" } else { "  " },
                    Style::new().fg(if bad { BAD } else { EMBER }).bg(bg),
                ),
                Span::styled(
                    format!(
                        "{:<w$}",
                        clip(
                            &match &m.problem {
                                Some(why) => format!("✗ {why}"),
                                None => m.origin.clone(),
                            },
                            width.saturating_sub(4)
                        ),
                        w = width.saturating_sub(4)
                    ),
                    Style::new().fg(if bad { BAD } else { FAINT }).bg(bg),
                ),
                Span::styled("  ", Style::new().bg(bg)),
            ]));
        }
        f.render_widget(Paragraph::new(lines), inner);
    }

    pub fn draw_picker(&self, f: &mut Frame) {
        let Some(p) = &self.picker else {
            return;
        };
        let area = f.area();
        let w = area.width.saturating_sub(8).min(76);
        let h = (p.options.len() as u16 + 4).min(area.height.saturating_sub(2));
        let rect = Rect::new(
            area.x + (area.width - w) / 2,
            area.y + (area.height - h) / 2,
            w,
            h,
        );
        f.render_widget(Clear, rect);
        let block = self.panel("Drafter", true);
        let inner = block.inner(rect);
        f.render_widget(block, rect);
        let rows = inner.height.saturating_sub(1) as usize;
        let top = (p.cur + 1).saturating_sub(rows);
        let width = inner.width as usize;
        let mut lines: Vec<Line> = p
            .options
            .iter()
            .enumerate()
            .skip(top)
            .take(rows)
            .map(|(i, o)| {
                let on = i == p.cur;
                let bg = if on { SELECTED } else { Color::Reset };
                let blocked = o.fit == Fit::Blocked;
                let mark = if o.fit == Fit::Fits {
                    "✓ fits model"
                } else {
                    ""
                };
                let note_w =
                    width.saturating_sub(o.label.chars().count() + mark.chars().count() + 8);
                let label_w = width
                    .saturating_sub(note_w.min(o.note.chars().count()) + mark.chars().count() + 7);
                Line::from(vec![
                    Span::styled(
                        if on { " ▌ " } else { "   " },
                        Style::new().fg(EMBER).bg(bg),
                    ),
                    Span::styled(
                        format!("{:<label_w$}", clip(&o.label, label_w)),
                        Style::new()
                            .fg(if blocked {
                                FAINT
                            } else if on {
                                Color::White
                            } else {
                                TEXT
                            })
                            .bg(bg),
                    ),
                    Span::styled(
                        format!(" {} ", clip(&o.note, note_w.max(10))),
                        Style::new().fg(if blocked { BAD } else { DIM }).bg(bg),
                    ),
                    Span::styled(format!("{mark} "), Style::new().fg(GOOD).bg(bg)),
                ])
            })
            .collect();
        lines.push(Line::styled(
            " ↑↓ choose · enter select · esc cancel",
            Style::new().fg(FAINT),
        ));
        f.render_widget(Paragraph::new(lines), inner);
    }

    pub fn draw_details(&self, f: &mut Frame, area: Rect) {
        let block = self.panel("Details", false);
        let inner = block.inner(area);
        f.render_widget(block, area);
        let Some(m) = self.models.get(self.cursor) else {
            return;
        };
        let info = m.info.as_ref();
        let row = |k: &str, v: String| {
            Line::from(vec![
                Span::styled(format!(" {k:<9}"), Style::new().fg(DIM)),
                Span::styled(v, Style::new().fg(TEXT)),
            ])
        };
        let w = (inner.width as usize).saturating_sub(11);
        let mut lines = vec![row(
            "name",
            clip(
                info.and_then(|i| i.name.clone())
                    .as_deref()
                    .unwrap_or(&m.name),
                w,
            ),
        )];
        match info {
            Some(i) => {
                let mut arch = i.arch.clone();
                if let Some(l) = i.layers {
                    arch += &format!(" · {l} layers");
                }
                if let Some(e) = i.experts {
                    arch += &format!(" · {e} experts");
                }
                lines.push(row("arch", clip(&arch, w)));
                if let Some(c) = i.train_ctx {
                    lines.push(row("trained", format!("{c} tokens")));
                }
                if let Some(v) = &m.vision {
                    let file = v.file_name().map(|n| n.to_string_lossy().into_owned());
                    lines.push(row(
                        "vision",
                        clip(&format!("👁 {}", file.unwrap_or_default()), w),
                    ));
                }
                lines.push(row("sampling", clip(&i.sampling, w)));
            }
            None => lines.push(row("arch", "unreadable GGUF header".into())),
        }
        lines.push(match &m.problem {
            None => row("size", format!("{} · ✓ supported", gib(m.size))),
            Some(why) => Line::from(vec![
                Span::styled(" status   ", Style::new().fg(DIM)),
                Span::styled(clip(&format!("✗ {why}"), w), Style::new().fg(BAD)),
            ]),
        });
        lines.push(row("path", clip(&tilde(&m.path), w)));
        f.render_widget(Paragraph::new(lines), inner);
    }

    pub fn draw_settings(&self, f: &mut Frame, area: Rect) {
        let focused = self.focus == Focus::Settings;
        let title = match self.mode {
            Mode::Chat => "Chat settings",
            Mode::Serve => "Server settings",
            Mode::Agent => "Agent settings",
        };
        let block = self.panel(title, focused);
        let inner = block.inner(area);
        f.render_widget(block, area);
        let visible = self.visible();
        // (line, is the selected row)
        let mut rows: Vec<(Line, bool)> = Vec::new();
        let mut group = "";
        let width = inner.width as usize;
        let mut selected_row = 0;
        for (n, &i) in visible.iter().enumerate() {
            let fl = &self.fields[i];
            if fl.group != group {
                group = fl.group;
                if !rows.is_empty() {
                    rows.push((Line::default(), false));
                }
                rows.push((
                    Line::from(vec![Span::styled(
                        format!("  {}", group.to_uppercase()),
                        Style::new().fg(DIM).add_modifier(Modifier::BOLD),
                    )]),
                    false,
                ));
            }
            let on = focused && n == self.sel;
            if n == self.sel {
                selected_row = rows.len();
            }
            rows.push((self.field_line(i, on, width), on));
        }
        rows.push((Line::default(), false));
        let launch_on = self.sel >= visible.len();
        if launch_on {
            selected_row = rows.len();
        }
        let label = match (self.mode, &self.origin) {
            (Mode::Chat, Origin::Custom) => "  ▶  Name & start chat",
            (Mode::Serve, Origin::Custom) => "  ▶  Name & start server",
            (Mode::Agent, Origin::Custom) => "  ▶  Name & start agent",
            (Mode::Chat, _) => "  ▶  Start chat",
            (Mode::Serve, _) => "  ▶  Start server",
            (Mode::Agent, _) => "  ▶  Start agent",
        };
        let style = if launch_on && focused {
            Style::new()
                .fg(Color::Black)
                .bg(EMBER)
                .add_modifier(Modifier::BOLD)
        } else {
            Style::new().fg(EMBER).add_modifier(Modifier::BOLD)
        };
        rows.push((
            Line::styled(format!("{label:<w$}", w = width.max(label.len())), style),
            false,
        ));
        let height = inner.height as usize;
        let top = (selected_row + 2).saturating_sub(height);
        let lines: Vec<Line> = rows
            .into_iter()
            .skip(top)
            .take(height)
            .map(|(l, _)| l)
            .collect();
        f.render_widget(Paragraph::new(lines), inner);
    }

    pub fn field_line(&self, i: usize, on: bool, width: usize) -> Line<'static> {
        let fl = &self.fields[i];
        let bg = if on { SELECTED } else { Color::Reset };
        let editing = matches!(self.editing, Editing::Field(e) if e == i);
        let (shown, set) = if editing {
            (format!("{}▏", self.buffer), true)
        } else if fl.value.is_empty() {
            (fl.default.to_string(), false)
        } else if fl.key == "draft" {
            (self.draft_label(&fl.value), true)
        } else if matches!(fl.kind, Kind::Text { secret: true, .. }) {
            ("•".repeat(fl.value.chars().count().min(16)), true)
        } else {
            (fl.value.clone(), true)
        };
        let adjustable = on && !editing && !matches!(fl.kind, Kind::Text { presets: [], .. });
        let value_style = if editing {
            Style::new().fg(Color::White).add_modifier(Modifier::BOLD)
        } else if set {
            Style::new().fg(EMBER).add_modifier(Modifier::BOLD)
        } else {
            Style::new().fg(DIM)
        };
        let label_w = 22;
        let room = width.saturating_sub(label_w + 8);
        let mut spans = vec![
            Span::styled(
                if on { " ▌ " } else { "   " },
                Style::new().fg(EMBER).bg(bg),
            ),
            Span::styled(
                format!("{:<label_w$}", fl.label),
                Style::new().fg(if on { Color::White } else { TEXT }).bg(bg),
            ),
            Span::styled(
                if adjustable { "‹ " } else { "  " },
                Style::new().fg(GOLD).bg(bg),
            ),
            Span::styled(clip(&shown, room), value_style.bg(bg)),
            Span::styled(
                if adjustable { " ›" } else { "  " },
                Style::new().fg(GOLD).bg(bg),
            ),
        ];
        let used: usize = spans.iter().map(|s| s.content.chars().count()).sum();
        spans.push(Span::styled(
            " ".repeat(width.saturating_sub(used)),
            Style::new().bg(bg),
        ));
        Line::from(spans)
    }

    pub fn draw_command(&self, f: &mut Frame, area: Rect) {
        let block = self.panel("Command", false);
        let inner = block.inner(area);
        f.render_widget(block, area);
        let (bin, args) = self.command();
        let mut cmd = vec![
            Span::styled(" $ ", Style::new().fg(EMBER)),
            Span::styled(bin, Style::new().fg(GOOD).add_modifier(Modifier::BOLD)),
        ];
        for a in &args {
            let shown = if a.starts_with('-') {
                Span::styled(format!(" {a}"), Style::new().fg(GOLD))
            } else {
                Span::styled(format!(" {}", shell_quote(a)), Style::new().fg(TEXT))
            };
            cmd.push(shown);
        }
        let hint = self
            .current_field()
            .filter(|_| self.focus == Focus::Settings)
            .map(|i| self.fields[i].help)
            .unwrap_or("Pick a model, adjust settings, then launch.");
        let note = match self
            .models
            .get(self.cursor)
            .and_then(|m| m.problem.as_ref())
        {
            Some(why) => Line::styled(
                format!(" ✗ can't run this model: {why}"),
                Style::new().fg(BAD),
            ),
            None => Line::styled(format!(" {hint}"), Style::new().fg(DIM)),
        };
        f.render_widget(
            Paragraph::new(vec![Line::from(cmd), note]).wrap(Wrap { trim: false }),
            inner,
        );
    }

    pub fn draw_footer(&self, f: &mut Frame, area: Rect) {
        let keys: &[(&str, &str)] = if matches!(self.editing, Editing::No) {
            &[
                ("↑↓", "move"),
                ("←→", "adjust"),
                ("enter", "edit"),
                ("⌫", "reset"),
                ("tab", "pane"),
                ("f", "save favorite"),
                ("a", "add path"),
                ("l", "launch"),
                ("esc", "back"),
            ]
        } else {
            &[("enter", "confirm"), ("esc", "cancel")]
        };
        keys_footer(f, area, keys, &self.status);
    }
}
