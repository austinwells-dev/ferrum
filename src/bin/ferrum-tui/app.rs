use crate::*;

impl App {
    pub fn new(preselect: Option<String>) -> Self {
        let saved: Json = fs::read_to_string(config_path())
            .ok()
            .and_then(|s| serde_json::from_str(&s).ok())
            .unwrap_or(Json::Null);
        let mut fields = fields();
        for f in &mut fields {
            if let Some(v) = saved["values"][f.key].as_str() {
                f.value = v.to_string();
            }
        }
        let mut extra: Vec<String> = saved["extra"]
            .as_array()
            .map(|a| {
                a.iter()
                    .filter_map(|v| v.as_str().map(String::from))
                    .collect()
            })
            .unwrap_or_default();
        if let Some(p) = &preselect {
            extra.push(p.clone());
        }
        let models = scan(&extra);
        let want = preselect
            .as_deref()
            .map(expand)
            .or_else(|| saved["model"].as_str().map(PathBuf::from));
        let cursor = want
            .and_then(|w| {
                models
                    .iter()
                    .position(|m| m.path == w && m.problem.is_none())
            })
            .unwrap_or(0);
        let drafters = scan_drafters(&extra);
        let favs = saved["favorites"]
            .as_array()
            .map(|a| {
                a.iter()
                    .map(|v| Fav {
                        name: v["name"].as_str().unwrap_or("favorite").to_string(),
                        model: v["model"].as_str().unwrap_or("").to_string(),
                        mode: Mode::parse(v["mode"].as_str().unwrap_or("chat")),
                        values: v["values"].as_object().cloned().unwrap_or_default(),
                    })
                    .collect()
            })
            .unwrap_or_default();
        let splash = saved["splash"].as_bool().unwrap_or(true);
        let skip = std::env::var_os("FERRUM_NO_SPLASH").is_some();
        let mut app = App {
            screen: if splash && !skip {
                Screen::Splash
            } else {
                Screen::Home
            },
            started: Instant::now(),
            quit: false,
            splash,
            search: search::Config::new(
                saved["search"]["provider"].as_str().unwrap_or("duckduckgo"),
                saved["search"]["url"].as_str().unwrap_or(""),
                saved["search"]["key"].as_str().unwrap_or(""),
            ),
            mode: Mode::Chat,
            origin: Origin::OneTime,
            home_sel: 0,
            pick_sel: 0,
            set_sel: 0,
            focus: Focus::Models,
            models,
            cursor,
            fields,
            sel: 0,
            editing: Editing::No,
            buffer: String::new(),
            pending_launch: false,
            fresh: false,
            extra,
            drafters,
            favs,
            picker: None,
            confirm: None,
            armed: None,
            status: None,
            chat: None,
            serve: None,
            bench: None,
            closing: Vec::new(),
            gfx: Default::default(),
            prefs: Prefs::load(&saved["layout"]),
            dash: Default::default(),
            overlay: Overlay::None,
            chat_box: true,
            panel_override: None,
            size: Default::default(),
            machine: bench::Machine::detect(),
            serve_gib: 0.0,
            keys_shown: Default::default(),
            bench_pane: 0,
            models_sel: 0,
            models_sort: models::Sort::Speed,
            models_detail: false,
        };
        app.load_info();
        app
    }

    pub fn values_map(&self) -> Map<String, Json> {
        let mut values = Map::new();
        for f in self.fields.iter().filter(|f| !f.value.is_empty()) {
            values.insert(f.key.into(), json!(f.value));
        }
        values
    }

    pub fn save(&self) {
        let config = json!({
            "splash": self.splash,
            "search": {"provider": self.search.provider, "url": self.search.url, "key": self.search.key},
            "model": self.selected().map(|m| m.path.display().to_string()),
            "values": self.values_map(),
            "extra": self.extra,
            "layout": self.prefs.to_json(),
            "favorites": self.favs.iter().map(|f| json!({
                "name": f.name,
                "model": f.model,
                "mode": f.mode.key(),
                "values": f.values,
            })).collect::<Vec<_>>(),
        });
        let _ = write_private(
            &config_path(),
            &serde_json::to_string_pretty(&config).unwrap_or_default(),
        );
    }

    /// The model under the cursor, if ferrum can run it.
    pub fn selected(&self) -> Option<&Model> {
        self.models.get(self.cursor).filter(|m| m.problem.is_none())
    }

    pub fn runnable(&self) -> usize {
        self.models.iter().filter(|m| m.problem.is_none()).count()
    }

    /// Drop a drafter the newly selected model cannot use.
    pub fn load_info(&mut self) {
        let Some(m) = self.selected() else {
            return;
        };
        let current = self.value("draft").to_string();
        let reason = match current.as_str() {
            "" | "off" => None,
            "mtp" => (!m.mtp).then(|| m.mtp_why.clone()),
            "lookup" => None,
            path => self
                .drafters
                .iter()
                .find(|d| d.path.display().to_string() == path)
                .and_then(|d| drafter_problem(d, m)),
        };
        if let Some(why) = reason {
            if let Some(f) = self.fields.iter_mut().find(|f| f.key == "draft") {
                f.value = "off".into();
            }
            self.status = Some((format!("drafter turned off: {why}"), false));
        }
    }

    /// The K/V cache format setting (FERRUM_KV_CACHE or f16 when unset).
    pub fn kv_cache(&self) -> Result<KvCache, String> {
        match self.value("kv_cache") {
            "" => Ok(KvCache::from_env()),
            v => v.parse().map_err(|e: ferrum::Error| e.to_string()),
        }
    }

    pub fn value(&self, key: &str) -> &str {
        self.fields
            .iter()
            .find(|f| f.key == key)
            .map(|f| f.value.as_str())
            .unwrap_or("")
    }

    /// The binary and arguments for the current selection.
    pub fn command(&self) -> (&'static str, Vec<String>) {
        self.command_for(self.mode, self.selected().map(|m| m.path.as_path()), &|k| {
            self.value(k).to_string()
        })
    }

    /// The command line for any set of values (the editor's, or a favorite's).
    pub fn command_for(
        &self,
        mode: Mode,
        model: Option<&Path>,
        value: &dyn Fn(&str) -> String,
    ) -> (&'static str, Vec<String>) {
        let serve = mode == Mode::Serve;
        let mut args: Vec<String> = Vec::new();
        if let Some(m) = model {
            args.extend(["--model".into(), m.display().to_string()]);
        }
        let draft = value("draft");
        let drafting = !draft.is_empty() && draft != "off";
        for f in &self.fields {
            let in_scope = f.scope.applies(mode);
            let v = value(f.key);
            let is_default = match &f.kind {
                Kind::Cycle(opts) => v.is_empty() || v == opts[0],
                _ => v.is_empty(),
            };
            if !in_scope || is_default {
                continue;
            }
            let flag = match f.key {
                "context" => "--context",
                "kv_cache" => "--kv-cache",
                "max_tokens" => "--max-tokens",
                "effort" => "--reasoning-effort",
                "budget" => "--reasoning-budget",
                "temperature" => "--temperature",
                "top_p" => "--top-p",
                "top_k" => "--top-k",
                "min_p" => "--min-p",
                "presence" => "--presence-penalty",
                "frequency" => "--frequency-penalty",
                "repeat" if serve => "--repeat-penalty",
                "repeat" => "--repetition-penalty",
                "system" => "--system",
                "host" => "--host",
                "port" => "--port",
                // Passed in the environment (Serve::start): any process,
                // the agent's sandbox included, can read another's arguments.
                "api_key" => continue,
                "alias" => "--alias",
                "reserve" => "--reserve-mib",
                "snapshots" => "--snapshots",
                "chunk" => "--chunk",
                "draft" if drafting => "--draft",
                "draft_max" if drafting => "--draft-max",
                "draft_quant" if drafting => "--draft-quant",
                "think" => {
                    args.push("--no-think".into());
                    continue;
                }
                "show" => {
                    args.push("--hide-thinking".into());
                    continue;
                }
                "verbose" => {
                    args.push("-v".into());
                    continue;
                }
                _ => continue,
            };
            args.push(flag.into());
            args.push(v);
        }
        (if serve { "ferrum-server" } else { "ferrum-cli" }, args)
    }

    /// The detected drafters, with GGUF drafters in `m`'s own folder first.
    pub fn drafters_for(&self, m: Option<&Model>) -> Vec<&Drafter> {
        let mut v: Vec<&Drafter> = self.drafters.iter().collect();
        if let Some(m) = m {
            v.sort_by_key(|d| !d.is_sibling(m));
        }
        v
    }

    pub fn draft_values(&self) -> Vec<String> {
        let mut v = vec!["off".to_string()];
        let model = self.selected();
        if model.is_some_and(|m| m.mtp) {
            v.push("mtp".to_string());
        }
        v.push("lookup".to_string());
        v.extend(
            self.drafters_for(model)
                .into_iter()
                .filter(|d| model.is_some_and(|m| drafter_problem(d, m).is_none()))
                .map(|d| d.path.display().to_string()),
        );
        v
    }

    pub fn draft_label(&self, value: &str) -> String {
        if value == "lookup" {
            return "Prompt lookup".into();
        }
        match self
            .drafters
            .iter()
            .find(|d| d.path.display().to_string() == value)
        {
            Some(d) => format!("{} ({})", d.label, d.kind),
            None => match value.rsplit('/').next() {
                Some(last) if value.contains('/') => format!("…/{last}"),
                _ => value.to_string(),
            },
        }
    }

    pub fn save_favorite(&mut self, name: &str) -> bool {
        let Some(model) = self.selected() else {
            self.status = Some(("pick a compatible model first".into(), false));
            return false;
        };
        let fav = Fav {
            name: name.to_string(),
            model: model.path.display().to_string(),
            mode: self.mode,
            values: self.values_map(),
        };
        match self.favs.iter().position(|f| f.name == name) {
            Some(i) => self.favs[i] = fav,
            None => self.favs.push(fav),
        }
        self.status = Some((format!("saved favorite {name:?}"), true));
        self.save();
        true
    }

    /// Apply a favorite's mode, settings and model. False when its model
    /// is gone or can no longer run (the settings are applied regardless).
    pub fn load_favorite(&mut self, i: usize) -> bool {
        let Some(fav) = self.favs.get(i) else {
            return false;
        };
        for f in &mut self.fields {
            f.value = fav
                .values
                .get(f.key)
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();
        }
        self.mode = fav.mode;
        let name = fav.name.clone();
        let found = self
            .models
            .iter()
            .position(|m| m.path.display().to_string() == fav.model);
        self.sel = 0;
        match found {
            Some(at) if self.models[at].problem.is_some() => {
                let why = self.models[at].problem.clone().unwrap_or_default();
                self.status = Some((format!("{name:?}: model unsupported ({why})"), false));
                false
            }
            Some(at) => {
                self.cursor = at;
                self.load_info();
                true
            }
            None => {
                self.status = Some((format!("{name:?}: model file not found"), false));
                false
            }
        }
    }

    pub fn rescan(&mut self, focus: Option<&Path>) {
        let keep = self.selected().map(|m| m.path.clone());
        self.models = scan(&self.extra);
        self.drafters = scan_drafters(&self.extra);
        let target = focus
            .and_then(|p| {
                self.models
                    .iter()
                    .position(|m| m.path == p || m.path.starts_with(p))
            })
            .or_else(|| keep.and_then(|k| self.models.iter().position(|m| m.path == k)));
        self.cursor = target
            .filter(|&t| self.models[t].problem.is_none())
            .unwrap_or(0);
        self.status = Some((
            format!(
                "{} models, {} supported",
                self.models.len(),
                self.runnable()
            ),
            true,
        ));
        self.load_info();
    }

    pub fn open_editor(&mut self, mode: Mode, origin: Origin) {
        self.mode = mode;
        if origin == Origin::Custom {
            for f in &mut self.fields {
                f.value.clear();
            }
        }
        self.origin = origin;
        self.screen = Screen::Editor;
        self.focus = Focus::Models;
        self.sel = 0;
    }

    /// Wait for chat workers that were told to stop, so their GPU memory is free.
    pub fn reap(&mut self) {
        for h in self.closing.drain(..) {
            let _ = h.join();
        }
    }

    /// Start chat or serve with the current model and settings.
    pub fn start(&mut self) {
        if self.selected().is_none() {
            let why = match self.models.get(self.cursor) {
                Some(m) => format!(
                    "{} can't run: {}",
                    m.name,
                    m.problem.clone().unwrap_or_default()
                ),
                None => "no model found: press a to add a path".into(),
            };
            self.status = Some((why, false));
            return;
        }
        let what = match self.mode {
            Mode::Chat => "this chat",
            Mode::Agent => "this agent",
            Mode::Serve => "this server",
        };
        if !self.make_room(what, false, false) {
            return;
        }
        self.save();
        self.reap();
        let size = self.selected().map_or(0, |m| m.size);
        let started = match self.mode {
            Mode::Chat | Mode::Agent => Chat::start(self).map(|c| {
                self.chat = Some(c);
                self.chat_box = true;
                Screen::Chat
            }),
            Mode::Serve => Serve::start(self).map(|s| {
                self.serve = Some(s);
                self.serve_gib = size as f64 / (1u64 << 30) as f64;
                Screen::Serve
            }),
        };
        match started {
            Ok(screen) => self.screen = screen,
            Err(e) => self.status = Some((e, false)),
        }
    }
}
