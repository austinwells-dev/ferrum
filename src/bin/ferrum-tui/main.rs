//! Interactive launcher: home screen, setups, chat and serve in one TUI.
//!
//! ferrum-tui [MODEL.gguf]   (aliased to `ferrum`)
#![allow(unused_imports)]

mod agent;
mod app;
mod attach;
mod bench;
mod benchui;
mod brain;
mod catalog;
mod chat;
mod coding;
mod editor;
mod fields;
mod home;
mod pick;
mod serve;
mod settings;
mod side;
mod splash;
mod store;
mod text;
mod theme;
mod ui;

pub use crossterm::event::{self, Event, KeyCode, KeyEvent, KeyEventKind, KeyModifiers};
pub use ferrum::{
    hybrid::{config::HybridConfig, draft::DraftConfig},
    loader::gguf::{GgufReader, MetadataValue},
};
pub use ratatui::{
    DefaultTerminal, Frame,
    layout::{Alignment, Constraint, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, BorderType, Borders, Clear, Paragraph, Wrap},
};
pub use serde_json::{Map, Value as Json, json};
pub use std::{
    collections::HashSet,
    fs,
    path::{Path, PathBuf},
    process::Command,
    time::{Duration, Instant},
};

pub use agent::ToolMode;
pub use benchui::BenchState;
pub use catalog::*;
pub use chat::Chat;
pub use fields::*;
pub use serve::Serve;
pub use text::*;
pub use theme::*;
pub use ui::*;

pub fn config_path() -> PathBuf {
    home().join(".config/ferrum/tui.json")
}

#[derive(Clone, Copy, PartialEq)]
pub enum Screen {
    Splash,
    Home,
    Pick,
    Editor,
    Settings,
    Chat,
    Serve,
    Bench,
}

/// Why the editor is open: decides what launching does.
#[derive(Clone, PartialEq)]
pub enum Origin {
    /// A new setup, saved as a favorite when launched.
    Custom,
    /// Run once, nothing saved.
    OneTime,
    /// Changing a saved favorite.
    Edit(String),
}

#[derive(PartialEq)]
pub enum Focus {
    Models,
    Settings,
}

/// A saved setup: model, mode and every setting, drafter included.
pub struct Fav {
    pub name: String,
    pub model: String,
    pub mode: Mode,
    pub values: Map<String, Json>,
}

#[derive(Clone, Copy, PartialEq)]
pub enum Fit {
    Plain,
    Fits,
    Blocked,
}

pub struct Opt {
    pub value: String,
    pub label: String,
    /// What the option is; for blocked options, why it cannot be used.
    pub note: String,
    pub fit: Fit,
}

pub struct Picker {
    pub options: Vec<Opt>,
    pub cur: usize,
}

pub enum Editing {
    No,
    Field(usize),
    AddPath,
    FavName,
    RenameFav(usize),
}

pub struct App {
    pub screen: Screen,
    pub started: Instant,
    pub quit: bool,
    pub splash: bool,
    pub mode: Mode,
    pub origin: Origin,
    pub home_sel: usize,
    pub pick_sel: usize,
    pub set_sel: usize,
    pub focus: Focus,
    pub models: Vec<Model>,
    pub cursor: usize,
    pub fields: Vec<Field>,
    /// Index into `visible()`; one past the end is the Launch row.
    pub sel: usize,
    pub editing: Editing,
    pub buffer: String,
    /// Open the favorite-name prompt first, then start (Custom setups).
    pub pending_launch: bool,
    /// The prompt's text was prefilled: the first typed character replaces it.
    pub fresh: bool,
    pub extra: Vec<String>,
    pub drafters: Vec<Drafter>,
    pub favs: Vec<Fav>,
    pub picker: Option<Picker>,
    /// A destructive key press waiting for its second press (cleared by any other key).
    pub confirm: Option<String>,
    pub armed: Option<String>,
    pub status: Option<(String, bool)>,
    pub chat: Option<Chat>,
    pub serve: Option<Serve>,
    pub bench: Option<BenchState>,
    /// Chat workers that were told to stop and may still be unloading.
    pub closing: Vec<std::thread::JoinHandle<()>>,
}

impl App {
    pub fn elapsed(&self) -> f32 {
        self.started.elapsed().as_secs_f32()
    }

    pub fn tick(&mut self) {
        match self.screen {
            Screen::Splash if self.elapsed() >= splash::SPLASH_SECS => self.screen = Screen::Home,
            Screen::Chat => {
                if let Some(c) = self.chat.as_mut() {
                    c.pump();
                }
            }
            Screen::Serve => {
                if let Some(s) = self.serve.as_mut() {
                    s.pump();
                }
            }
            Screen::Bench => self.bench_tick(),
            _ => {}
        }
    }

    pub fn poll_interval(&self) -> Duration {
        Duration::from_millis(match self.screen {
            Screen::Splash | Screen::Home | Screen::Chat | Screen::Serve | Screen::Bench => 33,
            _ => 200,
        })
    }

    pub fn draw(&self, f: &mut Frame) {
        match self.screen {
            Screen::Splash => splash::draw_splash(f, self.elapsed()),
            Screen::Home => self.draw_home(f),
            Screen::Pick => self.draw_pick(f),
            Screen::Editor => self.draw_editor(f),
            Screen::Settings => self.draw_settings_screen(f),
            Screen::Chat => self.draw_chat(f),
            Screen::Serve => self.draw_serve(f),
            Screen::Bench => self.draw_bench(f),
        }
        if self.picker.is_some() {
            self.draw_picker(f);
        }
        match self.editing {
            Editing::AddPath => input_popup(
                f,
                "Add model file or folder",
                "a .gguf file, or a folder to scan",
                &self.buffer,
            ),
            Editing::FavName => input_popup(
                f,
                "Save as favorite",
                "name this setup (same name overwrites)",
                &self.buffer,
            ),
            Editing::RenameFav(_) => input_popup(f, "Rename favorite", "new name", &self.buffer),
            _ => {}
        }
    }

    pub fn on_key(&mut self, key: KeyEvent) {
        let ctrl = key.modifiers.contains(KeyModifiers::CONTROL);
        self.armed = self.confirm.take();
        if self.picker.is_some() {
            self.picker_key(key.code);
            return;
        }
        if !matches!(self.editing, Editing::No) {
            self.text_key(key);
            return;
        }
        if ctrl && key.code == KeyCode::Char('c') {
            match self.screen {
                Screen::Chat => self.chat_key(key),
                Screen::Serve => self.leave_serve(),
                Screen::Bench => self.leave_bench(),
                _ => self.quit = true,
            }
            return;
        }
        self.status = None;
        match self.screen {
            Screen::Splash => self.screen = Screen::Home,
            Screen::Home => self.home_key(key),
            Screen::Pick => self.pick_key(key),
            Screen::Editor => self.editor_key(key),
            Screen::Settings => self.settings_key(key),
            Screen::Chat => self.chat_key(key),
            Screen::Serve => self.serve_key(key),
            Screen::Bench => self.bench_key(key),
        }
    }

    pub fn on_paste(&mut self, text: &str) {
        let clean = text.replace("\r\n", "\n").replace('\r', "\n");
        if !matches!(self.editing, Editing::No) {
            self.buffer.push_str(&clean.replace('\n', " "));
        } else if self.screen == Screen::Chat
            && let Some(c) = self.chat.as_mut()
        {
            let dropped = attach::parse_dropped(&clean);
            if !dropped.is_empty() && c.accepts_input() {
                c.attach_paths(&dropped);
            } else {
                c.insert_str(&clean);
            }
        }
    }

    /// Typing into the one-line prompts (add path, favorite name, field values).
    fn text_key(&mut self, key: KeyEvent) {
        let ctrl = key.modifiers.contains(KeyModifiers::CONTROL);
        match key.code {
            KeyCode::Enter => {
                self.fresh = false;
                self.commit()
            }
            KeyCode::Esc => {
                self.fresh = false;
                self.editing = Editing::No;
                self.pending_launch = false;
                self.buffer.clear();
            }
            KeyCode::Backspace => {
                self.fresh = false;
                self.buffer.pop();
            }
            KeyCode::Char('u') if ctrl => self.buffer.clear(),
            KeyCode::Char('w') if ctrl => {
                let t = self.buffer.trim_end().len();
                let cut = self.buffer[..t].rfind(' ').map_or(0, |i| i + 1);
                self.buffer.truncate(cut);
            }
            KeyCode::Char('c') if ctrl => {
                self.editing = Editing::No;
                self.buffer.clear();
            }
            KeyCode::Char(c) if !ctrl => {
                if std::mem::take(&mut self.fresh) {
                    self.buffer.clear();
                }
                self.buffer.push(c)
            }
            _ => {}
        }
    }

    pub fn go_home(&mut self) {
        self.screen = Screen::Home;
    }
}

fn run(terminal: &mut DefaultTerminal, app: &mut App) -> std::io::Result<()> {
    while !app.quit {
        app.tick();
        terminal.draw(|f| app.draw(f))?;
        if event::poll(app.poll_interval())? {
            loop {
                match event::read()? {
                    Event::Key(k) if k.kind != KeyEventKind::Release => app.on_key(k),
                    Event::Paste(s) => app.on_paste(&s),
                    _ => {}
                }
                if app.quit || !event::poll(Duration::ZERO)? {
                    break;
                }
            }
        }
    }
    Ok(())
}

fn main() {
    let mut preselect = None;
    for arg in std::env::args().skip(1) {
        match arg.as_str() {
            "-h" | "--help" => {
                println!(
                    "ferrum-tui: chat with local models, run the API server, manage setups\n\n\
                     usage: ferrum-tui [MODEL.gguf]\n\n\
                     Models are found in ./, ~/models, ~/Downloads, the Hugging Face and\n\
                     LM Studio caches, and any folders in FERRUM_MODELS (colon separated).\n\
                     Set FERRUM_NO_SPLASH=1 to skip the startup animation."
                );
                return;
            }
            _ => preselect = Some(arg),
        }
    }
    let mut app = App::new(preselect.clone());
    if preselect.is_some() {
        app.open_editor(Mode::Chat, Origin::OneTime);
    }
    let mut terminal = ratatui::init();
    let _ = crossterm::execute!(std::io::stdout(), crossterm::event::EnableBracketedPaste);
    let result = run(&mut terminal, &mut app);
    let _ = crossterm::execute!(std::io::stdout(), crossterm::event::DisableBracketedPaste);
    ratatui::restore();
    app.save();
    if let Some(s) = app.serve.as_mut() {
        s.stop();
    }
    if let Err(e) = result {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
    // The model thread may still be loading; leaving the process frees it.
    std::process::exit(0);
}
