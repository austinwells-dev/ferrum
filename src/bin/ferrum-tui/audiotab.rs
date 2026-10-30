//! Audio: play a WAV clip to LFM3-Audio and read what it heard, as a
//! transcript or as the answer to a question about the clip. Clips come from
//! a path (typed or dropped on the terminal) or from the microphone through
//! ffmpeg. Each run loads the model on its own thread, streams the reply to
//! the screen and unloads it again (the pair takes about 1.5 GB).
use crate::layout::{self, pane};
use crate::serve::fmt_dur;
use crate::*;
use ferrum::{
    hybrid::{
        plan::PlanOptions,
        runtime::{ChatHooks, ChatRequest, Delta, Runtime},
    },
    vision::{self, media},
};
use std::io::Write;
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
    mpsc::{Receiver, Sender, channel},
};

/// What to ask of the clip: (label, system prompt, sends the question).
/// The ASR prompts are the ones Liquid's card uses.
const TASKS: [(&str, Option<&str>, bool); 3] = [
    (
        "transcribe (English)",
        Some("Perform ASR in English."),
        false,
    ),
    ("transcribe (any language)", Some("Perform ASR."), false),
    ("ask a question", None, true),
];

/// Context for a run: a 120 s clip is 1500 tokens, the rest is the reply.
const CONTEXT: usize = 4096;
const MAX_REPLY: usize = 1024;
/// Peak level below which a recording is taken to be silence (about -50 dBFS).
const SILENT: f32 = 0.003;

#[derive(Clone, Copy, PartialEq)]
enum Row {
    Clip,
    Task,
    Question,
    Model,
}

const ROWS: [Row; 4] = [Row::Clip, Row::Task, Row::Question, Row::Model];

/// A finished reply's numbers.
pub struct Stats {
    seconds: f64,
    encode: Duration,
    prompt: usize,
    tokens: usize,
    decode: Duration,
}

enum Msg {
    Loaded,
    Text(String),
    Done(Result<Stats, String>),
}

struct Run {
    rx: Receiver<Msg>,
    cancel: Arc<AtomicBool>,
    loaded: bool,
    started: Instant,
}

/// ffmpeg writing the default microphone to a WAV file.
struct Recording {
    child: std::process::Child,
    path: PathBuf,
    /// When samples started reaching the file: opening the microphone takes
    /// ffmpeg a few seconds, and nothing said before then is kept.
    live: Option<Instant>,
    /// When `q` was sent; ffmpeg is killed if it has not finished soon after.
    stopping: Option<Instant>,
}

impl Drop for Recording {
    /// Quitting mid-recording must not leave ffmpeg listening.
    fn drop(&mut self) {
        if let Ok(None) = self.child.try_wait() {
            let _ = self.child.kill();
            let _ = self.child.wait();
        }
    }
}

pub struct AudioTab {
    pub clip: Option<PathBuf>,
    task: usize,
    pub question: String,
    /// The language model file; its projector comes from the model scan.
    model: Option<PathBuf>,
    row: usize,
    run: Option<Run>,
    recording: Option<Recording>,
    /// The reply so far, or the last one.
    pub text: String,
    stats: Option<Stats>,
    error: Option<String>,
}

fn config_path() -> PathBuf {
    home().join(".config/ferrum/audio.json")
}

/// Where recordings go: `$FERRUM_AUDIO`, else `~/Music/Ferrum`.
fn output_dir() -> PathBuf {
    std::env::var_os("FERRUM_AUDIO")
        .map(PathBuf::from)
        .unwrap_or_else(|| home().join("Music/Ferrum"))
}

/// Models the scan found with a speech projector next to them.
fn audio_models(models: &[Model]) -> Vec<&Model> {
    models
        .iter()
        .filter(|m| m.audio && m.problem.is_none() && m.vision.is_some())
        .collect()
}

impl AudioTab {
    pub fn load() -> Self {
        let saved: Json = fs::read_to_string(config_path())
            .ok()
            .and_then(|s| serde_json::from_str(&s).ok())
            .unwrap_or(Json::Null);
        Self {
            clip: saved["clip"].as_str().map(PathBuf::from),
            task: saved["task"]
                .as_u64()
                .map_or(0, |v| (v as usize).min(TASKS.len() - 1)),
            question: saved["question"].as_str().unwrap_or_default().to_string(),
            model: saved["model"].as_str().map(PathBuf::from),
            row: 0,
            run: None,
            recording: None,
            text: String::new(),
            stats: None,
            error: None,
        }
    }

    fn save(&self) {
        let value = json!({
            "clip": self.clip.as_ref().map(|p| p.display().to_string()),
            "task": self.task,
            "question": self.question,
            "model": self.model.as_ref().map(|p| p.display().to_string()),
        });
        let path = config_path();
        if let Some(dir) = path.parent() {
            let _ = fs::create_dir_all(dir);
        }
        let _ = fs::write(path, value.to_string());
    }

    pub fn running(&self) -> bool {
        self.run.is_some() || self.recording.is_some()
    }

    /// A word for the tab button while something is going on.
    pub fn progress(&self) -> Option<String> {
        if let Some(r) = &self.recording {
            return Some(match r.live {
                Some(t) => format!("recording {}", fmt_dur(t.elapsed())),
                None => "recording".into(),
            });
        }
        self.run.as_ref().map(|_| "listening".into())
    }

    /// Take what the run has reported, and stop a recording at its limit.
    pub fn pump(&mut self) {
        if let Some(r) = self.recording.as_mut() {
            if r.live.is_none() && fs::metadata(&r.path).is_ok_and(|m| m.len() > 44) {
                r.live = Some(Instant::now());
            }
            if r.stopping
                .is_some_and(|t| t.elapsed() > Duration::from_secs(3))
            {
                let _ = r.child.kill();
            }
            // Stopped, or ffmpeg ended by itself (the time limit, or an error).
            if let Ok(Some(_)) = r.child.try_wait() {
                let path = r.path.clone();
                self.recording = None;
                self.finish_recording(path);
            }
        }
        let Some(run) = self.run.as_mut() else {
            return;
        };
        let mut finished = None;
        while let Ok(msg) = run.rx.try_recv() {
            match msg {
                Msg::Loaded => run.loaded = true,
                Msg::Text(t) => self.text.push_str(&t),
                Msg::Done(r) => finished = Some(r),
            }
        }
        if let Some(result) = finished {
            self.run = None;
            match result {
                Ok(stats) => self.stats = Some(stats),
                Err(e) => self.error = Some(e),
            }
        }
    }

    /// The model the next run uses: the saved one if the scan still has it,
    /// else the first.
    fn pick<'a>(&self, models: &[&'a Model]) -> Option<&'a Model> {
        models
            .iter()
            .find(|m| Some(&m.path) == self.model.as_ref())
            .or(models.first())
            .copied()
    }

    fn set_clip(&mut self, path: PathBuf) {
        self.clip = Some(path);
        self.text.clear();
        self.stats = None;
        self.error = None;
        self.save();
    }

    fn start(&mut self, model: &Model) -> Result<(), String> {
        let clip = self
            .clip
            .clone()
            .ok_or("choose a clip first: enter on the clip row, or r to record")?;
        let bytes = fs::read(&clip).map_err(|e| format!("{}: {e}", tilde(&clip)))?;
        // Refuse a clip that is too long or not a WAV before loading anything.
        let seconds = vision::clip_seconds(&bytes).map_err(|e| e.to_string())?;
        let (_, system, asks) = TASKS[self.task];
        let question = self.question.trim().to_string();
        if asks && question.is_empty() {
            return Err("write the question first: enter on the question row".into());
        }
        let mut content = vec![media::audio_part_from_bytes(&bytes)];
        if asks {
            content.push(json!({"type": "text", "text": question}));
        }
        let mut messages = Vec::new();
        if let Some(system) = system {
            messages.push(json!({"role": "system", "content": system}));
        }
        messages.push(json!({"role": "user", "content": content}));
        let (path, mmproj) = (model.path.clone(), model.vision.clone());
        let (tx, rx) = channel();
        let cancel = Arc::new(AtomicBool::new(false));
        let stop = cancel.clone();
        std::thread::spawn(move || {
            let result = listen(&path, mmproj.as_deref(), messages, seconds, &tx, stop);
            let _ = tx.send(Msg::Done(result));
        });
        self.text.clear();
        self.stats = None;
        self.error = None;
        self.run = Some(Run {
            rx,
            cancel,
            loaded: false,
            started: Instant::now(),
        });
        Ok(())
    }

    fn record(&mut self) -> Result<(), String> {
        let ffmpeg = preview::which("ffmpeg")
            .ok_or("recording needs ffmpeg (brew install ffmpeg); or drop a WAV file here")?;
        let dir = output_dir();
        fs::create_dir_all(&dir).map_err(|e| format!("{}: {e}", tilde(&dir)))?;
        let stamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |t| t.as_secs());
        let path = dir.join(format!("ferrum-{stamp}.wav"));
        // 16 kHz mono 16-bit is what the encoder reads, so nothing is resampled.
        let child = std::process::Command::new(ffmpeg)
            // The system's default input: device 0 can be a virtual one
            // (BlackHole, Teams) that only ever gives silence.
            .args(["-hide_banner", "-loglevel", "error", "-f", "avfoundation"])
            .args([
                "-i",
                ":default",
                "-t",
                &vision::MAX_AUDIO_SECONDS.to_string(),
            ])
            .args(["-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le"])
            // Write as it records, so the screen can tell when audio arrives.
            .args(["-flush_packets", "1", "-y"])
            .arg(&path)
            .stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .spawn()
            .map_err(|e| format!("ffmpeg: {e}"))?;
        self.recording = Some(Recording {
            child,
            path,
            live: None,
            stopping: None,
        });
        Ok(())
    }

    /// Ask ffmpeg to finish the file (`q` on its input); `pump` sees it end.
    fn stop_recording(&mut self) {
        if let Some(r) = self.recording.as_mut()
            && r.stopping.is_none()
        {
            if let Some(stdin) = r.child.stdin.as_mut() {
                let _ = stdin.write_all(b"q").and_then(|_| stdin.flush());
            }
            r.stopping = Some(Instant::now());
        }
    }

    fn finish_recording(&mut self, path: PathBuf) {
        let samples = fs::read(&path)
            .ok()
            .and_then(|b| vision::audio::decode_wav(&b).ok())
            .unwrap_or_default();
        // A muted input or one the terminal may not use gives zeros or a
        // faint hiss, and the model would make up an answer from it.
        let peak = samples.iter().fold(0f32, |m, v| m.max(v.abs()));
        if peak < SILENT {
            self.error = Some(format!(
                "the recording is silent: check the input in System Settings › Sound, and that your terminal may use the microphone (Privacy & Security › Microphone). It is kept at {}",
                tilde(&path)
            ));
            return;
        }
        self.set_clip(path);
    }

    fn change(&mut self, forward: bool, models: &[&Model]) {
        let step = |v: usize, n: usize| (v + if forward { 1 } else { n - 1 }) % n;
        match ROWS[self.row] {
            Row::Task => self.task = step(self.task, TASKS.len()),
            Row::Model if !models.is_empty() => {
                let at = models
                    .iter()
                    .position(|m| Some(&m.path) == self.model.as_ref())
                    .unwrap_or(0);
                self.model = Some(models[step(at, models.len())].path.clone());
            }
            _ => return,
        }
        self.save();
    }
}

/// Sends the reply to the screen and stops when told to.
struct Hooks<'a> {
    tx: &'a Sender<Msg>,
    cancel: Arc<AtomicBool>,
}

impl ChatHooks for Hooks<'_> {
    fn delta(&mut self, delta: Delta) -> ferrum::Result<()> {
        if let Delta::Content(t) = delta {
            let _ = self.tx.send(Msg::Text(t));
        }
        Ok(())
    }
    fn prefill(&mut self, _processed: usize, _total: usize) -> bool {
        !self.cancel.load(Ordering::SeqCst)
    }
    fn cancelled(&self) -> bool {
        self.cancel.load(Ordering::SeqCst)
    }
}

/// The run's thread: load the pair, play it the clip, stream the reply.
fn listen(
    model: &Path,
    mmproj: Option<&Path>,
    messages: Vec<Json>,
    seconds: f64,
    tx: &Sender<Msg>,
    cancel: Arc<AtomicBool>,
) -> Result<Stats, String> {
    let options = PlanOptions {
        context: Some(CONTEXT),
        ..Default::default()
    };
    let mut rt =
        Runtime::load_multimodal(model, options, None, mmproj, 0).map_err(|e| e.to_string())?;
    if !rt.supports_audio() {
        return Err(format!("{} has no speech projector", tilde(model)));
    }
    let _ = tx.send(Msg::Loaded);
    let mut request = ChatRequest::new(messages, rt.default_sampling.clone());
    request.max_tokens = MAX_REPLY;
    let result = rt
        .chat(&request, Hooks { tx, cancel })
        .map_err(|e| e.to_string())?;
    let c = &result.completion;
    Ok(Stats {
        seconds,
        encode: rt.audio.as_ref().map_or(Duration::ZERO, |a| a.last_encode),
        prompt: c.prompt_tokens,
        tokens: c.tokens.len(),
        decode: c.decode,
    })
}

impl App {
    pub fn audio_key(&mut self, key: KeyEvent) {
        let models = audio_models(&self.models);
        let tab = &mut self.audio;
        if tab.recording.is_some() {
            match key.code {
                KeyCode::Esc | KeyCode::Enter | KeyCode::Char('r' | ' ') => tab.stop_recording(),
                _ => {}
            }
            return;
        }
        match key.code {
            KeyCode::Esc if tab.run.is_some() => {
                if let Some(run) = &tab.run {
                    run.cancel.store(true, Ordering::SeqCst);
                }
                self.status = Some(("stopping".into(), true));
            }
            KeyCode::Esc => self.go_tab(Tab::Home),
            KeyCode::Up | KeyCode::Char('k') => tab.row = (tab.row + ROWS.len() - 1) % ROWS.len(),
            KeyCode::Down | KeyCode::Char('j') | KeyCode::Tab => {
                tab.row = (tab.row + 1) % ROWS.len()
            }
            KeyCode::Left | KeyCode::Char('h') => tab.change(false, &models),
            KeyCode::Right | KeyCode::Char('l') => tab.change(true, &models),
            KeyCode::Enter if ROWS[tab.row] == Row::Clip => self.edit_audio(Editing::AudioClip),
            KeyCode::Enter if ROWS[tab.row] == Row::Question => {
                self.edit_audio(Editing::AudioQuestion)
            }
            KeyCode::Char('f') => self.edit_audio(Editing::AudioClip),
            KeyCode::Char('q') => self.edit_audio(Editing::AudioQuestion),
            KeyCode::Char('r') if tab.run.is_none() => {
                if let Err(e) = tab.record() {
                    self.status = Some((e, false));
                }
            }
            KeyCode::Enter | KeyCode::Char('g') if tab.run.is_none() => {
                let Some(model) = tab.pick(&models) else {
                    self.status = Some((
                        "no LFM3-Audio model found: put it and its mmproj file in a model folder"
                            .into(),
                        false,
                    ));
                    return;
                };
                if let Err(e) = tab.start(model) {
                    self.status = Some((e, false));
                }
            }
            KeyCode::Char('c') if !tab.text.is_empty() => {
                let copied = std::process::Command::new("pbcopy")
                    .stdin(std::process::Stdio::piped())
                    .spawn()
                    .and_then(|mut child| {
                        child
                            .stdin
                            .take()
                            .map(|mut s| s.write_all(tab.text.trim().as_bytes()))
                            .transpose()?;
                        child.wait()
                    });
                self.status = Some(match copied {
                    Ok(_) => ("copied".into(), true),
                    Err(e) => (format!("pbcopy: {e}"), false),
                });
            }
            KeyCode::Char('o') => match &tab.clip {
                Some(clip) => {
                    let _ = std::process::Command::new("open").arg(clip).spawn();
                }
                None => self.status = Some(("no clip yet".into(), false)),
            },
            _ => {}
        }
    }

    fn edit_audio(&mut self, editing: Editing) {
        self.buffer = match editing {
            Editing::AudioClip => self.audio.clip.as_deref().map(tilde).unwrap_or_default(),
            _ => self.audio.question.clone(),
        };
        self.fresh = true;
        self.editing = editing;
    }

    /// The clip or question box was confirmed.
    pub fn set_audio_text(&mut self) {
        match self.editing {
            Editing::AudioClip => {
                let text = self.buffer.trim();
                if text.is_empty() {
                    return;
                }
                let path = attach::parse_dropped(text)
                    .into_iter()
                    .next()
                    .unwrap_or_else(|| expand(text));
                if path.is_file() {
                    self.audio.set_clip(path);
                } else {
                    self.status = Some((format!("no file at {}", tilde(&path)), false));
                }
            }
            _ => {
                self.audio.question = self.buffer.trim().to_string();
                self.audio.save();
            }
        }
    }

    /// A file dropped on the terminal while the tab is open becomes the clip.
    pub fn audio_paste(&mut self, text: &str) {
        match attach::parse_dropped(text)
            .into_iter()
            .find(|p| p.is_file())
        {
            Some(path) => self.audio.set_clip(path),
            None => self.status = Some(("drop a .wav file to use it".into(), false)),
        }
    }

    pub fn draw_audio(&self, f: &mut Frame, body: Rect, footer: Rect) {
        let tab = &self.audio;
        let (settings, reply) = match layout::columns(body, 0.4, 44, 40, self.prefs.compact) {
            (l, Some(r)) => (l, r),
            (all, None) => {
                let [top, bottom] =
                    Layout::vertical([Constraint::Length(12), Constraint::Min(3)]).areas(all);
                (top, bottom)
            }
        };
        self.draw_audio_settings(f, settings);
        self.draw_audio_reply(f, reply);
        let keys = if tab.recording.is_some() {
            vec![("r", "stop recording")]
        } else if tab.run.is_some() {
            vec![("esc", "stop")]
        } else {
            vec![
                ("g", "listen"),
                ("f", "file"),
                ("r", "record"),
                ("q", "question"),
                ("←→", "change"),
                ("c", "copy"),
                ("o", "play"),
            ]
        };
        self.footer(f, footer, &keys);
    }

    fn draw_audio_settings(&self, f: &mut Frame, area: Rect) {
        let tab = &self.audio;
        let models = audio_models(&self.models);
        let state = match (&tab.recording, &tab.run) {
            (Some(Recording { live: Some(t), .. }), _) => format!("● rec {}", fmt_dur(t.elapsed())),
            (Some(_), _) => format!("{}", spinner(self.elapsed())),
            (_, Some(run)) => format!(
                "{} {}",
                spinner(self.elapsed()),
                fmt_dur(run.started.elapsed())
            ),
            _ => String::new(),
        };
        let inner = pane(f, area, "Audio", true, &state);
        let w = layout::inner_w(area).saturating_sub(12);
        let text = Style::new().fg(TEXT);
        let dim = Style::new().fg(DIM);
        let value = |row: Row| -> (String, Style) {
            match row {
                Row::Clip => match &tab.clip {
                    Some(p) if tilde(p).chars().count() <= w => (tilde(p), text),
                    Some(p) => (
                        clip(
                            &format!(
                                "…/{}",
                                p.file_name()
                                    .map_or_else(String::new, |n| n.to_string_lossy().into_owned())
                            ),
                            w,
                        ),
                        text,
                    ),
                    None => ("enter a path, drop a file or r".into(), dim),
                },
                Row::Task => (TASKS[tab.task].0.into(), text),
                Row::Question if !TASKS[tab.task].2 => ("(only when asking)".into(), dim),
                Row::Question if tab.question.is_empty() => ("enter to write one".into(), dim),
                Row::Question => (clip(&tab.question, w), text),
                Row::Model => match tab.pick(&models) {
                    Some(m) => (clip(&m.name, w), text),
                    None => ("no LFM3-Audio found".into(), Style::new().fg(BAD)),
                },
            }
        };
        let label = |row: Row| match row {
            Row::Clip => "clip",
            Row::Task => "task",
            Row::Question => "question",
            Row::Model => "model",
        };
        let mut lines: Vec<Line> = Vec::new();
        for (i, row) in ROWS.iter().enumerate() {
            let (v, style) = value(*row);
            let mut line = layout::kv(label(*row), v, style, 11);
            if i == tab.row {
                line = line.style(Style::new().bg(KEYCAP));
            }
            lines.push(line);
        }
        lines.push(Line::default());
        if let Some(r) = &tab.recording {
            let what = match (r.stopping, r.live) {
                (Some(_), _) => "finishing the recording".to_string(),
                (None, Some(_)) => format!(
                    "recording from the microphone (up to {} s); r stops",
                    vision::MAX_AUDIO_SECONDS
                ),
                (None, None) => "opening the microphone: wait to speak".into(),
            };
            lines.push(Line::styled(format!(" {what}"), Style::new().fg(EMBER)));
        } else if let Some(run) = &tab.run {
            let what = if run.loaded {
                "listening"
            } else {
                "loading the model"
            };
            lines.push(Line::styled(format!(" {what}"), Style::new().fg(GOLD)));
        } else if let Some(e) = &tab.error {
            lines.push(Line::styled(format!(" ✗ {e}"), Style::new().fg(BAD)));
        } else if let Some(s) = &tab.stats {
            lines.push(Line::styled(
                format!(
                    " ✓ {:.1} s of audio · {} prompt tokens",
                    s.seconds, s.prompt
                ),
                Style::new().fg(GOOD),
            ));
            lines.push(Line::styled(
                format!(
                    " encode {:.2} s · {} tokens at {:.0} tok/s",
                    s.encode.as_secs_f64(),
                    s.tokens,
                    s.tokens as f64 / s.decode.as_secs_f64().max(1e-9)
                ),
                dim,
            ));
        } else {
            lines.push(Line::styled(
                format!(" WAV clips up to {} s.", vision::MAX_AUDIO_SECONDS),
                dim,
            ));
            lines.push(Line::styled(
                " LFM3-Audio hears them; it does not speak.",
                dim,
            ));
        }
        f.render_widget(Paragraph::new(lines).wrap(Wrap { trim: false }), inner);
    }

    fn draw_audio_reply(&self, f: &mut Frame, area: Rect) {
        let tab = &self.audio;
        let title = if TASKS[tab.task].2 {
            "Answer"
        } else {
            "Transcript"
        };
        let inner = pane(f, area, title, false, "");
        if tab.text.is_empty() {
            let hint = if tab.run.is_some() {
                "the reply appears here"
            } else {
                "press g to listen"
            };
            f.render_widget(
                Paragraph::new(hint)
                    .style(Style::new().fg(DIM))
                    .alignment(Alignment::Center),
                centered(inner, inner.width, 1),
            );
            return;
        }
        // The end of a long reply stays in view while it streams.
        let text = tab.text.trim();
        let width = inner.width.saturating_sub(2).max(1) as usize;
        let chars: Vec<Sc> = text.chars().map(|c| (c, Style::new().fg(TEXT))).collect();
        let rows = wrap_styled(&chars, width, 0);
        let lines: Vec<Line> = rows
            .iter()
            .skip(rows.len().saturating_sub(inner.height as usize))
            .map(|row| to_line(row, vec![Span::raw(" ")]))
            .collect();
        f.render_widget(Paragraph::new(lines), inner);
    }
}

#[cfg(test)]
impl AudioTab {
    /// The tab idle, failed, listening and done, for the screen tests.
    pub fn demo(&mut self, state: usize) {
        self.clip = Some(PathBuf::from("/nonexistent/weather.wav"));
        self.question = "What city is mentioned?".into();
        self.run = None;
        self.recording = None;
        self.error = None;
        self.stats = None;
        self.text.clear();
        match state {
            1 => {
                self.error = Some(
                    "a 247.3 s clip is longer than the 120 s one clip may last; split it".into(),
                )
            }
            2 => {
                let (_tx, rx) = channel();
                self.run = Some(Run {
                    rx,
                    cancel: Arc::new(AtomicBool::new(false)),
                    loaded: true,
                    started: Instant::now(),
                });
                self.text = "Tomorrow will be mostly".into();
            }
            3 => {
                self.text = "Tomorrow will be mostly sunny in Chicago, with a high of 72 degrees and light winds from the west.".into();
                self.stats = Some(Stats {
                    seconds: 6.0,
                    encode: Duration::from_millis(30),
                    prompt: 95,
                    tokens: 23,
                    decode: Duration::from_millis(250),
                });
            }
            _ => {}
        }
    }
}
