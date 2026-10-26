//! Image: write a prompt, generate with Qwen-Image-2.1 and look at the
//! result, or edit the picture on screen with the prompt. The pipeline runs
//! on its own thread (loading one model at a time, so it needs the memory of
//! the chat model and server released first) and saves each picture as a
//! PNG.
use crate::layout::{self, pane};
use crate::preview::{self, Slot, Thumbs, Want};
use crate::serve::fmt_dur;
use crate::*;
use ferrum::{
    Error, MetalDevice,
    diffusion::pipeline::{InputImage, Observer, Pipeline, Request, Settings, Stage, Timings},
    hybrid::weights::Format,
};
use std::cell::RefCell;
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
    mpsc::{Receiver, Sender, channel},
};

/// Image sizes offered, width by height.
const SIZES: [(usize, usize); 7] = [
    (512, 512),
    (768, 768),
    (1024, 1024),
    (768, 1024),
    (1024, 768),
    (1024, 1536),
    (1536, 1024),
];

/// Weight precision choices: BF16 as stored, or quantized as they load.
const QUANTS: [(&str, Option<Format>); 3] = [
    ("bf16", None),
    ("q8_0", Some(Format::Q8_0)),
    ("q4_0", Some(Format::Q4_0)),
];

/// The rows of the settings list, top to bottom.
#[derive(Clone, Copy, PartialEq)]
enum Row {
    Prompt,
    /// A new picture, or an edit of one.
    Input,
    Model,
    Size,
    Seed,
    Transformer,
    Encoder,
}

const ROWS: [Row; 7] = [
    Row::Prompt,
    Row::Input,
    Row::Model,
    Row::Size,
    Row::Seed,
    Row::Transformer,
    Row::Encoder,
];

/// A finished picture.
pub struct Done {
    pub path: PathBuf,
    pub seed: u64,
    pub size: (usize, usize),
    pub timings: Timings,
}

enum Msg {
    Stage(Stage),
    Done(Result<Done, String>),
}

struct Run {
    rx: Receiver<Msg>,
    cancel: Arc<AtomicBool>,
    stage: Stage,
    started: Instant,
}

/// Sends each stage to the screen and stops when told to.
struct Watch {
    tx: Sender<Msg>,
    cancel: Arc<AtomicBool>,
}

impl Observer for Watch {
    fn stage(&mut self, stage: Stage) {
        let _ = self.tx.send(Msg::Stage(stage));
    }
    fn cancelled(&self) -> bool {
        self.cancel.load(Ordering::SeqCst)
    }
}

pub struct ImageTab {
    models: Vec<(String, PathBuf)>,
    model: usize,
    pub prompt: String,
    /// The picture to edit; `None` makes a new one.
    input: Option<PathBuf>,
    size: usize,
    /// `None` draws a new seed for every picture; otherwise this one.
    seed: Option<u64>,
    quant: usize,
    text_quant: usize,
    row: usize,
    run: Option<Run>,
    pub last: Option<Done>,
    error: Option<String>,
    thumbs: Thumbs,
    pub wants: RefCell<Vec<Want>>,
}

fn config_path() -> PathBuf {
    home().join(".config/ferrum/image.json")
}

/// Where pictures are saved: `$FERRUM_IMAGES`, else `~/Pictures/Ferrum`.
fn output_dir() -> PathBuf {
    std::env::var_os("FERRUM_IMAGES")
        .map(PathBuf::from)
        .unwrap_or_else(|| home().join("Pictures/Ferrum"))
}

/// A model directory is named by its Hugging Face repo (`models--A--B`
/// anywhere above it), else by its own name.
fn model_name(path: &Path) -> String {
    path.ancestors()
        .find_map(|p| {
            let name = p.file_name()?.to_string_lossy();
            name.strip_prefix("models--")
                .map(|r| r.replacen("--", "/", 1))
        })
        .or_else(|| path.file_name().map(|n| n.to_string_lossy().into_owned()))
        .unwrap_or_default()
}

fn is_qwen_image(dir: &Path) -> bool {
    fs::read_to_string(dir.join("model_index.json"))
        .is_ok_and(|s| s.contains("QwenImage21Pipeline"))
}

fn walk(dir: &Path, depth: usize, seen: &mut HashSet<PathBuf>, out: &mut Vec<(String, PathBuf)>) {
    if is_qwen_image(dir) {
        if seen.insert(dir.canonicalize().unwrap_or_else(|_| dir.to_path_buf())) {
            out.push((model_name(dir), dir.to_path_buf()));
        }
        return;
    }
    if depth == 0 {
        return;
    }
    let Ok(entries) = fs::read_dir(dir) else {
        return;
    };
    for e in entries.flatten() {
        let p = e.path();
        if p.is_dir() {
            walk(&p, depth - 1, seen, out);
        }
    }
}

/// Qwen-Image directories in the folders the model scan looks in.
fn find_models(extra: &[String]) -> Vec<(String, PathBuf)> {
    let (mut out, mut seen) = (Vec::new(), HashSet::new());
    for (root, depth) in catalog::roots(extra) {
        walk(&root, depth, &mut seen, &mut out);
    }
    out.sort();
    out
}

impl ImageTab {
    pub fn load(extra: &[String]) -> Self {
        let saved: Json = fs::read_to_string(config_path())
            .ok()
            .and_then(|s| serde_json::from_str(&s).ok())
            .unwrap_or(Json::Null);
        let models = find_models(extra);
        let at =
            |key: &str, max: usize| saved[key].as_u64().map_or(0, |v| (v as usize).min(max - 1));
        let model = saved["model"]
            .as_str()
            .and_then(|m| models.iter().position(|(_, p)| p.to_string_lossy() == m))
            .unwrap_or(0);
        Self {
            models,
            model,
            prompt: saved["prompt"].as_str().unwrap_or_default().to_string(),
            input: None,
            size: saved["size"]
                .as_u64()
                .map_or(2, |v| (v as usize).min(SIZES.len() - 1)),
            seed: None,
            quant: at("quant", QUANTS.len()),
            text_quant: at("text_quant", QUANTS.len()),
            row: 0,
            run: None,
            last: None,
            error: None,
            thumbs: Thumbs::default(),
            wants: RefCell::new(Vec::new()),
        }
    }

    pub fn running(&self) -> bool {
        self.run.is_some()
    }

    /// A word for the tab bar while a picture is being made.
    pub fn progress(&self) -> Option<String> {
        let run = self.run.as_ref()?;
        Some(match run.stage {
            Stage::Denoise { step, steps } => format!("image {step}/{steps}"),
            _ => "image …".into(),
        })
    }

    fn save(&self) {
        let value = json!({
            "model": self.models.get(self.model).map(|(_, p)| p.display().to_string()),
            "prompt": self.prompt,
            "size": self.size,
            "quant": self.quant,
            "text_quant": self.text_quant,
        });
        let path = config_path();
        if let Some(dir) = path.parent() {
            let _ = fs::create_dir_all(dir);
        }
        let _ = fs::write(path, value.to_string());
    }

    /// Take what the generation thread has reported.
    pub fn pump(&mut self) {
        self.thumbs.pump();
        let Some(run) = self.run.as_mut() else {
            return;
        };
        let mut finished = None;
        while let Ok(msg) = run.rx.try_recv() {
            match msg {
                Msg::Stage(s) => run.stage = s,
                Msg::Done(r) => finished = Some(r),
            }
        }
        if let Some(result) = finished {
            self.run = None;
            match result {
                Ok(done) => {
                    self.error = None;
                    self.last = Some(done);
                }
                Err(e) => self.error = Some(e),
            }
        }
    }

    /// The model and weights `generate_image` uses in chat: this tab's.
    pub fn tool_model(&self) -> Option<crate::imagetool::ImageModel> {
        let (_, dir) = self.models.get(self.model)?;
        Some(crate::imagetool::ImageModel {
            dir: dir.clone(),
            settings: self.settings(),
        })
    }

    fn settings(&self) -> Settings {
        Settings {
            text: QUANTS[self.text_quant].1,
            transformer: QUANTS[self.quant].1,
            transformer_file: None,
            text_file: None,
        }
    }

    fn start(&mut self) {
        let Some((_, dir)) = self.models.get(self.model).cloned() else {
            self.error = Some("no Qwen-Image model found: add its folder in Settings".into());
            return;
        };
        let mut request = Request::new(self.prompt.trim());
        match &self.input {
            // An edit keeps the picture's shape, at the chosen size's area.
            Some(path) => match fs::read(path)
                .map_err(|e| format!("{}: {e}", tilde(path)))
                .and_then(|b| InputImage::decode(&b).map_err(|e| e.to_string()))
            {
                Ok(image) => {
                    let (w, h) = SIZES[self.size];
                    request.resolution = ((w * h) as f64).sqrt().round() as usize;
                    request.images.push(image);
                }
                Err(e) => {
                    self.error = Some(e);
                    return;
                }
            },
            None => request.size = Some(SIZES[self.size]),
        }
        request.seed = self.seed.unwrap_or_else(|| {
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map_or(0, |t| t.as_nanos() as u64 & 0xffff_ffff)
        });
        let settings = self.settings();
        let (tx, rx) = channel();
        let cancel = Arc::new(AtomicBool::new(false));
        let watch = Watch {
            tx: tx.clone(),
            cancel: cancel.clone(),
        };
        std::thread::spawn(move || {
            let result = generate(&dir, settings, &request, watch);
            let _ = tx.send(Msg::Done(result));
        });
        self.error = None;
        self.run = Some(Run {
            rx,
            cancel,
            stage: Stage::LoadEncoder,
            started: Instant::now(),
        });
    }

    fn change(&mut self, forward: bool) {
        let step = |v: usize, n: usize| (v + if forward { 1 } else { n - 1 }) % n;
        match ROWS[self.row] {
            Row::Prompt => {}
            // New, or the picture on screen.
            Row::Input => {
                self.input = match (&self.input, &self.last) {
                    (None, Some(last)) => Some(last.path.clone()),
                    _ => None,
                }
            }
            Row::Model if !self.models.is_empty() => {
                self.model = step(self.model, self.models.len())
            }
            Row::Model => {}
            Row::Size => self.size = step(self.size, SIZES.len()),
            // Random, or the seed of the picture on screen.
            Row::Seed => {
                self.seed = match (self.seed, &self.last) {
                    (None, Some(last)) => Some(last.seed),
                    _ => None,
                }
            }
            Row::Transformer => self.quant = step(self.quant, QUANTS.len()),
            Row::Encoder => self.text_quant = step(self.text_quant, QUANTS.len()),
        }
        self.save();
    }
}

/// The generation thread: its own device, one picture, written to disk.
fn generate(
    dir: &Path,
    settings: Settings,
    request: &Request,
    mut watch: Watch,
) -> Result<Done, String> {
    let device = MetalDevice::new().map_err(|e| e.to_string())?;
    let pipeline = Pipeline::open(dir, settings).map_err(|e| e.to_string())?;
    let image = pipeline
        .generate(&device, request, &mut watch)
        .map_err(|e| match e {
            Error::Cancelled => "cancelled".to_string(),
            Error::Allocation(_) => "not enough free memory: pick a smaller size or q8_0".into(),
            e => e.to_string(),
        })?;
    let out = output_dir();
    fs::create_dir_all(&out).map_err(|e| format!("{}: {e}", tilde(&out)))?;
    let stamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |t| t.as_secs());
    let path = out.join(format!("ferrum-{stamp}-{}.png", image.seed));
    fs::write(&path, image.png().map_err(|e| e.to_string())?)
        .map_err(|e| format!("{}: {e}", tilde(&path)))?;
    Ok(Done {
        path,
        seed: image.seed,
        size: (image.width, image.height),
        timings: image.timings,
    })
}

pub fn stage_label(stage: Stage) -> String {
    match stage {
        Stage::EncodeImages => "reading the picture to edit".into(),
        Stage::LoadEncoder => "loading the text encoder".into(),
        Stage::Encode => "encoding the prompt".into(),
        Stage::LoadTransformer => "loading the transformer".into(),
        Stage::Denoise { step, steps } => format!("denoising, step {step}/{steps}"),
        Stage::LoadDecoder => "loading the decoder".into(),
        Stage::Decode => "decoding".into(),
    }
}

impl App {
    pub fn image_key(&mut self, key: KeyEvent) {
        let running = self.image.running();
        match key.code {
            KeyCode::Esc if running => {
                if let Some(run) = &self.image.run {
                    run.cancel.store(true, Ordering::SeqCst);
                }
                self.status = Some(("stopping after this step".into(), true));
            }
            KeyCode::Esc => self.go_tab(Tab::Home),
            KeyCode::Up | KeyCode::Char('k') => {
                self.image.row = (self.image.row + ROWS.len() - 1) % ROWS.len()
            }
            KeyCode::Down | KeyCode::Char('j') | KeyCode::Tab => {
                self.image.row = (self.image.row + 1) % ROWS.len()
            }
            KeyCode::Left | KeyCode::Char('h') => self.image.change(false),
            KeyCode::Right | KeyCode::Char('l') => self.image.change(true),
            KeyCode::Char('p') => self.edit_prompt(),
            KeyCode::Enter if ROWS[self.image.row] == Row::Prompt => self.edit_prompt(),
            KeyCode::Enter | KeyCode::Char('g') => self.generate_image(),
            KeyCode::Char('r') => {
                self.image.seed = None;
                self.status = Some(("a new seed for every picture".into(), true));
            }
            KeyCode::Char('o') => match &self.image.last {
                Some(done) => {
                    let _ = std::process::Command::new("open").arg(&done.path).spawn();
                }
                None => self.status = Some(("nothing generated yet".into(), false)),
            },
            _ => {}
        }
    }

    fn edit_prompt(&mut self) {
        self.buffer = self.image.prompt.clone();
        self.fresh = false;
        self.editing = Editing::ImagePrompt;
    }

    /// The prompt box was confirmed.
    pub fn set_image_prompt(&mut self) {
        self.image.prompt = self.buffer.trim().to_string();
        self.image.save();
    }

    fn generate_image(&mut self) {
        if self.image.running() {
            return;
        }
        if self.image.prompt.trim().is_empty() {
            self.status = Some(("write a prompt first: p".into(), false));
            return;
        }
        if !self.make_room("image generation", false, false) {
            return;
        }
        self.image.start();
    }

    pub fn draw_image(&self, f: &mut Frame, body: Rect, footer: Rect) {
        let tab = &self.image;
        let (left, right) = match layout::columns(body, 0.4, 44, 40, self.prefs.compact) {
            (l, Some(r)) => (l, Some(r)),
            (all, None) => (all, None),
        };
        // Wide: settings on the left, the picture on the right. Narrow: stacked.
        let (settings, picture) = match right {
            Some(r) => (left, r),
            None => {
                let [top, bottom] =
                    Layout::vertical([Constraint::Length(13), Constraint::Min(3)]).areas(body);
                (top, bottom)
            }
        };
        self.draw_image_settings(f, settings);
        self.draw_image_picture(f, picture);
        let mut keys = vec![
            ("g", "generate"),
            ("p", "prompt"),
            ("↑↓", "setting"),
            ("←→", "change"),
            ("r", "new seed"),
            ("o", "open"),
        ];
        if tab.running() {
            keys = vec![("esc", "stop")];
        }
        self.footer(f, footer, &keys);
    }

    fn draw_image_settings(&self, f: &mut Frame, area: Rect) {
        let tab = &self.image;
        let state = match &tab.run {
            Some(run) => format!(
                "{} {}",
                spinner(self.elapsed()),
                fmt_dur(run.started.elapsed())
            ),
            None => String::new(),
        };
        let inner = pane(f, area, "Image", true, &state);
        let w = layout::inner_w(area);
        let value = |row: Row| -> (String, Style) {
            match row {
                Row::Prompt if tab.prompt.is_empty() => {
                    ("press p to write one".into(), Style::new().fg(DIM))
                }
                Row::Prompt => (
                    clip(&tab.prompt, w.saturating_sub(12)),
                    Style::new().fg(TEXT),
                ),
                Row::Input => match &tab.input {
                    None => ("new picture".into(), Style::new().fg(TEXT)),
                    Some(path) => (
                        clip(
                            &format!(
                                "edit {}",
                                path.file_name()
                                    .map_or_else(String::new, |n| n.to_string_lossy().into_owned())
                            ),
                            w.saturating_sub(12),
                        ),
                        Style::new().fg(TEXT),
                    ),
                },
                Row::Model => match tab.models.get(tab.model) {
                    Some((name, _)) => (clip(name, w.saturating_sub(12)), Style::new().fg(TEXT)),
                    None => ("none found".into(), Style::new().fg(BAD)),
                },
                Row::Size => {
                    let (w, h) = SIZES[tab.size];
                    (format!("{w} × {h}"), Style::new().fg(TEXT))
                }
                Row::Seed => match tab.seed {
                    Some(s) => (format!("{s} (kept)"), Style::new().fg(TEXT)),
                    None => ("random".into(), Style::new().fg(TEXT)),
                },
                Row::Transformer => (QUANTS[tab.quant].0.into(), Style::new().fg(TEXT)),
                Row::Encoder => (QUANTS[tab.text_quant].0.into(), Style::new().fg(TEXT)),
            }
        };
        let label = |row: Row| match row {
            Row::Prompt => "prompt",
            Row::Input => "make",
            Row::Model => "model",
            Row::Size => "size",
            Row::Seed => "seed",
            Row::Transformer => "weights",
            Row::Encoder => "text weights",
        };
        let mut lines: Vec<Line> = Vec::new();
        for (i, row) in ROWS.iter().enumerate() {
            let (text, style) = value(*row);
            let mut line = layout::kv(label(*row), text, style, 13);
            if i == tab.row {
                line = line.style(Style::new().bg(KEYCAP));
            }
            lines.push(line);
        }
        lines.push(Line::default());
        if let Some(run) = &tab.run {
            lines.push(Line::styled(
                format!(" {}", stage_label(run.stage)),
                Style::new().fg(GOLD),
            ));
            if let Stage::Denoise { step, steps } = run.stage {
                let mut spans = vec![Span::raw(" ")];
                spans.extend(layout::bar(
                    step as f64 / steps as f64,
                    w.saturating_sub(2),
                    EMBER,
                ));
                lines.push(Line::from(spans));
            }
        } else if let Some(e) = &tab.error {
            lines.push(Line::styled(format!(" ✗ {e}"), Style::new().fg(BAD)));
        } else if let Some(d) = &tab.last {
            let t = d.timings;
            lines.push(Line::styled(
                format!(" ✓ {}×{} · seed {}", d.size.0, d.size.1, d.seed),
                Style::new().fg(GOOD),
            ));
            lines.push(Line::styled(
                format!(
                    " load {:.0}s · denoise {:.0}s · decode {:.1}s",
                    t.load.as_secs_f32(),
                    t.denoise.as_secs_f32(),
                    t.decode.as_secs_f32()
                ),
                Style::new().fg(DIM),
            ));
            lines.push(Line::styled(
                format!(" {}", clip(&tilde(&d.path), w.saturating_sub(1))),
                Style::new().fg(DIM),
            ));
        }
        f.render_widget(Paragraph::new(lines), inner);
    }

    fn draw_image_picture(&self, f: &mut Frame, area: Rect) {
        let tab = &self.image;
        tab.wants.borrow_mut().clear();
        let inner = pane(f, area, "Picture", false, "");
        let message = |f: &mut Frame, text: String, color: Color| {
            f.render_widget(
                Paragraph::new(text)
                    .style(Style::new().fg(color))
                    .alignment(Alignment::Center)
                    .wrap(Wrap { trim: false }),
                centered(inner, inner.width, 3),
            );
        };
        let Some(done) = &tab.last else {
            let hint = if tab.running() {
                "the picture appears here".to_string()
            } else {
                "press g to generate".to_string()
            };
            return message(f, hint, DIM);
        };
        match tab.thumbs.get(&done.path) {
            Slot::Ready(thumb) => {
                let (cols, rows) = thumb.size(tab.thumbs.mode, inner.width, inner.height);
                let at = centered(inner, cols, rows);
                preview::render(
                    f,
                    tab.thumbs.mode,
                    &tab.wants,
                    thumb.id,
                    &thumb.pic,
                    (at.x, at.y as i32),
                    (cols, rows),
                    at,
                );
            }
            Slot::Loading => message(f, format!("{} loading", spinner(self.elapsed())), DIM),
            Slot::Failed => message(f, format!("could not read {}", tilde(&done.path)), BAD),
        }
    }
}

#[cfg(test)]
impl ImageTab {
    /// The tab in one of four states (idle, failed, generating, done), for
    /// the screen tests.
    pub fn demo(&mut self, state: usize) {
        self.models = vec![(
            "Qwen/Qwen-Image-2.1-Turbo".into(),
            PathBuf::from("/models/qwen"),
        )];
        self.prompt = "A red fox sitting in fresh snow, golden hour light.".into();
        self.input = None;
        self.run = None;
        self.error = None;
        self.last = None;
        match state {
            1 => self.error = Some("not enough free memory: pick a smaller size or q8_0".into()),
            2 => {
                let (_tx, rx) = channel();
                self.run = Some(Run {
                    rx,
                    cancel: Arc::new(AtomicBool::new(false)),
                    stage: Stage::Denoise { step: 3, steps: 8 },
                    started: Instant::now(),
                });
            }
            3 => {
                self.last = Some(Done {
                    path: PathBuf::from("/nonexistent/fox.png"),
                    seed: 42,
                    size: (1024, 1024),
                    timings: Timings::default(),
                });
                self.input = Some(PathBuf::from("/nonexistent/fox.png"));
            }
            _ => {}
        }
    }
}

/// The prompt box: the whole prompt, wrapped, with room to read it.
pub fn prompt_popup(f: &mut Frame, buffer: &str) {
    let area = f.area();
    let w = area.width.saturating_sub(8).min(88);
    let rect = centered(area, w, 10.min(area.height));
    f.render_widget(Clear, rect);
    let block = panel("Prompt", true);
    let inner = block.inner(rect);
    f.render_widget(block, rect);
    let [text, hint] = Layout::vertical([Constraint::Min(1), Constraint::Length(1)]).areas(inner);
    // The tail of a long prompt stays in view with the cursor.
    let width = text.width.saturating_sub(2).max(1) as usize;
    let chars: Vec<char> = buffer.chars().collect();
    let rows: Vec<String> = chars
        .chunks(width)
        .map(|c| c.iter().collect::<String>())
        .collect();
    let keep = text.height as usize;
    let shown: Vec<Line> = rows
        .iter()
        .skip(rows.len().saturating_sub(keep))
        .map(|r| Line::styled(format!(" {r}"), Style::new().fg(Color::White)))
        .collect();
    f.render_widget(Paragraph::new(shown.clone()), text);
    f.render_widget(
        Paragraph::new(Line::styled(
            " enter confirm · esc cancel · ctrl-u clear",
            Style::new().fg(DIM),
        )),
        hint,
    );
    let last = rows.last().map_or(0, |r| r.chars().count());
    let y = text.y + (shown.len().max(1) - 1) as u16;
    let x = (text.x + 1 + last as u16).min(text.x + text.width.saturating_sub(1));
    f.set_cursor_position((x, y));
}
