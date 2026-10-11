//! Image and video previews for the chat: thumbnails inline in the
//! transcript and a full-screen viewer.
//!
//! Terminals that speak the kitty graphics protocol (kitty, Ghostty, WezTerm)
//! get the real pixels; everywhere else the picture is drawn with `▀` half
//! blocks in 24-bit colour. Stills and GIFs are decoded with the `image`
//! crate (HEIC and TIFF through `sips`). Videos play through `ffmpeg` when it
//! is installed, and otherwise show the Quick Look poster frame.
use crate::*;
use image::{AnimationDecoder, DynamicImage, ImageEncoder, RgbaImage};
use std::cell::{Cell, RefCell};
use std::collections::HashMap;
use std::io::{BufReader, Read, Write};
use std::rc::Rc;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc::{Receiver, Sender, SyncSender, TryRecvError, channel, sync_channel};

pub const VIDEO_EXTS: [&str; 6] = ["mp4", "mov", "m4v", "webm", "mkv", "avi"];
/// Kitty image id of the full-screen viewer; thumbnails count up from `THUMB_IDS`.
const VIEWER_ID: u32 = 0x0FE7;
const THUMB_IDS: u32 = 0x1000;
/// Longest side of a picture sent to the terminal (viewer, thumbnails).
const MAX_SIDE: u32 = 1600;
const THUMB_SIDE: u32 = 800;
/// Video playback: frames per second and widest frame decoded.
const VIDEO_FPS: u32 = 15;
const VIDEO_WIDTH: u32 = 960;
static SEQ: AtomicU64 = AtomicU64::new(1);

fn next_seq() -> u64 {
    SEQ.fetch_add(1, Ordering::Relaxed)
}

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Graphics {
    Kitty,
    Blocks,
}

/// Which way pictures are drawn in this terminal. `FERRUM_GRAPHICS=kitty|blocks` overrides.
pub fn graphics() -> Graphics {
    let var = |k: &str| std::env::var(k).unwrap_or_default();
    match var("FERRUM_GRAPHICS").as_str() {
        "kitty" => return Graphics::Kitty,
        "blocks" | "off" => return Graphics::Blocks,
        _ => {}
    }
    // tmux and screen swallow the protocol unless passthrough is set up.
    if !var("TMUX").is_empty() || var("TERM").starts_with("screen") {
        return Graphics::Blocks;
    }
    let term = var("TERM");
    let program = var("TERM_PROGRAM").to_lowercase();
    if !var("KITTY_WINDOW_ID").is_empty()
        || term.contains("kitty")
        || term.contains("ghostty")
        || !var("GHOSTTY_RESOURCES_DIR").is_empty()
        || !var("WEZTERM_EXECUTABLE").is_empty()
        || matches!(program.as_str(), "ghostty" | "wezterm")
    {
        Graphics::Kitty
    } else {
        Graphics::Blocks
    }
}

fn ext(path: &Path) -> String {
    path.extension()
        .map(|e| e.to_string_lossy().to_lowercase())
        .unwrap_or_default()
}

pub fn is_video(path: &Path) -> bool {
    VIDEO_EXTS.contains(&ext(path).as_str())
}

/// Files the preview can show.
pub fn previewable(path: &Path) -> bool {
    crate::attach::is_image(path) || is_video(path)
}

/// A decoded picture, ready to draw either way.
pub struct Picture {
    img: RgbaImage,
    /// Base64 PNG for the kitty protocol (only in kitty mode).
    kitty: Option<Arc<str>>,
    /// Changes whenever the pixels do.
    seq: u64,
    /// Half-block render cached for (cols, rows).
    blocks: RefCell<Option<(u16, u16, RgbaImage)>>,
}

impl Picture {
    fn new(img: RgbaImage, mode: Graphics) -> Picture {
        let kitty = (mode == Graphics::Kitty)
            .then(|| png_b64(&img))
            .flatten()
            .map(Arc::from);
        Picture::with(img, kitty)
    }

    fn with(img: RgbaImage, kitty: Option<Arc<str>>) -> Picture {
        Picture {
            img,
            kitty,
            seq: next_seq(),
            blocks: RefCell::new(None),
        }
    }
}

/// One image placement wanted this frame (kitty mode).
pub struct Want {
    id: u32,
    seq: u64,
    data: Arc<str>,
    at: Rect,
    /// Rows of the source image shown, in its pixels: (top, height).
    crop: (u32, u32),
}

/// Draw `pic` scaled to `cols`×`rows` cells with its top-left at (`x`, `y`),
/// showing only the part inside `clip` (so it can scroll off an edge).
#[allow(clippy::too_many_arguments)]
pub fn render(
    f: &mut Frame,
    mode: Graphics,
    wants: &RefCell<Vec<Want>>,
    id: u32,
    pic: &Picture,
    (x, y): (u16, i32),
    (cols, rows): (u16, u16),
    clip: Rect,
) {
    let top = y.max(clip.y as i32);
    let bottom = (y + rows as i32).min(clip.y as i32 + clip.height as i32);
    if top >= bottom || cols == 0 {
        return;
    }
    let skip = (top - y) as u32;
    let vis = (bottom - top) as u32;
    match mode {
        Graphics::Kitty => {
            let Some(data) = &pic.kitty else { return };
            let h = pic.img.height();
            let (y0, y1) = (skip * h / rows as u32, (skip + vis) * h / rows as u32);
            wants.borrow_mut().push(Want {
                id,
                seq: pic.seq,
                data: data.clone(),
                at: Rect::new(x, top as u16, cols, vis as u16),
                crop: (y0, (y1 - y0).max(1)),
            });
        }
        Graphics::Blocks => {
            let mut cache = pic.blocks.borrow_mut();
            if !matches!(&*cache, Some((c, r, _)) if *c == cols && *r == rows) {
                let small = image::imageops::resize(
                    &pic.img,
                    cols as u32,
                    rows as u32 * 2,
                    image::imageops::FilterType::Triangle,
                );
                *cache = Some((cols, rows, small));
            }
            let Some((_, _, small)) = &*cache else { return };
            let buf = f.buffer_mut();
            for r in 0..vis {
                let sy = (skip + r) * 2;
                for c in 0..cols as u32 {
                    let upper = rgb(small.get_pixel(c, sy));
                    let lower = rgb(small.get_pixel(c, sy + 1));
                    if let Some(cell) = buf.cell_mut((x + c as u16, top as u16 + r as u16)) {
                        cell.set_char('▀').set_fg(upper).set_bg(lower);
                    }
                }
            }
        }
    }
}

fn rgb(p: &image::Rgba<u8>) -> Color {
    // Transparent pixels fade into a dark background.
    let a = p[3] as u16;
    let m = |c: u8| ((c as u16 * a + 18 * (255 - a)) / 255) as u8;
    Color::Rgb(m(p[0]), m(p[1]), m(p[2]))
}

/// Size of one terminal cell in pixels (a 1:2 guess when the terminal won't say).
fn cell_pixels() -> (f32, f32) {
    match crossterm::terminal::window_size() {
        Ok(s) if s.width > 0 && s.height > 0 && s.columns > 0 && s.rows > 0 => (
            s.width as f32 / s.columns as f32,
            s.height as f32 / s.rows as f32,
        ),
        _ => (8.0, 16.0),
    }
}

/// Cells for a w×h picture inside `cols`×`rows`, keeping its shape and never
/// enlarging it past its own size.
fn cells(mode: Graphics, (w, h): (u32, u32), cols: u16, rows: u16) -> (u16, u16) {
    let (cw, ch) = cell_pixels();
    // Half blocks: two pixels per cell, one above the other.
    let ch = if mode == Graphics::Blocks {
        ch / 2.0
    } else {
        ch
    };
    let (w, h) = (w.max(1) as f32, h.max(1) as f32);
    let scale = (cols as f32 * cw / w).min(rows as f32 * ch / h).min(1.0);
    (
        ((w * scale / cw).round() as u16).clamp(1, cols.max(1)),
        ((h * scale / ch).round() as u16).clamp(1, rows.max(1)),
    )
}

// ---- thumbnails ----

pub struct Thumb {
    pub id: u32,
    pub pic: Picture,
    /// Size of the original, in pixels.
    pub orig: (u32, u32),
    pub video: bool,
}

impl Thumb {
    /// Cells it takes when allowed up to `cols`×`rows`.
    pub fn size(&self, mode: Graphics, cols: u16, rows: u16) -> (u16, u16) {
        cells(mode, self.orig, cols, rows)
    }
}

#[derive(Clone)]
pub enum Slot {
    Loading,
    Ready(Rc<Thumb>),
    Failed,
}

/// A finished thumbnail: (path, Some((pixels, original size))).
type Done = (PathBuf, Option<(RgbaImage, (u32, u32))>);

/// Thumbnails, decoded in the background and kept for the chat's lifetime.
pub struct Thumbs {
    pub mode: Graphics,
    map: RefCell<HashMap<PathBuf, Slot>>,
    tx: Sender<Done>,
    rx: Receiver<Done>,
    next: Cell<u32>,
}

impl Default for Thumbs {
    fn default() -> Self {
        let (tx, rx) = channel();
        Thumbs {
            mode: graphics(),
            map: RefCell::new(HashMap::new()),
            tx,
            rx,
            next: Cell::new(THUMB_IDS),
        }
    }
}

impl Thumbs {
    /// The thumbnail for `path`, starting to decode it the first time.
    pub fn get(&self, path: &Path) -> Slot {
        if let Some(s) = self.map.borrow().get(path) {
            return s.clone();
        }
        self.map
            .borrow_mut()
            .insert(path.to_path_buf(), Slot::Loading);
        let (tx, path) = (self.tx.clone(), path.to_path_buf());
        std::thread::spawn(move || {
            let result = thumbnail(&path);
            let _ = tx.send((path, result));
        });
        Slot::Loading
    }

    /// Take finished thumbnails.
    pub fn pump(&self) {
        while let Ok((path, result)) = self.rx.try_recv() {
            let slot = match result {
                Some((img, orig)) => {
                    let id = self.next.get();
                    self.next.set(id + 1);
                    Slot::Ready(Rc::new(Thumb {
                        id,
                        pic: Picture::new(img, self.mode),
                        orig,
                        video: is_video(&path),
                    }))
                }
                None => Slot::Failed,
            };
            self.map.borrow_mut().insert(path, slot);
        }
    }
}

fn thumbnail(path: &Path) -> Option<(RgbaImage, (u32, u32))> {
    let img = if is_video(path) {
        poster(path)?
    } else {
        still(path).ok()?
    };
    let orig = (img.width(), img.height());
    Some((shrink(img, THUMB_SIDE), orig))
}

/// A video's first frame (ffmpeg, else Quick Look).
fn poster(path: &Path) -> Option<DynamicImage> {
    if let Some(ffmpeg) = which("ffmpeg") {
        let out = Command::new(ffmpeg)
            .args(["-v", "error", "-nostdin", "-i"])
            .arg(path)
            .args(["-frames:v", "1", "-f", "image2pipe", "-vcodec", "png", "-"])
            .output()
            .ok()?;
        return image::load_from_memory_with_format(&out.stdout, image::ImageFormat::Png).ok();
    }
    let dir = scratch();
    Command::new("/usr/bin/qlmanage")
        .args(["-t", "-s", "1024", "-o"])
        .arg(&dir)
        .arg(path)
        .output()
        .ok()?;
    let name = path.file_name()?.to_string_lossy();
    let thumb = dir.join(format!("{name}.png"));
    let img = image::open(&thumb).ok();
    let _ = fs::remove_file(&thumb);
    img
}

// ---- the full-screen viewer ----

struct Pic {
    pic: Picture,
    delay: Duration,
}

enum Msg {
    Frame(Pic),
    /// Original size in pixels.
    Size(u32, u32),
    Info(String),
    Error(String),
}

pub struct Preview {
    pub items: Vec<PathBuf>,
    pub idx: usize,
    pub mode: Graphics,
    rx: Receiver<Msg>,
    current: Option<Pic>,
    orig: Option<(u32, u32)>,
    due: Instant,
    pub paused: bool,
    pub info: String,
    pub error: Option<String>,
    /// Frames shown so far (more than one: it is moving).
    frames: usize,
}

impl Preview {
    pub fn open(items: Vec<PathBuf>, idx: usize) -> Preview {
        let mode = graphics();
        let rx = load(items[idx].clone(), mode);
        Preview {
            items,
            idx,
            mode,
            rx,
            current: None,
            orig: None,
            due: Instant::now(),
            paused: false,
            info: String::new(),
            error: None,
            frames: 0,
        }
    }

    pub fn path(&self) -> &Path {
        &self.items[self.idx]
    }

    pub fn name(&self) -> String {
        self.path()
            .file_name()
            .map(|n| n.to_string_lossy().to_string())
            .unwrap_or_default()
    }

    pub fn moving(&self) -> bool {
        self.frames > 1
    }

    /// Show the next (or previous) item.
    pub fn step(&mut self, forward: bool) {
        let n = self.items.len();
        if n < 2 {
            return;
        }
        let idx = if forward {
            (self.idx + 1) % n
        } else {
            (self.idx + n - 1) % n
        };
        *self = Preview::open(std::mem::take(&mut self.items), idx);
    }

    /// Take the next frame when it is due.
    pub fn pump(&mut self) {
        loop {
            if self.paused && self.current.is_some() {
                return;
            }
            if self.current.is_some() && Instant::now() < self.due {
                return;
            }
            match self.rx.try_recv() {
                Ok(Msg::Frame(pic)) => {
                    // Catch up instead of drifting when frames arrive late.
                    self.due = (self.due + pic.delay).max(Instant::now());
                    self.frames += 1;
                    self.current = Some(pic);
                }
                Ok(Msg::Size(w, h)) => self.orig = Some((w, h)),
                Ok(Msg::Info(s)) => self.info = s,
                Ok(Msg::Error(e)) => self.error = Some(e),
                Err(TryRecvError::Empty | TryRecvError::Disconnected) => return,
            }
        }
    }

    pub fn draw(&self, f: &mut Frame, area: Rect, wants: &RefCell<Vec<Want>>) {
        let total = self.items.len();
        let title = if total > 1 {
            format!("{} · {}/{}", self.name(), self.idx + 1, total)
        } else {
            self.name()
        };
        let rect = centered(
            area,
            area.width.saturating_sub(4).max(20),
            area.height.saturating_sub(2).max(8),
        );
        f.render_widget(Clear, rect);
        let block = panel(&title, true);
        let inner = block.inner(rect);
        f.render_widget(block, rect);
        let [pic, foot] =
            Layout::vertical([Constraint::Min(1), Constraint::Length(1)]).areas(inner);

        let mut keys: Vec<(&str, &str)> = vec![("esc", "close")];
        if total > 1 {
            keys.push(("←→", "browse"));
        }
        if self.moving() {
            keys.push(("space", if self.paused { "play" } else { "pause" }));
        }
        keys.push(("o", "open"));
        let mut spans: Vec<Span> = Vec::new();
        for (k, w) in keys {
            spans.push(Span::styled(format!(" {k}"), Style::new().fg(GOLD)));
            spans.push(Span::styled(format!(" {w} "), Style::new().fg(DIM)));
        }
        if !self.info.is_empty() {
            spans.push(Span::styled(
                format!(" · {}", self.info),
                Style::new().fg(FAINT),
            ));
        }
        f.render_widget(Paragraph::new(Line::from(spans)), foot);

        let Some(cur) = &self.current else {
            let (msg, color) = match &self.error {
                Some(e) => (format!("✗ {e}"), BAD),
                None => (format!("{} loading…", spinner(clock())), DIM),
            };
            f.render_widget(
                Paragraph::new(msg)
                    .style(Style::new().fg(color))
                    .alignment(Alignment::Center)
                    .wrap(Wrap { trim: false }),
                centered(pic, pic.width, 3),
            );
            return;
        };
        let size = self
            .orig
            .unwrap_or((cur.pic.img.width(), cur.pic.img.height()));
        let (cols, rows) = cells(self.mode, size, pic.width, pic.height);
        let at = centered(pic, cols, rows);
        render(
            f,
            self.mode,
            wants,
            VIEWER_ID,
            &cur.pic,
            (at.x, at.y as i32),
            (cols, rows),
            at,
        );
    }
}

fn clock() -> f32 {
    static START: std::sync::OnceLock<Instant> = std::sync::OnceLock::new();
    START.get_or_init(Instant::now).elapsed().as_secs_f32()
}

// ---- the kitty protocol ----

/// Keeps the terminal's kitty images in step with what was drawn.
#[derive(Default)]
pub struct Kitty {
    /// Image data the terminal holds: id → seq.
    sent: HashMap<u32, u64>,
    /// Placements on screen: id → (cells, crop).
    shown: HashMap<u32, (Rect, (u32, u32))>,
}

impl Kitty {
    /// Call after every draw with the placements that frame wanted.
    pub fn sync(&mut self, out: &mut impl Write, wants: &[Want]) -> std::io::Result<()> {
        let mut wrote = false;
        let gone: Vec<u32> = self
            .shown
            .keys()
            .filter(|id| !wants.iter().any(|w| w.id == **id))
            .copied()
            .collect();
        for id in gone {
            write!(out, "\x1b_Ga=d,d=i,i={id},q=2\x1b\\")?;
            self.shown.remove(&id);
            wrote = true;
        }
        for w in wants {
            if self.sent.get(&w.id) != Some(&w.seq) {
                transmit(out, w.id, &w.data)?;
                self.sent.insert(w.id, w.seq);
                self.shown.remove(&w.id);
                wrote = true;
            }
            if self.shown.get(&w.id) != Some(&(w.at, w.crop)) {
                // Save the cursor, draw at the cell without moving it, restore.
                write!(
                    out,
                    "\x1b7\x1b[{};{}H\x1b_Ga=p,i={},p=1,y={},h={},c={},r={},C=1,q=2\x1b\\\x1b8",
                    w.at.y + 1,
                    w.at.x + 1,
                    w.id,
                    w.crop.0,
                    w.crop.1,
                    w.at.width,
                    w.at.height
                )?;
                self.shown.insert(w.id, (w.at, w.crop));
                wrote = true;
            }
        }
        // Nothing on screen: free the terminal's copies.
        if wants.is_empty() && !self.sent.is_empty() {
            for id in self.sent.keys() {
                write!(out, "\x1b_Ga=d,d=I,i={id},q=2\x1b\\")?;
            }
            self.sent.clear();
            wrote = true;
        }
        if wrote {
            out.flush()?;
        }
        Ok(())
    }

    /// Forget what is on screen (after a full clear, e.g. a resize).
    pub fn reset(&mut self) {
        self.shown.clear();
    }
}

/// Send image data (no placement) in 4 KiB chunks.
fn transmit(out: &mut impl Write, id: u32, data: &str) -> std::io::Result<()> {
    let chunks: Vec<&[u8]> = data.as_bytes().chunks(4096).collect();
    for (i, chunk) in chunks.iter().enumerate() {
        let more = u8::from(i + 1 < chunks.len());
        if i == 0 {
            write!(out, "\x1b_Ga=t,f=100,t=d,i={id},q=2,m={more};")?;
        } else {
            write!(out, "\x1b_Gm={more};")?;
        }
        out.write_all(chunk)?;
        out.write_all(b"\x1b\\")?;
    }
    Ok(())
}

pub fn base64(data: &[u8]) -> String {
    const T: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut s = String::with_capacity(data.len().div_ceil(3) * 4);
    for c in data.chunks(3) {
        let n = (c[0] as u32) << 16
            | (*c.get(1).unwrap_or(&0) as u32) << 8
            | *c.get(2).unwrap_or(&0) as u32;
        s.push(T[(n >> 18) as usize & 63] as char);
        s.push(T[(n >> 12) as usize & 63] as char);
        s.push(if c.len() > 1 {
            T[(n >> 6) as usize & 63] as char
        } else {
            '='
        });
        s.push(if c.len() > 2 {
            T[n as usize & 63] as char
        } else {
            '='
        });
    }
    s
}

fn png_b64(img: &RgbaImage) -> Option<String> {
    let mut buf = Vec::new();
    image::codecs::png::PngEncoder::new_with_quality(
        &mut buf,
        image::codecs::png::CompressionType::Fast,
        image::codecs::png::FilterType::Adaptive,
    )
    .write_image(
        img,
        img.width(),
        img.height(),
        image::ExtendedColorType::Rgba8,
    )
    .ok()?;
    Some(base64(&buf))
}

// ---- decoding ----

/// Decode `path` on a worker thread; frames arrive on the returned channel.
/// Dropping the receiver stops the worker.
fn load(path: PathBuf, mode: Graphics) -> Receiver<Msg> {
    let (tx, rx) = sync_channel(3);
    std::thread::spawn(move || {
        let result = if is_video(&path) {
            video(&path, mode, &tx)
        } else if ext(&path) == "gif" {
            gif(&path, mode, &tx)
        } else {
            still(&path).map(|img| {
                let _ = tx.send(Msg::Size(img.width(), img.height()));
                let _ = tx.send(Msg::Info(format!("{}×{}", img.width(), img.height())));
                let pic = Picture::new(shrink(img, MAX_SIDE), mode);
                let _ = tx.send(Msg::Frame(Pic {
                    pic,
                    delay: Duration::ZERO,
                }));
            })
        };
        if let Err(e) = result {
            let _ = tx.send(Msg::Error(e));
        }
    });
    rx
}

/// Fit within side×side (`thumbnail` alone would also enlarge small images).
fn shrink(img: DynamicImage, side: u32) -> RgbaImage {
    if img.width().max(img.height()) > side {
        img.thumbnail(side, side).to_rgba8()
    } else {
        img.to_rgba8()
    }
}

fn still(path: &Path) -> Result<DynamicImage, String> {
    let name = path.display();
    match ext(path).as_str() {
        "png" | "jpg" | "jpeg" | "webp" | "bmp" | "gif" => image::ImageReader::open(path)
            .and_then(|r| r.with_guessed_format())
            .map_err(|e| format!("{name}: {e}"))?
            .decode()
            .map_err(|e| format!("{name}: {e}")),
        // HEIC, TIFF and anything else macOS can read.
        _ => {
            let out = scratch().join(format!("sips-{}.png", next_seq()));
            let ok = Command::new("/usr/bin/sips")
                .args(["-s", "format", "png"])
                .arg(path)
                .arg("--out")
                .arg(&out)
                .output()
                .is_ok_and(|o| o.status.success());
            let img = ok.then(|| image::open(&out).ok()).flatten();
            let _ = fs::remove_file(&out);
            img.ok_or_else(|| format!("could not read {name}"))
        }
    }
}

fn scratch() -> PathBuf {
    let dir = std::env::temp_dir().join("ferrum-preview");
    let _ = fs::create_dir_all(&dir);
    dir
}

fn gif(path: &Path, mode: Graphics, tx: &SyncSender<Msg>) -> Result<(), String> {
    let file = fs::File::open(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let decoder = image::codecs::gif::GifDecoder::new(BufReader::new(file))
        .map_err(|e| format!("{}: {e}", path.display()))?;
    let mut frames = Vec::new();
    // Cap memory: long GIFs play their first 300 frames.
    for frame in decoder.into_frames().take(300) {
        let Ok(frame) = frame else { break };
        let (n, d) = frame.delay().numer_denom_ms();
        let ms = n.checked_div(d).map_or(100, |ms| ms.max(20));
        let img = shrink(DynamicImage::ImageRgba8(frame.into_buffer()), MAX_SIDE);
        frames.push((Picture::new(img, mode), Duration::from_millis(ms as u64)));
    }
    let Some((first, _)) = frames.first() else {
        return Err(format!("could not read {}", path.display()));
    };
    let (w, h) = (first.img.width(), first.img.height());
    let _ = tx.send(Msg::Size(w, h));
    let _ = tx.send(Msg::Info(format!("{w}×{h} · {} frames", frames.len())));
    loop {
        for (pic, delay) in &frames {
            let pic = Picture::with(pic.img.clone(), pic.kitty.clone());
            if tx.send(Msg::Frame(Pic { pic, delay: *delay })).is_err() {
                return Ok(());
            }
        }
        if frames.len() == 1 {
            return Ok(());
        }
    }
}

fn which(cmd: &str) -> Option<PathBuf> {
    let path = std::env::var_os("PATH").unwrap_or_default();
    std::env::split_paths(&path)
        .chain(["/opt/homebrew/bin".into(), "/usr/local/bin".into()])
        .map(|d| d.join(cmd))
        .find(|p| p.is_file())
}

fn video(path: &Path, mode: Graphics, tx: &SyncSender<Msg>) -> Result<(), String> {
    let Some(ffmpeg) = which("ffmpeg") else {
        let img = poster(path).ok_or_else(|| {
            "no preview: install ffmpeg (brew install ffmpeg) to play videos".to_string()
        })?;
        let _ = tx.send(Msg::Size(img.width(), img.height()));
        let _ = tx.send(Msg::Info(
            "poster frame · brew install ffmpeg to play".into(),
        ));
        let pic = Picture::new(shrink(img, MAX_SIDE), mode);
        let _ = tx.send(Msg::Frame(Pic {
            pic,
            delay: Duration::ZERO,
        }));
        return Ok(());
    };
    if let Some((w, h, info)) = probe(path) {
        let _ = tx.send(Msg::Size(w, h));
        let _ = tx.send(Msg::Info(info));
    }
    let delay = Duration::from_millis(1000 / VIDEO_FPS as u64);
    loop {
        let mut child = Command::new(&ffmpeg)
            .args(["-v", "error", "-nostdin", "-i"])
            .arg(path)
            .args([
                "-an",
                "-vf",
                &format!("fps={VIDEO_FPS},scale='min({VIDEO_WIDTH},iw)':-2"),
                "-f",
                "image2pipe",
                "-vcodec",
                "png",
                "-compression_level",
                "1",
                "-",
            ])
            .stdin(std::process::Stdio::null())
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .spawn()
            .map_err(|e| format!("ffmpeg: {e}"))?;
        let mut out = BufReader::new(child.stdout.take().ok_or("ffmpeg: no output")?);
        let mut count = 0;
        while let Some(png) = next_png(&mut out) {
            let Ok(img) = image::load_from_memory_with_format(&png, image::ImageFormat::Png) else {
                break;
            };
            // ffmpeg already wrote a PNG; send it as it is.
            let kitty = (mode == Graphics::Kitty).then(|| Arc::from(base64(&png)));
            let pic = Picture::with(img.to_rgba8(), kitty);
            if tx.send(Msg::Frame(Pic { pic, delay })).is_err() {
                let _ = child.kill();
                let _ = child.wait();
                return Ok(());
            }
            count += 1;
        }
        let mut err = String::new();
        if let Some(mut e) = child.stderr.take() {
            let _ = e.read_to_string(&mut err);
        }
        let _ = child.wait();
        if count == 0 {
            let first = err.lines().next().unwrap_or("could not decode the video");
            return Err(format!("ffmpeg: {first}"));
        }
        // Loop.
    }
}

/// (width, height, "1920×1080 · 0:42") from ffprobe.
fn probe(path: &Path) -> Option<(u32, u32, String)> {
    let out = Command::new(which("ffprobe")?)
        .args([
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height:format=duration",
            "-of",
            "json",
        ])
        .arg(path)
        .output()
        .ok()?;
    let j: Json = serde_json::from_slice(&out.stdout).ok()?;
    let s = &j["streams"][0];
    let (w, h) = (s["width"].as_u64()? as u32, s["height"].as_u64()? as u32);
    let secs = j["format"]["duration"]
        .as_str()
        .and_then(|d| d.parse::<f64>().ok())
        .unwrap_or(0.0) as u64;
    Some((w, h, format!("{w}×{h} · {}:{:02}", secs / 60, secs % 60)))
}

/// Read one PNG from a stream of concatenated PNGs (ffmpeg's image2pipe).
fn next_png(r: &mut impl Read) -> Option<Vec<u8>> {
    let mut png = vec![0u8; 8];
    r.read_exact(&mut png).ok()?;
    if &png[..] != b"\x89PNG\r\n\x1a\n" {
        return None;
    }
    loop {
        let mut head = [0u8; 8];
        r.read_exact(&mut head).ok()?;
        let len = u32::from_be_bytes([head[0], head[1], head[2], head[3]]) as usize;
        png.extend_from_slice(&head);
        let start = png.len();
        png.resize(start + len + 4, 0);
        r.read_exact(&mut png[start..]).ok()?;
        if &head[4..] == b"IEND" {
            return Some(png);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn base64_matches_rfc4648() {
        assert_eq!(base64(b""), "");
        assert_eq!(base64(b"f"), "Zg==");
        assert_eq!(base64(b"fo"), "Zm8=");
        assert_eq!(base64(b"foo"), "Zm9v");
        assert_eq!(base64(b"foobar"), "Zm9vYmFy");
    }

    #[test]
    fn splits_concatenated_pngs() {
        let img = RgbaImage::from_pixel(3, 2, image::Rgba([1, 2, 3, 255]));
        let mut one = Vec::new();
        image::codecs::png::PngEncoder::new(&mut one)
            .write_image(&img, 3, 2, image::ExtendedColorType::Rgba8)
            .unwrap();
        let stream = [one.clone(), one.clone()].concat();
        let mut r = &stream[..];
        assert_eq!(next_png(&mut r).unwrap(), one);
        assert_eq!(next_png(&mut r).unwrap(), one);
        assert!(next_png(&mut r).is_none());
    }

    #[test]
    fn kitty_sends_once_places_on_change_and_frees() {
        let pic = Picture::with(RgbaImage::new(4, 4), Some(Arc::from("AAAA")));
        let want = |y: u16| Want {
            id: 7,
            seq: pic.seq,
            data: pic.kitty.clone().unwrap(),
            at: Rect::new(2, y, 3, 2),
            crop: (0, 4),
        };
        let mut k = Kitty::default();
        let mut out = Vec::new();
        k.sync(&mut out, &[want(1)]).unwrap();
        let s = String::from_utf8(out).unwrap();
        assert!(s.contains("a=t,f=100,t=d,i=7"));
        assert!(s.contains("\x1b[2;3H\x1b_Ga=p,i=7,p=1,y=0,h=4,c=3,r=2"));
        // Same frame: nothing to say.
        let mut out = Vec::new();
        k.sync(&mut out, &[want(1)]).unwrap();
        assert!(out.is_empty());
        // Moved: placed again, not re-sent.
        let mut out = Vec::new();
        k.sync(&mut out, &[want(5)]).unwrap();
        let s = String::from_utf8(out).unwrap();
        assert!(!s.contains("a=t") && s.contains("a=p"));
        // Gone: placement deleted and data freed.
        let mut out = Vec::new();
        k.sync(&mut out, &[]).unwrap();
        let s = String::from_utf8(out).unwrap();
        assert!(s.contains("a=d,d=i,i=7") && s.contains("a=d,d=I,i=7"));
    }

    /// First frame from the worker, waiting up to 20 s.
    fn first_frame(path: PathBuf) -> (Pic, String) {
        let rx = load(path, Graphics::Kitty);
        let mut info = String::new();
        loop {
            match rx.recv_timeout(Duration::from_secs(20)).expect("no frame") {
                Msg::Frame(p) => return (p, info),
                Msg::Info(s) => info = s,
                Msg::Size(..) => {}
                Msg::Error(e) => panic!("{e}"),
            }
        }
    }

    #[test]
    fn decodes_gifs_and_videos() {
        let Some(ffmpeg) = which("ffmpeg") else {
            return eprintln!("ffmpeg not installed; skipped");
        };
        let dir = std::env::temp_dir().join(format!("ferrum-preview-test-{}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        for name in ["clip.mp4", "anim.gif"] {
            let ok = Command::new(&ffmpeg)
                .args(["-v", "error", "-y", "-f", "lavfi", "-i"])
                .arg("testsrc=size=320x240:rate=10:duration=1")
                .arg(dir.join(name))
                .status()
                .unwrap()
                .success();
            assert!(ok, "ffmpeg could not make {name}");
            let (p, info) = first_frame(dir.join(name));
            assert_eq!(
                (p.pic.img.width(), p.pic.img.height()),
                (320, 240),
                "{name}"
            );
            assert!(
                p.pic
                    .kitty
                    .as_deref()
                    .is_some_and(|k| k.starts_with("iVBOR")),
                "{name}"
            );
            assert!(info.starts_with("320×240"), "{name}: {info}");
            let (thumb, orig) = thumbnail(&dir.join(name)).expect("thumbnail");
            assert_eq!(orig, (320, 240));
            assert_eq!(thumb.width(), 320);
        }
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn knows_media_types() {
        assert!(is_video(Path::new("a.MOV")));
        assert!(!is_video(Path::new("a.gif")));
        assert!(previewable(Path::new("a.gif")));
        assert!(previewable(Path::new("a.heic")));
        assert!(!previewable(Path::new("a.txt")));
    }
}
