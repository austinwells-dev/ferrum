//! Attachments for the chat: files, dropped paths, clipboard contents and
//! screenshots. Everything is copied into the workspace (`.attachments/`) so
//! the agent's tools can reach it. Images are not shown to the model yet; text
//! found in them with macOS Vision is sent instead, and `Kind::Image` is the
//! place to hand the pixels to a vision model later.
use crate::agent::Sandbox;
use crate::*;

const CLIPBOARD_JS: &str = include_str!("clipboard.js");
const OCR_JS: &str = include_str!("ocr.js");
/// Largest file read into a message, and how much of it is pasted inline.
const MAX_BYTES: u64 = 8 << 20;
const INLINE_CHARS: usize = 24_000;
const IMAGE_EXTS: [&str; 9] = [
    "png", "jpg", "jpeg", "gif", "webp", "heic", "tiff", "tif", "bmp",
];

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Kind {
    Text,
    Image,
    Binary,
}

#[derive(Clone, Debug)]
pub struct Attachment {
    pub name: String,
    /// Where the copy lives, relative to the workspace.
    pub rel: String,
    pub kind: Kind,
    pub bytes: u64,
    pub text: Option<String>,
    /// Text recognised in an image.
    pub ocr: Option<String>,
}

impl Attachment {
    /// Short label for the chip row.
    pub fn chip(&self) -> String {
        let size = if self.bytes >= 1 << 20 {
            format!("{:.1} MB", self.bytes as f64 / (1u64 << 20) as f64)
        } else {
            format!("{} KB", self.bytes.div_ceil(1024).max(1))
        };
        format!("{} · {size}", self.name)
    }

    /// What the model receives for this attachment.
    pub fn block(&self) -> String {
        match self.kind {
            Kind::Text => {
                let text = self.text.as_deref().unwrap_or("");
                let total = text.chars().count();
                let shown: String = text.chars().take(INLINE_CHARS).collect();
                let note = if total > INLINE_CHARS {
                    format!(
                        "\n… [{} more characters; the whole file is at {}]",
                        total - INLINE_CHARS,
                        self.rel
                    )
                } else {
                    String::new()
                };
                format!(
                    "<attachment name=\"{}\" path=\"{}\">\n{shown}{note}\n</attachment>",
                    self.name, self.rel
                )
            }
            // Vision hook: replace this with the image itself once the runtime can take one.
            Kind::Image => match self.ocr.as_deref().filter(|t| !t.trim().is_empty()) {
                Some(text) => format!(
                    "<attachment name=\"{}\" path=\"{}\" type=\"image\">\n[You cannot see the image. Text recognised in it:]\n{}\n</attachment>",
                    self.name,
                    self.rel,
                    text.chars().take(INLINE_CHARS).collect::<String>()
                ),
                None => format!(
                    "<attachment name=\"{}\" path=\"{}\" type=\"image\">\n[An image; you cannot see it and no text was recognised in it.]\n</attachment>",
                    self.name, self.rel
                ),
            },
            Kind::Binary => format!(
                "<attachment name=\"{}\" path=\"{}\" type=\"binary\">\n[A binary file of {} bytes saved in the workspace.]\n</attachment>",
                self.name, self.rel, self.bytes
            ),
        }
    }
}

fn unique_dest(dir: &Path, name: &str) -> PathBuf {
    let clean: String = name
        .chars()
        .map(|c| {
            if c.is_alphanumeric() || "._- ".contains(c) {
                c
            } else {
                '_'
            }
        })
        .collect();
    let path = Path::new(&clean);
    let stem = path
        .file_stem()
        .map(|s| s.to_string_lossy().to_string())
        .unwrap_or_else(|| "file".into());
    let ext = path
        .extension()
        .map(|e| format!(".{}", e.to_string_lossy()))
        .unwrap_or_default();
    let mut dest = dir.join(&clean);
    let mut n = 2;
    while dest.exists() {
        dest = dir.join(format!("{stem}-{n}{ext}"));
        n += 1;
    }
    dest
}

fn stamp() -> String {
    let secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    format!(
        "{:02}{:02}{:02}",
        secs / 3600 % 24,
        secs / 60 % 60,
        secs % 60
    )
}

pub fn is_image(path: &Path) -> bool {
    path.extension()
        .map(|e| IMAGE_EXTS.contains(&e.to_string_lossy().to_lowercase().as_str()))
        .unwrap_or(false)
}

/// Recognise text in an image with the macOS Vision framework.
pub fn ocr(path: &Path) -> Option<String> {
    let out = Command::new("/usr/bin/osascript")
        .args(["-l", "JavaScript", "-e", OCR_JS])
        .arg(path)
        .output()
        .ok()?;
    let text = String::from_utf8_lossy(&out.stdout).trim().to_string();
    (out.status.success() && !text.is_empty()).then_some(text)
}

/// Describe a file that is already inside the attachments folder.
fn describe(sb: &Sandbox, stored: &Path, name: &str) -> Result<Attachment, String> {
    let meta = fs::metadata(stored).map_err(|e| e.to_string())?;
    let rel = stored
        .strip_prefix(&sb.workspace)
        .map(|p| p.display().to_string())
        .unwrap_or_else(|_| stored.display().to_string());
    let mut att = Attachment {
        name: name.to_string(),
        rel,
        kind: Kind::Binary,
        bytes: meta.len(),
        text: None,
        ocr: None,
    };
    if is_image(stored) {
        att.kind = Kind::Image;
        att.ocr = ocr(stored);
        return Ok(att);
    }
    let bytes = fs::read(stored).map_err(|e| e.to_string())?;
    if !bytes.contains(&0) {
        att.kind = Kind::Text;
        att.text = Some(String::from_utf8_lossy(&bytes).into_owned());
    }
    Ok(att)
}

/// Copy a file into the workspace and describe it.
pub fn add_file(sb: &Sandbox, src: &Path) -> Result<Attachment, String> {
    let meta = fs::metadata(src).map_err(|e| format!("{}: {e}", src.display()))?;
    if meta.is_dir() {
        return Err(format!("{} is a folder; attach files", src.display()));
    }
    if meta.len() > MAX_BYTES {
        return Err(format!(
            "{} is larger than {} MB",
            src.display(),
            MAX_BYTES >> 20
        ));
    }
    let name = src
        .file_name()
        .map(|n| n.to_string_lossy().to_string())
        .unwrap_or_else(|| "file".into());
    let dest = unique_dest(&sb.attachments(), &name);
    fs::copy(src, &dest).map_err(|e| format!("could not copy {}: {e}", src.display()))?;
    describe(sb, &dest, &name)
}

/// What is on the clipboard.
pub enum Clip {
    Files(Vec<PathBuf>),
    Image(Attachment),
    Text(String),
    Empty,
}

pub fn clipboard(sb: &Sandbox) -> Result<Clip, String> {
    let target = sb.attachments().join(format!("clipboard-{}.png", stamp()));
    let out = Command::new("/usr/bin/osascript")
        .args(["-l", "JavaScript", "-e", CLIPBOARD_JS])
        .arg(&target)
        .output()
        .map_err(|e| format!("could not read the clipboard: {e}"))?;
    if !out.status.success() {
        return Err("could not read the clipboard".into());
    }
    let json: Json = serde_json::from_slice(&out.stdout).map_err(|e| e.to_string())?;
    let files: Vec<PathBuf> = json["files"]
        .as_array()
        .map(|a| {
            a.iter()
                .filter_map(|v| v.as_str().map(PathBuf::from))
                .collect()
        })
        .unwrap_or_default();
    if !files.is_empty() {
        return Ok(Clip::Files(files));
    }
    if let Some(png) = json["image"].as_str() {
        let name = Path::new(png)
            .file_name()
            .map(|n| n.to_string_lossy().to_string())
            .unwrap_or_else(|| "clipboard.png".into());
        return describe(sb, Path::new(png), &name).map(Clip::Image);
    }
    match json["text"].as_str() {
        Some(t) if !t.is_empty() => Ok(Clip::Text(t.to_string())),
        _ => Ok(Clip::Empty),
    }
}

/// Let the user drag out a screen region. None when they cancel.
pub fn screenshot(sb: &Sandbox) -> Result<Option<Attachment>, String> {
    let name = format!("screenshot-{}.png", stamp());
    let dest = sb.attachments().join(&name);
    let status = Command::new("/usr/sbin/screencapture")
        .args(["-i", "-x"])
        .arg(&dest)
        .status()
        .map_err(|e| format!("could not run screencapture: {e}"))?;
    if !status.success() || !dest.exists() || fs::metadata(&dest).map(|m| m.len()).unwrap_or(0) == 0
    {
        let _ = fs::remove_file(&dest);
        return Ok(None);
    }
    describe(sb, &dest, &name).map(Some)
}

/// Paths in pasted text, as a terminal writes them when files are dropped on
/// it: quoted, backslash-escaped, `file://` URLs, one per line or space-separated.
pub fn parse_dropped(text: &str) -> Vec<PathBuf> {
    fn unescape(s: &str) -> String {
        let s = s.trim();
        let s = s.strip_prefix("file://").unwrap_or(s);
        let s = if (s.starts_with('\'') && s.ends_with('\'')
            || s.starts_with('"') && s.ends_with('"'))
            && s.len() >= 2
        {
            &s[1..s.len() - 1]
        } else {
            s
        };
        let mut out = String::new();
        let mut chars = s.chars();
        while let Some(c) = chars.next() {
            if c == '\\' {
                if let Some(n) = chars.next() {
                    out.push(n);
                }
            } else {
                out.push(c);
            }
        }
        out.replace("%20", " ")
    }
    fn shell_split(line: &str) -> Vec<String> {
        let (mut parts, mut cur, mut quote) = (Vec::new(), String::new(), None::<char>);
        let mut chars = line.chars();
        while let Some(c) = chars.next() {
            match (c, quote) {
                ('\\', _) => {
                    if let Some(n) = chars.next() {
                        cur.push(n);
                    }
                }
                ('\'' | '"', None) => quote = Some(c),
                (q, Some(open)) if q == open => quote = None,
                (' ', None) => {
                    if !cur.is_empty() {
                        parts.push(std::mem::take(&mut cur));
                    }
                }
                (c, _) => cur.push(c),
            }
        }
        if !cur.is_empty() {
            parts.push(cur);
        }
        parts
    }
    let mut found = Vec::new();
    for line in text.lines().map(str::trim).filter(|l| !l.is_empty()) {
        let whole = expand(&unescape(line));
        if whole.is_absolute() && whole.is_file() {
            found.push(whole);
            continue;
        }
        let parts: Vec<PathBuf> = shell_split(line)
            .iter()
            .map(|p| expand(&unescape(p)))
            .collect();
        if !parts.is_empty() && parts.iter().all(|p| p.is_absolute() && p.is_file()) {
            found.extend(parts);
        } else {
            return Vec::new();
        }
    }
    found
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn recognises_dropped_paths() {
        let dir = std::env::temp_dir().join(format!("ferrum-drop-test-{}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        let spaced = dir.join("my file.txt");
        let plain = dir.join("plain.txt");
        fs::write(&spaced, "a").unwrap();
        fs::write(&plain, "b").unwrap();
        let escaped = spaced.display().to_string().replace(' ', "\\ ");
        assert_eq!(parse_dropped(&escaped), vec![spaced.clone()]);
        assert_eq!(
            parse_dropped(&format!("'{}'", spaced.display())),
            vec![spaced.clone()]
        );
        assert_eq!(
            parse_dropped(&format!("{} {}", escaped, plain.display())),
            vec![spaced.clone(), plain.clone()]
        );
        assert_eq!(
            parse_dropped(&format!("file://{}", plain.display())),
            vec![plain.clone()]
        );
        assert!(parse_dropped("just some words I typed").is_empty());
        assert!(parse_dropped(&format!("see {}", plain.display())).is_empty());
        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn attaches_text_and_images() {
        let dir = std::env::temp_dir().join(format!("ferrum-att-test-{}", std::process::id()));
        let sb = Sandbox::new(&dir.join("ws").display().to_string(), false).unwrap();
        let src = dir.join("note.rs");
        fs::write(&src, "fn main() {}\n").unwrap();
        let a = add_file(&sb, &src).unwrap();
        assert_eq!(a.kind, Kind::Text);
        assert!(a.block().contains("fn main"), "{}", a.block());
        let b = add_file(&sb, &src).unwrap();
        assert_ne!(a.rel, b.rel, "second copy gets its own name");
        let bin = dir.join("data.bin");
        fs::write(&bin, [0u8, 1, 2]).unwrap();
        assert_eq!(add_file(&sb, &bin).unwrap().kind, Kind::Binary);
        assert!(add_file(&sb, &dir).is_err());
        let _ = fs::remove_dir_all(dir);
    }
}
