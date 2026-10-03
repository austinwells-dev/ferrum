//! Interactive launcher: pick a model, tune settings, then chat or serve.
//!
//! ferrum-tui [MODEL.gguf]   (aliased to `ferrum`)
//!
//! Scans the usual model folders (override with FERRUM_MODELS=dir:dir), shows
//! the exact command it will run, and hands the terminal over to `ferrum-cli`
//! or `ferrum-server`. Settings are remembered in ~/.config/ferrum/tui.json.
use crossterm::event::{self, Event, KeyCode, KeyEvent, KeyEventKind, KeyModifiers};
use ferrum::{
    hybrid::{config::HybridConfig, draft::DraftConfig},
    loader::gguf::{GgufReader, MetadataValue},
};
use ratatui::{
    DefaultTerminal, Frame,
    layout::{Alignment, Constraint, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, BorderType, Borders, Clear, Paragraph, Wrap},
};
use serde_json::{Map, Value as Json, json};
use std::{
    collections::HashSet,
    fs,
    os::unix::process::CommandExt,
    path::{Path, PathBuf},
    process::Command,
};

const EMBER: Color = Color::Rgb(255, 138, 61);
const GOLD: Color = Color::Rgb(255, 200, 120);
const TEXT: Color = Color::Rgb(226, 226, 236);
const DIM: Color = Color::Rgb(122, 122, 140);
const FAINT: Color = Color::Rgb(70, 70, 86);
const GOOD: Color = Color::Rgb(124, 220, 164);
const BAD: Color = Color::Rgb(255, 104, 104);
const SELECTED: Color = Color::Rgb(38, 33, 31);

#[derive(Clone, Copy, PartialEq)]
enum Mode {
    Chat,
    Serve,
}

#[derive(Clone, Copy, PartialEq)]
enum Scope {
    Both,
    Chat,
    Serve,
}

enum Kind {
    /// First option is the default and is never passed on the command line.
    Cycle(&'static [&'static str]),
    Num {
        step: f64,
        min: f64,
        max: f64,
        base: f64,
        int: bool,
    },
    Text {
        presets: &'static [&'static str],
        secret: bool,
    },
}

struct Field {
    key: &'static str,
    label: &'static str,
    group: &'static str,
    scope: Scope,
    kind: Kind,
    /// Empty means "not set": the binary uses its own default.
    value: String,
    default: &'static str,
    help: &'static str,
}

fn field(
    key: &'static str,
    label: &'static str,
    group: &'static str,
    scope: Scope,
    kind: Kind,
    default: &'static str,
    help: &'static str,
) -> Field {
    Field {
        key,
        label,
        group,
        scope,
        kind,
        value: String::new(),
        default,
        help,
    }
}

fn num(step: f64, min: f64, max: f64, base: f64, int: bool) -> Kind {
    Kind::Num {
        step,
        min,
        max,
        base,
        int,
    }
}

fn text(presets: &'static [&'static str]) -> Kind {
    Kind::Text {
        presets,
        secret: false,
    }
}

fn fields() -> Vec<Field> {
    use Scope::{Both, Chat, Serve};
    let m = "Model";
    let r = "Reasoning";
    let s = "Sampling";
    let d = "Speculative decoding";
    let v = "Server";
    vec![
        field(
            "context",
            "Context",
            m,
            Both,
            Kind::Cycle(&[
                "auto", "4096", "8192", "16384", "32768", "65536", "131072", "262144",
            ]),
            "auto",
            "Tokens of context. auto picks the largest that fits in memory.",
        ),
        field(
            "max_tokens",
            "Max tokens",
            m,
            Both,
            num(1024., 1., 1_000_000., 8192., true),
            "default",
            "Cap on tokens generated per reply.",
        ),
        field(
            "think",
            "Thinking",
            r,
            Both,
            Kind::Cycle(&["on", "off"]),
            "on",
            "Let the model reason before answering.",
        ),
        field(
            "effort",
            "Effort",
            r,
            Both,
            Kind::Cycle(&["default", "low", "medium", "high"]),
            "default",
            "Reasoning effort passed to the chat template.",
        ),
        field(
            "budget",
            "Reasoning budget",
            r,
            Both,
            num(256., 0., 100_000., 2048., true),
            "unlimited",
            "Force the end of thinking after this many tokens.",
        ),
        field(
            "show",
            "Show thinking",
            r,
            Chat,
            Kind::Cycle(&["show", "hide"]),
            "show",
            "Print the reasoning stream in the chat.",
        ),
        field(
            "temperature",
            "Temperature",
            s,
            Both,
            num(0.05, 0., 2., 0.7, false),
            "model default",
            "Higher is more random. Backspace resets to the model's default.",
        ),
        field(
            "top_p",
            "Top-p",
            s,
            Both,
            num(0.05, 0., 1., 0.95, false),
            "model default",
            "Nucleus sampling cutoff.",
        ),
        field(
            "top_k",
            "Top-k",
            s,
            Both,
            num(5., 0., 500., 20., true),
            "model default",
            "Keep only the k most likely tokens (0 disables).",
        ),
        field(
            "min_p",
            "Min-p",
            s,
            Both,
            num(0.01, 0., 1., 0.05, false),
            "model default",
            "Drop tokens below this fraction of the top token's probability.",
        ),
        field(
            "presence",
            "Presence penalty",
            s,
            Both,
            num(0.1, -2., 2., 0.5, false),
            "0",
            "Penalize tokens that already appeared.",
        ),
        field(
            "frequency",
            "Frequency penalty",
            s,
            Both,
            num(0.1, -2., 2., 0.5, false),
            "0",
            "Penalize tokens by how often they appeared.",
        ),
        field(
            "repeat",
            "Repeat penalty",
            s,
            Both,
            num(0.05, 0.5, 2., 1.1, false),
            "1",
            "Multiplicative repetition penalty.",
        ),
        field(
            "system",
            "System prompt",
            s,
            Chat,
            text(&[]),
            "none",
            "Enter to type a system prompt for the chat.",
        ),
        field(
            "draft",
            "Drafter",
            d,
            Both,
            text(&["off", "mtp"]),
            "off",
            "Lossless speedup. ←→ cycles, Enter opens the picker of detected DSpark/DFlash drafters.",
        ),
        field(
            "draft_max",
            "Drafts per step",
            d,
            Both,
            num(1., 1., 16., 3., true),
            "auto",
            "How many tokens the drafter proposes at a time.",
        ),
        field(
            "draft_quant",
            "Draft precision",
            d,
            Both,
            Kind::Cycle(&["q4_0", "q8_0"]),
            "q4_0",
            "Weight precision of the drafter.",
        ),
        field(
            "host",
            "Host",
            v,
            Serve,
            text(&["127.0.0.1", "0.0.0.0"]),
            "127.0.0.1",
            "127.0.0.1 is this Mac only; 0.0.0.0 exposes it to the network.",
        ),
        field(
            "port",
            "Port",
            v,
            Serve,
            num(1., 1., 65535., 8080., true),
            "8080",
            "Server listens on this port.",
        ),
        field(
            "api_key",
            "API key",
            v,
            Serve,
            Kind::Text {
                presets: &[],
                secret: true,
            },
            "none",
            "Require this bearer token from clients (Enter to type).",
        ),
        field(
            "alias",
            "Model alias",
            v,
            Serve,
            text(&[]),
            "file name",
            "Model name reported to clients.",
        ),
        field(
            "reserve",
            "Reserve (MiB)",
            v,
            Serve,
            num(256., 0., 65536., 1024., true),
            "1024",
            "GPU memory left free for the system.",
        ),
        field(
            "snapshots",
            "Snapshots",
            v,
            Serve,
            num(1., 0., 32., 2., true),
            "2",
            "Recurrent snapshots kept for prefix reuse.",
        ),
        field(
            "chunk",
            "Prefill chunk",
            v,
            Serve,
            num(128., 1., 8192., 512., true),
            "512",
            "Prompt tokens per forward pass.",
        ),
        field(
            "verbose",
            "Verbose log",
            v,
            Serve,
            Kind::Cycle(&["off", "on"]),
            "off",
            "Log request bodies and timings.",
        ),
    ]
}

struct Model {
    path: PathBuf,
    name: String,
    origin: String,
    size: u64,
    info: Option<Info>,
    hybrid: Option<HybridConfig>,
    /// Carries an MTP head ferrum can run (`--draft mtp`).
    mtp: bool,
    /// Why `--draft mtp` is unavailable.
    mtp_why: String,
    /// Why ferrum cannot run this file; None means it can.
    problem: Option<String>,
}

fn make_model(path: PathBuf, name: String, size: u64) -> Model {
    let checked = check(&path);
    Model {
        origin: origin_of(&path),
        path,
        name,
        size,
        info: checked.info,
        hybrid: checked.hybrid,
        mtp: checked.mtp,
        mtp_why: checked.mtp_why,
        problem: checked.problem,
    }
}

#[derive(Clone)]
struct Info {
    arch: String,
    name: Option<String>,
    layers: Option<u64>,
    experts: Option<u64>,
    train_ctx: Option<u64>,
    sampling: String,
}

fn home() -> PathBuf {
    std::env::var_os("HOME")
        .map(PathBuf::from)
        .unwrap_or_default()
}

fn tilde(path: &Path) -> String {
    let h = home();
    match path.strip_prefix(&h) {
        Ok(rest) if !h.as_os_str().is_empty() => format!("~/{}", rest.display()),
        _ => path.display().to_string(),
    }
}

fn expand(input: &str) -> PathBuf {
    match input.strip_prefix("~/") {
        Some(rest) => home().join(rest),
        None => PathBuf::from(input),
    }
}

fn gib(bytes: u64) -> String {
    format!("{:.1} GiB", bytes as f64 / (1u64 << 30) as f64)
}

fn clip(s: &str, width: usize) -> String {
    if s.chars().count() <= width {
        return s.to_string();
    }
    let keep = width.saturating_sub(1);
    format!("{}…", s.chars().take(keep).collect::<String>())
}

fn quant_of(name: &str) -> Option<String> {
    name.split(['-', '.', ' ']).find_map(|t| {
        let u = t.to_ascii_uppercase();
        let digit_after = |p: &str| {
            u.strip_prefix(p)
                .is_some_and(|r| r.starts_with(|c: char| c.is_ascii_digit()))
        };
        (digit_after("Q")
            || digit_after("IQ")
            || matches!(u.as_str(), "BF16" | "F16" | "F32")
            || u.starts_with("MXFP"))
        .then_some(u)
    })
}

/// The origin shown under a model: `org/repo` for Hugging Face cache entries.
fn origin_of(path: &Path) -> String {
    for part in path.components() {
        let s = part.as_os_str().to_string_lossy();
        if let Some(rest) = s.strip_prefix("models--") {
            return rest.replacen("--", "/", 1);
        }
    }
    path.parent().map(tilde).unwrap_or_default()
}

/// A GGUF found on disk, before its header is checked.
struct Cand {
    path: PathBuf,
    name: String,
    size: u64,
}

fn walk(dir: &Path, depth: usize, seen: &mut HashSet<PathBuf>, out: &mut Vec<Cand>) {
    let Ok(entries) = fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        let name = entry.file_name().to_string_lossy().to_string();
        if name.starts_with('.') {
            continue;
        }
        let Ok(meta) = fs::metadata(&path) else {
            continue;
        };
        if meta.is_dir() {
            if depth > 0 && !matches!(name.as_str(), "node_modules" | "target" | "Library") {
                walk(&path, depth - 1, seen, out);
            }
            continue;
        }
        let lower = name.to_ascii_lowercase();
        let later_shard = lower
            .split("-of-")
            .next()
            .and_then(|head| head.rsplit('-').next())
            .is_some_and(|n| lower.contains("-of-") && n.parse::<u32>().is_ok_and(|n| n > 1));
        if !lower.ends_with(".gguf")
            || lower.contains("vocab")
            || lower.contains("mmproj")
            || later_shard
            || is_drafter(&lower)
            || is_drafter(&origin_of(&path).to_ascii_lowercase())
            || meta.len() < (32 << 20)
        {
            continue;
        }
        let canonical = fs::canonicalize(&path).unwrap_or_else(|_| path.clone());
        if !seen.insert(canonical) {
            continue;
        }
        let stem = name
            .trim_end_matches(".gguf")
            .trim_end_matches(".GGUF")
            .to_string();
        out.push(Cand {
            path,
            name: stem,
            size: meta.len(),
        });
    }
}

/// Speculative-decoding checkpoints (and MTP heads) are not chat models.
fn is_drafter(lower: &str) -> bool {
    lower.contains("dspark")
        || lower.contains("dflash")
        || lower.contains("speculator")
        || lower.starts_with("mtp-")
}

struct Drafter {
    /// The checkpoint directory (an HF cache entry works as is).
    path: PathBuf,
    label: String,
    kind: &'static str,
    config: Option<DraftConfig>,
    /// Why this checkpoint cannot be loaded at all.
    problem: Option<String>,
}

fn read_draft(dir: &Path) -> (Option<DraftConfig>, Option<String>) {
    let root = if dir.join("config.json").exists() {
        dir.to_path_buf()
    } else {
        let rev = fs::read_to_string(dir.join("refs/main")).unwrap_or_default();
        dir.join("snapshots").join(rev.trim())
    };
    let Ok(raw) = fs::read(root.join("config.json")) else {
        return (None, Some("no downloaded snapshot (config.json)".into()));
    };
    let config = serde_json::from_slice::<Json>(&raw)
        .map_err(|e| e.to_string())
        .and_then(|j| DraftConfig::from_json(&j).map_err(|e| e.to_string()));
    match config {
        Err(e) => (None, Some(e)),
        Ok(_) if !root.join("model.safetensors").exists() => {
            (None, Some("no model.safetensors".into()))
        }
        Ok(c) => (Some(c), None),
    }
}

fn find_drafters(dir: &Path, depth: usize, seen: &mut HashSet<PathBuf>, out: &mut Vec<Drafter>) {
    let Ok(entries) = fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        let name = entry.file_name().to_string_lossy().to_string();
        if name.starts_with('.') || !path.is_dir() {
            continue;
        }
        let lower = name.to_ascii_lowercase();
        let hf = lower.starts_with("models--");
        if is_drafter(&lower) && (hf || path.join("config.json").exists()) {
            let canonical = fs::canonicalize(&path).unwrap_or_else(|_| path.clone());
            if seen.insert(canonical) {
                let (config, problem) = read_draft(&path);
                out.push(Drafter {
                    config,
                    problem,
                    label: origin_of(&path.join("x")),
                    kind: if lower.contains("dflash") {
                        "dflash"
                    } else {
                        "dspark"
                    },
                    path,
                });
            }
        } else if !hf
            && depth > 0
            && !matches!(name.as_str(), "node_modules" | "target" | "Library")
        {
            find_drafters(&path, depth - 1, seen, out);
        }
    }
}

fn scan_drafters(extra: &[String]) -> Vec<Drafter> {
    let (mut seen, mut out) = (HashSet::new(), Vec::new());
    for (root, depth) in roots(extra) {
        if root.is_dir() {
            find_drafters(&root, depth, &mut seen, &mut out);
        }
    }
    out.sort_by(|a, b| a.label.to_lowercase().cmp(&b.label.to_lowercase()));
    out
}

/// Why this drafter cannot serve this model (the checks `DraftModel::load`
/// makes against the target); None means it fits.
fn drafter_problem(d: &Drafter, m: &Model) -> Option<String> {
    if let Some(p) = &d.problem {
        return Some(p.clone());
    }
    let (c, t) = (d.config.as_ref()?, m.hybrid.as_ref()?);
    if c.hidden != t.hidden {
        return Some(format!("hidden size {} ≠ model's {}", c.hidden, t.hidden));
    }
    if c.target_layers.iter().any(|&l| l >= t.layers) {
        return Some("reads layers the model does not have".into());
    }
    if c.mask_token as usize >= t.vocab {
        return Some("mask token outside the model's vocabulary".into());
    }
    None
}

fn roots(extra: &[String]) -> Vec<(PathBuf, usize)> {
    let h = home();
    let mut roots: Vec<(PathBuf, usize)> = Vec::new();
    if let Ok(env) = std::env::var("FERRUM_MODELS") {
        roots.extend(
            env.split(':')
                .filter(|s| !s.is_empty())
                .map(|s| (expand(s), 6)),
        );
    }
    roots.extend(extra.iter().map(|s| (expand(s), 6)));
    if let Ok(cwd) = std::env::current_dir() {
        roots.push((cwd.clone(), 1));
        roots.push((cwd.join("models"), 3));
    }
    roots.extend([
        (h.join("models"), 4),
        (h.join("Models"), 4),
        (h.join("Downloads"), 2),
        (h.join(".cache/huggingface/hub"), 5),
        (h.join(".cache/lm-studio/models"), 4),
        (h.join(".lmstudio/models"), 4),
    ]);
    roots
}

fn scan(extra: &[String]) -> Vec<Model> {
    let mut seen = HashSet::new();
    let mut out = Vec::new();
    for (root, depth) in roots(extra) {
        if root.is_file() {
            walk_file(&root, &mut seen, &mut out);
        } else {
            walk(&root, depth, &mut seen, &mut out);
        }
    }
    // Reading a header means parsing the whole tokenizer vocabulary, so check files in parallel.
    let threads = std::thread::available_parallelism().map_or(4, |n| n.get());
    let chunk = out.len().div_ceil(threads).max(1);
    let mut models: Vec<Model> = std::thread::scope(|scope| {
        let jobs: Vec<_> = out
            .chunks(chunk)
            .map(|part| {
                scope.spawn(move || {
                    part.iter()
                        .map(|c| make_model(c.path.clone(), c.name.clone(), c.size))
                        .collect::<Vec<_>>()
                })
            })
            .collect();
        jobs.into_iter()
            .flat_map(|j| j.join().unwrap_or_default())
            .collect()
    });
    models.sort_by_key(|m| (m.problem.is_some(), m.name.to_lowercase()));
    models
}

fn walk_file(path: &Path, seen: &mut HashSet<PathBuf>, out: &mut Vec<Cand>) {
    let canonical = fs::canonicalize(path).unwrap_or_else(|_| path.to_path_buf());
    let Ok(meta) = fs::metadata(path) else {
        return;
    };
    if seen.insert(canonical) {
        let stem = path
            .file_stem()
            .map(|s| s.to_string_lossy().to_string())
            .unwrap_or_default();
        out.push(Cand {
            path: path.to_path_buf(),
            name: stem,
            size: meta.len(),
        });
    }
}

fn as_u64(v: &MetadataValue) -> Option<u64> {
    match v {
        MetadataValue::Uint8(x) => Some(*x as u64),
        MetadataValue::Uint16(x) => Some(*x as u64),
        MetadataValue::Uint32(x) => Some(*x as u64),
        MetadataValue::Uint64(x) => Some(*x),
        MetadataValue::Int32(x) => u64::try_from(*x).ok(),
        MetadataValue::Int64(x) => u64::try_from(*x).ok(),
        _ => None,
    }
}

struct Checked {
    info: Option<Info>,
    hybrid: Option<HybridConfig>,
    mtp: bool,
    mtp_why: String,
    problem: Option<String>,
}

fn type_name(id: u32) -> String {
    match id {
        0 => "F32",
        1 => "F16",
        2 => "Q4_0",
        3 => "Q4_1",
        6 => "Q5_0",
        7 => "Q5_1",
        8 => "Q8_0",
        9 => "Q8_1",
        10 => "Q2_K",
        11 => "Q3_K",
        12 => "Q4_K",
        13 => "Q5_K",
        14 => "Q6_K",
        15 => "Q8_K",
        16 => "IQ2_XXS",
        17 => "IQ2_XS",
        18 => "IQ3_XXS",
        19 => "IQ1_S",
        20 => "IQ4_NL",
        21 => "IQ3_S",
        22 => "IQ2_S",
        23 => "IQ4_XS",
        29 => "IQ1_M",
        30 => "BF16",
        n => return format!("type {n}"),
    }
    .to_string()
}

/// Weight types the hybrid engine has kernels for (see `hybrid::weights`).
const SUPPORTED: [u32; 8] = [0, 2, 8, 12, 13, 14, 21, 23];

/// Mirrors what `Runtime::load` requires, without reading any weights.
fn check(path: &Path) -> Checked {
    let blocked = |problem: String| Checked {
        info: None,
        hybrid: None,
        mtp: false,
        mtp_why: String::new(),
        problem: Some(problem),
    };
    let opened = fs::File::open(path)
        .map_err(|e| ferrum::Error::Gguf(format!("{}: {e}", path.display())))
        .and_then(|f| GgufReader::from_reader(std::io::BufReader::with_capacity(1 << 20, f)));
    let file = match opened {
        Ok(f) => f,
        Err(e) => {
            let text = e.to_string();
            let quant = text
                .split("unsupported GGML tensor type ")
                .nth(1)
                .and_then(|r| r.split(|c: char| !c.is_ascii_digit()).next())
                .and_then(|n| n.parse::<u32>().ok());
            return blocked(match quant {
                Some(id) => format!("unsupported quant {}", type_name(id)),
                None => format!("unreadable GGUF: {text}"),
            });
        }
    };
    let md = file.metadata();
    let text = |key: &str| match md.get(key) {
        Some(MetadataValue::String(s)) => Some(s.clone()),
        _ => None,
    };
    let Some(arch) = text("general.architecture") else {
        return blocked("no architecture in the GGUF header".into());
    };
    let get = |suffix: &str| md.get(&format!("{arch}.{suffix}")).and_then(as_u64);
    let float = |key: &str| match md.get(key) {
        Some(MetadataValue::Float32(v)) => Some(*v),
        Some(MetadataValue::Float64(v)) => Some(*v as f32),
        _ => None,
    };
    // The GGUF's own sampling advice, with the same fallbacks as `recommended_sampling`.
    let s = (
        float("general.sampling.temp").unwrap_or(1.0),
        float("general.sampling.top_p").unwrap_or(0.95),
        md.get("general.sampling.top_k")
            .and_then(as_u64)
            .unwrap_or(20),
        float("general.sampling.min_p").unwrap_or(0.),
    );
    let info = Info {
        name: text("general.name"),
        layers: get("block_count"),
        experts: get("expert_count").filter(|n| *n > 0),
        train_ctx: get("context_length"),
        sampling: format!(
            "t {} · p {} · k {}{}",
            s.0,
            s.1,
            s.2,
            if s.3 > 0. {
                format!(" · min-p {}", s.3)
            } else {
                String::new()
            }
        ),
        arch: arch.clone(),
    };
    let has_mtp = get("nextn_predict_layers").is_some_and(|n| n > 0);
    let mut result = Checked {
        info: Some(info),
        hybrid: None,
        mtp: false,
        mtp_why: "this GGUF has no MTP head".into(),
        problem: None,
    };
    if !matches!(arch.as_str(), "qwen35" | "qwen35moe") {
        result.problem = Some(format!("architecture {arch}: needs qwen35 or qwen35moe"));
        return result;
    }
    if text("tokenizer.ggml.model").as_deref() != Some("gpt2")
        || text("tokenizer.ggml.pre").as_deref() != Some("qwen35")
    {
        result.problem = Some("tokenizer is not the qwen35 gpt2 tokenizer".into());
        return result;
    }
    let config = match HybridConfig::from_gguf(&file) {
        Ok(c) => c,
        Err(e) => {
            result.problem = Some(e.to_string());
            return result;
        }
    };
    // Tensors of the trailing NextN/MTP block are only read for `--draft mtp`.
    let (mut trunk, mut head): (Vec<u32>, Vec<u32>) = (Vec::new(), Vec::new());
    for t in file.tensors().values() {
        let supported = if t.dimensions.len() <= 1 {
            t.type_id == 0
        } else {
            SUPPORTED.contains(&t.type_id)
        };
        if supported {
            continue;
        }
        let layer = t
            .name
            .strip_prefix("blk.")
            .and_then(|r| r.split('.').next())
            .and_then(|n| n.parse::<usize>().ok());
        if layer.is_some_and(|l| l >= config.layers) {
            head.push(t.type_id);
        } else {
            trunk.push(t.type_id);
        }
    }
    let names = |mut ids: Vec<u32>| {
        ids.sort_unstable();
        ids.dedup();
        ids.into_iter()
            .map(type_name)
            .collect::<Vec<_>>()
            .join(", ")
    };
    if !trunk.is_empty() {
        result.problem = Some(format!(
            "unsupported quant {} (runs Q4_0 Q8_0 Q4_K Q5_K Q6_K IQ3_S IQ4_XS)",
            names(trunk)
        ));
        return result;
    }
    if has_mtp && head.is_empty() {
        result.mtp = true;
    } else if has_mtp {
        result.mtp_why = format!("MTP head uses unsupported quant {}", names(head));
    }
    result.hybrid = Some(config);
    result
}

fn config_path() -> PathBuf {
    home().join(".config/ferrum/tui.json")
}

#[derive(PartialEq)]
enum Focus {
    Favorites,
    Models,
    Settings,
}

/// A saved setup: model, mode and every setting, drafter included.
struct Fav {
    name: String,
    model: String,
    serve: bool,
    values: Map<String, Json>,
}

#[derive(Clone, Copy, PartialEq)]
enum Fit {
    Plain,
    Fits,
    Blocked,
}

struct Opt {
    value: String,
    label: String,
    /// What the option is; for blocked options, why it cannot be used.
    note: String,
    fit: Fit,
}

struct Picker {
    options: Vec<Opt>,
    cur: usize,
}

enum Editing {
    No,
    Field(usize),
    AddPath,
    FavName,
}

enum Outcome {
    Quit,
    Launch(&'static str, Vec<String>),
}

struct App {
    mode: Mode,
    focus: Focus,
    models: Vec<Model>,
    cursor: usize,
    fields: Vec<Field>,
    /// Index into `visible()`; one past the end is the Launch row.
    sel: usize,
    editing: Editing,
    buffer: String,
    extra: Vec<String>,
    drafters: Vec<Drafter>,
    favs: Vec<Fav>,
    fav_cursor: usize,
    picker: Option<Picker>,
    status: Option<(String, bool)>,
}

impl App {
    fn new(preselect: Option<String>) -> Self {
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
                        serve: v["mode"] == "serve",
                        values: v["values"].as_object().cloned().unwrap_or_default(),
                    })
                    .collect()
            })
            .unwrap_or_default();
        let mut app = App {
            mode: if saved["mode"] == "serve" {
                Mode::Serve
            } else {
                Mode::Chat
            },
            focus: Focus::Models,
            models,
            cursor,
            fields,
            sel: 0,
            editing: Editing::No,
            buffer: String::new(),
            extra,
            drafters,
            favs,
            fav_cursor: 0,
            picker: None,
            status: None,
        };
        app.load_info();
        app
    }

    fn save(&self) {
        let mut values = Map::new();
        for f in self.fields.iter().filter(|f| !f.value.is_empty()) {
            values.insert(f.key.into(), json!(f.value));
        }
        let config = json!({
            "mode": if self.mode == Mode::Serve { "serve" } else { "chat" },
            "model": self.selected().map(|m| m.path.display().to_string()),
            "values": values,
            "extra": self.extra,
            "favorites": self.favs.iter().map(|f| json!({
                "name": f.name,
                "model": f.model,
                "mode": if f.serve { "serve" } else { "chat" },
                "values": f.values,
            })).collect::<Vec<_>>(),
        });
        let path = config_path();
        if let Some(dir) = path.parent() {
            fs::create_dir_all(dir).ok();
        }
        fs::write(
            path,
            serde_json::to_string_pretty(&config).unwrap_or_default(),
        )
        .ok();
    }

    /// The model under the cursor, if ferrum can run it.
    fn selected(&self) -> Option<&Model> {
        self.models.get(self.cursor).filter(|m| m.problem.is_none())
    }

    fn runnable(&self) -> usize {
        self.models.iter().filter(|m| m.problem.is_none()).count()
    }

    /// Drop a drafter the newly selected model cannot use.
    fn load_info(&mut self) {
        let Some(m) = self.selected() else {
            return;
        };
        let current = self.value("draft").to_string();
        let reason = match current.as_str() {
            "" | "off" => None,
            "mtp" => (!m.mtp).then(|| m.mtp_why.clone()),
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

    fn visible(&self) -> Vec<usize> {
        (0..self.fields.len())
            .filter(|&i| match self.fields[i].scope {
                Scope::Both => true,
                Scope::Chat => self.mode == Mode::Chat,
                Scope::Serve => self.mode == Mode::Serve,
            })
            .collect()
    }

    fn current_field(&self) -> Option<usize> {
        self.visible().get(self.sel).copied()
    }

    fn value(&self, key: &str) -> &str {
        self.fields
            .iter()
            .find(|f| f.key == key)
            .map(|f| f.value.as_str())
            .unwrap_or("")
    }

    /// The binary and arguments for the current selection.
    fn command(&self) -> (&'static str, Vec<String>) {
        let serve = self.mode == Mode::Serve;
        let mut args: Vec<String> = Vec::new();
        if let Some(m) = self.selected() {
            args.extend(["--model".into(), m.path.display().to_string()]);
        }
        let draft = self.value("draft");
        let drafting = !draft.is_empty() && draft != "off";
        for f in &self.fields {
            let in_scope = match f.scope {
                Scope::Both => true,
                Scope::Chat => !serve,
                Scope::Serve => serve,
            };
            let v = f.value.as_str();
            let is_default = match &f.kind {
                Kind::Cycle(opts) => v.is_empty() || v == opts[0],
                _ => v.is_empty(),
            };
            if !in_scope || is_default {
                continue;
            }
            let flag = match f.key {
                "context" => "--context",
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
                "api_key" => "--api-key",
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
            args.push(v.to_string());
        }
        (if serve { "ferrum-server" } else { "ferrum-cli" }, args)
    }

    fn draft_values(&self) -> Vec<String> {
        let mut v = vec!["off".to_string()];
        let model = self.selected();
        if model.is_some_and(|m| m.mtp) {
            v.push("mtp".to_string());
        }
        v.extend(
            self.drafters
                .iter()
                .filter(|d| model.is_some_and(|m| drafter_problem(d, m).is_none()))
                .map(|d| d.path.display().to_string()),
        );
        v
    }

    fn draft_label(&self, value: &str) -> String {
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

    fn open_picker(&mut self) {
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

    fn picker_key(&mut self, code: KeyCode) {
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

    fn save_favorite(&mut self, name: &str) {
        let Some(model) = self.selected() else {
            self.status = Some(("pick a compatible model first".into(), false));
            return;
        };
        let mut values = Map::new();
        for f in self.fields.iter().filter(|f| !f.value.is_empty()) {
            values.insert(f.key.into(), json!(f.value));
        }
        let fav = Fav {
            name: name.to_string(),
            model: model.path.display().to_string(),
            serve: self.mode == Mode::Serve,
            values,
        };
        match self.favs.iter().position(|f| f.name == name) {
            Some(i) => self.favs[i] = fav,
            None => {
                self.favs.push(fav);
                self.fav_cursor = self.favs.len() - 1;
            }
        }
        self.status = Some((format!("saved favorite {name:?}"), true));
        self.save();
    }

    fn load_favorite(&mut self, i: usize) {
        let Some(fav) = self.favs.get(i) else {
            return;
        };
        for f in &mut self.fields {
            f.value = fav
                .values
                .get(f.key)
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();
        }
        self.mode = if fav.serve { Mode::Serve } else { Mode::Chat };
        let name = fav.name.clone();
        let found = self
            .models
            .iter()
            .position(|m| m.path.display().to_string() == fav.model);
        match found {
            Some(at) if self.models[at].problem.is_some() => {
                let why = self.models[at].problem.clone().unwrap_or_default();
                self.status = Some((format!("{name:?}: model unsupported ({why})"), false));
            }
            Some(at) => {
                self.cursor = at;
                self.load_info();
                self.status = Some((format!("loaded {name:?}"), true));
            }
            None => self.status = Some((format!("{name:?}: model file not found"), false)),
        }
        self.sel = 0;
        self.focus = Focus::Settings;
    }

    fn step(&mut self, idx: usize, dir: i32) {
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

    fn commit(&mut self) {
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
                if !name.is_empty() {
                    self.save_favorite(&name);
                }
            }
            Editing::No => {}
        }
        self.buffer.clear();
    }

    fn rescan(&mut self, focus: Option<&Path>) {
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

    fn launch(&mut self) -> Option<Outcome> {
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
            return None;
        }
        self.save();
        let (bin, args) = self.command();
        Some(Outcome::Launch(bin, args))
    }

    fn on_key(&mut self, key: KeyEvent) -> Option<Outcome> {
        if key.modifiers.contains(KeyModifiers::CONTROL) && key.code == KeyCode::Char('c') {
            self.save();
            return Some(Outcome::Quit);
        }
        if self.picker.is_some() {
            self.picker_key(key.code);
            return None;
        }
        if !matches!(self.editing, Editing::No) {
            match key.code {
                KeyCode::Enter => self.commit(),
                KeyCode::Esc => {
                    self.editing = Editing::No;
                    self.buffer.clear();
                }
                KeyCode::Backspace => {
                    self.buffer.pop();
                }
                KeyCode::Char(c) if !key.modifiers.contains(KeyModifiers::CONTROL) => {
                    self.buffer.push(c)
                }
                _ => {}
            }
            return None;
        }
        self.status = None;
        match key.code {
            KeyCode::Char('q') | KeyCode::Esc => {
                self.save();
                return Some(Outcome::Quit);
            }
            KeyCode::Char('m') => {
                self.mode = if self.mode == Mode::Chat {
                    Mode::Serve
                } else {
                    Mode::Chat
                };
                self.sel = self.sel.min(self.visible().len());
            }
            KeyCode::Char('l') | KeyCode::F(5) => return self.launch(),
            KeyCode::Char('a') => {
                self.editing = Editing::AddPath;
                self.buffer.clear();
            }
            KeyCode::Char('r') => self.rescan(None),
            KeyCode::Char('f') => {
                self.buffer = self
                    .models
                    .get(self.cursor)
                    .map(|m| m.name.clone())
                    .unwrap_or_default();
                self.editing = Editing::FavName;
            }
            KeyCode::Tab | KeyCode::BackTab => {
                let order = [Focus::Favorites, Focus::Models, Focus::Settings];
                let at = order.iter().position(|f| *f == self.focus).unwrap_or(0);
                let back = key.code == KeyCode::BackTab;
                let next = (at + if back { 2 } else { 1 }) % 3;
                self.focus = order.into_iter().nth(next).unwrap_or(Focus::Models);
            }
            code => match self.focus {
                Focus::Favorites => match code {
                    KeyCode::Up | KeyCode::Char('k') => {
                        self.fav_cursor = self.fav_cursor.saturating_sub(1)
                    }
                    KeyCode::Down | KeyCode::Char('j') => {
                        self.fav_cursor =
                            (self.fav_cursor + 1).min(self.favs.len().saturating_sub(1))
                    }
                    KeyCode::Enter | KeyCode::Right => self.load_favorite(self.fav_cursor),
                    KeyCode::Char('x') | KeyCode::Delete | KeyCode::Backspace => {
                        if self.fav_cursor < self.favs.len() {
                            let gone = self.favs.remove(self.fav_cursor);
                            self.fav_cursor = self.fav_cursor.saturating_sub(1);
                            self.status = Some((format!("removed {:?}", gone.name), true));
                            self.save();
                        }
                    }
                    _ => {}
                },
                Focus::Models => match code {
                    KeyCode::Up | KeyCode::Char('k') => {
                        self.cursor = self.cursor.saturating_sub(1);
                        self.load_info();
                    }
                    KeyCode::Down | KeyCode::Char('j') => {
                        self.cursor = (self.cursor + 1).min(self.models.len().saturating_sub(1));
                        self.load_info();
                    }
                    KeyCode::Enter | KeyCode::Right | KeyCode::Char('l') => {
                        self.focus = Focus::Settings
                    }
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
                        KeyCode::Enter if on_launch => return self.launch(),
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
        None
    }

    // ---- drawing ----

    fn draw(&self, f: &mut Frame) {
        let [header, body, command, footer] = Layout::vertical([
            Constraint::Length(3),
            Constraint::Min(10),
            Constraint::Length(7),
            Constraint::Length(1),
        ])
        .areas(f.area());
        self.draw_header(f, header);
        let [left, right] =
            Layout::horizontal([Constraint::Percentage(40), Constraint::Percentage(60)])
                .areas(body);
        let [favs, list, details] = Layout::vertical([
            Constraint::Length(6),
            Constraint::Min(5),
            Constraint::Length(8),
        ])
        .areas(left);
        self.draw_favorites(f, favs);
        self.draw_models(f, list);
        self.draw_details(f, details);
        self.draw_settings(f, right);
        self.draw_command(f, command);
        self.draw_footer(f, footer);
        if self.picker.is_some() {
            self.draw_picker(f);
        }
        if matches!(self.editing, Editing::AddPath | Editing::FavName) {
            self.draw_add_path(f);
        }
    }

    fn panel(&self, title: &str, focused: bool) -> Block<'static> {
        let color = if focused { EMBER } else { FAINT };
        Block::bordered()
            .border_type(BorderType::Rounded)
            .border_style(Style::new().fg(color))
            .title(Line::from(vec![
                Span::raw(" "),
                Span::styled(
                    title.to_string(),
                    Style::new()
                        .fg(if focused { GOLD } else { DIM })
                        .add_modifier(Modifier::BOLD),
                ),
                Span::raw(" "),
            ]))
    }

    fn draw_header(&self, f: &mut Frame, area: Rect) {
        let block = Block::new()
            .borders(Borders::BOTTOM)
            .border_style(Style::new().fg(FAINT));
        let inner = block.inner(area);
        f.render_widget(block, area);
        let [title, pills] =
            Layout::horizontal([Constraint::Min(10), Constraint::Length(22)]).areas(inner);
        let word = "FERRUM";
        let mut spans = vec![Span::styled(" ◆ ", Style::new().fg(EMBER))];
        for (i, c) in word.chars().enumerate() {
            let t = i as f32 / (word.len() - 1) as f32;
            let color = Color::Rgb(255, (200.0 - 90.0 * t) as u8, (110.0 - 70.0 * t) as u8);
            spans.push(Span::styled(
                format!("{c} "),
                Style::new().fg(color).add_modifier(Modifier::BOLD),
            ));
        }
        spans.push(Span::styled(
            " local models on Apple Silicon",
            Style::new().fg(DIM),
        ));
        f.render_widget(
            Paragraph::new(vec![Line::from(spans), Line::default()]),
            title,
        );
        let pill = |label: &str, on: bool| {
            if on {
                Span::styled(
                    format!(" {label} "),
                    Style::new()
                        .fg(Color::Black)
                        .bg(EMBER)
                        .add_modifier(Modifier::BOLD),
                )
            } else {
                Span::styled(format!(" {label} "), Style::new().fg(DIM))
            }
        };
        f.render_widget(
            Paragraph::new(Line::from(vec![
                pill("CHAT", self.mode == Mode::Chat),
                Span::raw(" "),
                pill("SERVE", self.mode == Mode::Serve),
            ]))
            .alignment(Alignment::Right),
            pills,
        );
    }

    fn draw_models(&self, f: &mut Frame, area: Rect) {
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

    fn draw_favorites(&self, f: &mut Frame, area: Rect) {
        let focused = self.focus == Focus::Favorites;
        let block = self.panel(&format!("Favorites ({})", self.favs.len()), focused);
        let inner = block.inner(area);
        f.render_widget(block, area);
        if self.favs.is_empty() {
            f.render_widget(
                Paragraph::new(vec![
                    Line::styled(" Press f to save the current", Style::new().fg(DIM)),
                    Line::styled(
                        " model and settings, drafter included.",
                        Style::new().fg(DIM),
                    ),
                ]),
                inner,
            );
            return;
        }
        let per_page = inner.height as usize;
        let top = (self.fav_cursor + 1).saturating_sub(per_page);
        let width = inner.width as usize;
        let lines: Vec<Line> = self
            .favs
            .iter()
            .enumerate()
            .skip(top)
            .take(per_page)
            .map(|(i, fav)| {
                let on = focused && i == self.fav_cursor;
                let bg = if on { SELECTED } else { Color::Reset };
                let draft = match fav.values.get("draft").and_then(|v| v.as_str()) {
                    None | Some("off") | Some("") => String::new(),
                    Some("mtp") => " ⚡mtp".to_string(),
                    Some(p) => format!(
                        " ⚡{}",
                        self.drafters
                            .iter()
                            .find(|d| d.path.display().to_string() == p)
                            .map(|d| d.kind)
                            .unwrap_or("draft")
                    ),
                };
                let tail = format!("{}{} ", if fav.serve { "serve" } else { "chat" }, draft);
                let name_w = width.saturating_sub(tail.chars().count() + 4);
                Line::from(vec![
                    Span::styled(if on { " ▌" } else { "  " }, Style::new().fg(EMBER).bg(bg)),
                    Span::styled("★ ", Style::new().fg(GOLD).bg(bg)),
                    Span::styled(
                        format!("{:<name_w$}", clip(&fav.name, name_w)),
                        Style::new().fg(if on { Color::White } else { TEXT }).bg(bg),
                    ),
                    Span::styled(tail, Style::new().fg(DIM).bg(bg)),
                ])
            })
            .collect();
        f.render_widget(Paragraph::new(lines), inner);
    }

    fn draw_picker(&self, f: &mut Frame) {
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

    fn draw_details(&self, f: &mut Frame, area: Rect) {
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

    fn draw_settings(&self, f: &mut Frame, area: Rect) {
        let focused = self.focus == Focus::Settings;
        let title = match self.mode {
            Mode::Chat => "Chat settings",
            Mode::Serve => "Server settings",
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
        let label = match self.mode {
            Mode::Chat => "  ▶  Start chat",
            Mode::Serve => "  ▶  Start server",
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

    fn field_line(&self, i: usize, on: bool, width: usize) -> Line<'static> {
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

    fn draw_command(&self, f: &mut Frame, area: Rect) {
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

    fn draw_footer(&self, f: &mut Frame, area: Rect) {
        let key = |k: &'static str, what: &'static str| {
            [
                Span::styled(format!(" {k}"), Style::new().fg(GOLD)),
                Span::styled(format!(" {what} "), Style::new().fg(DIM)),
            ]
        };
        let mut spans: Vec<Span> = Vec::new();
        if matches!(self.editing, Editing::No) {
            for (k, w) in [
                ("↑↓", "move"),
                ("←→", "adjust"),
                ("enter", "edit"),
                ("⌫", "reset"),
                ("tab", "pane"),
                ("m", "chat/serve"),
                ("f", "favorite"),
                ("a", "add path"),
                ("l", "launch"),
                ("q", "quit"),
            ] {
                spans.extend(key(k, w));
            }
        } else {
            for (k, w) in [("enter", "confirm"), ("esc", "cancel")] {
                spans.extend(key(k, w));
            }
        }
        if let Some((msg, ok)) = &self.status {
            spans.push(Span::styled(
                format!(" · {msg}"),
                Style::new().fg(if *ok { GOOD } else { BAD }),
            ));
        }
        f.render_widget(Paragraph::new(Line::from(spans)), area);
    }

    fn draw_add_path(&self, f: &mut Frame) {
        let area = f.area();
        let w = area.width.saturating_sub(8).min(72);
        let rect = Rect::new(
            area.x + (area.width - w) / 2,
            area.y + area.height / 2 - 2,
            w,
            5,
        );
        f.render_widget(Clear, rect);
        let (title, hint) = match self.editing {
            Editing::FavName => (
                "Save as favorite",
                " name this setup (same name overwrites)",
            ),
            _ => (
                "Add model file or folder",
                " a .gguf file, or a folder to scan",
            ),
        };
        let block = self.panel(title, true);
        let inner = block.inner(rect);
        f.render_widget(block, rect);
        f.render_widget(
            Paragraph::new(vec![
                Line::from(vec![
                    Span::styled(" ❯ ", Style::new().fg(EMBER)),
                    Span::styled(format!("{}▏", self.buffer), Style::new().fg(Color::White)),
                ]),
                Line::styled(hint, Style::new().fg(DIM)),
            ]),
            inner,
        );
    }
}

fn fmt_num(n: f64, int: bool) -> String {
    if int {
        format!("{}", n.round() as i64)
    } else {
        let s = format!("{:.3}", (n * 1000.0).round() / 1000.0);
        s.trim_end_matches('0').trim_end_matches('.').to_string()
    }
}

fn shell_quote(s: &str) -> String {
    if !s.is_empty()
        && s.chars()
            .all(|c| c.is_ascii_alphanumeric() || "/._-:=,@~+".contains(c))
    {
        s.to_string()
    } else {
        format!("'{}'", s.replace('\'', "'\\''"))
    }
}

fn run(terminal: &mut DefaultTerminal, app: &mut App) -> std::io::Result<Outcome> {
    loop {
        terminal.draw(|f| app.draw(f))?;
        if let Event::Key(key) = event::read()?
            && key.kind != KeyEventKind::Release
            && let Some(outcome) = app.on_key(key)
        {
            return Ok(outcome);
        }
    }
}

fn main() {
    let mut preselect = None;
    for arg in std::env::args().skip(1) {
        match arg.as_str() {
            "-h" | "--help" => {
                println!(
                    "ferrum-tui: interactive launcher for ferrum-cli and ferrum-server\n\n\
                     usage: ferrum-tui [MODEL.gguf]\n\n\
                     Models are found in ./, ~/models, ~/Downloads, the Hugging Face and\n\
                     LM Studio caches, and any folders in FERRUM_MODELS (colon separated)."
                );
                return;
            }
            _ => preselect = Some(arg),
        }
    }
    let mut app = App::new(preselect);
    let mut terminal = ratatui::init();
    let outcome = run(&mut terminal, &mut app);
    ratatui::restore();
    match outcome {
        Ok(Outcome::Launch(bin, args)) => {
            let sibling = std::env::current_exe()
                .ok()
                .and_then(|p| p.parent().map(|d| d.join(bin)))
                .filter(|p| p.exists());
            let program = sibling.unwrap_or_else(|| PathBuf::from(bin));
            let shown: Vec<String> = args.iter().map(|a| shell_quote(a)).collect();
            eprintln!("\x1b[2m$ {bin} {}\x1b[0m", shown.join(" "));
            let err = Command::new(program).args(&args).exec();
            eprintln!("error: could not start {bin}: {err}");
            std::process::exit(1);
        }
        Ok(Outcome::Quit) => {}
        Err(e) => {
            eprintln!("error: {e}");
            std::process::exit(1);
        }
    }
}
