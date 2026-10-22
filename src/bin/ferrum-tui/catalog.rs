use crate::*;
use ferrum::{
    hybrid::draft::{
        DraftModel, DraftOptions, Kind as DraftKind, check_drafter_tokenizer, is_dflash,
    },
    loader::gguf::GgufFile,
};
use std::{
    collections::{BTreeMap, HashMap},
    sync::Mutex,
};

pub struct Model {
    pub path: PathBuf,
    pub name: String,
    pub origin: String,
    pub size: u64,
    pub info: Option<Info>,
    pub hybrid: Option<HybridConfig>,
    /// Carries an MTP head ferrum can run (`--draft mtp`).
    pub mtp: bool,
    /// Why `--draft mtp` is unavailable.
    pub mtp_why: String,
    /// A vision projector sits next to the file, so images can be attached.
    pub vision: Option<PathBuf>,
    /// Why ferrum cannot run this file; None means it can.
    pub problem: Option<String>,
    /// The header's scalar `tokenizer.*` keys, to check GGUF drafters against.
    pub tokenizer: BTreeMap<String, MetadataValue>,
}

pub fn make_model(path: PathBuf, name: String, size: u64) -> Model {
    let checked = check(&path);
    let vision = if checked.problem.is_none() {
        ferrum::vision::resolve_projector(&path, None)
            .ok()
            .flatten()
    } else {
        None
    };
    Model {
        vision,
        origin: origin_of(&path),
        path,
        name,
        size,
        info: checked.info,
        hybrid: checked.hybrid,
        mtp: checked.mtp,
        mtp_why: checked.mtp_why,
        problem: checked.problem,
        tokenizer: checked.tokenizer,
    }
}

#[derive(Clone)]
pub struct Info {
    pub arch: String,
    pub name: Option<String>,
    pub layers: Option<u64>,
    pub experts: Option<u64>,
    pub train_ctx: Option<u64>,
    pub sampling: String,
}

/// The origin shown under a model: `org/repo` for Hugging Face cache entries.
pub fn origin_of(path: &Path) -> String {
    for part in path.components() {
        let s = part.as_os_str().to_string_lossy();
        if let Some(rest) = s.strip_prefix("models--") {
            return rest.replacen("--", "/", 1);
        }
    }
    path.parent().map(tilde).unwrap_or_default()
}

/// A GGUF found on disk, before its header is checked.
pub struct Cand {
    pub path: PathBuf,
    pub name: String,
    pub size: u64,
}

pub fn walk(dir: &Path, depth: usize, seen: &mut HashSet<PathBuf>, out: &mut Vec<Cand>) {
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
pub fn is_drafter(lower: &str) -> bool {
    lower.contains("dspark")
        || lower.contains("dflash")
        || lower.contains("speculator")
        || lower.starts_with("mtp-")
}

pub struct Drafter {
    /// The checkpoint directory (an HF cache entry works as is).
    pub path: PathBuf,
    pub label: String,
    pub kind: &'static str,
    pub config: Option<DraftConfig>,
    /// Why this checkpoint cannot be loaded at all.
    pub problem: Option<String>,
    /// Rows in its Markov output head, i.e. the vocabulary it was trained on.
    pub head_rows: Option<usize>,
    /// GGUF drafters only: why it cannot serve each model checked so far.
    pub fits: Mutex<HashMap<PathBuf, Option<String>>>,
}

impl Drafter {
    /// A llama.cpp DFlash / DFlash2 GGUF rather than a checkpoint directory.
    pub fn is_gguf(&self) -> bool {
        self.path.is_file()
    }

    /// A GGUF drafter in the model's own folder, paired with it the way a
    /// sibling mmproj is.
    pub fn is_sibling(&self, m: &Model) -> bool {
        self.is_gguf() && self.path.parent() == m.path.parent()
    }
}

/// A DFlash / DFlash2 GGUF, told apart by its header alone.
pub fn gguf_drafter(path: &Path) -> Option<Drafter> {
    let file = fs::File::open(path).ok()?;
    let gguf = GgufFile::from_reader_skipping_arrays(file, 4096).ok()?;
    if !is_dflash(&gguf) {
        return None;
    }
    let (config, problem) = match DraftConfig::from_gguf(&gguf) {
        Ok(c) => (Some(c), None),
        Err(e) => (None, Some(e.to_string())),
    };
    Some(Drafter {
        kind: match config.as_ref().map(|c| c.kind) {
            Some(DraftKind::DFlash2 { .. }) => "dflash2",
            _ => "dflash",
        },
        label: path
            .file_stem()
            .map_or(String::new(), |s| s.to_string_lossy().into_owned()),
        path: path.to_path_buf(),
        config,
        problem,
        head_rows: None,
        fits: Mutex::default(),
    })
}

/// GGUFs worth opening as possible drafters: drafter-named, or small enough
/// to be one (headers decide).
fn maybe_gguf_drafter(lower: &str, size: u64) -> bool {
    lower.ends_with(".gguf")
        && !lower.contains("mmproj")
        && !lower.contains("vocab")
        && (is_drafter(lower) || size < (8 << 30))
}

/// The folder holding `config.json` (an HF cache entry resolves to its newest snapshot).
fn draft_root(dir: &Path) -> PathBuf {
    if dir.join("config.json").exists() {
        dir.to_path_buf()
    } else {
        let rev = fs::read_to_string(dir.join("refs/main")).unwrap_or_default();
        dir.join("snapshots").join(rev.trim())
    }
}

/// Vocabulary rows of the drafter's Markov head, read from the safetensors
/// header alone (no weights are loaded).
pub fn head_rows(dir: &Path) -> Option<usize> {
    use std::io::Read;
    let mut file = fs::File::open(draft_root(dir).join("model.safetensors")).ok()?;
    let mut len = [0u8; 8];
    file.read_exact(&mut len).ok()?;
    let n = u64::from_le_bytes(len) as usize;
    if n == 0 || n > 64 << 20 {
        return None;
    }
    let mut header = vec![0u8; n];
    file.read_exact(&mut header).ok()?;
    let json: Json = serde_json::from_slice(&header).ok()?;
    json.as_object()?
        .iter()
        .find(|(k, _)| k.ends_with("markov_w2.weight"))
        .and_then(|(_, v)| v["shape"][0].as_u64())
        .map(|r| r as usize)
}

pub fn read_draft(dir: &Path) -> (Option<DraftConfig>, Option<String>) {
    let root = draft_root(dir);
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

pub fn find_drafters(
    dir: &Path,
    depth: usize,
    seen: &mut HashSet<PathBuf>,
    out: &mut Vec<Drafter>,
) {
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
        let lower = name.to_ascii_lowercase();
        if meta.is_file() {
            if maybe_gguf_drafter(&lower, meta.len()) {
                push_gguf_drafter(&path, seen, out);
            }
            continue;
        }
        let hf = lower.starts_with("models--");
        // A `-GGUF` HF repo holds drafter files, not a checkpoint.
        let checkpoint = if hf {
            !lower.contains("gguf")
        } else {
            path.join("config.json").exists()
        };
        if is_drafter(&lower) && checkpoint {
            let canonical = fs::canonicalize(&path).unwrap_or_else(|_| path.clone());
            if seen.insert(canonical) {
                let (config, problem) = read_draft(&path);
                out.push(Drafter {
                    head_rows: head_rows(&path),
                    config,
                    problem,
                    label: origin_of(&path.join("x")),
                    kind: if lower.contains("dflash") {
                        "dflash"
                    } else {
                        "dspark"
                    },
                    path,
                    fits: Mutex::default(),
                });
            }
        } else if depth > 0
            && !matches!(
                name.as_str(),
                "node_modules" | "target" | "Library" | "blobs"
            )
        {
            find_drafters(&path, depth - 1, seen, out);
        }
    }
}

fn push_gguf_drafter(path: &Path, seen: &mut HashSet<PathBuf>, out: &mut Vec<Drafter>) {
    let canonical = fs::canonicalize(path).unwrap_or_else(|_| path.to_path_buf());
    if !seen.contains(&canonical)
        && let Some(d) = gguf_drafter(path)
    {
        seen.insert(canonical);
        out.push(d);
    }
}

pub fn scan_drafters(extra: &[String]) -> Vec<Drafter> {
    let (mut seen, mut out) = (HashSet::new(), Vec::new());
    for (root, depth) in roots(extra) {
        if root.is_dir() {
            find_drafters(&root, depth, &mut seen, &mut out);
        } else if root.is_file() {
            push_gguf_drafter(&root, &mut seen, &mut out);
        }
    }
    out.sort_by_key(|a| a.label.to_lowercase());
    out
}

/// Why this drafter cannot serve this model (the checks `DraftModel::load`
/// makes against the target); None means it fits.
pub fn drafter_problem(d: &Drafter, m: &Model) -> Option<String> {
    if let Some(p) = &d.problem {
        return Some(p.clone());
    }
    let (c, t) = (d.config.as_ref()?, m.hybrid.as_ref()?);
    if d.is_gguf() {
        // The loader's own checks; memoised, as lists redraw every frame.
        let mut fits = d.fits.lock().unwrap_or_else(|e| e.into_inner());
        return fits
            .entry(m.path.clone())
            .or_insert_with(|| {
                DraftModel::memory(&d.path, t, 512, DraftOptions::default())
                    .err()
                    .or_else(|| check_drafter_tokenizer(&d.path, &m.tokenizer).err())
                    .map(|e| e.to_string())
            })
            .clone();
    }
    if c.hidden != t.hidden {
        return Some(format!("hidden size {} ≠ model's {}", c.hidden, t.hidden));
    }
    if c.target_layers.iter().any(|&l| l >= t.layers) {
        return Some("reads layers the model does not have".into());
    }
    if c.mask_token as usize >= t.vocab {
        return Some("mask token outside the model's vocabulary".into());
    }
    if let Some(rows) = d.head_rows.filter(|&r| r < t.vocab) {
        return Some(format!(
            "trained on a {rows}-token vocabulary, the model has {}",
            t.vocab
        ));
    }
    None
}

pub fn roots(extra: &[String]) -> Vec<(PathBuf, usize)> {
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

pub fn scan(extra: &[String]) -> Vec<Model> {
    let mut seen = HashSet::new();
    let mut out = Vec::new();
    for (root, depth) in roots(extra) {
        if root.is_file() {
            walk_file(&root, &mut seen, &mut out);
        } else {
            walk(&root, depth, &mut seen, &mut out);
        }
    }
    // Headers are read in parallel; each still walks past the tokenizer arrays.
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
    // DFlash GGUFs are drafters (see `scan_drafters`), not chat models.
    models.retain(|m| m.info.as_ref().is_none_or(|i| i.arch != DFLASH_ARCH));
    models.sort_by_key(|m| (m.problem.is_some(), m.name.to_lowercase()));
    models
}

pub fn walk_file(path: &Path, seen: &mut HashSet<PathBuf>, out: &mut Vec<Cand>) {
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

pub fn as_u64(v: &MetadataValue) -> Option<u64> {
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

pub struct Checked {
    pub info: Option<Info>,
    pub hybrid: Option<HybridConfig>,
    pub mtp: bool,
    pub mtp_why: String,
    pub problem: Option<String>,
    pub tokenizer: BTreeMap<String, MetadataValue>,
}

/// `general.architecture` of llama.cpp's DFlash / DFlash2 drafters.
const DFLASH_ARCH: &str = "dflash";

pub fn type_name(id: u32) -> String {
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
pub const SUPPORTED: [u32; 10] = [0, 2, 8, 12, 13, 14, 18, 21, 22, 23];

/// Mirrors what `Runtime::load` requires, without reading any weights.
pub fn check(path: &Path) -> Checked {
    let blocked = |problem: String| Checked {
        info: None,
        hybrid: None,
        mtp: false,
        mtp_why: String::new(),
        problem: Some(problem),
        tokenizer: BTreeMap::new(),
    };
    let opened = fs::File::open(path)
        .map_err(|e| ferrum::Error::Gguf(format!("{}: {e}", path.display())))
        .and_then(|f| {
            // The tokenizer's big arrays are never looked at here.
            GgufReader::from_reader_skipping_arrays(
                std::io::BufReader::with_capacity(1 << 20, f),
                4096,
            )
        });
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
        tokenizer: md
            .iter()
            .filter(|(k, v)| {
                k.starts_with("tokenizer.") && !matches!(v, MetadataValue::Array { .. })
            })
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect(),
    };
    if arch == DFLASH_ARCH {
        result.problem = Some("a DFlash drafter, not a chat model".into());
        return result;
    }
    if !matches!(arch.as_str(), "qwen35" | "qwen35moe" | "flint") {
        result.problem = Some(format!(
            "architecture {arch}: needs qwen35, qwen35moe or flint"
        ));
        return result;
    }
    let pre = text("tokenizer.ggml.pre");
    let pre_ok = match arch.as_str() {
        "flint" => pre.as_deref() == Some("flint"),
        _ => pre.as_deref() == Some("qwen35"),
    };
    if text("tokenizer.ggml.model").as_deref() != Some("gpt2") || !pre_ok {
        result.problem = Some(format!("tokenizer is not the {arch} gpt2 tokenizer"));
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
        // BF16 routers are widened to F32 when the weights load.
        let bf16_router = t.type_id == 30 && t.name.ends_with("ffn_gate_inp.weight");
        let supported = if t.dimensions.len() <= 1 {
            t.type_id == 0
        } else {
            SUPPORTED.contains(&t.type_id) || bf16_router
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
            "unsupported quant {} (runs Q4_0 Q8_0 Q4_K Q5_K Q6_K IQ3_XXS IQ3_S IQ2_S IQ4_XS)",
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

#[cfg(test)]
mod tests {
    use super::*;

    /// The Flint GGUF from the Hugging Face cache, when it is present (it is
    /// 18.5 GB, so CI skips this). Its BF16 routers once made it "unsupported".
    #[test]
    fn cached_flint_gguf_passes_the_quant_check() {
        let cache = std::env::var_os("HOME")
            .map(std::path::PathBuf::from)
            .unwrap_or_default()
            .join(".cache/huggingface/hub/models--unsloth--flint-GGUF/snapshots");
        let Some(path) = std::fs::read_dir(&cache).ok().and_then(|dirs| {
            dirs.flatten()
                .map(|d| d.path().join("flint-UD-Q3_K_XL.gguf"))
                .find(|p| p.exists())
        }) else {
            return;
        };
        let checked = check(&path);
        assert_eq!(checked.problem, None);
        assert!(checked.hybrid.is_some());
    }
}
