use crate::*;

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
}

pub fn read_draft(dir: &Path) -> (Option<DraftConfig>, Option<String>) {
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

pub fn scan_drafters(extra: &[String]) -> Vec<Drafter> {
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
pub fn drafter_problem(d: &Drafter, m: &Model) -> Option<String> {
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
}

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
pub const SUPPORTED: [u32; 8] = [0, 2, 8, 12, 13, 14, 21, 23];

/// Mirrors what `Runtime::load` requires, without reading any weights.
pub fn check(path: &Path) -> Checked {
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
