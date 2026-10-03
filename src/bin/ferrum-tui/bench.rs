//! Benchmark engine: finds the fastest drafter, draft depth and prefill chunk
//! for one model on this Mac. Each drafter is loaded once and its depth swept
//! with `Session::depth_cap`; every run is greedy, so a correct speculative
//! run must reproduce the no-drafter output exactly.
use crate::*;
use ferrum::hybrid::{
    plan::PlanOptions,
    runtime::{ChatHooks, ChatRequest, Delta, DraftSource, Runtime, SpecOptions},
};
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
    mpsc::{Receiver, Sender, channel},
};
use std::thread::JoinHandle;

/// Prompts with different draft-friendliness: code, prose and rigid structure.
pub const PROMPTS: [(&str, &str); 3] = [
    (
        "code",
        "Write a Python module with a class LRUCache (get, put, delete, __len__) using an OrderedDict, with type hints and docstrings, followed by three unittest test cases for it.",
    ),
    (
        "prose",
        "Explain in plain language how a large language model generates text one token at a time, using an everyday analogy, in several paragraphs.",
    ),
    (
        "data",
        "Produce a JSON array of 10 objects describing planets of the solar system, each with the fields name, diameter_km, moons, has_rings and a one-sentence description.",
    ),
];

const LONG_PARAGRAPH: &str = "The scheduler walks the run queue, picks the task with the smallest virtual runtime, \
and lets it execute until its time slice ends or it blocks on input. Memory is managed in pages; \
a page table maps virtual addresses to physical frames, and a translation cache keeps recent mappings close to the core. \
When free frames run low the kernel evicts pages that were not used recently, writing dirty ones back to disk first. ";

/// Most MTP drafts worth trying.
const MTP_MAX_DEPTH: usize = 6;

#[derive(Clone, Debug, PartialEq)]
pub enum Source {
    Baseline,
    Mtp,
    Drafter {
        path: PathBuf,
        label: String,
        kind: &'static str,
        max_depth: usize,
    },
}

#[derive(Clone, Debug)]
pub struct Candidate {
    pub source: Source,
    /// Drafter weight precision: "q4_0" or "q8_0".
    pub quant: &'static str,
    pub on: bool,
}

impl Candidate {
    pub fn label(&self) -> String {
        match &self.source {
            Source::Baseline => "no drafter".into(),
            Source::Mtp => "MTP head".into(),
            Source::Drafter { label, kind, .. } => format!("{label} ({kind})"),
        }
    }

    /// Deepest draft worth sweeping (0 for the baseline).
    pub fn max_depth(&self) -> usize {
        match &self.source {
            Source::Baseline => 0,
            Source::Mtp => MTP_MAX_DEPTH,
            Source::Drafter { max_depth, .. } => *max_depth,
        }
    }

    /// The value of the `draft` setting that selects this candidate.
    pub fn draft_value(&self) -> String {
        match &self.source {
            Source::Baseline => "off".into(),
            Source::Mtp => "mtp".into(),
            Source::Drafter { path, .. } => path.display().to_string(),
        }
    }

    fn spec(&self, depth: Option<usize>) -> Option<SpecOptions> {
        let source = match &self.source {
            Source::Baseline => return None,
            Source::Mtp => DraftSource::Mtp,
            Source::Drafter { path, .. } => DraftSource::Checkpoint(path.clone()),
        };
        let mut spec = SpecOptions::new(source);
        spec.max_drafts = depth;
        spec.draft.quant = if self.quant == "q8_0" {
            ferrum::hybrid::draft::DraftQuant::Q8_0
        } else {
            ferrum::hybrid::draft::DraftQuant::Q4_0
        };
        Some(spec)
    }
}

/// The depths to try for a drafter that can propose up to `max`.
pub fn depth_list(max: usize, quick: bool) -> Vec<usize> {
    if max == 0 {
        return Vec::new();
    }
    let mut depths: Vec<usize> = if quick {
        vec![2, max / 2, max]
    } else if max <= 8 {
        (1..=max).collect()
    } else {
        vec![1, 2, 3, 4, 5, 6, 8, 10, 12, 14, max]
    };
    depths.retain(|d| (1..=max).contains(d));
    depths.sort_unstable();
    depths.dedup();
    depths
}

#[derive(Clone)]
pub struct Plan {
    pub model: PathBuf,
    pub candidates: Vec<Candidate>,
    pub quick: bool,
    /// Tokens generated per prompt.
    pub tokens: usize,
    /// Also sweep the prefill chunk for the winner.
    pub chunks: bool,
    pub context: usize,
}

impl Plan {
    /// Rough run time in seconds, assuming a typical 30 tokens/s decode.
    pub fn estimate_secs(&self) -> usize {
        let per_run = PROMPTS.len() * (self.tokens / 30 + 1);
        let mut total = 0;
        for c in self.candidates.iter().filter(|c| c.on) {
            let depths = match c.source {
                Source::Baseline => 1,
                _ => depth_list(c.max_depth(), self.quick).len(),
            };
            total += 25 + depths * per_run;
        }
        if self.chunks {
            total += 4 * 25;
        }
        total
    }
}

#[derive(Clone, Debug)]
pub struct Row {
    pub cand: usize,
    pub label: String,
    pub draft_value: String,
    pub quant: String,
    /// None: no drafter.
    pub depth: Option<usize>,
    /// Decoded tokens per second over all prompts.
    pub tps: f64,
    /// Mean tokens per verify step (1.0 = no gain).
    pub accept: Option<f64>,
    pub per_prompt: Vec<f64>,
    /// Output identical to the no-drafter run.
    pub same: Option<bool>,
    pub tokens: usize,
}

#[derive(Clone, Debug)]
pub struct ChunkRow {
    pub chunk: usize,
    pub prefill_tps: f64,
}

pub enum Event {
    Stage(String),
    Loaded {
        context: usize,
        weights_gib: f64,
        draft_gib: f64,
    },
    Row(Row),
    Chunk(ChunkRow),
    Skipped(String, String),
    Done,
}

struct Stop<'a>(&'a AtomicBool);

impl ChatHooks for Stop<'_> {
    fn delta(&mut self, _: Delta) -> ferrum::Result<()> {
        Ok(())
    }
    fn cancelled(&self) -> bool {
        self.0.load(Ordering::SeqCst)
    }
}

fn fnv(text: &str) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for b in text.bytes() {
        h = (h ^ b as u64).wrapping_mul(0x100000001b3);
    }
    h
}

struct Sample {
    tokens: usize,
    decode: f64,
    prefill_tps: f64,
    hash: u64,
}

/// Greedy, no-thinking generation of one prompt from a cold session.
fn generate(
    rt: &mut Runtime,
    prompt: &str,
    tokens: usize,
    stop: &AtomicBool,
) -> Result<Sample, String> {
    rt.session.reset();
    let mut vars = Map::new();
    vars.insert("enable_thinking".into(), Json::Bool(false));
    let request = ChatRequest {
        messages: vec![json!({"role": "user", "content": prompt})],
        tools: Vec::new(),
        max_tokens: tokens,
        sampling: ferrum::hybrid::session::SamplingParams::default(),
        template_vars: vars,
        stop: Vec::new(),
        reasoning_budget: None,
        raw_reasoning: false,
    };
    let r = rt.chat(&request, Stop(stop)).map_err(|e| e.to_string())?;
    let c = &r.completion;
    Ok(Sample {
        tokens: c.tokens.len(),
        decode: c.decode.as_secs_f64().max(1e-9),
        prefill_tps: c.prompt_tokens.saturating_sub(c.reused_tokens) as f64
            / c.prefill.as_secs_f64().max(1e-9),
        hash: fnv(&r.text),
    })
}

fn load(
    plan: &Plan,
    cand: &Candidate,
    depth: Option<usize>,
    chunk: Option<usize>,
) -> Result<Runtime, String> {
    let mut options = PlanOptions {
        context: Some(plan.context),
        ..Default::default()
    };
    if let Some(c) = chunk {
        options.chunk = c;
    }
    let attempt = |depth: Option<usize>| {
        let spec = cand.spec(depth);
        Runtime::load_speculative(&plan.model, options, spec.as_ref())
    };
    match attempt(depth) {
        Ok(rt) => Ok(rt),
        // A deep verify buffer may not fit next to the model: fall back to the drafter's own default.
        Err(_) if depth.is_some() => attempt(None).map_err(|e| e.to_string()),
        Err(e) => Err(e.to_string()),
    }
}

fn run(plan: Plan, tx: Sender<Event>, stop: Arc<AtomicBool>) {
    let send = |e: Event| {
        let _ = tx.send(e);
    };
    let mut baseline_hashes: Vec<u64> = Vec::new();
    let mut winner: Option<(Candidate, usize, f64)> = None;
    for (ci, cand) in plan.candidates.iter().filter(|c| c.on).enumerate() {
        if stop.load(Ordering::SeqCst) {
            break;
        }
        send(Event::Stage(format!("loading {}", cand.label())));
        let deepest = (cand.max_depth() > 0).then(|| cand.max_depth());
        let mut rt = match load(&plan, cand, deepest, None) {
            Ok(rt) => rt,
            Err(e) => {
                send(Event::Skipped(cand.label(), e));
                continue;
            }
        };
        if let Err(e) = rt.warmup() {
            send(Event::Skipped(cand.label(), e.to_string()));
            continue;
        }
        let reach = rt.session.drafter().map_or(0, |d| d.max_drafts());
        send(Event::Loaded {
            context: rt.loaded.plan.context,
            weights_gib: rt.loaded.plan.weights as f64 / (1u64 << 30) as f64,
            draft_gib: rt.draft_loaded_bytes as f64 / (1u64 << 30) as f64,
        });
        let depths: Vec<Option<usize>> = match cand.source {
            Source::Baseline => vec![None],
            _ => depth_list(reach, plan.quick)
                .into_iter()
                .map(Some)
                .collect(),
        };
        for depth in depths {
            if stop.load(Ordering::SeqCst) {
                break;
            }
            rt.session.depth_cap = depth;
            rt.session.speculate = depth.is_some();
            send(Event::Stage(format!(
                "{} · {}",
                cand.label(),
                depth.map_or("baseline".to_string(), |d| format!("depth {d}"))
            )));
            let (mut samples, mut per_prompt) = (Vec::new(), Vec::new());
            let (mut steps, mut accepted) = (0usize, 0usize);
            let mut failed = None;
            for (name, prompt) in PROMPTS {
                send(Event::Stage(format!(
                    "{} · {} · {name}",
                    cand.label(),
                    depth.map_or("baseline".to_string(), |d| format!("depth {d}"))
                )));
                match generate(&mut rt, prompt, plan.tokens, &stop) {
                    Ok(s) => {
                        per_prompt.push(s.tokens as f64 / s.decode);
                        let st = &rt.session.spec_stats;
                        steps += st.steps;
                        accepted += st.accepted;
                        samples.push(s);
                    }
                    Err(e) => {
                        failed = Some(e);
                        break;
                    }
                }
                if stop.load(Ordering::SeqCst) {
                    break;
                }
            }
            if let Some(e) = failed {
                send(Event::Skipped(cand.label(), e));
                break;
            }
            if samples.len() < PROMPTS.len() {
                break; // cancelled part-way: do not report a partial row
            }
            let tokens: usize = samples.iter().map(|s| s.tokens).sum();
            let secs: f64 = samples.iter().map(|s| s.decode).sum();
            let hashes: Vec<u64> = samples.iter().map(|s| s.hash).collect();
            let same = if cand.source == Source::Baseline {
                baseline_hashes = hashes;
                None
            } else if baseline_hashes.is_empty() {
                None
            } else {
                Some(baseline_hashes == hashes)
            };
            let tps = tokens as f64 / secs;
            if let Some(d) = depth
                && winner.as_ref().is_none_or(|(_, _, best)| tps > *best)
            {
                winner = Some((cand.clone(), d, tps));
            }
            send(Event::Row(Row {
                cand: ci,
                label: cand.label(),
                draft_value: cand.draft_value(),
                quant: if depth.is_some() {
                    cand.quant.into()
                } else {
                    String::new()
                },
                depth,
                tps,
                accept: (steps > 0).then(|| (accepted + steps) as f64 / steps as f64),
                per_prompt,
                same,
                tokens,
            }));
        }
        drop(rt);
    }
    if plan.chunks && !stop.load(Ordering::SeqCst) {
        // Prefill speed depends on how many prompt rows go through one pass.
        let (cand, depth) = match &winner {
            Some((c, d, _)) => (c.clone(), Some(*d)),
            None => (plan.candidates[0].clone(), None),
        };
        let long = LONG_PARAGRAPH.repeat(14);
        for chunk in [256usize, 512, 1024, 2048] {
            if stop.load(Ordering::SeqCst) {
                break;
            }
            send(Event::Stage(format!("prefill chunk {chunk}")));
            let Ok(mut rt) = load(&plan, &cand, depth, Some(chunk)) else {
                continue;
            };
            if rt.warmup().is_err() {
                continue;
            }
            rt.session.depth_cap = depth;
            rt.session.speculate = depth.is_some();
            if let Ok(s) = generate(&mut rt, &long, 1, &stop) {
                send(Event::Chunk(ChunkRow {
                    chunk,
                    prefill_tps: s.prefill_tps,
                }));
            }
        }
    }
    send(Event::Done);
}

pub struct Running {
    pub rx: Receiver<Event>,
    pub stop: Arc<AtomicBool>,
    pub handle: Option<JoinHandle<()>>,
}

pub fn spawn(plan: Plan) -> Result<Running, String> {
    let (tx, rx) = channel();
    let stop = Arc::new(AtomicBool::new(false));
    let flag = stop.clone();
    let handle = std::thread::Builder::new()
        .name("ferrum-bench".into())
        .spawn(move || run(plan, tx, flag))
        .map_err(|e| e.to_string())?;
    Ok(Running {
        rx,
        stop,
        handle: Some(handle),
    })
}

// ---- results ----

/// The fastest row whose output matches plain decoding (rows that differ are
/// only considered when nothing matches); on a near tie, the shallower, cheaper one.
pub fn best(rows: &[Row]) -> Option<&Row> {
    let exact = |r: &&Row| r.same != Some(false);
    let pool: Vec<&Row> = if rows.iter().any(|r| r.same != Some(false)) {
        rows.iter().filter(exact).collect()
    } else {
        rows.iter().collect()
    };
    let top = pool.iter().map(|r| r.tps).fold(0.0, f64::max);
    pool.into_iter()
        .filter(|r| r.tps >= top * 0.99)
        .min_by_key(|r| (r.depth.unwrap_or(0), r.quant == "q8_0"))
}

pub fn baseline(rows: &[Row]) -> Option<&Row> {
    rows.iter().find(|r| r.depth.is_none())
}

pub fn best_chunk(chunks: &[ChunkRow]) -> Option<usize> {
    let top = chunks.iter().map(|c| c.prefill_tps).fold(0.0, f64::max);
    chunks
        .iter()
        .filter(|c| c.prefill_tps >= top * 0.97)
        .map(|c| c.chunk)
        .min()
}

/// The settings that reproduce a row, as favorite values.
pub fn settings_for(row: &Row, chunk: Option<usize>) -> Map<String, Json> {
    let mut v = Map::new();
    v.insert("draft".into(), json!(row.draft_value));
    if let Some(d) = row.depth {
        v.insert("draft_max".into(), json!(d.to_string()));
        if row.quant == "q8_0" {
            v.insert("draft_quant".into(), json!("q8_0"));
        }
    }
    if let Some(c) = chunk {
        v.insert("chunk".into(), json!(c.to_string()));
    }
    v
}

// ---- machine and saved results ----

#[derive(Clone, Debug, PartialEq)]
pub struct Machine {
    pub chip: String,
    pub mem_gb: u64,
}

fn sysctl(key: &str) -> String {
    Command::new("/usr/sbin/sysctl")
        .args(["-n", key])
        .output()
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .unwrap_or_default()
}

impl Machine {
    pub fn detect() -> Self {
        let chip = sysctl("machdep.cpu.brand_string");
        let mem = sysctl("hw.memsize").parse::<u64>().unwrap_or(0);
        Self {
            chip: if chip.is_empty() {
                "this Mac".into()
            } else {
                chip
            },
            mem_gb: mem >> 30,
        }
    }

    pub fn label(&self) -> String {
        format!("{} · {} GB", self.chip, self.mem_gb)
    }
}

pub fn results_path() -> PathBuf {
    home().join(".config/ferrum/bench.json")
}

#[derive(Clone, Debug)]
pub struct Saved {
    pub machine: String,
    pub model: String,
    pub when: u64,
    pub best: String,
    pub speedup: f64,
    pub values: Map<String, Json>,
}

pub fn load_saved() -> Vec<Saved> {
    let Ok(text) = fs::read_to_string(results_path()) else {
        return Vec::new();
    };
    let json: Json = serde_json::from_str(&text).unwrap_or(Json::Null);
    json["runs"]
        .as_array()
        .map(|a| {
            a.iter()
                .map(|r| Saved {
                    machine: r["machine"].as_str().unwrap_or("").into(),
                    model: r["model"].as_str().unwrap_or("").into(),
                    when: r["when"].as_u64().unwrap_or(0),
                    best: r["best"].as_str().unwrap_or("").into(),
                    speedup: r["speedup"].as_f64().unwrap_or(1.0),
                    values: r["values"].as_object().cloned().unwrap_or_default(),
                })
                .collect()
        })
        .unwrap_or_default()
}

/// Keep the newest result per (machine, model), and at most 40.
pub fn record(saved: &mut Vec<Saved>, run: Saved) {
    saved.retain(|s| !(s.machine == run.machine && s.model == run.model));
    saved.insert(0, run);
    saved.truncate(40);
}

pub fn write_saved(saved: &[Saved]) {
    let runs: Vec<Json> = saved
        .iter()
        .map(|s| {
            json!({
                "machine": s.machine, "model": s.model, "when": s.when,
                "best": s.best, "speedup": s.speedup, "values": s.values,
            })
        })
        .collect();
    let path = results_path();
    if let Some(dir) = path.parent() {
        fs::create_dir_all(dir).ok();
    }
    fs::write(
        path,
        serde_json::to_string_pretty(&json!({"runs": runs})).unwrap_or_default(),
    )
    .ok();
}

pub fn now() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn row(label: &str, depth: Option<usize>, tps: f64) -> Row {
        Row {
            cand: 0,
            label: label.into(),
            draft_value: if depth.is_some() {
                "/d".into()
            } else {
                "off".into()
            },
            quant: if depth.is_some() {
                "q4_0".into()
            } else {
                String::new()
            },
            depth,
            tps,
            accept: depth.map(|_| 2.0),
            per_prompt: vec![tps; 3],
            same: None,
            tokens: 600,
        }
    }

    #[test]
    fn depth_lists() {
        assert_eq!(depth_list(0, false), Vec::<usize>::new());
        assert_eq!(depth_list(6, false), vec![1, 2, 3, 4, 5, 6]);
        assert_eq!(depth_list(6, true), vec![2, 3, 6]);
        assert_eq!(depth_list(1, true), vec![1]);
        let deep = depth_list(15, false);
        assert_eq!(deep.first(), Some(&1));
        assert_eq!(deep.last(), Some(&15));
        assert!(deep.windows(2).all(|w| w[0] < w[1]));
        assert!(depth_list(15, true).iter().all(|d| *d <= 15));
    }

    #[test]
    fn picks_the_best_and_prefers_shallow_on_ties() {
        let rows = vec![
            row("none", None, 40.0),
            row("a", Some(3), 80.0),
            row("a", Some(5), 99.0),
            row("a", Some(7), 100.0),
        ];
        // 99 is within 1% of 100: take the shallower one.
        assert_eq!(best(&rows).unwrap().depth, Some(5));
        assert_eq!(baseline(&rows).unwrap().tps, 40.0);
        // A faster row that changes the output is not recommended over an exact one.
        let mut off = row("a", Some(9), 120.0);
        off.same = Some(false);
        let mut ok = row("a", Some(2), 60.0);
        ok.same = Some(true);
        let mixed = vec![row("none", None, 40.0), ok, off];
        assert_eq!(best(&mixed).unwrap().depth, Some(2));
        let slow = vec![row("none", None, 40.0), row("a", Some(3), 30.0)];
        assert_eq!(
            best(&slow).unwrap().depth,
            None,
            "no drafter can be the answer"
        );
    }

    #[test]
    fn chunk_choice_and_settings() {
        let chunks = vec![
            ChunkRow {
                chunk: 256,
                prefill_tps: 700.0,
            },
            ChunkRow {
                chunk: 512,
                prefill_tps: 990.0,
            },
            ChunkRow {
                chunk: 1024,
                prefill_tps: 1000.0,
            },
            ChunkRow {
                chunk: 2048,
                prefill_tps: 1001.0,
            },
        ];
        assert_eq!(
            best_chunk(&chunks),
            Some(512),
            "within 3% of the top: the smallest memory"
        );
        let v = settings_for(&row("a", Some(4), 90.0), Some(512));
        assert_eq!(v["draft"], "/d");
        assert_eq!(v["draft_max"], "4");
        assert_eq!(v["chunk"], "512");
        assert!(!v.contains_key("draft_quant"));
        let v = settings_for(&row("none", None, 40.0), None);
        assert_eq!(v["draft"], "off");
        assert!(!v.contains_key("draft_max"));
    }

    #[test]
    fn estimates_grow_with_the_sweep() {
        let cand = |source| Candidate {
            source,
            quant: "q4_0",
            on: true,
        };
        let drafter = Source::Drafter {
            path: "/d".into(),
            label: "x".into(),
            kind: "dspark",
            max_depth: 8,
        };
        let mut plan = Plan {
            model: "/m".into(),
            candidates: vec![cand(Source::Baseline), cand(drafter)],
            quick: true,
            tokens: 192,
            chunks: false,
            context: 8192,
        };
        let quick = plan.estimate_secs();
        plan.quick = false;
        assert!(plan.estimate_secs() > quick);
        plan.chunks = true;
        assert!(plan.estimate_secs() > quick + 90);
    }

    #[test]
    fn saved_results_replace_per_model_and_machine() {
        let run = |model: &str, speed: f64| Saved {
            machine: "M5".into(),
            model: model.into(),
            when: 1,
            best: "x".into(),
            speedup: speed,
            values: Map::new(),
        };
        let mut saved = Vec::new();
        record(&mut saved, run("a", 1.5));
        record(&mut saved, run("b", 2.0));
        record(&mut saved, run("a", 1.8));
        assert_eq!(saved.len(), 2);
        assert_eq!(saved[0].model, "a");
        assert_eq!(saved[0].speedup, 1.8);
    }

    #[test]
    fn detects_this_machine() {
        let m = Machine::detect();
        assert!(m.mem_gb > 0 && !m.chip.is_empty());
        assert!(m.label().contains("GB"));
    }
}
