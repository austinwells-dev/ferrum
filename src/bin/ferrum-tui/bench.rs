//! Benchmark engine: finds the fastest drafter, draft depth and prefill chunk
//! for one model on this Mac. Each drafter is loaded once and its depth swept
//! with `Session::depth_cap`, then MTP and drafter models also try draft trees
//! (a fixed tree of 15 guesses, and KFlash's learned budget); every run is greedy, so a correct speculative
//! run reproduces the no-drafter output, up to near-tie flips from batched
//! verification (see [`Match`]). Near-tie rows that would win are then
//! asked the answer-check questions ([`eval`]) before they are recommended.
use crate::eval::{self, Verdict};
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
/// Most prompt-lookup drafts worth trying.
const LOOKUP_MAX_DEPTH: usize = 6;

/// A draft tree instead of a chain (`--draft-tree`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Tree {
    /// Up to this many guesses per step.
    Fixed(usize),
    /// KFlash: as many guesses as pay each cycle, learned and cached (for
    /// DFlash2 also a 16-row block over a reduced draft vocabulary).
    Auto,
}

impl Tree {
    /// The `--draft-tree` / `draft_tree` setting value.
    pub fn value(self) -> String {
        match self {
            Self::Fixed(n) => n.to_string(),
            Self::Auto => "auto".into(),
        }
    }

    pub fn label(self) -> String {
        match self {
            Self::Fixed(n) => format!("tree {n}"),
            Self::Auto => "KFlash trees".into(),
        }
    }

    /// For narrow columns.
    pub fn short(self) -> String {
        match self {
            Self::Fixed(n) => format!("T{n}"),
            Self::Auto => "auto".into(),
        }
    }
}

/// The trees the benchmark tries after a drafter's depth sweep.
pub const TREES: [Tree; 2] = [Tree::Fixed(15), Tree::Auto];

#[derive(Clone, Debug, PartialEq)]
pub enum Source {
    Baseline,
    Mtp,
    /// Prompt lookup: drafts copied from the context, no drafter model.
    Lookup,
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
            Source::Lookup => "prompt lookup".into(),
            Source::Drafter { label, kind, .. } => format!("{label} ({kind})"),
        }
    }

    /// The drafter precision as shown, or "–" for prompt lookup (no weights).
    pub fn quant_label(&self) -> &'static str {
        match self.source {
            Source::Lookup => "–",
            _ => self.quant,
        }
    }

    /// Deepest draft worth sweeping (0 for the baseline).
    pub fn max_depth(&self) -> usize {
        match &self.source {
            Source::Baseline => 0,
            Source::Mtp => MTP_MAX_DEPTH,
            Source::Lookup => LOOKUP_MAX_DEPTH,
            Source::Drafter { max_depth, .. } => *max_depth,
        }
    }

    /// The value of the `draft` setting that selects this candidate.
    pub fn draft_value(&self) -> String {
        match &self.source {
            Source::Baseline => "off".into(),
            Source::Mtp => "mtp".into(),
            Source::Lookup => "lookup".into(),
            Source::Drafter { path, .. } => path.display().to_string(),
        }
    }

    /// Whether this source can draft trees in the benchmark (prompt lookup's
    /// trees are a test vehicle, not offered).
    pub fn trees(&self) -> bool {
        matches!(self.source, Source::Mtp | Source::Drafter { .. })
    }

    fn spec(&self, depth: Option<usize>, tree: Option<Tree>) -> Option<SpecOptions> {
        let source = match &self.source {
            Source::Baseline => return None,
            Source::Mtp => DraftSource::Mtp,
            Source::Lookup => DraftSource::Lookup,
            Source::Drafter { path, .. } => DraftSource::Checkpoint(path.clone()),
        };
        let mut spec = SpecOptions::new(source);
        spec.max_drafts = depth;
        spec.draft.quant = if self.quant == "q8_0" {
            ferrum::hybrid::draft::DraftQuant::Q8_0
        } else {
            ferrum::hybrid::draft::DraftQuant::Q4_0
        };
        let mut spec = Some(spec);
        if let Some(tree) = tree {
            // Valid by construction (1..=31 nodes or auto).
            SpecOptions::apply_flag(&mut spec, "--draft-tree", &tree.value())
                .expect("a benchmark tree is a valid --draft-tree");
        }
        spec
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
    pub kv_cache: KvCache,
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
            } + if c.trees() { TREES.len() } else { 0 };
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
    /// Index into `Plan::candidates` (ticked or not), as the queue lists them.
    pub cand: usize,
    pub label: String,
    pub draft_value: String,
    pub quant: String,
    /// None: no drafter. For a tree, the drafter's deepest guess.
    pub depth: Option<usize>,
    /// Drafted as a tree (None: a chain of `depth`).
    pub tree: Option<Tree>,
    /// Decoded tokens per second over all prompts.
    pub tps: f64,
    /// Mean tokens per verify step (1.0 = no gain).
    pub accept: Option<f64>,
    pub per_prompt: Vec<f64>,
    /// Output compared with the no-drafter run.
    pub same: Option<Match>,
    /// Answer check of a `Close` row (None: not checked).
    pub check: Option<Verdict>,
    pub tokens: usize,
}

impl Row {
    /// How it drafts: "depth 4", "tree 15", "KFlash trees" (None: no drafter).
    pub fn shape(&self) -> Option<String> {
        match (self.tree, self.depth) {
            (Some(t), _) => Some(t.label()),
            (None, Some(d)) => Some(format!("depth {d}")),
            (None, None) => None,
        }
    }

    /// [`Row::shape`] for narrow columns: "4", "T15", "auto" ("–": none).
    pub fn shape_short(&self) -> String {
        match (self.tree, self.depth) {
            (Some(t), _) => t.short(),
            (None, Some(d)) => d.to_string(),
            (None, None) => "–".into(),
        }
    }
}

/// How a speculative run's output compares with plain decoding.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Match {
    Exact,
    /// Diverged, but only after `CLOSE_PREFIX` of every answer matched: a
    /// near-tie flip from batched verification rounding, not a broken drafter
    /// (which goes wrong within the first few tokens).
    Close,
    Differs,
}

/// Share of each baseline answer that must match before a divergence counts
/// as numerical noise.
pub const CLOSE_PREFIX: f64 = 0.25;

/// Compares per-prompt token ids with the baseline's.
pub fn compare(base: &[Vec<u32>], run: &[Vec<u32>]) -> Match {
    if base == run {
        return Match::Exact;
    }
    let close = base.len() == run.len()
        && base.iter().zip(run).all(|(b, r)| {
            let shared = b.iter().zip(r).take_while(|(x, y)| x == y).count();
            shared as f64 >= b.len() as f64 * CLOSE_PREFIX
        });
    if close { Match::Close } else { Match::Differs }
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
    /// The answer check of the row with this index (in `Row` order).
    Checked(usize, Verdict),
    Chunk(ChunkRow),
    Skipped(String, String),
    Done,
}

pub struct Sample {
    tokens: usize,
    decode: f64,
    prefill_tps: f64,
    ids: Vec<u32>,
    /// The visible answer.
    pub text: String,
    /// Tool calls the answer made.
    pub calls: Vec<ferrum::hybrid::chat::ToolCall>,
}

/// Greedy generation of one prompt from a cold session. `thinking` is a
/// reasoning budget (None: thinking off); it is added to `tokens`.
pub fn generate(
    rt: &mut Runtime,
    prompt: &str,
    tokens: usize,
    thinking: Option<usize>,
    stop: &AtomicBool,
) -> Result<Sample, String> {
    generate_with_tools(rt, prompt, &[], tokens, thinking, stop)
}

/// Passes each piece of the reply on as it streams: (reasoning, text).
struct Streaming<'a> {
    stop: &'a AtomicBool,
    on: &'a mut dyn FnMut(bool, String),
}

impl ChatHooks for Streaming<'_> {
    fn delta(&mut self, delta: Delta) -> ferrum::Result<()> {
        match delta {
            Delta::Reasoning(t) => (self.on)(true, t),
            Delta::Content(t) => (self.on)(false, t),
        }
        Ok(())
    }
    fn cancelled(&self) -> bool {
        self.stop.load(Ordering::SeqCst)
    }
}

/// [`generate`] with OpenAI-style tool definitions offered to the model.
pub fn generate_with_tools(
    rt: &mut Runtime,
    prompt: &str,
    tools: &[Json],
    tokens: usize,
    thinking: Option<usize>,
    stop: &AtomicBool,
) -> Result<Sample, String> {
    generate_streaming(rt, prompt, tools, tokens, thinking, stop, &mut |_, _| {})
}

/// [`generate_with_tools`], handing `on` the reply as it streams.
pub fn generate_streaming(
    rt: &mut Runtime,
    prompt: &str,
    tools: &[Json],
    tokens: usize,
    thinking: Option<usize>,
    stop: &AtomicBool,
    on: &mut dyn FnMut(bool, String),
) -> Result<Sample, String> {
    rt.session.reset();
    let mut vars = Map::new();
    vars.insert("enable_thinking".into(), Json::Bool(thinking.is_some()));
    let request = ChatRequest {
        messages: vec![json!({"role": "user", "content": prompt})],
        tools: tools.to_vec(),
        max_tokens: tokens + thinking.unwrap_or(0),
        sampling: ferrum::hybrid::session::SamplingParams::default(),
        template_vars: vars,
        stop: Vec::new(),
        reasoning_budget: thinking,
        raw_reasoning: false,
    };
    let r = rt
        .chat(&request, Streaming { stop, on })
        .map_err(|e| e.to_string())?;
    let c = &r.completion;
    Ok(Sample {
        tokens: c.tokens.len(),
        decode: c.decode.as_secs_f64().max(1e-9),
        prefill_tps: c.prompt_tokens.saturating_sub(c.reused_tokens) as f64
            / c.prefill.as_secs_f64().max(1e-9),
        ids: c.tokens.clone(),
        text: r.output.content.clone(),
        calls: r.output.tool_calls.clone(),
    })
}

pub fn load(
    plan: &Plan,
    cand: &Candidate,
    depth: Option<usize>,
    chunk: Option<usize>,
) -> Result<Runtime, String> {
    load_tree(plan, cand, depth, None, chunk)
}

/// [`load`], drafting trees when `tree` is set.
pub fn load_tree(
    plan: &Plan,
    cand: &Candidate,
    depth: Option<usize>,
    tree: Option<Tree>,
    chunk: Option<usize>,
) -> Result<Runtime, String> {
    let mut options = PlanOptions {
        context: Some(plan.context),
        kv_cache: plan.kv_cache,
        ..Default::default()
    };
    if let Some(c) = chunk {
        options.chunk = c;
    }
    let attempt = |depth: Option<usize>| {
        let spec = cand.spec(depth, tree);
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
    let mut baseline_ids: Vec<Vec<u32>> = Vec::new();
    let mut rows: Vec<Row> = Vec::new();
    // `Row::cand` is the candidate's place in `plan.candidates`, ticked or
    // not, as the queue on screen lists them.
    for (ci, cand) in plan.candidates.iter().enumerate().filter(|(_, c)| c.on) {
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
            let shape = depth.map_or("baseline".to_string(), |d| format!("depth {d}"));
            match measure(
                &mut rt,
                &plan,
                ci,
                depth,
                None,
                &shape,
                &baseline_ids,
                &send,
                &stop,
            ) {
                Ok(Some((row, ids))) => {
                    if cand.source == Source::Baseline {
                        baseline_ids = ids;
                    }
                    rows.push(row.clone());
                    send(Event::Row(row));
                }
                Ok(None) => break,
                Err(e) => {
                    send(Event::Skipped(cand.label(), e));
                    break;
                }
            }
        }
        drop(rt);
        // Trees, each on its own load (KFlash also sets the drafting block).
        for tree in TREES.into_iter().filter(|_| cand.trees()) {
            if stop.load(Ordering::SeqCst) {
                break;
            }
            let label = format!("{} · {}", cand.label(), tree.label());
            send(Event::Stage(format!("loading {label}")));
            let loaded = load_tree(&plan, cand, None, Some(tree), None)
                .and_then(|mut rt| rt.warmup().map(|_| rt).map_err(|e| e.to_string()));
            let mut rt = match loaded {
                Ok(rt) => rt,
                Err(e) => {
                    send(Event::Skipped(label, e));
                    continue;
                }
            };
            let reach = rt.session.drafter().map_or(0, |d| d.max_drafts());
            match measure(
                &mut rt,
                &plan,
                ci,
                Some(reach),
                Some(tree),
                &tree.label(),
                &baseline_ids,
                &send,
                &stop,
            ) {
                Ok(Some((row, _))) => {
                    rows.push(row.clone());
                    send(Event::Row(row));
                }
                Ok(None) => break,
                Err(e) => send(Event::Skipped(label, e)),
            }
        }
    }
    if !stop.load(Ordering::SeqCst) {
        check_close_rows(&plan, &mut rows, &send, &stop);
    }
    if plan.chunks && !stop.load(Ordering::SeqCst) {
        // Prefill speed depends on how many prompt rows go through one pass.
        let (cand, depth, tree) = match best(&rows) {
            Some(r) if r.depth.is_some() => (plan.candidates[r.cand].clone(), r.depth, r.tree),
            _ => (plan.candidates[0].clone(), None, None),
        };
        // A tree keeps the drafter's own depth.
        let depth = if tree.is_some() { None } else { depth };
        let long = LONG_PARAGRAPH.repeat(14);
        for chunk in [256usize, 512, 1024, 2048] {
            if stop.load(Ordering::SeqCst) {
                break;
            }
            send(Event::Stage(format!("prefill chunk {chunk}")));
            let Ok(mut rt) = load_tree(&plan, &cand, depth, tree, Some(chunk)) else {
                continue;
            };
            if rt.warmup().is_err() {
                continue;
            }
            rt.session.depth_cap = depth;
            rt.session.speculate = depth.is_some() || tree.is_some();
            if let Ok(s) = generate(&mut rt, &long, 1, None, &stop) {
                send(Event::Chunk(ChunkRow {
                    chunk,
                    prefill_tps: s.prefill_tps,
                }));
            }
        }
    }
    send(Event::Done);
}

/// A measured row and its runs' token ids, one list per prompt.
type Measured = (Row, Vec<Vec<u32>>);

/// Runs every prompt on `rt` as configured and makes its row, with the
/// runs' token ids (None: cancelled part-way, no partial row).
#[allow(clippy::too_many_arguments)]
fn measure(
    rt: &mut Runtime,
    plan: &Plan,
    ci: usize,
    depth: Option<usize>,
    tree: Option<Tree>,
    shape: &str,
    baseline_ids: &[Vec<u32>],
    send: &impl Fn(Event),
    stop: &AtomicBool,
) -> Result<Option<Measured>, String> {
    let cand = &plan.candidates[ci];
    send(Event::Stage(format!("{} · {shape}", cand.label())));
    let (mut samples, mut per_prompt) = (Vec::new(), Vec::new());
    let (mut steps, mut accepted) = (0usize, 0usize);
    for (name, prompt) in PROMPTS {
        send(Event::Stage(format!("{} · {shape} · {name}", cand.label())));
        let s = generate(rt, prompt, plan.tokens, None, stop)?;
        per_prompt.push(s.tokens as f64 / s.decode);
        let st = &rt.session.spec_stats;
        steps += st.steps;
        accepted += st.accepted;
        samples.push(s);
        if stop.load(Ordering::SeqCst) {
            break;
        }
    }
    if samples.len() < PROMPTS.len() {
        return Ok(None);
    }
    let tokens: usize = samples.iter().map(|s| s.tokens).sum();
    let secs: f64 = samples.iter().map(|s| s.decode).sum();
    let ids: Vec<Vec<u32>> = samples.iter().map(|s| s.ids.clone()).collect();
    let same = (cand.source != Source::Baseline && !baseline_ids.is_empty())
        .then(|| compare(baseline_ids, &ids));
    let row = Row {
        cand: ci,
        label: cand.label(),
        draft_value: cand.draft_value(),
        quant: if depth.is_some() && cand.source != Source::Lookup {
            cand.quant.into()
        } else {
            String::new()
        },
        depth,
        tree,
        tps: tokens as f64 / secs,
        accept: (steps > 0).then(|| (accepted + steps) as f64 / steps as f64),
        per_prompt,
        same,
        check: None,
        tokens,
    };
    Ok(Some((row, ids)))
}

/// Asks the answer-check questions of near-tie rows that would beat every
/// exactly matching row, fastest first, until one keeps plain decoding's
/// answers. Plain decoding's own answers are cached per model.
fn check_close_rows(plan: &Plan, rows: &mut [Row], send: &impl Fn(Event), stop: &AtomicBool) {
    let exact_top = rows
        .iter()
        .filter(|r| matches!(r.same, None | Some(Match::Exact)))
        .map(|r| r.tps)
        .fold(0.0, f64::max);
    let mut order: Vec<usize> = (0..rows.len())
        .filter(|&i| rows[i].same == Some(Match::Close) && rows[i].tps > exact_top)
        .collect();
    order.sort_by(|&a, &b| rows[b].tps.total_cmp(&rows[a].tps));
    if order.is_empty() {
        return;
    }
    let items = eval::check_items();
    let cache = eval::cache_path();
    let key = eval::CheckKey::new(&plan.model, plan.kv_cache);
    let mut base = eval::cached_baseline(&cache, &key).filter(|b| b.len() == items.len());
    for i in order {
        if stop.load(Ordering::SeqCst) {
            return;
        }
        if base.is_none() {
            let plain = Candidate {
                source: Source::Baseline,
                quant: "q4_0",
                on: true,
            };
            match answer_check(plan, &plain, None, None, &items, send, stop) {
                Ok(answers) => {
                    eval::store_baseline(&cache, &key, &answers);
                    base = Some(answers);
                }
                Err(e) => {
                    if !stop.load(Ordering::SeqCst) {
                        send(Event::Skipped("answer check".into(), e));
                    }
                    return;
                }
            }
        }
        let Some(base) = &base else { return };
        let row = &rows[i];
        let cand = &plan.candidates[row.cand];
        match answer_check(plan, cand, row.depth, row.tree, &items, send, stop) {
            Ok(answers) => {
                let verdict = eval::compare_answers(base, &answers);
                let passed = verdict.passed();
                rows[i].check = Some(verdict.clone());
                send(Event::Checked(i, verdict));
                if passed {
                    return;
                }
            }
            Err(_) if stop.load(Ordering::SeqCst) => return,
            Err(e) => send(Event::Skipped(
                format!("answer check of {}", cand.label()),
                e,
            )),
        }
    }
}

/// Loads `cand` at `depth` (or drafting `tree`) and grades its answers to
/// `items`.
pub fn answer_check(
    plan: &Plan,
    cand: &Candidate,
    depth: Option<usize>,
    tree: Option<Tree>,
    items: &[eval::Item],
    send: &impl Fn(Event),
    stop: &AtomicBool,
) -> Result<Vec<eval::Graded>, String> {
    let label = match (tree, depth) {
        (Some(t), _) => format!("{} · {} · {}", cand.label(), t.label(), cand.quant_label()),
        (None, Some(d)) => format!("{} · depth {d} · {}", cand.label(), cand.quant_label()),
        (None, None) => cand.label(),
    };
    send(Event::Stage(format!("answer check · loading {label}")));
    // A tree keeps the drafter's own depth.
    let depth = if tree.is_some() { None } else { depth };
    let mut rt = load_tree(plan, cand, depth, tree, None)?;
    rt.warmup().map_err(|e| e.to_string())?;
    rt.session.depth_cap = depth;
    rt.session.speculate = depth.is_some() || tree.is_some();
    eval::answer_items(&mut rt, items, stop, |n| {
        send(Event::Stage(format!(
            "answer check · {label} · question {}/{}",
            n + 1,
            items.len()
        )))
    })
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

/// The fastest row whose output matches plain decoding: exactly, or within
/// tolerance once the answer check passed (other rows are only considered
/// when nothing matches); on a near tie, a chain over a tree, then the
/// shallower, cheaper one.
pub fn best(rows: &[Row]) -> Option<&Row> {
    let ok = |r: &&Row| match r.same {
        Some(Match::Differs) => false,
        Some(Match::Close) => r.check.as_ref().is_some_and(Verdict::passed),
        _ => true,
    };
    let pool: Vec<&Row> = if rows.iter().any(|r| ok(&r)) {
        rows.iter().filter(ok).collect()
    } else {
        rows.iter().collect()
    };
    let top = pool.iter().map(|r| r.tps).fold(0.0, f64::max);
    pool.into_iter()
        .filter(|r| r.tps >= top * 0.99)
        .min_by_key(|r| (r.tree.is_some(), r.depth.unwrap_or(0), r.quant == "q8_0"))
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
    if let Some(t) = row.tree {
        // A tree's depth is the drafter's default for trees.
        v.insert("draft_tree".into(), json!(t.value()));
        if row.quant == "q8_0" {
            v.insert("draft_quant".into(), json!("q8_0"));
        }
    } else if let Some(d) = row.depth {
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
            tree: None,
            tps,
            accept: depth.map(|_| 2.0),
            per_prompt: vec![tps; 3],
            same: None,
            check: None,
            tokens: 600,
        }
    }

    #[test]
    fn compares_outputs_with_tolerance() {
        let base = vec![(0..100).collect::<Vec<u32>>(), (0..40).collect()];
        assert_eq!(compare(&base, &base), Match::Exact);
        // Prompt 0 flips at token 60, prompt 1 at token 10 (25% of 40).
        let mut late = base.clone();
        late[0][60] = 999;
        late[1][10] = 999;
        assert_eq!(compare(&base, &late), Match::Close);
        // One answer going wrong early is enough to differ.
        late[1][3] = 999;
        assert_eq!(compare(&base, &late), Match::Differs);
        // A shorter answer that matches as far as it goes is a late divergence.
        let mut short = base.clone();
        short[0].truncate(70);
        assert_eq!(compare(&base, &short), Match::Close);
        assert_eq!(compare(&base, &base[..1]), Match::Differs);
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
        off.same = Some(Match::Differs);
        let mut ok = row("a", Some(2), 60.0);
        ok.same = Some(Match::Exact);
        let mixed = vec![row("none", None, 40.0), ok.clone(), off.clone()];
        assert_eq!(best(&mixed).unwrap().depth, Some(2));
        // A late near-tie flip may win once it keeps its answers.
        let mut close = row("a", Some(6), 90.0);
        close.same = Some(Match::Close);
        let unchecked = vec![
            row("none", None, 40.0),
            ok.clone(),
            off.clone(),
            close.clone(),
        ];
        assert_eq!(
            best(&unchecked).unwrap().depth,
            Some(2),
            "≈? is not recommended"
        );
        let verdict = |lost| Verdict {
            n: 30,
            lost,
            gained: 0,
            same: 30 - lost,
        };
        close.check = Some(verdict(3));
        let failed = vec![
            row("none", None, 40.0),
            ok.clone(),
            off.clone(),
            close.clone(),
        ];
        assert_eq!(
            best(&failed).unwrap().depth,
            Some(2),
            "≈✗ is not recommended"
        );
        close.check = Some(verdict(0));
        let tolerant = vec![row("none", None, 40.0), ok, off, close];
        assert_eq!(best(&tolerant).unwrap().depth, Some(6));
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
        let mut tree = row("t", Some(15), 95.0);
        tree.tree = Some(Tree::Auto);
        let t = settings_for(&tree, None);
        assert_eq!(t["draft_tree"], "auto");
        assert!(!t.contains_key("draft_max"));
        let v = settings_for(&row("none", None, 40.0), None);
        assert_eq!(v["draft"], "off");
        assert!(!v.contains_key("draft_max"));
    }

    /// Prompt lookup selects `--draft lookup`, sweeps its own depths and
    /// carries no drafter precision into the saved settings.
    #[test]
    fn lookup_candidates_map_to_the_lookup_drafter() {
        let c = Candidate {
            source: Source::Lookup,
            quant: "q4_0",
            on: true,
        };
        assert_eq!(c.draft_value(), "lookup");
        assert_eq!(c.quant_label(), "–");
        assert_eq!(c.max_depth(), LOOKUP_MAX_DEPTH);
        let spec = c.spec(Some(3), None).expect("lookup speculates");
        assert!(matches!(spec.source, DraftSource::Lookup));
        assert_eq!(spec.max_drafts, Some(3));
    }

    #[test]
    #[ignore = "needs a model and a drafter on disk"]
    fn the_sweep_measures_trees() {
        let var = |k: &str| std::env::var(k).expect(k);
        let cand = |source| Candidate {
            source,
            quant: "q4_0",
            on: true,
        };
        let plan = Plan {
            model: var("FERRUM_CHECK_MODEL").into(),
            candidates: vec![
                cand(Source::Baseline),
                cand(Source::Drafter {
                    path: var("FERRUM_CHECK_DRAFTER").into(),
                    label: "drafter".into(),
                    kind: "drafter",
                    max_depth: 2,
                }),
            ],
            quick: true,
            tokens: 96,
            chunks: false,
            context: 8192,
            kv_cache: KvCache::F16,
        };
        let running = spawn(plan).unwrap();
        let mut rows = Vec::new();
        for event in running.rx.iter() {
            match event {
                Event::Row(r) => {
                    eprintln!(
                        "{} {}: {:.1} tok/s, same {:?}",
                        r.label,
                        r.shape_short(),
                        r.tps,
                        r.same
                    );
                    rows.push(r);
                }
                Event::Skipped(l, e) => panic!("{l} skipped: {e}"),
                Event::Done => break,
                _ => {}
            }
        }
        let trees: Vec<&Row> = rows.iter().filter(|r| r.tree.is_some()).collect();
        assert_eq!(trees.len(), TREES.len());
        // Greedy trees reproduce plain decoding (up to near-tie flips).
        assert!(trees.iter().all(|r| r.same != Some(Match::Differs)));
        eprintln!("best: {:?}", best(&rows).and_then(Row::shape));
    }

    #[test]
    fn tree_rows_draft_trees() {
        let mtp = Candidate {
            source: Source::Mtp,
            quant: "q4_0",
            on: true,
        };
        assert!(mtp.trees());
        let fixed = mtp.spec(None, Some(Tree::Fixed(15))).unwrap();
        assert_eq!((fixed.tree, fixed.tree_rule), (Some(15), false));
        let auto = mtp.spec(None, Some(Tree::Auto)).unwrap();
        assert!(auto.tree_rule && auto.draft.kflash);
        let lookup = Candidate {
            source: Source::Lookup,
            ..mtp
        };
        assert!(!lookup.trees());
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
            kv_cache: KvCache::F16,
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
