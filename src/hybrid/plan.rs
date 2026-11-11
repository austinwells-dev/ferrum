//! Memory planning: predict every allocation from GGUF metadata before
//! loading, and choose the largest context that fits the GPU working set.
#![forbid(unsafe_code)]
use super::{
    config::HybridConfig,
    engine::{kv_bytes, kv_ring_bytes, max_forward_rows, page, recurrent_bytes, scratch_bytes},
};
use crate::{Error, Result, loader::gguf::GgufFile};

/// Planning inputs. `context: None` fits the largest context automatically.
#[derive(Debug, Clone, Copy)]
pub struct PlanOptions {
    /// GPU memory the model may use. `None`: the device's recommended
    /// working set minus `reserve`.
    pub budget: Option<usize>,
    /// Headroom left for the OS, the window server and other GPU clients.
    pub reserve: usize,
    /// Requested context (positions); `None` = largest that fits.
    pub context: Option<usize>,
    /// Prompt rows per forward pass (scratch scales with it).
    pub chunk: usize,
    /// Rows of logits kept for evaluation (1 for generation).
    pub logit_rows: usize,
    /// Recurrent-state snapshots kept for prefix reuse across requests.
    pub snapshots: usize,
    /// Speculative drafter memory (`runtime::SpecOptions::memory`).
    pub draft: DraftMemory,
    /// Vision projector weights and encoder buffers (`vision::projector_bytes`).
    pub vision: usize,
    /// K/V cache format; Q8 roughly halves the per-token cost. Defaults to
    /// `FERRUM_KV_CACHE` (f16 or q8) when set, else F16.
    pub kv_cache: super::KvCache,
    /// Sessions of `context` positions each (the server's `--parallel`):
    /// every one has its own K/V cache, recurrent state and snapshots, and
    /// they share the weights, scratch, drafter and projector.
    pub parallel: usize,
}

/// Memory a speculative drafter adds: weights, verify and draft buffers,
/// plus any K/V that grows with the context (MTP).
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct DraftMemory {
    pub weights: usize,
    pub fixed: usize,
    pub per_token: f64,
}

impl Default for PlanOptions {
    fn default() -> Self {
        Self {
            budget: None,
            reserve: 1 << 30,
            context: None,
            chunk: 0,
            logit_rows: 1,
            snapshots: 2,
            draft: DraftMemory::default(),
            vision: 0,
            kv_cache: super::KvCache::from_env(),
            parallel: 1,
        }
    }
}

#[derive(Debug, Clone)]
pub struct MemoryPlan {
    pub weights: usize,
    pub scratch: usize,
    /// Fixed recurrent state (conv windows and delta-rule matrices).
    pub recurrent: usize,
    pub snapshots: usize,
    /// Allocation-rounded K+V bytes per cached position (full-attention layers).
    pub kv_per_token: f64,
    /// K+V rings of sliding-window layers: fixed, whatever the context.
    pub kv_ring: usize,
    /// Driver, pipeline and small runtime allocations (measured estimate).
    pub overhead: usize,
    pub budget: usize,
    pub max_context: usize,
    pub context: usize,
    pub trained_context: usize,
    /// Speculative drafter (zero without one).
    pub draft: DraftMemory,
    /// Vision projector (zero without one).
    pub vision: usize,
    /// Sessions, each with `recurrent`, `snapshots` and `kv(context)`.
    pub parallel: usize,
    /// Arena of routed experts streamed from the file (zero when they are
    /// all resident): what the budget leaves after everything else.
    pub arena: usize,
    /// Routed expert bytes in the file (zero when resident).
    pub streamed: usize,
}

/// Measured on the M5: pipelines, command buffers and small per-forward
/// buffers stay below this.
const RUNTIME_OVERHEAD: usize = 96 << 20;

impl MemoryPlan {
    /// Total predicted bytes at the chosen context.
    pub fn total(&self) -> usize {
        self.total_at(self.context)
    }
    fn total_at(&self, context: usize) -> usize {
        self.fixed() + self.parallel * self.kv(context) + self.draft_kv(context)
    }
    /// Drafter bytes at `context` (weights, buffers and context-sized K/V).
    pub fn draft_total(&self, context: usize) -> usize {
        self.draft.weights + self.draft.fixed + self.draft_kv(context)
    }
    fn draft_kv(&self, context: usize) -> usize {
        (self.draft.per_token * context.next_multiple_of(32) as f64).ceil() as usize
    }
    /// K+V bytes at `context` (upper bound; allocation pads to 32 positions,
    /// and contexts shorter than a ring keep linear caches no larger than it).
    pub fn kv(&self, context: usize) -> usize {
        (self.kv_per_token * context.next_multiple_of(32) as f64).ceil() as usize
            + self.kv_ring
            + (1 << 20)
    }
    fn fixed(&self) -> usize {
        self.arena
            + self.weights
            + self.scratch
            + self.parallel * (self.recurrent + self.snapshots)
            + self.overhead
            + self.draft.weights
            + self.draft.fixed
            + self.vision
    }
}

/// Weight bytes the loader will allocate: every trunk tensor, page-rounded,
/// plus a second embedding copy when the output projection is tied.
pub fn weight_bytes(file: &GgufFile, c: &HybridConfig) -> usize {
    let trunk = |name: &str| match name.strip_prefix("blk.") {
        Some(rest) => rest
            .split('.')
            .next()
            .and_then(|l| l.parse::<usize>().ok())
            .is_some_and(|l| l < c.layers),
        None => true,
    };
    let mut total: usize = file
        .tensors()
        .values()
        .filter(|t| trunk(&t.name))
        // qwen4exp's sparse-attention indexer is not loaded: below its
        // top-k of 2048 positions, its attention is dense attention.
        .filter(|t| !t.name.contains(".indexer."))
        .map(|t| match super::weights::quantizes_hc(&t.name, t.type_id) {
            // BF16 to Q8_0: 34 bytes per 32 values in place of 64.
            true => page(t.byte_len / 64 * 34),
            false => page(t.byte_len),
        })
        .sum();
    if !file.tensors().contains_key("output.weight")
        && let Ok(embedding) = file.tensor("token_embd.weight")
    {
        total += page(embedding.byte_len);
    }
    total
}

/// The part of `weight_bytes` that is routed experts (`*_exps` tensors): what
/// a model too large for the working set would have to stream from disk.
pub fn routed_expert_bytes(file: &GgufFile, c: &HybridConfig) -> usize {
    file.tensors()
        .values()
        .filter(|t| t.name.contains("_exps."))
        .filter(|t| {
            t.name
                .strip_prefix("blk.")
                .and_then(|rest| rest.split('.').next())
                .and_then(|l| l.parse::<usize>().ok())
                .is_some_and(|l| l < c.layers)
        })
        .map(|t| page(t.byte_len))
        .sum()
}

/// Prompt rows per forward when `PlanOptions::chunk` is 0.
pub const DEFAULT_CHUNK: usize = 512;
/// The same for a model whose experts stream: a larger chunk reads each
/// routed expert once for more rows, and its reads hide behind longer layers.
/// On Qwen3.8-Flash-Next it trades about 2 tok/s of decode (its scratch comes
/// out of the arena) for 25-30% faster prompts (docs/measurements/qwen4exp).
pub const STREAMED_CHUNK: usize = 2048;

/// `chunk`, or the model's default for 0.
pub fn model_chunk(c: &HybridConfig, chunk: usize) -> usize {
    match chunk {
        0 if super::streams_experts(c) => STREAMED_CHUNK,
        0 => DEFAULT_CHUNK,
        n => n,
    }
}

/// Default context of a model whose experts stream: past it, context takes
/// memory the expert arena would use to save SSD reads.
pub const STREAMED_CONTEXT: usize = 32768;
/// The smallest arena worth running: every expert of one layer (a prompt
/// chunk can route to all of them) twice over.
fn min_arena(c: &HybridConfig, streamed: usize) -> usize {
    let layers = c.layers.max(1);
    (2 * streamed / layers).min(streamed)
}

pub fn plan(
    file: &GgufFile,
    c: &HybridConfig,
    recommended_working_set: usize,
    options: PlanOptions,
) -> Result<MemoryPlan> {
    let mut options = options;
    options.chunk = model_chunk(c, options.chunk);
    let weights = weight_bytes(file, c);
    if !super::streams_experts(c) {
        return plan_for_weights(weights, c, recommended_working_set, options);
    }
    // Dense weights stay resident; the arena takes what the context leaves.
    let streamed = routed_expert_bytes(file, c);
    let floor = min_arena(c, streamed);
    let auto = options.context.is_none();
    let mut probe = plan_for_weights(weights - streamed + floor, c, recommended_working_set, {
        options.context = options.context.or(Some(256));
        options
    })?;
    if auto {
        probe.context = probe.max_context.clamp(256, STREAMED_CONTEXT);
    }
    let room = probe.budget.saturating_sub(probe.total());
    let wanted = match std::env::var("FERRUM_EXPERT_ARENA_GIB")
        .ok()
        .and_then(|v| v.parse::<f64>().ok())
    {
        Some(gib) => ((gib * (1u64 << 30) as f64) as usize).max(floor),
        None => floor + room,
    };
    probe.weights -= floor;
    probe.arena = wanted.min(streamed);
    probe.streamed = streamed;
    if probe.total() > probe.budget {
        return Err(Error::Config(format!(
            "an expert arena of {} MiB and a context of {} need {} MiB; the budget is {} MiB",
            probe.arena >> 20,
            probe.context,
            probe.total() >> 20,
            probe.budget >> 20
        )));
    }
    Ok(probe)
}

/// `plan` for a model whose weights take `weights` bytes (an MLX checkpoint's,
/// from `mlx::weight_bytes`).
pub fn plan_for_weights(
    weights: usize,
    c: &HybridConfig,
    recommended_working_set: usize,
    options: PlanOptions,
) -> Result<MemoryPlan> {
    let mut options = options;
    options.chunk = model_chunk(c, options.chunk);
    let budget = options
        .budget
        .unwrap_or_else(|| recommended_working_set.saturating_sub(options.reserve));
    let recurrent = recurrent_bytes(c);
    // Per-position K+V bytes over a large capacity, so page rounding amortizes.
    let probe = 1 << 16;
    let max_rows = max_forward_rows(options.chunk.max(1));
    let kv_ring = kv_ring_bytes(c, max_rows);
    let kv_per_token = (kv_bytes(c, probe, max_rows) - kv_ring) as f64 / probe as f64;
    let mut plan = MemoryPlan {
        weights,
        scratch: scratch_bytes(c, options.chunk, options.logit_rows),
        recurrent,
        // A snapshot copies the recurrent state and the sliding-window rings.
        snapshots: options.snapshots * (recurrent + kv_ring),
        kv_per_token,
        kv_ring,
        overhead: RUNTIME_OVERHEAD,
        budget,
        max_context: 0,
        context: 0,
        trained_context: c.trained_context,
        draft: options.draft,
        vision: options.vision,
        parallel: options.parallel.max(1),
        arena: 0,
        streamed: 0,
    };
    let free = budget.saturating_sub(plan.fixed() + plan.parallel * kv_ring);
    let per_token = plan.parallel as f64 * kv_per_token + options.draft.per_token;
    let fits = ((free as f64 / per_token) as usize / 256 * 256).min(c.trained_context);
    plan.max_context = fits;
    plan.context = match options.context {
        Some(requested) if requested > c.trained_context => {
            return Err(Error::Config(format!(
                "context {requested} exceeds the model's trained context {}",
                c.trained_context
            )));
        }
        Some(requested) if requested > fits => {
            return Err(Error::Config(format!(
                "context {requested} needs {} MiB but the budget is {} MiB; the largest context that fits is {fits}",
                plan.total_at(requested) >> 20,
                budget >> 20
            )));
        }
        Some(requested) => requested,
        None => fits,
    };
    if plan.context == 0 {
        return Err(Error::Config(format!(
            "model needs {} MiB before any context; budget is {} MiB",
            plan.fixed() >> 20,
            budget >> 20
        )));
    }
    Ok(plan)
}
