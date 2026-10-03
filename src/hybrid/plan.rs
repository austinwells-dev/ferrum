//! Memory planning: predict every allocation from GGUF metadata before
//! loading, and choose the largest context that fits the GPU working set.
#![forbid(unsafe_code)]
use super::{
    config::HybridConfig,
    engine::{kv_bytes, page, recurrent_bytes, scratch_bytes},
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
            chunk: 512,
            logit_rows: 1,
            snapshots: 2,
            draft: DraftMemory::default(),
            vision: 0,
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
    /// Allocation-rounded K+V bytes per cached position.
    pub kv_per_token: f64,
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
}

/// Measured on the M5: pipelines, command buffers and small per-forward
/// buffers stay below this.
const RUNTIME_OVERHEAD: usize = 96 << 20;

impl MemoryPlan {
    /// Total predicted bytes at the chosen context.
    pub fn total(&self) -> usize {
        self.fixed() + self.kv(self.context) + self.draft_kv(self.context)
    }
    /// Drafter bytes at `context` (weights, buffers and context-sized K/V).
    pub fn draft_total(&self, context: usize) -> usize {
        self.draft.weights + self.draft.fixed + self.draft_kv(context)
    }
    fn draft_kv(&self, context: usize) -> usize {
        (self.draft.per_token * context.next_multiple_of(32) as f64).ceil() as usize
    }
    /// K+V bytes at `context` (upper bound; allocation pads to 32 positions).
    pub fn kv(&self, context: usize) -> usize {
        (self.kv_per_token * context.next_multiple_of(32) as f64).ceil() as usize + (1 << 20)
    }
    fn fixed(&self) -> usize {
        self.weights
            + self.scratch
            + self.recurrent
            + self.snapshots
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
        .map(|t| page(t.byte_len))
        .sum();
    if !file.tensors().contains_key("output.weight")
        && let Ok(embedding) = file.tensor("token_embd.weight")
    {
        total += page(embedding.byte_len);
    }
    total
}

pub fn plan(
    file: &GgufFile,
    c: &HybridConfig,
    recommended_working_set: usize,
    options: PlanOptions,
) -> Result<MemoryPlan> {
    let budget = options
        .budget
        .unwrap_or_else(|| recommended_working_set.saturating_sub(options.reserve));
    let recurrent = recurrent_bytes(c);
    // Per-position K+V bytes over a large capacity, so page rounding amortizes.
    let probe = 1 << 16;
    let kv_per_token = kv_bytes(c, probe) as f64 / probe as f64;
    let mut plan = MemoryPlan {
        weights: weight_bytes(file, c),
        scratch: scratch_bytes(c, options.chunk, options.logit_rows),
        recurrent,
        snapshots: options.snapshots * recurrent,
        kv_per_token,
        overhead: RUNTIME_OVERHEAD,
        budget,
        max_context: 0,
        context: 0,
        trained_context: c.trained_context,
        draft: options.draft,
        vision: options.vision,
    };
    let free = budget.saturating_sub(plan.fixed());
    let per_token = kv_per_token + options.draft.per_token;
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
                (plan.fixed() + plan.kv(requested) + plan.draft_kv(requested)) >> 20,
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
