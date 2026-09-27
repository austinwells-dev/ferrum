//! Speculative decoding: the drafter interface and acceptance statistics.
//! The generation loop that drives a drafter lives in `session`.
#![forbid(unsafe_code)]
use super::engine::{HybridModel, SpecGeometry};
use crate::{MetalDevice, Result};

/// A draft model that proposes tokens for the target to verify.
///
/// Protocol: after every target forward (prefill chunk, plain decode step,
/// or `commit` after a verify) the session calls `ingest` with the rows that
/// are now part of the sequence, while the target's captured features for
/// those rows are still in place. `draft` then proposes continuations of the
/// anchor token, which is sampled but not yet in the target state.
pub trait Drafter {
    /// Target features this drafter reads, and the widest verify it needs.
    fn geometry(&self) -> SpecGeometry;
    /// Most drafts one step can propose.
    fn max_drafts(&self) -> usize;
    /// Rows `0..tokens.len()` of the target's last forward are committed at
    /// positions `pos..`.
    fn ingest(
        &mut self,
        d: &MetalDevice,
        target: &HybridModel,
        pos: usize,
        tokens: &[u32],
    ) -> Result<()>;
    /// Propose up to `max` tokens to follow `anchor` at position `pos` (the
    /// target's length).
    fn draft(
        &mut self,
        d: &MetalDevice,
        target: &HybridModel,
        pos: usize,
        anchor: u32,
        max: usize,
    ) -> Result<Vec<u32>>;
    /// The target sequence now ends at `len` (prefix reuse or reset).
    fn rewind(&mut self, d: &MetalDevice, target: &HybridModel, len: usize) -> Result<()>;
    /// The session saved a recurrent snapshot at `pos`.
    fn snapshot(&mut self, _d: &MetalDevice, _pos: usize) -> Result<()> {
        Ok(())
    }
    /// Short name for logs.
    fn name(&self) -> &str;
}

/// Acceptance statistics of one generation.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct SpecStats {
    /// Verify steps.
    pub steps: usize,
    /// Draft tokens proposed.
    pub drafted: usize,
    /// Draft tokens accepted.
    pub accepted: usize,
    /// Accepted drafts by position in the block.
    pub accepted_at: Vec<usize>,
    /// Seconds spent drafting and verifying.
    pub draft_seconds: f64,
    pub verify_seconds: f64,
}

impl SpecStats {
    /// Mean tokens per verify step, including the target's own token.
    pub fn acceptance_length(&self) -> f64 {
        if self.steps == 0 {
            return 0.;
        }
        (self.accepted + self.steps) as f64 / self.steps as f64
    }
    pub(crate) fn record(&mut self, drafted: usize, accepted: usize) {
        self.steps += 1;
        self.drafted += drafted;
        self.accepted += accepted;
        if self.accepted_at.len() < drafted {
            self.accepted_at.resize(drafted, 0);
        }
        for slot in &mut self.accepted_at[..accepted] {
            *slot += 1;
        }
    }
}
