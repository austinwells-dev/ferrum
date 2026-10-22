//! Prompt-lookup drafting: propose the tokens that followed the latest
//! earlier occurrence of the sequence's last few tokens. It needs no model
//! and no GPU work, and pays off where output repeats its context (code
//! edits, quoted text, tool-call arguments). A miss proposes nothing, so the
//! session takes a plain decode step.
#![forbid(unsafe_code)]
use super::{
    engine::{HybridModel, SpecGeometry},
    speculative::Drafter,
};
use crate::{MetalDevice, Result};

/// Longest and shortest suffix matched against the history, longest first.
const NGRAM_MAX: usize = 4;
const NGRAM_MIN: usize = 2;

pub struct Lookup {
    /// The target's committed tokens.
    tokens: Vec<u32>,
    max_drafts: usize,
}

impl Lookup {
    pub fn new(max_drafts: usize) -> Self {
        Self {
            tokens: Vec::new(),
            max_drafts: max_drafts.max(1),
        }
    }
}

/// Up to `max` tokens that followed the latest earlier occurrence of the
/// longest suffix of `history` (NGRAM_MAX down to NGRAM_MIN tokens) seen before.
pub(crate) fn propose(history: &[u32], max: usize) -> Vec<u32> {
    for n in (NGRAM_MIN..=NGRAM_MAX).rev() {
        if history.len() <= n {
            continue;
        }
        let tail = &history[history.len() - n..];
        let found = (0..history.len() - n)
            .rev()
            .find(|&start| &history[start..start + n] == tail);
        if let Some(start) = found {
            let from = start + n;
            return history[from..(from + max).min(history.len())].to_vec();
        }
    }
    Vec::new()
}

impl Drafter for Lookup {
    fn geometry(&self) -> SpecGeometry {
        SpecGeometry {
            rows: self.max_drafts + 1,
            aux_layers: Vec::new(),
            final_hidden: false,
        }
    }
    fn max_drafts(&self) -> usize {
        self.max_drafts
    }
    fn ingest(
        &mut self,
        _: &MetalDevice,
        _: &HybridModel,
        pos: usize,
        tokens: &[u32],
    ) -> Result<()> {
        self.tokens.truncate(pos);
        self.tokens.extend_from_slice(tokens);
        Ok(())
    }
    fn draft(
        &mut self,
        _: &MetalDevice,
        _: &HybridModel,
        pos: usize,
        anchor: u32,
        max: usize,
    ) -> Result<Vec<u32>> {
        self.tokens.truncate(pos);
        self.tokens.push(anchor);
        let drafts = propose(&self.tokens, max);
        self.tokens.pop();
        Ok(drafts)
    }
    fn rewind(&mut self, _: &MetalDevice, _: &HybridModel, len: usize) -> Result<()> {
        self.tokens.truncate(len);
        Ok(())
    }
    fn rewinds_anywhere(&self) -> bool {
        true
    }
    fn name(&self) -> &str {
        "lookup"
    }
}

#[cfg(test)]
mod tests {
    use super::propose;

    #[test]
    fn proposes_what_followed_the_latest_longest_match() {
        // "1 2 3 9 9 | 1 2 3 7 8 | 1 2 3": the 3-gram "1 2 3" last appeared
        // before "7 8", the more recent continuation.
        let h = [1, 2, 3, 9, 9, 1, 2, 3, 7, 8, 1, 2, 3];
        assert_eq!(propose(&h, 2), vec![7, 8]);
        assert_eq!(propose(&h, 8), vec![7, 8, 1, 2, 3]);
        // The 4-gram "5 1 2 3" wins over the more recent 3-gram.
        let h = [5, 1, 2, 3, 4, 6, 1, 2, 3, 0, 5, 1, 2, 3];
        assert_eq!(propose(&h, 1), vec![4]);
        // Single-token repeats are too weak to draft from.
        assert_eq!(propose(&[1, 2, 3, 1], 4), Vec::<u32>::new());
        assert_eq!(propose(&[], 4), Vec::<u32>::new());
        // A run proposes its own continuation.
        assert_eq!(propose(&[4, 4, 4, 4], 3), vec![4]);
    }
}
