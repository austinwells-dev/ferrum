//! Prompt-lookup drafting: propose the tokens that followed the latest
//! earlier occurrence of the sequence's last few tokens. It needs no model
//! and no GPU work, and pays off where output repeats its context (code
//! edits, quoted text, tool-call arguments). A miss proposes nothing, so the
//! session takes a plain decode step.
#![forbid(unsafe_code)]
use super::{
    engine::{HybridModel, SpecGeometry},
    speculative::{DraftTree, Drafter},
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

/// A draft tree of at most `budget` nodes, at most `max` deep: the
/// continuations of every earlier occurrence of the longest matching suffix
/// of `history`, latest first (so the first branch is `propose`'s chain),
/// then those of shorter suffixes, merged where they agree.
pub(crate) fn propose_tree(history: &[u32], max: usize, budget: usize) -> DraftTree {
    let mut tree = DraftTree::default();
    for n in (NGRAM_MIN..=NGRAM_MAX).rev() {
        if history.len() <= n {
            continue;
        }
        let tail = &history[history.len() - n..];
        for start in (0..history.len() - n).rev() {
            if &history[start..start + n] != tail {
                continue;
            }
            let from = start + n;
            let mut parent = None;
            for &token in &history[from..(from + max).min(history.len())] {
                parent = Some(match tree.child(parent, token) {
                    Some(node) => node,
                    None if tree.len() < budget => tree.push(token, parent),
                    None => return tree,
                });
            }
        }
    }
    tree
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
    fn draft_tree(
        &mut self,
        _: &MetalDevice,
        _: &HybridModel,
        pos: usize,
        anchor: u32,
        max: usize,
        budget: usize,
    ) -> Result<DraftTree> {
        self.tokens.truncate(pos);
        self.tokens.push(anchor);
        let tree = propose_tree(&self.tokens, max, budget);
        self.tokens.pop();
        Ok(tree)
    }
    fn rewind(&mut self, _: &MetalDevice, _: &HybridModel, len: usize) -> Result<()> {
        self.tokens.truncate(len);
        Ok(())
    }
    fn adopt(&mut self, _: &MetalDevice, _: &HybridModel, tokens: &[u32]) -> Result<()> {
        self.tokens = tokens.to_vec();
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
    use super::{DraftTree, propose, propose_tree};

    #[test]
    fn trees_merge_every_earlier_continuation() {
        // "1 2 3" was followed by 9 9, then by 7 8, then (now) the anchor.
        let h = [1, 2, 3, 9, 9, 1, 2, 3, 7, 8, 1, 2, 3];
        let tree = propose_tree(&h, 2, 8);
        // The latest continuation first (propose's chain), then the earlier one.
        assert_eq!(tree.tokens, vec![7, 8, 9, 9]);
        assert_eq!(tree.parents, vec![None, Some(0), None, Some(2)]);
        assert_eq!(&tree.tokens[..2], &propose(&h, 2)[..]);
        // The budget cuts the later branches.
        assert_eq!(propose_tree(&h, 2, 3).tokens, vec![7, 8, 9]);
        // Shared prefixes merge: "5 6" followed twice by 4, once by 2.
        let h = [5, 6, 4, 1, 5, 6, 4, 2, 5, 6, 2, 5, 6];
        let tree = propose_tree(&h, 2, 8);
        assert_eq!(tree.tokens, vec![2, 5, 4, 2, 1]);
        assert_eq!(tree.parents, vec![None, Some(0), None, Some(2), Some(2)]);
        assert_eq!(propose_tree(&[1, 2, 3], 4, 4), DraftTree::default());
    }

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
