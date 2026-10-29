//! Speculative decoding: the drafter interface and acceptance statistics.
//! The generation loop that drives a drafter lives in `session`.
#![forbid(unsafe_code)]
use super::engine::{HybridModel, SpecGeometry};
use super::kflash::Calibration;
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
    /// Rows `0..tokens.len()` of the target's last forward (after
    /// `commit_path`, the accepted path's rows) are committed at positions
    /// `pos..`.
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
    /// Propose a draft tree of at most `budget` nodes, at most `max` deep,
    /// under `anchor`. By default, the chain `draft`
    /// proposes.
    fn draft_tree(
        &mut self,
        d: &MetalDevice,
        target: &HybridModel,
        pos: usize,
        anchor: u32,
        max: usize,
        budget: usize,
    ) -> Result<DraftTree> {
        Ok(DraftTree::chain(&self.draft(
            d,
            target,
            pos,
            anchor,
            max.min(budget),
        )?))
    }
    /// The calibration lattice trees rank their nodes by (KFlash's budget
    /// learns it from the cycles). Drafters without lattice trees ignore it.
    fn set_calibration(&mut self, _calibration: &Calibration) {}
    /// The target sequence now ends at `len` (prefix reuse or reset).
    fn rewind(&mut self, d: &MetalDevice, target: &HybridModel, len: usize) -> Result<()>;
    /// Whether `rewind` accepts any length, not only snapshot positions, so an
    /// attention-only target can resume exactly where a prompt diverges.
    fn rewinds_anywhere(&self) -> bool {
        false
    }
    /// The target's cache was loaded from disk holding `tokens`, which this
    /// drafter never ingested. Drafts may be weaker until it ingests new
    /// tokens; verification keeps the output exact.
    fn adopt(&mut self, d: &MetalDevice, target: &HybridModel, tokens: &[u32]) -> Result<()>;
    /// The session saved a recurrent snapshot at `pos`.
    fn snapshot(&mut self, _d: &MetalDevice, _pos: usize) -> Result<()> {
        Ok(())
    }
    /// Short name for logs.
    fn name(&self) -> &str;
}

/// Drafts as a tree under the anchor: node i holds `tokens[i]` and follows
/// node `parents[i]`, or the anchor when that is `None`. Parents come before
/// their children.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct DraftTree {
    pub tokens: Vec<u32>,
    pub parents: Vec<Option<usize>>,
    /// Lattice trees (empty otherwise): each node's log-probability under the
    /// drafter given its parent, and the calibrated log-probability of its
    /// whole path (`kflash::Calibration`), non-increasing in node order, so
    /// the first n nodes are the builder's best tree of n.
    pub lq: Vec<f32>,
    pub score: Vec<f32>,
}

impl DraftTree {
    pub fn chain(tokens: &[u32]) -> Self {
        Self {
            tokens: tokens.to_vec(),
            parents: (0..tokens.len()).map(|i| i.checked_sub(1)).collect(),
            ..Default::default()
        }
    }
    /// Keep the first `n` nodes (a tree, since parents come first).
    pub fn truncate(&mut self, n: usize) {
        self.tokens.truncate(n);
        self.parents.truncate(n);
        self.lq.truncate(n);
        self.score.truncate(n);
    }
    /// Depth of each node (1: a child of the anchor).
    pub fn depths(&self) -> Vec<usize> {
        let mut depth = vec![0; self.len()];
        for i in 0..self.len() {
            depth[i] = self.parents[i].map_or(0, |p| depth[p]) + 1;
        }
        depth
    }
    pub fn len(&self) -> usize {
        self.tokens.len()
    }
    pub fn is_empty(&self) -> bool {
        self.tokens.is_empty()
    }
    /// Nodes on the longest root-to-leaf path.
    pub fn depth(&self) -> usize {
        self.depths().into_iter().max().unwrap_or(0)
    }
    /// Append a node under `parent` (None: the anchor) and return its index.
    pub fn push(&mut self, token: u32, parent: Option<usize>) -> usize {
        self.tokens.push(token);
        self.parents.push(parent);
        self.tokens.len() - 1
    }
    /// The child of `parent` holding `token`.
    pub fn child(&self, parent: Option<usize>, token: u32) -> Option<usize> {
        (0..self.len()).find(|&i| self.parents[i] == parent && self.tokens[i] == token)
    }
    /// Verify rows: the anchor at row 0, node i at row i + 1, and each row's
    /// parent row.
    pub fn rows(&self, anchor: u32) -> (Vec<u32>, Vec<Option<usize>>) {
        let mut tokens = vec![anchor];
        tokens.extend(&self.tokens);
        let mut parents = vec![None];
        parents.extend(self.parents.iter().map(|p| Some(p.map_or(0, |p| p + 1))));
        (tokens, parents)
    }
}

/// Accept walk over a verified tree: from the anchor, follow the
/// child whose token is the target's choice at its parent's row. Returns the
/// verify rows of the accepted path (starting at the anchor's 0) and the
/// target's token after it. `targets[r]` is the target's token at row r.
pub fn accept_walk(tree: &DraftTree, targets: &[u32]) -> (Vec<usize>, u32) {
    let mut path = vec![0];
    let mut node = None;
    while let Some(next) = tree.child(node, targets[*path.last().expect("root")]) {
        path.push(next + 1);
        node = Some(next);
    }
    let bonus = targets[*path.last().expect("root")];
    (path, bonus)
}

/// Natural log-probabilities of `candidates` (logits of the top tokens),
/// normalized over the candidates themselves: an approximation, since the
/// tail of the vocabulary is missing, that only shapes trees.
pub(crate) fn log_softmax(candidates: &[(u32, f32)]) -> Vec<(u32, f64)> {
    let max = candidates
        .iter()
        .fold(f64::NEG_INFINITY, |m, c| m.max(f64::from(c.1)));
    let total: f64 = candidates
        .iter()
        .map(|c| (f64::from(c.1) - max).exp())
        .sum();
    let lse = max + total.ln();
    candidates
        .iter()
        .map(|&(t, l)| (t, f64::from(l) - lse))
        .collect()
}

/// A draft tree of at most `budget` nodes from a drafter's chain and its
/// next-best guesses: `chain[i]` is the drafter's pick at depth i + 1, and
/// `candidates[i]` that step's top tokens with their logits. The chain
/// continues only through its own picks, so alternatives are leaves (a
/// sibling of `chain[i]`). Nodes are taken best first by the drafter's
/// probability of the whole path to them, with at most `siblings`
/// alternatives per depth.
pub fn chain_with_siblings(
    chain: &[u32],
    candidates: &[Vec<(u32, f32)>],
    budget: usize,
    siblings: usize,
) -> DraftTree {
    // Per depth: log-probability of the chain's pick, and the alternatives best first.
    let steps: Vec<(f64, Vec<(u32, f64)>)> = chain
        .iter()
        .zip(candidates)
        .map(|(&pick, cands)| {
            let mut probs = log_softmax(cands);
            probs.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
            let own = probs
                .iter()
                .find(|c| c.0 == pick)
                .map_or(f64::NEG_INFINITY, |c| c.1);
            let alternatives = probs
                .into_iter()
                .filter(|c| c.0 != pick)
                .take(siblings)
                .collect();
            (own, alternatives)
        })
        .collect();
    let mut tree = DraftTree::default();
    // Frontier entries: (path log-probability, depth, alternative rank or
    // None for the chain's pick). The chain node at depth i heads depth i + 1.
    let mut frontier: Vec<(f64, usize, Option<usize>)> = Vec::new();
    let mut chain_node: Vec<usize> = Vec::new();
    let open = |frontier: &mut Vec<(f64, usize, Option<usize>)>, depth: usize, base: f64| {
        if let Some((own, alternatives)) = steps.get(depth) {
            frontier.push((base + own, depth, None));
            for (rank, alt) in alternatives.iter().enumerate() {
                frontier.push((base + alt.1, depth, Some(rank)));
            }
        }
    };
    open(&mut frontier, 0, 0.);
    while tree.len() < budget {
        // Best first; ties go to the chain, then to shallower and better-ranked nodes.
        let Some(best) = (0..frontier.len()).max_by(|&a, &b| {
            let (x, y) = (frontier[a], frontier[b]);
            x.0.total_cmp(&y.0)
                .then(y.2.is_some().cmp(&x.2.is_some()))
                .then(y.1.cmp(&x.1))
                .then(y.2.cmp(&x.2))
        }) else {
            break;
        };
        let (score, depth, rank) = frontier.swap_remove(best);
        if score == f64::NEG_INFINITY {
            break;
        }
        let parent = depth.checked_sub(1).map(|p| chain_node[p]);
        match rank {
            None => {
                chain_node.push(tree.push(chain[depth], parent));
                open(&mut frontier, depth + 1, score);
            }
            Some(r) => {
                tree.push(steps[depth].1[r].0, parent);
            }
        }
    }
    tree
}

/// A draft tree of at most `budget` nodes from a block drafter's candidate
/// lattice: `candidates[i]` are the tokens one drafting pass proposes at
/// depth i + 1, and `cond(i, prev)` their log-probabilities (aligned with
/// `candidates[i]`) after `prev`, the parent's token (the anchor at depth 0).
/// Nodes are taken best first by the probability of the whole path, the
/// product of its conditionals, each mapped through `calibration` to the
/// probability that the target accepts the node (Kapitsa's KFlash builder;
/// the identity keeps the drafter's own); a node's children open when it is
/// taken, so parents come before children. With conditionals that ignore
/// `prev` the positions are independent.
pub fn lattice_best_first(
    anchor: u32,
    candidates: &[Vec<u32>],
    budget: usize,
    calibration: &Calibration,
    mut cond: impl FnMut(usize, u32) -> Vec<f64>,
) -> DraftTree {
    let mut tree = DraftTree::default();
    // Frontier entries: (path log-probability, depth, candidate index,
    // parent, own log q).
    type Entry = (f64, usize, usize, Option<usize>, f64);
    let mut frontier: Vec<Entry> = Vec::new();
    let mut open =
        |frontier: &mut Vec<Entry>, depth: usize, prev: u32, parent: Option<usize>, base: f64| {
            if depth < candidates.len() {
                for (j, lq) in cond(depth, prev).into_iter().enumerate() {
                    let lp = calibration.log_p(depth + 1, lq);
                    frontier.push((base + lp, depth, j, parent, lq));
                }
            }
        };
    open(&mut frontier, 0, anchor, None, 0.);
    while tree.len() < budget {
        // Likeliest first; ties to shallower nodes, then the drafter's ranking.
        let Some(best) = (0..frontier.len()).max_by(|&a, &b| {
            let (x, y) = (frontier[a], frontier[b]);
            x.0.total_cmp(&y.0).then(y.1.cmp(&x.1)).then(y.2.cmp(&x.2))
        }) else {
            break;
        };
        let (score, depth, j, parent, lq) = frontier.swap_remove(best);
        if score == f64::NEG_INFINITY {
            break;
        }
        let token = candidates[depth][j];
        let node = tree.push(token, parent);
        tree.lq.push(lq as f32);
        tree.score.push(score as f32);
        open(&mut frontier, depth + 1, token, Some(node), score);
    }
    tree
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
    /// Accepted drafts by position in the block (depth in a tree).
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
    /// One verify of `drafted` drafts at most `depth` deep (`drafted` for a
    /// chain), of which `accepted` were kept.
    pub(crate) fn record(&mut self, drafted: usize, depth: usize, accepted: usize) {
        self.steps += 1;
        self.drafted += drafted;
        self.accepted += accepted;
        if self.accepted_at.len() < depth {
            self.accepted_at.resize(depth, 0);
        }
        for slot in &mut self.accepted_at[..accepted] {
            *slot += 1;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn accept_walk_follows_the_targets_choices() {
        // anchor -> a(0) -> b(2); anchor -> c(1); a -> d(3)
        let mut tree = DraftTree::default();
        let a = tree.push(10, None);
        let c = tree.push(11, None);
        let b = tree.push(12, Some(a));
        tree.push(13, Some(a));
        assert_eq!((a, c, b), (0, 1, 2));
        let (tokens, parents) = tree.rows(99);
        assert_eq!(tokens, vec![99, 10, 11, 12, 13]);
        assert_eq!(parents, vec![None, Some(0), Some(0), Some(1), Some(1)]);
        // Target picks 10 after the anchor, 12 after a, then 7 after b.
        assert_eq!(accept_walk(&tree, &[10, 12, 0, 7, 0]), (vec![0, 1, 3], 7));
        // A sibling: 11 after the anchor, then 5.
        assert_eq!(accept_walk(&tree, &[11, 0, 5, 0, 0]), (vec![0, 2], 5));
        // Nothing accepted: the bonus is the anchor row's choice.
        assert_eq!(accept_walk(&tree, &[42, 0, 0, 0, 0]), (vec![0], 42));
        // A chain tree accepts exactly the matching prefix.
        assert_eq!(tree.depth(), 2);
        let chain = DraftTree::chain(&[1, 2, 3]);
        assert_eq!(chain.parents, vec![None, Some(0), Some(1)]);
        assert_eq!(chain.depth(), 3);
        assert_eq!(accept_walk(&chain, &[1, 2, 9, 9]), (vec![0, 1, 2], 9));
    }

    #[test]
    fn siblings_join_where_the_drafter_is_unsure() {
        // Depth 1 is a coin flip between 5 and 6, depth 2 is sure (7), depth 3
        // splits 8 / 9 at 60 / 40.
        let ln = |p: f64| p.ln() as f32;
        let chain = [5, 7, 8];
        let candidates = vec![
            vec![(5, ln(0.5)), (6, ln(0.5))],
            vec![(7, ln(0.98)), (1, ln(0.02))],
            vec![(8, ln(0.6)), (9, ln(0.4))],
        ];
        // 5 (0.5), 6 (0.5), 7 (0.49), 8 (0.294), 9 (0.196), 1 (0.01).
        let tree = chain_with_siblings(&chain, &candidates, 5, 2);
        assert_eq!(tree.tokens, vec![5, 6, 7, 8, 9]);
        assert_eq!(tree.parents, vec![None, None, Some(0), Some(2), Some(2)]);
        // A budget of the chain's length when the chain is likeliest.
        let tree = chain_with_siblings(&chain, &candidates, 3, 2);
        assert_eq!(tree.tokens, vec![5, 6, 7]);
        // No siblings allowed: the chain itself.
        assert_eq!(
            chain_with_siblings(&chain, &candidates, 8, 0),
            DraftTree::chain(&chain)
        );
        // Budget beyond every candidate: all of them, parents first.
        let tree = chain_with_siblings(&chain, &candidates, 20, 2);
        assert_eq!(tree.len(), 6);
        assert!(
            tree.parents
                .iter()
                .enumerate()
                .all(|(i, p)| p.is_none_or(|p| p < i))
        );
    }

    #[test]
    fn lattice_trees_follow_the_conditionals() {
        let ln = f64::ln;
        // Independent positions: depth 1 splits 5 / 6 at 0.6 / 0.4, depth 2
        // is 7 at 0.9 whatever comes first.
        let candidates = vec![vec![5, 6], vec![7, 8]];
        let independent = |i: usize, _| match i {
            0 => vec![ln(0.6), ln(0.4)],
            _ => vec![ln(0.9), ln(0.1)],
        };
        // 5 (0.6), 7 under 5 (0.54), 6 (0.4), 7 under 6 (0.36).
        let tree = lattice_best_first(1, &candidates, 4, &Calibration::default(), independent);
        assert_eq!(tree.tokens, vec![5, 7, 6, 7]);
        assert_eq!(tree.parents, vec![None, Some(0), None, Some(2)]);
        // Conditionals: after 6 the drafter is sure of 8; after 5 it splits.
        let conditional = |i: usize, prev: u32| match (i, prev) {
            (0, 1) => vec![ln(0.6), ln(0.4)],
            (1, 5) => vec![ln(0.5), ln(0.5)],
            (1, 6) => vec![ln(0.01), ln(0.99)],
            _ => unreachable!("only the anchor, 5 and 6 have children"),
        };
        // 5 (0.6), 6 (0.4), 8 under 6 (0.396), 7 under 5 (0.3), 8 under 5 (0.3).
        let tree = lattice_best_first(1, &candidates, 5, &Calibration::default(), conditional);
        assert_eq!(tree.tokens, vec![5, 6, 8, 7, 8]);
        assert_eq!(tree.parents, vec![None, None, Some(1), Some(0), Some(0)]);
        // The chain's next node can lose to a sibling of its first.
        let tree = lattice_best_first(1, &candidates, 2, &Calibration::default(), conditional);
        assert_eq!(tree.tokens, vec![5, 6]);
        let chain = lattice_best_first(1, &candidates, 2, &Calibration::default(), independent);
        let expected = DraftTree::chain(&[5, 7]);
        assert_eq!(
            (chain.tokens, chain.parents),
            (expected.tokens, expected.parents)
        );
        // Budget beyond the lattice: every path, parents first.
        let tree = lattice_best_first(1, &candidates, 20, &Calibration::default(), conditional);
        assert_eq!(tree.len(), 6);
        assert!(
            tree.parents
                .iter()
                .enumerate()
                .all(|(i, p)| p.is_none_or(|p| p < i))
        );
        // Path scores fall in build order, so every prefix is a best tree.
        assert!(tree.score.windows(2).all(|w| w[1] <= w[0]));
        assert_eq!(tree.lq.len(), tree.len());
    }
}
