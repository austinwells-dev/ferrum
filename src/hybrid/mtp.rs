//! Multi-token prediction with the target's own NextN block (Qwen3.5 MTP).
//!
//! MTP position p reads the embedding of token p and the target's
//! output-normed hidden state at p−1 and predicts token p+1; it keeps its own
//! K/V cache for every committed position. Drafting chains the block on its
//! own output: step i reads draft i and the block's hidden state from step
//! i−1. All draft steps run in one GPU submission.
//!
//! Draft trees grow best first: each pass runs the likeliest unexpanded
//! nodes together, each reading its parent's hidden state and attending to
//! its own ancestors only (tree-masked, at position anchor + depth), and
//! their top tokens join the frontier.
#![forbid(unsafe_code)]
use super::{
    draft::TreeSearch,
    engine::{
        HybridModel, Params, RopePlan, Scratch, SpecGeometry, gather_rows, page, tree_plan_rows,
    },
    speculative::{DraftTree, Drafter, chain_with_siblings, log_softmax},
    weights::{Ffn, MtpWeights, load_mtp},
};
use crate::{DType, Error, MetalDevice, Result, Tensor, loader::gguf::GgufFile};

pub struct Mtp {
    weights: MtpWeights,
    k: Tensor,
    v: Tensor,
    capacity: usize,
    max_drafts: usize,
    /// Positions whose K/V are committed.
    len: usize,
    /// Rows staged for the next run: tokens at positions `len..`, with their
    /// paired hidden states in rows `0..staged.len()` of `h_in`.
    staged: Vec<u32>,
    h_in: Tensor,
    cat: Tensor,
    /// Target hidden state at the last ingested position (pairs with the next
    /// token); zeros at the start of a sequence.
    prev_h: Tensor,
    /// Hidden state of the latest draft step (input of the next).
    h_out: Tensor,
    drafts: Tensor,
    logits: Tensor,
    /// `[max(max_drafts, TREE_WAVE), blocks * 64]`: each draft step's (or
    /// tree pass row's) top logits per 1,024-token block (`h_topk_blocks`).
    candidates: Tensor,
    /// Draft-tree nodes, the anchor included, a tree may hold.
    tree_slots: usize,
    /// `[tree_slots, hidden]`: the block's hidden state at each tree node
    /// (the input of its children), the anchor's first.
    node_h: Tensor,
    /// `[TREE_WAVE, head rows]`: logits of one tree pass.
    wave_logits: Tensor,
    search: TreeSearch,
    /// `prev_h` at recurrent-snapshot positions, for rewinds.
    saved: Vec<(usize, Vec<f32>)>,
    /// Draft LM head: a prefix of the target's (see `DraftOptions::vocab`).
    head: super::weights::Matrix,
}

/// Bytes an `Mtp` for `capacity` positions and draft trees of `tree` nodes
/// allocates beyond its weights.
pub fn mtp_state_bytes(
    c: &super::HybridConfig,
    capacity: usize,
    chunk: usize,
    max_drafts: usize,
    tree: usize,
) -> usize {
    let rows = chunk + 1;
    let kv = page(
        (capacity + max_drafts.max(tree) + 1).next_multiple_of(32)
            * c.kv_cache.row_bytes(c.kv_heads * c.head_dim),
    );
    2 * kv
        + page(rows * c.hidden * 4)
        + page(rows * 2 * c.hidden * 4)
        + 2 * page(c.hidden * 4)
        + page(max_drafts * 4)
        + page(c.vocab * 4)
        + page(max_drafts.max(TREE_WAVE) * c.vocab.div_ceil(1024) * 64 * 4)
        + page((tree + 1) * c.hidden * 4)
        + page(TREE_WAVE * c.vocab * 4)
}

/// Per draft step (or tree slot), its top (token, logit) candidates.
pub type StepCandidates = Vec<Vec<(u32, f32)>>;

/// Alternatives a draft tree may add beside each MTP step's own pick.
const TREE_SIBLINGS: usize = 3;
/// Most tree nodes one best-first pass expands: on the 27B, waves of 4 grow
/// trees too wide and shallow, and waves of 1 or 2 take too many passes
/// (Phase 11 journal, Experiment 5).
const TREE_WAVE: usize = 3;
/// Children of an expanded node that join the frontier.
const TREE_CHILDREN: usize = 4;

/// Bytes of the MTP block's weights in `file`.
pub fn mtp_weight_bytes(file: &GgufFile, c: &super::HybridConfig) -> usize {
    let prefix = format!("blk.{}.", c.layers);
    file.tensors()
        .values()
        .filter(|t| t.name.starts_with(&prefix))
        .map(|t| page(super::weights::loaded_bytes(t.type_id, t.byte_len)))
        .sum()
}

impl Mtp {
    /// Load the MTP block of the target GGUF at `path` for a session of
    /// `capacity` positions, drafting trees of up to `tree` nodes (0: chains
    /// only) grown by `search`.
    #[allow(clippy::too_many_arguments)]
    pub fn load(
        d: &MetalDevice,
        path: impl AsRef<std::path::Path>,
        target: &HybridModel,
        capacity: usize,
        max_drafts: usize,
        vocab: Option<usize>,
        tree: usize,
        search: TreeSearch,
    ) -> Result<Self> {
        let c = &target.config;
        let head_rows = vocab.map_or(c.vocab, |v| v.clamp(1024, c.vocab));
        let mut file = GgufFile::open(path)?;
        let weights = load_mtp(d, &mut file, c)?;
        // The block runs in the trunk's scratch, sized for the trunk's FFN.
        if let Ffn::Dense(f) = &weights.ffn
            && f.gate.rows != c.ffn
        {
            return Err(Error::Config(
                "MTP FFN width differs from the trunk's".into(),
            ));
        }
        let rows = target.chunk() + 1;
        let ahead = max_drafts.max(tree);
        let kv_rows = (capacity + ahead + 1).next_multiple_of(32);
        let kv_width = c.kv_heads * c.head_dim;
        let f = |dims: &[usize]| Tensor::zeros_resident(d, dims, DType::F32);
        Ok(Self {
            weights,
            k: c.kv_cache.alloc(d, kv_rows, kv_width)?,
            v: c.kv_cache.alloc(d, kv_rows, kv_width)?,
            capacity: capacity + ahead + 1,
            max_drafts,
            len: 0,
            staged: Vec::new(),
            h_in: f(&[rows, c.hidden])?,
            cat: f(&[rows, 2 * c.hidden])?,
            prev_h: f(&[1, c.hidden])?,
            h_out: f(&[1, c.hidden])?,
            drafts: f(&[max_drafts])?,
            logits: f(&[1, head_rows])?,
            candidates: f(&[max_drafts.max(TREE_WAVE), head_rows.div_ceil(1024) * 64])?,
            tree_slots: tree + 1,
            node_h: f(&[tree + 1, c.hidden])?,
            wave_logits: f(&[if tree > 0 { TREE_WAVE } else { 1 }, head_rows])?,
            search,
            saved: Vec::new(),
            head: target.weights.output.prefix_rows(head_rows),
        })
    }

    pub fn weight_bytes(&self) -> usize {
        self.weights.byte_size()
    }

    /// Diagnostics: the chain `draft` proposes, with each step's top
    /// candidates (the anchor's step first).
    pub fn draft_traced(
        &mut self,
        d: &MetalDevice,
        t: &HybridModel,
        pos: usize,
        anchor: u32,
        max: usize,
    ) -> Result<(Vec<u32>, StepCandidates)> {
        self.draft_steps(d, t, pos, anchor, max, true)
    }

    /// Diagnostics: a best-first draft tree, with the top candidates at the
    /// anchor (first) and at every expanded node, by tree slot (node i is
    /// slot i + 1; unexpanded nodes have none).
    pub fn draft_tree_traced(
        &mut self,
        d: &MetalDevice,
        t: &HybridModel,
        pos: usize,
        anchor: u32,
        max: usize,
        budget: usize,
    ) -> Result<(DraftTree, StepCandidates)> {
        let mut trace = Vec::new();
        let tree = self.best_first(d, t, pos, anchor, max, budget, Some(&mut trace))?;
        trace.resize(tree.len() + 1, Vec::new());
        Ok((tree, trace))
    }

    /// One MTP pass over `m` rows at `pos..`: `ids` (m token ids) and `h`
    /// (m hidden rows). With `out`, the last row's argmax goes to `out.0`
    /// and its hidden state to `out.1`, and with `top`, its top logits per
    /// block to `top`.
    #[allow(clippy::too_many_arguments)]
    fn run(
        &self,
        d: &MetalDevice,
        t: &HybridModel,
        ids: &Tensor,
        h: &Tensor,
        pos: usize,
        m: usize,
        out: Option<(&Tensor, &Tensor)>,
        top: Option<&Tensor>,
    ) -> Result<()> {
        let hd = t.config.hidden;
        let s = t.scratch.borrow();
        let xn = self.block(d, t, &s, ids, h, pos, m, &Default::default())?;
        if let Some((token, h_out)) = out {
            let last = xn.view((m - 1) * hd, [1, hd])?;
            HybridModel::project(d, &self.head, &last, 1, &self.logits)?;
            d.dispatch_hybrid(
                "h_argmax",
                &[self.logits.binding()],
                &[token.binding()],
                &Params::default().u(self.head.rows)?.0,
                [1, 1, 1],
                [1024, 1, 1],
                0,
            )?;
            if let Some(top) = top {
                self.top_blocks(d, &self.logits, top, 1)?;
            }
            copy_rows(d, &last, h_out, 1, hd)?;
        }
        Ok(())
    }

    /// One best-first tree pass: the `m` nodes in tree slots `rope`'s rows,
    /// with token ids `ids` and their parents' hidden states `h`, at cache
    /// position `pos` (the first node's slot past the anchor). Writes each
    /// node's top logits per block to candidate rows `0..m` and its hidden
    /// state to its `node_h` row.
    #[allow(clippy::too_many_arguments)]
    fn expand(
        &self,
        d: &MetalDevice,
        t: &HybridModel,
        ids: &Tensor,
        h: &Tensor,
        pos: usize,
        m: usize,
        rope: &RopePlan,
        slot: usize,
    ) -> Result<()> {
        let hd = t.config.hidden;
        let s = t.scratch.borrow();
        let xn = self.block(d, t, &s, ids, h, pos, m, rope)?;
        let logits = self.wave_logits.view(0, [m, self.head.rows])?;
        HybridModel::project(d, &self.head, &xn, m, &logits)?;
        self.top_blocks(d, &logits, &self.candidates, m)?;
        copy_rows(d, &xn, &self.node_h.view(slot * hd, [m, hd])?, m, hd)
    }

    /// The top 32 logits of every 1,024-token block of `rows` logit rows.
    fn top_blocks(
        &self,
        d: &MetalDevice,
        logits: &Tensor,
        top: &Tensor,
        rows: usize,
    ) -> Result<()> {
        d.dispatch_hybrid(
            "h_topk_blocks",
            &[logits.binding()],
            &[top.binding()],
            &Params::default().u(self.head.rows)?.0,
            [self.head.rows.div_ceil(1024), rows, 1],
            [256, 1, 1],
            0,
        )
    }

    /// The block (embedding, hidden-state mix, attention against the MTP
    /// cache, FFN, head norm) over `m` rows; returns the head-normed rows.
    #[allow(clippy::too_many_arguments)]
    fn block(
        &self,
        d: &MetalDevice,
        t: &HybridModel,
        s: &Scratch,
        ids: &Tensor,
        h: &Tensor,
        pos: usize,
        m: usize,
        rope: &RopePlan,
    ) -> Result<Tensor> {
        let c = &t.config;
        let hd = c.hidden;
        let w = &self.weights;
        let rows = |x: &Tensor, n: usize, width: usize| x.view(0, [n, width]);
        let x0 = rows(&s.x[0], m, hd)?;
        let x1 = rows(&s.x[1], m, hd)?;
        let xn = rows(&s.xn, m, hd)?;
        let mix = rows(&s.mix, m, hd)?;
        let cat = rows(&self.cat, m, 2 * hd)?;
        HybridModel::get_rows(d, &t.weights.embedding, ids, &x0)?;
        t.rmsnorm(d, &x0, &w.enorm, &xn, m)?;
        t.capture(d, &xn, &self.cat, 0, m)?;
        t.rmsnorm(d, h, &w.hnorm, &xn, m)?;
        t.capture(d, &xn, &self.cat, 1, m)?;
        HybridModel::project(d, &w.eh_proj, &cat, m, &x1)?;
        t.rmsnorm(d, &x1, &w.attn_norm, &xn, m)?;
        t.attention_with(
            d,
            s,
            &w.attention,
            &self.k,
            &self.v,
            0,
            c.layers,
            &xn,
            &mix,
            m,
            pos,
            rope,
        )?;
        t.add_rmsnorm(d, &x1, &mix, &w.post_norm, &x0, &xn, m)?;
        match &w.ffn {
            Ffn::Dense(f) => t.dense_ffn(d, s, f, &xn, &mix, m)?,
            Ffn::Moe(f) => t.moe_ffn(d, s, f, &xn, &mix, m)?,
        }
        t.add_rmsnorm(d, &x0, &mix, &w.head_norm, &x1, &xn, m)?;
        Ok(xn)
    }

    /// Draft up to `max` tokens after `anchor` at `pos` (the target's length)
    /// as one submission; with `top`, also every step's top candidates
    /// (token, logit).
    fn draft_steps(
        &mut self,
        d: &MetalDevice,
        t: &HybridModel,
        pos: usize,
        anchor: u32,
        max: usize,
        top: bool,
    ) -> Result<(Vec<u32>, StepCandidates)> {
        let max = max.min(self.max_drafts);
        if max == 0 {
            return Ok(Default::default());
        }
        if pos != self.len + self.staged.len() {
            return Err(Error::Cache(format!(
                "MTP draft at {pos} but its sequence ends at {}",
                self.len + self.staged.len()
            )));
        }
        if pos + max > self.capacity {
            return Ok(Default::default());
        }
        let hd = t.config.hidden;
        if self.staged.len() + 1 > t.chunk() {
            self.flush(d, t)?;
        }
        let at = self.staged.len();
        let mut ids = self.staged.clone();
        ids.push(anchor);
        let m = ids.len();
        let ids = ids_tensor(d, &ids)?;
        let width = self.candidate_width();
        let top_row = |i: usize| -> Result<Option<Tensor>> {
            top.then(|| self.candidates.view(i * width, [width]))
                .transpose()
        };
        let e = d.execution_with_shared_encoder(true)?;
        copy_rows(d, &self.prev_h, &self.h_in.view(at * hd, [1, hd])?, 1, hd)?;
        let first = self.drafts.view(0, [1])?;
        self.run(
            d,
            t,
            &ids,
            &self.h_in.view(0, [m, hd])?,
            self.len,
            m,
            Some((&first, &self.h_out)),
            top_row(0)?.as_ref(),
        )?;
        for i in 1..max {
            let prev = self.drafts.view(i - 1, [1])?;
            let next = self.drafts.view(i, [1])?;
            self.run(
                d,
                t,
                &prev,
                &self.h_out,
                pos + i,
                1,
                Some((&next, &self.h_out)),
                top_row(i)?.as_ref(),
            )?;
        }
        e.finish()?;
        // The staged rows are committed; the anchor and draft rows are not.
        self.len += at;
        self.staged.clear();
        let chain = self.drafts.to_f32()[..max]
            .iter()
            .map(|v| v.to_bits())
            .collect();
        let candidates = if top {
            self.read_candidates(max)
        } else {
            Vec::new()
        };
        Ok((chain, candidates))
    }

    /// Values per candidate row (`h_topk_blocks`' (logit, id) pairs).
    fn candidate_width(&self) -> usize {
        self.head.rows.div_ceil(1024) * 64
    }

    /// The (token, logit) pairs of candidate rows `0..rows`.
    fn read_candidates(&self, rows: usize) -> StepCandidates {
        let width = self.candidate_width();
        self.candidates.to_f32()[..rows * width]
            .chunks(width)
            .map(|row| {
                row.chunks_exact(2)
                    .filter(|pair| pair[0].is_finite())
                    .map(|pair| (pair[1].to_bits(), pair[0]))
                    .collect()
            })
            .collect()
    }

    /// Grow a draft tree of up to `budget` nodes, at most `max` deep, best
    /// first: each pass takes the likeliest frontier tokens (by the drafter's
    /// probability of the whole path), adds them, and, while budget remains,
    /// runs them together to find their children.
    #[allow(clippy::too_many_arguments)]
    fn best_first(
        &mut self,
        d: &MetalDevice,
        t: &HybridModel,
        pos: usize,
        anchor: u32,
        max: usize,
        budget: usize,
        mut trace: Option<&mut StepCandidates>,
    ) -> Result<DraftTree> {
        let budget = budget.min(self.tree_slots - 1);
        if budget == 0 || max == 0 || pos + budget + 1 > self.capacity {
            return Ok(DraftTree::default());
        }
        if pos != self.len + self.staged.len() {
            return Err(Error::Cache(format!(
                "MTP draft at {pos} but its sequence ends at {}",
                self.len + self.staged.len()
            )));
        }
        let hd = t.config.hidden;
        if self.staged.len() + 1 > t.chunk() {
            self.flush(d, t)?;
        }
        // The anchor (with the staged rows): its children and hidden state.
        let at = self.staged.len();
        let mut ids = self.staged.clone();
        ids.push(anchor);
        let m = ids.len();
        let ids = ids_tensor(d, &ids)?;
        let e = d.execution_with_shared_encoder(true)?;
        copy_rows(d, &self.prev_h, &self.h_in.view(at * hd, [1, hd])?, 1, hd)?;
        let top = self.candidates.view(0, [self.candidate_width()])?;
        self.run(
            d,
            t,
            &ids,
            &self.h_in.view(0, [m, hd])?,
            self.len,
            m,
            Some((&self.drafts.view(0, [1])?, &self.node_h.view(0, [1, hd])?)),
            Some(&top),
        )?;
        e.finish()?;
        self.len += at;
        self.staged.clear();

        // Frontier: (path log-probability, token, parent node, depth).
        let mut frontier: Vec<(f64, u32, Option<usize>, usize)> = Vec::new();
        let mut score: Vec<f64> = Vec::new();
        let mut depth: Vec<usize> = Vec::new();
        let mut tree = DraftTree::default();
        let push_children = |frontier: &mut Vec<(f64, u32, Option<usize>, usize)>,
                             candidates: &[(u32, f32)],
                             parent: Option<usize>,
                             base: f64,
                             at_depth: usize| {
            if at_depth >= max {
                return;
            }
            let mut probs = log_softmax(candidates);
            probs.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
            for (token, lp) in probs.into_iter().take(TREE_CHILDREN) {
                frontier.push((base + lp, token, parent, at_depth + 1));
            }
        };
        let root = self.read_candidates(1).swap_remove(0);
        push_children(&mut frontier, &root, None, 0., 0);
        if let Some(trace) = trace.as_deref_mut() {
            *trace = vec![root];
        }
        while tree.len() < budget && !frontier.is_empty() {
            // Likeliest first; ties to shallower nodes, then lower ids.
            frontier.sort_by(|a, b| b.0.total_cmp(&a.0).then(a.3.cmp(&b.3)).then(a.1.cmp(&b.1)));
            let take = TREE_WAVE.min(budget - tree.len()).min(frontier.len());
            let first = tree.len();
            for (lp, token, parent, at_depth) in frontier.drain(..take) {
                tree.push(token, parent);
                score.push(lp);
                depth.push(at_depth);
            }
            if tree.len() == budget {
                break;
            }
            // Expand the new nodes (tree slots first + 1 ..): each reads its
            // parent's hidden state and sees only its ancestors.
            let new = first..tree.len();
            let (tokens, parents) = tree.rows(anchor);
            let slots = new.start + 1..new.end + 1;
            let parent_slots: Vec<usize> = slots
                .clone()
                .map(|s| parents[s].expect("a node has a parent slot"))
                .collect();
            let rope = tree_plan_rows(d, pos, pos as i32, &parents, slots.clone())?;
            let m = slots.len();
            let ids = ids_tensor(d, &tokens[slots.clone()])?;
            let parent_ids = Tensor::from_le_bytes(
                d,
                [m],
                DType::F32,
                &parent_slots
                    .iter()
                    .flat_map(|&s| (s as u32).to_le_bytes())
                    .collect::<Vec<_>>(),
            )?;
            let h = self.h_in.view(0, [m, hd])?;
            let e = d.execution_with_shared_encoder(true)?;
            gather_rows(d, &self.node_h, &parent_ids, &h, m, hd, hd)?;
            self.expand(d, t, &ids, &h, pos + slots.start, m, &rope, slots.start)?;
            e.finish()?;
            for (row, candidates) in self.read_candidates(m).into_iter().enumerate() {
                let node = new.start + row;
                push_children(
                    &mut frontier,
                    &candidates,
                    Some(node),
                    score[node],
                    depth[node],
                );
                if let Some(trace) = trace.as_deref_mut() {
                    trace.resize(node + 2, Vec::new());
                    trace[node + 1] = candidates;
                }
            }
        }
        Ok(tree)
    }

    /// Run the staged rows (no output) and commit them.
    fn flush(&mut self, d: &MetalDevice, t: &HybridModel) -> Result<()> {
        if self.staged.is_empty() {
            return Ok(());
        }
        let m = self.staged.len();
        let ids = ids_tensor(d, &self.staged)?;
        let e = d.execution_with_shared_encoder(true)?;
        self.run(
            d,
            t,
            &ids,
            &self.h_in.view(0, [m, t.config.hidden])?,
            self.len,
            m,
            None,
            None,
        )?;
        e.finish()?;
        self.len += m;
        self.staged.clear();
        Ok(())
    }
}

fn ids_tensor(d: &MetalDevice, ids: &[u32]) -> Result<Tensor> {
    Tensor::from_le_bytes(
        d,
        [ids.len()],
        DType::F32,
        &ids.iter().flat_map(|t| t.to_le_bytes()).collect::<Vec<_>>(),
    )
}

/// `dst[0..n] = src[0..n]` as `n` rows of `width` (contiguous).
fn copy_rows(d: &MetalDevice, src: &Tensor, dst: &Tensor, n: usize, width: usize) -> Result<()> {
    d.dispatch_hybrid(
        "h_copy_rows",
        &[src.binding()],
        &[dst.binding()],
        &Params::default()
            .u(n)?
            .u(width)?
            .u(width)?
            .u(width)?
            .u(0)?
            .0,
        [(n * width).div_ceil(256), 1, 1],
        [256, 1, 1],
        0,
    )
}

impl Drafter for Mtp {
    fn geometry(&self) -> SpecGeometry {
        SpecGeometry {
            rows: self.max_drafts + 1,
            aux_layers: Vec::new(),
            final_hidden: true,
        }
    }

    fn max_drafts(&self) -> usize {
        self.max_drafts
    }

    fn name(&self) -> &str {
        "mtp"
    }

    fn ingest(
        &mut self,
        d: &MetalDevice,
        t: &HybridModel,
        pos: usize,
        tokens: &[u32],
    ) -> Result<()> {
        if pos != self.len + self.staged.len() {
            return Err(Error::Cache(format!(
                "MTP ingest at {pos} but its sequence ends at {}",
                self.len + self.staged.len()
            )));
        }
        let hd = t.config.hidden;
        let r = tokens.len();
        // A run shares the target's scratch, so it covers at most one chunk.
        if self.staged.len() + r > t.chunk() {
            self.flush(d, t)?;
        }
        let spec = t.spec.borrow();
        let hidden = spec
            .as_ref()
            .and_then(|s| s.hidden.as_ref())
            .ok_or_else(|| Error::Parameter("target does not capture its hidden state".into()))?;
        let at = self.staged.len();
        let e = d.execution_with_shared_encoder(true)?;
        // Row j pairs token pos+j with the hidden state at pos+j-1.
        copy_rows(d, &self.prev_h, &self.h_in.view(at * hd, [1, hd])?, 1, hd)?;
        if r > 1 {
            copy_rows(
                d,
                hidden,
                &self.h_in.view((at + 1) * hd, [r - 1, hd])?,
                r - 1,
                hd,
            )?;
        }
        copy_rows(d, &hidden.view((r - 1) * hd, [1, hd])?, &self.prev_h, 1, hd)?;
        e.finish()?;
        self.staged.extend_from_slice(tokens);
        Ok(())
    }

    fn draft(
        &mut self,
        d: &MetalDevice,
        t: &HybridModel,
        pos: usize,
        anchor: u32,
        max: usize,
    ) -> Result<Vec<u32>> {
        Ok(self.draft_steps(d, t, pos, anchor, max, false)?.0)
    }

    /// A draft tree of up to `budget` nodes, at most `max` deep: grown best
    /// first (`TreeSearch::BestFirst`), or MTP's chain of up to `max` picks
    /// with each pick's siblings (the step's next-best tokens) as leaves.
    fn draft_tree(
        &mut self,
        d: &MetalDevice,
        t: &HybridModel,
        pos: usize,
        anchor: u32,
        max: usize,
        budget: usize,
    ) -> Result<DraftTree> {
        if self.search == TreeSearch::BestFirst {
            return self.best_first(d, t, pos, anchor, max, budget, None);
        }
        let (chain, candidates) = self.draft_steps(d, t, pos, anchor, max, true)?;
        Ok(chain_with_siblings(
            &chain,
            &candidates,
            budget,
            TREE_SIBLINGS,
        ))
    }

    fn rewind(&mut self, d: &MetalDevice, t: &HybridModel, len: usize) -> Result<()> {
        if len == self.len + self.staged.len() {
            return Ok(());
        }
        if len > self.len + self.staged.len() {
            return Err(Error::Cache("MTP rewind beyond its sequence".into()));
        }
        if len > self.len {
            self.flush(d, t)?;
        }
        self.staged.clear();
        self.len = len;
        let saved = self
            .saved
            .iter()
            .find(|(p, _)| *p == len)
            .map(|(_, h)| h.clone());
        let h = saved.unwrap_or_else(|| vec![0.; t.config.hidden]);
        // The hidden state at len-1 is only known at snapshot positions; a
        // zero state degrades the next draft but never correctness.
        let src = Tensor::from_f32(d, [1, t.config.hidden], DType::F32, &h)?;
        let e = d.execution_with_shared_encoder(true)?;
        copy_rows(d, &src, &self.prev_h, 1, t.config.hidden)?;
        e.finish()?;
        self.saved.retain(|(p, _)| *p <= len);
        Ok(())
    }

    fn snapshot(&mut self, d: &MetalDevice, pos: usize) -> Result<()> {
        if pos != self.len + self.staged.len() {
            return Ok(());
        }
        d.synchronize()?;
        let h = self.prev_h.to_f32();
        self.saved.retain(|(p, _)| *p != pos);
        self.saved.push((pos, h));
        if self.saved.len() > 8 {
            self.saved.remove(0);
        }
        Ok(())
    }
}
