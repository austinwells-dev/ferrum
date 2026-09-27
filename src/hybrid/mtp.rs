//! Multi-token prediction with the target's own NextN block (Qwen3.5 MTP).
//!
//! MTP position p reads the embedding of token p and the target's
//! output-normed hidden state at p−1 and predicts token p+1; it keeps its own
//! K/V cache for every committed position. Drafting chains the block on its
//! own output: step i reads draft i and the block's hidden state from step
//! i−1. All draft steps run in one GPU submission.
#![forbid(unsafe_code)]
use super::{
    engine::{HybridModel, Params, SpecGeometry, page},
    speculative::Drafter,
    weights::{MtpWeights, load_mtp},
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
    /// `prev_h` at recurrent-snapshot positions, for rewinds.
    saved: Vec<(usize, Vec<f32>)>,
}

/// Bytes an `Mtp` for `capacity` positions allocates beyond its weights.
pub fn mtp_state_bytes(
    c: &super::HybridConfig,
    capacity: usize,
    chunk: usize,
    max_drafts: usize,
) -> usize {
    let rows = chunk + 1;
    let kv = page((capacity + max_drafts + 1).next_multiple_of(32) * c.kv_heads * c.head_dim * 2);
    2 * kv
        + page(rows * c.hidden * 4)
        + page(rows * 2 * c.hidden * 4)
        + 2 * page(c.hidden * 4)
        + page(max_drafts * 4)
        + page(c.vocab * 4)
}

/// Bytes of the MTP block's weights in `file`.
pub fn mtp_weight_bytes(file: &GgufFile, c: &super::HybridConfig) -> usize {
    let prefix = format!("blk.{}.", c.layers);
    file.tensors()
        .values()
        .filter(|t| t.name.starts_with(&prefix))
        .map(|t| page(t.byte_len))
        .sum()
}

impl Mtp {
    /// Load the MTP block of the target GGUF at `path` for a session of
    /// `capacity` positions.
    pub fn load(
        d: &MetalDevice,
        path: impl AsRef<std::path::Path>,
        target: &HybridModel,
        capacity: usize,
        max_drafts: usize,
    ) -> Result<Self> {
        let c = &target.config;
        let mut file = GgufFile::open(path)?;
        let weights = load_mtp(d, &mut file, c)?;
        if weights.ffn.gate.rows != c.ffn {
            return Err(Error::Config(
                "MTP FFN width differs from the trunk's".into(),
            ));
        }
        let rows = target.chunk() + 1;
        let kv_rows = (capacity + max_drafts + 1).next_multiple_of(32);
        let kv_width = c.kv_heads * c.head_dim;
        let f = |dims: &[usize]| Tensor::zeros_resident(d, dims, DType::F32);
        Ok(Self {
            weights,
            k: Tensor::zeros_resident(d, [kv_rows, kv_width], DType::F16)?,
            v: Tensor::zeros_resident(d, [kv_rows, kv_width], DType::F16)?,
            capacity: capacity + max_drafts + 1,
            max_drafts,
            len: 0,
            staged: Vec::new(),
            h_in: f(&[rows, c.hidden])?,
            cat: f(&[rows, 2 * c.hidden])?,
            prev_h: f(&[1, c.hidden])?,
            h_out: f(&[1, c.hidden])?,
            drafts: f(&[max_drafts])?,
            logits: f(&[1, c.vocab])?,
            saved: Vec::new(),
        })
    }

    pub fn weight_bytes(&self) -> usize {
        self.weights.byte_size()
    }

    /// One MTP pass over `m` rows at `pos..`: `ids` (m token ids) and `h`
    /// (m hidden rows). With `out`, the last row's argmax goes to `out.0`
    /// and its hidden state to `out.1`.
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
    ) -> Result<()> {
        let c = &t.config;
        let hd = c.hidden;
        let w = &self.weights;
        let s = t.scratch.borrow();
        let rows = |x: &Tensor, n: usize, width: usize| x.view(0, [n, width]);
        let x0 = rows(&s.x[0], m, hd)?;
        let x1 = rows(&s.x[1], m, hd)?;
        let xn = rows(&s.xn, m, hd)?;
        let mix = rows(&s.mix, m, hd)?;
        let cat = rows(&self.cat, m, 2 * hd)?;
        t.get_rows(d, &t.weights.embedding, ids, &x0)?;
        t.rmsnorm(d, &x0, &w.enorm, &xn, m)?;
        t.capture(d, &xn, &self.cat, 0, m)?;
        t.rmsnorm(d, h, &w.hnorm, &xn, m)?;
        t.capture(d, &xn, &self.cat, 1, m)?;
        t.project(d, &w.eh_proj, &cat, m, &x1)?;
        t.rmsnorm(d, &x1, &w.attn_norm, &xn, m)?;
        t.attention_with(
            d,
            &s,
            &w.attention,
            &self.k,
            &self.v,
            c.layers,
            &xn,
            &mix,
            m,
            pos,
        )?;
        t.add_rmsnorm(d, &x1, &mix, &w.post_norm, &x0, &xn, m)?;
        t.dense_ffn(d, &s, &w.ffn, &xn, &mix, m)?;
        t.add_rmsnorm(d, &x0, &mix, &w.head_norm, &x1, &xn, m)?;
        if let Some((token, h_out)) = out {
            let last = xn.view((m - 1) * hd, [1, hd])?;
            t.project(d, &t.weights.output, &last, 1, &self.logits)?;
            d.dispatch_hybrid(
                "h_argmax",
                &[self.logits.binding()],
                &[token.binding()],
                &Params::default().u(c.vocab)?.0,
                [1, 1, 1],
                [1024, 1, 1],
                0,
            )?;
            copy_rows(d, &last, h_out, 1, hd)?;
        }
        Ok(())
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
        let max = max.min(self.max_drafts);
        if max == 0 {
            return Ok(Vec::new());
        }
        if pos != self.len + self.staged.len() {
            return Err(Error::Cache(format!(
                "MTP draft at {pos} but its sequence ends at {}",
                self.len + self.staged.len()
            )));
        }
        if pos + max > self.capacity {
            return Ok(Vec::new());
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
            )?;
        }
        e.finish()?;
        // The staged rows are committed; the anchor and draft rows are not.
        self.len += at;
        self.staged.clear();
        Ok(self.drafts.to_f32()[..max]
            .iter()
            .map(|v| v.to_bits())
            .collect())
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
