//! The qwen4exp (Qwen3.8-Flash-Next) layer stack: four residual streams
//! joined by hyper-connections, and the n-gram embedding on layer 1. The
//! token mixers and experts are the Qwen3.5 ones of the parent module.
//!
//! Per layer, with `res` the `streams × hidden` residual of each row:
//!   (layer 1) res += ple(res)
//!   x = mix(res, hc[0]); res = write(res, mixer(x), inject)
//!   x = mix(res, hc[1]); res = write(res, moe(x), inject)
//! and before the head, x = mix(res, hc_head) in place of the output norm.
//! `mix` normalizes each stream, gates it with `up(silu(down(·) / streams))`
//! and averages the streams; `write` adds the sublayer output to stream `s`
//! scaled by `2 sigmoid(inject_s / streams)`.
use super::*;
use crate::hybrid::{ple, weights::HcWeights, weights::PleWeights};

pub(crate) struct HcScratch {
    res: Tensor,
    xn: Tensor,
    gate: Tensor,
    lo: Tensor,
    inject: Tensor,
    key: Tensor,
    value: Tensor,
    gated: Tensor,
    normed: Tensor,
}

impl HcScratch {
    pub(crate) fn new(d: &MetalDevice, c: &HybridConfig, chunk: usize) -> Result<Option<Self>> {
        let Some(q) = &c.qwen4exp else {
            return Ok(None);
        };
        let f = |dims: &[usize]| Tensor::zeros_resident(d, dims, DType::F32);
        let wide = q.hc_streams * c.hidden;
        Ok(Some(Self {
            res: f(&[chunk, wide])?,
            xn: f(&[chunk, wide])?,
            gate: f(&[chunk, wide])?,
            lo: f(&[chunk, q.hc_low_rank])?,
            inject: f(&[chunk, q.hc_streams])?,
            key: f(&[chunk, wide])?,
            value: f(&[chunk, c.hidden])?,
            gated: f(&[chunk, wide])?,
            normed: f(&[chunk, wide])?,
        }))
    }

    pub(crate) fn byte_size(&self) -> usize {
        [
            &self.res,
            &self.xn,
            &self.gate,
            &self.lo,
            &self.inject,
            &self.key,
            &self.value,
            &self.gated,
            &self.normed,
        ]
        .iter()
        .map(|t| page(t.byte_size()))
        .sum()
    }
}

/// Width of the n-gram conv input of one row (zero without hyper-connections).
pub(crate) fn ple_width(c: &HybridConfig) -> usize {
    c.qwen4exp.as_ref().map_or(0, |q| q.hc_streams * c.hidden)
}

/// `HcScratch` bytes for prompt chunks of `chunk` rows (zero without
/// hyper-connections).
pub(crate) fn scratch_bytes(c: &HybridConfig, chunk: usize) -> usize {
    let Some(q) = &c.qwen4exp else {
        return 0;
    };
    let wide = chunk * q.hc_streams * c.hidden;
    let f32s = |n: usize| page(n * 4);
    6 * f32s(wide)
        + f32s(chunk * q.hc_low_rank)
        + f32s(chunk * q.hc_streams)
        + f32s(chunk * c.hidden)
}

/// The n-gram embedding's conv history (zero without one).
pub(crate) fn ple_state_bytes(c: &HybridConfig) -> usize {
    match &c.qwen4exp {
        Some(q) if !q.ple.layers.is_empty() => {
            page((q.ple.conv_kernel - 1) * q.ple.ngram * q.hc_streams * c.hidden * 4)
        }
        _ => 0,
    }
}

impl HybridModel {
    /// The layer stack of a hyper-connection model over `m` rows of `x0`
    /// (embeddings); leaves the head's mix of the final residual in `xn`.
    /// A verify (`spec` set) tapes what `commit` replays and leaves the
    /// recurrent state and the n-gram context as they were.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn hc_layers(
        &self,
        d: &MetalDevice,
        s: &Scratch,
        state: &mut HybridState,
        tokens: &[u32],
        x0: &Tensor,
        xn: &Tensor,
        mix: &Tensor,
        m: usize,
        pos: usize,
        rope: &RopePlan,
        verify: bool,
        spec: Option<&Spec>,
    ) -> Result<()> {
        let c = &self.config;
        let q = c.qwen4exp.as_ref().expect("hyper-connection geometry");
        let hs = s.hc.as_ref().expect("hyper-connection scratch");
        let (h, streams) = (c.hidden, q.hc_streams);
        let wide = streams * h;
        let res = rows(&hs.res, 0, m, wide)?;
        let hc = Params::default().u(h)?.u(streams)?.u(m)?.0;
        d.dispatch_hybrid(
            "h_hc_expand",
            &[x0.binding()],
            &[res.binding()],
            &hc,
            [(m * h).div_ceil(256), 1, 1],
            [256, 1, 1],
            0,
        )?;
        // The n-gram rows depend only on the tokens: gather them up front.
        let emb = match &self.weights.ple_table {
            Some(table) if !q.ple.layers.is_empty() => {
                let ids = ple::rows(&q.ple, &state.ple_tokens, tokens);
                let mut values = vec![0f32; ids.len() * q.ple.row_dim];
                table.gather(&ids, &mut values)?;
                Some(Tensor::from_f32(
                    d,
                    [m, q.ple.heads() * q.ple.row_dim],
                    DType::F32,
                    &values,
                )?)
            }
            _ => None,
        };
        for (i, layer) in self.weights.layers.iter().enumerate() {
            let pair = layer
                .hc
                .as_ref()
                .expect("every layer has hyper-connections");
            if let (Some(p), Some(emb)) = (&layer.ple, &emb) {
                let tape = match (verify, spec) {
                    (true, Some(sp)) => sp.tape_ple.as_ref(),
                    _ => None,
                };
                self.ple_add(d, hs, p, state, emb, &res, m, tape)?;
            }
            self.hc_mix(d, hs, &pair[0], &res, xn, m)?;
            match &layer.mixer {
                Mixer::Attention(a) => self.attention(d, s, a, state, i, xn, mix, m, pos, rope)?,
                Mixer::Delta(w) => {
                    let tape = match (verify, spec) {
                        (true, Some(sp)) => {
                            let r = sp.recurrent_index[i].expect("delta layer is recurrent");
                            Some((&sp.tape_qkv[r], &sp.tape_g[r], &sp.tape_b[r]))
                        }
                        _ => None,
                    };
                    self.delta(d, s, w, state, i, xn, mix, m, tape, None)?
                }
                Mixer::ShortConv(_) => {
                    return Err(Error::Config(
                        "short convolutions have no hyper-connections".into(),
                    ));
                }
            }
            self.hc_write(d, hs, mix, &res, m)?;
            self.hc_mix(d, hs, &pair[1], &res, xn, m)?;
            match &layer.ffn {
                Ffn::Moe(f) => self.moe_ffn(d, s, f, Some(i), xn, mix, m)?,
                Ffn::Dense(f) => self.dense_ffn(d, s, f, xn, mix, m)?,
            }
            self.hc_write(d, hs, mix, &res, m)?;
            self.dump(d, "l_out", i, &res)?;
            // Drafters read the mean of the streams a layer leaves.
            if let Some(sp) = spec
                && let Some(features) = &sp.features
                && let Some(column) = sp.aux.iter().position(|&l| l == i)
            {
                let mean = rows(&hs.value, 0, m, h)?;
                d.dispatch_hybrid(
                    "h_hc_mean",
                    &[res.binding()],
                    &[mean.binding()],
                    &hc,
                    [(m * h).div_ceil(256), 1, 1],
                    [256, 1, 1],
                    0,
                )?;
                self.capture(d, &mean, features, column, m)?;
            }
        }
        let head = self.weights.hc_head.as_ref().expect("head mixer");
        self.hc_mix(d, hs, head, &res, xn, m)?;
        if let Some(hidden) = spec.and_then(|sp| sp.hidden.as_ref()) {
            self.capture(d, xn, hidden, 0, m)?;
        }
        match verify {
            // `commit` appends the accepted ones.
            true => state.ple_pending = tokens.to_vec(),
            false => state.push_ple_tokens(tokens),
        }
        Ok(())
    }

    /// `out = mix(res, w)`; `hs.inject` takes the write weights when `w` has them.
    fn hc_mix(
        &self,
        d: &MetalDevice,
        hs: &HcScratch,
        w: &HcWeights,
        res: &Tensor,
        out: &Tensor,
        m: usize,
    ) -> Result<()> {
        let c = &self.config;
        let q = c.qwen4exp.as_ref().expect("hyper-connection geometry");
        let (h, streams) = (c.hidden, q.hc_streams);
        let wide = streams * h;
        let xn = rows(&hs.xn, 0, m, wide)?;
        let lo = rows(&hs.lo, 0, m, q.hc_low_rank)?;
        let gate = rows(&hs.gate, 0, m, wide)?;
        d.dispatch_hybrid(
            "h_group_rmsnorm",
            &[res.binding(), w.norm.binding()],
            &[xn.binding()],
            &Params::default()
                .u(h)?
                .f(c.eps)
                .u(m * streams)?
                .u(streams)?
                .0,
            [m * streams, 1, 1],
            [256, 1, 1],
            0,
        )?;
        Self::project(d, &w.down, &xn, m, &lo)?;
        let n = m * q.hc_low_rank;
        d.dispatch_hybrid(
            "h_scaled_silu",
            &[],
            &[lo.binding()],
            &Params::default().u(n)?.f(1. / streams as f32).0,
            [n.div_ceil(256), 1, 1],
            [256, 1, 1],
            0,
        )?;
        Self::project(d, &w.up, &lo, m, &gate)?;
        d.dispatch_hybrid(
            "h_hc_mix",
            &[xn.binding(), gate.binding()],
            &[out.binding()],
            &Params::default().u(h)?.u(streams)?.u(m)?.0,
            [(m * h).div_ceil(256), 1, 1],
            [256, 1, 1],
            0,
        )?;
        if let Some(inject) = &w.inject {
            Self::project(d, inject, &xn, m, &rows(&hs.inject, 0, m, streams)?)?;
        }
        Ok(())
    }

    /// `res = write(res, out, hs.inject)`.
    fn hc_write(
        &self,
        d: &MetalDevice,
        hs: &HcScratch,
        out: &Tensor,
        res: &Tensor,
        m: usize,
    ) -> Result<()> {
        let c = &self.config;
        let streams = c.qwen4exp.as_ref().expect("geometry").hc_streams;
        d.dispatch_hybrid(
            "h_hc_write",
            &[out.binding(), rows(&hs.inject, 0, m, streams)?.binding()],
            &[res.binding()],
            &Params::default().u(c.hidden)?.u(streams)?.u(m)?.0,
            [(m * c.hidden).div_ceil(256), 1, 1],
            [256, 1, 1],
            0,
        )
    }

    /// The n-gram embedding block: `res += gated value + silu(conv(norm))`.
    /// With `tape` (a verify) the conv inputs go there and the history is
    /// left for `commit_ple`.
    #[allow(clippy::too_many_arguments)]
    fn ple_add(
        &self,
        d: &MetalDevice,
        hs: &HcScratch,
        p: &PleWeights,
        state: &HybridState,
        emb: &Tensor,
        res: &Tensor,
        m: usize,
        tape: Option<&Tensor>,
    ) -> Result<()> {
        let c = &self.config;
        let q = c.qwen4exp.as_ref().expect("geometry");
        let (h, streams) = (c.hidden, q.hc_streams);
        let wide = streams * h;
        let key = rows(&hs.key, 0, m, wide)?;
        let value = rows(&hs.value, 0, m, h)?;
        let gated = rows(&hs.gated, 0, m, wide)?;
        let normed = rows(&hs.normed, 0, m, wide)?;
        let hist = state.ple.as_ref().expect("n-gram conv history");
        Self::project(d, &p.key, emb, m, &key)?;
        Self::project(d, &p.value, emb, m, &value)?;
        d.dispatch_hybrid(
            "h_ple_gate",
            &[
                key.binding(),
                res.binding(),
                value.binding(),
                p.norm_key.binding(),
                p.norm_query.binding(),
                p.norm_conv.binding(),
            ],
            &[gated.binding(), normed.binding()],
            &Params::default().u(h)?.u(streams)?.u(m)?.f(c.eps).0,
            [m * streams, 1, 1],
            [256, 1, 1],
            0,
        )?;
        d.dispatch_hybrid(
            "h_ple_conv",
            &[normed.binding(), gated.binding(), p.conv.binding()],
            &[hist.binding(), res.binding()],
            &Params::default()
                .u(wide)?
                .u(m)?
                .u(q.ple.conv_kernel)?
                .u(q.ple.ngram)?
                .u(usize::from(tape.is_some()))?
                .0,
            [wide.div_ceil(256), 1, 1],
            [256, 1, 1],
            0,
        )?;
        match tape {
            Some(tape) => copy(d, &normed, &rows(tape, 0, m, wide)?),
            None => Ok(()),
        }
    }

    /// Commit `n` rows of a verify to the n-gram conv history and context.
    pub(super) fn commit_ple(
        &self,
        d: &MetalDevice,
        spec: &Spec,
        state: &mut HybridState,
        n: usize,
    ) -> Result<()> {
        let (Some(tape), Some(q)) = (&spec.tape_ple, &self.config.qwen4exp) else {
            return Ok(());
        };
        let wide = q.hc_streams * self.config.hidden;
        let hist = state.ple.as_ref().expect("n-gram conv history");
        d.dispatch_hybrid(
            "h_ple_push",
            &[rows(tape, 0, n, wide)?.binding()],
            &[hist.binding()],
            &Params::default()
                .u(wide)?
                .u(n)?
                .u((q.ple.conv_kernel - 1) * q.ple.ngram)?
                .0,
            [wide.div_ceil(256), 1, 1],
            [256, 1, 1],
            0,
        )?;
        let accepted: Vec<u32> = state.ple_pending.drain(..).take(n).collect();
        state.push_ple_tokens(&accepted);
        Ok(())
    }
}

/// `tape_ple` rows for verifies of `rows` (zero without an n-gram embedding).
pub(crate) fn ple_tape_bytes(c: &HybridConfig, rows: usize) -> usize {
    match &c.qwen4exp {
        Some(q) if !q.ple.layers.is_empty() => page(rows * q.hc_streams * c.hidden * 4),
        _ => 0,
    }
}
