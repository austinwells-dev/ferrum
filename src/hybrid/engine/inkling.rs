//! The Inkling 2 layer stack. Each visit (48 over 24 stored layers: 6, then
//! three passes over layers 6..18, then 6) runs attention and experts with
//! no RoPE and no plain residual stream across blocks:
//!
//! - Inputs are attention over depth (`h_ink_mix`): learned queries weigh
//!   the RMS-normed block outputs so far and the current row. A layer whose
//!   index is a multiple of `block_size` starts a block: its input is pushed
//!   to the stack and its attention output replaces the row instead of
//!   adding to it.
//! - Attention: Q, gate and K/V projections (the last two through short
//!   causal convolutions, input-dependent on full-attention layers), L2
//!   clips, a learned bias by distance and a per-row sink logit
//!   (`flash_attention_in` with `RelBias`), then the output projection and
//!   one more convolution. Full-attention layers scale q and the position
//!   vector by a log of the position.
//! - Experts: sigmoid routing with a selection bias, a fused gate/up matrix,
//!   SiTU activation and per-expert output scales, then a convolution.
//! - The looped core: each pass after the first restarts from the state the
//!   first one entered with and reads the previous pass through messages,
//!   mixed from its blocks and blended into the Q, K, V and expert inputs by
//!   low-rank controllers.
//!
//! Each visit keeps the last three inputs of its four convolutions
//! (`HybridState::conv`). Speculative verifies are not supported yet.
use super::*;
use crate::hybrid::{
    config::inkling_conv_channels,
    weights::{InklingController, InklingLayer, InklingNorms},
};

/// RMS epsilon of the depth mixes and the controllers' inputs.
const MIX_EPS: f32 = 1e-6;
/// Most streams `h_ink_mix` reads (blocks plus the current row).
const MIX_SOURCES: usize = crate::hybrid::config::INKLING_MIX_SOURCES;

pub(crate) struct InkScratch {
    /// The residual row of every token.
    cur: Tensor,
    /// The block stack, the first pass's entry state, and the previous
    /// pass's bank and final row.
    blocks: Vec<Tensor>,
    prelude: Tensor,
    bank: Vec<Tensor>,
    bank_last: Tensor,
    /// `[3][chunk][hidden]` planes: mixed inputs, controller outputs, normed inputs.
    inputs: Tensor,
    ctl: Tensor,
    normed: Tensor,
    /// `[4][chunk][hidden]`: the Q, K, V and expert messages.
    msgs: Tensor,
    z_in: Tensor,
    z: Tensor,
    rel: Tensor,
    rel_out: Tensor,
    sink: Tensor,
    dyn_k: Tensor,
    dyn_v: Tensor,
    kconv: Tensor,
    vconv: Tensor,
    tmp: Tensor,
    gate_up: Tensor,
}

/// Every tensor `InkScratch::new` allocates, as F32 element counts.
fn scratch_sizes(c: &HybridConfig, chunk: usize) -> Vec<usize> {
    let Some(ink) = &c.inkling else {
        return Vec::new();
    };
    let (h, kv) = (c.hidden, c.kv_heads * c.head_dim);
    let (blocks, bank) = ink.stack_depth();
    let moe = c.moe.expect("inkling experts");
    let mut sizes = vec![chunk * h; blocks + bank.max(1) + 3];
    sizes.extend([
        3 * chunk * h,
        3 * chunk * h,
        3 * chunk * h,
        4 * chunk * h,
        chunk * 2 * h,
        chunk * ink.gate_rank,
        chunk * c.heads * ink.rel_len,
        chunk * c.heads * ink.rel_len,
        chunk * c.heads,
        chunk * c.kv_heads * 4,
        chunk * c.kv_heads * 4,
        chunk * kv,
        chunk * kv,
        chunk * h,
        chunk * moe.experts_used * 2 * moe.expert_ffn,
    ]);
    sizes
}

/// `InkScratch` bytes for prompt chunks of `chunk` rows (zero for other models).
pub(crate) fn scratch_bytes(c: &HybridConfig, chunk: usize) -> usize {
    scratch_sizes(c, chunk)
        .into_iter()
        .map(|n| page(n * 4))
        .sum()
}

impl InkScratch {
    pub(crate) fn new(d: &MetalDevice, c: &HybridConfig, chunk: usize) -> Result<Option<Self>> {
        let Some(ink) = &c.inkling else {
            return Ok(None);
        };
        let mut t = scratch_sizes(c, chunk)
            .into_iter()
            .map(|n| Tensor::zeros_resident(d, [n], DType::F32))
            .collect::<Result<Vec<_>>>()?
            .into_iter();
        let mut next = || t.next().expect("one tensor per size");
        let (blocks, bank) = ink.stack_depth();
        Ok(Some(Self {
            blocks: (0..blocks).map(|_| next()).collect(),
            bank: (0..bank.max(1)).map(|_| next()).collect(),
            cur: next(),
            prelude: next(),
            bank_last: next(),
            inputs: next(),
            ctl: next(),
            normed: next(),
            msgs: next(),
            z_in: next(),
            z: next(),
            rel: next(),
            rel_out: next(),
            sink: next(),
            dyn_k: next(),
            dyn_v: next(),
            kconv: next(),
            vconv: next(),
            tmp: next(),
            gate_up: next(),
        }))
    }

    pub(crate) fn byte_size(&self) -> usize {
        self.blocks
            .iter()
            .chain(&self.bank)
            .chain([
                &self.cur,
                &self.prelude,
                &self.bank_last,
                &self.inputs,
                &self.ctl,
                &self.normed,
                &self.msgs,
                &self.z_in,
                &self.z,
                &self.rel,
                &self.rel_out,
                &self.sink,
                &self.dyn_k,
                &self.dyn_v,
                &self.kconv,
                &self.vconv,
                &self.tmp,
                &self.gate_up,
            ])
            .map(|t| page(t.byte_size()))
            .sum()
    }
}

/// Plane `j` (of `m` rows) of a `[planes][chunk][hidden]` buffer.
fn plane(t: &Tensor, j: usize, m: usize, h: usize) -> Result<Tensor> {
    t.view(j * m * h, [m, h])
}

impl HybridModel {
    /// The Inkling 2 stack over the `m` embedded rows in `x0` (image rows
    /// hold the vision encoder's output); leaves `output_norm` of the final
    /// mix in `xn`.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn inkling_layers(
        &self,
        d: &MetalDevice,
        s: &Scratch,
        state: &mut HybridState,
        x0: &Tensor,
        xn: &Tensor,
        m: usize,
        pos: usize,
        tapes: Option<&[Tensor]>,
    ) -> Result<()> {
        let c = &self.config;
        let ink = c.inkling.as_ref().expect("inkling geometry");
        let w = self.weights.inkling.as_ref().expect("inkling weights");
        let is = s.ink.as_ref().expect("inkling scratch");
        let h = c.hidden;
        let cur = rows(&is.cur, 0, m, h)?;
        // Token rows are normed; image rows enter as the encoder left them.
        self.rmsnorm(d, x0, &w.embedding_norm, &cur, m)?;
        for span in state
            .media
            .iter()
            .filter(|sp| sp.start < pos + m && sp.end() > pos)
        {
            let (from, to) = (span.start.max(pos), span.end().min(pos + m));
            let n = to - from;
            d.dispatch_hybrid(
                "h_copy_rows",
                &[rows(x0, from - pos, n, h)?.binding()],
                &[rows(&cur, from - pos, n, h)?.binding()],
                &Params::default().u(n)?.u(h)?.u(h)?.u(h)?.u(0)?.0,
                [(n * h).div_ceil(256), 1, 1],
                [256, 1, 1],
                0,
            )?;
        }
        // The expert combine adds a shared expert Inkling does not have.
        let ms = s.moe.as_ref().expect("inkling has MoE scratch");
        self.fill_zero(d, &ms.shared_gate.view(0, [m])?)?;
        let block = |i: usize| rows(&is.blocks[i], 0, m, h);
        let (mut depth, mut prelude, mut bank) = (0usize, 0usize, 0usize);
        for v in 0..c.layers {
            // A verify records each convolution's inputs for `commit` and
            // leaves the visit's history alone.
            let tape = tapes.map(|t| &t[v]);
            let ip = ink.layer_of(v);
            let layer = &w.layers[ip];
            let pass = ink.pass_of(v);
            if ip == ink.core_start {
                if pass == 0 {
                    prelude = depth;
                    copy(d, &cur, &rows(&is.prelude, 0, m, h)?)?;
                } else {
                    // The blocks the previous pass pushed after its first.
                    bank = depth.saturating_sub(prelude + 1);
                    for k in 0..bank {
                        copy(d, &block(prelude + 1 + k)?, &rows(&is.bank[k], 0, m, h)?)?;
                    }
                    copy(d, &cur, &rows(&is.bank_last, 0, m, h)?)?;
                    depth = prelude;
                    copy(d, &rows(&is.prelude, 0, m, h)?, &cur)?;
                }
            }
            // Messages from the previous pass: Q, K, V, expert.
            let messages = match (&layer.loop_ctrl, pass) {
                (Some((_, queries)), 1..) => {
                    let mut sources: Vec<Tensor> = (0..bank)
                        .map(|k| rows(&is.bank[k], 0, m, h))
                        .collect::<Result<_>>()?;
                    sources.push(rows(&is.bank_last, 0, m, h)?);
                    self.ink_mix(d, &sources, queries, 4, &is.msgs, m)?;
                    true
                }
                _ => false,
            };
            let msg = |j: usize| plane(&is.msgs, j, m, h);
            let spec = attn_spec(c, v);
            let sliding = spec.window.is_some();
            // Attention inputs: one mix on sliding layers, Q, K and V on full ones.
            let mut sources: Vec<Tensor> = (0..depth).map(block).collect::<Result<_>>()?;
            sources.push(cur.clone());
            let mut inputs: [Tensor; 3] = match &layer.attn_query {
                Some(q) => {
                    let n = if sliding { 1 } else { 3 };
                    self.ink_mix(d, &sources, q, n, &is.inputs, m)?;
                    let p = |j: usize| plane(&is.inputs, if sliding { 0 } else { j }, m, h);
                    [p(0)?, p(1)?, p(2)?]
                }
                None => [cur.clone(), cur.clone(), cur.clone()],
            };
            if messages {
                let (ctrl, _) = layer.loop_ctrl.as_ref().expect("core layer");
                for (j, input) in inputs.iter_mut().enumerate() {
                    let out = plane(&is.ctl, j, m, h)?;
                    self.ink_control(d, is, &ctrl[j], input, &msg(j)?, &out, m)?;
                    *input = out;
                }
            }
            let normed = |j: usize| plane(&is.normed, j, m, h);
            let n: [Tensor; 3] = match &layer.norms {
                InklingNorms::Sliding(norm) if messages => {
                    for (j, input) in inputs.iter().enumerate() {
                        self.rmsnorm(d, input, norm, &normed(j)?, m)?;
                    }
                    [normed(0)?, normed(1)?, normed(2)?]
                }
                InklingNorms::Sliding(norm) => {
                    self.rmsnorm(d, &inputs[0], norm, &normed(0)?, m)?;
                    [normed(0)?, normed(0)?, normed(0)?]
                }
                InklingNorms::Global { q, k, v: vn, .. } => {
                    for (j, norm) in [q, k, vn].into_iter().enumerate() {
                        self.rmsnorm(d, &inputs[j], norm, &normed(j)?, m)?;
                    }
                    [normed(0)?, normed(1)?, normed(2)?]
                }
            };
            let out = rows(&s.mix, 0, m, h)?;
            self.ink_attention(d, s, is, layer, state, v, &n, &out, m, pos, tape)?;
            let conv = state.conv[v].as_ref().expect("inkling visit conv state");
            let kv = c.kv_heads * c.head_dim;
            if ip.is_multiple_of(ink.block_size) {
                // A new block: its input joins the stack, the row restarts.
                copy(d, &cur, &block(depth)?)?;
                depth += 1;
                self.ink_conv(
                    d,
                    &is.z_in,
                    tape,
                    &out,
                    &layer.attn_conv,
                    None,
                    &cur,
                    conv,
                    2 * kv,
                    m,
                )?;
            } else {
                let tmp = rows(&is.tmp, 0, m, h)?;
                self.ink_conv(
                    d,
                    &is.z_in,
                    tape,
                    &out,
                    &layer.attn_conv,
                    None,
                    &tmp,
                    conv,
                    2 * kv,
                    m,
                )?;
                self.ink_acc(d, &tmp, &cur)?;
            }
            // Experts.
            let mut sources: Vec<Tensor> = (0..depth).map(block).collect::<Result<_>>()?;
            sources.push(cur.clone());
            let mut f = plane(&is.inputs, 0, m, h)?;
            self.ink_mix(d, &sources, &layer.ffn_query, 1, &is.inputs, m)?;
            if messages {
                let (ctrl, _) = layer.loop_ctrl.as_ref().expect("core layer");
                let out = plane(&is.ctl, 0, m, h)?;
                self.ink_control(d, is, &ctrl[3], &f, &msg(3)?, &out, m)?;
                f = out;
            }
            self.rmsnorm(d, &f, &layer.ffn_norm, xn, m)?;
            self.ink_experts(d, s, is, layer, xn, &out, m)?;
            let tmp = rows(&is.tmp, 0, m, h)?;
            self.ink_conv(
                d,
                &is.z_in,
                tape,
                &out,
                &layer.ffn_conv,
                None,
                &tmp,
                conv,
                2 * kv + h,
                m,
            )?;
            self.ink_acc(d, &tmp, &cur)?;
            self.dump(d, "l_out", v, &cur)?;
        }
        let mut sources: Vec<Tensor> = (0..depth).map(block).collect::<Result<_>>()?;
        sources.push(cur.clone());
        self.ink_mix(d, &sources, &w.output_query, 1, &is.inputs, m)?;
        self.rmsnorm(
            d,
            &plane(&is.inputs, 0, m, h)?,
            &self.weights.output_norm,
            xn,
            m,
        )
    }

    /// `out[j]` = query `j`'s softmax-weighted mix of the RMS-normed `sources`.
    fn ink_mix(
        &self,
        d: &MetalDevice,
        sources: &[Tensor],
        query: &Tensor,
        queries: usize,
        out: &Tensor,
        m: usize,
    ) -> Result<()> {
        let h = self.config.hidden;
        if sources.is_empty() || sources.len() > MIX_SOURCES || queries > 4 {
            return Err(Error::Shape(format!(
                "a depth mix of {} sources and {queries} queries",
                sources.len()
            )));
        }
        let inputs: Vec<Binding<'_>> = (0..MIX_SOURCES)
            .map(|i| sources.get(i).unwrap_or(&sources[0]).binding())
            .chain([query.binding()])
            .collect();
        d.dispatch_hybrid(
            "h_ink_mix",
            &inputs,
            &[out.view(0, [queries * m * h])?.binding()],
            &Params::default()
                .u(h)?
                .u(m)?
                .u(sources.len())?
                .u(queries)?
                .f(MIX_EPS)
                .0,
            [m, 1, 1],
            [256, 1, 1],
            0,
        )
    }

    /// `out = x - tanh(F z) x + tanh(G z) msg`, `z = down [rms(x); rms(msg)]`.
    #[allow(clippy::too_many_arguments)]
    fn ink_control(
        &self,
        d: &MetalDevice,
        is: &InkScratch,
        ctrl: &InklingController,
        x: &Tensor,
        msg: &Tensor,
        out: &Tensor,
        m: usize,
    ) -> Result<()> {
        let h = self.config.hidden;
        let rank = ctrl.down.rows;
        let args = Params::default().u(h)?.u(m)?.u(rank)?.f(MIX_EPS).0;
        let z_in = rows(&is.z_in, 0, m, 2 * h)?;
        let z = rows(&is.z, 0, m, rank)?;
        d.dispatch_hybrid(
            "h_ink_ctrl_in",
            &[x.binding(), msg.binding()],
            &[z_in.binding()],
            &args,
            [m, 1, 1],
            [256, 1, 1],
            0,
        )?;
        Self::project(d, &ctrl.down, &z_in, m, &z)?;
        d.dispatch_hybrid(
            "h_ink_ctrl_out",
            &[
                x.binding(),
                msg.binding(),
                z.binding(),
                ctrl.gain.binding(),
                ctrl.forget.binding(),
            ],
            &[out.binding()],
            &args,
            [(m * h).div_ceil(256), 1, 1],
            [256, 1, 1],
            0,
        )
    }

    /// Kernel-4 causal convolution of `x` (`[m, channels]`) into `y`, with
    /// the visit's history at channel `offset` of its state row. `dynamic`
    /// adds per-head input-dependent taps (`[m, heads * 4]`). With `tape` (a
    /// verify) each row's input is recorded there at the same channel and the
    /// history is left alone; otherwise `spare` is bound, unwritten.
    #[allow(clippy::too_many_arguments)]
    fn ink_conv(
        &self,
        d: &MetalDevice,
        spare: &Tensor,
        tape: Option<&Tensor>,
        x: &Tensor,
        taps: &Tensor,
        dynamic: Option<&Tensor>,
        y: &Tensor,
        state: &Tensor,
        offset: usize,
        m: usize,
    ) -> Result<()> {
        let c = &self.config;
        let channels = x.numel() / m;
        let heads = dynamic.map_or(0, |t| t.numel() / m / 4);
        let width = channels.checked_div(heads).unwrap_or(1);
        let scale = 1. / (if heads > 0 { 9f32 } else { 5f32 }).sqrt();
        d.dispatch_hybrid(
            "h_ink_conv",
            &[
                x.binding(),
                taps.binding(),
                dynamic.unwrap_or(taps).binding(),
            ],
            &[
                y.binding(),
                state.binding(),
                tape.unwrap_or(spare).binding(),
            ],
            &Params::default()
                .u(m)?
                .u(channels)?
                .u(channels)?
                .u(inkling_conv_channels(c))?
                .u(offset)?
                .u(heads)?
                .u(width)?
                .u(usize::from(tape.is_some()))?
                .f(scale)
                .0,
            [channels.div_ceil(256), 1, 1],
            [256, 1, 1],
            0,
        )
    }

    /// After a verify, the first `n` rows of each visit's convolution
    /// inputs join its history, which keeps the newest `kernel - 1` rows.
    pub(super) fn commit_inkling(
        &self,
        d: &MetalDevice,
        spec: &Spec,
        state: &HybridState,
        n: usize,
    ) -> Result<()> {
        let c = &self.config;
        let channels = inkling_conv_channels(c);
        for (v, tape) in spec.tape_ink.iter().enumerate() {
            let history = state.conv[v].as_ref().expect("inkling visit conv state");
            d.dispatch_hybrid(
                "h_ple_push",
                &[rows(tape, 0, n, channels)?.binding()],
                &[history.binding()],
                &Params::default().u(channels)?.u(n)?.u(c.conv_kernel - 1)?.0,
                [channels.div_ceil(256), 1, 1],
                [256, 1, 1],
                0,
            )?;
        }
        Ok(())
    }

    /// `y += x`.
    fn ink_acc(&self, d: &MetalDevice, x: &Tensor, y: &Tensor) -> Result<()> {
        let n = x.numel();
        d.dispatch_hybrid(
            "h_ink_acc",
            &[x.binding()],
            &[y.binding()],
            &Params::default().u(n)?.f(0.).0,
            [n.div_ceil(256), 1, 1],
            [256, 1, 1],
            0,
        )
    }

    /// Attention of visit `v` from its normed Q, K and V inputs `n` into `out`.
    #[allow(clippy::too_many_arguments)]
    fn ink_attention(
        &self,
        d: &MetalDevice,
        s: &Scratch,
        is: &InkScratch,
        layer: &InklingLayer,
        state: &HybridState,
        v: usize,
        n: &[Tensor; 3],
        out: &Tensor,
        m: usize,
        pos: usize,
        tape: Option<&Tensor>,
    ) -> Result<()> {
        let c = &self.config;
        let ink = c.inkling.as_ref().expect("inkling geometry");
        let spec = attn_spec(c, v);
        let (q_width, kv) = (c.heads * c.head_dim, c.kv_heads * c.head_dim);
        let r_width = c.heads * ink.rel_len;
        let qg = rows(&s.qg, 0, m, 2 * q_width)?;
        let k = rows(&s.k, 0, m, kv)?;
        let vv = rows(&s.v, 0, m, kv)?;
        let rel = rows(&is.rel, 0, m, r_width)?;
        let rel_out = rows(&is.rel_out, 0, m, r_width)?;
        let sink = rows(&is.sink, 0, m, c.heads)?;
        let kconv = rows(&is.kconv, 0, m, kv)?;
        let vconv = rows(&is.vconv, 0, m, kv)?;
        let q = s.q.view(0, [m, q_width])?;
        let att = rows(&s.att, 0, m, q_width)?;
        Self::project(d, &layer.qg, &n[0], m, &qg)?;
        Self::project(d, &layer.k, &n[1], m, &k)?;
        Self::project(d, &layer.v, &n[2], m, &vv)?;
        Self::project(d, &layer.rel, &n[0], m, &rel)?;
        Self::project(d, &layer.sink, &n[0], m, &sink)?;
        let conv = state.conv[v].as_ref().expect("inkling visit conv state");
        let (dyn_k, dyn_v) = match &layer.norms {
            InklingNorms::Global { k_taps, v_taps, .. } => {
                let dk = rows(&is.dyn_k, 0, m, c.kv_heads * 4)?;
                let dv = rows(&is.dyn_v, 0, m, c.kv_heads * 4)?;
                Self::project(d, k_taps, &n[1], m, &dk)?;
                Self::project(d, v_taps, &n[2], m, &dv)?;
                (Some(dk), Some(dv))
            }
            InklingNorms::Sliding(_) => (None, None),
        };
        self.ink_conv(
            d,
            &is.z_in,
            tape,
            &k,
            &layer.k_conv,
            dyn_k.as_ref(),
            &kconv,
            conv,
            0,
            m,
        )?;
        self.ink_conv(
            d,
            &is.z_in,
            tape,
            &vv,
            &layer.v_conv,
            dyn_v.as_ref(),
            &vconv,
            conv,
            kv,
            m,
        )?;
        let kc = state.k[v].as_ref().expect("attention visit has K cache");
        let vc = state.v[v].as_ref().expect("attention visit has V cache");
        let ring = state.ring[v];
        let q8 = c.kv_cache == KvCache::Q8;
        d.dispatch_hybrid(
            if q8 { "h_ink_prep_q8" } else { "h_ink_prep" },
            &[
                qg.binding(),
                kconv.binding(),
                vconv.binding(),
                layer.q_clip.binding(),
                rel.binding(),
                layer.rel_clip.binding(),
            ],
            &[q.binding(), kc.binding(), vc.binding(), rel_out.binding()],
            &Params::default()
                .u(m)?
                .u(c.heads)?
                .u(c.kv_heads)?
                .u(pos)?
                .u(kv)?
                .u(kc.shape().dimensions()[0])?
                .u(ring)?
                .u(usize::from(spec.window.is_none()))?
                .f(ink.qk_clip)
                .f(ink.rel_clip)
                .f(c.eps)
                .f(1. / (c.head_dim as f32).sqrt())
                .f(ink.log_floor)
                .f(ink.log_alpha)
                .0,
            [2 * c.heads + 2 * c.kv_heads, m, 1],
            [c.head_dim, 1, 1],
            0,
        )?;
        self.dump(d, "Rcur", v, &rel_out)?;
        self.flash_attention_in(
            d,
            s,
            c.kv_cache,
            &q,
            kc,
            vc,
            &qg,
            &att,
            m,
            pos,
            spec,
            Some(&sink),
            ring,
            None,
            Some(RelBias {
                rel: &rel_out,
                table: &layer.rel_bias,
                extent: layer.rel_bias.numel() / ink.rel_len,
            }),
        )?;
        Self::project(d, &layer.o, &att, m, out)
    }

    /// Routed experts of `xn` into `out` (no shared expert).
    #[allow(clippy::too_many_arguments)]
    fn ink_experts(
        &self,
        d: &MetalDevice,
        s: &Scratch,
        is: &InkScratch,
        layer: &InklingLayer,
        xn: &Tensor,
        out: &Tensor,
        m: usize,
    ) -> Result<()> {
        let c = &self.config;
        let ink = c.inkling.as_ref().expect("inkling geometry");
        let moe = c.moe.expect("inkling experts");
        let ms = s.moe.as_ref().expect("MoE scratch");
        let (experts, used, h, ef) = (moe.experts, moe.experts_used, c.hidden, moe.expert_ffn);
        let routes = m * used;
        let logits = rows(&ms.logits, 0, m, experts)?;
        let ids = ms.ids.view(0, [routes])?;
        let weights = ms.weights.view(0, [routes])?;
        let gu = rows(&is.gate_up, 0, routes, 2 * ef)?;
        let act = rows(&ms.act, 0, routes, ef)?;
        let down = rows(&ms.down, 0, routes, h)?;
        Self::project(d, &layer.router, xn, m, &logits)?;
        d.dispatch_hybrid(
            "h_ink_route",
            &[logits.binding(), layer.router_bias.binding()],
            &[ids.binding(), weights.binding()],
            &Params::default()
                .u(experts)?
                .u(used)?
                .u(m)?
                .f(ink.expert_weights_scale)
                .0,
            [m, 1, 1],
            [experts, 1, 1],
            0,
        )?;
        self.dump(d, "ffn_moe_weights", 0, &weights)?;
        let situ = |sorted: Option<&Tensor>| -> Result<()> {
            let scaled = layer.gate_up_scale.is_some() && layer.down_scale.is_some();
            let gs = layer.gate_up_scale.as_ref().unwrap_or(&layer.router_bias);
            let ds = layer.down_scale.as_ref().unwrap_or(&layer.router_bias);
            d.dispatch_hybrid(
                "h_ink_situ",
                &[
                    gu.binding(),
                    ids.binding(),
                    sorted.unwrap_or(&ids).binding(),
                    gs.binding(),
                    ds.binding(),
                ],
                &[act.binding(), weights.binding()],
                &Params::default()
                    .u(ef)?
                    .u(routes)?
                    .u(usize::from(sorted.is_some()))?
                    .f(ink.situ_beta)
                    .f(ink.situ_linear_beta)
                    .u(usize::from(scaled))?
                    .0,
                [(routes * ef).div_ceil(256), 1, 1],
                [256, 1, 1],
                0,
            )
        };
        let identity = routes <= MV_ID_MAX_ROUTES || exact_math();
        let position = ms.position.view(0, [routes])?;
        if identity {
            Self::project_id(d, &layer.gate_up, xn, &ids, routes, used, &gu)?;
            situ(None)?;
            Self::project_id(d, &layer.down, &act, &ids, routes, 1, &down)?;
        } else {
            let offsets = ms.offsets.view(0, [experts + 1])?;
            let sorted = ms.sorted.view(0, [routes])?;
            let xg = rows(&ms.xg, 0, routes, h)?;
            d.dispatch_hybrid(
                "h_moe_map",
                &[ids.binding()],
                &[offsets.binding(), sorted.binding(), position.binding()],
                &Params::default().u(experts)?.u(routes)?.0,
                [1, 1, 1],
                [experts.min(1024), 1, 1],
                0,
            )?;
            d.dispatch_hybrid(
                "h_moe_gather",
                &[xn.binding(), sorted.binding()],
                &[xg.binding()],
                &Params::default().u(h)?.u(used)?.u(routes)?.0,
                [h.div_ceil(256), routes, 1],
                [256, 1, 1],
                0,
            )?;
            Self::project_mm_id_in(d, &layer.gate_up, &xg, &offsets, None, m, &gu)?;
            situ(Some(&sorted))?;
            Self::project_mm_id_in(d, &layer.down, &act, &offsets, None, m, &down)?;
        }
        d.dispatch_hybrid(
            "h_moe_combine",
            &[
                down.binding(),
                weights.binding(),
                if identity {
                    ids.binding()
                } else {
                    position.binding()
                },
                rows(&ms.shared, 0, m, h)?.binding(),
                ms.shared_gate.view(0, [m])?.binding(),
            ],
            &[out.binding()],
            &Params::default()
                .u(h)?
                .u(used)?
                .u(m)?
                .u(usize::from(identity))?
                .0,
            [(m * h).div_ceil(256), 1, 1],
            [256, 1, 1],
            0,
        )
    }

    /// Apply the final logit soft cap (`softcap * tanh(x / softcap)`) to
    /// `logits` when the model has one.
    pub(super) fn softcap(&self, d: &MetalDevice, logits: &Tensor) -> Result<()> {
        let Some(cap) = self.config.inkling.as_ref().map(|i| i.softcap) else {
            return Ok(());
        };
        if cap == 0. {
            return Ok(());
        }
        let n = logits.numel();
        d.dispatch_hybrid(
            "h_ink_softcap",
            &[],
            &[logits.binding()],
            &Params::default().u(n)?.f(cap).0,
            [n.div_ceil(256), 1, 1],
            [256, 1, 1],
            0,
        )
    }
}
