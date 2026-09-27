//! Forward pass for Qwen3.5-family hybrid models on fixed, preallocated storage.
#![forbid(unsafe_code)]
use super::{
    config::HybridConfig,
    weights::{
        AttentionWeights, DeltaWeights, DenseFfn, Ffn, Format, Matrix, Mixer, MoeFfn, Weights,
    },
};
use crate::{DType, Error, MetalDevice, Result, Tensor, metal::MetalBuffer};
use std::cell::RefCell;

type Binding<'a> = (&'a MetalBuffer, usize, usize);

/// Little-endian kernel parameter block.
#[derive(Default)]
pub(crate) struct Params(pub(crate) Vec<u8>);
impl Params {
    pub(crate) fn u(mut self, v: usize) -> Result<Self> {
        let v =
            u32::try_from(v).map_err(|_| Error::Shape("kernel parameter exceeds u32".into()))?;
        self.0.extend_from_slice(&v.to_le_bytes());
        Ok(self)
    }
    pub(crate) fn f(mut self, v: f32) -> Self {
        self.0.extend_from_slice(&v.to_le_bytes());
        self
    }
}

fn rows(t: &Tensor, start_row: usize, rows: usize, width: usize) -> Result<Tensor> {
    t.view(start_row * width, [rows, width])
}

/// Conversation state: F16 K/V for attention layers and F32 recurrent state
/// for delta-rule layers, all allocated once for `capacity` positions.
pub struct HybridState {
    capacity: usize,
    len: usize,
    valid: bool,
    k: Vec<Option<Tensor>>,
    v: Vec<Option<Tensor>>,
    conv: Vec<Option<Tensor>>,
    ssm: Vec<Option<Tensor>>,
}

impl HybridState {
    pub fn new(d: &MetalDevice, c: &HybridConfig, capacity: usize) -> Result<Self> {
        if capacity == 0 {
            return Err(Error::Cache("context capacity must be positive".into()));
        }
        let kv_row = c.kv_heads * c.head_dim;
        // Attention reads keys in blocks of 32; pad so a block never leaves storage.
        let kv_rows = capacity.next_multiple_of(32);
        let mut state = Self {
            capacity,
            len: 0,
            valid: true,
            k: Vec::new(),
            v: Vec::new(),
            conv: Vec::new(),
            ssm: Vec::new(),
        };
        for layer in 0..c.layers {
            if c.is_attention(layer) {
                state
                    .k
                    .push(Some(Tensor::zeros(d, [kv_rows, kv_row], DType::F16)?));
                state
                    .v
                    .push(Some(Tensor::zeros(d, [kv_rows, kv_row], DType::F16)?));
                state.conv.push(None);
                state.ssm.push(None);
            } else {
                state.k.push(None);
                state.v.push(None);
                state.conv.push(Some(Tensor::zeros(
                    d,
                    [(c.conv_kernel - 1) * c.conv_channels()],
                    DType::F32,
                )?));
                state.ssm.push(Some(Tensor::zeros(
                    d,
                    [c.ssm_v_heads * c.ssm_head_dim * c.ssm_head_dim],
                    DType::F32,
                )?));
            }
        }
        Ok(state)
    }
    pub fn len(&self) -> usize {
        self.len
    }
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
    pub fn capacity(&self) -> usize {
        self.capacity
    }
    pub fn byte_size(&self) -> usize {
        [&self.k, &self.v, &self.conv, &self.ssm]
            .iter()
            .flat_map(|v| v.iter().flatten())
            .map(Tensor::byte_size)
            .sum()
    }
    /// Forget the conversation. Recurrent state is zeroed on the next forward.
    pub fn reset(&mut self) {
        self.len = 0;
        self.valid = true;
    }
}

/// Per-chunk activation buffers, allocated once for the largest chunk.
pub(crate) struct Scratch {
    chunk: usize,
    x: [Tensor; 2],
    xn: Tensor,
    mix: Tensor,
    qg: Tensor,
    k: Tensor,
    v: Tensor,
    q: Tensor,
    att: Tensor,
    attn_partial: Tensor,
    attn_ml: Tensor,
    qkv: Tensor,
    z: Tensor,
    alpha: Tensor,
    beta: Tensor,
    g: Tensor,
    b: Tensor,
    conv: Tensor,
    delta: Tensor,
    gated: Tensor,
    ffn_gate: Tensor,
    ffn_up: Tensor,
    ffn_act: Tensor,
    logits: Tensor,
    token: Tensor,
    moe: Option<MoeScratch>,
}

/// Routing and expert buffers for `chunk * experts_used` routes.
struct MoeScratch {
    logits: Tensor,
    ids: Tensor,
    weights: Tensor,
    shared_gate: Tensor,
    offsets: Tensor,
    sorted: Tensor,
    position: Tensor,
    xg: Tensor,
    gate: Tensor,
    up: Tensor,
    act: Tensor,
    down: Tensor,
    shared: Tensor,
}

impl MoeScratch {
    fn tensors(&self) -> [&Tensor; 13] {
        [
            &self.logits,
            &self.ids,
            &self.weights,
            &self.shared_gate,
            &self.offsets,
            &self.sorted,
            &self.position,
            &self.xg,
            &self.gate,
            &self.up,
            &self.act,
            &self.down,
            &self.shared,
        ]
    }
}

impl Scratch {
    fn new(d: &MetalDevice, c: &HybridConfig, chunk: usize, logit_rows: usize) -> Result<Self> {
        let f = |dims: &[usize]| Tensor::zeros(d, dims, DType::F32);
        let ffn = match c.moe {
            Some(moe) => moe.shared_ffn.max(moe.expert_ffn),
            None => c.ffn,
        };
        Ok(Self {
            chunk,
            x: [f(&[chunk, c.hidden])?, f(&[chunk, c.hidden])?],
            xn: f(&[chunk, c.hidden])?,
            mix: f(&[chunk, c.hidden])?,
            qg: f(&[chunk, 2 * c.heads * c.head_dim])?,
            k: f(&[chunk, c.kv_heads * c.head_dim])?,
            v: f(&[chunk, c.kv_heads * c.head_dim])?,
            q: Tensor::zeros(d, [chunk, c.heads * c.head_dim], DType::F16)?,
            att: f(&[chunk, c.heads * c.head_dim])?,
            attn_partial: f(&[MAX_SPLITS * MAX_SPLIT_ROWS * c.head_dim])?,
            attn_ml: f(&[MAX_SPLITS * MAX_SPLIT_ROWS * 2])?,
            qkv: f(&[chunk, c.conv_channels()])?,
            z: f(&[chunk, c.ssm_value_dim()])?,
            alpha: f(&[chunk, c.ssm_v_heads])?,
            beta: f(&[chunk, c.ssm_v_heads])?,
            g: f(&[chunk, c.ssm_v_heads])?,
            b: f(&[chunk, c.ssm_v_heads])?,
            conv: f(&[chunk, c.conv_channels()])?,
            delta: f(&[chunk, c.ssm_value_dim()])?,
            gated: f(&[chunk, c.ssm_value_dim()])?,
            ffn_gate: f(&[chunk, ffn])?,
            ffn_up: f(&[chunk, ffn])?,
            ffn_act: f(&[chunk, ffn])?,
            logits: f(&[logit_rows, c.vocab])?,
            token: f(&[1])?,
            moe: match c.moe {
                None => None,
                Some(moe) => {
                    let routes = chunk * moe.experts_used;
                    Some(MoeScratch {
                        logits: f(&[chunk, moe.experts])?,
                        ids: f(&[routes])?,
                        weights: f(&[routes])?,
                        shared_gate: f(&[chunk])?,
                        offsets: f(&[moe.experts + 1])?,
                        sorted: f(&[routes])?,
                        position: f(&[routes])?,
                        xg: f(&[routes, c.hidden])?,
                        gate: f(&[routes, moe.expert_ffn])?,
                        up: f(&[routes, moe.expert_ffn])?,
                        act: f(&[routes, moe.expert_ffn])?,
                        down: f(&[routes, c.hidden])?,
                        shared: f(&[chunk, c.hidden])?,
                    })
                }
            },
        })
    }
    fn byte_size(&self) -> usize {
        [
            &self.x[0],
            &self.x[1],
            &self.xn,
            &self.mix,
            &self.qg,
            &self.k,
            &self.v,
            &self.q,
            &self.att,
            &self.attn_partial,
            &self.attn_ml,
            &self.qkv,
            &self.z,
            &self.alpha,
            &self.beta,
            &self.g,
            &self.b,
            &self.conv,
            &self.delta,
            &self.gated,
            &self.ffn_gate,
            &self.ffn_up,
            &self.ffn_act,
            &self.logits,
            &self.token,
        ]
        .iter()
        .map(|t| t.byte_size())
        .sum::<usize>()
            + self
                .moe
                .as_ref()
                .map_or(0, |m| m.tensors().iter().map(|t| t.byte_size()).sum())
    }
}

/// What a forward call returns for its final token(s).
pub enum Output {
    /// Nothing (prefill of a prefix that will be continued).
    None,
    /// Greedy next token, selected on the GPU.
    Argmax,
    /// Final-position logits copied to the host.
    Logits,
}

pub enum Produced {
    None,
    Token(u32),
    Logits(Vec<f32>),
}

pub struct HybridModel {
    pub config: HybridConfig,
    pub(crate) weights: Weights,
    pub(crate) scratch: RefCell<Scratch>,
    /// Rows of logits the scratch holds (1 unless evaluation asked for more).
    logit_rows: usize,
}

impl HybridModel {
    pub fn new(
        d: &MetalDevice,
        config: HybridConfig,
        weights: Weights,
        chunk: usize,
        logit_rows: usize,
    ) -> Result<Self> {
        let chunk = chunk.max(1);
        let logit_rows = logit_rows.clamp(1, chunk);
        let scratch = Scratch::new(d, &config, chunk, logit_rows)?;
        Ok(Self {
            config,
            weights,
            scratch: RefCell::new(scratch),
            logit_rows,
        })
    }
    pub fn weight_bytes(&self) -> usize {
        self.weights.byte_size()
    }
    pub fn scratch_bytes(&self) -> usize {
        self.scratch.borrow().byte_size()
    }
    pub fn chunk(&self) -> usize {
        self.scratch.borrow().chunk
    }

    /// Append `tokens` to `state` and produce `output` for the last token.
    pub fn forward(
        &self,
        d: &MetalDevice,
        state: &mut HybridState,
        tokens: &[u32],
        output: Output,
    ) -> Result<Produced> {
        if tokens.is_empty() {
            return Err(Error::Parameter(
                "forward requires at least one token".into(),
            ));
        }
        if !state.valid {
            return Err(Error::Cache(
                "state was invalidated by a failed forward; reset it".into(),
            ));
        }
        if state.len + tokens.len() > state.capacity {
            return Err(Error::Cache(format!(
                "context overflow: {} cached + {} new exceeds capacity {}",
                state.len,
                tokens.len(),
                state.capacity
            )));
        }
        if let Some(&id) = tokens.iter().find(|&&id| id as usize >= self.config.vocab) {
            return Err(Error::Token {
                id,
                vocab: self.config.vocab,
            });
        }
        let chunk = self.chunk();
        let mut produced = Produced::None;
        let pieces = tokens.chunks(chunk).count();
        for (index, piece) in tokens.chunks(chunk).enumerate() {
            let last = index + 1 == pieces;
            let out = if last { &output } else { &Output::None };
            state.valid = false;
            produced = self.forward_chunk(d, state, piece, out, None)?;
            state.len += piece.len();
            state.valid = true;
        }
        Ok(produced)
    }

    /// Evaluation: append `tokens` (at most `logit_rows` per call) and return
    /// the logits of every position as rows of `vocab` values.
    pub fn forward_all_logits(
        &self,
        d: &MetalDevice,
        state: &mut HybridState,
        tokens: &[u32],
    ) -> Result<Vec<f32>> {
        if tokens.len() > self.logit_rows || tokens.is_empty() {
            return Err(Error::Parameter(format!(
                "evaluation forward takes 1..={} tokens",
                self.logit_rows
            )));
        }
        if !state.valid || state.len + tokens.len() > state.capacity {
            return Err(Error::Cache("state invalid or full".into()));
        }
        state.valid = false;
        let mut all = Vec::new();
        self.forward_chunk(d, state, tokens, &Output::None, Some(&mut all))?;
        state.len += tokens.len();
        state.valid = true;
        Ok(all)
    }

    fn forward_chunk(
        &self,
        d: &MetalDevice,
        state: &mut HybridState,
        tokens: &[u32],
        output: &Output,
        all_logits: Option<&mut Vec<f32>>,
    ) -> Result<Produced> {
        let c = &self.config;
        let s = self.scratch.borrow();
        let m = tokens.len();
        let pos = state.len;
        let h = c.hidden;
        let execution = d.execution_with_shared_encoder(true)?;
        if pos == 0 {
            for t in state.conv.iter().chain(&state.ssm).flatten() {
                self.fill_zero(d, t)?;
            }
        }
        let ids = Tensor::from_le_bytes(
            d,
            [m],
            DType::F32,
            &tokens
                .iter()
                .flat_map(|t| t.to_le_bytes())
                .collect::<Vec<_>>(),
        )?;
        let x0 = rows(&s.x[0], 0, m, h)?;
        let x1 = rows(&s.x[1], 0, m, h)?;
        let xn = rows(&s.xn, 0, m, h)?;
        let mix = rows(&s.mix, 0, m, h)?;
        self.get_rows(d, &self.weights.embedding, &ids, &x0)?;
        self.rmsnorm(d, &x0, &self.weights.layers[0].attn_norm, &xn, m)?;
        let (mut cur, mut next) = (&x0, &x1);
        for (i, layer) in self.weights.layers.iter().enumerate() {
            match &layer.mixer {
                Mixer::Attention(a) => self.attention(d, &s, a, state, i, &xn, &mix, m, pos)?,
                Mixer::Delta(w) => self.delta(d, &s, w, state, i, &xn, &mix, m)?,
            }
            self.add_rmsnorm(d, cur, &mix, &layer.post_norm, next, &xn, m)?;
            std::mem::swap(&mut cur, &mut next);
            match &layer.ffn {
                Ffn::Dense(f) => self.dense_ffn(d, &s, f, &xn, &mix, m)?,
                Ffn::Moe(f) => {
                    self.dump(d, "attn_post_norm", i, &xn)?;
                    self.moe_ffn(d, &s, f, &xn, &mix, m)?;
                    self.dump(d, "ffn_out", i, &mix)?;
                }
            }
            let norm = self
                .weights
                .layers
                .get(i + 1)
                .map_or(&self.weights.output_norm, |l| &l.attn_norm);
            self.add_rmsnorm(d, cur, &mix, norm, next, &xn, m)?;
            std::mem::swap(&mut cur, &mut next);
            self.dump(d, "l_out", i, cur)?;
        }
        // xn now holds output_norm(final hidden) for every position.
        let produced = if let Some(all) = all_logits {
            let logits = rows(&s.logits, 0, m, c.vocab)?;
            self.project(d, &self.weights.output, &xn, m, &logits)?;
            execution.finish()?;
            *all = logits.to_f32();
            Produced::None
        } else {
            match output {
                Output::None => {
                    execution.finish()?;
                    Produced::None
                }
                Output::Argmax | Output::Logits => {
                    let last = rows(&xn, m - 1, 1, h)?;
                    let logits = rows(&s.logits, 0, 1, c.vocab)?;
                    self.project(d, &self.weights.output, &last, 1, &logits)?;
                    if matches!(output, Output::Argmax) {
                        d.dispatch_hybrid(
                            "h_argmax",
                            &[logits.binding()],
                            &[s.token.binding()],
                            &Params::default().u(c.vocab)?.0,
                            [1, 1, 1],
                            [1024, 1, 1],
                            0,
                        )?;
                        execution.finish()?;
                        Produced::Token(s.token.to_f32()[0].to_bits())
                    } else {
                        execution.finish()?;
                        Produced::Logits(logits.to_f32())
                    }
                }
            }
        };
        Ok(produced)
    }

    #[allow(clippy::too_many_arguments)]
    fn attention(
        &self,
        d: &MetalDevice,
        s: &Scratch,
        a: &AttentionWeights,
        state: &HybridState,
        layer: usize,
        xn: &Tensor,
        out: &Tensor,
        m: usize,
        pos: usize,
    ) -> Result<()> {
        let c = &self.config;
        let q_width = c.heads * c.head_dim;
        let kv_width = c.kv_heads * c.head_dim;
        let qg = rows(&s.qg, 0, m, 2 * q_width)?;
        let k = rows(&s.k, 0, m, kv_width)?;
        let v = rows(&s.v, 0, m, kv_width)?;
        let q = s.q.view(0, [m, q_width])?;
        let att = rows(&s.att, 0, m, q_width)?;
        self.project(d, &a.q, xn, m, &qg)?;
        self.project(d, &a.k, xn, m, &k)?;
        self.project(d, &a.v, xn, m, &v)?;
        let kc = state.k[layer]
            .as_ref()
            .expect("attention layer has K cache");
        let vc = state.v[layer]
            .as_ref()
            .expect("attention layer has V cache");
        let prep = Params::default()
            .u(m)?
            .u(c.heads)?
            .u(c.kv_heads)?
            .u(c.head_dim)?
            .u(c.rope_dims)?
            .u(pos)?
            .f(c.rope_theta)
            .f(c.eps)
            .u(kv_width)?
            .f(1. / (c.head_dim as f32).sqrt());
        d.dispatch_hybrid(
            "h_attn_prep",
            &[
                qg.binding(),
                k.binding(),
                v.binding(),
                a.q_norm.binding(),
                a.k_norm.binding(),
            ],
            &[q.binding(), kc.binding()],
            &prep.0,
            [c.heads + c.kv_heads, m, 1],
            [c.head_dim, 1, 1],
            0,
        )?;
        d.dispatch_hybrid(
            "h_store_v",
            &[v.binding()],
            &[vc.binding()],
            &prep.0,
            [(m * kv_width).div_ceil(256), 1, 1],
            [256, 1, 1],
            0,
        )?;
        self.dump(d, "Qcur_full", layer, &qg)?;
        self.dump(d, "Kcur", layer, &k)?;
        self.dump(d, "Vcur", layer, &v)?;
        self.flash_attention(d, s, &q, kc, vc, &qg, &att, m, pos)?;
        self.dump(d, "attn_gated", layer, &att)?;
        self.project(d, &a.o, &att, m, out)?;
        self.dump(d, "attn_output", layer, out)
    }

    /// Causal attention of `m` new queries against the cache, gated by
    /// sigmoid(gate) from `qg`, written to `att`.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn flash_attention(
        &self,
        d: &MetalDevice,
        s: &Scratch,
        q: &Tensor,
        kc: &Tensor,
        vc: &Tensor,
        qg: &Tensor,
        att: &Tensor,
        m: usize,
        pos: usize,
    ) -> Result<()> {
        let c = &self.config;
        let group = c.heads / c.kv_heads;
        let rows = m * group;
        let keys = pos + m;
        if std::env::var_os("FERRUM_HYBRID_REFERENCE_ATTENTION").is_some() {
            let args = Params::default()
                .u(m)?
                .u(c.heads)?
                .u(c.kv_heads)?
                .u(c.head_dim)?
                .u(pos)?
                .u(c.kv_heads * c.head_dim)?
                .f(1.);
            return d.dispatch_hybrid(
                "h_attention",
                &[q.binding(), kc.binding(), vc.binding(), qg.binding()],
                &[att.binding()],
                &args.0,
                [c.heads, m, 1],
                [c.head_dim, 1, 1],
                0,
            );
        }
        if m >= MPP_ATTENTION_MIN_TOKENS
            && std::env::var_os("FERRUM_HYBRID_SIMD_ATTENTION").is_none()
        {
            let args = Params::default()
                .u(m)?
                .u(c.heads)?
                .u(c.kv_heads)?
                .u(group)?
                .u(pos)?
                .u(c.kv_heads * c.head_dim)?
                .u(0)?
                .u(1)?
                .0;
            return d.dispatch_hybrid(
                "h_flash_attn_mpp",
                &[q.binding(), kc.binding(), vc.binding(), qg.binding()],
                &[att.binding()],
                &args,
                [m.div_ceil(64), c.heads, 1],
                [128, 1, 1],
                64 * 64 * 4 + 64 * 64 * 2 + 3 * 64 * 4,
            );
        }
        if m == 1 && group <= 8 && std::env::var_os("FERRUM_HYBRID_SIMD_ATTENTION").is_none() {
            let target: usize = std::env::var("FERRUM_HYBRID_SPLIT_TARGET")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(DECODE_SPLIT_TARGET);
            let min_keys: usize = std::env::var("FERRUM_HYBRID_SPLIT_MIN_KEYS")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(DECODE_SPLIT_MIN_KEYS);
            let splits = (target / c.kv_heads)
                .max(1)
                .min(keys.div_ceil(min_keys))
                .clamp(1, MAX_SPLITS);
            let keys_per_split = keys.div_ceil(splits).next_multiple_of(32);
            let splits = keys.div_ceil(keys_per_split);
            let args = Params::default()
                .u(1)?
                .u(c.heads)?
                .u(c.kv_heads)?
                .u(group)?
                .u(pos)?
                .u(c.kv_heads * c.head_dim)?
                .u(keys_per_split)?
                .u(splits)?
                .0;
            d.dispatch_hybrid(
                "h_attn_decode",
                &[q.binding(), kc.binding(), vc.binding(), qg.binding()],
                &[att.binding(), s.attn_partial.binding(), s.attn_ml.binding()],
                &args,
                [c.kv_heads, splits, 1],
                [128, 1, 1],
                8 * 256 * 2 + 4 * 8 * 2 * 4 + 2 * 8 * 256 * 4,
            )?;
            if splits > 1 {
                d.dispatch_hybrid(
                    "h_flash_reduce",
                    &[s.attn_partial.binding(), s.attn_ml.binding(), qg.binding()],
                    &[att.binding()],
                    &args,
                    [c.heads, 1, 1],
                    [c.head_dim, 1, 1],
                    0,
                )?;
            }
            return Ok(());
        }
        let simds = if rows <= 8 { 1 } else { 4 };
        let row_blocks = rows.div_ceil(8 * simds);
        // Split long key ranges across threadgroups when few query rows exist.
        let splits = if m * c.heads <= MAX_SPLIT_ROWS {
            let target: usize = std::env::var("FERRUM_HYBRID_SPLIT_TARGET")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(SPLIT_TARGET_GROUPS);
            let min_keys: usize = std::env::var("FERRUM_HYBRID_SPLIT_MIN_KEYS")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(SPLIT_MIN_KEYS);
            let wanted = (target / (row_blocks * c.kv_heads)).max(1);
            wanted.min(keys.div_ceil(min_keys)).clamp(1, MAX_SPLITS)
        } else {
            1
        };
        let keys_per_split = keys.div_ceil(splits).next_multiple_of(32);
        let splits = keys.div_ceil(keys_per_split);
        let args = Params::default()
            .u(m)?
            .u(c.heads)?
            .u(c.kv_heads)?
            .u(group)?
            .u(pos)?
            .u(c.kv_heads * c.head_dim)?
            .u(keys_per_split)?
            .u(splits)?
            .0;
        d.dispatch_hybrid(
            if simds == 1 {
                "h_flash_attn_1"
            } else {
                "h_flash_attn_4"
            },
            &[q.binding(), kc.binding(), vc.binding(), qg.binding()],
            &[att.binding(), s.attn_partial.binding(), s.attn_ml.binding()],
            &args,
            [row_blocks, c.kv_heads, splits],
            [32 * simds, 1, 1],
            simds * 5888,
        )?;
        if splits > 1 {
            d.dispatch_hybrid(
                "h_flash_reduce",
                &[s.attn_partial.binding(), s.attn_ml.binding(), qg.binding()],
                &[att.binding()],
                &args,
                [m * c.heads, 1, 1],
                [c.head_dim, 1, 1],
                0,
            )?;
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn delta(
        &self,
        d: &MetalDevice,
        s: &Scratch,
        w: &DeltaWeights,
        state: &HybridState,
        layer: usize,
        xn: &Tensor,
        out: &Tensor,
        m: usize,
    ) -> Result<()> {
        let c = &self.config;
        let channels = c.conv_channels();
        let vd = c.ssm_value_dim();
        let heads = c.ssm_v_heads;
        let qkv = rows(&s.qkv, 0, m, channels)?;
        let z = rows(&s.z, 0, m, vd)?;
        let alpha = rows(&s.alpha, 0, m, heads)?;
        let beta = rows(&s.beta, 0, m, heads)?;
        let g = rows(&s.g, 0, m, heads)?;
        let b = rows(&s.b, 0, m, heads)?;
        let conv = rows(&s.conv, 0, m, channels)?;
        let delta = rows(&s.delta, 0, m, vd)?;
        let gated = rows(&s.gated, 0, m, vd)?;
        self.dump(d, "attn_norm", layer, xn)?;
        self.project(d, &w.qkv, xn, m, &qkv)?;
        self.project(d, &w.z, xn, m, &z)?;
        self.project(d, &w.alpha, xn, m, &alpha)?;
        self.project(d, &w.beta, xn, m, &beta)?;
        self.dump(d, "linear_attn_qkv_mixed", layer, &qkv)?;
        self.dump(d, "z", layer, &z)?;
        self.dump(d, "alpha", layer, &alpha)?;
        self.dump(d, "beta", layer, &beta)?;
        d.dispatch_hybrid(
            "h_gdn_gates",
            &[
                alpha.binding(),
                beta.binding(),
                w.dt_bias.binding(),
                w.a.binding(),
            ],
            &[g.binding(), b.binding()],
            &Params::default().u(heads)?.u(m)?.0,
            [(heads * m).div_ceil(64), 1, 1],
            [64, 1, 1],
            0,
        )?;
        let conv_state = state.conv[layer]
            .as_ref()
            .expect("delta layer has conv state");
        d.dispatch_hybrid(
            "h_gdn_conv",
            &[qkv.binding(), w.conv.binding()],
            &[conv.binding(), conv_state.binding()],
            &Params::default()
                .u(channels)?
                .u(m)?
                .u(c.ssm_key_dim())?
                .u(c.ssm_head_dim)?
                .f(c.eps)
                .0,
            [channels / c.ssm_head_dim, 1, 1],
            [c.ssm_head_dim, 1, 1],
            0,
        )?;
        let ssm = state.ssm[layer]
            .as_ref()
            .expect("delta layer has recurrent state");
        const NSG: usize = 4;
        d.dispatch_hybrid(
            "h_gdn_recurrent",
            &[conv.binding(), g.binding(), b.binding()],
            &[delta.binding(), ssm.binding()],
            &Params::default()
                .u(m)?
                .u(heads)?
                .u(c.ssm_k_heads)?
                .u(channels)?
                .u(c.ssm_key_dim())?
                .f(1. / (c.ssm_head_dim as f32).sqrt())
                .0,
            [c.ssm_head_dim / NSG, heads, 1],
            [32, NSG, 1],
            0,
        )?;
        self.dump(d, "attn_output", layer, &delta)?;
        d.dispatch_hybrid(
            "h_gated_rmsnorm",
            &[delta.binding(), z.binding(), w.norm.binding()],
            &[gated.binding()],
            &Params::default().u(heads)?.u(c.ssm_head_dim)?.f(c.eps).0,
            [m * heads, 1, 1],
            [c.ssm_head_dim, 1, 1],
            0,
        )?;
        self.dump(d, "final_output", layer, &gated)?;
        self.project(d, &w.out, &gated, m, out)?;
        self.dump(d, "linear_attn_out", layer, out)
    }

    #[allow(clippy::too_many_arguments)]
    fn moe_ffn(
        &self,
        d: &MetalDevice,
        s: &Scratch,
        f: &MoeFfn,
        xn: &Tensor,
        out: &Tensor,
        m: usize,
    ) -> Result<()> {
        let c = &self.config;
        let moe = c.moe.expect("MoE layer has expert geometry");
        let ms = s.moe.as_ref().expect("MoE model has MoE scratch");
        let (experts, used, hidden, ef) = (moe.experts, moe.experts_used, c.hidden, moe.expert_ffn);
        let routes = m * used;
        let logits = rows(&ms.logits, 0, m, experts)?;
        let ids = ms.ids.view(0, [routes])?;
        let weights = ms.weights.view(0, [routes])?;
        let shared_gate = ms.shared_gate.view(0, [m])?;
        let shared = rows(&ms.shared, 0, m, hidden)?;
        let gate = rows(&ms.gate, 0, routes, ef)?;
        let up = rows(&ms.up, 0, routes, ef)?;
        let act = rows(&ms.act, 0, routes, ef)?;
        let down = rows(&ms.down, 0, routes, hidden)?;
        self.project(d, &f.router, xn, m, &logits)?;
        d.dispatch_hybrid(
            "h_moe_route",
            &[logits.binding(), xn.binding(), f.shared_gate_inp.binding()],
            &[ids.binding(), weights.binding(), shared_gate.binding()],
            &Params::default().u(experts)?.u(used)?.u(hidden)?.u(m)?.0,
            [m, 1, 1],
            [experts, 1, 1],
            0,
        )?;
        self.dense_ffn(d, s, &f.shared, xn, &shared, m)?;
        if std::env::var_os("FERRUM_HYBRID_DUMP").is_some() {
            d.synchronize()?;
            let picked: Vec<u32> = ids.to_f32().iter().map(|v| v.to_bits()).collect();
            eprintln!(
                "dump ffn_moe_topk: sum {} ids {:?}",
                picked.iter().map(|&v| v as u64).sum::<u64>(),
                picked
            );
        }
        self.dump(d, "ffn_moe_logits", 0, &logits)?;
        self.dump(d, "ffn_moe_weights", 0, &weights)?;
        self.dump(d, "ffn_shexp", 0, &shared)?;
        self.dump(d, "shared_expert_gate_sigmoid", 0, &shared_gate)?;
        let identity = routes <= MV_ID_MAX_ROUTES || exact_math();
        if identity {
            self.project_id(d, &f.gate_exps, xn, &ids, routes, used, &gate)?;
            self.project_id(d, &f.up_exps, xn, &ids, routes, used, &up)?;
            self.swiglu(d, &gate, &up, &act)?;
            self.project_id(d, &f.down_exps, &act, &ids, routes, 1, &down)?;
        } else {
            let offsets = ms.offsets.view(0, [experts + 1])?;
            let sorted = ms.sorted.view(0, [routes])?;
            let position = ms.position.view(0, [routes])?;
            let xg = rows(&ms.xg, 0, routes, hidden)?;
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
                &Params::default().u(hidden)?.u(used)?.u(routes)?.0,
                [hidden.div_ceil(256), routes, 1],
                [256, 1, 1],
                0,
            )?;
            self.project_mm_id(d, &f.gate_exps, &xg, &offsets, m, &gate)?;
            self.project_mm_id(d, &f.up_exps, &xg, &offsets, m, &up)?;
            self.swiglu(d, &gate, &up, &act)?;
            self.project_mm_id(d, &f.down_exps, &act, &offsets, m, &down)?;
        }
        let position = ms.position.view(0, [routes])?;
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
                shared.binding(),
                shared_gate.binding(),
            ],
            &[out.binding()],
            &Params::default()
                .u(hidden)?
                .u(used)?
                .u(m)?
                .u(usize::from(identity))?
                .0,
            [(m * hidden).div_ceil(256), 1, 1],
            [256, 1, 1],
            0,
        )
    }

    fn swiglu(&self, d: &MetalDevice, gate: &Tensor, up: &Tensor, out: &Tensor) -> Result<()> {
        let n = gate.numel();
        d.dispatch_hybrid(
            "h_swiglu",
            &[gate.binding(), up.binding()],
            &[out.binding()],
            &Params::default().u(n)?.0,
            [n.div_ceil(256), 1, 1],
            [256, 1, 1],
            0,
        )
    }

    /// Expert GEMV: route r multiplies activation row r / x_div by expert ids[r].
    #[allow(clippy::too_many_arguments)]
    fn project_id(
        &self,
        d: &MetalDevice,
        w: &Matrix,
        x: &Tensor,
        ids: &Tensor,
        routes: usize,
        x_div: usize,
        y: &Tensor,
    ) -> Result<()> {
        let rows_e = w.expert_rows();
        if y.numel() != routes * rows_e || x.numel() * x_div != routes * w.cols {
            return Err(Error::Shape("expert projection geometry".into()));
        }
        let args = Params::default()
            .u(w.cols)?
            .u(rows_e)?
            .u(routes)?
            .u(w.row_bytes)?
            .u(w.cols)?
            .u(rows_e)?
            .u(rows_e)?
            .u(x_div)?
            .0;
        let (name, rows_per_group, simds, shared) = mv_kernel(w.format, true);
        d.dispatch_hybrid(
            name,
            &[w.binding(), x.binding(), ids.binding()],
            &[y.binding()],
            &args,
            [rows_e.div_ceil(rows_per_group), routes, 1],
            [32, simds, 1],
            shared,
        )
    }

    /// Expert GEMM over expert-sorted rows delimited by `offsets`.
    fn project_mm_id(
        &self,
        d: &MetalDevice,
        w: &Matrix,
        x: &Tensor,
        offsets: &Tensor,
        max_rows: usize,
        y: &Tensor,
    ) -> Result<()> {
        let rows_e = w.expert_rows();
        let args = Params::default()
            .u(w.cols)?
            .u(rows_e)?
            .u(0)?
            .u(w.row_bytes)?
            .u(w.cols)?
            .u(rows_e)?
            .u(rows_e)?
            .0;
        let name = match w.format {
            Format::Q4K => "h_mm_id_q4_k",
            Format::Q5K => "h_mm_id_q5_k",
            Format::Q6K => "h_mm_id_q6_k",
            Format::Q8_0 => "h_mm_id_q8_0",
            Format::Q4_0 => "h_mm_id_q4_0",
            Format::Iq4Xs => "h_mm_id_iq4_xs",
            Format::Iq3S => "h_mm_id_iq3_s",
            Format::F32 => return Err(Error::Gguf("F32 experts are unsupported".into())),
        };
        d.dispatch_hybrid(
            name,
            &[w.binding(), x.binding(), offsets.binding()],
            &[y.binding()],
            &args,
            [max_rows.div_ceil(32), rows_e.div_ceil(64), w.experts],
            [128, 1, 1],
            64 * 32 * 2,
        )
    }

    fn dense_ffn(
        &self,
        d: &MetalDevice,
        s: &Scratch,
        f: &DenseFfn,
        xn: &Tensor,
        out: &Tensor,
        m: usize,
    ) -> Result<()> {
        let width = f.gate.rows;
        let gate = rows(&s.ffn_gate, 0, m, width)?;
        let up = rows(&s.ffn_up, 0, m, width)?;
        let act = rows(&s.ffn_act, 0, m, width)?;
        self.project(d, &f.gate, xn, m, &gate)?;
        self.project(d, &f.up, xn, m, &up)?;
        d.dispatch_hybrid(
            "h_swiglu",
            &[gate.binding(), up.binding()],
            &[act.binding()],
            &Params::default().u(m * width)?.0,
            [(m * width).div_ceil(256), 1, 1],
            [256, 1, 1],
            0,
        )?;
        self.project(d, &f.down, &act, m, out)
    }

    /// Debug aid (`FERRUM_HYBRID_DUMP=1`): complete queued work and print the
    /// sum of `t`, named like llama.cpp's eval-callback tensors.
    fn dump(&self, d: &MetalDevice, name: &str, layer: usize, t: &Tensor) -> Result<()> {
        if std::env::var_os("FERRUM_HYBRID_DUMP").is_some() {
            d.synchronize()?;
            let values = t.to_f32();
            let sum: f64 = values.iter().map(|&v| v as f64).sum();
            eprintln!(
                "dump {name}-{layer}: sum {sum:.6} first {:?}",
                &values[..values.len().min(4)]
            );
        }
        Ok(())
    }

    fn fill_zero(&self, d: &MetalDevice, t: &Tensor) -> Result<()> {
        // Zero through the GPU so the write is ordered with queued work.
        let n = t.numel();
        d.dispatch_hybrid(
            "h_zero",
            &[],
            &[t.binding()],
            &Params::default().u(n)?.0,
            [n.div_ceil(256), 1, 1],
            [256, 1, 1],
            0,
        )
    }

    fn get_rows(&self, d: &MetalDevice, w: &Matrix, ids: &Tensor, y: &Tensor) -> Result<()> {
        let tokens = ids.numel();
        let name = match w.format {
            Format::Q4K => "h_get_rows_q4_k",
            Format::Q5K => "h_get_rows_q5_k",
            Format::Q6K => "h_get_rows_q6_k",
            Format::Q8_0 => "h_get_rows_q8_0",
            Format::Q4_0 => "h_get_rows_q4_0",
            Format::Iq4Xs => "h_get_rows_iq4_xs",
            f => return Err(Error::Gguf(format!("no embedding lookup for {}", f.name()))),
        };
        d.dispatch_hybrid(
            name,
            &[w.binding(), ids.binding()],
            &[y.binding()],
            &Params::default().u(w.cols)?.u(w.row_bytes)?.u(tokens)?.0,
            [(w.cols / 16).div_ceil(32), tokens, 1],
            [32, 1, 1],
            0,
        )
    }

    fn rmsnorm(&self, d: &MetalDevice, x: &Tensor, w: &Tensor, y: &Tensor, m: usize) -> Result<()> {
        let n = self.config.hidden;
        d.dispatch_hybrid(
            "h_rmsnorm",
            &[x.binding(), w.binding()],
            &[y.binding()],
            &Params::default().u(n)?.f(self.config.eps).u(m)?.0,
            [m, 1, 1],
            [256, 1, 1],
            0,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn add_rmsnorm(
        &self,
        d: &MetalDevice,
        x: &Tensor,
        r: &Tensor,
        w: &Tensor,
        h: &Tensor,
        y: &Tensor,
        m: usize,
    ) -> Result<()> {
        let n = self.config.hidden;
        d.dispatch_hybrid(
            "h_add_rmsnorm",
            &[x.binding(), r.binding(), w.binding()],
            &[h.binding(), y.binding()],
            &Params::default().u(n)?.f(self.config.eps).u(m)?.0,
            [m, 1, 1],
            [256, 1, 1],
            0,
        )
    }

    /// y[m][rows] = x[m][cols] · Wᵀ.
    pub(crate) fn project(
        &self,
        d: &MetalDevice,
        w: &Matrix,
        x: &Tensor,
        m: usize,
        y: &Tensor,
    ) -> Result<()> {
        if x.numel() != m * w.cols || y.numel() != m * w.rows {
            return Err(Error::Shape(format!(
                "projection [{m}, {}] x [{}, {}]ᵀ into {} values",
                w.cols,
                w.rows,
                w.cols,
                y.numel()
            )));
        }
        let args = Params::default()
            .u(w.cols)?
            .u(w.rows)?
            .u(m)?
            .u(w.row_bytes)?
            .u(w.cols)?
            .u(w.rows)?
            .0;
        let inputs: [Binding; 2] = [w.binding(), x.binding()];
        let outputs: [Binding; 1] = [y.binding()];
        if m > MV_MAX_ROWS && !exact_math() {
            let name = match w.format {
                Format::Q4K => "h_mm_q4_k",
                Format::Q5K => "h_mm_q5_k",
                Format::Q6K => "h_mm_q6_k",
                Format::Q8_0 => "h_mm_q8_0",
                Format::Q4_0 => "h_mm_q4_0",
                Format::Iq4Xs => "h_mm_iq4_xs",
                Format::F32 => "h_mm_f32",
                Format::Iq3S => "h_mm_iq3_s",
            };
            return d.dispatch_hybrid(
                name,
                &inputs,
                &outputs,
                &args,
                [m.div_ceil(128), w.rows.div_ceil(64), 1],
                [128, 1, 1],
                64 * 32 * 2,
            );
        }
        let (mut name, mut rows_per_group, mut simds, shared) = mv_kernel(w.format, false);
        if let Some(variant) = kernel_variant(w.format) {
            (name, rows_per_group, simds) = variant;
        }
        d.dispatch_hybrid(
            name,
            &inputs,
            &outputs,
            &args,
            [w.rows.div_ceil(rows_per_group), m, 1],
            [32, simds, 1],
            shared,
        )
    }
}

/// `FERRUM_HYBRID_EXACT=1`: every projection uses the F32-activation GEMV
/// kernels (no F16 GEMM tiles). Slow; a high-precision reference for evaluation.
fn exact_math() -> bool {
    static EXACT: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *EXACT.get_or_init(|| std::env::var_os("FERRUM_HYBRID_EXACT").is_some_and(|v| v != "0"))
}

/// Experiment hook: `FERRUM_HYBRID_MV_Q4K=r1s4` (etc.) selects a GEMV shape.
fn kernel_variant(format: Format) -> Option<(&'static str, usize, usize)> {
    let key = match format {
        Format::Q4K => "FERRUM_HYBRID_MV_Q4K",
        Format::Q6K => "FERRUM_HYBRID_MV_Q6K",
        _ => return None,
    };
    let v = std::env::var(key).ok()?;
    let q4 = format == Format::Q4K;
    Some(match (v.as_str(), q4) {
        ("r1s2", true) => ("h_mv_q4_k_r1s2", 2, 2),
        ("r1s4", true) => ("h_mv_q4_k_r1s4", 4, 4),
        ("r2s4", true) => ("h_mv_q4_k_r2s4", 8, 4),
        ("r4s2", true) => ("h_mv_q4_k_r4s2", 8, 2),
        ("r4s1", true) => ("h_mv_q4_k_r4s1", 4, 1),
        ("r1s2", false) => ("h_mv_q6_k_r1s2", 2, 2),
        ("r1s4", false) => ("h_mv_q6_k_r1s4", 4, 4),
        ("r2s4", false) => ("h_mv_q6_k_r2s4", 8, 4),
        ("r4s2", false) => ("h_mv_q6_k_r4s2", 8, 2),
        _ => return None,
    })
}

/// Query tokens from which attention uses the TensorOps flash kernel.
const MPP_ATTENTION_MIN_TOKENS: usize = 16;

/// Most key splits for few-row attention, and (token, head) rows they may cover.
const MAX_SPLITS: usize = 256;
/// Few-row attention aims for this many threadgroups, each with at least
/// `SPLIT_MIN_KEYS` keys, so long-context decode fills the GPU.
const SPLIT_TARGET_GROUPS: usize = 64;
const SPLIT_MIN_KEYS: usize = 256;
/// Single-token decode attention: target threadgroups and keys per split.
const DECODE_SPLIT_TARGET: usize = 128;
const DECODE_SPLIT_MIN_KEYS: usize = 256;
const MAX_SPLIT_ROWS: usize = 64;
/// Activation rows at or below which projections use the GEMV kernels.
const MV_MAX_ROWS: usize = 4;
/// Expert routes at or below which experts use per-route GEMVs.
const MV_ID_MAX_ROUTES: usize = 32;

/// GEMV kernel, output rows per threadgroup, SIMD groups, threadgroup bytes.
fn mv_kernel(format: Format, expert: bool) -> (&'static str, usize, usize, usize) {
    let pick = |plain, id| if expert { id } else { plain };
    match format {
        Format::Q4K => (pick("h_mv_q4_k", "h_mv_id_q4_k"), 4, 2, 0),
        Format::Q5K => (pick("h_mv_q5_k", "h_mv_id_q5_k"), 2, 2, 0),
        Format::Q6K => (pick("h_mv_q6_k", "h_mv_id_q6_k"), 4, 2, 0),
        Format::Q4_0 => (pick("h_mv_q4_0", "h_mv_id_q4_0"), 8, 2, 0),
        Format::F32 => ("h_mv_f32", 8, 2, 0),
        Format::Q8_0 => (pick("h_mv_q8_0", "h_mv_id_q8_0"), 2, 4, 32 * 2 * 4),
        Format::Iq4Xs => (pick("h_mv_iq4_xs", "h_mv_id_iq4_xs"), 4, 2, 32 * 4),
        Format::Iq3S => (pick("h_mv_iq3_s", "h_mv_id_iq3_s"), 8, 2, 512 * 4),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn random(n: usize, seed: u64) -> Vec<f32> {
        let mut x = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
        (0..n)
            .map(|_| {
                x ^= x << 13;
                x ^= x >> 7;
                x ^= x << 17;
                ((x >> 40) as f32 / (1u64 << 24) as f32) * 2. - 1.
            })
            .collect()
    }

    /// Flash attention (both SIMD-group shapes, split and unsplit) against the
    /// reference online-softmax kernel for several GQA geometries.
    #[test]
    fn flash_attention_matches_reference() {
        let d = MetalDevice::new().unwrap();
        let cases: [(usize, usize, usize, usize); 8] = [
            (24, 4, 1, 0),
            (24, 4, 1, 700),
            (16, 2, 1, 300),
            (16, 2, 5, 40),
            (16, 2, 37, 0),
            (24, 4, 37, 90),
            (16, 2, 64, 128),
            (24, 4, 150, 70),
        ];
        for (heads, kv_heads, m, pos) in cases {
            let dim = 256;
            let stride = kv_heads * dim;
            let keys = pos + m;
            let cap = keys.next_multiple_of(32);
            let q = Tensor::from_f32(
                &d,
                [m, heads * dim],
                DType::F16,
                &random(m * heads * dim, 1)
                    .iter()
                    .map(|v| v / 16.)
                    .collect::<Vec<_>>(),
            )
            .unwrap();
            let kc =
                Tensor::from_f32(&d, [cap, stride], DType::F16, &random(cap * stride, 2)).unwrap();
            let vc =
                Tensor::from_f32(&d, [cap, stride], DType::F16, &random(cap * stride, 3)).unwrap();
            let qg = Tensor::from_f32(
                &d,
                [m, 2 * heads * dim],
                DType::F32,
                &random(m * 2 * heads * dim, 4),
            )
            .unwrap();
            let reference = Tensor::zeros(&d, [m, heads * dim], DType::F32).unwrap();
            let args = Params::default()
                .u(m)
                .unwrap()
                .u(heads)
                .unwrap()
                .u(kv_heads)
                .unwrap()
                .u(dim)
                .unwrap()
                .u(pos)
                .unwrap()
                .u(stride)
                .unwrap()
                .f(1.);
            {
                let e = d.execution_with_shared_encoder(true).unwrap();
                d.dispatch_hybrid(
                    "h_attention",
                    &[q.binding(), kc.binding(), vc.binding(), qg.binding()],
                    &[reference.binding()],
                    &args.0,
                    [heads, m, 1],
                    [dim, 1, 1],
                    0,
                )
                .unwrap();
                e.finish().unwrap();
            }
            let expected = reference.to_f32();
            let group = heads / kv_heads;
            for simds in [1usize, 4] {
                for splits_wanted in [1usize, 3] {
                    let out = Tensor::zeros(&d, [m, heads * dim], DType::F32).unwrap();
                    let partial = Tensor::zeros(&d, [3 * m * heads * dim], DType::F32).unwrap();
                    let ml = Tensor::zeros(&d, [3 * m * heads * 2], DType::F32).unwrap();
                    let kps = keys.div_ceil(splits_wanted).next_multiple_of(32);
                    let splits = keys.div_ceil(kps);
                    let fa = Params::default()
                        .u(m)
                        .unwrap()
                        .u(heads)
                        .unwrap()
                        .u(kv_heads)
                        .unwrap()
                        .u(group)
                        .unwrap()
                        .u(pos)
                        .unwrap()
                        .u(stride)
                        .unwrap()
                        .u(kps)
                        .unwrap()
                        .u(splits)
                        .unwrap();
                    let e = d.execution_with_shared_encoder(true).unwrap();
                    d.dispatch_hybrid(
                        if simds == 1 {
                            "h_flash_attn_1"
                        } else {
                            "h_flash_attn_4"
                        },
                        &[q.binding(), kc.binding(), vc.binding(), qg.binding()],
                        &[out.binding(), partial.binding(), ml.binding()],
                        &fa.0,
                        [(m * group).div_ceil(8 * simds), kv_heads, splits],
                        [32 * simds, 1, 1],
                        simds * 5888,
                    )
                    .unwrap();
                    if splits > 1 {
                        d.dispatch_hybrid(
                            "h_flash_reduce",
                            &[partial.binding(), ml.binding(), qg.binding()],
                            &[out.binding()],
                            &fa.0,
                            [m * heads, 1, 1],
                            [dim, 1, 1],
                            0,
                        )
                        .unwrap();
                    }
                    e.finish().unwrap();
                    let got = out.to_f32();
                    let worst = got
                        .iter()
                        .zip(&expected)
                        .map(|(a, b)| (a - b).abs())
                        .fold(0f32, f32::max);
                    assert!(
                        worst < 2e-3,
                        "heads {heads}/{kv_heads} m {m} pos {pos} simds {simds} splits {splits}: max error {worst}"
                    );
                }
            }
            // Vector decode kernel (one query token).
            if m == 1 {
                for splits_wanted in [1usize, 5] {
                    let out = Tensor::zeros(&d, [m, heads * dim], DType::F32).unwrap();
                    let partial = Tensor::zeros(&d, [8 * heads * dim], DType::F32).unwrap();
                    let ml = Tensor::zeros(&d, [8 * heads * 2], DType::F32).unwrap();
                    let kps = keys.div_ceil(splits_wanted).next_multiple_of(32);
                    let splits = keys.div_ceil(kps);
                    let fa = Params::default()
                        .u(1)
                        .unwrap()
                        .u(heads)
                        .unwrap()
                        .u(kv_heads)
                        .unwrap()
                        .u(group)
                        .unwrap()
                        .u(pos)
                        .unwrap()
                        .u(stride)
                        .unwrap()
                        .u(kps)
                        .unwrap()
                        .u(splits)
                        .unwrap();
                    let e = d.execution_with_shared_encoder(true).unwrap();
                    d.dispatch_hybrid(
                        "h_attn_decode",
                        &[q.binding(), kc.binding(), vc.binding(), qg.binding()],
                        &[out.binding(), partial.binding(), ml.binding()],
                        &fa.0,
                        [kv_heads, splits, 1],
                        [128, 1, 1],
                        8 * 256 * 2 + 4 * 8 * 2 * 4 + 2 * 8 * 256 * 4,
                    )
                    .unwrap();
                    if splits > 1 {
                        d.dispatch_hybrid(
                            "h_flash_reduce",
                            &[partial.binding(), ml.binding(), qg.binding()],
                            &[out.binding()],
                            &fa.0,
                            [heads, 1, 1],
                            [dim, 1, 1],
                            0,
                        )
                        .unwrap();
                    }
                    e.finish().unwrap();
                    let worst = out
                        .to_f32()
                        .iter()
                        .zip(&expected)
                        .map(|(a, b)| (a - b).abs())
                        .fold(0f32, f32::max);
                    assert!(
                        worst < 2e-3,
                        "decode heads {heads}/{kv_heads} pos {pos} splits {splits}: max error {worst}"
                    );
                }
            }
            // TensorOps kernel (prompt chunks).
            let out = Tensor::zeros(&d, [m, heads * dim], DType::F32).unwrap();
            let fa = Params::default()
                .u(m)
                .unwrap()
                .u(heads)
                .unwrap()
                .u(kv_heads)
                .unwrap()
                .u(group)
                .unwrap()
                .u(pos)
                .unwrap()
                .u(stride)
                .unwrap()
                .u(0)
                .unwrap()
                .u(1)
                .unwrap();
            let e = d.execution_with_shared_encoder(true).unwrap();
            d.dispatch_hybrid(
                "h_flash_attn_mpp",
                &[q.binding(), kc.binding(), vc.binding(), qg.binding()],
                &[out.binding()],
                &fa.0,
                [m.div_ceil(64), heads, 1],
                [128, 1, 1],
                64 * 64 * 4 + 64 * 64 * 2 + 3 * 64 * 4,
            )
            .unwrap();
            e.finish().unwrap();
            let worst = out
                .to_f32()
                .iter()
                .zip(&expected)
                .map(|(a, b)| (a - b).abs())
                .fold(0f32, f32::max);
            assert!(
                worst < 2e-3,
                "TensorOps heads {heads}/{kv_heads} m {m} pos {pos}: max error {worst}"
            );
        }
    }
}
