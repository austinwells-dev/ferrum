//! Forward pass for Qwen3.5-family hybrid models on fixed, preallocated storage.
#![forbid(unsafe_code)]
use super::{
    config::HybridConfig,
    weights::{AttentionWeights, DeltaWeights, DenseFfn, Ffn, Format, Matrix, Mixer, Weights},
};
use crate::{DType, Error, MetalDevice, Result, Tensor, metal::MetalBuffer};
use std::cell::RefCell;

type Binding<'a> = (&'a MetalBuffer, usize, usize);

/// Little-endian kernel parameter block.
#[derive(Default)]
struct Params(Vec<u8>);
impl Params {
    fn u(mut self, v: usize) -> Result<Self> {
        let v =
            u32::try_from(v).map_err(|_| Error::Shape("kernel parameter exceeds u32".into()))?;
        self.0.extend_from_slice(&v.to_le_bytes());
        Ok(self)
    }
    fn f(mut self, v: f32) -> Self {
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
                    .push(Some(Tensor::zeros(d, [capacity, kv_row], DType::F16)?));
                state
                    .v
                    .push(Some(Tensor::zeros(d, [capacity, kv_row], DType::F16)?));
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
struct Scratch {
    chunk: usize,
    x: [Tensor; 2],
    xn: Tensor,
    mix: Tensor,
    qg: Tensor,
    k: Tensor,
    v: Tensor,
    q: Tensor,
    att: Tensor,
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
            q: f(&[chunk, c.heads * c.head_dim])?,
            att: f(&[chunk, c.heads * c.head_dim])?,
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
        .sum()
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
    weights: Weights,
    scratch: RefCell<Scratch>,
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
                Ffn::Moe(_) => {
                    return Err(Error::Config(
                        "MoE feed-forward is not implemented yet".into(),
                    ));
                }
            }
            let norm = self
                .weights
                .layers
                .get(i + 1)
                .map_or(&self.weights.output_norm, |l| &l.attn_norm);
            self.add_rmsnorm(d, cur, &mix, norm, next, &xn, m)?;
            std::mem::swap(&mut cur, &mut next);
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
        let q = rows(&s.q, 0, m, q_width)?;
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
            .u(kv_width)?;
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
        let args = Params::default()
            .u(m)?
            .u(c.heads)?
            .u(c.kv_heads)?
            .u(c.head_dim)?
            .u(pos)?
            .u(kv_width)?
            .f(1. / (c.head_dim as f32).sqrt());
        d.dispatch_hybrid(
            "h_attention",
            &[q.binding(), kc.binding(), vc.binding(), qg.binding()],
            &[att.binding()],
            &args.0,
            [c.heads, m, 1],
            [c.head_dim, 1, 1],
            0,
        )?;
        self.project(d, &a.o, &att, m, out)
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
        self.project(d, &w.qkv, xn, m, &qkv)?;
        self.project(d, &w.z, xn, m, &z)?;
        self.project(d, &w.alpha, xn, m, &alpha)?;
        self.project(d, &w.beta, xn, m, &beta)?;
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
        d.dispatch_hybrid(
            "h_gated_rmsnorm",
            &[delta.binding(), z.binding(), w.norm.binding()],
            &[gated.binding()],
            &Params::default().u(heads)?.u(c.ssm_head_dim)?.f(c.eps).0,
            [m * heads, 1, 1],
            [c.ssm_head_dim, 1, 1],
            0,
        )?;
        self.project(d, &w.out, &gated, m, out)
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
    fn project(&self, d: &MetalDevice, w: &Matrix, x: &Tensor, m: usize, y: &Tensor) -> Result<()> {
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
        if m > MV_MAX_ROWS {
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
        // (kernel, output rows per threadgroup, SIMD groups, threadgroup bytes)
        let (name, rows_per_group, simds, shared) = match w.format {
            Format::Q4K => ("h_mv_q4_k", 4, 2, 0),
            Format::Q5K => ("h_mv_q5_k", 2, 2, 0),
            Format::Q6K => ("h_mv_q6_k", 4, 2, 0),
            Format::Q4_0 => ("h_mv_q4_0", 8, 2, 0),
            Format::F32 => ("h_mv_f32", 8, 2, 0),
            Format::Q8_0 => ("h_mv_q8_0", 2, 4, 32 * 2 * 4),
            Format::Iq4Xs => ("h_mv_iq4_xs", 4, 2, 32 * 4),
            Format::Iq3S => ("h_mv_iq3_s", 8, 2, 512 * 4),
        };
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

/// Activation rows at or below which projections use the GEMV kernels.
const MV_MAX_ROWS: usize = 4;
