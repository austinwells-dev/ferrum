//! LFM3-Audio encoder (`clip.audio.projector_type = lfm3a`), after llama.cpp's
//! `clip_graph_lfm3a`.
//!
//! Log-mel frames `[time, mel]` go through three stride-2 3x3 convs over time
//! and frequency (a full conv to 256 channels, then two depthwise plus
//! pointwise pairs, each followed by SiLU), so 8 frames become one row. The
//! flattened channels-by-frequency row is projected to the encoder width and
//! runs through a causal LFM2-style stack: gated short convs (kernel 3) and
//! grouped-query attention with QK RMSNorm and NEOX RoPE at base 1e6 over a
//! window of the last `window` rows. A final RMSNorm, a LayerNorm and a GELU
//! MLP give one language-model row per 8 frames (80 ms).
//!
//! The whole clip is encoded at once, as llama.cpp does with a zero streaming
//! state. Speech output (`lfm3a_gen`) is not this path.
#![forbid(unsafe_code)]

use super::{Gemm, ImageEmbedding, Source, run_add, run_gemm};
use crate::{DType, Error, MetalDevice, Result, Tensor, hybrid::engine::Params};

/// Hardcoded in llama.cpp's loader; the converter checks it.
const ROPE_THETA: f64 = 1_000_000.;
/// The projector LayerNorm's epsilon, fixed in llama.cpp's graph.
const PROJECTOR_EPS: f32 = 1e-5;
/// Short-conv taps (kernel 3, two of history).
const CONV_STATE: usize = 2;

#[derive(Debug, Clone)]
pub(super) struct Config {
    pub n_mel: usize,
    pub hidden: usize,
    pub heads: usize,
    pub kv_heads: usize,
    pub ffn: usize,
    pub layers: usize,
    pub window: usize,
    pub eps: f32,
    pub out_dim: usize,
}

impl Config {
    fn head_dim(&self) -> usize {
        self.hidden / self.heads
    }
}

/// Rows a clip of `frames` mel frames becomes: three halvings, rounding up.
pub(super) fn tokens_for(frames: usize) -> usize {
    frames.div_ceil(2).div_ceil(2).div_ceil(2)
}

struct Proj {
    weight: Tensor,
    bias: Option<Tensor>,
}

struct Stem {
    channels: usize,
    /// 3x3 taps per output channel, `[channel][kf][kt]`.
    w0: Tensor,
    b0: Tensor,
    w1: Tensor,
    b1: Tensor,
    pw1: Proj,
    w2: Tensor,
    b2: Tensor,
    pw2: Proj,
    /// Columns reordered from ggml's channel-major flattening to the
    /// frequency-major rows the convs leave.
    out: Proj,
}

struct Ffn {
    ln2: Tensor,
    gate: Tensor,
    up: Tensor,
    down: Tensor,
}

struct Conv {
    ln1: Tensor,
    in_proj: Tensor,
    /// `[hidden, 3]`, tap 0 the oldest.
    taps: Tensor,
    out_proj: Tensor,
    ffn: Ffn,
}

struct Attn {
    ln1: Tensor,
    q: Tensor,
    k: Tensor,
    v: Tensor,
    q_norm: Tensor,
    k_norm: Tensor,
    out: Tensor,
    ffn: Ffn,
}

/// Both arms are boxed: each block is wider than the enum discriminant.
enum Block {
    Conv(Box<Conv>),
    Attn(Box<Attn>),
}

/// Device bytes `Encoder::encode` allocates for the activations of a clip of
/// `frames` mel frames: one buffer per entry, page-rounded as Metal allocates
/// them, the returned rows included. Keep in step with `encode`; `hybrid_audio_check --memory` compares the two
/// for a full-length clip (small buffers share pages, so a tiny test cannot).
pub(super) fn activation_bytes(c: &Config, channels: usize, mid: usize, frames: usize) -> usize {
    let (h, hd) = (c.hidden, c.head_dim());
    let t2 = frames.div_ceil(2).div_ceil(2);
    let n = tokens_for(frames);
    let f2 = c.n_mel.div_ceil(2).div_ceil(2);
    let f3 = f2.div_ceil(2);
    let kv = c.kv_heads * hd;
    let counts = [
        frames * c.n_mel,
        t2 * f2 * channels,
        t2 * f2 * channels,
        n * f3 * channels,
        n * f3 * channels,
        n * h,
        n * h,
        n * h,
        n * 3 * h,
        n * h,
        n * h,
        n * h,
        n * kv,
        n * kv,
        n * kv,
        n * h,
        n * c.ffn,
        n * c.ffn,
        n * c.ffn,
        n * n,
        n * hd / 2,
        n * hd / 2,
        n * n,
        n * mid,
        n * c.out_dim,
    ];
    counts
        .iter()
        .map(|&e| crate::hybrid::engine::page(e * 4))
        .sum()
}

/// Device bytes of the rows `encode` returns for a clip of `frames` frames,
/// which outlive it (the runtime caches them).
pub(super) fn output_bytes(c: &Config, frames: usize) -> usize {
    crate::hybrid::engine::page(tokens_for(frames) * c.out_dim * 4)
}

pub(super) struct Encoder {
    pub config: Config,
    stem: Stem,
    blocks: Vec<Block>,
    post_norm: Tensor,
    mm_norm: (Tensor, Tensor),
    mm1: Proj,
    mm3: Proj,
    /// Width of the projector's hidden layer.
    mid: usize,
    /// Placeholder bias and the zero short-conv history: every clip starts
    /// from silence, as llama.cpp's zero streaming state.
    zero: Tensor,
    conv_state: Tensor,
    /// Bound as the unused B*x tape of `h_short_conv`.
    no_tape: Tensor,
    pub weight_bytes: usize,
    /// Device bytes the latest clip's activations took (Metal's allocated
    /// size before and after they were made).
    pub activation_bytes: std::cell::Cell<usize>,
}

pub(super) fn load(d: &MetalDevice, config: Config, source: &mut dyn Source) -> Result<Encoder> {
    let bad = |what: String| Error::Config(format!("unsupported lfm3a encoder: {what}"));
    let c = &config;
    let h = c.hidden;
    if c.heads == 0 || !h.is_multiple_of(c.heads) || !c.head_dim().is_multiple_of(2) {
        return Err(bad(format!("{} heads for width {h}", c.heads)));
    }
    if c.kv_heads == 0 || !c.heads.is_multiple_of(c.kv_heads) {
        return Err(bad(format!(
            "{} KV heads for {} query heads",
            c.kv_heads, c.heads
        )));
    }
    if c.window == 0 || c.n_mel == 0 {
        return Err(bad(format!(
            "window {} with {} mel bins",
            c.window, c.n_mel
        )));
    }
    let hd = c.head_dim();
    let bytes = std::cell::Cell::new(0usize);
    let values = |source: &mut dyn Source, name: &str, len: usize| -> Result<Vec<f32>> {
        let v = source.values(name)?;
        if v.len() != len {
            return Err(Error::Weight {
                name: name.into(),
                message: format!("{} values, expected {len}", v.len()),
            });
        }
        bytes.set(bytes.get() + len * 4);
        Ok(v)
    };
    let vector = |v: Vec<f32>| Tensor::from_f32(d, [v.len()], DType::F32, &v);
    // 1x1 convs are 4-D in the file, so they are read as values.
    let dense = |v: &[f32], rows: usize, cols: usize| {
        bytes.set(bytes.get() + rows * cols * 2 - v.len() * 4);
        Tensor::from_f32(d, [rows, cols], DType::F16, v)
    };
    let matrix = |source: &mut dyn Source, name: &str, rows: usize, cols: usize| {
        bytes.set(bytes.get() + rows * cols * 2);
        source.matrix(d, name, rows, cols)
    };
    let width = |source: &mut dyn Source, name: &str| {
        source
            .values(name)
            .map(|v| v.len())
            .map_err(|_| bad(format!("no {name}")))
    };

    let ch = width(source, "a.conv1d.0.bias")?;
    let f_out = c.n_mel.div_ceil(2).div_ceil(2).div_ceil(2);
    let flat = ch * f_out;
    let taps = |source: &mut dyn Source, i: usize| -> Result<(Tensor, Tensor)> {
        let w = values(source, &format!("a.conv1d.{i}.weight"), ch * 9)?;
        let b = values(source, &format!("a.conv1d.{i}.bias"), ch)?;
        Ok((vector(w)?, vector(b)?))
    };
    let (w0, b0) = taps(source, 0)?;
    let (w1, b1) = taps(source, 2)?;
    let (w2, b2) = taps(source, 5)?;
    let pointwise = |source: &mut dyn Source, i: usize| -> Result<Proj> {
        let w = values(source, &format!("a.conv1d.{i}.weight"), ch * ch)?;
        let b = values(source, &format!("a.conv1d.{i}.bias"), ch)?;
        Ok(Proj {
            weight: dense(&w, ch, ch)?,
            bias: Some(vector(b)?),
        })
    };
    let pw1 = pointwise(source, 3)?;
    let pw2 = pointwise(source, 6)?;
    let out = {
        let w = values(source, "a.pre_encode.out.weight", h * flat)?;
        let mut permuted = vec![0f32; w.len()];
        for (row, dst) in w.chunks_exact(flat).zip(permuted.chunks_exact_mut(flat)) {
            for k in 0..ch {
                for f in 0..f_out {
                    dst[f * ch + k] = row[k * f_out + f];
                }
            }
        }
        Proj {
            weight: dense(&permuted, h, flat)?,
            bias: Some(vector(values(source, "a.pre_encode.out.bias", h)?)?),
        }
    };
    let stem = Stem {
        channels: ch,
        w0,
        b0,
        w1,
        b1,
        pw1,
        w2,
        b2,
        pw2,
        out,
    };

    let mut blocks = Vec::with_capacity(c.layers);
    for i in 0..c.layers {
        let n = |s: &str| format!("a.blk.{i}.{s}");
        let conv = source.contains(&n("conv_dw.weight"));
        let attn = source.contains(&n("attn_q.weight"));
        if conv == attn {
            return Err(bad(format!(
                "layer {i} is not exactly one of conv or attention"
            )));
        }
        let ln1 = vector(values(source, &n("ln1.weight"), h)?)?;
        let ffn = Ffn {
            ln2: vector(values(source, &n("ln2.weight"), h)?)?,
            gate: matrix(source, &n("ffn_gate.weight"), c.ffn, h)?,
            up: matrix(source, &n("ffn_up.weight"), c.ffn, h)?,
            down: matrix(source, &n("ffn_down.weight"), h, c.ffn)?,
        };
        blocks.push(if conv {
            Block::Conv(Box::new(Conv {
                ln1,
                in_proj: matrix(source, &n("conv_pw1.weight"), 3 * h, h)?,
                taps: vector(values(source, &n("conv_dw.weight"), h * (CONV_STATE + 1))?)?,
                out_proj: matrix(source, &n("conv_pw2.weight"), h, h)?,
                ffn,
            }))
        } else {
            Block::Attn(Box::new(Attn {
                ln1,
                q: matrix(source, &n("attn_q.weight"), h, h)?,
                k: matrix(source, &n("attn_k.weight"), c.kv_heads * hd, h)?,
                v: matrix(source, &n("attn_v.weight"), c.kv_heads * hd, h)?,
                q_norm: vector(values(source, &n("attn_q_norm.weight"), hd)?)?,
                k_norm: vector(values(source, &n("attn_k_norm.weight"), hd)?)?,
                out: matrix(source, &n("attn_out.weight"), h, h)?,
                ffn,
            }))
        });
    }
    let post_norm = vector(values(source, "a.post_ln.weight", h)?)?;
    let mm_norm = (
        vector(values(source, "mm.a.mlp.0.weight", h)?)?,
        vector(values(source, "mm.a.mlp.0.bias", h)?)?,
    );
    let mid = width(source, "mm.a.mlp.1.bias")?;
    let mm1 = Proj {
        weight: matrix(source, "mm.a.mlp.1.weight", mid, h)?,
        bias: Some(vector(values(source, "mm.a.mlp.1.bias", mid)?)?),
    };
    let mm3 = Proj {
        weight: matrix(source, "mm.a.mlp.3.weight", c.out_dim, mid)?,
        bias: Some(vector(values(source, "mm.a.mlp.3.bias", c.out_dim)?)?),
    };
    Ok(Encoder {
        stem,
        blocks,
        post_norm,
        mm_norm,
        mm1,
        mm3,
        zero: Tensor::zeros(d, [1], DType::F32)?,
        conv_state: Tensor::zeros(d, [CONV_STATE * h], DType::F32)?,
        no_tape: Tensor::zeros(d, [1], DType::F32)?,
        mid,
        weight_bytes: bytes.get(),
        activation_bytes: std::cell::Cell::new(0),
        config,
    })
}

impl Encoder {
    /// Encode `features`, `[frames, n_mel]` log-mel rows, into
    /// `tokens_for(frames)` language-model rows.
    pub(super) fn encode(
        &self,
        d: &MetalDevice,
        features: &[f32],
        hash: u64,
    ) -> Result<ImageEmbedding> {
        let c = &self.config;
        let frames = features.len() / c.n_mel;
        if frames == 0 || features.len() != frames * c.n_mel {
            return Err(Error::Parameter(format!(
                "{} mel values are not whole frames of {}",
                features.len(),
                c.n_mel
            )));
        }
        let (h, hd, ch) = (c.hidden, c.head_dim(), self.stem.channels);
        let t1 = frames.div_ceil(2);
        let (t2, t3) = (t1.div_ceil(2), tokens_for(frames));
        let f1 = c.n_mel.div_ceil(2);
        let (f2, f3) = (f1.div_ceil(2), f1.div_ceil(2).div_ceil(2));
        let n = t3;
        let before = d.allocated_bytes();
        let zeros = |dims: &[usize]| Tensor::zeros(d, dims, DType::F32);
        let mel = Tensor::from_f32(d, [frames, c.n_mel], DType::F32, features)?;
        let s2 = zeros(&[t2 * f2, ch])?;
        let p2 = zeros(&[t2 * f2, ch])?;
        let s3 = zeros(&[t3 * f3, ch])?;
        let p3 = zeros(&[t3 * f3, ch])?;
        let x = zeros(&[n, h])?;
        let xn = zeros(&[n, h])?;
        let tmp = zeros(&[n, h])?;
        let packed = zeros(&[n, 3 * h])?;
        let conv = zeros(&[n, h])?;
        let kv_width = c.kv_heads * hd;
        let q = zeros(&[n, h])?;
        let qn = zeros(&[n, h])?;
        let k = zeros(&[n, kv_width])?;
        let kn = zeros(&[n, kv_width])?;
        let v = zeros(&[n, kv_width])?;
        let att = zeros(&[n, h])?;
        let hid = zeros(&[n, c.ffn])?;
        let up = zeros(&[n, c.ffn])?;
        let prod = zeros(&[n, c.ffn])?;
        let scores = zeros(&[n * n])?;
        let (cos, sin) = rope_tables(n, hd);
        let cos = Tensor::from_f32(d, [n, hd / 2], DType::F32, &cos)?;
        let sin = Tensor::from_f32(d, [n, hd / 2], DType::F32, &sin)?;
        let mask = Tensor::from_f32(d, [n * n], DType::F32, &window_mask(n, c.window))?;
        let mid = zeros(&[n, self.mid])?;
        let out = zeros(&[n, c.out_dim])?;
        // Every buffer encode uses is allocated above, so this is the clip's
        // peak (`out` is returned and outlives the rest).
        self.activation_bytes
            .set(d.allocated_bytes().saturating_sub(before));
        let scale = 1. / (hd as f32).sqrt();
        let execution = d.execution_with_shared_encoder(true)?;

        let stem = &self.stem;
        d.dispatch_hybrid(
            "a_conv_stem",
            &[
                mel.binding(),
                stem.w0.binding(),
                stem.b0.binding(),
                stem.w1.binding(),
                stem.b1.binding(),
            ],
            &[s2.binding()],
            &Params::default()
                .u(frames)?
                .u(c.n_mel)?
                .u(t1)?
                .u(f1)?
                .u(t2)?
                .u(f2)?
                .u(ch)?
                .0,
            [(t2 * f2 * ch).div_ceil(256), 1, 1],
            [256, 1, 1],
            0,
        )?;
        self.linear(d, &s2, &stem.pw1, &p2, t2 * f2, ch, ch, 3)?;
        d.dispatch_hybrid(
            "a_dwconv_s2",
            &[p2.binding(), stem.w2.binding(), stem.b2.binding()],
            &[s3.binding()],
            &Params::default().u(t2)?.u(f2)?.u(t3)?.u(f3)?.u(ch)?.0,
            [(t3 * f3 * ch).div_ceil(256), 1, 1],
            [256, 1, 1],
            0,
        )?;
        self.linear(d, &s3, &stem.pw2, &p3, t3 * f3, ch, ch, 3)?;
        self.linear(d, &p3, &stem.out, &x, n, h, f3 * ch, 0)?;

        for block in &self.blocks {
            match block {
                Block::Conv(b) => {
                    self.rmsnorm(d, &x, &b.ln1, &xn, n, h)?;
                    self.matmul(d, &xn, &b.in_proj, &packed, n, 3 * h, h)?;
                    d.dispatch_hybrid(
                        "h_short_conv",
                        &[packed.binding(), b.taps.binding()],
                        &[
                            conv.binding(),
                            self.conv_state.binding(),
                            self.no_tape.view(0, [0])?.binding(),
                        ],
                        &Params::default().u(n)?.u(h)?.u(CONV_STATE)?.u(0)?.u(0)?.0,
                        [h.div_ceil(256), 1, 1],
                        [256, 1, 1],
                        0,
                    )?;
                    self.matmul(d, &conv, &b.out_proj, &tmp, n, h, h)?;
                    run_add(d, &tmp, &x, n * h)?;
                    self.ffn(d, &x, &xn, &hid, &up, &prod, &tmp, &b.ffn, n)?;
                }
                Block::Attn(b) => {
                    self.rmsnorm(d, &x, &b.ln1, &xn, n, h)?;
                    self.matmul(d, &xn, &b.q, &q, n, h, h)?;
                    self.matmul(d, &xn, &b.k, &k, n, kv_width, h)?;
                    self.matmul(d, &xn, &b.v, &v, n, kv_width, h)?;
                    self.rmsnorm(
                        d,
                        &q.reshape([n * c.heads, hd])?,
                        &b.q_norm,
                        &qn.reshape([n * c.heads, hd])?,
                        n * c.heads,
                        hd,
                    )?;
                    self.rmsnorm(
                        d,
                        &k.reshape([n * c.kv_heads, hd])?,
                        &b.k_norm,
                        &kn.reshape([n * c.kv_heads, hd])?,
                        n * c.kv_heads,
                        hd,
                    )?;
                    self.rope(d, &cos, &sin, &qn, n, c.heads, hd, h)?;
                    self.rope(d, &cos, &sin, &kn, n, c.kv_heads, hd, kv_width)?;
                    let q_per = c.heads / c.kv_heads;
                    for head in 0..c.heads {
                        let kv = head / q_per;
                        run_gemm(
                            d,
                            &self.zero,
                            &Gemm {
                                m: n,
                                n,
                                k: hd,
                                lda: h,
                                ldb_n: kv_width,
                                ldb_k: 1,
                                ldc: n,
                                offsets: (head * hd, kv * hd, 0),
                                alpha: scale,
                                ..Gemm::default()
                            },
                            &qn,
                            &kn,
                            None,
                            &scores,
                        )?;
                        run_add(d, &mask, &scores, n * n)?;
                        d.dispatch_hybrid(
                            "v_softmax",
                            &[],
                            &[scores.binding()],
                            &Params::default().u(n)?.u(n)?.0,
                            [n, 1, 1],
                            [256, 1, 1],
                            0,
                        )?;
                        run_gemm(
                            d,
                            &self.zero,
                            &Gemm {
                                m: n,
                                n: hd,
                                k: n,
                                lda: n,
                                ldb_n: 1,
                                ldb_k: kv_width,
                                ldc: h,
                                offsets: (0, kv * hd, head * hd),
                                ..Gemm::default()
                            },
                            &scores,
                            &v,
                            None,
                            &att,
                        )?;
                    }
                    self.matmul(d, &att, &b.out, &tmp, n, h, h)?;
                    run_add(d, &tmp, &x, n * h)?;
                    self.ffn(d, &x, &xn, &hid, &up, &prod, &tmp, &b.ffn, n)?;
                }
            }
        }
        self.rmsnorm(d, &x, &self.post_norm, &xn, n, h)?;
        d.dispatch_hybrid(
            "v_layernorm",
            &[
                xn.binding(),
                self.mm_norm.0.binding(),
                self.mm_norm.1.binding(),
            ],
            &[tmp.binding()],
            &Params::default().u(n)?.u(h)?.f(PROJECTOR_EPS).0,
            [n, 1, 1],
            [256, 1, 1],
            0,
        )?;
        self.linear(d, &tmp, &self.mm1, &mid, n, self.mid, h, 2)?;
        self.linear(d, &mid, &self.mm3, &out, n, c.out_dim, self.mid, 0)?;
        execution.finish()?;
        Ok(ImageEmbedding {
            hash,
            grid: (1, n),
            rows: std::rc::Rc::new(out),
            deepstack: Vec::new(),
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn ffn(
        &self,
        d: &MetalDevice,
        x: &Tensor,
        xn: &Tensor,
        hid: &Tensor,
        up: &Tensor,
        prod: &Tensor,
        tmp: &Tensor,
        w: &Ffn,
        n: usize,
    ) -> Result<()> {
        let (h, f) = (self.config.hidden, self.config.ffn);
        self.rmsnorm(d, x, &w.ln2, xn, n, h)?;
        self.gemm(d, xn, &w.gate, None, hid, n, f, h, 3)?;
        self.gemm(d, xn, &w.up, None, up, n, f, h, 0)?;
        d.dispatch_hybrid(
            "v_mul",
            &[hid.binding(), up.binding()],
            &[prod.binding()],
            &Params::default().u(n)?.u(f)?.u(f)?.u(0)?.u(f)?.u(0)?.0,
            [(n * f).div_ceil(256), 1, 1],
            [256, 1, 1],
            0,
        )?;
        self.matmul(d, prod, &w.down, tmp, n, h, f)?;
        run_add(d, tmp, x, n * h)
    }

    #[allow(clippy::too_many_arguments)]
    fn linear(
        &self,
        d: &MetalDevice,
        a: &Tensor,
        w: &Proj,
        out: &Tensor,
        m: usize,
        n: usize,
        k: usize,
        epilogue: u32,
    ) -> Result<()> {
        self.gemm(d, a, &w.weight, w.bias.as_ref(), out, m, n, k, epilogue)
    }

    #[allow(clippy::too_many_arguments)]
    fn matmul(
        &self,
        d: &MetalDevice,
        a: &Tensor,
        w: &Tensor,
        out: &Tensor,
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<()> {
        self.gemm(d, a, w, None, out, m, n, k, 0)
    }

    #[allow(clippy::too_many_arguments)]
    fn gemm(
        &self,
        d: &MetalDevice,
        a: &Tensor,
        w: &Tensor,
        bias: Option<&Tensor>,
        out: &Tensor,
        m: usize,
        n: usize,
        k: usize,
        epilogue: u32,
    ) -> Result<()> {
        run_gemm(
            d,
            &self.zero,
            &Gemm {
                m,
                n,
                k,
                lda: k,
                ldb_n: k,
                ldb_k: 1,
                ldc: n,
                epilogue,
                ..Gemm::default()
            },
            a,
            w,
            bias,
            out,
        )
    }

    fn rmsnorm(
        &self,
        d: &MetalDevice,
        x: &Tensor,
        w: &Tensor,
        out: &Tensor,
        rows: usize,
        width: usize,
    ) -> Result<()> {
        d.dispatch_hybrid(
            "v_rmsnorm",
            &[x.binding(), w.binding()],
            &[out.binding()],
            &Params::default().u(rows)?.u(width)?.f(self.config.eps).0,
            [rows, 1, 1],
            [256, 1, 1],
            0,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn rope(
        &self,
        d: &MetalDevice,
        cos: &Tensor,
        sin: &Tensor,
        x: &Tensor,
        tokens: usize,
        heads: usize,
        head_dim: usize,
        stride: usize,
    ) -> Result<()> {
        d.dispatch_hybrid(
            "v_rope1",
            &[cos.binding(), sin.binding()],
            &[x.binding()],
            &Params::default()
                .u(tokens)?
                .u(heads)?
                .u(head_dim)?
                .u(stride)?
                .0,
            [(tokens * heads * (head_dim / 2)).div_ceil(256), 1, 1],
            [256, 1, 1],
            0,
        )
    }
}

/// NEOX RoPE tables for positions `0..n`: pair `i` rotates `(i, i + hd/2)`.
fn rope_tables(n: usize, head_dim: usize) -> (Vec<f32>, Vec<f32>) {
    let half = head_dim / 2;
    let mut cos = Vec::with_capacity(n * half);
    let mut sin = Vec::with_capacity(n * half);
    for p in 0..n {
        for i in 0..half {
            let angle = p as f64 * ROPE_THETA.powf(-2. * i as f64 / head_dim as f64);
            cos.push(angle.cos() as f32);
            sin.push(angle.sin() as f32);
        }
    }
    (cos, sin)
}

/// Additive `[n, n]` mask: row `q` sees keys `q - window + 1 ..= q`.
fn window_mask(n: usize, window: usize) -> Vec<f32> {
    let mut mask = vec![f32::NEG_INFINITY; n * n];
    for q in 0..n {
        for j in q.saturating_sub(window - 1)..=q {
            mask[q * n + j] = 0.;
        }
    }
    mask
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{Rng, SeedableRng};
    use rand_chacha::ChaCha8Rng;
    use std::collections::HashMap;

    /// Tensors by GGUF name, `dims` in ggml order.
    struct MapSource(HashMap<String, (Vec<usize>, Vec<f32>)>);

    impl Source for MapSource {
        fn contains(&mut self, name: &str) -> bool {
            self.0.contains_key(name)
        }
        fn values(&mut self, name: &str) -> Result<Vec<f32>> {
            self.0
                .get(name)
                .map(|t| t.1.clone())
                .ok_or_else(|| Error::Gguf(format!("no tensor {name}")))
        }
        fn matrix(
            &mut self,
            d: &MetalDevice,
            name: &str,
            rows: usize,
            cols: usize,
        ) -> Result<Tensor> {
            let (dims, v) = self.0.get(name).ok_or_else(|| Error::Gguf(name.into()))?;
            assert_eq!(dims, &[cols, rows], "{name}");
            Tensor::from_f32(d, [rows, cols], DType::F16, v)
        }
    }

    fn config() -> Config {
        Config {
            n_mel: 12,
            hidden: 32,
            heads: 4,
            kv_heads: 2,
            ffn: 48,
            layers: 4,
            window: 3,
            eps: 1e-5,
            out_dim: 24,
        }
    }

    /// Random weights, rounded to F16 so the reference sees what the GPU does.
    fn source(c: &Config, ch: usize) -> MapSource {
        let mut rng = ChaCha8Rng::seed_from_u64(5);
        let mut m = HashMap::new();
        let mut put = |name: &str, dims: &[usize], scale: f32, base: f32| {
            let n = dims.iter().product();
            let v = (0..n)
                .map(|_| half::f16::from_f32(base + rng.random_range(-scale..scale)).to_f32())
                .collect();
            m.insert(name.to_string(), (dims.to_vec(), v));
        };
        let (h, hd) = (c.hidden, c.head_dim());
        let f3 = c.n_mel.div_ceil(8);
        put("a.conv1d.0.weight", &[3, 3, 1, ch], 0.5, 0.);
        put("a.conv1d.0.bias", &[1, 1, ch], 0.1, 0.);
        for i in [2, 5] {
            put(&format!("a.conv1d.{i}.weight"), &[3, 3, 1, ch], 0.5, 0.);
            put(&format!("a.conv1d.{i}.bias"), &[1, 1, ch], 0.1, 0.);
        }
        for i in [3, 6] {
            put(&format!("a.conv1d.{i}.weight"), &[1, 1, ch, ch], 0.35, 0.);
            put(&format!("a.conv1d.{i}.bias"), &[1, 1, ch], 0.1, 0.);
        }
        put("a.pre_encode.out.weight", &[ch * f3, h], 0.2, 0.);
        put("a.pre_encode.out.bias", &[h], 0.1, 0.);
        for i in 0..c.layers {
            let n = |s: &str| format!("a.blk.{i}.{s}");
            put(&n("ln1.weight"), &[h], 0.1, 1.);
            put(&n("ln2.weight"), &[h], 0.1, 1.);
            put(&n("ffn_gate.weight"), &[h, c.ffn], 0.18, 0.);
            put(&n("ffn_up.weight"), &[h, c.ffn], 0.18, 0.);
            put(&n("ffn_down.weight"), &[c.ffn, h], 0.15, 0.);
            if i % 2 == 0 {
                put(&n("conv_pw1.weight"), &[h, 3 * h], 0.18, 0.);
                put(&n("conv_dw.weight"), &[3, h], 0.5, 0.);
                put(&n("conv_pw2.weight"), &[h, h], 0.18, 0.);
            } else {
                put(&n("attn_q.weight"), &[h, h], 0.3, 0.);
                put(&n("attn_k.weight"), &[h, c.kv_heads * hd], 0.3, 0.);
                put(&n("attn_v.weight"), &[h, c.kv_heads * hd], 0.18, 0.);
                put(&n("attn_q_norm.weight"), &[hd], 0.1, 1.);
                put(&n("attn_k_norm.weight"), &[hd], 0.1, 1.);
                put(&n("attn_out.weight"), &[h, h], 0.18, 0.);
            }
        }
        put("a.post_ln.weight", &[h], 0.1, 1.);
        put("mm.a.mlp.0.weight", &[h], 0.1, 1.);
        put("mm.a.mlp.0.bias", &[h], 0.1, 0.);
        put("mm.a.mlp.1.weight", &[h, 40], 0.18, 0.);
        put("mm.a.mlp.1.bias", &[40], 0.1, 0.);
        put("mm.a.mlp.3.weight", &[40, c.out_dim], 0.16, 0.);
        put("mm.a.mlp.3.bias", &[c.out_dim], 0.1, 0.);
        MapSource(m)
    }

    fn silu(x: f32) -> f32 {
        x / (1. + (-x).exp())
    }

    fn gelu_erf(x: f32) -> f32 {
        // erf via the same Abramowitz & Stegun form as the kernel is not
        // needed at this tolerance; f64 erf from the series is.
        let x = f64::from(x);
        let z = x / std::f64::consts::SQRT_2;
        let t = 1. / (1. + 0.5 * z.abs());
        let y = 1.
            - t * (-z * z - 1.26551223
                + t * (1.00002368
                    + t * (0.37409196
                        + t * (0.09678418
                            + t * (-0.18628806
                                + t * (0.27886807
                                    + t * (-1.13520398
                                        + t * (1.48851587
                                            + t * (-0.82215223 + t * 0.17087277)))))))))
                .exp();
        (0.5 * x * (1. + y.copysign(z))) as f32
    }

    /// `[m, n] = a[m, k] * w[n, k]^T + b`.
    fn linear(a: &[f32], w: &[f32], b: Option<&[f32]>, m: usize, n: usize, k: usize) -> Vec<f32> {
        let mut out = vec![0f32; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut acc = b.map_or(0., |b| b[j]);
                for l in 0..k {
                    acc += a[i * k + l] * w[j * k + l];
                }
                out[i * n + j] = acc;
            }
        }
        out
    }

    fn rms(x: &[f32], w: &[f32], width: usize, eps: f32) -> Vec<f32> {
        x.chunks(width)
            .flat_map(|r| {
                let ss = r.iter().map(|v| v * v).sum::<f32>() / width as f32;
                let inv = 1. / (ss + eps).sqrt();
                r.iter().zip(w).map(move |(v, w)| v * inv * w)
            })
            .collect()
    }

    /// Stride-2 3x3 conv over `[t, f, cin]`, ggml layout `[cout][kf][kt]`;
    /// depthwise when `cin == cout`, else `cin == 1`.
    fn conv_s2(
        x: &[f32],
        t: usize,
        f: usize,
        cin: usize,
        w: &[f32],
        b: &[f32],
        cout: usize,
    ) -> (Vec<f32>, usize, usize) {
        let (to, fo) = (t.div_ceil(2), f.div_ceil(2));
        let mut out = vec![0f32; to * fo * cout];
        for ot in 0..to {
            for of in 0..fo {
                for c in 0..cout {
                    let ci = if cin == 1 { 0 } else { c };
                    let mut acc = b[c];
                    for kf in 0..3 {
                        for kt in 0..3 {
                            let (st, sf) = (2 * ot + kt, 2 * of + kf);
                            if st == 0 || sf == 0 || st > t || sf > f {
                                continue;
                            }
                            acc += x[((st - 1) * f + sf - 1) * cin + ci] * w[c * 9 + kf * 3 + kt];
                        }
                    }
                    out[(ot * fo + of) * cout + c] = acc;
                }
            }
        }
        (out, to, fo)
    }

    /// llama.cpp's graph, in llama.cpp's own (channel-major) flattening.
    fn reference(
        m: &HashMap<String, (Vec<usize>, Vec<f32>)>,
        c: &Config,
        mel: &[f32],
        frames: usize,
    ) -> Vec<f32> {
        let g = |name: &str| m[name].1.clone();
        let ch = g("a.conv1d.0.bias").len();
        let (h, hd) = (c.hidden, c.head_dim());
        let (x, t, f) = conv_s2(
            mel,
            frames,
            c.n_mel,
            1,
            &g("a.conv1d.0.weight"),
            &g("a.conv1d.0.bias"),
            ch,
        );
        let x: Vec<f32> = x.into_iter().map(silu).collect();
        let (x, t, f) = conv_s2(
            &x,
            t,
            f,
            ch,
            &g("a.conv1d.2.weight"),
            &g("a.conv1d.2.bias"),
            ch,
        );
        let x: Vec<f32> = linear(
            &x,
            &g("a.conv1d.3.weight"),
            Some(&g("a.conv1d.3.bias")),
            t * f,
            ch,
            ch,
        )
        .into_iter()
        .map(silu)
        .collect();
        let (x, t, f) = conv_s2(
            &x,
            t,
            f,
            ch,
            &g("a.conv1d.5.weight"),
            &g("a.conv1d.5.bias"),
            ch,
        );
        let x: Vec<f32> = linear(
            &x,
            &g("a.conv1d.6.weight"),
            Some(&g("a.conv1d.6.bias")),
            t * f,
            ch,
            ch,
        )
        .into_iter()
        .map(silu)
        .collect();
        let n = t;
        // ggml flattens channel-major: column k * f + freq.
        let mut flat = vec![0f32; n * ch * f];
        for p in 0..n {
            for fr in 0..f {
                for k in 0..ch {
                    flat[p * ch * f + k * f + fr] = x[(p * f + fr) * ch + k];
                }
            }
        }
        let mut x = linear(
            &flat,
            &g("a.pre_encode.out.weight"),
            Some(&g("a.pre_encode.out.bias")),
            n,
            h,
            ch * f,
        );
        let (cos, sin) = rope_tables(n, hd);
        let rope = |v: &mut [f32], heads: usize| {
            for p in 0..n {
                for hh in 0..heads {
                    let r = &mut v[p * heads * hd + hh * hd..][..hd];
                    for i in 0..hd / 2 {
                        let (co, si) = (cos[p * hd / 2 + i], sin[p * hd / 2 + i]);
                        let (a, b) = (r[i], r[i + hd / 2]);
                        r[i] = a * co - b * si;
                        r[i + hd / 2] = b * co + a * si;
                    }
                }
            }
        };
        for i in 0..c.layers {
            let n_ = |s: &str| format!("a.blk.{i}.{s}");
            let xn = rms(&x, &g(&n_("ln1.weight")), h, c.eps);
            let y = if m.contains_key(&n_("conv_dw.weight")) {
                let g = |name: &str| m[name].1.clone();
                let bcx = linear(&xn, &g(&n_("conv_pw1.weight")), None, n, 3 * h, h);
                let w = g(&n_("conv_dw.weight"));
                let bx: Vec<f32> = (0..n * h)
                    .map(|i| bcx[(i / h) * 3 * h + i % h] * bcx[(i / h) * 3 * h + 2 * h + i % h])
                    .collect();
                let mut y = vec![0f32; n * h];
                for p in 0..n {
                    for k in 0..h {
                        let mut acc = 0.;
                        for tap in 0..3 {
                            if p + tap >= 2 {
                                acc += w[k * 3 + tap] * bx[(p + tap - 2) * h + k];
                            }
                        }
                        y[p * h + k] = acc * bcx[p * 3 * h + h + k];
                    }
                }
                linear(&y, &g(&n_("conv_pw2.weight")), None, n, h, h)
            } else {
                let g = |name: &str| m[name].1.clone();
                let kvw = c.kv_heads * hd;
                let mut q = rms(
                    &linear(&xn, &g(&n_("attn_q.weight")), None, n, h, h),
                    &g(&n_("attn_q_norm.weight")),
                    hd,
                    c.eps,
                );
                let mut k = rms(
                    &linear(&xn, &g(&n_("attn_k.weight")), None, n, kvw, h),
                    &g(&n_("attn_k_norm.weight")),
                    hd,
                    c.eps,
                );
                let v = linear(&xn, &g(&n_("attn_v.weight")), None, n, kvw, h);
                rope(&mut q, c.heads);
                rope(&mut k, c.kv_heads);
                let mut att = vec![0f32; n * h];
                for hh in 0..c.heads {
                    let kv = hh / (c.heads / c.kv_heads);
                    for p in 0..n {
                        let lo = (p + 1).saturating_sub(c.window);
                        let s: Vec<f32> = (lo..=p)
                            .map(|j| {
                                (0..hd)
                                    .map(|l| q[p * h + hh * hd + l] * k[j * kvw + kv * hd + l])
                                    .sum::<f32>()
                                    / (hd as f32).sqrt()
                            })
                            .collect();
                        let mx = s.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                        let e: Vec<f32> = s.iter().map(|v| (v - mx).exp()).collect();
                        let sum: f32 = e.iter().sum();
                        for l in 0..hd {
                            att[p * h + hh * hd + l] = (lo..=p)
                                .zip(&e)
                                .map(|(j, w)| w / sum * v[j * kvw + kv * hd + l])
                                .sum();
                        }
                    }
                }
                linear(&att, &g(&n_("attn_out.weight")), None, n, h, h)
            };
            let g = |name: &str| m[name].1.clone();
            x.iter_mut().zip(&y).for_each(|(a, b)| *a += b);
            let xn = rms(&x, &g(&n_("ln2.weight")), h, c.eps);
            let gate = linear(&xn, &g(&n_("ffn_gate.weight")), None, n, c.ffn, h);
            let up = linear(&xn, &g(&n_("ffn_up.weight")), None, n, c.ffn, h);
            let prod: Vec<f32> = gate.iter().zip(&up).map(|(a, b)| silu(*a) * b).collect();
            let y = linear(&prod, &g(&n_("ffn_down.weight")), None, n, h, c.ffn);
            x.iter_mut().zip(&y).for_each(|(a, b)| *a += b);
        }
        let g = |name: &str| m[name].1.clone();
        let x = rms(&x, &g("a.post_ln.weight"), h, c.eps);
        let (w, b) = (g("mm.a.mlp.0.weight"), g("mm.a.mlp.0.bias"));
        let x: Vec<f32> = x
            .chunks(h)
            .flat_map(|r| {
                let mean = r.iter().sum::<f32>() / h as f32;
                let var = r.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / h as f32;
                let inv = 1. / (var + PROJECTOR_EPS).sqrt();
                r.iter()
                    .zip(w.iter().zip(&b))
                    .map(move |(v, (w, b))| (v - mean) * inv * w + b)
                    .collect::<Vec<_>>()
            })
            .collect();
        let mid: Vec<f32> = linear(
            &x,
            &g("mm.a.mlp.1.weight"),
            Some(&g("mm.a.mlp.1.bias")),
            n,
            40,
            h,
        )
        .into_iter()
        .map(gelu_erf)
        .collect();
        linear(
            &mid,
            &g("mm.a.mlp.3.weight"),
            Some(&g("mm.a.mlp.3.bias")),
            n,
            c.out_dim,
            40,
        )
    }

    #[test]
    fn encoder_matches_cpu_reference() {
        let d = MetalDevice::new().unwrap();
        let c = config();
        // 37 frames: odd at every stage, so each conv uses its right padding.
        let frames = 37;
        let mut rng = ChaCha8Rng::seed_from_u64(9);
        let mel: Vec<f32> = (0..frames * c.n_mel)
            .map(|_| rng.random_range(-1.5..0.5))
            .collect();
        let mut src = source(&c, 8);
        let encoder = load(&d, c.clone(), &mut src).unwrap();
        let want = reference(&src.0, &c, &mel, frames);
        let got = encoder.encode(&d, &mel, 1).unwrap();
        assert_eq!(got.grid, (1, tokens_for(frames)));
        assert_eq!(tokens_for(frames), 5);
        let got = got.rows.to_f32();
        assert_eq!(got.len(), want.len());
        let scale = want.iter().map(|v| v.abs()).fold(0., f32::max).max(1e-3);
        let worst = got
            .iter()
            .zip(&want)
            .map(|(a, b)| (a - b).abs())
            .fold(0., f32::max);
        assert!(
            worst <= 0.02 * scale,
            "worst error {worst} against output scale {scale}"
        );
    }

    #[test]
    fn window_mask_is_causal_and_bounded() {
        let m = window_mask(5, 3);
        let seen = |q: usize| (0..5).filter(|&j| m[q * 5 + j] == 0.).collect::<Vec<_>>();
        assert_eq!(seen(0), vec![0]);
        assert_eq!(seen(1), vec![0, 1]);
        assert_eq!(seen(4), vec![2, 3, 4]);
        assert_eq!(tokens_for(1), 1);
        assert_eq!(tokens_for(8), 1);
        assert_eq!(tokens_for(9), 2);
    }
}
