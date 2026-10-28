//! LFM3-VL still-image tower (`clip.projector_type = lfm3vl`).
//!
//! Patches stay in raster order through the encoder. Each block is a gated
//! depthwise conv or full bidirectional attention, then a SwiGLU. The
//! projector layer-normalises each patch, interleaves every 2x2 into one row
//! (top-left, top-right, bottom-left, bottom-right) and runs a GELU MLP.
//! Video frames and the audio projectors are not this path.
#![forbid(unsafe_code)]

use super::{Gemm, ImageEmbedding, Prepared, SCORE_BYTES, Source, VisionConfig, VisionModel};
use crate::{DType, Error, MetalDevice, Result, Tensor, hybrid::engine::Params};

const ROPE_THETA: f32 = 1_000_000.;
/// Hardcoded in llama.cpp's LFM3-VL graph, not the encoder epsilon.
const PROJECTOR_EPS: f32 = 1e-6;

struct Proj {
    weight: Tensor,
    bias: Tensor,
}

struct Conv {
    ln1: Tensor,
    in_proj: Proj,
    spatial: Tensor,
    out_proj: Proj,
    ln2: Tensor,
    gate: Proj,
    up: Proj,
    down: Proj,
}

struct Attn {
    ln1: Tensor,
    q: Proj,
    k: Proj,
    v: Proj,
    q_norm: Tensor,
    k_norm: Tensor,
    out_proj: Proj,
    ln2: Tensor,
    gate: Proj,
    up: Proj,
    down: Proj,
}

/// Both arms are boxed: each block is wider than the enum discriminant, and
/// a layer is only ever one of them.
enum Block {
    Conv(Box<Conv>),
    Attn(Box<Attn>),
}

pub(super) struct Tower {
    kv_heads: usize,
    embed_norm: Tensor,
    patch: Tensor,
    blocks: Vec<Block>,
    proj_norm: (Tensor, Tensor),
    fc1: Proj,
    fc2: Proj,
    pub weight_bytes: usize,
}

pub(super) struct Limits {
    pub min_tokens: usize,
    pub max_tokens: usize,
}

/// Most language tokens whose attention scores fit in `cap` bytes.
pub(super) fn score_language_cap(merge: usize, cap: usize) -> usize {
    let patches = ((cap / 4) as f64).sqrt().floor() as usize;
    (patches / (merge * merge)).max(1)
}

pub(super) fn load(
    d: &MetalDevice,
    config: &VisionConfig,
    source: &mut dyn Source,
    requested: usize,
    kv_heads: usize,
    min_pixels: usize,
    max_pixels: usize,
) -> Result<(Tower, Limits)> {
    let bad = |what: String| Error::Config(format!("unsupported lfm3vl tower: {what}"));
    let h = config.hidden;
    let hd = config.head_dim();
    if kv_heads == 0 || !config.heads.is_multiple_of(kv_heads) {
        return Err(bad(format!(
            "{kv_heads} KV heads for {} query heads",
            config.heads
        )));
    }
    let area = config.patch * config.patch * config.merge * config.merge;
    if area == 0 || min_pixels == 0 || max_pixels < min_pixels {
        return Err(bad(format!("pixel limits {min_pixels}/{max_pixels}")));
    }
    let meta_max = (max_pixels / area).min(4096);
    let max_tokens = requested
        .clamp(4, meta_max)
        .min(score_language_cap(config.merge, SCORE_BYTES));
    let min_tokens = (min_pixels / area).clamp(1, max_tokens);
    let mut bytes = 0usize;
    let proj =
        |source: &mut dyn Source, bytes: &mut usize, name: &str, rows: usize, cols: usize| {
            *bytes += rows * cols * 2;
            let weight = source.matrix(d, &format!("{name}.weight"), rows, cols)?;
            let bias_name = format!("{name}.bias");
            let bias = if source.contains(&bias_name) {
                let v = source.values(&bias_name)?;
                if v.len() != rows {
                    return Err(Error::Weight {
                        name: bias_name,
                        message: format!("{} values, expected {rows}", v.len()),
                    });
                }
                *bytes += rows * 4;
                Tensor::from_f32(d, [rows], DType::F32, &v)?
            } else {
                Tensor::zeros(d, [rows], DType::F32)?
            };
            Ok(Proj { weight, bias })
        };
    let rms = |source: &mut dyn Source, bytes: &mut usize, name: &str| {
        if source.contains(&format!("{name}.bias")) {
            return Err(bad(format!(
                "{name} has a bias; the encoder norms are RMSNorm"
            )));
        }
        let v = source.values(&format!("{name}.weight"))?;
        if v.len() != h {
            return Err(Error::Weight {
                name: format!("{name}.weight"),
                message: format!("{} values, expected {h}", v.len()),
            });
        }
        *bytes += h * 4;
        Tensor::from_f32(d, [h], DType::F32, &v)
    };
    if source.contains("v.patch_embd.bias") {
        return Err(bad("a patch-embedding bias".into()));
    }
    let (w0, w1) = (
        source.values("v.patch_embd.weight")?,
        if source.contains("v.patch_embd.weight.1") {
            source.values("v.patch_embd.weight.1")?
        } else {
            vec![0.; h * config.patch_len()]
        },
    );
    if w0.len() != h * config.patch_len() || w1.len() != w0.len() {
        return Err(bad("patch embedding shape".into()));
    }
    // A still image fills both temporal taps, so the kernels add.
    let summed: Vec<f32> = w0.iter().zip(&w1).map(|(a, b)| a + b).collect();
    let patch = Tensor::from_f32(d, [h, config.patch_len()], DType::F16, &summed)?;
    bytes += summed.len() * 2;
    let embed_norm = rms(source, &mut bytes, "v.embedding_norm")?;
    let mut blocks = Vec::with_capacity(config.layers);
    for i in 0..config.layers {
        let n = |s: &str| format!("v.blk.{i}.{s}");
        let conv = source.contains(&format!("v.blk.{i}.conv2d_in_proj.weight"));
        let attn = source.contains(&format!("v.blk.{i}.attn_q.weight"));
        if conv == attn {
            return Err(bad(format!(
                "layer {i} is not exactly one of conv or attention"
            )));
        }
        let ln1 = rms(source, &mut bytes, &n("ln1"))?;
        let ln2 = rms(source, &mut bytes, &n("ln2"))?;
        let gate = proj(source, &mut bytes, &n("ffn_gate"), config.ffn, h)?;
        let up = proj(source, &mut bytes, &n("ffn_up"), config.ffn, h)?;
        let down = proj(source, &mut bytes, &n("ffn_down"), h, config.ffn)?;
        if conv {
            let spatial = source.values(&n("conv2d_spatial.weight"))?;
            if spatial.len() != h * 9 {
                return Err(bad(format!("layer {i} spatial kernel is not 3x3")));
            }
            bytes += spatial.len() * 4;
            blocks.push(Block::Conv(Box::new(Conv {
                ln1,
                in_proj: proj(source, &mut bytes, &n("conv2d_in_proj"), 3 * h, h)?,
                spatial: Tensor::from_f32(d, [h, 9], DType::F32, &spatial)?,
                out_proj: proj(source, &mut bytes, &n("conv2d_out_proj"), h, h)?,
                ln2,
                gate,
                up,
                down,
            })));
        } else {
            let mut head_w = |name: &str| -> Result<Tensor> {
                let v = source.values(&n(name))?;
                if v.len() != hd {
                    return Err(Error::Weight {
                        name: n(name),
                        message: format!("{} values, expected {hd}", v.len()),
                    });
                }
                bytes += hd * 4;
                Tensor::from_f32(d, [hd], DType::F32, &v)
            };
            let q_norm = head_w("attn_q_norm.weight")?;
            let k_norm = head_w("attn_k_norm.weight")?;
            blocks.push(Block::Attn(Box::new(Attn {
                ln1,
                q: proj(source, &mut bytes, &n("attn_q"), h, h)?,
                k: proj(source, &mut bytes, &n("attn_k"), kv_heads * hd, h)?,
                v: proj(source, &mut bytes, &n("attn_v"), kv_heads * hd, h)?,
                q_norm,
                k_norm,
                out_proj: proj(source, &mut bytes, &n("attn_out"), h, h)?,
                ln2,
                gate,
                up,
                down,
            })));
        }
    }
    let proj_norm = {
        let w = source.values("mm.input_norm.weight")?;
        let b = source.values("mm.input_norm.bias")?;
        if w.len() != h || b.len() != h {
            return Err(bad("projector input norm shape".into()));
        }
        bytes += h * 8;
        (
            Tensor::from_f32(d, [h], DType::F32, &w)?,
            Tensor::from_f32(d, [h], DType::F32, &b)?,
        )
    };
    let merged = h * config.merge * config.merge;
    let fc1 = proj(source, &mut bytes, "mm.0", merged, merged)?;
    let fc2 = proj(source, &mut bytes, "mm.2", config.out_dim, merged)?;
    Ok((
        Tower {
            kv_heads,
            embed_norm,
            patch,
            blocks,
            proj_norm,
            fc1,
            fc2,
            weight_bytes: bytes,
        },
        Limits {
            min_tokens,
            max_tokens,
        },
    ))
}

impl Tower {
    pub(super) fn encode(
        &self,
        model: &VisionModel,
        d: &MetalDevice,
        image: &Prepared,
    ) -> Result<ImageEmbedding> {
        let c = &model.config;
        let (gh, gw) = (image.height / c.patch, image.width / c.patch);
        if gh == 0
            || gw == 0
            || !gh.is_multiple_of(c.merge)
            || !gw.is_multiple_of(c.merge)
            || image.width != gw * c.patch
            || image.height != gh * c.patch
        {
            return Err(Error::Parameter(format!(
                "a {}x{} image is not a whole grid of {}x{} patches",
                image.width, image.height, c.patch, c.patch
            )));
        }
        let n = gh * gw;
        if n.saturating_mul(n).saturating_mul(4) > model.score_bytes {
            return Err(Error::Parameter(format!(
                "a {gh}x{gw} patch image needs one attention matrix larger than the encoder allows; lower --image-tokens"
            )));
        }
        let h = c.hidden;
        let hd = c.head_dim();
        let (cos, sin) = rope_tables(gh, gw, hd);
        let patches = model.patchify_raster(image);
        let tensor = |dims: &[usize], data: &[f32]| Tensor::from_f32(d, dims, DType::F32, data);
        let patches = tensor(&[n, c.patch_len()], &patches)?;
        let cos = tensor(&[n, hd / 2], &cos)?;
        let sin = tensor(&[n, hd / 2], &sin)?;
        let x = Tensor::zeros(d, [n, h], DType::F32)?;
        let xn = Tensor::zeros(d, [n, h], DType::F32)?;
        let tmp = Tensor::zeros(d, [n, h], DType::F32)?;
        let proj = Tensor::zeros(d, [n, 3 * h], DType::F32)?;
        let bx = Tensor::zeros(d, [n, h], DType::F32)?;
        let conv = Tensor::zeros(d, [n, h], DType::F32)?;
        let kv_width = self.kv_heads * hd;
        let q = Tensor::zeros(d, [n, h], DType::F32)?;
        let k = Tensor::zeros(d, [n, kv_width], DType::F32)?;
        let kn = Tensor::zeros(d, [n, kv_width], DType::F32)?;
        let v = Tensor::zeros(d, [n, kv_width], DType::F32)?;
        let att = Tensor::zeros(d, [n, h], DType::F32)?;
        let hid = Tensor::zeros(d, [n, c.ffn], DType::F32)?;
        let up = Tensor::zeros(d, [n, c.ffn], DType::F32)?;
        let scores = Tensor::zeros(d, [n * n], DType::F32)?;
        let prod = Tensor::zeros(d, [n, c.ffn], DType::F32)?;
        let scale = 1. / (hd as f32).sqrt();
        let execution = d.execution_with_shared_encoder(true)?;

        self.linear(
            model,
            d,
            &patches,
            &self.patch,
            None,
            &x,
            n,
            h,
            c.patch_len(),
            0,
        )?;
        self.rmsnorm(d, &x, &self.embed_norm, &xn, n, h, c.eps)?;
        self.copy(d, &xn, &x, n, h)?;
        for block in &self.blocks {
            match block {
                Block::Conv(block) => {
                    let Conv {
                        ln1,
                        in_proj,
                        spatial,
                        out_proj,
                        ln2,
                        gate,
                        up: up_w,
                        down,
                    } = block.as_ref();
                    self.rmsnorm(d, &x, ln1, &xn, n, h, c.eps)?;
                    self.linear(
                        model,
                        d,
                        &xn,
                        &in_proj.weight,
                        Some(&in_proj.bias),
                        &proj,
                        n,
                        3 * h,
                        h,
                        0,
                    )?;
                    // B and X are the first and third thirds; their product is what the 3x3 sees.
                    self.mul(d, &proj, &proj, &bx, n, h, 3 * h, 0, 3 * h, 2 * h)?;
                    self.dwconv(d, &bx, spatial, &conv, gw, gh, h)?;
                    self.mul(d, &proj, &conv, &tmp, n, h, 3 * h, h, h, 0)?;
                    self.linear(
                        model,
                        d,
                        &tmp,
                        &out_proj.weight,
                        Some(&out_proj.bias),
                        &xn,
                        n,
                        h,
                        h,
                        0,
                    )?;
                    self.add(model, d, &xn, &x, n * h)?;
                    self.ffn(
                        model, d, &x, &xn, &hid, &up, &prod, &tmp, ln2, gate, up_w, down, n,
                    )?;
                }
                Block::Attn(block) => {
                    let Attn {
                        ln1,
                        q: qw,
                        k: kw,
                        v: vw,
                        q_norm,
                        k_norm,
                        out_proj,
                        ln2,
                        gate,
                        up: up_w,
                        down,
                    } = block.as_ref();
                    self.rmsnorm(d, &x, ln1, &xn, n, h, c.eps)?;
                    self.linear(model, d, &xn, &qw.weight, Some(&qw.bias), &q, n, h, h, 0)?;
                    self.linear(
                        model,
                        d,
                        &xn,
                        &kw.weight,
                        Some(&kw.bias),
                        &k,
                        n,
                        kv_width,
                        h,
                        0,
                    )?;
                    self.linear(
                        model,
                        d,
                        &xn,
                        &vw.weight,
                        Some(&vw.bias),
                        &v,
                        n,
                        kv_width,
                        h,
                        0,
                    )?;
                    self.rmsnorm(
                        d,
                        &q.reshape([n * c.heads, hd])?,
                        q_norm,
                        &tmp.reshape([n * c.heads, hd])?,
                        n * c.heads,
                        hd,
                        c.eps,
                    )?;
                    self.rmsnorm(
                        d,
                        &k.reshape([n * self.kv_heads, hd])?,
                        k_norm,
                        &kn.reshape([n * self.kv_heads, hd])?,
                        n * self.kv_heads,
                        hd,
                        c.eps,
                    )?;
                    self.rope(d, &cos, &sin, &tmp, n, c.heads, hd, h)?;
                    self.rope(d, &cos, &sin, &kn, n, self.kv_heads, hd, kv_width)?;
                    let q_per = c.heads / self.kv_heads;
                    for head in 0..c.heads {
                        let kv = head / q_per;
                        model.gemm(
                            d,
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
                            &tmp,
                            &kn,
                            None,
                            &scores,
                        )?;
                        d.dispatch_hybrid(
                            "v_softmax",
                            &[],
                            &[scores.binding()],
                            &Params::default().u(n)?.u(n)?.0,
                            [n, 1, 1],
                            [256, 1, 1],
                            0,
                        )?;
                        model.gemm(
                            d,
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
                    self.linear(
                        model,
                        d,
                        &att,
                        &out_proj.weight,
                        Some(&out_proj.bias),
                        &xn,
                        n,
                        h,
                        h,
                        0,
                    )?;
                    self.add(model, d, &xn, &x, n * h)?;
                    self.ffn(
                        model, d, &x, &xn, &hid, &up, &prod, &tmp, ln2, gate, up_w, down, n,
                    )?;
                }
            }
        }
        self.norm(d, &x, &self.proj_norm, &xn, n, h, PROJECTOR_EPS)?;
        let rows = n / (c.merge * c.merge);
        let merged = h * 4;
        let grouped = Tensor::zeros(d, [rows, merged], DType::F32)?;
        d.dispatch_hybrid(
            "v_interleave",
            &[xn.binding()],
            &[grouped.binding()],
            &Params::default().u(gw)?.u(gh)?.u(h)?.0,
            [(rows * merged).div_ceil(256), 1, 1],
            [256, 1, 1],
            0,
        )?;
        let mid = Tensor::zeros(d, [rows, merged], DType::F32)?;
        let out = Tensor::zeros(d, [rows, c.out_dim], DType::F32)?;
        self.linear(
            model,
            d,
            &grouped,
            &self.fc1.weight,
            Some(&self.fc1.bias),
            &mid,
            rows,
            merged,
            merged,
            2,
        )?;
        self.linear(
            model,
            d,
            &mid,
            &self.fc2.weight,
            Some(&self.fc2.bias),
            &out,
            rows,
            c.out_dim,
            merged,
            0,
        )?;
        execution.finish()?;
        Ok(ImageEmbedding {
            hash: image.hash,
            grid: (gh / c.merge, gw / c.merge),
            rows: std::rc::Rc::new(out),
            deepstack: Vec::new(),
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn ffn(
        &self,
        model: &VisionModel,
        d: &MetalDevice,
        x: &Tensor,
        xn: &Tensor,
        hid: &Tensor,
        up: &Tensor,
        prod: &Tensor,
        tmp: &Tensor,
        ln2: &Tensor,
        gate: &Proj,
        up_w: &Proj,
        down: &Proj,
        n: usize,
    ) -> Result<()> {
        let c = &model.config;
        self.rmsnorm(d, x, ln2, xn, n, c.hidden, c.eps)?;
        self.linear(
            model,
            d,
            xn,
            &gate.weight,
            Some(&gate.bias),
            hid,
            n,
            c.ffn,
            c.hidden,
            3,
        )?;
        self.linear(
            model,
            d,
            xn,
            &up_w.weight,
            Some(&up_w.bias),
            up,
            n,
            c.ffn,
            c.hidden,
            0,
        )?;
        // The product cannot reuse `hid`: a kernel may not read and write one buffer.
        self.mul(d, hid, up, prod, n, c.ffn, c.ffn, 0, c.ffn, 0)?;
        self.linear(
            model,
            d,
            prod,
            &down.weight,
            Some(&down.bias),
            tmp,
            n,
            c.hidden,
            c.ffn,
            0,
        )?;
        self.add(model, d, tmp, x, n * c.hidden)
    }

    #[allow(clippy::too_many_arguments)]
    fn linear(
        &self,
        model: &VisionModel,
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
        model.gemm(
            d,
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

    #[allow(clippy::too_many_arguments)]
    fn rmsnorm(
        &self,
        d: &MetalDevice,
        x: &Tensor,
        w: &Tensor,
        out: &Tensor,
        rows: usize,
        width: usize,
        eps: f32,
    ) -> Result<()> {
        d.dispatch_hybrid(
            "v_rmsnorm",
            &[x.binding(), w.binding()],
            &[out.binding()],
            &Params::default().u(rows)?.u(width)?.f(eps).0,
            [rows, 1, 1],
            [256, 1, 1],
            0,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn norm(
        &self,
        d: &MetalDevice,
        x: &Tensor,
        w: &(Tensor, Tensor),
        out: &Tensor,
        rows: usize,
        width: usize,
        eps: f32,
    ) -> Result<()> {
        d.dispatch_hybrid(
            "v_layernorm",
            &[x.binding(), w.0.binding(), w.1.binding()],
            &[out.binding()],
            &Params::default().u(rows)?.u(width)?.f(eps).0,
            [rows, 1, 1],
            [256, 1, 1],
            0,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn mul(
        &self,
        d: &MetalDevice,
        a: &Tensor,
        b: &Tensor,
        out: &Tensor,
        rows: usize,
        cols: usize,
        a_stride: usize,
        a_off: usize,
        b_stride: usize,
        b_off: usize,
    ) -> Result<()> {
        d.dispatch_hybrid(
            "v_mul",
            &[a.binding(), b.binding()],
            &[out.binding()],
            &Params::default()
                .u(rows)?
                .u(cols)?
                .u(a_stride)?
                .u(a_off)?
                .u(b_stride)?
                .u(b_off)?
                .0,
            [(rows * cols).div_ceil(256), 1, 1],
            [256, 1, 1],
            0,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn dwconv(
        &self,
        d: &MetalDevice,
        input: &Tensor,
        kern: &Tensor,
        out: &Tensor,
        gw: usize,
        gh: usize,
        hidden: usize,
    ) -> Result<()> {
        d.dispatch_hybrid(
            "v_dwconv3",
            &[input.binding(), kern.binding()],
            &[out.binding()],
            &Params::default().u(gw)?.u(gh)?.u(hidden)?.0,
            [(gh * gw * hidden).div_ceil(256), 1, 1],
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
        q: &Tensor,
        tokens: usize,
        heads: usize,
        head_dim: usize,
        stride: usize,
    ) -> Result<()> {
        d.dispatch_hybrid(
            "v_rope1",
            &[cos.binding(), sin.binding()],
            &[q.binding()],
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

    fn copy(
        &self,
        d: &MetalDevice,
        src: &Tensor,
        dst: &Tensor,
        rows: usize,
        width: usize,
    ) -> Result<()> {
        d.dispatch_hybrid(
            "h_copy_rows",
            &[src.binding()],
            &[dst.binding()],
            &Params::default()
                .u(rows)?
                .u(width)?
                .u(width)?
                .u(width)?
                .u(0)?
                .0,
            [(rows * width).div_ceil(256), 1, 1],
            [256, 1, 1],
            0,
        )
    }

    fn add(
        &self,
        model: &VisionModel,
        d: &MetalDevice,
        y: &Tensor,
        acc: &Tensor,
        count: usize,
    ) -> Result<()> {
        model.add(d, y, acc, count)
    }
}

/// Vision RoPE (`GGML_ROPE_TYPE_VISION`): the first `head_dim/4` pairs use the
/// row, the next `head_dim/4` use the column, and the base resets per section.
/// Tokens are raster, x fastest. Theta is 1e6.
fn rope_tables(gh: usize, gw: usize, head_dim: usize) -> (Vec<f32>, Vec<f32>) {
    let half = head_dim / 2;
    let axis = head_dim / 4;
    let freq = |j: usize| ROPE_THETA.powf(-2. * j as f32 / half as f32);
    let mut cos = Vec::with_capacity(gh * gw * half);
    let mut sin = Vec::with_capacity(gh * gw * half);
    for y in 0..gh {
        for x in 0..gw {
            for i in 0..half {
                let (pos, f) = if i < axis {
                    (y, freq(i))
                } else {
                    (x, freq(i - axis))
                };
                let angle = pos as f32 * f;
                cos.push(angle.cos());
                sin.push(angle.sin());
            }
        }
    }
    (cos, sin)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vision::{VisionConfig, VisionModel};
    use rand::{Rng, SeedableRng};
    use rand_chacha::ChaCha8Rng;

    fn rand_vec(rng: &mut ChaCha8Rng, n: usize, scale: f32) -> Vec<f32> {
        (0..n).map(|_| rng.random_range(-scale..scale)).collect()
    }

    fn mat(d: &MetalDevice, rng: &mut ChaCha8Rng, rows: usize, cols: usize) -> Tensor {
        let scale = 1. / (cols as f32).sqrt();
        Tensor::from_f32(
            d,
            [rows, cols],
            DType::F16,
            &rand_vec(rng, rows * cols, scale),
        )
        .unwrap()
    }

    fn vec_f(d: &MetalDevice, rng: &mut ChaCha8Rng, n: usize, base: f32) -> Tensor {
        let v: Vec<f32> = rand_vec(rng, n, 0.1)
            .into_iter()
            .map(|x| x + base)
            .collect();
        Tensor::from_f32(d, [n], DType::F32, &v).unwrap()
    }

    fn proj(d: &MetalDevice, rng: &mut ChaCha8Rng, rows: usize, cols: usize) -> Proj {
        Proj {
            weight: mat(d, rng, rows, cols),
            bias: vec_f(d, rng, rows, 0.),
        }
    }

    fn tiny(d: &MetalDevice) -> (VisionModel, VisionConfig) {
        let config = VisionConfig {
            patch: 4,
            hidden: 32,
            ffn: 48,
            layers: 3,
            heads: 4,
            eps: 1e-5,
            merge: 2,
            out_dim: 16,
            table: 1,
            mean: [0.5; 3],
            std: [0.5; 3],
        };
        let mut rng = ChaCha8Rng::seed_from_u64(11);
        let h = config.hidden;
        let hd = h / config.heads;
        let kv = 2;
        let block = |rng: &mut ChaCha8Rng, conv: bool| {
            let ln1 = vec_f(d, rng, h, 1.);
            let ln2 = vec_f(d, rng, h, 1.);
            let gate = proj(d, rng, config.ffn, h);
            let up = proj(d, rng, config.ffn, h);
            let down = proj(d, rng, h, config.ffn);
            if conv {
                Block::Conv(Box::new(Conv {
                    ln1,
                    in_proj: proj(d, rng, 3 * h, h),
                    spatial: Tensor::from_f32(d, [h, 9], DType::F32, &rand_vec(rng, h * 9, 0.1))
                        .unwrap(),
                    out_proj: proj(d, rng, h, h),
                    ln2,
                    gate,
                    up,
                    down,
                }))
            } else {
                Block::Attn(Box::new(Attn {
                    ln1,
                    q: proj(d, rng, h, h),
                    k: proj(d, rng, kv * hd, h),
                    v: proj(d, rng, kv * hd, h),
                    q_norm: vec_f(d, rng, hd, 1.),
                    k_norm: vec_f(d, rng, hd, 1.),
                    out_proj: proj(d, rng, h, h),
                    ln2,
                    gate,
                    up,
                    down,
                }))
            }
        };
        let tower = Tower {
            kv_heads: kv,
            embed_norm: vec_f(d, &mut rng, h, 1.),
            patch: mat(d, &mut rng, h, config.patch_len()),
            blocks: vec![
                block(&mut rng, true),
                block(&mut rng, false),
                block(&mut rng, true),
            ],
            proj_norm: (vec_f(d, &mut rng, h, 1.), vec_f(d, &mut rng, h, 0.)),
            fc1: proj(d, &mut rng, h * 4, h * 4),
            fc2: proj(d, &mut rng, config.out_dim, h * 4),
            weight_bytes: 0,
        };
        let z = || Tensor::zeros(d, [1], DType::F32).unwrap();
        let m = || Tensor::zeros(d, [1, 1], DType::F16).unwrap();
        let model = VisionModel {
            config: config.clone(),
            max_tokens: 16,
            min_tokens: 1,
            layers: Vec::new(),
            patch: (m(), z()),
            position: Vec::new(),
            merger: super::super::Merger {
                norm: (z(), z()),
                fc1: (m(), z()),
                fc2: (m(), z()),
                post_shuffle: false,
            },
            deepstack: Vec::new(),
            zero: z(),
            weight_bytes: 0,
            score_bytes: SCORE_BYTES,
            lfm3: Some(tower),
            marker: "<|image|>",
            filter: image::imageops::FilterType::CatmullRom,
        };
        (model, config)
    }

    fn silu(x: f32) -> f32 {
        x / (1. + (-x).exp())
    }

    /// `v_gelu_erf`: Abramowitz erf of `x / sqrt(2)`, same constants as the kernel.
    fn gelu_erf(x: f32) -> f32 {
        let z = x * 0.707_106_77;
        let t = 1. / (1. + 0.327_591_1 * z.abs());
        let y = 1.
            - (((((1.061_405_4 * t - 1.453_152) * t) + 1.421_413_7) * t - 0.284_496_74) * t
                + 0.254_829_6)
                * t
                * (-z * z).exp();
        0.5 * x * (1. + y.copysign(z))
    }

    fn linear(a: &[f32], w: &[f32], b: &[f32], m: usize, n: usize, k: usize) -> Vec<f32> {
        let mut out = vec![0.; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut s = b[j];
                for t in 0..k {
                    s += a[i * k + t] * w[j * k + t];
                }
                out[i * n + j] = s;
            }
        }
        out
    }

    fn rms(x: &[f32], w: &[f32], width: usize, eps: f32) -> Vec<f32> {
        let mut out = Vec::with_capacity(x.len());
        for row in x.chunks(width) {
            let ss = row.iter().map(|v| v * v).sum::<f32>() / width as f32;
            let inv = 1. / (ss + eps).sqrt();
            out.extend(row.iter().zip(w).map(|(v, g)| v * inv * g));
        }
        out
    }

    fn layernorm(x: &[f32], w: &[f32], b: &[f32], width: usize, eps: f32) -> Vec<f32> {
        let mut out = Vec::with_capacity(x.len());
        for row in x.chunks(width) {
            let mean = row.iter().sum::<f32>() / width as f32;
            let var = row.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / width as f32;
            let inv = 1. / (var + eps).sqrt();
            out.extend(
                row.iter()
                    .zip(w)
                    .zip(b)
                    .map(|((v, g), bias)| (v - mean) * inv * g + bias),
            );
        }
        out
    }

    fn reference(model: &VisionModel, image: &Prepared) -> Vec<f32> {
        let tower = model.lfm3.as_ref().unwrap();
        let c = &model.config;
        let (gh, gw) = (image.height / c.patch, image.width / c.patch);
        let (n, h, hd) = (gh * gw, c.hidden, c.head_dim());
        let w = |p: &Proj| (p.weight.to_f32(), p.bias.to_f32());
        let mut x = linear(
            &model.patchify_raster(image),
            &tower.patch.to_f32(),
            &vec![0.; h],
            n,
            h,
            c.patch_len(),
        );
        x = rms(&x, &tower.embed_norm.to_f32(), h, c.eps);
        let (cos, sin) = rope_tables(gh, gw, hd);
        for block in &tower.blocks {
            let (ln1, ln2, gate, up_w, down, branch) = match block {
                Block::Conv(block) => {
                    let Conv {
                        ln1,
                        in_proj,
                        spatial,
                        out_proj,
                        ln2,
                        gate,
                        up,
                        down,
                    } = block.as_ref();
                    let xn = rms(&x, &ln1.to_f32(), h, c.eps);
                    let (iw, ib) = w(in_proj);
                    let proj = linear(&xn, &iw, &ib, n, 3 * h, h);
                    let mut bx = vec![0.; n * h];
                    for t in 0..n {
                        for c0 in 0..h {
                            bx[t * h + c0] = proj[t * 3 * h + c0] * proj[t * 3 * h + 2 * h + c0];
                        }
                    }
                    let kern = spatial.to_f32();
                    let mut conv = vec![0.; n * h];
                    for y in 0..gh {
                        for x0 in 0..gw {
                            for c0 in 0..h {
                                let mut acc = 0.;
                                for ky in 0..3 {
                                    for kx in 0..3 {
                                        let sy = y as isize + ky as isize - 1;
                                        let sx = x0 as isize + kx as isize - 1;
                                        if sy < 0
                                            || sx < 0
                                            || sy >= gh as isize
                                            || sx >= gw as isize
                                        {
                                            continue;
                                        }
                                        acc += bx[(sy as usize * gw + sx as usize) * h + c0]
                                            * kern[c0 * 9 + ky * 3 + kx];
                                    }
                                }
                                conv[(y * gw + x0) * h + c0] = acc;
                            }
                        }
                    }
                    let mut mixed = vec![0.; n * h];
                    for t in 0..n {
                        for c0 in 0..h {
                            mixed[t * h + c0] = proj[t * 3 * h + h + c0] * conv[t * h + c0];
                        }
                    }
                    let (ow, ob) = w(out_proj);
                    (ln1, ln2, gate, up, down, linear(&mixed, &ow, &ob, n, h, h))
                }
                Block::Attn(block) => {
                    let Attn {
                        ln1,
                        q,
                        k,
                        v,
                        q_norm,
                        k_norm,
                        out_proj,
                        ln2,
                        gate,
                        up,
                        down,
                    } = block.as_ref();
                    let xn = rms(&x, &ln1.to_f32(), h, c.eps);
                    let (qw, qb) = w(q);
                    let (kw, kb) = w(k);
                    let (vw, vb) = w(v);
                    let kv = tower.kv_heads;
                    let qv = rms(&linear(&xn, &qw, &qb, n, h, h), &q_norm.to_f32(), hd, c.eps);
                    let mut kvv = rms(
                        &linear(&xn, &kw, &kb, n, kv * hd, h),
                        &k_norm.to_f32(),
                        hd,
                        c.eps,
                    );
                    let vv = linear(&xn, &vw, &vb, n, kv * hd, h);
                    let rotate = |row: &mut [f32], heads: usize, tok: usize| {
                        for head in 0..heads {
                            let base = head * hd;
                            for i in 0..hd / 2 {
                                let (co, si) = (cos[tok * hd / 2 + i], sin[tok * hd / 2 + i]);
                                let (x0, x1) = (row[base + i], row[base + i + hd / 2]);
                                row[base + i] = x0 * co - x1 * si;
                                row[base + i + hd / 2] = x1 * co + x0 * si;
                            }
                        }
                    };
                    let mut qv = qv;
                    for t in 0..n {
                        rotate(&mut qv[t * h..(t + 1) * h], c.heads, t);
                        rotate(&mut kvv[t * kv * hd..(t + 1) * kv * hd], kv, t);
                    }
                    let mut att = vec![0.; n * h];
                    let q_per = c.heads / kv;
                    for head in 0..c.heads {
                        let khead = head / q_per;
                        for qi in 0..n {
                            let mut scores = vec![0.; n];
                            for ki in 0..n {
                                let mut s = 0.;
                                for i in 0..hd {
                                    s += qv[qi * h + head * hd + i]
                                        * kvv[ki * kv * hd + khead * hd + i];
                                }
                                scores[ki] = s / (hd as f32).sqrt();
                            }
                            let max = scores.iter().copied().fold(f32::MIN, f32::max);
                            let mut sum = 0.;
                            for s in &mut scores {
                                *s = (*s - max).exp();
                                sum += *s;
                            }
                            for i in 0..hd {
                                att[qi * h + head * hd + i] = (0..n)
                                    .map(|ki| scores[ki] / sum * vv[ki * kv * hd + khead * hd + i])
                                    .sum();
                            }
                        }
                    }
                    let (ow, ob) = w(out_proj);
                    (ln1, ln2, gate, up, down, linear(&att, &ow, &ob, n, h, h))
                }
            };
            let _ = ln1;
            for (a, b) in x.iter_mut().zip(&branch) {
                *a += b;
            }
            let xn = rms(&x, &ln2.to_f32(), h, c.eps);
            let (gw, gb) = w(gate);
            let (uw, ub) = w(up_w);
            let gated: Vec<f32> = linear(&xn, &gw, &gb, n, c.ffn, h)
                .into_iter()
                .map(silu)
                .collect();
            let upv = linear(&xn, &uw, &ub, n, c.ffn, h);
            let prod: Vec<f32> = gated.iter().zip(&upv).map(|(a, b)| a * b).collect();
            let (dw, db) = w(down);
            for (a, b) in x.iter_mut().zip(linear(&prod, &dw, &db, n, h, c.ffn)) {
                *a += b;
            }
        }
        let (pw, pb) = (&tower.proj_norm.0.to_f32(), &tower.proj_norm.1.to_f32());
        let xn = layernorm(&x, pw, pb, h, PROJECTOR_EPS);
        let rows = n / 4;
        let mut grouped = vec![0.; rows * h * 4];
        for yb in 0..gh / 2 {
            for x2 in 0..gw / 2 {
                let tok = yb * (gw / 2) + x2;
                for (which, (dx, dy)) in [(0, 0), (1, 0), (0, 1), (1, 1)].iter().enumerate() {
                    let src = ((2 * yb + dy) * gw + (2 * x2 + dx)) * h;
                    grouped[tok * h * 4 + which * h..tok * h * 4 + (which + 1) * h]
                        .copy_from_slice(&xn[src..src + h]);
                }
            }
        }
        let (w1, b1) = w(&tower.fc1);
        let mid: Vec<f32> = linear(&grouped, &w1, &b1, rows, h * 4, h * 4)
            .into_iter()
            .map(gelu_erf)
            .collect();
        let (w2, b2) = w(&tower.fc2);
        linear(&mid, &w2, &b2, rows, c.out_dim, h * 4)
    }

    #[test]
    fn encoder_matches_cpu_reference() {
        let d = MetalDevice::new().unwrap();
        let (mut model, config) = tiny(&d);
        let image = Prepared {
            width: 16,
            height: 16,
            pixels: rand_vec(&mut ChaCha8Rng::seed_from_u64(3), 3 * 16 * 16, 1.),
            hash: 3,
        };
        let want = reference(&model, &image);
        let got = model.encode_prepared(&d, &image).unwrap();
        assert_eq!(got.grid, (2, 2));
        assert!(got.deepstack.is_empty());
        let got = got.rows.to_f32();
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
        model.score_bytes = 4;
        assert!(model.encode_prepared(&d, &image).is_err());
        let _ = config;
    }

    #[test]
    fn score_cap_stays_under_the_scratch_budget() {
        let cap = score_language_cap(2, SCORE_BYTES);
        let n = cap * 4;
        assert!(n * n * 4 <= SCORE_BYTES);
        assert!((n + 4) * (n + 4) * 4 > SCORE_BYTES);
    }
}
