//! The Qwen-Image-2.1 diffusion transformer: 32 single-stream blocks over the
//! text tokens followed by the image tokens, all modulated by one shared
//! projection of the timestep.
//!
//! Text tokens are modulated from `t = 0` and attend causally, so every
//! layer's text keys and values are the same at each step. `prefill` computes
//! them once; `step` then runs only the image tokens, which attend to the
//! cached text and bidirectionally to each other. That is the arithmetic of
//! diffusers' KV-cached pipeline, whose first step also runs the text.
use super::{
    checkpoint::Checkpoint,
    config::DitConfig,
    ops::{self, HEAD_DIM, KeyValues, QkvIn},
    rope::RopeTable,
};
use crate::{
    DType, Error, MetalDevice, Result, Tensor,
    hybrid::{
        engine::{HybridModel, Params, page},
        weights::{Format, Matrix},
    },
};

const TIME_DIM: usize = 256;

struct Block {
    q: Matrix,
    k: Matrix,
    v: Matrix,
    o: Matrix,
    q_norm: Tensor,
    k_norm: Tensor,
    gate: Matrix,
    up: Matrix,
    down: Matrix,
}

pub struct Dit {
    cfg: DitConfig,
    img_in: Matrix,
    txt_norm: Tensor,
    txt_in: Matrix,
    txt_out: Matrix,
    time_in: Matrix,
    time_out: Matrix,
    modulation: Matrix,
    norm_out: Matrix,
    proj_out: Matrix,
    blocks: Vec<Block>,
}

impl Dit {
    /// Read the transformer from a diffusers checkpoint. `quantize` (Q8_0 or
    /// Q4_0) quantizes the blocks' matrices as they load; the embeddings,
    /// the modulation and the output projection stay BF16.
    pub fn load(
        device: &MetalDevice,
        ckpt: &Checkpoint,
        cfg: &DitConfig,
        quantize: Option<Format>,
    ) -> Result<Self> {
        let (h, f) = (cfg.hidden(), cfg.ffn());
        if cfg.head_dim != HEAD_DIM {
            return Err(Error::Config(format!(
                "the image kernels are written for {HEAD_DIM}-wide heads, not {}",
                cfg.head_dim
            )));
        }
        let m = |name: &str, rows, cols| ckpt.matrix(device, name, rows, cols, None);
        let blocks = (0..cfg.layers)
            .map(|i| {
                let n = |s: &str| format!("transformer_blocks.{i}.{s}.weight");
                let b = |s: &str, rows, cols| ckpt.matrix(device, &n(s), rows, cols, quantize);
                Ok(Block {
                    q: b("attn.to_q", h, h)?,
                    k: b("attn.to_k", h, h)?,
                    v: b("attn.to_v", h, h)?,
                    o: b("attn.to_out.0", h, h)?,
                    q_norm: ckpt.vector(device, &n("attn.norm_q"), HEAD_DIM, 0.)?,
                    k_norm: ckpt.vector(device, &n("attn.norm_k"), HEAD_DIM, 0.)?,
                    gate: b("img_mlp.gate_layer", f, h)?,
                    up: b("img_mlp.proj", f, h)?,
                    down: b("img_mlp.out", h, f)?,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            img_in: m("img_in.weight", h, cfg.in_channels)?,
            // The text norm stores `scale - 1`.
            txt_norm: ckpt.vector(device, "txt_in.text_norm.weight", cfg.context, 1.)?,
            txt_in: m("txt_in.in_layer.weight", h, cfg.context)?,
            txt_out: m("txt_in.out_layer.weight", h, h)?,
            time_in: m(
                "time_text_embed.timestep_embedder.linear_1.weight",
                h,
                TIME_DIM,
            )?,
            time_out: m("time_text_embed.timestep_embedder.linear_2.weight", h, h)?,
            modulation: m("modulation.1.weight", 4 * h, h)?,
            norm_out: m("norm_out.linear.weight", h, h)?,
            proj_out: m("proj_out.weight", cfg.out_channels(), h)?,
            blocks,
            cfg: cfg.clone(),
        })
    }

    /// Weight bytes `load` allocates.
    pub fn estimate_bytes(cfg: &DitConfig, quantize: Option<Format>) -> usize {
        let (h, f) = (cfg.hidden(), cfg.ffn());
        let stored = |rows, cols, q| Matrix::bf16_stored_bytes(rows, cols, q);
        let block = 4 * stored(h, h, quantize)
            + 2 * stored(f, h, quantize)
            + stored(h, f, quantize)
            + 2 * page(HEAD_DIM * 4);
        let fixed = stored(h, cfg.in_channels, None)
            + stored(h, cfg.context, None)
            + stored(h, h, None) * 3
            + stored(h, TIME_DIM, None)
            + stored(4 * h, h, None)
            + stored(cfg.out_channels(), h, None)
            + page(cfg.context * 4);
        fixed + cfg.layers * block
    }

    pub fn config(&self) -> &DitConfig {
        &self.cfg
    }

    /// Device bytes of the weights.
    pub fn weight_bytes(&self) -> usize {
        let b = |b: &Block| {
            [&b.q, &b.k, &b.v, &b.o, &b.gate, &b.up, &b.down]
                .iter()
                .map(|m| m.byte_size())
                .sum::<usize>()
                + b.q_norm.byte_size()
                + b.k_norm.byte_size()
        };
        [
            &self.img_in,
            &self.txt_in,
            &self.txt_out,
            &self.time_in,
            &self.time_out,
            &self.modulation,
            &self.norm_out,
            &self.proj_out,
        ]
        .iter()
        .map(|m| m.byte_size())
        .sum::<usize>()
            + self.txt_norm.byte_size()
            + self.blocks.iter().map(b).sum::<usize>()
    }
}

/// Working tensors for up to `rows` tokens, shared by the text prefill and
/// every step.
pub struct Scratch {
    rows: usize,
    x: Tensor,
    h: Tensor,
    q: Tensor,
    k: Tensor,
    v: Tensor,
    att: Tensor,
    o: Tensor,
    qh: Tensor,
    kh: Tensor,
    vh: Tensor,
    gate: Tensor,
    up: Tensor,
    act: Tensor,
}

impl Scratch {
    pub fn new(d: &MetalDevice, cfg: &DitConfig, rows: usize) -> Result<Self> {
        let (h, f) = (cfg.hidden(), cfg.ffn());
        let wide = |dtype, w| Tensor::zeros(d, [rows, w], dtype);
        Ok(Self {
            rows,
            x: wide(DType::F32, h)?,
            h: wide(DType::F32, h)?,
            q: wide(DType::F32, h)?,
            k: wide(DType::F32, h)?,
            v: wide(DType::F32, h)?,
            att: wide(DType::F32, h)?,
            o: wide(DType::F32, h)?,
            qh: wide(DType::F16, h)?,
            kh: wide(DType::F16, h)?,
            vh: wide(DType::F16, h)?,
            gate: wide(DType::F32, f)?,
            up: wide(DType::F32, f)?,
            act: wide(DType::F32, f)?,
        })
    }

    /// Device bytes for `rows` tokens (what `new` allocates).
    pub fn bytes(cfg: &DitConfig, rows: usize) -> usize {
        let (h, f) = (cfg.hidden(), cfg.ffn());
        rows * (7 * h * 4 + 3 * h * 2 + 3 * f * 4)
    }
}

/// Rotary rows of one token run, as the kernels read them.
pub struct RopeRows {
    cos: Tensor,
    sin: Tensor,
}

/// Rotary tables of the prompt and of the image.
pub struct Rope {
    text: RopeRows,
    image: RopeRows,
}

impl Rope {
    pub fn new(
        d: &MetalDevice,
        cfg: &DitConfig,
        text: usize,
        grid: (usize, usize),
    ) -> Result<Self> {
        let table = RopeTable::new(cfg, text, grid);
        let rows = |first: usize, n: usize| -> Result<RopeRows> {
            let pairs = table.pairs();
            let make = |x: &[f32]| Tensor::from_f32(d, [n, pairs], DType::F32, &x[..n * pairs]);
            Ok(RopeRows {
                cos: make(table.cos(first))?,
                sin: make(table.sin(first))?,
            })
        };
        Ok(Self {
            text: rows(0, text)?,
            image: rows(text, grid.0 * grid.1)?,
        })
    }
}

/// The text's keys and values in every layer, `[rows, hidden]` as F16.
pub struct TextPrefix {
    rows: usize,
    k: Vec<Tensor>,
    v: Vec<Tensor>,
}

impl TextPrefix {
    pub fn rows(&self) -> usize {
        self.rows
    }
    pub fn bytes(cfg: &DitConfig, rows: usize) -> usize {
        cfg.layers * 2 * rows * cfg.hidden() * 2
    }
}

/// The timestep's shared modulation (scale and gate of both sublayers) and
/// the final norm's scale.
struct Conditioning {
    modulation: Tensor,
    norm_scale: Tensor,
}

/// Called with each layer's output rows after the layer ran (for tests).
pub type Probe<'a> = Option<&'a mut dyn FnMut(usize, &Tensor)>;

enum Pass<'a> {
    /// Text rows: causal, writing their K/V into the prefix.
    Fill(&'a TextPrefix),
    /// Image rows: bidirectional, over the cached text and themselves.
    Use(&'a TextPrefix),
}

impl Dit {
    /// The sinusoidal embedding of `sigma * 1000` (cosines, then sines)
    /// through the timestep MLP, then the shared modulation and the final
    /// norm's scale from its SiLU.
    fn condition(&self, d: &MetalDevice, sigma: f32) -> Result<Conditioning> {
        let h = self.cfg.hidden();
        let half = TIME_DIM / 2;
        let t = sigma * 1000.;
        let mut emb = vec![0f32; TIME_DIM];
        for j in 0..half {
            let angle = t * (-(10_000f32).ln() * j as f32 / half as f32).exp();
            emb[j] = angle.cos();
            emb[half + j] = angle.sin();
        }
        let input = Tensor::from_f32(d, [1, TIME_DIM], DType::F32, &emb)?;
        let mid = Tensor::zeros(d, [1, h], DType::F32)?;
        let act = Tensor::zeros(d, [1, h], DType::F32)?;
        let temb = Tensor::zeros(d, [1, h], DType::F32)?;
        let silu = Tensor::zeros(d, [1, h], DType::F32)?;
        let modulation = Tensor::zeros(d, [1, 4 * h], DType::F32)?;
        let norm_scale = Tensor::zeros(d, [1, h], DType::F32)?;
        HybridModel::project(d, &self.time_in, &input, 1, &mid)?;
        ops::silu(d, &mid, &act)?;
        HybridModel::project(d, &self.time_out, &act, 1, &temb)?;
        ops::silu(d, &temb, &silu)?;
        HybridModel::project(d, &self.modulation, &silu, 1, &modulation)?;
        HybridModel::project(d, &self.norm_out, &silu, 1, &norm_scale)?;
        Ok(Conditioning {
            modulation: modulation.reshape([4 * h])?,
            norm_scale: norm_scale.reshape([h])?,
        })
    }

    /// Run the text through every layer, keeping each layer's keys and values.
    pub fn prefill(
        &self,
        d: &MetalDevice,
        s: &Scratch,
        embeds: &Tensor,
        rope: &Rope,
        probe: Probe,
    ) -> Result<TextPrefix> {
        let rows = embeds.shape().dimensions()[0];
        let h = self.cfg.hidden();
        if rows > s.rows || embeds.shape().dimensions() != [rows, self.cfg.context] {
            return Err(Error::Shape(format!(
                "{rows} text rows against scratch for {}",
                s.rows
            )));
        }
        let prefix = TextPrefix {
            rows,
            k: (0..self.cfg.layers)
                .map(|_| Tensor::zeros(d, [rows, h], DType::F16))
                .collect::<Result<_>>()?,
            v: (0..self.cfg.layers)
                .map(|_| Tensor::zeros(d, [rows, h], DType::F16))
                .collect::<Result<_>>()?,
        };
        // The text projection: norm, linear, GELU, linear.
        let at = |t: &Tensor| t.view(0, [rows, h]);
        let (x, hid, mid, act) = (at(&s.x)?, at(&s.h)?, at(&s.q)?, at(&s.k)?);
        d.dispatch_hybrid(
            "h_rmsnorm",
            &[embeds.binding(), self.txt_norm.binding()],
            &[hid.binding()],
            &Params::default()
                .u(self.cfg.context)?
                .f(self.cfg.eps)
                .u(rows)?
                .0,
            [rows, 1, 1],
            [256, 1, 1],
            0,
        )?;
        HybridModel::project(d, &self.txt_in, &hid, rows, &mid)?;
        ops::gelu_tanh(d, &mid, &act)?;
        HybridModel::project(d, &self.txt_out, &act, rows, &x)?;
        let cond = self.condition(d, 0.)?;
        self.layers(d, s, rows, &cond, &rope.text, Pass::Fill(&prefix), probe)?;
        Ok(prefix)
    }

    /// One denoising step: the velocity the model predicts for `latents`
    /// (`[tokens, in_channels]`) at `sigma`, into `pred`.
    #[allow(clippy::too_many_arguments)]
    pub fn step(
        &self,
        d: &MetalDevice,
        s: &Scratch,
        prefix: &TextPrefix,
        rope: &Rope,
        latents: &Tensor,
        sigma: f32,
        pred: &Tensor,
        probe: Probe,
    ) -> Result<()> {
        let rows = latents.shape().dimensions()[0];
        let h = self.cfg.hidden();
        if rows > s.rows || latents.numel() != rows * self.cfg.in_channels {
            return Err(Error::Shape(format!(
                "{rows} image rows against scratch for {}",
                s.rows
            )));
        }
        let x = s.x.view(0, [rows, h])?;
        HybridModel::project(d, &self.img_in, latents, rows, &x)?;
        let cond = self.condition(d, sigma)?;
        self.layers(d, s, rows, &cond, &rope.image, Pass::Use(prefix), probe)?;
        let hid = s.h.view(0, [rows, h])?;
        ops::layernorm_scale(d, &x, &cond.norm_scale, &hid, rows, self.cfg.eps, 1.)?;
        HybridModel::project(d, &self.proj_out, &hid, rows, pred)
    }

    #[allow(clippy::too_many_arguments)]
    fn layers(
        &self,
        d: &MetalDevice,
        s: &Scratch,
        rows: usize,
        cond: &Conditioning,
        rope: &RopeRows,
        pass: Pass,
        mut probe: Probe,
    ) -> Result<()> {
        let (h, f, heads) = (self.cfg.hidden(), self.cfg.ffn(), self.cfg.heads);
        let at = |t: &Tensor, w: usize| t.view(0, [rows, w]);
        let (x, hid, q, k, v) = (
            at(&s.x, h)?,
            at(&s.h, h)?,
            at(&s.q, h)?,
            at(&s.k, h)?,
            at(&s.v, h)?,
        );
        let (att, o, qh) = (at(&s.att, h)?, at(&s.o, h)?, at(&s.qh, h)?);
        let (gate, up, act) = (at(&s.gate, f)?, at(&s.up, f)?, at(&s.act, f)?);
        let seg = |i: usize| cond.modulation.view(i * h, [h]);
        let (scale1, gate1, scale2, gate2) = (seg(0)?, seg(1)?, seg(2)?, seg(3)?);
        let eps = self.cfg.eps;
        for (l, b) in self.blocks.iter().enumerate() {
            ops::layernorm_scale(d, &x, &scale1, &hid, rows, eps, 1.)?;
            HybridModel::project(d, &b.q, &hid, rows, &q)?;
            HybridModel::project(d, &b.k, &hid, rows, &k)?;
            HybridModel::project(d, &b.v, &hid, rows, &v)?;
            let (kd, vd, kc, vc, prefix, keys, causal) = match &pass {
                Pass::Fill(p) => (p.k[l].clone(), p.v[l].clone(), None, None, 0, rows, true),
                Pass::Use(p) => (
                    at(&s.kh, h)?,
                    at(&s.vh, h)?,
                    Some(&p.k[l]),
                    Some(&p.v[l]),
                    p.rows,
                    p.rows + rows,
                    false,
                ),
            };
            let input = QkvIn {
                q: &q,
                k: &k,
                v: &v,
                q_norm: &b.q_norm,
                k_norm: &b.k_norm,
                cos: &rope.cos,
                sin: &rope.sin,
            };
            ops::qkv_prep(d, &input, [&qh, &kd, &vd], rows, heads, eps)?;
            let values = KeyValues {
                prefix_k: kc.unwrap_or(&kd),
                prefix_v: vc.unwrap_or(&vd),
                prefix,
                k: &kd,
                v: &vd,
            };
            ops::attention(d, &qh, &values, &att, rows, heads, keys, causal)?;
            HybridModel::project(d, &b.o, &att, rows, &o)?;
            ops::gate_add(d, &o, &gate1, &x)?;

            ops::layernorm_scale(d, &x, &scale2, &hid, rows, eps, 1.)?;
            HybridModel::project(d, &b.gate, &hid, rows, &gate)?;
            HybridModel::project(d, &b.up, &hid, rows, &up)?;
            ops::swiglu(d, &gate, &up, &act)?;
            HybridModel::project(d, &b.down, &act, rows, &o)?;
            ops::gate_add(d, &o, &gate2, &x)?;
            if let Some(probe) = probe.as_mut() {
                probe(l, &x);
            }
        }
        Ok(())
    }
}
