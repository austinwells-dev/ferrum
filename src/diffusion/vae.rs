//! The image decoder of `AutoencoderKLQwenImage21`: latents `[h * w, 64]` to
//! pixels, 16 times larger on each side. A single image is a one-frame video
//! for the 3D causal convolutions, which then reduce to plain 2D ones with
//! zero padding, and the temporal upsampling layers are bypassed; both are
//! checked against diffusers' output in `reference_tests`.
//!
//! Activations are F16 and NHWC (`conv`); every layer is a kernel in
//! `diffusion_vae.metal`.
use super::{
    checkpoint::Checkpoint,
    config::VaeConfig,
    conv::{self, Conv, Operand, Residual},
};
use crate::{DType, Error, MetalDevice, Result, Tensor};

/// Residual block: `x + conv2(silu(rms(conv1(silu(rms(x))))))`, with a 1x1
/// convolution on the skip path when the channel count changes.
struct Resnet {
    norm1: Tensor,
    conv1: Conv,
    norm2: Tensor,
    conv2: Conv,
    shortcut: Option<Conv>,
}

/// Single-head self-attention over all pixels, as a residual.
struct Attention {
    norm: Tensor,
    qkv: Conv,
    proj: Conv,
}

struct UpBlock {
    resnets: Vec<Resnet>,
    /// Nearest-neighbour 2x then a 3x3 convolution; the last block has none.
    upsampler: Option<Conv>,
    /// The pixel-shuffled skip path around the block (`Residual::Shuffled`).
    frames: usize,
    repeats: usize,
}

pub struct Vae {
    post_quant: Conv,
    conv_in: Conv,
    mid: [Resnet; 2],
    attention: Attention,
    ups: Vec<UpBlock>,
    norm_out: Tensor,
    conv_out: Conv,
    mean: Tensor,
    std: Tensor,
}

/// The decoded image, NHWC with `channels` values per pixel in `[-1, 1]`.
pub struct Image {
    pub width: usize,
    pub height: usize,
    pub channels: usize,
    pub pixels: Vec<f32>,
}

fn gamma(d: &MetalDevice, c: &Checkpoint, name: &str, len: usize) -> Result<Tensor> {
    c.vector(d, &format!("{name}.gamma"), len, 0.)
}

impl Resnet {
    fn load(d: &MetalDevice, c: &Checkpoint, name: &str, cin: usize, cout: usize) -> Result<Self> {
        Ok(Self {
            norm1: gamma(d, c, &format!("{name}.norm1"), cin)?,
            conv1: Conv::load(d, c, &format!("{name}.conv1"), cin, cout, 3)?,
            norm2: gamma(d, c, &format!("{name}.norm2"), cout)?,
            conv2: Conv::load(d, c, &format!("{name}.conv2"), cout, cout, 3)?,
            shortcut: (cin != cout)
                .then(|| Conv::load(d, c, &format!("{name}.conv_shortcut"), cin, cout, 1))
                .transpose()?,
        })
    }

    fn forward(&self, d: &MetalDevice, x: &Tensor, (h, w): (usize, usize)) -> Result<Tensor> {
        let px = h * w;
        let (cin, cout) = (self.conv1.cin, self.conv1.cout);
        let act = Tensor::zeros(d, [px, cin], DType::F16)?;
        conv::channel_rms(d, x, &self.norm1, &act, px, true)?;
        let mid = Tensor::zeros(d, [px, cout], DType::F16)?;
        conv::conv(d, &self.conv1, &act, &mid, (h, w), 1, Residual::None)?;
        drop(act);
        let act = Tensor::zeros(d, [px, cout], DType::F16)?;
        conv::channel_rms(d, &mid, &self.norm2, &act, px, true)?;
        drop(mid);
        let skip = match &self.shortcut {
            Some(s) => {
                let t = Tensor::zeros(d, [px, cout], DType::F16)?;
                conv::conv(d, s, x, &t, (h, w), 1, Residual::None)?;
                Some(t)
            }
            None => None,
        };
        let out = Tensor::zeros(d, [px, cout], DType::F16)?;
        let res = Residual::Same(skip.as_ref().unwrap_or(x));
        conv::conv(d, &self.conv2, &act, &out, (h, w), 1, res)?;
        Ok(out)
    }
}

impl Attention {
    fn forward(&self, d: &MetalDevice, x: &Tensor, (h, w): (usize, usize)) -> Result<Tensor> {
        let (p, dim) = (h * w, self.proj.cout);
        let row = 3 * dim;
        let normed = Tensor::zeros(d, [p, dim], DType::F16)?;
        conv::channel_rms(d, x, &self.norm, &normed, p, false)?;
        let qkv = Tensor::zeros(d, [p, row], DType::F16)?;
        conv::conv(d, &self.qkv, &normed, &qkv, (h, w), 1, Residual::None)?;
        drop(normed);

        let k = qkv.view(dim, [p * row - dim])?;
        let v = qkv.view(2 * dim, [p * row - 2 * dim])?;
        let vt = Tensor::zeros(d, [dim, p], DType::F16)?;
        conv::transpose(d, &v, &vt, p, dim, row)?;
        let keys = Operand {
            t: &k,
            rows: p,
            stride_row: row,
            stride_k: 1,
        };
        let values = Operand {
            t: &vt,
            rows: dim,
            stride_row: p,
            stride_k: 1,
        };
        // Queries go through in chunks so the P x P score matrix stays a
        // fixed size however large the image is.
        let chunk = ((1 << 26) / p).clamp(64, p);
        let scores = Tensor::zeros(d, [chunk, p], DType::F32)?;
        let weights = Tensor::zeros(d, [chunk, p], DType::F16)?;
        let attended = Tensor::zeros(d, [p, dim], DType::F16)?;
        let scale = 1. / (dim as f32).sqrt();
        for start in (0..p).step_by(chunk) {
            let rows = chunk.min(p - start);
            let q = qkv.view(start * row, [(p - start) * row])?;
            let queries = Operand {
                t: &q,
                rows,
                stride_row: row,
                stride_k: 1,
            };
            conv::gemm(d, &queries, &keys, &scores, dim, p)?;
            conv::softmax_rows(d, &scores, &weights, rows, p, scale)?;
            let out = attended.view(start * dim, [(p - start) * dim])?;
            let probs = Operand {
                t: &weights,
                rows,
                stride_row: p,
                stride_k: 1,
            };
            conv::gemm(d, &probs, &values, &out, p, dim)?;
        }
        let y = Tensor::zeros(d, [p, dim], DType::F16)?;
        conv::conv(d, &self.proj, &attended, &y, (h, w), 1, Residual::Same(x))?;
        Ok(y)
    }
}

impl Vae {
    pub fn load(d: &MetalDevice, c: &Checkpoint, cfg: &VaeConfig) -> Result<Self> {
        let z = cfg.z_dim;
        // Channel counts from the deepest level outwards: [1152, 1152, 1152, 576, 288, 144].
        let widths = cfg.dim_mult.iter().map(|m| m * cfg.decoder_base_dim);
        let deepest = widths
            .clone()
            .next_back()
            .ok_or_else(|| Error::Config("empty dim_mult".into()))?;
        let dims: Vec<usize> = std::iter::once(deepest).chain(widths.rev()).collect();
        let top = dims[0];
        let mut ups = Vec::new();
        for i in 0..cfg.dim_mult.len() {
            let (cin, cout) = (dims[i], dims[i + 1]);
            let name = format!("decoder.up_blocks.{i}");
            let resnets = (0..=cfg.num_res_blocks)
                .map(|j| {
                    let from = if j == 0 { cin } else { cout };
                    Resnet::load(d, c, &format!("{name}.resnets.{j}"), from, cout)
                })
                .collect::<Result<Vec<_>>>()?;
            let last = i + 1 == cfg.dim_mult.len();
            // The temporal upsampling of the first three blocks doubles the
            // frames a shortcut channel is spread over; a single image uses
            // the last frame of each.
            let frames = if i + 1 < cfg.dim_mult.len() - 1 { 2 } else { 1 };
            ups.push(UpBlock {
                resnets,
                upsampler: (!last)
                    .then(|| {
                        Conv::load(d, c, &format!("{name}.upsampler.resample.1"), cout, cout, 3)
                    })
                    .transpose()?,
                frames,
                repeats: cout * frames * 4 / cin,
            });
        }
        let stats = |v: &[f32]| Tensor::from_f32(d, [v.len()], DType::F32, v);
        Ok(Self {
            post_quant: Conv::load(d, c, "post_quant_conv", z, z, 1)?,
            conv_in: Conv::load(d, c, "decoder.conv_in", z, top, 3)?,
            mid: [
                Resnet::load(d, c, "decoder.mid_block.resnets.0", top, top)?,
                Resnet::load(d, c, "decoder.mid_block.resnets.1", top, top)?,
            ],
            attention: Attention {
                norm: gamma(d, c, "decoder.mid_block.attentions.0.norm", top)?,
                qkv: Conv::load(
                    d,
                    c,
                    "decoder.mid_block.attentions.0.to_qkv",
                    top,
                    3 * top,
                    1,
                )?,
                proj: Conv::load(d, c, "decoder.mid_block.attentions.0.proj", top, top, 1)?,
            },
            ups,
            norm_out: gamma(d, c, "decoder.norm_out", *dims.last().unwrap_or(&0))?,
            conv_out: Conv::load(
                d,
                c,
                "decoder.conv_out",
                *dims.last().unwrap_or(&0),
                cfg.out_channels,
                3,
            )?,
            mean: stats(&cfg.latents_mean)?,
            std: stats(&cfg.latents_std)?,
        })
    }

    /// Weight bytes on the device.
    pub fn weight_bytes(&self) -> usize {
        let resnet = |r: &Resnet| {
            r.conv1.byte_size()
                + r.conv2.byte_size()
                + r.shortcut.as_ref().map_or(0, Conv::byte_size)
        };
        self.post_quant.byte_size()
            + self.conv_in.byte_size()
            + self.conv_out.byte_size()
            + self.mid.iter().map(resnet).sum::<usize>()
            + self.attention.qkv.byte_size()
            + self.attention.proj.byte_size()
            + self
                .ups
                .iter()
                .map(|u| {
                    u.resnets.iter().map(resnet).sum::<usize>()
                        + u.upsampler.as_ref().map_or(0, Conv::byte_size)
                })
                .sum::<usize>()
    }

    /// The transformer's F32 latents `[h * w, z]` as the decoder's F16 input
    /// (`x * std + mean`).
    pub fn denormalize(&self, d: &MetalDevice, latents: &Tensor) -> Result<Tensor> {
        let z = Tensor::zeros(
            d,
            [latents.numel() / self.mean.numel(), self.mean.numel()],
            DType::F16,
        )?;
        conv::denormalize(d, latents, &self.mean, &self.std, &z)?;
        Ok(z)
    }

    /// Decode de-normalised latents `[h * w, z]` (F16) into an
    /// `(h * scale) x (w * scale)` image.
    pub fn decode(&self, d: &MetalDevice, z: &Tensor, (h, w): (usize, usize)) -> Result<Image> {
        let top = self.conv_in.cout;
        let zc = self.post_quant.cin;
        let mapped = Tensor::zeros(d, [h * w, zc], DType::F16)?;
        conv::conv(d, &self.post_quant, z, &mapped, (h, w), 1, Residual::None)?;
        let mut x = Tensor::zeros(d, [h * w, top], DType::F16)?;
        conv::conv(d, &self.conv_in, &mapped, &x, (h, w), 1, Residual::None)?;
        drop(mapped);
        x = self.mid[0].forward(d, &x, (h, w))?;
        x = self.attention.forward(d, &x, (h, w))?;
        x = self.mid[1].forward(d, &x, (h, w))?;

        let (mut h, mut w) = (h, w);
        for block in &self.ups {
            let input = x;
            let mut cur = block.resnets[0].forward(d, &input, (h, w))?;
            for r in &block.resnets[1..] {
                cur = r.forward(d, &cur, (h, w))?;
            }
            x = match &block.upsampler {
                Some(up) => {
                    let (nh, nw) = (h * 2, w * 2);
                    let out = Tensor::zeros(d, [nh * nw, up.cout], DType::F16)?;
                    let skip = Residual::Shuffled {
                        src: &input,
                        channels: input.numel() / (h * w),
                        frames: block.frames,
                        repeats: block.repeats,
                    };
                    conv::conv(d, up, &cur, &out, (nh, nw), 2, skip)?;
                    (h, w) = (nh, nw);
                    out
                }
                None => cur,
            };
        }
        let c = self.norm_out.numel();
        let act = Tensor::zeros(d, [h * w, c], DType::F16)?;
        conv::channel_rms(d, &x, &self.norm_out, &act, h * w, true)?;
        drop(x);
        let out = Tensor::zeros(d, [h * w, self.conv_out.cout], DType::F16)?;
        conv::conv(d, &self.conv_out, &act, &out, (h, w), 1, Residual::None)?;
        let mut pixels = out.to_f32();
        for p in &mut pixels {
            *p = p.clamp(-1., 1.);
        }
        Ok(Image {
            width: w,
            height: h,
            channels: self.conv_out.cout,
            pixels,
        })
    }
}
