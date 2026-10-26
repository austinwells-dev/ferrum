//! The image encoder of `AutoencoderKLQwenImage21`, for condition images:
//! RGBA pixels to normalised latents `[h * w, 64]`, 16 times smaller on each
//! side. Like the decoder it treats an image as one frame, so the causal
//! convolutions are plain 2D ones and the temporal convolutions of the
//! downsamplers never run (diffusers skips them on a video's first frame).
//!
//! Each of the five levels is residual blocks, a stride-2 convolution (not on
//! the last), and an averaged space-to-depth shortcut around the whole level.
//! Then the middle blocks, the output convolution and `quant_conv`, whose
//! first 64 channels are the posterior's mode.
use super::{
    checkpoint::Checkpoint,
    config::VaeConfig,
    conv::{self, Conv, Residual, Sample},
    vae::{Attention, Resnet, gamma},
};
use crate::{DType, Error, MetalDevice, Result, Tensor, hybrid::engine::page};

/// The convolution kernels read input channels 16 at a time.
const INPUT_WIDTH: usize = 16;

struct DownBlock {
    resnets: Vec<Resnet>,
    /// Stride-2 3x3 convolution; the last level has none.
    downsampler: Option<Conv>,
    /// The shortcut's input and output channels and whether it averages a
    /// zero frame in (the levels that halve frames).
    cin: usize,
    cout: usize,
    frames: usize,
}

pub struct VaeEncoder {
    conv_in: Conv,
    downs: Vec<DownBlock>,
    mid: [Resnet; 2],
    attention: Attention,
    norm_out: Tensor,
    conv_out: Conv,
    quant: Conv,
    z: usize,
    mean: Vec<f32>,
    std: Vec<f32>,
    scale: usize,
}

/// Channels per level: `[base, base * mult...]`.
fn encoder_dims(cfg: &VaeConfig) -> Vec<usize> {
    std::iter::once(cfg.base_dim)
        .chain(cfg.dim_mult.iter().map(|m| m * cfg.base_dim))
        .collect()
}

impl VaeEncoder {
    pub fn load(d: &MetalDevice, c: &Checkpoint, cfg: &VaeConfig) -> Result<Self> {
        if cfg.in_channels > INPUT_WIDTH {
            return Err(Error::Config(format!(
                "{} image channels into the encoder",
                cfg.in_channels
            )));
        }
        let dims = encoder_dims(cfg);
        let levels = cfg.dim_mult.len();
        let mut downs = Vec::with_capacity(levels);
        for i in 0..levels {
            let (cin, cout) = (dims[i], dims[i + 1]);
            let name = format!("encoder.down_blocks.{i}");
            let resnets = (0..cfg.num_res_blocks)
                .map(|j| {
                    let from = if j == 0 { cin } else { cout };
                    Resnet::load(d, c, &format!("{name}.resnets.{j}"), from, cout)
                })
                .collect::<Result<Vec<_>>>()?;
            let last = i + 1 == levels;
            downs.push(DownBlock {
                resnets,
                downsampler: (!last)
                    .then(|| {
                        Conv::load(
                            d,
                            c,
                            &format!("{name}.downsampler.resample.1"),
                            cout,
                            cout,
                            3,
                        )
                    })
                    .transpose()?,
                cin,
                cout,
                frames: if !last && cfg.temperal_downsample[i] {
                    2
                } else {
                    1
                },
            });
        }
        let top = dims[levels];
        Ok(Self {
            conv_in: Conv::load_padded(
                d,
                c,
                "encoder.conv_in",
                cfg.in_channels,
                INPUT_WIDTH,
                dims[0],
                3,
            )?,
            downs,
            mid: [
                Resnet::load(d, c, "encoder.mid_block.resnets.0", top, top)?,
                Resnet::load(d, c, "encoder.mid_block.resnets.1", top, top)?,
            ],
            attention: Attention::load(d, c, "encoder.mid_block.attentions.0", top)?,
            norm_out: gamma(d, c, "encoder.norm_out", top)?,
            conv_out: Conv::load(d, c, "encoder.conv_out", top, 2 * cfg.z_dim, 3)?,
            quant: Conv::load(d, c, "quant_conv", 2 * cfg.z_dim, 2 * cfg.z_dim, 1)?,
            z: cfg.z_dim,
            mean: cfg.latents_mean.clone(),
            std: cfg.latents_std.clone(),
            scale: cfg.scale_factor_spatial,
        })
    }

    /// Device bytes to encode a `width x height` image: the weights and the
    /// activations of the first level, where they are largest.
    pub fn estimate_bytes(cfg: &VaeConfig, width: usize, height: usize) -> usize {
        let dims = encoder_dims(cfg);
        let levels = cfg.dim_mult.len();
        let top = dims[levels];
        let mut weights = Conv::bytes(INPUT_WIDTH, dims[0], 3)
            + 2 * Resnet::bytes(top, top)
            + Conv::bytes(top, 3 * top, 1)
            + Conv::bytes(top, top, 1)
            + page(top * 4)
            + Conv::bytes(top, 2 * cfg.z_dim, 3)
            + Conv::bytes(2 * cfg.z_dim, 2 * cfg.z_dim, 1);
        for i in 0..levels {
            let (cin, cout) = (dims[i], dims[i + 1]);
            weights += Resnet::bytes(cin, cout)
                + (cfg.num_res_blocks - 1) * Resnet::bytes(cout, cout)
                + if i + 1 < levels {
                    Conv::bytes(cout, cout, 3)
                } else {
                    0
                };
        }
        // Live at once in a first-level residual block: the padded input,
        // the level's input (held for its shortcut) and the block's input,
        // normalised copy, middle, shortcut and output.
        let pixels = width * height;
        weights + pixels * 2 * (INPUT_WIDTH + 6 * dims[1].max(dims[0]))
    }

    /// Encode `width x height` RGBA pixels (row-major, 8 bits a channel;
    /// sides multiples of the VAE's scale) into the transformer's normalised
    /// latents: F32 `[h * w, z]` over the `(height / scale, width / scale)`
    /// grid, row-major.
    pub fn encode(
        &self,
        d: &MetalDevice,
        rgba: &[u8],
        width: usize,
        height: usize,
    ) -> Result<Tensor> {
        let step = self.scale;
        if width == 0
            || height == 0
            || !width.is_multiple_of(step)
            || !height.is_multiple_of(step)
            || rgba.len() != width * height * 4
        {
            return Err(Error::Shape(format!(
                "a {width} x {height} image of {} bytes for the encoder (sides in multiples of {step})",
                rgba.len()
            )));
        }
        let channels = self.conv_in.cin;
        let mut input = vec![0f32; width * height * channels];
        for (px, chunk) in rgba.chunks_exact(4).enumerate() {
            for (c, &v) in chunk.iter().enumerate() {
                input[px * channels + c] = f32::from(v) / 255. * 2. - 1.;
            }
        }
        let (mut h, mut w) = (height, width);
        let input = Tensor::from_f32(d, [h * w, channels], DType::F16, &input)?;
        let mut x = Tensor::zeros(d, [h * w, self.conv_in.cout], DType::F16)?;
        conv::conv(
            d,
            &self.conv_in,
            &input,
            &x,
            (h, w),
            Sample::Same,
            Residual::None,
        )?;
        drop(input);

        for block in &self.downs {
            let (nh, nw, side) = match block.downsampler {
                Some(_) => (h / 2, w / 2, 2),
                None => (h, w, 1),
            };
            let shortcut = Tensor::zeros(d, [nh * nw, block.cout], DType::F16)?;
            conv::avg_down(
                d,
                &x,
                &shortcut,
                (nh, nw),
                block.cin,
                block.cout,
                block.frames,
                side,
            )?;
            for r in &block.resnets {
                x = r.forward(d, &x, (h, w))?;
            }
            x = match &block.downsampler {
                Some(down) => {
                    let out = Tensor::zeros(d, [nh * nw, block.cout], DType::F16)?;
                    let res = Residual::Same(&shortcut);
                    conv::conv(d, down, &x, &out, (nh, nw), Sample::Down, res)?;
                    out
                }
                None => {
                    conv::add(d, &shortcut, &x)?;
                    x
                }
            };
            (h, w) = (nh, nw);
        }
        x = self.mid[0].forward(d, &x, (h, w))?;
        x = self.attention.forward(d, &x, (h, w))?;
        x = self.mid[1].forward(d, &x, (h, w))?;

        let top = self.norm_out.numel();
        let act = Tensor::zeros(d, [h * w, top], DType::F16)?;
        conv::channel_rms(d, &x, &self.norm_out, &act, h * w, true)?;
        drop(x);
        let moments = Tensor::zeros(d, [h * w, self.conv_out.cout], DType::F16)?;
        conv::conv(
            d,
            &self.conv_out,
            &act,
            &moments,
            (h, w),
            Sample::Same,
            Residual::None,
        )?;
        drop(act);
        let quantized = Tensor::zeros(d, [h * w, self.quant.cout], DType::F16)?;
        conv::conv(
            d,
            &self.quant,
            &moments,
            &quantized,
            (h, w),
            Sample::Same,
            Residual::None,
        )?;
        // The mode is the mean half of each pixel's moments, normalised as
        // the transformer was trained on.
        let moments = quantized.to_f32();
        let latents: Vec<f32> = moments
            .chunks_exact(self.quant.cout)
            .flat_map(|px| {
                px[..self.z]
                    .iter()
                    .zip(self.mean.iter().zip(&self.std))
                    .map(|(&v, (m, s))| (v - m) / s)
            })
            .collect();
        if latents.iter().any(|v| !v.is_finite()) {
            return Err(Error::Validation(
                "the image encoder produced non-finite latents".into(),
            ));
        }
        Tensor::from_f32(d, [h * w, self.z], DType::F32, &latents)
    }
}
