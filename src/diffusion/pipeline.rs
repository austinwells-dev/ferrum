//! Prompt to image: encode the prompt, denoise latents, decode them. The three
//! models are loaded one at a time and released before the next, which is
//! what fits a 7B transformer and an 8B encoder on a 32 GB Mac.
use super::{
    checkpoint::Checkpoint,
    config::ModelDir,
    dit::{Dit, Rope, Scratch, TextPrefix},
    schedule::Schedule,
    text::TextEncoder,
    vae::{Image, Vae},
};
use crate::{
    DType, Error, MetalDevice, Result, Tensor, hybrid::weights::Format, tokenizer::Tokenizer,
};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use std::{
    path::Path,
    time::{Duration, Instant},
};

/// The system prompt of the text-to-image template.
const SYSTEM_PROMPT: &str = "Comprehend and analyze the provided prompt.";

/// Longest prompt, in tokens after the system turn.
pub const MAX_PROMPT_TOKENS: usize = 1024;

/// Image sides are multiples of this (16 pixels per latent token, in pairs).
pub const SIZE_STEP: usize = 32;
pub const MIN_SIDE: usize = 256;
pub const MAX_SIDE: usize = 2048;

/// How the weights are stored on the device.
#[derive(Debug, Clone, Copy, Default)]
pub struct Settings {
    /// Quantize the encoder's layer matrices as they load (BF16 when `None`).
    pub text: Option<Format>,
    /// The same for the transformer's block matrices.
    pub transformer: Option<Format>,
}

/// `bf16`, `q8_0` or `q4_0` (what the BF16 checkpoints can be quantized to
/// as they load) as a `Settings` field.
pub fn parse_quantization(s: &str) -> Result<Option<Format>> {
    match s.to_ascii_lowercase().as_str() {
        "bf16" | "none" => Ok(None),
        "q8_0" | "q8" => Ok(Some(Format::Q8_0)),
        "q4_0" | "q4" => Ok(Some(Format::Q4_0)),
        other => Err(Error::Parameter(format!(
            "unknown quantization {other:?} (expected bf16, q8_0 or q4_0)"
        ))),
    }
}

#[derive(Debug, Clone)]
pub struct Request {
    pub prompt: String,
    pub width: usize,
    pub height: usize,
    pub seed: u64,
    /// Denoising steps; `None` takes the checkpoint's own schedule (8 for
    /// Turbo, 40 for the base model).
    pub steps: Option<usize>,
}

impl Request {
    pub fn new(prompt: impl Into<String>) -> Self {
        Self {
            prompt: prompt.into(),
            width: 1024,
            height: 1024,
            seed: 0,
            steps: None,
        }
    }
}

/// Where a generation is, for progress reporting.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Stage {
    LoadEncoder,
    Encode,
    LoadTransformer,
    /// Step `step` of `steps` (1-based) just finished.
    Denoise {
        step: usize,
        steps: usize,
    },
    LoadDecoder,
    Decode,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct Timings {
    pub load: Duration,
    pub encode: Duration,
    pub denoise: Duration,
    pub decode: Duration,
}

pub struct Generated {
    pub width: usize,
    pub height: usize,
    /// RGBA, 8 bits a channel, row-major.
    pub rgba: Vec<u8>,
    pub timings: Timings,
    pub seed: u64,
}

impl Generated {
    pub fn png(&self) -> Result<Vec<u8>> {
        let mut out = Vec::new();
        let image =
            image::RgbaImage::from_raw(self.width as u32, self.height as u32, self.rgba.clone())
                .ok_or_else(|| Error::Shape("pixel buffer does not match the image size".into()))?;
        image
            .write_to(&mut std::io::Cursor::new(&mut out), image::ImageFormat::Png)
            .map_err(|e| Error::Parameter(format!("PNG encoding: {e}")))?;
        Ok(out)
    }
}

pub struct Pipeline {
    model: ModelDir,
    tokenizer: Tokenizer,
    settings: Settings,
    /// Tokens of the system turn, dropped from the encoder's output.
    drop_idx: usize,
}

impl Pipeline {
    pub fn open(dir: impl AsRef<Path>, settings: Settings) -> Result<Self> {
        let model = ModelDir::open(dir)?;
        let tokenizer = Tokenizer::from_file(model.path("processor/tokenizer.json"))?;
        let drop_idx = tokenizer.encode(&system_turn())?.len();
        Ok(Self {
            model,
            tokenizer,
            settings,
            drop_idx,
        })
    }

    pub fn model(&self) -> &ModelDir {
        &self.model
    }

    /// The template's token ids for `prompt`, and how many lead the system turn.
    pub fn tokenize(&self, prompt: &str) -> Result<(Vec<u32>, usize)> {
        let ids = self.tokenizer.encode(&template(prompt))?;
        if ids.len() - self.drop_idx > MAX_PROMPT_TOKENS {
            return Err(Error::Parameter(format!(
                "the prompt is {} tokens; the limit is {MAX_PROMPT_TOKENS}",
                ids.len() - self.drop_idx
            )));
        }
        Ok((ids, self.drop_idx))
    }

    /// Latent tokens per side and the sigma schedule for `request`.
    fn geometry(&self, request: &Request) -> Result<((usize, usize), Schedule)> {
        for (name, side) in [("width", request.width), ("height", request.height)] {
            if !(MIN_SIDE..=MAX_SIDE).contains(&side) || !side.is_multiple_of(SIZE_STEP) {
                return Err(Error::Parameter(format!(
                    "{name} {side} must be a multiple of {SIZE_STEP} between {MIN_SIDE} and {MAX_SIDE}"
                )));
            }
        }
        let scale = self.model.vae.scale_factor_spatial;
        let grid = (request.height / scale, request.width / scale);
        let schedule = match (&self.model.sample_sigmas, request.steps) {
            (Some(sigmas), None) => Schedule::explicit(sigmas),
            (Some(sigmas), Some(n)) if n == sigmas.len() => Schedule::explicit(sigmas),
            (Some(sigmas), Some(n)) => {
                return Err(Error::Parameter(format!(
                    "this checkpoint is distilled for {} steps, not {n}",
                    sigmas.len()
                )));
            }
            (None, steps) => {
                Schedule::dynamic(steps.unwrap_or(40), grid.0 * grid.1, &self.model.scheduler)
            }
        };
        Ok((grid, schedule))
    }

    /// Device bytes each stage needs at its peak, for `request`.
    pub fn stage_bytes(&self, request: &Request) -> Result<[(&'static str, usize); 3]> {
        let (grid, _) = self.geometry(request)?;
        let tokens = grid.0 * grid.1;
        let cfg = &self.model.dit;
        let (ids, skip) = self.tokenize(&request.prompt)?;
        let text_rows = ids.len() - skip;
        let text = TextEncoder::estimate_bytes(&self.model.text, ids.len(), self.settings.text);
        let dit = Dit::estimate_bytes(cfg, self.settings.transformer)
            + Scratch::bytes(cfg, tokens.max(text_rows))
            + TextPrefix::bytes(cfg, text_rows);
        let vae = Vae::estimate_bytes(&self.model.vae, grid);
        Ok([("encoder", text), ("transformer", dit), ("decoder", vae)])
    }

    pub fn generate(
        &self,
        d: &MetalDevice,
        request: &Request,
        progress: &mut dyn FnMut(Stage),
    ) -> Result<Generated> {
        let (grid, schedule) = self.geometry(request)?;
        let budget = d.recommended_max_working_set() as usize;
        for (name, bytes) in self.stage_bytes(request)? {
            if bytes > budget {
                return Err(Error::Allocation(bytes)).map_err(|_| {
                    Error::Parameter(format!(
                        "the {name} stage needs about {:.1} GB, over this device's {:.1} GB working set",
                        bytes as f64 / 1e9,
                        budget as f64 / 1e9
                    ))
                });
            }
        }
        let mut timings = Timings::default();
        let tokens = grid.0 * grid.1;

        let (ids, skip) = self.tokenize(&request.prompt)?;
        let embeds = {
            progress(Stage::LoadEncoder);
            let started = Instant::now();
            let ckpt = Checkpoint::open(self.model.path("text_encoder"))?;
            let encoder =
                TextEncoder::load(d, &ckpt, &self.model.text, ids.len(), self.settings.text)?;
            timings.load += started.elapsed();
            progress(Stage::Encode);
            let started = Instant::now();
            let embeds = encoder.encode(d, &ids, skip)?;
            timings.encode = started.elapsed();
            embeds
        };
        let text_rows = embeds.shape().dimensions()[0];

        let latents = {
            progress(Stage::LoadTransformer);
            let started = Instant::now();
            let ckpt = Checkpoint::open(self.model.path("transformer"))?;
            let dit = Dit::load(d, &ckpt, &self.model.dit, self.settings.transformer)?;
            let scratch = Scratch::new(d, &self.model.dit, text_rows.max(tokens))?;
            let rope = Rope::new(d, &self.model.dit, text_rows, grid)?;
            timings.load += started.elapsed();

            let started = Instant::now();
            let prefix = dit.prefill(d, &scratch, &embeds, &rope, None)?;
            drop(embeds);
            let channels = self.model.dit.in_channels;
            let latents = Tensor::from_f32(
                d,
                [tokens, channels],
                DType::F32,
                &gaussian(request.seed, tokens * channels),
            )?;
            let pred = Tensor::zeros(d, [tokens, self.model.dit.out_channels()], DType::F32)?;
            for i in 0..schedule.steps() {
                dit.step(
                    d,
                    &scratch,
                    &prefix,
                    &rope,
                    &latents,
                    schedule.sigma(i),
                    &pred,
                    None,
                )?;
                schedule.advance(d, i, &pred, &latents)?;
                progress(Stage::Denoise {
                    step: i + 1,
                    steps: schedule.steps(),
                });
            }
            timings.denoise = started.elapsed();
            latents
        };

        progress(Stage::LoadDecoder);
        let started = Instant::now();
        let ckpt = Checkpoint::open(self.model.path("vae"))?;
        let vae = Vae::load(d, &ckpt, &self.model.vae)?;
        timings.load += started.elapsed();
        progress(Stage::Decode);
        let started = Instant::now();
        let z = vae.denormalize(d, &latents)?;
        let image = vae.decode(d, &z, grid)?;
        timings.decode = started.elapsed();
        Ok(Generated {
            width: image.width,
            height: image.height,
            rgba: rgba8(&image)?,
            timings,
            seed: request.seed,
        })
    }
}

fn system_turn() -> String {
    format!("<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n")
}

/// The text-to-image prompt template (not the tokenizer's chat template: the
/// checkpoint was trained on this string).
pub fn template(prompt: &str) -> String {
    format!(
        "{}<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n",
        system_turn()
    )
}

/// `n` standard normal values from `seed` (Box-Muller over ChaCha8). These
/// are not torch's, so the same seed gives a different image than diffusers.
pub fn gaussian(seed: u64, n: usize) -> Vec<f32> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut out = Vec::with_capacity(n + 1);
    while out.len() < n {
        let (u1, u2) = (1. - rng.random::<f64>(), rng.random::<f64>());
        let r = (-2. * u1.ln()).sqrt();
        let angle = std::f64::consts::TAU * u2;
        out.push((r * angle.cos()) as f32);
        out.push((r * angle.sin()) as f32);
    }
    out.truncate(n);
    out
}

/// `[-1, 1]` floats to 8-bit RGBA; a three-channel decoder gets opaque alpha.
fn rgba8(image: &Image) -> Result<Vec<u8>> {
    if !matches!(image.channels, 3 | 4) {
        return Err(Error::Shape(format!("{} image channels", image.channels)));
    }
    let byte = |v: f32| ((v * 0.5 + 0.5).clamp(0., 1.) * 255.).round() as u8;
    Ok(image
        .pixels
        .chunks_exact(image.channels)
        .flat_map(|p| {
            [
                byte(p[0]),
                byte(p[1]),
                byte(p[2]),
                p.get(3).map_or(255, |&a| byte(a)),
            ]
        })
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn noise_is_standard_normal_and_repeatable() {
        let a = gaussian(7, 100_001);
        assert_eq!(a, gaussian(7, 100_001));
        assert_ne!(a[..16], gaussian(8, 16)[..]);
        let mean = a.iter().map(|&v| f64::from(v)).sum::<f64>() / a.len() as f64;
        let var = a
            .iter()
            .map(|&v| (f64::from(v) - mean).powi(2))
            .sum::<f64>()
            / a.len() as f64;
        assert!(
            mean.abs() < 0.02 && (var - 1.).abs() < 0.03,
            "mean {mean} var {var}"
        );
    }

    #[test]
    fn template_wraps_the_prompt_in_the_chat_turns() {
        assert_eq!(
            template("a fox"),
            "<|im_start|>system\nComprehend and analyze the provided prompt.<|im_end|>\n\
             <|im_start|>user\na fox<|im_end|>\n<|im_start|>assistant\n"
        );
    }

    #[test]
    fn pixels_convert_with_rounding_and_clamping() {
        let image = Image {
            width: 2,
            height: 1,
            channels: 4,
            pixels: vec![-1., 0., 1., 1., -2., 0.5, 3., 0.],
        };
        assert_eq!(rgba8(&image).unwrap(), [0, 128, 255, 255, 0, 191, 255, 128]);
    }
}
