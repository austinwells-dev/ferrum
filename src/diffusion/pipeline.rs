//! Prompt (and condition images, for an edit) to image: encode the images
//! and the prompt, denoise latents, decode them. The models are loaded one
//! at a time and released before the next, which is what fits a 7B
//! transformer and an 8B encoder on a 32 GB Mac.
//!
//! An edit sizes each condition image to about `resolution²` pixels; the
//! vision tower reads it inside the prompt, and the VAE's latents of it sit
//! in the transformer's sequence where the prompt held the image, so the
//! target attends to both.
use super::{
    checkpoint::{Checkpoint, GgufWeights},
    config::ModelDir,
    dit::{Dit, Prefix, Rope, Scratch},
    rope::{Layout, Segment},
    schedule::Schedule,
    text::{self, TextEncoder},
    vae::{Image, Vae},
    vae_encoder::VaeEncoder,
};
use crate::{
    DType, Error, MetalDevice, Result, Tensor,
    hybrid::{engine::ImageSpan, weights::Format},
    tokenizer::Tokenizer,
};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use std::{
    path::{Path, PathBuf},
    time::{Duration, Instant},
};

/// The system prompt of the text-to-image template.
const SYSTEM_PROMPT: &str = "Comprehend and analyze the provided prompt.";

/// Longest prompt, in text tokens after the system turn.
pub const MAX_PROMPT_TOKENS: usize = 1024;
/// Most condition images one edit takes.
pub const MAX_IMAGES: usize = 4;
/// Most denoising steps a request may ask for.
pub const MAX_STEPS: usize = 200;
/// The side an edit sizes its condition images (and, by default, its
/// output) to the area of.
pub const DEFAULT_RESOLUTION: usize = 1024;

/// Image sides are multiples of this (16 pixels per latent token, in pairs).
pub const SIZE_STEP: usize = 32;
pub const MIN_SIDE: usize = 256;
pub const MAX_SIDE: usize = 2048;

/// Sides must be multiples of `SIZE_STEP` within `MIN_SIDE..=MAX_SIDE`.
pub fn check_size(width: usize, height: usize) -> Result<()> {
    for (name, side) in [("width", width), ("height", height)] {
        if !(MIN_SIDE..=MAX_SIDE).contains(&side) || !side.is_multiple_of(SIZE_STEP) {
            return Err(Error::Parameter(format!(
                "{name} {side} must be a multiple of {SIZE_STEP} between {MIN_SIDE} and {MAX_SIDE}"
            )));
        }
    }
    Ok(())
}

/// How the weights are stored on the device.
#[derive(Debug, Clone, Default)]
pub struct Settings {
    /// Quantize the encoder's layer matrices as they load (BF16 when `None`).
    pub text: Option<Format>,
    /// The same for the transformer's block matrices.
    pub transformer: Option<Format>,
    /// Take the transformer from this GGUF instead of the model directory,
    /// keeping the encoding it is stored in (so `transformer` must be unset).
    pub transformer_file: Option<PathBuf>,
    /// Take the text encoder's language model from a Qwen3-VL GGUF, keeping
    /// its encoding (so `text` must be unset).
    pub text_file: Option<PathBuf>,
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

/// A condition image: 8-bit RGBA, row-major.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InputImage {
    pub width: usize,
    pub height: usize,
    pub rgba: Vec<u8>,
}

impl InputImage {
    /// Decode an encoded image (PNG, JPEG, GIF, WebP, BMP).
    pub fn decode(bytes: &[u8]) -> Result<Self> {
        let rgba = crate::vision::decode(bytes)?;
        Ok(Self {
            width: rgba.width() as usize,
            height: rgba.height() as usize,
            rgba: rgba.into_raw(),
        })
    }

    /// Resized (Lanczos) to about `resolution²` pixels at this aspect ratio,
    /// sides multiples of `SIZE_STEP`, as diffusers sizes condition images.
    fn sized(&self, resolution: usize) -> Result<image::RgbaImage> {
        let (w, h) = dimensions(resolution, self.width, self.height);
        let source =
            image::RgbaImage::from_raw(self.width as u32, self.height as u32, self.rgba.clone())
                .ok_or_else(|| Error::Shape("pixel buffer does not match the image size".into()))?;
        if (w, h) == (self.width, self.height) {
            return Ok(source);
        }
        Ok(image::imageops::resize(
            &source,
            w as u32,
            h as u32,
            image::imageops::FilterType::Lanczos3,
        ))
    }
}

/// The `(width, height)` of about `resolution²` pixels at the aspect ratio
/// of `width x height`, each a multiple of `SIZE_STEP` (diffusers'
/// `calculate_dimensions`, whose `round` takes ties to even).
pub fn dimensions(resolution: usize, width: usize, height: usize) -> (usize, usize) {
    let ratio = width as f64 / height as f64;
    let w = ((resolution * resolution) as f64 * ratio).sqrt();
    let h = w / ratio;
    let step = SIZE_STEP as f64;
    let side = |v: f64| (((v / step).round_ties_even() as usize) * SIZE_STEP).max(SIZE_STEP);
    (side(w), side(h))
}

#[derive(Debug, Clone)]
pub struct Request {
    pub prompt: String,
    /// Output `(width, height)`; `None` follows the last condition image's
    /// aspect ratio at `resolution²` pixels, or is `resolution` square.
    pub size: Option<(usize, usize)>,
    /// Condition images: an edit when there are any.
    pub images: Vec<InputImage>,
    /// The side whose square area condition images are sized to.
    pub resolution: usize,
    pub seed: u64,
    /// Denoising steps; `None` takes the checkpoint's own schedule (8 for
    /// Turbo, 40 for the base model).
    pub steps: Option<usize>,
}

impl Request {
    pub fn new(prompt: impl Into<String>) -> Self {
        Self {
            prompt: prompt.into(),
            size: None,
            images: Vec::new(),
            resolution: DEFAULT_RESOLUTION,
            seed: 0,
            steps: None,
        }
    }

    /// The `(width, height)` this request generates (before validation).
    pub fn output_size(&self) -> (usize, usize) {
        match (self.size, self.images.last()) {
            (Some(size), _) => size,
            (None, Some(last)) => dimensions(self.resolution, last.width, last.height),
            (None, None) => (self.resolution, self.resolution),
        }
    }
}

/// Where a generation is, for progress reporting.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Stage {
    /// The condition images go through the vision tower and the VAE.
    EncodeImages,
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

/// Watches a generation: told each stage as it starts, and asked between
/// stages and steps whether to carry on. Any `FnMut(Stage)` is an observer
/// that never cancels.
pub trait Observer {
    fn stage(&mut self, stage: Stage);
    fn cancelled(&self) -> bool {
        false
    }
}

impl<F: FnMut(Stage)> Observer for F {
    fn stage(&mut self, stage: Stage) {
        self(stage);
    }
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
        if (settings.transformer_file.is_some() && settings.transformer.is_some())
            || (settings.text_file.is_some() && settings.text.is_some())
        {
            return Err(Error::Parameter(
                "a GGUF keeps its own encoding; drop the quantization for the same model".into(),
            ));
        }
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

    /// The template's token ids for `prompt` with condition images of these
    /// sized `(width, height)`s (their slots widened to their tokens), and
    /// how many lead the system turn.
    pub fn tokenize(&self, prompt: &str, images: &[(usize, usize)]) -> Result<(Vec<u32>, usize)> {
        let p = self.prompt(prompt, images)?;
        Ok((p.ids, p.skip))
    }

    /// The encoder's input for `prompt` with condition images of these
    /// (sized) `(width, height)`s: each `<|image_pad|>` widened to the
    /// image's vision tokens, and the transformer's sequence it becomes.
    pub(crate) fn prompt(&self, prompt: &str, images: &[(usize, usize)]) -> Result<Prompt> {
        let ids = self.tokenizer.encode(&template(prompt, images.len()))?;
        let pad = self.model.vision.as_ref().map(|v| v.image_token);
        let vision_step = self
            .model
            .vision
            .as_ref()
            .map_or(SIZE_STEP, |v| v.config.patch * v.config.merge);
        let latent = self.model.vae.scale_factor_spatial;
        let mut expanded = Vec::with_capacity(ids.len());
        let (mut spans, mut prefix) = (Vec::new(), Vec::new());
        let mut text = 0;
        for (i, &id) in ids.iter().enumerate() {
            if Some(id) != pad {
                expanded.push(id);
                text += usize::from(i >= self.drop_idx);
                continue;
            }
            let &(w, h) = images.get(spans.len()).ok_or_else(|| {
                Error::Parameter("the prompt holds more image slots than images".into())
            })?;
            if text > 0 {
                prefix.push(Segment::Text(text));
                text = 0;
            }
            prefix.push(Segment::Image(h / latent, w / latent));
            let grid = (h / vision_step, w / vision_step);
            spans.push((expanded.len(), grid));
            expanded.extend(std::iter::repeat_n(id, grid.0 * grid.1));
        }
        if spans.len() != images.len() {
            return Err(Error::Parameter(format!(
                "{} images for {} image slots in the prompt",
                images.len(),
                spans.len()
            )));
        }
        if text > 0 {
            prefix.push(Segment::Text(text));
        }
        let layout = Layout {
            prefix,
            target: (0, 0),
        };
        if layout.text_rows() > MAX_PROMPT_TOKENS {
            return Err(Error::Parameter(format!(
                "the prompt is {} tokens; the limit is {MAX_PROMPT_TOKENS}",
                layout.text_rows()
            )));
        }
        Ok(Prompt {
            ids: expanded,
            skip: self.drop_idx,
            spans,
            layout,
        })
    }

    /// The output's latent grid and the sigma schedule for `request`, after
    /// checking its parameters.
    fn geometry(&self, request: &Request) -> Result<((usize, usize), Schedule)> {
        let (width, height) = request.output_size();
        check_size(width, height)?;
        if !(MIN_SIDE..=MAX_SIDE).contains(&request.resolution) {
            return Err(Error::Parameter(format!(
                "resolution {} must be between {MIN_SIDE} and {MAX_SIDE}",
                request.resolution
            )));
        }
        if request.images.len() > MAX_IMAGES {
            return Err(Error::Parameter(format!(
                "{} condition images; the limit is {MAX_IMAGES}",
                request.images.len()
            )));
        }
        if !request.images.is_empty() && self.model.vision.is_none() {
            return Err(Error::Config(
                "this checkpoint's text encoder has no vision tower, so it cannot edit".into(),
            ));
        }
        if let Some(steps) = request.steps
            && !(1..=MAX_STEPS).contains(&steps)
        {
            return Err(Error::Parameter(format!(
                "steps {steps} must be between 1 and {MAX_STEPS}"
            )));
        }
        let scale = self.model.vae.scale_factor_spatial;
        let grid = (height / scale, width / scale);
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
                Schedule::dynamic(steps.unwrap_or(40), grid.0 * grid.1, &self.model.scheduler)?
            }
        };
        Ok((grid, schedule))
    }

    /// The condition images sized as the encoders read them.
    fn conditions(&self, request: &Request) -> Result<Vec<image::RgbaImage>> {
        request
            .images
            .iter()
            .map(|i| i.sized(request.resolution))
            .collect()
    }

    /// Device bytes each stage needs at its peak, for `request`, in the
    /// order they run.
    pub fn stage_bytes(&self, request: &Request) -> Result<Vec<(&'static str, usize)>> {
        let (grid, _) = self.geometry(request)?;
        let sizes: Vec<_> = request
            .images
            .iter()
            .map(|i| dimensions(request.resolution, i.width, i.height))
            .collect();
        let prompt = self.prompt(&request.prompt, &sizes)?;
        self.plan(&prompt, &sizes, grid)
    }

    /// Whether every stage of `request` fits beside what `d` has allocated
    /// now (another model in this process counts), without saying why not.
    pub fn fits_now(&self, d: &MetalDevice, request: &Request) -> Result<bool> {
        let free = (d.recommended_max_working_set() as usize).saturating_sub(d.allocated_bytes());
        Ok(self
            .stage_bytes(request)?
            .iter()
            .all(|&(_, bytes)| bytes <= free))
    }

    fn plan(
        &self,
        prompt: &Prompt,
        sizes: &[(usize, usize)],
        grid: (usize, usize),
    ) -> Result<Vec<(&'static str, usize)>> {
        use crate::hybrid::engine::page;
        let mut stages = Vec::new();
        // The embeddings and DeepStack features of every image wait for the
        // text encoder, which reads them.
        let mut features = 0;
        if let Some(vision) = self.model.vision.as_ref().filter(|_| !sizes.is_empty()) {
            let tokens = prompt.spans.iter().map(|(_, g)| g.0 * g.1);
            stages.push((
                "vision",
                text::vision_bytes(vision, tokens.clone().max().unwrap_or(0)),
            ));
            features = prompt
                .spans
                .iter()
                .map(|(_, g)| {
                    (1 + vision.deepstack.len()) * page(g.0 * g.1 * self.model.text.hidden * 4)
                })
                .sum();
            let largest = sizes
                .iter()
                .map(|&(w, h)| VaeEncoder::estimate_bytes(&self.model.vae, w, h));
            stages.push(("image encoder", largest.max().unwrap_or(0) + features));
        }
        let ids = prompt.ids.len();
        let encoder = match &self.settings.text_file {
            Some(path) => TextEncoder::estimate_bytes_gguf(&self.model.text, ids, path)?,
            None => TextEncoder::estimate_bytes(&self.model.text, ids, self.settings.text),
        };
        stages.push(("encoder", encoder + features));
        let cfg = &self.model.dit;
        let weights = match &self.settings.transformer_file {
            Some(path) => GgufWeights::open(path)?.estimate_bytes(),
            None => Dit::estimate_bytes(cfg, self.settings.transformer),
        };
        let (prefix, text, tokens) = (
            prompt.layout.prefix_rows(),
            prompt.layout.text_rows(),
            grid.0 * grid.1,
        );
        // Beside the weights, scratch and prefix: the prompt embeddings and
        // condition latents (until the prefill ends), the rotary tables, the
        // key limits, the latents and the prediction.
        let inputs = page(text * cfg.context * 4) + page((prefix - text) * cfg.in_channels * 4);
        let rope = 2 * page((prefix + tokens) * cfg.head_dim / 2 * 4);
        let state = 2 * page(tokens * cfg.in_channels * 4) + page(prefix * 4);
        stages.push((
            "transformer",
            weights
                + Scratch::bytes(cfg, tokens.max(prefix))
                + Prefix::bytes(cfg, prefix)
                + inputs
                + rope
                + state,
        ));
        stages.push(("decoder", Vae::estimate_bytes(&self.model.vae, grid)));
        Ok(stages)
    }

    pub fn generate(
        &self,
        d: &MetalDevice,
        request: &Request,
        observer: &mut dyn Observer,
    ) -> Result<Generated> {
        let (grid, schedule) = self.geometry(request)?;
        let conditions = self.conditions(request)?;
        let sizes: Vec<_> = conditions
            .iter()
            .map(|c| (c.width() as usize, c.height() as usize))
            .collect();
        let prompt = self.prompt(&request.prompt, &sizes)?;
        let plan = self.plan(&prompt, &sizes, grid)?;
        // Each stage is checked up front against what is resident now (a
        // chat model counts), and again as it starts, since others may have
        // allocated since.
        let budget = d.recommended_max_working_set() as usize;
        for &(name, bytes) in &plan {
            fits(d, budget, name, bytes)?;
        }
        let need = |name: &str| plan.iter().find(|(n, _)| *n == name).map_or(0, |p| p.1);
        let mut stage = |s: Stage| -> Result<()> {
            if observer.cancelled() {
                return Err(Error::Cancelled);
            }
            observer.stage(s);
            Ok(())
        };
        let mut timings = Timings::default();
        let tokens = grid.0 * grid.1;

        // Condition images: vision embeddings for the prompt, latents for
        // the transformer.
        let mut media = Vec::new();
        let mut condition_latents = Vec::new();
        if let Some(vision) = self
            .model
            .vision
            .as_ref()
            .filter(|_| !conditions.is_empty())
        {
            stage(Stage::EncodeImages)?;
            fits(d, budget, "vision", need("vision"))?;
            let started = Instant::now();
            let ckpt = Checkpoint::open(self.model.path("text_encoder"))?;
            let tower = text::load_vision(d, &ckpt, vision)?;
            for (image, &(start, grid)) in conditions.iter().zip(&prompt.spans) {
                let embedded = tower.encode_prepared(d, &tower.prepare_rgba(image)?)?;
                if embedded.grid != grid {
                    return Err(Error::Shape(format!(
                        "the vision tower made a {:?} grid for {grid:?} slots",
                        embedded.grid
                    )));
                }
                media.push(ImageSpan {
                    start,
                    grid,
                    embedding: embedded.rows,
                    deepstack: embedded.deepstack,
                    hash: embedded.hash,
                });
            }
            drop(tower);
            fits(d, budget, "image encoder", need("image encoder"))?;
            let encoder = VaeEncoder::load(
                d,
                &Checkpoint::open(self.model.path("vae"))?,
                &self.model.vae,
            )?;
            for image in &conditions {
                let (w, h) = (image.width() as usize, image.height() as usize);
                condition_latents.extend(encoder.encode(d, image.as_raw(), w, h)?.to_f32());
            }
            timings.encode += started.elapsed();
        }

        let embeds = {
            stage(Stage::LoadEncoder)?;
            fits(d, budget, "encoder", need("encoder"))?;
            let started = Instant::now();
            let ids = prompt.ids.len();
            let encoder = match &self.settings.text_file {
                Some(path) => TextEncoder::load_gguf(d, path, &self.model.text, ids)?,
                None => {
                    let ckpt = Checkpoint::open(self.model.path("text_encoder"))?;
                    TextEncoder::load(d, &ckpt, &self.model.text, ids, self.settings.text)?
                }
            };
            timings.load += started.elapsed();
            stage(Stage::Encode)?;
            let started = Instant::now();
            let rows = encoder.encode(d, &prompt.ids, prompt.skip, std::mem::take(&mut media))?;
            let embeds = prompt.text_rows(d, &rows)?;
            timings.encode += started.elapsed();
            embeds
        };

        let latents = {
            stage(Stage::LoadTransformer)?;
            fits(d, budget, "transformer", need("transformer"))?;
            let started = Instant::now();
            let dit = match &self.settings.transformer_file {
                Some(path) => {
                    let gguf = GgufWeights::open(path)?;
                    Dit::load(d, &gguf, &self.model.dit, None)?
                }
                None => {
                    let ckpt = Checkpoint::open(self.model.path("transformer"))?;
                    Dit::load(d, &ckpt, &self.model.dit, self.settings.transformer)?
                }
            };
            let layout = Layout {
                target: grid,
                ..prompt.layout.clone()
            };
            let scratch = Scratch::new(d, &self.model.dit, layout.prefix_rows().max(tokens))?;
            let rope = Rope::new(d, &self.model.dit, &layout)?;
            timings.load += started.elapsed();

            let started = Instant::now();
            let channels = self.model.dit.in_channels;
            let conditions = (!condition_latents.is_empty())
                .then(|| {
                    Tensor::from_f32(
                        d,
                        [condition_latents.len() / channels, channels],
                        DType::F32,
                        &condition_latents,
                    )
                })
                .transpose()?;
            let prefix = dit.prefill(
                d,
                &scratch,
                &layout,
                &embeds,
                conditions.as_ref(),
                &rope,
                None,
            )?;
            drop((embeds, conditions));
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
                stage(Stage::Denoise {
                    step: i + 1,
                    steps: schedule.steps(),
                })?;
            }
            timings.denoise = started.elapsed();
            latents
        };
        // A blown-up activation leaves NaN latents, which would decode to a
        // plausible-looking black image.
        if latents.to_f32().iter().any(|v| !v.is_finite()) {
            return Err(Error::Validation(
                "denoising produced non-finite latents".into(),
            ));
        }

        stage(Stage::LoadDecoder)?;
        fits(d, budget, "decoder", need("decoder"))?;
        let started = Instant::now();
        let ckpt = Checkpoint::open(self.model.path("vae"))?;
        let vae = Vae::load(d, &ckpt, &self.model.vae)?;
        timings.load += started.elapsed();
        stage(Stage::Decode)?;
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

/// The encoder's input and how its output becomes the transformer's prefix.
pub(crate) struct Prompt {
    /// Template tokens with each image slot widened to its vision tokens.
    pub ids: Vec<u32>,
    /// Leading tokens (the system turn) the transformer does not read.
    pub skip: usize,
    /// Where each image's tokens start in `ids`, and its vision grid.
    pub spans: Vec<(usize, (usize, usize))>,
    /// The transformer's prefix (its target is set later).
    pub layout: Layout,
}

impl Prompt {
    /// The encoder's rows after `skip` at text positions only: the
    /// transformer reads condition latents at the image positions instead.
    pub fn text_rows(&self, d: &MetalDevice, rows: &Tensor) -> Result<Tensor> {
        if self.spans.is_empty() {
            return Ok(rows.clone());
        }
        let hidden = rows.numel() / (self.ids.len() - self.skip);
        let all = rows.to_f32();
        let in_image = |i: usize| {
            self.spans
                .iter()
                .any(|&(start, g)| (start..start + g.0 * g.1).contains(&(i + self.skip)))
        };
        let kept: Vec<f32> = all
            .chunks_exact(hidden)
            .enumerate()
            .filter(|&(i, _)| !in_image(i))
            .flat_map(|(_, row)| row.iter().copied())
            .collect();
        Tensor::from_f32(d, [kept.len() / hidden, hidden], DType::F32, &kept)
    }
}

/// Whether a stage of `bytes` fits beside what is allocated now.
fn fits(d: &MetalDevice, budget: usize, name: &str, bytes: usize) -> Result<()> {
    let resident = d.allocated_bytes();
    if resident + bytes > budget {
        eprintln!(
            "the {name} stage needs about {:.1} GiB with {:.1} GiB already in use, over this device's {:.1} GiB working set",
            bytes as f64 / GIB,
            resident as f64 / GIB,
            budget as f64 / GIB
        );
        return Err(Error::Allocation(bytes));
    }
    Ok(())
}

const GIB: f64 = (1u64 << 30) as f64;

fn system_turn() -> String {
    format!("<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n")
}

/// The prompt template with `images` condition-image slots before the
/// prompt (not the tokenizer's chat template: the checkpoint was trained on
/// this string).
pub fn template(prompt: &str, images: usize) -> String {
    let slots: Vec<String> = (1..=images)
        .map(|i| format!("<image{i}><|vision_start|><|image_pad|><|vision_end|>"))
        .collect();
    format!(
        "{}<|im_start|>user\n{}{prompt}<|im_end|>\n<|im_start|>assistant\n",
        system_turn(),
        slots.join(" ")
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
            template("a fox", 0),
            "<|im_start|>system\nComprehend and analyze the provided prompt.<|im_end|>\n\
             <|im_start|>user\na fox<|im_end|>\n<|im_start|>assistant\n"
        );
        assert_eq!(
            template("swap them", 2),
            "<|im_start|>system\nComprehend and analyze the provided prompt.<|im_end|>\n\
             <|im_start|>user\n<image1><|vision_start|><|image_pad|><|vision_end|> \
             <image2><|vision_start|><|image_pad|><|vision_end|>swap them<|im_end|>\n\
             <|im_start|>assistant\n"
        );
    }

    /// diffusers' `calculate_dimensions` for the same inputs.
    #[test]
    fn condition_sizes_follow_diffusers() {
        assert_eq!(dimensions(1024, 1024, 1024), (1024, 1024));
        assert_eq!(dimensions(1024, 1920, 1080), (1376, 768));
        assert_eq!(dimensions(512, 300, 400), (448, 576));
        // A side that rounds to nothing stays one step (diffusers gives 0).
        assert_eq!(dimensions(256, 1, 4000), (32, 16192));
    }

    #[test]
    fn requests_size_from_the_last_image() {
        let mut r = Request::new("x");
        assert_eq!(r.output_size(), (1024, 1024));
        r.images.push(InputImage {
            width: 1920,
            height: 1080,
            rgba: vec![0; 1920 * 1080 * 4],
        });
        assert_eq!(r.output_size(), (1376, 768));
        r.size = Some((512, 512));
        assert_eq!(r.output_size(), (512, 512));
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
