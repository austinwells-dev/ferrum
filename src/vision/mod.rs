//! Vision encoders for still images. Qwen3-VL projectors
//! (`clip.projector_type = qwen3vl_merger`), the `mmproj` file next to
//! Qwen3.5/3.6/3.8-family checkpoints, are the path in this file: an image
//! becomes a grid of embeddings that replace `<|image_pad|>` rows.
//! LFM3-VL (`clip.vision.projector_type = lfm3vl`) is the private `lfm3` module. It repeats
//! `<|image|>` and gives each of those tokens its own position, not a 2-D grid.
//! LFM3-Audio's speech encoder (`clip.audio.projector_type = lfm3a`) is
//! `AudioModel` (the private `lfm3a` and public `audio` modules): its rows
//! replace repeated `<|audio|>` tokens the same way. Speech output
//! (`lfm3a_gen`) and video are refused.
//!
//! Pipeline: decode, resize to a multiple of `patch * merge`, normalise, cut
//! into 16x16 patches (ordered so each 2x2 merge block is contiguous), embed
//! (the two temporal conv taps are folded into one matrix because a still image
//! repeats its frame), add the bilinearly resized learned position table, run
//! the ViT blocks with 2-D rotary attention, then merge 2x2 patches through the
//! MLP projector into the language model's hidden width.
//!
//! The same tower also loads from a Hugging Face Qwen3-VL checkpoint
//! (`model.visual.*`, as the image generator's text encoder ships it), where
//! it has DeepStack mergers: features of a few intermediate blocks, merged
//! like the output, that the language model adds to the image rows after its
//! first layers.
#![forbid(unsafe_code)]
use crate::{
    DType, Error, MetalDevice, Result, Tensor,
    hybrid::engine::Params,
    loader::gguf::{GgufFile, MetadataValue},
};
use std::{
    collections::BTreeMap,
    hash::{Hash, Hasher},
    path::Path,
};

/// Default floor on language-model tokens per image (a 256x256 image).
pub const DEFAULT_MIN_TOKENS: usize = 64;
/// Default cap on language-model tokens per image.
pub const DEFAULT_MAX_TOKENS: usize = 1024;
/// Cap for `detail: "low"` images: a 512x512 picture, as OpenAI's low detail.
pub const LOW_DETAIL_TOKENS: usize = 256;
/// Hard cap on one image's longest decoded side before resizing.
const MAX_SOURCE_SIDE: u32 = 16384;
/// Attention score scratch (all heads of a group) stays under this many bytes.
pub const SCORE_BYTES: usize = 192 << 20;

#[derive(Debug, Clone)]
pub struct VisionConfig {
    pub patch: usize,
    pub hidden: usize,
    pub ffn: usize,
    pub layers: usize,
    pub heads: usize,
    pub eps: f32,
    pub merge: usize,
    /// Embedding width of the language model the projector feeds.
    pub out_dim: usize,
    /// Side of the learned position table (`image_size / patch`).
    pub table: usize,
    pub mean: [f32; 3],
    pub std: [f32; 3],
}

impl VisionConfig {
    fn head_dim(&self) -> usize {
        self.hidden / self.heads
    }
    fn patch_len(&self) -> usize {
        3 * self.patch * self.patch
    }
    /// Pixels of one merged language-model token (a `patch*merge` square).
    fn factor(&self) -> usize {
        self.patch * self.merge
    }
}

/// Token limits one request asks for an image, within the projector's own
/// (`VisionModel::limits`). `None` keeps the projector's setting.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct ImageBudget {
    pub min_tokens: Option<usize>,
    pub max_tokens: Option<usize>,
}

impl ImageBudget {
    /// Each limit this budget leaves unset, taken from `fallback`.
    pub fn or(self, fallback: Self) -> Self {
        Self {
            min_tokens: self.min_tokens.or(fallback.min_tokens),
            max_tokens: self.max_tokens.or(fallback.max_tokens),
        }
    }
}

/// A decoded, resized and normalised image ready to be cut into patches.
pub struct Prepared {
    /// Resized width and height in pixels (multiples of `patch * merge`).
    pub width: usize,
    pub height: usize,
    /// Channel-first `[3, height, width]`, normalised.
    pub pixels: Vec<f32>,
    /// Identity of the source bytes, for caches and prefix reuse.
    pub hash: u64,
}

/// What an image contributes to the prompt.
pub struct ImageEmbedding {
    pub hash: u64,
    /// Language-model token grid (rows, columns).
    pub grid: (usize, usize),
    /// `[rows * columns, out_dim]` F32, in raster order.
    pub rows: std::rc::Rc<Tensor>,
    /// DeepStack features, one per merger, each shaped like `rows`: entry
    /// `i` is added to the image rows after language-model layer `i`.
    pub deepstack: Vec<std::rc::Rc<Tensor>>,
}

impl ImageEmbedding {
    pub fn tokens(&self) -> usize {
        self.grid.0 * self.grid.1
    }
}

/// Norm, linear, GELU, linear from a `merge x merge` block of patches to one
/// language-model row. The output merger normalises each patch before the
/// blocks are concatenated; DeepStack mergers normalise the concatenation.
struct Merger {
    norm: (Tensor, Tensor),
    fc1: (Tensor, Tensor),
    fc2: (Tensor, Tensor),
    post_shuffle: bool,
}

struct Layer {
    ln1: (Tensor, Tensor),
    ln2: (Tensor, Tensor),
    qkv: (Tensor, Tensor),
    out: (Tensor, Tensor),
    up: (Tensor, Tensor),
    down: (Tensor, Tensor),
}

pub struct VisionModel {
    pub config: VisionConfig,
    /// Most language-model tokens one image may take.
    pub max_tokens: usize,
    /// Fewest tokens an image is scaled up to (Qwen-VL grounding is more
    /// accurate with larger inputs, at the cost of encode time).
    pub min_tokens: usize,
    layers: Vec<Layer>,
    patch: (Tensor, Tensor),
    position: Vec<f32>,
    merger: Merger,
    /// DeepStack mergers and the block whose output each reads.
    deepstack: Vec<(usize, Merger)>,
    zero: Tensor,
    weight_bytes: usize,
    /// Attention score scratch per head group.
    score_bytes: usize,
    /// LFM3-VL tower. Absent on the Qwen path.
    lfm3: Option<lfm3::Tower>,
    /// Placeholder the chat template emits once per image (`<|image_pad|>` or
    /// `<|image|>`). Repeated once per language-model token.
    marker: &'static str,
    /// Resize filter. Qwen is Catmull-Rom; LFM3 follows the file.
    filter: image::imageops::FilterType,
}

fn meta_uint(md: &BTreeMap<String, MetadataValue>, key: &str) -> Result<usize> {
    crate::hybrid::config::uint(md, key)
        .map_err(|_| Error::Config(format!("mmproj is missing GGUF metadata {key}")))
}

fn meta_float(md: &BTreeMap<String, MetadataValue>, key: &str) -> Result<f32> {
    match md.get(key) {
        Some(MetadataValue::Float32(v)) => Ok(*v),
        Some(MetadataValue::Float64(v)) => Ok(*v as f32),
        _ => Err(Error::Config(format!(
            "mmproj is missing GGUF metadata {key}"
        ))),
    }
}

fn meta_floats(md: &BTreeMap<String, MetadataValue>, key: &str) -> Result<[f32; 3]> {
    let bad = || Error::Config(format!("mmproj metadata {key} must be three numbers"));
    let Some(MetadataValue::Array { values, .. }) = md.get(key) else {
        return Err(bad());
    };
    let mut out = [0.; 3];
    if values.len() != 3 {
        return Err(bad());
    }
    for (o, v) in out.iter_mut().zip(values) {
        *o = match v {
            MetadataValue::Float32(f) => *f,
            MetadataValue::Float64(f) => *f as f32,
            _ => return Err(bad()),
        };
    }
    Ok(out)
}

/// Identity of an encoded image, for caches and prefix reuse.
pub fn content_hash(bytes: &[u8]) -> u64 {
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    bytes.hash(&mut hasher);
    hasher.finish()
}

/// Does `path` look like a vision projector (`general.architecture = clip`)?
pub fn is_projector(path: &Path) -> bool {
    GgufFile::open(path).is_ok_and(|f| {
        matches!(
            f.metadata_value("general.architecture"),
            Some(MetadataValue::String(a)) if a == "clip"
        )
    })
}

/// Sibling `mmproj*.gguf` files of `model`, best match first.
pub fn find_projectors(model: &Path) -> Vec<std::path::PathBuf> {
    let (Some(dir), Some(stem)) = (model.parent(), model.file_stem()) else {
        return Vec::new();
    };
    let stem = stem.to_string_lossy().to_lowercase();
    let Ok(entries) = std::fs::read_dir(dir) else {
        return Vec::new();
    };
    let mut found: Vec<_> = entries
        .flatten()
        .map(|e| e.path())
        .filter(|p| {
            let name = p.file_name().map(|n| n.to_string_lossy().to_lowercase());
            name.is_some_and(|n| n.contains("mmproj") && n.ends_with(".gguf"))
        })
        .collect();
    // Prefer a projector named after the model, then the smallest filename.
    found.sort_by_key(|p| {
        let name = p.file_name().map(|n| n.to_string_lossy().to_lowercase());
        let named = name.is_some_and(|n| n.contains(&stem));
        (!named, p.clone())
    });
    found
}

/// Does the projector at `projector` produce embeddings of `model`'s width?
fn fits_model(projector: &Path, model: &Path) -> bool {
    let (Ok(p), Ok(m)) = (GgufFile::open(projector), GgufFile::open(model)) else {
        return false;
    };
    let Some(MetadataValue::String(arch)) = m.metadata_value("general.architecture") else {
        return false;
    };
    let width = |f: &GgufFile, key: &str| match f.metadata_value(key) {
        Some(MetadataValue::Uint32(v)) => Some(u64::from(*v)),
        Some(MetadataValue::Uint64(v)) => Some(*v),
        _ => None,
    };
    let wanted =
        width(&p, "clip.vision.projection_dim").or_else(|| width(&p, "clip.audio.projection_dim"));
    wanted.is_some() && wanted == width(&m, &format!("{arch}.embedding_length"))
}

/// Choose the projector for `model` from a `--mmproj` value: absent or
/// `auto` finds a sibling `mmproj*.gguf` of the right width, `none` disables
/// vision, anything else must be a projector file.
pub fn resolve_projector(model: &Path, flag: Option<&str>) -> Result<Option<std::path::PathBuf>> {
    match flag {
        None | Some("auto") => Ok(find_projectors(model)
            .into_iter()
            .find(|p| is_projector(p) && fits_model(p, model))),
        Some("none" | "off") => Ok(None),
        Some(path) => {
            let path = std::path::PathBuf::from(path);
            if !is_projector(&path) {
                return Err(Error::Parameter(format!(
                    "{} is not a vision projector (mmproj) GGUF",
                    path.display()
                )));
            }
            Ok(Some(path))
        }
    }
}

/// Device bytes `VisionModel::load` or `AudioModel::load` will take, for
/// memory planning.
pub fn projector_bytes(path: &Path) -> usize {
    let audio = is_audio_projector(path);
    GgufFile::open(path).map_or(0, |f| {
        let weights: usize = f
            .tensors()
            .values()
            .filter(|t| {
                if audio {
                    (t.name.starts_with("a.") && !t.name.starts_with("a.gen."))
                        || t.name.starts_with("mm.a.")
                } else {
                    t.name.starts_with("v.blk.") || t.name.starts_with("mm.")
                }
            })
            .map(|t| t.dimensions.iter().product::<usize>() * 2)
            .sum();
        // Weights plus the transient activations of a full-size image, or of
        // a `MAX_AUDIO_SECONDS` clip (its downsampler stages dominate).
        weights
            + if audio {
                AUDIO_ACTIVATION_BYTES
            } else {
                640 << 20
            }
    })
}

/// Longest clip one request may attach. The encoder runs a clip in one pass,
/// so its activations grow with the clip.
pub const MAX_AUDIO_SECONDS: usize = 120;
/// Encoder activations of a `MAX_AUDIO_SECONDS` clip (about 380 MB measured
/// from the buffer sizes in `lfm3a::Encoder::encode`), rounded up.
const AUDIO_ACTIVATION_BYTES: usize = 400 << 20;

/// Is `path` a speech-input projector with no vision tower (LFM3-Audio's
/// `mmproj`)? Combined files (LFM3-Omni) load as vision projectors.
pub fn is_audio_projector(path: &Path) -> bool {
    GgufFile::open(path).is_ok_and(|f| {
        let md = f.metadata();
        matches!(md.get("clip.audio.projector_type"), Some(MetadataValue::String(t)) if t == "lfm3a")
            && !matches!(md.get("clip.has_vision_encoder"), Some(MetadataValue::Bool(true)))
    })
}

/// LFM3-Audio's speech encoder: WAV in, one language-model row per 80 ms.
pub struct AudioModel {
    encoder: lfm3a::Encoder,
    mel: audio::MelBank,
}

impl AudioModel {
    pub fn load(d: &MetalDevice, path: impl AsRef<Path>) -> Result<Self> {
        let file = GgufFile::open(path.as_ref())?;
        let md = file.metadata().clone();
        let unsupported = |what: String| {
            Err(Error::Config(format!(
                "unsupported audio projector: {what}"
            )))
        };
        match md.get("general.architecture") {
            Some(MetadataValue::String(a)) if a == "clip" => {}
            _ => return unsupported("not a clip/mmproj GGUF".into()),
        }
        match md.get("clip.audio.projector_type") {
            Some(MetadataValue::String(t)) if t == "lfm3a" => {}
            Some(MetadataValue::String(t)) => {
                return unsupported(format!("projector type {t} (only lfm3a is implemented)"));
            }
            _ => return unsupported("missing clip.audio.projector_type".into()),
        }
        let key = |k: &str| meta_uint(&md, &format!("clip.audio.{k}"));
        let config = lfm3a::Config {
            n_mel: key("num_mel_bins")?,
            hidden: key("embedding_length")?,
            heads: key("attention.head_count")?,
            kv_heads: key("attention.head_count_kv")?,
            ffn: key("feed_forward_length")?,
            layers: key("block_count")?,
            window: key("window_size")?,
            eps: meta_float(&md, "clip.audio.attention.layer_norm_epsilon")?,
            out_dim: key("projection_dim")?,
        };
        let mut source = GgufSource(file);
        let encoder = lfm3a::load(d, config, &mut source)?;
        Ok(Self {
            mel: audio::MelBank::new(encoder.config.n_mel),
            encoder,
        })
    }

    /// Placeholder the chat template emits once per clip, repeated once per
    /// language-model row.
    pub fn marker(&self) -> &'static str {
        "<|audio|>"
    }

    /// Embedding width of the language model this projector feeds.
    pub fn out_dim(&self) -> usize {
        self.encoder.config.out_dim
    }

    pub fn weight_bytes(&self) -> usize {
        self.encoder.weight_bytes
    }

    /// Most language-model rows one clip can take.
    pub fn max_tokens(&self) -> usize {
        lfm3a::tokens_for(audio::MelBank::frames(
            MAX_AUDIO_SECONDS * audio::SAMPLE_RATE,
        ))
    }

    /// Encode a WAV file.
    pub fn encode(&self, d: &MetalDevice, bytes: &[u8]) -> Result<ImageEmbedding> {
        let samples = audio::decode_wav(bytes)?;
        self.encode_samples(d, &samples, content_hash(bytes))
    }

    /// Encode 16 kHz mono samples.
    pub fn encode_samples(
        &self,
        d: &MetalDevice,
        samples: &[f32],
        hash: u64,
    ) -> Result<ImageEmbedding> {
        let seconds = samples.len() as f64 / audio::SAMPLE_RATE as f64;
        if seconds > MAX_AUDIO_SECONDS as f64 {
            return Err(Error::Parameter(format!(
                "a {seconds:.1} s clip is longer than the {MAX_AUDIO_SECONDS} s one request may attach; split it"
            )));
        }
        self.encoder.encode(d, &self.mel.features(samples), hash)
    }

    /// Run a short silent clip through the encoder so its kernels are
    /// compiled before the first request.
    pub fn warm(&self, d: &MetalDevice) -> Result<()> {
        self.encode_samples(d, &[0.; audio::SAMPLE_RATE / 4], 0)
            .map(|_| ())
    }
}

impl VisionModel {
    pub fn load(d: &MetalDevice, path: impl AsRef<Path>, max_tokens: usize) -> Result<Self> {
        let path = path.as_ref();
        let file = GgufFile::open(path)?;
        let md = file.metadata().clone();
        let unsupported =
            |what: String| Err(Error::Config(format!("unsupported projector: {what}")));
        match md.get("general.architecture") {
            Some(MetadataValue::String(a)) if a == "clip" => {}
            _ => return unsupported("not a clip/mmproj GGUF".into()),
        }
        // `clip.projector_type` wins when both keys are set, so a combined
        // audio+vision file that names `lfm3a` there is refused instead of
        // loading the vision half.
        let projector = match (
            md.get("clip.projector_type"),
            md.get("clip.vision.projector_type"),
        ) {
            (Some(MetadataValue::String(t)), _) | (_, Some(MetadataValue::String(t))) => t.as_str(),
            _ => return unsupported("missing clip.projector_type".into()),
        };
        match projector {
            "qwen3vl_merger" | "lfm3vl" => {}
            "lfm3a" | "lfm3a_gen" => {
                return unsupported(format!(
                    "projector type {projector} (audio is not implemented; still images only)"
                ));
            }
            other => {
                return unsupported(format!(
                    "projector type {other} (only qwen3vl_merger and lfm3vl are implemented)"
                ));
            }
        }
        if let Some(MetadataValue::Array { values, .. }) = md.get("clip.vision.is_deepstack_layers")
            && values
                .iter()
                .any(|v| matches!(v, MetadataValue::Bool(true)))
        {
            return unsupported("DeepStack vision layers".into());
        }
        let image_size = meta_uint(&md, "clip.vision.image_size")?;
        let mut config = VisionConfig {
            patch: meta_uint(&md, "clip.vision.patch_size")?,
            hidden: meta_uint(&md, "clip.vision.embedding_length")?,
            ffn: meta_uint(&md, "clip.vision.feed_forward_length")?,
            layers: meta_uint(&md, "clip.vision.block_count")?,
            heads: meta_uint(&md, "clip.vision.attention.head_count")?,
            eps: meta_float(&md, "clip.vision.attention.layer_norm_epsilon")?,
            merge: meta_uint(&md, "clip.vision.spatial_merge_size")?,
            out_dim: meta_uint(&md, "clip.vision.projection_dim")?,
            table: 0,
            mean: meta_floats(&md, "clip.vision.image_mean")?,
            std: meta_floats(&md, "clip.vision.image_std")?,
        };
        if config.patch == 0 || config.heads == 0 || config.merge == 0 || image_size == 0 {
            return unsupported("zero-sized geometry".into());
        }
        config.table = image_size / config.patch;
        if !config.hidden.is_multiple_of(config.heads)
            || !config.head_dim().is_multiple_of(4)
            || !config.hidden.is_multiple_of(8)
            || config.merge != 2
            || config.std.iter().any(|&s| s <= 0.)
        {
            return unsupported("vision geometry".into());
        }
        let mut source = GgufSource(file);
        if projector == "lfm3vl" {
            return Self::load_lfm3(d, config, &md, &mut source, max_tokens);
        }
        Self::build(d, config, &[], &mut source, max_tokens)
    }

    /// LFM3-VL still-image tower. The Qwen fields stay empty; encoding goes
    /// through `lfm3` and language positions are linear.
    fn load_lfm3(
        d: &MetalDevice,
        mut config: VisionConfig,
        md: &BTreeMap<String, MetadataValue>,
        source: &mut dyn Source,
        max_tokens: usize,
    ) -> Result<Self> {
        let unsupported =
            |what: String| Err(Error::Config(format!("unsupported projector: {what}")));
        match md.get("clip.use_silu") {
            Some(MetadataValue::Bool(true)) => {}
            _ => return unsupported("lfm3vl encoder without clip.use_silu".into()),
        }
        let filter = match md.get("clip.vision.image_resize_algo") {
            Some(MetadataValue::String(s)) if s == "bicubic" => {
                image::imageops::FilterType::CatmullRom
            }
            Some(MetadataValue::String(s)) if s == "bilinear" => {
                image::imageops::FilterType::Triangle
            }
            Some(MetadataValue::String(s)) => {
                return unsupported(format!("image resize {s} (bicubic or bilinear)"));
            }
            _ => return unsupported("missing clip.vision.image_resize_algo".into()),
        };
        let kv_heads = meta_uint(md, "clip.vision.attention.head_count_kv")?;
        let min_pixels = meta_uint(md, "clip.vision.image_min_pixels")?;
        let max_pixels = meta_uint(md, "clip.vision.image_max_pixels")?;
        // The position table is a Qwen tensor. LFM3 does not have one.
        config.table = config.table.max(1);
        let (tower, limits) = lfm3::load(
            d, &config, source, max_tokens, kv_heads, min_pixels, max_pixels,
        )?;
        let z = || Tensor::zeros(d, [1], DType::F32);
        let m = || Tensor::zeros(d, [1, 1], DType::F16);
        Ok(Self {
            config,
            max_tokens: limits.max_tokens,
            min_tokens: limits.min_tokens,
            layers: Vec::new(),
            patch: (m()?, z()?),
            position: Vec::new(),
            merger: Merger {
                norm: (z()?, z()?),
                fc1: (m()?, z()?),
                fc2: (m()?, z()?),
                post_shuffle: false,
            },
            deepstack: Vec::new(),
            zero: z()?,
            weight_bytes: tower.weight_bytes,
            score_bytes: SCORE_BYTES,
            lfm3: Some(tower),
            marker: "<|image|>",
            filter,
        })
    }

    /// Load the tower of a Hugging Face Qwen3-VL checkpoint, whose tensors
    /// are under `prefix` (`model.visual.`), with DeepStack mergers reading
    /// the outputs of blocks `deepstack`.
    pub fn load_safetensors(
        d: &MetalDevice,
        ckpt: &crate::diffusion::checkpoint::Checkpoint,
        prefix: &str,
        config: VisionConfig,
        deepstack: &[usize],
        max_tokens: usize,
    ) -> Result<Self> {
        Self::build(
            d,
            config,
            deepstack,
            &mut SafetensorsSource { ckpt, prefix },
            max_tokens,
        )
    }

    fn build(
        d: &MetalDevice,
        config: VisionConfig,
        deepstack: &[usize],
        source: &mut dyn Source,
        max_tokens: usize,
    ) -> Result<Self> {
        let unsupported =
            |what: String| Err(Error::Config(format!("unsupported vision tower: {what}")));
        if deepstack.iter().any(|&l| l >= config.layers) {
            return unsupported(format!("DeepStack layers {deepstack:?}"));
        }
        let h = config.hidden;
        let mut load = Loader {
            d,
            source,
            bytes: 0,
        };
        let patch_len = config.patch_len();
        // The two temporal taps see the same frame, so their sum is the kernel.
        let (w0, w1) = (
            load.source.values("v.patch_embd.weight")?,
            load.source.values("v.patch_embd.weight.1")?,
        );
        if w0.len() != h * patch_len || w1.len() != w0.len() {
            return unsupported("patch embedding shape".into());
        }
        let summed: Vec<f32> = w0.iter().zip(&w1).map(|(a, b)| a + b).collect();
        let patch = (
            Tensor::from_f32(d, [h, patch_len], DType::F16, &summed)?,
            load.vector("v.patch_embd.bias", h)?,
        );
        load.bytes += summed.len() * 2;
        let position = load.source.values("v.position_embd.weight")?;
        if position.len() != config.table * config.table * h {
            return unsupported("position table shape".into());
        }
        let mut layers = Vec::with_capacity(config.layers);
        for i in 0..config.layers {
            let n = |s: &str| format!("v.blk.{i}.{s}");
            layers.push(Layer {
                ln1: load.affine(&n("ln1"), h, None)?,
                ln2: load.affine(&n("ln2"), h, None)?,
                qkv: load.affine(&n("attn_qkv"), 3 * h, Some(h))?,
                out: load.affine(&n("attn_out"), h, Some(h))?,
                up: load.affine(&n("ffn_up"), config.ffn, Some(h))?,
                down: load.affine(&n("ffn_down"), h, Some(config.ffn))?,
            });
        }
        let merged = h * config.merge * config.merge;
        let merger = Merger {
            norm: load.affine("v.post_ln", h, None)?,
            fc1: load.affine("mm.0", merged, Some(merged))?,
            fc2: load.affine("mm.2", config.out_dim, Some(merged))?,
            post_shuffle: false,
        };
        let deepstack = deepstack
            .iter()
            .enumerate()
            .map(|(i, &layer)| {
                let n = |s: &str| format!("v.deepstack.{i}.{s}");
                Ok((
                    layer,
                    Merger {
                        norm: load.affine(&n("norm"), merged, None)?,
                        fc1: load.affine(&n("fc1"), merged, Some(merged))?,
                        fc2: load.affine(&n("fc2"), config.out_dim, Some(merged))?,
                        post_shuffle: true,
                    },
                ))
            })
            .collect::<Result<Vec<_>>>()?;
        let weight_bytes = load.bytes;
        let zero = Tensor::zeros(d, [1], DType::F32)?;
        Ok(Self {
            config,
            max_tokens: max_tokens.max(4),
            min_tokens: DEFAULT_MIN_TOKENS,
            layers,
            patch,
            position,
            merger,
            deepstack,
            zero,
            weight_bytes,
            score_bytes: SCORE_BYTES,
            lfm3: None,
            marker: "<|image_pad|>",
            filter: image::imageops::FilterType::CatmullRom,
        })
    }

    /// Placeholder repeated once per image token.
    pub fn marker(&self) -> &'static str {
        self.marker
    }

    /// Language-model positions are one per token (LFM3), not a 2-D grid.
    pub fn linear_positions(&self) -> bool {
        self.lfm3.is_some()
    }

    pub fn weight_bytes(&self) -> usize {
        self.weight_bytes
    }

    /// Run a minimal image through the encoder so its kernels are compiled
    /// before the first request.
    pub fn warm(&self, d: &MetalDevice) -> Result<()> {
        let f = self.config.factor();
        let image = Prepared {
            width: f,
            height: f,
            pixels: vec![0.; 3 * f * f],
            hash: 0,
        };
        self.encode_prepared(d, &image).map(drop)
    }

    /// The (fewest, most) tokens an image with `budget` takes. A request may
    /// lower the cap but never raise it past `max_tokens`, which the memory
    /// plan budgets the encoder's activations for.
    pub fn limits(&self, budget: ImageBudget) -> (usize, usize) {
        let max = budget
            .max_tokens
            .unwrap_or(self.max_tokens)
            .clamp(4, self.max_tokens);
        (budget.min_tokens.unwrap_or(self.min_tokens).min(max), max)
    }

    /// Pixel size an image of `width` x `height` is resized to.
    pub fn target_size(&self, width: usize, height: usize) -> (usize, usize) {
        self.target_size_with(width, height, ImageBudget::default())
    }

    /// `target_size` within a request's `budget`.
    pub fn target_size_with(
        &self,
        width: usize,
        height: usize,
        budget: ImageBudget,
    ) -> (usize, usize) {
        let f = self.config.factor();
        let (min, max) = self.limits(budget);
        smart_resize(height, width, f, min * f * f, max * f * f)
    }

    /// Decode and resize an encoded image (PNG, JPEG, GIF, WebP, BMP).
    pub fn prepare(&self, bytes: &[u8]) -> Result<Prepared> {
        self.prepare_with(bytes, ImageBudget::default())
    }

    /// `prepare` within a request's `budget`.
    pub fn prepare_with(&self, bytes: &[u8], budget: ImageBudget) -> Result<Prepared> {
        let hash = content_hash(bytes);
        let rgb = on_white(&decode(bytes)?);
        let (width, height) =
            self.target_size_with(rgb.width() as usize, rgb.height() as usize, budget);
        let rgb = if (rgb.width() as usize, rgb.height() as usize) == (width, height) {
            rgb
        } else {
            image::imageops::resize(&rgb, width as u32, height as u32, self.filter)
        };
        Ok(self.normalise(&rgb, hash))
    }

    /// An RGBA image already sized to whole merged tokens, as is: the image
    /// generator sizes condition images itself so both of its encoders see
    /// the same pixels.
    pub fn prepare_rgba(&self, rgba: &image::RgbaImage) -> Result<Prepared> {
        let f = self.config.factor();
        let (w, h) = (rgba.width() as usize, rgba.height() as usize);
        if w == 0 || h == 0 || !w.is_multiple_of(f) || !h.is_multiple_of(f) {
            return Err(Error::Parameter(format!(
                "a {w} x {h} image is not whole {f}-pixel tokens"
            )));
        }
        Ok(self.normalise(&on_white(rgba), content_hash(rgba.as_raw())))
    }

    /// Channel-first pixels in the encoder's normalisation.
    fn normalise(&self, rgb: &image::RgbImage, hash: u64) -> Prepared {
        let c = &self.config;
        let (width, height) = (rgb.width() as usize, rgb.height() as usize);
        let mut pixels = vec![0f32; 3 * width * height];
        for (i, p) in rgb.pixels().enumerate() {
            for ch in 0..3 {
                pixels[ch * width * height + i] =
                    (f32::from(p[ch]) / 255. - c.mean[ch]) / c.std[ch];
            }
        }
        Prepared {
            width,
            height,
            pixels,
            hash,
        }
    }

    /// Language-model tokens an image of this prepared size produces.
    pub fn grid(&self, prepared: &Prepared) -> (usize, usize) {
        let f = self.config.factor();
        (prepared.height / f, prepared.width / f)
    }

    /// Encode one image into language-model embeddings.
    pub fn encode(&self, d: &MetalDevice, bytes: &[u8]) -> Result<ImageEmbedding> {
        self.encode_with(d, bytes, ImageBudget::default())
    }

    /// `encode` within a request's `budget`.
    pub fn encode_with(
        &self,
        d: &MetalDevice,
        bytes: &[u8],
        budget: ImageBudget,
    ) -> Result<ImageEmbedding> {
        let prepared = self.prepare_with(bytes, budget)?;
        self.encode_prepared(d, &prepared)
    }

    pub fn encode_prepared(&self, d: &MetalDevice, image: &Prepared) -> Result<ImageEmbedding> {
        if let Some(tower) = &self.lfm3 {
            return tower.encode(self, d, image);
        }
        let c = &self.config;
        let (gh, gw) = (image.height / c.patch, image.width / c.patch);
        let n = gh * gw;
        let h = c.hidden;
        let hd = c.head_dim();
        let patches = self.patchify(image);
        let position = self.position_rows(gh, gw);
        let (cos, sin) = rope_tables(gh, gw, c.merge, hd);
        let tensor = |dims: &[usize], data: &[f32]| Tensor::from_f32(d, dims, DType::F32, data);
        let patches = tensor(&[n, c.patch_len()], &patches)?;
        let cos = tensor(&[n, hd / 2], &cos)?;
        let sin = tensor(&[n, hd / 2], &sin)?;

        let x = Tensor::zeros(d, [n, h], DType::F32)?;
        let xn = Tensor::zeros(d, [n, h], DType::F32)?;
        let qkv = Tensor::zeros(d, [n, 3 * h], DType::F32)?;
        let att = Tensor::zeros(d, [n, h], DType::F32)?;
        let hid = Tensor::zeros(d, [n, c.ffn], DType::F32)?;
        let tmp = Tensor::zeros(d, [n, h], DType::F32)?;
        let group = (self.score_bytes / (n * n * 4)).clamp(1, c.heads);
        let scores = Tensor::zeros(d, [group * n * n], DType::F32)?;
        let scale = 1. / (hd as f32).sqrt();
        let execution = d.execution_with_shared_encoder(true)?;

        self.linear(d, &patches, &self.patch, &x, n, h, c.patch_len(), 0)?;
        let pos = tensor(&[n, h], &position)?;
        self.add(d, &pos, &x, n * h)?;
        let mut deepstack = Vec::with_capacity(self.deepstack.len());
        for (index, layer) in self.layers.iter().enumerate() {
            self.norm(d, &x, &layer.ln1, &xn, n, h)?;
            self.linear(d, &xn, &layer.qkv, &qkv, n, 3 * h, h, 0)?;
            d.dispatch_hybrid(
                "v_rope",
                &[cos.binding(), sin.binding()],
                &[qkv.binding()],
                &Params::default().u(n)?.u(c.heads)?.u(hd)?.u(3 * h)?.u(h)?.0,
                [(n * 2 * c.heads * (hd / 2)).div_ceil(256), 1, 1],
                [256, 1, 1],
                0,
            )?;
            let mut head = 0;
            while head < c.heads {
                let g = group.min(c.heads - head);
                // scores = scale * Q K^T per head
                self.gemm(
                    d,
                    &Gemm {
                        m: n,
                        n,
                        k: hd,
                        lda: 3 * h,
                        ldb_n: 3 * h,
                        ldb_k: 1,
                        ldc: n,
                        batch: g,
                        strides: (hd, hd, n * n),
                        offsets: (head * hd, h + head * hd, 0),
                        alpha: scale,
                        ..Gemm::default()
                    },
                    &qkv,
                    &qkv,
                    None,
                    &scores,
                )?;
                d.dispatch_hybrid(
                    "v_softmax",
                    &[],
                    &[scores.binding()],
                    &Params::default().u(g * n)?.u(n)?.0,
                    [g * n, 1, 1],
                    [256, 1, 1],
                    0,
                )?;
                // out = P V per head, written into this head's columns
                self.gemm(
                    d,
                    &Gemm {
                        m: n,
                        n: hd,
                        k: n,
                        lda: n,
                        ldb_n: 1,
                        ldb_k: 3 * h,
                        ldc: h,
                        batch: g,
                        strides: (n * n, hd, hd),
                        offsets: (0, 2 * h + head * hd, head * hd),
                        ..Gemm::default()
                    },
                    &scores,
                    &qkv,
                    None,
                    &att,
                )?;
                head += g;
            }
            self.linear(d, &att, &layer.out, &tmp, n, h, h, 0)?;
            self.add(d, &tmp, &x, n * h)?;
            self.norm(d, &x, &layer.ln2, &xn, n, h)?;
            self.linear(d, &xn, &layer.up, &hid, n, c.ffn, h, 1)?;
            self.linear(d, &hid, &layer.down, &tmp, n, h, c.ffn, 0)?;
            self.add(d, &tmp, &x, n * h)?;
            for (_, merger) in self.deepstack.iter().filter(|(l, _)| *l == index) {
                deepstack.push(std::rc::Rc::new(self.merge(d, merger, &x, n)?));
            }
        }
        let out = self.merge(d, &self.merger, &x, n)?;
        execution.finish()?;
        Ok(ImageEmbedding {
            hash: image.hash,
            grid: (gh / c.merge, gw / c.merge),
            rows: std::rc::Rc::new(out),
            deepstack,
        })
    }

    /// Merge `n` patch rows of `x` (block order) through `m`: `[n / merge², out_dim]`.
    fn merge(&self, d: &MetalDevice, m: &Merger, x: &Tensor, n: usize) -> Result<Tensor> {
        let c = &self.config;
        let h = c.hidden;
        // Four consecutive rows are one merge block: a reshape concatenates them.
        let merged_rows = n / (c.merge * c.merge);
        let merged_width = h * c.merge * c.merge;
        let normed = Tensor::zeros(d, [n, h], DType::F32)?;
        if m.post_shuffle {
            let grouped = x.reshape([merged_rows, merged_width])?;
            let out = normed.reshape([merged_rows, merged_width])?;
            self.norm(d, &grouped, &m.norm, &out, merged_rows, merged_width)?;
        } else {
            self.norm(d, x, &m.norm, &normed, n, h)?;
        }
        let grouped = normed.reshape([merged_rows, merged_width])?;
        let mid = Tensor::zeros(d, [merged_rows, merged_width], DType::F32)?;
        let out = Tensor::zeros(d, [merged_rows, c.out_dim], DType::F32)?;
        self.linear(
            d,
            &grouped,
            &m.fc1,
            &mid,
            merged_rows,
            merged_width,
            merged_width,
            2,
        )?;
        self.linear(
            d,
            &mid,
            &m.fc2,
            &out,
            merged_rows,
            c.out_dim,
            merged_width,
            0,
        )?;
        Ok(out)
    }

    /// Patch vectors `[tokens, 3*patch*patch]`, channel-major, with the
    /// patches of each `merge x merge` block adjacent.
    fn patchify(&self, image: &Prepared) -> Vec<f32> {
        let c = &self.config;
        let (gh, gw) = (image.height / c.patch, image.width / c.patch);
        let p = c.patch;
        let mut out = Vec::with_capacity(gh * gw * c.patch_len());
        for (row, col) in block_order(gh, gw, c.merge) {
            for ch in 0..3 {
                for y in 0..p {
                    let start =
                        ch * image.width * image.height + (row * p + y) * image.width + col * p;
                    out.extend_from_slice(&image.pixels[start..start + p]);
                }
            }
        }
        out
    }

    /// Patch vectors in raster order (row, then column). LFM3 keeps this order
    /// through the encoder; Qwen regroups into merge blocks ([`patchify`]).
    fn patchify_raster(&self, image: &Prepared) -> Vec<f32> {
        let c = &self.config;
        let (gh, gw) = (image.height / c.patch, image.width / c.patch);
        let p = c.patch;
        let mut out = Vec::with_capacity(gh * gw * c.patch_len());
        for row in 0..gh {
            for col in 0..gw {
                for ch in 0..3 {
                    for y in 0..p {
                        let start =
                            ch * image.width * image.height + (row * p + y) * image.width + col * p;
                        out.extend_from_slice(&image.pixels[start..start + p]);
                    }
                }
            }
        }
        out
    }

    /// The learned position table resized (bilinear, corners aligned) to the
    /// patch grid and laid out in block order.
    fn position_rows(&self, gh: usize, gw: usize) -> Vec<f32> {
        let c = &self.config;
        let (t, h) = (c.table, c.hidden);
        let axis = |len: usize| -> Vec<(usize, usize, f32)> {
            (0..len)
                .map(|i| {
                    let at = if len == 1 {
                        0.
                    } else {
                        i as f32 * (t - 1) as f32 / (len - 1) as f32
                    };
                    let lo = (at.floor() as usize).min(t - 1);
                    (lo, (lo + 1).min(t - 1), at - lo as f32)
                })
                .collect()
        };
        let (rows, cols) = (axis(gh), axis(gw));
        let mut out = Vec::with_capacity(gh * gw * h);
        for (row, col) in block_order(gh, gw, c.merge) {
            let (r0, r1, dr) = rows[row];
            let (c0, c1, dc) = cols[col];
            let corner = |r: usize, k: usize| &self.position[(r * t + k) * h..(r * t + k + 1) * h];
            let (a, b, cc, e) = (
                corner(r0, c0),
                corner(r0, c1),
                corner(r1, c0),
                corner(r1, c1),
            );
            for i in 0..h {
                out.push(
                    a[i] * (1. - dr) * (1. - dc)
                        + b[i] * (1. - dr) * dc
                        + cc[i] * dr * (1. - dc)
                        + e[i] * dr * dc,
                );
            }
        }
        out
    }

    #[allow(clippy::too_many_arguments)]
    fn linear(
        &self,
        d: &MetalDevice,
        a: &Tensor,
        w: &(Tensor, Tensor),
        c: &Tensor,
        m: usize,
        n: usize,
        k: usize,
        epilogue: u32,
    ) -> Result<()> {
        self.gemm(
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
            &w.0,
            Some(&w.1),
            c,
        )
    }

    fn gemm(
        &self,
        d: &MetalDevice,
        g: &Gemm,
        a: &Tensor,
        b: &Tensor,
        bias: Option<&Tensor>,
        c: &Tensor,
    ) -> Result<()> {
        run_gemm(d, &self.zero, g, a, b, bias, c)
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
    ) -> Result<()> {
        d.dispatch_hybrid(
            "v_layernorm",
            &[x.binding(), w.0.binding(), w.1.binding()],
            &[out.binding()],
            &Params::default().u(rows)?.u(width)?.f(self.config.eps).0,
            [rows, 1, 1],
            [256, 1, 1],
            0,
        )
    }

    /// `acc += y` over `count` values.
    fn add(&self, d: &MetalDevice, y: &Tensor, acc: &Tensor, count: usize) -> Result<()> {
        run_add(d, y, acc, count)
    }
}

/// `c = epilogue(alpha * a b^T + bias)` through the `v_gemm_*` kernels. `zero`
/// is bound in place of a missing bias.
fn run_gemm(
    d: &MetalDevice,
    zero: &Tensor,
    g: &Gemm,
    a: &Tensor,
    b: &Tensor,
    bias: Option<&Tensor>,
    c: &Tensor,
) -> Result<()> {
    let kernel = match b.dtype() {
        DType::F16 => "v_gemm_h",
        DType::F32 => "v_gemm_f",
        _ => return Err(Error::DType),
    };
    let params = Params::default()
        .u(g.m)?
        .u(g.n)?
        .u(g.k)?
        .u(g.lda)?
        .u(g.ldb_n)?
        .u(g.ldb_k)?
        .u(g.ldc)?
        .u(g.strides.0)?
        .u(g.strides.1)?
        .u(g.strides.2)?
        .u(g.offsets.0)?
        .u(g.offsets.1)?
        .u(g.offsets.2)?
        .u(g.epilogue as usize)?
        .u(usize::from(bias.is_some()))?
        .f(g.alpha);
    d.dispatch_hybrid(
        kernel,
        &[a.binding(), b.binding(), bias.unwrap_or(zero).binding()],
        &[c.binding()],
        &params.0,
        [g.n.div_ceil(32), g.m.div_ceil(32), g.batch],
        [128, 1, 1],
        0,
    )
}

/// `acc += y` over `count` values.
fn run_add(d: &MetalDevice, y: &Tensor, acc: &Tensor, count: usize) -> Result<()> {
    d.dispatch_hybrid(
        "v_add",
        &[y.binding()],
        &[acc.binding()],
        &Params::default().u(count)?.u(1)?.0,
        [count.div_ceil(256), 1, 1],
        [256, 1, 1],
        0,
    )
}

/// Strided batched GEMM description for the `v_gemm_*` kernels.
struct Gemm {
    m: usize,
    n: usize,
    k: usize,
    lda: usize,
    ldb_n: usize,
    ldb_k: usize,
    ldc: usize,
    batch: usize,
    /// Per-batch element strides of A, B and C.
    strides: (usize, usize, usize),
    /// Element offsets of batch zero in A, B and C.
    offsets: (usize, usize, usize),
    epilogue: u32,
    alpha: f32,
}

impl Default for Gemm {
    fn default() -> Self {
        Self {
            m: 0,
            n: 0,
            k: 0,
            lda: 0,
            ldb_n: 0,
            ldb_k: 1,
            ldc: 0,
            batch: 1,
            strides: (0, 0, 0),
            offsets: (0, 0, 0),
            epilogue: 0,
            alpha: 1.,
        }
    }
}

/// Where the tower's tensors come from, by their GGUF names
/// (`v.blk.0.ln1.weight`).
trait Source {
    fn contains(&mut self, name: &str) -> bool;
    /// Any float tensor's values.
    fn values(&mut self, name: &str) -> Result<Vec<f32>>;
    /// A `[rows, cols]` matrix as F16 on the device.
    fn matrix(&mut self, d: &MetalDevice, name: &str, rows: usize, cols: usize) -> Result<Tensor>;
}

struct GgufSource(GgufFile);

impl GgufSource {
    fn raw(&mut self, name: &str) -> Result<(u32, Vec<u8>)> {
        let info = self.0.tensor(name)?.clone();
        let mut data = vec![0u8; info.byte_len];
        self.0.read_tensor_into(name, &mut data)?;
        Ok((info.type_id, data))
    }
}

impl Source for GgufSource {
    fn contains(&mut self, name: &str) -> bool {
        self.0.tensors().contains_key(name)
    }

    fn values(&mut self, name: &str) -> Result<Vec<f32>> {
        let (ty, data) = self.raw(name)?;
        Ok(match ty {
            0 => data
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                .collect(),
            1 => data
                .chunks_exact(2)
                .map(|b| half::f16::from_bits(u16::from_le_bytes([b[0], b[1]])).to_f32())
                .collect(),
            8 => dequant_q8(name, &data)?,
            30 => data
                .chunks_exact(2)
                .map(|b| half::bf16::from_bits(u16::from_le_bytes([b[0], b[1]])).to_f32())
                .collect(),
            t => {
                return Err(Error::Gguf(format!(
                    "projector tensor {name} has unsupported type {t}"
                )));
            }
        })
    }

    fn matrix(&mut self, d: &MetalDevice, name: &str, rows: usize, cols: usize) -> Result<Tensor> {
        let info = self.0.tensor(name)?.clone();
        if info.dimensions != [cols, rows] {
            return Err(Error::Gguf(format!(
                "projector tensor {name} has shape {:?}, expected [{cols}, {rows}]",
                info.dimensions
            )));
        }
        if info.type_id == 1 {
            let file = &mut self.0;
            return Tensor::from_reader(d, [rows, cols], DType::F16, |dst| {
                file.read_tensor_into(name, dst)
            });
        }
        let values = self.values(name)?;
        if values.len() != rows * cols {
            return Err(Error::Gguf(format!(
                "projector tensor {name} has {} values, expected {}",
                values.len(),
                rows * cols
            )));
        }
        Tensor::from_f32(d, [rows, cols], DType::F16, &values)
    }
}

/// ggml Q8_0: 32 int8 values and one F16 scale per 34-byte block.
fn dequant_q8(name: &str, data: &[u8]) -> Result<Vec<f32>> {
    if !data.len().is_multiple_of(34) {
        return Err(Error::Gguf(format!(
            "projector tensor {name} has {} bytes, not a whole Q8_0 block",
            data.len()
        )));
    }
    let mut out = Vec::with_capacity(data.len() / 34 * 32);
    for block in data.chunks_exact(34) {
        let scale = half::f16::from_le_bytes([block[0], block[1]]).to_f32();
        for &q in &block[2..] {
            out.push(f32::from(q as i8) * scale);
        }
    }
    Ok(out)
}

/// A Hugging Face Qwen3-VL tower under `prefix`.
struct SafetensorsSource<'a> {
    ckpt: &'a crate::diffusion::checkpoint::Checkpoint,
    prefix: &'a str,
}

impl SafetensorsSource<'_> {
    fn name(&self, gguf: &str) -> String {
        transformers_name(self.prefix, gguf)
    }
}

/// The transformers checkpoint's name, under `prefix`, for a tower tensor's
/// GGUF name.
fn transformers_name(prefix: &str, gguf: &str) -> String {
    {
        let (stem, kind) = gguf.rsplit_once('.').unwrap_or((gguf, ""));
        let tail = match stem.split('.').collect::<Vec<_>>()[..] {
            ["v", "patch_embd"] => "patch_embed.proj".to_string(),
            ["v", "position_embd"] => "pos_embed".to_string(),
            ["v", "post_ln"] => "merger.norm".to_string(),
            ["mm", "0"] => "merger.linear_fc1".to_string(),
            ["mm", "2"] => "merger.linear_fc2".to_string(),
            ["v", "blk", i, part] => format!(
                "blocks.{i}.{}",
                match part {
                    "ln1" => "norm1",
                    "ln2" => "norm2",
                    "attn_qkv" => "attn.qkv",
                    "attn_out" => "attn.proj",
                    "ffn_up" => "mlp.linear_fc1",
                    "ffn_down" => "mlp.linear_fc2",
                    other => other,
                }
            ),
            ["v", "deepstack", i, part] => format!(
                "deepstack_merger_list.{i}.{}",
                match part {
                    "fc1" => "linear_fc1",
                    "fc2" => "linear_fc2",
                    other => other,
                }
            ),
            _ => stem.to_string(),
        };
        format!("{prefix}{tail}.{kind}")
    }
}

impl Source for SafetensorsSource<'_> {
    fn contains(&mut self, name: &str) -> bool {
        self.ckpt.entry(&self.name(name)).is_ok()
    }

    fn values(&mut self, name: &str) -> Result<Vec<f32>> {
        // The patch kernel is one `[out, 3, 2, p, p]` tensor; GGUF splits
        // its two temporal taps.
        let tap = match name {
            "v.patch_embd.weight" => Some(0),
            "v.patch_embd.weight.1" => Some(1),
            _ => None,
        };
        let Some(tap) = tap else {
            return self.ckpt.read_f32(&self.name(name));
        };
        let full = self.name("v.patch_embd.weight");
        let shape = self.ckpt.entry(&full)?.shape.clone();
        let [out, channels, 2, ph, pw] = shape[..] else {
            return Err(Error::Weight {
                name: full,
                message: format!("expected a [out, 3, 2, p, p] kernel, found {shape:?}"),
            });
        };
        let all = self.ckpt.read_f32(&full)?;
        let taps = ph * pw;
        Ok((0..out * channels)
            .flat_map(|oc| &all[(oc * 2 + tap) * taps..][..taps])
            .copied()
            .collect())
    }

    fn matrix(&mut self, d: &MetalDevice, name: &str, rows: usize, cols: usize) -> Result<Tensor> {
        let full = self.name(name);
        let shape = &self.ckpt.entry(&full)?.shape;
        if shape[..] != [rows, cols] {
            return Err(Error::Weight {
                name: full,
                message: format!("expected [{rows}, {cols}], found {shape:?}"),
            });
        }
        Tensor::from_f32(d, [rows, cols], DType::F16, &self.ckpt.read_f32(&full)?)
    }
}

struct Loader<'a> {
    d: &'a MetalDevice,
    source: &'a mut dyn Source,
    bytes: usize,
}

impl Loader<'_> {
    fn vector(&mut self, name: &str, len: usize) -> Result<Tensor> {
        let v = self.source.values(name)?;
        if v.len() != len {
            return Err(Error::Weight {
                name: name.into(),
                message: format!("{} values, expected {len}", v.len()),
            });
        }
        self.bytes += len * 4;
        Tensor::from_f32(self.d, [len], DType::F32, &v)
    }

    /// `{name}.weight` and `{name}.bias`: a `[rows, cols]` matrix, or with
    /// no `cols` a norm's `rows` scales.
    fn affine(&mut self, name: &str, rows: usize, cols: Option<usize>) -> Result<(Tensor, Tensor)> {
        let weight = format!("{name}.weight");
        let weight = match cols {
            Some(cols) => {
                self.bytes += rows * cols * 2;
                self.source.matrix(self.d, &weight, rows, cols)?
            }
            None => self.vector(&weight, rows)?,
        };
        Ok((weight, self.vector(&format!("{name}.bias"), rows)?))
    }
}

/// Decode an encoded image (PNG, JPEG, GIF, WebP, BMP) of a supported size.
pub fn decode(bytes: &[u8]) -> Result<image::RgbaImage> {
    let decoded = image::load_from_memory(bytes)
        .map_err(|e| Error::Parameter(format!("cannot decode image: {e}")))?;
    if decoded.width().max(decoded.height()) > MAX_SOURCE_SIDE
        || decoded.width() == 0
        || decoded.height() == 0
    {
        return Err(Error::Parameter(format!(
            "image of {}x{} pixels is unsupported",
            decoded.width(),
            decoded.height()
        )));
    }
    Ok(decoded.to_rgba8())
}

/// Transparent areas shown on white, as a viewer would (and as Qwen-Image
/// was trained to read them).
fn on_white(rgba: &image::RgbaImage) -> image::RgbImage {
    let mut rgb = image::RgbImage::new(rgba.width(), rgba.height());
    for (o, p) in rgb.pixels_mut().zip(rgba.pixels()) {
        let a = u32::from(p[3]);
        let mix = |c: u8| ((u32::from(c) * a + 255 * (255 - a) + 127) / 255) as u8;
        *o = image::Rgb([mix(p[0]), mix(p[1]), mix(p[2])]);
    }
    rgb
}

/// Patch coordinates (row, column) in merge-block order: each `merge x merge`
/// block of patches is contiguous, blocks run in raster order.
fn block_order(gh: usize, gw: usize, merge: usize) -> impl Iterator<Item = (usize, usize)> {
    (0..gh / merge).flat_map(move |br| {
        (0..gw / merge).flat_map(move |bc| {
            (0..merge)
                .flat_map(move |dy| (0..merge).map(move |dx| (br * merge + dy, bc * merge + dx)))
        })
    })
}

/// Cosine and sine of the 2-D rotary angles, `[tokens, head_dim/2]` each:
/// the first half of the pairs rotates with the row, the second with the column.
fn rope_tables(gh: usize, gw: usize, merge: usize, head_dim: usize) -> (Vec<f32>, Vec<f32>) {
    let half = head_dim / 2;
    let axis = half / 2;
    let freq: Vec<f32> = (0..axis)
        .map(|j| 10000f32.powf(-2. * j as f32 / half as f32))
        .collect();
    let mut cos = Vec::with_capacity(gh * gw * half);
    let mut sin = Vec::with_capacity(gh * gw * half);
    for (row, col) in block_order(gh, gw, merge) {
        for i in 0..half {
            let (pos, f) = if i < axis {
                (row, freq[i])
            } else {
                (col, freq[i - axis])
            };
            let angle = pos as f32 * f;
            cos.push(angle.cos());
            sin.push(angle.sin());
        }
    }
    (cos, sin)
}

/// Qwen2-VL `smart_resize`: round each side to a multiple of `factor`, then
/// scale into `[min_pixels, max_pixels]` keeping the aspect ratio.
pub fn smart_resize(
    height: usize,
    width: usize,
    factor: usize,
    min_pixels: usize,
    max_pixels: usize,
) -> (usize, usize) {
    let f = factor as f64;
    let (h, w) = (height.max(1) as f64, width.max(1) as f64);
    let mut hb = ((h / f).round() * f).max(f);
    let mut wb = ((w / f).round() * f).max(f);
    if hb * wb > max_pixels as f64 {
        let beta = (h * w / max_pixels as f64).sqrt();
        hb = ((h / beta / f).floor() * f).max(f);
        wb = ((w / beta / f).floor() * f).max(f);
    } else if hb * wb < min_pixels as f64 {
        let beta = (min_pixels as f64 / (h * w)).sqrt();
        hb = (h * beta / f).ceil() * f;
        wb = (w * beta / f).ceil() * f;
    }
    // (width, height)
    (wb as usize, hb as usize)
}

pub mod audio;
mod lfm3;
mod lfm3a;
pub mod media;

#[cfg(test)]
mod tests;
