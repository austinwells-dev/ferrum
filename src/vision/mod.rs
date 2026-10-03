//! Vision encoder for Qwen3-VL style projectors (`clip` GGUF with
//! `clip.projector_type = qwen3vl_merger`), the `mmproj` file that ships next
//! to Qwen3.5/3.6/3.8-family checkpoints. An image becomes a grid of
//! language-model embeddings that replace `<|image_pad|>` rows in the prompt.
//!
//! Pipeline: decode, resize to a multiple of `patch * merge`, normalise, cut
//! into 16x16 patches (ordered so each 2x2 merge block is contiguous), embed
//! (the two temporal conv taps are folded into one matrix because a still image
//! repeats its frame), add the bilinearly resized learned position table, run
//! the ViT blocks with 2-D rotary attention, then merge 2x2 patches through the
//! MLP projector into the language model's hidden width.
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
/// Hard cap on one image's longest decoded side before resizing.
const MAX_SOURCE_SIDE: u32 = 16384;
/// Attention score scratch (all heads of a group) stays under this many bytes.
const SCORE_BYTES: usize = 192 << 20;

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
}

impl ImageEmbedding {
    pub fn tokens(&self) -> usize {
        self.grid.0 * self.grid.1
    }
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
    post: (Tensor, Tensor),
    merge1: (Tensor, Tensor),
    merge2: (Tensor, Tensor),
    zero: Tensor,
    weight_bytes: usize,
    /// Attention score scratch per head group.
    score_bytes: usize,
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
    let wanted = width(&p, "clip.vision.projection_dim");
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

/// Device bytes `VisionModel::load` will take, for memory planning.
pub fn projector_bytes(path: &Path) -> usize {
    GgufFile::open(path).map_or(0, |f| {
        let weights: usize = f
            .tensors()
            .values()
            .filter(|t| t.name.starts_with("v.blk.") || t.name.starts_with("mm."))
            .map(|t| t.dimensions.iter().product::<usize>() * 2)
            .sum();
        // Weights plus the transient activations of a full-size image.
        weights + (640 << 20)
    })
}

impl VisionModel {
    pub fn load(d: &MetalDevice, path: impl AsRef<Path>, max_tokens: usize) -> Result<Self> {
        let path = path.as_ref();
        let mut file = GgufFile::open(path)?;
        let md = file.metadata().clone();
        let unsupported =
            |what: String| Err(Error::Config(format!("unsupported projector: {what}")));
        match md.get("general.architecture") {
            Some(MetadataValue::String(a)) if a == "clip" => {}
            _ => return unsupported("not a clip/mmproj GGUF".into()),
        }
        match md.get("clip.projector_type") {
            Some(MetadataValue::String(t)) if t == "qwen3vl_merger" => {}
            Some(MetadataValue::String(t)) => {
                return unsupported(format!(
                    "projector type {t} (only qwen3vl_merger is implemented)"
                ));
            }
            _ => return unsupported("missing clip.projector_type".into()),
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
        let h = config.hidden;
        let mut load = Loader {
            d,
            file: &mut file,
            bytes: 0,
        };
        let patch_len = config.patch_len();
        // The two temporal taps see the same frame, so their sum is the kernel.
        let (w0, w1) = (
            load.f32s("v.patch_embd.weight")?,
            load.f32s("v.patch_embd.weight.1")?,
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
        let position = load.f32s("v.position_embd.weight")?;
        if position.len() != config.table * config.table * h {
            return unsupported("position table shape".into());
        }
        let mut layers = Vec::with_capacity(config.layers);
        for i in 0..config.layers {
            let n = |s: &str| format!("v.blk.{i}.{s}");
            layers.push(Layer {
                ln1: (
                    load.vector(&n("ln1.weight"), h)?,
                    load.vector(&n("ln1.bias"), h)?,
                ),
                ln2: (
                    load.vector(&n("ln2.weight"), h)?,
                    load.vector(&n("ln2.bias"), h)?,
                ),
                qkv: (
                    load.matrix(&n("attn_qkv.weight"), 3 * h, h)?,
                    load.vector(&n("attn_qkv.bias"), 3 * h)?,
                ),
                out: (
                    load.matrix(&n("attn_out.weight"), h, h)?,
                    load.vector(&n("attn_out.bias"), h)?,
                ),
                up: (
                    load.matrix(&n("ffn_up.weight"), config.ffn, h)?,
                    load.vector(&n("ffn_up.bias"), config.ffn)?,
                ),
                down: (
                    load.matrix(&n("ffn_down.weight"), h, config.ffn)?,
                    load.vector(&n("ffn_down.bias"), h)?,
                ),
            });
        }
        let merged = h * config.merge * config.merge;
        let post = (
            load.vector("v.post_ln.weight", h)?,
            load.vector("v.post_ln.bias", h)?,
        );
        let merge1 = (
            load.matrix("mm.0.weight", merged, merged)?,
            load.vector("mm.0.bias", merged)?,
        );
        let merge2 = (
            load.matrix("mm.2.weight", config.out_dim, merged)?,
            load.vector("mm.2.bias", config.out_dim)?,
        );
        let weight_bytes = load.bytes;
        let zero = Tensor::zeros(d, [1], DType::F32)?;
        Ok(Self {
            config,
            max_tokens: max_tokens.max(4),
            min_tokens: DEFAULT_MIN_TOKENS,
            layers,
            patch,
            position,
            post,
            merge1,
            merge2,
            zero,
            weight_bytes,
            score_bytes: SCORE_BYTES,
        })
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

    /// Pixel size an image of `width` x `height` is resized to.
    pub fn target_size(&self, width: usize, height: usize) -> (usize, usize) {
        let f = self.config.factor();
        smart_resize(
            height,
            width,
            f,
            self.min_tokens.min(self.max_tokens) * f * f,
            self.max_tokens * f * f,
        )
    }

    /// Decode and resize an encoded image (PNG, JPEG, GIF, WebP, BMP).
    pub fn prepare(&self, bytes: &[u8]) -> Result<Prepared> {
        let hash = content_hash(bytes);
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
        // Transparent areas are shown on white, as a viewer would.
        let rgba = decoded.to_rgba8();
        let mut rgb = image::RgbImage::new(rgba.width(), rgba.height());
        for (o, p) in rgb.pixels_mut().zip(rgba.pixels()) {
            let a = u32::from(p[3]);
            let mix = |c: u8| ((u32::from(c) * a + 255 * (255 - a) + 127) / 255) as u8;
            *o = image::Rgb([mix(p[0]), mix(p[1]), mix(p[2])]);
        }
        let (width, height) = self.target_size(rgb.width() as usize, rgb.height() as usize);
        let rgb = if (rgb.width() as usize, rgb.height() as usize) == (width, height) {
            rgb
        } else {
            image::imageops::resize(
                &rgb,
                width as u32,
                height as u32,
                image::imageops::FilterType::CatmullRom,
            )
        };
        let c = &self.config;
        let mut pixels = vec![0f32; 3 * width * height];
        for (i, p) in rgb.pixels().enumerate() {
            for ch in 0..3 {
                pixels[ch * width * height + i] =
                    (f32::from(p[ch]) / 255. - c.mean[ch]) / c.std[ch];
            }
        }
        Ok(Prepared {
            width,
            height,
            pixels,
            hash,
        })
    }

    /// Language-model tokens an image of this prepared size produces.
    pub fn grid(&self, prepared: &Prepared) -> (usize, usize) {
        let f = self.config.factor();
        (prepared.height / f, prepared.width / f)
    }

    /// Encode one image into language-model embeddings.
    pub fn encode(&self, d: &MetalDevice, bytes: &[u8]) -> Result<ImageEmbedding> {
        let prepared = self.prepare(bytes)?;
        self.encode_prepared(d, &prepared)
    }

    pub fn encode_prepared(&self, d: &MetalDevice, image: &Prepared) -> Result<ImageEmbedding> {
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
        for layer in &self.layers {
            self.norm(d, &x, &layer.ln1, &xn, n)?;
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
            self.norm(d, &x, &layer.ln2, &xn, n)?;
            self.linear(d, &xn, &layer.up, &hid, n, c.ffn, h, 1)?;
            self.linear(d, &hid, &layer.down, &tmp, n, h, c.ffn, 0)?;
            self.add(d, &tmp, &x, n * h)?;
        }
        self.norm(d, &x, &self.post, &xn, n)?;
        // Four consecutive rows are one merge block: a reshape concatenates them.
        let merged_rows = n / (c.merge * c.merge);
        let merged_width = h * c.merge * c.merge;
        let grouped = xn.reshape([merged_rows, merged_width])?;
        let mid = Tensor::zeros(d, [merged_rows, merged_width], DType::F32)?;
        let out = Tensor::zeros(d, [merged_rows, c.out_dim], DType::F32)?;
        self.linear(
            d,
            &grouped,
            &self.merge1,
            &mid,
            merged_rows,
            merged_width,
            merged_width,
            2,
        )?;
        self.linear(
            d,
            &mid,
            &self.merge2,
            &out,
            merged_rows,
            c.out_dim,
            merged_width,
            0,
        )?;
        execution.finish()?;
        Ok(ImageEmbedding {
            hash: image.hash,
            grid: (gh / c.merge, gw / c.merge),
            rows: std::rc::Rc::new(out),
        })
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
            &[
                a.binding(),
                b.binding(),
                bias.unwrap_or(&self.zero).binding(),
            ],
            &[c.binding()],
            &params.0,
            [g.n.div_ceil(32), g.m.div_ceil(32), g.batch],
            [128, 1, 1],
            0,
        )
    }

    fn norm(
        &self,
        d: &MetalDevice,
        x: &Tensor,
        w: &(Tensor, Tensor),
        out: &Tensor,
        rows: usize,
    ) -> Result<()> {
        d.dispatch_hybrid(
            "v_layernorm",
            &[x.binding(), w.0.binding(), w.1.binding()],
            &[out.binding()],
            &Params::default()
                .u(rows)?
                .u(self.config.hidden)?
                .f(self.config.eps)
                .0,
            [rows, 1, 1],
            [256, 1, 1],
            0,
        )
    }

    /// `acc += y` over `count` values.
    fn add(&self, d: &MetalDevice, y: &Tensor, acc: &Tensor, count: usize) -> Result<()> {
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

struct Loader<'a> {
    d: &'a MetalDevice,
    file: &'a mut GgufFile,
    bytes: usize,
}

impl Loader<'_> {
    fn raw(&mut self, name: &str) -> Result<(u32, Vec<usize>, Vec<u8>)> {
        let info = self.file.tensor(name)?.clone();
        let mut data = vec![0u8; info.byte_len];
        self.file.read_tensor_into(name, &mut data)?;
        Ok((info.type_id, info.dimensions, data))
    }

    /// Any float tensor as F32 values.
    fn f32s(&mut self, name: &str) -> Result<Vec<f32>> {
        let (ty, _, data) = self.raw(name)?;
        Ok(match ty {
            0 => data
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                .collect(),
            1 => data
                .chunks_exact(2)
                .map(|b| half::f16::from_bits(u16::from_le_bytes([b[0], b[1]])).to_f32())
                .collect(),
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

    fn vector(&mut self, name: &str, len: usize) -> Result<Tensor> {
        let v = self.f32s(name)?;
        if v.len() != len {
            return Err(Error::Gguf(format!(
                "projector tensor {name} has {} values, expected {len}",
                v.len()
            )));
        }
        self.bytes += len * 4;
        Tensor::from_f32(self.d, [len], DType::F32, &v)
    }

    /// A `[rows, cols]` weight stored as F16.
    fn matrix(&mut self, name: &str, rows: usize, cols: usize) -> Result<Tensor> {
        let info = self.file.tensor(name)?.clone();
        if info.dimensions != [cols, rows] {
            return Err(Error::Gguf(format!(
                "projector tensor {name} has shape {:?}, expected [{cols}, {rows}]",
                info.dimensions
            )));
        }
        self.bytes += rows * cols * 2;
        if info.type_id == 1 {
            let file = &mut *self.file;
            return Tensor::from_reader(self.d, [rows, cols], DType::F16, |dst| {
                file.read_tensor_into(name, dst)
            });
        }
        let values = self.f32s(name)?;
        Tensor::from_f32(self.d, [rows, cols], DType::F16, &values)
    }
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

pub mod media;

#[cfg(test)]
mod tests;
