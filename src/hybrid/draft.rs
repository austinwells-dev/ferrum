//! Block-diffusion drafters from Hugging Face checkpoints: DFlash, DFlash2
//! (grouped dynamic convolutions and a candidate-path selector) and DSpark
//! (a low-rank Markov bias chained across the block and a confidence head).
//!
//! The drafter reads the target's residual stream at a few layers. Every
//! committed position is fused (`fc`, `hidden_norm`) and projected to K/V for
//! each draft layer, then stored in a ring of `window` slots tagged with its
//! position. A draft step embeds `[anchor, mask, ...]` with the target's
//! embedding, runs the draft layers with non-causal attention over the valid
//! context slots and the block itself, and reads draft tokens through the
//! target's LM head.
//!
//! Drafters load from an HF checkpoint directory (`config.json` +
//! `model.safetensors`) or from a GGUF written by llama.cpp's converter
//! (`general.architecture` `dflash`; see `gguf`).
//!
//! References: the HF `dflash.py`/`dspark.py` shipped with
//! `RadixArk/Qwen3.8-27B-DSpark`, and llama.cpp 9710a32 (`src/models/dflash.cpp`,
//! `common/speculative.cpp`, `conversion/qwen.py`).
#![forbid(unsafe_code)]
mod gguf;
#[cfg(test)]
mod tests;
pub use gguf::{check_tokenizer, is_dflash};

use super::{
    HybridConfig,
    engine::{HybridModel, Params, SpecGeometry, page},
    kflash::Calibration,
    speculative::{DraftTree, Drafter, lattice_best_first, log_softmax},
    weights::{Format, Matrix, gguf_matrix},
};
use crate::{
    DType, Error, MetalDevice, Result, Tensor,
    loader::gguf::{GgufFile, MetadataValue, TensorInfo, block_layout},
};
use serde_json::Value as Json;
use std::{
    collections::HashMap,
    fs::File,
    io::{Read, Seek, SeekFrom},
    path::{Path, PathBuf},
};

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Kind {
    /// Mask slots predict their own position: `block_size - 1` drafts.
    DFlash,
    /// DFlash with dynamic convolutions and the top-k path selector.
    DFlash2 { top_k: usize, rank: usize },
    /// Slot i predicts position anchor + i + 1 with a Markov bias:
    /// `block_size` drafts.
    DSpark { rank: usize, confidence: bool },
}

/// Draft architecture read from `config.json` or GGUF metadata.
#[derive(Debug, Clone, PartialEq)]
pub struct DraftConfig {
    pub kind: Kind,
    pub hidden: usize,
    pub layers: usize,
    pub heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub ffn: usize,
    pub eps: f32,
    pub block_size: usize,
    pub mask_token: u32,
    /// Target layers whose outputs are fused, in `fc` column order.
    pub target_layers: Vec<usize>,
    /// Sliding window of the draft's context attention (None: full).
    pub window: Option<usize>,
    /// RoPE inverse frequencies (head_dim / 2) and the cos/sin scale.
    pub inv_freq: Vec<f32>,
    pub rope_scale: f32,
    /// DFlash2 convolution (kernel, group size).
    pub conv: Option<(usize, usize)>,
}

impl DraftConfig {
    pub fn from_json(root: &Json) -> Result<Self> {
        let bad = |m: &str| Error::Config(format!("draft config: {m}"));
        // speculators nests the transformer config; SpecForge keeps it flat.
        let layer = root.get("transformer_layer_config").unwrap_or(root);
        let dflash = root.get("dflash_config").cloned().unwrap_or(Json::Null);
        let get = |v: &Json, k: &str| v.get(k).and_then(Json::as_u64).map(|x| x as usize);
        let need = |k: &str| get(layer, k).ok_or_else(|| bad(&format!("missing {k}")));
        let hidden = need("hidden_size")?;
        let heads = need("num_attention_heads")?;
        let head_dim = get(layer, "head_dim").unwrap_or(hidden / heads);
        let layers = need("num_hidden_layers")?;
        let has = |k: &str| root.get(k).is_some() || dflash.get(k).is_some();
        let pick = |k: &str| get(&dflash, k).or_else(|| get(root, k));
        let block_size = pick("block_size").ok_or_else(|| bad("missing block_size"))?;
        let mask_token = pick("mask_token_id").ok_or_else(|| bad("missing mask_token_id"))? as u32;
        let target_layers: Vec<usize> = if let Some(ids) = root.get("aux_hidden_state_layer_ids") {
            // speculators: the input of layer i, i.e. the output of layer i - 1.
            ids.as_array()
                .ok_or_else(|| bad("aux_hidden_state_layer_ids"))?
                .iter()
                .map(|v| v.as_u64().map(|i| i as usize - 1))
                .collect::<Option<_>>()
                .ok_or_else(|| bad("aux_hidden_state_layer_ids"))?
        } else {
            dflash
                .get("target_layer_ids")
                .or_else(|| root.get("target_layer_ids"))
                .and_then(Json::as_array)
                .ok_or_else(|| bad("missing target_layer_ids"))?
                .iter()
                .map(|v| v.as_u64().map(|i| i as usize))
                .collect::<Option<_>>()
                .ok_or_else(|| bad("target_layer_ids"))?
        };
        let kind = if has("markov_rank") && pick("markov_rank").unwrap_or(0) > 0 {
            let anchor = root
                .get("sample_from_anchor")
                .and_then(Json::as_bool)
                .unwrap_or(
                    root.get("transformer_layer_config").is_none()
                        && root.get("aux_hidden_state_layer_ids").is_none(),
                );
            if !anchor {
                return Err(bad("DSpark with a bonus-anchor block is not supported"));
            }
            let markov = root
                .get("markov_head_type")
                .and_then(Json::as_str)
                .unwrap_or("vanilla");
            if markov != "vanilla" {
                return Err(bad(&format!("markov head {markov:?}")));
            }
            Kind::DSpark {
                rank: pick("markov_rank").unwrap_or(0),
                confidence: root
                    .get("enable_confidence_head")
                    .and_then(Json::as_bool)
                    .unwrap_or(false),
            }
        } else if let Some(top_k) = get(&dflash, "selector_top_k") {
            Kind::DFlash2 {
                top_k,
                rank: get(&dflash, "selector_rank").ok_or_else(|| bad("selector_rank"))?,
            }
        } else {
            Kind::DFlash
        };
        let conv = match kind {
            Kind::DFlash2 { .. } => Some((
                get(&dflash, "conv_kernel_size").ok_or_else(|| bad("conv_kernel_size"))?,
                get(&dflash, "conv_group_size").ok_or_else(|| bad("conv_group_size"))?,
            )),
            _ => None,
        };
        let sliding = layer
            .get("use_sliding_window")
            .and_then(Json::as_bool)
            .unwrap_or(false);
        let types_sliding = layer
            .get("layer_types")
            .and_then(Json::as_array)
            .is_some_and(|t| t.iter().all(|x| x.as_str() == Some("sliding_attention")));
        let window = match get(layer, "sliding_window") {
            Some(w) if sliding || types_sliding => Some(w),
            _ => None,
        };
        let rope = layer
            .get("rope_parameters")
            .or_else(|| layer.get("rope_scaling"))
            .cloned()
            .unwrap_or(Json::Null);
        let theta = rope
            .get("rope_theta")
            .or_else(|| layer.get("rope_theta"))
            .and_then(Json::as_f64)
            .ok_or_else(|| bad("missing rope_theta"))?;
        let partial = rope
            .get("partial_rotary_factor")
            .and_then(Json::as_f64)
            .unwrap_or(1.);
        if partial != 1. {
            return Err(bad("partial rotary drafts are not supported"));
        }
        let (inv_freq, rope_scale) = match rope.get("rope_type").and_then(Json::as_str) {
            None | Some("default") => (
                (0..head_dim / 2)
                    .map(|i| (theta.powf(-2. * i as f64 / head_dim as f64)) as f32)
                    .collect(),
                1.,
            ),
            Some("yarn") => yarn(
                &rope,
                get(layer, "max_position_embeddings"),
                theta,
                head_dim,
            )?,
            Some(other) => return Err(bad(&format!("rope type {other:?}"))),
        };
        if head_dim % 32 != 0 || head_dim > 256 || heads % need("num_key_value_heads")? != 0 {
            return Err(bad("attention geometry"));
        }
        Ok(Self {
            kind,
            hidden,
            layers,
            heads,
            kv_heads: need("num_key_value_heads")?,
            head_dim,
            ffn: need("intermediate_size")?,
            eps: layer
                .get("rms_norm_eps")
                .and_then(Json::as_f64)
                .unwrap_or(1e-6) as f32,
            block_size,
            mask_token,
            target_layers,
            window,
            inv_freq,
            rope_scale,
            conv,
        })
    }

    /// Most drafts one block yields.
    pub fn max_drafts(&self) -> usize {
        match self.kind {
            Kind::DSpark { .. } => self.block_size,
            _ => self.block_size - 1,
        }
    }
}

/// Hugging Face `_compute_yarn_parameters` (truncated correction range).
fn yarn(rope: &Json, max_pos: Option<usize>, theta: f64, dim: usize) -> Result<(Vec<f32>, f32)> {
    let f = |k: &str, d: f64| rope.get(k).and_then(Json::as_f64).unwrap_or(d);
    let original = f("original_max_position_embeddings", 0.);
    let factor = match (max_pos, original > 0.) {
        (Some(m), true) => m as f64 / original,
        _ => f("factor", 1.),
    };
    let original = if original > 0. {
        original
    } else {
        max_pos.unwrap_or(0) as f64
    };
    let (beta_fast, beta_slow) = (f("beta_fast", 32.), f("beta_slow", 1.));
    let mscale = if factor <= 1. {
        1.
    } else {
        0.1 * factor.ln() + 1.
    };
    let attention_factor = rope
        .get("attention_factor")
        .and_then(Json::as_f64)
        .unwrap_or(mscale);
    let correction = |rotations: f64| {
        dim as f64 * (original / (rotations * 2. * std::f64::consts::PI)).ln() / (2. * theta.ln())
    };
    let low = correction(beta_fast).floor().max(0.);
    let high = correction(beta_slow).ceil().min(dim as f64 - 1.);
    let high = if low == high { high + 0.001 } else { high };
    let inv = (0..dim / 2)
        .map(|i| {
            let pos_freq = theta.powf(2. * i as f64 / dim as f64);
            let extrapolation = 1. / pos_freq;
            let interpolation = 1. / (factor * pos_freq);
            let ramp = ((i as f64 - low) / (high - low)).clamp(0., 1.);
            let keep = 1. - ramp;
            (interpolation * (1. - keep) + extrapolation * keep) as f32
        })
        .collect();
    Ok((inv, attention_factor as f32))
}

/// Weight precision for the draft's projections. Q4_0 halves the draft's
/// memory and bandwidth at no measurable acceptance cost (Phase 9 journal,
/// Experiment 7); the Markov, convolution and selector projections stay Q8_0.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DraftQuant {
    Q8_0,
    #[default]
    Q4_0,
}

#[derive(Debug, Clone, Copy)]
pub struct DraftOptions {
    pub quant: DraftQuant,
    /// Context slots for drafts trained without a sliding window.
    pub context_cap: usize,
    /// DSpark: stop the block where the confidence head falls below this.
    pub confidence_min: f32,
    /// Draft only among token ids below this (the LM head and Markov rows
    /// are prefixes; byte-level BPE ids follow merge order, roughly by
    /// frequency). None: the full vocabulary. Verification always uses the
    /// full vocabulary, so this trades acceptance for draft speed only.
    pub vocab: Option<usize>,
    /// DFlash and DFlash2: rows of the drafting block (the anchor and its
    /// mask slots) when wider than the trained block. One pass then drafts
    /// a deeper candidate lattice for trees (Kapitsa runs DFlash2 drafters
    /// trained on 8 rows with 16). None: the trained block.
    pub block: Option<usize>,
    /// Tune DFlash2 for KFlash trees (`--draft-tree auto`), as Kapitsa does:
    /// unless `block` and `vocab` are set, a 16-row drafting block over the
    /// first 98,304 token ids.
    pub kflash: bool,
    /// How MTP grows draft trees.
    pub tree_search: TreeSearch,
}

/// How MTP grows a draft tree.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum TreeSearch {
    /// Expand the likeliest nodes, a few per MTP pass, wherever they are.
    #[default]
    BestFirst,
    /// Keep MTP's chain and add each step's next-best tokens as leaves.
    Siblings,
}

impl Default for DraftOptions {
    fn default() -> Self {
        Self {
            quant: DraftQuant::Q4_0,
            context_cap: 8192,
            confidence_min: 0.,
            vocab: None,
            block: None,
            kflash: false,
            tree_search: TreeSearch::default(),
        }
    }
}

/// Candidates per depth in a DFlash or DSpark draft tree (DFlash2 keeps its
/// selector's top-k).
const TREE_CANDIDATES: usize = 16;

/// Per draft row, (token, logit) candidates best first.
type Candidates = Vec<Vec<(u32, f32)>>;

/// Safetensors index over one file (BF16 tensors are read on demand).
struct Safetensors {
    file: File,
    base: u64,
    tensors: HashMap<String, (String, Vec<usize>, usize, usize)>,
}

impl Safetensors {
    fn open(path: &Path) -> Result<Self> {
        let mut file =
            File::open(path).map_err(|e| Error::Config(format!("{}: {e}", path.display())))?;
        let mut len = [0u8; 8];
        file.read_exact(&mut len).map_err(io)?;
        let n = u64::from_le_bytes(len) as usize;
        let mut header = vec![0u8; n];
        file.read_exact(&mut header).map_err(io)?;
        let header: serde_json::Map<String, Json> = serde_json::from_slice(&header)
            .map_err(|e| Error::Config(format!("safetensors header: {e}")))?;
        let mut tensors = HashMap::new();
        for (name, info) in header {
            if name == "__metadata__" {
                continue;
            }
            let dtype = info["dtype"].as_str().unwrap_or_default().to_owned();
            let shape = info["shape"]
                .as_array()
                .map(|a| {
                    a.iter()
                        .filter_map(|v| v.as_u64())
                        .map(|v| v as usize)
                        .collect()
                })
                .unwrap_or_default();
            let offsets = info["data_offsets"].as_array().cloned().unwrap_or_default();
            let (a, b) = (
                offsets.first().and_then(Json::as_u64).unwrap_or(0) as usize,
                offsets.get(1).and_then(Json::as_u64).unwrap_or(0) as usize,
            );
            tensors.insert(name, (dtype, shape, a, b - a));
        }
        Ok(Self {
            file,
            base: 8 + n as u64,
            tensors,
        })
    }

    fn has(&self, name: &str) -> bool {
        self.tensors.contains_key(name)
    }

    /// Raw BF16 bytes of `name`, checking its shape.
    fn bf16(&mut self, name: &str, shape: &[usize]) -> Result<Vec<u8>> {
        let (dtype, found, offset, len) =
            self.tensors
                .get(name)
                .cloned()
                .ok_or_else(|| Error::Weight {
                    name: name.into(),
                    message: "missing from the draft checkpoint".into(),
                })?;
        if dtype != "BF16" || found != shape {
            return Err(Error::Weight {
                name: name.into(),
                message: format!("expected BF16 {shape:?}, found {dtype} {found:?}"),
            });
        }
        let mut bytes = vec![0u8; len];
        self.file
            .seek(SeekFrom::Start(self.base + offset as u64))
            .map_err(io)?;
        self.file.read_exact(&mut bytes).map_err(io)?;
        Ok(bytes)
    }

    fn f32s(&mut self, name: &str, shape: &[usize]) -> Result<Vec<f32>> {
        Ok(bf16_to_f32(&self.bf16(name, shape)?))
    }
}

fn io(e: std::io::Error) -> Error {
    Error::Safetensors(e.to_string())
}

fn bf16_to_f32(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(2)
        .map(|b| f32::from_bits(u32::from(u16::from_le_bytes([b[0], b[1]])) << 16))
        .collect()
}

/// Where a drafter's tensors are read from. Names are the HF checkpoint's;
/// GGUF reads map them with `gguf::tensor_name`.
enum Source {
    Safetensors(Safetensors),
    Gguf(GgufFile),
}

/// A GGUF tensor under its GGUF name, its dims checked against the HF
/// (row-major) `shape`.
fn gguf_info(file: &GgufFile, name: &str, shape: &[usize]) -> Result<(String, TensorInfo)> {
    let g = gguf::tensor_name(name).ok_or_else(|| Error::Weight {
        name: name.into(),
        message: "has no GGUF name".into(),
    })?;
    let info = file.tensor(&g).map_err(|_| Error::Weight {
        name: g.clone(),
        message: "missing from the draft GGUF".into(),
    })?;
    let dims: Vec<usize> = info.dimensions.iter().rev().copied().collect();
    if dims != shape {
        return Err(Error::Weight {
            name: g,
            message: format!("expected {shape:?}, found {dims:?}"),
        });
    }
    Ok((g, info.clone()))
}

/// A GGUF tensor's values as F32.
fn gguf_f32s(file: &mut GgufFile, name: &str, shape: &[usize]) -> Result<Vec<f32>> {
    let (g, info) = gguf_info(file, name, shape)?;
    let mut raw = vec![0u8; info.byte_len];
    file.read_tensor_into(&g, &mut raw)?;
    let mut out = vec![0f32; shape.iter().product()];
    gguf::dequantize(info.type_id, &raw, &mut out)?;
    Ok(out)
}

impl Source {
    fn has(&self, name: &str) -> bool {
        match self {
            Self::Safetensors(st) => st.has(name),
            Self::Gguf(file) => gguf::tensor_name(name).is_some_and(|g| file.tensor(&g).is_ok()),
        }
    }

    /// Shape in HF (row-major) order.
    fn shape(&self, name: &str) -> Result<Vec<usize>> {
        let missing = || Error::Weight {
            name: name.into(),
            message: "missing from the draft checkpoint".into(),
        };
        match self {
            Self::Safetensors(st) => st
                .tensors
                .get(name)
                .map(|t| t.1.clone())
                .ok_or_else(missing),
            Self::Gguf(file) => {
                let g = gguf::tensor_name(name).ok_or_else(missing)?;
                let info = file.tensor(&g).map_err(|_| missing())?;
                Ok(info.dimensions.iter().rev().copied().collect())
            }
        }
    }

    fn f32s(&mut self, name: &str, shape: &[usize]) -> Result<Vec<f32>> {
        match self {
            Self::Safetensors(st) => st.f32s(name, shape),
            Self::Gguf(file) => gguf_f32s(file, name, shape),
        }
    }

    /// BF16 bit patterns of `name` (GGUF tensors in other encodings are
    /// decoded and rounded to BF16).
    fn bf16_words(&mut self, name: &str, shape: &[usize]) -> Result<Vec<u16>> {
        let words = |b: &[u8]| -> Vec<u16> {
            b.chunks_exact(2)
                .map(|b| u16::from_le_bytes([b[0], b[1]]))
                .collect()
        };
        match self {
            Self::Safetensors(st) => Ok(words(&st.bf16(name, shape)?)),
            Self::Gguf(file) => {
                let (g, info) = gguf_info(file, name, shape)?;
                if info.type_id == 30 {
                    let mut raw = vec![0u8; info.byte_len];
                    file.read_tensor_into(&g, &mut raw)?;
                    return Ok(words(&raw));
                }
                Ok(gguf_f32s(file, name, shape)?
                    .into_iter()
                    .map(gguf::to_bf16)
                    .collect())
            }
        }
    }
}

/// Reads draft tensors, quantizing matrices, and counts their device bytes.
struct WeightLoader<'a> {
    d: &'a MetalDevice,
    src: &'a mut Source,
    bytes: usize,
}

impl WeightLoader<'_> {
    fn matrix(&mut self, name: &str, rows: usize, cols: usize, format: Format) -> Result<Matrix> {
        let m = match &mut *self.src {
            Source::Safetensors(st) => {
                Matrix::from_bf16(self.d, rows, cols, &st.bf16(name, &[rows, cols])?, format)?
            }
            Source::Gguf(file) => gguf_weight(self.d, file, name, rows, cols, format)?,
        };
        self.bytes += page(m.byte_size());
        Ok(m)
    }

    fn vector(&mut self, name: &str, len: usize) -> Result<Tensor> {
        self.bytes += page(len * 4);
        Tensor::from_f32(self.d, [len], DType::F32, &self.src.f32s(name, &[len])?)
    }

    fn conv(&mut self, conv: Option<(usize, usize)>, prefix: &str) -> Result<Option<Conv>> {
        let Some((kernel, group)) = conv else {
            return Ok(None);
        };
        let base = format!("{prefix}.base_kernel");
        let hidden = self.src.shape(&base)?.last().copied().unwrap_or(0);
        let base = self.src.f32s(&base, &[2, kernel, hidden])?;
        self.bytes += page(base.len() * 4);
        Ok(Some(Conv {
            base: Tensor::from_f32(self.d, [base.len()], DType::F32, &base)?,
            proj: self.matrix(
                &format!("{prefix}.kernel_projection.weight"),
                2 * kernel * (hidden / group),
                hidden,
                Format::Q8_0,
            )?,
        }))
    }

    /// A BF16 `[rows, cols]` table kept in host memory.
    fn codebook(&mut self, name: &str, rows: usize, cols: usize) -> Result<Vec<u16>> {
        self.src.bf16_words(name, &[rows, cols])
    }
}

/// A GGUF matrix in `format`: its stored blocks when they already are, else
/// decoded and re-quantized row by row.
fn gguf_weight(
    d: &MetalDevice,
    file: &mut GgufFile,
    name: &str,
    rows: usize,
    cols: usize,
    format: Format,
) -> Result<Matrix> {
    let (g, info) = gguf_info(file, name, &[rows, cols])?;
    let stored = matches!(
        (format, info.type_id),
        (Format::F32, 0) | (Format::Q4_0, 2) | (Format::Q8_0, 8)
    );
    if stored && cols.is_multiple_of(256) {
        return gguf_matrix(d, file, &g, 1, rows, cols, false);
    }
    let layout = block_layout(info.type_id)?;
    let row_bytes = cols / layout.elements * layout.bytes;
    let mut raw = vec![0u8; info.byte_len];
    file.read_tensor_into(&g, &mut raw)?;
    // Decode one row up front so the parallel fill below cannot fail.
    gguf::dequantize(info.type_id, &raw[..row_bytes], &mut vec![0f32; cols])?;
    Matrix::from_rows(d, rows, cols, format, |r, row| {
        gguf::dequantize(info.type_id, &raw[r * row_bytes..(r + 1) * row_bytes], row)
            .expect("row decoding checked above");
    })
}

struct Conv {
    /// `[2 sides][kernel][hidden]` F32.
    base: Tensor,
    /// `[2 * kernel * groups, hidden]`.
    proj: Matrix,
}

struct Layer {
    in_norm: Tensor,
    post_norm: Tensor,
    q: Matrix,
    k: Matrix,
    v: Matrix,
    o: Matrix,
    q_norm: Tensor,
    k_norm: Tensor,
    gate: Matrix,
    up: Matrix,
    down: Matrix,
    attn_conv: Option<Conv>,
    mlp_conv: Option<Conv>,
}

struct Markov {
    /// `[vocab, rank]` (row lookup) and the draft-vocabulary prefix of the
    /// `[vocab, rank]` projection.
    w1: Matrix,
    w2_head: Matrix,
    /// Confidence head over `[hidden ; rank]` and its bias.
    confidence: Option<(Tensor, f32)>,
}

struct Selector {
    hidden: Matrix,
    /// Host BF16 codebooks `[vocab][rank]`.
    prev: Vec<u16>,
    next: Vec<u16>,
}

struct Buffers {
    // Context ingestion, `chunk` rows.
    g: Tensor,
    gn: Tensor,
    kx: Tensor,
    vx: Tensor,
    // Block rows.
    x: Tensor,
    hn: Tensor,
    hc: Tensor,
    tmp: Tensor,
    q32: Tensor,
    k32: Tensor,
    v32: Tensor,
    qh: Tensor,
    kh: Tensor,
    vh: Tensor,
    att: Tensor,
    dyn_: Tensor,
    gate: Tensor,
    up: Tensor,
    act: Tensor,
    logits: Tensor,
    candidates: Tensor,
    sel_gate: Tensor,
    ids: Tensor,
    w1p: Tensor,
    bias: Tensor,
    conf: Tensor,
    /// DSpark trees: gathered Markov rows, `[2 * block * (TREE_CANDIDATES + 1) + 1, rank]`.
    lattice: Tensor,
}

/// A drafter's weights, `bytes` of device memory.
struct DraftWeights {
    fc: Matrix,
    hidden_norm: Tensor,
    norm: Tensor,
    layers: Vec<Layer>,
    markov: Option<Markov>,
    selector: Option<Selector>,
    bytes: usize,
}

/// Read and encode the weights of `c` for a target of `target_hidden` and
/// `vocab` (the Markov head keeps the `vocab_limit` prefix).
fn load_weights(
    d: &MetalDevice,
    c: &DraftConfig,
    src: &mut Source,
    target_hidden: usize,
    vocab: usize,
    vocab_limit: usize,
    precision: DraftQuant,
) -> Result<DraftWeights> {
    let quant = match precision {
        DraftQuant::Q8_0 => Format::Q8_0,
        DraftQuant::Q4_0 => Format::Q4_0,
    };
    let (h, hd) = (c.hidden, c.head_dim);
    let mut l = WeightLoader { d, src, bytes: 0 };
    let fc = l.matrix("fc.weight", h, c.target_layers.len() * target_hidden, quant)?;
    let hidden_norm = l.vector("hidden_norm.weight", h)?;
    let norm = l.vector("norm.weight", h)?;
    let mut layers = Vec::with_capacity(c.layers);
    for i in 0..c.layers {
        let n = |s: &str| format!("layers.{i}.{s}");
        let q_rows = c.heads * hd;
        let kv_rows = c.kv_heads * hd;
        layers.push(Layer {
            in_norm: l.vector(&n("input_layernorm.weight"), h)?,
            post_norm: l.vector(&n("post_attention_layernorm.weight"), h)?,
            q: l.matrix(&n("self_attn.q_proj.weight"), q_rows, h, quant)?,
            k: l.matrix(&n("self_attn.k_proj.weight"), kv_rows, h, quant)?,
            v: l.matrix(&n("self_attn.v_proj.weight"), kv_rows, h, quant)?,
            o: l.matrix(&n("self_attn.o_proj.weight"), h, q_rows, quant)?,
            q_norm: l.vector(&n("self_attn.q_norm.weight"), hd)?,
            k_norm: l.vector(&n("self_attn.k_norm.weight"), hd)?,
            gate: l.matrix(&n("mlp.gate_proj.weight"), c.ffn, h, quant)?,
            up: l.matrix(&n("mlp.up_proj.weight"), c.ffn, h, quant)?,
            down: l.matrix(&n("mlp.down_proj.weight"), h, c.ffn, quant)?,
            attn_conv: l.conv(c.conv, &n("attention_conv"))?,
            mlp_conv: l.conv(c.conv, &n("mlp_conv"))?,
        });
    }
    let markov = match c.kind {
        Kind::DSpark { rank, confidence } => Some(Markov {
            w1: l.matrix("markov_head.markov_w1.weight", vocab, rank, Format::Q8_0)?,
            w2_head: l
                .matrix("markov_head.markov_w2.weight", vocab, rank, Format::Q8_0)?
                .prefix_rows(vocab_limit),
            confidence: if confidence && l.src.has("confidence_head.proj.weight") {
                let w = l.src.f32s("confidence_head.proj.weight", &[1, h + rank])?;
                let b = l.src.f32s("confidence_head.proj.bias", &[1])?[0];
                Some((Tensor::from_f32(d, [h + rank], DType::F32, &w)?, b))
            } else {
                None
            },
        }),
        _ => None,
    };
    let selector = match c.kind {
        Kind::DFlash2 { rank, .. } => Some(Selector {
            hidden: l.matrix(
                "candidate_selector.hidden_projection.weight",
                rank,
                h,
                Format::Q8_0,
            )?,
            prev: l.codebook("candidate_selector.predecessor_codebook", vocab, rank)?,
            next: l.codebook("candidate_selector.successor_codebook", vocab, rank)?,
        }),
        _ => None,
    };
    Ok(DraftWeights {
        fc,
        hidden_norm,
        norm,
        layers,
        markov,
        selector,
        bytes: l.bytes,
    })
}

pub struct DraftModel {
    pub config: DraftConfig,
    name: String,
    fc: Matrix,
    hidden_norm: Tensor,
    norm: Tensor,
    layers: Vec<Layer>,
    markov: Option<Markov>,
    selector: Option<Selector>,
    inv_freq: Tensor,
    ring: usize,
    window: usize,
    kc: Vec<Tensor>,
    vc: Vec<Tensor>,
    tags: Tensor,
    b: Buffers,
    max_drafts: usize,
    confidence_min: f32,
    weight_bytes: usize,
    /// Draft LM head (a prefix of the target's) and its vocabulary.
    head: Matrix,
    vocab: usize,
    /// How lattice trees rank their nodes (`Drafter::set_calibration`).
    calibration: Calibration,
}

impl DraftModel {
    /// Load a drafter for `target`: a checkpoint directory (`config.json` +
    /// `model.safetensors`, or its HF cache entry) or a DFlash GGUF.
    /// `max_drafts` caps the block (default: the trained size).
    pub fn load(
        d: &MetalDevice,
        path: impl AsRef<Path>,
        target: &HybridModel,
        capacity: usize,
        max_drafts: Option<usize>,
        options: DraftOptions,
    ) -> Result<Self> {
        let (mut config, mut src, label) = open(path.as_ref())?;
        let options = kflash_defaults(&config, options);
        widen_block(&mut config, options.block)?;
        let t = &target.config;
        check_target(&config, t)?;
        let vocab = t.vocab;
        let vocab_limit = options.vocab.map_or(vocab, |v| v.clamp(1024, vocab));
        let DraftWeights {
            fc,
            hidden_norm,
            norm,
            layers,
            markov,
            selector,
            bytes: weight_bytes,
        } = load_weights(
            d,
            &config,
            &mut src,
            t.hidden,
            vocab,
            vocab_limit,
            options.quant,
        )?;
        let c = &config;
        let (h, hd) = (c.hidden, c.head_dim);
        let window = c.window.unwrap_or(options.context_cap.min(capacity));
        let ring = window.min(capacity).max(1);
        let block = c.block_size;
        let max_drafts = max_drafts
            .unwrap_or(c.max_drafts())
            .clamp(1, c.max_drafts());
        let f = |dims: &[usize]| Tensor::zeros_resident(d, dims, DType::F32);
        let hf = |dims: &[usize]| Tensor::zeros_resident(d, dims, DType::F16);
        let chunk = target.chunk();
        let kvw = c.kv_heads * hd;
        let groups = c.conv.map_or(1, |(k, g)| 2 * k * (h / g));
        let rank = match c.kind {
            Kind::DSpark { rank, .. } | Kind::DFlash2 { rank, .. } => rank,
            Kind::DFlash => 1,
        };
        let b = Buffers {
            g: f(&[chunk, h])?,
            gn: f(&[chunk, h])?,
            kx: f(&[chunk, kvw])?,
            vx: f(&[chunk, kvw])?,
            x: f(&[block, h])?,
            hn: f(&[block, h])?,
            hc: f(&[block, h])?,
            tmp: f(&[block, h])?,
            q32: f(&[block, c.heads * hd])?,
            k32: f(&[block, kvw])?,
            v32: f(&[block, kvw])?,
            qh: hf(&[block, c.heads * hd])?,
            kh: hf(&[block, kvw])?,
            vh: hf(&[block, kvw])?,
            att: f(&[block, c.heads * hd])?,
            dyn_: f(&[block, groups])?,
            gate: f(&[block, c.ffn])?,
            up: f(&[block, c.ffn])?,
            act: f(&[block, c.ffn])?,
            logits: f(&[block, vocab])?,
            candidates: f(&[block, vocab.div_ceil(1024) * 64])?,
            sel_gate: f(&[block, rank])?,
            ids: f(&[block + 1])?,
            w1p: f(&[block, rank])?,
            bias: f(&[1, vocab])?,
            conf: f(&[block])?,
            lattice: f(&[2 * block * (TREE_CANDIDATES + 1) + 1, rank])?,
        };
        let mut kc = Vec::new();
        let mut vc = Vec::new();
        for _ in 0..c.layers {
            kc.push(hf(&[ring, kvw])?);
            vc.push(hf(&[ring, kvw])?);
        }
        let tags = Tensor::from_le_bytes(d, [ring], DType::F32, &vec![0xFF; ring * 4])?;
        let name = format!(
            "{} ({})",
            match c.kind {
                Kind::DFlash => "dflash",
                Kind::DFlash2 { .. } => "dflash2",
                Kind::DSpark { .. } => "dspark",
            },
            label
        );
        Ok(Self {
            inv_freq: Tensor::from_f32(d, [c.inv_freq.len()], DType::F32, &c.inv_freq)?,
            config,
            name,
            fc,
            hidden_norm,
            norm,
            layers,
            markov,
            selector,
            ring,
            window,
            kc,
            vc,
            tags,
            b,
            max_drafts,
            confidence_min: options.confidence_min,
            weight_bytes,
            head: target.weights.output.prefix_rows(vocab_limit),
            vocab: vocab_limit,
            calibration: Calibration::default(),
        })
    }

    pub fn weight_bytes(&self) -> usize {
        self.weight_bytes
    }

    /// Context slots (and so the attention window) of the draft.
    pub fn window(&self) -> usize {
        self.window
    }

    fn rmsnorm(&self, d: &MetalDevice, x: &Tensor, w: &Tensor, y: &Tensor, m: usize) -> Result<()> {
        d.dispatch_hybrid(
            "h_rmsnorm",
            &[x.binding(), w.binding()],
            &[y.binding()],
            &Params::default()
                .u(self.config.hidden)?
                .f(self.config.eps)
                .u(m)?
                .0,
            [m, 1, 1],
            [256, 1, 1],
            0,
        )
    }

    fn add(&self, d: &MetalDevice, x: &Tensor, r: &Tensor, n: usize) -> Result<()> {
        d.dispatch_hybrid(
            "h_accumulate",
            &[r.binding()],
            &[x.binding()],
            &Params::default().u(n)?.0,
            [n.div_ceil(256), 1, 1],
            [256, 1, 1],
            0,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn prep(
        &self,
        d: &MetalDevice,
        x: &Tensor,
        norm: &Tensor,
        dst: &Tensor,
        heads: usize,
        rows: usize,
        position: usize,
        ring: usize,
        scale: f32,
    ) -> Result<()> {
        d.dispatch_hybrid(
            "d_qk_prep",
            &[x.binding(), norm.binding(), self.inv_freq.binding()],
            &[dst.binding(), self.tags.binding()],
            &Params::default()
                .u(rows)?
                .u(heads)?
                .u(self.config.head_dim)?
                .f(self.config.eps)
                .u(position)?
                .u(ring)?
                .f(scale)
                .0,
            [heads, rows, 1],
            [self.config.head_dim, 1, 1],
            0,
        )
    }

    fn store_v(
        &self,
        d: &MetalDevice,
        v: &Tensor,
        dst: &Tensor,
        rows: usize,
        position: usize,
        ring: usize,
    ) -> Result<()> {
        let c = &self.config;
        let n = rows * c.kv_heads * c.head_dim;
        d.dispatch_hybrid(
            "d_store_v",
            &[v.binding()],
            &[dst.binding()],
            &Params::default()
                .u(rows)?
                .u(c.kv_heads)?
                .u(c.head_dim)?
                .f(c.eps)
                .u(position)?
                .u(ring)?
                .f(1.)
                .0,
            [n.div_ceil(256), 1, 1],
            [256, 1, 1],
            0,
        )
    }

    /// DFlash2 conv of `m` rows of `x` into `y` with coefficients in `dyn_`.
    fn conv(
        &self,
        d: &MetalDevice,
        conv: &Conv,
        x: &Tensor,
        y: &Tensor,
        m: usize,
        side: usize,
    ) -> Result<()> {
        let (kernel, group) = self.config.conv.expect("conv layers imply a conv config");
        let n = m * self.config.hidden;
        d.dispatch_hybrid(
            "d_dyn_conv",
            &[x.binding(), self.b.dyn_.binding(), conv.base.binding()],
            &[y.binding()],
            &Params::default()
                .u(m)?
                .u(self.config.hidden)?
                .u(kernel)?
                .u(group)?
                .u(side)?
                .0,
            [n.div_ceil(256), 1, 1],
            [256, 1, 1],
            0,
        )
    }

    /// The block forward: `m` rows at `pos..`, leaving normalized outputs in
    /// `b.hn` and logits of rows `from..m` in `b.logits`.
    fn block(
        &self,
        d: &MetalDevice,
        t: &HybridModel,
        tokens: &[u32],
        pos: usize,
        from: usize,
    ) -> Result<()> {
        let c = &self.config;
        let b = &self.b;
        let m = tokens.len();
        let (h, hd) = (c.hidden, c.head_dim);
        let rows = |x: &Tensor, width: usize| x.view(0, [m, width]);
        let ids = Tensor::from_le_bytes(
            d,
            [m],
            DType::F32,
            &tokens
                .iter()
                .flat_map(|t| t.to_le_bytes())
                .collect::<Vec<_>>(),
        )?;
        let x = rows(&b.x, h)?;
        let hn = rows(&b.hn, h)?;
        let hc = rows(&b.hc, h)?;
        let tmp = rows(&b.tmp, h)?;
        let q32 = rows(&b.q32, c.heads * hd)?;
        let k32 = rows(&b.k32, c.kv_heads * hd)?;
        let v32 = rows(&b.v32, c.kv_heads * hd)?;
        let att = rows(&b.att, c.heads * hd)?;
        let gate = rows(&b.gate, c.ffn)?;
        let up = rows(&b.up, c.ffn)?;
        let act = rows(&b.act, c.ffn)?;
        HybridModel::get_rows(d, &t.weights.embedding, &ids, &x)?;
        for (li, layer) in self.layers.iter().enumerate() {
            self.rmsnorm(d, &x, &layer.in_norm, &hn, m)?;
            let input = match &layer.attn_conv {
                Some(conv) => {
                    let dyn_ = b.dyn_.view(0, [m, conv.proj.rows])?;
                    HybridModel::project(d, &conv.proj, &hn, m, &dyn_)?;
                    self.conv(d, conv, &hn, &hc, m, 0)?;
                    &hc
                }
                None => &hn,
            };
            HybridModel::project(d, &layer.q, input, m, &q32)?;
            HybridModel::project(d, &layer.k, input, m, &k32)?;
            HybridModel::project(d, &layer.v, input, m, &v32)?;
            let q_scale = c.rope_scale / (hd as f32).sqrt();
            self.prep(d, &q32, &layer.q_norm, &b.qh, c.heads, m, pos, 0, q_scale)?;
            self.prep(
                d,
                &k32,
                &layer.k_norm,
                &b.kh,
                c.kv_heads,
                m,
                pos,
                0,
                c.rope_scale,
            )?;
            self.store_v(d, &v32, &b.vh, m, pos, 0)?;
            d.dispatch_hybrid(
                "d_attention",
                &[
                    b.qh.binding(),
                    self.kc[li].binding(),
                    self.vc[li].binding(),
                    self.tags.binding(),
                    b.kh.binding(),
                    b.vh.binding(),
                ],
                &[att.binding()],
                &Params::default()
                    .u(m)?
                    .u(c.heads)?
                    .u(c.kv_heads)?
                    .u(hd)?
                    .u(pos)?
                    .u(self.ring)?
                    .u(self.window)?
                    .0,
                [c.heads, m, 1],
                [128, 1, 1],
                0,
            )?;
            HybridModel::project(d, &layer.o, &att, m, &tmp)?;
            if let Some(conv) = &layer.attn_conv {
                self.conv(d, conv, &tmp, &hc, m, 1)?;
                self.add(d, &x, &hc, m * h)?;
            } else {
                self.add(d, &x, &tmp, m * h)?;
            }
            self.rmsnorm(d, &x, &layer.post_norm, &hn, m)?;
            let input = match &layer.mlp_conv {
                Some(conv) => {
                    let dyn_ = b.dyn_.view(0, [m, conv.proj.rows])?;
                    HybridModel::project(d, &conv.proj, &hn, m, &dyn_)?;
                    self.conv(d, conv, &hn, &hc, m, 0)?;
                    &hc
                }
                None => &hn,
            };
            HybridModel::project(d, &layer.gate, input, m, &gate)?;
            HybridModel::project(d, &layer.up, input, m, &up)?;
            d.dispatch_hybrid(
                "h_swiglu",
                &[gate.binding(), up.binding()],
                &[act.binding()],
                &Params::default().u(m * c.ffn)?.0,
                [(m * c.ffn).div_ceil(256), 1, 1],
                [256, 1, 1],
                0,
            )?;
            HybridModel::project(d, &layer.down, &act, m, &tmp)?;
            if let Some(conv) = &layer.mlp_conv {
                self.conv(d, conv, &tmp, &hc, m, 1)?;
                self.add(d, &x, &hc, m * h)?;
            } else {
                self.add(d, &x, &tmp, m * h)?;
            }
        }
        self.rmsnorm(d, &x, &self.norm, &hn, m)?;
        let n = m - from;
        let out = hn.view(from * h, [n, h])?;
        HybridModel::project(d, &self.head, &out, n, &b.logits.view(0, [n, self.vocab])?)
    }

    /// Top (logit, id) pairs per 1,024-token block of logit rows `0..rows`.
    fn top_blocks(&self, d: &MetalDevice, rows: usize) -> Result<()> {
        d.dispatch_hybrid(
            "h_topk_blocks",
            &[self.b.logits.binding()],
            &[self.b.candidates.binding()],
            &Params::default().u(self.vocab)?.0,
            [self.vocab.div_ceil(1024), rows, 1],
            [256, 1, 1],
            0,
        )
    }

    /// The best `keep` (token, logit) pairs of candidate rows `0..rows`,
    /// best first (ties to lower ids).
    fn read_candidates(&self, rows: usize, keep: usize) -> Candidates {
        let width = self.vocab.div_ceil(1024) * 64;
        self.b.candidates.to_f32()[..rows * width]
            .chunks(width)
            .map(|row| {
                let mut all: Vec<(u32, f32)> = row
                    .chunks_exact(2)
                    .filter(|p| p[0].is_finite())
                    .map(|p| (p[1].to_bits(), p[0]))
                    .collect();
                all.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
                all.truncate(keep);
                all
            })
            .collect()
    }

    /// One DFlash2 block: each draft row's top-k candidates and its
    /// selector gate (`[max, rank]`).
    fn dflash2_lattice(
        &self,
        d: &MetalDevice,
        t: &HybridModel,
        pos: usize,
        anchor: u32,
        max: usize,
    ) -> Result<(Candidates, Vec<f32>)> {
        let c = &self.config;
        let Kind::DFlash2 { top_k, rank } = c.kind else {
            unreachable!("a DFlash2 lattice needs DFlash2")
        };
        let mut tokens = vec![c.mask_token; max + 1];
        tokens[0] = anchor;
        let b = &self.b;
        let sel = self.selector.as_ref().expect("DFlash2 has a selector");
        let e = d.execution_with_shared_encoder(true)?;
        self.block(d, t, &tokens, pos, 1)?;
        self.top_blocks(d, max)?;
        let out = b.hn.view(c.hidden, [max, c.hidden])?;
        HybridModel::project(d, &sel.hidden, &out, max, &b.sel_gate.view(0, [max, rank])?)?;
        e.finish()?;
        let mut gates = b.sel_gate.to_f32();
        gates.truncate(max * rank);
        Ok((self.read_candidates(max, top_k), gates))
    }

    /// DFlash2 path scores of `cands` after `prev`, for a row whose selector
    /// gate is `gate`: the drafter's logit plus the selector's bilinear term.
    fn path_scores(&self, prev: u32, cands: &[(u32, f32)], gate: &[f32]) -> Vec<(u32, f32)> {
        let sel = self.selector.as_ref().expect("DFlash2 has a selector");
        let rank = gate.len();
        fn row(book: &[u16], token: u32, rank: usize) -> impl Iterator<Item = f32> + '_ {
            let start = token as usize * rank;
            book[start..start + rank]
                .iter()
                .map(|&v| f32::from_bits(u32::from(v) << 16))
        }
        let cond: Vec<f32> = row(&sel.prev, prev, rank)
            .zip(gate)
            .map(|(p, g)| p * g)
            .collect();
        cands
            .iter()
            .map(|&(token, logit)| {
                let score = row(&sel.next, token, rank)
                    .zip(&cond)
                    .map(|(s, c)| s * c)
                    .sum::<f32>()
                    + logit;
                (token, score)
            })
            .collect()
    }

    /// Greedy DFlash2 path through each row's top-k candidates.
    fn walk(&self, anchor: u32, candidates: &[Vec<(u32, f32)>], gates: &[f32]) -> Vec<u32> {
        let rank = gates.len() / candidates.len().max(1);
        let mut out = Vec::with_capacity(candidates.len());
        let mut prev = anchor;
        for (i, cands) in candidates.iter().enumerate() {
            let gate = &gates[i * rank..(i + 1) * rank];
            let mut best = (f32::NEG_INFINITY, 0u32);
            for (token, score) in self.path_scores(prev, cands, gate) {
                if score > best.0 {
                    best = (score, token);
                }
            }
            out.push(best.1);
            prev = best.1;
        }
        out
    }

    /// One DSpark block and its Markov chain: up to `max` drafts, cut where
    /// the confidence head falls below the threshold. With `lattice`, the
    /// candidate rows first take each slot's top logits before the Markov
    /// bias (and the logit rows keep the chain's biased ones).
    fn dspark_chain(
        &self,
        d: &MetalDevice,
        t: &HybridModel,
        pos: usize,
        anchor: u32,
        max: usize,
        lattice: bool,
    ) -> Result<Vec<u32>> {
        let c = &self.config;
        let Kind::DSpark { rank, .. } = c.kind else {
            unreachable!("a DSpark chain needs DSpark")
        };
        let vocab = self.vocab;
        // Slot i predicts position pos + i + 1.
        let mut tokens = vec![c.mask_token; max];
        tokens[0] = anchor;
        let e = d.execution_with_shared_encoder(true)?;
        self.block(d, t, &tokens, pos, 0)?;
        if lattice {
            self.top_blocks(d, max)?;
        }
        let markov = self.markov.as_ref().expect("DSpark has a Markov head");
        let b = &self.b;
        // Token ids are uint bit patterns: never route them through float ops.
        let anchor_t = Tensor::from_le_bytes(d, [1], DType::F32, &anchor.to_le_bytes())?;
        for i in 0..max {
            let prev = if i == 0 {
                anchor_t.clone()
            } else {
                b.ids.view(i - 1, [1])?
            };
            let w1p = b.w1p.view(i * rank, [1, rank])?;
            HybridModel::get_rows(d, &markov.w1, &prev, &w1p)?;
            HybridModel::project(d, &markov.w2_head, &w1p, 1, &b.bias.view(0, [1, vocab])?)?;
            let row = b.logits.view(i * vocab, [1, vocab])?;
            d.dispatch_hybrid(
                "h_accumulate",
                &[b.bias.binding()],
                &[row.binding()],
                &Params::default().u(vocab)?.0,
                [vocab.div_ceil(256), 1, 1],
                [256, 1, 1],
                0,
            )?;
            d.dispatch_hybrid(
                "h_argmax",
                &[row.binding()],
                &[b.ids.view(i, [1])?.binding()],
                &Params::default().u(vocab)?.0,
                [1, 1, 1],
                [1024, 1, 1],
                0,
            )?;
            if let Some((w, bias)) = &markov.confidence {
                d.dispatch_hybrid(
                    "d_confidence",
                    &[
                        b.hn.view(i * c.hidden, [1, c.hidden])?.binding(),
                        w1p.binding(),
                        w.binding(),
                    ],
                    &[b.conf.view(i, [1])?.binding()],
                    &Params::default().u(c.hidden)?.u(rank)?.f(*bias).0,
                    [1, 1, 1],
                    [256, 1, 1],
                    0,
                )?;
            }
        }
        e.finish()?;
        let ids = b.ids.to_f32();
        let conf = b.conf.to_f32();
        let mut out = Vec::with_capacity(max);
        for i in 0..max {
            if markov.confidence.is_some() && conf[i] < self.confidence_min {
                break;
            }
            out.push(ids[i].to_bits());
        }
        Ok(out)
    }

    /// DSpark's lattice after `dspark_chain(.., lattice = true)` drafted
    /// `chain`: per depth, the slot's top tokens before the Markov bias (and
    /// the chain's pick) with those logits, and each depth's conditional
    /// log-probabilities after a given previous token, base logit plus
    /// `w2[token] . w1[prev]`, normalized over the depth's candidates.
    #[allow(clippy::type_complexity)]
    fn dspark_lattice(
        &self,
        d: &MetalDevice,
        anchor: u32,
        chain: &[u32],
    ) -> Result<(Vec<Vec<u32>>, Box<dyn Fn(usize, u32) -> Vec<f64> + '_>)> {
        let Kind::DSpark { rank, .. } = self.config.kind else {
            unreachable!("a DSpark lattice needs DSpark")
        };
        let markov = self.markov.as_ref().expect("DSpark has a Markov head");
        let depth = chain.len();
        let vocab = self.vocab;
        let mut cands = self.read_candidates(depth, TREE_CANDIDATES);
        // The chain's pick may lie outside the slot's unbiased top tokens:
        // its base logit is its biased one less the bias, found below.
        let mut missing = Vec::new();
        for (i, (row, &pick)) in cands.iter_mut().zip(chain).enumerate() {
            if !row.iter().any(|c| c.0 == pick) {
                let biased = self.b.logits.view(i * vocab + pick as usize, [1])?.to_f32()[0];
                missing.push((i, row.len()));
                row.push((pick, biased));
            }
        }
        // Rows to gather: w1 of every possible previous token (the anchor,
        // then each depth's candidates but the last), w2 of every candidate.
        let prevs: Vec<u32> = std::iter::once(anchor)
            .chain(cands[..depth - 1].iter().flatten().map(|c| c.0))
            .collect();
        let nexts: Vec<u32> = cands.iter().flatten().map(|c| c.0).collect();
        let ids = |v: &[u32]| {
            Tensor::from_le_bytes(
                d,
                [v.len()],
                DType::F32,
                &v.iter().flat_map(|t| t.to_le_bytes()).collect::<Vec<_>>(),
            )
        };
        let w1 = self.b.lattice.view(0, [prevs.len(), rank])?;
        let w2 = self
            .b
            .lattice
            .view(prevs.len() * rank, [nexts.len(), rank])?;
        let e = d.execution_with_shared_encoder(true)?;
        HybridModel::get_rows(d, &markov.w1, &ids(&prevs)?, &w1)?;
        HybridModel::get_rows(d, &markov.w2_head, &ids(&nexts)?, &w2)?;
        e.finish()?;
        let (w1, w2) = (w1.to_f32(), w2.to_f32());
        // Rows by (depth, previous token) and the first w2 row of each depth.
        let mut prev_row = HashMap::new();
        prev_row.insert((0, anchor), 0);
        let mut next_start = Vec::with_capacity(depth);
        let mut n = 0;
        for (i, row) in cands.iter().enumerate() {
            next_start.push(n);
            if i + 1 < depth {
                for c in row {
                    prev_row.entry((i + 1, c.0)).or_insert(1 + n);
                    n += 1;
                }
            } else {
                n += row.len();
            }
        }
        let dot = move |p: usize, q: usize| -> f32 {
            w1[p * rank..(p + 1) * rank]
                .iter()
                .zip(&w2[q * rank..(q + 1) * rank])
                .map(|(a, b)| a * b)
                .sum()
        };
        for (i, j) in missing {
            let prev = if i == 0 { anchor } else { chain[i - 1] };
            cands[i][j].1 -= dot(prev_row[&(i, prev)], next_start[i] + j);
        }
        let tokens = cands
            .iter()
            .map(|r| r.iter().map(|c| c.0).collect())
            .collect();
        let cond = move |i: usize, prev: u32| -> Vec<f64> {
            let p = prev_row[&(i, prev)];
            let scored: Vec<(u32, f32)> = cands[i]
                .iter()
                .enumerate()
                .map(|(j, &(token, base))| (token, base + dot(p, next_start[i] + j)))
                .collect();
            log_softmax(&scored).into_iter().map(|c| c.1).collect()
        };
        Ok((tokens, Box::new(cond)))
    }
}

/// A Hugging Face cache directory resolves to its newest snapshot.
/// A checkpoint directory holding `config.json` (an HF cache entry resolves
/// to its `refs/main` snapshot).
fn resolve(dir: &Path) -> Result<PathBuf> {
    if dir.join("config.json").exists() {
        return Ok(dir.to_owned());
    }
    let snapshots = dir.join("snapshots");
    let main = dir.join("refs").join("main");
    if let Ok(rev) = std::fs::read_to_string(&main) {
        let p = snapshots.join(rev.trim());
        if p.join("config.json").exists() {
            return Ok(p);
        }
    }
    Err(Error::Config(format!(
        "{} is not a draft checkpoint (no config.json) or DFlash GGUF",
        dir.display()
    )))
}

fn is_gguf(path: &Path) -> bool {
    path.is_file()
        && path
            .extension()
            .is_some_and(|e| e.eq_ignore_ascii_case("gguf"))
}

/// A drafter's GGUF header, without the tokenizer's vocabulary and merges.
fn open_gguf(path: &Path) -> Result<GgufFile> {
    let file =
        std::fs::File::open(path).map_err(|e| Error::Gguf(format!("{}: {e}", path.display())))?;
    GgufFile::from_reader_skipping_arrays(file, 4096)
}

/// The drafter at `path`: its config, tensor source and a short label.
fn open(path: &Path) -> Result<(DraftConfig, Source, String)> {
    if is_gguf(path) {
        let file = open_gguf(path)?;
        let config = DraftConfig::from_gguf(&file)?;
        let label = path
            .file_stem()
            .map_or(String::new(), |n| n.to_string_lossy().into_owned());
        return Ok((config, Source::Gguf(file), label));
    }
    let dir = resolve(path)?;
    let json: Json = serde_json::from_slice(&std::fs::read(dir.join("config.json")).map_err(io)?)
        .map_err(|e| Error::Config(format!("draft config.json: {e}")))?;
    let config = DraftConfig::from_json(&json)?;
    let st = Safetensors::open(&dir.join("model.safetensors"))?;
    let label = dir
        .file_name()
        .map_or(String::new(), |n| n.to_string_lossy().into_owned());
    Ok((config, Source::Safetensors(st), label))
}

/// Check a GGUF drafter's tokenizer against the target's metadata (a
/// checkpoint directory carries none, so it passes).
pub fn check_drafter_tokenizer(
    path: impl AsRef<Path>,
    target: &std::collections::BTreeMap<String, MetadataValue>,
) -> Result<()> {
    let path = path.as_ref();
    if !is_gguf(path) {
        return Ok(());
    }
    check_tokenizer(open_gguf(path)?.metadata(), target)
}

fn check_target(c: &DraftConfig, t: &HybridConfig) -> Result<()> {
    if c.hidden != t.hidden {
        return Err(Error::Config(format!(
            "draft hidden size {} differs from the target's {}",
            c.hidden, t.hidden
        )));
    }
    if let Some(&bad) = c.target_layers.iter().find(|&&l| l >= t.layers) {
        return Err(Error::Config(format!(
            "draft reads target layer {bad} of {}",
            t.layers
        )));
    }
    if c.mask_token as usize >= t.vocab {
        return Err(Error::Config(
            "mask token beyond the target vocabulary".into(),
        ));
    }
    Ok(())
}

/// `DraftOptions::kflash`'s defaults for a DFlash2 drafter.
fn kflash_defaults(c: &DraftConfig, mut options: DraftOptions) -> DraftOptions {
    if options.kflash && matches!(c.kind, Kind::DFlash2 { .. }) {
        options.block.get_or_insert(c.block_size.max(16));
        options.vocab.get_or_insert(98_304);
    }
    options
}

/// Widen a DFlash or DFlash2 drafting block to `block` rows. Mask slots
/// carry no per-position weights (positions enter through RoPE), so a wider
/// block is the same forward over more rows; DSpark's Markov chain and
/// confidence head are trained on its block and keep it.
fn widen_block(c: &mut DraftConfig, block: Option<usize>) -> Result<()> {
    let Some(block) = block.filter(|&b| b != c.block_size) else {
        return Ok(());
    };
    if matches!(c.kind, Kind::DSpark { .. }) {
        return Err(Error::Config("DSpark keeps its trained block".into()));
    }
    if !(c.block_size..=32).contains(&block) {
        return Err(Error::Config(format!(
            "drafting block {block}: {}..=32 rows",
            c.block_size
        )));
    }
    c.block_size = block;
    Ok(())
}

/// Bytes a `DraftModel` allocates beyond its weights: the context ring and
/// the block and ingestion buffers.
fn state_bytes(c: &DraftConfig, t: &HybridConfig, ring: usize, chunk: usize) -> usize {
    let kvw = c.kv_heads * c.head_dim;
    let block = c.block_size;
    let f = |n: usize| page(n * 4);
    let groups = c.conv.map_or(1, |(k, g)| 2 * k * (c.hidden / g));
    let rank = match c.kind {
        Kind::DSpark { rank, .. } | Kind::DFlash2 { rank, .. } => rank,
        Kind::DFlash => 1,
    };
    c.layers * 2 * page(ring * kvw * 2)
        + page(ring * 4)
        + page(c.inv_freq.len() * 4)
        + 2 * f(chunk * c.hidden)
        + 2 * f(chunk * kvw)
        + 4 * f(block * c.hidden)
        + 2 * f(block * c.heads * c.head_dim)
        + 2 * f(block * kvw)
        + page(block * c.heads * c.head_dim * 2)
        + 2 * page(block * kvw * 2)
        + f(block * groups)
        + 3 * f(block * c.ffn)
        + f(block * t.vocab)
        + f(block * t.vocab.div_ceil(1024) * 64)
        + 2 * f(block * rank)
        + f(block + 1)
        + f(t.vocab)
        + f(block)
        + f((2 * block * (TREE_CANDIDATES + 1) + 1) * rank)
}

impl DraftModel {
    /// Device bytes a drafter will take for `target` without loading it:
    /// (weights, buffers and context ring). Context rings of drafts without
    /// a window are sized for `options.context_cap`.
    pub fn memory(
        path: impl AsRef<Path>,
        target: &HybridConfig,
        chunk: usize,
        options: DraftOptions,
    ) -> Result<(DraftConfig, usize, usize)> {
        let (mut config, src, _) = open(path.as_ref())?;
        let options = kflash_defaults(&config, options);
        widen_block(&mut config, options.block)?;
        check_target(&config, target)?;
        let quantized = |cols: usize| match options.quant {
            DraftQuant::Q8_0 => cols / 32 * 34,
            DraftQuant::Q4_0 => cols / 32 * 18,
        };
        let mut weights = 0;
        let st = match src {
            Source::Safetensors(st) => st,
            Source::Gguf(file) => {
                for (name, info) in file.tensors() {
                    let dims = &info.dimensions;
                    weights += if gguf::host_tensor(name) || gguf::target_tensor(name) {
                        0
                    } else if dims.len() == 2 && dims[1] > 1 {
                        let (rows, cols) = (dims[1], dims[0]);
                        let q8 = name.contains("conv_proj") || name.starts_with("selector_");
                        page(rows * if q8 { cols / 32 * 34 } else { quantized(cols) })
                    } else {
                        page(dims.iter().product::<usize>() * 4)
                    };
                }
                let ring = config.window.unwrap_or(options.context_cap);
                let state = state_bytes(&config, target, ring, chunk);
                return Ok((config, weights, state));
            }
        };
        for (name, (_, shape, _, _)) in &st.tensors {
            let numel: usize = shape.iter().product();
            // Host-side codebooks and scalars, and the target's own
            // embedding / LM head, take no draft device memory.
            let host = name.ends_with("_codebook")
                || name == "confidence_head.proj.bias"
                || name.starts_with("lm_head")
                || name.starts_with("embed_tokens")
                || name == "d2t"
                || name == "t2d";
            weights += if host {
                0
            } else if shape.len() == 2 && shape[0] > 1 {
                let (rows, cols) = (shape[0], shape[1]);
                let q8 = name.starts_with("markov_head")
                    || name.contains("kernel_projection")
                    || name.starts_with("candidate_selector");
                page(rows * if q8 { cols / 32 * 34 } else { quantized(cols) })
            } else {
                page(numel * 4)
            };
        }
        let ring = config.window.unwrap_or(options.context_cap);
        let state = state_bytes(&config, target, ring, chunk);
        Ok((config, weights, state))
    }
}

impl Drafter for DraftModel {
    fn geometry(&self) -> SpecGeometry {
        SpecGeometry {
            rows: self.max_drafts + 1,
            aux_layers: self.config.target_layers.clone(),
            final_hidden: false,
        }
    }

    fn max_drafts(&self) -> usize {
        self.max_drafts
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn ingest(
        &mut self,
        d: &MetalDevice,
        t: &HybridModel,
        pos: usize,
        tokens: &[u32],
    ) -> Result<()> {
        let c = &self.config;
        let r = tokens.len();
        let spec = t.spec.borrow();
        let features = spec
            .as_ref()
            .and_then(|s| s.features.as_ref())
            .ok_or_else(|| Error::Parameter("target does not capture draft features".into()))?;
        let fw = c.target_layers.len() * t.config.hidden;
        let (h, kvw) = (c.hidden, c.kv_heads * c.head_dim);
        // Only the last `ring` rows can matter to a future draft.
        let skip = r.saturating_sub(self.ring);
        let (r, pos) = (r - skip, pos + skip);
        let e = d.execution_with_shared_encoder(true)?;
        let f = features.view(skip * fw, [r, fw])?;
        let g = self.b.g.view(0, [r, h])?;
        let gn = self.b.gn.view(0, [r, h])?;
        let kx = self.b.kx.view(0, [r, kvw])?;
        let vx = self.b.vx.view(0, [r, kvw])?;
        HybridModel::project(d, &self.fc, &f, r, &g)?;
        self.rmsnorm(d, &g, &self.hidden_norm, &gn, r)?;
        for (i, layer) in self.layers.iter().enumerate() {
            HybridModel::project(d, &layer.k, &gn, r, &kx)?;
            HybridModel::project(d, &layer.v, &gn, r, &vx)?;
            self.prep(
                d,
                &kx,
                &layer.k_norm,
                &self.kc[i],
                c.kv_heads,
                r,
                pos,
                self.ring,
                c.rope_scale,
            )?;
            self.store_v(d, &vx, &self.vc[i], r, pos, self.ring)?;
        }
        e.finish()
    }

    fn draft(
        &mut self,
        d: &MetalDevice,
        t: &HybridModel,
        pos: usize,
        anchor: u32,
        max: usize,
    ) -> Result<Vec<u32>> {
        let c = &self.config;
        let max = max.min(self.max_drafts);
        if max == 0 {
            return Ok(Vec::new());
        }
        match c.kind {
            Kind::DSpark { .. } => self.dspark_chain(d, t, pos, anchor, max, false),
            Kind::DFlash => {
                let mut tokens = vec![c.mask_token; max + 1];
                tokens[0] = anchor;
                let e = d.execution_with_shared_encoder(true)?;
                self.block(d, t, &tokens, pos, 1)?;
                d.dispatch_hybrid(
                    "h_argmax",
                    &[self.b.logits.binding()],
                    &[self.b.ids.binding()],
                    &Params::default().u(self.vocab)?.0,
                    [max, 1, 1],
                    [1024, 1, 1],
                    0,
                )?;
                e.finish()?;
                Ok(self.b.ids.to_f32()[..max]
                    .iter()
                    .map(|v| v.to_bits())
                    .collect())
            }
            Kind::DFlash2 { .. } => {
                let (candidates, gates) = self.dflash2_lattice(d, t, pos, anchor, max)?;
                Ok(self.walk(anchor, &candidates, &gates))
            }
        }
    }

    /// A tree of up to `budget` nodes, at most `max` deep, from the one
    /// block pass a chain takes: each depth's top candidates, joined by
    /// DFlash2's selector or DSpark's Markov bias (independent for DFlash),
    /// grown best first.
    fn draft_tree(
        &mut self,
        d: &MetalDevice,
        t: &HybridModel,
        pos: usize,
        anchor: u32,
        max: usize,
        budget: usize,
    ) -> Result<DraftTree> {
        let c = &self.config;
        let max = max.min(self.max_drafts);
        if max == 0 || budget == 0 {
            return Ok(DraftTree::default());
        }
        let tokens = |cands: &[Vec<(u32, f32)>]| -> Vec<Vec<u32>> {
            cands
                .iter()
                .map(|r| r.iter().map(|c| c.0).collect())
                .collect()
        };
        let probs = |scored: &[(u32, f32)]| -> Vec<f64> {
            log_softmax(scored).into_iter().map(|c| c.1).collect()
        };
        match c.kind {
            Kind::DSpark { .. } => {
                // The chain's confidence cut bounds the depth.
                let chain = self.dspark_chain(d, t, pos, anchor, max, true)?;
                if chain.is_empty() {
                    return Ok(DraftTree::default());
                }
                let (lattice, cond) = self.dspark_lattice(d, anchor, &chain)?;
                Ok(lattice_best_first(
                    anchor,
                    &lattice,
                    budget,
                    &self.calibration,
                    cond,
                ))
            }
            Kind::DFlash => {
                let mut ids = vec![c.mask_token; max + 1];
                ids[0] = anchor;
                let e = d.execution_with_shared_encoder(true)?;
                self.block(d, t, &ids, pos, 1)?;
                self.top_blocks(d, max)?;
                e.finish()?;
                let cands = self.read_candidates(max, TREE_CANDIDATES);
                let marginals: Vec<Vec<f64>> = cands.iter().map(|r| probs(r)).collect();
                Ok(lattice_best_first(
                    anchor,
                    &tokens(&cands),
                    budget,
                    &self.calibration,
                    |i, _| marginals[i].clone(),
                ))
            }
            Kind::DFlash2 { rank, .. } => {
                let (cands, gates) = self.dflash2_lattice(d, t, pos, anchor, max)?;
                let mut memo: HashMap<(usize, u32), Vec<f64>> = HashMap::new();
                Ok(lattice_best_first(
                    anchor,
                    &tokens(&cands),
                    budget,
                    &self.calibration,
                    |i, prev| {
                        memo.entry((i, prev))
                            .or_insert_with(|| {
                                let gate = &gates[i * rank..(i + 1) * rank];
                                probs(&self.path_scores(prev, &cands[i], gate))
                            })
                            .clone()
                    },
                ))
            }
        }
    }

    fn set_calibration(&mut self, calibration: &Calibration) {
        self.calibration.clone_from(calibration);
    }

    fn rewind(&mut self, _d: &MetalDevice, _t: &HybridModel, _len: usize) -> Result<()> {
        // Slots are tagged with their positions and a query only reads tags
        // below its block, so stale slots past `len` are never read.
        Ok(())
    }

    fn adopt(&mut self, _d: &MetalDevice, _t: &HybridModel, _tokens: &[u32]) -> Result<()> {
        // The loaded positions have no slots of their own: drafts read
        // whatever earlier tags remain until new rows are ingested.
        Ok(())
    }
}
