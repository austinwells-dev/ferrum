//! DFlash drafters packaged by llama.cpp's converter (`general.architecture`
//! `dflash`): the config from GGUF metadata, HF → GGUF tensor names, and host
//! dequantization for tensors the draft re-encodes.
//!
//! Reference: llama.cpp 7fe450e (`conversion/qwen.py` `DFlashModel`,
//! `src/models/dflash.cpp`, `gguf-py/gguf/tensor_mapping.py`).
#![forbid(unsafe_code)]
use super::{DraftConfig, Kind, yarn};
use crate::{
    Error, Result,
    hybrid::config::uint,
    loader::gguf::{GgufFile, MetadataValue, block_layout},
};
use serde_json::{Value as Json, json};
use std::collections::BTreeMap;

const ARCH: &str = "dflash";

/// Metadata of DFlash variants (DSpark heads, DeepSeek-V4 hyper-connections,
/// output scaling) the draft kernels do not implement.
const UNSUPPORTED_KEYS: &[&str] = &[
    "sample_from_anchor",
    "has_confidence_head",
    "hyper_connection.count",
    "logit_scale",
    "final_logit_softcapping",
    "embedding_scale",
    "attention.scale",
];

/// Tensors of those variants and of Gemma-style post norms, sinks and
/// learned RoPE (prefixes of GGUF names after any `blk.N.`).
const UNSUPPORTED_TENSORS: &[&str] = &[
    "markov_w1",
    "markov_w2",
    "conf_proj",
    "d2t",
    "fc.scale",
    "attn_sinks",
    "post_attention_norm",
    "post_ffw_norm",
    "layer_output_scale",
    "rope_freqs",
    "hc_",
];

type Metadata = BTreeMap<String, MetadataValue>;

fn bad(m: &str) -> Error {
    Error::Config(format!("draft GGUF: {m}"))
}

fn key(k: &str) -> String {
    format!("{ARCH}.{k}")
}

fn opt_uint(md: &Metadata, k: &str) -> Option<usize> {
    md.contains_key(k).then(|| uint(md, k).ok()).flatten()
}

fn opt_float(md: &Metadata, k: &str) -> Option<f64> {
    match md.get(k) {
        Some(MetadataValue::Float32(v)) => Some(f64::from(*v)),
        Some(MetadataValue::Float64(v)) => Some(*v),
        _ => None,
    }
}

fn ints(md: &Metadata, k: &str) -> Option<Vec<i64>> {
    let Some(MetadataValue::Array { values, .. }) = md.get(k) else {
        return None;
    };
    values
        .iter()
        .map(|v| match v {
            MetadataValue::Int32(x) => Some(i64::from(*x)),
            MetadataValue::Uint32(x) => Some(i64::from(*x)),
            MetadataValue::Int64(x) => Some(*x),
            MetadataValue::Uint64(x) => i64::try_from(*x).ok(),
            _ => None,
        })
        .collect()
}

fn bools(md: &Metadata, k: &str) -> Option<Vec<bool>> {
    let Some(MetadataValue::Array { values, .. }) = md.get(k) else {
        return None;
    };
    values
        .iter()
        .map(|v| match v {
            MetadataValue::Bool(b) => Some(*b),
            _ => None,
        })
        .collect()
}

fn text<'a>(md: &'a Metadata, k: &str) -> Option<&'a str> {
    match md.get(k) {
        Some(MetadataValue::String(s)) => Some(s),
        _ => None,
    }
}

/// Whether a GGUF file is a DFlash drafter, from its header alone.
pub fn is_dflash(file: &GgufFile) -> bool {
    text(file.metadata(), "general.architecture") == Some(ARCH)
}

impl DraftConfig {
    /// Read a llama.cpp DFlash / DFlash2 GGUF's metadata (the converter writes
    /// the HF `dflash_config` under `dflash.*`).
    pub fn from_gguf(file: &GgufFile) -> Result<Self> {
        let md = file.metadata();
        match text(md, "general.architecture") {
            Some(ARCH) => {}
            other => {
                return Err(bad(&format!(
                    "architecture {} is not a DFlash drafter",
                    other.unwrap_or("(none)")
                )));
            }
        }
        for k in UNSUPPORTED_KEYS {
            if md.contains_key(&key(k)) {
                return Err(bad(&format!("{} is not supported", key(k))));
            }
        }
        for name in file.tensors().keys() {
            let local = name
                .strip_prefix("blk.")
                .and_then(|r| r.split_once('.'))
                .map_or(name.as_str(), |(_, rest)| rest);
            if UNSUPPORTED_TENSORS.iter().any(|p| local.starts_with(p)) {
                return Err(bad(&format!("tensor {name} is not supported")));
            }
        }
        if let Some(act) = text(md, &key("hidden_activation"))
            && act != "silu"
        {
            return Err(bad(&format!("hidden activation {act:?}")));
        }
        if let Some(MetadataValue::Bool(true)) = md.get(&key("attention.causal")) {
            return Err(bad("causal drafts are not supported"));
        }
        let need = |k: &str| uint(md, &key(k));
        let hidden = need("embedding_length")?;
        let heads = need("attention.head_count")?;
        let kv_heads = need("attention.head_count_kv")?;
        let head_dim = opt_uint(md, &key("attention.key_length")).unwrap_or(hidden / heads);
        if opt_uint(md, &key("attention.value_length")).is_some_and(|v| v != head_dim) {
            return Err(bad("value heads differ from key heads"));
        }
        let layers = need("block_count")?;
        let block_size = need("block_size")?;
        let mask_token = uint(md, "tokenizer.ggml.mask_token_id")? as u32;
        // The converter stores the extraction point (HF layer id + 1).
        let target_layers = ints(md, &key("target_layers"))
            .ok_or_else(|| bad("missing dflash.target_layers"))?
            .into_iter()
            .map(|i| usize::try_from(i - 1).map_err(|_| bad("target layer 0")))
            .collect::<Result<Vec<_>>>()?;
        let kind = match opt_uint(md, &key("selector_rank")) {
            Some(rank) if rank > 0 => Kind::DFlash2 {
                top_k: need("selector_top_k")?,
                rank,
            },
            _ => Kind::DFlash,
        };
        let conv = match kind {
            Kind::DFlash2 { .. } => Some((need("conv_kernel_size")?, need("conv_group_size")?)),
            _ => None,
        };
        let window = match (
            opt_uint(md, &key("attention.sliding_window")),
            bools(md, &key("attention.sliding_window_pattern")),
        ) {
            (Some(w), None) if w > 0 => Some(w),
            (Some(w), Some(p)) if w > 0 && p.iter().all(|&s| s) => Some(w),
            (Some(w), Some(p)) if w > 0 && p.iter().any(|&s| s) => {
                return Err(bad("mixed sliding and full attention layers"));
            }
            _ => None,
        };
        if opt_uint(md, &key("rope.dimension_count")).is_some_and(|n| n != head_dim) {
            return Err(bad("partial rotary drafts are not supported"));
        }
        // M-RoPE targets: the draft ropes on the temporal section only, which
        // the converter writes as [head_dim / 2, 0, 0, 0] (plain RoPE).
        if let Some(s) = ints(md, &key("rope.dimension_sections"))
            && (s.first().copied() != Some(head_dim as i64 / 2) || s[1..].iter().any(|&x| x != 0))
        {
            return Err(bad(&format!("rope sections {s:?}")));
        }
        let theta =
            opt_float(md, &key("rope.freq_base")).ok_or_else(|| bad("missing rope.freq_base"))?;
        let (inv_freq, rope_scale) = match text(md, &key("rope.scaling.type")) {
            None | Some("none") => (
                (0..head_dim / 2)
                    .map(|i| (theta.powf(-2. * i as f64 / head_dim as f64)) as f32)
                    .collect(),
                1.,
            ),
            Some("yarn") => {
                let f = |k: &str| opt_float(md, &key(k));
                let original = opt_uint(md, &key("rope.scaling.original_context_length"));
                let mut rope = json!({});
                for (field, value) in [
                    ("factor", f("rope.scaling.factor")),
                    ("beta_fast", f("rope.scaling.yarn_beta_fast")),
                    ("beta_slow", f("rope.scaling.yarn_beta_slow")),
                    ("attention_factor", f("rope.scaling.attn_factor")),
                    (
                        "original_max_position_embeddings",
                        original.map(|o| o as f64),
                    ),
                ] {
                    if let Some(v) = value {
                        rope[field] = Json::from(v);
                    }
                }
                // With the original length known, the factor key decides
                // (`yarn` derives it from the context length otherwise).
                let max_pos = match original {
                    Some(_) => None,
                    None => opt_uint(md, &key("context_length")),
                };
                yarn(&rope, max_pos, theta, head_dim)?
            }
            Some(other) => return Err(bad(&format!("rope scaling {other:?}"))),
        };
        if !head_dim.is_multiple_of(32)
            || head_dim > 256
            || kv_heads == 0
            || !heads.is_multiple_of(kv_heads)
        {
            return Err(bad("attention geometry"));
        }
        Ok(Self {
            kind,
            hidden,
            layers,
            heads,
            kv_heads,
            head_dim,
            ffn: need("feed_forward_length")?,
            eps: opt_float(md, &key("attention.layer_norm_rms_epsilon")).unwrap_or(1e-6) as f32,
            block_size,
            mask_token,
            target_layers,
            window,
            inv_freq,
            rope_scale,
            conv,
        })
    }
}

/// The GGUF name of an HF draft tensor (as `DraftModel::load` names them).
pub(super) fn tensor_name(hf: &str) -> Option<String> {
    let global = match hf {
        "fc.weight" => Some("fc.weight"),
        "hidden_norm.weight" => Some("enc.output_norm.weight"),
        "norm.weight" => Some("output_norm.weight"),
        "candidate_selector.hidden_projection.weight" => Some("selector_hidden.weight"),
        "candidate_selector.predecessor_codebook" => Some("selector_predecessor.weight"),
        "candidate_selector.successor_codebook" => Some("selector_successor.weight"),
        _ => None,
    };
    if let Some(g) = global {
        return Some(g.into());
    }
    let rest = hf.strip_prefix("layers.")?;
    let (layer, rest) = rest.split_once('.')?;
    let local = match rest {
        "input_layernorm.weight" => "attn_norm.weight",
        "post_attention_layernorm.weight" => "ffn_norm.weight",
        "self_attn.q_proj.weight" => "attn_q.weight",
        "self_attn.k_proj.weight" => "attn_k.weight",
        "self_attn.v_proj.weight" => "attn_v.weight",
        "self_attn.o_proj.weight" => "attn_output.weight",
        "self_attn.q_norm.weight" => "attn_q_norm.weight",
        "self_attn.k_norm.weight" => "attn_k_norm.weight",
        "mlp.gate_proj.weight" => "ffn_gate.weight",
        "mlp.up_proj.weight" => "ffn_up.weight",
        "mlp.down_proj.weight" => "ffn_down.weight",
        "attention_conv.base_kernel" => "attn_conv_base",
        "attention_conv.kernel_projection.weight" => "attn_conv_proj.weight",
        "mlp_conv.base_kernel" => "ffn_conv_base",
        "mlp_conv.kernel_projection.weight" => "ffn_conv_proj.weight",
        _ => return None,
    };
    Some(format!("blk.{layer}.{local}"))
}

/// Whether a GGUF draft tensor stays in host memory (the selector codebooks).
pub(super) fn host_tensor(name: &str) -> bool {
    name.starts_with("selector_predecessor") || name.starts_with("selector_successor")
}

/// Whether a GGUF draft tensor is the draft's own copy of the target's
/// embedding or LM head (the draft uses the target's).
pub(super) fn target_tensor(name: &str) -> bool {
    name.starts_with("token_embd.") || name.starts_with("output.")
}

/// Decode `out.len()` values of GGML `type_id` from `src`.
pub(super) fn dequantize(type_id: u32, src: &[u8], out: &mut [f32]) -> Result<()> {
    let layout = block_layout(type_id)?;
    if !out.len().is_multiple_of(layout.elements)
        || src.len() != out.len() / layout.elements * layout.bytes
    {
        return Err(Error::Shape(format!(
            "{} GGML type {type_id} values from {} bytes",
            out.len(),
            src.len()
        )));
    }
    let half = |b: &[u8]| half::f16::from_le_bytes([b[0], b[1]]).to_f32();
    match type_id {
        0 => {
            for (o, b) in out.iter_mut().zip(src.chunks_exact(4)) {
                *o = f32::from_le_bytes([b[0], b[1], b[2], b[3]]);
            }
        }
        1 => {
            for (o, b) in out.iter_mut().zip(src.chunks_exact(2)) {
                *o = half(b);
            }
        }
        30 => {
            for (o, b) in out.iter_mut().zip(src.chunks_exact(2)) {
                *o = f32::from_bits(u32::from(u16::from_le_bytes([b[0], b[1]])) << 16);
            }
        }
        8 => {
            for (o, b) in out.chunks_exact_mut(32).zip(src.chunks_exact(34)) {
                let d = half(b);
                for (v, &q) in o.iter_mut().zip(&b[2..]) {
                    *v = f32::from(q as i8) * d;
                }
            }
        }
        2 => {
            for (o, b) in out.chunks_exact_mut(32).zip(src.chunks_exact(18)) {
                let d = half(b);
                for j in 0..16 {
                    o[j] = (f32::from(b[2 + j] & 15) - 8.) * d;
                    o[j + 16] = (f32::from(b[2 + j] >> 4) - 8.) * d;
                }
            }
        }
        other => {
            return Err(Error::Gguf(format!(
                "draft tensors of GGML type {other} are not supported"
            )));
        }
    }
    Ok(())
}

/// Round to the nearest BF16 (ties to even), as `torch` does.
pub(super) fn to_bf16(v: f32) -> u16 {
    let bits = v.to_bits();
    if v.is_nan() {
        return ((bits >> 16) | 0x40) as u16;
    }
    ((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16) as u16
}

/// A drafter's tokenizer must be the target's: the draft embeds and reads
/// out through the target's vocabulary.
pub fn check_tokenizer(draft: &Metadata, target: &Metadata) -> Result<()> {
    for k in [
        "tokenizer.ggml.model",
        "tokenizer.ggml.pre",
        "tokenizer.ggml.bos_token_id",
        "tokenizer.ggml.eos_token_id",
    ] {
        if let (Some(a), Some(b)) = (draft.get(k), target.get(k))
            && a != b
        {
            return Err(Error::Config(format!(
                "draft {k} {a:?} differs from the target's {b:?}"
            )));
        }
    }
    Ok(())
}
