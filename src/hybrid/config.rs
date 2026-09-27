//! Qwen3.5-family hybrid geometry read from GGUF metadata.
#![forbid(unsafe_code)]
use crate::{
    Error, Result,
    loader::gguf::{GgufFile, MetadataValue},
};
use std::collections::BTreeMap;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Variant {
    /// `qwen35`: every layer has a dense SwiGLU feed-forward block.
    Dense,
    /// `qwen35moe`: routed experts plus a sigmoid-gated shared expert.
    Moe,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MoeConfig {
    pub experts: usize,
    pub experts_used: usize,
    pub expert_ffn: usize,
    pub shared_ffn: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub struct HybridConfig {
    pub variant: Variant,
    pub name: String,
    pub vocab: usize,
    pub hidden: usize,
    /// Trunk layers executed per token. The trailing NextN/MTP block is excluded.
    pub layers: usize,
    /// Layers with full attention (the rest are Gated DeltaNet).
    pub attention_layers: Vec<bool>,
    pub heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub rope_dims: usize,
    pub rope_theta: f32,
    pub eps: f32,
    /// Dense FFN width (unused by MoE layers).
    pub ffn: usize,
    pub conv_kernel: usize,
    /// Per-head state width of the delta rule (both K and V head dims).
    pub ssm_head_dim: usize,
    pub ssm_k_heads: usize,
    pub ssm_v_heads: usize,
    pub moe: Option<MoeConfig>,
    pub trained_context: usize,
}

impl HybridConfig {
    pub fn from_gguf(file: &GgufFile) -> Result<Self> {
        let md = file.metadata();
        let arch = string(md, "general.architecture")?;
        let variant = match arch {
            "qwen35" => Variant::Dense,
            "qwen35moe" => Variant::Moe,
            other => {
                return Err(Error::Config(format!(
                    "GGUF architecture {other:?} is not a Qwen3.5 hybrid"
                )));
            }
        };
        let key = |suffix: &str| format!("{arch}.{suffix}");
        let blocks = uint(md, &key("block_count"))?;
        let nextn = match md.get(&key("nextn_predict_layers")) {
            Some(_) => uint(md, &key("nextn_predict_layers"))?,
            None => 0,
        };
        let layers = blocks
            .checked_sub(nextn)
            .filter(|&l| l > 0)
            .ok_or_else(|| Error::Config("no trunk layers".into()))?;
        let attention_layers = match md.get(&key("attention.recurrent_layers")) {
            Some(MetadataValue::Array { values, .. }) => values
                .iter()
                .take(layers)
                .map(|v| match v {
                    MetadataValue::Bool(recurrent) => Ok(!recurrent),
                    _ => Err(Error::Config("recurrent_layers must be booleans".into())),
                })
                .collect::<Result<Vec<_>>>()?,
            _ => {
                let interval = match md.get(&key("full_attention_interval")) {
                    Some(_) => uint(md, &key("full_attention_interval"))?,
                    None => 4,
                };
                (0..layers).map(|i| (i + 1) % interval == 0).collect()
            }
        };
        if attention_layers.len() != layers {
            return Err(Error::Config("recurrent layer mask is too short".into()));
        }
        let head_dim = uint(md, &key("attention.key_length"))?;
        if uint(md, &key("attention.value_length"))? != head_dim {
            return Err(Error::Config("K/V head dimensions differ".into()));
        }
        let ssm_head_dim = uint(md, &key("ssm.state_size"))?;
        let ssm_v_heads = uint(md, &key("ssm.time_step_rank"))?;
        let ssm_k_heads = uint(md, &key("ssm.group_count"))?;
        if uint(md, &key("ssm.inner_size"))? != ssm_v_heads * ssm_head_dim {
            return Err(Error::Config(
                "ssm.inner_size must equal time_step_rank * state_size".into(),
            ));
        }
        let moe = match variant {
            Variant::Dense => None,
            Variant::Moe => Some(MoeConfig {
                experts: uint(md, &key("expert_count"))?,
                experts_used: uint(md, &key("expert_used_count"))?,
                expert_ffn: uint(md, &key("expert_feed_forward_length"))?,
                shared_ffn: uint(md, &key("expert_shared_feed_forward_length"))?,
            }),
        };
        let embedding = file.tensor("token_embd.weight")?.logical_matrix_shape()?;
        let config = Self {
            variant,
            name: md
                .get("general.name")
                .and_then(|v| match v {
                    MetadataValue::String(s) => Some(s.clone()),
                    _ => None,
                })
                .unwrap_or_default(),
            vocab: embedding[0],
            hidden: uint(md, &key("embedding_length"))?,
            layers,
            attention_layers,
            heads: uint(md, &key("attention.head_count"))?,
            kv_heads: uint(md, &key("attention.head_count_kv"))?,
            head_dim,
            rope_dims: uint(md, &key("rope.dimension_count"))?,
            rope_theta: float(md, &key("rope.freq_base"))?,
            eps: float(md, &key("attention.layer_norm_rms_epsilon"))?,
            ffn: match variant {
                Variant::Dense => uint(md, &key("feed_forward_length"))?,
                Variant::Moe => 0,
            },
            conv_kernel: uint(md, &key("ssm.conv_kernel"))?,
            ssm_head_dim,
            ssm_k_heads,
            ssm_v_heads,
            moe,
            trained_context: uint(md, &key("context_length"))?,
        };
        config.validate()?;
        Ok(config)
    }

    fn validate(&self) -> Result<()> {
        let bad = |m: &str| Err(Error::Config(format!("unsupported hybrid geometry: {m}")));
        if embedding_mismatch(self) {
            return bad("embedding width");
        }
        if self.head_dim != 256
            || self.rope_dims > self.head_dim
            || !self.rope_dims.is_multiple_of(2)
        {
            return bad("attention head dimension must be 256 with even RoPE width");
        }
        if self.kv_heads == 0 || !self.heads.is_multiple_of(self.kv_heads) {
            return bad("grouped-query head ratio");
        }
        if self.conv_kernel != 4 {
            return bad("convolution kernel must be 4");
        }
        if self.ssm_head_dim != 128
            || self.ssm_k_heads == 0
            || !self.ssm_v_heads.is_multiple_of(self.ssm_k_heads)
        {
            return bad("delta-rule heads must be 128 wide with V heads a multiple of K heads");
        }
        if !self.hidden.is_multiple_of(256)
            || (self.variant == Variant::Dense && !self.ffn.is_multiple_of(256))
        {
            return bad("projection widths must be multiples of 256");
        }
        if let Some(moe) = self.moe
            && (moe.experts == 0
                || moe.experts_used == 0
                || moe.experts_used > moe.experts
                || !moe.expert_ffn.is_multiple_of(256)
                || !moe.shared_ffn.is_multiple_of(256))
        {
            return bad("expert geometry");
        }
        Ok(())
    }

    pub fn is_attention(&self, layer: usize) -> bool {
        self.attention_layers[layer]
    }
    pub fn attention_layer_count(&self) -> usize {
        self.attention_layers.iter().filter(|&&a| a).count()
    }
    pub fn recurrent_layer_count(&self) -> usize {
        self.layers - self.attention_layer_count()
    }
    /// Q and K channels of the delta-rule convolution input.
    pub fn ssm_key_dim(&self) -> usize {
        self.ssm_k_heads * self.ssm_head_dim
    }
    pub fn ssm_value_dim(&self) -> usize {
        self.ssm_v_heads * self.ssm_head_dim
    }
    pub fn conv_channels(&self) -> usize {
        2 * self.ssm_key_dim() + self.ssm_value_dim()
    }
    /// Bytes of K plus V for one cached position across all attention layers (F16).
    pub fn kv_bytes_per_token(&self) -> usize {
        self.attention_layer_count() * 2 * self.kv_heads * self.head_dim * 2
    }
    /// Fixed recurrent state bytes (conv window + delta-rule matrices, F32).
    pub fn recurrent_state_bytes(&self) -> usize {
        let conv = (self.conv_kernel - 1) * self.conv_channels() * 4;
        let ssm = self.ssm_v_heads * self.ssm_head_dim * self.ssm_head_dim * 4;
        self.recurrent_layer_count() * (conv + ssm)
    }
}

fn embedding_mismatch(c: &HybridConfig) -> bool {
    c.hidden == 0 || c.vocab == 0
}

fn string<'a>(md: &'a BTreeMap<String, MetadataValue>, key: &str) -> Result<&'a str> {
    match md.get(key) {
        Some(MetadataValue::String(s)) => Ok(s),
        _ => Err(Error::Config(format!("missing GGUF metadata {key}"))),
    }
}

pub(crate) fn uint(md: &BTreeMap<String, MetadataValue>, key: &str) -> Result<usize> {
    let value = match md.get(key) {
        Some(MetadataValue::Uint32(v)) => u64::from(*v),
        Some(MetadataValue::Uint64(v)) => *v,
        Some(MetadataValue::Int32(v)) if *v >= 0 => *v as u64,
        _ => return Err(Error::Config(format!("missing GGUF metadata {key}"))),
    };
    usize::try_from(value).map_err(|_| Error::Config(format!("{key} exceeds usize")))
}

fn float(md: &BTreeMap<String, MetadataValue>, key: &str) -> Result<f32> {
    match md.get(key) {
        Some(MetadataValue::Float32(v)) if v.is_finite() && *v > 0. => Ok(*v),
        Some(MetadataValue::Float64(v)) if v.is_finite() && *v > 0. => Ok(*v as f32),
        _ => Err(Error::Config(format!(
            "missing or invalid GGUF metadata {key}"
        ))),
    }
}
