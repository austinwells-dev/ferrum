//! Strict local adapter for the inspected dense Qwen2.5 checkpoint contract.
#![forbid(unsafe_code)]
use super::{ModelConfig, Transformer, weights};
use crate::{DType, Error, MetalDevice, Result, loader::Weights};
use serde::Deserialize;
use std::{collections::BTreeMap, path::Path};

pub const REVISION: &str = "7ae557604adf67be50417f59c2c2f167def9a775";
#[derive(Debug, Deserialize)]
pub struct QwenConfig {
    pub architectures: Vec<String>,
    pub model_type: String,
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub rms_norm_eps: f32,
    pub rope_theta: f32,
    pub max_position_embeddings: usize,
    pub tie_word_embeddings: bool,
    pub torch_dtype: String,
    pub hidden_act: String,
    pub use_sliding_window: bool,
    pub bos_token_id: u32,
    pub eos_token_id: u32,
    #[serde(flatten)]
    pub extra: BTreeMap<String, serde_json::Value>,
}
impl QwenConfig {
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let bytes =
            std::fs::read(path).map_err(|e| Error::Config(format!("{}: {e}", path.display())))?;
        Self::from_bytes(&bytes)
    }
    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        serde_json::from_slice(bytes).map_err(|e| Error::Config(format!("config.json: {e}")))
    }
    pub fn convert(&self) -> Result<ModelConfig> {
        let fail = |s: &str| Error::Config(s.into());
        if self.architectures != ["Qwen2ForCausalLM"] || self.model_type != "qwen2" {
            return Err(fail("supported architecture is Qwen2ForCausalLM / qwen2"));
        }
        if self.hidden_act != "silu" || self.use_sliding_window {
            return Err(fail(
                "only silu and full attention (use_sliding_window=false) are supported",
            ));
        }
        for (key, value) in &self.extra {
            let supported = match key.as_str() {
                // Training/serialization metadata, and inactive sliding-window settings.
                "initializer_range"
                | "transformers_version"
                | "_name_or_path"
                | "max_window_layers"
                | "sliding_window" => true,
                "attention_dropout" => value.as_f64() == Some(0.),
                "use_cache" | "attention_bias" => value.as_bool() == Some(true),
                "mlp_bias" => value.as_bool() == Some(false),
                "rope_scaling" => value.is_null(),
                "head_dim" => value.as_u64().is_some_and(|d| {
                    d.checked_mul(self.num_attention_heads as u64) == Some(self.hidden_size as u64)
                }),
                _ => false,
            };
            if !supported {
                return Err(fail(&format!("unsupported config field {key}={value}")));
            }
        }
        if self.torch_dtype != "bfloat16" {
            return Err(fail("Qwen checkpoint storage must be bfloat16"));
        }
        if self.num_attention_heads == 0
            || !self.hidden_size.is_multiple_of(self.num_attention_heads)
        {
            return Err(fail("hidden_size must be divisible by num_attention_heads"));
        }
        if self.bos_token_id as usize >= self.vocab_size
            || self.eos_token_id as usize >= self.vocab_size
        {
            return Err(fail("BOS/EOS ID outside model vocabulary"));
        }
        let c = ModelConfig {
            vocab_size: self.vocab_size,
            hidden_size: self.hidden_size,
            intermediate_size: self.intermediate_size,
            num_layers: self.num_hidden_layers,
            num_attention_heads: self.num_attention_heads,
            num_key_value_heads: self.num_key_value_heads,
            head_dim: self.hidden_size / self.num_attention_heads,
            rms_norm_epsilon: self.rms_norm_eps,
            rope_theta: self.rope_theta,
            max_context_length: self.max_position_embeddings,
            tie_word_embeddings: self.tie_word_embeddings,
            dtype: DType::BF16,
        };
        c.validate()?;
        Ok(c)
    }

    /// Convert the supported mlx-community affine 4-bit, group-64 Qwen config.
    /// MLX's published config keeps the source checkpoint's `torch_dtype` even
    /// though its safetensors activations and dense tensors use f16.
    pub(crate) fn convert_mlx_affine4(&self) -> Result<ModelConfig> {
        let fail = |s: &str| Error::Config(s.into());
        if self.architectures != ["Qwen2ForCausalLM"] || self.model_type != "qwen2" {
            return Err(fail("supported architecture is Qwen2ForCausalLM / qwen2"));
        }
        if self.hidden_act != "silu" || self.use_sliding_window {
            return Err(fail(
                "only silu and full attention (use_sliding_window=false) are supported",
            ));
        }
        for (key, value) in &self.extra {
            let supported = match key.as_str() {
                "initializer_range"
                | "transformers_version"
                | "_name_or_path"
                | "max_window_layers"
                | "sliding_window" => true,
                "attention_dropout" => value.as_f64() == Some(0.),
                "use_cache" | "attention_bias" => value.as_bool() == Some(true),
                "mlp_bias" => value.as_bool() == Some(false),
                "rope_scaling" => value.is_null(),
                "head_dim" => value.as_u64().is_some_and(|d| {
                    d.checked_mul(self.num_attention_heads as u64) == Some(self.hidden_size as u64)
                }),
                "quantization" => {
                    let Some(params) = value.as_object() else {
                        return Err(fail("MLX quantization config must be an object"));
                    };
                    params.get("bits").and_then(serde_json::Value::as_u64) == Some(4)
                        && params.get("group_size").and_then(serde_json::Value::as_u64) == Some(64)
                        && params.get("mode").is_none_or(|mode| mode == "affine")
                        && params
                            .keys()
                            .all(|name| matches!(name.as_str(), "bits" | "group_size" | "mode"))
                }
                _ => false,
            };
            if !supported {
                return Err(fail(&format!(
                    "unsupported MLX Qwen config field {key}={value}"
                )));
            }
        }
        if !self.extra.contains_key("quantization") {
            return Err(fail("MLX Qwen config is missing quantization metadata"));
        }
        if self.torch_dtype != "bfloat16" {
            return Err(fail("MLX source config must identify the BF16 Qwen base"));
        }
        if self.num_attention_heads == 0
            || !self.hidden_size.is_multiple_of(self.num_attention_heads)
        {
            return Err(fail("hidden_size must be divisible by num_attention_heads"));
        }
        if self.bos_token_id as usize >= self.vocab_size
            || self.eos_token_id as usize >= self.vocab_size
        {
            return Err(fail("BOS/EOS ID outside model vocabulary"));
        }
        let config = ModelConfig {
            vocab_size: self.vocab_size,
            hidden_size: self.hidden_size,
            intermediate_size: self.intermediate_size,
            num_layers: self.num_hidden_layers,
            num_attention_heads: self.num_attention_heads,
            num_key_value_heads: self.num_key_value_heads,
            head_dim: self.hidden_size / self.num_attention_heads,
            rms_norm_epsilon: self.rms_norm_eps,
            rope_theta: self.rope_theta,
            max_context_length: self.max_position_embeddings,
            tie_word_embeddings: self.tie_word_embeddings,
            dtype: DType::F16,
        };
        config.validate()?;
        Ok(config)
    }
}
/// (official name, Phase 2 canonical name, exact shape). Q/K/V bias is mandatory.
pub fn specifications(c: &ModelConfig) -> Vec<(String, String, Vec<usize>)> {
    let mut out = Vec::new();
    for (canonical, shape) in weights::specifications(c) {
        let official = official_name(&canonical);
        out.push((official, canonical, shape));
    }
    for layer in 0..c.num_layers {
        for (p, heads) in [
            ("q", c.num_attention_heads),
            ("k", c.num_key_value_heads),
            ("v", c.num_key_value_heads),
        ] {
            out.push((
                format!("model.layers.{layer}.self_attn.{p}_proj.bias"),
                format!("layers.{layer}.{p}.bias"),
                vec![heads * c.head_dim],
            ));
        }
    }
    out
}
pub(crate) fn official_name(canonical: &str) -> String {
    if canonical == "embedding.weight" {
        "model.embed_tokens.weight".into()
    } else if canonical == "final_norm.weight" {
        "model.norm.weight".into()
    } else if canonical == "lm_head.weight" {
        canonical.into()
    } else {
        let mut parts = canonical.split('.');
        let _ = parts.next();
        let layer = parts.next().unwrap_or_default();
        let component = parts.next().unwrap_or_default();
        let component = match component {
            "input_norm" => "input_layernorm",
            "post_norm" => "post_attention_layernorm",
            "q" => "self_attn.q_proj",
            "k" => "self_attn.k_proj",
            "v" => "self_attn.v_proj",
            "o" => "self_attn.o_proj",
            "q_norm" => "self_attn.q_norm",
            "k_norm" => "self_attn.k_norm",
            "gate" => "mlp.gate_proj",
            "up" => "mlp.up_proj",
            "down" => "mlp.down_proj",
            _ => component,
        };
        format!("model.layers.{layer}.{component}.weight")
    }
}
pub fn map_weights(d: &MetalDevice, c: &ModelConfig, source: &Weights) -> Result<Weights> {
    c.validate()?;
    let specs = specifications(c);
    // Validate all official names before any construction/transposition dispatch.
    for (official, _, shape) in &specs {
        weights::checked(source, official, shape, c, d)?;
    }
    for name in source.names() {
        if !specs.iter().any(|(official, _, _)| official == name) {
            return Err(Error::Weight {
                name: name.into(),
                message: "unexpected tensor for supported Qwen contract".into(),
            });
        }
    }
    source.remap(
        specs
            .iter()
            .map(|(official, canonical, _)| (official.as_str(), canonical.as_str())),
    )
}
pub fn construct(d: &MetalDevice, c: ModelConfig, source: &Weights) -> Result<Transformer> {
    let mapped = map_weights(d, &c, source)?;
    Transformer::from_weights(d, c, &mapped)
}
