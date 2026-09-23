//! Official LFM2 short-convolution/attention hybrid adapter on shared execution.
#![forbid(unsafe_code)]

use super::{
    ModelConfig, Transformer,
    architecture::{
        ArchitecturePolicy, LayerFeedForwardPolicy, LayerOperatorPolicy, LayerPolicy,
        ProjectionBias, QkNormLayout,
    },
    weights::{self, ModelWeight, ModelWeights},
};
use crate::{DType, Error, MetalDevice, Result, loader::Weights, tokenizer::Tokenizer};
use serde::Deserialize;
use std::{
    collections::BTreeMap,
    path::Path,
    time::{Duration, Instant},
};

const DOCUMENTED_CONTEXT_LENGTH: usize = 32_768;

#[derive(Debug, Deserialize)]
struct RopeParameters {
    rope_theta: f32,
    rope_type: String,
    #[serde(flatten)]
    extra: BTreeMap<String, serde_json::Value>,
}

#[derive(Debug, Deserialize)]
pub struct Lfm2Config {
    architectures: Vec<String>,
    model_type: String,
    vocab_size: usize,
    hidden_size: usize,
    intermediate_size: usize,
    num_hidden_layers: usize,
    num_attention_heads: usize,
    num_key_value_heads: usize,
    num_heads: usize,
    max_position_embeddings: usize,
    norm_eps: f32,
    rope_parameters: RopeParameters,
    layer_types: Vec<String>,
    #[serde(rename = "conv_L_cache")]
    conv_l_cache: usize,
    conv_bias: bool,
    conv_dim: usize,
    block_dim: usize,
    block_ff_dim: usize,
    block_norm_eps: f32,
    block_use_swiglu: bool,
    #[serde(rename = "block__name_mlp")]
    block_name_mlp: String,
    use_pos_enc: bool,
    tie_word_embeddings: bool,
    tie_embedding: bool,
    dtype: String,
    use_cache: bool,
    bos_token_id: u32,
    eos_token_id: u32,
    pad_token_id: u32,
    #[serde(flatten)]
    extra: BTreeMap<String, serde_json::Value>,
}

impl Lfm2Config {
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let bytes =
            std::fs::read(path).map_err(|e| Error::Config(format!("{}: {e}", path.display())))?;
        serde_json::from_slice(&bytes).map_err(|e| Error::Config(format!("config.json: {e}")))
    }

    pub fn convert(&self) -> Result<(ModelConfig, ArchitecturePolicy)> {
        let bad = |reason: &str| Error::Config(format!("unsupported LFM2 policy: {reason}"));
        if self.architectures != ["Lfm2ForCausalLM"] || self.model_type != "lfm2" {
            return Err(bad("architecture identifier"));
        }
        if self.layer_types.len() != self.num_hidden_layers
            || self.num_heads != self.num_attention_heads
            || self.conv_dim != self.hidden_size
            || self.block_dim != self.hidden_size
            || self.block_ff_dim != self.intermediate_size
            || self.block_norm_eps != self.norm_eps
            || !self.block_use_swiglu
            || self.block_name_mlp != "parallel_mlp_merged"
            || self.conv_bias
            || !self.use_pos_enc
            || !self.use_cache
            || !self.tie_embedding
            || !self.tie_word_embeddings
            || self.dtype != "bfloat16"
            || self.conv_l_cache != 3
            || self.rope_parameters.rope_type != "default"
            || !self.rope_parameters.extra.is_empty()
        {
            return Err(bad(
                "unsupported projection, convolution, MLP, or storage policy",
            ));
        }
        if self.rope_parameters.rope_theta <= 0. || !self.rope_parameters.rope_theta.is_finite() {
            return Err(bad("RoPE theta must be finite and positive"));
        }
        if self
            .layer_types
            .iter()
            .any(|layer| !matches!(layer.as_str(), "conv" | "full_attention"))
        {
            return Err(bad("layer schedule contains an unsupported operator"));
        }
        for (key, value) in &self.extra {
            let known = match key.as_str() {
                "block_auto_adjust_ff_dim"
                | "block_ffn_te_autocast"
                | "block_ffn_use_quantized_params"
                | "block_sequence_parallel_norm_across_tp"
                | "block_use_xavier_init"
                | "conv_use_xavier_init"
                | "ffn_te_autocast"
                | "ffn_use_quantized_params"
                | "sequence_parallel_norm_across_tp" => value.is_boolean(),
                "block_ffn_dim_multiplier"
                | "block_mlp_init_scale"
                | "block_multiple_of"
                | "block_out_init_scale"
                | "initializer_range" => value.is_number(),
                "transformers_version" => value.as_str().is_some(),
                "rope_parameters" | "architectures" | "model_type" => false,
                _ => false,
            };
            if !known {
                return Err(bad(&format!("unknown config field {key}")));
            }
        }
        if self.extra.get("block_auto_adjust_ff_dim") != Some(&serde_json::Value::Bool(false))
            || self.extra.get("block_ffn_use_quantized_params")
                != Some(&serde_json::Value::Bool(false))
            || self.extra.get("ffn_use_quantized_params") != Some(&serde_json::Value::Bool(false))
            || self.extra.get("block_ffn_dim_multiplier") != Some(&serde_json::Value::from(1.0))
        {
            return Err(bad(
                "FFN dimensions must use the declared dense SwiGLU shape",
            ));
        }
        for id in [self.bos_token_id, self.eos_token_id, self.pad_token_id] {
            if id as usize >= self.vocab_size {
                return Err(bad("special token ID outside vocabulary"));
            }
        }
        if self.num_attention_heads == 0
            || !self.hidden_size.is_multiple_of(self.num_attention_heads)
        {
            return Err(bad("hidden width is not divisible by attention heads"));
        }

        let config = ModelConfig {
            vocab_size: self.vocab_size,
            hidden_size: self.hidden_size,
            intermediate_size: self.intermediate_size,
            num_layers: self.num_hidden_layers,
            num_attention_heads: self.num_attention_heads,
            num_key_value_heads: self.num_key_value_heads,
            head_dim: self.hidden_size / self.num_attention_heads,
            rms_norm_epsilon: self.norm_eps,
            rope_theta: self.rope_parameters.rope_theta,
            // The model card documents 32K context; config.json's 128K limit
            // is retained as upstream metadata but not claimed as validated.
            max_context_length: self.max_position_embeddings.min(DOCUMENTED_CONTEXT_LENGTH),
            tie_word_embeddings: true,
            dtype: DType::BF16,
        };
        config.validate()?;
        let layer_policies = self
            .layer_types
            .iter()
            .map(|kind| LayerPolicy {
                operator: match kind.as_str() {
                    "conv" => LayerOperatorPolicy::ShortConv {
                        kernel_size: self.conv_l_cache,
                    },
                    "full_attention" => LayerOperatorPolicy::Attention,
                    _ => unreachable!("layer types validated above"),
                },
                feed_forward: LayerFeedForwardPolicy::Dense {
                    intermediate_size: self.intermediate_size,
                },
            })
            .collect();
        let policy = ArchitecturePolicy {
            qk_norm_epsilon: Some(self.norm_eps),
            qk_norm_layout: QkNormLayout::PerHead,
            qkv_bias: ProjectionBias::Forbidden,
            layer_policies: Some(layer_policies),
            ..ArchitecturePolicy::default()
        };
        policy.validate_for(&config)?;
        Ok((config, policy))
    }
}

fn official_name(canonical: &str) -> Result<String> {
    if canonical == "embedding.weight" {
        return Ok("model.embed_tokens.weight".into());
    }
    if canonical == "final_norm.weight" {
        return Ok("model.embedding_norm.weight".into());
    }
    let parts: Vec<_> = canonical.split('.').collect();
    if parts.len() == 4 && parts[0] == "layers" && parts[3] == "weight" {
        let component = match parts[2] {
            "input_norm" => "operator_norm",
            "post_norm" => "ffn_norm",
            "q" => "self_attn.q_proj",
            "k" => "self_attn.k_proj",
            "v" => "self_attn.v_proj",
            "o" => "self_attn.out_proj",
            "q_norm" => "self_attn.q_layernorm",
            "k_norm" => "self_attn.k_layernorm",
            "gate" => "feed_forward.w1",
            "up" => "feed_forward.w3",
            "down" => "feed_forward.w2",
            _ => {
                return Err(Error::Weight {
                    name: canonical.into(),
                    message: "unmapped LFM2 layer tensor".into(),
                });
            }
        };
        return Ok(format!("model.layers.{}.{component}.weight", parts[1]));
    }
    if parts.len() == 5 && parts[0] == "layers" && parts[2] == "conv" && parts[4] == "weight" {
        let component = match parts[3] {
            "in_proj" => "conv.in_proj",
            "depthwise" => "conv.conv",
            "out_proj" => "conv.out_proj",
            _ => {
                return Err(Error::Weight {
                    name: canonical.into(),
                    message: "unmapped LFM2 convolution tensor".into(),
                });
            }
        };
        return Ok(format!("model.layers.{}.{component}.weight", parts[1]));
    }
    Err(Error::Weight {
        name: canonical.into(),
        message: "unmapped LFM2 tensor".into(),
    })
}

pub(crate) fn map_weights(
    device: &MetalDevice,
    config: &ModelConfig,
    policy: &ArchitecturePolicy,
    source: &Weights,
) -> Result<ModelWeights> {
    let specs = weights::specifications_with_policy(config, policy);
    let mut expected_sources = std::collections::BTreeSet::new();
    let mut mapped = Vec::with_capacity(specs.len());
    for (canonical, shape) in specs {
        let official = official_name(&canonical)?;
        expected_sources.insert(official.clone());
        weights::checked(source, &official, &shape, config, device)?;
        mapped.push((official, canonical));
    }
    for name in source.names() {
        if !expected_sources.contains(name) {
            return Err(Error::Weight {
                name: name.into(),
                message: "unexpected tensor for LFM2 architecture".into(),
            });
        }
    }
    let mut result = ModelWeights::default();
    for (official, canonical) in mapped {
        result.insert(
            canonical,
            ModelWeight::Dense(source.get(&official)?.clone()),
        )?;
    }
    Ok(result)
}

pub struct LoadedLfm2 {
    pub config: ModelConfig,
    pub model: Transformer,
    pub tokenizer: Tokenizer,
    pub eos_ids: Vec<u32>,
    pub source_tensor_bytes: usize,
    pub tensor_count: usize,
    pub parameter_count: usize,
    pub config_tokenizer_load: Duration,
    pub weight_load: Duration,
    pub construction: Duration,
}

pub fn load(device: &MetalDevice, dir: impl AsRef<Path>) -> Result<LoadedLfm2> {
    let dir = dir.as_ref();
    let started = Instant::now();
    let source_config = Lfm2Config::from_file(dir.join("config.json"))?;
    let (config, policy) = source_config.convert()?;
    let tokenizer_config: serde_json::Value = serde_json::from_slice(
        &std::fs::read(dir.join("tokenizer_config.json"))
            .map_err(|e| Error::Tokenizer(format!("tokenizer_config.json: {e}")))?,
    )
    .map_err(|e| Error::Tokenizer(format!("tokenizer_config.json: {e}")))?;
    let generation_config: serde_json::Value = serde_json::from_slice(
        &std::fs::read(dir.join("generation_config.json"))
            .map_err(|e| Error::Tokenizer(format!("generation_config.json: {e}")))?,
    )
    .map_err(|e| Error::Tokenizer(format!("generation_config.json: {e}")))?;
    if tokenizer_config["tokenizer_class"] != "TokenizersBackend"
        || tokenizer_config["use_default_system_prompt"] != false
        || tokenizer_config["bos_token"] != "<|startoftext|>"
        || tokenizer_config["eos_token"] != "<|im_end|>"
        || tokenizer_config["pad_token"] != "<|pad|>"
        || generation_config["bos_token_id"] != source_config.bos_token_id
        || generation_config["eos_token_id"] != source_config.eos_token_id
        || generation_config["pad_token_id"] != source_config.pad_token_id
    {
        return Err(Error::Tokenizer(
            "LFM2 tokenizer/generation policy mismatch".into(),
        ));
    }
    let tokenizer = Tokenizer::from_file(dir.join("tokenizer.json"))?;
    if tokenizer.vocab_size() > config.vocab_size
        || tokenizer.encode("<|startoftext|>")? != [source_config.bos_token_id]
        || tokenizer.encode("<|im_end|>")? != [source_config.eos_token_id]
        || tokenizer.encode("<|pad|>")? != [source_config.pad_token_id]
    {
        return Err(Error::Tokenizer(
            "LFM2 special-token IDs differ from config".into(),
        ));
    }
    let config_tokenizer_load = started.elapsed();
    let started = Instant::now();
    let source = Weights::from_directory(device, dir)?;
    let source_tensor_bytes = source.bytes();
    let tensor_count = source.names().count();
    let weight_load = started.elapsed();
    let started = Instant::now();
    let mapped = map_weights(device, &config, &policy, &source)?;
    let model =
        Transformer::from_model_weights_with_policy(device, config.clone(), policy, &mapped)?;
    let construction = started.elapsed();
    Ok(LoadedLfm2 {
        config,
        model,
        tokenizer,
        eos_ids: vec![source_config.eos_token_id],
        source_tensor_bytes,
        tensor_count,
        parameter_count: source_tensor_bytes / DType::BF16.size_bytes(),
        config_tokenizer_load,
        weight_load,
        construction,
    })
}
