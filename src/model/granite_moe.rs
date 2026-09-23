//! Official Granite 3.1 sparse MoE adapter using the shared decoder runtime.
#![forbid(unsafe_code)]

use super::{
    ModelConfig, Transformer,
    architecture::{ArchitecturePolicy, MoeRoutingPolicy, ProjectionBias},
    weights::{self, ModelWeight, ModelWeights},
};
use crate::{DType, Error, MetalDevice, Result, loader::Weights, tokenizer::Tokenizer};
use serde::Deserialize;
use std::{
    collections::BTreeMap,
    path::Path,
    time::{Duration, Instant},
};

#[derive(Debug, Deserialize)]
pub struct GraniteMoeConfig {
    architectures: Vec<String>,
    model_type: String,
    vocab_size: usize,
    hidden_size: usize,
    intermediate_size: usize,
    num_hidden_layers: usize,
    num_attention_heads: usize,
    num_key_value_heads: usize,
    max_position_embeddings: usize,
    rms_norm_eps: f32,
    rope_theta: f32,
    rope_scaling: Option<serde_json::Value>,
    attention_multiplier: f32,
    embedding_multiplier: f32,
    residual_multiplier: f32,
    logits_scaling: f32,
    attention_bias: bool,
    attention_dropout: f32,
    hidden_act: String,
    num_experts_per_tok: usize,
    num_local_experts: usize,
    output_router_logits: bool,
    tie_word_embeddings: bool,
    torch_dtype: String,
    use_cache: bool,
    bos_token_id: u32,
    eos_token_id: u32,
    pad_token_id: u32,
    #[serde(flatten)]
    extra: BTreeMap<String, serde_json::Value>,
}

impl GraniteMoeConfig {
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let bytes =
            std::fs::read(path).map_err(|e| Error::Config(format!("{}: {e}", path.display())))?;
        serde_json::from_slice(&bytes).map_err(|e| Error::Config(format!("config.json: {e}")))
    }

    pub fn convert(&self) -> Result<(ModelConfig, ArchitecturePolicy)> {
        let bad = |reason: &str| Error::Config(format!("unsupported Granite MoE policy: {reason}"));
        if self.architectures != ["GraniteMoeForCausalLM"] || self.model_type != "granitemoe" {
            return Err(bad("architecture identifier"));
        }
        if self.num_local_experts == 0
            || self.num_experts_per_tok == 0
            || self.num_experts_per_tok > self.num_local_experts
            || self.hidden_size == 0
            || !self
                .hidden_size
                .is_multiple_of(self.num_attention_heads.max(1))
        {
            return Err(bad("expert or attention dimensions"));
        }
        if self.attention_bias
            || self.attention_dropout != 0.
            || self.hidden_act != "silu"
            || self.rope_scaling.is_some()
            || !self.tie_word_embeddings
            || self.torch_dtype != "bfloat16"
            || !self.use_cache
            || self.output_router_logits
        {
            return Err(bad("attention, activation, output, or storage policy"));
        }
        for id in [self.bos_token_id, self.eos_token_id, self.pad_token_id] {
            if id as usize >= self.vocab_size {
                return Err(bad("special token outside vocabulary"));
            }
        }
        for (name, value) in &self.extra {
            let known = match name.as_str() {
                "initializer_range" | "router_aux_loss_coef" => value.as_f64().is_some(),
                "transformers_version" => value.as_str().is_some(),
                _ => false,
            };
            if !known {
                return Err(bad(&format!("unknown field {name}")));
            }
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
            tie_word_embeddings: true,
            dtype: DType::BF16,
        };
        config.validate()?;
        let policy = ArchitecturePolicy {
            qkv_bias: ProjectionBias::Forbidden,
            attention_scale: Some(self.attention_multiplier),
            embedding_multiplier: self.embedding_multiplier,
            residual_multiplier: self.residual_multiplier,
            logits_divisor: self.logits_scaling,
            moe: Some(MoeRoutingPolicy {
                experts: self.num_local_experts,
                top_k: self.num_experts_per_tok,
            }),
            ..ArchitecturePolicy::default()
        };
        policy.validate()?;
        Ok((config, policy))
    }
}

fn official_name(canonical: &str) -> Result<String> {
    if canonical == "embedding.weight" {
        return Ok("model.embed_tokens.weight".into());
    }
    if canonical == "final_norm.weight" {
        return Ok("model.norm.weight".into());
    }
    let parts: Vec<_> = canonical.split('.').collect();
    if parts.len() == 4 && parts[0] == "layers" && parts[3] == "weight" {
        let component = match parts[2] {
            "input_norm" => "input_layernorm",
            "post_norm" => "post_attention_layernorm",
            "q" => "self_attn.q_proj",
            "k" => "self_attn.k_proj",
            "v" => "self_attn.v_proj",
            "o" => "self_attn.o_proj",
            _ => {
                return Err(Error::Weight {
                    name: canonical.into(),
                    message: "unmapped Granite MoE component".into(),
                });
            }
        };
        return Ok(format!("model.layers.{}.{component}.weight", parts[1]));
    }
    let parts: Vec<_> = canonical.split('.').collect();
    if parts.len() == 5 && parts[0] == "layers" && parts[2] == "moe" && parts[4] == "weight" {
        let component = match parts[3] {
            "router" => "block_sparse_moe.router.layer",
            "input" => "block_sparse_moe.input_linear",
            "output" => "block_sparse_moe.output_linear",
            _ => {
                return Err(Error::Weight {
                    name: canonical.into(),
                    message: "unmapped Granite expert weight".into(),
                });
            }
        };
        return Ok(format!("model.layers.{}.{component}.weight", parts[1]));
    }
    Err(Error::Weight {
        name: canonical.into(),
        message: "unmapped Granite MoE weight".into(),
    })
}

pub(crate) fn map_weights(
    device: &MetalDevice,
    config: &ModelConfig,
    policy: ArchitecturePolicy,
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
                message: "unexpected tensor for Granite MoE architecture".into(),
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

pub struct LoadedGraniteMoe {
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

pub fn load(device: &MetalDevice, dir: impl AsRef<Path>) -> Result<LoadedGraniteMoe> {
    let dir = dir.as_ref();
    let started = Instant::now();
    let source_config = GraniteMoeConfig::from_file(dir.join("config.json"))?;
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
    let template = tokenizer_config["chat_template"]
        .as_str()
        .ok_or_else(|| Error::Tokenizer("Granite MoE chat template is missing".into()))?;
    if tokenizer_config["tokenizer_class"] != "GPT2Tokenizer"
        || tokenizer_config["add_bos_token"] != false
        || tokenizer_config["bos_token"] != "<|end_of_text|>"
        || tokenizer_config["eos_token"] != "<|end_of_text|>"
        || tokenizer_config["pad_token"] != "<|end_of_text|>"
        || generation_config["bos_token_id"] != source_config.bos_token_id
        || generation_config["eos_token_id"] != source_config.eos_token_id
        || generation_config["pad_token_id"] != source_config.pad_token_id
        || !template.contains("<|start_of_role|>")
        || !template.contains("<|end_of_role|>")
        || !template.contains("add_generation_prompt")
    {
        return Err(Error::Tokenizer(
            "Granite MoE tokenizer policy mismatch".into(),
        ));
    }
    let tokenizer = Tokenizer::from_file(dir.join("tokenizer.json"))?;
    if tokenizer.vocab_size() > config.vocab_size
        || tokenizer.encode("<|end_of_text|>")? != [source_config.eos_token_id]
    {
        return Err(Error::Tokenizer(
            "Granite MoE token IDs differ from config".into(),
        ));
    }
    let config_tokenizer_load = started.elapsed();
    let started = Instant::now();
    let source = Weights::from_directory(device, dir)?;
    let source_tensor_bytes = source.bytes();
    let tensor_count = source.names().count();
    let weight_load = started.elapsed();
    let started = Instant::now();
    let mapped = map_weights(device, &config, policy, &source)?;
    let model =
        Transformer::from_model_weights_with_policy(device, config.clone(), policy, &mapped)?;
    let construction = started.elapsed();
    let parameter_count = source_tensor_bytes / DType::BF16.size_bytes();
    Ok(LoadedGraniteMoe {
        config,
        model,
        tokenizer,
        eos_ids: vec![source_config.eos_token_id],
        source_tensor_bytes,
        tensor_count,
        parameter_count,
        config_tokenizer_load,
        weight_load,
        construction,
    })
}
