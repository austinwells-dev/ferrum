//! Official Granite 4 dense checkpoint mapping into the shared transformer.
#![forbid(unsafe_code)]

use super::{
    ModelConfig, Transformer,
    architecture::{ArchitecturePolicy, ProjectionBias},
    weights,
};
use crate::{DType, Error, MetalDevice, Result, loader::Weights, tokenizer::Tokenizer};
use serde::Deserialize;
use std::{
    collections::BTreeMap,
    path::Path,
    time::{Duration, Instant},
};

#[derive(Debug, Deserialize)]
pub struct GraniteConfig {
    architectures: Vec<String>,
    model_type: String,
    vocab_size: usize,
    hidden_size: usize,
    intermediate_size: usize,
    shared_intermediate_size: usize,
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
    normalization_function: String,
    position_embedding_type: String,
    layer_types: Vec<String>,
    num_local_experts: usize,
    num_experts_per_tok: usize,
    tie_word_embeddings: bool,
    torch_dtype: String,
    use_cache: bool,
    bos_token_id: u32,
    eos_token_id: u32,
    pad_token_id: u32,
    #[serde(flatten)]
    extra: BTreeMap<String, serde_json::Value>,
}

impl GraniteConfig {
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let bytes =
            std::fs::read(path).map_err(|e| Error::Config(format!("{}: {e}", path.display())))?;
        serde_json::from_slice(&bytes).map_err(|e| Error::Config(format!("config.json: {e}")))
    }

    pub fn convert(&self) -> Result<(ModelConfig, ArchitecturePolicy)> {
        let bad =
            |reason: &str| Error::Config(format!("unsupported Granite dense policy: {reason}"));
        if self.architectures != ["GraniteMoeHybridForCausalLM"]
            || self.model_type != "granitemoehybrid"
            || self.layer_types.len() != self.num_hidden_layers
            || self.layer_types.iter().any(|layer| layer != "attention")
            || self.num_local_experts != 0
            || self.num_experts_per_tok != 0
        {
            return Err(bad("expected all-attention layers without active experts"));
        }
        if self.intermediate_size != self.shared_intermediate_size
            || self.hidden_act != "silu"
            || self.normalization_function != "rmsnorm"
            || self.position_embedding_type != "rope"
            || self.rope_scaling.is_some()
            || self.attention_bias
            || self.attention_dropout != 0.
            || !self.tie_word_embeddings
            || self.torch_dtype != "bfloat16"
            || !self.use_cache
        {
            return Err(bad("activation, normalization, attention, or storage"));
        }
        for id in [self.bos_token_id, self.eos_token_id, self.pad_token_id] {
            if id as usize >= self.vocab_size {
                return Err(bad("special token ID outside vocabulary"));
            }
        }
        for (key, value) in &self.extra {
            let known = match key.as_str() {
                "init_method" => value.as_str().is_some(),
                "initializer_range" | "router_aux_loss_coef" => value.as_f64().is_some(),
                "mamba_chunk_size" | "mamba_d_conv" | "mamba_d_head" | "mamba_d_state"
                | "mamba_expand" | "mamba_n_groups" | "mamba_n_heads" => value.as_u64().is_some(),
                "mamba_conv_bias" | "mamba_proj_bias" | "output_router_logits" => {
                    value.is_boolean()
                }
                "transformers_version" => value.as_str().is_some(),
                _ => false,
            };
            if !known {
                return Err(bad(&format!("unknown field {key}")));
            }
        }
        if self.num_attention_heads == 0
            || !self.hidden_size.is_multiple_of(self.num_attention_heads)
        {
            return Err(bad("hidden width is not divisible by Q heads"));
        }
        let config = ModelConfig {
            vocab_size: self.vocab_size,
            hidden_size: self.hidden_size,
            intermediate_size: self.shared_intermediate_size,
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
    if parts.len() != 4 || parts[0] != "layers" || parts[3] != "weight" {
        return Err(Error::Weight {
            name: canonical.into(),
            message: "unmapped Granite weight".into(),
        });
    }
    let component = match parts[2] {
        "input_norm" => "input_layernorm",
        "post_norm" => "post_attention_layernorm",
        "q" => "self_attn.q_proj",
        "k" => "self_attn.k_proj",
        "v" => "self_attn.v_proj",
        "o" => "self_attn.o_proj",
        "down" => "shared_mlp.output_linear",
        "gate" | "up" => "shared_mlp.input_linear",
        _ => {
            return Err(Error::Weight {
                name: canonical.into(),
                message: "unmapped Granite component".into(),
            });
        }
    };
    Ok(format!("model.layers.{}.{component}.weight", parts[1]))
}

pub fn map_weights(
    device: &MetalDevice,
    config: &ModelConfig,
    policy: &ArchitecturePolicy,
    source: &Weights,
) -> Result<Weights> {
    let specs = weights::specifications_with_policy(config, policy);
    let mut ordinary = Vec::new();
    let mut expected_source = std::collections::BTreeSet::new();
    for (canonical, shape) in &specs {
        let name = official_name(canonical)?;
        expected_source.insert(name.clone());
        if canonical.ends_with(".gate.weight") || canonical.ends_with(".up.weight") {
            continue;
        }
        weights::checked(source, &name, shape, config, device)?;
        ordinary.push((name, canonical.clone()));
    }
    for layer in 0..config.num_layers {
        let name = format!("model.layers.{layer}.shared_mlp.input_linear.weight");
        weights::checked(
            source,
            &name,
            &[2 * config.intermediate_size, config.hidden_size],
            config,
            device,
        )?;
    }
    for name in source.names() {
        if !expected_source.contains(name) {
            return Err(Error::Weight {
                name: name.into(),
                message: "unexpected tensor for Granite dense architecture".into(),
            });
        }
    }
    let mut mapped = source.remap(
        ordinary
            .iter()
            .map(|(official, canonical)| (official.as_str(), canonical.as_str())),
    )?;
    for layer in 0..config.num_layers {
        let packed = source.get(&format!(
            "model.layers.{layer}.shared_mlp.input_linear.weight"
        ))?;
        let half = config.intermediate_size * config.hidden_size;
        mapped.insert(
            format!("layers.{layer}.gate.weight"),
            packed.view(0, [config.intermediate_size, config.hidden_size])?,
        )?;
        mapped.insert(
            format!("layers.{layer}.up.weight"),
            packed.view(half, [config.intermediate_size, config.hidden_size])?,
        )?;
    }
    Ok(mapped)
}

pub struct LoadedGranite {
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

pub fn load(device: &MetalDevice, dir: impl AsRef<Path>) -> Result<LoadedGranite> {
    let dir = dir.as_ref();
    let started = Instant::now();
    let source_config = GraniteConfig::from_file(dir.join("config.json"))?;
    let (config, policy) = source_config.convert()?;
    let tc: serde_json::Value = serde_json::from_slice(
        &std::fs::read(dir.join("tokenizer_config.json"))
            .map_err(|e| Error::Tokenizer(format!("tokenizer_config.json: {e}")))?,
    )
    .map_err(|e| Error::Tokenizer(format!("tokenizer_config.json: {e}")))?;
    let gc: serde_json::Value = serde_json::from_slice(
        &std::fs::read(dir.join("generation_config.json"))
            .map_err(|e| Error::Tokenizer(format!("generation_config.json: {e}")))?,
    )
    .map_err(|e| Error::Tokenizer(format!("generation_config.json: {e}")))?;
    let template = std::fs::read_to_string(dir.join("chat_template.jinja"))
        .map_err(|e| Error::Tokenizer(format!("chat_template.jinja: {e}")))?;
    if tc["tokenizer_class"] != "GPT2Tokenizer"
        || tc["add_bos_token"] != false
        || tc["bos_token"] != "<|end_of_text|>"
        || tc["eos_token"] != "<|end_of_text|>"
        || tc["pad_token"] != "<|pad|>"
        || gc["bos_token_id"] != source_config.bos_token_id
        || gc["eos_token_id"] != source_config.eos_token_id
        || gc["pad_token_id"] != source_config.pad_token_id
        || !template.contains("<|start_of_role|>")
        || !template.contains("<|end_of_role|>")
        || !template.contains("<|end_of_text|>")
        || !template.contains("add_generation_prompt")
    {
        return Err(Error::Tokenizer(
            "Granite tokenizer/generation policy mismatch".into(),
        ));
    }
    let tokenizer = Tokenizer::from_file(dir.join("tokenizer.json"))?;
    if tokenizer.vocab_size() > config.vocab_size
        || tokenizer.encode("<|end_of_text|>")? != [source_config.eos_token_id]
        || tokenizer.encode("<|pad|>")? != [source_config.pad_token_id]
    {
        return Err(Error::Tokenizer(
            "Granite tokenizer IDs differ from config".into(),
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
    let model = Transformer::from_weights_with_policy(device, config.clone(), policy, &mapped)?;
    let construction = started.elapsed();
    let parameter_count = source_tensor_bytes / DType::BF16.size_bytes();
    Ok(LoadedGranite {
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
