//! Official OLMo 2 weight mapping and post-normalized block policy.
#![forbid(unsafe_code)]

use super::{
    ModelConfig, Transformer,
    architecture::{ArchitecturePolicy, ProjectionBias, QkNormLayout, ResidualTopology},
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
pub struct Olmo2Config {
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
    attention_bias: bool,
    attention_dropout: f32,
    hidden_act: String,
    tie_word_embeddings: bool,
    torch_dtype: String,
    use_cache: bool,
    eos_token_id: u32,
    pad_token_id: u32,
    #[serde(flatten)]
    extra: BTreeMap<String, serde_json::Value>,
}

impl Olmo2Config {
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let bytes =
            std::fs::read(path).map_err(|e| Error::Config(format!("{}: {e}", path.display())))?;
        serde_json::from_slice(&bytes).map_err(|e| Error::Config(format!("config.json: {e}")))
    }

    pub fn convert(&self) -> Result<(ModelConfig, ArchitecturePolicy)> {
        let bad = |reason: &str| Error::Config(format!("unsupported OLMo 2 policy: {reason}"));
        if self.architectures != ["Olmo2ForCausalLM"] || self.model_type != "olmo2" {
            return Err(bad("architecture identifier"));
        }
        if self.attention_bias
            || self.attention_dropout != 0.
            || self.hidden_act != "silu"
            || self.tie_word_embeddings
            || self.torch_dtype != "float32"
            || self.rope_scaling.is_some()
            || !self.use_cache
        {
            return Err(bad("attention, activation, embedding, or storage"));
        }
        if self.num_attention_heads == 0
            || !self.hidden_size.is_multiple_of(self.num_attention_heads)
        {
            return Err(bad("hidden size/head count mismatch"));
        }
        if [self.eos_token_id, self.pad_token_id]
            .iter()
            .any(|&id| id as usize >= self.vocab_size)
        {
            return Err(bad("special token outside vocabulary"));
        }
        for (name, value) in &self.extra {
            let known = match name.as_str() {
                "initializer_range" => value.as_f64().is_some(),
                "transformers_version" => value.as_str().is_some(),
                _ => false,
            };
            if !known {
                return Err(bad(&format!("unknown config field {name}")));
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
            tie_word_embeddings: false,
            dtype: DType::F32,
        };
        config.validate()?;
        let policy = ArchitecturePolicy {
            qk_norm_epsilon: Some(self.rms_norm_eps),
            qk_norm_layout: QkNormLayout::Projection,
            qkv_bias: ProjectionBias::Forbidden,
            residual_topology: ResidualTopology::PostNorm,
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
    if canonical == "lm_head.weight" {
        return Ok("lm_head.weight".into());
    }
    let parts: Vec<_> = canonical.split('.').collect();
    if parts.len() != 4 || parts[0] != "layers" || parts[3] != "weight" {
        return Err(Error::Weight {
            name: canonical.into(),
            message: "unmapped OLMo 2 weight".into(),
        });
    }
    let component = match parts[2] {
        "input_norm" => "post_attention_layernorm",
        "post_norm" => "post_feedforward_layernorm",
        "q" => "self_attn.q_proj",
        "k" => "self_attn.k_proj",
        "v" => "self_attn.v_proj",
        "o" => "self_attn.o_proj",
        "q_norm" => "self_attn.q_norm",
        "k_norm" => "self_attn.k_norm",
        "gate" => "mlp.gate_proj",
        "up" => "mlp.up_proj",
        "down" => "mlp.down_proj",
        _ => {
            return Err(Error::Weight {
                name: canonical.into(),
                message: "unmapped OLMo 2 component".into(),
            });
        }
    };
    Ok(format!("model.layers.{}.{component}.weight", parts[1]))
}

pub fn map_weights(
    device: &MetalDevice,
    config: &ModelConfig,
    policy: ArchitecturePolicy,
    source: &Weights,
) -> Result<Weights> {
    let specs = weights::specifications_with_policy(config, policy)
        .into_iter()
        .map(|(canonical, shape)| Ok((official_name(&canonical)?, canonical, shape)))
        .collect::<Result<Vec<_>>>()?;
    for (official, _, shape) in &specs {
        weights::checked(source, official, shape, config, device)?;
    }
    for name in source.names() {
        if !specs.iter().any(|(official, _, _)| official == name) {
            return Err(Error::Weight {
                name: name.into(),
                message: "unexpected tensor for OLMo 2 architecture".into(),
            });
        }
    }
    source.remap(
        specs
            .iter()
            .map(|(official, canonical, _)| (official.as_str(), canonical.as_str())),
    )
}

pub struct LoadedOlmo2 {
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

pub fn load(device: &MetalDevice, dir: impl AsRef<Path>) -> Result<LoadedOlmo2> {
    let dir = dir.as_ref();
    let started = Instant::now();
    let source_config = Olmo2Config::from_file(dir.join("config.json"))?;
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
    if tc["tokenizer_class"] != "GPT2Tokenizer"
        || tc["bos_token"] != "<|endoftext|>"
        || tc["eos_token"] != "<|endoftext|>"
        || tc["pad_token"] != "<|pad|>"
        || gc["eos_token_id"] != source_config.eos_token_id
        || gc["pad_token_id"] != source_config.pad_token_id
    {
        return Err(Error::Tokenizer(
            "OLMo 2 tokenizer/generation policy mismatch".into(),
        ));
    }
    let tokenizer = Tokenizer::from_file(dir.join("tokenizer.json"))?;
    if tokenizer.vocab_size() > config.vocab_size
        || tokenizer.encode("<|endoftext|>")? != [source_config.eos_token_id]
        || tokenizer.encode("<|pad|>")? != [source_config.pad_token_id]
    {
        return Err(Error::Tokenizer(
            "OLMo 2 tokenizer IDs differ from config".into(),
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
    let model = Transformer::from_weights_with_policy(device, config.clone(), policy, &mapped)?;
    let construction = started.elapsed();
    let parameter_count = source_tensor_bytes / DType::F32.size_bytes();
    Ok(LoadedOlmo2 {
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
