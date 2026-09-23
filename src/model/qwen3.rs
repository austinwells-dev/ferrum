//! Qwen3 metadata and weight mapping on the shared dense execution path.
#![forbid(unsafe_code)]
use super::{
    ModelConfig, Transformer,
    architecture::{ArchitecturePolicy, ProjectionBias},
    qwen, weights,
};
use crate::{DType, Error, MetalDevice, Result, loader::Weights, tokenizer::qwen::QwenTokenizer};
use serde::Deserialize;
use std::{
    collections::BTreeMap,
    path::Path,
    time::{Duration, Instant},
};

#[derive(Debug, Deserialize)]
pub struct Qwen3Config {
    pub architectures: Vec<String>,
    pub model_type: String,
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub rms_norm_eps: f32,
    pub rope_theta: f32,
    pub max_position_embeddings: usize,
    pub tie_word_embeddings: bool,
    pub torch_dtype: String,
    pub hidden_act: String,
    pub attention_bias: bool,
    pub attention_dropout: f32,
    pub use_sliding_window: bool,
    pub sliding_window: Option<usize>,
    pub rope_scaling: Option<serde_json::Value>,
    pub use_cache: bool,
    pub bos_token_id: u32,
    pub eos_token_id: u32,
    #[serde(flatten)]
    pub extra: BTreeMap<String, serde_json::Value>,
}

impl Qwen3Config {
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let bytes =
            std::fs::read(path).map_err(|e| Error::Config(format!("{}: {e}", path.display())))?;
        serde_json::from_slice(&bytes).map_err(|e| Error::Config(format!("config.json: {e}")))
    }

    pub fn convert(&self) -> Result<(ModelConfig, ArchitecturePolicy)> {
        let bad = |message: &str| Error::Config(message.into());
        if self.architectures != ["Qwen3ForCausalLM"] || self.model_type != "qwen3" {
            return Err(bad("expected Qwen3ForCausalLM / qwen3"));
        }
        if self.hidden_act != "silu"
            || self.attention_bias
            || self.attention_dropout != 0.
            || self.use_sliding_window
            || self.sliding_window.is_some()
            || self.rope_scaling.is_some()
            || !self.use_cache
        {
            return Err(bad(
                "unsupported Qwen3 attention, activation, or cache policy",
            ));
        }
        if self.torch_dtype != "bfloat16" {
            return Err(bad("Qwen3 checkpoint storage must be bfloat16"));
        }
        if self.bos_token_id as usize >= self.vocab_size
            || self.eos_token_id as usize >= self.vocab_size
        {
            return Err(bad("BOS/EOS ID outside model vocabulary"));
        }
        for (name, value) in &self.extra {
            let supported = match name.as_str() {
                "initializer_range" => value.as_f64().is_some(),
                "transformers_version" => value.as_str().is_some(),
                "max_window_layers" => value.as_u64().is_some(),
                _ => false,
            };
            if !supported {
                return Err(bad(&format!(
                    "unsupported Qwen3 config field {name}={value}"
                )));
            }
        }
        let config = ModelConfig {
            vocab_size: self.vocab_size,
            hidden_size: self.hidden_size,
            intermediate_size: self.intermediate_size,
            num_layers: self.num_hidden_layers,
            num_attention_heads: self.num_attention_heads,
            num_key_value_heads: self.num_key_value_heads,
            head_dim: self.head_dim,
            rms_norm_epsilon: self.rms_norm_eps,
            rope_theta: self.rope_theta,
            max_context_length: self.max_position_embeddings,
            tie_word_embeddings: self.tie_word_embeddings,
            dtype: DType::BF16,
        };
        config.validate()?;
        let policy = ArchitecturePolicy {
            qk_norm_epsilon: Some(self.rms_norm_eps),
            qkv_bias: ProjectionBias::Forbidden,
        };
        policy.validate()?;
        Ok((config, policy))
    }
}

pub fn specifications(
    config: &ModelConfig,
    policy: ArchitecturePolicy,
) -> Vec<(String, String, Vec<usize>)> {
    weights::specifications_with_policy(config, policy)
        .into_iter()
        .map(|(canonical, shape)| (qwen::official_name(&canonical), canonical, shape))
        .collect()
}

pub fn map_weights(
    device: &MetalDevice,
    config: &ModelConfig,
    policy: ArchitecturePolicy,
    source: &Weights,
) -> Result<Weights> {
    config.validate()?;
    policy.validate()?;
    let specs = specifications(config, policy);
    for (official, _, shape) in &specs {
        weights::checked(source, official, shape, config, device)?;
    }
    let redundant_tied_head =
        config.tie_word_embeddings && source.optional("lm_head.weight").is_some();
    if redundant_tied_head {
        weights::checked(
            source,
            "lm_head.weight",
            &[config.vocab_size, config.hidden_size],
            config,
            device,
        )?;
        if !source.payload_equal("model.embed_tokens.weight", "lm_head.weight")? {
            return Err(Error::Weight {
                name: "lm_head.weight".into(),
                message: "tied LM head differs from input embedding".into(),
            });
        }
    }
    for name in source.names() {
        if !(specs.iter().any(|(official, _, _)| official == name)
            || redundant_tied_head && name == "lm_head.weight")
        {
            return Err(Error::Weight {
                name: name.into(),
                message: "unexpected tensor for Qwen3 architecture".into(),
            });
        }
    }
    source.remap(
        specs
            .iter()
            .map(|(official, canonical, _)| (official.as_str(), canonical.as_str())),
    )
}

pub fn construct(
    device: &MetalDevice,
    config: ModelConfig,
    policy: ArchitecturePolicy,
    source: &Weights,
) -> Result<Transformer> {
    let mapped = map_weights(device, &config, policy, source)?;
    Transformer::from_weights_with_policy(device, config, policy, &mapped)
}

pub struct LoadedQwen3 {
    pub config: ModelConfig,
    pub model: Transformer,
    pub tokenizer: QwenTokenizer,
    pub source_tensor_bytes: usize,
    pub tensor_count: usize,
    pub parameter_count: usize,
    pub config_tokenizer_load: Duration,
    pub weight_load: Duration,
    pub construction: Duration,
}

pub fn load(device: &MetalDevice, dir: impl AsRef<Path>) -> Result<LoadedQwen3> {
    let dir = dir.as_ref();
    let start = Instant::now();
    let source_config = Qwen3Config::from_file(dir.join("config.json"))?;
    let (config, policy) = source_config.convert()?;
    let tokenizer = QwenTokenizer::load_qwen3(
        dir,
        config.vocab_size,
        source_config.bos_token_id,
        source_config.eos_token_id,
    )?;
    let config_tokenizer_load = start.elapsed();
    let start = Instant::now();
    let source = Weights::from_directory(device, dir)?;
    let source_tensor_bytes = source.bytes();
    let tensor_count = source.names().count();
    let weight_load = start.elapsed();
    let start = Instant::now();
    let model = construct(device, config.clone(), policy, &source)?;
    // A tied checkpoint may serialize a byte-identical LM head as a second
    // tensor. Count the model's unique retained parameters, not that copy.
    let parameter_count = model.weight_bytes() / DType::BF16.size_bytes();
    let construction = start.elapsed();
    Ok(LoadedQwen3 {
        config,
        model,
        tokenizer,
        source_tensor_bytes,
        tensor_count,
        parameter_count,
        config_tokenizer_load,
        weight_load,
        construction,
    })
}
