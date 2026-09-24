//! Official LFM2-MoE hybrid adapter using shared attention, convolution, and MoE execution.
#![forbid(unsafe_code)]

use super::{
    ModelConfig, Transformer,
    architecture::{
        ArchitecturePolicy, LayerFeedForwardPolicy, LayerOperatorPolicy, LayerPolicy,
        MoeRoutingPolicy, MoeScoringFunction, ProjectionBias, QkNormLayout,
    },
    weights::{self, ModelWeight, ModelWeights},
};
use crate::{DType, Error, MetalDevice, Result, Tensor, loader::Weights, tokenizer::Tokenizer};
use serde::Deserialize;
use std::{
    collections::{BTreeMap, BTreeSet},
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
pub struct Lfm2MoeConfig {
    architectures: Vec<String>,
    model_type: String,
    vocab_size: usize,
    hidden_size: usize,
    intermediate_size: usize,
    moe_intermediate_size: usize,
    num_hidden_layers: usize,
    num_attention_heads: usize,
    num_key_value_heads: usize,
    num_dense_layers: usize,
    num_experts: usize,
    num_experts_per_tok: usize,
    max_position_embeddings: usize,
    norm_eps: f32,
    rope_parameters: RopeParameters,
    layer_types: Vec<String>,
    #[serde(rename = "conv_L_cache")]
    conv_l_cache: usize,
    conv_bias: bool,
    norm_topk_prob: bool,
    routed_scaling_factor: f32,
    use_expert_bias: bool,
    tie_word_embeddings: bool,
    dtype: String,
    use_cache: bool,
    bos_token_id: u32,
    eos_token_id: u32,
    pad_token_id: u32,
    #[serde(flatten)]
    extra: BTreeMap<String, serde_json::Value>,
}

impl Lfm2MoeConfig {
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let bytes =
            std::fs::read(path).map_err(|e| Error::Config(format!("{}: {e}", path.display())))?;
        serde_json::from_slice(&bytes).map_err(|e| Error::Config(format!("config.json: {e}")))
    }

    pub fn convert(&self) -> Result<(ModelConfig, ArchitecturePolicy)> {
        let bad = |reason: &str| Error::Config(format!("unsupported LFM2-MoE policy: {reason}"));
        if self.architectures != ["Lfm2MoeForCausalLM"] || self.model_type != "lfm2_moe" {
            return Err(bad("architecture identifier"));
        }
        if self.layer_types.len() != self.num_hidden_layers
            || self.num_dense_layers > self.num_hidden_layers
            || !self.tie_word_embeddings
            || !self.use_cache
            || self.dtype != "bfloat16"
            || self.conv_bias
            || self.conv_l_cache != 3
            || self.rope_parameters.rope_type != "default"
            || !self.rope_parameters.extra.is_empty()
            || self.num_experts == 0
            || self.num_experts_per_tok == 0
            || self.num_experts_per_tok > self.num_experts
            || self.moe_intermediate_size == 0
            || !self.routed_scaling_factor.is_finite()
            || self.routed_scaling_factor <= 0.
        {
            return Err(bad("unsupported routing, convolution, or storage policy"));
        }
        if !self.rope_parameters.rope_theta.is_finite() || self.rope_parameters.rope_theta <= 0. {
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
                "initializer_range" => value.is_number(),
                "transformers_version" => value.as_str().is_some(),
                _ => false,
            };
            if !known {
                return Err(bad(&format!("unknown config field {key}")));
            }
        }
        if self.num_attention_heads == 0
            || !self.hidden_size.is_multiple_of(self.num_attention_heads)
            || self.num_key_value_heads == 0
            || !self
                .num_attention_heads
                .is_multiple_of(self.num_key_value_heads)
        {
            return Err(bad(
                "hidden width or grouped-query head geometry is invalid",
            ));
        }
        for id in [self.bos_token_id, self.eos_token_id, self.pad_token_id] {
            if id as usize >= self.vocab_size {
                return Err(bad("special token ID outside vocabulary"));
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
            rms_norm_epsilon: self.norm_eps,
            rope_theta: self.rope_parameters.rope_theta,
            // The card documents 32K although config.json declares 128K.
            max_context_length: self.max_position_embeddings.min(DOCUMENTED_CONTEXT_LENGTH),
            tie_word_embeddings: true,
            dtype: DType::BF16,
        };
        config.validate()?;

        let routing = MoeRoutingPolicy {
            experts: self.num_experts,
            top_k: self.num_experts_per_tok,
            scoring_function: MoeScoringFunction::Sigmoid,
            normalize_top_k_prob: self.norm_topk_prob,
            normalization_epsilon: 1e-6,
            routed_scaling_factor: self.routed_scaling_factor,
            use_expert_bias: self.use_expert_bias,
        };
        let layer_policies = self
            .layer_types
            .iter()
            .enumerate()
            .map(|(index, kind)| LayerPolicy {
                operator: match kind.as_str() {
                    "conv" => LayerOperatorPolicy::ShortConv {
                        kernel_size: self.conv_l_cache,
                    },
                    "full_attention" => LayerOperatorPolicy::Attention,
                    _ => unreachable!("layer types validated above"),
                },
                feed_forward: if index < self.num_dense_layers {
                    LayerFeedForwardPolicy::Dense {
                        intermediate_size: self.intermediate_size,
                    }
                } else {
                    LayerFeedForwardPolicy::Sparse {
                        routing,
                        intermediate_size: self.moe_intermediate_size,
                    }
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
                    message: "unmapped LFM2-MoE layer tensor".into(),
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
                    message: "unmapped LFM2-MoE convolution tensor".into(),
                });
            }
        };
        return Ok(format!("model.layers.{}.{component}.weight", parts[1]));
    }
    Err(Error::Weight {
        name: canonical.into(),
        message: "unmapped LFM2-MoE tensor".into(),
    })
}

fn add_expected_sources(
    canonical: &str,
    routing: Option<MoeRoutingPolicy>,
    expected: &mut BTreeSet<String>,
) -> Result<()> {
    let parts: Vec<_> = canonical.split('.').collect();
    if parts.len() == 5 && parts[0] == "layers" && parts[2] == "moe" {
        match parts[3] {
            "router" => {
                expected.insert(format!(
                    "model.layers.{}.feed_forward.gate.weight",
                    parts[1]
                ));
            }
            "router_bias" => {
                expected.insert(format!(
                    "model.layers.{}.feed_forward.expert_bias",
                    parts[1]
                ));
            }
            "input" | "output" => {
                let routing =
                    routing.ok_or_else(|| Error::Config("missing layer routing".into()))?;
                let components: &[&str] = if parts[3] == "input" {
                    &["w1", "w3"]
                } else {
                    &["w2"]
                };
                for expert in 0..routing.experts {
                    for component in components {
                        expected.insert(format!(
                            "model.layers.{}.feed_forward.experts.{expert}.{component}.weight",
                            parts[1]
                        ));
                    }
                }
            }
            _ => {
                return Err(Error::Weight {
                    name: canonical.into(),
                    message: "unmapped LFM2-MoE expert tensor".into(),
                });
            }
        }
    } else {
        expected.insert(official_name(canonical)?);
    }
    Ok(())
}

fn pack_experts(
    device: &MetalDevice,
    source: &mut Weights,
    layer: usize,
    routing: MoeRoutingPolicy,
    hidden_size: usize,
    intermediate_size: usize,
    input: bool,
) -> Result<Tensor> {
    let rows_per_expert = if input {
        intermediate_size
            .checked_mul(2)
            .ok_or_else(|| Error::Shape("expert input width overflow".into()))?
    } else {
        hidden_size
    };
    let columns = if input {
        hidden_size
    } else {
        intermediate_size
    };
    let elements = routing
        .experts
        .checked_mul(rows_per_expert)
        .and_then(|v| v.checked_mul(columns))
        .ok_or_else(|| Error::Shape("expert tensor size overflow".into()))?;
    let byte_len = elements
        .checked_mul(DType::BF16.size_bytes())
        .ok_or_else(|| Error::Shape("expert tensor byte size overflow".into()))?;
    let mut data = Vec::new();
    data.try_reserve_exact(byte_len)
        .map_err(|e| Error::Shape(format!("cannot allocate packed expert bytes: {e}")))?;

    for expert in 0..routing.experts {
        let components: &[&str] = if input { &["w1", "w3"] } else { &["w2"] };
        for component in components {
            let name =
                format!("model.layers.{layer}.feed_forward.experts.{expert}.{component}.weight");
            let shape = if component == &"w2" {
                vec![hidden_size, intermediate_size]
            } else {
                vec![intermediate_size, hidden_size]
            };
            let tensor = source.get(&name)?;
            if tensor.shape().dimensions() != shape || tensor.dtype() != DType::BF16 {
                return Err(Error::Weight {
                    name,
                    message: format!("expected BF16 tensor with shape {shape:?}"),
                });
            }
            let (buffer, offset, len) = tensor.binding();
            if len != byte_len / routing.experts / components.len() {
                return Err(Error::Weight {
                    name,
                    message: "expert byte count differs from configured shape".into(),
                });
            }
            buffer.with_bytes(offset, len, |bytes| data.extend_from_slice(bytes));
            drop(source.remove(&name)?);
        }
    }
    if data.len() != byte_len {
        return Err(Error::Shape("packed expert byte count mismatch".into()));
    }
    let shape = if input {
        vec![routing.experts, rows_per_expert, columns]
    } else {
        vec![routing.experts, hidden_size, intermediate_size]
    };
    Tensor::from_le_bytes(device, shape, DType::BF16, &data)
}

pub(crate) fn map_weights(
    device: &MetalDevice,
    config: &ModelConfig,
    policy: &ArchitecturePolicy,
    source: &mut Weights,
) -> Result<ModelWeights> {
    let specs = weights::specifications_with_policy(config, policy);
    let mut expected_sources = BTreeSet::new();
    for (canonical, _) in &specs {
        let parts: Vec<_> = canonical.split('.').collect();
        let layer_routing = if parts.len() > 1 && parts[0] == "layers" {
            let index = parts[1].parse::<usize>().map_err(|e| Error::Weight {
                name: canonical.clone(),
                message: format!("invalid layer index: {e}"),
            })?;
            match policy.layer(index, config.intermediate_size)?.feed_forward {
                LayerFeedForwardPolicy::Sparse { routing, .. } => Some(routing),
                LayerFeedForwardPolicy::Dense { .. } => None,
            }
        } else {
            None
        };
        add_expected_sources(canonical, layer_routing, &mut expected_sources)?;
    }
    for name in source.names() {
        if !expected_sources.contains(name) {
            return Err(Error::Weight {
                name: name.into(),
                message: "unexpected tensor for LFM2-MoE architecture".into(),
            });
        }
    }

    let mut result = ModelWeights::default();
    for (canonical, shape) in specs {
        let parts: Vec<_> = canonical.split('.').collect();
        let layer = if parts.first() == Some(&"layers") {
            parts[1].parse::<usize>().map_err(|e| Error::Weight {
                name: canonical.clone(),
                message: format!("invalid layer index: {e}"),
            })?
        } else {
            usize::MAX
        };
        match parts.as_slice() {
            ["layers", _, "moe", "input", "weight"] => {
                let LayerFeedForwardPolicy::Sparse {
                    routing,
                    intermediate_size,
                } = policy.layer(layer, config.intermediate_size)?.feed_forward
                else {
                    return Err(Error::Config(
                        "expert tensor assigned to a dense layer".into(),
                    ));
                };
                let tensor = pack_experts(
                    device,
                    source,
                    layer,
                    routing,
                    config.hidden_size,
                    intermediate_size,
                    true,
                )?;
                result.insert(canonical, ModelWeight::Dense(tensor))?;
            }
            ["layers", _, "moe", "output", "weight"] => {
                let LayerFeedForwardPolicy::Sparse {
                    routing,
                    intermediate_size,
                } = policy.layer(layer, config.intermediate_size)?.feed_forward
                else {
                    return Err(Error::Config(
                        "expert tensor assigned to a dense layer".into(),
                    ));
                };
                let tensor = pack_experts(
                    device,
                    source,
                    layer,
                    routing,
                    config.hidden_size,
                    intermediate_size,
                    false,
                )?;
                result.insert(canonical, ModelWeight::Dense(tensor))?;
            }
            ["layers", _, "moe", "router", "weight"] => {
                let source_name = format!("model.layers.{layer}.feed_forward.gate.weight");
                let tensor = weights::checked(source, &source_name, &shape, config, device)?;
                result.insert(canonical, ModelWeight::Dense(tensor))?;
            }
            ["layers", _, "moe", "router_bias", "weight"] => {
                let source_name = format!("model.layers.{layer}.feed_forward.expert_bias");
                let tensor = source.get(&source_name)?;
                if tensor.dtype() != DType::F32
                    || tensor.shape().dimensions() != shape
                    || !device.owns(tensor.buffer())
                {
                    return Err(Error::Weight {
                        name: source_name,
                        message: format!("expected F32 expert bias with shape {shape:?}"),
                    });
                }
                result.insert(canonical, ModelWeight::Dense(tensor.clone()))?;
            }
            _ => {
                let source_name = official_name(&canonical)?;
                let tensor = weights::checked(source, &source_name, &shape, config, device)?;
                result.insert(canonical, ModelWeight::Dense(tensor))?;
            }
        }
    }
    Ok(result)
}

pub struct LoadedLfm2Moe {
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

pub fn load(device: &MetalDevice, dir: impl AsRef<Path>) -> Result<LoadedLfm2Moe> {
    let dir = dir.as_ref();
    let started = Instant::now();
    let source_config = Lfm2MoeConfig::from_file(dir.join("config.json"))?;
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
    if tokenizer_config["backend"] != "tokenizers"
        || tokenizer_config["use_default_system_prompt"] != false
        || tokenizer_config["bos_token"] != "<|startoftext|>"
        || tokenizer_config["eos_token"] != "<|im_end|>"
        || tokenizer_config["pad_token"] != "<|pad|>"
        || generation_config["bos_token_id"] != source_config.bos_token_id
        || generation_config["eos_token_id"] != source_config.eos_token_id
        || generation_config["pad_token_id"] != source_config.pad_token_id
    {
        return Err(Error::Tokenizer(
            "LFM2-MoE tokenizer/generation policy mismatch".into(),
        ));
    }
    let tokenizer = Tokenizer::from_file(dir.join("tokenizer.json"))?;
    if tokenizer.vocab_size() > config.vocab_size
        || tokenizer.encode("<|startoftext|>")? != [source_config.bos_token_id]
        || tokenizer.encode("<|im_end|>")? != [source_config.eos_token_id]
        || tokenizer.encode("<|pad|>")? != [source_config.pad_token_id]
    {
        return Err(Error::Tokenizer(
            "LFM2-MoE special-token IDs differ from config".into(),
        ));
    }
    let config_tokenizer_load = started.elapsed();

    let started = Instant::now();
    let mut source = Weights::from_directory(device, dir)?;
    let source_tensor_bytes = source.bytes();
    let tensor_count = source.names().count();
    let weight_load = started.elapsed();

    let started = Instant::now();
    let bf16_expert_bias_bytes = if source_config.use_expert_bias {
        (source_config.num_hidden_layers - source_config.num_dense_layers)
            .checked_mul(source_config.num_experts)
            .and_then(|count| count.checked_mul(DType::F32.size_bytes()))
            .ok_or_else(|| Error::Shape("expert bias byte count overflow".into()))?
    } else {
        0
    };
    let parameter_count = source_tensor_bytes
        .checked_sub(bf16_expert_bias_bytes)
        .filter(|bytes| bytes.is_multiple_of(DType::BF16.size_bytes()))
        .ok_or_else(|| Error::Safetensors("unexpected LFM2-MoE parameter byte count".into()))?
        / DType::BF16.size_bytes();
    let mapped = map_weights(device, &config, &policy, &mut source)?;
    let model =
        Transformer::from_model_weights_with_policy(device, config.clone(), policy, &mapped)?;
    let construction = started.elapsed();

    Ok(LoadedLfm2Moe {
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

#[cfg(test)]
mod tests {
    use super::{MoeRoutingPolicy, pack_experts};
    use crate::{DType, MetalDevice, loader::Weights};

    #[test]
    fn packed_expert_weights_preserve_official_gate_up_down_order() {
        let device = MetalDevice::new().unwrap();
        let source_values = [
            ("experts.0.w1.weight", vec![1., 2.]),
            ("experts.0.w3.weight", vec![3., 4.]),
            ("experts.0.w2.weight", vec![5., 6.]),
            ("experts.1.w1.weight", vec![7., 8.]),
            ("experts.1.w3.weight", vec![9., 10.]),
            ("experts.1.w2.weight", vec![11., 12.]),
        ];
        let mut owned = Vec::new();
        for (name, values) in source_values {
            let data = values
                .iter()
                .flat_map(|value| half::bf16::from_f32(*value).to_bits().to_le_bytes())
                .collect::<Vec<_>>();
            let (shape, name) = if name.ends_with("w2.weight") {
                (vec![2, 1], format!("model.layers.2.feed_forward.{name}"))
            } else {
                (vec![1, 2], format!("model.layers.2.feed_forward.{name}"))
            };
            owned.push((name, shape, data));
        }
        let views = owned
            .iter()
            .map(|(name, shape, data)| {
                (
                    name.as_str(),
                    safetensors::tensor::TensorView::new(
                        safetensors::Dtype::BF16,
                        shape.clone(),
                        data,
                    )
                    .unwrap(),
                )
            })
            .collect::<Vec<_>>();
        let bytes = safetensors::serialize(views, None).unwrap();
        let mut source = Weights::from_bytes(&device, &bytes).unwrap();
        let routing = MoeRoutingPolicy::softmax(2, 1);

        let input = pack_experts(&device, &mut source, 2, routing, 2, 1, true).unwrap();
        assert_eq!(input.shape().dimensions(), [2, 2, 2]);
        assert_eq!(input.dtype(), DType::BF16);
        assert_eq!(input.to_f32(), [1., 2., 3., 4., 7., 8., 9., 10.]);
        let output = pack_experts(&device, &mut source, 2, routing, 2, 1, false).unwrap();
        assert_eq!(output.shape().dimensions(), [2, 2, 1]);
        assert_eq!(output.to_f32(), [5., 6., 11., 12.]);
        assert_eq!(source.names().count(), 0);
    }
}
