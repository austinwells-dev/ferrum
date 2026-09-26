use super::{
    ModelConfig,
    architecture::{ArchitecturePolicy, ResidualTopology},
    weights::{self, DecoderLayer},
};
use crate::{
    Error, MetalDevice, Result, Tensor,
    loader::Weights,
    nn::{
        Embedding, Linear, RmsNorm,
        attention::{Trace, record},
        kv_cache::{CacheLayerKind, KvCache},
    },
};
impl DecoderLayer {
    fn apply_operator(
        &self,
        d: &MetalDevice,
        x: &Tensor,
        cache: &mut KvCache,
        layer: usize,
        mut trace: Option<&mut Trace>,
    ) -> Result<Tensor> {
        match &self.operator {
            weights::LayerOperator::Attention(attention) => {
                attention.forward(d, x, cache, layer, trace)
            }
            weights::LayerOperator::ShortConv(convolution) => {
                let output = convolution.forward(d, x, cache, layer)?;
                record(&mut trace, format!("layer.{layer}.short_conv"), &output);
                Ok(output)
            }
        }
    }

    pub fn forward(
        &self,
        d: &MetalDevice,
        x: &Tensor,
        cache: &mut KvCache,
        layer: usize,
        policy: &ArchitecturePolicy,
        mut trace: Option<&mut Trace>,
    ) -> Result<Tensor> {
        let output = match policy.residual_topology {
            ResidualTopology::PreNorm => {
                let norm = self.input_norm.forward(d, x)?;
                record(&mut trace, format!("layer.{layer}.norm"), &norm);
                let operator = self.apply_operator(d, &norm, cache, layer, trace.as_deref_mut())?;
                let operator = if policy.residual_multiplier == 1. {
                    operator
                } else {
                    d.scale(&operator, policy.residual_multiplier)?.tensor
                };
                let residual = d.add(x, &operator)?.tensor;
                let norm = self.post_norm.forward(d, &residual)?;
                record(&mut trace, format!("layer.{layer}.post_norm"), &norm);
                let mlp = self.feed_forward.forward(d, &norm)?;
                record(&mut trace, format!("layer.{layer}.mlp"), &mlp);
                let mlp = if policy.residual_multiplier == 1. {
                    mlp
                } else {
                    d.scale(&mlp, policy.residual_multiplier)?.tensor
                };
                d.add(&residual, &mlp)?.tensor
            }
            ResidualTopology::PostNorm => {
                let operator = self.apply_operator(d, x, cache, layer, trace.as_deref_mut())?;
                let operator = self.input_norm.forward(d, &operator)?;
                record(&mut trace, format!("layer.{layer}.norm"), &operator);
                let residual = d.add(x, &operator)?.tensor;
                let mlp = self.feed_forward.forward(d, &residual)?;
                let mlp = self.post_norm.forward(d, &mlp)?;
                record(&mut trace, format!("layer.{layer}.mlp"), &mlp);
                d.add(&residual, &mlp)?.tensor
            }
        };
        record(&mut trace, format!("layer.{layer}.output"), &output);
        Ok(output)
    }
}
pub struct Transformer {
    embedding: Embedding,
    layers: Vec<DecoderLayer>,
    final_norm: RmsNorm,
    lm_head: Linear,
    config: ModelConfig,
    policy: ArchitecturePolicy,
    weight_bytes: usize,
    // Measured on M5: dense models gain from one shared concurrent encoder per
    // forward; sparse-MoE models lose prefill latency, so they keep one per kernel.
    shared_encoder: bool,
}
impl Transformer {
    pub fn from_weights(d: &MetalDevice, config: ModelConfig, w: &Weights) -> Result<Self> {
        Self::from_weights_with_policy(d, config, ArchitecturePolicy::default(), w)
    }
    pub fn from_weights_with_policy(
        d: &MetalDevice,
        config: ModelConfig,
        policy: ArchitecturePolicy,
        w: &Weights,
    ) -> Result<Self> {
        config.validate()?;
        policy.validate_for(&config)?;
        let (embedding, layers, final_norm, lm_head) = weights::construct(d, &config, &policy, w)?;
        Self::from_parts(config, policy, embedding, layers, final_norm, lm_head)
    }
    pub(crate) fn from_model_weights(
        d: &MetalDevice,
        config: ModelConfig,
        w: &weights::ModelWeights,
    ) -> Result<Self> {
        Self::from_model_weights_with_policy(d, config, ArchitecturePolicy::default(), w)
    }
    pub(crate) fn from_model_weights_with_policy(
        d: &MetalDevice,
        config: ModelConfig,
        policy: ArchitecturePolicy,
        w: &weights::ModelWeights,
    ) -> Result<Self> {
        config.validate()?;
        policy.validate_for(&config)?;
        let (embedding, layers, final_norm, lm_head) =
            weights::construct_mixed(d, &config, &policy, w)?;
        Self::from_parts(config, policy, embedding, layers, final_norm, lm_head)
    }
    fn from_parts(
        config: ModelConfig,
        policy: ArchitecturePolicy,
        embedding: Embedding,
        layers: Vec<DecoderLayer>,
        final_norm: RmsNorm,
        lm_head: Linear,
    ) -> Result<Self> {
        // Sum actual retained tensor payloads, including the tied LM transpose allocation.
        let mut weight_bytes =
            embedding.weight_bytes() + final_norm.weight.byte_size() + lm_head.weight_bytes();
        if config.tie_word_embeddings {
            weight_bytes -= embedding.weight_bytes();
        }
        for layer in &layers {
            weight_bytes +=
                layer.input_norm.weight.byte_size() + layer.post_norm.weight.byte_size();
            weight_bytes += layer.operator.weight_bytes() + layer.feed_forward.weight_bytes();
        }

        let shared_encoder = !layers
            .iter()
            .any(|layer| matches!(layer.feed_forward, weights::FeedForward::Sparse(_)));
        Ok(Self {
            shared_encoder,
            embedding,
            layers,
            final_norm,
            lm_head,
            config,
            policy,
            weight_bytes,
        })
    }
    pub fn config(&self) -> &ModelConfig {
        &self.config
    }
    pub fn policy(&self) -> &ArchitecturePolicy {
        &self.policy
    }
    pub fn weight_bytes(&self) -> usize {
        self.weight_bytes
    }
    pub fn take_moe_stats(&self) -> crate::nn::moe::MoeStats {
        self.layers
            .iter()
            .fold(Default::default(), |mut total, layer| {
                if let Some(stats) = layer.feed_forward.take_moe_stats() {
                    total.router_projection_enqueue += stats.router_projection_enqueue;
                    total.routing += stats.routing;
                    total.routing_boundary_wait += stats.routing_boundary_wait;
                    total.expert_dispatch += stats.expert_dispatch;
                    total.combine_dispatch += stats.combine_dispatch;
                    total.active_experts += stats.active_experts;
                    total.active_experts_known &= stats.active_experts_known;
                    total.assignments += stats.assignments;
                    total.peak_temporary_bytes =
                        total.peak_temporary_bytes.max(stats.peak_temporary_bytes);
                }
                total
            })
    }
    pub fn new_cache(&self) -> Result<KvCache> {
        let c = &self.config;
        let kinds = (0..c.num_layers)
            .map(
                |layer| match self.policy.layer(layer, c.intermediate_size)?.operator {
                    super::architecture::LayerOperatorPolicy::Attention => {
                        Ok(CacheLayerKind::KeyValue)
                    }
                    super::architecture::LayerOperatorPolicy::ShortConv { kernel_size } => {
                        Ok(CacheLayerKind::Convolution { kernel_size })
                    }
                },
            )
            .collect::<Result<Vec<_>>>()?;
        KvCache::with_layout(
            c.num_layers,
            c.max_context_length,
            c.num_key_value_heads,
            c.head_dim,
            c.hidden_size,
            c.dtype,
            kinds,
        )
    }
    /// Empty-cache prefill. Returns all position logits [S,vocab] and populated cache.
    pub fn forward_prefill(&self, d: &MetalDevice, tokens: &[u32]) -> Result<(Tensor, KvCache)> {
        let mut cache = self.new_cache()?;
        let logits = self.forward(d, tokens, &mut cache, None)?;
        Ok((logits, cache))
    }
    /// Generation prefill computes only the requested final-position logits.
    /// All transformer positions and cache updates still execute inside this call.
    pub fn forward_prefill_last(
        &self,
        d: &MetalDevice,
        tokens: &[u32],
    ) -> Result<(Tensor, KvCache)> {
        let mut cache = self.new_cache()?;
        let logits = self.forward_impl(d, tokens, &mut cache, None, true, false)?;
        Ok((logits, cache))
    }
    /// Greedy prefill: final-position argmax selected on Metal in the same
    /// submission. Returns the `[1,2]` carrier from `argmax_rows`.
    pub(crate) fn forward_prefill_argmax(
        &self,
        d: &MetalDevice,
        tokens: &[u32],
    ) -> Result<(Tensor, KvCache)> {
        let mut cache = self.new_cache()?;
        let selected = self.forward_impl(d, tokens, &mut cache, None, true, true)?;
        Ok((selected, cache))
    }
    /// Greedy decode step with on-device argmax; see `forward_prefill_argmax`.
    pub(crate) fn forward_decode_argmax(
        &self,
        d: &MetalDevice,
        token: u32,
        cache: &mut KvCache,
    ) -> Result<Tensor> {
        self.forward_impl(d, &[token], cache, None, true, true)
    }
    /// One new token, returning [1,vocab]; an empty cache is also supported.
    pub fn forward_decode(
        &self,
        d: &MetalDevice,
        token: u32,
        cache: &mut KvCache,
    ) -> Result<Tensor> {
        self.forward(d, &[token], cache, None)
    }
    /// Supports chunked appends. Cache publication is atomic with respect to errors.
    pub fn forward(
        &self,
        d: &MetalDevice,
        tokens: &[u32],
        cache: &mut KvCache,
        trace: Option<&mut Trace>,
    ) -> Result<Tensor> {
        self.forward_impl(d, tokens, cache, trace, false, false)
    }
    fn forward_impl(
        &self,
        d: &MetalDevice,
        tokens: &[u32],
        cache: &mut KvCache,
        mut trace: Option<&mut Trace>,
        last_only: bool,
        argmax: bool,
    ) -> Result<Tensor> {
        if tokens.is_empty() {
            return Err(Error::Parameter(
                "transformer requires at least one token".into(),
            ));
        }
        let offset = cache.validate_for(&self.config, tokens.len())?;
        let expected_layout = (0..self.config.num_layers)
            .map(
                |layer| match self.policy.layer(layer, self.config.intermediate_size)? {
                    super::architecture::LayerPolicy {
                        operator: super::architecture::LayerOperatorPolicy::Attention,
                        ..
                    } => Ok(CacheLayerKind::KeyValue),
                    super::architecture::LayerPolicy {
                        operator:
                            super::architecture::LayerOperatorPolicy::ShortConv { kernel_size },
                        ..
                    } => Ok(CacheLayerKind::Convolution { kernel_size }),
                },
            )
            .collect::<Result<Vec<_>>>()?;
        cache.validate_layout(&expected_layout)?;
        for &id in tokens {
            if id as usize >= self.config.vocab_size {
                return Err(Error::Token {
                    id,
                    vocab: self.config.vocab_size,
                });
            }
        }
        let execution = d.execution_with_shared_encoder(self.shared_encoder)?;
        let mut staged = cache.clone();
        let mut x = self.embedding.forward(d, tokens)?;
        if self.policy.embedding_multiplier != 1. {
            x = d.scale(&x, self.policy.embedding_multiplier)?.tensor;
        }
        record(&mut trace, "embedding", &x);
        for (l, layer) in self.layers.iter().enumerate() {
            x = layer.forward(d, &x, &mut staged, l, &self.policy, trace.as_deref_mut())?;
        }
        let x = self.final_norm.forward(d, &x)?;
        record(&mut trace, "final_hidden", &x);
        let head_input = if last_only {
            x.view(
                (tokens.len() - 1) * self.config.hidden_size,
                [1, self.config.hidden_size],
            )?
        } else {
            x
        };
        let logits = d.profile_projection("lm_head", || self.lm_head.forward(d, &head_input))?;
        let logits = if self.policy.logits_divisor == 1. {
            logits
        } else {
            d.scale(&logits, self.policy.logits_divisor.recip())?.tensor
        };
        record(&mut trace, "logits", &logits);
        let logits = if argmax {
            d.argmax_rows(&logits)?
        } else {
            logits
        };
        staged.set_sequence_len(offset + tokens.len())?;
        execution.finish()?;
        *cache = staged;
        Ok(logits)
    }
}

#[cfg(test)]
mod hybrid_state_tests {
    use super::*;
    use crate::model::architecture::{LayerFeedForwardPolicy, LayerOperatorPolicy, LayerPolicy};

    #[test]
    fn failed_short_conv_forward_keeps_published_state_and_cache_retryable() {
        let device = MetalDevice::new().unwrap();
        let config = ModelConfig::tiny(crate::DType::BF16);
        let policy = ArchitecturePolicy {
            layer_policies: Some(vec![
                LayerPolicy {
                    operator: LayerOperatorPolicy::ShortConv { kernel_size: 3 },
                    feed_forward: LayerFeedForwardPolicy::Dense {
                        intermediate_size: config.intermediate_size,
                    },
                },
                LayerPolicy {
                    operator: LayerOperatorPolicy::Attention,
                    feed_forward: LayerFeedForwardPolicy::Dense {
                        intermediate_size: config.intermediate_size,
                    },
                },
            ]),
            ..ArchitecturePolicy::default()
        };
        let mut model_weights = weights::ModelWeights::default();
        for (name, shape) in weights::specifications_with_policy(&config, &policy) {
            let count = shape.iter().product();
            let values = vec![0.01; count];
            let tensor = Tensor::from_f32(&device, &shape, config.dtype, &values).unwrap();
            model_weights
                .insert(name, weights::ModelWeight::Dense(tensor))
                .unwrap();
        }
        let model =
            Transformer::from_model_weights_with_policy(&device, config, policy, &model_weights)
                .unwrap();
        let (_, mut cache) = model.forward_prefill_last(&device, &[1, 2]).unwrap();
        let before_len = cache.len().unwrap();
        let before_state = cache.conv_state(0).unwrap().unwrap().to_f32();
        assert_eq!(before_len, 2);

        // The first six dispatches cover token embedding, operator norm,
        // in-projection, B/C/X split, B*X, and short convolution/state output.
        // The following output projection fails after new state was staged.
        device.inject_dispatch_failure_after(Some(device.counters().dispatches + 6));
        assert!(model.forward_decode(&device, 3, &mut cache).is_err());
        device.inject_dispatch_failure_after(None);

        assert_eq!(cache.len().unwrap(), before_len);
        assert_eq!(cache.conv_state(0).unwrap().unwrap().to_f32(), before_state);
        let mut branch_a = cache.clone();
        let mut branch_b = cache.clone();
        let logits_a = model.forward_decode(&device, 3, &mut branch_a).unwrap();
        let logits_b = model.forward_decode(&device, 3, &mut branch_b).unwrap();
        assert_eq!(logits_a.to_f32(), logits_b.to_f32());
    }
}
