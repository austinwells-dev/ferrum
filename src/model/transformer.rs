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
        kv_cache::KvCache,
    },
};
impl DecoderLayer {
    pub fn forward(
        &self,
        d: &MetalDevice,
        x: &Tensor,
        cache: &mut KvCache,
        layer: usize,
        policy: ArchitecturePolicy,
        mut trace: Option<&mut Trace>,
    ) -> Result<Tensor> {
        let output = match policy.residual_topology {
            ResidualTopology::PreNorm => {
                let norm = self.input_norm.forward(d, x)?;
                record(&mut trace, format!("layer.{layer}.norm"), &norm);
                let attn = self
                    .attention
                    .forward(d, &norm, cache, layer, trace.as_deref_mut())?;
                let attn = if policy.residual_multiplier == 1. {
                    attn
                } else {
                    d.scale(&attn, policy.residual_multiplier)?.tensor
                };
                let residual = d.add(x, &attn)?.tensor;
                let norm = self.post_norm.forward(d, &residual)?;
                record(&mut trace, format!("layer.{layer}.post_norm"), &norm);
                let mlp = self.mlp.forward(d, &norm)?;
                record(&mut trace, format!("layer.{layer}.mlp"), &mlp);
                let mlp = if policy.residual_multiplier == 1. {
                    mlp
                } else {
                    d.scale(&mlp, policy.residual_multiplier)?.tensor
                };
                d.add(&residual, &mlp)?.tensor
            }
            ResidualTopology::PostNorm => {
                let attn = self
                    .attention
                    .forward(d, x, cache, layer, trace.as_deref_mut())?;
                let attn = self.input_norm.forward(d, &attn)?;
                record(&mut trace, format!("layer.{layer}.norm"), &attn);
                let residual = d.add(x, &attn)?.tensor;
                let mlp = self.mlp.forward(d, &residual)?;
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
        policy.validate()?;
        let (embedding, layers, final_norm, lm_head) = weights::construct(d, &config, policy, w)?;
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
        policy.validate()?;
        let (embedding, layers, final_norm, lm_head) =
            weights::construct_mixed(d, &config, policy, w)?;
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
            weight_bytes += layer
                .attention
                .q_norm
                .as_ref()
                .map_or(0, |norm| norm.weight.byte_size());
            weight_bytes += layer
                .attention
                .k_norm
                .as_ref()
                .map_or(0, |norm| norm.weight.byte_size());
            for linear in [
                &layer.attention.q,
                &layer.attention.k,
                &layer.attention.v,
                &layer.attention.output,
                &layer.mlp.gate,
                &layer.mlp.up,
                &layer.mlp.down,
            ] {
                weight_bytes += linear.weight_bytes();
            }
        }

        Ok(Self {
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
    pub fn policy(&self) -> ArchitecturePolicy {
        self.policy
    }
    pub fn weight_bytes(&self) -> usize {
        self.weight_bytes
    }
    pub fn new_cache(&self) -> Result<KvCache> {
        let c = &self.config;
        KvCache::new(
            c.num_layers,
            c.max_context_length,
            c.num_key_value_heads,
            c.head_dim,
            c.dtype,
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
        let logits = self.forward_impl(d, tokens, &mut cache, None, true)?;
        Ok((logits, cache))
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
        self.forward_impl(d, tokens, cache, trace, false)
    }
    fn forward_impl(
        &self,
        d: &MetalDevice,
        tokens: &[u32],
        cache: &mut KvCache,
        mut trace: Option<&mut Trace>,
        last_only: bool,
    ) -> Result<Tensor> {
        if tokens.is_empty() {
            return Err(Error::Parameter(
                "transformer requires at least one token".into(),
            ));
        }
        cache.validate_for(&self.config, tokens.len())?;
        for &id in tokens {
            if id as usize >= self.config.vocab_size {
                return Err(Error::Token {
                    id,
                    vocab: self.config.vocab_size,
                });
            }
        }
        let execution = d.execution()?;
        let mut staged = cache.clone();
        let mut x = self.embedding.forward(d, tokens)?;
        if self.policy.embedding_multiplier != 1. {
            x = d.scale(&x, self.policy.embedding_multiplier)?.tensor;
        }
        record(&mut trace, "embedding", &x);
        for (l, layer) in self.layers.iter().enumerate() {
            x = layer.forward(d, &x, &mut staged, l, self.policy, trace.as_deref_mut())?;
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
        execution.finish()?;
        *cache = staged;
        Ok(logits)
    }
}
