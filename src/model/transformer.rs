use super::{
    ModelConfig,
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
        mut trace: Option<&mut Trace>,
    ) -> Result<Tensor> {
        let norm = self.input_norm.forward(d, x)?;
        record(&mut trace, format!("layer.{layer}.norm"), &norm);
        let attn = self
            .attention
            .forward(d, &norm, cache, layer, trace.as_deref_mut())?;
        let residual = d.add(x, &attn)?.tensor;
        let norm = self.post_norm.forward(d, &residual)?;
        record(&mut trace, format!("layer.{layer}.post_norm"), &norm);
        let mlp = self.mlp.forward(d, &norm)?;
        record(&mut trace, format!("layer.{layer}.mlp"), &mlp);
        let output = d.add(&residual, &mlp)?.tensor;
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
    weight_bytes: usize,
}
impl Transformer {
    pub fn from_weights(d: &MetalDevice, config: ModelConfig, w: &Weights) -> Result<Self> {
        config.validate()?;
        let (embedding, layers, final_norm, lm_head) = weights::construct(d, &config, w)?;
        // Sum actual retained tensor payloads, including the tied LM transpose allocation.
        let mut weight_bytes =
            embedding.weight_bytes() + final_norm.weight.byte_size() + lm_head.weight_bytes();
        for layer in &layers {
            weight_bytes +=
                layer.input_norm.weight.byte_size() + layer.post_norm.weight.byte_size();
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
            weight_bytes,
        })
    }
    pub fn config(&self) -> &ModelConfig {
        &self.config
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
        mut trace: Option<&mut Trace>,
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
        let mut staged = cache.clone();
        let mut x = self.embedding.forward(d, tokens)?;
        record(&mut trace, "embedding", &x);
        for (l, layer) in self.layers.iter().enumerate() {
            x = layer.forward(d, &x, &mut staged, l, trace.as_deref_mut())?;
        }
        let x = self.final_norm.forward(d, &x)?;
        record(&mut trace, "final_hidden", &x);
        let logits = self.lm_head.forward(d, &x)?;
        record(&mut trace, "logits", &logits);
        *cache = staged;
        Ok(logits)
    }
}
