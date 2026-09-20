use crate::{DType, Error, Result, tensor::Shape};
#[derive(Debug, Clone, PartialEq)]
pub struct ModelConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub rms_norm_epsilon: f32,
    pub rope_theta: f32,
    pub max_context_length: usize,
    pub tie_word_embeddings: bool,
    pub dtype: DType,
}
impl ModelConfig {
    pub fn validate(&self) -> Result<()> {
        let c = self;
        if [
            c.vocab_size,
            c.hidden_size,
            c.intermediate_size,
            c.num_layers,
            c.num_attention_heads,
            c.num_key_value_heads,
            c.head_dim,
            c.max_context_length,
        ]
        .contains(&0)
        {
            return Err(Error::Config("dimensions must be positive".into()));
        }
        if !c.head_dim.is_multiple_of(2)
            || !c.num_attention_heads.is_multiple_of(c.num_key_value_heads)
        {
            return Err(Error::Config(
                "even head dimension and integral Q/KV head ratio required".into(),
            ));
        }
        if !c.rms_norm_epsilon.is_finite()
            || c.rms_norm_epsilon <= 0.
            || !c.rope_theta.is_finite()
            || c.rope_theta <= 0.
        {
            return Err(Error::Config(
                "epsilon and theta must be finite and positive".into(),
            ));
        }
        for dims in [
            vec![c.vocab_size, c.hidden_size],
            vec![c.intermediate_size, c.hidden_size],
            vec![c.num_attention_heads, c.head_dim, c.hidden_size],
            vec![c.max_context_length, c.num_attention_heads, c.head_dim],
            vec![c.max_context_length, c.max_context_length],
            vec![c.num_layers],
        ] {
            let shape = Shape::new(dims).map_err(|e| Error::Config(e.to_string()))?;
            if shape.numel() > u32::MAX as usize {
                return Err(Error::Config("dimensions exceed kernel indexing".into()));
            }
            shape
                .byte_size(c.dtype)
                .map_err(|e| Error::Config(e.to_string()))?;
        }
        Ok(())
    }
    pub fn tiny(dtype: DType) -> Self {
        Self {
            vocab_size: 32,
            hidden_size: 16,
            intermediate_size: 32,
            num_layers: 2,
            num_attention_heads: 4,
            num_key_value_heads: 2,
            head_dim: 4,
            rms_norm_epsilon: 1e-5,
            rope_theta: 10000.,
            max_context_length: 16,
            tie_word_embeddings: false,
            dtype,
        }
    }
}
