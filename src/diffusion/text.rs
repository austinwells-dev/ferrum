//! The prompt encoder: the language model of Qwen3-VL-8B, run on the hybrid
//! engine as an attention-only model (grouped-query attention with QK norm,
//! SwiGLU, no recurrent layers). The diffusion transformer conditions on the
//! hidden state *before* the final norm, so the last layer's output is read
//! from the engine's feature capture rather than from logits.
//!
//! Text-only prompts position every token with one index on all three
//! multimodal rotary axes, which makes Qwen3-VL's interleaved RoPE plain
//! RoPE. The vision tower is not loaded.
use super::{checkpoint::Checkpoint, config::TextConfig};
use crate::{
    DType, Error, MetalDevice, Result, Tensor,
    hybrid::{
        AttnSpec, HybridConfig, HybridModel, HybridState, KvCache, Output, Variant,
        engine::SpecGeometry,
        weights::{AttentionWeights, DenseFfn, Ffn, Format, Layer, Matrix, Mixer, Weights},
    },
};

pub struct TextEncoder {
    model: HybridModel,
    max_tokens: usize,
}

impl TextEncoder {
    /// Load the language model from `text_encoder/model.safetensors`, with
    /// the layer matrices quantized to `quantize` (Q8_0, Q4_0) as they are
    /// read; the embedding stays BF16. Prompts may be up to `max_tokens` long.
    pub fn load(
        d: &MetalDevice,
        ckpt: &Checkpoint,
        cfg: &TextConfig,
        max_tokens: usize,
        quantize: Option<Format>,
    ) -> Result<Self> {
        let config = HybridConfig {
            variant: Variant::Flint,
            name: "qwen3-vl-text".into(),
            vocab: cfg.vocab,
            hidden: cfg.hidden,
            layers: cfg.layers,
            attention_layers: vec![true; cfg.layers],
            attn: vec![
                AttnSpec {
                    window: None,
                    rope: true,
                    sinks: false,
                    gated: false,
                };
                cfg.layers
            ],
            leading_dense: cfg.layers,
            heads: cfg.heads,
            kv_heads: cfg.kv_heads,
            head_dim: cfg.head_dim,
            rope_dims: cfg.head_dim,
            rope_theta: cfg.rope_theta,
            swa_rope: None,
            rope_sections: [0; 4],
            eps: cfg.eps,
            ffn: cfg.ffn,
            gelu_ffn: false,
            conv_kernel: 4,
            ssm_head_dim: 0,
            ssm_k_heads: 0,
            ssm_v_heads: 0,
            ssm_grouped_heads: false,
            moe: None,
            trained_context: cfg.context,
            kv_cache: KvCache::F16,
        };
        config.validate()?;
        let (hidden, q, kv) = (
            cfg.hidden,
            cfg.heads * cfg.head_dim,
            cfg.kv_heads * cfg.head_dim,
        );
        let matrix = |name: &str, rows, cols| ckpt.matrix(d, name, rows, cols, quantize);
        let mut layers = Vec::with_capacity(cfg.layers);
        for i in 0..cfg.layers {
            let p = format!("model.language_model.layers.{i}.");
            let a = format!("{p}self_attn.");
            let vector = |name: String, len| ckpt.vector(d, &name, len, 0.);
            layers.push(Layer {
                attn_norm: vector(format!("{p}input_layernorm.weight"), hidden)?,
                post_norm: vector(format!("{p}post_attention_layernorm.weight"), hidden)?,
                mixer: Mixer::Attention(AttentionWeights {
                    q: matrix(&format!("{a}q_proj.weight"), q, hidden)?,
                    k: matrix(&format!("{a}k_proj.weight"), kv, hidden)?,
                    v: matrix(&format!("{a}v_proj.weight"), kv, hidden)?,
                    o: matrix(&format!("{a}o_proj.weight"), hidden, q)?,
                    q_norm: Some(vector(format!("{a}q_norm.weight"), cfg.head_dim)?),
                    k_norm: Some(vector(format!("{a}k_norm.weight"), cfg.head_dim)?),
                    sinks: None,
                    head_gate: None,
                }),
                ffn: Ffn::Dense(DenseFfn {
                    gate: matrix(&format!("{p}mlp.gate_proj.weight"), cfg.ffn, hidden)?,
                    up: matrix(&format!("{p}mlp.up_proj.weight"), cfg.ffn, hidden)?,
                    down: matrix(&format!("{p}mlp.down_proj.weight"), hidden, cfg.ffn)?,
                }),
            });
        }
        let weights = Weights {
            embedding: ckpt.matrix(
                d,
                "model.language_model.embed_tokens.weight",
                cfg.vocab,
                hidden,
                None,
            )?,
            output_norm: ckpt.vector(d, "model.language_model.norm.weight", hidden, 0.)?,
            // Logits are never computed, so the LM head is not read.
            output: Matrix::bf16_with(d, 1, hidden, |_| Ok(()))?,
            layers,
        };
        let model = HybridModel::new(d, config, weights, max_tokens, 1)?;
        model.enable_speculation(
            d,
            &SpecGeometry {
                rows: 2,
                aux_layers: vec![cfg.layers - 1],
                final_hidden: false,
            },
        )?;
        Ok(Self { model, max_tokens })
    }

    /// Weight bytes on the device.
    pub fn weight_bytes(&self) -> usize {
        self.model.weight_bytes()
    }

    /// The last layer's output (before the final norm) for `tokens`, dropping
    /// the first `skip` rows: F32 `[tokens.len() - skip, hidden]`.
    pub fn encode(&self, d: &MetalDevice, tokens: &[u32], skip: usize) -> Result<Tensor> {
        if tokens.len() > self.max_tokens || skip >= tokens.len() {
            return Err(Error::Parameter(format!(
                "{} prompt tokens (skipping {skip}) for an encoder of {} tokens",
                tokens.len(),
                self.max_tokens
            )));
        }
        let mut state = HybridState::new(
            d,
            &self.model.config,
            tokens.len(),
            self.model.max_forward_rows(),
        )?;
        self.model.forward(d, &mut state, tokens, Output::None)?;
        let spec = self.model.spec.borrow();
        let features = spec
            .as_ref()
            .and_then(|s| s.features.as_ref())
            .ok_or_else(|| Error::Parameter("feature capture is off".into()))?;
        let hidden = self.model.config.hidden;
        let rows = tokens.len() - skip;
        let kept = features.view(skip * hidden, [rows, hidden])?;
        Tensor::from_f32(d, [rows, hidden], DType::F32, &kept.to_f32())
    }
}
