//! The prompt encoder: the language model of Qwen3-VL-8B, run on the hybrid
//! engine as an attention-only model (grouped-query attention with QK norm,
//! SwiGLU, no recurrent layers). The diffusion transformer conditions on the
//! hidden state *before* the final norm, so the last layer's output is read
//! from the engine's feature capture rather than from logits.
//!
//! Condition images (editing) go through Qwen3-VL's vision tower, loaded
//! separately by `load_vision`; their embeddings replace the `<|image_pad|>`
//! rows, sit on a 2-D grid of the interleaved multimodal rotary axes, and
//! bring DeepStack features the first layers add to them. Text-only prompts
//! position every token with one index on all three axes, which makes the
//! rotation plain RoPE.
use super::{
    checkpoint::Checkpoint,
    config::{EncoderVision, TextConfig},
};
use crate::{
    DType, Error, MetalDevice, Result, Tensor,
    hybrid::{
        AttnSpec, HybridConfig, HybridModel, HybridState, KvCache, Output, Variant,
        engine::{ImageSpan, SpecGeometry},
        weights::{self, AttentionWeights, DenseFfn, Ffn, Format, Layer, Matrix, Mixer, Weights},
    },
    loader::gguf::{GgufFile, MetadataValue},
    vision::VisionModel,
};
use std::path::Path;

fn hybrid_config(cfg: &TextConfig) -> Result<HybridConfig> {
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
        rope_sections: [
            cfg.mrope_sections[0],
            cfg.mrope_sections[1],
            cfg.mrope_sections[2],
            0,
        ],
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
    Ok(config)
}

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
        let config = hybrid_config(cfg)?;
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
                post_attn: None,
                post_ffw: None,
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
            rope_freqs: None,
            layers,
        };
        Self::with_weights(d, config, weights, cfg, max_tokens)
    }

    /// Load the language model from a Qwen3-VL GGUF (`blk.N.attn_q.weight`
    /// and so on), keeping each tensor in the encoding it is stored in. The
    /// vision tower and the LM head in the file are not read.
    pub fn load_gguf(
        d: &MetalDevice,
        path: impl AsRef<Path>,
        cfg: &TextConfig,
        max_tokens: usize,
    ) -> Result<Self> {
        let config = hybrid_config(cfg)?;
        let mut file = GgufFile::open(path)?;
        check_gguf(&file, cfg)?;
        let weights = weights::load_without_head(d, &mut file, &config)?;
        Self::with_weights(d, config, weights, cfg, max_tokens)
    }

    fn with_weights(
        d: &MetalDevice,
        config: HybridConfig,
        weights: Weights,
        cfg: &TextConfig,
        max_tokens: usize,
    ) -> Result<Self> {
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

    /// Device bytes of an encoder for `max_tokens`-token prompts: weights,
    /// engine scratch, K/V cache and the feature capture.
    pub fn estimate_bytes(cfg: &TextConfig, max_tokens: usize, quantize: Option<Format>) -> usize {
        use crate::hybrid::engine::page;
        let stored = |rows, cols| Matrix::bf16_stored_bytes(rows, cols, quantize);
        let (h, q, kv) = (
            cfg.hidden,
            cfg.heads * cfg.head_dim,
            cfg.kv_heads * cfg.head_dim,
        );
        let layer = stored(q, h)
            + 2 * stored(kv, h)
            + stored(h, q)
            + 2 * stored(cfg.ffn, h)
            + stored(h, cfg.ffn)
            + 2 * page(h * 4)
            + 2 * page(cfg.head_dim * 4);
        let weights = Matrix::bf16_stored_bytes(cfg.vocab, h, None) + cfg.layers * layer;
        weights + Self::runtime_bytes(cfg, max_tokens)
    }

    /// The same for a GGUF at `path`: the tensors `load_gguf` reads, as stored.
    pub fn estimate_bytes_gguf(
        cfg: &TextConfig,
        max_tokens: usize,
        path: impl AsRef<Path>,
    ) -> Result<usize> {
        use crate::hybrid::engine::page;
        let file = GgufFile::open(path)?;
        let loaded = |name: &str| {
            let tail = name
                .strip_prefix("blk.")
                .map_or(name, |n| n.split_once('.').map_or(n, |(_, tail)| tail));
            matches!(
                tail,
                "token_embd.weight"
                    | "output_norm.weight"
                    | "attn_q.weight"
                    | "attn_k.weight"
                    | "attn_v.weight"
                    | "attn_output.weight"
                    | "ffn_gate.weight"
                    | "ffn_up.weight"
                    | "ffn_down.weight"
                    | "attn_norm.weight"
                    | "ffn_norm.weight"
                    | "attn_q_norm.weight"
                    | "attn_k_norm.weight"
            )
        };
        let weights: usize = file
            .tensors()
            .iter()
            .filter(|(name, _)| loaded(name))
            .map(|(_, t)| match t.dimensions.len() {
                1 => page(t.dimensions[0] * 4),
                _ => page(t.byte_len),
            })
            .sum();
        Ok(weights + Self::runtime_bytes(cfg, max_tokens))
    }

    /// Engine scratch, K/V cache and feature capture.
    fn runtime_bytes(cfg: &TextConfig, max_tokens: usize) -> usize {
        use crate::hybrid::engine::{page, scratch_bytes};
        let kv = cfg.kv_heads * cfg.head_dim;
        let config = hybrid_config(cfg).expect("a validated geometry");
        let kv_cache = cfg.layers * 2 * page(max_tokens.next_multiple_of(32) * kv * 2);
        // The captured features, the rows `encode` copies out of them and
        // the text rows the pipeline then takes from those.
        let rows = 3 * page(max_tokens.max(2) * cfg.hidden * 4);
        scratch_bytes(&config, max_tokens, 1) + kv_cache + rows
    }

    /// Weight bytes on the device.
    pub fn weight_bytes(&self) -> usize {
        self.model.weight_bytes()
    }

    /// The last layer's output (before the final norm) for `tokens`, dropping
    /// the first `skip` rows: F32 `[tokens.len() - skip, hidden]`. `media`
    /// places image embeddings over their pad tokens.
    pub fn encode(
        &self,
        d: &MetalDevice,
        tokens: &[u32],
        skip: usize,
        media: Vec<ImageSpan>,
    ) -> Result<Tensor> {
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
        state.set_media(media);
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

/// The language model a GGUF holds must be Qwen3-VL's of this geometry.
fn check_gguf(file: &GgufFile, cfg: &TextConfig) -> Result<()> {
    let arch = match file.metadata_value("general.architecture") {
        Some(MetadataValue::String(a)) => a.clone(),
        _ => String::new(),
    };
    let uint = |key: &str| match file.metadata_value(&format!("{arch}.{key}")) {
        Some(MetadataValue::Uint32(v)) => Some(*v as usize),
        Some(MetadataValue::Uint64(v)) => Some(*v as usize),
        _ => None,
    };
    if arch != "qwen3vl"
        || uint("embedding_length") != Some(cfg.hidden)
        || uint("block_count") != Some(cfg.layers)
    {
        return Err(Error::Config(format!(
            "the text encoder GGUF is a {arch:?} model of width {:?} and {:?} layers, not Qwen3-VL's {} x {}",
            uint("embedding_length"),
            uint("block_count"),
            cfg.hidden,
            cfg.layers
        )));
    }
    Ok(())
}

/// The encoder's vision tower from `text_encoder/model.safetensors`.
pub fn load_vision(
    d: &MetalDevice,
    ckpt: &Checkpoint,
    vision: &EncoderVision,
) -> Result<VisionModel> {
    VisionModel::load_safetensors(
        d,
        ckpt,
        "model.visual.",
        vision.config.clone(),
        &vision.deepstack,
        usize::MAX,
    )
}

/// Device bytes `load_vision` and one image's encoding take: F16 matrices,
/// the position table, and the activations and attention scores of
/// `tokens` merged tokens.
pub fn vision_bytes(vision: &EncoderVision, tokens: usize) -> usize {
    let c = &vision.config;
    let (h, merged) = (c.hidden, c.hidden * c.merge * c.merge);
    let mergers = 1 + vision.deepstack.len();
    let matrices = c.layers * (4 * h * h + 2 * h * c.ffn)
        + h * 3 * c.patch * c.patch
        + mergers * (merged * merged + c.out_dim * merged);
    // What `encode_prepared` holds at once (its execution keeps every
    // buffer until the end): per patch the pixels, rotary tables, position
    // rows, the block's seven hidden-wide rows and its MLP row, and each
    // merger's normed and middle rows; then the merged outputs, and the
    // attention scores, which hold at least one head.
    let patches = tokens * c.merge * c.merge;
    let per_patch = 3 * c.patch * c.patch + h / c.heads + 8 * h + c.ffn + mergers * 2 * h;
    let scores = crate::vision::SCORE_BYTES.max(patches * patches * 4);
    let activations = patches * per_patch * 4 + scores + mergers * tokens * c.out_dim * 4;
    matrices * 2 + c.table * c.table * h * 4 + activations
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::diffusion::checkpoint::tests::{Meta, write};

    #[test]
    fn only_a_qwen3_vl_gguf_of_the_encoders_geometry_is_taken() {
        let cfg = TextConfig::from_json(
            r#"{"hidden_size":4096,"num_hidden_layers":36,"num_attention_heads":32,
            "num_key_value_heads":8,"head_dim":128,"intermediate_size":12288,
            "vocab_size":151936,"rope_theta":5000000,"max_position_embeddings":262144}"#,
        )
        .unwrap();
        let path = std::env::temp_dir().join(format!("ferrum-te-{}.gguf", std::process::id()));
        let check = |arch: &'static str, width, layers| {
            let (w, l) = (
                format!("{arch}.embedding_length"),
                format!("{arch}.block_count"),
            );
            let meta = [
                ("general.architecture", Meta::Str(arch)),
                (w.as_str(), Meta::U32(width)),
                (l.as_str(), Meta::U32(layers)),
            ];
            write(&path, &meta, &[]);
            check_gguf(&GgufFile::open(&path).unwrap(), &cfg)
        };
        assert!(check("qwen3vl", 4096, 36).is_ok());
        // Qwen3-8B has the same geometry but no image tokens or DeepStack.
        assert!(check("qwen3", 4096, 36).is_err());
        assert!(check("qwen3vl", 2560, 36).is_err());
        assert!(check("qwen3vl", 4096, 28).is_err());
        // No geometry to check is a refusal too.
        write(
            &path,
            &[("general.architecture", Meta::Str("qwen3vl"))],
            &[],
        );
        assert!(check_gguf(&GgufFile::open(&path).unwrap(), &cfg).is_err());
        std::fs::remove_file(path).unwrap();
    }
}
