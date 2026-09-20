use super::ModelConfig;
use crate::{
    Error, MetalDevice, Result, Tensor,
    loader::Weights,
    nn::{Embedding, Linear, Mlp, RmsNorm, attention::Attention},
};
pub struct DecoderLayer {
    pub input_norm: RmsNorm,
    pub attention: Attention,
    pub post_norm: RmsNorm,
    pub mlp: Mlp,
}
pub(crate) fn specifications(c: &ModelConfig) -> Vec<(String, Vec<usize>)> {
    let h = c.hidden_size;
    let q = c.num_attention_heads * c.head_dim;
    let k = c.num_key_value_heads * c.head_dim;
    let i = c.intermediate_size;
    let mut specs = vec![
        ("embedding.weight".into(), vec![c.vocab_size, h]),
        ("final_norm.weight".into(), vec![h]),
    ];
    if !c.tie_word_embeddings {
        specs.push(("lm_head.weight".into(), vec![c.vocab_size, h]));
    }
    for l in 0..c.num_layers {
        for (name, shape) in [
            ("input_norm.weight", vec![h]),
            ("post_norm.weight", vec![h]),
            ("q.weight", vec![q, h]),
            ("k.weight", vec![k, h]),
            ("v.weight", vec![k, h]),
            ("o.weight", vec![h, q]),
            ("gate.weight", vec![i, h]),
            ("up.weight", vec![i, h]),
            ("down.weight", vec![h, i]),
        ] {
            specs.push((format!("layers.{l}.{name}"), shape));
        }
    }
    specs
}
pub(crate) fn checked(
    w: &Weights,
    name: &str,
    shape: &[usize],
    c: &ModelConfig,
    d: &MetalDevice,
) -> Result<Tensor> {
    let t = w.get(name)?;
    let message = if t.shape().dimensions() != shape {
        Some(format!(
            "expected shape {shape:?}, got {:?}",
            t.shape().dimensions()
        ))
    } else if t.dtype() != c.dtype {
        Some(format!("expected dtype {:?}, got {:?}", c.dtype, t.dtype()))
    } else if !d.owns(t.buffer()) {
        Some("different MetalDevice context".into())
    } else {
        None
    };
    if let Some(message) = message {
        return Err(Error::Weight {
            name: name.into(),
            message,
        });
    }
    Ok(t.clone())
}
pub(crate) fn construct(
    d: &MetalDevice,
    c: &ModelConfig,
    w: &Weights,
) -> Result<(Embedding, Vec<DecoderLayer>, RmsNorm, Linear)> {
    // Complete validation precedes any preprocessing dispatch.
    for (name, shape) in specifications(c) {
        checked(w, &name, &shape, c, d)?;
    }
    for l in 0..c.num_layers {
        for (name, out) in [
            ("q", c.num_attention_heads * c.head_dim),
            ("k", c.num_key_value_heads * c.head_dim),
            ("v", c.num_key_value_heads * c.head_dim),
            ("o", c.hidden_size),
            ("gate", c.intermediate_size),
            ("up", c.intermediate_size),
            ("down", c.hidden_size),
        ] {
            let name = format!("layers.{l}.{name}.bias");
            if w.optional(&name).is_some() {
                checked(w, &name, &[out], c, d)?;
            }
        }
    }
    if w.optional("lm_head.bias").is_some() {
        checked(w, "lm_head.bias", &[c.vocab_size], c, d)?;
    }
    let norm = |name: &str| -> Result<RmsNorm> {
        Ok(RmsNorm {
            weight: w.get(name)?.clone(),
            epsilon: c.rms_norm_epsilon,
        })
    };
    let linear = |name: &str| -> Result<Linear> {
        Linear::new(
            d,
            w.get(&format!("{name}.weight"))?.clone(),
            w.optional(&format!("{name}.bias")).cloned(),
        )
    };
    let mut layers = Vec::new();
    for l in 0..c.num_layers {
        let p = format!("layers.{l}");
        layers.push(DecoderLayer {
            input_norm: norm(&format!("{p}.input_norm.weight"))?,
            post_norm: norm(&format!("{p}.post_norm.weight"))?,
            attention: Attention {
                q: linear(&format!("{p}.q"))?,
                k: linear(&format!("{p}.k"))?,
                v: linear(&format!("{p}.v"))?,
                output: linear(&format!("{p}.o"))?,
                q_heads: c.num_attention_heads,
                kv_heads: c.num_key_value_heads,
                head_dim: c.head_dim,
                theta: c.rope_theta,
            },
            mlp: Mlp {
                gate: linear(&format!("{p}.gate"))?,
                up: linear(&format!("{p}.up"))?,
                down: linear(&format!("{p}.down"))?,
            },
        });
    }
    let embedding = Embedding::new(w.get("embedding.weight")?.clone())?;
    let lm = if c.tie_word_embeddings {
        Linear::new(
            d,
            w.get("embedding.weight")?.clone(),
            w.optional("lm_head.bias").cloned(),
        )?
    } else {
        linear("lm_head")?
    };
    Ok((embedding, layers, norm("final_norm.weight")?, lm))
}
