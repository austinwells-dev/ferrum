use super::{
    ModelConfig,
    architecture::{ArchitecturePolicy, ProjectionBias, QkNormLayout},
};
use crate::{
    Error, MetalDevice, Result, Tensor,
    loader::Weights,
    nn::{Embedding, Linear, Mlp, RmsNorm, attention::Attention},
    quantization::{QuantizationFormat, QuantizedMatrix},
};
use std::collections::BTreeMap;
pub struct DecoderLayer {
    pub input_norm: RmsNorm,
    pub attention: Attention,
    pub post_norm: RmsNorm,
    pub mlp: Mlp,
}

#[derive(Clone)]
pub(crate) enum ModelWeight {
    Dense(Tensor),
    Quantized(QuantizedMatrix),
}
impl ModelWeight {
    fn dimensions(&self) -> Vec<usize> {
        match self {
            Self::Dense(tensor) => tensor.shape().dimensions().to_vec(),
            Self::Quantized(tensor) => vec![tensor.rows(), tensor.columns()],
        }
    }
}

#[derive(Default)]
pub(crate) struct ModelWeights {
    tensors: BTreeMap<String, ModelWeight>,
}
impl ModelWeights {
    pub(crate) fn insert(&mut self, name: String, weight: ModelWeight) -> Result<()> {
        if self.tensors.insert(name.clone(), weight).is_some() {
            return Err(Error::Weight {
                name,
                message: "duplicate mapped tensor".into(),
            });
        }
        Ok(())
    }
    fn get(&self, name: &str) -> Result<&ModelWeight> {
        self.tensors.get(name).ok_or_else(|| Error::Weight {
            name: name.into(),
            message: "missing mapped tensor".into(),
        })
    }
}
pub(crate) fn specifications(c: &ModelConfig) -> Vec<(String, Vec<usize>)> {
    specifications_with_policy(c, ArchitecturePolicy::default())
}
pub(crate) fn specifications_with_policy(
    c: &ModelConfig,
    policy: ArchitecturePolicy,
) -> Vec<(String, Vec<usize>)> {
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
        if policy.qk_norm_epsilon.is_some() {
            for (name, heads) in [
                ("q_norm.weight", c.num_attention_heads),
                ("k_norm.weight", c.num_key_value_heads),
            ] {
                let width = match policy.qk_norm_layout {
                    QkNormLayout::PerHead => c.head_dim,
                    QkNormLayout::Projection => heads * c.head_dim,
                };
                specs.push((format!("layers.{l}.{name}"), vec![width]));
            }
        }
    }
    specs
}
fn check_qkv_bias(policy: ArchitecturePolicy, present: bool, name: &str) -> Result<()> {
    match (policy.qkv_bias, present) {
        (ProjectionBias::Required, false) => Err(Error::Weight {
            name: name.into(),
            message: "required Q/K/V projection bias is missing".into(),
        }),
        (ProjectionBias::Forbidden, true) => Err(Error::Weight {
            name: name.into(),
            message: "Q/K/V projection bias is forbidden by architecture policy".into(),
        }),
        _ => Ok(()),
    }
}
pub(crate) fn checked(
    w: &Weights,
    name: &str,
    shape: &[usize],
    c: &ModelConfig,
    d: &MetalDevice,
) -> Result<Tensor> {
    let t = w.optional(name).ok_or_else(|| Error::Weight {
        name: name.into(),
        message: format!(
            "missing tensor; expected shape {shape:?}, expected dtype {:?}",
            c.dtype
        ),
    })?;
    let message = if t.shape().dimensions() != shape || t.dtype() != c.dtype {
        Some(format!(
            "expected shape {shape:?}, actual shape {:?}; expected dtype {:?}, actual dtype {:?}",
            t.shape().dimensions(),
            c.dtype,
            t.dtype()
        ))
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
    policy: ArchitecturePolicy,
    w: &Weights,
) -> Result<(Embedding, Vec<DecoderLayer>, RmsNorm, Linear)> {
    // Complete validation precedes any preprocessing dispatch.
    for (name, shape) in specifications_with_policy(c, policy) {
        checked(w, &name, &shape, c, d)?;
    }
    for l in 0..c.num_layers {
        for (component, out) in [
            ("q", c.num_attention_heads * c.head_dim),
            ("k", c.num_key_value_heads * c.head_dim),
            ("v", c.num_key_value_heads * c.head_dim),
            ("o", c.hidden_size),
            ("gate", c.intermediate_size),
            ("up", c.intermediate_size),
            ("down", c.hidden_size),
        ] {
            let name = format!("layers.{l}.{component}.bias");
            if matches!(component, "q" | "k" | "v") {
                check_qkv_bias(policy, w.optional(&name).is_some(), &name)?;
            }
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
    let head_norm = |name: &str| -> Result<Option<RmsNorm>> {
        policy
            .qk_norm_epsilon
            .map(|epsilon| {
                Ok(RmsNorm {
                    weight: w.get(name)?.clone(),
                    epsilon,
                })
            })
            .transpose()
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
                q_norm: head_norm(&format!("{p}.q_norm.weight"))?,
                k_norm: head_norm(&format!("{p}.k_norm.weight"))?,
                q_heads: c.num_attention_heads,
                kv_heads: c.num_key_value_heads,
                head_dim: c.head_dim,
                theta: c.rope_theta,
                qk_norm_layout: policy.qk_norm_layout,
                scale: policy
                    .attention_scale
                    .unwrap_or_else(|| (c.head_dim as f32).sqrt().recip()),
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

/// Construct a transformer from validated mixed dense and packed matrices.
/// Dense values must already use the activation dtype; quantized matrices are
/// retained as packed Metal allocations.
pub(crate) fn construct_mixed(
    d: &MetalDevice,
    c: &ModelConfig,
    policy: ArchitecturePolicy,
    w: &ModelWeights,
) -> Result<(Embedding, Vec<DecoderLayer>, RmsNorm, Linear)> {
    c.validate()?;
    for (name, shape) in specifications_with_policy(c, policy) {
        let weight = w.get(&name)?;
        if weight.dimensions() != shape {
            return Err(Error::Weight {
                name,
                message: format!("expected shape {shape:?}, actual {:?}", weight.dimensions()),
            });
        }
        match weight {
            ModelWeight::Dense(tensor) => {
                if tensor.dtype() != c.dtype {
                    return Err(Error::Weight {
                        name,
                        message: format!(
                            "dense model tensor must be {:?}, found {:?}",
                            c.dtype,
                            tensor.dtype()
                        ),
                    });
                }
                if !d.owns(tensor.buffer()) {
                    return Err(Error::DeviceMismatch);
                }
            }
            ModelWeight::Quantized(tensor) => {
                if !matches!(
                    tensor.format(),
                    QuantizationFormat::Q4_0
                        | QuantizationFormat::Q5_0
                        | QuantizationFormat::Q5_1
                        | QuantizationFormat::Q4_K
                        | QuantizationFormat::Q5_K
                        | QuantizationFormat::Q8_0
                        | QuantizationFormat::Q6_K
                        | QuantizationFormat::MlxAffine4Group64
                ) {
                    return Err(Error::Weight {
                        name,
                        message: format!("unsupported execution format {:?}", tensor.format()),
                    });
                }
                if !d.owns(tensor.buffer()) {
                    return Err(Error::DeviceMismatch);
                }
            }
        }
    }
    for l in 0..c.num_layers {
        for (name, out) in [
            ("q", c.num_attention_heads * c.head_dim),
            ("k", c.num_key_value_heads * c.head_dim),
            ("v", c.num_key_value_heads * c.head_dim),
        ] {
            let name = format!("layers.{l}.{name}.bias");
            check_qkv_bias(policy, w.tensors.contains_key(&name), &name)?;
            if let Some(weight) = w.tensors.get(&name) {
                check_dense_vector(d, &name, weight, out, c.dtype)?;
            }
        }
    }
    if let Some(weight) = w.tensors.get("lm_head.bias") {
        check_dense_vector(d, "lm_head.bias", weight, c.vocab_size, c.dtype)?;
    }

    let dense = |name: &str| -> Result<Tensor> {
        match w.get(name)? {
            ModelWeight::Dense(tensor) => Ok(tensor.clone()),
            ModelWeight::Quantized(_) => Err(Error::Weight {
                name: name.into(),
                message: "normalization tensors must be dense".into(),
            }),
        }
    };
    let norm = |name: &str| -> Result<RmsNorm> {
        Ok(RmsNorm {
            weight: dense(name)?,
            epsilon: c.rms_norm_epsilon,
        })
    };
    let linear = |name: &str| -> Result<Linear> {
        let weight_name = format!("{name}.weight");
        let bias_name = format!("{name}.bias");
        let bias = w
            .tensors
            .get(&bias_name)
            .map(|value| match value {
                ModelWeight::Dense(tensor) => Ok(tensor.clone()),
                ModelWeight::Quantized(_) => Err(Error::Weight {
                    name: bias_name.clone(),
                    message: "linear bias must be dense".into(),
                }),
            })
            .transpose()?;
        match w.get(&weight_name)? {
            ModelWeight::Dense(tensor) => Linear::new(d, tensor.clone(), bias),
            ModelWeight::Quantized(tensor) => {
                Linear::new_quantized(d, tensor.clone(), bias, c.dtype)
            }
        }
    };
    let head_norm = |name: &str| -> Result<Option<RmsNorm>> {
        policy
            .qk_norm_epsilon
            .map(|epsilon| {
                Ok(RmsNorm {
                    weight: dense(name)?,
                    epsilon,
                })
            })
            .transpose()
    };

    let mut layers = Vec::with_capacity(c.num_layers);
    for layer in 0..c.num_layers {
        let prefix = format!("layers.{layer}");
        layers.push(DecoderLayer {
            input_norm: norm(&format!("{prefix}.input_norm.weight"))?,
            post_norm: norm(&format!("{prefix}.post_norm.weight"))?,
            attention: Attention {
                q: linear(&format!("{prefix}.q"))?,
                k: linear(&format!("{prefix}.k"))?,
                v: linear(&format!("{prefix}.v"))?,
                output: linear(&format!("{prefix}.o"))?,
                q_norm: head_norm(&format!("{prefix}.q_norm.weight"))?,
                k_norm: head_norm(&format!("{prefix}.k_norm.weight"))?,
                q_heads: c.num_attention_heads,
                kv_heads: c.num_key_value_heads,
                head_dim: c.head_dim,
                theta: c.rope_theta,
                qk_norm_layout: policy.qk_norm_layout,
                scale: policy
                    .attention_scale
                    .unwrap_or_else(|| (c.head_dim as f32).sqrt().recip()),
            },
            mlp: Mlp {
                gate: linear(&format!("{prefix}.gate"))?,
                up: linear(&format!("{prefix}.up"))?,
                down: linear(&format!("{prefix}.down"))?,
            },
        });
    }
    let embedding = match w.get("embedding.weight")? {
        ModelWeight::Dense(tensor) => Embedding::new(tensor.clone())?,
        ModelWeight::Quantized(tensor) => Embedding::new_quantized(d, tensor.clone(), c.dtype)?,
    };
    let lm_head = if c.tie_word_embeddings {
        let bias = w
            .tensors
            .get("lm_head.bias")
            .map(|weight| match weight {
                ModelWeight::Dense(tensor) => Ok(tensor.clone()),
                ModelWeight::Quantized(_) => Err(Error::Weight {
                    name: "lm_head.bias".into(),
                    message: "linear bias must be dense".into(),
                }),
            })
            .transpose()?;
        match w.get("embedding.weight")? {
            ModelWeight::Dense(tensor) => Linear::new(d, tensor.clone(), bias)?,
            ModelWeight::Quantized(tensor) => {
                Linear::new_quantized(d, tensor.clone(), bias, c.dtype)?
            }
        }
    } else {
        linear("lm_head")?
    };
    Ok((embedding, layers, norm("final_norm.weight")?, lm_head))
}

fn check_dense_vector(
    d: &MetalDevice,
    name: &str,
    weight: &ModelWeight,
    length: usize,
    dtype: crate::DType,
) -> Result<()> {
    let tensor = match weight {
        ModelWeight::Dense(tensor) => tensor,
        ModelWeight::Quantized(_) => {
            return Err(Error::Weight {
                name: name.into(),
                message: "one-dimensional model tensors must be dense".into(),
            });
        }
    };
    if tensor.shape().dimensions() != [length] || tensor.dtype() != dtype {
        return Err(Error::Weight {
            name: name.into(),
            message: format!(
                "expected dense [{length}] {:?}, found {:?} {:?}",
                dtype,
                tensor.shape().dimensions(),
                tensor.dtype()
            ),
        });
    }
    if !d.owns(tensor.buffer()) {
        return Err(Error::DeviceMismatch);
    }
    Ok(())
}
