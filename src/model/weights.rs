use super::{
    ModelConfig,
    architecture::{
        ArchitecturePolicy, LayerFeedForwardPolicy, LayerOperatorPolicy, ProjectionBias,
        QkNormLayout,
    },
};
use crate::{
    Error, MetalDevice, Result, Tensor,
    loader::Weights,
    nn::{
        Embedding, Linear, Mlp, RmsNorm, attention::Attention, kv_cache::KvCache, moe::SparseMoe,
    },
    quantization::{QuantizationFormat, QuantizedMatrix},
};
use std::collections::BTreeMap;
pub struct DecoderLayer {
    pub input_norm: RmsNorm,
    pub operator: LayerOperator,
    pub post_norm: RmsNorm,
    pub(crate) feed_forward: FeedForward,
}

pub enum LayerOperator {
    Attention(Box<Attention>),
    ShortConv(Box<ShortConv>),
}

pub struct ShortConv {
    pub input_projection: Linear,
    pub depthwise_weight: Tensor,
    pub output_projection: Linear,
    pub kernel_size: usize,
}

impl ShortConv {
    pub(crate) fn forward(
        &self,
        d: &MetalDevice,
        x: &Tensor,
        cache: &mut KvCache,
        layer: usize,
    ) -> Result<Tensor> {
        let dims = x.shape().dimensions();
        if dims.len() != 2 || self.depthwise_weight.shape().dimensions().first() != Some(&dims[1]) {
            return Err(Error::Shape(
                "short convolution input must be [S,hidden]".into(),
            ));
        }
        let hidden = dims[1];
        let projected = self.input_projection.forward(d, x)?;
        let (b, c, value) = d.lfm2_split3(&projected)?;
        let bx = d.mul(&b, &value)?.tensor;
        let previous = match cache.conv_state(layer)? {
            Some(state) => state.clone(),
            None => Tensor::zeros(d, [self.kernel_size, hidden], x.dtype())?,
        };
        let (convolved, next_state) =
            d.lfm2_short_conv(&bx, &c, &self.depthwise_weight, &previous, self.kernel_size)?;
        cache.set_conv_state(d, layer, next_state)?;
        self.output_projection.forward(d, &convolved)
    }

    pub(crate) fn weight_bytes(&self) -> usize {
        self.input_projection.weight_bytes()
            + self.depthwise_weight.byte_size()
            + self.output_projection.weight_bytes()
    }
}

impl LayerOperator {
    pub(crate) fn weight_bytes(&self) -> usize {
        match self {
            Self::Attention(attention) => {
                let mut bytes = 0;
                for linear in [&attention.q, &attention.k, &attention.v, &attention.output] {
                    bytes += linear.weight_bytes();
                }
                bytes
                    + attention
                        .q_norm
                        .as_ref()
                        .map_or(0, |norm| norm.weight.byte_size())
                    + attention
                        .k_norm
                        .as_ref()
                        .map_or(0, |norm| norm.weight.byte_size())
            }
            Self::ShortConv(convolution) => convolution.weight_bytes(),
        }
    }
}

pub(crate) enum FeedForward {
    Dense(Mlp),
    Sparse(SparseMoe),
}

impl FeedForward {
    pub(crate) fn forward(&self, d: &MetalDevice, x: &Tensor) -> Result<Tensor> {
        match self {
            Self::Dense(mlp) => mlp.forward(d, x),
            Self::Sparse(moe) => moe.forward(d, x),
        }
    }

    pub(crate) fn weight_bytes(&self) -> usize {
        match self {
            Self::Dense(mlp) => [&mlp.gate, &mlp.up, &mlp.down]
                .iter()
                .map(|linear| linear.weight_bytes())
                .sum(),
            Self::Sparse(moe) => moe.weight_bytes(),
        }
    }

    pub(crate) fn take_moe_stats(&self) -> Option<crate::nn::moe::MoeStats> {
        match self {
            Self::Dense(_) => None,
            Self::Sparse(moe) => Some(moe.take_stats()),
        }
    }
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
    specifications_with_policy(c, &ArchitecturePolicy::default())
}
pub(crate) fn specifications_with_policy(
    c: &ModelConfig,
    policy: &ArchitecturePolicy,
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
        specs.extend([
            (format!("layers.{l}.input_norm.weight"), vec![h]),
            (format!("layers.{l}.post_norm.weight"), vec![h]),
        ]);
        let Ok(layer) = policy.layer(l, i) else {
            continue;
        };
        match layer.operator {
            LayerOperatorPolicy::Attention => {
                specs.extend([
                    (format!("layers.{l}.q.weight"), vec![q, h]),
                    (format!("layers.{l}.k.weight"), vec![k, h]),
                    (format!("layers.{l}.v.weight"), vec![k, h]),
                    (format!("layers.{l}.o.weight"), vec![h, q]),
                ]);
            }
            LayerOperatorPolicy::ShortConv { kernel_size } => {
                specs.extend([
                    (format!("layers.{l}.conv.in_proj.weight"), vec![3 * h, h]),
                    (
                        format!("layers.{l}.conv.depthwise.weight"),
                        vec![h, 1, kernel_size],
                    ),
                    (format!("layers.{l}.conv.out_proj.weight"), vec![h, h]),
                ]);
            }
        }
        match layer.feed_forward {
            LayerFeedForwardPolicy::Dense { intermediate_size } => {
                specs.extend([
                    (
                        format!("layers.{l}.gate.weight"),
                        vec![intermediate_size, h],
                    ),
                    (format!("layers.{l}.up.weight"), vec![intermediate_size, h]),
                    (
                        format!("layers.{l}.down.weight"),
                        vec![h, intermediate_size],
                    ),
                ]);
            }
            LayerFeedForwardPolicy::Sparse {
                routing,
                intermediate_size,
            } => specs.extend([
                (
                    format!("layers.{l}.moe.router.weight"),
                    vec![routing.experts, h],
                ),
                (
                    format!("layers.{l}.moe.input.weight"),
                    vec![routing.experts, 2 * intermediate_size, h],
                ),
                (
                    format!("layers.{l}.moe.output.weight"),
                    vec![routing.experts, h, intermediate_size],
                ),
            ]),
        }
        if matches!(layer.operator, LayerOperatorPolicy::Attention)
            && policy.qk_norm_epsilon.is_some()
        {
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
fn check_qkv_bias(policy: &ArchitecturePolicy, present: bool, name: &str) -> Result<()> {
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
    policy: &ArchitecturePolicy,
    w: &Weights,
) -> Result<(Embedding, Vec<DecoderLayer>, RmsNorm, Linear)> {
    policy.validate_for(c)?;
    if policy.moe.is_some()
        || policy.layer_policies.as_ref().is_some_and(|layers| {
            layers.iter().any(|layer| {
                !matches!(layer.operator, LayerOperatorPolicy::Attention)
                    || !matches!(layer.feed_forward, LayerFeedForwardPolicy::Dense { .. })
            })
        })
    {
        return Err(Error::Config(
            "sparse models must be constructed from mixed model weights".into(),
        ));
    }
    // Complete validation precedes any preprocessing dispatch.
    for (name, shape) in specifications_with_policy(c, policy) {
        checked(w, &name, &shape, c, d)?;
    }
    for l in 0..c.num_layers {
        let layer = policy.layer(l, c.intermediate_size)?;
        let mut components = Vec::new();
        if matches!(layer.operator, LayerOperatorPolicy::Attention) {
            components.extend([
                ("q", c.num_attention_heads * c.head_dim),
                ("k", c.num_key_value_heads * c.head_dim),
                ("v", c.num_key_value_heads * c.head_dim),
                ("o", c.hidden_size),
            ]);
        }
        if let LayerFeedForwardPolicy::Dense { intermediate_size } = layer.feed_forward {
            components.extend([
                ("gate", intermediate_size),
                ("up", intermediate_size),
                ("down", c.hidden_size),
            ]);
        }
        for (component, out) in components {
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
        let layer_policy = policy.layer(l, c.intermediate_size)?;
        let operator = match layer_policy.operator {
            LayerOperatorPolicy::Attention => LayerOperator::Attention(Box::new(Attention {
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
            })),
            LayerOperatorPolicy::ShortConv { kernel_size } => {
                LayerOperator::ShortConv(Box::new(ShortConv {
                    input_projection: linear(&format!("{p}.conv.in_proj"))?,
                    depthwise_weight: w.get(&format!("{p}.conv.depthwise.weight"))?.clone(),
                    output_projection: linear(&format!("{p}.conv.out_proj"))?,
                    kernel_size,
                }))
            }
        };
        let LayerFeedForwardPolicy::Dense { .. } = layer_policy.feed_forward else {
            return Err(Error::Config(
                "direct weight construction supports dense feed-forward layers only".into(),
            ));
        };
        layers.push(DecoderLayer {
            input_norm: norm(&format!("{p}.input_norm.weight"))?,
            post_norm: norm(&format!("{p}.post_norm.weight"))?,
            operator,
            feed_forward: FeedForward::Dense(Mlp {
                gate: linear(&format!("{p}.gate"))?,
                up: linear(&format!("{p}.up"))?,
                down: linear(&format!("{p}.down"))?,
            }),
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
    policy: &ArchitecturePolicy,
    w: &ModelWeights,
) -> Result<(Embedding, Vec<DecoderLayer>, RmsNorm, Linear)> {
    c.validate()?;
    policy.validate_for(c)?;
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
        if matches!(
            policy.layer(l, c.intermediate_size)?.operator,
            LayerOperatorPolicy::Attention
        ) {
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
        let layer_policy = policy.layer(layer, c.intermediate_size)?;
        let feed_forward = match layer_policy.feed_forward {
            LayerFeedForwardPolicy::Sparse {
                routing,
                intermediate_size: _,
            } => {
                let router = dense(&format!("{prefix}.moe.router.weight"))?;
                let input_experts = dense(&format!("{prefix}.moe.input.weight"))?;
                let output_experts = dense(&format!("{prefix}.moe.output.weight"))?;
                FeedForward::Sparse(SparseMoe::new(
                    d,
                    router,
                    input_experts,
                    output_experts,
                    routing.top_k,
                )?)
            }
            LayerFeedForwardPolicy::Dense { .. } => FeedForward::Dense(Mlp {
                gate: linear(&format!("{prefix}.gate"))?,
                up: linear(&format!("{prefix}.up"))?,
                down: linear(&format!("{prefix}.down"))?,
            }),
        };
        let operator = match layer_policy.operator {
            LayerOperatorPolicy::Attention => LayerOperator::Attention(Box::new(Attention {
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
            })),
            LayerOperatorPolicy::ShortConv { kernel_size } => {
                LayerOperator::ShortConv(Box::new(ShortConv {
                    input_projection: linear(&format!("{prefix}.conv.in_proj"))?,
                    depthwise_weight: dense(&format!("{prefix}.conv.depthwise.weight"))?,
                    output_projection: linear(&format!("{prefix}.conv.out_proj"))?,
                    kernel_size,
                }))
            }
        };
        layers.push(DecoderLayer {
            input_norm: norm(&format!("{prefix}.input_norm.weight"))?,
            post_norm: norm(&format!("{prefix}.post_norm.weight"))?,
            operator,
            feed_forward,
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
