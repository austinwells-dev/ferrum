//! Execution policies selected while constructing a model from observed architecture metadata.
#![forbid(unsafe_code)]
use crate::{Error, Result};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ProjectionBias {
    #[default]
    Optional,
    Required,
    Forbidden,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LayerOperatorPolicy {
    Attention,
    ShortConv { kernel_size: usize },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LayerFeedForwardPolicy {
    Dense {
        intermediate_size: usize,
    },
    Sparse {
        routing: MoeRoutingPolicy,
        intermediate_size: usize,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LayerPolicy {
    pub operator: LayerOperatorPolicy,
    pub feed_forward: LayerFeedForwardPolicy,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ArchitecturePolicy {
    /// Normalize each Q and K head after projection and before RoPE.
    pub qk_norm_epsilon: Option<f32>,
    pub qk_norm_layout: QkNormLayout,
    pub qkv_bias: ProjectionBias,
    pub residual_topology: ResidualTopology,
    pub attention_scale: Option<f32>,
    pub embedding_multiplier: f32,
    pub residual_multiplier: f32,
    pub logits_divisor: f32,
    /// Optional layer-by-layer block composition. When absent, every layer is
    /// attention plus a dense MLP, or sparse MLP when `moe` is selected.
    pub layer_policies: Option<Vec<LayerPolicy>>,
    /// Compatibility shorthand for architectures using sparse experts in every layer.
    pub moe: Option<MoeRoutingPolicy>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum QkNormLayout {
    #[default]
    PerHead,
    Projection,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ResidualTopology {
    #[default]
    PreNorm,
    PostNorm,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MoeRoutingPolicy {
    pub experts: usize,
    pub top_k: usize,
}

impl Default for ArchitecturePolicy {
    fn default() -> Self {
        Self {
            qk_norm_epsilon: None,
            qk_norm_layout: QkNormLayout::PerHead,
            qkv_bias: ProjectionBias::Optional,
            residual_topology: ResidualTopology::PreNorm,
            attention_scale: None,
            embedding_multiplier: 1.,
            residual_multiplier: 1.,
            logits_divisor: 1.,
            layer_policies: None,
            moe: None,
        }
    }
}

impl ArchitecturePolicy {
    pub fn validate(&self) -> Result<()> {
        if self
            .qk_norm_epsilon
            .is_some_and(|epsilon| !epsilon.is_finite() || epsilon <= 0.)
        {
            return Err(Error::Config(
                "Q/K normalization epsilon must be finite and positive".into(),
            ));
        }
        if self.residual_topology == ResidualTopology::PostNorm && self.residual_multiplier != 1. {
            return Err(Error::Config(
                "post-normalized residual multiplier is unsupported".into(),
            ));
        }
        if self.moe.is_some_and(|routing| {
            routing.experts == 0 || routing.top_k == 0 || routing.top_k > routing.experts
        }) {
            return Err(Error::Config(
                "MoE requires positive expert count and top-k within expert count".into(),
            ));
        }
        if self.moe.is_some() && self.layer_policies.is_some() {
            return Err(Error::Config(
                "global MoE routing cannot be combined with per-layer policies".into(),
            ));
        }
        if self.layer_policies.as_ref().is_some_and(|layers| {
            layers.iter().any(|layer| {
                matches!(
                    layer.operator,
                    LayerOperatorPolicy::ShortConv { kernel_size: 0 }
                ) || match layer.feed_forward {
                    LayerFeedForwardPolicy::Dense { intermediate_size } => intermediate_size == 0,
                    LayerFeedForwardPolicy::Sparse {
                        routing,
                        intermediate_size,
                    } => {
                        routing.experts == 0
                            || routing.top_k == 0
                            || routing.top_k > routing.experts
                            || intermediate_size == 0
                    }
                }
            })
        }) {
            return Err(Error::Config(
                "per-layer policies require positive dimensions and valid expert routing".into(),
            ));
        }
        if self
            .attention_scale
            .is_some_and(|scale| !scale.is_finite() || scale <= 0.)
            || [
                self.embedding_multiplier,
                self.residual_multiplier,
                self.logits_divisor,
            ]
            .iter()
            .any(|value| !value.is_finite() || *value <= 0.)
        {
            return Err(Error::Config(
                "architecture scale factors must be finite and positive".into(),
            ));
        }
        Ok(())
    }

    pub fn validate_for(&self, config: &super::ModelConfig) -> Result<()> {
        self.validate()?;
        if self
            .layer_policies
            .as_ref()
            .is_some_and(|layers| layers.len() != config.num_layers)
        {
            return Err(Error::Config(format!(
                "architecture policy has {} layer entries for a {}-layer model",
                self.layer_policies.as_ref().map_or(0, Vec::len),
                config.num_layers
            )));
        }
        Ok(())
    }

    pub fn layer(&self, index: usize, default_intermediate_size: usize) -> Result<LayerPolicy> {
        if let Some(layers) = &self.layer_policies {
            return layers
                .get(index)
                .copied()
                .ok_or_else(|| Error::Config("layer policy index out of range".into()));
        }
        let feed_forward = if let Some(routing) = self.moe {
            LayerFeedForwardPolicy::Sparse {
                routing,
                intermediate_size: default_intermediate_size,
            }
        } else {
            LayerFeedForwardPolicy::Dense {
                intermediate_size: default_intermediate_size,
            }
        };
        Ok(LayerPolicy {
            operator: LayerOperatorPolicy::Attention,
            feed_forward,
        })
    }
}
