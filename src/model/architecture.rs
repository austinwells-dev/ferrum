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

#[derive(Debug, Clone, Copy, PartialEq)]
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
            moe: None,
        }
    }
}

impl ArchitecturePolicy {
    pub fn validate(self) -> Result<()> {
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
}
