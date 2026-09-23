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

#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct ArchitecturePolicy {
    /// Normalize each Q and K head after projection and before RoPE.
    pub qk_norm_epsilon: Option<f32>,
    pub qkv_bias: ProjectionBias,
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
        Ok(())
    }
}
