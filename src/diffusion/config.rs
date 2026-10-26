//! Geometry and sampling settings read from a Hugging Face diffusers
//! directory (`model_index.json`, `transformer/config.json`, `vae/config.json`).
use crate::{Error, Result};
use serde::Deserialize;
use std::path::{Path, PathBuf};

/// The diffusion transformer (`QwenImage21Transformer2DModel`).
#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct DitConfig {
    #[serde(rename = "num_layers")]
    pub layers: usize,
    #[serde(rename = "num_attention_heads")]
    pub heads: usize,
    #[serde(rename = "attention_head_dim")]
    pub head_dim: usize,
    /// Width of the text encoder's embeddings.
    #[serde(rename = "context_in_dim")]
    pub context: usize,
    pub in_channels: usize,
    #[serde(default)]
    pub out_channels: Option<usize>,
    pub mlp_ratio: usize,
    #[serde(default = "default_eps")]
    pub eps: f32,
    /// Rotary widths of the frame, height and width axes.
    #[serde(rename = "axes_dims_rope")]
    pub rope_axes: [usize; 3],
    /// Text tokens are modulated from `t = 0`, which is what makes their K/V
    /// the same at every step.
    pub causal_condition: bool,
    #[serde(default = "one")]
    pub patch_size: usize,
}

fn default_eps() -> f32 {
    1e-6
}

fn one() -> usize {
    1
}

impl DitConfig {
    pub fn from_json(json: &str) -> Result<Self> {
        let c: Self = serde_json::from_str(json)
            .map_err(|e| Error::Config(format!("transformer config: {e}")))?;
        if c.rope_axes.iter().sum::<usize>() != c.head_dim {
            return Err(Error::Config(format!(
                "rotary axes {:?} do not add up to the head width {}",
                c.rope_axes, c.head_dim
            )));
        }
        if c.patch_size != 1 || !c.causal_condition {
            return Err(Error::Config(
                "only unpatched, causal-condition transformers are supported".into(),
            ));
        }
        Ok(c)
    }
    /// Model width.
    pub fn hidden(&self) -> usize {
        self.heads * self.head_dim
    }
    /// SwiGLU width.
    pub fn ffn(&self) -> usize {
        self.hidden() * self.mlp_ratio
    }
    pub fn out_channels(&self) -> usize {
        self.out_channels.unwrap_or(self.in_channels)
    }
}

/// The latent decoder (`AutoencoderKLQwenImage21`).
#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct VaeConfig {
    pub z_dim: usize,
    pub decoder_base_dim: usize,
    pub dim_mult: Vec<usize>,
    pub num_res_blocks: usize,
    /// Pixels per latent token along one side.
    pub scale_factor_spatial: usize,
    pub latents_mean: Vec<f32>,
    pub latents_std: Vec<f32>,
    pub out_channels: usize,
}

impl VaeConfig {
    pub fn from_json(json: &str) -> Result<Self> {
        let c: Self =
            serde_json::from_str(json).map_err(|e| Error::Config(format!("vae config: {e}")))?;
        if c.latents_mean.len() != c.z_dim || c.latents_std.len() != c.z_dim {
            return Err(Error::Config("latent statistics do not match z_dim".into()));
        }
        Ok(c)
    }
}

/// The text encoder's language model (the `text_config` of a
/// `Qwen3VLForConditionalGeneration` checkpoint; its vision tower is unused).
#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct TextConfig {
    #[serde(rename = "hidden_size")]
    pub hidden: usize,
    #[serde(rename = "num_hidden_layers")]
    pub layers: usize,
    #[serde(rename = "num_attention_heads")]
    pub heads: usize,
    #[serde(rename = "num_key_value_heads")]
    pub kv_heads: usize,
    pub head_dim: usize,
    #[serde(rename = "intermediate_size")]
    pub ffn: usize,
    #[serde(rename = "vocab_size")]
    pub vocab: usize,
    #[serde(rename = "rms_norm_eps", default = "default_eps")]
    pub eps: f32,
    pub rope_theta: f32,
    #[serde(rename = "max_position_embeddings")]
    pub context: usize,
}

impl TextConfig {
    pub fn from_json(json: &str) -> Result<Self> {
        let root: serde_json::Value = serde_json::from_str(json)
            .map_err(|e| Error::Config(format!("text encoder config: {e}")))?;
        let text = root.get("text_config").unwrap_or(&root);
        serde_json::from_value(text.clone())
            .map_err(|e| Error::Config(format!("text encoder config: {e}")))
    }
}

/// The sampler settings of `scheduler/scheduler_config.json` that shape the
/// sigma schedule.
#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct SchedulerConfig {
    #[serde(default = "half")]
    pub base_shift: f32,
    #[serde(default = "one_f")]
    pub max_shift: f32,
    #[serde(default = "base_len")]
    pub base_image_seq_len: usize,
    #[serde(default = "max_len")]
    pub max_image_seq_len: usize,
    #[serde(default)]
    pub use_dynamic_shifting: bool,
    #[serde(default = "one_f")]
    pub shift: f32,
    #[serde(default)]
    pub shift_terminal: Option<f32>,
}

fn half() -> f32 {
    0.5
}
fn one_f() -> f32 {
    1.
}
fn base_len() -> usize {
    256
}
fn max_len() -> usize {
    4096
}

/// A diffusers model directory.
#[derive(Debug, Clone)]
pub struct ModelDir {
    root: PathBuf,
    pub dit: DitConfig,
    pub vae: VaeConfig,
    pub text: TextConfig,
    pub scheduler: SchedulerConfig,
    /// The checkpoint's own sampling sigmas (Turbo ships an 8-step list in
    /// `model_index.json`); the base model samples a linear grid instead.
    pub sample_sigmas: Option<Vec<f32>>,
}

impl ModelDir {
    pub fn open(root: impl AsRef<Path>) -> Result<Self> {
        let root = root.as_ref().to_path_buf();
        let read = |file: &str| {
            std::fs::read_to_string(root.join(file))
                .map_err(|e| Error::Config(format!("{}: {e}", root.join(file).display())))
        };
        let index: serde_json::Value = serde_json::from_str(&read("model_index.json")?)
            .map_err(|e| Error::Config(format!("model_index.json: {e}")))?;
        match index["_class_name"].as_str() {
            Some("QwenImage21Pipeline") => {}
            other => {
                return Err(Error::Config(format!(
                    "{} is a {other:?}, not a QwenImage21Pipeline",
                    root.display()
                )));
            }
        }
        let sample_sigmas = match index.get("sample_sigmas") {
            Some(v) => Some(
                serde_json::from_value(v.clone())
                    .map_err(|e| Error::Config(format!("sample_sigmas: {e}")))?,
            ),
            None => None,
        };
        Ok(Self {
            dit: DitConfig::from_json(&read("transformer/config.json")?)?,
            vae: VaeConfig::from_json(&read("vae/config.json")?)?,
            text: TextConfig::from_json(&read("text_encoder/config.json")?)?,
            scheduler: serde_json::from_str(&read("scheduler/scheduler_config.json")?)
                .map_err(|e| Error::Config(format!("scheduler config: {e}")))?,
            sample_sigmas,
            root,
        })
    }
    pub fn path(&self, relative: &str) -> PathBuf {
        self.root.join(relative)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const DIT: &str = r#"{"_class_name":"QwenImage21Transformer2DModel","attention_head_dim":128,
        "axes_dims_rope":[16,56,56],"causal_condition":true,"context_in_dim":4096,"eps":1e-6,
        "in_channels":64,"mlp_ratio":3,"num_attention_heads":32,"num_layers":32,
        "out_channels":64,"patch_size":1}"#;

    #[test]
    fn reads_the_published_transformer_geometry() {
        let c = DitConfig::from_json(DIT).unwrap();
        assert_eq!((c.layers, c.hidden(), c.ffn()), (32, 4096, 12288));
        assert_eq!(c.out_channels(), 64);
    }

    #[test]
    fn rejects_rotary_axes_that_miss_the_head_width() {
        let bad = DIT.replace("[16,56,56]", "[16,56,48]");
        assert!(DitConfig::from_json(&bad).is_err());
    }
}
