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
    /// Channels of the encoder's first level.
    pub base_dim: usize,
    pub decoder_base_dim: usize,
    pub dim_mult: Vec<usize>,
    pub num_res_blocks: usize,
    /// Pixels per latent token along one side.
    pub scale_factor_spatial: usize,
    pub latents_mean: Vec<f32>,
    pub latents_std: Vec<f32>,
    /// Image channels in (RGBA) and out.
    pub in_channels: usize,
    pub out_channels: usize,
    /// Per encoder level, whether it halves frames (which, for one image,
    /// only changes how its shortcut averages).
    pub temperal_downsample: Vec<bool>,
}

impl VaeConfig {
    pub fn from_json(json: &str) -> Result<Self> {
        let c: Self =
            serde_json::from_str(json).map_err(|e| Error::Config(format!("vae config: {e}")))?;
        if c.latents_mean.len() != c.z_dim || c.latents_std.len() != c.z_dim {
            return Err(Error::Config("latent statistics do not match z_dim".into()));
        }
        if c.temperal_downsample.len() + 1 != c.dim_mult.len() {
            return Err(Error::Config(
                "temperal_downsample needs one entry per downsampling level".into(),
            ));
        }
        Ok(c)
    }
}

/// The text encoder's language model (the `text_config` of a
/// `Qwen3VLForConditionalGeneration` checkpoint).
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
    /// Rotary pairs of the (time, height, width) axes, interleaved; image
    /// tokens are placed on a 2-D grid through them.
    #[serde(skip)]
    pub mrope_sections: [usize; 3],
}

impl TextConfig {
    pub fn from_json(json: &str) -> Result<Self> {
        let fail = |e: serde_json::Error| Error::Config(format!("text encoder config: {e}"));
        let root: serde_json::Value = serde_json::from_str(json).map_err(fail)?;
        let text = root.get("text_config").unwrap_or(&root);
        let mut c: Self = serde_json::from_value(text.clone()).map_err(fail)?;
        let section = ["rope_scaling", "rope_parameters"]
            .iter()
            .find_map(|k| text.get(k)?.get("mrope_section").cloned());
        if let Some(section) = section {
            c.mrope_sections = serde_json::from_value(section).map_err(fail)?;
            if c.mrope_sections.iter().sum::<usize>() != c.head_dim / 2 {
                return Err(Error::Config(format!(
                    "mrope_section {:?} does not cover {} rotary pairs",
                    c.mrope_sections,
                    c.head_dim / 2
                )));
            }
        }
        Ok(c)
    }
}

/// The text encoder's vision tower and the prompt's image tokens.
#[derive(Debug, Clone)]
pub struct EncoderVision {
    pub config: crate::vision::VisionConfig,
    /// Blocks whose outputs feed the DeepStack mergers.
    pub deepstack: Vec<usize>,
    pub image_token: u32,
}

#[derive(Deserialize)]
struct RawVision {
    depth: usize,
    hidden_size: usize,
    intermediate_size: usize,
    num_heads: usize,
    out_hidden_size: usize,
    patch_size: usize,
    spatial_merge_size: usize,
    num_position_embeddings: usize,
    #[serde(default)]
    deepstack_visual_indexes: Vec<usize>,
}

impl EncoderVision {
    /// From `text_encoder/config.json` and the image processor's settings
    /// in `processor/processor_config.json`.
    pub fn from_json(encoder: &str, processor: &str) -> Result<Self> {
        let fail = |e: serde_json::Error| Error::Config(format!("vision config: {e}"));
        let root: serde_json::Value = serde_json::from_str(encoder).map_err(fail)?;
        let v: RawVision = serde_json::from_value(root["vision_config"].clone()).map_err(fail)?;
        let image_token = root["image_token_id"]
            .as_u64()
            .ok_or_else(|| Error::Config("text encoder config has no image_token_id".into()))?;
        let processor: serde_json::Value = serde_json::from_str(processor).map_err(fail)?;
        let p = processor.get("image_processor").unwrap_or(&processor);
        let stats = |key: &str| -> Result<[f32; 3]> {
            serde_json::from_value(p[key].clone()).map_err(fail)
        };
        let table = (v.num_position_embeddings as f64).sqrt() as usize;
        if table * table != v.num_position_embeddings
            || p["patch_size"].as_u64() != Some(v.patch_size as u64)
            || p["merge_size"].as_u64() != Some(v.spatial_merge_size as u64)
        {
            return Err(Error::Config(
                "the vision tower and the image processor disagree on its geometry".into(),
            ));
        }
        Ok(Self {
            config: crate::vision::VisionConfig {
                patch: v.patch_size,
                hidden: v.hidden_size,
                ffn: v.intermediate_size,
                layers: v.depth,
                heads: v.num_heads,
                eps: 1e-6,
                merge: v.spatial_merge_size,
                out_dim: v.out_hidden_size,
                table,
                mean: stats("image_mean")?,
                std: stats("image_std")?,
            },
            deepstack: v.deepstack_visual_indexes,
            image_token: image_token as u32,
        })
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
    /// The encoder's vision tower, which editing needs.
    pub vision: Option<EncoderVision>,
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
            vision: match read("processor/processor_config.json") {
                Ok(processor) => Some(EncoderVision::from_json(
                    &read("text_encoder/config.json")?,
                    &processor,
                )?),
                Err(_) => None,
            },
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
