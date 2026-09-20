//! Deterministic synthetic fixture, never a pretrained model.
use super::{ModelConfig, weights::specifications};
use crate::{DType, Error, Result};
use std::collections::BTreeMap;
pub type CpuWeights = BTreeMap<String, (Vec<usize>, Vec<f32>)>;
pub fn weights(c: &ModelConfig) -> Result<CpuWeights> {
    c.validate()?;
    let mut specs = specifications(c);
    for l in 0..c.num_layers {
        for (name, n) in [
            ("q", c.num_attention_heads * c.head_dim),
            ("k", c.num_key_value_heads * c.head_dim),
            ("v", c.num_key_value_heads * c.head_dim),
        ] {
            specs.push((format!("layers.{l}.{name}.bias"), vec![n]));
        }
    }
    let mut out = BTreeMap::new();
    for (name, shape) in specs {
        let seed = name
            .bytes()
            .fold(17u64, |a, b| a.wrapping_mul(31).wrapping_add(b as u64));
        let norm = name.contains("norm");
        let bias = name.ends_with("bias");
        let values = (0..shape.iter().product())
            .map(|i: usize| {
                let mut z = (i as u64)
                    .wrapping_add(seed)
                    .wrapping_add(0x9e3779b97f4a7c15);
                z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
                z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
                z ^= z >> 31;
                let x = (z % 2001) as f32 / 1000. - 1.;
                c.dtype.round(if norm {
                    1. + 0.1 * x
                } else if bias {
                    0.02 * x
                } else {
                    0.15 * x
                })
            })
            .collect();
        out.insert(name, (shape, values));
    }
    Ok(out)
}
pub fn serialize(c: &ModelConfig, w: &CpuWeights) -> Result<Vec<u8>> {
    use safetensors::{Dtype, tensor::TensorView};
    let dtype = match c.dtype {
        DType::F32 => Dtype::F32,
        DType::F16 => Dtype::F16,
        DType::BF16 => Dtype::BF16,
    };
    let data: Vec<_> = w
        .iter()
        .map(|(name, (shape, values))| {
            let bytes: Vec<u8> = values
                .iter()
                .flat_map(|&v| match c.dtype {
                    DType::F32 => v.to_le_bytes().to_vec(),
                    DType::F16 => half::f16::from_f32(v).to_bits().to_le_bytes().to_vec(),
                    DType::BF16 => half::bf16::from_f32(v).to_bits().to_le_bytes().to_vec(),
                })
                .collect();
            (name, shape, bytes)
        })
        .collect();
    let views = data
        .iter()
        .map(|(name, shape, bytes)| {
            TensorView::new(dtype, (*shape).clone(), bytes).map(|v| (name.as_str(), v))
        })
        .collect::<std::result::Result<Vec<_>, _>>()
        .map_err(|e| Error::Safetensors(e.to_string()))?;
    safetensors::serialize(views, None).map_err(|e| Error::Safetensors(e.to_string()))
}
