//! Strict Qwen2/Qwen3 GGUF adapter. GGUF indexing stays model-agnostic in
//! `loader::gguf`; this module maps Qwen names and metadata into Ferrum weights.
#![forbid(unsafe_code)]

use super::{
    ModelConfig, Transformer,
    architecture::{ArchitecturePolicy, ProjectionBias},
    weights,
};
use crate::{
    DType, Error, MetalDevice, Result, Tensor,
    loader::gguf::{GgufFile, MetadataType, MetadataValue, TensorInfo},
    quantization::{QuantizationFormat, QuantizedMatrix},
    tokenizer::qwen::QwenTokenizer,
};
use std::{collections::HashSet, path::Path};

pub struct LoadedQwenGguf {
    pub architecture: String,
    pub config: ModelConfig,
    pub model: Transformer,
    pub tokenizer: QwenTokenizer,
    pub source_tensor_bytes: usize,
    pub quantized_tensor_bytes: usize,
    pub tensor_count: usize,
    pub parameter_count: usize,
}

pub fn load(device: &MetalDevice, path: impl AsRef<Path>) -> Result<LoadedQwenGguf> {
    let mut reader = GgufFile::open(path)?;
    let architecture = metadata_string(reader.metadata(), "general.architecture")?.to_owned();
    if architecture != "qwen2" && architecture != "qwen3" {
        return Err(Error::Config(format!(
            "GGUF architecture {architecture:?} is unsupported; expected qwen2 or qwen3"
        )));
    }
    let is_qwen3 = architecture == "qwen3";
    let tied_embeddings = if reader.tensors().contains_key("output.weight") {
        reader.tensors_equal_payload("token_embd.weight", "output.weight")?
    } else {
        true
    };
    let metadata = reader.metadata();
    if metadata_string(metadata, "tokenizer.ggml.model")? != "gpt2"
        || metadata_string(metadata, "tokenizer.ggml.pre")? != "qwen2"
    {
        return Err(Error::Tokenizer(
            "GGUF tokenizer must declare model=gpt2 and pre=qwen2".into(),
        ));
    }
    if metadata.get("tokenizer.ggml.add_bos_token") != Some(&MetadataValue::Bool(false)) {
        return Err(Error::Tokenizer(
            "Qwen GGUF must disable automatic BOS insertion".into(),
        ));
    }

    let embedding_info = reader.tensor("token_embd.weight")?.clone();
    let embedding_shape = embedding_info.logical_matrix_shape()?;
    let hidden_size = metadata_usize(metadata, &format!("{architecture}.embedding_length"))?;
    if embedding_shape[1] != hidden_size {
        return Err(Error::Config(format!(
            "token_embd.weight hidden dimension {} differs from metadata {hidden_size}",
            embedding_shape[1]
        )));
    }
    let num_attention_heads =
        metadata_usize(metadata, &format!("{architecture}.attention.head_count"))?;
    let num_key_value_heads =
        metadata_usize(metadata, &format!("{architecture}.attention.head_count_kv"))?;
    let head_dim = if is_qwen3 {
        let key = metadata_usize(metadata, "qwen3.attention.key_length")?;
        let value = metadata_usize(metadata, "qwen3.attention.value_length")?;
        if key != value {
            return Err(Error::Config("Qwen3 K/V head dimensions must match".into()));
        }
        key
    } else {
        match metadata.get("qwen2.rope.dimension_count") {
            Some(_) => metadata_usize(metadata, "qwen2.rope.dimension_count")?,
            None if num_attention_heads != 0 && hidden_size.is_multiple_of(num_attention_heads) => {
                hidden_size / num_attention_heads
            }
            None => {
                return Err(Error::Config(
                    "cannot infer head dimension from Qwen2 metadata".into(),
                ));
            }
        }
    };
    let rms_norm_epsilon = metadata_f32(
        metadata,
        &format!("{architecture}.attention.layer_norm_rms_epsilon"),
    )?;
    let policy = if is_qwen3 {
        ArchitecturePolicy {
            qk_norm_epsilon: Some(rms_norm_epsilon),
            qkv_bias: ProjectionBias::Forbidden,
            ..ArchitecturePolicy::default()
        }
    } else {
        ArchitecturePolicy {
            qk_norm_epsilon: None,
            qkv_bias: ProjectionBias::Required,
            ..ArchitecturePolicy::default()
        }
    };
    let config = ModelConfig {
        vocab_size: embedding_shape[0],
        hidden_size,
        intermediate_size: metadata_usize(
            metadata,
            &format!("{architecture}.feed_forward_length"),
        )?,
        num_layers: metadata_usize(metadata, &format!("{architecture}.block_count"))?,
        num_attention_heads,
        num_key_value_heads,
        head_dim,
        rms_norm_epsilon,
        rope_theta: metadata_f32(metadata, &format!("{architecture}.rope.freq_base"))?,
        max_context_length: metadata_usize(metadata, &format!("{architecture}.context_length"))?,
        tie_word_embeddings: tied_embeddings,
        dtype: DType::BF16,
    };
    config.validate()?;
    policy.validate()?;

    let tokens = metadata_string_array(metadata, "tokenizer.ggml.tokens")?;
    let merges = metadata_string_array(metadata, "tokenizer.ggml.merges")?;
    let token_types = metadata_i32_array(metadata, "tokenizer.ggml.token_type")?;
    let bos_id = metadata_usize(metadata, "tokenizer.ggml.bos_token_id")?;
    let eos_id = metadata_usize(metadata, "tokenizer.ggml.eos_token_id")?;
    let pad_id = match metadata.get("tokenizer.ggml.padding_token_id") {
        Some(_) => metadata_usize(metadata, "tokenizer.ggml.padding_token_id")?,
        None => bos_id,
    };
    let tokenizer = QwenTokenizer::from_gguf(
        &tokens,
        &merges,
        &token_types,
        config.vocab_size,
        u32::try_from(bos_id).map_err(|_| Error::Tokenizer("BOS ID exceeds u32".into()))?,
        u32::try_from(eos_id).map_err(|_| Error::Tokenizer("EOS ID exceeds u32".into()))?,
        u32::try_from(pad_id).map_err(|_| Error::Tokenizer("PAD ID exceeds u32".into()))?,
    )?;

    let mut source_names = HashSet::new();
    let mut expected = Vec::new();
    for (canonical, shape) in weights::specifications_with_policy(&config, policy) {
        let source = gguf_weight_name(&canonical)?;
        source_names.insert(source.clone());
        expected.push((source, canonical, shape));
    }
    let redundant_output = tied_embeddings && reader.tensors().contains_key("output.weight");
    if redundant_output {
        source_names.insert("output.weight".into());
    }
    let mut biases = Vec::with_capacity(config.num_layers * 3 + 1);
    if !is_qwen3 {
        for layer in 0..config.num_layers {
            for (short_name, length) in [
                ("q", config.num_attention_heads * config.head_dim),
                ("k", config.num_key_value_heads * config.head_dim),
                ("v", config.num_key_value_heads * config.head_dim),
            ] {
                let source = format!("blk.{layer}.attn_{short_name}.bias");
                let canonical = format!("layers.{layer}.{short_name}.bias");
                source_names.insert(source.clone());
                biases.push((source, canonical, vec![length]));
            }
        }
    }
    if reader.tensors().contains_key("output.bias") {
        if is_qwen3 {
            return Err(Error::Weight {
                name: "output.bias".into(),
                message: "Qwen3 output projection has no bias".into(),
            });
        }
        source_names.insert("output.bias".into());
        biases.push((
            "output.bias".into(),
            "lm_head.bias".into(),
            vec![config.vocab_size],
        ));
    }
    for source in source_names.iter() {
        if redundant_output && source == "output.weight" {
            continue;
        }
        let info = reader.tensor(source)?;
        let shape = source_shape(info)?;
        let expected_shape = expected
            .iter()
            .find(|(name, _, _)| name == source)
            .map(|(_, _, shape)| shape.as_slice())
            .or_else(|| {
                biases
                    .iter()
                    .find(|(name, _, _)| name == source)
                    .map(|(_, _, shape)| shape.as_slice())
            })
            .ok_or_else(|| Error::Weight {
                name: source.clone(),
                message: "internal GGUF mapping invariant".into(),
            })?;
        if shape != expected_shape {
            return Err(Error::Weight {
                name: source.clone(),
                message: format!("expected shape {expected_shape:?}, found {shape:?}"),
            });
        }
        validate_tensor_type(info, expected_shape.len(), source)?;
    }
    for source in reader.tensors().keys() {
        if !source_names.contains(source) {
            return Err(Error::Weight {
                name: source.clone(),
                message: "unexpected tensor for supported Qwen GGUF contract".into(),
            });
        }
    }

    let mut mixed = weights::ModelWeights::default();
    let source_tensor_bytes = reader
        .tensors()
        .values()
        .try_fold(0usize, |total, tensor| {
            total
                .checked_add(tensor.byte_len)
                .ok_or_else(|| Error::Gguf("total source tensor byte count overflow".into()))
        })?;
    let mut quantized_tensor_bytes = 0usize;
    for (source, canonical, shape) in expected.into_iter().chain(biases) {
        let info = reader.tensor(&source)?.clone();
        if matches!(info.type_id, 2 | 6 | 7 | 8 | 12 | 13 | 14) {
            if shape.len() != 2 {
                return Err(Error::Weight {
                    name: source,
                    message: "quantized GGUF types are only accepted for matrix weights".into(),
                });
            }
            let [rows, columns] = [shape[0], shape[1]];
            let format = QuantizationFormat::from_ggml_type(info.type_id)?;
            let tensor =
                QuantizedMatrix::from_reader(device, rows, columns, format, |destination| {
                    let mut written = 0usize;
                    reader.stream_tensor(&source, format.block_bytes() * 4096, |chunk| {
                        validate_quantized_payload(&source, format, chunk)?;
                        let end = written + chunk.len();
                        destination[written..end].copy_from_slice(chunk);
                        written = end;
                        Ok(())
                    })?;
                    if written != destination.len() {
                        return Err(Error::Gguf(format!(
                            "tensor {source} streamed {written} bytes, expected {}",
                            destination.len()
                        )));
                    }
                    Ok(())
                })?;
            quantized_tensor_bytes = quantized_tensor_bytes
                .checked_add(tensor.byte_size())
                .ok_or_else(|| Error::Gguf("quantized byte count overflow".into()))?;
            mixed.insert(canonical, weights::ModelWeight::Quantized(tensor))?;
        } else {
            let tensor = load_dense_bf16(device, &mut reader, &source, &info, &shape)?;
            mixed.insert(canonical, weights::ModelWeight::Dense(tensor))?;
        }
    }
    let parameter_count = reader
        .tensors()
        .values()
        .filter(|tensor| !(redundant_output && tensor.name == "output.weight"))
        .try_fold(0usize, |total, tensor| {
            let elements = tensor
                .dimensions
                .iter()
                .try_fold(1usize, |count, &dimension| count.checked_mul(dimension))
                .ok_or_else(|| {
                    Error::Gguf(format!("tensor {} element count overflow", tensor.name))
                })?;
            total
                .checked_add(elements)
                .ok_or_else(|| Error::Gguf("total parameter count overflow".into()))
        })?;
    let model =
        Transformer::from_model_weights_with_policy(device, config.clone(), policy, &mixed)?;
    Ok(LoadedQwenGguf {
        architecture,
        config,
        model,
        tokenizer,
        source_tensor_bytes,
        quantized_tensor_bytes,
        tensor_count: reader.tensors().len(),
        parameter_count,
    })
}

fn validate_quantized_payload(
    name: &str,
    format: QuantizationFormat,
    payload: &[u8],
) -> Result<()> {
    if !payload.len().is_multiple_of(format.block_bytes()) {
        return Err(Error::Gguf(format!(
            "tensor {name} {} stream ended inside a block",
            format_name(format)
        )));
    }
    for block in payload.chunks_exact(format.block_bytes()) {
        let half_at = |offset| {
            half::f16::from_bits(u16::from_le_bytes([block[offset], block[offset + 1]])).to_f32()
        };
        match format {
            QuantizationFormat::Q4_0 => {
                if !half_at(0).is_finite() {
                    return Err(invalid_quantized_scale(name, format));
                }
            }
            QuantizationFormat::Q5_0 => {
                if !half_at(0).is_finite() {
                    return Err(invalid_quantized_scale(name, format));
                }
            }
            QuantizationFormat::Q5_1 => {
                if !half_at(0).is_finite() || !half_at(2).is_finite() {
                    return Err(invalid_quantized_scale(name, format));
                }
            }
            QuantizationFormat::Q4_K | QuantizationFormat::Q5_K => {
                if !half_at(0).is_finite() || !half_at(2).is_finite() {
                    return Err(invalid_quantized_scale(name, format));
                }
            }
            QuantizationFormat::MlxAffine4Group64 => {
                if !half_at(0).is_finite() || !half_at(2).is_finite() {
                    return Err(invalid_quantized_scale(name, format));
                }
            }
            QuantizationFormat::Q8_0 => {
                let scale = half_at(0);
                if !scale.is_finite() || scale < 0.0 {
                    return Err(invalid_quantized_scale(name, format));
                }
                if block[2..].contains(&0x80) {
                    return Err(Error::Weight {
                        name: name.into(),
                        message: "Q8_0 value -128 is outside the GGML range".into(),
                    });
                }
            }
            QuantizationFormat::Q6_K => {
                if !half_at(208).is_finite() {
                    return Err(invalid_quantized_scale(name, format));
                }
            }
        }
    }
    Ok(())
}

fn invalid_quantized_scale(name: &str, format: QuantizationFormat) -> Error {
    Error::Weight {
        name: name.into(),
        message: format!("{} scale must be finite", format_name(format)),
    }
}

fn format_name(format: QuantizationFormat) -> &'static str {
    match format {
        QuantizationFormat::Q4_0 => "Q4_0",
        QuantizationFormat::Q5_0 => "Q5_0",
        QuantizationFormat::Q5_1 => "Q5_1",
        QuantizationFormat::Q4_K => "Q4_K",
        QuantizationFormat::Q5_K => "Q5_K",
        QuantizationFormat::Q8_0 => "Q8_0",
        QuantizationFormat::Q6_K => "Q6_K",
        QuantizationFormat::MlxAffine4Group64 => "MLX affine 4-bit group-64",
    }
}

fn gguf_weight_name(canonical: &str) -> Result<String> {
    if canonical == "embedding.weight" {
        return Ok("token_embd.weight".into());
    }
    if canonical == "final_norm.weight" {
        return Ok("output_norm.weight".into());
    }
    if canonical == "lm_head.weight" {
        return Ok("output.weight".into());
    }
    let parts = canonical.split('.').collect::<Vec<_>>();
    if parts.len() != 4 || parts[0] != "layers" {
        return Err(Error::Weight {
            name: canonical.into(),
            message: "cannot map canonical Qwen weight".into(),
        });
    }
    let layer = parts[1];
    let component = match parts[2] {
        "input_norm" => "attn_norm",
        "post_norm" => "ffn_norm",
        "q" => "attn_q",
        "k" => "attn_k",
        "v" => "attn_v",
        "o" => "attn_output",
        "q_norm" => "attn_q_norm",
        "k_norm" => "attn_k_norm",
        "gate" => "ffn_gate",
        "up" => "ffn_up",
        "down" => "ffn_down",
        _ => {
            return Err(Error::Weight {
                name: canonical.into(),
                message: "unsupported Qwen weight component".into(),
            });
        }
    };
    Ok(format!("blk.{layer}.{component}.{}", parts[3]))
}

fn source_shape(info: &TensorInfo) -> Result<Vec<usize>> {
    match info.dimensions.len() {
        1 => Ok(info.dimensions.clone()),
        2 => Ok(vec![info.dimensions[1], info.dimensions[0]]),
        _ => Err(Error::Weight {
            name: info.name.clone(),
            message: format!("unsupported Qwen tensor rank {}", info.dimensions.len()),
        }),
    }
}

fn validate_tensor_type(info: &TensorInfo, rank: usize, name: &str) -> Result<()> {
    match info.type_id {
        0 | 1 | 30 => Ok(()),
        2 | 6 | 7 | 8 | 12 | 13 | 14 if rank == 2 => Ok(()),
        type_id => Err(Error::Weight {
            name: name.into(),
            message: format!("unsupported GGML type {type_id} for Qwen tensor"),
        }),
    }
}

fn load_dense_bf16(
    device: &MetalDevice,
    reader: &mut GgufFile,
    name: &str,
    info: &TensorInfo,
    dimensions: &[usize],
) -> Result<Tensor> {
    if info.type_id == 30 {
        return Tensor::from_reader(device, dimensions, DType::BF16, |destination| {
            reader.read_tensor_into(name, destination)
        });
    }
    let source_width = match info.type_id {
        0 => 4,
        1 => 2,
        _ => {
            return Err(Error::Weight {
                name: name.into(),
                message: format!("expected a dense GGML type, found {}", info.type_id),
            });
        }
    };
    Tensor::from_reader(device, dimensions, DType::BF16, |destination| {
        let mut written = 0usize;
        reader.stream_tensor(name, 1024 * 1024, |chunk| {
            if !chunk.len().is_multiple_of(source_width) {
                return Err(Error::Gguf(format!(
                    "tensor {name} stream split a dense element"
                )));
            }
            for source in chunk.chunks_exact(source_width) {
                let value = if source_width == 4 {
                    f32::from_le_bytes([source[0], source[1], source[2], source[3]])
                } else {
                    half::f16::from_bits(u16::from_le_bytes([source[0], source[1]])).to_f32()
                };
                if !value.is_finite() {
                    return Err(Error::Weight {
                        name: name.into(),
                        message: "dense GGUF weight contains a non-finite value".into(),
                    });
                }
                let end = written + 2;
                if end > destination.len() {
                    return Err(Error::Gguf(format!(
                        "tensor {name} conversion exceeds destination"
                    )));
                }
                destination[written..end]
                    .copy_from_slice(&half::bf16::from_f32(value).to_bits().to_le_bytes());
                written = end;
            }
            Ok(())
        })?;
        if written != destination.len() {
            return Err(Error::Gguf(format!(
                "tensor {name} conversion wrote {written} bytes, expected {}",
                destination.len()
            )));
        }
        Ok(())
    })
}

fn metadata_string<'a>(
    metadata: &'a std::collections::BTreeMap<String, MetadataValue>,
    key: &str,
) -> Result<&'a str> {
    match metadata.get(key) {
        Some(MetadataValue::String(value)) => Ok(value),
        _ => Err(Error::Config(format!(
            "missing or invalid GGUF metadata {key}"
        ))),
    }
}

fn metadata_usize(
    metadata: &std::collections::BTreeMap<String, MetadataValue>,
    key: &str,
) -> Result<usize> {
    let value = match metadata.get(key) {
        Some(MetadataValue::Uint32(value)) => u64::from(*value),
        Some(MetadataValue::Uint64(value)) => *value,
        _ => {
            return Err(Error::Config(format!(
                "missing or invalid GGUF metadata {key}"
            )));
        }
    };
    usize::try_from(value).map_err(|_| Error::Config(format!("GGUF metadata {key} exceeds usize")))
}

fn metadata_f32(
    metadata: &std::collections::BTreeMap<String, MetadataValue>,
    key: &str,
) -> Result<f32> {
    match metadata.get(key) {
        Some(MetadataValue::Float32(value)) => Ok(*value),
        Some(MetadataValue::Float64(value)) => Ok(*value as f32),
        _ => Err(Error::Config(format!(
            "missing or invalid GGUF metadata {key}"
        ))),
    }
}

fn metadata_string_array(
    metadata: &std::collections::BTreeMap<String, MetadataValue>,
    key: &str,
) -> Result<Vec<String>> {
    match metadata.get(key) {
        Some(MetadataValue::Array {
            element_type: MetadataType::String,
            values,
        }) => values
            .iter()
            .map(|value| match value {
                MetadataValue::String(value) => Ok(value.clone()),
                _ => Err(Error::Tokenizer(format!("invalid string array {key}"))),
            })
            .collect(),
        _ => Err(Error::Tokenizer(format!(
            "missing or invalid GGUF tokenizer array {key}"
        ))),
    }
}

fn metadata_i32_array(
    metadata: &std::collections::BTreeMap<String, MetadataValue>,
    key: &str,
) -> Result<Vec<i32>> {
    match metadata.get(key) {
        Some(MetadataValue::Array {
            element_type: MetadataType::Int32,
            values,
        }) => values
            .iter()
            .map(|value| match value {
                MetadataValue::Int32(value) => Ok(*value),
                _ => Err(Error::Tokenizer(format!("invalid int32 array {key}"))),
            })
            .collect(),
        _ => Err(Error::Tokenizer(format!(
            "missing or invalid GGUF tokenizer array {key}"
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn q8_block(scale: f32, quantized: u8) -> Vec<u8> {
        let mut block = Vec::with_capacity(34);
        block.extend_from_slice(&half::f16::from_f32(scale).to_bits().to_le_bytes());
        block.extend(std::iter::repeat_n(quantized, 32));
        block
    }

    #[test]
    fn q8_0_loader_checks_scale_range_and_block_boundary() {
        assert!(
            validate_quantized_payload("weight", QuantizationFormat::Q8_0, &q8_block(0.25, 127))
                .is_ok()
        );
        assert!(
            validate_quantized_payload("weight", QuantizationFormat::Q8_0, &q8_block(0.0, 0))
                .is_ok()
        );
        assert!(
            validate_quantized_payload(
                "weight",
                QuantizationFormat::Q8_0,
                &q8_block(f32::INFINITY, 0)
            )
            .is_err()
        );
        assert!(
            validate_quantized_payload("weight", QuantizationFormat::Q8_0, &q8_block(-0.25, 0))
                .is_err()
        );
        assert!(
            validate_quantized_payload("weight", QuantizationFormat::Q8_0, &q8_block(0.25, 0x80))
                .is_err()
        );
        assert!(validate_quantized_payload("weight", QuantizationFormat::Q8_0, &[0; 33]).is_err());
    }

    #[test]
    fn q4_0_loader_checks_scale_and_block_boundary() {
        let mut block = vec![0u8; 18];
        block[..2].copy_from_slice(&half::f16::from_f32(0.125).to_bits().to_le_bytes());
        assert!(validate_quantized_payload("weight", QuantizationFormat::Q4_0, &block).is_ok());
        block[..2].copy_from_slice(&half::f16::from_f32(-0.125).to_bits().to_le_bytes());
        assert!(validate_quantized_payload("weight", QuantizationFormat::Q4_0, &block).is_ok());
        block[..2].copy_from_slice(&half::f16::from_f32(f32::INFINITY).to_bits().to_le_bytes());
        assert!(validate_quantized_payload("weight", QuantizationFormat::Q4_0, &block).is_err());
        assert!(validate_quantized_payload("weight", QuantizationFormat::Q4_0, &[0; 17]).is_err());
    }

    #[test]
    fn q5_0_loader_checks_scale_and_block_boundary() {
        let mut block = vec![0u8; 22];
        block[..2].copy_from_slice(&half::f16::from_f32(-0.125).to_bits().to_le_bytes());
        assert!(validate_quantized_payload("weight", QuantizationFormat::Q5_0, &block).is_ok());
        block[..2].copy_from_slice(&half::f16::from_f32(f32::INFINITY).to_bits().to_le_bytes());
        assert!(validate_quantized_payload("weight", QuantizationFormat::Q5_0, &block).is_err());
        assert!(validate_quantized_payload("weight", QuantizationFormat::Q5_0, &[0; 21]).is_err());
    }

    #[test]
    fn q5_1_loader_checks_scale_minimum_and_block_boundary() {
        let mut block = vec![0u8; 24];
        block[..2].copy_from_slice(&half::f16::from_f32(0.125).to_bits().to_le_bytes());
        block[2..4].copy_from_slice(&half::f16::from_f32(-0.25).to_bits().to_le_bytes());
        assert!(validate_quantized_payload("weight", QuantizationFormat::Q5_1, &block).is_ok());
        block[2..4].copy_from_slice(&half::f16::from_f32(f32::INFINITY).to_bits().to_le_bytes());
        assert!(validate_quantized_payload("weight", QuantizationFormat::Q5_1, &block).is_err());
        assert!(validate_quantized_payload("weight", QuantizationFormat::Q5_1, &[0; 23]).is_err());
    }

    #[test]
    fn q4_k_loader_checks_both_superblock_scales_and_boundary() {
        let mut block = vec![0u8; 144];
        block[..2].copy_from_slice(&half::f16::from_f32(0.125).to_bits().to_le_bytes());
        block[2..4].copy_from_slice(&half::f16::from_f32(0.25).to_bits().to_le_bytes());
        assert!(validate_quantized_payload("weight", QuantizationFormat::Q4_K, &block).is_ok());
        block[2..4].copy_from_slice(&half::f16::from_f32(f32::INFINITY).to_bits().to_le_bytes());
        assert!(validate_quantized_payload("weight", QuantizationFormat::Q4_K, &block).is_err());
        assert!(validate_quantized_payload("weight", QuantizationFormat::Q4_K, &[0; 143]).is_err());
    }

    #[test]
    fn q5_k_loader_checks_both_superblock_scales_and_boundary() {
        let mut block = vec![0u8; 176];
        block[..2].copy_from_slice(&half::f16::from_f32(0.125).to_bits().to_le_bytes());
        block[2..4].copy_from_slice(&half::f16::from_f32(0.25).to_bits().to_le_bytes());
        assert!(validate_quantized_payload("weight", QuantizationFormat::Q5_K, &block).is_ok());
        block[..2].copy_from_slice(&half::f16::from_f32(f32::NAN).to_bits().to_le_bytes());
        assert!(validate_quantized_payload("weight", QuantizationFormat::Q5_K, &block).is_err());
        assert!(validate_quantized_payload("weight", QuantizationFormat::Q5_K, &[0; 175]).is_err());
    }

    #[test]
    fn q6_k_loader_checks_superblock_scale_and_boundary() {
        let mut block = vec![0u8; 210];
        block[208..210].copy_from_slice(&half::f16::from_f32(0.125).to_bits().to_le_bytes());
        assert!(validate_quantized_payload("weight", QuantizationFormat::Q6_K, &block).is_ok());
        block[208..210]
            .copy_from_slice(&half::f16::from_f32(f32::INFINITY).to_bits().to_le_bytes());
        assert!(validate_quantized_payload("weight", QuantizationFormat::Q6_K, &block).is_err());
        assert!(validate_quantized_payload("weight", QuantizationFormat::Q6_K, &[0; 209]).is_err());
    }
}
