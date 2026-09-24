//! Packed GGUF adapter for the shared LFM2 hybrid/MoE decoder.
#![forbid(unsafe_code)]

use super::*;
use crate::{
    loader::gguf::{GgufFile, MetadataType, MetadataValue, TensorInfo},
    model::weights::{self, ModelWeight, ModelWeights},
    quantization::{
        QuantizationFormat, QuantizedExpertMatrix, QuantizedMatrix, validate_gguf_quantized_payload,
    },
};
use std::{collections::BTreeSet, path::Path, time::Instant};

/// Load the official LFM2.5-8B-A1B GGUF weights with the matching official
/// Transformers config/tokenizer directory. Quantized expert blocks remain
/// packed and use the same MoE routing, hybrid state, and decoder as BF16.
pub fn load_gguf(
    device: &MetalDevice,
    gguf_path: impl AsRef<Path>,
    metadata_dir: impl AsRef<Path>,
) -> Result<LoadedLfm2Moe> {
    let prepared = prepare_metadata(metadata_dir.as_ref())?;
    let mut reader = GgufFile::open(gguf_path)?;
    validate_metadata(&reader, &prepared.source_config, &prepared.config)?;

    let redundant_output =
        prepared.config.tie_word_embeddings && reader.tensors().contains_key("output.weight");
    if redundant_output && !reader.tensors_equal_payload("token_embd.weight", "output.weight")? {
        return Err(Error::Weight {
            name: "output.weight".into(),
            message: "tied GGUF output weights differ from token embeddings".into(),
        });
    }
    let source_tensor_bytes = reader.tensors().values().try_fold(0usize, |total, info| {
        total
            .checked_add(info.byte_len)
            .ok_or_else(|| Error::Gguf("total source tensor bytes overflow".into()))
    })?;
    let tensor_count = reader.tensors().len();
    let parameter_count = reader
        .tensors()
        .values()
        .filter(|info| {
            !(info.name.ends_with(".exp_probs_b.bias")
                || redundant_output && info.name == "output.weight")
        })
        .try_fold(0usize, |total, info| {
            let elements = info
                .dimensions
                .iter()
                .try_fold(1usize, |count, &dimension| count.checked_mul(dimension))
                .ok_or_else(|| {
                    Error::Gguf(format!("tensor {} element count overflow", info.name))
                })?;
            total
                .checked_add(elements)
                .ok_or_else(|| Error::Gguf("total parameter count overflow".into()))
        })?;

    let weight_start = Instant::now();
    let (mapped, quantized_tensor_bytes) = map_gguf_weights(
        device,
        &prepared.config,
        &prepared.policy,
        &mut reader,
        redundant_output,
    )?;
    let weight_load = weight_start.elapsed();

    let construction_start = Instant::now();
    let model = Transformer::from_model_weights_with_policy(
        device,
        prepared.config.clone(),
        prepared.policy,
        &mapped,
    )?;
    let construction = construction_start.elapsed();

    Ok(LoadedLfm2Moe {
        config: prepared.config,
        model,
        tokenizer: prepared.tokenizer,
        eos_ids: vec![prepared.source_config.eos_token_id],
        source_tensor_bytes,
        quantized_tensor_bytes,
        tensor_count,
        parameter_count,
        config_tokenizer_load: prepared.config_tokenizer_load,
        weight_load,
        construction,
    })
}

fn validate_metadata(
    reader: &GgufFile,
    source: &Lfm2MoeConfig,
    config: &ModelConfig,
) -> Result<()> {
    let metadata = reader.metadata();
    if metadata_string(metadata, "general.architecture")? != "lfm2moe"
        || metadata_usize(metadata, "lfm2moe.vocab_size")? != config.vocab_size
        || metadata_usize(metadata, "lfm2moe.embedding_length")? != config.hidden_size
        || metadata_usize(metadata, "lfm2moe.feed_forward_length")? != config.intermediate_size
        || metadata_usize(metadata, "lfm2moe.expert_feed_forward_length")?
            != source.moe_intermediate_size
        || metadata_usize(metadata, "lfm2moe.block_count")? != config.num_layers
        || metadata_usize(metadata, "lfm2moe.attention.head_count")? != config.num_attention_heads
        || metadata_usize(metadata, "lfm2moe.expert_count")? != source.num_experts
        || metadata_usize(metadata, "lfm2moe.expert_used_count")? != source.num_experts_per_tok
        || metadata_usize(metadata, "lfm2moe.leading_dense_block_count")? != source.num_dense_layers
        || metadata_usize(metadata, "lfm2moe.context_length")? != source.max_position_embeddings
        || metadata_usize(metadata, "lfm2moe.shortconv.l_cache")? != source.conv_l_cache
        || metadata_f32(metadata, "lfm2moe.attention.layer_norm_rms_epsilon")?
            != config.rms_norm_epsilon
        || metadata_f32(metadata, "lfm2moe.rope.freq_base")? != source.rope_parameters.rope_theta
        || metadata_usize(metadata, "lfm2moe.expert_gating_func")? != 2
    {
        return Err(Error::Config(
            "LFM2-MoE GGUF metadata differs from the companion official config".into(),
        ));
    }
    let kv_heads = metadata_i32_array(metadata, "lfm2moe.attention.head_count_kv")?;
    if kv_heads.len() != source.layer_types.len()
        || kv_heads
            .iter()
            .zip(&source.layer_types)
            .any(|(&actual, kind)| {
                actual
                    != if kind == "full_attention" {
                        config.num_key_value_heads as i32
                    } else {
                        0
                    }
            })
    {
        return Err(Error::Config(
            "LFM2-MoE GGUF per-layer KV-head schedule differs from config".into(),
        ));
    }
    if metadata_string(metadata, "tokenizer.ggml.model")? != "gpt2"
        || metadata_string(metadata, "tokenizer.ggml.pre")? != "lfm2"
        || metadata_usize(metadata, "tokenizer.ggml.bos_token_id")? != source.bos_token_id as usize
        || metadata_usize(metadata, "tokenizer.ggml.eos_token_id")? != source.eos_token_id as usize
        || metadata_usize(metadata, "tokenizer.ggml.padding_token_id")?
            != source.pad_token_id as usize
        || metadata_array_len(metadata, "tokenizer.ggml.tokens")? != config.vocab_size
        || metadata
            .get("tokenizer.ggml.add_bos_token")
            .is_some_and(|value| value != &MetadataValue::Bool(false))
    {
        return Err(Error::Tokenizer(
            "LFM2-MoE GGUF tokenizer metadata differs from official tokenizer config".into(),
        ));
    }
    Ok(())
}

fn map_gguf_weights(
    device: &MetalDevice,
    config: &ModelConfig,
    policy: &ArchitecturePolicy,
    reader: &mut GgufFile,
    redundant_output: bool,
) -> Result<(ModelWeights, usize)> {
    let specifications = weights::specifications_with_policy(config, policy);
    let mut expected_sources = BTreeSet::new();
    for (canonical, _) in &specifications {
        for source in source_names(canonical)? {
            expected_sources.insert(source);
        }
    }
    if redundant_output {
        expected_sources.insert("output.weight".into());
    }
    for name in reader.tensors().keys() {
        if !expected_sources.contains(name) {
            return Err(Error::Weight {
                name: name.clone(),
                message: "unexpected tensor for the supported LFM2-MoE GGUF contract".into(),
            });
        }
    }

    let mut mapped = ModelWeights::default();
    let mut quantized_bytes = 0usize;
    for (canonical, expected_shape) in specifications {
        match canonical.as_str() {
            name if name.ends_with(".moe.input.weight") => {
                let layer = layer_index(&canonical)?;
                let LayerFeedForwardPolicy::Sparse {
                    routing,
                    intermediate_size,
                } = policy.layer(layer, config.intermediate_size)?.feed_forward
                else {
                    return Err(Error::Config(
                        "GGUF expert tensor targets a dense layer".into(),
                    ));
                };
                let expert_sources = source_names(&canonical)?;
                if expert_sources.len() != 2 {
                    return Err(Error::Config(
                        "invalid LFM2-MoE input tensor mapping".into(),
                    ));
                }
                let gate_name = expert_sources[0].clone();
                let up_name = expert_sources[1].clone();
                let expert_shape = [routing.experts, intermediate_size, config.hidden_size];
                let gate = reader.tensor(&gate_name)?.clone();
                let up = reader.tensor(&up_name)?.clone();
                validate_expert_component(&gate, &expert_shape, &gate_name)?;
                validate_expert_component(&up, &expert_shape, &up_name)?;
                let format = expert_format(&gate)?;
                if expert_format(&up)? != format {
                    return Err(Error::Weight {
                        name: canonical,
                        message: "gate/up expert tensors use different quantization formats".into(),
                    });
                }
                let experts = load_quantized_input_experts(
                    device,
                    reader,
                    &gate_name,
                    &up_name,
                    expert_shape,
                    format,
                )?;
                quantized_bytes = checked_add_bytes(quantized_bytes, experts.byte_size())?;
                if vec![
                    experts.experts(),
                    experts.rows_per_expert(),
                    experts.columns(),
                ] != expected_shape
                {
                    return Err(Error::Weight {
                        name: canonical.clone(),
                        message: "packed gate/up expert shape differs from model policy".into(),
                    });
                }
                mapped.insert(canonical, ModelWeight::QuantizedExperts(experts))?;
            }
            name if name.ends_with(".moe.output.weight") => {
                let layer = layer_index(&canonical)?;
                let LayerFeedForwardPolicy::Sparse {
                    routing,
                    intermediate_size,
                } = policy.layer(layer, config.intermediate_size)?.feed_forward
                else {
                    return Err(Error::Config(
                        "GGUF expert tensor targets a dense layer".into(),
                    ));
                };
                let source = source_names(&canonical)?.remove(0);
                let info = reader.tensor(&source)?.clone();
                validate_expert_component(
                    &info,
                    &[routing.experts, config.hidden_size, intermediate_size],
                    &source,
                )?;
                let format = expert_format(&info)?;
                let experts = load_quantized_experts(
                    device,
                    reader,
                    &source,
                    routing.experts,
                    config.hidden_size,
                    intermediate_size,
                    format,
                )?;
                quantized_bytes = checked_add_bytes(quantized_bytes, experts.byte_size())?;
                mapped.insert(canonical, ModelWeight::QuantizedExperts(experts))?;
            }
            _ => {
                let source = source_names(&canonical)?.remove(0);
                let info = reader.tensor(&source)?.clone();
                let shape = source_shape(&canonical, &info)?;
                if shape != expected_shape {
                    return Err(Error::Weight {
                        name: source,
                        message: format!(
                            "expected logical shape {expected_shape:?}, found {shape:?}"
                        ),
                    });
                }
                let value = match info.type_id {
                    0 | 1 | 30 => {
                        let dtype = if canonical.ends_with(".moe.router_bias.weight") {
                            DType::F32
                        } else {
                            config.dtype
                        };
                        ModelWeight::Dense(load_dense(
                            device, reader, &source, &info, &shape, dtype,
                        )?)
                    }
                    2 | 6 | 7 | 8 | 12 | 13 | 14 => {
                        if shape.len() != 2 {
                            return Err(Error::Weight {
                                name: source,
                                message: "quantized non-expert GGUF tensors must be matrices"
                                    .into(),
                            });
                        }
                        let format = QuantizationFormat::from_ggml_type(info.type_id)?;
                        let matrix = load_quantized_matrix(device, reader, &source, &info, format)?;
                        quantized_bytes = checked_add_bytes(quantized_bytes, matrix.byte_size())?;
                        ModelWeight::Quantized(matrix)
                    }
                    type_id => {
                        return Err(Error::Weight {
                            name: source,
                            message: format!("unsupported GGML tensor type {type_id}"),
                        });
                    }
                };
                mapped.insert(canonical, value)?;
            }
        }
    }
    Ok((mapped, quantized_bytes))
}

fn source_names(canonical: &str) -> Result<Vec<String>> {
    if canonical == "embedding.weight" {
        return Ok(vec!["token_embd.weight".into()]);
    }
    if canonical == "final_norm.weight" {
        return Ok(vec!["token_embd_norm.weight".into()]);
    }
    let parts: Vec<_> = canonical.split('.').collect();
    if parts.len() == 5 && parts[0] == "layers" && parts[2] == "moe" {
        return match parts[3] {
            "router" => Ok(vec![format!("blk.{}.ffn_gate_inp.weight", parts[1])]),
            "router_bias" => Ok(vec![format!("blk.{}.exp_probs_b.bias", parts[1])]),
            "input" => Ok(vec![
                format!("blk.{}.ffn_gate_exps.weight", parts[1]),
                format!("blk.{}.ffn_up_exps.weight", parts[1]),
            ]),
            "output" => Ok(vec![format!("blk.{}.ffn_down_exps.weight", parts[1])]),
            _ => Err(unmapped_name(canonical)),
        };
    }
    if parts.first() != Some(&"layers") {
        return Err(unmapped_name(canonical));
    }
    let component = match parts.as_slice() {
        [_, _, "input_norm", "weight"] => "attn_norm",
        [_, _, "post_norm", "weight"] => "ffn_norm",
        [_, _, "q", "weight"] => "attn_q",
        [_, _, "k", "weight"] => "attn_k",
        [_, _, "v", "weight"] => "attn_v",
        [_, _, "o", "weight"] => "attn_output",
        [_, _, "q_norm", "weight"] => "attn_q_norm",
        [_, _, "k_norm", "weight"] => "attn_k_norm",
        [_, _, "gate", "weight"] => "ffn_gate",
        [_, _, "up", "weight"] => "ffn_up",
        [_, _, "down", "weight"] => "ffn_down",
        [_, _, "conv", "in_proj", "weight"] => "shortconv.in_proj",
        [_, _, "conv", "depthwise", "weight"] => "shortconv.conv",
        [_, _, "conv", "out_proj", "weight"] => "shortconv.out_proj",
        _ => return Err(unmapped_name(canonical)),
    };
    Ok(vec![format!("blk.{}.{}.weight", parts[1], component)])
}

fn unmapped_name(canonical: &str) -> Error {
    Error::Weight {
        name: canonical.into(),
        message: "unmapped LFM2-MoE GGUF tensor".into(),
    }
}

fn layer_index(canonical: &str) -> Result<usize> {
    canonical
        .split('.')
        .nth(1)
        .ok_or_else(|| unmapped_name(canonical))?
        .parse()
        .map_err(|e| Error::Weight {
            name: canonical.into(),
            message: format!("invalid layer index: {e}"),
        })
}

fn source_shape(canonical: &str, info: &TensorInfo) -> Result<Vec<usize>> {
    match info.dimensions.len() {
        1 => Ok(info.dimensions.clone()),
        2 if canonical.ends_with(".conv.depthwise.weight") => {
            Ok(vec![info.dimensions[1], 1, info.dimensions[0]])
        }
        2 => Ok(vec![info.dimensions[1], info.dimensions[0]]),
        3 => Ok(vec![
            info.dimensions[2],
            info.dimensions[1],
            info.dimensions[0],
        ]),
        rank => Err(Error::Weight {
            name: info.name.clone(),
            message: format!("unsupported GGUF tensor rank {rank}"),
        }),
    }
}

fn validate_expert_component(info: &TensorInfo, expected: &[usize; 3], name: &str) -> Result<()> {
    if info.logical_expert_shape()? != *expected {
        return Err(Error::Weight {
            name: name.into(),
            message: format!(
                "expected logical expert shape {expected:?}, found {:?}",
                info.logical_expert_shape()?
            ),
        });
    }
    if !matches!(info.type_id, 12..=14) {
        return Err(Error::Weight {
            name: name.into(),
            message: "LFM2-MoE GGUF expert projections must use Q4_K, Q5_K, or Q6_K".into(),
        });
    }
    Ok(())
}

fn expert_format(info: &TensorInfo) -> Result<QuantizationFormat> {
    match QuantizationFormat::from_ggml_type(info.type_id)? {
        format @ (QuantizationFormat::Q4_K
        | QuantizationFormat::Q5_K
        | QuantizationFormat::Q6_K) => Ok(format),
        format => Err(Error::Weight {
            name: info.name.clone(),
            message: format!("unsupported packed expert format {format:?}"),
        }),
    }
}

fn load_quantized_matrix(
    device: &MetalDevice,
    reader: &mut GgufFile,
    name: &str,
    info: &TensorInfo,
    format: QuantizationFormat,
) -> Result<QuantizedMatrix> {
    let [rows, columns] = info.logical_matrix_shape()?;
    QuantizedMatrix::from_reader(device, rows, columns, format, |destination| {
        stream_packed_tensor(reader, name, info, format, destination)
    })
}

fn load_quantized_experts(
    device: &MetalDevice,
    reader: &mut GgufFile,
    name: &str,
    experts: usize,
    rows_per_expert: usize,
    columns: usize,
    format: QuantizationFormat,
) -> Result<QuantizedExpertMatrix> {
    let rows = experts
        .checked_mul(rows_per_expert)
        .ok_or_else(|| Error::Shape("quantized expert row count overflow".into()))?;
    let matrix = QuantizedMatrix::from_reader(device, rows, columns, format, |destination| {
        let info = reader.tensor(name)?.clone();
        stream_packed_tensor(reader, name, &info, format, destination)
    })?;
    QuantizedExpertMatrix::new(matrix, experts, rows_per_expert)
}

fn load_quantized_input_experts(
    device: &MetalDevice,
    reader: &mut GgufFile,
    gate_name: &str,
    up_name: &str,
    expert_shape: [usize; 3],
    format: QuantizationFormat,
) -> Result<QuantizedExpertMatrix> {
    let [experts, intermediate, hidden] = expert_shape;
    let rows_per_expert = intermediate
        .checked_mul(2)
        .ok_or_else(|| Error::Shape("quantized expert input width overflow".into()))?;
    let rows = experts
        .checked_mul(rows_per_expert)
        .ok_or_else(|| Error::Shape("quantized expert input row count overflow".into()))?;
    let row_bytes = packed_row_bytes(hidden, format)?;
    let component_bytes = experts
        .checked_mul(intermediate)
        .and_then(|count| count.checked_mul(row_bytes))
        .ok_or_else(|| Error::Shape("quantized expert source byte count overflow".into()))?;
    let matrix = QuantizedMatrix::from_reader(device, rows, hidden, format, |destination| {
        for (source, component_offset) in [(gate_name, 0usize), (up_name, intermediate)] {
            let info = reader.tensor(source)?.clone();
            if info.byte_len != component_bytes {
                return Err(Error::Weight {
                    name: source.into(),
                    message: format!("expected {component_bytes} packed expert bytes"),
                });
            }
            let mut written = 0usize;
            reader.stream_tensor(source, row_bytes.saturating_mul(64), |chunk| {
                validate_gguf_quantized_payload(source, format, chunk)?;
                if !chunk.len().is_multiple_of(row_bytes) {
                    return Err(Error::Gguf(format!(
                        "tensor {source} stream ended inside an expert row"
                    )));
                }
                let first_row = written / row_bytes;
                for (offset, row) in chunk.chunks_exact(row_bytes).enumerate() {
                    let source_row = first_row + offset;
                    let expert = source_row / intermediate;
                    let within_expert = source_row % intermediate;
                    if expert >= experts {
                        return Err(Error::Gguf(format!(
                            "tensor {source} contains too many expert rows"
                        )));
                    }
                    let destination_row =
                        expert * rows_per_expert + component_offset + within_expert;
                    let start = destination_row
                        .checked_mul(row_bytes)
                        .ok_or_else(|| Error::Shape("packed expert row offset overflow".into()))?;
                    destination[start..start + row_bytes].copy_from_slice(row);
                }
                written = written
                    .checked_add(chunk.len())
                    .ok_or_else(|| Error::Shape("packed expert byte count overflow".into()))?;
                Ok(())
            })?;
            if written != component_bytes {
                return Err(Error::Gguf(format!(
                    "tensor {source} streamed {written} bytes, expected {component_bytes}"
                )));
            }
        }
        Ok(())
    })?;
    QuantizedExpertMatrix::new(matrix, experts, rows_per_expert)
}

fn stream_packed_tensor(
    reader: &mut GgufFile,
    name: &str,
    info: &TensorInfo,
    format: QuantizationFormat,
    destination: &mut [u8],
) -> Result<()> {
    let mut written = 0usize;
    reader.stream_tensor(name, format.block_bytes() * 4096, |chunk| {
        validate_gguf_quantized_payload(name, format, chunk)?;
        let end = written
            .checked_add(chunk.len())
            .filter(|&end| end <= destination.len())
            .ok_or_else(|| Error::Gguf(format!("tensor {name} stream exceeds destination")))?;
        destination[written..end].copy_from_slice(chunk);
        written = end;
        Ok(())
    })?;
    if written != info.byte_len || written != destination.len() {
        return Err(Error::Gguf(format!(
            "tensor {name} streamed {written} bytes, expected {}",
            destination.len()
        )));
    }
    Ok(())
}

fn packed_row_bytes(columns: usize, format: QuantizationFormat) -> Result<usize> {
    if !columns.is_multiple_of(format.block_elements()) {
        return Err(Error::Shape(format!(
            "expert columns {columns} are not divisible by {:?} block width {}",
            format,
            format.block_elements()
        )));
    }
    (columns / format.block_elements())
        .checked_mul(format.block_bytes())
        .ok_or_else(|| Error::Shape("quantized expert row byte count overflow".into()))
}

fn load_dense(
    device: &MetalDevice,
    reader: &mut GgufFile,
    name: &str,
    info: &TensorInfo,
    dimensions: &[usize],
    dtype: DType,
) -> Result<Tensor> {
    let elements = dimensions
        .iter()
        .try_fold(1usize, |count, &dimension| count.checked_mul(dimension))
        .ok_or_else(|| Error::Shape("GGUF tensor element count overflow".into()))?;
    let source_width = match info.type_id {
        0 => 4,
        1 | 30 => 2,
        type_id => {
            return Err(Error::Weight {
                name: name.into(),
                message: format!("expected dense GGML tensor, found type {type_id}"),
            });
        }
    };
    if elements.checked_mul(source_width) != Some(info.byte_len) {
        return Err(Error::Weight {
            name: name.into(),
            message: "dense GGUF tensor byte count differs from shape".into(),
        });
    }
    if dtype == DType::F32 && info.type_id != 0 {
        return Err(Error::Weight {
            name: name.into(),
            message: "F32 routing bias must use GGML F32 storage".into(),
        });
    }
    if dtype == DType::BF16 && info.type_id == 30 {
        return Tensor::from_reader(device, dimensions, dtype, |destination| {
            reader.read_tensor_into(name, destination)
        });
    }
    Tensor::from_reader(device, dimensions, dtype, |destination| {
        let mut written = 0usize;
        reader.stream_tensor(name, source_width * 262_144, |chunk| {
            if !chunk.len().is_multiple_of(source_width) {
                return Err(Error::Gguf(format!(
                    "tensor {name} stream ended inside a dense value"
                )));
            }
            let values = chunk.len() / source_width;
            let destination_width = dtype.size_bytes();
            let end = written
                .checked_add(values * destination_width)
                .filter(|&end| end <= destination.len())
                .ok_or_else(|| {
                    Error::Gguf(format!("tensor {name} conversion exceeds destination"))
                })?;
            for (source, output) in chunk
                .chunks_exact(source_width)
                .zip(destination[written..end].chunks_exact_mut(destination_width))
            {
                let value = match info.type_id {
                    0 => f32::from_le_bytes(source.try_into().expect("4-byte F32")),
                    1 => half::f16::from_bits(u16::from_le_bytes([source[0], source[1]])).to_f32(),
                    30 => {
                        half::bf16::from_bits(u16::from_le_bytes([source[0], source[1]])).to_f32()
                    }
                    _ => unreachable!("dense type validated"),
                };
                if !value.is_finite() {
                    return Err(Error::Weight {
                        name: name.into(),
                        message: "dense GGUF weight contains a non-finite value".into(),
                    });
                }
                match dtype {
                    DType::F32 => output.copy_from_slice(&value.to_le_bytes()),
                    DType::F16 => {
                        output.copy_from_slice(&half::f16::from_f32(value).to_bits().to_le_bytes())
                    }
                    DType::BF16 => {
                        output.copy_from_slice(&half::bf16::from_f32(value).to_bits().to_le_bytes())
                    }
                }
            }
            written = end;
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
                _ => Err(Error::Config(format!(
                    "invalid i32 GGUF metadata array {key}"
                ))),
            })
            .collect(),
        _ => Err(Error::Config(format!(
            "missing or invalid GGUF metadata {key}"
        ))),
    }
}

fn metadata_array_len(
    metadata: &std::collections::BTreeMap<String, MetadataValue>,
    key: &str,
) -> Result<usize> {
    match metadata.get(key) {
        Some(MetadataValue::Array {
            element_type: MetadataType::String,
            values,
        }) if values
            .iter()
            .all(|value| matches!(value, MetadataValue::String(_))) =>
        {
            Ok(values.len())
        }
        _ => Err(Error::Tokenizer(format!(
            "missing or invalid GGUF tokenizer array {key}"
        ))),
    }
}

fn checked_add_bytes(total: usize, bytes: usize) -> Result<usize> {
    total
        .checked_add(bytes)
        .ok_or_else(|| Error::Gguf("quantized byte count overflow".into()))
}
