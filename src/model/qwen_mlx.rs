//! Strict loader for MLX affine 4-bit, group-64 Qwen2 safetensors.
//!
//! MLX stores packed U32 words and separate F16 scales/biases. Ferrum repacks
//! those exact quantized values into 64-value blocks (scale, bias, 32 Q4 bytes)
//! while streaming one source tensor at a time. It never expands a matrix.
#![forbid(unsafe_code)]

use super::{ModelConfig, Transformer, qwen, weights};
use crate::{
    DType, Error, MetalDevice, Result, Tensor,
    model::weights::ModelWeight,
    quantization::{QuantizationFormat, QuantizedMatrix},
    tokenizer::qwen::QwenTokenizer,
};
use half::f16;
use serde_json::Value;
use std::{
    collections::{BTreeMap, HashSet},
    fs::File,
    io::{Read, Seek, SeekFrom},
    path::Path,
    time::{Duration, Instant},
};

const MAX_SAFETENSORS_HEADER_BYTES: usize = 64 * 1024 * 1024;
const MLX_GROUP_SIZE: usize = 64;
const MLX_PACK_FACTOR: usize = 8;
const MLX_BLOCK_BYTES: usize = 36;

pub struct LoadedQwenMlx {
    pub config: ModelConfig,
    pub model: Transformer,
    pub tokenizer: QwenTokenizer,
    pub source_tensor_bytes: usize,
    pub quantized_tensor_bytes: usize,
    pub tensor_count: usize,
    pub parameter_count: usize,
    pub config_tokenizer_load: Duration,
    pub weight_load: Duration,
    pub construction: Duration,
}

#[derive(Clone, Debug)]
struct TensorIndex {
    dtype: String,
    shape: Vec<usize>,
    start: usize,
    end: usize,
    byte_len: usize,
}

/// Load the supported MLX affine Q4 Qwen checkpoint without an MLX runtime.
pub fn load(device: &MetalDevice, path: impl AsRef<Path>) -> Result<LoadedQwenMlx> {
    let path = path.as_ref();
    if !path.is_dir() {
        return Err(Error::Config(format!(
            "MLX Qwen model path must be a directory: {}",
            path.display()
        )));
    }
    let config_tokenizer_start = Instant::now();
    let source_config = qwen::QwenConfig::from_file(path.join("config.json"))?;
    let config = source_config.convert_mlx_affine4()?;
    if !config.tie_word_embeddings {
        return Err(Error::Config(
            "MLX affine Q4 loader currently requires tied Qwen embeddings".into(),
        ));
    }
    let tokenizer = QwenTokenizer::load_mlx_affine4(path, &source_config)?;
    let config_tokenizer_load = config_tokenizer_start.elapsed();

    let weight_load_start = Instant::now();
    let weight_path = path.join("model.safetensors");
    let mut file = File::open(&weight_path)
        .map_err(|e| Error::Safetensors(format!("{}: {e}", weight_path.display())))?;
    let file_len = file
        .metadata()
        .map_err(|e| Error::Safetensors(format!("{}: {e}", weight_path.display())))?
        .len();
    let mut length_bytes = [0u8; 8];
    file.read_exact(&mut length_bytes)
        .map_err(|e| Error::Safetensors(format!("{}: {e}", weight_path.display())))?;
    let header_len = usize::try_from(u64::from_le_bytes(length_bytes))
        .map_err(|_| Error::Safetensors("safetensors header length exceeds usize".into()))?;
    if header_len > MAX_SAFETENSORS_HEADER_BYTES {
        return Err(Error::Safetensors(format!(
            "safetensors header is too large: {header_len} bytes"
        )));
    }
    let data_start = 8usize
        .checked_add(header_len)
        .ok_or_else(|| Error::Safetensors("safetensors header offset overflow".into()))?;
    let data_start_u64 = u64::try_from(data_start)
        .map_err(|_| Error::Safetensors("safetensors header offset exceeds u64".into()))?;
    if data_start_u64 > file_len {
        return Err(Error::Safetensors(
            "safetensors header extends beyond end of file".into(),
        ));
    }
    let data_len = usize::try_from(file_len - data_start_u64)
        .map_err(|_| Error::Safetensors("safetensors data size exceeds usize".into()))?;
    let mut header = vec![0u8; header_len];
    file.read_exact(&mut header)
        .map_err(|e| Error::Safetensors(format!("{}: {e}", weight_path.display())))?;
    let index = parse_index(&header, data_len)?;

    // Validate every name, shape and dtype before allocating any Metal weights.
    let specifications = qwen::specifications(&config);
    let mut expected_names = HashSet::with_capacity(index.len());
    let mut parameter_count = 0usize;
    for (source, _canonical, shape) in &specifications {
        let info = index.get(source).ok_or_else(|| Error::Weight {
            name: source.clone(),
            message: "missing MLX safetensors entry".into(),
        })?;
        parameter_count = parameter_count
            .checked_add(numel(shape)?)
            .ok_or_else(|| Error::Shape("Qwen parameter count overflow".into()))?;
        if source.ends_with(".weight") && info.dtype == "U32" {
            if shape.len() != 2 || !shape[1].is_multiple_of(MLX_GROUP_SIZE) {
                return Err(Error::Weight {
                    name: source.clone(),
                    message: "MLX affine Q4 matrix dimensions must use group size 64".into(),
                });
            }
            let packed_shape = [shape[0], shape[1] / MLX_PACK_FACTOR];
            check_tensor_info(source, info, "U32", &packed_shape)?;
            let groups = shape[1] / MLX_GROUP_SIZE;
            let scales_name = companion_name(source, "scales");
            let biases_name = companion_name(source, "biases");
            check_tensor_info(
                &scales_name,
                required_info(&index, &scales_name)?,
                "F16",
                &[shape[0], groups],
            )?;
            check_tensor_info(
                &biases_name,
                required_info(&index, &biases_name)?,
                "F16",
                &[shape[0], groups],
            )?;
            expected_names.insert(scales_name.clone());
            expected_names.insert(biases_name.clone());
        } else {
            check_tensor_info(source, info, "F16", shape)?;
        }
        expected_names.insert(source.clone());
    }
    if let Some(extra) = index.keys().find(|name| !expected_names.contains(*name)) {
        return Err(Error::Weight {
            name: extra.clone(),
            message: "unexpected tensor for supported MLX affine Q4 Qwen contract".into(),
        });
    }
    if expected_names.len() != index.len() {
        return Err(Error::Safetensors(
            "MLX Qwen tensor index contains duplicate or unreferenced entries".into(),
        ));
    }

    let mut model_weights = weights::ModelWeights::default();
    let mut quantized_tensor_bytes = 0usize;
    for (source, canonical, shape) in &specifications {
        let info = required_info(&index, source)?;
        if source.ends_with(".weight") && info.dtype == "U32" {
            let weight_bytes = read_tensor(&mut file, data_start, &index, source)?;
            let scales_name = companion_name(source, "scales");
            let biases_name = companion_name(source, "biases");
            let scales = read_tensor(&mut file, data_start, &index, &scales_name)?;
            let biases = read_tensor(&mut file, data_start, &index, &biases_name)?;
            // MLX affine scales are signed in published checkpoints; the
            // affine contract only requires finite scale and bias values.
            validate_f16_metadata(&scales_name, &scales, false)?;
            validate_f16_metadata(&biases_name, &biases, false)?;
            let tensor =
                repack_affine4(device, shape[0], shape[1], &weight_bytes, &scales, &biases)?;
            quantized_tensor_bytes = quantized_tensor_bytes
                .checked_add(tensor.byte_size())
                .ok_or_else(|| Error::Shape("MLX packed-weight byte count overflow".into()))?;
            model_weights.insert(canonical.clone(), ModelWeight::Quantized(tensor))?;
        } else {
            let bytes = read_tensor(&mut file, data_start, &index, source)?;
            validate_f16_metadata(source, &bytes, false)?;
            let tensor = Tensor::from_le_bytes(device, shape, DType::F16, &bytes).map_err(|e| {
                Error::Weight {
                    name: source.clone(),
                    message: e.to_string(),
                }
            })?;
            model_weights.insert(canonical.clone(), ModelWeight::Dense(tensor))?;
        }
    }
    let weight_load = weight_load_start.elapsed();
    let construction_start = Instant::now();
    let model = Transformer::from_model_weights(device, config.clone(), &model_weights)?;
    let construction = construction_start.elapsed();
    Ok(LoadedQwenMlx {
        config,
        model,
        tokenizer,
        source_tensor_bytes: data_len,
        quantized_tensor_bytes,
        tensor_count: index.len(),
        parameter_count,
        config_tokenizer_load,
        weight_load,
        construction,
    })
}

fn parse_index(header: &[u8], data_len: usize) -> Result<BTreeMap<String, TensorIndex>> {
    let value: Value = serde_json::from_slice(header)
        .map_err(|e| Error::Safetensors(format!("invalid safetensors header JSON: {e}")))?;
    let object = value
        .as_object()
        .ok_or_else(|| Error::Safetensors("safetensors header must be a JSON object".into()))?;
    let mut index = BTreeMap::new();
    for (name, value) in object {
        if name == "__metadata__" {
            if !value
                .as_object()
                .is_some_and(|metadata| metadata.values().all(serde_json::Value::is_string))
            {
                return Err(Error::Safetensors(
                    "safetensors __metadata__ must contain string values".into(),
                ));
            }
            continue;
        }
        let entry = value.as_object().ok_or_else(|| {
            Error::Safetensors(format!("safetensors entry {name:?} must be an object"))
        })?;
        let dtype = entry
            .get("dtype")
            .and_then(Value::as_str)
            .ok_or_else(|| Error::Safetensors(format!("safetensors entry {name:?} has no dtype")))?
            .to_owned();
        let item_size = dtype_size(&dtype).ok_or_else(|| {
            Error::Safetensors(format!(
                "unsupported MLX safetensors dtype {dtype:?} in tensor {name:?}"
            ))
        })?;
        let dims = entry
            .get("shape")
            .and_then(Value::as_array)
            .ok_or_else(|| {
                Error::Safetensors(format!("safetensors entry {name:?} has no shape"))
            })?;
        let mut shape = Vec::with_capacity(dims.len());
        for dim in dims {
            let dim = dim.as_u64().ok_or_else(|| {
                Error::Safetensors(format!("invalid dimension in tensor {name:?}"))
            })?;
            shape.push(usize::try_from(dim).map_err(|_| {
                Error::Safetensors(format!("dimension exceeds usize in tensor {name:?}"))
            })?);
        }
        if shape.contains(&0) {
            return Err(Error::Safetensors(format!(
                "zero-length dimensions are unsupported in tensor {name:?}"
            )));
        }
        let offsets = entry
            .get("data_offsets")
            .and_then(Value::as_array)
            .filter(|offsets| offsets.len() == 2)
            .ok_or_else(|| {
                Error::Safetensors(format!("invalid data_offsets in tensor {name:?}"))
            })?;
        let start = offsets[0]
            .as_u64()
            .and_then(|v| usize::try_from(v).ok())
            .ok_or_else(|| {
                Error::Safetensors(format!("invalid start offset in tensor {name:?}"))
            })?;
        let end = offsets[1]
            .as_u64()
            .and_then(|v| usize::try_from(v).ok())
            .ok_or_else(|| Error::Safetensors(format!("invalid end offset in tensor {name:?}")))?;
        if end < start {
            return Err(Error::Safetensors(format!(
                "reversed data offsets in tensor {name:?}"
            )));
        }
        let byte_len = numel(&shape)?
            .checked_mul(item_size)
            .ok_or_else(|| Error::Safetensors(format!("byte size overflow in tensor {name:?}")))?;
        if end - start != byte_len || end > data_len {
            return Err(Error::Safetensors(format!(
                "tensor {name:?} range has {} bytes, expected {byte_len}, data size is {data_len}",
                end - start
            )));
        }
        index.insert(
            name.clone(),
            TensorIndex {
                dtype,
                shape,
                start,
                end,
                byte_len,
            },
        );
    }
    if index.is_empty() {
        return Err(Error::Safetensors("safetensors file has no tensors".into()));
    }
    let mut ranges: Vec<_> = index.iter().collect();
    ranges.sort_by_key(|(_, info)| info.start);
    let mut cursor = 0usize;
    for (name, info) in ranges {
        if info.start != cursor {
            return Err(Error::Safetensors(format!(
                "tensor {name:?} overlaps or leaves a gap in safetensors data"
            )));
        }
        cursor = info.end;
    }
    if cursor != data_len {
        return Err(Error::Safetensors(format!(
            "safetensors data has {} unindexed trailing bytes",
            data_len - cursor
        )));
    }
    Ok(index)
}

fn dtype_size(dtype: &str) -> Option<usize> {
    match dtype {
        "F16" => Some(2),
        "U32" => Some(4),
        _ => None,
    }
}

fn numel(shape: &[usize]) -> Result<usize> {
    shape.iter().try_fold(1usize, |n, &dim| {
        n.checked_mul(dim)
            .ok_or_else(|| Error::Shape("safetensors tensor element count overflow".into()))
    })
}

fn required_info<'a>(
    index: &'a BTreeMap<String, TensorIndex>,
    name: &str,
) -> Result<&'a TensorIndex> {
    index.get(name).ok_or_else(|| Error::Weight {
        name: name.into(),
        message: "missing MLX quantization tensor".into(),
    })
}

fn check_tensor_info(name: &str, info: &TensorIndex, dtype: &str, shape: &[usize]) -> Result<()> {
    if info.dtype != dtype || info.shape != shape {
        return Err(Error::Weight {
            name: name.into(),
            message: format!(
                "expected safetensors {dtype} {shape:?}, found {} {:?}",
                info.dtype, info.shape
            ),
        });
    }
    Ok(())
}

fn companion_name(weight: &str, suffix: &str) -> String {
    format!("{}.{suffix}", weight.trim_end_matches(".weight"))
}

fn read_tensor(
    file: &mut File,
    data_start: usize,
    index: &BTreeMap<String, TensorIndex>,
    name: &str,
) -> Result<Vec<u8>> {
    let info = required_info(index, name)?;
    let offset = data_start
        .checked_add(info.start)
        .ok_or_else(|| Error::Safetensors(format!("file offset overflow in tensor {name:?}")))?;
    file.seek(SeekFrom::Start(u64::try_from(offset).map_err(|_| {
        Error::Safetensors(format!("file offset exceeds u64 in tensor {name:?}"))
    })?))
    .map_err(|e| Error::Safetensors(format!("seek tensor {name:?}: {e}")))?;
    let mut bytes = vec![0u8; info.byte_len];
    file.read_exact(&mut bytes)
        .map_err(|e| Error::Safetensors(format!("read tensor {name:?}: {e}")))?;
    Ok(bytes)
}

fn validate_f16_metadata(name: &str, bytes: &[u8], nonnegative: bool) -> Result<()> {
    if !bytes.len().is_multiple_of(2) {
        return Err(Error::Weight {
            name: name.into(),
            message: "F16 tensor byte length is not aligned".into(),
        });
    }
    for chunk in bytes.chunks_exact(2) {
        let value = f16::from_bits(u16::from_le_bytes([chunk[0], chunk[1]])).to_f32();
        if !value.is_finite() || (nonnegative && value < 0.0) {
            return Err(Error::Weight {
                name: name.into(),
                message: "MLX affine scale/bias contains an invalid F16 value".into(),
            });
        }
    }
    Ok(())
}

fn repack_affine4(
    device: &MetalDevice,
    rows: usize,
    columns: usize,
    weight: &[u8],
    scales: &[u8],
    biases: &[u8],
) -> Result<QuantizedMatrix> {
    if columns == 0 || !columns.is_multiple_of(MLX_GROUP_SIZE) {
        return Err(Error::Shape(
            "MLX affine Q4 columns must be divisible by group size 64".into(),
        ));
    }
    let groups = columns / MLX_GROUP_SIZE;
    let expected_weight = rows
        .checked_mul(columns / 2)
        .ok_or_else(|| Error::Shape("MLX packed weight size overflow".into()))?;
    let expected_metadata = rows
        .checked_mul(groups)
        .and_then(|count| count.checked_mul(2))
        .ok_or_else(|| Error::Shape("MLX scale/bias size overflow".into()))?;
    if weight.len() != expected_weight
        || scales.len() != expected_metadata
        || biases.len() != expected_metadata
    {
        return Err(Error::Shape(format!(
            "invalid MLX affine Q4 payload lengths: weight {}, scales {}, biases {}; expected {expected_weight}, {expected_metadata}, {expected_metadata}",
            weight.len(),
            scales.len(),
            biases.len()
        )));
    }
    // Safetensors stores U32 words, eight Q4 codes per word. In the little-
    // endian payload those codes are byte-identical to two nibbles per byte.
    let row_source_bytes = columns / 2;
    QuantizedMatrix::from_reader(
        device,
        rows,
        columns,
        QuantizationFormat::MlxAffine4Group64,
        |destination| {
            let mut output = 0usize;
            for row in 0..rows {
                for group in 0..groups {
                    let metadata_index = (row * groups + group) * 2;
                    destination[output..output + 2]
                        .copy_from_slice(&scales[metadata_index..metadata_index + 2]);
                    destination[output + 2..output + 4]
                        .copy_from_slice(&biases[metadata_index..metadata_index + 2]);
                    let source = row * row_source_bytes + group * (MLX_GROUP_SIZE / 2);
                    destination[output + 4..output + MLX_BLOCK_BYTES]
                        .copy_from_slice(&weight[source..source + MLX_GROUP_SIZE / 2]);
                    output += MLX_BLOCK_BYTES;
                }
            }
            debug_assert_eq!(output, destination.len());
            Ok(())
        },
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn safetensors_index_checks_dtype_geometry_ranges_and_trailing_bytes() {
        let header = json!({
            "first": {"dtype":"U32", "shape":[2], "data_offsets":[0,8]},
            "second": {"dtype":"F16", "shape":[2], "data_offsets":[8,12]},
            "__metadata__": {"format":"mlx"}
        });
        let bytes = serde_json::to_vec(&header).unwrap();
        let parsed = parse_index(&bytes, 12).unwrap();
        assert_eq!(parsed.len(), 2);
        assert!(parse_index(&bytes, 13).is_err());

        let overlap = json!({
            "first": {"dtype":"U32", "shape":[2], "data_offsets":[0,8]},
            "second": {"dtype":"F16", "shape":[2], "data_offsets":[4,8]}
        });
        assert!(parse_index(&serde_json::to_vec(&overlap).unwrap(), 8).is_err());

        let unsupported = json!({
            "weight": {"dtype":"BF16", "shape":[1], "data_offsets":[0,2]}
        });
        assert!(parse_index(&serde_json::to_vec(&unsupported).unwrap(), 2).is_err());
    }

    #[test]
    fn mlx_affine_repack_preserves_u4_codes_and_f16_metadata() {
        let device = MetalDevice::new().unwrap();
        let rows = 2;
        let columns = 128;
        let weight = (0..rows * columns / 2)
            .map(|i| (i * 17 + 3) as u8)
            .collect::<Vec<_>>();
        let scales = [0.125, 0.25, 0.375, 0.5]
            .into_iter()
            .map(f16::from_f32)
            .flat_map(|v| v.to_bits().to_le_bytes())
            .collect::<Vec<_>>();
        let biases = [-0.5, -0.25, -0.125, 0.0]
            .into_iter()
            .map(f16::from_f32)
            .flat_map(|v| v.to_bits().to_le_bytes())
            .collect::<Vec<_>>();
        let matrix = repack_affine4(&device, rows, columns, &weight, &scales, &biases).unwrap();
        assert_eq!(matrix.byte_size(), rows * 2 * MLX_BLOCK_BYTES);
        matrix.with_bytes(|packed| {
            assert_eq!(&packed[..4], &[0, 0x30, 0, 0xb8]);
            assert_eq!(&packed[4..36], &weight[..32]);
            assert_eq!(&packed[36..40], &[0, 0x34, 0, 0xb4]);
            assert_eq!(&packed[40..72], &weight[32..64]);
            assert_eq!(matrix.format(), QuantizationFormat::MlxAffine4Group64);
        });
        assert!(
            repack_affine4(&device, rows, columns, &weight[..], &scales[..6], &biases).is_err()
        );
    }

    #[test]
    fn mlx_affine_accepts_signed_finite_scales_and_rejects_nonfinite_metadata() {
        let signed_scale = f16::from_f32(-0.125).to_bits().to_le_bytes();
        assert!(validate_f16_metadata("scale", &signed_scale, false).is_ok());

        let nan = f16::NAN.to_bits().to_le_bytes();
        assert!(validate_f16_metadata("scale", &nan, false).is_err());
        let infinity = f16::INFINITY.to_bits().to_le_bytes();
        assert!(validate_f16_metadata("bias", &infinity, false).is_err());
    }
}
