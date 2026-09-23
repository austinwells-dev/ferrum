//! Bounds-checked GGUF v2/v3 index and streaming tensor reads.
//!
//! The parser knows the physical block sizes of common GGML types so it can
//! validate every tensor range. Decoding support is a separate runtime concern:
//! callers must explicitly match a tensor's type before loading it.
#![forbid(unsafe_code)]

use crate::{Error, Result};
use std::{
    collections::BTreeMap,
    fs::File,
    io::{Read, Seek, SeekFrom},
    path::Path,
};

const MAGIC: &[u8; 4] = b"GGUF";
const MAX_TENSORS: u64 = 1_000_000;
const MAX_METADATA: u64 = 1_000_000;
const MAX_ARRAY_ITEMS: u64 = 4_000_000;
const MAX_STRING_BYTES: u64 = 16 * 1024 * 1024;
const MAX_TENSOR_NAME_BYTES: u64 = 64;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum MetadataType {
    Uint8 = 0,
    Int8 = 1,
    Uint16 = 2,
    Int16 = 3,
    Uint32 = 4,
    Int32 = 5,
    Float32 = 6,
    Bool = 7,
    String = 8,
    Array = 9,
    Uint64 = 10,
    Int64 = 11,
    Float64 = 12,
}
impl TryFrom<u32> for MetadataType {
    type Error = Error;
    fn try_from(value: u32) -> Result<Self> {
        match value {
            0 => Ok(Self::Uint8),
            1 => Ok(Self::Int8),
            2 => Ok(Self::Uint16),
            3 => Ok(Self::Int16),
            4 => Ok(Self::Uint32),
            5 => Ok(Self::Int32),
            6 => Ok(Self::Float32),
            7 => Ok(Self::Bool),
            8 => Ok(Self::String),
            9 => Ok(Self::Array),
            10 => Ok(Self::Uint64),
            11 => Ok(Self::Int64),
            12 => Ok(Self::Float64),
            _ => Err(Error::Gguf(format!("unsupported metadata type {value}"))),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum MetadataValue {
    Uint8(u8),
    Int8(i8),
    Uint16(u16),
    Int16(i16),
    Uint32(u32),
    Int32(i32),
    Float32(f32),
    Bool(bool),
    String(String),
    Array {
        element_type: MetadataType,
        values: Vec<MetadataValue>,
    },
    Uint64(u64),
    Int64(i64),
    Float64(f64),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BlockLayout {
    pub elements: usize,
    pub bytes: usize,
}

/// Physical GGML block layout for the commonly used standard tensor types.
/// Unknown/removed encodings are rejected before tensor offsets are trusted.
pub fn block_layout(type_id: u32) -> Result<BlockLayout> {
    let (elements, bytes) = match type_id {
        0 => (1, 4),   // F32
        1 => (1, 2),   // F16
        2 => (32, 18), // Q4_0
        3 => (32, 20), // Q4_1
        6 => (32, 22), // Q5_0
        7 => (32, 24), // Q5_1
        8 => (32, 34), // Q8_0
        9 => (32, 36), // Q8_1
        10 => (256, 84),
        11 => (256, 110),
        12 => (256, 144),
        13 => (256, 176),
        14 => (256, 210),
        15 => (256, 292),
        30 => (1, 2), // BF16
        _ => {
            return Err(Error::Gguf(format!(
                "unsupported GGML tensor type {type_id}"
            )));
        }
    };
    Ok(BlockLayout { elements, bytes })
}

#[derive(Debug, Clone)]
pub struct TensorInfo {
    pub name: String,
    /// GGUF order: dimension zero is the contiguous/fastest dimension.
    pub dimensions: Vec<usize>,
    pub type_id: u32,
    /// Relative to the aligned beginning of the tensor-data section.
    pub offset: u64,
    pub byte_len: usize,
}
impl TensorInfo {
    pub fn logical_matrix_shape(&self) -> Result<[usize; 2]> {
        if self.dimensions.len() != 2 {
            return Err(Error::Gguf(format!(
                "tensor {} is rank {}, expected a matrix",
                self.name,
                self.dimensions.len()
            )));
        }
        Ok([self.dimensions[1], self.dimensions[0]])
    }
}

/// A seekable GGUF reader. Tensor payloads are read on demand, so opening a
/// multi-gigabyte model does not create a second full-file CPU copy.
#[derive(Debug)]
pub struct GgufReader<R> {
    reader: R,
    file_len: u64,
    tensor_data_offset: u64,
    version: u32,
    alignment: u32,
    metadata: BTreeMap<String, MetadataValue>,
    tensors: BTreeMap<String, TensorInfo>,
}
pub type GgufFile = GgufReader<File>;

impl GgufFile {
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let file = File::open(path).map_err(|e| Error::Gguf(format!("{}: {e}", path.display())))?;
        Self::from_reader(file)
    }
}

impl<R: Read + Seek> GgufReader<R> {
    pub fn from_reader(mut reader: R) -> Result<Self> {
        let file_len = reader
            .seek(SeekFrom::End(0))
            .map_err(|e| io_error("seek to end", e))?;
        reader
            .seek(SeekFrom::Start(0))
            .map_err(|e| io_error("seek to start", e))?;

        let magic = read_array::<4, _>(&mut reader, "magic")?;
        if &magic != MAGIC {
            return Err(Error::Gguf("invalid magic; expected GGUF".into()));
        }
        let version = read_u32(&mut reader, "version")?;
        if !matches!(version, 2 | 3) {
            return Err(Error::Gguf(format!("unsupported version {version}")));
        }
        let tensor_count = read_u64(&mut reader, "tensor count")?;
        let metadata_count = read_u64(&mut reader, "metadata count")?;
        if tensor_count > MAX_TENSORS || metadata_count > MAX_METADATA {
            return Err(Error::Gguf(format!(
                "unreasonable descriptor counts: {tensor_count} tensors, {metadata_count} metadata entries"
            )));
        }

        let mut metadata = BTreeMap::new();
        for _ in 0..metadata_count {
            let key = read_string(&mut reader, file_len, "metadata key")?;
            let type_id = read_u32(&mut reader, "metadata value type")?;
            let value_type = MetadataType::try_from(type_id)?;
            let value = read_metadata_value(&mut reader, value_type, file_len)?;
            if metadata.insert(key.clone(), value).is_some() {
                return Err(Error::Gguf(format!("duplicate metadata key {key}")));
            }
        }

        let alignment = match metadata.get("general.alignment") {
            None => 32,
            Some(MetadataValue::Uint32(value)) => *value,
            Some(_) => {
                return Err(Error::Gguf(
                    "general.alignment must have uint32 type".into(),
                ));
            }
        };
        if alignment < 8 || !alignment.is_multiple_of(8) {
            return Err(Error::Gguf(format!(
                "general.alignment must be a positive multiple of 8, got {alignment}"
            )));
        }
        if !matches!(
            metadata.get("general.architecture"),
            Some(MetadataValue::String(_))
        ) {
            return Err(Error::Gguf(
                "missing or invalid general.architecture metadata".into(),
            ));
        }

        let mut tensors = BTreeMap::new();
        let mut ranges = Vec::new();
        let mut contains_quantized = false;
        for _ in 0..tensor_count {
            let name = read_string(&mut reader, file_len, "tensor name")?;
            if name.len() as u64 > MAX_TENSOR_NAME_BYTES {
                return Err(Error::Gguf(format!(
                    "tensor name exceeds {MAX_TENSOR_NAME_BYTES} bytes"
                )));
            }
            let dimensions_count = read_u32(&mut reader, "tensor rank")?;
            if !(1..=4).contains(&dimensions_count) {
                return Err(Error::Gguf(format!(
                    "tensor {name} has unsupported rank {dimensions_count}"
                )));
            }
            let mut dimensions = Vec::with_capacity(dimensions_count as usize);
            for _ in 0..dimensions_count {
                let dim = usize::try_from(read_u64(&mut reader, "tensor dimension")?)
                    .map_err(|_| Error::Gguf(format!("tensor {name} dimension exceeds usize")))?;
                if dim == 0 {
                    return Err(Error::Gguf(format!("tensor {name} has an empty dimension")));
                }
                dimensions.push(dim);
            }
            let type_id = read_u32(&mut reader, "tensor type")?;
            let offset = read_u64(&mut reader, "tensor data offset")?;
            let layout = block_layout(type_id).map_err(|e| match e {
                Error::Gguf(message) => Error::Gguf(format!("tensor {name}: {message}")),
                other => other,
            })?;
            let elements = dimensions
                .iter()
                .try_fold(1usize, |n, &d| n.checked_mul(d))
                .ok_or_else(|| Error::Gguf(format!("tensor {name} element count overflow")))?;
            if !dimensions[0].is_multiple_of(layout.elements) {
                return Err(Error::Gguf(format!(
                    "tensor {name} fastest dimension {} is not divisible by block size {} for type {type_id}",
                    dimensions[0], layout.elements
                )));
            }
            let byte_len = elements
                .checked_div(layout.elements)
                .and_then(|blocks| blocks.checked_mul(layout.bytes))
                .ok_or_else(|| Error::Gguf(format!("tensor {name} byte size overflow")))?;
            if !offset.is_multiple_of(u64::from(alignment)) {
                return Err(Error::Gguf(format!(
                    "tensor {name} offset {offset} is not aligned to {alignment}"
                )));
            }
            contains_quantized |= !matches!(type_id, 0 | 1 | 30);
            let info = TensorInfo {
                name: name.clone(),
                dimensions,
                type_id,
                offset,
                byte_len,
            };
            if tensors.insert(name.clone(), info).is_some() {
                return Err(Error::Gguf(format!("duplicate tensor name {name}")));
            }
        }

        let descriptors_end = reader
            .stream_position()
            .map_err(|e| io_error("read tensor descriptors", e))?;
        let tensor_data_offset = align_up(descriptors_end, u64::from(alignment))?;
        // Recalculate ranges relative to the final data section (which starts
        // after all descriptors), then reject overlap even if entries were
        // intentionally listed out of file order.
        ranges.clear();
        for tensor in tensors.values() {
            let start = tensor_data_offset
                .checked_add(tensor.offset)
                .ok_or_else(|| Error::Gguf(format!("tensor {} offset overflow", tensor.name)))?;
            let end = start
                .checked_add(tensor.byte_len as u64)
                .ok_or_else(|| Error::Gguf(format!("tensor {} end overflow", tensor.name)))?;
            if end > file_len {
                return Err(Error::Gguf(format!(
                    "tensor {} range {start}..{end} exceeds file length {file_len}",
                    tensor.name
                )));
            }
            ranges.push((start, end, tensor.name.clone()));
        }
        ranges.sort_unstable_by_key(|range| range.0);
        for adjacent in ranges.windows(2) {
            if adjacent[0].1 > adjacent[1].0 {
                return Err(Error::Gguf(format!(
                    "tensor ranges overlap: {} and {}",
                    adjacent[0].2, adjacent[1].2
                )));
            }
        }
        if contains_quantized
            && !matches!(
                metadata.get("general.quantization_version"),
                Some(MetadataValue::Uint32(_))
            )
        {
            return Err(Error::Gguf(
                "quantized tensor file is missing uint32 general.quantization_version".into(),
            ));
        }

        Ok(Self {
            reader,
            file_len,
            tensor_data_offset,
            version,
            alignment,
            metadata,
            tensors,
        })
    }

    pub fn version(&self) -> u32 {
        self.version
    }
    pub fn alignment(&self) -> u32 {
        self.alignment
    }
    pub fn file_len(&self) -> u64 {
        self.file_len
    }
    pub fn metadata(&self) -> &BTreeMap<String, MetadataValue> {
        &self.metadata
    }
    pub fn metadata_value(&self, key: &str) -> Option<&MetadataValue> {
        self.metadata.get(key)
    }
    pub fn tensors(&self) -> &BTreeMap<String, TensorInfo> {
        &self.tensors
    }
    pub fn tensor(&self, name: &str) -> Result<&TensorInfo> {
        self.tensors
            .get(name)
            .ok_or_else(|| Error::Gguf(format!("missing tensor {name}")))
    }

    /// Compare same-shape, same-format tensor payloads exactly using bounded
    /// scratch space. This lets a model adapter verify a serialized alias
    /// (such as tied input/output embeddings) without buffering either tensor.
    pub fn tensors_equal_payload(&mut self, left: &str, right: &str) -> Result<bool> {
        let left = self.tensor(left)?.clone();
        let right = self.tensor(right)?.clone();
        if left.dimensions != right.dimensions
            || left.type_id != right.type_id
            || left.byte_len != right.byte_len
        {
            return Ok(false);
        }
        let mut left_buffer = vec![0; 256 * 1024];
        let mut right_buffer = vec![0; 256 * 1024];
        let mut offset = 0usize;
        while offset < left.byte_len {
            let length = (left.byte_len - offset).min(left_buffer.len());
            self.read_tensor_range(&left, offset, &mut left_buffer[..length])?;
            self.read_tensor_range(&right, offset, &mut right_buffer[..length])?;
            if left_buffer[..length] != right_buffer[..length] {
                return Ok(false);
            }
            offset += length;
        }
        Ok(true)
    }

    fn read_tensor_range(
        &mut self,
        tensor: &TensorInfo,
        relative_offset: usize,
        destination: &mut [u8],
    ) -> Result<()> {
        let range_end = relative_offset
            .checked_add(destination.len())
            .ok_or_else(|| Error::Gguf(format!("tensor {} range overflow", tensor.name)))?;
        if range_end > tensor.byte_len {
            return Err(Error::Gguf(format!(
                "tensor {} range exceeds payload",
                tensor.name
            )));
        }
        let position = self
            .tensor_data_offset
            .checked_add(tensor.offset)
            .and_then(|base| base.checked_add(relative_offset as u64))
            .ok_or_else(|| Error::Gguf(format!("tensor {} offset overflow", tensor.name)))?;
        self.reader
            .seek(SeekFrom::Start(position))
            .map_err(|e| io_error(&format!("seek to tensor {}", tensor.name), e))?;
        self.reader
            .read_exact(destination)
            .map_err(|e| io_error(&format!("read tensor {}", tensor.name), e))
    }

    /// Fill a caller-owned destination directly from the tensor's file range.
    /// This supports streaming into Metal shared storage without a per-tensor
    /// staging `Vec`.
    pub fn read_tensor_into(&mut self, name: &str, destination: &mut [u8]) -> Result<()> {
        let tensor = self.tensor(name)?;
        if destination.len() != tensor.byte_len {
            return Err(Error::Gguf(format!(
                "tensor {name} has {} bytes, destination has {}",
                tensor.byte_len,
                destination.len()
            )));
        }
        let position = self
            .tensor_data_offset
            .checked_add(tensor.offset)
            .ok_or_else(|| Error::Gguf(format!("tensor {name} offset overflow")))?;
        self.reader
            .seek(SeekFrom::Start(position))
            .map_err(|e| io_error(&format!("seek to tensor {name}"), e))?;
        self.reader
            .read_exact(destination)
            .map_err(|e| io_error(&format!("read tensor {name}"), e))
    }

    /// Stream a tensor through a bounded scratch buffer. The callback may
    /// transform values directly into a Metal allocation without retaining a
    /// full CPU copy of the tensor.
    pub fn stream_tensor(
        &mut self,
        name: &str,
        chunk_bytes: usize,
        mut consume: impl FnMut(&[u8]) -> Result<()>,
    ) -> Result<()> {
        if chunk_bytes == 0 {
            return Err(Error::Gguf(
                "tensor stream chunk size must be positive".into(),
            ));
        }
        let tensor = self.tensor(name)?;
        let byte_len = tensor.byte_len;
        let position = self
            .tensor_data_offset
            .checked_add(tensor.offset)
            .ok_or_else(|| Error::Gguf(format!("tensor {name} offset overflow")))?;
        self.reader
            .seek(SeekFrom::Start(position))
            .map_err(|e| io_error(&format!("seek to tensor {name}"), e))?;
        let mut scratch = vec![0; chunk_bytes.min(byte_len)];
        let mut remaining = byte_len;
        while remaining > 0 {
            let length = remaining.min(scratch.len());
            self.reader
                .read_exact(&mut scratch[..length])
                .map_err(|e| io_error(&format!("read tensor {name}"), e))?;
            consume(&scratch[..length])?;
            remaining -= length;
        }
        Ok(())
    }
}

fn read_metadata_value<R: Read>(
    reader: &mut R,
    ty: MetadataType,
    file_len: u64,
) -> Result<MetadataValue> {
    let value = match ty {
        MetadataType::Uint8 => MetadataValue::Uint8(read_u8(reader, "uint8 metadata")?),
        MetadataType::Int8 => MetadataValue::Int8(read_i8(reader, "int8 metadata")?),
        MetadataType::Uint16 => MetadataValue::Uint16(read_u16(reader, "uint16 metadata")?),
        MetadataType::Int16 => MetadataValue::Int16(read_i16(reader, "int16 metadata")?),
        MetadataType::Uint32 => MetadataValue::Uint32(read_u32(reader, "uint32 metadata")?),
        MetadataType::Int32 => MetadataValue::Int32(read_i32(reader, "int32 metadata")?),
        MetadataType::Float32 => MetadataValue::Float32(read_f32(reader, "float32 metadata")?),
        MetadataType::Bool => {
            let value = read_i8(reader, "bool metadata")?;
            match value {
                0 => MetadataValue::Bool(false),
                1 => MetadataValue::Bool(true),
                _ => return Err(Error::Gguf(format!("invalid bool value {value}"))),
            }
        }
        MetadataType::String => {
            MetadataValue::String(read_string(reader, file_len, "string metadata")?)
        }
        MetadataType::Array => {
            let element_type_id = read_u32(reader, "array element type")?;
            let element_type = MetadataType::try_from(element_type_id)?;
            if element_type == MetadataType::Array {
                return Err(Error::Gguf("nested metadata arrays are unsupported".into()));
            }
            let count = read_u64(reader, "array element count")?;
            if count > MAX_ARRAY_ITEMS || count > file_len {
                return Err(Error::Gguf(format!(
                    "unreasonable metadata array length {count}"
                )));
            }
            let mut values = Vec::with_capacity(count as usize);
            for _ in 0..count {
                values.push(read_metadata_value(reader, element_type, file_len)?);
            }
            MetadataValue::Array {
                element_type,
                values,
            }
        }
        MetadataType::Uint64 => MetadataValue::Uint64(read_u64(reader, "uint64 metadata")?),
        MetadataType::Int64 => MetadataValue::Int64(read_i64(reader, "int64 metadata")?),
        MetadataType::Float64 => MetadataValue::Float64(read_f64(reader, "float64 metadata")?),
    };
    Ok(value)
}

fn read_string<R: Read>(reader: &mut R, file_len: u64, what: &str) -> Result<String> {
    let length = read_u64(reader, "string length")?;
    if length > MAX_STRING_BYTES || length > file_len {
        return Err(Error::Gguf(format!("unreasonable {what} length {length}")));
    }
    let length = usize::try_from(length)
        .map_err(|_| Error::Gguf(format!("{what} length exceeds address space")))?;
    let mut bytes = vec![0; length];
    reader
        .read_exact(&mut bytes)
        .map_err(|e| io_error(&format!("read {what}"), e))?;
    String::from_utf8(bytes).map_err(|e| Error::Gguf(format!("invalid UTF-8 in {what}: {e}")))
}

fn align_up(value: u64, alignment: u64) -> Result<u64> {
    let remainder = value % alignment;
    if remainder == 0 {
        Ok(value)
    } else {
        value
            .checked_add(alignment - remainder)
            .ok_or_else(|| Error::Gguf("alignment overflow".into()))
    }
}

fn io_error(context: &str, error: std::io::Error) -> Error {
    Error::Gguf(format!("{context}: {error}"))
}

fn read_array<const N: usize, R: Read>(reader: &mut R, what: &str) -> Result<[u8; N]> {
    let mut bytes = [0; N];
    reader
        .read_exact(&mut bytes)
        .map_err(|e| io_error(&format!("read {what}"), e))?;
    Ok(bytes)
}
fn read_u8<R: Read>(r: &mut R, what: &str) -> Result<u8> {
    Ok(read_array::<1, _>(r, what)?[0])
}
fn read_i8<R: Read>(r: &mut R, what: &str) -> Result<i8> {
    Ok(read_u8(r, what)? as i8)
}
fn read_u16<R: Read>(r: &mut R, what: &str) -> Result<u16> {
    Ok(u16::from_le_bytes(read_array(r, what)?))
}
fn read_i16<R: Read>(r: &mut R, what: &str) -> Result<i16> {
    Ok(i16::from_le_bytes(read_array(r, what)?))
}
fn read_u32<R: Read>(r: &mut R, what: &str) -> Result<u32> {
    Ok(u32::from_le_bytes(read_array(r, what)?))
}
fn read_i32<R: Read>(r: &mut R, what: &str) -> Result<i32> {
    Ok(i32::from_le_bytes(read_array(r, what)?))
}
fn read_u64<R: Read>(r: &mut R, what: &str) -> Result<u64> {
    Ok(u64::from_le_bytes(read_array(r, what)?))
}
fn read_i64<R: Read>(r: &mut R, what: &str) -> Result<i64> {
    Ok(i64::from_le_bytes(read_array(r, what)?))
}
fn read_f32<R: Read>(r: &mut R, what: &str) -> Result<f32> {
    Ok(f32::from_bits(read_u32(r, what)?))
}
fn read_f64<R: Read>(r: &mut R, what: &str) -> Result<f64> {
    Ok(f64::from_bits(read_u64(r, what)?))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    fn push_u32(bytes: &mut Vec<u8>, value: u32) {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    fn push_u64(bytes: &mut Vec<u8>, value: u64) {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    fn push_string(bytes: &mut Vec<u8>, value: &str) {
        push_u64(bytes, value.len() as u64);
        bytes.extend_from_slice(value.as_bytes());
    }
    fn fixture(type_id: u32, dims: &[u64], offsets: &[u64], names: &[&str]) -> Vec<u8> {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(MAGIC);
        push_u32(&mut bytes, 3);
        push_u64(&mut bytes, names.len() as u64);
        push_u64(&mut bytes, 2);
        push_string(&mut bytes, "general.architecture");
        push_u32(&mut bytes, MetadataType::String as u32);
        push_string(&mut bytes, "qwen2");
        push_string(&mut bytes, "general.quantization_version");
        push_u32(&mut bytes, MetadataType::Uint32 as u32);
        push_u32(&mut bytes, 2);
        for (index, name) in names.iter().enumerate() {
            push_string(&mut bytes, name);
            push_u32(&mut bytes, dims.len() as u32);
            for &dim in dims {
                push_u64(&mut bytes, dim);
            }
            push_u32(&mut bytes, type_id);
            push_u64(&mut bytes, offsets[index]);
        }
        while !bytes.len().is_multiple_of(32) {
            bytes.push(0);
        }
        let layout = block_layout(type_id).unwrap_or(BlockLayout {
            elements: 32,
            bytes: 34,
        });
        let payload_bytes = layout.bytes * dims.iter().product::<u64>() as usize / layout.elements;
        let max_end = offsets.iter().copied().max().unwrap_or(0) as usize + payload_bytes;
        bytes.resize(bytes.len() + max_end, 0);
        for (index, name) in names.iter().enumerate() {
            let fill = u8::try_from(index + 1).unwrap();
            let start = bytes.len() - max_end + offsets[index] as usize;
            let end = start + payload_bytes;
            bytes[start..end].fill(fill);
            assert!(!name.is_empty());
        }
        bytes
    }

    #[test]
    fn reads_aligned_q8_0_tensor_without_reordering_payload() {
        let bytes = fixture(8, &[32, 2], &[0], &["blk.0.attn_q.weight"]);
        let mut file = GgufReader::from_reader(Cursor::new(bytes)).unwrap();
        let tensor = file.tensor("blk.0.attn_q.weight").unwrap();
        assert_eq!(tensor.type_id, 8);
        assert_eq!(tensor.byte_len, 68);
        assert_eq!(tensor.logical_matrix_shape().unwrap(), [2, 32]);
        let mut destination = vec![0; 68];
        file.read_tensor_into("blk.0.attn_q.weight", &mut destination)
            .unwrap();
        assert_eq!(destination, vec![1; 68]);
    }

    #[test]
    fn compares_tensor_aliases_with_bounded_reads() {
        let bytes = fixture(8, &[32, 2], &[0, 96], &["input.weight", "output.weight"]);
        let mut file = GgufReader::from_reader(Cursor::new(bytes)).unwrap();
        let data_start = file.tensor_data_offset as usize;
        let source = file.reader.get_ref()[data_start..data_start + 68].to_vec();
        file.reader.get_mut()[data_start + 96..data_start + 164].copy_from_slice(&source);
        assert!(
            file.tensors_equal_payload("input.weight", "output.weight")
                .unwrap()
        );
        file.reader.get_mut()[data_start + 120] ^= 1;
        assert!(
            !file
                .tensors_equal_payload("input.weight", "output.weight")
                .unwrap()
        );
    }

    #[test]
    fn rejects_bad_magic_and_truncated_header() {
        let mut bad = fixture(8, &[32, 2], &[0], &["weight"]);
        bad[0] = b'X';
        assert!(GgufReader::from_reader(Cursor::new(bad)).is_err());
        assert!(GgufReader::from_reader(Cursor::new(b"GGUF\x03\0".to_vec())).is_err());
    }

    #[test]
    fn rejects_unaligned_offsets_and_non_block_dimensions() {
        let unaligned = fixture(8, &[32, 2], &[1], &["weight"]);
        assert!(
            GgufReader::from_reader(Cursor::new(unaligned))
                .unwrap_err()
                .to_string()
                .contains("not aligned")
        );
        let tail = fixture(8, &[31, 2], &[0], &["weight"]);
        assert!(
            GgufReader::from_reader(Cursor::new(tail))
                .unwrap_err()
                .to_string()
                .contains("block size")
        );
    }

    #[test]
    fn rejects_unknown_quantization_and_overlapping_ranges() {
        let unsupported = fixture(99, &[32, 2], &[0], &["weight"]);
        assert!(
            GgufReader::from_reader(Cursor::new(unsupported))
                .unwrap_err()
                .to_string()
                .contains("unsupported GGML tensor type 99")
        );
        let overlap = fixture(8, &[32, 2], &[0, 0], &["weight.a", "weight.b"]);
        assert!(
            GgufReader::from_reader(Cursor::new(overlap))
                .unwrap_err()
                .to_string()
                .contains("ranges overlap")
        );
    }

    #[test]
    fn standard_block_sizes_match_the_ggml_layouts() {
        assert_eq!(
            block_layout(8).unwrap(),
            BlockLayout {
                elements: 32,
                bytes: 34
            }
        );
        assert_eq!(
            block_layout(14).unwrap(),
            BlockLayout {
                elements: 256,
                bytes: 210
            }
        );
        assert_eq!(
            block_layout(13).unwrap(),
            BlockLayout {
                elements: 256,
                bytes: 176
            }
        );
        assert_eq!(
            block_layout(12).unwrap(),
            BlockLayout {
                elements: 256,
                bytes: 144
            }
        );
    }
}
