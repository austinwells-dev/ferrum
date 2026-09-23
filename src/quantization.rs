//! Explicit packed quantization formats and model-weight storage.
#![forbid(unsafe_code)]

use crate::{Error, MetalDevice, Result, tensor::PackedStorage};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuantizationFormat {
    Q4_0,
    #[allow(non_camel_case_types)]
    Q5_0,
    #[allow(non_camel_case_types)]
    Q5_1,
    #[allow(non_camel_case_types)]
    Q4_K,
    #[allow(non_camel_case_types)]
    Q5_K,
    Q8_0,
    #[allow(non_camel_case_types)]
    Q6_K,
    /// MLX affine 4-bit weights, repacked as 64-value groups with f16 scale/bias.
    MlxAffine4Group64,
}
impl QuantizationFormat {
    pub fn from_ggml_type(type_id: u32) -> Result<Self> {
        match type_id {
            2 => Ok(Self::Q4_0),
            6 => Ok(Self::Q5_0),
            7 => Ok(Self::Q5_1),
            8 => Ok(Self::Q8_0),
            12 => Ok(Self::Q4_K),
            13 => Ok(Self::Q5_K),
            14 => Ok(Self::Q6_K),
            _ => Err(Error::Gguf(format!(
                "unsupported quantized weight type {type_id}"
            ))),
        }
    }
    pub const fn ggml_type(self) -> Option<u32> {
        match self {
            Self::Q4_0 => Some(2),
            Self::Q5_0 => Some(6),
            Self::Q5_1 => Some(7),
            Self::Q4_K => Some(12),
            Self::Q5_K => Some(13),
            Self::Q8_0 => Some(8),
            Self::Q6_K => Some(14),
            Self::MlxAffine4Group64 => None,
        }
    }
    pub const fn block_elements(self) -> usize {
        match self {
            Self::Q4_0 => 32,
            Self::Q5_0 => 32,
            Self::Q5_1 => 32,
            Self::Q4_K => 256,
            Self::Q5_K => 256,
            Self::Q8_0 => 32,
            Self::Q6_K => 256,
            Self::MlxAffine4Group64 => 64,
        }
    }
    pub const fn block_bytes(self) -> usize {
        match self {
            Self::Q4_0 => 18,
            Self::Q5_0 => 22,
            Self::Q5_1 => 24,
            Self::Q4_K => 144,
            Self::Q5_K => 176,
            Self::Q8_0 => 34,
            Self::Q6_K => 210,
            Self::MlxAffine4Group64 => 36,
        }
    }
}

/// A row-major logical `[rows, columns]` matrix whose source blocks remain
/// packed for the full model lifetime.
#[derive(Clone)]
pub struct QuantizedMatrix {
    rows: usize,
    columns: usize,
    format: QuantizationFormat,
    storage: PackedStorage,
}
impl QuantizedMatrix {
    pub(crate) fn from_reader(
        device: &MetalDevice,
        rows: usize,
        columns: usize,
        format: QuantizationFormat,
        fill: impl FnOnce(&mut [u8]) -> Result<()>,
    ) -> Result<Self> {
        if rows == 0 || columns == 0 || !columns.is_multiple_of(format.block_elements()) {
            return Err(Error::Shape(format!(
                "{} matrix requires nonempty rows and columns divisible by block size {}",
                format_name(format),
                format.block_elements()
            )));
        }
        let byte_len = rows
            .checked_mul(columns / format.block_elements())
            .and_then(|blocks| blocks.checked_mul(format.block_bytes()))
            .ok_or_else(|| Error::Shape("packed matrix byte size overflow".into()))?;
        let storage = PackedStorage::from_reader(device, byte_len, fill)?;
        Ok(Self {
            rows,
            columns,
            format,
            storage,
        })
    }
    pub fn rows(&self) -> usize {
        self.rows
    }
    pub fn columns(&self) -> usize {
        self.columns
    }
    pub fn format(&self) -> QuantizationFormat {
        self.format
    }
    pub fn byte_size(&self) -> usize {
        self.storage.byte_len()
    }
    pub(crate) fn binding(&self) -> (&crate::metal::MetalBuffer, usize, usize) {
        self.storage.binding()
    }
    pub(crate) fn buffer(&self) -> &crate::metal::MetalBuffer {
        self.storage.buffer()
    }
    #[cfg(test)]
    pub(crate) fn with_bytes<T>(&self, f: impl FnOnce(&[u8]) -> T) -> T {
        self.storage.with_bytes(f)
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

#[cfg(test)]
mod tests {
    use super::QuantizationFormat;

    #[test]
    fn q5_formats_match_ggml_type_ids_and_block_geometry() {
        assert_eq!(
            QuantizationFormat::from_ggml_type(6).unwrap(),
            QuantizationFormat::Q5_0
        );
        assert_eq!(
            QuantizationFormat::from_ggml_type(7).unwrap(),
            QuantizationFormat::Q5_1
        );
        for (format, bytes) in [
            (QuantizationFormat::Q5_0, 22),
            (QuantizationFormat::Q5_1, 24),
        ] {
            assert_eq!(
                format.ggml_type().unwrap(),
                if format == QuantizationFormat::Q5_0 {
                    6
                } else {
                    7
                }
            );
            assert_eq!(format.block_elements(), 32);
            assert_eq!(format.block_bytes(), bytes);
        }
        assert!(QuantizationFormat::from_ggml_type(5).is_err());
    }

    #[test]
    fn k_block_formats_match_ggml_type_ids_and_block_geometry() {
        for (type_id, format, bytes) in [
            (12, QuantizationFormat::Q4_K, 144),
            (13, QuantizationFormat::Q5_K, 176),
        ] {
            assert_eq!(QuantizationFormat::from_ggml_type(type_id).unwrap(), format);
            assert_eq!(format.ggml_type(), Some(type_id));
            assert_eq!(format.block_elements(), 256);
            assert_eq!(format.block_bytes(), bytes);
        }
    }

    #[test]
    fn mlx_affine_runtime_layout_is_explicitly_not_a_ggml_type() {
        let format = QuantizationFormat::MlxAffine4Group64;
        assert_eq!(format.ggml_type(), None);
        assert_eq!(format.block_elements(), 64);
        assert_eq!(format.block_bytes(), 36);
    }
}
