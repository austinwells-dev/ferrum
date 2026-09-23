//! Explicit GGML quantization formats and packed model-weight storage.
#![forbid(unsafe_code)]

use crate::{Error, MetalDevice, Result, tensor::PackedStorage};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuantizationFormat {
    Q8_0,
}
impl QuantizationFormat {
    pub fn from_ggml_type(type_id: u32) -> Result<Self> {
        match type_id {
            8 => Ok(Self::Q8_0),
            _ => Err(Error::Gguf(format!(
                "unsupported quantized weight type {type_id}"
            ))),
        }
    }
    pub const fn ggml_type(self) -> u32 {
        match self {
            Self::Q8_0 => 8,
        }
    }
    pub const fn block_elements(self) -> usize {
        match self {
            Self::Q8_0 => 32,
        }
    }
    pub const fn block_bytes(self) -> usize {
        match self {
            Self::Q8_0 => 34,
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
        QuantizationFormat::Q8_0 => "Q8_0",
    }
}
