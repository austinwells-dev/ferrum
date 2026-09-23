//! Local safetensors parsing, independent of model naming.
#![forbid(unsafe_code)]
pub mod gguf;
use crate::{DType, Error, MetalDevice, Result, Tensor};
use safetensors::SafeTensors;
use std::{collections::BTreeMap, path::Path};

pub struct Weights {
    tensors: BTreeMap<String, Tensor>,
}
impl Weights {
    pub fn from_file(device: &MetalDevice, path: impl AsRef<Path>) -> Result<Self> {
        let bytes = std::fs::read(path.as_ref())
            .map_err(|e| Error::Safetensors(format!("{}: {e}", path.as_ref().display())))?;
        Self::from_bytes(device, &bytes)
    }
    pub fn from_bytes(device: &MetalDevice, bytes: &[u8]) -> Result<Self> {
        let file =
            SafeTensors::deserialize(bytes).map_err(|e| Error::Safetensors(e.to_string()))?;
        let mut tensors = BTreeMap::new();
        for (name, view) in file.tensors() {
            let result = (|| {
                let dtype = match view.dtype() {
                    safetensors::Dtype::F32 => DType::F32,
                    safetensors::Dtype::F16 => DType::F16,
                    safetensors::Dtype::BF16 => DType::BF16,
                    other => {
                        return Err(Error::Safetensors(format!("unsupported dtype {other:?}")));
                    }
                };
                Tensor::from_le_bytes(device, view.shape(), dtype, view.data())
            })()
            .map_err(|e| Error::Weight {
                name: name.clone(),
                message: e.to_string(),
            })?;
            tensors.insert(name, result);
        }
        Ok(Self { tensors })
    }
    /// Rename validated entries by sharing immutable storage; no payload copy.
    pub(crate) fn remap<'a>(
        &self,
        names: impl IntoIterator<Item = (&'a str, &'a str)>,
    ) -> Result<Self> {
        let mut tensors = BTreeMap::new();
        for (source, target) in names {
            if tensors
                .insert(target.to_owned(), self.get(source)?.clone())
                .is_some()
            {
                return Err(Error::Weight {
                    name: target.into(),
                    message: "duplicate mapping".into(),
                });
            }
        }
        Ok(Self { tensors })
    }
    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.tensors.keys().map(String::as_str)
    }
    pub fn get(&self, name: &str) -> Result<&Tensor> {
        self.tensors.get(name).ok_or_else(|| Error::Weight {
            name: name.into(),
            message: "missing tensor".into(),
        })
    }
    pub fn optional(&self, name: &str) -> Option<&Tensor> {
        self.tensors.get(name)
    }
    pub fn bytes(&self) -> usize {
        self.tensors.values().map(Tensor::byte_size).sum()
    }
}
