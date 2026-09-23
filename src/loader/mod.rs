//! Local safetensors parsing, independent of model naming.
#![forbid(unsafe_code)]
pub mod gguf;
use crate::{DType, Error, MetalDevice, Result, Tensor};
use safetensors::SafeTensors;
use std::{
    collections::{BTreeMap, BTreeSet},
    path::{Component, Path},
};

pub struct Weights {
    tensors: BTreeMap<String, Tensor>,
}
impl Weights {
    /// Load a single official safetensors file or an indexed set of shards.
    /// The index must name every tensor exactly once and match each shard.
    pub fn from_directory(device: &MetalDevice, dir: impl AsRef<Path>) -> Result<Self> {
        let dir = dir.as_ref();
        let single = dir.join("model.safetensors");
        let index_path = dir.join("model.safetensors.index.json");
        if single.is_file() && !index_path.exists() {
            return Self::from_file(device, single);
        }
        let bytes = std::fs::read(&index_path)
            .map_err(|e| Error::Safetensors(format!("{}: {e}", index_path.display())))?;
        let index: serde_json::Value = serde_json::from_slice(&bytes)
            .map_err(|e| Error::Safetensors(format!("invalid safetensors index: {e}")))?;
        let map = index["weight_map"]
            .as_object()
            .filter(|map| !map.is_empty())
            .ok_or_else(|| {
                Error::Safetensors("safetensors index needs a nonempty weight_map".into())
            })?;
        let mut files = BTreeSet::new();
        for (tensor, filename) in map {
            let filename = filename.as_str().ok_or_else(|| {
                Error::Safetensors(format!("non-string shard for tensor {tensor}"))
            })?;
            let mut parts = Path::new(filename).components();
            if !filename.ends_with(".safetensors")
                || !matches!(parts.next(), Some(Component::Normal(_)))
                || parts.next().is_some()
            {
                return Err(Error::Safetensors(format!(
                    "invalid shard filename {filename:?}"
                )));
            }
            files.insert(filename.to_owned());
        }
        let mut tensors = BTreeMap::new();
        for filename in files {
            let shard = Self::from_file(device, dir.join(&filename))?;
            for (name, tensor) in shard.tensors {
                if map.get(&name).and_then(serde_json::Value::as_str) != Some(&filename) {
                    return Err(Error::Weight {
                        name,
                        message: format!(
                            "tensor is absent from index or assigned to a different shard than {filename}"
                        ),
                    });
                }
                if tensors.insert(name.clone(), tensor).is_some() {
                    return Err(Error::Weight {
                        name,
                        message: "tensor occurs in multiple shards".into(),
                    });
                }
            }
        }
        if let Some(name) = map.keys().find(|name| !tensors.contains_key(*name)) {
            return Err(Error::Weight {
                name: name.clone(),
                message: "indexed tensor is missing from shard".into(),
            });
        }
        let result = Self { tensors };
        if let Some(expected) = index["metadata"]["total_size"].as_u64()
            && result.bytes() as u64 != expected
        {
            return Err(Error::Safetensors(format!(
                "index total_size {expected} differs from tensor bytes {}",
                result.bytes()
            )));
        }
        Ok(result)
    }

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
    pub(crate) fn insert(&mut self, name: String, tensor: Tensor) -> Result<()> {
        if self.tensors.insert(name.clone(), tensor).is_some() {
            return Err(Error::Weight {
                name,
                message: "duplicate mapped tensor".into(),
            });
        }
        Ok(())
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
    /// Check a published tied-weight duplicate without copying either tensor to host memory.
    pub(crate) fn payload_equal(&self, first: &str, second: &str) -> Result<bool> {
        let a = self.get(first)?;
        let b = self.get(second)?;
        if a.shape() != b.shape() || a.dtype() != b.dtype() {
            return Ok(false);
        }
        let (a_buffer, a_offset, a_len) = a.binding();
        let (b_buffer, b_offset, b_len) = b.binding();
        Ok(a_buffer.with_bytes(a_offset, a_len, |a_bytes| {
            b_buffer.with_bytes(b_offset, b_len, |b_bytes| a_bytes == b_bytes)
        }))
    }
}
