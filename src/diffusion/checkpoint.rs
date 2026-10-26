//! Random access to the tensors of a diffusers safetensors directory: one
//! file or several shards, each self-describing, so the index file isn't read.
use crate::{
    DType, Error, MetalDevice, Result, Tensor,
    hybrid::{
        engine::page,
        weights::{Format, Matrix},
    },
    loader::gguf::GgufFile,
};
use serde::Deserialize;
use std::{cell::RefCell, collections::BTreeMap, fs::File, os::unix::fs::FileExt, path::Path};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Dtype {
    F32,
    F16,
    Bf16,
    /// A raw payload (the packed codes of MLX quantized weights).
    U32,
}

impl Dtype {
    fn parse(s: &str) -> Option<Self> {
        Some(match s {
            "F32" => Self::F32,
            "F16" => Self::F16,
            "BF16" => Self::Bf16,
            "U32" => Self::U32,
            _ => return None,
        })
    }
    pub fn bytes(self) -> usize {
        match self {
            Self::F32 | Self::U32 => 4,
            Self::F16 | Self::Bf16 => 2,
        }
    }
}

#[derive(Debug, Clone)]
pub struct Entry {
    pub dtype: Dtype,
    pub shape: Vec<usize>,
    shard: usize,
    start: u64,
    len: usize,
}

impl Entry {
    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }
}

#[derive(Deserialize)]
struct RawEntry {
    dtype: String,
    shape: Vec<usize>,
    data_offsets: [u64; 2],
}

pub struct Checkpoint {
    shards: Vec<File>,
    tensors: BTreeMap<String, Entry>,
}

impl Checkpoint {
    /// Every `*.safetensors` file in `dir`.
    pub fn open(dir: impl AsRef<Path>) -> Result<Self> {
        let dir = dir.as_ref();
        let fail = |what: String| Error::Safetensors(format!("{}: {what}", dir.display()));
        let mut paths: Vec<_> = std::fs::read_dir(dir)
            .map_err(|e| fail(e.to_string()))?
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.extension().is_some_and(|e| e == "safetensors"))
            .collect();
        paths.sort();
        if paths.is_empty() {
            return Err(fail("no .safetensors files".into()));
        }
        let mut shards = Vec::new();
        let mut tensors = BTreeMap::new();
        for path in paths {
            let file = File::open(&path).map_err(|e| fail(format!("{}: {e}", path.display())))?;
            let size = file.metadata().map_err(|e| fail(e.to_string()))?.len();
            let mut len = [0u8; 8];
            file.read_exact_at(&mut len, 0)
                .map_err(|e| fail(e.to_string()))?;
            let header_len = u64::from_le_bytes(len);
            if header_len > 100_000_000 || 8 + header_len > size {
                return Err(fail(format!("{}: bad header length", path.display())));
            }
            let mut header = vec![0u8; header_len as usize];
            file.read_exact_at(&mut header, 8)
                .map_err(|e| fail(e.to_string()))?;
            let raw: BTreeMap<String, serde_json::Value> = serde_json::from_slice(&header)
                .map_err(|e| fail(format!("{}: {e}", path.display())))?;
            let base = 8 + header_len;
            for (name, value) in raw {
                if name == "__metadata__" {
                    continue;
                }
                let e: RawEntry = serde_json::from_value(value)
                    .map_err(|e| fail(format!("tensor {name}: {e}")))?;
                let dtype = Dtype::parse(&e.dtype).ok_or_else(|| Error::Weight {
                    name: name.clone(),
                    message: format!("unsupported dtype {}", e.dtype),
                })?;
                let [from, to] = e.data_offsets;
                let entry = Entry {
                    dtype,
                    shard: shards.len(),
                    start: base + from,
                    len: (to - from) as usize,
                    shape: e.shape,
                };
                if entry.len != entry.numel() * dtype.bytes() || base + to > size {
                    return Err(Error::Weight {
                        name,
                        message: "byte range differs from shape and dtype".into(),
                    });
                }
                if tensors.insert(name.clone(), entry).is_some() {
                    return Err(Error::Weight {
                        name,
                        message: "tensor occurs in more than one shard".into(),
                    });
                }
            }
            shards.push(file);
        }
        Ok(Self { shards, tensors })
    }

    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.tensors.keys().map(String::as_str)
    }

    pub fn entry(&self, name: &str) -> Result<&Entry> {
        self.tensors.get(name).ok_or_else(|| Error::Weight {
            name: name.into(),
            message: "missing tensor".into(),
        })
    }

    /// `name`'s payload into `dst`, which must be exactly its length.
    pub fn read_into(&self, name: &str, dst: &mut [u8]) -> Result<()> {
        let e = self.entry(name)?;
        if dst.len() != e.len {
            return Err(Error::Weight {
                name: name.into(),
                message: format!("read of {} bytes into {}", e.len, dst.len()),
            });
        }
        self.shards[e.shard]
            .read_exact_at(dst, e.start)
            .map_err(|e| Error::Safetensors(format!("{name}: {e}")))
    }

    /// `name` as F32 values, whatever its stored float type.
    pub fn read_f32(&self, name: &str) -> Result<Vec<f32>> {
        let e = self.entry(name)?;
        let mut bytes = vec![0u8; e.len];
        self.read_into(name, &mut bytes)?;
        Ok(match e.dtype {
            Dtype::F32 => bytes
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                .collect(),
            Dtype::F16 => bytes
                .chunks_exact(2)
                .map(|b| half::f16::from_le_bytes([b[0], b[1]]).to_f32())
                .collect(),
            Dtype::Bf16 => bytes
                .chunks_exact(2)
                .map(|b| half::bf16::from_le_bytes([b[0], b[1]]).to_f32())
                .collect(),
            Dtype::U32 => {
                return Err(Error::Weight {
                    name: name.into(),
                    message: "packed integer tensor is not a float".into(),
                });
            }
        })
    }
}

impl Checkpoint {
    /// A `[rows, cols]` BF16 matrix on the device, kept as stored or (for
    /// `quantize` = Q8_0 or Q4_0) quantized row by row as it is read.
    pub fn matrix(
        &self,
        device: &MetalDevice,
        name: &str,
        rows: usize,
        cols: usize,
        quantize: Option<Format>,
    ) -> Result<Matrix> {
        let e = self.entry(name)?;
        if e.dtype != Dtype::Bf16 || e.shape != [rows, cols] {
            return Err(Error::Weight {
                name: name.into(),
                message: format!(
                    "expected a BF16 [{rows}, {cols}] matrix, found {:?} {:?}",
                    e.dtype, e.shape
                ),
            });
        }
        match quantize {
            None => Matrix::bf16_with(device, rows, cols, |dst| self.read_into(name, dst)),
            Some(format) => {
                let mut bytes = vec![0u8; e.len];
                self.read_into(name, &mut bytes)?;
                Matrix::from_bf16(device, rows, cols, &bytes, format)
            }
        }
    }

    /// A float vector of `len` values as F32, plus `add` to each (the
    /// zero-centred norm weights store `scale - 1`).
    pub fn vector(&self, device: &MetalDevice, name: &str, len: usize, add: f32) -> Result<Tensor> {
        let mut v = self.read_f32(name)?;
        if v.len() != len {
            return Err(Error::Weight {
                name: name.into(),
                message: format!("expected {len} values, found {}", v.len()),
            });
        }
        for x in &mut v {
            *x += add;
        }
        Tensor::from_f32(device, [len], DType::F32, &v)
    }
}

/// Where the transformer's tensors come from. Names are the diffusers ones
/// (`transformer_blocks.0.attn.to_q.weight`) whatever the file calls them.
pub trait Weights {
    /// A `[rows, cols]` matrix, quantized as it loads when the file holds BF16
    /// and `quantize` is set.
    fn matrix(
        &self,
        device: &MetalDevice,
        name: &str,
        rows: usize,
        cols: usize,
        quantize: Option<Format>,
    ) -> Result<Matrix>;

    /// A float vector as F32, plus `add` to each value.
    fn vector(&self, device: &MetalDevice, name: &str, len: usize, add: f32) -> Result<Tensor>;

    /// One block's SwiGLU input matrices, `(gate, up)`, each `[ffn, hidden]`.
    fn mlp_in(
        &self,
        device: &MetalDevice,
        layer: usize,
        ffn: usize,
        hidden: usize,
        quantize: Option<Format>,
    ) -> Result<(Matrix, Matrix)>;
}

impl Weights for Checkpoint {
    fn matrix(
        &self,
        device: &MetalDevice,
        name: &str,
        rows: usize,
        cols: usize,
        quantize: Option<Format>,
    ) -> Result<Matrix> {
        Checkpoint::matrix(self, device, name, rows, cols, quantize)
    }

    fn vector(&self, device: &MetalDevice, name: &str, len: usize, add: f32) -> Result<Tensor> {
        Checkpoint::vector(self, device, name, len, add)
    }

    fn mlp_in(
        &self,
        device: &MetalDevice,
        layer: usize,
        ffn: usize,
        hidden: usize,
        quantize: Option<Format>,
    ) -> Result<(Matrix, Matrix)> {
        let n = |s: &str| format!("transformer_blocks.{layer}.img_mlp.{s}.weight");
        Ok((
            Checkpoint::matrix(self, device, &n("gate_layer"), ffn, hidden, quantize)?,
            Checkpoint::matrix(self, device, &n("proj"), ffn, hidden, quantize)?,
        ))
    }
}

/// A quantized (or BF16/F32) transformer in a ComfyUI-style GGUF: the
/// denoiser alone, with no metadata, under `model.diffusion_model.`, and each
/// block's gate and up matrices stacked in one `gate_up` tensor. Matrices keep
/// the encoding they are stored in.
pub struct GgufWeights {
    file: RefCell<GgufFile>,
}

const GGUF_PREFIX: &str = "model.diffusion_model.";
const GGML_F32: u32 = 0;
const GGML_BF16: u32 = 30;

impl GgufWeights {
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        Ok(Self {
            file: RefCell::new(GgufFile::open_untagged(path)?),
        })
    }

    /// Device bytes `Dit::load` allocates for the weights in this file.
    pub fn estimate_bytes(&self) -> usize {
        self.file
            .borrow()
            .tensors()
            .iter()
            .map(|(name, t)| match t.dimensions.len() {
                1 => page(t.dimensions[0] * 4),
                _ if name.ends_with(".gate_up.weight") => 2 * page(t.byte_len / 2),
                _ => page(t.byte_len),
            })
            .sum()
    }

    fn read(&self, name: &str, dims: &[usize]) -> Result<(u32, Vec<u8>)> {
        let full = format!("{GGUF_PREFIX}{name}");
        let mut file = self.file.borrow_mut();
        let info = file.tensor(&full)?.clone();
        if info.dimensions != dims {
            return Err(Error::Weight {
                name: name.into(),
                message: format!("expected GGUF dims {dims:?}, found {:?}", info.dimensions),
            });
        }
        let mut bytes = vec![0u8; info.byte_len];
        file.read_tensor_into(&full, &mut bytes)?;
        Ok((info.type_id, bytes))
    }
}

impl Weights for GgufWeights {
    fn matrix(
        &self,
        device: &MetalDevice,
        name: &str,
        rows: usize,
        cols: usize,
        _quantize: Option<Format>,
    ) -> Result<Matrix> {
        let (type_id, bytes) = self.read(name, &[cols, rows])?;
        Matrix::from_ggml(device, name, rows, cols, type_id, &bytes)
    }

    fn vector(&self, device: &MetalDevice, name: &str, len: usize, add: f32) -> Result<Tensor> {
        let (type_id, bytes) = self.read(name, &[len])?;
        let mut v: Vec<f32> = match type_id {
            GGML_F32 => bytes
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                .collect(),
            GGML_BF16 => bytes
                .chunks_exact(2)
                .map(|b| half::bf16::from_le_bytes([b[0], b[1]]).to_f32())
                .collect(),
            t => {
                return Err(Error::Weight {
                    name: name.into(),
                    message: format!("GGML type {t} is not a float vector"),
                });
            }
        };
        for x in &mut v {
            *x += add;
        }
        Tensor::from_f32(device, [len], DType::F32, &v)
    }

    fn mlp_in(
        &self,
        device: &MetalDevice,
        layer: usize,
        ffn: usize,
        hidden: usize,
        _quantize: Option<Format>,
    ) -> Result<(Matrix, Matrix)> {
        let name = format!("transformer_blocks.{layer}.img_mlp.gate_up.weight");
        let (type_id, bytes) = self.read(&name, &[hidden, 2 * ffn])?;
        let (gate, up) = bytes.split_at(bytes.len() / 2);
        Ok((
            Matrix::from_ggml(device, &name, ffn, hidden, type_id, gate)?,
            Matrix::from_ggml(device, &name, ffn, hidden, type_id, up)?,
        ))
    }
}
