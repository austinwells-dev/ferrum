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
const GGML_F16: u32 = 1;
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
            GGML_F16 => bytes
                .chunks_exact(2)
                .map(|b| half::f16::from_le_bytes([b[0], b[1]]).to_f32())
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

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::diffusion::ops::check_projectable;

    pub(crate) enum Meta {
        U32(u32),
        Str(&'static str),
    }

    /// A GGUF of `(name, ggml dims, type, bytes)` tensors, each name taken
    /// as is. ComfyUI's transformers carry no metadata.
    pub(crate) fn write(
        path: &Path,
        meta: &[(&str, Meta)],
        tensors: &[(String, Vec<usize>, u32, Vec<u8>)],
    ) {
        let string = |b: &mut Vec<u8>, v: &str| {
            b.extend_from_slice(&(v.len() as u64).to_le_bytes());
            b.extend_from_slice(v.as_bytes());
        };
        let mut b = b"GGUF".to_vec();
        b.extend_from_slice(&3u32.to_le_bytes());
        b.extend_from_slice(&(tensors.len() as u64).to_le_bytes());
        b.extend_from_slice(&(meta.len() as u64).to_le_bytes());
        for (key, value) in meta {
            string(&mut b, key);
            match value {
                Meta::U32(v) => {
                    b.extend_from_slice(&4u32.to_le_bytes());
                    b.extend_from_slice(&v.to_le_bytes());
                }
                Meta::Str(v) => {
                    b.extend_from_slice(&8u32.to_le_bytes());
                    string(&mut b, v);
                }
            }
        }
        let mut offset = 0u64;
        for (name, dims, type_id, bytes) in tensors {
            string(&mut b, name);
            b.extend_from_slice(&(dims.len() as u32).to_le_bytes());
            for &d in dims {
                b.extend_from_slice(&(d as u64).to_le_bytes());
            }
            b.extend_from_slice(&type_id.to_le_bytes());
            b.extend_from_slice(&offset.to_le_bytes());
            offset += (bytes.len() as u64).div_ceil(32) * 32;
        }
        b.resize(b.len().div_ceil(32) * 32, 0);
        for (_, _, _, bytes) in tensors {
            b.extend_from_slice(bytes);
            b.resize(b.len().div_ceil(32) * 32, 0);
        }
        std::fs::write(path, b).unwrap();
    }

    fn bf16(values: impl IntoIterator<Item = f32>) -> Vec<u8> {
        values
            .into_iter()
            .flat_map(|v| half::bf16::from_f32(v).to_le_bytes())
            .collect()
    }

    #[test]
    fn a_gguf_transformer_loads_by_diffusers_names_or_says_why_not() {
        let d = MetalDevice::new().unwrap();
        let path = std::env::temp_dir().join(format!("ferrum-dit-{}.gguf", std::process::id()));
        let (hidden, ffn) = (64usize, 32usize);
        let q = bf16((0..16 * hidden).map(|i| i as f32 / 64.));
        // The stacked SwiGLU input: gate rows, then up rows.
        let gate_up = bf16((0..2 * ffn * hidden).map(|i| if i < ffn * hidden { 1. } else { 2. }));
        let norm: Vec<u8> = (0..hidden)
            .flat_map(|i| half::f16::from_f32(i as f32).to_le_bytes())
            .collect();
        let tensors = [
            ("blocks.q.weight", vec![hidden, 16], 30, q.clone()),
            (
                "transformer_blocks.0.img_mlp.gate_up.weight",
                vec![hidden, 2 * ffn],
                30,
                gate_up,
            ),
            ("norm.weight", vec![hidden], 1, norm),
            // Q8_0, a matrix encoding, where a float vector belongs.
            ("scale", vec![32], 8, vec![0; 34]),
            // F16 matrices have no kernel; IQ2_S has one, but not in the
            // image transformer's GEMM.
            ("f16.weight", vec![hidden, 2], 1, vec![0; 4 * hidden]),
            ("iq2_s.weight", vec![256, 1], 22, vec![0; 82]),
        ]
        .map(|(name, dims, t, bytes)| (format!("{GGUF_PREFIX}{name}"), dims, t, bytes));
        write(&path, &[], &tensors);
        let w = GgufWeights::open(&path).unwrap();
        let loaded = w.matrix(&d, "blocks.q.weight", 16, hidden, None).unwrap();
        assert_eq!(loaded.format, Format::Bf16);
        assert_eq!(loaded.packed_bytes(), q);
        // Rows and columns swapped is a shape error, not a transposed load.
        assert!(w.matrix(&d, "blocks.q.weight", hidden, 16, None).is_err());
        assert!(w.matrix(&d, "missing.weight", 16, hidden, None).is_err());

        let (gate, up) = w.mlp_in(&d, 0, ffn, hidden, None).unwrap();
        assert_eq!(gate.packed_bytes(), bf16(vec![1.; ffn * hidden]));
        assert_eq!(up.packed_bytes(), bf16(vec![2.; ffn * hidden]));

        let norm = w.vector(&d, "norm.weight", hidden, 1.).unwrap().to_f32();
        assert_eq!(norm, (0..hidden).map(|i| i as f32 + 1.).collect::<Vec<_>>());
        assert!(w.vector(&d, "scale", 32, 0.).is_err());

        assert!(w.matrix(&d, "f16.weight", 2, hidden, None).is_err());
        let iq = w.matrix(&d, "iq2_s.weight", 1, 256, None).unwrap();
        let message = check_projectable(&iq, "iq2_s.weight")
            .unwrap_err()
            .to_string();
        assert!(message.contains("iq2_s.weight"), "{message}");
        std::fs::remove_file(path).unwrap();
    }
}
