//! Packed GGUF weights for the hybrid engine. Matrices keep their GGML block
//! encoding for the model lifetime and are read directly into Metal storage.
#![forbid(unsafe_code)]
use super::config::{HybridConfig, Variant};
use crate::{
    DType, Error, MetalDevice, Result, Tensor,
    loader::gguf::{GgufFile, block_layout},
    metal::MetalBuffer,
    tensor::PackedStorage,
};

/// GGML encodings with hybrid-engine kernels.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Format {
    F32,
    Q4_0,
    Q8_0,
    Q4K,
    Q5K,
    Q6K,
    Iq3S,
    Iq4Xs,
}

impl Format {
    fn from_type(type_id: u32) -> Option<Self> {
        Some(match type_id {
            0 => Self::F32,
            2 => Self::Q4_0,
            8 => Self::Q8_0,
            12 => Self::Q4K,
            13 => Self::Q5K,
            14 => Self::Q6K,
            21 => Self::Iq3S,
            23 => Self::Iq4Xs,
            _ => return None,
        })
    }
    pub fn name(self) -> &'static str {
        match self {
            Self::F32 => "f32",
            Self::Q4_0 => "q4_0",
            Self::Q8_0 => "q8_0",
            Self::Q4K => "q4_k",
            Self::Q5K => "q5_k",
            Self::Q6K => "q6_k",
            Self::Iq3S => "iq3_s",
            Self::Iq4Xs => "iq4_xs",
        }
    }
}

/// Row-major logical `[rows, cols]` weight (GGUF dims reversed), optionally
/// stacked as `experts` equal slices along the row axis.
pub struct Matrix {
    pub rows: usize,
    pub cols: usize,
    pub experts: usize,
    pub format: Format,
    pub row_bytes: usize,
    storage: PackedStorage,
}

impl Matrix {
    pub(crate) fn binding(&self) -> (&MetalBuffer, usize, usize) {
        self.storage.binding()
    }
    pub fn byte_size(&self) -> usize {
        self.storage.byte_len()
    }
    /// Rows of one expert slice (all rows for a plain matrix).
    pub fn expert_rows(&self) -> usize {
        self.rows / self.experts
    }
}

pub struct AttentionWeights {
    pub q: Matrix,
    pub k: Matrix,
    pub v: Matrix,
    pub o: Matrix,
    pub q_norm: Tensor,
    pub k_norm: Tensor,
}

pub struct DeltaWeights {
    pub qkv: Matrix,
    pub z: Matrix,
    pub alpha: Matrix,
    pub beta: Matrix,
    pub out: Matrix,
    pub conv: Tensor,
    pub a: Tensor,
    pub dt_bias: Tensor,
    pub norm: Tensor,
}

#[allow(clippy::large_enum_variant)] // one per layer; never moved in hot paths
pub enum Mixer {
    Attention(AttentionWeights),
    Delta(DeltaWeights),
}

pub struct DenseFfn {
    pub gate: Matrix,
    pub up: Matrix,
    pub down: Matrix,
}

pub struct MoeFfn {
    pub router: Matrix,
    pub gate_exps: Matrix,
    pub up_exps: Matrix,
    pub down_exps: Matrix,
    pub shared_gate_inp: Tensor,
    pub shared: DenseFfn,
}

#[allow(clippy::large_enum_variant)]
pub enum Ffn {
    Dense(DenseFfn),
    Moe(MoeFfn),
}

pub struct Layer {
    pub attn_norm: Tensor,
    pub post_norm: Tensor,
    pub mixer: Mixer,
    pub ffn: Ffn,
}

pub struct Weights {
    pub embedding: Matrix,
    pub output_norm: Tensor,
    pub output: Matrix,
    pub layers: Vec<Layer>,
}

impl Weights {
    pub fn byte_size(&self) -> usize {
        let m = |x: &Matrix| x.byte_size();
        let dense = |f: &DenseFfn| m(&f.gate) + m(&f.up) + m(&f.down);
        let mut total = m(&self.embedding) + m(&self.output) + self.output_norm.byte_size();
        for layer in &self.layers {
            total += layer.attn_norm.byte_size() + layer.post_norm.byte_size();
            total += match &layer.mixer {
                Mixer::Attention(a) => {
                    m(&a.q)
                        + m(&a.k)
                        + m(&a.v)
                        + m(&a.o)
                        + a.q_norm.byte_size()
                        + a.k_norm.byte_size()
                }
                Mixer::Delta(d) => {
                    m(&d.qkv)
                        + m(&d.z)
                        + m(&d.alpha)
                        + m(&d.beta)
                        + m(&d.out)
                        + d.conv.byte_size()
                        + d.a.byte_size()
                        + d.dt_bias.byte_size()
                        + d.norm.byte_size()
                }
            };
            total += match &layer.ffn {
                Ffn::Dense(f) => dense(f),
                Ffn::Moe(f) => {
                    m(&f.router)
                        + m(&f.gate_exps)
                        + m(&f.up_exps)
                        + m(&f.down_exps)
                        + f.shared_gate_inp.byte_size()
                        + dense(&f.shared)
                }
            };
        }
        total
    }
}

struct Loader<'a> {
    device: &'a MetalDevice,
    file: &'a mut GgufFile,
}

impl Loader<'_> {
    fn matrix(&mut self, name: &str, rows: usize, cols: usize) -> Result<Matrix> {
        self.stacked(name, 1, rows, cols)
    }

    /// Load `[experts][rows][cols]` (GGUF dims `[cols, rows, experts]`).
    fn stacked(&mut self, name: &str, experts: usize, rows: usize, cols: usize) -> Result<Matrix> {
        let info = self.file.tensor(name)?.clone();
        let expected: Vec<usize> = if experts == 1 {
            vec![cols, rows]
        } else {
            vec![cols, rows, experts]
        };
        if info.dimensions != expected {
            return Err(Error::Weight {
                name: name.into(),
                message: format!(
                    "expected GGUF dims {expected:?}, found {:?}",
                    info.dimensions
                ),
            });
        }
        let format = Format::from_type(info.type_id).ok_or_else(|| Error::Weight {
            name: name.into(),
            message: format!("GGML type {} has no hybrid kernel", info.type_id),
        })?;
        let layout = block_layout(info.type_id)?;
        if !cols.is_multiple_of(layout.elements)
            || (format != Format::F32 && !cols.is_multiple_of(256))
        {
            return Err(Error::Weight {
                name: name.into(),
                message: "row width is not a whole number of blocks".into(),
            });
        }
        let row_bytes = cols / layout.elements * layout.bytes;
        if row_bytes * rows * experts != info.byte_len {
            return Err(Error::Weight {
                name: name.into(),
                message: "byte length differs from geometry".into(),
            });
        }
        let file = &mut *self.file;
        let storage = PackedStorage::from_reader(self.device, info.byte_len, |dst| {
            file.read_tensor_into(name, dst)
        })?;
        Ok(Matrix {
            rows: rows * experts,
            cols,
            experts,
            format,
            row_bytes,
            storage,
        })
    }

    fn vector(&mut self, name: &str, len: usize) -> Result<Tensor> {
        let info = self.file.tensor(name)?.clone();
        if info.type_id != 0 || info.dimensions.iter().product::<usize>() != len {
            return Err(Error::Weight {
                name: name.into(),
                message: format!("expected {len} F32 values"),
            });
        }
        let file = &mut *self.file;
        let tensor = Tensor::from_reader(self.device, [len], DType::F32, |dst| {
            file.read_tensor_into(name, dst)
        })?;
        if tensor.to_f32().iter().any(|v| !v.is_finite()) {
            return Err(Error::Weight {
                name: name.into(),
                message: "non-finite value".into(),
            });
        }
        Ok(tensor)
    }
}

pub fn load(device: &MetalDevice, file: &mut GgufFile, c: &HybridConfig) -> Result<Weights> {
    let mut l = Loader { device, file };
    let embedding = l.matrix("token_embd.weight", c.vocab, c.hidden)?;
    let output_norm = l.vector("output_norm.weight", c.hidden)?;
    let output = if l.file.tensors().contains_key("output.weight") {
        l.matrix("output.weight", c.vocab, c.hidden)?
    } else {
        l.matrix("token_embd.weight", c.vocab, c.hidden)?
    };
    let mut layers = Vec::with_capacity(c.layers);
    for i in 0..c.layers {
        let n = |s: &str| format!("blk.{i}.{s}");
        let attn_norm = l.vector(&n("attn_norm.weight"), c.hidden)?;
        let post_norm = l.vector(&n("post_attention_norm.weight"), c.hidden)?;
        let mixer = if c.is_attention(i) {
            let q = c.heads * c.head_dim;
            let kv = c.kv_heads * c.head_dim;
            Mixer::Attention(AttentionWeights {
                q: l.matrix(&n("attn_q.weight"), 2 * q, c.hidden)?,
                k: l.matrix(&n("attn_k.weight"), kv, c.hidden)?,
                v: l.matrix(&n("attn_v.weight"), kv, c.hidden)?,
                o: l.matrix(&n("attn_output.weight"), c.hidden, q)?,
                q_norm: l.vector(&n("attn_q_norm.weight"), c.head_dim)?,
                k_norm: l.vector(&n("attn_k_norm.weight"), c.head_dim)?,
            })
        } else {
            let v_heads = c.ssm_v_heads;
            Mixer::Delta(DeltaWeights {
                qkv: l.matrix(&n("attn_qkv.weight"), c.conv_channels(), c.hidden)?,
                z: l.matrix(&n("attn_gate.weight"), c.ssm_value_dim(), c.hidden)?,
                alpha: l.matrix(&n("ssm_alpha.weight"), v_heads, c.hidden)?,
                beta: l.matrix(&n("ssm_beta.weight"), v_heads, c.hidden)?,
                out: l.matrix(&n("ssm_out.weight"), c.hidden, c.ssm_value_dim())?,
                conv: l.vector(&n("ssm_conv1d.weight"), c.conv_channels() * c.conv_kernel)?,
                a: l.vector(&n("ssm_a"), v_heads)?,
                dt_bias: l.vector(&n("ssm_dt.bias"), v_heads)?,
                norm: l.vector(&n("ssm_norm.weight"), c.ssm_head_dim)?,
            })
        };
        let ffn = match c.variant {
            Variant::Dense => Ffn::Dense(DenseFfn {
                gate: l.matrix(&n("ffn_gate.weight"), c.ffn, c.hidden)?,
                up: l.matrix(&n("ffn_up.weight"), c.ffn, c.hidden)?,
                down: l.matrix(&n("ffn_down.weight"), c.hidden, c.ffn)?,
            }),
            Variant::Moe => {
                let moe = c.moe.expect("MoE variant carries expert geometry");
                Ffn::Moe(MoeFfn {
                    router: l.matrix(&n("ffn_gate_inp.weight"), moe.experts, c.hidden)?,
                    gate_exps: l.stacked(
                        &n("ffn_gate_exps.weight"),
                        moe.experts,
                        moe.expert_ffn,
                        c.hidden,
                    )?,
                    up_exps: l.stacked(
                        &n("ffn_up_exps.weight"),
                        moe.experts,
                        moe.expert_ffn,
                        c.hidden,
                    )?,
                    down_exps: l.stacked(
                        &n("ffn_down_exps.weight"),
                        moe.experts,
                        c.hidden,
                        moe.expert_ffn,
                    )?,
                    shared_gate_inp: l.vector(&n("ffn_gate_inp_shexp.weight"), c.hidden)?,
                    shared: DenseFfn {
                        gate: l.matrix(&n("ffn_gate_shexp.weight"), moe.shared_ffn, c.hidden)?,
                        up: l.matrix(&n("ffn_up_shexp.weight"), moe.shared_ffn, c.hidden)?,
                        down: l.matrix(&n("ffn_down_shexp.weight"), c.hidden, moe.shared_ffn)?,
                    },
                })
            }
        };
        layers.push(Layer {
            attn_norm,
            post_norm,
            mixer,
            ffn,
        });
    }
    Ok(Weights {
        embedding,
        output_norm,
        output,
        layers,
    })
}
