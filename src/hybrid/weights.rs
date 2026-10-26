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
    /// Only MoE routers, embeddings, LM heads and Spark's head gates load as
    /// BF16 (see `Loader::matrix_or_bf16`).
    Bf16,
    Q4_0,
    Q8_0,
    Q4K,
    Q5K,
    Q6K,
    Iq3Xxs,
    Iq3S,
    Iq2S,
    Iq4Xs,
    /// MLX affine 4-bit, group 64: each row holds its `cols / 8` code words,
    /// then its `cols / 64` bf16 scales, then as many bf16 biases
    /// (`Matrix::from_mlx`).
    Mlx4,
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
            18 => Self::Iq3Xxs,
            21 => Self::Iq3S,
            22 => Self::Iq2S,
            23 => Self::Iq4Xs,
            _ => return None,
        })
    }
    pub fn name(self) -> &'static str {
        match self {
            Self::F32 => "f32",
            Self::Bf16 => "bf16",
            Self::Q4_0 => "q4_0",
            Self::Q8_0 => "q8_0",
            Self::Q4K => "q4_k",
            Self::Q5K => "q5_k",
            Self::Q6K => "q6_k",
            Self::Iq3Xxs => "iq3_xxs",
            Self::Iq3S => "iq3_s",
            Self::Iq2S => "iq2_s",
            Self::Iq4Xs => "iq4_xs",
            Self::Mlx4 => "mlx4",
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
    /// The first `rows` rows, sharing storage (a draft vocabulary prefix of
    /// an LM head).
    pub(crate) fn prefix_rows(&self, rows: usize) -> Matrix {
        Matrix {
            rows: rows.min(self.rows),
            cols: self.cols,
            experts: 1,
            format: self.format,
            row_bytes: self.row_bytes,
            storage: self.storage.clone(),
        }
    }
}

pub struct AttentionWeights {
    pub q: Matrix,
    pub k: Matrix,
    pub v: Matrix,
    pub o: Matrix,
    /// Per-head RMS norms of Q and K (absent for Spark, which has none).
    pub q_norm: Option<Tensor>,
    pub k_norm: Option<Tensor>,
    /// Per-head sink logits, for layers whose `AttnSpec::sinks` is set.
    pub sinks: Option<Tensor>,
    /// Spark's `[heads, hidden]` output gate: each head's attention output is
    /// scaled by sigmoid(gate · attn_norm(x)).
    pub head_gate: Option<Matrix>,
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
    /// Sigmoid gate of the shared expert (absent when `MoeConfig::shared_gate` is off).
    pub shared_gate_inp: Option<Tensor>,
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
                        + a.q_norm
                            .iter()
                            .chain(&a.k_norm)
                            .chain(&a.sinks)
                            .map(Tensor::byte_size)
                            .sum::<usize>()
                        + a.head_gate.as_ref().map_or(0, Matrix::byte_size)
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
                        + f.shared_gate_inp.as_ref().map_or(0, Tensor::byte_size)
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

    fn stacked(&mut self, name: &str, experts: usize, rows: usize, cols: usize) -> Result<Matrix> {
        gguf_matrix(self.device, self.file, name, experts, rows, cols)
    }

    /// A plain matrix (router, embedding, LM head, head gate) that may be
    /// BF16. BF16 keeps its encoding; the BF16 kernels widen each value to F32
    /// as they read it. Other encodings load as any matrix does.
    fn matrix_or_bf16(&mut self, name: &str, rows: usize, cols: usize) -> Result<Matrix> {
        let info = self.file.tensor(name)?.clone();
        if info.type_id != 30 {
            return self.matrix(name, rows, cols);
        }
        if info.dimensions != [cols, rows]
            || info.byte_len != rows * cols * 2
            || !cols.is_multiple_of(256)
        {
            return Err(Error::Weight {
                name: name.into(),
                message: format!("expected a BF16 [{rows}, {cols}] matrix"),
            });
        }
        let file = &mut *self.file;
        let storage = PackedStorage::from_reader(self.device, info.byte_len, |dst| {
            file.read_tensor_into(name, dst)
        })?;
        Ok(Matrix {
            rows,
            cols,
            experts: 1,
            format: Format::Bf16,
            row_bytes: cols * 2,
            storage,
        })
    }

    /// A row-stacked matrix (Spark's fused `attn_qkv`) split into consecutive
    /// blocks of `parts` rows, each copied into its own storage.
    fn split(&mut self, name: &str, parts: &[usize], cols: usize) -> Result<Vec<Matrix>> {
        let info = self.file.tensor(name)?.clone();
        let rows: usize = parts.iter().sum();
        let bad = |message: String| Error::Weight {
            name: name.into(),
            message,
        };
        let (format, row_bytes) = match info.type_id {
            30 => (Format::Bf16, cols * 2),
            t => {
                let format = Format::from_type(t)
                    .ok_or_else(|| bad(format!("GGML type {t} has no hybrid kernel")))?;
                let layout = block_layout(t)?;
                (format, cols / layout.elements * layout.bytes)
            }
        };
        if info.dimensions != [cols, rows] {
            return Err(bad(format!(
                "expected GGUF dims {:?}, found {:?}",
                [cols, rows],
                info.dimensions
            )));
        }
        if !cols.is_multiple_of(256) || row_bytes * rows != info.byte_len {
            return Err(bad("byte length differs from geometry".into()));
        }
        let mut bytes = vec![0; info.byte_len];
        self.file.read_tensor_into(name, &mut bytes)?;
        let mut start = 0;
        parts
            .iter()
            .map(|&n| {
                let block = &bytes[start * row_bytes..(start + n) * row_bytes];
                start += n;
                Ok(Matrix {
                    rows: n,
                    cols,
                    experts: 1,
                    format,
                    row_bytes,
                    storage: PackedStorage::from_reader(self.device, block.len(), |dst| {
                        dst.copy_from_slice(block);
                        Ok(())
                    })?,
                })
            })
            .collect()
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

/// Load `[experts][rows][cols]` (GGUF dims `[cols, rows, experts]`) in its
/// stored encoding.
pub(crate) fn gguf_matrix(
    device: &MetalDevice,
    file: &mut GgufFile,
    name: &str,
    experts: usize,
    rows: usize,
    cols: usize,
) -> Result<Matrix> {
    let info = file.tensor(name)?.clone();
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
    if !cols.is_multiple_of(layout.elements) || (format != Format::F32 && !cols.is_multiple_of(256))
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
    let storage = PackedStorage::from_reader(device, info.byte_len, |dst| {
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

pub fn load(device: &MetalDevice, file: &mut GgufFile, c: &HybridConfig) -> Result<Weights> {
    let mut l = Loader { device, file };
    let embedding = l.matrix_or_bf16("token_embd.weight", c.vocab, c.hidden)?;
    let output_norm = l.vector("output_norm.weight", c.hidden)?;
    let output = if l.file.tensors().contains_key("output.weight") {
        l.matrix_or_bf16("output.weight", c.vocab, c.hidden)?
    } else {
        l.matrix_or_bf16("token_embd.weight", c.vocab, c.hidden)?
    };
    let mut layers = Vec::with_capacity(c.layers);
    for i in 0..c.layers {
        let n = |s: &str| format!("blk.{i}.{s}");
        let attn_norm = l.vector(&n("attn_norm.weight"), c.hidden)?;
        // Flint and Spark have no separate post-attention norm: `ffn_norm` sits there.
        let post_norm = match c.variant {
            Variant::Flint | Variant::Spark => l.vector(&n("ffn_norm.weight"), c.hidden)?,
            Variant::Dense | Variant::Moe => {
                l.vector(&n("post_attention_norm.weight"), c.hidden)?
            }
        };
        let mixer = if c.is_attention(i) {
            let q = c.heads * c.head_dim;
            let kv = c.kv_heads * c.head_dim;
            let spec = c.attn[i];
            let spark = c.variant == Variant::Spark;
            let (wq, wk, wv) = if spark {
                let mut qkv = l.split(&n("attn_qkv.weight"), &[q, kv, kv], c.hidden)?;
                let wv = qkv.pop().expect("three parts");
                let wk = qkv.pop().expect("three parts");
                (qkv.pop().expect("three parts"), wk, wv)
            } else {
                (
                    l.matrix(
                        &n("attn_q.weight"),
                        if spec.gated { 2 * q } else { q },
                        c.hidden,
                    )?,
                    l.matrix(&n("attn_k.weight"), kv, c.hidden)?,
                    l.matrix(&n("attn_v.weight"), kv, c.hidden)?,
                )
            };
            Mixer::Attention(AttentionWeights {
                q: wq,
                k: wk,
                v: wv,
                o: l.matrix(&n("attn_output.weight"), c.hidden, q)?,
                q_norm: match spark {
                    true => None,
                    false => Some(l.vector(&n("attn_q_norm.weight"), c.head_dim)?),
                },
                k_norm: match spark {
                    true => None,
                    false => Some(l.vector(&n("attn_k_norm.weight"), c.head_dim)?),
                },
                sinks: match spec.sinks {
                    true => Some(l.vector(&n("attn_sinks.weight"), c.heads)?),
                    false => None,
                },
                head_gate: match spark {
                    true => Some(l.matrix_or_bf16(&n("attn_gate.weight"), c.heads, c.hidden)?),
                    false => None,
                },
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
        let ffn = if c.dense_ffn(i) {
            Ffn::Dense(DenseFfn {
                gate: l.matrix(&n("ffn_gate.weight"), c.ffn, c.hidden)?,
                up: l.matrix(&n("ffn_up.weight"), c.ffn, c.hidden)?,
                down: l.matrix(&n("ffn_down.weight"), c.hidden, c.ffn)?,
            })
        } else {
            let moe = c.moe.expect("expert layer carries expert geometry");
            Ffn::Moe(MoeFfn {
                // `HybridConfig::from_gguf` refuses a selection bias (`exp_probs_b`).
                router: l.matrix_or_bf16(&n("ffn_gate_inp.weight"), moe.experts, c.hidden)?,
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
                shared_gate_inp: match moe.shared_gate {
                    true => Some(l.vector(&n("ffn_gate_inp_shexp.weight"), c.hidden)?),
                    false => None,
                },
                shared: DenseFfn {
                    gate: l.matrix(&n("ffn_gate_shexp.weight"), moe.shared_ffn, c.hidden)?,
                    up: l.matrix(&n("ffn_up_shexp.weight"), moe.shared_ffn, c.hidden)?,
                    down: l.matrix(&n("ffn_down_shexp.weight"), c.hidden, moe.shared_ffn)?,
                },
            })
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

/// The NextN/MTP block after the trunk (`blk.{layers}`): the token embedding
/// and the previous hidden state are normalized, concatenated and projected,
/// then pass one gated-attention layer with a dense FFN; `head_norm` feeds
/// the shared LM head.
pub struct MtpWeights {
    pub eh_proj: Matrix,
    pub enorm: Tensor,
    pub hnorm: Tensor,
    pub head_norm: Tensor,
    pub attn_norm: Tensor,
    pub post_norm: Tensor,
    pub attention: AttentionWeights,
    pub ffn: DenseFfn,
}

impl MtpWeights {
    pub fn byte_size(&self) -> usize {
        let a = &self.attention;
        [
            &self.eh_proj,
            &a.q,
            &a.k,
            &a.v,
            &a.o,
            &self.ffn.gate,
            &self.ffn.up,
            &self.ffn.down,
        ]
        .iter()
        .map(|m| m.byte_size())
        .sum::<usize>()
            + [
                &self.enorm,
                &self.hnorm,
                &self.head_norm,
                &self.attn_norm,
                &self.post_norm,
            ]
            .into_iter()
            .chain(&a.q_norm)
            .chain(&a.k_norm)
            .map(|t| t.byte_size())
            .sum::<usize>()
    }
}

/// Load the first MTP block, if the GGUF has one with a dense FFN.
pub fn load_mtp(device: &MetalDevice, file: &mut GgufFile, c: &HybridConfig) -> Result<MtpWeights> {
    let i = c.layers;
    let n = |s: &str| format!("blk.{i}.{s}");
    if !file.tensors().contains_key(&n("nextn.eh_proj.weight")) {
        return Err(Error::Gguf("the GGUF has no NextN/MTP block".into()));
    }
    if file.tensors().contains_key(&n("ffn_gate_exps.weight")) {
        return Err(Error::Gguf(
            "MTP blocks with routed experts are not supported".into(),
        ));
    }
    let ffn_width = file.tensor(&n("ffn_gate.weight"))?.dimensions[1];
    let mut l = Loader { device, file };
    let q = c.heads * c.head_dim;
    let kv = c.kv_heads * c.head_dim;
    Ok(MtpWeights {
        eh_proj: l.matrix(&n("nextn.eh_proj.weight"), c.hidden, 2 * c.hidden)?,
        enorm: l.vector(&n("nextn.enorm.weight"), c.hidden)?,
        hnorm: l.vector(&n("nextn.hnorm.weight"), c.hidden)?,
        head_norm: l.vector(&n("nextn.shared_head_norm.weight"), c.hidden)?,
        attn_norm: l.vector(&n("attn_norm.weight"), c.hidden)?,
        post_norm: l.vector(&n("post_attention_norm.weight"), c.hidden)?,
        attention: AttentionWeights {
            q: l.matrix(&n("attn_q.weight"), 2 * q, c.hidden)?,
            k: l.matrix(&n("attn_k.weight"), kv, c.hidden)?,
            v: l.matrix(&n("attn_v.weight"), kv, c.hidden)?,
            o: l.matrix(&n("attn_output.weight"), c.hidden, q)?,
            q_norm: Some(l.vector(&n("attn_q_norm.weight"), c.head_dim)?),
            k_norm: Some(l.vector(&n("attn_k_norm.weight"), c.head_dim)?),
            sinks: None,
            head_gate: None,
        },
        ffn: DenseFfn {
            gate: l.matrix(&n("ffn_gate.weight"), ffn_width, c.hidden)?,
            up: l.matrix(&n("ffn_up.weight"), ffn_width, c.hidden)?,
            down: l.matrix(&n("ffn_down.weight"), c.hidden, ffn_width)?,
        },
    })
}

impl Matrix {
    /// A BF16 `[rows, cols]` matrix kept in its stored encoding, filled by
    /// `fill` straight into device memory (`Format::Bf16`).
    pub(crate) fn bf16_with(
        device: &MetalDevice,
        rows: usize,
        cols: usize,
        fill: impl FnOnce(&mut [u8]) -> Result<()>,
    ) -> Result<Self> {
        if !cols.is_multiple_of(16) {
            return Err(Error::Shape(format!(
                "BF16 matrix [{rows}, {cols}]: width is not a multiple of 16"
            )));
        }
        let row_bytes = cols * 2;
        let storage = PackedStorage::from_reader(device, rows * row_bytes, fill)?;
        Ok(Matrix {
            rows,
            cols,
            experts: 1,
            format: Format::Bf16,
            row_bytes,
            storage,
        })
    }

    /// Quantize a row-major BF16 `[rows, cols]` matrix (little-endian bytes)
    /// into `format` (Q8_0, Q4_0 or F32), rows in parallel.
    pub(crate) fn from_bf16(
        device: &MetalDevice,
        rows: usize,
        cols: usize,
        bf16: &[u8],
        format: Format,
    ) -> Result<Self> {
        if bf16.len() != rows * cols * 2 {
            return Err(Error::Shape(format!(
                "BF16 matrix [{rows}, {cols}] from {} bytes",
                bf16.len()
            )));
        }
        Self::from_rows(device, rows, cols, format, |r, row| {
            let src = &bf16[r * cols * 2..(r + 1) * cols * 2];
            for (v, b) in row.iter_mut().zip(src.chunks_exact(2)) {
                *v = f32::from_bits(u32::from(u16::from_le_bytes([b[0], b[1]])) << 16);
            }
        })
    }

    /// Quantize `rows` rows of `cols` values into `format` (Q8_0, Q4_0 or
    /// F32), in parallel; `fill(r, row)` writes row `r` as F32.
    pub(crate) fn from_rows(
        device: &MetalDevice,
        rows: usize,
        cols: usize,
        format: Format,
        fill: impl Fn(usize, &mut [f32]) + Sync,
    ) -> Result<Self> {
        if !cols.is_multiple_of(32) {
            return Err(Error::Shape(format!(
                "matrix [{rows}, {cols}]: width is not a multiple of 32"
            )));
        }
        let row_bytes = match format {
            Format::Q8_0 => cols / 32 * 34,
            Format::Q4_0 => cols / 32 * 18,
            Format::F32 => cols * 4,
            f => {
                return Err(Error::Parameter(format!("cannot quantize to {}", f.name())));
            }
        };
        let fill = &fill;
        let storage = PackedStorage::from_reader(device, rows * row_bytes, |dst| {
            let threads = std::thread::available_parallelism().map_or(4, |n| n.get());
            let per = rows.div_ceil(threads).max(1);
            std::thread::scope(|scope| {
                for (index, out) in dst.chunks_mut(per * row_bytes).enumerate() {
                    let first = index * per;
                    scope.spawn(move || {
                        let mut row = vec![0f32; cols];
                        for (r, out_row) in out.chunks_mut(row_bytes).enumerate() {
                            fill(first + r, &mut row);
                            quantize_row(&row, out_row, format);
                        }
                    });
                }
            });
            Ok(())
        })?;
        Ok(Matrix {
            rows,
            cols,
            experts: 1,
            format,
            row_bytes,
            storage,
        })
    }

    /// `experts` stacked slices of `rows` rows each, from raw packed bytes in
    /// `format`, for kernel tests.
    #[cfg(test)]
    pub(crate) fn from_packed(
        device: &MetalDevice,
        rows: usize,
        cols: usize,
        experts: usize,
        format: Format,
        bytes: &[u8],
    ) -> Result<Self> {
        let type_id = match format {
            Format::Iq3Xxs => 18,
            Format::Iq3S => 21,
            Format::Iq2S => 22,
            Format::Bf16 => 30,
            f => {
                return Err(Error::Parameter(format!(
                    "no raw test path for {}",
                    f.name()
                )));
            }
        };
        let layout = block_layout(type_id)?;
        let row_bytes = cols / layout.elements * layout.bytes;
        if row_bytes * rows * experts != bytes.len() {
            return Err(Error::Shape(format!(
                "{rows} x {experts} rows of {row_bytes} bytes against {} bytes",
                bytes.len()
            )));
        }
        let storage = PackedStorage::from_reader(device, bytes.len(), |dst| {
            dst.copy_from_slice(bytes);
            Ok(())
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

    /// An MLX affine 4-bit, group-64 matrix from its three arrays (`codes`
    /// `[rows, cols / 8]` little-endian u32 words, `scales` and `biases`
    /// `[rows, cols / 64]` bf16), repacked so each row holds its codes, then
    /// its scales, then its biases (`Format::Mlx4`).
    pub(crate) fn from_mlx(
        device: &MetalDevice,
        rows: usize,
        cols: usize,
        codes: &[u8],
        scales: &[u8],
        biases: &[u8],
    ) -> Result<Self> {
        if !cols.is_multiple_of(64)
            || codes.len() != rows * cols / 2
            || scales.len() != rows * cols / 32
            || biases.len() != rows * cols / 32
        {
            return Err(Error::Shape(format!(
                "MLX 4-bit [{rows}, {cols}]: {} code, {} scale and {} bias bytes",
                codes.len(),
                scales.len(),
                biases.len()
            )));
        }
        let (code_bytes, group_bytes) = (cols / 2, cols / 32);
        let row_bytes = code_bytes + 2 * group_bytes;
        let storage = PackedStorage::from_reader(device, rows * row_bytes, |dst| {
            for (r, out) in dst.chunks_exact_mut(row_bytes).enumerate() {
                out[..code_bytes].copy_from_slice(&codes[r * code_bytes..][..code_bytes]);
                out[code_bytes..][..group_bytes]
                    .copy_from_slice(&scales[r * group_bytes..][..group_bytes]);
                out[code_bytes + group_bytes..]
                    .copy_from_slice(&biases[r * group_bytes..][..group_bytes]);
            }
            Ok(())
        })?;
        Ok(Matrix {
            rows,
            cols,
            experts: 1,
            format: Format::Mlx4,
            row_bytes,
            storage,
        })
    }

    /// The packed bytes, for tests that compare two loads.
    #[cfg(test)]
    pub(crate) fn packed_bytes(&self) -> Vec<u8> {
        self.storage.with_bytes(<[u8]>::to_vec)
    }
}

/// ggml's reference Q8_0 / Q4_0 row quantization.
fn quantize_row(x: &[f32], out: &mut [u8], format: Format) {
    match format {
        Format::F32 => {
            for (o, v) in out.chunks_exact_mut(4).zip(x) {
                o.copy_from_slice(&v.to_le_bytes());
            }
        }
        Format::Q8_0 => {
            for (block, o) in x.chunks_exact(32).zip(out.chunks_exact_mut(34)) {
                let amax = block.iter().fold(0f32, |m, v| m.max(v.abs()));
                let d = amax / 127.;
                let id = if d != 0. { 1. / d } else { 0. };
                o[..2].copy_from_slice(&half::f16::from_f32(d).to_le_bytes());
                for (q, v) in o[2..].iter_mut().zip(block) {
                    *q = ((v * id).round() as i8) as u8;
                }
            }
        }
        Format::Q4_0 => {
            for (block, o) in x.chunks_exact(32).zip(out.chunks_exact_mut(18)) {
                let max = block
                    .iter()
                    .fold(0f32, |m, &v| if v.abs() > m.abs() { v } else { m });
                let d = max / -8.;
                let id = if d != 0. { 1. / d } else { 0. };
                o[..2].copy_from_slice(&half::f16::from_f32(d).to_le_bytes());
                for j in 0..16 {
                    let lo = ((block[j] * id + 8.5) as i32).clamp(0, 15) as u8;
                    let hi = ((block[j + 16] * id + 8.5) as i32).clamp(0, 15) as u8;
                    o[2 + j] = lo | (hi << 4);
                }
            }
        }
        _ => unreachable!("checked by the caller"),
    }
}
