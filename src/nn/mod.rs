#![forbid(unsafe_code)]
pub mod attention;
pub mod kv_cache;
pub mod moe;
use crate::{DType, Error, MetalDevice, Result, Tensor, quantization::QuantizedMatrix};

enum MatrixWeight {
    Dense(Tensor),
    Quantized(QuantizedMatrix),
}
impl MatrixWeight {
    fn byte_size(&self) -> usize {
        match self {
            Self::Dense(t) => t.byte_size(),
            Self::Quantized(t) => t.byte_size(),
        }
    }
}

pub struct Embedding {
    weight: MatrixWeight,
    output_dtype: DType,
}
impl Embedding {
    pub fn new(weight: Tensor) -> Result<Self> {
        if weight.shape().rank() != 2 || weight.shape().dimensions().contains(&0) {
            return Err(Error::Shape(
                "embedding requires nonempty [vocab,hidden]".into(),
            ));
        }
        let output_dtype = weight.dtype();
        Ok(Self {
            weight: MatrixWeight::Dense(weight),
            output_dtype,
        })
    }
    pub(crate) fn new_quantized(
        d: &MetalDevice,
        weight: QuantizedMatrix,
        output_dtype: DType,
    ) -> Result<Self> {
        if weight.rows() == 0 || weight.columns() == 0 {
            return Err(Error::Shape(
                "quantized embedding requires nonempty [vocab,hidden]".into(),
            ));
        }
        if !d.owns(weight.buffer()) {
            return Err(Error::DeviceMismatch);
        }
        Ok(Self {
            weight: MatrixWeight::Quantized(weight),
            output_dtype,
        })
    }
    pub fn weight_bytes(&self) -> usize {
        self.weight.byte_size()
    }
    pub fn forward(&self, d: &MetalDevice, tokens: &[u32]) -> Result<Tensor> {
        let output = match &self.weight {
            MatrixWeight::Dense(weight) => d.embedding_gather(weight, tokens)?,
            MatrixWeight::Quantized(weight) => {
                d.embedding_gather_quantized(weight, tokens, self.output_dtype)?
            }
        };
        Ok(output.tensor)
    }
}
pub struct Linear {
    weight: MatrixWeight,
    bias: Option<Tensor>,
    input: usize,
    output: usize,
}
impl Linear {
    pub fn new(d: &MetalDevice, weight: Tensor, bias: Option<Tensor>) -> Result<Self> {
        let dims = weight.shape().dimensions();
        if dims.len() != 2 || dims.contains(&0) {
            return Err(Error::Shape(
                "linear weight requires nonempty [out,in]".into(),
            ));
        }
        if let Some(b) = &bias {
            if b.shape().dimensions() != [dims[0]] {
                return Err(Error::Shape("linear bias requires [out]".into()));
            }
            if b.dtype() != weight.dtype() {
                return Err(Error::DType);
            }
            if !d.owns(b.buffer()) {
                return Err(Error::DeviceMismatch);
            }
        }
        if !d.owns(weight.buffer()) {
            return Err(Error::DeviceMismatch);
        }
        Ok(Self {
            input: dims[1],
            output: dims[0],
            weight: MatrixWeight::Dense(weight),
            bias,
        })
    }
    pub(crate) fn new_quantized(
        d: &MetalDevice,
        weight: QuantizedMatrix,
        bias: Option<Tensor>,
        output_dtype: DType,
    ) -> Result<Self> {
        if weight.rows() == 0 || weight.columns() == 0 {
            return Err(Error::Shape(
                "quantized linear weight requires nonempty [out,in]".into(),
            ));
        }
        if let Some(b) = &bias {
            if b.shape().dimensions() != [weight.rows()] {
                return Err(Error::Shape("linear bias requires [out]".into()));
            }
            if b.dtype() != output_dtype {
                return Err(Error::DType);
            }
            if !d.owns(b.buffer()) {
                return Err(Error::DeviceMismatch);
            }
        }
        if !d.owns(weight.buffer()) {
            return Err(Error::DeviceMismatch);
        }
        Ok(Self {
            input: weight.columns(),
            output: weight.rows(),
            weight: MatrixWeight::Quantized(weight),
            bias,
        })
    }
    pub fn weight_bytes(&self) -> usize {
        self.weight.byte_size() + self.bias.as_ref().map_or(0, Tensor::byte_size)
    }
    /// The projection without its bias, for callers that fold the bias into
    /// a following kernel; returns the same `[.., out]` shape as `forward`.
    pub(crate) fn forward_unbiased(&self, d: &MetalDevice, x: &Tensor) -> Result<Tensor> {
        let dims = x.shape().dimensions();
        if dims.is_empty() || dims.last() != Some(&self.input) {
            return Err(Error::Shape(
                "linear input last dimension differs from weight".into(),
            ));
        }
        let mut out_dims = dims.to_vec();
        *out_dims
            .last_mut()
            .ok_or_else(|| Error::Shape("linear rank".into()))? = self.output;
        let x = x.reshape([x.numel() / self.input, self.input])?;
        let projected = match &self.weight {
            MatrixWeight::Dense(weight) => d.project(&x, weight)?,
            MatrixWeight::Quantized(weight) => d.project_quantized(&x, weight)?,
        };
        projected.tensor.reshape(out_dims)
    }
    pub(crate) fn bias(&self) -> Option<&Tensor> {
        self.bias.as_ref()
    }
    pub fn forward(&self, d: &MetalDevice, x: &Tensor) -> Result<Tensor> {
        let dims = x.shape().dimensions();
        if dims.is_empty() || dims.last() != Some(&self.input) {
            return Err(Error::Shape(
                "linear input last dimension differs from weight".into(),
            ));
        }
        let mut out_dims = dims.to_vec();
        *out_dims
            .last_mut()
            .ok_or_else(|| Error::Shape("linear rank".into()))? = self.output;
        let x = x.reshape([x.numel() / self.input, self.input])?;
        let projected = match &self.weight {
            MatrixWeight::Dense(weight) => d.project(&x, weight)?,
            MatrixWeight::Quantized(weight) => d.project_quantized(&x, weight)?,
        };
        let mut y = projected.tensor;
        if let Some(b) = &self.bias {
            y = d.bias_add(&y, b)?.tensor;
        }
        y.reshape(out_dims)
    }
}
pub struct RmsNorm {
    pub weight: Tensor,
    pub epsilon: f32,
}
impl RmsNorm {
    pub fn forward(&self, d: &MetalDevice, x: &Tensor) -> Result<Tensor> {
        Ok(d.rmsnorm(x, &self.weight, self.epsilon)?.tensor)
    }
}
pub struct Mlp {
    pub gate: Linear,
    pub up: Linear,
    pub down: Linear,
}
impl Mlp {
    pub fn forward(&self, d: &MetalDevice, x: &Tensor) -> Result<Tensor> {
        if let (
            MatrixWeight::Quantized(gate),
            MatrixWeight::Quantized(up),
            None,
            None,
            [.., last],
        ) = (
            &self.gate.weight,
            &self.up.weight,
            &self.gate.bias,
            &self.up.bias,
            x.shape().dimensions(),
        ) {
            let flat = x.reshape([x.numel() / last, *last])?;
            if d.can_fuse_swiglu(&flat, gate, up) {
                let mut dims = x.shape().dimensions().to_vec();
                if let Some(last) = dims.last_mut() {
                    *last = self.gate.output;
                }
                let hidden = d.profile_projection("gate_up_proj", || {
                    d.swiglu_q5_0(&flat, gate, up)?.tensor.reshape(dims)
                })?;
                return d.profile_projection("down_proj", || self.down.forward(d, &hidden));
            }
        }
        let gate = d.profile_projection("gate_proj", || self.gate.forward(d, x))?;
        let up = d.profile_projection("up_proj", || self.up.forward(d, x))?;
        let hidden = d.silu_mul(&gate, &up)?.tensor;
        d.profile_projection("down_proj", || self.down.forward(d, &hidden))
    }
}
