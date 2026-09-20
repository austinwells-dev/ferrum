#![forbid(unsafe_code)]
pub mod attention;
pub mod kv_cache;
use crate::{Error, MetalDevice, Result, Tensor};

pub struct Embedding {
    weight: Tensor,
}
impl Embedding {
    pub fn new(weight: Tensor) -> Result<Self> {
        if weight.shape().rank() != 2 || weight.shape().dimensions().contains(&0) {
            return Err(Error::Shape(
                "embedding requires nonempty [vocab,hidden]".into(),
            ));
        }
        Ok(Self { weight })
    }
    pub fn weight_bytes(&self) -> usize {
        self.weight.byte_size()
    }
    pub fn forward(&self, d: &MetalDevice, tokens: &[u32]) -> Result<Tensor> {
        Ok(d.embedding_gather(&self.weight, tokens)?.tensor)
    }
}
pub struct Linear {
    weight: Tensor,
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
            weight,
            bias,
        })
    }
    pub fn weight_bytes(&self) -> usize {
        self.weight.byte_size() + self.bias.as_ref().map_or(0, Tensor::byte_size)
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
        let mut y = d.project(&x, &self.weight)?.tensor;
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
        let gate = self.gate.forward(d, x)?;
        let up = self.up.forward(d, x)?;
        let hidden = d.mul(&d.silu(&gate)?.tensor, &up)?.tensor;
        self.down.forward(d, &hidden)
    }
}
