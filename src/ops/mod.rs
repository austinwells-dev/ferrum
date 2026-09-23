#![forbid(unsafe_code)]
use crate::{
    DType, Error, MetalDevice, Result, Tensor, metal::DispatchTiming, quantization::QuantizedMatrix,
};
#[derive(Debug, Clone)]
pub struct Metrics {
    pub operation: &'static str,
    pub shape: crate::tensor::Shape,
    pub dtype: DType,
    pub bytes_read: usize,
    pub bytes_written: usize,
    pub allocation_bytes: usize,
    pub timing: DispatchTiming,
}
pub struct Output {
    pub tensor: Tensor,
    pub metrics: Metrics,
}
impl MetalDevice {
    pub(crate) fn run_quantized(
        &self,
        name: &'static str,
        a: &Tensor,
        weight: &QuantizedMatrix,
        dims: &[usize],
        mut p: [u32; 9],
        grid: [usize; 2],
    ) -> Result<Output> {
        let profile_start = self.profiling().then(std::time::Instant::now);
        let wait_before = profile_start
            .map(|_| self.counters().wait)
            .unwrap_or_default();
        if !self.owns(a.buffer()) || !self.owns(weight.buffer()) {
            return Err(Error::DeviceMismatch);
        }
        index(a.numel())?;
        let shape = crate::tensor::Shape::new(dims)?;
        index(shape.numel())?;
        let allocation_start = profile_start.map(|_| std::time::Instant::now());
        let tensor = if shape.numel() == 0 {
            Tensor::zeros(self, dims, a.dtype())?
        } else {
            Tensor::output(self, dims, a.dtype())?
        };
        let allocation_time = allocation_start.map(|s| s.elapsed()).unwrap_or_default();
        p[0] = index(a.numel())?;
        p[4] = a.dtype() as u32;
        let timing = self.dispatch(
            name,
            &[a.binding(), weight.binding(), tensor.binding()],
            &p,
            grid,
            false,
        )?;
        let metrics = Metrics {
            operation: name,
            shape,
            dtype: a.dtype(),
            bytes_read: a.byte_size() + weight.byte_size(),
            bytes_written: tensor.byte_size(),
            allocation_bytes: tensor.storage_info().allocation_bytes,
            timing,
        };
        if let Some(start) = profile_start {
            self.record_profile(
                name,
                start
                    .elapsed()
                    .saturating_sub(self.counters().wait - wait_before),
                allocation_time,
                metrics.allocation_bytes,
                &metrics.timing,
            );
        }
        Ok(Output { tensor, metrics })
    }

    pub(crate) fn run_quantized_embedding(
        &self,
        weight: &QuantizedMatrix,
        ids: &Tensor,
        output_dims: &[usize],
        output_dtype: DType,
        mut p: [u32; 9],
    ) -> Result<Output> {
        let name = match weight.format() {
            crate::quantization::QuantizationFormat::Q4_0 => "embedding_gather_q4_0",
            crate::quantization::QuantizationFormat::Q5_0 => "embedding_gather_q5_0",
            crate::quantization::QuantizationFormat::Q5_1 => "embedding_gather_q5_1",
            crate::quantization::QuantizationFormat::Q4_K => "embedding_gather_q4_k",
            crate::quantization::QuantizationFormat::Q5_K => "embedding_gather_q5_k",
            crate::quantization::QuantizationFormat::Q8_0 => "embedding_gather_q8_0",
            crate::quantization::QuantizationFormat::Q6_K => "embedding_gather_q6_k",
            crate::quantization::QuantizationFormat::MlxAffine4Group64 => {
                "embedding_gather_mlx_affine4"
            }
        };
        let profile_start = self.profiling().then(std::time::Instant::now);
        let wait_before = profile_start
            .map(|_| self.counters().wait)
            .unwrap_or_default();
        if !self.owns(ids.buffer()) || !self.owns(weight.buffer()) {
            return Err(Error::DeviceMismatch);
        }
        let shape = crate::tensor::Shape::new(output_dims)?;
        index(shape.numel())?;
        let allocation_start = profile_start.map(|_| std::time::Instant::now());
        let tensor = if shape.numel() == 0 {
            Tensor::zeros(self, output_dims, output_dtype)?
        } else {
            Tensor::output(self, output_dims, output_dtype)?
        };
        let allocation_time = allocation_start.map(|s| s.elapsed()).unwrap_or_default();
        p[0] = index(shape.numel())?;
        p[4] = output_dtype as u32;
        let timing = self.dispatch(
            name,
            &[weight.binding(), ids.binding(), tensor.binding()],
            &p,
            [shape.numel(), 1],
            false,
        )?;
        let metrics = Metrics {
            operation: name,
            shape,
            dtype: output_dtype,
            bytes_read: ids.byte_size() + weight.byte_size(),
            bytes_written: tensor.byte_size(),
            allocation_bytes: tensor.storage_info().allocation_bytes,
            timing,
        };
        if let Some(start) = profile_start {
            self.record_profile(
                name,
                start
                    .elapsed()
                    .saturating_sub(self.counters().wait - wait_before),
                allocation_time,
                metrics.allocation_bytes,
                &metrics.timing,
            );
        }
        Ok(Output { tensor, metrics })
    }

    fn run(
        &self,
        name: &'static str,
        a: &Tensor,
        b: Option<&Tensor>,
        dims: &[usize],
        mut p: [u32; 9],
        grid: [usize; 2],
    ) -> Result<Output> {
        let profile_start = self.profiling().then(std::time::Instant::now);
        let wait_before = profile_start
            .map(|_| self.counters().wait)
            .unwrap_or_default();
        if !self.owns(a.buffer()) || b.is_some_and(|b| !self.owns(b.buffer())) {
            return Err(Error::DeviceMismatch);
        }
        if b.is_some_and(|b| b.dtype() != a.dtype()) {
            return Err(Error::DType);
        }
        index(a.numel())?;
        if let Some(b) = b {
            index(b.numel())?;
        }
        let shape = crate::tensor::Shape::new(dims)?;
        index(shape.numel())?;
        let allocation_start = profile_start.map(|_| std::time::Instant::now());
        let tensor = if shape.numel() == 0 {
            Tensor::zeros(self, dims, a.dtype())?
        } else {
            Tensor::output(self, dims, a.dtype())?
        };
        let allocation_time = allocation_start.map(|s| s.elapsed()).unwrap_or_default();
        p[0] = index(a.numel())?;
        p[4] = a.dtype() as u32;
        let timing = self.dispatch(
            name,
            &[a.binding(), b.unwrap_or(a).binding(), tensor.binding()],
            &p,
            grid,
            matches!(name, "matmul" | "matmul_nt"),
        )?;
        let metrics = Metrics {
            operation: name,
            shape,
            dtype: a.dtype(),
            bytes_read: a.byte_size() + b.map_or(0, Tensor::byte_size),
            bytes_written: tensor.byte_size(),
            allocation_bytes: tensor.storage_info().allocation_bytes,
            timing,
        };
        if let Some(start) = profile_start {
            self.record_profile(
                name,
                start
                    .elapsed()
                    .saturating_sub(self.counters().wait - wait_before),
                allocation_time,
                metrics.allocation_bytes,
                &metrics.timing,
            );
        }
        Ok(Output { tensor, metrics })
    }
    fn binary(&self, name: &'static str, a: &Tensor, b: &Tensor) -> Result<Output> {
        if a.shape() != b.shape() {
            return Err(Error::Shape("elementwise shapes must match exactly".into()));
        }
        self.run(
            name,
            a,
            Some(b),
            a.shape().dimensions(),
            [0; 9],
            [a.numel(), 1],
        )
    }
    pub fn add(&self, a: &Tensor, b: &Tensor) -> Result<Output> {
        self.binary("add", a, b)
    }
    pub fn mul(&self, a: &Tensor, b: &Tensor) -> Result<Output> {
        self.binary("mul", a, b)
    }
    pub(crate) fn silu_mul(&self, a: &Tensor, b: &Tensor) -> Result<Output> {
        self.binary("silu_mul", a, b)
    }
    pub fn silu(&self, a: &Tensor) -> Result<Output> {
        self.run(
            "silu",
            a,
            None,
            a.shape().dimensions(),
            [0; 9],
            [a.numel(), 1],
        )
    }
    pub fn rmsnorm(&self, a: &Tensor, weight: &Tensor, eps: f32) -> Result<Output> {
        let w = width(a)?;
        if w > u32::MAX as usize - 256 {
            return Err(Error::Shape("reduction width exceeds index limit".into()));
        }
        if weight.shape().dimensions() != [w] {
            return Err(Error::Shape(
                "RMSNorm weight must match the last dimension".into(),
            ));
        }
        if !eps.is_finite() || eps <= 0. {
            return Err(Error::Parameter(
                "epsilon must be finite and positive".into(),
            ));
        }
        let mut p = [0; 9];
        p[1] = index(w)?;
        p[7] = eps.to_bits();
        self.run(
            if self.reference_math() && a.dtype() == DType::F32 {
                "rmsnorm_ordered"
            } else {
                "rmsnorm"
            },
            a,
            Some(weight),
            a.shape().dimensions(),
            p,
            [a.numel() / w, 1],
        )
    }
    pub fn softmax(&self, a: &Tensor) -> Result<Output> {
        let w = width(a)?;
        if w > u32::MAX as usize - 256 {
            return Err(Error::Shape("reduction width exceeds index limit".into()));
        }
        let mut p = [0; 9];
        p[1] = index(w)?;
        self.run(
            if self.reference_math() && a.dtype() == DType::F32 {
                "softmax_ordered"
            } else {
                "softmax"
            },
            a,
            None,
            a.shape().dimensions(),
            p,
            [a.numel() / w, 1],
        )
    }
    pub fn rope(&self, a: &Tensor, head_dim: usize, position: u32, theta: f32) -> Result<Output> {
        let w = width(a)?;
        if w > u32::MAX as usize - 256 {
            return Err(Error::Shape("reduction width exceeds index limit".into()));
        }
        if head_dim == 0 || !head_dim.is_multiple_of(2) || !w.is_multiple_of(head_dim) {
            return Err(Error::Parameter(
                "head dimension must be positive, even and divide the last axis".into(),
            ));
        }
        if !theta.is_finite() || theta <= 0. {
            return Err(Error::Parameter("theta must be finite and positive".into()));
        }
        let mut p = [0; 9];
        p[5] = position;
        p[6] = index(head_dim)?;
        p[8] = theta.to_bits();
        self.run(
            "rope",
            a,
            None,
            a.shape().dimensions(),
            p,
            [a.numel() / 2, 1],
        )
    }
    pub fn matmul(&self, a: &Tensor, b: &Tensor) -> Result<Output> {
        let ad = a.shape().dimensions();
        let bd = b.shape().dimensions();
        if ad.len() != 2 || bd.len() != 2 || ad[1] != bd[0] {
            return Err(Error::Shape("matmul requires [M,K] and [K,N]".into()));
        }
        let mut p = [0; 9];
        p[1] = index(ad[0])?;
        p[2] = index(ad[1])?;
        p[3] = index(bd[1])?;
        // The tiled loop increments K by 16; prevent uint wrap in the final increment.
        if ad[1] > u32::MAX as usize - 16 {
            return Err(Error::Shape("K exceeds tiled kernel index limit".into()));
        }
        self.run("matmul", a, Some(b), &[ad[0], bd[1]], p, [bd[1], ad[0]])
    }
    pub fn warm_up(&self) -> Result<()> {
        for name in [
            "add",
            "mul",
            "silu",
            "rmsnorm",
            "softmax",
            "rope",
            "matmul",
            "q4_0_gemv",
            "q4_0_gemm",
            "q5_0_gemv",
            "q5_0_gemm",
            "q5_0_gemm_mpp",
            "q5_1_gemv",
            "q5_1_gemm",
            "q5_1_gemm_mpp",
            "q4_k_gemv",
            "q4_k_gemm",
            "q4_k_gemm_mpp",
            "q5_k_gemv",
            "q5_k_gemm",
            "q5_k_gemm_mpp",
            "q6_k_gemm_mpp",
            "q6_k_gemv",
            "q6_k_gemm",
            "q8_0_gemv",
            "q8_0_gemm",
            "mlx_affine4_gemv",
            "mlx_affine4_gemm",
            "mlx_affine4_gemm_mpp",
            "embedding_gather_q4_0",
            "embedding_gather_q5_0",
            "embedding_gather_q5_1",
            "embedding_gather_q4_k",
            "embedding_gather_q5_k",
            "embedding_gather_q6_k",
            "embedding_gather_q8_0",
            "embedding_gather_mlx_affine4",
        ] {
            self.builtin(name)?;
        }
        Ok(())
    }
}
fn width(a: &Tensor) -> Result<usize> {
    a.shape()
        .dimensions()
        .last()
        .copied()
        .filter(|&w| w > 0)
        .ok_or_else(|| Error::Shape("operation needs a nonempty last axis".into()))
}
fn index(n: usize) -> Result<u32> {
    u32::try_from(n).map_err(|_| Error::Shape("kernel index exceeds u32".into()))
}

mod transformer;
