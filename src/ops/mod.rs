#![forbid(unsafe_code)]
use crate::{DType, Error, MetalDevice, Result, Tensor, metal::DispatchTiming};
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
        let tensor = Tensor::zeros(self, dims, a.dtype())?;
        let allocation_time = allocation_start.map(|s| s.elapsed()).unwrap_or_default();
        p[0] = index(a.numel())?;
        p[4] = a.dtype() as u32;
        let timing = self.dispatch(
            name,
            &[a.buffer(), b.unwrap_or(a).buffer(), tensor.buffer()],
            &p,
            grid,
            name == "matmul",
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
                start.elapsed(),
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
            "rmsnorm",
            a,
            Some(weight),
            a.shape().dimensions(),
            p,
            [a.numel() / w, 1],
        )
    }
    pub fn softmax(&self, a: &Tensor) -> Result<Output> {
        let w = width(a)?;
        let mut p = [0; 9];
        p[1] = index(w)?;
        self.run(
            "softmax",
            a,
            None,
            a.shape().dimensions(),
            p,
            [a.numel() / w, 1],
        )
    }
    pub fn rope(&self, a: &Tensor, head_dim: usize, position: u32, theta: f32) -> Result<Output> {
        let w = width(a)?;
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
        for name in ["add", "mul", "silu", "rmsnorm", "softmax", "rope", "matmul"] {
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
