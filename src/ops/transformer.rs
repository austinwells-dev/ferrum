//! Narrow contiguous copies and transformer operations; no strided views.
use super::*;
/// Dense GGUF TensorOps kernels that accept the paired-M-tile flag.
const DENSE_MPP_PAIRED: &[&str] = &[
    "q4_0_gemm_mpp",
    "q5_0_gemm_mpp",
    "q5_1_gemm_mpp",
    "q5_1_gemm_mpp_k64",
    "q4_k_gemm_mpp",
    "q4_k_gemm_mpp_k64",
    "q5_k_gemm_mpp",
    "q5_k_gemm_mpp_k64",
    "q6_k_gemm_mpp",
    "q8_0_gemm_mpp",
    "q8_0_gemm_mpp_k64",
];

impl MetalDevice {
    /// Whether `swiglu_q5_0` can replace gate, up, and `silu_mul` for `x`.
    pub(crate) fn can_fuse_swiglu(
        &self,
        x: &Tensor,
        gate: &crate::quantization::QuantizedMatrix,
        up: &crate::quantization::QuantizedMatrix,
    ) -> bool {
        use crate::quantization::QuantizationFormat;
        let dims = x.shape().dimensions();
        self.fuse_swiglu()
            && self.q5_0_ksplit() > 0
            && !self.reference_math()
            && dims.len() == 2
            && dims[0] == 1
            && dims[1] == gate.columns()
            && dims[1] <= u32::MAX as usize - 32
            && gate.format() == QuantizationFormat::Q5_0
            && up.format() == QuantizationFormat::Q5_0
            && gate.rows() == up.rows()
            && gate.columns() == up.columns()
    }
    /// Decode SwiGLU: `silu(x Gᵀ) * (x Uᵀ)` for Q5_0 gate/up in one kernel,
    /// bit-identical to the K-split projections followed by `silu_mul`.
    pub(crate) fn swiglu_q5_0(
        &self,
        x: &Tensor,
        gate: &crate::quantization::QuantizedMatrix,
        up: &crate::quantization::QuantizedMatrix,
    ) -> Result<Output> {
        if !self.can_fuse_swiglu(x, gate, up) {
            return Err(Error::Shape("fused SwiGLU operands do not match".into()));
        }
        if !self.owns(x.buffer()) || !self.owns(gate.buffer()) || !self.owns(up.buffer()) {
            return Err(Error::DeviceMismatch);
        }
        let profile_start = self.profiling().then(std::time::Instant::now);
        let wait_before = profile_start
            .map(|_| self.counters().wait)
            .unwrap_or_default();
        let dims = [1, gate.rows()];
        let tensor = Tensor::output(self, &dims, x.dtype())?;
        let mut p = [0; 9];
        p[0] = index(x.numel())?;
        p[1] = 1;
        p[2] = index(gate.columns())?;
        p[3] = index(gate.rows())?;
        p[4] = x.dtype() as u32;
        p[8] = index(self.q5_0_ksplit())?;
        let timing = self.dispatch(
            "q5_0_gemv_swiglu",
            &[x.binding(), gate.binding(), up.binding(), tensor.binding()],
            &p,
            [gate.rows(), 1],
            false,
        )?;
        let metrics = Metrics {
            operation: "q5_0_gemv_swiglu",
            shape: crate::tensor::Shape::new(dims)?,
            dtype: x.dtype(),
            bytes_read: x.byte_size() + gate.byte_size() + up.byte_size(),
            bytes_written: tensor.byte_size(),
            allocation_bytes: tensor.storage_info().allocation_bytes,
            timing,
        };
        if let Some(start) = profile_start {
            self.record_profile(
                "q5_0_gemv_swiglu",
                start
                    .elapsed()
                    .saturating_sub(self.counters().wait - wait_before),
                std::time::Duration::ZERO,
                metrics.allocation_bytes,
                &metrics.timing,
            );
        }
        Ok(Output { tensor, metrics })
    }
    pub(crate) fn project_quantized(
        &self,
        a: &Tensor,
        weight: &crate::quantization::QuantizedMatrix,
    ) -> Result<Output> {
        use crate::quantization::QuantizationFormat;
        let ad = a.shape().dimensions();
        if ad.len() != 2 || ad[1] != weight.columns() || ad[1] > u32::MAX as usize - 32 {
            return Err(Error::Shape(
                "quantized projection requires [M,K] matching [N,K]".into(),
            ));
        }
        let mut p = [0; 9];
        p[1] = index(ad[0])?;
        p[2] = index(ad[1])?;
        p[3] = index(weight.rows())?;
        let automatic_min_mpp_rows = match weight.format() {
            // M5 crossover points from paired MSL/MPP measurements.
            QuantizationFormat::Q4_0 | QuantizationFormat::Q4_K => 4,
            QuantizationFormat::Q5_K | QuantizationFormat::Q6_K => 8,
            QuantizationFormat::Q8_0 => 12,
            // Q5_1 MPP wins from M=8; Q5_0 remains at its unmeasured boundary.
            QuantizationFormat::Q5_1 => 8,
            QuantizationFormat::Q5_0 | QuantizationFormat::MlxAffine4Group64 => 16,
        };
        let min_mpp_rows = if weight.format() == QuantizationFormat::MlxAffine4Group64 {
            16
        } else {
            let q5_1_override = (weight.format() == QuantizationFormat::Q5_1)
                .then(|| self.gguf_mpp_q5_1_min_rows())
                .flatten();
            q5_1_override
                .or_else(|| self.gguf_mpp_min_rows())
                .unwrap_or(automatic_min_mpp_rows)
        };
        let supports_mpp = ad[0] >= min_mpp_rows
            && ((weight.format() == QuantizationFormat::MlxAffine4Group64
                && a.dtype() == DType::F16)
                || (weight.format() != QuantizationFormat::MlxAffine4Group64
                    && a.dtype() == DType::BF16))
            && ad[1].is_multiple_of(128)
            && self.mpp_projection();
        let name = match (weight.format(), ad[0] == 1, supports_mpp) {
            (QuantizationFormat::Q8_0, true, _) => {
                if self.q8_0_gemv_k_split(weight.rows(), weight.columns()) {
                    match self.q8_0_gemv_rows() {
                        8 => "q8_0_gemv_k_split8",
                        4 => "q8_0_gemv_k_split4",
                        _ => "q8_0_gemv_k_split",
                    }
                } else if self.q8_0_gemv_8rows(weight.rows()) {
                    "q8_0_gemv_8rows"
                } else {
                    "q8_0_gemv"
                }
            }
            (QuantizationFormat::Q8_0, false, true) => {
                if self.q8_0_mpp_tile_k64(ad[0]) {
                    "q8_0_gemm_mpp_k64"
                } else {
                    "q8_0_gemm_mpp"
                }
            }
            (QuantizationFormat::Q8_0, false, false) => "q8_0_gemm",
            (QuantizationFormat::Q4_0, true, _) => {
                if self.use_q4_0_gemv_8rows(weight.rows()) {
                    "q4_0_gemv_8rows"
                } else {
                    "q4_0_gemv"
                }
            }
            (QuantizationFormat::Q4_0, false, true) => "q4_0_gemm_mpp",
            (QuantizationFormat::Q4_0, false, false) => "q4_0_gemm",
            (QuantizationFormat::Q5_0, true, _) => {
                if self.q5_0_ksplit() > 0 {
                    "q5_0_gemv_ksplit"
                } else if self.use_q5_0_gemv_n4(weight.rows()) {
                    "q5_0_gemv_n4"
                } else {
                    "q5_0_gemv"
                }
            }
            (QuantizationFormat::Q5_0, false, true) => "q5_0_gemm_mpp",
            (QuantizationFormat::Q5_0, false, false) => "q5_0_gemm",
            (QuantizationFormat::Q5_1, true, _) => {
                if self.use_q5_1_gemv_n4(weight.rows()) {
                    "q5_1_gemv_n4"
                } else {
                    "q5_1_gemv"
                }
            }
            (QuantizationFormat::Q5_1, false, true) => {
                if self.q5_1_mpp_tile_k64(ad[0]) {
                    "q5_1_gemm_mpp_k64"
                } else {
                    "q5_1_gemm_mpp"
                }
            }
            (QuantizationFormat::Q5_1, false, false) => "q5_1_gemm",
            (QuantizationFormat::Q4_K, true, _) => {
                if self.use_q4_k_factored(weight.columns()) && self.k_quant_gemv_ksplit() > 0 {
                    "q4_k_gemv_ksplit"
                } else if self.use_q4_k_factored(weight.columns()) {
                    "q4_k_gemv_factored"
                } else if self.use_q4_k_gemv_8rows(weight.rows()) {
                    "q4_k_gemv_8rows"
                } else {
                    "q4_k_gemv"
                }
            }
            (QuantizationFormat::Q4_K, false, true) => {
                if self.q4_k_mpp_tile_m128(ad[0]) {
                    "q4_k_gemm_mpp_k64_m128"
                } else if self.q4_k_mpp_tile_k64(ad[0]) {
                    "q4_k_gemm_mpp_k64"
                } else {
                    "q4_k_gemm_mpp"
                }
            }
            (QuantizationFormat::Q4_K, false, false) => "q4_k_gemm",
            (QuantizationFormat::Q5_K, true, _) => {
                if self.use_q5_k_gemv_8rows(weight.rows()) {
                    "q5_k_gemv_8rows"
                } else {
                    "q5_k_gemv"
                }
            }
            (QuantizationFormat::Q5_K, false, true) => {
                if self.q5_k_mpp_tile_k64(ad[0]) {
                    "q5_k_gemm_mpp_k64"
                } else {
                    "q5_k_gemm_mpp"
                }
            }
            (QuantizationFormat::Q5_K, false, false) => "q5_k_gemm",
            (QuantizationFormat::Q6_K, true, _) => {
                if self.use_q6_k_factored(weight.columns()) && self.k_quant_gemv_ksplit() > 0 {
                    "q6_k_gemv_ksplit"
                } else if self.use_q6_k_factored(weight.columns()) {
                    "q6_k_gemv_factored"
                } else if self.use_q6_k_gemv_8rows(weight.rows()) {
                    "q6_k_gemv_8rows"
                } else {
                    "q6_k_gemv"
                }
            }
            (QuantizationFormat::Q6_K, false, true) => "q6_k_gemm_mpp",
            (QuantizationFormat::Q6_K, false, false) => "q6_k_gemm",
            (QuantizationFormat::MlxAffine4Group64, true, _) => {
                if self.use_mlx_affine4_gemv_quad(weight.rows(), weight.columns()) {
                    "mlx_affine4_gemv_quad"
                } else {
                    "mlx_affine4_gemv"
                }
            }
            (QuantizationFormat::MlxAffine4Group64, false, true) => {
                if self.mlx_affine4_mpp_tile_k64(ad[0]) {
                    "mlx_affine4_gemm_mpp_k64"
                } else {
                    "mlx_affine4_gemm_mpp"
                }
            }
            (QuantizationFormat::MlxAffine4Group64, false, false) => "mlx_affine4_gemm",
        };
        match name {
            "q5_0_gemv_ksplit" => p[8] = index(self.q5_0_ksplit())?,
            "q4_k_gemv_ksplit" | "q6_k_gemv_ksplit" => {
                // Q4_K lanes cover four super-blocks per pass, Q6_K two.
                let per_pass = if name == "q4_k_gemv_ksplit" { 4 } else { 2 };
                let passes = (weight.columns() / 256)
                    .div_ceil(per_pass)
                    .next_power_of_two();
                let cap = if weight.rows() >= 4096 { 2 } else { 8 };
                p[8] = index(passes.min(cap).min(self.k_quant_gemv_ksplit()))?;
            }
            _ => {}
        }
        if name.starts_with("q4_k_gemm_mpp")
            || name.starts_with("q8_0_gemm_mpp")
            || matches!(name, "q6_k_gemm_mpp" | "q5_0_gemm_mpp")
        {
            p[8] = u32::from(self.mpp_fast_dequant());
        }
        // Paired M tiles: one threadgroup covers two consecutive 64-row tiles.
        // Narrow outputs and prompts under 512 rows keep single tiles so enough
        // threadgroups stay resident (measured: 128 rows lose, 512+ gain).
        let pairs = self.dense_mpp_tile_pairs()
            && weight.rows() >= 512
            && ad[0] >= 512
            && DENSE_MPP_PAIRED.contains(&name);
        if pairs {
            p[8] |= 2;
        }
        self.run_quantized(
            name,
            a,
            weight,
            &[ad[0], weight.rows()],
            p,
            [weight.rows(), if pairs { ad[0].div_ceil(2) } else { ad[0] }],
        )
    }

    pub(crate) fn embedding_gather_quantized(
        &self,
        weight: &crate::quantization::QuantizedMatrix,
        tokens: &[u32],
        output_dtype: DType,
    ) -> Result<Output> {
        use crate::quantization::QuantizationFormat;
        if !matches!(
            weight.format(),
            QuantizationFormat::Q4_0
                | QuantizationFormat::Q5_0
                | QuantizationFormat::Q5_1
                | QuantizationFormat::Q4_K
                | QuantizationFormat::Q5_K
                | QuantizationFormat::Q8_0
                | QuantizationFormat::Q6_K
                | QuantizationFormat::MlxAffine4Group64
        ) || weight.columns() > u32::MAX as usize
        {
            return Err(Error::Gguf(format!(
                "no quantized embedding gather for {:?}",
                weight.format()
            )));
        }
        for &id in tokens {
            if id as usize >= weight.rows() {
                return Err(Error::Token {
                    id,
                    vocab: weight.rows(),
                });
            }
        }
        let ids = Tensor::from_le_bytes(
            self,
            [tokens.len()],
            DType::F32,
            &tokens
                .iter()
                .flat_map(|id| id.to_le_bytes())
                .collect::<Vec<_>>(),
        )?;
        if ids.storage_info().alignment < 4 {
            return Err(Error::Range(
                "token ID carrier requires u32 alignment".into(),
            ));
        }
        let mut p = [0; 9];
        p[1] = index(weight.columns())?;
        p[2] = index(weight.rows())?;
        self.run_quantized_embedding(
            weight,
            &ids,
            &[tokens.len(), weight.columns()],
            output_dtype,
            p,
        )
    }

    /// Integer IDs never pass through floating-point conversion.
    pub fn embedding_gather(&self, weight: &Tensor, tokens: &[u32]) -> Result<Output> {
        let dims = weight.shape().dimensions();
        if dims.len() != 2 || dims.contains(&0) {
            return Err(Error::Shape("embedding requires [vocab,hidden]".into()));
        }
        if !self.owns(weight.buffer()) {
            return Err(Error::DeviceMismatch);
        }
        for &id in tokens {
            if id as usize >= dims[0] {
                return Err(Error::Token { id, vocab: dims[0] });
            }
        }
        let shape = crate::tensor::Shape::new([tokens.len(), dims[1]])?;
        index(shape.numel())?;
        let bytes_len = tokens
            .len()
            .checked_mul(4)
            .ok_or_else(|| Error::Shape("token bytes overflow".into()))?;
        let bytes: Vec<u8> = tokens.iter().flat_map(|id| id.to_le_bytes()).collect();
        let ids = Tensor::from_le_bytes(
            self,
            [bytes_len / weight.dtype().size_bytes()],
            weight.dtype(),
            &bytes,
        )?;
        if ids.storage_info().alignment < 4 {
            return Err(Error::Range(
                "token ID carrier requires u32 alignment".into(),
            ));
        }
        let mut p = [0; 9];
        p[1] = index(dims[1])?;
        self.run(
            "embedding_gather",
            weight,
            Some(&ids),
            shape.dimensions(),
            p,
            [shape.numel(), 1],
        )
    }

    pub fn copy_range(&self, a: &Tensor, start: usize, dims: &[usize]) -> Result<Output> {
        let n = crate::tensor::Shape::new(dims)?.numel();
        if start.checked_add(n).is_none_or(|end| end > a.numel()) {
            return Err(Error::Range("copy exceeds input".into()));
        }
        let mut p = [0; 9];
        p[1] = index(start)?;
        self.run("copy_range", a, None, dims, p, [n, 1])
    }
    pub fn concat_first(&self, a: &Tensor, b: &Tensor) -> Result<Output> {
        let ad = a.shape().dimensions();
        let bd = b.shape().dimensions();
        if ad.is_empty() || ad.len() != bd.len() || ad[1..] != bd[1..] {
            return Err(Error::Shape(
                "concat requires matching trailing axes".into(),
            ));
        }
        let mut dims = ad.to_vec();
        dims[0] = ad[0]
            .checked_add(bd[0])
            .ok_or_else(|| Error::Shape("concat overflow".into()))?;
        let n = crate::tensor::Shape::new(&dims)?.numel();
        self.run("concat_flat", a, Some(b), &dims, [0; 9], [n, 1])
    }
    pub fn transpose2(&self, a: &Tensor) -> Result<Output> {
        let d = a.shape().dimensions();
        if d.len() != 2 {
            return Err(Error::Shape("transpose requires rank two".into()));
        }
        let mut p = [0; 9];
        p[1] = index(d[0])?;
        p[2] = index(d[1])?;
        self.run("transpose2", a, None, &[d[1], d[0]], p, [a.numel(), 1])
    }
    pub fn swap01(&self, a: &Tensor) -> Result<Output> {
        let d = a.shape().dimensions();
        if d.len() != 3 {
            return Err(Error::Shape("swap01 requires rank three".into()));
        }
        let mut p = [0; 9];
        for j in 0..3 {
            p[j + 1] = index(d[j])?;
        }
        self.run("swap01", a, None, &[d[1], d[0], d[2]], p, [a.numel(), 1])
    }
    pub fn select_head(&self, a: &Tensor, head: usize) -> Result<Output> {
        let d = a.shape().dimensions();
        if d.len() != 3 || head >= d[1] {
            return Err(Error::Shape("head index out of range for [S,H,D]".into()));
        }
        let mut p = [0; 9];
        p[1] = index(head)?;
        p[2] = index(d[1])?;
        p[3] = index(d[2])?;
        self.run("select_head", a, None, &[d[0], d[2]], p, [d[0] * d[2], 1])
    }
    pub fn bias_add(&self, a: &Tensor, bias: &Tensor) -> Result<Output> {
        let w = width(a)?;
        if bias.shape().dimensions() != [w] {
            return Err(Error::Shape("bias must match last dimension".into()));
        }
        let mut p = [0; 9];
        p[1] = index(w)?;
        self.run(
            "bias_add",
            a,
            Some(bias),
            a.shape().dimensions(),
            p,
            [a.numel(), 1],
        )
    }
    pub fn scale(&self, a: &Tensor, value: f32) -> Result<Output> {
        if !value.is_finite() {
            return Err(Error::Parameter("scale must be finite".into()));
        }
        let mut p = [0; 9];
        p[7] = value.to_bits();
        self.run("scale", a, None, a.shape().dimensions(), p, [a.numel(), 1])
    }
    pub fn causal_mask(&self, a: &Tensor, offset: usize) -> Result<Output> {
        let d = a.shape().dimensions();
        if d.len() != 2 || d[1] == 0 || offset.checked_add(d[0]) != Some(d[1]) {
            return Err(Error::Parameter(
                "causal scores require [M,offset+M]".into(),
            ));
        }
        let mut p = [0; 9];
        p[1] = index(d[1])?;
        p[5] = index(offset)?;
        self.run("causal_mask", a, None, d, p, [a.numel(), 1])
    }
    pub fn rope_split(&self, a: &Tensor, offset: usize, theta: f32) -> Result<Output> {
        let d = a.shape().dimensions();
        if d.len() != 3
            || d[1] == 0
            || d[2] == 0
            || !d[2].is_multiple_of(2)
            || !theta.is_finite()
            || theta <= 0.
        {
            return Err(Error::Parameter(
                "split RoPE requires [S,H,even D], positive heads and theta".into(),
            ));
        }
        index(
            offset
                .checked_add(d[0])
                .ok_or_else(|| Error::Parameter("position overflow".into()))?,
        )?;
        let mut p = [0; 9];
        p[1] = index(d[1])?;
        p[5] = index(offset)?;
        p[6] = index(d[2])?;
        p[8] = theta.to_bits();
        if self.batching_active() && self.rope_table_enabled() {
            let table = self.rope_table(a, offset, d[0], d[2], theta)?;
            return self.run(
                "rope_split_table",
                a,
                Some(&table),
                d,
                p,
                [a.numel() / 2, 1],
            );
        }
        self.run("rope_split", a, None, d, p, [a.numel() / 2, 1])
    }
    /// Table-driven RoPE into a caller-provided tensor (for example a KV-cache
    /// slot), optionally folding a projection bias in first. Bit-identical to
    /// `bias_add` followed by `rope_split` and a copy into `out`.
    pub(crate) fn rope_split_into(
        &self,
        a: &Tensor,
        bias: Option<&Tensor>,
        offset: usize,
        theta: f32,
        out: &Tensor,
    ) -> Result<()> {
        let d = a.shape().dimensions();
        if d.len() != 3 || d[2] == 0 || !d[2].is_multiple_of(2) || out.shape() != a.shape() {
            return Err(Error::Parameter(
                "RoPE destination must match [S,H,even D]".into(),
            ));
        }
        if out.dtype() != a.dtype()
            || bias
                .is_some_and(|b| b.dtype() != a.dtype() || b.shape().dimensions() != [d[1] * d[2]])
        {
            return Err(Error::DType);
        }
        let table = self.rope_table(a, offset, d[0], d[2], theta)?;
        let mut p = [0; 9];
        p[0] = index(a.numel())?;
        p[1] = index(d[1])?;
        p[4] = a.dtype() as u32;
        p[5] = index(offset)?;
        p[6] = index(d[2])?;
        p[8] = theta.to_bits();
        match bias {
            Some(bias) => self.dispatch(
                "rope_split_table_bias",
                &[a.binding(), table.binding(), bias.binding(), out.binding()],
                &p,
                [a.numel() / 2, 1],
                false,
            )?,
            None => self.dispatch(
                "rope_split_table",
                &[a.binding(), table.binding(), out.binding()],
                &p,
                [a.numel() / 2, 1],
                false,
            )?,
        };
        Ok(())
    }
    /// `bias_add` writing into a caller-provided tensor (a KV-cache slot).
    pub(crate) fn bias_add_into(&self, a: &Tensor, bias: &Tensor, out: &Tensor) -> Result<()> {
        let w = width(a)?;
        if bias.shape().dimensions() != [w] || out.numel() != a.numel() || out.dtype() != a.dtype()
        {
            return Err(Error::Shape("bias must match last dimension".into()));
        }
        let mut p = [0; 9];
        p[0] = index(a.numel())?;
        p[1] = index(w)?;
        p[4] = a.dtype() as u32;
        self.dispatch(
            "bias_add",
            &[a.binding(), bias.binding(), out.binding()],
            &p,
            [a.numel(), 1],
            false,
        )?;
        Ok(())
    }
    /// Cos/sin table for positions `offset..offset+rows`, built once per
    /// execution and shared by every q/k rotation with the same geometry.
    fn rope_table(
        &self,
        carrier: &Tensor,
        offset: usize,
        rows: usize,
        head_dim: usize,
        theta: f32,
    ) -> Result<Tensor> {
        let key = (offset, rows, head_dim, theta.to_bits());
        if let Some((cached, table)) = self.rope_table_cache().borrow().as_ref()
            && *cached == key
        {
            return Ok(table.clone());
        }
        let entries = rows
            .checked_mul(head_dim / 2)
            .ok_or_else(|| Error::Shape("RoPE table size overflow".into()))?;
        let table = Tensor::output(self, &[entries, 2], DType::F32)?;
        let mut p = [0; 9];
        p[0] = index(entries)?;
        p[5] = index(offset)?;
        p[6] = index(head_dim)?;
        p[8] = theta.to_bits();
        self.dispatch(
            "rope_table",
            &[carrier.binding(), carrier.binding(), table.binding()],
            &p,
            [entries, 1],
            false,
        )
        .map(|_| ())?;
        *self.rope_table_cache().borrow_mut() = Some((key, table.clone()));
        Ok(table)
    }

    /// Causal depthwise convolution used by LFM2 short-convolution blocks.
    /// `bx` is the projected B*X activation, while `gate` is C. The convolution
    /// result is rounded to the activation dtype before multiplying C, matching
    /// the two storage boundaries in the upstream implementation. State output
    /// is a fresh immutable tensor so cache snapshots remain branchable.
    pub(crate) fn lfm2_short_conv(
        &self,
        bx: &Tensor,
        gate: &Tensor,
        weight: &Tensor,
        previous_state: &Tensor,
        kernel_size: usize,
    ) -> Result<(Tensor, Tensor)> {
        let input_dims = bx.shape().dimensions();
        if input_dims.len() != 2 || input_dims.contains(&0) {
            return Err(Error::Shape("short convolution requires [S,H]".into()));
        }
        let (sequence, hidden) = (input_dims[0], input_dims[1]);
        if gate.shape() != bx.shape()
            || weight.shape().dimensions() != [hidden, 1, kernel_size]
            || previous_state.shape().dimensions() != [kernel_size, hidden]
            || kernel_size == 0
        {
            return Err(Error::Shape(
                "short convolution weight, gate, or state shape mismatch".into(),
            ));
        }
        if [gate, weight, previous_state]
            .iter()
            .any(|tensor| tensor.dtype() != bx.dtype())
        {
            return Err(Error::DType);
        }
        if [bx, gate, weight, previous_state]
            .iter()
            .any(|tensor| !self.owns(tensor.buffer()))
        {
            return Err(Error::DeviceMismatch);
        }

        let output = Tensor::output(self, &[sequence, hidden], bx.dtype())?;
        let next_state = Tensor::output(self, &[kernel_size, hidden], bx.dtype())?;
        let output_elements = sequence
            .checked_mul(hidden)
            .ok_or_else(|| Error::Shape("short convolution output size overflow".into()))?;
        let state_elements = kernel_size
            .checked_mul(hidden)
            .ok_or_else(|| Error::Shape("short convolution state size overflow".into()))?;
        let mut p = [0; 9];
        p[0] = index(output_elements)?;
        p[1] = index(hidden)?;
        p[2] = index(kernel_size)?;
        p[3] = index(sequence)?;
        p[4] = bx.dtype() as u32;
        let start = self.profiling().then(std::time::Instant::now);
        let timing = self.dispatch(
            "lfm2_short_conv",
            &[
                bx.binding(),
                gate.binding(),
                weight.binding(),
                previous_state.binding(),
                output.binding(),
                next_state.binding(),
            ],
            &p,
            [output_elements.max(state_elements), 1],
            false,
        )?;
        if let Some(start) = start {
            self.record_profile(
                "short_conv",
                start
                    .elapsed()
                    .saturating_sub(timing.synchronized.saturating_sub(timing.submission)),
                std::time::Duration::ZERO,
                output.byte_size() + next_state.byte_size(),
                &timing,
            );
        }
        Ok((output, next_state))
    }

    /// Split a row-major `[S,3H]` LFM2 input projection into contiguous B, C,
    /// and X tensors. The three channel groups are adjacent within each token
    /// row, so they cannot be represented as three contiguous tensor views.
    pub(crate) fn lfm2_split3(&self, projected: &Tensor) -> Result<(Tensor, Tensor, Tensor)> {
        let dims = projected.shape().dimensions();
        if dims.len() != 2 || dims.contains(&0) || !dims[1].is_multiple_of(3) {
            return Err(Error::Shape("LFM2 projection must be [S,3H]".into()));
        }
        if !self.owns(projected.buffer()) {
            return Err(Error::DeviceMismatch);
        }
        let (sequence, hidden) = (dims[0], dims[1] / 3);
        let elements = sequence
            .checked_mul(hidden)
            .ok_or_else(|| Error::Shape("LFM2 split size overflow".into()))?;
        let first = Tensor::output(self, &[sequence, hidden], projected.dtype())?;
        let second = Tensor::output(self, &[sequence, hidden], projected.dtype())?;
        let third = Tensor::output(self, &[sequence, hidden], projected.dtype())?;
        let mut p = [0; 9];
        p[0] = index(elements)?;
        p[1] = index(hidden)?;
        p[4] = projected.dtype() as u32;
        let start = self.profiling().then(std::time::Instant::now);
        let timing = self.dispatch(
            "lfm2_split3",
            &[
                projected.binding(),
                first.binding(),
                second.binding(),
                third.binding(),
            ],
            &p,
            [elements, 1],
            false,
        )?;
        if let Some(start) = start {
            self.record_profile(
                "short_conv_split",
                start
                    .elapsed()
                    .saturating_sub(timing.synchronized.saturating_sub(timing.submission)),
                std::time::Duration::ZERO,
                first.byte_size() + second.byte_size() + third.byte_size(),
                &timing,
            );
        }
        Ok((first, second, third))
    }
}

impl MetalDevice {
    /// Row-wise greedy selection: `[rows, 2]` F32 carrier holding u32 bits for
    /// the first maximal index and a non-finite flag.
    pub(crate) fn argmax_rows(&self, logits: &Tensor) -> Result<Tensor> {
        let dims = logits.shape().dimensions();
        if dims.len() != 2 || dims.contains(&0) || dims[1] > u32::MAX as usize {
            return Err(Error::Shape("argmax requires nonempty [rows,width]".into()));
        }
        if !self.owns(logits.buffer()) {
            return Err(Error::DeviceMismatch);
        }
        let output = Tensor::output(self, &[dims[0], 2], DType::F32)?;
        let mut p = [0; 9];
        p[0] = index(logits.numel())?;
        p[1] = index(dims[1])?;
        p[4] = logits.dtype() as u32;
        self.dispatch(
            "argmax_rows",
            &[logits.binding(), logits.binding(), output.binding()],
            &p,
            [dims[0], 1],
            false,
        )?;
        Ok(output)
    }
    /// Internal append-only KV write. Caller reserves an unpublished suffix before
    /// calling; no published tensor may cover the destination range.
    pub(crate) fn write_kv(&self, source: &Tensor, suffix: &Tensor) -> Result<()> {
        if source.dtype() != suffix.dtype() || source.numel() != suffix.numel() {
            return Err(Error::Cache("KV write shape/dtype mismatch".into()));
        }
        let mut p = [0; 9];
        p[0] = index(source.numel())?;
        p[4] = source.dtype() as u32;
        let start = self.profiling().then(std::time::Instant::now);
        let timing = self.dispatch(
            "copy_range",
            &[source.binding(), source.binding(), suffix.binding()],
            &p,
            [source.numel(), 1],
            false,
        )?;
        if let Some(start) = start {
            self.record_profile(
                "kv_append",
                start
                    .elapsed()
                    .saturating_sub(timing.synchronized.saturating_sub(timing.submission)),
                std::time::Duration::ZERO,
                0,
                &timing,
            );
        }
        Ok(())
    }
}

impl MetalDevice {
    /// Grouped products read sequence-major Q/K directly, retaining the original
    /// dot-product and storage-rounding boundaries without head materialization.
    pub(crate) fn attention_scores(&self, q: &Tensor, k: &Tensor) -> Result<Output> {
        let a = q.shape().dimensions();
        let b = k.shape().dimensions();
        if a.len() != 3
            || b.len() != 3
            || a[2] != b[2]
            || a.contains(&0)
            || b.contains(&0)
            || !a[1].is_multiple_of(b[1])
        {
            return Err(Error::Shape("grouped Q/K shapes".into()));
        }
        let mut p = [0; 9];
        for (i, n) in [(1, a[0]), (2, b[0]), (3, a[1]), (5, b[1]), (6, a[2])] {
            p[i] = index(n)?;
        }
        let dims = [a[1], a[0], b[0]];
        let n = crate::tensor::Shape::new(dims)?.numel();
        if self.mpp_projection()
            && matches!(q.dtype(), DType::BF16 | DType::F16)
            && a[0] > 1
            && [q.numel(), k.numel(), n]
                .iter()
                .all(|&x| x <= i32::MAX as usize)
        {
            let name = if q.dtype() == DType::F16 {
                "attention_scores_mpp_f16"
            } else {
                "attention_scores_mpp"
            };
            self.run(name, q, Some(k), &dims, p, [b[0], a[0]])
        } else if self.native_matmul()
            && q.dtype() == DType::BF16
            && a[0] > 1
            && a[0] <= u32::MAX as usize - 8
            && a[2] <= u32::MAX as usize - 8
        {
            let rows = a[0]
                .div_ceil(8)
                .checked_mul(8)
                .and_then(|n| n.checked_mul(a[1]))
                .ok_or_else(|| Error::Shape("attention grid overflow".into()))?;
            self.run(
                "attention_scores_matrix",
                q,
                Some(k),
                &dims,
                p,
                [b[0], rows],
            )
        } else {
            // Four-wide BF16 loads need 8-byte aligned rows: head dimension and
            // both tensor offsets divisible by four elements.
            let vector = q.dtype() == DType::BF16
                && self.attention_scores_vector()
                && a[2].is_multiple_of(4)
                && q.binding().1.is_multiple_of(8)
                && k.binding().1.is_multiple_of(8);
            p[8] = u32::from(vector);
            self.run("attention_scores", q, Some(k), &dims, p, [n, 1])
        }
    }
    pub(crate) fn attention_mask(&self, scores: &Tensor, offset: usize) -> Result<Output> {
        let dims = scores.shape().dimensions();
        if dims.len() != 3 || dims[1] == 0 || offset.checked_add(dims[1]) != Some(dims[2]) {
            return Err(Error::Shape("grouped causal scores".into()));
        }
        let mut p = [0; 9];
        p[1] = index(dims[1])?;
        p[2] = index(dims[2])?;
        p[5] = index(offset)?;
        self.run("attention_mask", scores, None, dims, p, [scores.numel(), 1])
    }
    pub(crate) fn attention_softmax(
        &self,
        scores: &Tensor,
        offset: usize,
        scale: f32,
    ) -> Result<Output> {
        let dims = scores.shape().dimensions();
        if dims.len() != 3
            || dims[1] == 0
            || offset.checked_add(dims[1]) != Some(dims[2])
            || dims[2] > u32::MAX as usize - 256
            || !scale.is_finite()
        {
            return Err(Error::Shape("grouped causal softmax scores".into()));
        }
        let mut p = [0; 9];
        p[1] = index(dims[2])?;
        p[2] = index(dims[1])?;
        p[5] = index(offset)?;
        p[7] = scale.to_bits();
        let kernel = if self.attention_softmax_prefix_reuse(dims[1], dims[2]) {
            "attention_softmax_prefix_reuse"
        } else if self.attention_softmax_prefix(dims[1], dims[2]) {
            "attention_softmax_prefix"
        } else {
            "attention_softmax"
        };
        self.run(kernel, scores, None, dims, p, [scores.numel() / dims[2], 1])
    }
    /// Whether `attention_decode` handles one query row against this cache.
    pub(crate) fn can_flash_decode(&self, q: &Tensor, k: &Tensor, v: &Tensor) -> bool {
        let (qd, kd) = (q.shape().dimensions(), k.shape().dimensions());
        self.flash_decode()
            && !self.reference_math()
            && qd.len() == 3
            && qd[0] == 1
            && kd.len() == 3
            && k.shape() == v.shape()
            && q.dtype() == k.dtype()
            && k.dtype() == v.dtype()
            && q.dtype() != DType::F32
            && qd[2] == kd[2]
            && kd[1] > 0
            && qd[1].is_multiple_of(kd[1])
            && kd[2].is_multiple_of(32)
            && kd[2] <= 256
            && kd[0] > 0
            && kd[0] <= u32::MAX as usize / 2
    }
    /// Single-row decode attention, `softmax(scale * q Kᵀ) V`, as a
    /// split-context pass plus a combine. Returns `[1, heads, head_dim]`.
    pub(crate) fn attention_decode(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        scale: f32,
    ) -> Result<Tensor> {
        if !self.can_flash_decode(q, k, v) || !scale.is_finite() {
            return Err(Error::Shape("flash decode operands".into()));
        }
        if [q, k, v].iter().any(|t| !self.owns(t.buffer())) {
            return Err(Error::DeviceMismatch);
        }
        let (heads, tokens, kv_heads, dim) = (
            q.shape().dimensions()[1],
            k.shape().dimensions()[0],
            k.shape().dimensions()[1],
            k.shape().dimensions()[2],
        );
        // Enough (head, split) groups to fill the GPU, at least 32 tokens each.
        let splits = tokens.div_ceil(32).min(96usize.div_ceil(heads)).max(1);
        let partial = Tensor::output(self, &[heads * splits, dim + 2], DType::F32)?;
        let out = Tensor::output(self, &[1, heads, dim], q.dtype())?;
        let mut p = [0; 9];
        p[1] = index(tokens)?;
        p[2] = index(splits)?;
        p[3] = index(heads)?;
        p[4] = q.dtype() as u32;
        p[5] = index(kv_heads)?;
        p[6] = index(dim)?;
        p[7] = scale.to_bits();
        let start = self.profiling().then(std::time::Instant::now);
        self.dispatch(
            "attention_decode_split",
            &[q.binding(), k.binding(), v.binding(), partial.binding()],
            &p,
            [heads * splits, 1],
            false,
        )?;
        let timing = self.dispatch(
            "attention_decode_reduce",
            &[partial.binding(), out.binding()],
            &p,
            [heads, 1],
            false,
        )?;
        if let Some(start) = start {
            self.record_profile(
                "attention_decode",
                start.elapsed(),
                std::time::Duration::ZERO,
                out.storage_info().allocation_bytes,
                &timing,
            );
        }
        Ok(out)
    }
    pub(crate) fn attention_context(&self, probs: &Tensor, v: &Tensor) -> Result<Output> {
        let a = probs.shape().dimensions();
        let b = v.shape().dimensions();
        if a.len() != 3
            || b.len() != 3
            || a.contains(&0)
            || b.contains(&0)
            || a[2] != b[0]
            || !a[0].is_multiple_of(b[1])
        {
            return Err(Error::Shape("grouped probabilities/V shapes".into()));
        }
        let mut p = [0; 9];
        for (i, n) in [(1, a[1]), (2, b[0]), (3, a[0]), (5, b[1]), (6, b[2])] {
            p[i] = index(n)?;
        }
        let dims = [a[1], a[0], b[2]];
        let n = crate::tensor::Shape::new(dims)?.numel();
        // The 1,024-thread kernel replaces the scalar path below 128 positions and,
        // above it, wins only while (head, column tile) groups underfill the GPU.
        let wide_groups = b[2]
            .div_ceil(32)
            .checked_mul(a[0])
            .ok_or_else(|| Error::Shape("decode context grid overflow".into()))?;
        if self.attention_context_decode_wide() && a[1] == 1 && (b[0] < 128 || wide_groups < 64) {
            let groups = wide_groups;
            self.run(
                "attention_context_decode_wide",
                probs,
                Some(v),
                &dims,
                p,
                [groups, 1],
            )
        } else if a[1] == 1
            && b[0] >= 128
            && b[0] <= u32::MAX as usize - 8
            && b[2] <= u32::MAX as usize - 32
            && probs.dtype() != DType::F32
        {
            let groups = b[2]
                .div_ceil(32)
                .checked_mul(a[0])
                .ok_or_else(|| Error::Shape("decode context grid overflow".into()))?;
            self.run(
                "attention_context_decode",
                probs,
                Some(v),
                &dims,
                p,
                [groups, 1],
            )
        } else if self.mpp_projection()
            && matches!(probs.dtype(), DType::BF16 | DType::F16)
            && a[1] > 1
            && [probs.numel(), v.numel(), n]
                .iter()
                .all(|&x| x <= i32::MAX as usize)
        {
            let name = if probs.dtype() == DType::F16 {
                "attention_context_mpp_f16"
            } else {
                "attention_context_mpp"
            };
            self.run(name, probs, Some(v), &dims, p, [b[2], a[1]])
        } else if self.native_matmul()
            && probs.dtype() == DType::BF16
            && a[1] > 1
            && a[1] <= u32::MAX as usize - 8
            && b[0] <= u32::MAX as usize - 8
        {
            let rows = a[1]
                .div_ceil(8)
                .checked_mul(8)
                .and_then(|n| n.checked_mul(a[0]))
                .ok_or_else(|| Error::Shape("attention grid overflow".into()))?;
            self.run(
                "attention_context_matrix",
                probs,
                Some(v),
                &dims,
                p,
                [b[2], rows],
            )
        } else {
            self.run("attention_context", probs, Some(v), &dims, p, [n, 1])
        }
    }
}

impl MetalDevice {
    /// Linear projection with retained row-major [N,K] weights.
    pub(crate) fn project(&self, a: &Tensor, weight: &Tensor) -> Result<Output> {
        let ad = a.shape().dimensions();
        let bd = weight.shape().dimensions();
        if ad.len() != 2 || bd.len() != 2 || ad[1] != bd[1] || ad[1] > u32::MAX as usize - 32 {
            return Err(Error::Shape("projection requires [M,K] and [N,K]".into()));
        }
        let mut p = [0; 9];
        p[1] = index(ad[0])?;
        p[2] = index(ad[1])?;
        p[3] = index(bd[0])?;
        let f32_batched_gemv = a.dtype() == DType::F32
            && ad[0] > 1
            && ad[0] <= 32
            && ad[1] >= 512
            && bd[0] >= 512
            && ad[1].is_multiple_of(4)
            && a.storage_info().alignment >= 8
            && weight.storage_info().alignment >= 8;
        let name = if f32_batched_gemv {
            "gemv_wide"
        } else if ad[0] > 1 && self.native_matmul() && a.dtype() != DType::F32 {
            if ad[0] >= 32
                && ad[1] > 0
                && a.numel() <= i32::MAX as usize
                && weight.numel() <= i32::MAX as usize
                && ad[0]
                    .checked_mul(bd[0])
                    .is_some_and(|n| n <= i32::MAX as usize)
                && a.dtype() == DType::BF16
                && self.mpp_projection()
                && [ad[0], ad[1], bd[0]]
                    .iter()
                    .all(|&x| x <= i32::MAX as usize)
            {
                "project_mpp"
            } else if ad[0] >= 32 {
                if a.dtype() == DType::BF16 {
                    "project_wide_bf16"
                } else {
                    "project_wide_f16"
                }
            } else if a.dtype() == DType::BF16 {
                "project_bf16"
            } else {
                "project_f16"
            }
        } else if ad[0] == 1 {
            if self.split_k_gemv()
                && matches!(a.dtype(), DType::BF16 | DType::F32)
                && ad[1] >= 512
                && bd[0] >= 512
                && ad[1].is_multiple_of(4)
                && a.storage_info().alignment >= 8
                && weight.storage_info().alignment >= 8
            {
                "gemv_wide"
            } else if ad[1].is_multiple_of(4)
                && a.storage_info().alignment >= 8
                && weight.storage_info().alignment >= 8
            {
                "gemv_vector"
            } else {
                "gemv"
            }
        } else {
            "matmul_nt"
        };
        self.run(name, a, Some(weight), &[ad[0], bd[0]], p, [bd[0], ad[0]])
    }
}

#[cfg(test)]
mod fusion_tests {
    use super::*;
    fn pseudo(n: usize, seed: usize) -> Vec<f32> {
        (0..n)
            .map(|i| (((i * 2654435761 + seed * 40503) >> 7) % 2001) as f32 / 1000.0 - 1.0)
            .collect()
    }
    #[test]
    fn flash_decode_matches_reference_and_unfused_path() {
        let d = MetalDevice::new().unwrap();
        d.set_flash_decode(true).unwrap();
        for (heads, kv_heads, dim, tokens) in [
            (14, 2, 64, 1),
            (14, 2, 64, 37),
            (14, 2, 64, 1024),
            (16, 8, 128, 300),
            (32, 8, 64, 5),
            (4, 4, 256, 70),
        ] {
            let scale = 1.0 / (dim as f32).sqrt() * 3.0;
            let dtype = DType::BF16;
            let q = Tensor::from_f32(&d, [1, heads, dim], dtype, &pseudo(heads * dim, 1)).unwrap();
            let kv = [tokens, kv_heads, dim];
            let k = Tensor::from_f32(&d, kv, dtype, &pseudo(tokens * kv_heads * dim, 2)).unwrap();
            let v = Tensor::from_f32(&d, kv, dtype, &pseudo(tokens * kv_heads * dim, 3)).unwrap();
            let (qv, kv_, vv) = (q.to_f32(), k.to_f32(), v.to_f32());
            let group = heads / kv_heads;
            let mut reference = vec![0.0f32; heads * dim];
            for h in 0..heads {
                let kvh = h / group;
                let scores: Vec<f32> = (0..tokens)
                    .map(|t| {
                        scale
                            * (0..dim)
                                .map(|j| qv[h * dim + j] * kv_[(t * kv_heads + kvh) * dim + j])
                                .sum::<f32>()
                    })
                    .collect();
                let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let weights: Vec<f32> = scores.iter().map(|x| (x - max).exp()).collect();
                let total: f32 = weights.iter().sum();
                for j in 0..dim {
                    reference[h * dim + j] = (0..tokens)
                        .map(|t| weights[t] * vv[(t * kv_heads + kvh) * dim + j])
                        .sum::<f32>()
                        / total;
                }
            }
            let flash = d.attention_decode(&q, &k, &v, scale).unwrap().to_f32();
            let scores = d
                .attention_scores(&q.reshape([1, heads, dim]).unwrap(), &k)
                .unwrap()
                .tensor;
            let probs = d
                .attention_softmax(&scores, tokens - 1, scale)
                .unwrap()
                .tensor;
            let unfused = d.attention_context(&probs, &v).unwrap().tensor.to_f32();
            let error = |x: &[f32]| {
                x.iter()
                    .zip(&reference)
                    .map(|(a, b)| (a - b).abs())
                    .fold(0.0f32, f32::max)
            };
            let (flash_error, unfused_error) = (error(&flash), error(&unfused));
            assert!(
                // Both paths round scores to storage; flash must be no worse.
                flash_error <= 5e-2 && flash_error <= unfused_error * 1.25 + 2e-3,
                "({heads},{kv_heads},{dim},{tokens}): flash {flash_error} unfused {unfused_error}"
            );
        }
    }
    #[test]
    fn fused_silu_multiply_preserves_storage_rounding() {
        let d = MetalDevice::new().unwrap();
        for dtype in [DType::F32, DType::F16, DType::BF16] {
            let values: Vec<_> = (-500..501).map(|i| i as f32 / 10.).collect();
            let x = Tensor::from_f32(&d, [values.len()], dtype, &values).unwrap();
            let y = Tensor::from_f32(
                &d,
                [values.len()],
                dtype,
                &crate::reference::deterministic(values.len()),
            )
            .unwrap();
            let staged = d.mul(&d.silu(&x).unwrap().tensor, &y).unwrap().tensor;
            let fused = d.silu_mul(&x, &y).unwrap().tensor;
            assert_eq!(staged.to_f32(), fused.to_f32(), "{dtype:?}");
        }
    }
    #[test]
    fn decode_context_partitions_cover_threshold_and_column_tails() {
        let d = MetalDevice::new().unwrap();
        for dtype in [DType::BF16, DType::F16] {
            for (t, width) in [(127, 13), (128, 13), (1025, 65)] {
                let p = Tensor::from_f32(
                    &d,
                    [6, 1, t],
                    dtype,
                    &crate::reference::deterministic(6 * t),
                )
                .unwrap();
                let p = d.softmax(&p).unwrap().tensor;
                let v = Tensor::from_f32(
                    &d,
                    [t, 2, width],
                    dtype,
                    &crate::reference::deterministic(t * 2 * width),
                )
                .unwrap();
                let pv = p.to_f32();
                let vv = v.to_f32();
                let mut expected = Vec::new();
                for h in 0..6 {
                    for col in 0..width {
                        let mut sum = 0f32;
                        for pos in 0..t {
                            sum += pv[h * t + pos] * vv[(pos * 2 + h / 3) * width + col];
                        }
                        expected.push(dtype.round(sum));
                    }
                }
                let y = d.attention_context(&p, &v).unwrap().tensor;
                let (atol, rtol) = if dtype == DType::BF16 {
                    (1.6e-2, 1e-2)
                } else {
                    (2e-3, 2e-3)
                };
                crate::reference::check(&y.to_f32(), &expected, atol, rtol).unwrap();
            }
        }
    }
    #[test]
    fn argmax_rows_matches_host_ties_tails_and_nonfinite_flags() {
        let d = MetalDevice::new().unwrap();
        for dtype in [DType::F32, DType::F16, DType::BF16] {
            for width in [1, 7, 255, 256, 257, 151_936] {
                let rows = 3;
                let mut values: Vec<f32> = crate::reference::deterministic(rows * width)
                    .iter()
                    .map(|x| (x * 8.).round() / 8.)
                    .collect();
                // Row 1: an exact tie at a later index must lose to the earlier one.
                if width > 3 {
                    values[width + 1] = 100.;
                    values[width + width - 1] = 100.;
                }
                let logits = Tensor::from_f32(&d, [rows, width], dtype, &values).unwrap();
                let selected = d.argmax_rows(&logits).unwrap().to_f32();
                let rounded = logits.to_f32();
                for row in 0..rows {
                    let expected =
                        crate::generation::argmax(&rounded[row * width..(row + 1) * width])
                            .unwrap();
                    assert_eq!(
                        selected[row * 2].to_bits(),
                        expected,
                        "{dtype:?} {width} {row}"
                    );
                    assert_eq!(selected[row * 2 + 1].to_bits(), 0);
                }
            }
            let mut values = vec![0.5f32; 300];
            values[299] = f32::NAN;
            let logits = Tensor::from_f32(&d, [1, 300], dtype, &values).unwrap();
            assert_eq!(d.argmax_rows(&logits).unwrap().to_f32()[1].to_bits(), 1);
            values[299] = f32::INFINITY;
            let logits = Tensor::from_f32(&d, [1, 300], dtype, &values).unwrap();
            assert_eq!(d.argmax_rows(&logits).unwrap().to_f32()[1].to_bits(), 1);
        }
    }
    #[test]
    fn fused_add_rmsnorm_matches_separate_kernels_bitwise() {
        let d = MetalDevice::new().unwrap();
        let bits = |t: &Tensor| t.to_f32().iter().map(|v| v.to_bits()).collect::<Vec<_>>();
        for dtype in [DType::F32, DType::F16, DType::BF16] {
            for (rows, width) in [(1, 896), (37, 2048), (5, 64), (3, 1000)] {
                let n = rows * width;
                let a = Tensor::from_f32(
                    &d,
                    [rows, width],
                    dtype,
                    &crate::reference::deterministic(n),
                )
                .unwrap();
                let b = Tensor::from_f32(
                    &d,
                    [rows, width],
                    dtype,
                    &crate::reference::deterministic(n + 11)[11..],
                )
                .unwrap();
                let w = Tensor::from_f32(
                    &d,
                    [width],
                    dtype,
                    &crate::reference::deterministic(width + 3)[3..]
                        .iter()
                        .map(|v| 1. + v)
                        .collect::<Vec<_>>(),
                )
                .unwrap();
                let residual = d.add(&a, &b).unwrap().tensor;
                let normed = d.rmsnorm(&residual, &w, 1e-6).unwrap().tensor;
                let (fused_residual, fused_normed) = d.add_rmsnorm(&a, &b, &w, 1e-6).unwrap();
                assert_eq!(
                    bits(&fused_residual),
                    bits(&residual),
                    "{dtype:?} {rows}x{width}"
                );
                assert_eq!(
                    bits(&fused_normed),
                    bits(&normed),
                    "{dtype:?} {rows}x{width}"
                );
            }
        }
    }
    #[test]
    fn rope_table_matches_per_element_rotation_bitwise() {
        let d = MetalDevice::new().unwrap();
        let bits = |t: &Tensor| t.to_f32().iter().map(|v| v.to_bits()).collect::<Vec<_>>();
        for dtype in [DType::BF16, DType::F16, DType::F32] {
            for (rows, q_heads, kv_heads, dim, offset, theta) in [
                (37, 14, 2, 64, 5, 1_000_000.0f32),
                (1, 16, 8, 128, 1023, 10_000.0),
                (130, 32, 8, 64, 0, 1_000_000.0),
            ] {
                let q = Tensor::from_f32(
                    &d,
                    [rows, q_heads, dim],
                    dtype,
                    &crate::reference::deterministic(rows * q_heads * dim),
                )
                .unwrap();
                let k = Tensor::from_f32(
                    &d,
                    [rows, kv_heads, dim],
                    dtype,
                    &crate::reference::deterministic(rows * kv_heads * dim),
                )
                .unwrap();
                let mut outputs = Vec::new();
                for table in [false, true] {
                    d.set_rope_table(table).unwrap();
                    let execution = d.execution().unwrap();
                    let rq = d.rope_split(&q, offset, theta).unwrap();
                    let rk = d.rope_split(&k, offset, theta).unwrap();
                    let expected = if table {
                        "rope_split_table"
                    } else {
                        "rope_split"
                    };
                    assert_eq!(rq.metrics.operation, expected);
                    assert_eq!(rk.metrics.operation, expected);
                    execution.finish().unwrap();
                    outputs.push((bits(&rq.tensor), bits(&rk.tensor)));
                }
                assert_eq!(outputs[0], outputs[1], "{dtype:?} rows={rows} dim={dim}");
            }
        }
    }
    #[test]
    fn vector_bf16_attention_scores_are_bit_identical() {
        let d = MetalDevice::new().unwrap();
        let bits = |t: &Tensor| t.to_f32().iter().map(|v| v.to_bits()).collect::<Vec<_>>();
        for (heads, kv, width, t) in [
            (32, 8, 64, 1025),
            (14, 2, 64, 37),
            (6, 6, 80, 5),
            (16, 8, 128, 300),
            (4, 2, 33, 9),
        ] {
            let q = Tensor::from_f32(
                &d,
                [1, heads, width],
                DType::BF16,
                &crate::reference::deterministic(heads * width),
            )
            .unwrap();
            let k = Tensor::from_f32(
                &d,
                [t + 1, kv, width],
                DType::BF16,
                &crate::reference::deterministic((t + 1) * kv * width),
            )
            .unwrap();
            // Aligned view and one offset by a single row (still 8-byte aligned when width%4==0),
            // plus an unaligned two-element offset that must fall back to the scalar loop.
            for (start, rows) in [(0, t), (kv * width, t)] {
                let k = k.view(start, [rows, kv, width]).unwrap();
                d.set_attention_scores_vector(false).unwrap();
                let scalar = d.attention_scores(&q, &k).unwrap().tensor;
                d.set_attention_scores_vector(true).unwrap();
                let vector = d.attention_scores(&q, &k).unwrap().tensor;
                assert_eq!(
                    bits(&vector),
                    bits(&scalar),
                    "H={heads} KV={kv} D={width} T={t}"
                );
            }
            let flat = Tensor::from_f32(
                &d,
                [heads * width + 2],
                DType::BF16,
                &crate::reference::deterministic(heads * width + 2),
            )
            .unwrap();
            let q_unaligned = flat.view(2, [1, heads, width]).unwrap();
            let k0 = k.view(0, [t, kv, width]).unwrap();
            d.set_attention_scores_vector(false).unwrap();
            let scalar = d.attention_scores(&q_unaligned, &k0).unwrap().tensor;
            d.set_attention_scores_vector(true).unwrap();
            let vector = d.attention_scores(&q_unaligned, &k0).unwrap().tensor;
            assert_eq!(
                bits(&vector),
                bits(&scalar),
                "unaligned H={heads} D={width}"
            );
        }
    }
    #[test]
    fn wide_decode_context_matches_existing_kernels() {
        let d = MetalDevice::new().unwrap();
        for dtype in [DType::BF16, DType::F16, DType::F32] {
            let (atol, rtol) = match dtype {
                DType::BF16 => (1.6e-2, 1e-2),
                DType::F16 => (2e-3, 2e-3),
                DType::F32 => (1e-5, 1e-5),
            };
            // GQA groups 7, 2, 1 and 10; odd widths; short (scalar) and long (partitioned) controls.
            for (heads, kv, width, t) in [
                (14, 2, 64, 1),
                (14, 2, 64, 1025),
                (16, 8, 128, 37),
                (6, 6, 80, 300),
                (20, 2, 33, 129),
            ] {
                let q = Tensor::from_f32(
                    &d,
                    [1, heads, width],
                    dtype,
                    &crate::reference::deterministic(heads * width),
                )
                .unwrap();
                let k = Tensor::from_f32(
                    &d,
                    [t, kv, width],
                    dtype,
                    &crate::reference::deterministic(t * kv * width),
                )
                .unwrap();
                let scores = d.attention_scores(&q, &k).unwrap().tensor;
                let probs = d
                    .attention_softmax(&scores, t - 1, (width as f32).sqrt().recip())
                    .unwrap()
                    .tensor;
                d.set_attention_context_decode_wide(false).unwrap();
                let slow = d.attention_context(&probs, &k).unwrap().tensor;
                d.set_attention_context_decode_wide(true).unwrap();
                let fast = d.attention_context(&probs, &k).unwrap().tensor;
                crate::reference::check(&fast.to_f32(), &slow.to_f32(), atol, rtol).unwrap();
            }
        }
    }
    #[test]
    fn tiled_attention_products_match_scalar_for_grouping_and_tails() {
        let d = MetalDevice::new().unwrap();
        for dtype in [DType::BF16, DType::F16] {
            let (atol, rtol) = if dtype == DType::F16 {
                (2e-3, 2e-3)
            } else {
                (1.6e-2, 1e-2)
            };
            for (s, t, width) in [(3, 5, 12), (17, 33, 64), (3, 1027, 64)] {
                let q = Tensor::from_f32(
                    &d,
                    [s, 6, width],
                    dtype,
                    &crate::reference::deterministic(s * 6 * width),
                )
                .unwrap();
                let k = Tensor::from_f32(
                    &d,
                    [t, 2, width],
                    dtype,
                    &crate::reference::deterministic(t * 2 * width),
                )
                .unwrap();
                d.set_native_matmul(true).unwrap();
                let fast = d.attention_scores(&q, &k).unwrap().tensor;
                d.set_native_matmul(false).unwrap();
                let slow = d.attention_scores(&q, &k).unwrap().tensor;
                crate::reference::check(&fast.to_f32(), &slow.to_f32(), atol, rtol).unwrap();
                let probs = d
                    .attention_softmax(&slow, t - s, (width as f32).sqrt().recip())
                    .unwrap()
                    .tensor;
                let slow = d.attention_context(&probs, &k).unwrap().tensor;
                d.set_native_matmul(true).unwrap();
                let fast = d.attention_context(&probs, &k).unwrap().tensor;
                crate::reference::check(&fast.to_f32(), &slow.to_f32(), atol, rtol).unwrap();
            }
        }
    }
    #[test]
    fn attention_softmax_preserves_staged_rounding_and_causal_offsets() {
        let d = MetalDevice::new().unwrap();
        for dtype in [DType::F32, DType::F16, DType::BF16] {
            for (s, offset) in [(1, 0), (5, 3), (17, 256), (3, 1024)] {
                let shape = [2, s, s + offset];
                let values = crate::reference::deterministic(shape.iter().product());
                let x = Tensor::from_f32(&d, shape, dtype, &values).unwrap();
                let scale = 64f32.sqrt().recip();
                let a = d.scale(&x, scale).unwrap().tensor;
                let a = d.attention_mask(&a, offset).unwrap().tensor;
                let a = d.softmax(&a).unwrap().tensor;
                let b = d.attention_softmax(&x, offset, scale).unwrap().tensor;
                assert_eq!(a.to_f32(), b.to_f32(), "{dtype:?} S={s} P={offset}");
            }
        }
    }

    #[test]
    fn attention_softmax_prefix_matches_full_scan_for_causal_tails() {
        let d = MetalDevice::new().unwrap();
        d.set_attention_softmax_prefix(true).unwrap();
        d.set_attention_softmax_prefix_reuse(false).unwrap();
        for dtype in [DType::F32, DType::F16, DType::BF16] {
            for (s, offset) in [
                (1, 0),
                (5, 3),
                (17, 256),
                (64, 64),
                (128, 0),
                (128, 128),
                (256, 0),
                (256, 256),
            ] {
                let shape = [2, s, s + offset];
                let values = crate::reference::deterministic(shape.iter().product());
                let x = Tensor::from_f32(&d, shape, dtype, &values).unwrap();
                let expected = d
                    .attention_softmax(&x, offset, 64f32.sqrt().recip())
                    .unwrap();
                let kernel = if s >= 256 && offset == 0 {
                    "attention_softmax_prefix"
                } else {
                    "attention_softmax"
                };
                assert_eq!(
                    expected.metrics.operation, kernel,
                    "{dtype:?} S={s} P={offset}"
                );
                d.set_attention_softmax_prefix(false).unwrap();
                let reference = d
                    .attention_softmax(&x, offset, 64f32.sqrt().recip())
                    .unwrap();
                d.set_attention_softmax_prefix(true).unwrap();
                let actual = expected.tensor.to_f32();
                let reference = reference.tensor.to_f32();
                let mismatch = actual
                    .iter()
                    .zip(&reference)
                    .position(|(actual, reference)| actual != reference);
                assert!(
                    mismatch.is_none(),
                    "{dtype:?} S={s} P={offset} mismatch={:?}",
                    mismatch.map(|i| (i, actual[i], reference[i]))
                );
            }
        }
    }

    #[test]
    fn attention_softmax_prefix_reuse_preserves_results_through_width_1024() {
        let d = MetalDevice::new().unwrap();
        d.set_attention_softmax_prefix(true).unwrap();
        assert!(!d.attention_softmax_prefix_reuse(255, 255));
        assert!(!d.attention_softmax_prefix_reuse(256, 512));
        assert!(!d.attention_softmax_prefix_reuse(1025, 1025));
        for dtype in [DType::F32, DType::F16, DType::BF16] {
            for sequence in [256, 512, 768, 1024] {
                let shape = [2, sequence, sequence];
                let values = crate::reference::deterministic(shape.iter().product());
                let scores = Tensor::from_f32(&d, shape, dtype, &values).unwrap();
                d.set_attention_softmax_prefix_reuse(false).unwrap();
                let control = d
                    .attention_softmax(&scores, 0, 64f32.sqrt().recip())
                    .unwrap();
                assert_eq!(control.metrics.operation, "attention_softmax_prefix");
                d.set_attention_softmax_prefix_reuse(true).unwrap();
                let candidate = d
                    .attention_softmax(&scores, 0, 64f32.sqrt().recip())
                    .unwrap();
                assert_eq!(
                    candidate.metrics.operation,
                    "attention_softmax_prefix_reuse"
                );
                assert_eq!(candidate.tensor.to_f32(), control.tensor.to_f32());
            }
        }
    }
}

#[cfg(test)]
mod q8_0_tests {
    use super::*;
    use crate::quantization::{QuantizationFormat, QuantizedMatrix};

    fn packed(device: &MetalDevice, rows: usize, columns: usize) -> QuantizedMatrix {
        let mut bytes = Vec::new();
        for row in 0..rows {
            for _ in 0..columns / 32 {
                bytes.extend_from_slice(&half::f16::from_f32(0.03125).to_bits().to_le_bytes());
                for column in 0..32 {
                    let q = ((row * 11 + column * 7) % 25) as i8 - 12;
                    bytes.push(q as u8);
                }
            }
        }
        let expected_len = rows * (columns / 32) * 34;
        assert_eq!(bytes.len(), expected_len);
        let matrix =
            QuantizedMatrix::from_reader(device, rows, columns, QuantizationFormat::Q8_0, |dst| {
                dst.copy_from_slice(&bytes);
                Ok(())
            })
            .unwrap();
        assert_eq!(matrix.byte_size(), expected_len);
        assert_eq!(matrix.with_bytes(|data| data.len()), expected_len);
        matrix
    }

    fn q8_value(row: usize, column: usize) -> f32 {
        let q = ((row * 11 + (column % 32) * 7) % 25) as i8 - 12;
        0.03125 * f32::from(q)
    }

    fn input(rows: usize, columns: usize) -> Vec<f32> {
        (0..rows * columns)
            .map(|i| ((i * 29 + 3) % 101) as f32 / 73.0 - 0.65)
            .collect()
    }

    fn expected(input: &[f32], m: usize, n: usize, k: usize) -> Vec<f32> {
        (0..m)
            .flat_map(|row| {
                (0..n).map(move |column| {
                    (0..k)
                        .map(|i| input[row * k + i] * q8_value(column, i))
                        .sum::<f32>()
                })
            })
            .collect()
    }

    #[test]
    fn q8_0_gemv_k_split_row_counts_are_bit_identical() {
        let d = MetalDevice::new().unwrap();
        for (n, k) in [(131, 896), (130, 1024), (129, 3072)] {
            let weight = packed(&d, n, k);
            let x = Tensor::from_f32(&d, [1, k], DType::BF16, &input(1, k)).unwrap();
            let mut outputs = Vec::new();
            for rows in [2, 4, 8] {
                d.set_q8_0_gemv_rows(rows).unwrap();
                let output = d.project_quantized(&x, &weight).unwrap();
                let name = match rows {
                    8 => "q8_0_gemv_k_split8",
                    4 => "q8_0_gemv_k_split4",
                    _ => "q8_0_gemv_k_split",
                };
                assert_eq!(output.metrics.operation, name);
                outputs.push(output.tensor.to_f32());
            }
            for output in &outputs[1..] {
                for (index, (a, b)) in outputs[0].iter().zip(output).enumerate() {
                    assert_eq!(a.to_bits(), b.to_bits(), "({n}, {k}) index {index}");
                }
            }
        }
        d.set_q8_0_gemv_rows(2).unwrap();
    }

    /// Steady-state Q8_0 GEMV: Qwen2.5 lm_head and Qwen3-0.6B shapes.
    #[test]
    #[ignore = "microbenchmark; run with --ignored --nocapture"]
    fn bench_q8_0_gemv() {
        let d = MetalDevice::new().unwrap();
        for (n, k, copies) in [
            (151936, 896, 2),
            (3072, 1024, 24),
            (1024, 3072, 24),
            (2048, 1024, 24),
        ] {
            let weights: Vec<_> = (0..copies).map(|_| packed(&d, n, k)).collect();
            let x = Tensor::from_f32(&d, [1, k], DType::BF16, &input(1, k)).unwrap();
            let bytes = (weights[0].byte_size() * copies) as f64;
            for rows in [2, 4, 8] {
                d.set_q8_0_gemv_rows(rows).unwrap();
                let mut best = f64::MAX;
                for _ in 0..7 {
                    let start = std::time::Instant::now();
                    let execution = d.execution().unwrap();
                    let mut last = None;
                    for _ in 0..4 {
                        for weight in &weights {
                            last = Some(d.project_quantized(&x, weight).unwrap().tensor);
                        }
                    }
                    drop(execution);
                    std::hint::black_box(last.unwrap().to_f32());
                    best = best.min(start.elapsed().as_secs_f64() / 4.0);
                }
                println!(
                    "Q8_0 {n}x{k} rows={rows}: {:.1} us/projection, {:.1} GB/s",
                    best * 1e6 / copies as f64,
                    bytes / best / 1e9
                );
            }
        }
        d.set_q8_0_gemv_rows(2).unwrap();
    }

    #[test]
    fn q8_0_gemv_and_gemm_cover_output_tails_and_odd_batch_sizes() {
        let d = MetalDevice::new().unwrap();
        assert!(d.q8_0_gemv_k_split(128, 768));
        d.set_q8_0_gemv_k_split(false).unwrap();
        assert!(!d.q8_0_gemv_8rows(5));
        assert!(!d.q8_0_gemv_8rows(127));
        assert!(d.q8_0_gemv_8rows(128));
        assert!(d.q8_0_gemv_8rows(896));
        assert!(!d.q8_0_gemv_8rows(129));
        assert!(!d.q8_0_gemv_k_split(128, 767));
        assert!(!d.q8_0_gemv_k_split(127, 768));
        assert!(!d.q8_0_gemv_k_split(128, 768));

        let n = 5;
        let k = 64;
        let weight = packed(&d, n, k);
        for m in [1, 2, 3, 4, 5, 7] {
            let values = input(m, k);
            let x = Tensor::from_f32(&d, [m, k], DType::F32, &values).unwrap();
            let actual = d.project_quantized(&x, &weight).unwrap().tensor.to_f32();
            let expected = expected(&values, m, n, k);
            for (index, (actual, expected)) in actual.iter().zip(&expected).enumerate() {
                assert!(
                    (actual - expected).abs() <= 2.0e-5,
                    "index {index}: actual={actual}, expected={expected}"
                );
            }
        }
        let n = 128;
        let weight = packed(&d, n, k);
        let values = input(1, k);
        let x = Tensor::from_f32(&d, [1, k], DType::F32, &values).unwrap();
        let output = d.project_quantized(&x, &weight).unwrap();
        assert_eq!(output.metrics.operation, "q8_0_gemv_8rows");
        let expected_eight_rows = expected(&values, 1, n, k);
        for (index, (actual, expected)) in output
            .tensor
            .to_f32()
            .iter()
            .zip(expected_eight_rows)
            .enumerate()
        {
            assert!(
                (actual - expected).abs() <= 2.0e-5,
                "eight-row GEMV index {index}: actual={actual}, expected={expected}"
            );
        }

        let (n, k) = (129, 1056);
        let weight = packed(&d, n, k);
        let values = input(1, k);
        let x = Tensor::from_f32(&d, [1, k], DType::F32, &values).unwrap();
        d.set_q8_0_gemv_k_split(true).unwrap();
        assert!(!d.q8_0_gemv_k_split(n, 767));
        assert!(d.q8_0_gemv_k_split(n, k));
        let output = d.project_quantized(&x, &weight).unwrap();
        assert_eq!(output.metrics.operation, "q8_0_gemv_k_split4");
        let expected_k_split = expected(&values, 1, n, k);
        for (index, (actual, expected)) in output
            .tensor
            .to_f32()
            .iter()
            .zip(expected_k_split)
            .enumerate()
        {
            assert!(
                (actual - expected).abs() <= 2.0e-5,
                "K-split GEMV index {index}: actual={actual}, expected={expected}"
            );
        }
        d.set_q8_0_gemv_k_split(false).unwrap();
        let output = d.project_quantized(&x, &weight).unwrap();
        assert_eq!(output.metrics.operation, "q8_0_gemv");
    }

    #[test]
    fn q8_0_tensorops_paired_tiles_are_bit_identical() {
        let d = MetalDevice::new().unwrap();
        if !d.mpp_projection() {
            return;
        }
        let bits = |t: &Tensor| t.to_f32().iter().map(|v| v.to_bits()).collect::<Vec<_>>();
        let check = |weight: &QuantizedMatrix, m: usize, k: usize, kernel: &str| {
            let x = Tensor::from_f32(&d, [m, k], DType::BF16, &input(m, k)).unwrap();
            d.set_dense_mpp_tile_pairs(false).unwrap();
            let control = d.project_quantized(&x, weight).unwrap();
            assert_eq!(control.metrics.operation, kernel);
            d.set_dense_mpp_tile_pairs(true).unwrap();
            let paired = d.project_quantized(&x, weight).unwrap();
            assert_eq!(
                bits(&paired.tensor),
                bits(&control.tensor),
                "{kernel} M={m}"
            );
        };
        let weight = packed(&d, 517, 384);
        check(&weight, 200, 384, "q8_0_gemm_mpp");
        check(&weight, 1025, 384, "q8_0_gemm_mpp_k64");
    }

    #[test]
    fn q8_0_tensorops_fast_dequant_is_bit_identical() {
        let d = MetalDevice::new().unwrap();
        if !d.mpp_projection() {
            return;
        }
        let bits = |t: &Tensor| t.to_f32().iter().map(|v| v.to_bits()).collect::<Vec<_>>();
        // K=128 tile below 512 rows; K=64 tile at 513 rows; column and row tails.
        for (m, n, k, kernel) in [
            (37, 133, 384, "q8_0_gemm_mpp"),
            (513, 65, 384, "q8_0_gemm_mpp_k64"),
        ] {
            let weight = packed(&d, n, k);
            let x = Tensor::from_f32(&d, [m, k], DType::BF16, &input(m, k)).unwrap();
            d.set_mpp_fast_dequant(false).unwrap();
            let control = d.project_quantized(&x, &weight).unwrap();
            assert_eq!(control.metrics.operation, kernel);
            d.set_mpp_fast_dequant(true).unwrap();
            let fast = d.project_quantized(&x, &weight).unwrap();
            assert_eq!(bits(&fast.tensor), bits(&control.tensor), "{kernel}");
        }
    }

    #[test]
    fn q8_0_mpp_gemm_dequantizes_only_a_tiled_weight_block() {
        let d = MetalDevice::new().unwrap();
        if !d.mpp_projection() {
            return;
        }
        let (small_n, small_k) = (65, 256);
        let small_weight = packed(&d, small_n, small_k);
        for m in [11, 12, 15] {
            let values = input(m, small_k);
            let x = Tensor::from_f32(&d, [m, small_k], DType::BF16, &values).unwrap();
            let rounded_input = x.to_f32();
            let reference = expected(&rounded_input, m, small_n, small_k);
            let reference = Tensor::from_f32(&d, [m, small_n], DType::BF16, &reference).unwrap();
            let output = d.project_quantized(&x, &small_weight).unwrap();
            assert_eq!(
                output.metrics.operation,
                if m < 12 { "q8_0_gemm" } else { "q8_0_gemm_mpp" },
                "M={m}"
            );
            for (index, (actual, expected)) in output
                .tensor
                .to_f32()
                .iter()
                .zip(reference.to_f32())
                .enumerate()
            {
                assert!(
                    (actual - expected).abs() <= 0.06,
                    "M={m} index {index}: actual={actual}, expected={expected}"
                );
            }
        }
        assert!(d.q8_0_mpp_tile_k64(512));
        assert!(!d.q8_0_mpp_tile_k64(128));
        d.set_q8_0_mpp_tile_k64(false).unwrap();
        let (m, n, k) = (515, 65, 256);
        let weight = packed(&d, n, k);
        let values = input(m, k);
        let x = Tensor::from_f32(&d, [m, k], DType::BF16, &values).unwrap();
        let rounded_input = x.to_f32();
        let reference = expected(&rounded_input, m, n, k);
        let reference = Tensor::from_f32(&d, [m, n], DType::BF16, &reference).unwrap();
        let output = d.project_quantized(&x, &weight).unwrap();
        assert_eq!(output.metrics.operation, "q8_0_gemm_mpp");
        for (index, (actual, expected)) in output
            .tensor
            .to_f32()
            .iter()
            .zip(reference.to_f32())
            .enumerate()
        {
            assert!(
                (actual - expected).abs() <= 0.06,
                "index {index}: actual={actual}, expected={expected}"
            );
        }

        d.set_q8_0_mpp_tile_k64(true).unwrap();
        let candidate = d.project_quantized(&x, &weight).unwrap();
        assert_eq!(candidate.metrics.operation, "q8_0_gemm_mpp_k64");
        for (index, (actual, expected)) in candidate
            .tensor
            .to_f32()
            .iter()
            .zip(reference.to_f32())
            .enumerate()
        {
            assert!(
                (actual - expected).abs() <= 0.06,
                "K=64 index {index}: actual={actual}, expected={expected}"
            );
        }
    }

    #[test]
    fn q8_0_embedding_gather_decodes_only_requested_rows() {
        let d = MetalDevice::new().unwrap();
        let vocab = 5;
        let hidden = 64;
        let weight = packed(&d, vocab, hidden);
        let ids = [4, 0, 3];
        let actual = d
            .embedding_gather_quantized(&weight, &ids, DType::BF16)
            .unwrap()
            .tensor
            .to_f32();
        let expected = ids
            .iter()
            .flat_map(|&row| {
                (0..hidden).map(move |column| DType::BF16.round(q8_value(row as usize, column)))
            })
            .collect::<Vec<_>>();
        assert_eq!(actual, expected);
    }

    #[test]
    fn q8_0_rejects_columns_that_cannot_form_upstream_blocks() {
        let d = MetalDevice::new().unwrap();
        let result = QuantizedMatrix::from_reader(&d, 3, 31, QuantizationFormat::Q8_0, |_| Ok(()));
        assert!(result.is_err());
    }
}

#[cfg(test)]
mod q4_0_tests {
    use super::*;
    use crate::quantization::{QuantizationFormat, QuantizedMatrix};

    fn packed(device: &MetalDevice, rows: usize, columns: usize) -> QuantizedMatrix {
        let mut bytes = Vec::new();
        for row in 0..rows {
            for block in 0..columns / 32 {
                bytes.extend_from_slice(&half::f16::from_f32(0.125).to_bits().to_le_bytes());
                for byte in 0..16 {
                    let low = (row * 5 + block * 2 + byte * 3) % 16;
                    let high = (row * 9 + block * 3 + byte * 7) % 16;
                    bytes.push((low | (high << 4)) as u8);
                }
            }
        }
        let expected_len = rows * (columns / 32) * 18;
        assert_eq!(bytes.len(), expected_len);
        let matrix =
            QuantizedMatrix::from_reader(device, rows, columns, QuantizationFormat::Q4_0, |dst| {
                dst.copy_from_slice(&bytes);
                Ok(())
            })
            .unwrap();
        assert_eq!(matrix.byte_size(), expected_len);
        matrix
    }

    fn q4_value(row: usize, column: usize) -> f32 {
        let block = column / 32;
        let within = column % 32;
        let byte = within % 16;
        let q = if within < 16 {
            (row * 5 + block * 2 + byte * 3) % 16
        } else {
            (row * 9 + block * 3 + byte * 7) % 16
        };
        0.125 * (q as f32 - 8.0)
    }

    fn input(rows: usize, columns: usize) -> Vec<f32> {
        (0..rows * columns)
            .map(|i| ((i * 17 + 5) % 89) as f32 / 61.0 - 0.7)
            .collect()
    }

    fn expected(input: &[f32], m: usize, n: usize, k: usize) -> Vec<f32> {
        (0..m)
            .flat_map(|row| {
                (0..n).map(move |column| {
                    (0..k)
                        .map(|i| input[row * k + i] * q4_value(column, i))
                        .sum::<f32>()
                })
            })
            .collect()
    }

    #[test]
    fn q4_0_gemv_and_gemm_cover_output_and_batch_tails() {
        let d = MetalDevice::new().unwrap();
        let (n, k) = (5, 64);
        let weight = packed(&d, n, k);
        for m in [1, 2, 3, 4, 5, 7] {
            let values = input(m, k);
            let x = Tensor::from_f32(&d, [m, k], DType::F32, &values).unwrap();
            let actual = d.project_quantized(&x, &weight).unwrap().tensor.to_f32();
            let expected = expected(&values, m, n, k);
            for (index, (actual, expected)) in actual.iter().zip(&expected).enumerate() {
                assert!(
                    (actual - expected).abs() <= 2.0e-5,
                    "index {index}: actual={actual}, expected={expected}"
                );
            }
        }
    }

    #[test]
    fn q4_0_gemv_8rows_reuses_inputs_and_covers_output_tail() {
        let d = MetalDevice::new().unwrap();
        let (n, k) = (131, 64);
        let weight = packed(&d, n, k);
        let values = input(1, k);
        let x = Tensor::from_f32(&d, [1, k], DType::F32, &values).unwrap();
        assert!(d.use_q4_0_gemv_8rows(n));
        let output = d.project_quantized(&x, &weight).unwrap();
        assert_eq!(output.metrics.operation, "q4_0_gemv_8rows");
        let expected = expected(&values, 1, n, k);
        for (index, (actual, expected)) in output.tensor.to_f32().iter().zip(expected).enumerate() {
            assert!(
                (actual - expected).abs() <= 2.0e-5,
                "index {index}: actual={actual}, expected={expected}"
            );
        }
    }

    #[test]
    fn q4_0_tensorops_paired_tiles_are_bit_identical() {
        let d = MetalDevice::new().unwrap();
        if !d.mpp_projection() {
            return;
        }
        let bits = |t: &Tensor| t.to_f32().iter().map(|v| v.to_bits()).collect::<Vec<_>>();
        let weight = packed(&d, 517, 384);
        let x = Tensor::from_f32(&d, [600, 384], DType::BF16, &input(600, 384)).unwrap();
        d.set_dense_mpp_tile_pairs(false).unwrap();
        let control = d.project_quantized(&x, &weight).unwrap();
        assert_eq!(control.metrics.operation, "q4_0_gemm_mpp");
        d.set_dense_mpp_tile_pairs(true).unwrap();
        let paired = d.project_quantized(&x, &weight).unwrap();
        assert_eq!(bits(&paired.tensor), bits(&control.tensor));
    }

    #[test]
    fn q4_0_mpp_gemm_decodes_a_bounded_weight_tile() {
        let d = MetalDevice::new().unwrap();
        if !d.mpp_projection() {
            return;
        }
        let (m, n, k) = (35, 65, 128);
        let weight = packed(&d, n, k);
        let values = input(m, k);
        let x = Tensor::from_f32(&d, [m, k], DType::BF16, &values).unwrap();
        let rounded_input = x.to_f32();
        let reference = expected(&rounded_input, m, n, k);
        let reference = Tensor::from_f32(&d, [m, n], DType::BF16, &reference).unwrap();
        let output = d.project_quantized(&x, &weight).unwrap();
        assert_eq!(output.metrics.operation, "q4_0_gemm_mpp");
        for (index, (actual, expected)) in output
            .tensor
            .to_f32()
            .iter()
            .zip(reference.to_f32())
            .enumerate()
        {
            assert!(
                (actual - expected).abs() <= 0.06,
                "index {index}: actual={actual}, expected={expected}"
            );
        }
    }

    #[test]
    fn q4_0_small_batch_mpp_covers_row_and_column_tails() {
        let d = MetalDevice::new().unwrap();
        if !d.mpp_projection() {
            return;
        }
        let (n, k) = (65, 128);
        let weight = packed(&d, n, k);
        for m in [4, 8, 15] {
            let values = input(m, k);
            let x = Tensor::from_f32(&d, [m, k], DType::BF16, &values).unwrap();
            let expected = expected(&x.to_f32(), m, n, k);
            let output = d.project_quantized(&x, &weight).unwrap();
            assert_eq!(output.metrics.operation, "q4_0_gemm_mpp");
            for (index, (actual, expected)) in
                output.tensor.to_f32().iter().zip(&expected).enumerate()
            {
                assert!(
                    (actual - expected).abs() <= 0.06,
                    "M={m} Q4_0 MPP index {index}: actual={actual}, expected={expected}"
                );
            }
        }
        d.set_gguf_mpp_min_rows(2).unwrap();
        let values = input(1, k);
        let x = Tensor::from_f32(&d, [1, k], DType::BF16, &values).unwrap();
        let output = d.project_quantized(&x, &weight).unwrap();
        assert_eq!(output.metrics.operation, "q4_0_gemv");
        assert_eq!(d.gguf_mpp_min_rows(), Some(2));
        d.clear_gguf_mpp_min_rows().unwrap();
        assert_eq!(d.gguf_mpp_min_rows(), None);
    }

    #[test]
    fn q4_0_gemv_honors_signed_block_scale() {
        let d = MetalDevice::new().unwrap();
        let mut bytes = Vec::with_capacity(18);
        bytes.extend_from_slice(&half::f16::from_f32(-0.25).to_bits().to_le_bytes());
        bytes.extend([0x87; 16]);
        let weight =
            QuantizedMatrix::from_reader(&d, 1, 32, QuantizationFormat::Q4_0, |destination| {
                destination.copy_from_slice(&bytes);
                Ok(())
            })
            .unwrap();
        let input = Tensor::from_f32(&d, [1, 32], DType::F32, &[1.0; 32]).unwrap();
        let result = d.project_quantized(&input, &weight).unwrap();
        assert_eq!(result.tensor.to_f32(), [4.0]);
    }

    #[test]
    fn q4_0_embedding_gather_decodes_only_requested_rows() {
        let d = MetalDevice::new().unwrap();
        let (vocab, hidden) = (5, 64);
        let weight = packed(&d, vocab, hidden);
        let ids = [4, 0, 3];
        let actual = d
            .embedding_gather_quantized(&weight, &ids, DType::BF16)
            .unwrap()
            .tensor
            .to_f32();
        let expected = ids
            .iter()
            .flat_map(|&row| {
                (0..hidden).map(move |column| DType::BF16.round(q4_value(row as usize, column)))
            })
            .collect::<Vec<_>>();
        assert_eq!(actual, expected);
    }

    #[test]
    fn q4_0_rejects_columns_that_do_not_form_ggml_blocks() {
        let d = MetalDevice::new().unwrap();
        assert!(
            QuantizedMatrix::from_reader(&d, 3, 31, QuantizationFormat::Q4_0, |_| Ok(())).is_err()
        );
    }
}

#[cfg(test)]
mod q5_tests {
    use super::*;
    use crate::quantization::{QuantizationFormat, QuantizedMatrix};

    fn block_parameters(format: QuantizationFormat, row: usize, block: usize) -> (f32, f32) {
        match format {
            QuantizationFormat::Q5_0 => {
                let d = if (row + block).is_multiple_of(2) {
                    0.0625
                } else {
                    -0.03125
                };
                (d, 0.0)
            }
            QuantizationFormat::Q5_1 => (
                0.03125 + ((row + block) % 3) as f32 * 0.0078125,
                -0.125 + block as f32 * 0.03125,
            ),
            _ => unreachable!(),
        }
    }

    fn qh_byte(row: usize, block: usize, byte: usize) -> u8 {
        ((row * 37 + block * 19 + byte * 53 + 0xa5) & 255) as u8
    }

    fn qs_byte(row: usize, block: usize, byte: usize) -> u8 {
        ((row * 11 + block * 7 + byte * 29 + 0x36) & 255) as u8
    }

    fn packed(
        device: &MetalDevice,
        rows: usize,
        columns: usize,
        format: QuantizationFormat,
    ) -> QuantizedMatrix {
        let bytes = packed_bytes(rows, columns, format);
        QuantizedMatrix::from_reader(device, rows, columns, format, |destination| {
            destination.copy_from_slice(&bytes);
            Ok(())
        })
        .unwrap()
    }

    fn packed_bytes(rows: usize, columns: usize, format: QuantizationFormat) -> Vec<u8> {
        assert!(matches!(
            format,
            QuantizationFormat::Q5_0 | QuantizationFormat::Q5_1
        ));
        let block_bytes = format.block_bytes();
        let mut bytes = Vec::with_capacity(rows * columns / 32 * block_bytes);
        for row in 0..rows {
            for block_index in 0..columns / 32 {
                let (d, minimum) = block_parameters(format, row, block_index);
                bytes.extend_from_slice(&half::f16::from_f32(d).to_bits().to_le_bytes());
                if format == QuantizationFormat::Q5_1 {
                    bytes.extend_from_slice(&half::f16::from_f32(minimum).to_bits().to_le_bytes());
                }
                let qh_offset = if format == QuantizationFormat::Q5_0 {
                    2
                } else {
                    4
                };
                for byte in 0..4 {
                    bytes.push(qh_byte(row, block_index, byte));
                }
                for byte in 0..16 {
                    bytes.push(qs_byte(row, block_index, byte));
                }
                assert_eq!(bytes.len() % block_bytes, 0);
                assert_eq!(
                    bytes.len() - (row * (columns / 32) + block_index) * block_bytes,
                    block_bytes
                );
                assert_eq!(qh_offset + 4 + 16, block_bytes);
            }
        }
        bytes
    }

    fn q5_value(row: usize, column: usize, format: QuantizationFormat) -> f32 {
        let block = column / 32;
        let within = column % 32;
        let qs = qs_byte(row, block, within & 15);
        let nibble = if within < 16 { qs & 15 } else { qs >> 4 };
        let high_bit = (qh_byte(row, block, within >> 3) >> (within & 7)) & 1;
        let q = u32::from(nibble) + (u32::from(high_bit) << 4);
        let (d, minimum) = block_parameters(format, row, block);
        match format {
            QuantizationFormat::Q5_0 => d * (q as f32 - 16.0),
            QuantizationFormat::Q5_1 => d * q as f32 + minimum,
            _ => unreachable!(),
        }
    }

    fn input(rows: usize, columns: usize) -> Vec<f32> {
        (0..rows * columns)
            .map(|i| ((i * 23 + 5) % 97) as f32 / 89.0 - 0.52)
            .collect()
    }

    fn expected(
        input: &[f32],
        m: usize,
        n: usize,
        k: usize,
        format: QuantizationFormat,
    ) -> Vec<f32> {
        (0..m)
            .flat_map(|row| {
                (0..n).map(move |column| {
                    (0..k)
                        .map(|index| input[row * k + index] * q5_value(column, index, format))
                        .sum::<f32>()
                })
            })
            .collect()
    }

    #[test]
    fn q5_0_and_q5_1_gemv_and_gemm_cover_output_and_batch_tails() {
        let d = MetalDevice::new().unwrap();
        for format in [QuantizationFormat::Q5_0, QuantizationFormat::Q5_1] {
            let (n, k) = (5, 64);
            let weight = packed(&d, n, k, format);
            assert_eq!(weight.byte_size(), n * 2 * format.block_bytes());
            for m in [1, 2, 3, 4, 5, 7] {
                let values = input(m, k);
                let x = Tensor::from_f32(&d, [m, k], DType::F32, &values).unwrap();
                let actual = d.project_quantized(&x, &weight).unwrap().tensor.to_f32();
                let expected = expected(&values, m, n, k, format);
                for (index, (actual, expected)) in actual.iter().zip(&expected).enumerate() {
                    assert!(
                        (actual - expected).abs() <= 3.0e-5,
                        "{format:?} index {index}: actual={actual}, expected={expected}"
                    );
                }
            }
        }
    }

    #[test]
    fn q5_0_gemv_n4_reuses_activations_and_covers_output_tail() {
        let d = MetalDevice::new().unwrap();
        d.set_q5_0_gemv_ksplit(0).unwrap();
        assert!(d.use_q5_0_gemv_n4(128));
        assert!(!d.use_q5_0_gemv_n4(127));
        for (n, k) in [(131, 512), (4864, 896)] {
            let weight = packed(&d, n, k, QuantizationFormat::Q5_0);
            let values = input(1, k);
            let x = Tensor::from_f32(&d, [1, k], DType::F32, &values).unwrap();
            let output = d.project_quantized(&x, &weight).unwrap();
            assert_eq!(output.metrics.operation, "q5_0_gemv_n4");
            let expected = expected(&values, 1, n, k, QuantizationFormat::Q5_0);
            for (index, (actual, expected)) in
                output.tensor.to_f32().iter().zip(expected).enumerate()
            {
                assert!(
                    (actual - expected).abs() <= 5.0e-5,
                    "Q5_0 ({n}, {k}) index {index}: actual={actual}, expected={expected}"
                );
            }
        }
    }

    #[test]
    fn q5_0_swiglu_matches_separate_kernels_bitwise() {
        let d = MetalDevice::new().unwrap();
        for (n, k) in [(131, 896), (4864, 896), (6, 1440)] {
            let gate = packed(&d, n, k, QuantizationFormat::Q5_0);
            let up = {
                // A second, distinct weight: rows shifted by one.
                let bytes = packed_bytes(n + 1, k, QuantizationFormat::Q5_0);
                let row = k / 32 * 22;
                QuantizedMatrix::from_reader(&d, n, k, QuantizationFormat::Q5_0, |dst| {
                    dst.copy_from_slice(&bytes[row..]);
                    Ok(())
                })
                .unwrap()
            };
            let x = Tensor::from_f32(&d, [1, k], DType::BF16, &input(1, k)).unwrap();
            for simds in [1, 2, 4, 8] {
                d.set_q5_0_gemv_ksplit(simds).unwrap();
                let g = d.project_quantized(&x, &gate).unwrap().tensor;
                let u = d.project_quantized(&x, &up).unwrap().tensor;
                let separate = d.silu_mul(&g, &u).unwrap().tensor.to_f32();
                let fused = d.swiglu_q5_0(&x, &gate, &up).unwrap();
                assert_eq!(fused.metrics.operation, "q5_0_gemv_swiglu");
                let fused = fused.tensor.to_f32();
                for (index, (a, b)) in fused.iter().zip(&separate).enumerate() {
                    assert_eq!(
                        a.to_bits(),
                        b.to_bits(),
                        "({n}, {k}) simds {simds} index {index}"
                    );
                }
            }
            d.set_q5_0_gemv_ksplit(2).unwrap();
        }
    }

    /// Decode MLP front half at Qwen2.5-0.5B shape: separate gate/up/silu_mul
    /// against the fused SwiGLU kernel, 24 distinct weight pairs.
    #[test]
    #[ignore = "microbenchmark; run with --ignored --nocapture"]
    fn bench_q5_0_swiglu() {
        let d = MetalDevice::new().unwrap();
        let (n, k) = (4864, 896);
        let pairs: Vec<_> = (0..24)
            .map(|_| {
                (
                    packed(&d, n, k, QuantizationFormat::Q5_0),
                    packed(&d, n, k, QuantizationFormat::Q5_0),
                )
            })
            .collect();
        let x = Tensor::from_f32(&d, [1, k], DType::BF16, &input(1, k)).unwrap();
        for fused in [false, true] {
            let mut best = f64::MAX;
            for _ in 0..7 {
                let start = std::time::Instant::now();
                let execution = d.execution().unwrap();
                let mut last = None;
                for _ in 0..8 {
                    for (gate, up) in &pairs {
                        last = Some(if fused {
                            d.swiglu_q5_0(&x, gate, up).unwrap().tensor
                        } else {
                            let g = d.project_quantized(&x, gate).unwrap().tensor;
                            let u = d.project_quantized(&x, up).unwrap().tensor;
                            d.silu_mul(&g, &u).unwrap().tensor
                        });
                    }
                }
                drop(execution);
                std::hint::black_box(last.unwrap().to_f32());
                best = best.min(start.elapsed().as_secs_f64() / 8.0);
            }
            println!(
                "swiglu fused={fused}: {:.1} us/layer",
                best * 1e6 / pairs.len() as f64
            );
        }
    }

    #[test]
    fn q5_0_gemv_ksplit_covers_row_and_block_tails() {
        let d = MetalDevice::new().unwrap();
        assert!(d.set_q5_0_gemv_ksplit(3).is_err());
        // 131 rows leave a partial four-row tile; 28 and 45 blocks leave
        // partial 16-block strides for every SIMD group count.
        for (n, k) in [(131, 896), (5, 1440), (128, 512)] {
            let weight = packed(&d, n, k, QuantizationFormat::Q5_0);
            let values = input(1, k);
            let x = Tensor::from_f32(&d, [1, k], DType::F32, &values).unwrap();
            let expected = expected(&values, 1, n, k, QuantizationFormat::Q5_0);
            for simds in [1, 2, 4, 8] {
                d.set_q5_0_gemv_ksplit(simds).unwrap();
                let output = d.project_quantized(&x, &weight).unwrap();
                assert_eq!(output.metrics.operation, "q5_0_gemv_ksplit");
                for (index, (actual, expected)) in
                    output.tensor.to_f32().iter().zip(&expected).enumerate()
                {
                    assert!(
                        (actual - expected).abs() <= 5.0e-5,
                        "Q5_0 K-split {simds} ({n}, {k}) index {index}: actual={actual}, expected={expected}"
                    );
                }
            }
        }
    }

    /// Steady-state Q5_0 GEMV bandwidth at Qwen2.5-0.5B shapes: 24 distinct
    /// weights (like 24 layers, beyond the system cache) projected back to back.
    #[test]
    #[ignore = "microbenchmark; run with --ignored --nocapture"]
    fn bench_q5_0_gemv_bandwidth() {
        let d = MetalDevice::new().unwrap();
        for (label, n, k) in [("gate", 4864, 896), ("q", 896, 896), ("kv", 128, 896)] {
            let x = Tensor::from_f32(&d, [1, k], DType::BF16, &input(1, k)).unwrap();
            for simds in [0usize, 2, 4] {
                let bytes = packed_bytes(n, k, QuantizationFormat::Q5_0);
                let weights: Vec<_> = (0..24)
                    .map(|_| {
                        QuantizedMatrix::from_reader(&d, n, k, QuantizationFormat::Q5_0, |dst| {
                            dst.copy_from_slice(&bytes);
                            Ok(())
                        })
                        .unwrap()
                    })
                    .collect();
                d.set_q5_0_gemv_ksplit(simds).unwrap();
                let check = d
                    .project_quantized(&x, &weights[0])
                    .unwrap()
                    .tensor
                    .to_f32();
                println!(
                    "{label} simds={simds} checksum={:08x}",
                    check
                        .iter()
                        .fold(0u32, |h, v| h.rotate_left(5) ^ v.to_bits())
                );
                let bytes = (n * k / 32 * 22 * weights.len()) as f64;
                let mut best = f64::MAX;
                for _ in 0..7 {
                    let start = std::time::Instant::now();
                    let execution = d.execution().unwrap();
                    let mut last = None;
                    for _ in 0..8 {
                        for weight in &weights {
                            last = Some(d.project_quantized(&x, weight).unwrap().tensor);
                        }
                    }
                    drop(execution);
                    std::hint::black_box(last.unwrap().to_f32());
                    best = best.min(start.elapsed().as_secs_f64() / 8.0);
                }
                println!(
                    "{label} simds={simds} n={n} k={k}: {:.1} us/projection, {:.1} GB/s",
                    best * 1e6 / weights.len() as f64,
                    bytes / best / 1e9
                );
            }
            d.set_q5_0_gemv_ksplit(2).unwrap();
        }
    }

    #[test]
    fn q5_1_gemv_n4_reuses_activations_and_covers_output_tail() {
        let d = MetalDevice::new().unwrap();
        assert!(d.use_q5_1_gemv_n4(128));
        assert!(!d.use_q5_1_gemv_n4(127));
        d.set_q5_1_gemv_n4(false).unwrap();
        assert!(!d.use_q5_1_gemv_n4(128));
        d.set_q5_1_gemv_n4(true).unwrap();
        for (n, k) in [(131, 512), (4864, 896)] {
            let weight = packed(&d, n, k, QuantizationFormat::Q5_1);
            let values = input(1, k);
            let x = Tensor::from_f32(&d, [1, k], DType::F32, &values).unwrap();
            let output = d.project_quantized(&x, &weight).unwrap();
            assert_eq!(output.metrics.operation, "q5_1_gemv_n4");
            let expected = expected(&values, 1, n, k, QuantizationFormat::Q5_1);
            for (index, (actual, expected)) in
                output.tensor.to_f32().iter().zip(expected).enumerate()
            {
                assert!(
                    (actual - expected).abs() <= 5.0e-5,
                    "Q5_1 ({n}, {k}) index {index}: actual={actual}, expected={expected}"
                );
            }
        }
    }

    #[test]
    fn q5_1_tensorops_paired_tiles_are_bit_identical() {
        let d = MetalDevice::new().unwrap();
        if !d.mpp_projection() {
            return;
        }
        let bits = |t: &Tensor| t.to_f32().iter().map(|v| v.to_bits()).collect::<Vec<_>>();
        let check = |weight: &QuantizedMatrix, m: usize, k: usize, kernel: &str| {
            let x = Tensor::from_f32(&d, [m, k], DType::BF16, &input(m, k)).unwrap();
            d.set_dense_mpp_tile_pairs(false).unwrap();
            let control = d.project_quantized(&x, weight).unwrap();
            assert_eq!(control.metrics.operation, kernel);
            d.set_dense_mpp_tile_pairs(true).unwrap();
            let paired = d.project_quantized(&x, weight).unwrap();
            assert_eq!(
                bits(&paired.tensor),
                bits(&control.tensor),
                "{kernel} M={m}"
            );
        };
        let weight = packed(&d, 517, 384, QuantizationFormat::Q5_1);
        check(&weight, 200, 384, "q5_1_gemm_mpp");
        check(&weight, 1025, 384, "q5_1_gemm_mpp_k64");
        let weight = packed(&d, 517, 384, QuantizationFormat::Q5_0);
        check(&weight, 600, 384, "q5_0_gemm_mpp");
    }

    #[test]
    fn q5_0_tensorops_fast_dequant_and_pairs_are_bit_identical() {
        let d = MetalDevice::new().unwrap();
        if !d.mpp_projection() {
            return;
        }
        let bits = |t: &Tensor| t.to_f32().iter().map(|v| v.to_bits()).collect::<Vec<_>>();
        for m in [37, 600] {
            let (n, k) = (133, 384);
            let weight = packed(&d, n, k, QuantizationFormat::Q5_0);
            let x = Tensor::from_f32(&d, [m, k], DType::BF16, &input(m, k)).unwrap();
            d.set_mpp_fast_dequant(false).unwrap();
            d.set_dense_mpp_tile_pairs(false).unwrap();
            let control = d.project_quantized(&x, &weight).unwrap();
            assert_eq!(control.metrics.operation, "q5_0_gemm_mpp");
            for (fast, pairs) in [(true, false), (false, true), (true, true)] {
                d.set_mpp_fast_dequant(fast).unwrap();
                d.set_dense_mpp_tile_pairs(pairs).unwrap();
                let candidate = d.project_quantized(&x, &weight).unwrap();
                assert_eq!(
                    bits(&candidate.tensor),
                    bits(&control.tensor),
                    "M={m} fast={fast} pairs={pairs}"
                );
            }
        }
    }

    #[test]
    fn q5_0_and_q5_1_mpp_gemm_cover_batch_and_output_tails() {
        let d = MetalDevice::new().unwrap();
        if !d.mpp_projection() {
            return;
        }
        let (n, k) = (65, 128);
        d.set_gguf_mpp_min_rows(8).unwrap();
        for format in [QuantizationFormat::Q5_0, QuantizationFormat::Q5_1] {
            let weight = packed(&d, n, k, format);
            for m in [8, 15, 35] {
                let values = input(m, k);
                let x = Tensor::from_f32(&d, [m, k], DType::BF16, &values).unwrap();
                let reference_values = expected(&x.to_f32(), m, n, k, format);
                let reference =
                    Tensor::from_f32(&d, [m, n], DType::BF16, &reference_values).unwrap();
                let output = d.project_quantized(&x, &weight).unwrap();
                let expected_kernel = match format {
                    QuantizationFormat::Q5_0 => "q5_0_gemm_mpp",
                    QuantizationFormat::Q5_1 => "q5_1_gemm_mpp",
                    _ => unreachable!(),
                };
                assert_eq!(output.metrics.operation, expected_kernel);
                for (index, (actual, expected)) in output
                    .tensor
                    .to_f32()
                    .iter()
                    .zip(reference.to_f32())
                    .enumerate()
                {
                    assert!(
                        (actual - expected).abs() <= 0.08,
                        "{format:?} M={m} index {index}: actual={actual}, expected={expected}"
                    );
                }
            }
        }
    }

    #[test]
    fn q5_1_mpp_min_rows_default_only_changes_q5_1_and_keeps_m1_gemv() {
        let d = MetalDevice::new().unwrap();
        if !d.mpp_projection() {
            return;
        }
        let (n, k, m) = (65, 128, 8);
        assert_eq!(d.gguf_mpp_min_rows(), None);
        assert_eq!(d.gguf_mpp_q5_1_min_rows(), None);

        let values = input(m, k);
        let x = Tensor::from_f32(&d, [m, k], DType::BF16, &values).unwrap();
        for (format, expected_kernel) in [
            (QuantizationFormat::Q5_1, "q5_1_gemm_mpp"),
            (QuantizationFormat::Q5_0, "q5_0_gemm"),
        ] {
            let weight = packed(&d, n, k, format);
            let output = d.project_quantized(&x, &weight).unwrap();
            assert_eq!(output.metrics.operation, expected_kernel);
        }
        let mut q8_bytes = Vec::with_capacity(n * k / 32 * 34);
        for _ in 0..n {
            for _ in 0..k / 32 {
                q8_bytes.extend_from_slice(&half::f16::from_f32(0.03125).to_bits().to_le_bytes());
                q8_bytes.extend(std::iter::repeat_n(0u8, 32));
            }
        }
        let q8_weight =
            QuantizedMatrix::from_reader(&d, n, k, QuantizationFormat::Q8_0, |destination| {
                destination.copy_from_slice(&q8_bytes);
                Ok(())
            })
            .unwrap();
        assert_eq!(
            d.project_quantized(&x, &q8_weight)
                .unwrap()
                .metrics
                .operation,
            "q8_0_gemm"
        );

        d.set_gguf_mpp_q5_1_min_rows(16).unwrap();
        let q5_1_weight = packed(&d, n, k, QuantizationFormat::Q5_1);
        assert_eq!(
            d.project_quantized(&x, &q5_1_weight)
                .unwrap()
                .metrics
                .operation,
            "q5_1_gemm"
        );
        d.clear_gguf_mpp_q5_1_min_rows().unwrap();
        assert_eq!(d.gguf_mpp_q5_1_min_rows(), None);

        let values = input(1, k);
        let x = Tensor::from_f32(&d, [1, k], DType::BF16, &values).unwrap();
        assert_eq!(
            d.project_quantized(&x, &q5_1_weight)
                .unwrap()
                .metrics
                .operation,
            "q5_1_gemv"
        );
    }

    #[test]
    fn q5_1_mpp_k64_covers_batch_output_and_k_tiles() {
        let d = MetalDevice::new().unwrap();
        if !d.mpp_projection() {
            return;
        }
        assert!(!d.q5_1_mpp_tile_k64(511));
        assert!(d.q5_1_mpp_tile_k64(512));
        d.set_q5_1_mpp_tile_k64(false).unwrap();
        let (m, n, k) = (515, 65, 256);
        let weight = packed(&d, n, k, QuantizationFormat::Q5_1);
        let values = input(m, k);
        let x = Tensor::from_f32(&d, [m, k], DType::BF16, &values).unwrap();
        let reference_values = expected(&x.to_f32(), m, n, k, QuantizationFormat::Q5_1);
        let reference = Tensor::from_f32(&d, [m, n], DType::BF16, &reference_values).unwrap();

        let control = d.project_quantized(&x, &weight).unwrap();
        assert_eq!(control.metrics.operation, "q5_1_gemm_mpp");
        d.set_q5_1_mpp_tile_k64(true).unwrap();
        let candidate = d.project_quantized(&x, &weight).unwrap();
        assert_eq!(candidate.metrics.operation, "q5_1_gemm_mpp_k64");
        for (index, (actual, expected)) in candidate
            .tensor
            .to_f32()
            .iter()
            .zip(reference.to_f32())
            .enumerate()
        {
            assert!(
                (actual - expected).abs() <= 0.08,
                "Q5_1 K=64 index {index}: actual={actual}, expected={expected}"
            );
        }
    }

    #[test]
    fn q5_0_and_q5_1_embedding_gather_decode_selected_rows() {
        let d = MetalDevice::new().unwrap();
        for format in [QuantizationFormat::Q5_0, QuantizationFormat::Q5_1] {
            let (vocab, hidden) = (5, 64);
            let weight = packed(&d, vocab, hidden, format);
            let ids = [4, 0, 3];
            let actual = d
                .embedding_gather_quantized(&weight, &ids, DType::BF16)
                .unwrap()
                .tensor
                .to_f32();
            let expected = ids
                .iter()
                .flat_map(|&row| {
                    (0..hidden).map(move |column| {
                        DType::BF16.round(q5_value(row as usize, column, format))
                    })
                })
                .collect::<Vec<_>>();
            assert_eq!(actual, expected, "{format:?}");
        }
    }

    #[test]
    fn q5_0_and_q5_1_reject_partial_blocks() {
        let d = MetalDevice::new().unwrap();
        for format in [QuantizationFormat::Q5_0, QuantizationFormat::Q5_1] {
            assert!(QuantizedMatrix::from_reader(&d, 3, 31, format, |_| Ok(())).is_err());
        }
    }
}

#[cfg(test)]
mod qk_tests {
    use super::*;
    use crate::quantization::{QuantizationFormat, QuantizedExpertMatrix, QuantizedMatrix};

    fn scales(row: usize, block: usize) -> [u8; 12] {
        std::array::from_fn(|i| ((row * 47 + block * 31 + i * 53 + 0x9d) & 255) as u8)
    }

    fn qbyte(row: usize, block: usize, index: usize) -> u8 {
        ((row * 17 + block * 43 + index * 29 + 0x6b) & 255) as u8
    }

    fn parameters(format: QuantizationFormat, row: usize, block: usize) -> (f32, f32) {
        match format {
            QuantizationFormat::Q4_K => (0.00390625 + row as f32 * 0.00048828125, 0.015625),
            QuantizationFormat::Q5_K => (0.001953125 + block as f32 / 4096.0, 0.0078125),
            _ => unreachable!(),
        }
    }

    fn packed(
        device: &MetalDevice,
        rows: usize,
        columns: usize,
        format: QuantizationFormat,
    ) -> QuantizedMatrix {
        assert!(matches!(
            format,
            QuantizationFormat::Q4_K | QuantizationFormat::Q5_K | QuantizationFormat::Q6_K
        ));
        let mut bytes = Vec::with_capacity(rows * columns / 256 * format.block_bytes());
        for row in 0..rows {
            for block_index in 0..columns / 256 {
                let block_start = bytes.len();
                if format == QuantizationFormat::Q6_K {
                    for index in 0..128 {
                        bytes.push(qbyte(row, block_index, index));
                    }
                    for index in 0..64 {
                        bytes.push(qbyte(row + 5, block_index, index));
                    }
                    for index in 0..16 {
                        bytes
                            .push((((row * 7 + block_index * 3 + index * 5) % 15) as i8 - 7) as u8);
                    }
                    let d = 0.015625 + row as f32 * 0.000244140625;
                    bytes.extend_from_slice(&half::f16::from_f32(d).to_bits().to_le_bytes());
                    assert_eq!(bytes.len() - block_start, format.block_bytes());
                    continue;
                }
                let (d, dmin) = parameters(format, row, block_index);
                bytes.extend_from_slice(&half::f16::from_f32(d).to_bits().to_le_bytes());
                bytes.extend_from_slice(&half::f16::from_f32(dmin).to_bits().to_le_bytes());
                bytes.extend_from_slice(&scales(row, block_index));
                match format {
                    QuantizationFormat::Q4_K => {}
                    QuantizationFormat::Q5_K => {
                        for index in 0..32 {
                            bytes.push(qbyte(row + 7, block_index, index));
                        }
                    }
                    _ => unreachable!(),
                }
                for index in 0..128 {
                    bytes.push(qbyte(row, block_index, index));
                }
                assert_eq!(bytes.len() - block_start, format.block_bytes());
            }
        }
        QuantizedMatrix::from_reader(device, rows, columns, format, |destination| {
            destination.copy_from_slice(&bytes);
            Ok(())
        })
        .unwrap()
    }

    fn unpack_scale_min(scales: &[u8], group: usize) -> (u32, u32) {
        if group < 4 {
            (
                u32::from(scales[group] & 63),
                u32::from(scales[group + 4] & 63),
            )
        } else {
            (
                u32::from(scales[group + 4] & 15) | (u32::from(scales[group - 4] >> 6) << 4),
                u32::from(scales[group + 4] >> 4) | (u32::from(scales[group] >> 6) << 4),
            )
        }
    }

    fn qk_value(row: usize, column: usize, format: QuantizationFormat) -> f32 {
        let block = column / 256;
        let within = column % 256;
        let group = within / 32;
        let lane = within % 32;
        if format == QuantizationFormat::Q6_K {
            let half_block = group / 4;
            let slice = group % 4;
            let ql_index = half_block * 64 + (slice & 1) * 32 + lane;
            let packed = qbyte(row, block, ql_index);
            let low = if slice < 2 { packed & 15 } else { packed >> 4 };
            let high = (qbyte(row + 5, block, half_block * 32 + lane) >> (slice * 2)) & 3;
            let q = i32::from(low | (high << 4)) - 32;
            let scale_index = half_block * 8 + slice * 2 + lane / 16;
            let scale = ((row * 7 + block * 3 + scale_index * 5) % 15) as i8 - 7;
            let d = 0.015625 + row as f32 * 0.000244140625;
            return d * f32::from(scale) * q as f32;
        }
        let chunk = within / 64;
        let packed = qbyte(row, block, chunk * 32 + lane);
        let nibble = if group.is_multiple_of(2) {
            packed & 15
        } else {
            packed >> 4
        };
        let (d, dmin) = parameters(format, row, block);
        let sc = scales(row, block);
        let (scale, minimum) = unpack_scale_min(&sc, group);
        let q = match format {
            QuantizationFormat::Q4_K => u32::from(nibble),
            QuantizationFormat::Q5_K => {
                let high = (qbyte(row + 7, block, lane) >> group) & 1;
                u32::from(nibble) | (u32::from(high) << 4)
            }
            _ => unreachable!(),
        };
        d * scale as f32 * q as f32 - dmin * minimum as f32
    }

    fn input(rows: usize, columns: usize) -> Vec<f32> {
        (0..rows * columns)
            .map(|i| ((i * 41 + 11) % 127) as f32 / 113.0 - 0.55)
            .collect()
    }

    fn expected(
        input: &[f32],
        m: usize,
        n: usize,
        k: usize,
        format: QuantizationFormat,
    ) -> Vec<f32> {
        (0..m)
            .flat_map(|row| {
                (0..n).map(move |column| {
                    (0..k)
                        .map(|index| input[row * k + index] * qk_value(column, index, format))
                        .sum::<f32>()
                })
            })
            .collect()
    }

    #[test]
    fn q4_k_and_q5_k_gemv_gemm_cover_blocks_and_batch_tails() {
        let d = MetalDevice::new().unwrap();
        for format in [QuantizationFormat::Q4_K, QuantizationFormat::Q5_K] {
            let (n, k) = (5, 512);
            let weight = packed(&d, n, k, format);
            assert_eq!(weight.byte_size(), n * 2 * format.block_bytes());
            for m in [1, 2, 3, 4, 5, 7] {
                let values = input(m, k);
                let x = Tensor::from_f32(&d, [m, k], DType::F32, &values).unwrap();
                let actual = d.project_quantized(&x, &weight).unwrap().tensor.to_f32();
                let reference = expected(&values, m, n, k, format);
                for (index, (actual, reference)) in actual.iter().zip(&reference).enumerate() {
                    assert!(
                        (actual - reference).abs() <= 2.0e-4,
                        "{format:?} index {index}: actual={actual}, expected={reference}"
                    );
                }
            }
        }
    }

    #[test]
    fn q4_k_gemv_ksplit_covers_row_and_block_tails() {
        let d = MetalDevice::new().unwrap();
        assert!(d.set_k_quant_gemv_ksplit(3).is_err());
        // 133 rows leave a partial four-row tile; 10 and 19 super-blocks
        // leave partial strides for every SIMD group count.
        for (n, k) in [(133, 2560), (6, 4864)] {
            let weight = packed(&d, n, k, QuantizationFormat::Q4_K);
            let values = input(1, k);
            let x = Tensor::from_f32(&d, [1, k], DType::F32, &values).unwrap();
            let reference = expected(&values, 1, n, k, QuantizationFormat::Q4_K);
            for simds in [1, 2, 4, 8] {
                d.set_k_quant_gemv_ksplit(simds).unwrap();
                let output = d.project_quantized(&x, &weight).unwrap();
                assert_eq!(output.metrics.operation, "q4_k_gemv_ksplit");
                for (index, (actual, reference)) in
                    output.tensor.to_f32().iter().zip(&reference).enumerate()
                {
                    assert!(
                        (actual - reference).abs() <= 1.0e-3 * reference.abs().max(1.0),
                        "Q4_K K-split {simds} ({n}, {k}) index {index}: actual={actual}, expected={reference}"
                    );
                }
            }
        }
    }

    /// Steady-state down-projection GEMV bandwidth (896 x 4864, 24 weights).
    #[test]
    #[ignore = "microbenchmark; run with --ignored --nocapture"]
    fn bench_q4_k_gemv_ksplit_bandwidth() {
        let d = MetalDevice::new().unwrap();
        for (n, k) in [(896, 4864), (2048, 2048), (8192, 2048)] {
            let weights: Vec<_> = (0..24)
                .map(|_| packed(&d, n, k, QuantizationFormat::Q4_K))
                .collect();
            let x = Tensor::from_f32(&d, [1, k], DType::BF16, &input(1, k)).unwrap();
            let bytes = (weights[0].byte_size() * weights.len()) as f64;
            for simds in [0usize, 2, 4, 8] {
                d.set_k_quant_gemv_ksplit(simds).unwrap();
                let mut best = f64::MAX;
                for _ in 0..7 {
                    let start = std::time::Instant::now();
                    let execution = d.execution().unwrap();
                    let mut last = None;
                    for _ in 0..8 {
                        for weight in &weights {
                            last = Some(d.project_quantized(&x, weight).unwrap().tensor);
                        }
                    }
                    drop(execution);
                    std::hint::black_box(last.unwrap().to_f32());
                    best = best.min(start.elapsed().as_secs_f64() / 8.0);
                }
                println!(
                    "Q4_K {n}x{k} simds={simds}: {:.1} us/projection, {:.1} GB/s",
                    best * 1e6 / weights.len() as f64,
                    bytes / best / 1e9
                );
            }
        }
        d.set_k_quant_gemv_ksplit(0).unwrap();
    }

    #[test]
    fn q4_k_eight_row_gemv_reuses_activations_and_covers_output_tail() {
        let d = MetalDevice::new().unwrap();
        d.set_q4_k_factored(false).unwrap();
        assert!(!d.use_q4_k_gemv_8rows(127));
        d.set_q4_k_gemv_8rows(true).unwrap();
        let (n, k) = (133, 512);
        assert!(d.use_q4_k_gemv_8rows(n));
        let weight = packed(&d, n, k, QuantizationFormat::Q4_K);
        let values = input(1, k);
        let x = Tensor::from_f32(&d, [1, k], DType::F32, &values).unwrap();
        let output = d.project_quantized(&x, &weight).unwrap();
        assert_eq!(output.metrics.operation, "q4_k_gemv_8rows");
        let reference = expected(&values, 1, n, k, QuantizationFormat::Q4_K);
        for (index, (actual, reference)) in output.tensor.to_f32().iter().zip(reference).enumerate()
        {
            assert!(
                (actual - reference).abs() <= 2.0e-4,
                "eight-row GEMV index {index}: actual={actual}, expected={reference}"
            );
        }
    }

    #[test]
    fn q5_k_eight_row_gemv_reuses_activations_and_covers_output_tail() {
        let d = MetalDevice::new().unwrap();
        assert!(d.use_q5_k_gemv_8rows(128));
        assert!(!d.use_q5_k_gemv_8rows(127));
        d.set_q5_k_gemv_8rows(false).unwrap();
        assert!(!d.use_q5_k_gemv_8rows(128));
        d.set_q5_k_gemv_8rows(true).unwrap();
        let (n, k) = (133, 512);
        let weight = packed(&d, n, k, QuantizationFormat::Q5_K);
        let values = input(1, k);
        let x = Tensor::from_f32(&d, [1, k], DType::F32, &values).unwrap();
        let output = d.project_quantized(&x, &weight).unwrap();
        assert_eq!(output.metrics.operation, "q5_k_gemv_8rows");
        let actual_values = output.tensor.to_f32();
        let reference = expected(&values, 1, n, k, QuantizationFormat::Q5_K);
        for (index, (actual, reference)) in actual_values.iter().zip(reference).enumerate() {
            assert!(
                (actual - reference).abs() <= 2.0e-4,
                "eight-row Q5_K GEMV index {index}: actual={actual}, expected={reference}"
            );
        }
    }

    #[test]
    fn q4_k_and_q5_k_mpp_gemm_cover_row_output_and_group_tails() {
        let d = MetalDevice::new().unwrap();
        if !d.mpp_projection() {
            return;
        }
        let (n, k) = (65, 512);
        for format in [QuantizationFormat::Q4_K, QuantizationFormat::Q5_K] {
            let weight = packed(&d, n, k, format);
            let batch_sizes: &[usize] = &[4, 8, 15, 35];
            for &m in batch_sizes {
                let values = input(m, k);
                let x = Tensor::from_f32(&d, [m, k], DType::BF16, &values).unwrap();
                let rounded_input = x.to_f32();
                let rounded_input = &rounded_input;
                let expected = (0..m)
                    .flat_map(|row| {
                        (0..n).map(move |column| {
                            (0..k)
                                .map(|index| {
                                    rounded_input[row * k + index]
                                        * DType::BF16.round(qk_value(column, index, format))
                                })
                                .sum::<f32>()
                        })
                    })
                    .collect::<Vec<_>>();
                let reference = Tensor::from_f32(&d, [m, n], DType::BF16, &expected).unwrap();
                let output = d.project_quantized(&x, &weight).unwrap();
                let use_mpp = match format {
                    QuantizationFormat::Q4_K => m >= 4,
                    QuantizationFormat::Q5_K => m >= 8,
                    _ => unreachable!(),
                };
                assert_eq!(
                    output.metrics.operation,
                    match (format, use_mpp) {
                        (QuantizationFormat::Q4_K, true) => "q4_k_gemm_mpp",
                        (QuantizationFormat::Q5_K, true) => "q5_k_gemm_mpp",
                        (QuantizationFormat::Q5_K, false) => "q5_k_gemm",
                        _ => unreachable!(),
                    }
                );
                if !use_mpp {
                    continue;
                }
                for (index, (actual, reference)) in output
                    .tensor
                    .to_f32()
                    .iter()
                    .zip(reference.to_f32())
                    .enumerate()
                {
                    assert!(
                        (actual - reference).abs() <= 0.04,
                        "{format:?} M={m} index {index}: actual={actual}, expected={reference}"
                    );
                }
            }
        }
    }

    #[test]
    fn q4_k_mpp_k64_covers_batch_output_and_k_tiles() {
        let d = MetalDevice::new().unwrap();
        if !d.mpp_projection() {
            return;
        }
        assert!(!d.q4_k_mpp_tile_k64(512));
        assert!(d.q4_k_mpp_tile_k64(1024));
        assert!(!d.q4_k_mpp_tile_m128(1023));
        assert!(d.q4_k_mpp_tile_m128(1024));
        d.set_q4_k_mpp_tile_m128(false).unwrap();
        d.set_q4_k_mpp_tile_k64(false).unwrap();
        let (m, n, k) = (1025, 65, 256);
        let weight = packed(&d, n, k, QuantizationFormat::Q4_K);
        let values = input(m, k);
        let x = Tensor::from_f32(&d, [m, k], DType::BF16, &values).unwrap();
        let rounded_input = x.to_f32();
        let rounded_input = &rounded_input;
        let expected = (0..m)
            .flat_map(|row| {
                (0..n).map(move |column| {
                    (0..k)
                        .map(|index| {
                            rounded_input[row * k + index]
                                * DType::BF16.round(qk_value(
                                    column,
                                    index,
                                    QuantizationFormat::Q4_K,
                                ))
                        })
                        .sum::<f32>()
                })
            })
            .collect::<Vec<_>>();
        let reference = Tensor::from_f32(&d, [m, n], DType::BF16, &expected).unwrap();

        let control = d.project_quantized(&x, &weight).unwrap();
        assert_eq!(control.metrics.operation, "q4_k_gemm_mpp");
        d.set_q4_k_mpp_tile_k64(true).unwrap();
        let candidate = d.project_quantized(&x, &weight).unwrap();
        assert_eq!(candidate.metrics.operation, "q4_k_gemm_mpp_k64");
        for (index, (actual, expected)) in candidate
            .tensor
            .to_f32()
            .iter()
            .zip(reference.to_f32())
            .enumerate()
        {
            assert!(
                (actual - expected).abs() <= 0.04,
                "Q4_K K=64 index {index}: actual={actual}, expected={expected}"
            );
        }

        assert!(!d.q4_k_mpp_tile_m128(1023));
        assert!(!d.q4_k_mpp_tile_m128(1024));
        d.set_q4_k_mpp_tile_m128(true).unwrap();
        assert!(!d.q4_k_mpp_tile_m128(1023));
        assert!(d.q4_k_mpp_tile_m128(1024));
        let candidate_m128 = d.project_quantized(&x, &weight).unwrap();
        assert_eq!(candidate_m128.metrics.operation, "q4_k_gemm_mpp_k64_m128");
        for (index, (actual, expected)) in candidate_m128
            .tensor
            .to_f32()
            .iter()
            .zip(reference.to_f32())
            .enumerate()
        {
            assert!(
                (actual - expected).abs() <= 0.04,
                "Q4_K M=128 K=64 index {index}: actual={actual}, expected={expected}"
            );
        }
    }

    #[test]
    fn q5_k_mpp_k64_covers_batch_output_and_k_tiles() {
        let d = MetalDevice::new().unwrap();
        if !d.mpp_projection() {
            return;
        }
        assert!(!d.q5_k_mpp_tile_k64(1023));
        d.set_q5_k_mpp_tile_k64(true).unwrap();
        assert!(!d.q5_k_mpp_tile_k64(1023));
        assert!(d.q5_k_mpp_tile_k64(1024));

        let (m, n, k) = (1025, 65, 256);
        let weight = packed(&d, n, k, QuantizationFormat::Q5_K);
        let values = input(m, k);
        let x = Tensor::from_f32(&d, [m, k], DType::BF16, &values).unwrap();
        let rounded_input = x.to_f32();
        let rounded_input = &rounded_input;
        let expected = (0..m)
            .flat_map(|row| {
                (0..n).map(move |column| {
                    (0..k)
                        .map(|index| {
                            rounded_input[row * k + index]
                                * DType::BF16.round(qk_value(
                                    column,
                                    index,
                                    QuantizationFormat::Q5_K,
                                ))
                        })
                        .sum::<f32>()
                })
            })
            .collect::<Vec<_>>();
        let reference = Tensor::from_f32(&d, [m, n], DType::BF16, &expected).unwrap();

        d.set_q5_k_mpp_tile_k64(false).unwrap();
        let control = d.project_quantized(&x, &weight).unwrap();
        assert_eq!(control.metrics.operation, "q5_k_gemm_mpp");
        d.set_q5_k_mpp_tile_k64(true).unwrap();
        let candidate = d.project_quantized(&x, &weight).unwrap();
        assert_eq!(candidate.metrics.operation, "q5_k_gemm_mpp_k64");
        for (index, (actual, expected)) in candidate
            .tensor
            .to_f32()
            .iter()
            .zip(reference.to_f32())
            .enumerate()
        {
            assert!(
                (actual - expected).abs() <= 0.04,
                "Q5_K K=64 index {index}: actual={actual}, expected={expected}"
            );
        }
    }

    #[test]
    fn q4_k_and_q5_k_embedding_gather_decodes_selected_rows() {
        let d = MetalDevice::new().unwrap();
        for format in [QuantizationFormat::Q4_K, QuantizationFormat::Q5_K] {
            let (vocab, hidden) = (5, 512);
            let weight = packed(&d, vocab, hidden, format);
            let ids = [4, 0, 3];
            let actual = d
                .embedding_gather_quantized(&weight, &ids, DType::BF16)
                .unwrap()
                .tensor
                .to_f32();
            let expected = ids
                .iter()
                .flat_map(|&row| {
                    (0..hidden).map(move |column| {
                        DType::BF16.round(qk_value(row as usize, column, format))
                    })
                })
                .collect::<Vec<_>>();
            assert_eq!(actual, expected, "{format:?}");
        }
    }

    #[test]
    fn q4_k_and_q5_k_expert_projection_uses_assignment_metadata() {
        let d = MetalDevice::new().unwrap();
        let (experts, rows_per_expert, columns) = (3, 5, 512);
        let expert_ids = [2usize, 1, 2, 0];
        for format in [QuantizationFormat::Q4_K, QuantizationFormat::Q5_K] {
            let packed = packed(&d, experts * rows_per_expert, columns, format);
            let weight = QuantizedExpertMatrix::new(packed, experts, rows_per_expert).unwrap();
            let values = input(expert_ids.len(), columns);
            let x =
                Tensor::from_f32(&d, [expert_ids.len(), columns], DType::BF16, &values).unwrap();
            let rounded_input = x.to_f32();
            let metadata = expert_ids
                .iter()
                .enumerate()
                .flat_map(|(assignment, &expert)| {
                    [
                        f32::from_bits(expert as u32),
                        f32::from_bits(assignment as u32),
                        1.0,
                    ]
                })
                .collect::<Vec<_>>();
            let metadata =
                Tensor::from_f32(&d, [expert_ids.len(), 3], DType::F32, &metadata).unwrap();
            let actual = d
                .expert_project_quantized(&x, &metadata, &weight, columns, experts)
                .unwrap()
                .to_f32();
            let expected = (0..expert_ids.len())
                .flat_map(|assignment| {
                    (0..rows_per_expert).map({
                        let rounded_input = &rounded_input;
                        move |row| {
                            DType::BF16.round(
                                (0..columns)
                                    .map(|column| {
                                        rounded_input[assignment * columns + column]
                                            * qk_value(
                                                expert_ids[assignment] * rows_per_expert + row,
                                                column,
                                                format,
                                            )
                                    })
                                    .sum::<f32>(),
                            )
                        }
                    })
                })
                .collect::<Vec<_>>();
            for (index, (actual, expected)) in actual.iter().zip(&expected).enumerate() {
                assert!(
                    (actual - expected).abs() <= 0.06,
                    "{format:?} index {index}: actual={actual}, expected={expected}"
                );
            }
        }
    }

    #[test]
    fn q4_k_expert_row_reuse_and_pair_traversal_covers_tail() {
        let d = MetalDevice::new().unwrap();
        d.set_q4_k_factored(false).unwrap();
        assert!(d.use_q4_k_expert_project_8rows(133, 512));
        d.set_moe_expert_tensorops(false).unwrap();
        assert!(!d.use_q4_k_expert_project_8rows(127, 512));
        assert!(!d.use_q4_k_expert_project_8rows(128, 255));
        d.set_q4_k_expert_project_8rows(false).unwrap();
        assert!(!d.use_q4_k_expert_project_8rows(133, 512));
        d.set_q4_k_expert_project_8rows(true).unwrap();
        assert!(d.use_q4_k_expert_project_8rows(133, 512));
        assert!(d.use_q4_k_expert_project_16rows(133, 512));
        d.set_q4_k_expert_project_16rows(false).unwrap();
        assert!(!d.use_q4_k_expert_project_16rows(133, 512));
        d.set_q4_k_expert_project_16rows(true).unwrap();
        assert!(d.use_q4_k_expert_project_16rows(133, 512));
        assert!(d.use_q4_k_expert_project_16rows_pairs(133, 512));
        d.set_q4_k_expert_project_16rows_pairs(false).unwrap();
        assert!(!d.use_q4_k_expert_project_16rows_pairs(133, 512));
        d.set_q4_k_expert_project_16rows_pairs(true).unwrap();
        assert!(d.use_q4_k_expert_project_16rows_pairs(133, 512));
        for columns in [256, 512, 1024, 1792, 2048, 4096] {
            assert!(d.use_q4_k_expert_project_16rows_pairs(133, columns));
        }
        assert!(!d.use_q4_k_expert_project_16rows(127, 512));
        assert!(!d.use_q4_k_expert_project_16rows(133, 255));

        for columns in [256, 512, 1024, 1792, 2048, 4096] {
            let (experts, rows_per_expert) = (3, 133);
            let expert_ids = [2usize, 1, 2, 0];
            let packed = packed(
                &d,
                experts * rows_per_expert,
                columns,
                QuantizationFormat::Q4_K,
            );
            let weight = QuantizedExpertMatrix::new(packed, experts, rows_per_expert).unwrap();
            let values = input(expert_ids.len(), columns);
            let x =
                Tensor::from_f32(&d, [expert_ids.len(), columns], DType::BF16, &values).unwrap();
            let rounded_input = x.to_f32();
            let metadata = expert_ids
                .iter()
                .enumerate()
                .flat_map(|(assignment, &expert)| {
                    [
                        f32::from_bits(expert as u32),
                        f32::from_bits(assignment as u32),
                        1.0,
                    ]
                })
                .collect::<Vec<_>>();
            let metadata =
                Tensor::from_f32(&d, [expert_ids.len(), 3], DType::F32, &metadata).unwrap();
            d.set_q4_k_expert_project_16rows(false).unwrap();
            d.set_q4_k_expert_project_8rows(false).unwrap();
            let control = d
                .expert_project_quantized(&x, &metadata, &weight, columns, experts)
                .unwrap()
                .to_f32();
            d.set_q4_k_expert_project_8rows(true).unwrap();
            let output = d
                .expert_project_quantized(&x, &metadata, &weight, columns, experts)
                .unwrap();
            assert_eq!(
                output.to_f32(),
                control,
                "optimized order must match control"
            );
            d.set_q4_k_expert_project_16rows(true).unwrap();
            let output = d
                .expert_project_quantized(&x, &metadata, &weight, columns, experts)
                .unwrap();
            assert_eq!(
                output.to_f32(),
                control,
                "sixteen-row reuse must match control and cover the output tail"
            );
            d.set_q4_k_expert_project_16rows_pairs(true).unwrap();
            let output = d
                .expert_project_quantized(&x, &metadata, &weight, columns, experts)
                .unwrap();
            assert_eq!(
                output.to_f32(),
                control,
                "block-and-pair traversal must match the scalar control and cover the output tail"
            );
            let expected = (0..expert_ids.len())
                .flat_map(|assignment| {
                    (0..rows_per_expert).map({
                        let rounded_input = &rounded_input;
                        move |row| {
                            DType::BF16.round(
                                (0..columns)
                                    .map(|column| {
                                        rounded_input[assignment * columns + column]
                                            * qk_value(
                                                expert_ids[assignment] * rows_per_expert + row,
                                                column,
                                                QuantizationFormat::Q4_K,
                                            )
                                    })
                                    .sum::<f32>(),
                            )
                        }
                    })
                })
                .collect::<Vec<_>>();
            for (index, (actual, expected)) in output.to_f32().iter().zip(&expected).enumerate() {
                assert!(
                    (actual - expected).abs() <= 0.06,
                    "Q4_K expert row-pair index {index}: actual={actual}, expected={expected}"
                );
            }
            // Factored scales reorder the f32 sum; results stay within BF16 rounding.
            d.set_q4_k_factored(true).unwrap();
            let factored = d
                .expert_project_quantized(&x, &metadata, &weight, columns, experts)
                .unwrap()
                .to_f32();
            d.set_q4_k_factored(false).unwrap();
            for (index, ((actual, control), expected)) in
                factored.iter().zip(&control).zip(&expected).enumerate()
            {
                assert!(
                    (actual - expected).abs() <= 0.06
                        && (actual - control).abs() <= control.abs().max(1.) / 128.,
                    "Q4_K factored expert index {index}: actual={actual}, control={control}, expected={expected}"
                );
            }
        }
    }

    #[test]
    fn q4_k_factored_gemv_matches_reference_across_blocks_and_row_tails() {
        let d = MetalDevice::new().unwrap();
        d.set_k_quant_gemv_ksplit(0).unwrap();
        assert!(d.use_q4_k_factored(256));
        assert!(!d.use_q4_k_factored(384));
        for (n, k) in [(1, 256), (7, 512), (133, 1024), (130, 2048), (9, 4864)] {
            let weight = packed(&d, n, k, QuantizationFormat::Q4_K);
            let values = input(1, k);
            let x = Tensor::from_f32(&d, [1, k], DType::F32, &values).unwrap();
            let output = d.project_quantized(&x, &weight).unwrap();
            assert_eq!(output.metrics.operation, "q4_k_gemv_factored");
            let reference = expected(&values, 1, n, k, QuantizationFormat::Q4_K);
            for (index, (actual, reference)) in
                output.tensor.to_f32().iter().zip(reference).enumerate()
            {
                assert!(
                    (actual - reference).abs()
                        <= 1e-4 * reference.abs().max(1.) * (k as f32 / 256.),
                    "factored GEMV N={n} K={k} index {index}: actual={actual}, expected={reference}"
                );
            }
        }
    }

    #[test]
    fn q6_k_expert_eight_row_projection_reuses_activations_and_covers_tail() {
        let d = MetalDevice::new().unwrap();
        d.set_q6_k_factored(false).unwrap();
        assert!(d.use_q6_k_expert_project_8rows(133, 512));
        d.set_moe_expert_tensorops(false).unwrap();
        assert!(!d.use_q6_k_expert_project_8rows(127, 512));
        assert!(!d.use_q6_k_expert_project_8rows(128, 255));
        d.set_q6_k_expert_project_8rows(true).unwrap();
        assert!(d.use_q6_k_expert_project_8rows(133, 512));

        let (experts, rows_per_expert, columns) = (3, 133, 512);
        let expert_ids = [2usize, 1, 2, 0];
        let packed = packed(
            &d,
            experts * rows_per_expert,
            columns,
            QuantizationFormat::Q6_K,
        );
        let weight = QuantizedExpertMatrix::new(packed, experts, rows_per_expert).unwrap();
        let values = input(expert_ids.len(), columns);
        let x = Tensor::from_f32(&d, [expert_ids.len(), columns], DType::BF16, &values).unwrap();
        let rounded_input = x.to_f32();
        let metadata = expert_ids
            .iter()
            .enumerate()
            .flat_map(|(assignment, &expert)| {
                [
                    f32::from_bits(expert as u32),
                    f32::from_bits(assignment as u32),
                    1.0,
                ]
            })
            .collect::<Vec<_>>();
        let metadata = Tensor::from_f32(&d, [expert_ids.len(), 3], DType::F32, &metadata).unwrap();
        d.set_q6_k_expert_project_8rows(false).unwrap();
        let control = d
            .expert_project_quantized(&x, &metadata, &weight, columns, experts)
            .unwrap()
            .to_f32();
        d.set_q6_k_expert_project_8rows(true).unwrap();
        let output = d
            .expert_project_quantized(&x, &metadata, &weight, columns, experts)
            .unwrap();
        assert_eq!(
            output.to_f32(),
            control,
            "optimized order must match control"
        );
        let expected = (0..expert_ids.len())
            .flat_map(|assignment| {
                (0..rows_per_expert).map({
                    let rounded_input = &rounded_input;
                    move |row| {
                        DType::BF16.round(
                            (0..columns)
                                .map(|column| {
                                    rounded_input[assignment * columns + column]
                                        * qk_value(
                                            expert_ids[assignment] * rows_per_expert + row,
                                            column,
                                            QuantizationFormat::Q6_K,
                                        )
                                })
                                .sum::<f32>(),
                        )
                    }
                })
            })
            .collect::<Vec<_>>();
        for (index, (actual, expected)) in output.to_f32().iter().zip(&expected).enumerate() {
            assert!(
                (actual - expected).abs() <= 0.06,
                "Q6_K expert eight-row index {index}: actual={actual}, expected={expected}"
            );
        }
        // Factored slice scales reorder the f32 sum; results stay within BF16 rounding.
        d.set_q6_k_factored(true).unwrap();
        let factored = d
            .expert_project_quantized(&x, &metadata, &weight, columns, experts)
            .unwrap()
            .to_f32();
        for (index, ((actual, control), expected)) in
            factored.iter().zip(&control).zip(&expected).enumerate()
        {
            assert!(
                (actual - expected).abs() <= 0.06
                    && (actual - control).abs() <= control.abs().max(1.) / 128.,
                "Q6_K factored expert index {index}: actual={actual}, control={control}, expected={expected}"
            );
        }
    }

    #[test]
    fn q4_k_and_q5_k_expert_tensorops_group_routes_and_restore_assignment_order() {
        let d = MetalDevice::new().unwrap();
        if !d.mpp_projection() {
            return;
        }
        d.set_profiling(true);
        let (assignments, experts, rows_per_expert, columns) = (197, 3, 65, 512);
        let expert_ids = (0..assignments)
            .map(|assignment| (assignment + 1) % experts)
            .collect::<Vec<_>>();
        let values = input(assignments, columns);
        let x = Tensor::from_f32(&d, [assignments, columns], DType::BF16, &values).unwrap();
        let rounded_input = x.to_f32();
        let metadata_values = expert_ids
            .iter()
            .enumerate()
            .flat_map(|(assignment, &expert)| {
                [
                    f32::from_bits(expert as u32),
                    f32::from_bits(assignment as u32),
                    1.0,
                ]
            })
            .collect::<Vec<_>>();
        let metadata =
            Tensor::from_f32(&d, [assignments, 3], DType::F32, &metadata_values).unwrap();

        for format in [QuantizationFormat::Q4_K, QuantizationFormat::Q5_K] {
            let packed = packed(&d, experts * rows_per_expert, columns, format);
            let weight = QuantizedExpertMatrix::new(packed, experts, rows_per_expert).unwrap();
            let actual = d
                .expert_project_quantized(&x, &metadata, &weight, columns, experts)
                .unwrap()
                .to_f32();
            let expected = (0..assignments)
                .flat_map(|assignment| {
                    (0..rows_per_expert).map({
                        let rounded_input = &rounded_input;
                        let expert_ids = &expert_ids;
                        move |row| {
                            DType::BF16.round(
                                (0..columns)
                                    .map(|column| {
                                        rounded_input[assignment * columns + column]
                                            * DType::BF16.round(qk_value(
                                                expert_ids[assignment] * rows_per_expert + row,
                                                column,
                                                format,
                                            ))
                                    })
                                    .sum::<f32>(),
                            )
                        }
                    })
                })
                .collect::<Vec<_>>();
            for (index, (actual, expected)) in actual.iter().zip(&expected).enumerate() {
                assert!(
                    (actual - expected).abs() <= 0.08,
                    "{format:?} index {index}: actual={actual}, expected={expected}"
                );
            }
            let profile = d.take_profile();
            let operation = match format {
                QuantizationFormat::Q4_K => "expert_project_q4_k_mpp_k64",
                QuantizationFormat::Q5_K => "expert_project_q5_k_mpp",
                _ => unreachable!(),
            };
            assert_eq!(profile.get(operation).unwrap().calls, 1);
        }
    }

    #[test]
    fn q4_k_and_q5_k_tensorops_paired_tiles_are_bit_identical() {
        let d = MetalDevice::new().unwrap();
        if !d.mpp_projection() {
            return;
        }
        let bits = |t: &Tensor| t.to_f32().iter().map(|v| v.to_bits()).collect::<Vec<_>>();
        let check = |weight: &QuantizedMatrix, m: usize, k: usize, kernel: &str| {
            let x = Tensor::from_f32(&d, [m, k], DType::BF16, &input(m, k)).unwrap();
            d.set_dense_mpp_tile_pairs(false).unwrap();
            let control = d.project_quantized(&x, weight).unwrap();
            assert_eq!(control.metrics.operation, kernel);
            d.set_dense_mpp_tile_pairs(true).unwrap();
            let paired = d.project_quantized(&x, weight).unwrap();
            assert_eq!(
                bits(&paired.tensor),
                bits(&control.tensor),
                "{kernel} M={m}"
            );
        };
        let weight = packed(&d, 517, 512, QuantizationFormat::Q4_K);
        check(&weight, 600, 512, "q4_k_gemm_mpp");
        d.set_q4_k_mpp_tile_m128(false).unwrap();
        check(&weight, 1025, 512, "q4_k_gemm_mpp_k64");
        d.set_q4_k_mpp_tile_m128(true).unwrap();
        let weight = packed(&d, 517, 512, QuantizationFormat::Q5_K);
        check(&weight, 600, 512, "q5_k_gemm_mpp");
        check(&weight, 1025, 512, "q5_k_gemm_mpp_k64");
    }

    #[test]
    fn q4_k_grouped_experts_match_direct_bitwise() {
        let d = MetalDevice::new().unwrap();
        d.set_moe_expert_tensorops(false).unwrap();
        let bits = |t: &Tensor| t.to_f32().iter().map(|v| v.to_bits()).collect::<Vec<_>>();
        let (experts, rows_per_expert, columns) = (5, 133, 768);
        let weight = QuantizedExpertMatrix::new(
            packed(
                &d,
                experts * rows_per_expert,
                columns,
                QuantizationFormat::Q4_K,
            ),
            experts,
            rows_per_expert,
        )
        .unwrap();
        for assignments in [16, 44, 97] {
            let expert_ids = (0..assignments)
                .map(|a| if a % 7 == 0 { 4 } else { a % 3 })
                .collect::<Vec<_>>();
            let x = Tensor::from_f32(
                &d,
                [assignments, columns],
                DType::BF16,
                &input(assignments, columns),
            )
            .unwrap();
            let metadata = expert_ids
                .iter()
                .enumerate()
                .flat_map(|(a, &e)| [f32::from_bits(e as u32), f32::from_bits(a as u32), 1.0])
                .collect::<Vec<_>>();
            let metadata = Tensor::from_f32(&d, [assignments, 3], DType::F32, &metadata).unwrap();
            d.set_moe_expert_grouped(false).unwrap();
            let direct = d
                .expert_project_quantized(&x, &metadata, &weight, columns, experts)
                .unwrap();
            d.set_moe_expert_grouped(true).unwrap();
            let grouped = d
                .expert_project_quantized(&x, &metadata, &weight, columns, experts)
                .unwrap();
            assert_eq!(bits(&grouped), bits(&direct), "assignments={assignments}");
        }
    }

    #[test]
    fn q4_k_tensorops_fast_dequant_is_bit_identical() {
        let d = MetalDevice::new().unwrap();
        if !d.mpp_projection() {
            return;
        }
        let bits = |t: &Tensor| t.to_f32().iter().map(|v| v.to_bits()).collect::<Vec<_>>();
        // Dense: K=128 tile (M=37), K=64 tile (M=1,025 without M128), and K=64/M128.
        for (m, n, k, m128) in [
            (37, 133, 512, true),
            (1025, 65, 512, false),
            (1025, 65, 512, true),
        ] {
            d.set_q4_k_mpp_tile_m128(m128).unwrap();
            let weight = packed(&d, n, k, QuantizationFormat::Q4_K);
            let x = Tensor::from_f32(&d, [m, k], DType::BF16, &input(m, k)).unwrap();
            d.set_mpp_fast_dequant(false).unwrap();
            let control = d.project_quantized(&x, &weight).unwrap();
            assert!(control.metrics.operation.starts_with("q4_k_gemm_mpp"));
            d.set_mpp_fast_dequant(true).unwrap();
            let fast = d.project_quantized(&x, &weight).unwrap();
            assert_eq!(fast.metrics.operation, control.metrics.operation);
            assert_eq!(
                bits(&fast.tensor),
                bits(&control.tensor),
                "M={m} N={n} K={k}"
            );
        }
        d.set_q4_k_mpp_tile_m128(true).unwrap();
        // Experts: uneven segments (one expert above one 32-row tile) and a row tail.
        let (assignments, experts, rows_per_expert, columns) = (197, 3, 133, 512);
        let expert_ids = (0..assignments)
            .map(|a| if a % 5 == 0 { 0 } else { 1 + a % 2 })
            .collect::<Vec<_>>();
        let x = Tensor::from_f32(
            &d,
            [assignments, columns],
            DType::BF16,
            &input(assignments, columns),
        )
        .unwrap();
        let metadata = expert_ids
            .iter()
            .enumerate()
            .flat_map(|(a, &e)| [f32::from_bits(e as u32), f32::from_bits(a as u32), 1.0])
            .collect::<Vec<_>>();
        let metadata = Tensor::from_f32(&d, [assignments, 3], DType::F32, &metadata).unwrap();
        let packed = packed(
            &d,
            experts * rows_per_expert,
            columns,
            QuantizationFormat::Q4_K,
        );
        let weight = QuantizedExpertMatrix::new(packed, experts, rows_per_expert).unwrap();
        for k64 in [false, true] {
            d.set_moe_expert_tensorops_tile_k64(k64).unwrap();
            d.set_mpp_fast_dequant(false).unwrap();
            d.set_moe_expert_tile_pairs(false).unwrap();
            let control = d
                .expert_project_quantized(&x, &metadata, &weight, columns, experts)
                .unwrap();
            for (fast, pairs) in [(true, false), (false, true), (true, true)] {
                d.set_mpp_fast_dequant(fast).unwrap();
                d.set_moe_expert_tile_pairs(pairs).unwrap();
                let candidate = d
                    .expert_project_quantized(&x, &metadata, &weight, columns, experts)
                    .unwrap();
                assert_eq!(
                    bits(&candidate),
                    bits(&control),
                    "k64={k64} fast={fast} pairs={pairs}"
                );
            }
        }
    }

    #[test]
    fn q4_k_expert_tensorops_k64_matches_reference() {
        let d = MetalDevice::new().unwrap();
        if !d.mpp_projection() {
            return;
        }
        d.set_profiling(true);
        d.set_moe_expert_tensorops_tile_k64(true).unwrap();
        let (assignments, experts, rows_per_expert, columns) = (197, 3, 65, 512);
        let expert_ids = (0..assignments)
            .map(|assignment| (assignment + 1) % experts)
            .collect::<Vec<_>>();
        let values = input(assignments, columns);
        let x = Tensor::from_f32(&d, [assignments, columns], DType::BF16, &values).unwrap();
        let rounded_input = x.to_f32();
        let metadata_values = expert_ids
            .iter()
            .enumerate()
            .flat_map(|(assignment, &expert)| {
                [
                    f32::from_bits(expert as u32),
                    f32::from_bits(assignment as u32),
                    1.0,
                ]
            })
            .collect::<Vec<_>>();
        let metadata =
            Tensor::from_f32(&d, [assignments, 3], DType::F32, &metadata_values).unwrap();

        for format in [QuantizationFormat::Q4_K] {
            let packed = packed(&d, experts * rows_per_expert, columns, format);
            let weight = QuantizedExpertMatrix::new(packed, experts, rows_per_expert).unwrap();
            let actual = d
                .expert_project_quantized(&x, &metadata, &weight, columns, experts)
                .unwrap()
                .to_f32();
            let expected = (0..assignments)
                .flat_map(|assignment| {
                    (0..rows_per_expert).map({
                        let rounded_input = &rounded_input;
                        let expert_ids = &expert_ids;
                        move |row| {
                            DType::BF16.round(
                                (0..columns)
                                    .map(|column| {
                                        rounded_input[assignment * columns + column]
                                            * DType::BF16.round(qk_value(
                                                expert_ids[assignment] * rows_per_expert + row,
                                                column,
                                                format,
                                            ))
                                    })
                                    .sum::<f32>(),
                            )
                        }
                    })
                })
                .collect::<Vec<_>>();
            for (index, (actual, expected)) in actual.iter().zip(&expected).enumerate() {
                assert!(
                    (actual - expected).abs() <= 0.08,
                    "{format:?} K=64 expert index {index}: actual={actual}, expected={expected}"
                );
            }
            let operation = "expert_project_q4_k_mpp_k64";
            let profile = d.take_profile();
            assert_eq!(profile.get(operation).unwrap().calls, 1);
        }
    }

    #[test]
    fn q4_k_and_q5_k_expert_tensorops_cover_twelve_routes_per_expert() {
        let d = MetalDevice::new().unwrap();
        if !d.mpp_projection() {
            return;
        }
        d.set_profiling(true);
        assert!(d.set_moe_expert_tensorops_min_routes_per_expert(0).is_err());
        d.set_moe_expert_tensorops_min_routes_per_expert(12)
            .unwrap();
        let (assignments, experts, rows_per_expert, columns) = (37, 3, 65, 512);
        let expert_ids = (0..assignments)
            .map(|assignment| (assignment + 1) % experts)
            .collect::<Vec<_>>();
        let values = input(assignments, columns);
        let x = Tensor::from_f32(&d, [assignments, columns], DType::BF16, &values).unwrap();
        let rounded_input = x.to_f32();
        let metadata_values = expert_ids
            .iter()
            .enumerate()
            .flat_map(|(assignment, &expert)| {
                [
                    f32::from_bits(expert as u32),
                    f32::from_bits(assignment as u32),
                    1.0,
                ]
            })
            .collect::<Vec<_>>();
        let metadata =
            Tensor::from_f32(&d, [assignments, 3], DType::F32, &metadata_values).unwrap();

        for format in [QuantizationFormat::Q4_K, QuantizationFormat::Q5_K] {
            let packed = packed(&d, experts * rows_per_expert, columns, format);
            let weight = QuantizedExpertMatrix::new(packed, experts, rows_per_expert).unwrap();
            let actual = d
                .expert_project_quantized(&x, &metadata, &weight, columns, experts)
                .unwrap()
                .to_f32();
            let expected = (0..assignments)
                .flat_map(|assignment| {
                    (0..rows_per_expert).map({
                        let rounded_input = &rounded_input;
                        let expert_ids = &expert_ids;
                        move |row| {
                            DType::BF16.round(
                                (0..columns)
                                    .map(|column| {
                                        rounded_input[assignment * columns + column]
                                            * DType::BF16.round(qk_value(
                                                expert_ids[assignment] * rows_per_expert + row,
                                                column,
                                                format,
                                            ))
                                    })
                                    .sum::<f32>(),
                            )
                        }
                    })
                })
                .collect::<Vec<_>>();
            for (index, (actual, expected)) in actual.iter().zip(&expected).enumerate() {
                assert!(
                    (actual - expected).abs() <= 0.08,
                    "{format:?} index {index}: actual={actual}, expected={expected}"
                );
            }
            let profile = d.take_profile();
            let operation = match format {
                QuantizationFormat::Q4_K => "expert_project_q4_k_mpp_k64",
                QuantizationFormat::Q5_K => "expert_project_q5_k_mpp",
                _ => unreachable!(),
            };
            assert_eq!(profile.get(operation).unwrap().calls, 1);
        }
    }

    #[test]
    fn q4_k_and_q5_k_expert_tensorops_cover_eight_routes_per_expert() {
        let d = MetalDevice::new().unwrap();
        if !d.mpp_projection() {
            return;
        }
        d.set_profiling(true);
        assert_eq!(d.moe_expert_tensorops_min_routes_per_expert(), 8);
        d.set_moe_expert_tensorops_min_routes_per_expert(8).unwrap();
        let (assignments, experts, rows_per_expert, columns) = (25, 3, 65, 512);
        let expert_ids = (0..assignments)
            .map(|assignment| (assignment + 1) % experts)
            .collect::<Vec<_>>();
        let values = input(assignments, columns);
        let x = Tensor::from_f32(&d, [assignments, columns], DType::BF16, &values).unwrap();
        let rounded_input = x.to_f32();
        let metadata_values = expert_ids
            .iter()
            .enumerate()
            .flat_map(|(assignment, &expert)| {
                [
                    f32::from_bits(expert as u32),
                    f32::from_bits(assignment as u32),
                    1.0,
                ]
            })
            .collect::<Vec<_>>();
        let metadata =
            Tensor::from_f32(&d, [assignments, 3], DType::F32, &metadata_values).unwrap();

        for format in [QuantizationFormat::Q4_K, QuantizationFormat::Q5_K] {
            let packed = packed(&d, experts * rows_per_expert, columns, format);
            let weight = QuantizedExpertMatrix::new(packed, experts, rows_per_expert).unwrap();
            let actual = d
                .expert_project_quantized(&x, &metadata, &weight, columns, experts)
                .unwrap()
                .to_f32();
            let expected = (0..assignments)
                .flat_map(|assignment| {
                    (0..rows_per_expert).map({
                        let rounded_input = &rounded_input;
                        let expert_ids = &expert_ids;
                        move |row| {
                            DType::BF16.round(
                                (0..columns)
                                    .map(|column| {
                                        rounded_input[assignment * columns + column]
                                            * DType::BF16.round(qk_value(
                                                expert_ids[assignment] * rows_per_expert + row,
                                                column,
                                                format,
                                            ))
                                    })
                                    .sum::<f32>(),
                            )
                        }
                    })
                })
                .collect::<Vec<_>>();
            for (index, (actual, expected)) in actual.iter().zip(&expected).enumerate() {
                assert!(
                    (actual - expected).abs() <= 0.08,
                    "{format:?} index {index}: actual={actual}, expected={expected}"
                );
            }
            let profile = d.take_profile();
            let operation = match format {
                QuantizationFormat::Q4_K => "expert_project_q4_k_mpp_k64",
                QuantizationFormat::Q5_K => "expert_project_q5_k_mpp",
                _ => unreachable!(),
            };
            assert_eq!(profile.get(operation).unwrap().calls, 1);
        }
    }

    #[test]
    fn q4_k_and_q5_k_reject_partial_superblocks() {
        let d = MetalDevice::new().unwrap();
        for format in [QuantizationFormat::Q4_K, QuantizationFormat::Q5_K] {
            assert!(QuantizedMatrix::from_reader(&d, 3, 255, format, |_| Ok(())).is_err());
        }
    }
}

#[cfg(test)]
mod q6_k_tests {
    use super::*;
    use crate::quantization::{QuantizationFormat, QuantizedExpertMatrix, QuantizedMatrix};

    fn packed_byte(row: usize, block: usize, index: usize) -> u8 {
        ((row * 13 + block * 7 + index * 17 + 3) & 255) as u8
    }

    fn scale_byte(row: usize, block: usize, index: usize) -> i8 {
        ((row * 3 + block * 5 + index * 7) % 9) as i8 - 4
    }

    fn block_scale(row: usize, block: usize) -> f32 {
        if (row + block).is_multiple_of(2) {
            0.125
        } else {
            -0.0625
        }
    }

    fn packed(device: &MetalDevice, rows: usize, columns: usize) -> QuantizedMatrix {
        let mut bytes = Vec::new();
        for row in 0..rows {
            for block_index in 0..columns / 256 {
                let mut block = [0u8; 210];
                for (index, byte) in block[..128].iter_mut().enumerate() {
                    *byte = packed_byte(row, block_index, index);
                }
                for (index, byte) in block[128..192].iter_mut().enumerate() {
                    *byte = packed_byte(row + 11, block_index, index);
                }
                for (index, byte) in block[192..208].iter_mut().enumerate() {
                    *byte = scale_byte(row, block_index, index) as u8;
                }
                block[208..210].copy_from_slice(
                    &half::f16::from_f32(block_scale(row, block_index))
                        .to_bits()
                        .to_le_bytes(),
                );
                bytes.extend_from_slice(&block);
            }
        }
        QuantizedMatrix::from_reader(
            device,
            rows,
            columns,
            QuantizationFormat::Q6_K,
            |destination| {
                destination.copy_from_slice(&bytes);
                Ok(())
            },
        )
        .unwrap()
    }

    fn q6_value(row: usize, column: usize) -> f32 {
        let block = column / 256;
        let within = column % 256;
        let group = within / 32;
        let lane = within % 32;
        let half_block = group / 4;
        let slice = group % 4;
        let ql_index = half_block * 64 + (slice % 2) * 32 + lane;
        let ql = packed_byte(row, block, ql_index);
        let ql_bits = if slice < 2 { ql & 15 } else { ql >> 4 };
        let qh = packed_byte(row + 11, block, half_block * 32 + lane);
        let high = (qh >> (slice * 2)) & 3;
        let quant = i32::from(ql_bits | (high << 4)) - 32;
        let scale_index = half_block * 8 + slice * 2 + lane / 16;
        let scale = f32::from(scale_byte(row, block, scale_index));
        block_scale(row, block) * scale * quant as f32
    }

    fn input(rows: usize, columns: usize) -> Vec<f32> {
        (0..rows * columns)
            .map(|index| ((index * 19 + 7) % 127) as f32 / 113.0 - 0.56)
            .collect()
    }

    fn expected(input: &[f32], m: usize, n: usize, k: usize) -> Vec<f32> {
        (0..m)
            .flat_map(|row| {
                (0..n).map(move |column| {
                    (0..k)
                        .map(|index| input[row * k + index] * q6_value(column, index))
                        .sum::<f32>()
                })
            })
            .collect()
    }

    #[test]
    fn q6_k_gemv_ksplit_covers_row_and_block_tails() {
        let d = MetalDevice::new().unwrap();
        assert!(d.set_k_quant_gemv_ksplit(3).is_err());
        // 133 rows leave a partial four-row tile; 10 and 19 super-blocks
        // leave partial strides for every SIMD group count.
        for (n, k) in [(133, 2560), (6, 4864)] {
            let weight = packed(&d, n, k);
            let values = input(1, k);
            let x = Tensor::from_f32(&d, [1, k], DType::F32, &values).unwrap();
            let reference = expected(&values, 1, n, k);
            for simds in [1, 2, 4, 8] {
                d.set_k_quant_gemv_ksplit(simds).unwrap();
                let output = d.project_quantized(&x, &weight).unwrap();
                assert_eq!(output.metrics.operation, "q6_k_gemv_ksplit");
                for (index, (actual, reference)) in
                    output.tensor.to_f32().iter().zip(&reference).enumerate()
                {
                    assert!(
                        (actual - reference).abs() <= 1.0e-3 * reference.abs().max(1.0),
                        "Q6_K K-split {simds} ({n}, {k}) index {index}: actual={actual}, expected={reference}"
                    );
                }
            }
        }
    }

    /// Steady-state down-projection GEMV bandwidth (896 x 4864, 24 weights).
    #[test]
    #[ignore = "microbenchmark; run with --ignored --nocapture"]
    fn bench_q6_k_gemv_ksplit_bandwidth() {
        let d = MetalDevice::new().unwrap();
        for (n, k) in [(896, 4864), (2048, 2048), (8192, 2048)] {
            let weights: Vec<_> = (0..24).map(|_| packed(&d, n, k)).collect();
            let x = Tensor::from_f32(&d, [1, k], DType::BF16, &input(1, k)).unwrap();
            let bytes = (weights[0].byte_size() * weights.len()) as f64;
            for simds in [0usize, 2, 4, 8] {
                d.set_k_quant_gemv_ksplit(simds).unwrap();
                let mut best = f64::MAX;
                for _ in 0..7 {
                    let start = std::time::Instant::now();
                    let execution = d.execution().unwrap();
                    let mut last = None;
                    for _ in 0..8 {
                        for weight in &weights {
                            last = Some(d.project_quantized(&x, weight).unwrap().tensor);
                        }
                    }
                    drop(execution);
                    std::hint::black_box(last.unwrap().to_f32());
                    best = best.min(start.elapsed().as_secs_f64() / 8.0);
                }
                println!(
                    "Q6_K {n}x{k} simds={simds}: {:.1} us/projection, {:.1} GB/s",
                    best * 1e6 / weights.len() as f64,
                    bytes / best / 1e9
                );
            }
        }
        d.set_k_quant_gemv_ksplit(0).unwrap();
    }

    #[test]
    fn q6_k_gemv_and_gemm_cover_superblock_and_batch_tails() {
        let d = MetalDevice::new().unwrap();
        let (n, k) = (5, 512);
        let weight = packed(&d, n, k);
        assert_eq!(weight.byte_size(), n * 2 * 210);
        for m in [1, 2, 3, 4, 5, 7] {
            let values = input(m, k);
            let x = Tensor::from_f32(&d, [m, k], DType::F32, &values).unwrap();
            let actual = d.project_quantized(&x, &weight).unwrap().tensor.to_f32();
            let expected = expected(&values, m, n, k);
            for (index, (actual, expected)) in actual.iter().zip(&expected).enumerate() {
                assert!(
                    (actual - expected).abs() <= 0.02,
                    "index {index}: actual={actual}, expected={expected}"
                );
            }
        }
    }

    #[test]
    fn q6_k_eight_row_gemv_reuses_activations_and_covers_output_tail() {
        let d = MetalDevice::new().unwrap();
        d.set_q6_k_factored(false).unwrap();
        assert!(!d.use_q6_k_gemv_8rows(127));
        let (n, k) = (133, 512);
        assert!(d.use_q6_k_gemv_8rows(n));
        let weight = packed(&d, n, k);
        let values = input(1, k);
        let x = Tensor::from_f32(&d, [1, k], DType::F32, &values).unwrap();
        let output = d.project_quantized(&x, &weight).unwrap();
        assert_eq!(output.metrics.operation, "q6_k_gemv_8rows");
        let reference = expected(&values, 1, n, k);
        for (index, (actual, reference)) in output.tensor.to_f32().iter().zip(reference).enumerate()
        {
            assert!(
                (actual - reference).abs() <= 0.02,
                "eight-row GEMV index {index}: actual={actual}, expected={reference}"
            );
        }
    }

    #[test]
    fn q6_k_factored_gemv_matches_reference_across_blocks_and_row_tails() {
        let d = MetalDevice::new().unwrap();
        d.set_k_quant_gemv_ksplit(0).unwrap();
        assert!(d.use_q6_k_factored(256));
        assert!(!d.use_q6_k_factored(384));
        for (n, k) in [(1, 256), (7, 512), (133, 1792), (9, 2048)] {
            let weight = packed(&d, n, k);
            let values = input(1, k);
            let x = Tensor::from_f32(&d, [1, k], DType::F32, &values).unwrap();
            let output = d.project_quantized(&x, &weight).unwrap();
            assert_eq!(output.metrics.operation, "q6_k_gemv_factored");
            let reference = expected(&values, 1, n, k);
            for (index, (actual, reference)) in
                output.tensor.to_f32().iter().zip(reference).enumerate()
            {
                assert!(
                    (actual - reference).abs() <= 0.02,
                    "factored GEMV N={n} K={k} index {index}: actual={actual}, expected={reference}"
                );
            }
        }
    }

    #[test]
    fn q6_k_mpp_gemm_decodes_bounded_tiles_with_row_and_k_tails() {
        let d = MetalDevice::new().unwrap();
        if !d.mpp_projection() {
            return;
        }
        let (n, k) = (65, 512);
        let weight = packed(&d, n, k);
        for m in [4, 8, 15, 35] {
            let values = input(m, k);
            let x = Tensor::from_f32(&d, [m, k], DType::BF16, &values).unwrap();
            let rounded_input = x.to_f32();
            let reference = expected(&rounded_input, m, n, k);
            let reference = Tensor::from_f32(&d, [m, n], DType::BF16, &reference).unwrap();
            let output = d.project_quantized(&x, &weight).unwrap();
            assert_eq!(
                output.metrics.operation,
                if m < 8 { "q6_k_gemm" } else { "q6_k_gemm_mpp" }
            );
            for (index, (actual, expected)) in output
                .tensor
                .to_f32()
                .iter()
                .zip(reference.to_f32())
                .enumerate()
            {
                assert!(
                    (actual - expected).abs() <= 0.12,
                    "M={m} index {index}: actual={actual}, expected={expected}"
                );
            }
        }
    }

    #[test]
    fn q6_k_embedding_gather_decodes_only_requested_rows() {
        let d = MetalDevice::new().unwrap();
        let (vocab, hidden) = (5, 512);
        let weight = packed(&d, vocab, hidden);
        let ids = [4, 0, 3];
        let actual = d
            .embedding_gather_quantized(&weight, &ids, DType::BF16)
            .unwrap()
            .tensor
            .to_f32();
        let expected = ids
            .iter()
            .flat_map(|&row| {
                (0..hidden).map(move |column| DType::BF16.round(q6_value(row as usize, column)))
            })
            .collect::<Vec<_>>();
        assert_eq!(actual, expected);
    }

    #[test]
    fn q6_k_expert_projection_uses_assignment_metadata() {
        let d = MetalDevice::new().unwrap();
        d.set_q6_k_factored(false).unwrap();
        let (experts, rows_per_expert, columns) = (3, 5, 512);
        let expert_ids = [2usize, 1, 2, 0];
        let packed = packed(&d, experts * rows_per_expert, columns);
        let weight = QuantizedExpertMatrix::new(packed, experts, rows_per_expert).unwrap();
        let values = input(expert_ids.len(), columns);
        let x = Tensor::from_f32(&d, [expert_ids.len(), columns], DType::BF16, &values).unwrap();
        let rounded_input = x.to_f32();
        let metadata = expert_ids
            .iter()
            .enumerate()
            .flat_map(|(assignment, &expert)| {
                [
                    f32::from_bits(expert as u32),
                    f32::from_bits(assignment as u32),
                    1.0,
                ]
            })
            .collect::<Vec<_>>();
        let metadata = Tensor::from_f32(&d, [expert_ids.len(), 3], DType::F32, &metadata).unwrap();
        let actual = d
            .expert_project_quantized(&x, &metadata, &weight, columns, experts)
            .unwrap()
            .to_f32();
        let expected = (0..expert_ids.len())
            .flat_map(|assignment| {
                (0..rows_per_expert).map({
                    let rounded_input = &rounded_input;
                    move |row| {
                        DType::BF16.round(
                            (0..columns)
                                .map(|column| {
                                    rounded_input[assignment * columns + column]
                                        * q6_value(
                                            expert_ids[assignment] * rows_per_expert + row,
                                            column,
                                        )
                                })
                                .sum::<f32>(),
                        )
                    }
                })
            })
            .collect::<Vec<_>>();
        for (index, (actual, expected)) in actual.iter().zip(&expected).enumerate() {
            assert!(
                (actual - expected).abs() <= 0.12,
                "Q6_K index {index}: actual={actual}, expected={expected}"
            );
        }
    }

    #[test]
    fn q6_k_tensorops_paired_tiles_are_bit_identical() {
        let d = MetalDevice::new().unwrap();
        if !d.mpp_projection() {
            return;
        }
        let bits = |t: &Tensor| t.to_f32().iter().map(|v| v.to_bits()).collect::<Vec<_>>();
        let check = |weight: &QuantizedMatrix, m: usize, k: usize, kernel: &str| {
            let x = Tensor::from_f32(&d, [m, k], DType::BF16, &input(m, k)).unwrap();
            d.set_dense_mpp_tile_pairs(false).unwrap();
            let control = d.project_quantized(&x, weight).unwrap();
            assert_eq!(control.metrics.operation, kernel);
            d.set_dense_mpp_tile_pairs(true).unwrap();
            let paired = d.project_quantized(&x, weight).unwrap();
            assert_eq!(
                bits(&paired.tensor),
                bits(&control.tensor),
                "{kernel} M={m}"
            );
        };
        let weight = packed(&d, 517, 768);
        check(&weight, 600, 768, "q6_k_gemm_mpp");
    }

    #[test]
    fn q6_k_grouped_experts_match_direct_bitwise() {
        let d = MetalDevice::new().unwrap();
        d.set_moe_expert_tensorops(false).unwrap();
        let bits = |t: &Tensor| t.to_f32().iter().map(|v| v.to_bits()).collect::<Vec<_>>();
        let (experts, rows_per_expert, columns) = (5, 133, 768);
        let weight = QuantizedExpertMatrix::new(
            packed(&d, experts * rows_per_expert, columns),
            experts,
            rows_per_expert,
        )
        .unwrap();
        for assignments in [16, 44, 97] {
            let expert_ids = (0..assignments)
                .map(|a| if a % 7 == 0 { 4 } else { a % 3 })
                .collect::<Vec<_>>();
            let x = Tensor::from_f32(
                &d,
                [assignments, columns],
                DType::BF16,
                &input(assignments, columns),
            )
            .unwrap();
            let metadata = expert_ids
                .iter()
                .enumerate()
                .flat_map(|(a, &e)| [f32::from_bits(e as u32), f32::from_bits(a as u32), 1.0])
                .collect::<Vec<_>>();
            let metadata = Tensor::from_f32(&d, [assignments, 3], DType::F32, &metadata).unwrap();
            d.set_moe_expert_grouped(false).unwrap();
            let direct = d
                .expert_project_quantized(&x, &metadata, &weight, columns, experts)
                .unwrap();
            d.set_moe_expert_grouped(true).unwrap();
            let grouped = d
                .expert_project_quantized(&x, &metadata, &weight, columns, experts)
                .unwrap();
            assert_eq!(bits(&grouped), bits(&direct), "assignments={assignments}");
        }
    }

    #[test]
    fn q6_k_tensorops_fast_dequant_is_bit_identical() {
        let d = MetalDevice::new().unwrap();
        if !d.mpp_projection() {
            return;
        }
        let bits = |t: &Tensor| t.to_f32().iter().map(|v| v.to_bits()).collect::<Vec<_>>();
        let (m, n, k) = (37, 133, 768);
        let weight = packed(&d, n, k);
        let x = Tensor::from_f32(&d, [m, k], DType::BF16, &input(m, k)).unwrap();
        d.set_mpp_fast_dequant(false).unwrap();
        let control = d.project_quantized(&x, &weight).unwrap();
        assert_eq!(control.metrics.operation, "q6_k_gemm_mpp");
        d.set_mpp_fast_dequant(true).unwrap();
        let fast = d.project_quantized(&x, &weight).unwrap();
        assert_eq!(bits(&fast.tensor), bits(&control.tensor));

        let (assignments, experts, rows_per_expert, columns) = (197, 3, 133, 768);
        let expert_ids = (0..assignments)
            .map(|a| if a % 5 == 0 { 0 } else { 1 + a % 2 })
            .collect::<Vec<_>>();
        let x = Tensor::from_f32(
            &d,
            [assignments, columns],
            DType::BF16,
            &input(assignments, columns),
        )
        .unwrap();
        let metadata = expert_ids
            .iter()
            .enumerate()
            .flat_map(|(a, &e)| [f32::from_bits(e as u32), f32::from_bits(a as u32), 1.0])
            .collect::<Vec<_>>();
        let metadata = Tensor::from_f32(&d, [assignments, 3], DType::F32, &metadata).unwrap();
        let weight = QuantizedExpertMatrix::new(
            packed(&d, experts * rows_per_expert, columns),
            experts,
            rows_per_expert,
        )
        .unwrap();
        for k64 in [false, true] {
            d.set_moe_expert_tensorops_tile_k64(k64).unwrap();
            d.set_mpp_fast_dequant(false).unwrap();
            d.set_moe_expert_tile_pairs(false).unwrap();
            let control = d
                .expert_project_quantized(&x, &metadata, &weight, columns, experts)
                .unwrap();
            for (fast, pairs) in [(true, false), (false, true), (true, true)] {
                d.set_mpp_fast_dequant(fast).unwrap();
                d.set_moe_expert_tile_pairs(pairs).unwrap();
                let candidate = d
                    .expert_project_quantized(&x, &metadata, &weight, columns, experts)
                    .unwrap();
                assert_eq!(
                    bits(&candidate),
                    bits(&control),
                    "k64={k64} fast={fast} pairs={pairs}"
                );
            }
        }
    }

    #[test]
    fn q6_k_expert_tensorops_k64_matches_reference() {
        let d = MetalDevice::new().unwrap();
        if !d.mpp_projection() {
            return;
        }
        d.set_profiling(true);
        d.set_moe_expert_tensorops_tile_k64(true).unwrap();
        let (assignments, experts, rows_per_expert, columns) = (197, 3, 65, 512);
        let expert_ids = (0..assignments)
            .map(|assignment| (assignment + 1) % experts)
            .collect::<Vec<_>>();
        let packed = packed(&d, experts * rows_per_expert, columns);
        let weight = QuantizedExpertMatrix::new(packed, experts, rows_per_expert).unwrap();
        let values = input(assignments, columns);
        let x = Tensor::from_f32(&d, [assignments, columns], DType::BF16, &values).unwrap();
        let rounded_input = x.to_f32();
        let metadata_values = expert_ids
            .iter()
            .enumerate()
            .flat_map(|(assignment, &expert)| {
                [
                    f32::from_bits(expert as u32),
                    f32::from_bits(assignment as u32),
                    1.0,
                ]
            })
            .collect::<Vec<_>>();
        let metadata =
            Tensor::from_f32(&d, [assignments, 3], DType::F32, &metadata_values).unwrap();
        let actual = d
            .expert_project_quantized(&x, &metadata, &weight, columns, experts)
            .unwrap()
            .to_f32();
        let expected = (0..assignments)
            .flat_map(|assignment| {
                (0..rows_per_expert).map({
                    let rounded_input = &rounded_input;
                    let expert_ids = &expert_ids;
                    move |row| {
                        DType::BF16.round(
                            (0..columns)
                                .map(|column| {
                                    rounded_input[assignment * columns + column]
                                        * DType::BF16.round(q6_value(
                                            expert_ids[assignment] * rows_per_expert + row,
                                            column,
                                        ))
                                })
                                .sum::<f32>(),
                        )
                    }
                })
            })
            .collect::<Vec<_>>();
        for (index, (actual, expected)) in actual.iter().zip(&expected).enumerate() {
            assert!(
                (actual - expected).abs() <= 0.16,
                "Q6_K K=64 expert index {index}: actual={actual}, expected={expected}"
            );
        }
        assert_eq!(
            d.take_profile()
                .get("expert_project_q6_k_mpp_k64")
                .unwrap()
                .calls,
            1
        );
    }

    #[test]
    fn q6_k_expert_tensorops_groups_routes_and_restores_assignment_order() {
        let d = MetalDevice::new().unwrap();
        if !d.mpp_projection() {
            return;
        }
        d.set_profiling(true);
        let (assignments, experts, rows_per_expert, columns) = (197, 3, 65, 512);
        let expert_ids = (0..assignments)
            .map(|assignment| (assignment + 1) % experts)
            .collect::<Vec<_>>();
        let packed = packed(&d, experts * rows_per_expert, columns);
        let weight = QuantizedExpertMatrix::new(packed, experts, rows_per_expert).unwrap();
        let values = input(assignments, columns);
        let x = Tensor::from_f32(&d, [assignments, columns], DType::BF16, &values).unwrap();
        let rounded_input = x.to_f32();
        let metadata_values = expert_ids
            .iter()
            .enumerate()
            .flat_map(|(assignment, &expert)| {
                [
                    f32::from_bits(expert as u32),
                    f32::from_bits(assignment as u32),
                    1.0,
                ]
            })
            .collect::<Vec<_>>();
        let metadata =
            Tensor::from_f32(&d, [assignments, 3], DType::F32, &metadata_values).unwrap();
        let actual = d
            .expert_project_quantized(&x, &metadata, &weight, columns, experts)
            .unwrap()
            .to_f32();
        let expected = (0..assignments)
            .flat_map(|assignment| {
                (0..rows_per_expert).map({
                    let rounded_input = &rounded_input;
                    let expert_ids = &expert_ids;
                    move |row| {
                        DType::BF16.round(
                            (0..columns)
                                .map(|column| {
                                    rounded_input[assignment * columns + column]
                                        * DType::BF16.round(q6_value(
                                            expert_ids[assignment] * rows_per_expert + row,
                                            column,
                                        ))
                                })
                                .sum::<f32>(),
                        )
                    }
                })
            })
            .collect::<Vec<_>>();
        for (index, (actual, expected)) in actual.iter().zip(&expected).enumerate() {
            assert!(
                (actual - expected).abs() <= 0.16,
                "Q6_K TensorOps index {index}: actual={actual}, expected={expected}"
            );
        }
        assert_eq!(
            d.take_profile()
                .get("expert_project_q6_k_mpp_k64")
                .unwrap()
                .calls,
            1
        );
    }

    #[test]
    fn q6_k_expert_tensorops_covers_twelve_routes_per_expert() {
        let d = MetalDevice::new().unwrap();
        if !d.mpp_projection() {
            return;
        }
        d.set_profiling(true);
        d.set_moe_expert_tensorops_min_routes_per_expert(12)
            .unwrap();
        let (assignments, experts, rows_per_expert, columns) = (37, 3, 65, 512);
        let expert_ids = (0..assignments)
            .map(|assignment| (assignment + 1) % experts)
            .collect::<Vec<_>>();
        let packed = packed(&d, experts * rows_per_expert, columns);
        let weight = QuantizedExpertMatrix::new(packed, experts, rows_per_expert).unwrap();
        let values = input(assignments, columns);
        let x = Tensor::from_f32(&d, [assignments, columns], DType::BF16, &values).unwrap();
        let rounded_input = x.to_f32();
        let metadata_values = expert_ids
            .iter()
            .enumerate()
            .flat_map(|(assignment, &expert)| {
                [
                    f32::from_bits(expert as u32),
                    f32::from_bits(assignment as u32),
                    1.0,
                ]
            })
            .collect::<Vec<_>>();
        let metadata =
            Tensor::from_f32(&d, [assignments, 3], DType::F32, &metadata_values).unwrap();
        let actual = d
            .expert_project_quantized(&x, &metadata, &weight, columns, experts)
            .unwrap()
            .to_f32();
        let expected = (0..assignments)
            .flat_map(|assignment| {
                (0..rows_per_expert).map({
                    let rounded_input = &rounded_input;
                    let expert_ids = &expert_ids;
                    move |row| {
                        DType::BF16.round(
                            (0..columns)
                                .map(|column| {
                                    rounded_input[assignment * columns + column]
                                        * DType::BF16.round(q6_value(
                                            expert_ids[assignment] * rows_per_expert + row,
                                            column,
                                        ))
                                })
                                .sum::<f32>(),
                        )
                    }
                })
            })
            .collect::<Vec<_>>();
        for (index, (actual, expected)) in actual.iter().zip(&expected).enumerate() {
            assert!(
                (actual - expected).abs() <= 0.16,
                "Q6_K index {index}: actual={actual}, expected={expected}"
            );
        }
        assert_eq!(
            d.take_profile()
                .get("expert_project_q6_k_mpp_k64")
                .unwrap()
                .calls,
            1
        );
    }

    #[test]
    fn q6_k_expert_tensorops_covers_eight_routes_per_expert() {
        let d = MetalDevice::new().unwrap();
        if !d.mpp_projection() {
            return;
        }
        d.set_profiling(true);
        d.set_moe_expert_tensorops_min_routes_per_expert(8).unwrap();
        let (assignments, experts, rows_per_expert, columns) = (25, 3, 65, 512);
        let expert_ids = (0..assignments)
            .map(|assignment| (assignment + 1) % experts)
            .collect::<Vec<_>>();
        let packed = packed(&d, experts * rows_per_expert, columns);
        let weight = QuantizedExpertMatrix::new(packed, experts, rows_per_expert).unwrap();
        let values = input(assignments, columns);
        let x = Tensor::from_f32(&d, [assignments, columns], DType::BF16, &values).unwrap();
        let rounded_input = x.to_f32();
        let metadata_values = expert_ids
            .iter()
            .enumerate()
            .flat_map(|(assignment, &expert)| {
                [
                    f32::from_bits(expert as u32),
                    f32::from_bits(assignment as u32),
                    1.0,
                ]
            })
            .collect::<Vec<_>>();
        let metadata =
            Tensor::from_f32(&d, [assignments, 3], DType::F32, &metadata_values).unwrap();
        let actual = d
            .expert_project_quantized(&x, &metadata, &weight, columns, experts)
            .unwrap()
            .to_f32();
        let expected = (0..assignments)
            .flat_map(|assignment| {
                (0..rows_per_expert).map({
                    let rounded_input = &rounded_input;
                    let expert_ids = &expert_ids;
                    move |row| {
                        DType::BF16.round(
                            (0..columns)
                                .map(|column| {
                                    rounded_input[assignment * columns + column]
                                        * DType::BF16.round(q6_value(
                                            expert_ids[assignment] * rows_per_expert + row,
                                            column,
                                        ))
                                })
                                .sum::<f32>(),
                        )
                    }
                })
            })
            .collect::<Vec<_>>();
        for (index, (actual, expected)) in actual.iter().zip(&expected).enumerate() {
            assert!(
                (actual - expected).abs() <= 0.16,
                "Q6_K index {index}: actual={actual}, expected={expected}"
            );
        }
        assert_eq!(
            d.take_profile()
                .get("expert_project_q6_k_mpp_k64")
                .unwrap()
                .calls,
            1
        );
    }

    #[test]
    fn q6_k_rejects_columns_that_do_not_form_ggml_superblocks() {
        let d = MetalDevice::new().unwrap();
        assert!(
            QuantizedMatrix::from_reader(&d, 3, 255, QuantizationFormat::Q6_K, |_| Ok(())).is_err()
        );
    }
}

#[cfg(test)]
mod mlx_affine4_tests {
    use super::*;
    use crate::quantization::{QuantizationFormat, QuantizedMatrix};
    use half::f16;

    fn scale(row: usize, group: usize) -> f32 {
        let magnitude = 1 + (row * 3 + group * 5) % 5;
        let sign = if (row + group).is_multiple_of(2) {
            1.0
        } else {
            -1.0
        };
        f16::from_f32(sign * magnitude as f32 / 512.0).to_f32()
    }

    fn bias(row: usize, group: usize) -> f32 {
        f16::from_f32(((row + 2 * group) % 5) as f32 / 128.0 - 0.015625).to_f32()
    }

    fn quant(row: usize, column: usize) -> u8 {
        ((row * 3 + column * 7 + 5) & 15) as u8
    }

    fn packed(device: &MetalDevice, rows: usize, columns: usize) -> QuantizedMatrix {
        assert!(columns.is_multiple_of(64));
        let mut bytes = Vec::with_capacity(rows * columns / 64 * 36);
        for row in 0..rows {
            for group in 0..columns / 64 {
                bytes.extend_from_slice(&f16::from_f32(scale(row, group)).to_bits().to_le_bytes());
                bytes.extend_from_slice(&f16::from_f32(bias(row, group)).to_bits().to_le_bytes());
                for pair in 0..32 {
                    let even = quant(row, group * 64 + pair * 2);
                    let odd = quant(row, group * 64 + pair * 2 + 1);
                    bytes.push(even | (odd << 4));
                }
            }
        }
        QuantizedMatrix::from_reader(
            device,
            rows,
            columns,
            QuantizationFormat::MlxAffine4Group64,
            |destination| {
                destination.copy_from_slice(&bytes);
                Ok(())
            },
        )
        .unwrap()
    }

    fn value(row: usize, column: usize) -> f32 {
        let group = column / 64;
        scale(row, group) * f32::from(quant(row, column)) + bias(row, group)
    }

    fn input(rows: usize, columns: usize) -> Vec<f32> {
        (0..rows * columns)
            .map(|index| ((index * 29 + 11) % 113) as f32 / 127.0 - 0.42)
            .collect()
    }

    #[test]
    fn affine4_direct_gemv_and_gemm_match_signed_scale_reference_with_tails() {
        let device = MetalDevice::new().unwrap();
        let (n, k) = (5, 128);
        let weight = packed(&device, n, k);
        for m in [1, 2, 3, 5] {
            let values = input(m, k);
            let x = Tensor::from_f32(&device, [m, k], DType::F16, &values).unwrap();
            let rounded_input = x.to_f32();
            let rounded_input = &rounded_input;
            let actual = device.project_quantized(&x, &weight).unwrap();
            assert_eq!(
                actual.metrics.operation,
                if m == 1 {
                    "mlx_affine4_gemv"
                } else {
                    "mlx_affine4_gemm"
                }
            );
            let expected = (0..m)
                .flat_map(|row| {
                    (0..n).map(move |column| {
                        (0..k)
                            .map(|index| rounded_input[row * k + index] * value(column, index))
                            .sum::<f32>()
                    })
                })
                .collect::<Vec<_>>();
            for (index, (actual, expected)) in actual
                .tensor
                .to_f32()
                .iter()
                .zip(expected.iter().map(|&v| DType::F16.round(v)))
                .enumerate()
            {
                assert!(
                    (actual - expected).abs() <= 0.015,
                    "M={m}, index {index}: actual={actual}, expected={expected}"
                );
            }
        }
    }

    #[test]
    fn affine4_quad_gemv_reuses_activations_across_rows_and_covers_tails() {
        let device = MetalDevice::new().unwrap();
        let (n, k) = (2051, 192);
        let weight = packed(&device, n, k);
        let values = input(1, k);
        let x = Tensor::from_f32(&device, [1, k], DType::F16, &values).unwrap();
        assert!(device.use_mlx_affine4_gemv_quad(n, k));
        assert!(!device.use_mlx_affine4_gemv_quad(896, 896));
        assert!(!device.use_mlx_affine4_gemv_quad(896, 4864));
        let rounded_input = x.to_f32();
        let expected = (0..n)
            .map(|row| {
                (0..k)
                    .map(|column| rounded_input[column] * value(row, column))
                    .sum::<f32>()
            })
            .map(|value| DType::F16.round(value))
            .collect::<Vec<_>>();
        let candidate = device.project_quantized(&x, &weight).unwrap();
        assert_eq!(candidate.metrics.operation, "mlx_affine4_gemv_quad");
        device.set_mlx_affine4_gemv_quad(false).unwrap();
        let baseline = device.project_quantized(&x, &weight).unwrap();
        assert_eq!(baseline.metrics.operation, "mlx_affine4_gemv");
        for (index, ((actual, baseline), expected)) in candidate
            .tensor
            .to_f32()
            .iter()
            .zip(baseline.tensor.to_f32())
            .zip(expected)
            .enumerate()
        {
            assert!(
                (actual - expected).abs() <= 0.015 && (actual - baseline).abs() <= 0.015,
                "index {index}: candidate={actual}, baseline={baseline}, expected={expected}"
            );
        }
    }

    #[test]
    fn affine4_mpp_gemm_handles_batch_and_output_tails_across_k_tiles() {
        let device = MetalDevice::new().unwrap();
        if !device.mpp_projection() {
            return;
        }
        let (m, n, k) = (19, 65, 256);
        let weight = packed(&device, n, k);
        let values = input(m, k);
        let x = Tensor::from_f32(&device, [m, k], DType::F16, &values).unwrap();
        let rounded_input = x.to_f32();
        let rounded_input = &rounded_input;
        let expected = (0..m)
            .flat_map(|row| {
                (0..n).map(move |column| {
                    (0..k)
                        .map(|index| {
                            rounded_input[row * k + index] * DType::F16.round(value(column, index))
                        })
                        .sum::<f32>()
                })
            })
            .map(|value| DType::F16.round(value))
            .collect::<Vec<_>>();
        for (tile_k64, operation) in [
            (false, "mlx_affine4_gemm_mpp"),
            (true, "mlx_affine4_gemm_mpp_k64"),
        ] {
            device.set_mlx_affine4_mpp_tile_k64(Some(tile_k64)).unwrap();
            let actual = device.project_quantized(&x, &weight).unwrap();
            assert_eq!(actual.metrics.operation, operation);
            for (index, (actual, expected)) in
                actual.tensor.to_f32().iter().zip(&expected).enumerate()
            {
                assert!(
                    (actual - expected).abs() <= 0.06,
                    "tile_k64={tile_k64}, index {index}: actual={actual}, expected={expected}"
                );
            }
        }
        device.set_mlx_affine4_mpp_tile_k64(None).unwrap();
        assert!(!device.mlx_affine4_mpp_tile_k64(511));
        assert!(device.mlx_affine4_mpp_tile_k64(512));
        assert!(device.mlx_affine4_mpp_tile_k64(1024));
        assert!(!device.mlx_affine4_mpp_tile_k64(1025));
    }

    #[test]
    fn affine4_embedding_gather_dequantizes_only_requested_rows() {
        let device = MetalDevice::new().unwrap();
        let (vocab, hidden) = (5, 128);
        let weight = packed(&device, vocab, hidden);
        let ids = [4, 0, 3];
        let actual = device
            .embedding_gather_quantized(&weight, &ids, DType::F16)
            .unwrap()
            .tensor
            .to_f32();
        let expected = ids
            .iter()
            .flat_map(|&row| {
                (0..hidden).map(move |column| DType::F16.round(value(row as usize, column)))
            })
            .collect::<Vec<_>>();
        assert_eq!(actual, expected);
    }
}

#[cfg(test)]
mod lfm2_short_conv_tests {
    use crate::{DType, MetalDevice, Tensor};

    fn exercise_short_conv(dtype: DType, sequence: usize, hidden: usize, kernel: usize) {
        let device = MetalDevice::new().unwrap();
        let bx_values = (0..sequence * hidden)
            .map(|i| (i as i32 % 11 - 5) as f32 / 8.)
            .collect::<Vec<_>>();
        let gate_values = (0..sequence * hidden)
            .map(|i| (i % 7 + 1) as f32 / 4.)
            .collect::<Vec<_>>();
        let weight_values = (0..hidden * kernel)
            .map(|i| (i as i32 % 5 - 2) as f32 / 8.)
            .collect::<Vec<_>>();
        let state_values = (0..kernel * hidden)
            .map(|i| (i as i32 % 9 - 4) as f32 / 8.)
            .collect::<Vec<_>>();
        let bx = Tensor::from_f32(&device, [sequence, hidden], dtype, &bx_values).unwrap();
        let gate = Tensor::from_f32(&device, [sequence, hidden], dtype, &gate_values).unwrap();
        let weight = Tensor::from_f32(&device, [hidden, 1, kernel], dtype, &weight_values).unwrap();
        let previous = Tensor::from_f32(&device, [kernel, hidden], dtype, &state_values).unwrap();
        let bx_values = bx.to_f32();
        let gate_values = gate.to_f32();
        let weight_values = weight.to_f32();
        let state_values = previous.to_f32();

        let mut expected_output = vec![0.; sequence * hidden];
        for token in 0..sequence {
            for channel in 0..hidden {
                let mut sum = 0.;
                for tap in 0..kernel {
                    let joined = token + 1 + tap;
                    let value = if joined < kernel {
                        state_values[joined * hidden + channel]
                    } else {
                        bx_values[(joined - kernel) * hidden + channel]
                    };
                    sum += value * weight_values[channel * kernel + tap];
                }
                let conv_value = dtype.round(sum);
                expected_output[token * hidden + channel] =
                    dtype.round(conv_value * gate_values[token * hidden + channel]);
            }
        }
        let mut expected_state = vec![0.; kernel * hidden];
        for position in 0..kernel {
            for channel in 0..hidden {
                let joined = sequence + position;
                expected_state[position * hidden + channel] = if joined < kernel {
                    state_values[joined * hidden + channel]
                } else {
                    bx_values[(joined - kernel) * hidden + channel]
                };
            }
        }

        let (actual_output, actual_state) = device
            .lfm2_short_conv(&bx, &gate, &weight, &previous, kernel)
            .unwrap();
        assert_eq!(actual_output.to_f32(), expected_output);
        assert_eq!(actual_state.to_f32(), expected_state);
    }

    #[test]
    fn lfm2_short_conv_matches_causal_reference_and_updates_fresh_state() {
        for dtype in [DType::F32, DType::F16, DType::BF16] {
            for (sequence, hidden, kernel) in [(1, 3, 3), (4, 7, 3), (2, 5, 1)] {
                exercise_short_conv(dtype, sequence, hidden, kernel);
            }
        }
    }

    #[test]
    fn lfm2_split3_deinterleaves_each_token_row() {
        let device = MetalDevice::new().unwrap();
        for dtype in [DType::F16, DType::BF16] {
            let (sequence, hidden) = (3, 5);
            let source = (0..sequence * hidden * 3)
                .map(|index| (index as f32 - 13.) / 16.)
                .collect::<Vec<_>>();
            let projected =
                Tensor::from_f32(&device, [sequence, 3 * hidden], dtype, &source).unwrap();
            let rounded = projected.to_f32();
            let (b, c, x) = device.lfm2_split3(&projected).unwrap();
            for (part, actual) in [(0, b), (1, c), (2, x)] {
                let expected = (0..sequence)
                    .flat_map(|token| {
                        let rounded = &rounded;
                        (0..hidden).map(move |channel| {
                            rounded[token * 3 * hidden + part * hidden + channel]
                        })
                    })
                    .collect::<Vec<_>>();
                assert_eq!(actual.to_f32(), expected);
            }
        }
    }
}
