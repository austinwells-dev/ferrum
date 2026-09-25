#![forbid(unsafe_code)]
use crate::{
    DType, Error, MetalDevice, Result, Tensor,
    metal::DispatchTiming,
    model::architecture::{MoeRoutingPolicy, MoeScoringFunction},
    quantization::QuantizedMatrix,
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
    pub(crate) fn moe_route(
        &self,
        logits: &Tensor,
        selection_bias: Option<&Tensor>,
        policy: MoeRoutingPolicy,
    ) -> Result<Tensor> {
        let dims = logits.shape().dimensions();
        if dims.len() != 2
            || dims[0] == 0
            || dims[1] == 0
            || dims[1] != policy.experts
            || policy.top_k == 0
            || policy.top_k > 16
            || policy.top_k > policy.experts
            || !policy.normalization_epsilon.is_finite()
            || policy.normalization_epsilon < 0.
            || !policy.routed_scaling_factor.is_finite()
            || policy.routed_scaling_factor <= 0.
            || !self.owns(logits.buffer())
            || selection_bias.is_some_and(|bias| {
                bias.shape().dimensions() != [policy.experts]
                    || bias.dtype() != DType::F32
                    || !self.owns(bias.buffer())
            })
        {
            return Err(Error::Shape("invalid sparse routing geometry".into()));
        }
        let assignments = dims[0]
            .checked_mul(policy.top_k)
            .ok_or_else(|| Error::Shape("expert assignment count overflow".into()))?;
        let output = Tensor::output(self, &[assignments, 3], DType::F32)?;
        let mut params = [0; 9];
        params[0] = policy.normalization_epsilon.to_bits();
        params[1] = index(dims[0])?;
        params[2] = index(dims[1])?;
        params[3] = index(policy.top_k)?;
        params[4] = logits.dtype() as u32;
        params[5] = u32::from(selection_bias.is_some());
        params[6] = u32::from(policy.scoring_function == MoeScoringFunction::Sigmoid);
        params[7] = u32::from(policy.normalize_top_k_prob);
        params[8] = policy.routed_scaling_factor.to_bits();
        let bias = selection_bias.unwrap_or(logits);
        self.dispatch(
            "moe_route",
            &[logits.binding(), bias.binding(), output.binding()],
            &params,
            [dims[0], 1],
            false,
        )?;
        Ok(output)
    }

    pub(crate) fn expert_assign(
        &self,
        hidden: &Tensor,
        metadata: &Tensor,
        top_k: usize,
    ) -> Result<Tensor> {
        let x = hidden.shape().dimensions();
        let meta = metadata.shape().dimensions();
        if x.len() != 2
            || x[0] == 0
            || x[1] == 0
            || meta.len() != 2
            || meta[1] != 3
            || top_k == 0
            || x[0].checked_mul(top_k) != Some(meta[0])
            || metadata.dtype() != DType::F32
        {
            return Err(Error::Shape(
                "expert assignments require [S,H] and [S*top_k,3]".into(),
            ));
        }
        let dims = [meta[0], x[1]];
        let elements = dims[0]
            .checked_mul(dims[1])
            .ok_or_else(|| Error::Shape("expert assignment size overflow".into()))?;
        let mut p = [0; 9];
        p[1] = index(x[0])?;
        p[2] = index(x[1])?;
        p[3] = index(top_k)?;
        p[5] = DType::F32 as u32;
        Ok(self
            .run(
                "expert_assign",
                hidden,
                Some(metadata),
                &dims,
                p,
                [elements, 1],
            )?
            .tensor)
    }

    pub(crate) fn expert_project(
        &self,
        input: &Tensor,
        metadata: &Tensor,
        weight: &Tensor,
        features: usize,
        experts: usize,
    ) -> Result<Tensor> {
        let x = input.shape().dimensions();
        let meta = metadata.shape().dimensions();
        let w = weight.shape().dimensions();
        if x.len() != 2
            || x[0] == 0
            || x[1] != features
            || meta != [x[0], 3]
            || w.len() != 3
            || w[0] != experts
            || w[2] != features
            || input.dtype() != weight.dtype()
            || metadata.dtype() != DType::F32
            || experts == 0
            || features == 0
            || w[1] == 0
        {
            return Err(Error::Shape("expert projection geometry mismatch".into()));
        }
        let rows = w[1];
        let dims = [x[0], rows];
        let mut p = [0; 9];
        p[0] = index(input.numel())?;
        p[1] = index(x[0])?;
        p[2] = index(features)?;
        p[3] = index(rows)?;
        p[4] = input.dtype() as u32;
        p[5] = index(experts)?;
        p[6] = DType::F32 as u32;
        let profile_start = self.profiling().then(std::time::Instant::now);
        let wait_before = profile_start
            .map(|_| self.counters().wait)
            .unwrap_or_default();
        let allocation_start = profile_start.map(|_| std::time::Instant::now());
        let tensor = Tensor::output(self, &dims, input.dtype())?;
        let allocation_time = allocation_start
            .map(|start| start.elapsed())
            .unwrap_or_default();
        let timing = self.dispatch(
            "expert_project",
            &[
                input.binding(),
                weight.binding(),
                metadata.binding(),
                tensor.binding(),
            ],
            &p,
            [rows, x[0]],
            false,
        )?;
        if let Some(start) = profile_start {
            self.record_profile(
                "expert_project",
                start
                    .elapsed()
                    .saturating_sub(self.counters().wait - wait_before),
                allocation_time,
                tensor.storage_info().allocation_bytes,
                &timing,
            );
        }
        Ok(tensor)
    }

    pub(crate) fn expert_project_quantized(
        &self,
        input: &Tensor,
        metadata: &Tensor,
        weight: &crate::quantization::QuantizedExpertMatrix,
        features: usize,
        experts: usize,
    ) -> Result<Tensor> {
        use crate::quantization::QuantizationFormat;

        let x = input.shape().dimensions();
        let meta = metadata.shape().dimensions();
        if x.len() != 2
            || x[0] == 0
            || x[1] != features
            || meta != [x[0], 3]
            || metadata.dtype() != DType::F32
            || experts == 0
            || weight.experts() != experts
            || weight.columns() != features
            || weight.rows_per_expert() == 0
            || !self.owns(input.buffer())
            || !self.owns(metadata.buffer())
            || !self.owns(weight.buffer())
        {
            return Err(Error::Shape(
                "quantized expert projection geometry mismatch".into(),
            ));
        }
        let (name, ggml_type) = match weight.format() {
            QuantizationFormat::Q4_K => ("expert_project_q4_k", 12),
            QuantizationFormat::Q5_K => ("expert_project_q5_k", 13),
            QuantizationFormat::Q6_K => ("expert_project_q6_k", 14),
            format => {
                return Err(Error::Gguf(format!(
                    "no quantized expert projection for {format:?}"
                )));
            }
        };
        let rows = weight.rows_per_expert();
        let dims = [x[0], rows];
        let shape = crate::tensor::Shape::new(dims)?;
        index(shape.numel())?;
        let tensorops_min_rows = experts
            .checked_mul(self.moe_expert_tensorops_min_routes_per_expert())
            .unwrap_or(usize::MAX);
        let use_tensorops = input.dtype() == DType::BF16
            && self.moe_expert_tensorops_enabled()
            && x[0] >= tensorops_min_rows
            && rows >= 64
            && features >= 128
            && features.is_multiple_of(128)
            && x[0] <= i32::MAX as usize
            && rows <= i32::MAX as usize
            && features <= i32::MAX as usize
            && self.mpp_projection();
        if use_tensorops {
            let tile_k64 = self.moe_expert_tensorops_tile_k64();
            let mpp_name = match (ggml_type, tile_k64) {
                (12, false) => "expert_project_q4_k_mpp",
                (13, false) => "expert_project_q5_k_mpp",
                (14, false) => "expert_project_q6_k_mpp",
                (12, true) => "expert_project_q4_k_mpp_k64",
                (13, true) => "expert_project_q5_k_mpp",
                (14, true) => "expert_project_q6_k_mpp_k64",
                _ => unreachable!(),
            };
            return self.expert_project_quantized_mpp(
                input, metadata, weight, experts, features, rows, ggml_type, mpp_name,
            );
        }
        let dispatch_name = match ggml_type {
            12 if self.use_q4_k_expert_project_8rows(rows, features) => "expert_project_q4_k_8rows",
            14 if self.use_q6_k_expert_project_8rows(rows, features) => "expert_project_q6_k_8rows",
            _ => name,
        };
        let mut p = [0; 9];
        p[0] = index(shape.numel())?;
        p[1] = index(x[0])?;
        p[2] = index(features)?;
        p[3] = index(rows)?;
        p[4] = input.dtype() as u32;
        p[5] = index(experts)?;
        p[6] = DType::F32 as u32;
        p[7] = ggml_type;
        let profile_start = self.profiling().then(std::time::Instant::now);
        let wait_before = profile_start
            .map(|_| self.counters().wait)
            .unwrap_or_default();
        let allocation_start = profile_start.map(|_| std::time::Instant::now());
        let tensor = Tensor::output(self, &dims, input.dtype())?;
        let allocation_time = allocation_start
            .map(|start| start.elapsed())
            .unwrap_or_default();
        let timing = self.dispatch(
            dispatch_name,
            &[
                input.binding(),
                weight.matrix().binding(),
                metadata.binding(),
                tensor.binding(),
            ],
            &p,
            [rows, x[0]],
            false,
        )?;
        if let Some(start) = profile_start {
            self.record_profile(
                dispatch_name,
                start
                    .elapsed()
                    .saturating_sub(self.counters().wait - wait_before),
                allocation_time,
                tensor.storage_info().allocation_bytes,
                &timing,
            );
        }
        Ok(tensor)
    }

    fn expert_project_quantized_mpp(
        &self,
        input: &Tensor,
        metadata: &Tensor,
        weight: &crate::quantization::QuantizedExpertMatrix,
        experts: usize,
        features: usize,
        rows: usize,
        ggml_type: u32,
        name: &'static str,
    ) -> Result<Tensor> {
        let assignments = input.shape().dimensions()[0];
        let padded_capacity = assignments
            .checked_add(
                experts
                    .checked_mul(31)
                    .ok_or_else(|| Error::Shape("expert TensorOps padding overflow".into()))?,
            )
            .ok_or_else(|| Error::Shape("expert TensorOps capacity overflow".into()))?;
        let compact_elements = padded_capacity
            .checked_mul(features)
            .ok_or_else(|| Error::Shape("expert TensorOps input size overflow".into()))?;
        let output_dims = [assignments, rows];
        let output_shape = crate::tensor::Shape::new(output_dims)?;
        index(padded_capacity)?;
        index(compact_elements)?;
        index(output_shape.numel())?;

        let profile_start = self.profiling().then(std::time::Instant::now);
        let wait_before = profile_start
            .map(|_| self.counters().wait)
            .unwrap_or_default();
        let allocation_start = profile_start.map(|_| std::time::Instant::now());
        let counts = Tensor::output(self, &[experts], DType::F32)?;
        let bases = Tensor::output(self, &[experts], DType::F32)?;
        let cursors = Tensor::output(self, &[experts], DType::F32)?;
        let compact = Tensor::output(self, &[padded_capacity, features], DType::BF16)?;
        let assignment_ids = Tensor::output(self, &[padded_capacity], DType::F32)?;
        let output = Tensor::output(self, &output_dims, DType::BF16)?;
        let allocation_time = allocation_start
            .map(|start| start.elapsed())
            .unwrap_or_default();
        let allocation_bytes = [
            counts.storage_info().allocation_bytes,
            bases.storage_info().allocation_bytes,
            cursors.storage_info().allocation_bytes,
            compact.storage_info().allocation_bytes,
            assignment_ids.storage_info().allocation_bytes,
            output.storage_info().allocation_bytes,
        ]
        .into_iter()
        .sum();

        let mut segments_params = [0; 9];
        segments_params[1] = index(assignments)?;
        segments_params[2] = index(experts)?;
        segments_params[3] = DType::F32 as u32;
        let segments_timing = self.dispatch(
            "expert_segments",
            &[
                metadata.binding(),
                counts.binding(),
                bases.binding(),
                cursors.binding(),
            ],
            &segments_params,
            [1, 1],
            false,
        )?;

        let mut compact_params = [0; 9];
        compact_params[1] = index(assignments)?;
        compact_params[2] = index(features)?;
        compact_params[3] = index(experts)?;
        compact_params[4] = DType::F32 as u32;
        let compact_timing = self.dispatch(
            "expert_assign_compact",
            &[
                input.binding(),
                metadata.binding(),
                cursors.binding(),
                compact.binding(),
                assignment_ids.binding(),
            ],
            &compact_params,
            [1, assignments],
            false,
        )?;

        let mut mpp_params = [0; 9];
        mpp_params[0] = index(output_shape.numel())?;
        mpp_params[1] = index(assignments)?;
        mpp_params[2] = index(features)?;
        mpp_params[3] = index(rows)?;
        mpp_params[4] = DType::BF16 as u32;
        mpp_params[5] = index(experts)?;
        mpp_params[6] = DType::F32 as u32;
        mpp_params[7] = ggml_type;
        let mpp_timing = self.dispatch(
            name,
            &[
                compact.binding(),
                weight.matrix().binding(),
                assignment_ids.binding(),
                counts.binding(),
                bases.binding(),
                output.binding(),
            ],
            &mpp_params,
            [rows, assignments],
            false,
        )?;

        if let Some(start) = profile_start {
            let mut timing = DispatchTiming::default();
            let mut gpu = std::time::Duration::ZERO;
            let mut has_gpu = true;
            for part in [segments_timing, compact_timing, mpp_timing] {
                timing.submission += part.submission;
                timing.synchronized += part.synchronized;
                timing.dispatches += part.dispatches;
                if let Some(part_gpu) = part.gpu {
                    gpu += part_gpu;
                } else {
                    has_gpu = false;
                }
            }
            timing.gpu = has_gpu.then_some(gpu);
            self.record_profile(
                name,
                start
                    .elapsed()
                    .saturating_sub(self.counters().wait - wait_before),
                allocation_time,
                allocation_bytes,
                &timing,
            );
        }
        Ok(output)
    }

    pub(crate) fn expert_silu_mul(&self, input: &Tensor, intermediate: usize) -> Result<Tensor> {
        let x = input.shape().dimensions();
        if x.len() != 2 || intermediate.checked_mul(2) != Some(x[1]) || intermediate == 0 {
            return Err(Error::Shape(
                "packed expert activation geometry mismatch".into(),
            ));
        }
        let dims = [x[0], intermediate];
        let mut p = [0; 9];
        p[1] = index(x[0])?;
        p[2] = index(intermediate)?;
        Ok(self
            .run(
                "expert_silu_mul",
                input,
                None,
                &dims,
                p,
                [dims[0] * dims[1], 1],
            )?
            .tensor)
    }

    pub(crate) fn expert_combine(
        &self,
        input: &Tensor,
        metadata: &Tensor,
        tokens: usize,
        hidden: usize,
        top_k: usize,
    ) -> Result<Tensor> {
        let x = input.shape().dimensions();
        if top_k == 0
            || hidden == 0
            || x != [
                tokens
                    .checked_mul(top_k)
                    .ok_or_else(|| Error::Shape("expert combine geometry overflow".into()))?,
                hidden,
            ]
            || metadata.shape().dimensions() != [x[0], 3]
            || metadata.dtype() != DType::F32
        {
            return Err(Error::Shape("expert combine geometry mismatch".into()));
        }
        let dims = [tokens, hidden];
        let mut p = [0; 9];
        p[1] = index(tokens)?;
        p[2] = index(hidden)?;
        p[3] = index(top_k)?;
        p[5] = DType::F32 as u32;
        let elements = tokens
            .checked_mul(hidden)
            .ok_or_else(|| Error::Shape("expert combine output size overflow".into()))?;
        Ok(self
            .run(
                "expert_combine",
                input,
                Some(metadata),
                &dims,
                p,
                [elements, 1],
            )?
            .tensor)
    }

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
        let f32_route_metadata = matches!(name, "expert_assign" | "expert_combine")
            && b.is_some_and(|b| b.dtype() == DType::F32);
        if b.is_some_and(|b| b.dtype() != a.dtype()) && !f32_route_metadata {
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
            "q4_k_gemv_8rows",
            "q4_k_gemm",
            "q4_k_gemm_mpp",
            "q4_k_gemm_mpp_k64",
            "q5_k_gemv",
            "q5_k_gemm",
            "q5_k_gemm_mpp",
            "q6_k_gemm_mpp",
            "q6_k_gemv",
            "q6_k_gemv_8rows",
            "q6_k_gemm",
            "q8_0_gemv",
            "q8_0_gemv_8rows",
            "q8_0_gemm",
            "q8_0_gemm_mpp_k64",
            "mlx_affine4_gemv",
            "mlx_affine4_gemm",
            "mlx_affine4_gemm_mpp",
            "mlx_affine4_gemm_mpp_k64",
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
