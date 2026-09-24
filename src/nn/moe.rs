//! Shared sparse top-k expert routing and grouped expert projection runtime.
#![forbid(unsafe_code)]

use crate::model::architecture::{MoeRoutingPolicy, MoeScoringFunction};
use crate::{Error, MetalDevice, Result, Tensor, quantization::QuantizedExpertMatrix};
use std::{
    cell::Cell,
    time::{Duration, Instant},
};

const ROUTING_TEMPORARY_LIMIT: usize = 16 * 1024 * 1024;

#[derive(Clone)]
pub(crate) enum ExpertMatrix {
    Dense(Tensor),
    Quantized(QuantizedExpertMatrix),
}

impl ExpertMatrix {
    fn geometry(&self) -> Option<(usize, usize, usize)> {
        match self {
            Self::Dense(tensor) => {
                let dims = tensor.shape().dimensions();
                (dims.len() == 3).then(|| (dims[0], dims[1], dims[2]))
            }
            Self::Quantized(tensor) => {
                Some((tensor.experts(), tensor.rows_per_expert(), tensor.columns()))
            }
        }
    }

    fn dtype(&self) -> Option<crate::DType> {
        match self {
            Self::Dense(tensor) => Some(tensor.dtype()),
            Self::Quantized(_) => None,
        }
    }

    fn byte_size(&self) -> usize {
        match self {
            Self::Dense(tensor) => tensor.byte_size(),
            Self::Quantized(tensor) => tensor.byte_size(),
        }
    }

    fn is_owned_by(&self, device: &MetalDevice) -> bool {
        match self {
            Self::Dense(tensor) => device.owns(tensor.buffer()),
            Self::Quantized(tensor) => device.owns(tensor.buffer()),
        }
    }

    fn project(
        &self,
        device: &MetalDevice,
        input: &Tensor,
        metadata: &Tensor,
        features: usize,
        experts: usize,
    ) -> Result<Tensor> {
        match self {
            Self::Dense(tensor) => {
                device.expert_project(input, metadata, tensor, features, experts)
            }
            Self::Quantized(tensor) => {
                device.expert_project_quantized(input, metadata, tensor, features, experts)
            }
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct MoeStats {
    pub router_projection_enqueue: Duration,
    pub routing: Duration,
    pub routing_boundary_wait: Duration,
    pub expert_dispatch: Duration,
    pub combine_dispatch: Duration,
    pub active_experts: usize,
    pub active_experts_known: bool,
    pub assignments: usize,
    pub peak_temporary_bytes: usize,
}

impl Default for MoeStats {
    fn default() -> Self {
        Self {
            router_projection_enqueue: Duration::default(),
            routing: Duration::default(),
            routing_boundary_wait: Duration::default(),
            expert_dispatch: Duration::default(),
            combine_dispatch: Duration::default(),
            active_experts: 0,
            active_experts_known: true,
            assignments: 0,
            peak_temporary_bytes: 0,
        }
    }
}

impl MoeStats {
    fn merge(&mut self, other: Self) {
        self.router_projection_enqueue += other.router_projection_enqueue;
        self.routing += other.routing;
        self.routing_boundary_wait += other.routing_boundary_wait;
        self.expert_dispatch += other.expert_dispatch;
        self.combine_dispatch += other.combine_dispatch;
        self.active_experts += other.active_experts;
        self.active_experts_known &= other.active_experts_known;
        self.assignments += other.assignments;
        self.peak_temporary_bytes = self.peak_temporary_bytes.max(other.peak_temporary_bytes);
    }
}

pub struct SparseMoe {
    router: crate::nn::Linear,
    input_experts: ExpertMatrix,
    output_experts: ExpertMatrix,
    selection_bias: Option<Vec<f32>>,
    selection_bias_tensor: Option<Tensor>,
    expert_count: usize,
    routing: MoeRoutingPolicy,
    hidden_size: usize,
    intermediate_size: usize,
    stats: Cell<MoeStats>,
}

impl SparseMoe {
    pub(crate) fn new(
        device: &MetalDevice,
        router_weight: Tensor,
        input_experts: ExpertMatrix,
        output_experts: ExpertMatrix,
        routing: MoeRoutingPolicy,
        selection_bias: Option<Tensor>,
    ) -> Result<Self> {
        let router_dims = router_weight.shape().dimensions();
        let input_dims = input_experts.geometry();
        let output_dims = output_experts.geometry();
        if router_dims.len() != 2
            || input_dims.is_none()
            || output_dims.is_none()
            || input_dims.is_some_and(|dims| {
                router_dims[0] != dims.0
                    || dims.1 == 0
                    || !dims.1.is_multiple_of(2)
                    || router_dims[1] != dims.2
            })
            || output_dims.is_some_and(|dims| router_dims[0] != dims.0 || router_dims[1] != dims.1)
            || input_dims
                .zip(output_dims)
                .is_some_and(|(input, output)| output.2.checked_mul(2) != Some(input.1))
            || routing.experts != router_dims[0]
            || routing.top_k == 0
            || routing.top_k > router_dims[0]
            || !routing.normalization_epsilon.is_finite()
            || routing.normalization_epsilon < 0.
            || !routing.routed_scaling_factor.is_finite()
            || routing.routed_scaling_factor <= 0.
            || routing.use_expert_bias != selection_bias.is_some()
            || input_experts
                .dtype()
                .is_some_and(|dtype| dtype != router_weight.dtype())
            || output_experts
                .dtype()
                .is_some_and(|dtype| dtype != router_weight.dtype())
            || !device.owns(router_weight.buffer())
            || !input_experts.is_owned_by(device)
            || !output_experts.is_owned_by(device)
            || selection_bias.as_ref().is_some_and(|bias| {
                bias.shape().dimensions() != [router_dims.first().copied().unwrap_or(0)]
                    || bias.dtype() != crate::DType::F32
                    || !device.owns(bias.buffer())
            })
        {
            return Err(Error::Config(
                "invalid sparse expert weight geometry".into(),
            ));
        }
        let expert_count = router_dims[0];
        let hidden_size = router_dims[1];
        let intermediate_size = input_dims.expect("validated input expert geometry").1 / 2;
        let selection_bias_values = selection_bias.as_ref().map(Tensor::to_f32);
        if selection_bias_values
            .as_ref()
            .is_some_and(|bias| bias.iter().any(|value| !value.is_finite()))
        {
            return Err(Error::Config("MoE selection bias must be finite".into()));
        }
        let router = crate::nn::Linear::new(device, router_weight, None)?;
        Ok(Self {
            router,
            input_experts,
            output_experts,
            selection_bias: selection_bias_values,
            selection_bias_tensor: selection_bias,
            expert_count,
            routing,
            hidden_size,
            intermediate_size,
            stats: Cell::new(MoeStats::default()),
        })
    }

    pub fn weight_bytes(&self) -> usize {
        self.router.weight_bytes()
            + self.input_experts.byte_size()
            + self.output_experts.byte_size()
            + self
                .selection_bias
                .as_ref()
                .map_or(0, |bias| bias.len() * std::mem::size_of::<f32>())
    }

    pub fn take_stats(&self) -> MoeStats {
        self.stats.replace(MoeStats::default())
    }

    pub fn forward(&self, device: &MetalDevice, hidden: &Tensor) -> Result<Tensor> {
        let dims = hidden.shape().dimensions();
        if dims.len() != 2 || dims[1] != self.hidden_size || dims[0] == 0 {
            return Err(Error::Shape(
                "sparse expert input requires [tokens,hidden]".into(),
            ));
        }
        if !device.owns(hidden.buffer()) {
            return Err(Error::DeviceMismatch);
        }
        let activation_elements_per_assignment = self
            .hidden_size
            .checked_mul(2)
            .and_then(|v| v.checked_add(self.intermediate_size.checked_mul(3)?))
            .ok_or_else(|| Error::Shape("expert temporary size overflow".into()))?;
        let bytes_per_assignment = activation_elements_per_assignment
            .checked_mul(hidden.dtype().size_bytes())
            .and_then(|v| v.checked_add(3 * std::mem::size_of::<f32>()))
            .ok_or_else(|| Error::Shape("expert temporary size overflow".into()))?;
        let bytes_per_token = bytes_per_assignment
            .checked_mul(self.routing.top_k)
            .and_then(|v| {
                v.checked_add(
                    self.expert_count
                        .checked_add(self.hidden_size)?
                        .checked_mul(hidden.dtype().size_bytes())?,
                )
            })
            .ok_or_else(|| Error::Shape("expert temporary size overflow".into()))?;
        let chunk_tokens = (ROUTING_TEMPORARY_LIMIT / bytes_per_token)
            .max(1)
            .min(dims[0]);
        let chunks = dims[0].div_ceil(chunk_tokens);
        let output_elements = dims[0]
            .checked_mul(self.hidden_size)
            .ok_or_else(|| Error::Shape("expert output size overflow".into()))?;
        let mut all_outputs = Vec::with_capacity(output_elements);
        let mut total_stats = MoeStats::default();

        for chunk_start in (0..dims[0]).step_by(chunk_tokens) {
            let count = chunk_tokens.min(dims[0] - chunk_start);
            let input = hidden.view(chunk_start * self.hidden_size, [count, self.hidden_size])?;
            let route_start = Instant::now();
            let router_logits =
                device.profile_projection("moe.router", || self.router.forward(device, &input))?;
            total_stats.router_projection_enqueue += route_start.elapsed();
            let assignment_count = count
                .checked_mul(self.routing.top_k)
                .ok_or_else(|| Error::Shape("expert assignment count overflow".into()))?;
            total_stats.assignments += assignment_count;
            let metadata_tensor = if device.use_moe_gpu_routing(self.routing.top_k, dims[0]) {
                let route_start = Instant::now();
                let metadata = device.profile_projection("moe.route", || {
                    device.moe_route(
                        &router_logits,
                        self.selection_bias_tensor.as_ref(),
                        self.routing,
                    )
                })?;
                total_stats.routing += route_start.elapsed();
                total_stats.active_experts_known = false;
                metadata
            } else {
                // Router choice is data-dependent. Complete the GPU projection before
                // reading it; all staged cache changes remain local to the caller.
                let boundary_start = Instant::now();
                device.synchronize()?;
                total_stats.routing_boundary_wait += boundary_start.elapsed();
                let route_start = Instant::now();
                let logits = router_logits.to_f32();
                let metadata = self.select_routes(&logits, count)?;
                total_stats.active_experts += metadata
                    .chunks_exact(3)
                    .map(|item| item[0].to_bits() as usize)
                    .collect::<std::collections::BTreeSet<_>>()
                    .len();
                total_stats.routing += route_start.elapsed();
                Tensor::from_f32(device, [assignment_count, 3], crate::DType::F32, &metadata)?
            };
            let temp_bytes = assignment_count
                .checked_mul(bytes_per_assignment)
                .and_then(|v| {
                    v.checked_add(
                        count
                            .checked_mul(self.expert_count.checked_add(self.hidden_size)?)?
                            .checked_mul(hidden.dtype().size_bytes())?,
                    )
                })
                .ok_or_else(|| Error::Shape("expert temporary size overflow".into()))?;
            total_stats.peak_temporary_bytes = total_stats.peak_temporary_bytes.max(temp_bytes);

            let dispatch_start = Instant::now();
            let assigned = device.profile_projection("moe.assign", || {
                device.expert_assign(&input, &metadata_tensor, self.routing.top_k)
            })?;
            let projected = device.profile_projection("moe.input_projection", || {
                self.input_experts.project(
                    device,
                    &assigned,
                    &metadata_tensor,
                    self.hidden_size,
                    self.expert_count,
                )
            })?;
            let activated = device.profile_projection("moe.silu_mul", || {
                device.expert_silu_mul(&projected, self.intermediate_size)
            })?;
            let expert_output = device.profile_projection("moe.output_projection", || {
                self.output_experts.project(
                    device,
                    &activated,
                    &metadata_tensor,
                    self.intermediate_size,
                    self.expert_count,
                )
            })?;
            total_stats.expert_dispatch += dispatch_start.elapsed();
            let combine_start = Instant::now();
            let combined = device.profile_projection("moe.combine", || {
                device.expert_combine(
                    &expert_output,
                    &metadata_tensor,
                    count,
                    self.hidden_size,
                    self.routing.top_k,
                )
            })?;
            total_stats.combine_dispatch += combine_start.elapsed();

            if chunks == 1 {
                let mut accumulated = self.stats.get();
                accumulated.merge(total_stats);
                self.stats.set(accumulated);
                return Ok(combined);
            }
            device.synchronize()?;
            all_outputs.extend(combined.to_f32());
        }
        let mut accumulated = self.stats.get();
        accumulated.merge(total_stats);
        self.stats.set(accumulated);
        Tensor::from_f32(
            device,
            [dims[0], self.hidden_size],
            hidden.dtype(),
            &all_outputs,
        )
    }

    fn select_routes(&self, logits: &[f32], tokens: usize) -> Result<Vec<f32>> {
        if tokens.checked_mul(self.expert_count) != Some(logits.len())
            || logits.iter().any(|value| !value.is_finite())
        {
            return Err(Error::Validation("router produced invalid logits".into()));
        }
        let capacity = tokens
            .checked_mul(self.routing.top_k)
            .and_then(|value| value.checked_mul(3))
            .ok_or_else(|| Error::Shape("expert routing metadata overflow".into()))?;
        let mut metadata = Vec::with_capacity(capacity);
        for token in 0..tokens {
            let row = &logits[token * self.expert_count..(token + 1) * self.expert_count];
            let scores = match self.routing.scoring_function {
                MoeScoringFunction::Softmax => {
                    let maximum = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                    let mut values: Vec<_> =
                        row.iter().map(|score| (score - maximum).exp()).collect();
                    let normalizer: f32 = values.iter().sum();
                    for value in &mut values {
                        *value /= normalizer;
                    }
                    values
                }
                MoeScoringFunction::Sigmoid => {
                    row.iter().map(|score| 1. / (1. + (-score).exp())).collect()
                }
            };
            let mut order: Vec<usize> = (0..self.expert_count).collect();
            order.sort_by(|&a, &b| {
                let score_a = scores[a] + self.selection_bias.as_ref().map_or(0., |bias| bias[a]);
                let score_b = scores[b] + self.selection_bias.as_ref().map_or(0., |bias| bias[b]);
                score_b.total_cmp(&score_a).then_with(|| a.cmp(&b))
            });
            order.truncate(self.routing.top_k);
            let mut selected: Vec<_> = order
                .into_iter()
                .map(|expert| (expert, scores[expert]))
                .collect();
            if self.routing.normalize_top_k_prob {
                let normalizer: f32 = selected.iter().map(|(_, weight)| weight).sum();
                let denominator = normalizer + self.routing.normalization_epsilon;
                if !denominator.is_finite() || denominator <= 0. {
                    return Err(Error::Validation(
                        "router produced invalid selected scores".into(),
                    ));
                }
                for (_, weight) in &mut selected {
                    *weight /= denominator;
                }
            }
            for (_, weight) in &mut selected {
                *weight *= self.routing.routed_scaling_factor;
            }
            // Grouped expert execution uses the reference's expert-ID ordering
            // when combining the selected contributions for each token.
            selected.sort_by_key(|(expert, _)| *expert);
            for (expert, weight) in selected {
                let expert_id = u32::try_from(expert)
                    .map_err(|_| Error::Shape("expert ID exceeds routing index range".into()))?;
                let token_id = u32::try_from(token)
                    .map_err(|_| Error::Shape("token ID exceeds routing index range".into()))?;
                metadata.extend([f32::from_bits(expert_id), f32::from_bits(token_id), weight]);
            }
        }
        Ok(metadata)
    }
}

#[cfg(test)]
mod tests {
    use super::{ExpertMatrix, MoeRoutingPolicy, ROUTING_TEMPORARY_LIMIT, SparseMoe};
    use crate::model::architecture::MoeScoringFunction;
    use crate::{DType, MetalDevice, Tensor};

    #[test]
    fn grouped_experts_preserve_long_token_indices_and_match_cpu() {
        let device = MetalDevice::new().unwrap();
        let experts = 3;
        let hidden = 3;
        let intermediate = 2;
        let tokens = 300;
        let top_k = 2;

        let router_weights = vec![
            1., 0., 0., // Expert 0 ranks first for positive x[0].
            0.5, 0., 0., // Expert 1 ranks second.
            -1., 0., 0., // Expert 2 is not selected.
        ];
        let mut input_weights = vec![0.; experts * 2 * intermediate * hidden];
        let mut set_input = |expert: usize, row: usize, values: [f32; 3]| {
            let start = (expert * 2 * intermediate + row) * hidden;
            input_weights[start..start + hidden].copy_from_slice(&values);
        };
        set_input(0, 0, [1., 0., 0.]);
        set_input(0, 1, [0., 1., 0.]);
        set_input(0, 2, [1., 0., 0.]);
        set_input(0, 3, [0., 1., 0.]);
        set_input(1, 0, [1., 1., 0.]);
        set_input(1, 1, [0.5, -1., 0.]);
        set_input(1, 2, [0., 0., 1.]);
        set_input(1, 3, [0., 1., 0.]);

        let mut output_weights = vec![0.; experts * hidden * intermediate];
        let mut set_output = |expert: usize, row: usize, values: [f32; 2]| {
            let start = (expert * hidden + row) * intermediate;
            output_weights[start..start + intermediate].copy_from_slice(&values);
        };
        set_output(0, 0, [1., 0.]);
        set_output(0, 1, [0., 1.]);
        set_output(0, 2, [0.5, 0.5]);
        set_output(1, 0, [0., 1.]);
        set_output(1, 1, [1., 0.]);
        set_output(1, 2, [0.25, 0.5]);

        let router =
            Tensor::from_f32(&device, [experts, hidden], DType::F32, &router_weights).unwrap();
        let input_experts = Tensor::from_f32(
            &device,
            [experts, 2 * intermediate, hidden],
            DType::F32,
            &input_weights,
        )
        .unwrap();
        let output_experts = Tensor::from_f32(
            &device,
            [experts, hidden, intermediate],
            DType::F32,
            &output_weights,
        )
        .unwrap();
        let moe = SparseMoe::new(
            &device,
            router,
            ExpertMatrix::Dense(input_experts),
            ExpertMatrix::Dense(output_experts),
            MoeRoutingPolicy::softmax(experts, top_k),
            None,
        )
        .unwrap();

        let mut input_values = Vec::with_capacity(tokens * hidden);
        for token in 0..tokens {
            input_values.extend([token as f32 + 1., 1., 2.]);
        }
        let input = Tensor::from_f32(&device, [tokens, hidden], DType::F32, &input_values).unwrap();
        let actual = moe.forward(&device, &input).unwrap().to_f32();

        let mut expected = Vec::with_capacity(tokens * hidden);
        for token in 0..tokens {
            let x = [token as f32 + 1., 1., 2.];
            let mut logits: Vec<_> = (0..experts)
                .map(|expert| {
                    let row = &router_weights[expert * hidden..(expert + 1) * hidden];
                    (expert, row.iter().zip(x).map(|(a, b)| a * b).sum::<f32>())
                })
                .collect();
            logits.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
            logits.truncate(top_k);
            let max = logits[0].1;
            let mut gates: Vec<_> = logits
                .iter()
                .map(|(_, value)| (value - max).exp())
                .collect();
            let denominator: f32 = gates.iter().sum();
            for gate in &mut gates {
                *gate /= denominator;
            }
            let mut y = [0.; 3];
            for ((expert, _), gate) in logits.iter().zip(gates) {
                let base = expert * 2 * intermediate * hidden;
                let projected: Vec<f32> = (0..2 * intermediate)
                    .map(|row| {
                        input_weights[base + row * hidden..base + (row + 1) * hidden]
                            .iter()
                            .zip(x)
                            .map(|(a, b)| a * b)
                            .sum()
                    })
                    .collect();
                let activation: Vec<f32> = (0..intermediate)
                    .map(|j| {
                        let value = projected[j];
                        (value / (1. + (-value).exp())) * projected[intermediate + j]
                    })
                    .collect();
                for (row, total) in y.iter_mut().enumerate() {
                    let start = (expert * hidden + row) * intermediate;
                    let value = output_weights[start..start + intermediate]
                        .iter()
                        .zip(&activation)
                        .map(|(a, b)| a * b)
                        .sum::<f32>();
                    *total += gate * value;
                }
            }
            expected.extend(y);
        }

        for (index, (actual, expected)) in actual.iter().zip(&expected).enumerate() {
            let tolerance = expected.abs().max(1.) * 2e-5;
            assert!(
                (actual - expected).abs() <= tolerance,
                "element {index}: {actual} vs {expected}, tolerance {tolerance}"
            );
        }
        let stats = moe.take_stats();
        assert_eq!(stats.assignments, tokens * top_k);
        assert_eq!(stats.active_experts, 2);
        assert!(stats.peak_temporary_bytes < ROUTING_TEMPORARY_LIMIT);
    }

    #[test]
    fn sigmoid_top_k_routes_normalize_and_apply_scale() {
        let device = MetalDevice::new().unwrap();
        let routing = MoeRoutingPolicy {
            experts: 3,
            top_k: 2,
            scoring_function: MoeScoringFunction::Sigmoid,
            normalize_top_k_prob: true,
            normalization_epsilon: 1e-6,
            routed_scaling_factor: 1.5,
            use_expert_bias: false,
        };
        let router = Tensor::from_f32(&device, [3, 2], DType::F32, &[0.; 6]).unwrap();
        let input = Tensor::from_f32(&device, [3, 4, 2], DType::F32, &[0.; 24]).unwrap();
        let output = Tensor::from_f32(&device, [3, 2, 2], DType::F32, &[0.; 12]).unwrap();
        let moe = SparseMoe::new(
            &device,
            router,
            ExpertMatrix::Dense(input),
            ExpertMatrix::Dense(output),
            routing,
            None,
        )
        .unwrap();
        let routes = moe.select_routes(&[0., 2., -2.], 1).unwrap();

        assert_eq!(routes[0].to_bits(), 0);
        assert_eq!(routes[1].to_bits(), 0);
        assert_eq!(routes[3].to_bits(), 1);
        let sigmoid_zero = 0.5;
        let sigmoid_two = 1. / (1. + (-2f32).exp());
        let denominator = sigmoid_zero + sigmoid_two + 1e-6;
        let expected_zero = sigmoid_zero / denominator * 1.5;
        let expected_two = sigmoid_two / denominator * 1.5;
        assert!((routes[2] - expected_zero).abs() < 1e-6);
        assert!((routes[5] - expected_two).abs() < 1e-6);
        assert!((routes[2] + routes[5] - 1.5).abs() < 2e-6);
    }

    #[test]
    fn sigmoid_expert_bias_changes_selection_without_changing_route_scores() {
        let device = MetalDevice::new().unwrap();
        let routing = MoeRoutingPolicy {
            experts: 3,
            top_k: 2,
            scoring_function: MoeScoringFunction::Sigmoid,
            normalize_top_k_prob: true,
            normalization_epsilon: 1e-6,
            routed_scaling_factor: 1.,
            use_expert_bias: true,
        };
        let router = Tensor::from_f32(&device, [3, 1], DType::F32, &[0.; 3]).unwrap();
        let input = Tensor::from_f32(&device, [3, 2, 1], DType::F32, &[0.; 6]).unwrap();
        let output = Tensor::from_f32(&device, [3, 1, 1], DType::F32, &[0.; 3]).unwrap();
        let bias = Tensor::from_f32(&device, [3], DType::F32, &[-10., 0., 2.]).unwrap();
        let moe = SparseMoe::new(
            &device,
            router,
            ExpertMatrix::Dense(input),
            ExpertMatrix::Dense(output),
            routing,
            Some(bias),
        )
        .unwrap();
        let routes = moe.select_routes(&[2., 1., 0.], 1).unwrap();

        let sigmoid_one = 1. / (1. + (-1f32).exp());
        let sigmoid_zero = 0.5;
        let normalizer = sigmoid_one + sigmoid_zero + 1e-6;
        assert_eq!(routes[0].to_bits(), 1);
        assert!((routes[2] - sigmoid_one / normalizer).abs() < 1e-6);
        assert_eq!(routes[3].to_bits(), 2);
        assert!((routes[5] - sigmoid_zero / normalizer).abs() < 1e-6);
    }

    #[test]
    fn metal_router_preserves_cpu_top_k_order_and_weights() {
        let device = MetalDevice::new().unwrap();
        let experts = 5;
        let hidden = 1;
        let top_k = 3;
        let router = Tensor::from_f32(&device, [experts, hidden], DType::F32, &[0.; 5]).unwrap();
        let input_experts =
            Tensor::from_f32(&device, [experts, 2, hidden], DType::F32, &[0.; 10]).unwrap();
        let output_experts =
            Tensor::from_f32(&device, [experts, hidden, 1], DType::F32, &[0.; 5]).unwrap();
        let bias =
            Tensor::from_f32(&device, [experts], DType::F32, &[0.1, 0.05, 0.1, -0.3, 0.1]).unwrap();
        let logits = Tensor::from_f32(
            &device,
            [2, experts],
            DType::BF16,
            &[0., 0., 0., 0., 0., 2., 0.5, -1., 1.5, 0.],
        )
        .unwrap();
        let logits_values = logits.to_f32();

        for (scoring_function, use_expert_bias) in [
            (MoeScoringFunction::Sigmoid, true),
            (MoeScoringFunction::Softmax, true),
            (MoeScoringFunction::Softmax, false),
        ] {
            let policy = MoeRoutingPolicy {
                experts,
                top_k,
                scoring_function,
                normalize_top_k_prob: true,
                normalization_epsilon: 1e-6,
                routed_scaling_factor: 1.25,
                use_expert_bias,
            };
            let selection_bias = use_expert_bias.then(|| bias.clone());
            let moe = SparseMoe::new(
                &device,
                router.clone(),
                ExpertMatrix::Dense(input_experts.clone()),
                ExpertMatrix::Dense(output_experts.clone()),
                policy,
                selection_bias.clone(),
            )
            .unwrap();
            let expected = moe.select_routes(&logits_values, 2).unwrap();
            let execution = device.execution().unwrap();
            let actual_tensor = device
                .moe_route(&logits, selection_bias.as_ref(), policy)
                .unwrap();
            let hidden = Tensor::from_f32(&device, [2, 1], DType::BF16, &[1., 2.]).unwrap();
            let assigned = device
                .expert_assign(&hidden, &actual_tensor, top_k)
                .unwrap();
            execution.finish().unwrap();
            let actual = actual_tensor.to_f32();
            let assigned_values = assigned.to_f32();

            for (index, (actual, expected)) in actual.iter().zip(&expected).enumerate() {
                if index % 3 != 2 {
                    assert_eq!(
                        actual.to_bits(),
                        expected.to_bits(),
                        "metadata index {index}"
                    );
                } else {
                    assert!(
                        (actual - expected).abs() <= 2e-6,
                        "weight index {index}: {actual} vs {expected}"
                    );
                }
            }
            for (assignment, route) in expected.chunks_exact(3).enumerate() {
                let token = route[1].to_bits() as usize;
                assert_eq!(assigned_values[assignment], [1., 2.][token]);
            }
        }
    }
}
