//! Kernel selection and tuning switches used only by the general engine
//! (`ops`, behind the `legacy-engine` feature), with its dispatcher and
//! transient output arena. The hybrid engine dispatches through
//! `MetalDevice::dispatch_hybrid` and never reads these.
use super::*;

/// The general engine's kernel switches, with their measured defaults.
pub(super) struct Tuning {
    pub(super) gguf_mpp_min_rows: Cell<Option<usize>>,
    pub(super) gguf_mpp_q5_1_min_rows: Cell<Option<usize>>,
    pub(super) attention_softmax_prefix: Cell<bool>,
    pub(super) attention_softmax_prefix_reuse: Cell<bool>,
    pub(super) attention_context_decode_wide: Cell<bool>,
    pub(super) attention_scores_vector: Cell<bool>,
    pub(super) q8_0_mpp_tile_k64: Cell<bool>,
    pub(super) q8_0_gemv_k_split: Cell<bool>,
    pub(super) q5_1_mpp_tile_k64: Cell<bool>,
    pub(super) q4_k_mpp_tile_k64: Cell<bool>,
    pub(super) q4_k_mpp_tile_m128: Cell<bool>,
    pub(super) q5_k_mpp_tile_k64: Cell<bool>,
    pub(super) mlx_affine4_mpp_tile_k64: Cell<Option<bool>>,
    pub(super) mlx_affine4_gemv_quad: Cell<bool>,
    pub(super) split_k_gemv: Cell<bool>,
    pub(super) q5_0_gemv_n4: Cell<bool>,
    pub(super) q5_0_ksplit: Cell<usize>,
    pub(super) k_quant_ksplit: Cell<usize>,
    pub(super) fuse_swiglu: Cell<bool>,
    pub(super) q8_0_rows: Cell<usize>,
    pub(super) flash_decode: Cell<bool>,
    pub(super) q5_1_gemv_n4: Cell<bool>,
    pub(super) q4_0_gemv_8rows: Cell<bool>,
    pub(super) q4_k_gemv_8rows: Cell<bool>,
    pub(super) q4_k_factored: Cell<bool>,
    pub(super) q6_k_factored: Cell<bool>,
    pub(super) mpp_fast_dequant: Cell<bool>,
    pub(super) dense_mpp_tile_pairs: Cell<bool>,
    pub(super) rope_table: Cell<bool>,
    pub(super) arena_epoch_reuse: Cell<bool>,
    pub(super) moe_expert_grouped: Cell<bool>,
    pub(super) fuse_add_rmsnorm: Cell<bool>,
    pub(super) fuse_rope_cache: Cell<bool>,
    pub(super) moe_expert_tile_pairs: Cell<bool>,
    pub(super) moe_routing_temporary_limit: Cell<usize>,
    pub(super) q4_k_expert_project_8rows: Cell<bool>,
    pub(super) q4_k_expert_project_16rows: Cell<bool>,
    pub(super) q4_k_expert_project_16rows_pairs: Cell<bool>,
    pub(super) q6_k_expert_project_8rows: Cell<bool>,
    pub(super) q5_k_gemv_8rows: Cell<bool>,
    pub(super) q6_k_gemv_8rows: Cell<bool>,
    pub(super) moe_gpu_routing: Cell<bool>,
    pub(super) moe_gpu_routing_prefill: Cell<bool>,
    pub(super) moe_expert_tensorops: Cell<bool>,
    pub(super) moe_expert_tensorops_tile_k64: Cell<bool>,
    pub(super) moe_expert_tensorops_min_routes_per_expert: Cell<usize>,
    pub(super) moe_chunk_device_copy: Cell<bool>,
    pub(super) reference_math: Cell<bool>,
}
impl Default for Tuning {
    fn default() -> Self {
        Self {
            gguf_mpp_min_rows: Cell::new(None),
            gguf_mpp_q5_1_min_rows: Cell::new(None),
            attention_softmax_prefix: Cell::new(true),
            attention_softmax_prefix_reuse: Cell::new(true),
            attention_context_decode_wide: Cell::new(true),
            attention_scores_vector: Cell::new(true),
            q8_0_mpp_tile_k64: Cell::new(true),
            q8_0_gemv_k_split: Cell::new(true),
            q5_1_mpp_tile_k64: Cell::new(true),
            q4_k_mpp_tile_k64: Cell::new(true),
            q4_k_mpp_tile_m128: Cell::new(true),
            q5_k_mpp_tile_k64: Cell::new(true),
            mlx_affine4_mpp_tile_k64: Cell::new(None),
            mlx_affine4_gemv_quad: Cell::new(true),
            split_k_gemv: Cell::new(true),
            q5_0_gemv_n4: Cell::new(true),
            q5_0_ksplit: Cell::new(2),
            k_quant_ksplit: Cell::new(8),
            fuse_swiglu: Cell::new(true),
            q8_0_rows: Cell::new(4),
            flash_decode: Cell::new(false),
            q5_1_gemv_n4: Cell::new(true),
            q4_0_gemv_8rows: Cell::new(true),
            q4_k_gemv_8rows: Cell::new(true),
            q4_k_factored: Cell::new(true),
            q6_k_factored: Cell::new(true),
            mpp_fast_dequant: Cell::new(true),
            dense_mpp_tile_pairs: Cell::new(true),
            rope_table: Cell::new(true),
            arena_epoch_reuse: Cell::new(true),
            moe_expert_grouped: Cell::new(true),
            fuse_add_rmsnorm: Cell::new(false),
            fuse_rope_cache: Cell::new(true),
            moe_expert_tile_pairs: Cell::new(true),
            moe_routing_temporary_limit: Cell::new(128 * 1024 * 1024),
            q4_k_expert_project_8rows: Cell::new(true),
            q4_k_expert_project_16rows: Cell::new(true),
            q4_k_expert_project_16rows_pairs: Cell::new(true),
            q6_k_expert_project_8rows: Cell::new(true),
            q5_k_gemv_8rows: Cell::new(true),
            q6_k_gemv_8rows: Cell::new(true),
            moe_gpu_routing: Cell::new(true),
            moe_gpu_routing_prefill: Cell::new(true),
            moe_expert_tensorops: Cell::new(true),
            moe_expert_tensorops_tile_k64: Cell::new(true),
            moe_expert_tensorops_min_routes_per_expert: Cell::new(8),
            moe_chunk_device_copy: Cell::new(true),
            reference_math: Cell::new(false),
        }
    }
}

impl MetalDevice {
    pub fn set_native_matmul(&self, enabled: bool) -> Result<()> {
        if self.batching.get() || (enabled && !self.raw.supportsFamily(MTLGPUFamily::Apple7)) {
            return Err(Error::Parameter(
                "native SIMD matrix support required".into(),
            ));
        }
        self.native_matmul.set(enabled);
        Ok(())
    }
    /// Override the automatic, measured per-format batch boundary for GGUF Metal 4 GEMM.
    ///
    /// A newly created device uses its automatic format-specific boundary until this is set.
    pub fn set_gguf_mpp_min_rows(&self, rows: usize) -> Result<()> {
        if self.batching.get() || !(2..=16).contains(&rows) || (rows < 16 && !self.mpp_projection())
        {
            return Err(Error::Parameter(
                "GGUF MPP minimum rows must be 2..=16 on an idle compatible device".into(),
            ));
        }
        self.tuning.gguf_mpp_min_rows.set(Some(rows));
        Ok(())
    }
    /// Restore automatic, measured per-format GGUF MPP row thresholds.
    pub fn clear_gguf_mpp_min_rows(&self) -> Result<()> {
        if self.batching.get() {
            return Err(Error::Parameter(
                "cannot clear GGUF MPP row override during execution".into(),
            ));
        }
        self.tuning.gguf_mpp_min_rows.set(None);
        Ok(())
    }
    /// Returns the explicit GGUF MPP row override, or `None` for automatic dispatch.
    pub fn gguf_mpp_min_rows(&self) -> Option<usize> {
        self.tuning.gguf_mpp_min_rows.get()
    }
    /// Override only the Q5_1 GGUF MPP row boundary for controlled measurements.
    pub fn set_gguf_mpp_q5_1_min_rows(&self, rows: usize) -> Result<()> {
        if self.batching.get() || !(2..=16).contains(&rows) || !self.mpp_projection() {
            return Err(Error::Parameter(
                "Q5_1 GGUF MPP minimum rows must be 2..=16 on an idle compatible device".into(),
            ));
        }
        self.tuning.gguf_mpp_q5_1_min_rows.set(Some(rows));
        Ok(())
    }
    /// Clear the Q5_1-only boundary override.
    pub fn clear_gguf_mpp_q5_1_min_rows(&self) -> Result<()> {
        if self.batching.get() {
            return Err(Error::Parameter(
                "cannot clear Q5_1 GGUF MPP row override during execution".into(),
            ));
        }
        self.tuning.gguf_mpp_q5_1_min_rows.set(None);
        Ok(())
    }
    /// Returns the explicit Q5_1 row override, or `None` for automatic dispatch.
    pub fn gguf_mpp_q5_1_min_rows(&self) -> Option<usize> {
        self.tuning.gguf_mpp_q5_1_min_rows.get()
    }
    /// Enable or disable prefix-bounded causal softmax for measured full-prefill shapes.
    /// The default is enabled; dispatch uses it only when M equals context width and M>=256.
    pub fn set_attention_softmax_prefix(&self, enabled: bool) -> Result<()> {
        if self.batching.get() {
            return Err(Error::Parameter(
                "attention softmax variant can only change on an idle device".into(),
            ));
        }
        self.tuning.attention_softmax_prefix.set(enabled);
        Ok(())
    }
    pub(crate) fn attention_softmax_prefix(&self, rows: usize, width: usize) -> bool {
        self.tuning.attention_softmax_prefix.get() && rows >= 256 && rows == width
    }
    /// Enable register reuse for full-prefill prefix softmax rows up to width 1,024.
    pub fn set_attention_softmax_prefix_reuse(&self, enabled: bool) -> Result<()> {
        if self.batching.get() {
            return Err(Error::Parameter(
                "attention softmax variant can only change on an idle device".into(),
            ));
        }
        self.tuning.attention_softmax_prefix_reuse.set(enabled);
        Ok(())
    }
    pub(crate) fn attention_softmax_prefix_reuse(&self, rows: usize, width: usize) -> bool {
        self.tuning.attention_softmax_prefix.get()
            && self.tuning.attention_softmax_prefix_reuse.get()
            && (256..=1024).contains(&rows)
            && rows == width
    }
    /// Select the 1,024-thread M=1 attention context kernel for all widths.
    pub fn set_attention_context_decode_wide(&self, enabled: bool) -> Result<()> {
        if self.batching.get() {
            return Err(Error::Parameter(
                "attention decode variant can only change on an idle device".into(),
            ));
        }
        self.tuning.attention_context_decode_wide.set(enabled);
        Ok(())
    }
    pub fn attention_context_decode_wide(&self) -> bool {
        self.tuning.attention_context_decode_wide.get()
    }
    /// Use four-wide BF16 loads in the scalar attention score kernel.
    pub fn set_attention_scores_vector(&self, enabled: bool) -> Result<()> {
        if self.batching.get() {
            return Err(Error::Parameter(
                "attention score variant can only change on an idle device".into(),
            ));
        }
        self.tuning.attention_scores_vector.set(enabled);
        Ok(())
    }
    pub(crate) fn attention_scores_vector(&self) -> bool {
        self.tuning.attention_scores_vector.get()
    }
    /// Select 64-element K tiles for Q8_0 MPP prompt GEMM.
    pub fn set_q8_0_mpp_tile_k64(&self, enabled: bool) -> Result<()> {
        if self.batching.get() || (enabled && !self.mpp_projection()) {
            return Err(Error::Parameter(
                "Q8_0 MPP K=64 requires an idle compatible device".into(),
            ));
        }
        self.tuning.q8_0_mpp_tile_k64.set(enabled);
        Ok(())
    }
    pub(crate) fn q8_0_mpp_tile_k64(&self, batch_rows: usize) -> bool {
        self.tuning.q8_0_mpp_tile_k64.get() && batch_rows >= 512
    }
    /// Select 64-element K tiles for Q5_1 MPP prompt GEMM.
    pub fn set_q5_1_mpp_tile_k64(&self, enabled: bool) -> Result<()> {
        if self.batching.get() || (enabled && !self.mpp_projection()) {
            return Err(Error::Parameter(
                "Q5_1 MPP K=64 requires an idle compatible device".into(),
            ));
        }
        self.tuning.q5_1_mpp_tile_k64.set(enabled);
        Ok(())
    }
    pub(crate) fn q5_1_mpp_tile_k64(&self, batch_rows: usize) -> bool {
        self.tuning.q5_1_mpp_tile_k64.get() && batch_rows >= 512
    }
    /// Select 64-element K tiles for Q4_K MPP prompt GEMM.
    pub fn set_q4_k_mpp_tile_k64(&self, enabled: bool) -> Result<()> {
        if self.batching.get() || (enabled && !self.mpp_projection()) {
            return Err(Error::Parameter(
                "Q4_K MPP K=64 requires an idle compatible device".into(),
            ));
        }
        self.tuning.q4_k_mpp_tile_k64.set(enabled);
        Ok(())
    }
    pub(crate) fn q4_k_mpp_tile_k64(&self, batch_rows: usize) -> bool {
        self.tuning.q4_k_mpp_tile_k64.get() && batch_rows >= 1024
    }
    /// Select the measured 128-row output tile for Q4_K K=64 prompt GEMM.
    pub fn set_q4_k_mpp_tile_m128(&self, enabled: bool) -> Result<()> {
        if self.batching.get() || (enabled && !self.mpp_projection()) {
            return Err(Error::Parameter(
                "Q4_K MPP M=128 requires an idle compatible device".into(),
            ));
        }
        self.tuning.q4_k_mpp_tile_m128.set(enabled);
        Ok(())
    }
    pub(crate) fn q4_k_mpp_tile_m128(&self, batch_rows: usize) -> bool {
        self.tuning.q4_k_mpp_tile_m128.get() && batch_rows >= 1024
    }
    /// Select 64-element K tiles for Q5_K MPP prompt GEMM.
    pub fn set_q5_k_mpp_tile_k64(&self, enabled: bool) -> Result<()> {
        if self.batching.get() || (enabled && !self.mpp_projection()) {
            return Err(Error::Parameter(
                "Q5_K MPP K=64 requires an idle compatible device".into(),
            ));
        }
        self.tuning.q5_k_mpp_tile_k64.set(enabled);
        Ok(())
    }
    pub(crate) fn q5_k_mpp_tile_k64(&self, batch_rows: usize) -> bool {
        self.tuning.q5_k_mpp_tile_k64.get() && batch_rows >= 1024
    }
    /// Override the shape-aware MLX affine-Q4 MPP K-tile choice.
    /// `Some(false)` selects K=128, `Some(true)` selects K=64, and `None`
    /// restores the production heuristic.
    pub fn set_mlx_affine4_mpp_tile_k64(&self, enabled: Option<bool>) -> Result<()> {
        if self.batching.get() || (enabled == Some(true) && !self.mpp_projection()) {
            return Err(Error::Parameter(
                "MLX affine-Q4 MPP tile override requires an idle compatible device".into(),
            ));
        }
        self.tuning.mlx_affine4_mpp_tile_k64.set(enabled);
        Ok(())
    }
    pub(crate) fn mlx_affine4_mpp_tile_k64(&self, batch_rows: usize) -> bool {
        self.tuning
            .mlx_affine4_mpp_tile_k64
            .get()
            .unwrap_or((512..=1024).contains(&batch_rows))
    }
    /// Select the four-lane, eight-row affine-Q4 M=1 GEMV candidate for wide outputs.
    pub fn set_mlx_affine4_gemv_quad(&self, enabled: bool) -> Result<()> {
        if self.batching.get() || (enabled && !self.mpp_projection()) {
            return Err(Error::Parameter(
                "affine-Q4 quad GEMV requires an idle compatible device".into(),
            ));
        }
        self.tuning.mlx_affine4_gemv_quad.set(enabled);
        Ok(())
    }
    pub(crate) fn use_mlx_affine4_gemv_quad(
        &self,
        output_rows: usize,
        input_columns: usize,
    ) -> bool {
        self.tuning.mlx_affine4_gemv_quad.get()
            && self.mpp_projection()
            && output_rows >= 2048
            && input_columns <= 2048
    }
    pub(crate) fn q8_0_gemv_8rows(&self, output_rows: usize) -> bool {
        output_rows >= 128 && output_rows.is_multiple_of(8)
    }
    /// Select the four-SIMD-group K-reduction Q8_0 M=1 GEMV path.
    pub fn set_q8_0_gemv_k_split(&self, enabled: bool) -> Result<()> {
        if self.batching.get() {
            return Err(Error::Parameter(
                "cannot change Q8_0 K-split GEMV during execution".into(),
            ));
        }
        self.tuning.q8_0_gemv_k_split.set(enabled);
        Ok(())
    }
    pub(crate) fn q8_0_gemv_k_split(&self, output_rows: usize, input_columns: usize) -> bool {
        self.tuning.q8_0_gemv_k_split.get() && output_rows >= 128 && input_columns >= 768
    }
    /// Select the four-output-row-per-SIMD Q5_0 M=1 candidate for wide outputs.
    pub fn set_q5_0_gemv_n4(&self, enabled: bool) -> Result<()> {
        if self.batching.get() {
            return Err(Error::Parameter(
                "cannot change Q5_0 GEMV during execution".into(),
            ));
        }
        self.tuning.q5_0_gemv_n4.set(enabled);
        Ok(())
    }
    pub(crate) fn use_q5_0_gemv_n4(&self, output_rows: usize) -> bool {
        self.tuning.q5_0_gemv_n4.get() && output_rows >= 128
    }
    /// K-split Q5_0 M=1 GEMV: SIMD groups sharing each four-row tile
    /// (0 restores the row-owning kernels; 1, 2, 4, or 8 otherwise).
    pub fn set_q5_0_gemv_ksplit(&self, simds: usize) -> Result<()> {
        if self.batching.get() {
            return Err(Error::Parameter(
                "cannot change Q5_0 GEMV during execution".into(),
            ));
        }
        if !matches!(simds, 0 | 1 | 2 | 4 | 8) {
            return Err(Error::Parameter(
                "Q5_0 K-split SIMD groups must be 0, 1, 2, 4, or 8".into(),
            ));
        }
        self.tuning.q5_0_ksplit.set(simds);
        Ok(())
    }
    pub(crate) fn q5_0_ksplit(&self) -> usize {
        self.tuning.q5_0_ksplit.get()
    }
    /// K-split factored Q4_K/Q6_K M=1 GEMV: the most SIMD groups that may share
    /// each four-row tile (0 restores the row-owning kernels; 1, 2, 4, or 8).
    /// Each projection uses enough groups to cover its super-blocks in about
    /// one pass, and at most two once there are 4,096 or more output rows.
    pub fn set_k_quant_gemv_ksplit(&self, simds: usize) -> Result<()> {
        if self.batching.get() {
            return Err(Error::Parameter(
                "cannot change K-quant GEMV during execution".into(),
            ));
        }
        if !matches!(simds, 0 | 1 | 2 | 4 | 8) {
            return Err(Error::Parameter(
                "K-quant K-split SIMD groups must be 0, 1, 2, 4, or 8".into(),
            ));
        }
        self.tuning.k_quant_ksplit.set(simds);
        Ok(())
    }
    pub(crate) fn k_quant_gemv_ksplit(&self) -> usize {
        self.tuning.k_quant_ksplit.get()
    }
    /// Fuse decode gate/up Q5_0 projections with SiLU-multiply (bit-identical).
    pub fn set_fuse_swiglu(&self, enabled: bool) -> Result<()> {
        if self.batching.get() {
            return Err(Error::Parameter(
                "cannot change fusion policy during execution".into(),
            ));
        }
        self.tuning.fuse_swiglu.set(enabled);
        Ok(())
    }
    pub(crate) fn fuse_swiglu(&self) -> bool {
        self.tuning.fuse_swiglu.get()
    }
    /// Output rows per threadgroup of the K-split Q8_0 M=1 GEMV: 2, 4, or 8.
    /// Every count is bit-identical; more rows reuse each activation load.
    pub fn set_q8_0_gemv_rows(&self, rows: usize) -> Result<()> {
        if self.batching.get() {
            return Err(Error::Parameter(
                "cannot change Q8_0 GEMV during execution".into(),
            ));
        }
        if !matches!(rows, 2 | 4 | 8) {
            return Err(Error::Parameter("Q8_0 GEMV rows must be 2, 4, or 8".into()));
        }
        self.tuning.q8_0_rows.set(rows);
        Ok(())
    }
    pub(crate) fn q8_0_gemv_rows(&self) -> usize {
        self.tuning.q8_0_rows.get()
    }
    /// Single-row decode attention as split-context flash kernels. Off by
    /// default: it skips the unfused path's BF16 probability rounding, which
    /// moves the Qwen2.5 BF16 reference check from 0.516 to 0.562 (limit 0.55).
    pub fn set_flash_decode(&self, enabled: bool) -> Result<()> {
        if self.batching.get() {
            return Err(Error::Parameter(
                "cannot change attention policy during execution".into(),
            ));
        }
        self.tuning.flash_decode.set(enabled);
        Ok(())
    }
    pub(crate) fn flash_decode(&self) -> bool {
        self.tuning.flash_decode.get()
    }
    /// K-split GEMVs carry their SIMD groups per threadgroup in `params[8]`.
    fn ksplit_simds(name: &str, params: &[u32; 9]) -> Option<usize> {
        matches!(
            name,
            "q5_0_gemv_ksplit" | "q5_0_gemv_swiglu" | "q4_k_gemv_ksplit" | "q6_k_gemv_ksplit"
        )
        .then(|| (params[8] as usize).clamp(1, 8))
    }
    /// Select the four-output-row-per-SIMD Q5_1 M=1 GEMV path.
    pub fn set_q5_1_gemv_n4(&self, enabled: bool) -> Result<()> {
        if self.batching.get() {
            return Err(Error::Parameter(
                "cannot change Q5_1 GEMV during execution".into(),
            ));
        }
        self.tuning.q5_1_gemv_n4.set(enabled);
        Ok(())
    }
    pub(crate) fn use_q5_1_gemv_n4(&self, output_rows: usize) -> bool {
        self.tuning.q5_1_gemv_n4.get() && output_rows >= 128
    }
    /// Select the two-output-row-per-SIMD Q4_0 M=1 kernel for wide projections.
    pub fn set_q4_0_gemv_8rows(&self, enabled: bool) -> Result<()> {
        if self.batching.get() {
            return Err(Error::Parameter(
                "cannot change Q4_0 GEMV during execution".into(),
            ));
        }
        self.tuning.q4_0_gemv_8rows.set(enabled);
        Ok(())
    }
    pub(crate) fn use_q4_0_gemv_8rows(&self, output_rows: usize) -> bool {
        self.tuning.q4_0_gemv_8rows.get() && output_rows >= 128
    }
    /// Select the two-output-row-per-SIMD Q4_K M=1 kernel for wide projections.
    pub fn set_q4_k_gemv_8rows(&self, enabled: bool) -> Result<()> {
        if self.batching.get() {
            return Err(Error::Parameter(
                "cannot change Q4_K GEMV during execution".into(),
            ));
        }
        self.tuning.q4_k_gemv_8rows.set(enabled);
        Ok(())
    }
    pub(crate) fn use_q4_k_gemv_8rows(&self, output_rows: usize) -> bool {
        self.tuning.q4_k_gemv_8rows.get() && output_rows >= 128
    }
    /// Select factored-scale Q4_K M=1 kernels (dense GEMV and sparse experts).
    pub fn set_q4_k_factored(&self, enabled: bool) -> Result<()> {
        if self.batching.get() {
            return Err(Error::Parameter(
                "cannot change Q4_K GEMV during execution".into(),
            ));
        }
        self.tuning.q4_k_factored.set(enabled);
        Ok(())
    }
    pub(crate) fn use_q4_k_factored(&self, features: usize) -> bool {
        self.tuning.q4_k_factored.get() && features >= 256 && features.is_multiple_of(256)
    }
    /// Decode Q4_K/Q6_K/Q8_0/Q5_0 TensorOps weight tiles one block or chunk per work item.
    pub fn set_mpp_fast_dequant(&self, enabled: bool) -> Result<()> {
        if self.batching.get() {
            return Err(Error::Parameter(
                "cannot change TensorOps dequantization during execution".into(),
            ));
        }
        self.tuning.mpp_fast_dequant.set(enabled);
        Ok(())
    }
    pub(crate) fn mpp_fast_dequant(&self) -> bool {
        self.tuning.mpp_fast_dequant.get()
    }
    /// Bound per-chunk routing temporaries for quantized MoE experts; larger
    /// chunks give each expert more routed rows per TensorOps pass. Dense
    /// experts keep the 16 MiB Phase 6 bound.
    pub fn set_moe_routing_temporary_limit(&self, bytes: usize) -> Result<()> {
        if self.batching.get() || bytes == 0 {
            return Err(Error::Parameter(
                "MoE routing limit must be positive and set on an idle device".into(),
            ));
        }
        self.tuning.moe_routing_temporary_limit.set(bytes);
        Ok(())
    }
    pub fn moe_routing_temporary_limit(&self) -> usize {
        self.tuning.moe_routing_temporary_limit.get()
    }
    /// Write rotated K (and V, with folded biases) straight into KV-cache slots.
    /// Active only inside a batched execution with the RoPE table enabled.
    pub fn set_fuse_rope_cache(&self, enabled: bool) -> Result<()> {
        if self.batching.get() {
            return Err(Error::Parameter(
                "cannot change fusion policy during execution".into(),
            ));
        }
        self.tuning.fuse_rope_cache.set(enabled);
        Ok(())
    }
    pub(crate) fn fuse_rope_cache(&self) -> bool {
        self.tuning.fuse_rope_cache.get() && self.batching.get() && self.tuning.rope_table.get()
    }
    /// Fuse each residual add with the RMSNorm that consumes it. Off by default:
    /// bit-identical but performance-neutral on Qwen2.5 and Qwen3 (Experiment 66).
    pub fn set_fuse_add_rmsnorm(&self, enabled: bool) -> Result<()> {
        if self.batching.get() {
            return Err(Error::Parameter(
                "cannot change fusion policy during execution".into(),
            ));
        }
        self.tuning.fuse_add_rmsnorm.set(enabled);
        Ok(())
    }
    pub(crate) fn fuse_add_rmsnorm(&self) -> bool {
        self.tuning.fuse_add_rmsnorm.get()
    }
    /// Group small-batch quantized expert projections by expert.
    pub fn set_moe_expert_grouped(&self, enabled: bool) -> Result<()> {
        if self.batching.get() {
            return Err(Error::Parameter(
                "cannot change expert grouping during execution".into(),
            ));
        }
        self.tuning.moe_expert_grouped.set(enabled);
        Ok(())
    }
    pub(crate) fn moe_expert_grouped(&self) -> bool {
        self.tuning.moe_expert_grouped.get()
    }
    /// Recycle transient storage retired earlier in the current encoding epoch.
    pub fn set_arena_epoch_reuse(&self, enabled: bool) -> Result<()> {
        if self.batching.get() {
            return Err(Error::Parameter(
                "cannot change arena policy during execution".into(),
            ));
        }
        self.tuning.arena_epoch_reuse.set(enabled);
        Ok(())
    }
    /// Share one RoPE cos/sin table across the q/k rotations of an execution.
    pub fn set_rope_table(&self, enabled: bool) -> Result<()> {
        if self.batching.get() {
            return Err(Error::Parameter(
                "cannot change RoPE table policy during execution".into(),
            ));
        }
        self.tuning.rope_table.set(enabled);
        Ok(())
    }
    pub(crate) fn rope_table_enabled(&self) -> bool {
        self.tuning.rope_table.get()
    }
    pub(crate) fn rope_table_cache(&self) -> &RefCell<Option<RopeTableEntry>> {
        &self.rope_table_cache
    }
    pub(crate) fn batching_active(&self) -> bool {
        self.batching.get()
    }
    /// Let each dense GGUF TensorOps threadgroup cover two consecutive M tiles,
    /// sharing each dequantized weight tile.
    pub fn set_dense_mpp_tile_pairs(&self, enabled: bool) -> Result<()> {
        if self.batching.get() {
            return Err(Error::Parameter(
                "cannot change TensorOps tiling during execution".into(),
            ));
        }
        self.tuning.dense_mpp_tile_pairs.set(enabled);
        Ok(())
    }
    pub(crate) fn dense_mpp_tile_pairs(&self) -> bool {
        self.tuning.dense_mpp_tile_pairs.get()
    }
    /// Let each expert TensorOps threadgroup cover two 32-row tiles of one
    /// expert, sharing each dequantized weight tile.
    pub fn set_moe_expert_tile_pairs(&self, enabled: bool) -> Result<()> {
        if self.batching.get() {
            return Err(Error::Parameter(
                "cannot change expert TensorOps tiling during execution".into(),
            ));
        }
        self.tuning.moe_expert_tile_pairs.set(enabled);
        Ok(())
    }
    pub(crate) fn moe_expert_tile_pairs(&self) -> bool {
        self.tuning.moe_expert_tile_pairs.get()
    }
    /// Select factored-scale Q6_K M=1 kernels (dense GEMV and sparse experts).
    pub fn set_q6_k_factored(&self, enabled: bool) -> Result<()> {
        if self.batching.get() {
            return Err(Error::Parameter(
                "cannot change Q6_K GEMV during execution".into(),
            ));
        }
        self.tuning.q6_k_factored.set(enabled);
        Ok(())
    }
    pub(crate) fn use_q6_k_factored(&self, features: usize) -> bool {
        self.tuning.q6_k_factored.get() && features >= 256 && features.is_multiple_of(256)
    }
    /// Select the two-output-row-per-SIMD Q4_K kernel for sparse expert batches.
    pub fn set_q4_k_expert_project_8rows(&self, enabled: bool) -> Result<()> {
        if self.batching.get() {
            return Err(Error::Parameter(
                "cannot change Q4_K expert projection during execution".into(),
            ));
        }
        self.tuning.q4_k_expert_project_8rows.set(enabled);
        Ok(())
    }
    pub(crate) fn use_q4_k_expert_project_8rows(&self, rows: usize, features: usize) -> bool {
        self.tuning.q4_k_expert_project_8rows.get()
            && rows >= 128
            && features >= 256
            && features.is_multiple_of(256)
    }
    /// Select the four-output-row-per-SIMD Q4_K expert kernel.
    pub fn set_q4_k_expert_project_16rows(&self, enabled: bool) -> Result<()> {
        if self.batching.get() {
            return Err(Error::Parameter(
                "cannot change Q4_K expert projection during execution".into(),
            ));
        }
        self.tuning.q4_k_expert_project_16rows.set(enabled);
        Ok(())
    }
    pub(crate) fn use_q4_k_expert_project_16rows(&self, rows: usize, features: usize) -> bool {
        self.tuning.q4_k_expert_project_16rows.get()
            && rows >= 128
            && features >= 256
            && features.is_multiple_of(256)
    }
    /// Select the measured block-and-pair traversal for direct Q4_K expert GEMV.
    pub fn set_q4_k_expert_project_16rows_pairs(&self, enabled: bool) -> Result<()> {
        if self.batching.get() {
            return Err(Error::Parameter(
                "cannot change Q4_K expert pair traversal during execution".into(),
            ));
        }
        self.tuning.q4_k_expert_project_16rows_pairs.set(enabled);
        Ok(())
    }
    pub(crate) fn use_q4_k_expert_project_16rows_pairs(
        &self,
        rows: usize,
        features: usize,
    ) -> bool {
        self.tuning.q4_k_expert_project_16rows_pairs.get()
            && self.use_q4_k_expert_project_16rows(rows, features)
    }
    /// Select the two-output-row-per-SIMD Q6_K kernel for sparse expert batches.
    pub fn set_q6_k_expert_project_8rows(&self, enabled: bool) -> Result<()> {
        if self.batching.get() {
            return Err(Error::Parameter(
                "cannot change Q6_K expert projection during execution".into(),
            ));
        }
        self.tuning.q6_k_expert_project_8rows.set(enabled);
        Ok(())
    }
    pub(crate) fn use_q6_k_expert_project_8rows(&self, rows: usize, features: usize) -> bool {
        self.tuning.q6_k_expert_project_8rows.get()
            && rows >= 128
            && features >= 256
            && features.is_multiple_of(256)
    }
    /// Select the two-output-row-per-SIMD Q5_K M=1 GEMV path.
    pub fn set_q5_k_gemv_8rows(&self, enabled: bool) -> Result<()> {
        if self.batching.get() {
            return Err(Error::Parameter(
                "cannot change Q5_K GEMV during execution".into(),
            ));
        }
        self.tuning.q5_k_gemv_8rows.set(enabled);
        Ok(())
    }
    pub(crate) fn use_q5_k_gemv_8rows(&self, output_rows: usize) -> bool {
        self.tuning.q5_k_gemv_8rows.get() && output_rows >= 128
    }
    /// Select the two-output-row-per-SIMD Q6_K M=1 kernel for wide projections.
    pub fn set_q6_k_gemv_8rows(&self, enabled: bool) -> Result<()> {
        if self.batching.get() {
            return Err(Error::Parameter(
                "cannot change Q6_K GEMV during execution".into(),
            ));
        }
        self.tuning.q6_k_gemv_8rows.set(enabled);
        Ok(())
    }
    pub(crate) fn use_q6_k_gemv_8rows(&self, output_rows: usize) -> bool {
        self.tuning.q6_k_gemv_8rows.get() && output_rows >= 128
    }
    /// Route sparse experts on Metal to avoid a router-logit readback boundary.
    pub fn set_moe_gpu_routing(&self, enabled: bool) -> Result<()> {
        if self.batching.get() {
            return Err(Error::Parameter(
                "cannot change MoE routing during execution".into(),
            ));
        }
        self.tuning.moe_gpu_routing.set(enabled);
        Ok(())
    }
    /// Enable or disable GPU top-k routing on measured prompt shapes.
    pub fn set_moe_gpu_routing_prefill(&self, enabled: bool) -> Result<()> {
        if self.batching.get() {
            return Err(Error::Parameter(
                "cannot change prompt GPU routing during execution".into(),
            ));
        }
        self.tuning.moe_gpu_routing_prefill.set(enabled);
        Ok(())
    }
    pub(crate) fn use_moe_gpu_routing(&self, top_k: usize, token_count: usize) -> bool {
        let measured_prefill_shape = token_count > 1 && (token_count <= 32 || token_count >= 512);
        self.tuning.moe_gpu_routing.get()
            && top_k <= 16
            && (token_count == 1
                || (self.tuning.moe_gpu_routing_prefill.get() && measured_prefill_shape))
    }
    /// Enable grouped TensorOps for sufficiently large quantized expert batches.
    pub fn set_moe_expert_tensorops(&self, enabled: bool) -> Result<()> {
        if self.batching.get() {
            return Err(Error::Parameter(
                "cannot change expert TensorOps during execution".into(),
            ));
        }
        self.tuning.moe_expert_tensorops.set(enabled);
        Ok(())
    }
    pub fn moe_expert_tensorops_enabled(&self) -> bool {
        self.tuning.moe_expert_tensorops.get()
    }
    /// Select a 64-wide K tile for grouped Q4_K and Q6_K expert TensorOps.
    pub fn set_moe_expert_tensorops_tile_k64(&self, enabled: bool) -> Result<()> {
        if self.batching.get() || (enabled && !self.mpp_projection()) {
            return Err(Error::Parameter(
                "expert TensorOps K=64 requires an idle compatible device".into(),
            ));
        }
        self.tuning.moe_expert_tensorops_tile_k64.set(enabled);
        Ok(())
    }
    pub fn moe_expert_tensorops_tile_k64(&self) -> bool {
        self.tuning.moe_expert_tensorops_tile_k64.get()
    }
    /// Set the minimum average route count per expert for grouped expert TensorOps.
    pub fn set_moe_expert_tensorops_min_routes_per_expert(&self, routes: usize) -> Result<()> {
        if self.batching.get() {
            return Err(Error::Parameter(
                "cannot change expert TensorOps threshold during execution".into(),
            ));
        }
        if routes == 0 {
            return Err(Error::Parameter(
                "expert TensorOps threshold must be positive".into(),
            ));
        }
        self.tuning
            .moe_expert_tensorops_min_routes_per_expert
            .set(routes);
        Ok(())
    }
    pub fn moe_expert_tensorops_min_routes_per_expert(&self) -> usize {
        self.tuning.moe_expert_tensorops_min_routes_per_expert.get()
    }
    /// Assemble chunked MoE outputs on Metal instead of through host readback.
    pub fn set_moe_chunk_device_copy(&self, enabled: bool) -> Result<()> {
        if self.batching.get() {
            return Err(Error::Parameter(
                "cannot change MoE chunk assembly during execution".into(),
            ));
        }
        self.tuning.moe_chunk_device_copy.set(enabled);
        Ok(())
    }
    pub fn moe_chunk_device_copy(&self) -> bool {
        self.tuning.moe_chunk_device_copy.get()
    }
    /// Select the multi-SIMD split-K BF16 GEMV for large aligned rows.
    pub fn set_split_k_gemv(&self, enabled: bool) -> Result<()> {
        if self.batching.get() {
            return Err(Error::Parameter(
                "cannot change GEMV during execution".into(),
            ));
        }
        self.tuning.split_k_gemv.set(enabled);
        Ok(())
    }
    pub(crate) fn split_k_gemv(&self) -> bool {
        self.tuning.split_k_gemv.get()
    }
    pub(crate) fn native_matmul(&self) -> bool {
        self.native_matmul.get()
    }
    /// Validation-only ordered reductions reproduce the immutable F32 diagnostic.
    /// Ordinary inference uses the parallel kernels.
    pub fn set_reference_math(&self, enabled: bool) -> Result<()> {
        if self.batching.get() {
            return Err(Error::Parameter("cannot change math in execution".into()));
        }
        self.tuning.reference_math.set(enabled);
        Ok(())
    }
    pub(crate) fn reference_math(&self) -> bool {
        self.tuning.reference_math.get()
    }
    pub(crate) fn profile_projection<T>(
        &self,
        stage: &'static str,
        f: impl FnOnce() -> Result<T>,
    ) -> Result<T> {
        if !self.profiling.get() {
            return f();
        }
        let _scope = ProfileStageScope {
            stage: &self.profile_stage,
            previous: self.profile_stage.replace(Some(stage)),
        };
        f()
    }
    /// Only fully overwritten trusted kernel outputs use the bounded transient arena.
    /// Fresh storage remains zero-initialized. Reused storage is already initialized,
    /// so skipping redundant zeroing cannot expose uninitialized Rust bytes.
    pub(crate) fn allocate_output(&self, bytes: usize) -> Result<MetalBuffer> {
        if !self.batching.get() {
            return self.allocate(bytes);
        }
        let length = bytes
            .max(4)
            .checked_next_power_of_two()
            .ok_or(Error::Allocation(bytes))?;
        // Bound resources retained by one encoding epoch independently of sequence
        // length. Completing earlier work makes retired tensors reusable; live
        // arguments/outputs may individually exceed this soft working-set budget.
        if self.arena.borrow().live.saturating_add(length) > 256 * 1024 * 1024 {
            self.flush()?;
        }
        let mut arena = self.arena.borrow_mut();
        // Storage retired earlier in the current, still-encoding epoch is also
        // reusable: its tensor is gone, so no later dispatch reads the old
        // contents, and every earlier reader is ordered before the new writer
        // (serial encoder order, the concurrent encoder's range barriers, or
        // Metal's tracked hazards between encoders).
        let current_epoch = self
            .tuning
            .arena_epoch_reuse
            .get()
            .then(|| self.pending.borrow().as_ref().map(|s| s.ready.clone()))
            .flatten();
        let bin = arena.free.entry(length).or_default();
        let available = bin.iter().rposition(|(_, ready, _)| {
            ready.as_ref().is_none_or(|r| {
                r.get() == Completion::Completed
                    || current_epoch
                        .as_ref()
                        .is_some_and(|epoch| Rc::ptr_eq(epoch, r))
            })
        });
        let mut still_counted = false;
        let mut buffer = if let Some(i) = available {
            let (raw, _, counted) = bin.swap_remove(i);
            still_counted = counted;
            let mut counters = self.counters.get();
            counters.reused_bytes += length;
            self.counters.set(counters);
            MetalBuffer {
                raw,
                len: bytes,
                owner: self.owner.clone(),
                ready: RefCell::new(None),
                writes: RefCell::new(Vec::new()),
                arena: None,
                resident: None,
            }
        } else {
            arena.make_room(length);
            let mut buffer = self.allocate(length)?;
            buffer.len = bytes;
            arena.capacity += length;
            buffer
        };
        // A pending buffer stays counted as live until its epoch completes.
        if !still_counted {
            arena.acquire(length);
        }
        buffer.arena = Some(self.arena.clone());
        Ok(buffer)
    }
    /// Only trusted, validated built-in operations can dispatch. Custom MSL is compile-only.
    pub(crate) fn dispatch(
        &self,
        name: &'static str,
        buffers: &[(&MetalBuffer, usize, usize)],
        params: &[u32; 9],
        grid: [usize; 2],
        tiled: bool,
    ) -> Result<DispatchTiming> {
        let (input_count, output_count) = match name {
            "expert_segments" => (1, 3),
            "expert_assign_compact" => (2, 3),
            "expert_project"
            | "expert_project_q4_k"
            | "expert_project_q4_k_8rows"
            | "expert_project_q4_k_16rows"
            | "expert_project_q4_k_16rows_pairs"
            | "expert_project_q4_k_factored"
            | "expert_project_q6_k_factored"
            | "expert_project_q6_k_8rows"
            | "expert_project_q5_k"
            | "expert_project_q6_k" => (3, 1),
            "expert_project_q4_k_mpp"
            | "expert_project_q5_k_mpp"
            | "expert_project_q6_k_mpp"
            | "expert_project_q4_k_mpp_k64"
            | "expert_project_q6_k_mpp_k64"
            | "expert_project_q4_k_grouped"
            | "expert_project_q6_k_grouped" => (5, 1),
            "lfm2_short_conv" => (4, 2),
            "add_rmsnorm" => (3, 2),
            "q5_0_gemv_swiglu" => (3, 1),
            "attention_decode_split" => (3, 1),
            "attention_decode_reduce" => (1, 1),
            "rope_split_table_bias" => (3, 1),
            "lfm2_split3" => (1, 3),
            _ => (2, 1),
        };
        if buffers.len() != input_count + output_count {
            return Err(Error::Dispatch(format!("invalid buffer count for {name}")));
        }
        self.check_bindings(name, buffers, input_count)?;
        if grid.contains(&0) {
            return Ok(DispatchTiming::default());
        }
        debug_assert!(params[4] <= 2);
        let p = self.builtin(name)?;
        let required = if matches!(name, "attention_context_decode_wide" | "argmax_rows") {
            1024
        } else if tiled
            || matches!(
                name,
                "rmsnorm"
                    | "add_rmsnorm"
                    | "softmax"
                    | "attention_softmax"
                    | "attention_softmax_prefix"
                    | "attention_softmax_prefix_reuse"
                    | "attention_context_decode"
            )
        {
            256
        } else if matches!(
            name,
            "gemv"
                | "gemv_vector"
                | "gemv_wide"
                | "expert_project"
                | "expert_project_q4_k"
                | "expert_project_q5_k"
                | "expert_project_q6_k"
                | "project_wide_bf16"
                | "project_wide_f16"
                | "project_mpp"
                | "q4_0_gemm_mpp"
                | "q5_0_gemm_mpp"
                | "q5_1_gemm_mpp"
                | "q5_1_gemm_mpp_k64"
                | "q4_k_gemm_mpp"
                | "q4_k_gemm_mpp_k64"
                | "q4_k_gemm_mpp_k64_m128"
                | "q5_k_gemm_mpp"
                | "q5_k_gemm_mpp_k64"
                | "mlx_affine4_gemm_mpp"
                | "mlx_affine4_gemm_mpp_k64"
                | "q6_k_gemm_mpp"
                | "q8_0_gemm_mpp"
                | "q8_0_gemm_mpp_k64"
                | "expert_project_q4_k_mpp"
                | "expert_project_q5_k_mpp"
                | "expert_project_q6_k_mpp"
                | "expert_project_q4_k_mpp_k64"
                | "expert_project_q6_k_mpp_k64"
                | "attention_scores_mpp"
                | "attention_context_mpp"
                | "attention_scores_mpp_f16"
                | "attention_context_mpp_f16"
                | "q8_0_gemv"
                | "q8_0_gemv_8rows"
                | "q8_0_gemv_k_split"
                | "q8_0_gemv_k_split4"
                | "q8_0_gemv_k_split8"
                | "q4_0_gemv_8rows"
                | "q4_k_gemv_8rows"
                | "q4_k_gemv_factored"
                | "q6_k_gemv_factored"
                | "q5_k_gemv_8rows"
                | "q6_k_gemv_8rows"
                | "q4_0_gemv"
                | "q5_0_gemv"
                | "q5_1_gemv"
                | "q4_k_gemv"
                | "q5_k_gemv"
                | "q4_0_gemm"
                | "q5_0_gemm"
                | "q5_1_gemm"
                | "q4_k_gemm"
                | "q5_k_gemm"
                | "q6_k_gemv"
                | "q6_k_gemm"
                | "q8_0_gemm"
                | "mlx_affine4_gemv"
                | "mlx_affine4_gemm"
        ) {
            128
        } else if matches!(name, "q5_0_gemv_n4" | "q5_1_gemv_n4") {
            64
        } else if let Some(simds) = Self::ksplit_simds(name, params) {
            32 * simds
        } else {
            32
        };
        if p.raw.maxTotalThreadsPerThreadgroup() < required || p.raw.threadExecutionWidth() != 32 {
            return Err(Error::Dispatch(
                "required modern Apple SIMD configuration unavailable".into(),
            ));
        }
        let param_bytes = params.map(u32::to_le_bytes).concat();
        self.encode_dispatch(&p, buffers, input_count, &param_bytes, |encoder| {
            if matches!(
                name,
                "project_mpp"
                    | "q4_0_gemm_mpp"
                    | "q5_0_gemm_mpp"
                    | "q5_1_gemm_mpp"
                    | "q5_1_gemm_mpp_k64"
                    | "q4_k_gemm_mpp"
                    | "q4_k_gemm_mpp_k64"
                    | "q4_k_gemm_mpp_k64_m128"
                    | "q5_k_gemm_mpp"
                    | "q5_k_gemm_mpp_k64"
                    | "mlx_affine4_gemm_mpp"
                    | "mlx_affine4_gemm_mpp_k64"
                    | "q6_k_gemm_mpp"
                    | "q8_0_gemm_mpp"
                    | "q8_0_gemm_mpp_k64"
                    | "expert_project_q4_k_mpp"
                    | "expert_project_q5_k_mpp"
                    | "expert_project_q6_k_mpp"
                    | "expert_project_q4_k_mpp_k64"
                    | "expert_project_q6_k_mpp_k64"
                    | "attention_scores_mpp"
                    | "attention_context_mpp"
                    | "attention_scores_mpp_f16"
                    | "attention_context_mpp_f16"
            ) {
                if p.raw.threadExecutionWidth() != 32 {
                    return Err(Error::Dispatch("MPP requires 32-wide SIMD".into()));
                }
                encoder.dispatchThreadgroups_threadsPerThreadgroup(
                    MTLSize {
                        width: grid[0].div_ceil(64),
                        height: grid[1].div_ceil(
                            if matches!(
                                name,
                                "expert_project_q4_k_mpp"
                                    | "expert_project_q5_k_mpp"
                                    | "expert_project_q6_k_mpp"
                                    | "expert_project_q4_k_mpp_k64"
                                    | "expert_project_q6_k_mpp_k64"
                            ) {
                                32
                            } else if name == "q4_k_gemm_mpp_k64_m128" {
                                128
                            } else {
                                64
                            },
                        ),
                        depth: if matches!(
                            name,
                            "project_mpp"
                                | "q4_0_gemm_mpp"
                                | "q5_0_gemm_mpp"
                                | "q5_1_gemm_mpp"
                                | "q5_1_gemm_mpp_k64"
                                | "q4_k_gemm_mpp"
                                | "q4_k_gemm_mpp_k64"
                                | "q4_k_gemm_mpp_k64_m128"
                                | "q5_k_gemm_mpp"
                                | "q5_k_gemm_mpp_k64"
                                | "mlx_affine4_gemm_mpp"
                                | "mlx_affine4_gemm_mpp_k64"
                                | "q6_k_gemm_mpp"
                                | "q8_0_gemm_mpp"
                                | "q8_0_gemm_mpp_k64"
                        ) {
                            1
                        } else if matches!(
                            name,
                            "expert_project_q4_k_mpp"
                                | "expert_project_q5_k_mpp"
                                | "expert_project_q6_k_mpp"
                                | "expert_project_q4_k_mpp_k64"
                                | "expert_project_q6_k_mpp_k64"
                        ) {
                            params[5] as usize
                        } else {
                            params[3] as usize
                        },
                    },
                    MTLSize {
                        width: 128,
                        height: 1,
                        depth: 1,
                    },
                );
            } else if matches!(name, "attention_decode_split" | "attention_decode_reduce") {
                // Split: 128 threads per (head, split). Reduce: one thread per
                // head dimension (params[6], a multiple of 32).
                let threads = if name == "attention_decode_split" {
                    128
                } else {
                    params[6] as usize
                };
                if p.raw.threadExecutionWidth() != 32
                    || p.raw.maxTotalThreadsPerThreadgroup() < threads
                {
                    return Err(Error::Dispatch(
                        "flash decode requires 32-wide SIMD groups".into(),
                    ));
                }
                encoder.dispatchThreadgroups_threadsPerThreadgroup(
                    MTLSize {
                        width: grid[0],
                        height: 1,
                        depth: 1,
                    },
                    MTLSize {
                        width: threads,
                        height: 1,
                        depth: 1,
                    },
                );
            } else if name == "argmax_rows" {
                if p.raw.threadExecutionWidth() != 32
                    || p.raw.maxTotalThreadsPerThreadgroup() < 1024
                {
                    return Err(Error::Dispatch(
                        "argmax requires 32-wide SIMD and 1,024-thread groups".into(),
                    ));
                }
                encoder.dispatchThreadgroups_threadsPerThreadgroup(
                    MTLSize {
                        width: grid[0],
                        height: 1,
                        depth: 1,
                    },
                    MTLSize {
                        width: 1024,
                        height: 1,
                        depth: 1,
                    },
                );
            } else if name == "attention_context_decode_wide" {
                if p.raw.threadExecutionWidth() != 32
                    || p.raw.maxTotalThreadsPerThreadgroup() < 1024
                {
                    return Err(Error::Dispatch(
                        "decode attention requires 32-wide SIMD and 1,024-thread groups".into(),
                    ));
                }
                encoder.dispatchThreadgroups_threadsPerThreadgroup(
                    MTLSize {
                        width: grid[0],
                        height: 1,
                        depth: 1,
                    },
                    MTLSize {
                        width: 1024,
                        height: 1,
                        depth: 1,
                    },
                );
            } else if matches!(
                name,
                "expert_project_q4_k_grouped" | "expert_project_q6_k_grouped"
            ) {
                if p.raw.threadExecutionWidth() != 32 || p.raw.maxTotalThreadsPerThreadgroup() < 128
                {
                    return Err(Error::Dispatch(
                        "grouped expert projection requires 32-wide SIMD".into(),
                    ));
                }
                encoder.dispatchThreadgroups_threadsPerThreadgroup(
                    MTLSize {
                        width: grid[0].div_ceil(16),
                        height: 1,
                        depth: params[5] as usize,
                    },
                    MTLSize {
                        width: 128,
                        height: 1,
                        depth: 1,
                    },
                );
            } else if name == "mlx_affine4_gemv_quad" {
                if p.raw.threadExecutionWidth() != 32 || p.raw.maxTotalThreadsPerThreadgroup() < 32
                {
                    return Err(Error::Dispatch(
                        "affine-Q4 quad GEMV requires 32-wide SIMD".into(),
                    ));
                }
                encoder.dispatchThreadgroups_threadsPerThreadgroup(
                    MTLSize {
                        width: grid[0].div_ceil(64),
                        height: 1,
                        depth: 1,
                    },
                    MTLSize {
                        width: 32,
                        height: 1,
                        depth: 1,
                    },
                );
            } else if matches!(name, "project_wide_bf16" | "project_wide_f16") {
                if p.raw.threadExecutionWidth() != 32 {
                    return Err(Error::Dispatch(
                        "native matrix requires 32-wide SIMD".into(),
                    ));
                }
                encoder.dispatchThreadgroups_threadsPerThreadgroup(
                    MTLSize {
                        width: grid[0].div_ceil(32),
                        height: grid[1].div_ceil(16),
                        depth: 1,
                    },
                    MTLSize {
                        width: 128,
                        height: 1,
                        depth: 1,
                    },
                );
            } else if matches!(
                name,
                "project_bf16"
                    | "project_f16"
                    | "attention_scores_matrix"
                    | "attention_context_matrix"
            ) {
                if p.raw.threadExecutionWidth() != 32 {
                    return Err(Error::Dispatch(
                        "native matrix requires 32-wide SIMD".into(),
                    ));
                }
                encoder.dispatchThreadgroups_threadsPerThreadgroup(
                    MTLSize {
                        width: grid[0].div_ceil(8),
                        height: grid[1].div_ceil(8),
                        depth: 1,
                    },
                    MTLSize {
                        width: 32,
                        height: 1,
                        depth: 1,
                    },
                );
            } else if matches!(
                name,
                "rmsnorm"
                    | "add_rmsnorm"
                    | "softmax"
                    | "attention_softmax"
                    | "attention_softmax_prefix"
                    | "attention_softmax_prefix_reuse"
                    | "attention_context_decode"
            ) {
                if p.raw.threadExecutionWidth() != 32 || p.raw.maxTotalThreadsPerThreadgroup() < 256
                {
                    return Err(Error::Dispatch(
                        "parallel reduction requires 32-wide SIMD and 256-thread groups".into(),
                    ));
                }
                encoder.dispatchThreadgroups_threadsPerThreadgroup(
                    MTLSize {
                        width: grid[0],
                        height: 1,
                        depth: 1,
                    },
                    MTLSize {
                        width: 256,
                        height: 1,
                        depth: 1,
                    },
                );
            } else if matches!(name, "gemv" | "gemv_vector") {
                if p.raw.threadExecutionWidth() != 32 || p.raw.maxTotalThreadsPerThreadgroup() < 128
                {
                    return Err(Error::Dispatch(
                        "GEMV requires 32-wide SIMD and 128-thread groups".into(),
                    ));
                }
                encoder.dispatchThreadgroups_threadsPerThreadgroup(
                    MTLSize {
                        width: grid[0].div_ceil(4),
                        height: 1,
                        depth: 1,
                    },
                    MTLSize {
                        width: 128,
                        height: 1,
                        depth: 1,
                    },
                );
            } else if let Some(simds) = Self::ksplit_simds(name, params) {
                encoder.dispatchThreadgroups_threadsPerThreadgroup(
                    MTLSize {
                        width: grid[0].div_ceil(4),
                        height: 1,
                        depth: 1,
                    },
                    MTLSize {
                        width: 32 * simds,
                        height: 1,
                        depth: 1,
                    },
                );
            } else if matches!(name, "q5_0_gemv_n4" | "q5_1_gemv_n4") {
                if p.raw.threadExecutionWidth() != 32 || p.raw.maxTotalThreadsPerThreadgroup() < 64
                {
                    return Err(Error::Dispatch(
                        "Q5_0/Q5_1 four-row GEMV requires two 32-wide SIMD groups".into(),
                    ));
                }
                encoder.dispatchThreadgroups_threadsPerThreadgroup(
                    MTLSize {
                        width: grid[0].div_ceil(8),
                        height: 1,
                        depth: 1,
                    },
                    MTLSize {
                        width: 64,
                        height: 1,
                        depth: 1,
                    },
                );
            } else if matches!(
                name,
                "q4_0_gemv"
                    | "q5_0_gemv"
                    | "q5_1_gemv"
                    | "q4_k_gemv"
                    | "q5_k_gemv"
                    | "q5_k_gemv_8rows"
                    | "q6_k_gemv"
                    | "q8_0_gemv"
                    | "q8_0_gemv_8rows"
                    | "q8_0_gemv_k_split"
                    | "q8_0_gemv_k_split4"
                    | "q8_0_gemv_k_split8"
                    | "q4_0_gemv_8rows"
                    | "q4_k_gemv_8rows"
                    | "q4_k_gemv_factored"
                    | "q6_k_gemv_8rows"
                    | "q6_k_gemv_factored"
                    | "mlx_affine4_gemv"
            ) {
                if p.raw.threadExecutionWidth() != 32 || p.raw.maxTotalThreadsPerThreadgroup() < 128
                {
                    return Err(Error::Dispatch(
                        "quantized GEMV requires 32-wide SIMD and 128-thread groups".into(),
                    ));
                }
                encoder.dispatchThreadgroups_threadsPerThreadgroup(
                    MTLSize {
                        width: grid[0].div_ceil(if matches!(name, "q8_0_gemv_k_split") {
                            2
                        } else if matches!(name, "q4_k_gemv_factored" | "q6_k_gemv_factored") {
                            16
                        } else if matches!(
                            name,
                            "q8_0_gemv_8rows"
                                | "q8_0_gemv_k_split8"
                                | "q4_0_gemv_8rows"
                                | "q4_k_gemv_8rows"
                                | "q5_k_gemv_8rows"
                                | "q6_k_gemv_8rows"
                        ) {
                            8
                        } else {
                            4
                        }),
                        height: 1,
                        depth: 1,
                    },
                    MTLSize {
                        width: 128,
                        height: 1,
                        depth: 1,
                    },
                );
            } else if matches!(
                name,
                "q4_0_gemm"
                    | "q5_0_gemm"
                    | "q5_1_gemm"
                    | "q4_k_gemm"
                    | "q5_k_gemm"
                    | "q6_k_gemm"
                    | "q8_0_gemm"
                    | "mlx_affine4_gemm"
            ) {
                if p.raw.threadExecutionWidth() != 32 || p.raw.maxTotalThreadsPerThreadgroup() < 128
                {
                    return Err(Error::Dispatch(
                        "quantized GEMM requires 32-wide SIMD and 128-thread groups".into(),
                    ));
                }
                encoder.dispatchThreadgroups_threadsPerThreadgroup(
                    MTLSize {
                        width: grid[0].div_ceil(4),
                        height: grid[1].div_ceil(4),
                        depth: 1,
                    },
                    MTLSize {
                        width: 128,
                        height: 1,
                        depth: 1,
                    },
                );
            } else if matches!(
                name,
                "expert_project"
                    | "expert_project_q4_k"
                    | "expert_project_q5_k"
                    | "expert_project_q6_k"
            ) {
                if p.raw.threadExecutionWidth() != 32 || p.raw.maxTotalThreadsPerThreadgroup() < 128
                {
                    return Err(Error::Dispatch(
                        "expert projection requires 32-wide SIMD and 128-thread groups".into(),
                    ));
                }
                encoder.dispatchThreadgroups_threadsPerThreadgroup(
                    MTLSize {
                        width: grid[0].div_ceil(4),
                        height: grid[1],
                        depth: 1,
                    },
                    MTLSize {
                        width: 128,
                        height: 1,
                        depth: 1,
                    },
                );
            } else if name == "expert_project_q4_k_8rows"
                || name == "expert_project_q4_k_16rows"
                || name == "expert_project_q4_k_16rows_pairs"
                || name == "expert_project_q4_k_factored"
                || name == "expert_project_q6_k_factored"
                || name == "expert_project_q6_k_8rows"
            {
                if p.raw.threadExecutionWidth() != 32 || p.raw.maxTotalThreadsPerThreadgroup() < 128
                {
                    return Err(Error::Dispatch(
                        "quantized expert projection requires 32-wide SIMD and 128-thread groups"
                            .into(),
                    ));
                }
                encoder.dispatchThreadgroups_threadsPerThreadgroup(
                    MTLSize {
                        width: grid[0].div_ceil(
                            if name == "expert_project_q4_k_16rows"
                                || name == "expert_project_q4_k_16rows_pairs"
                                || name == "expert_project_q4_k_factored"
                                || name == "expert_project_q6_k_factored"
                            {
                                16
                            } else {
                                8
                            },
                        ),
                        height: grid[1],
                        depth: 1,
                    },
                    MTLSize {
                        width: 128,
                        height: 1,
                        depth: 1,
                    },
                );
            } else if name == "expert_assign_compact" {
                if p.raw.threadExecutionWidth() != 32 || p.raw.maxTotalThreadsPerThreadgroup() < 128
                {
                    return Err(Error::Dispatch(
                        "expert compaction requires 32-wide SIMD and 128-thread groups".into(),
                    ));
                }
                encoder.dispatchThreadgroups_threadsPerThreadgroup(
                    MTLSize {
                        width: 1,
                        height: grid[1],
                        depth: 1,
                    },
                    MTLSize {
                        width: 128,
                        height: 1,
                        depth: 1,
                    },
                );
            } else if name == "gemv_wide" {
                if p.raw.threadExecutionWidth() != 32 || p.raw.maxTotalThreadsPerThreadgroup() < 128
                {
                    return Err(Error::Dispatch(
                        "wide GEMV requires 32-wide SIMD and 128-thread groups".into(),
                    ));
                }
                encoder.dispatchThreadgroups_threadsPerThreadgroup(
                    MTLSize {
                        width: grid[0],
                        height: grid[1],
                        depth: 1,
                    },
                    MTLSize {
                        width: 128,
                        height: 1,
                        depth: 1,
                    },
                );
            } else if tiled {
                if p.raw.maxTotalThreadsPerThreadgroup() < 256 {
                    return Err(Error::Dispatch("16x16 tile unsupported".into()));
                }
                encoder.dispatchThreadgroups_threadsPerThreadgroup(
                    MTLSize {
                        width: grid[0].div_ceil(16),
                        height: grid[1].div_ceil(16),
                        depth: 1,
                    },
                    MTLSize {
                        width: 16,
                        height: 16,
                        depth: 1,
                    },
                );
            } else {
                let width = p.raw.maxTotalThreadsPerThreadgroup().min(256);
                encoder.dispatchThreads_threadsPerThreadgroup(
                    MTLSize {
                        width: grid[0],
                        height: grid[1],
                        depth: 1,
                    },
                    MTLSize {
                        width,
                        height: 1,
                        depth: 1,
                    },
                );
            }
            Ok(())
        })
    }
}
