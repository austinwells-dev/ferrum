//! The only module allowed to interact with Objective-C or raw memory.
use crate::{Error, Result};
use objc2::{
    rc::{Retained, autoreleasepool},
    runtime::ProtocolObject,
};
use objc2_foundation::NSString;
use objc2_metal::*;
use std::{
    cell::{Cell, RefCell},
    collections::HashMap,
    ptr::NonNull,
    rc::Rc,
    time::{Duration, Instant},
};

#[link(name = "CoreGraphics", kind = "framework")]
// Linking CoreGraphics permits default-device discovery in a command-line process.
unsafe extern "C" {}

pub struct ComputePipeline {
    raw: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
}
/// Shared, owned allocation. No mapped pointers escape this module.
/// Rc deliberately makes storage and the device !Send and !Sync in Phase 1.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Completion {
    Encoding,
    Completed,
    Failed,
}
type RetiredBuffer = (
    Retained<ProtocolObject<dyn MTLBuffer>>,
    Option<Rc<Cell<Completion>>>,
    bool,
);
#[derive(Default)]
struct Arena {
    free: HashMap<usize, Vec<RetiredBuffer>>,
    live: usize,
    peak: usize,
    capacity: usize,
    high_water: usize,
}
impl Arena {
    // Reclaim completed, unowned storage before a fresh allocation would overflow
    // the retention budget. In-flight and live buffers are never candidates.
    fn make_room(&mut self, requested: usize) {
        let target = (256usize * 1024 * 1024).saturating_sub(requested);
        if self.capacity <= target {
            return;
        }
        let mut sizes: Vec<_> = self.free.keys().copied().collect();
        sizes.sort_unstable_by(|a, b| b.cmp(a));
        for size in sizes {
            let bin = self.free.get_mut(&size).unwrap();
            bin.retain(|(_, ready, counted)| {
                let completed = ready
                    .as_ref()
                    .is_none_or(|x| x.get() == Completion::Completed);
                if self.capacity > target && completed && !counted {
                    self.capacity -= size;
                    false
                } else {
                    true
                }
            });
            if self.capacity <= target {
                break;
            }
        }
    }
    fn complete(&mut self) {
        let mut retired = 0;
        let mut released = 0;
        let over_capacity = self.capacity > 256 * 1024 * 1024;
        for (bytes, bin) in &mut self.free {
            bin.retain_mut(|(_, ready, counted)| {
                let state = ready.as_ref().map_or(Completion::Completed, |r| r.get());
                if state != Completion::Encoding && *counted {
                    retired += *bytes;
                    *counted = false;
                }
                if state == Completion::Failed || (over_capacity && state == Completion::Completed)
                {
                    released += *bytes;
                    false
                } else {
                    true
                }
            });
        }
        self.live -= retired;
        self.capacity -= released;
    }
    fn acquire(&mut self, bytes: usize) {
        self.live += bytes;
        self.peak = self.peak.max(self.live);
        self.high_water = self.high_water.max(self.capacity);
    }
}
type WriteRange = (usize, usize, Rc<Cell<Completion>>);
pub struct MetalBuffer {
    raw: Retained<ProtocolObject<dyn MTLBuffer>>,
    len: usize,
    owner: Rc<()>,
    ready: RefCell<Option<Rc<Cell<Completion>>>>,
    writes: RefCell<Vec<WriteRange>>,
    arena: Option<Rc<RefCell<Arena>>>,
}
impl Drop for MetalBuffer {
    fn drop(&mut self) {
        if let Some(arena) = &self.arena {
            let mut arena = arena.borrow_mut();
            let bytes = self.raw.length();
            let state = self
                .ready
                .borrow()
                .as_ref()
                .map_or(Completion::Completed, |r| r.get());
            let pending = state == Completion::Encoding;
            if !pending {
                arena.live -= bytes;
            }
            // A pool entry exists only after the final Tensor owner is gone.
            // A pending epoch keeps it ineligible until successful completion.
            if pending || (state == Completion::Completed && arena.capacity <= 256 * 1024 * 1024) {
                arena.free.entry(bytes).or_default().push((
                    self.raw.clone(),
                    self.ready.borrow().clone(),
                    pending,
                ));
            } else {
                arena.capacity -= bytes;
            }
        }
    }
}
impl MetalBuffer {
    pub fn len_bytes(&self) -> usize {
        self.len
    }
    pub fn allocation_bytes(&self) -> usize {
        self.raw.length()
    }
    pub fn alignment(&self) -> usize {
        1usize << self.raw.contents().as_ptr().addr().trailing_zeros()
    }
    pub(crate) fn with_bytes_mut<T>(&mut self, f: impl FnOnce(&mut [u8]) -> T) -> T {
        // SAFETY: the allocation owns len initialized, writable shared bytes. Exclusive
        // access is only granted before publication as a Tensor; no GPU work is in flight.
        // The closure's borrow cannot escape, and no raw pointer leaves this module.
        let bytes = unsafe {
            std::slice::from_raw_parts_mut(self.raw.contents().as_ptr().cast::<u8>(), self.len)
        };
        f(bytes)
    }
    fn readable(&self, offset: usize, length: usize) -> bool {
        self.writes.borrow().iter().all(|(start, end, state)| {
            offset + length <= *start || offset >= *end || state.get() == Completion::Completed
        })
    }
    pub(crate) fn with_bytes<T>(
        &self,
        offset: usize,
        length: usize,
        f: impl FnOnce(&[u8]) -> T,
    ) -> T {
        assert!(
            offset
                .checked_add(length)
                .is_some_and(|end| end <= self.len)
        );
        assert!(
            self.readable(offset, length),
            "GPU output range is not successfully completed"
        );
        // SAFETY: all bytes are initialized; dispatch waits before returning on both success
        // and GPU error. Rc prevents cross-thread use; only read-only tensors are published.
        let bytes = unsafe {
            std::slice::from_raw_parts(
                self.raw.contents().as_ptr().cast::<u8>().add(offset),
                length,
            )
        };
        f(bytes)
    }
}
#[derive(Debug, Clone, Default)]
pub struct DispatchTiming {
    pub submission: Duration,
    pub synchronized: Duration,
    pub gpu: Option<Duration>,
    pub dispatches: usize,
}
/// Single-threaded synchronous execution context with one queue and cached pipelines.
#[derive(Clone, Copy, Debug, Default, serde::Serialize)]
pub struct Counters {
    pub allocations: usize,
    pub allocated_bytes: usize,
    pub dispatches: usize,
    pub command_buffers: usize,
    pub completion_waits: usize,
    pub encode: Duration,
    pub wait: Duration,
    pub gpu: Duration,
    pub allocation_time: Duration,
    pub reused_bytes: usize,
    pub transient_live_bytes: usize,
    pub transient_peak_bytes: usize,
    pub arena_capacity: usize,
    pub arena_high_water: usize,
}

/// Opt-in aggregate operation measurements, with no tensor retention.
#[derive(Clone, Debug, Default, serde::Serialize)]
pub struct ProfileEntry {
    pub calls: usize,
    pub wall: Duration,
    pub allocation: Duration,
    pub submission: Duration,
    pub synchronized: Duration,
    pub gpu: Duration,
    pub gpu_samples: usize,
    pub allocation_bytes: usize,
}
pub type Profile = std::collections::BTreeMap<&'static str, ProfileEntry>;
/// An encoding epoch owns every referenced Metal resource through completion.
/// No tensor carrying this epoch may be mapped until successful completion.
struct Submission {
    command: Retained<ProtocolObject<dyn MTLCommandBuffer>>,
    resources: Vec<Retained<ProtocolObject<dyn MTLBuffer>>>,
    ready: Rc<Cell<Completion>>,
    dispatches: usize,
}
/// Private scope: model code must finish before publishing cache or results.
/// Unwinding and early errors still drain submitted work.
pub(crate) struct Execution<'a> {
    device: &'a MetalDevice,
}
impl Execution<'_> {
    pub(crate) fn finish(self) -> Result<()> {
        self.device.flush()
    }
}
impl Drop for Execution<'_> {
    fn drop(&mut self) {
        let _ = self.device.flush();
        self.device.batching.set(false);
    }
}
struct ProfileScope<'a> {
    flag: &'a Cell<bool>,
    previous: bool,
}
impl Drop for ProfileScope<'_> {
    fn drop(&mut self) {
        self.flag.set(self.previous);
    }
}
pub struct MetalDevice {
    lm_head_profile: Cell<bool>,
    #[cfg(test)]
    fail_after: Cell<Option<usize>>,
    #[cfg(test)]
    fail_completion: Cell<bool>,
    arena: Rc<RefCell<Arena>>,
    native_matmul: Cell<bool>,
    split_k_gemv: Cell<bool>,
    reference_math: Cell<bool>,
    batching: Cell<bool>,
    batch_limit: Cell<usize>,
    pending: RefCell<Option<Submission>>,
    profiling: Cell<bool>,
    profile: RefCell<Profile>,
    counters: Cell<Counters>,
    raw: Retained<ProtocolObject<dyn MTLDevice>>,
    queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
    name: String,
    owner: Rc<()>,
    libraries: RefCell<HashMap<String, Retained<ProtocolObject<dyn MTLLibrary>>>>,
    pipelines: RefCell<HashMap<(String, String), Rc<ComputePipeline>>>,
    builtins: RefCell<HashMap<&'static str, Rc<ComputePipeline>>>,
}
impl MetalDevice {
    pub fn new() -> Result<Self> {
        let raw = MTLCreateSystemDefaultDevice()
            .ok_or_else(|| Error::Initialization("no default GPU".into()))?;
        if !raw.hasUnifiedMemory() {
            return Err(Error::Initialization("unified memory GPU required".into()));
        }
        let queue = raw
            .newCommandQueue()
            .ok_or_else(|| Error::Initialization("command queue creation failed".into()))?;
        Ok(Self {
            lm_head_profile: Cell::new(false),
            #[cfg(test)]
            fail_after: Cell::new(None),
            #[cfg(test)]
            fail_completion: Cell::new(false),
            arena: Rc::new(RefCell::new(Arena::default())),
            native_matmul: Cell::new(raw.supportsFamily(MTLGPUFamily::Apple7)),
            split_k_gemv: Cell::new(true),
            reference_math: Cell::new(false),
            batching: Cell::new(false),
            batch_limit: Cell::new(1024),
            pending: RefCell::new(None),
            counters: Cell::new(Counters::default()),
            profiling: Cell::new(false),
            profile: RefCell::new(Profile::new()),
            name: raw.name().to_string(),
            raw,
            queue,
            owner: Rc::new(()),
            libraries: RefCell::new(HashMap::new()),
            pipelines: RefCell::new(HashMap::new()),
            builtins: RefCell::new(HashMap::new()),
        })
    }
    pub fn set_native_matmul(&self, enabled: bool) -> Result<()> {
        if self.batching.get() || (enabled && !self.raw.supportsFamily(MTLGPUFamily::Apple7)) {
            return Err(Error::Parameter(
                "native SIMD matrix support required".into(),
            ));
        }
        self.native_matmul.set(enabled);
        Ok(())
    }
    /// Select the multi-SIMD split-K BF16 GEMV for large aligned rows.
    pub fn set_split_k_gemv(&self, enabled: bool) -> Result<()> {
        if self.batching.get() {
            return Err(Error::Parameter(
                "cannot change GEMV during execution".into(),
            ));
        }
        self.split_k_gemv.set(enabled);
        Ok(())
    }
    pub(crate) fn split_k_gemv(&self) -> bool {
        self.split_k_gemv.get()
    }
    pub(crate) fn mpp_projection(&self) -> bool {
        self.native_matmul.get()
            && self.raw.supportsFamily(MTLGPUFamily::Apple10)
            && self.raw.supportsFamily(MTLGPUFamily::Metal4)
    }
    pub(crate) fn native_matmul(&self) -> bool {
        self.native_matmul.get()
    }
    pub fn capabilities(&self) -> Result<serde_json::Value> {
        let pipeline = self.builtin("gemv")?;
        Ok(
            serde_json::json!({"apple_families":self.apple_families(),"metal4":self.raw.supportsFamily(MTLGPUFamily::Metal4),"msl_requested":"3.1","mpp_msl_requested":"4.0","mpp_available":self.mpp_projection(),"mpp_compiled":self.mpp_projection() && self.builtin("project_mpp").is_ok(),"simdgroup_matrix":self.raw.supportsFamily(MTLGPUFamily::Apple7),"native_bfloat_compiled":self.builtin("project_bf16").is_ok(),"thread_execution_width":pipeline.raw.threadExecutionWidth(),"pipeline_max_threads":pipeline.raw.maxTotalThreadsPerThreadgroup(),"device_max_threads":self.raw.maxThreadsPerThreadgroup().width,"recommended_working_set":self.recommended_max_working_set(),"storage":"shared","hazards":"tracked"}),
        )
    }
    /// Validation-only ordered reductions reproduce the immutable F32 diagnostic.
    /// Ordinary inference uses the parallel kernels.
    pub fn set_reference_math(&self, enabled: bool) -> Result<()> {
        if self.batching.get() {
            return Err(Error::Parameter("cannot change math in execution".into()));
        }
        self.reference_math.set(enabled);
        Ok(())
    }
    pub(crate) fn reference_math(&self) -> bool {
        self.reference_math.get()
    }
    /// Maximum kernels per completion boundary in model execution. One is the control.
    pub fn set_batch_limit(&self, limit: usize) -> Result<()> {
        if limit == 0 || self.batching.get() {
            return Err(Error::Parameter("invalid active batch limit".into()));
        }
        self.batch_limit.set(limit);
        Ok(())
    }
    pub(crate) fn execution(&self) -> Result<Execution<'_>> {
        if self.batching.replace(true) {
            return Err(Error::Dispatch("nested execution".into()));
        }
        Ok(Execution { device: self })
    }
    fn flush(&self) -> Result<()> {
        let Some(submission) = self.pending.borrow_mut().take() else {
            return Ok(());
        };
        let start = Instant::now();
        submission.command.commit();
        submission.command.waitUntilCompleted();
        let mut counters = self.counters.get();
        counters.command_buffers += 1;
        counters.completion_waits += 1;
        counters.wait += start.elapsed();
        let seconds = submission.command.GPUEndTime() - submission.command.GPUStartTime();
        if seconds.is_finite() && seconds > 0. {
            counters.gpu += Duration::from_secs_f64(seconds);
        }
        self.counters.set(counters);
        let succeeded = submission.command.status() == MTLCommandBufferStatus::Completed;
        #[cfg(test)]
        let succeeded = succeeded && !self.fail_completion.replace(false);
        if !succeeded {
            submission.ready.set(Completion::Failed);
            self.arena.borrow_mut().complete();
            return Err(Error::Synchronization(
                submission
                    .command
                    .error()
                    .map(|e| e.to_string())
                    .unwrap_or_else(|| format!("status {:?}", submission.command.status())),
            ));
        }
        submission.ready.set(Completion::Completed);
        self.arena.borrow_mut().complete();
        // submission.resources drops only after the wait and status check.
        Ok(())
    }
    pub(crate) fn profile_lm_head<T>(&self, f: impl FnOnce() -> Result<T>) -> Result<T> {
        let _scope = ProfileScope {
            flag: &self.lm_head_profile,
            previous: self.lm_head_profile.replace(true),
        };
        f()
    }
    pub fn set_profiling(&self, enabled: bool) {
        self.profiling.set(enabled);
        self.profile.borrow_mut().clear();
    }
    pub(crate) fn profiling(&self) -> bool {
        self.profiling.get()
    }
    pub fn take_profile(&self) -> Profile {
        std::mem::take(&mut *self.profile.borrow_mut())
    }
    pub(crate) fn record_profile(
        &self,
        name: &'static str,
        wall: Duration,
        allocation: Duration,
        bytes: usize,
        timing: &DispatchTiming,
    ) {
        let name = if self.lm_head_profile.get() {
            match name {
                "gemv" | "gemv_vector" | "gemv_wide" | "q4_0_gemv" | "q5_0_gemv" | "q5_1_gemv"
                | "q4_k_gemv" | "q5_k_gemv" | "q6_k_gemv" | "q8_0_gemv" | "mlx_affine4_gemv" => {
                    "lm_head_gemv"
                }
                "project_bf16"
                | "project_f16"
                | "project_wide_bf16"
                | "project_wide_f16"
                | "project_mpp"
                | "q4_0_gemm_mpp"
                | "q5_0_gemm_mpp"
                | "q5_1_gemm_mpp"
                | "q4_k_gemm_mpp"
                | "q5_k_gemm_mpp"
                | "q6_k_gemm_mpp"
                | "q8_0_gemm_mpp"
                | "matmul_nt"
                | "q4_0_gemm"
                | "q5_0_gemm"
                | "q5_1_gemm"
                | "q4_k_gemm"
                | "q5_k_gemm"
                | "q6_k_gemm"
                | "q8_0_gemm"
                | "mlx_affine4_gemm"
                | "mlx_affine4_gemm_mpp" => "lm_head_matmul",
                _ => name,
            }
        } else {
            name
        };
        let mut profile = self.profile.borrow_mut();
        let entry = profile.entry(name).or_default();
        entry.calls += 1;
        entry.wall += wall;
        entry.allocation += allocation;
        entry.submission += timing.submission;
        entry.synchronized += timing.synchronized;
        entry.allocation_bytes += bytes;
        if let Some(gpu) = timing.gpu {
            entry.gpu += gpu;
            entry.gpu_samples += 1;
        }
    }
    pub fn counters(&self) -> Counters {
        let mut counters = self.counters.get();
        let arena = self.arena.borrow();
        counters.transient_live_bytes = arena.live;
        counters.transient_peak_bytes = arena.peak;
        counters.arena_capacity = arena.capacity;
        counters.arena_high_water = arena.high_water;
        counters
    }
    /// Reset the peak observation to the currently live transient payload.
    pub fn reset_transient_peak(&self) {
        let mut arena = self.arena.borrow_mut();
        arena.peak = arena.live;
    }
    pub fn name(&self) -> &str {
        &self.name
    }
    pub fn has_unified_memory(&self) -> bool {
        self.raw.hasUnifiedMemory()
    }
    pub fn recommended_max_working_set(&self) -> u64 {
        self.raw.recommendedMaxWorkingSetSize()
    }
    pub fn max_buffer_length(&self) -> usize {
        self.raw.maxBufferLength()
    }
    pub fn apple_families(&self) -> Vec<usize> {
        (1..=10)
            .filter(|i| self.raw.supportsFamily(MTLGPUFamily(1000 + *i as isize)))
            .collect()
    }
    pub fn cached_pipeline_count(&self) -> usize {
        self.pipelines.borrow().len()
    }
    pub fn allocate(&self, bytes: usize) -> Result<MetalBuffer> {
        let start = Instant::now();
        let length = bytes.max(4);
        if length > self.max_buffer_length() {
            return Err(Error::Allocation(bytes));
        }
        let raw = self
            .raw
            .newBufferWithLength_options(length, MTLResourceOptions::StorageModeShared)
            .ok_or(Error::Allocation(bytes))?;
        // SAFETY: newBuffer owns length writable shared bytes and is not submitted anywhere.
        unsafe {
            std::ptr::write_bytes(raw.contents().as_ptr().cast::<u8>(), 0, length);
        }
        let mut counters = self.counters.get();
        counters.allocations += 1;
        counters.allocated_bytes += length;
        counters.allocation_time += start.elapsed();
        self.counters.set(counters);
        Ok(MetalBuffer {
            raw,
            len: bytes,
            owner: self.owner.clone(),
            ready: RefCell::new(None),
            writes: RefCell::new(Vec::new()),
            arena: None,
        })
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
        let bin = arena.free.entry(length).or_default();
        let available = bin.iter().rposition(|(_, ready, _)| {
            ready
                .as_ref()
                .is_none_or(|r| r.get() == Completion::Completed)
        });
        let mut buffer = if let Some(i) = available {
            let (raw, _, _) = bin.swap_remove(i);
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
            }
        } else {
            arena.make_room(length);
            let mut buffer = self.allocate(length)?;
            buffer.len = bytes;
            arena.capacity += length;
            buffer
        };
        arena.acquire(length);
        buffer.arena = Some(self.arena.clone());
        Ok(buffer)
    }
    pub fn compile_kernel(&self, source: &str, name: &str) -> Result<Rc<ComputePipeline>> {
        self.compile_kernel_version(source, name, MTLLanguageVersion::Version3_1)
    }
    fn compile_kernel_version(
        &self,
        source: &str,
        name: &str,
        version: MTLLanguageVersion,
    ) -> Result<Rc<ComputePipeline>> {
        autoreleasepool(|_| {
            let library_key = format!("{}:{source}", version.0);
            let key = (library_key.clone(), name.to_owned());
            if let Some(p) = self.pipelines.borrow().get(&key) {
                return Ok(p.clone());
            }
            let mut libraries = self.libraries.borrow_mut();
            if !libraries.contains_key(&library_key) {
                let options = MTLCompileOptions::new();
                options.setLanguageVersion(version);
                options.setMathMode(MTLMathMode::Safe);
                options.setMathFloatingPointFunctions(MTLMathFloatingPointFunctions::Precise);
                let lib = self
                    .raw
                    .newLibraryWithSource_options_error(&NSString::from_str(source), Some(&options))
                    .map_err(|e| Error::Compilation(e.to_string()))?;
                libraries.insert(library_key.clone(), lib);
            }
            let lib = libraries
                .get(&library_key)
                .ok_or_else(|| Error::Compilation("library cache invariant".into()))?;
            let function = lib
                .newFunctionWithName(&NSString::from_str(name))
                .ok_or_else(|| Error::MissingKernel(name.into()))?;
            let raw = self
                .raw
                .newComputePipelineStateWithFunction_error(&function)
                .map_err(|e| Error::Pipeline {
                    name: name.into(),
                    message: e.to_string(),
                })?;
            let p = Rc::new(ComputePipeline { raw });
            self.pipelines.borrow_mut().insert(key, p.clone());
            Ok(p)
        })
    }
    pub(crate) fn builtin(&self, name: &'static str) -> Result<Rc<ComputePipeline>> {
        if let Some(p) = self.builtins.borrow().get(name) {
            return Ok(p.clone());
        }
        let p = if matches!(
            name,
            "project_mpp"
                | "q4_0_gemm_mpp"
                | "q5_0_gemm_mpp"
                | "q5_1_gemm_mpp"
                | "q4_k_gemm_mpp"
                | "q5_k_gemm_mpp"
                | "mlx_affine4_gemm_mpp"
                | "q6_k_gemm_mpp"
                | "q8_0_gemm_mpp"
                | "attention_scores_mpp"
                | "attention_context_mpp"
        ) {
            self.compile_kernel_version(
                include_str!("shaders/project_mpp.metal"),
                name,
                MTLLanguageVersion::Version4_0,
            )?
        } else {
            self.compile_kernel(include_str!("shaders/ops.metal"), name)?
        };
        self.builtins.borrow_mut().insert(name, p.clone());
        Ok(p)
    }
    pub(crate) fn owns(&self, buffer: &MetalBuffer) -> bool {
        Rc::ptr_eq(&self.owner, &buffer.owner)
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
        #[cfg(test)]
        if self
            .fail_after
            .get()
            .is_some_and(|n| self.counters.get().dispatches >= n)
        {
            return Err(Error::Dispatch("injected encoder failure".into()));
        }
        for (b, offset, length) in buffers {
            if offset
                .checked_add(*length)
                .is_none_or(|end| end > b.len_bytes())
            {
                return Err(Error::Range("binding offset exceeds allocation".into()));
            }
            if !self.owns(b) {
                return Err(Error::DeviceMismatch);
            }
        }
        for (buffer, offset, length) in &buffers[..buffers.len().min(2)] {
            if buffer.writes.borrow().iter().any(|(start, end, state)| {
                *offset < *end && offset + length > *start && state.get() == Completion::Failed
            }) {
                return Err(Error::Synchronization(
                    "input range belongs to a failed submission".into(),
                ));
            }
        }
        if grid.contains(&0) {
            return Ok(DispatchTiming::default());
        }
        debug_assert_eq!(buffers.len(), 3);
        debug_assert!(params[4] <= 2);
        let p = self.builtin(name)?;
        let required = if tiled
            || matches!(
                name,
                "rmsnorm" | "softmax" | "attention_softmax" | "attention_context_decode"
            ) {
            256
        } else if matches!(
            name,
            "gemv"
                | "gemv_vector"
                | "gemv_wide"
                | "project_wide_bf16"
                | "project_wide_f16"
                | "project_mpp"
                | "q4_0_gemm_mpp"
                | "q5_0_gemm_mpp"
                | "q5_1_gemm_mpp"
                | "q4_k_gemm_mpp"
                | "q5_k_gemm_mpp"
                | "mlx_affine4_gemm_mpp"
                | "q6_k_gemm_mpp"
                | "q8_0_gemm_mpp"
                | "attention_scores_mpp"
                | "attention_context_mpp"
                | "q8_0_gemv"
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
        } else {
            32
        };
        if p.raw.maxTotalThreadsPerThreadgroup() < required || p.raw.threadExecutionWidth() != 32 {
            return Err(Error::Dispatch(
                "required modern Apple SIMD configuration unavailable".into(),
            ));
        }
        autoreleasepool(|_| {
            let start = Instant::now();
            if self.pending.borrow().is_none() {
                let command = self
                    .queue
                    .commandBuffer()
                    .ok_or_else(|| Error::Dispatch("command buffer creation failed".into()))?;
                *self.pending.borrow_mut() = Some(Submission {
                    command,
                    resources: Vec::new(),
                    ready: Rc::new(Cell::new(Completion::Encoding)),
                    dispatches: 0,
                });
            }
            let mut pending = self.pending.borrow_mut();
            let submission = pending
                .as_mut()
                .ok_or_else(|| Error::Dispatch("missing submission".into()))?;
            let command = &submission.command;
            let encoder = command
                .computeCommandEncoder()
                .ok_or_else(|| Error::Dispatch("compute encoder creation failed".into()))?;
            encoder.setComputePipelineState(&p.raw);
            // SAFETY: crate-private callers validate dimensions, dtypes and lengths against the
            // embedded kernel ABI. Buffers stay alive through completion; params is copied by Metal.
            unsafe {
                for (i, (b, offset, _)) in buffers.iter().enumerate() {
                    encoder.setBuffer_offset_atIndex(Some(&b.raw), *offset, i);
                }
                encoder.setBytes_length_atIndex(
                    NonNull::from(params).cast(),
                    size_of_val(params),
                    3,
                );
            }
            if matches!(
                name,
                "project_mpp"
                    | "q4_0_gemm_mpp"
                    | "q5_0_gemm_mpp"
                    | "q5_1_gemm_mpp"
                    | "q4_k_gemm_mpp"
                    | "q5_k_gemm_mpp"
                    | "mlx_affine4_gemm_mpp"
                    | "q6_k_gemm_mpp"
                    | "q8_0_gemm_mpp"
                    | "attention_scores_mpp"
                    | "attention_context_mpp"
            ) {
                if p.raw.threadExecutionWidth() != 32 {
                    return Err(Error::Dispatch("MPP requires 32-wide SIMD".into()));
                }
                encoder.dispatchThreadgroups_threadsPerThreadgroup(
                    MTLSize {
                        width: grid[0].div_ceil(64),
                        height: grid[1].div_ceil(64),
                        depth: if matches!(
                            name,
                            "project_mpp"
                                | "q4_0_gemm_mpp"
                                | "q5_0_gemm_mpp"
                                | "q5_1_gemm_mpp"
                                | "q4_k_gemm_mpp"
                                | "q5_k_gemm_mpp"
                                | "mlx_affine4_gemm_mpp"
                                | "q6_k_gemm_mpp"
                                | "q8_0_gemm_mpp"
                        ) {
                            1
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
                "rmsnorm" | "softmax" | "attention_softmax" | "attention_context_decode"
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
            } else if matches!(
                name,
                "q4_0_gemv"
                    | "q5_0_gemv"
                    | "q5_1_gemv"
                    | "q4_k_gemv"
                    | "q5_k_gemv"
                    | "q6_k_gemv"
                    | "q8_0_gemv"
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
                        height: 1,
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
            encoder.endEncoding();
            submission
                .resources
                .extend(buffers.iter().map(|(b, _, _)| b.raw.clone()));
            *buffers[2].0.ready.borrow_mut() = Some(submission.ready.clone());
            let (output, offset, length) = buffers[2];
            let mut writes = output.writes.borrow_mut();
            writes.retain(|(_, _, state)| state.get() != Completion::Completed);
            writes.push((offset, offset + length, submission.ready.clone()));
            drop(writes);
            submission.dispatches += 1;
            let flush = !self.batching.get() || submission.dispatches >= self.batch_limit.get();
            let mut counters = self.counters.get();
            counters.dispatches += 1;
            let submission_time = start.elapsed();
            counters.encode += submission_time;
            self.counters.set(counters);
            drop(pending);
            let before_gpu = counters.gpu;
            if flush {
                self.flush()?;
            }
            let synchronized = start.elapsed();
            let gpu = flush.then(|| self.counters.get().gpu - before_gpu);
            Ok(DispatchTiming {
                submission: submission_time,
                synchronized: if self.batching.get() && self.batch_limit.get() != 1 {
                    submission_time
                } else {
                    synchronized
                },
                gpu: if self.batching.get() && self.batch_limit.get() != 1 {
                    None
                } else {
                    gpu
                },
                dispatches: 1,
            })
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{DType, Tensor};

    #[test]
    fn execution_owns_dropped_intermediates_and_drains_on_error() {
        let d = MetalDevice::new().unwrap();
        d.set_batch_limit(32).unwrap();
        let a = Tensor::from_f32(&d, [257], DType::F32, &[1.; 257]).unwrap();
        for _ in 0..20 {
            let before = d.counters();
            let scope = d.execution().unwrap();
            let mut x = a.clone();
            for _ in 0..70 {
                x = d.add(&x, &a).unwrap().tensor;
            }
            scope.finish().unwrap();
            assert_eq!(x.to_f32(), vec![71.; 257]);
            assert_eq!(d.counters().completion_waits - before.completion_waits, 3);
        }
        let x;
        {
            let _scope = d.execution().unwrap();
            x = d.add(&a, &a).unwrap().tensor;
            assert!(d.matmul(&a, &a).is_err());
            // Drop completes pending work even when the caller returns an error.
        }
        assert_eq!(x.to_f32(), vec![2.; 257]);
        assert_eq!(a.to_f32(), vec![1.; 257]);
    }

    #[test]
    fn mapping_is_excluded_until_scope_completion() {
        let d = MetalDevice::new().unwrap();
        let a = Tensor::from_f32(&d, [1], DType::F32, &[2.]).unwrap();
        let scope = d.execution().unwrap();
        let x = d.add(&a, &a).unwrap().tensor;
        assert!(std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| x.to_f32())).is_err());
        scope.finish().unwrap();
        assert_eq!(x.to_f32(), vec![4.]);
    }
}

#[cfg(test)]
mod transaction_tests {
    use super::*;
    use crate::{
        DType,
        loader::Weights,
        model::{ModelConfig, Transformer, tiny},
    };
    #[test]
    fn late_encoder_failure_preserves_cache_and_retry() {
        let d = MetalDevice::new().unwrap();
        d.set_batch_limit(8).unwrap();
        let c = ModelConfig::tiny(DType::F32);
        let w = Weights::from_bytes(
            &d,
            &tiny::serialize(&c, &tiny::weights(&c).unwrap()).unwrap(),
        )
        .unwrap();
        let model = Transformer::from_weights(&d, c, &w).unwrap();
        let (_, mut cache) = model.forward_prefill(&d, &[1, 2]).unwrap();
        let before = cache.active(0).unwrap().unwrap().0.to_f32();
        let mut expected_cache = cache.clone();
        let expected = model
            .forward_decode(&d, 3, &mut expected_cache)
            .unwrap()
            .to_f32();
        d.fail_after.set(Some(d.counters().dispatches + 35));
        assert!(model.forward_decode(&d, 3, &mut cache).is_err());
        assert_eq!(cache.len().unwrap(), 2);
        assert_eq!(cache.active(0).unwrap().unwrap().0.to_f32(), before);
        d.fail_after.set(None);
        assert_eq!(
            model.forward_decode(&d, 3, &mut cache).unwrap().to_f32(),
            expected
        );
    }
}

#[cfg(test)]
mod arena_tests {
    use super::*;
    use crate::{DType, Tensor};
    #[test]
    fn arena_never_recycles_live_or_pending_tensors() {
        let d = MetalDevice::new().unwrap();
        let a = Tensor::from_f32(&d, [257], DType::F32, &[1.; 257]).unwrap();
        let mut retained = Vec::new();
        for round in 0..4 {
            let scope = d.execution().unwrap();
            let before = d.counters();
            let x = d.add(&a, &a).unwrap().tensor;
            let first = d.counters().allocations;
            let y = d.add(&x, &a).unwrap().tensor;
            drop(x);
            let z = d.add(&y, &a).unwrap().tensor;
            if round == 0 {
                assert!(d.counters().allocations > first);
            }
            scope.finish().unwrap();
            assert_eq!(y.to_f32(), vec![3.; 257]);
            assert_eq!(z.to_f32(), vec![4.; 257]);
            retained.push(z);
            if round > 1 {
                assert!(d.counters().reused_bytes > before.reused_bytes);
            }
        }
        for z in retained {
            assert_eq!(z.to_f32(), vec![4.; 257]);
        }
    }
}

#[cfg(test)]
mod failed_completion_tests {
    use super::*;
    use crate::{
        DType,
        loader::Weights,
        model::{ModelConfig, Transformer, tiny},
        nn::attention::Trace,
    };
    #[test]
    fn failed_suffix_does_not_poison_published_cache_prefix() {
        let d = MetalDevice::new().unwrap();
        let c = ModelConfig::tiny(DType::F32);
        let w = Weights::from_bytes(
            &d,
            &tiny::serialize(&c, &tiny::weights(&c).unwrap()).unwrap(),
        )
        .unwrap();
        let model = Transformer::from_weights(&d, c, &w).unwrap();
        let (_, mut cache) = model.forward_prefill(&d, &[1, 2]).unwrap();
        let prefix = cache.active(0).unwrap().unwrap().0.to_f32();
        let mut trace = Trace::new();
        d.fail_completion.set(true);
        assert!(
            model
                .forward(&d, &[3], &mut cache, Some(&mut trace))
                .is_err()
        );
        assert_eq!(cache.len().unwrap(), 2);
        assert_eq!(cache.active(0).unwrap().unwrap().0.to_f32(), prefix);
        assert!(
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| trace["logits"].to_f32()))
                .is_err()
        );
        let retry = model.forward_decode(&d, 3, &mut cache).unwrap();
        let full = model.forward_prefill(&d, &[1, 2, 3]).unwrap().0;
        crate::reference::check(
            &retry.to_f32(),
            &full.view(64, [1, 32]).unwrap().to_f32(),
            3e-5,
            3e-5,
        )
        .unwrap();
    }
    #[test]
    fn bounded_epochs_preserve_live_dependency_chain() {
        use crate::{DType, Tensor};
        let d = MetalDevice::new().unwrap();
        let mut x = Tensor::from_f32(
            &d,
            [2 * 1024 * 1024],
            DType::F32,
            &vec![0.01; 2 * 1024 * 1024],
        )
        .unwrap();
        let before = d.counters();
        d.reset_transient_peak();
        let execution = d.execution().unwrap();
        for _ in 0..40 {
            x = d.add(&x, &x).unwrap().tensor;
        }
        execution.finish().unwrap();
        let after = d.counters();
        assert!(after.command_buffers - before.command_buffers >= 2);
        assert!(after.transient_peak_bytes <= 256 * 1024 * 1024);
        assert!(x.to_f32().iter().all(|&v| v == 0.01f32 * 2f32.powi(40)));
    }
}
