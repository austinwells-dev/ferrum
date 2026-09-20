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
type RetiredBuffer = (
    Retained<ProtocolObject<dyn MTLBuffer>>,
    Option<Rc<Cell<bool>>>,
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
    fn acquire(&mut self, bytes: usize) {
        self.live += bytes;
        self.peak = self.peak.max(self.live);
        self.high_water = self.high_water.max(self.capacity);
    }
}
pub struct MetalBuffer {
    raw: Retained<ProtocolObject<dyn MTLBuffer>>,
    len: usize,
    owner: Rc<()>,
    ready: RefCell<Option<Rc<Cell<bool>>>>,
    arena: Option<Rc<RefCell<Arena>>>,
}
impl Drop for MetalBuffer {
    fn drop(&mut self) {
        if let Some(arena) = &self.arena {
            let mut arena = arena.borrow_mut();
            let bytes = self.raw.length();
            arena.live -= bytes;
            // A pool entry exists only after the final Tensor owner is gone.
            // A pending epoch keeps it ineligible until successful completion.
            if arena.capacity <= 64 * 1024 * 1024 {
                arena
                    .free
                    .entry(bytes)
                    .or_default()
                    .push((self.raw.clone(), self.ready.borrow().clone()));
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
    pub(crate) fn with_bytes<T>(&self, f: impl FnOnce(&[u8]) -> T) -> T {
        assert!(
            self.ready.borrow().as_ref().is_none_or(|r| r.get()),
            "GPU output is not successfully completed"
        );
        // SAFETY: all bytes are initialized; dispatch waits before returning on both success
        // and GPU error. Rc prevents cross-thread use; only read-only tensors are published.
        let bytes = unsafe {
            std::slice::from_raw_parts(self.raw.contents().as_ptr().cast::<u8>(), self.len)
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
#[derive(Clone, Copy, Debug, Default)]
pub struct Counters {
    pub allocations: usize,
    pub allocated_bytes: usize,
    pub dispatches: usize,
    pub command_buffers: usize,
    pub completion_waits: usize,
    pub encode: Duration,
    pub wait: Duration,
    pub gpu: Duration,
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
    ready: Rc<Cell<bool>>,
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
pub struct MetalDevice {
    arena: Rc<RefCell<Arena>>,
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
            arena: Rc::new(RefCell::new(Arena::default())),
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
        if submission.command.status() != MTLCommandBufferStatus::Completed {
            return Err(Error::Synchronization(
                submission
                    .command
                    .error()
                    .map(|e| e.to_string())
                    .unwrap_or_else(|| format!("status {:?}", submission.command.status())),
            ));
        }
        submission.ready.set(true);
        // submission.resources drops only after the wait and status check.
        Ok(())
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
        self.counters.set(counters);
        Ok(MetalBuffer {
            raw,
            len: bytes,
            owner: self.owner.clone(),
            ready: RefCell::new(None),
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
        let mut arena = self.arena.borrow_mut();
        let bin = arena.free.entry(length).or_default();
        let available = bin
            .iter()
            .rposition(|(_, ready)| ready.as_ref().is_none_or(|r| r.get()));
        let mut buffer = if let Some(i) = available {
            let (raw, _) = bin.swap_remove(i);
            let mut counters = self.counters.get();
            counters.reused_bytes += length;
            self.counters.set(counters);
            MetalBuffer {
                raw,
                len: bytes,
                owner: self.owner.clone(),
                ready: RefCell::new(None),
                arena: None,
            }
        } else {
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
        autoreleasepool(|_| {
            let key = (source.to_owned(), name.to_owned());
            if let Some(p) = self.pipelines.borrow().get(&key) {
                return Ok(p.clone());
            }
            let mut libraries = self.libraries.borrow_mut();
            if !libraries.contains_key(source) {
                let options = MTLCompileOptions::new();
                options.setMathMode(MTLMathMode::Safe);
                options.setMathFloatingPointFunctions(MTLMathFloatingPointFunctions::Precise);
                let lib = self
                    .raw
                    .newLibraryWithSource_options_error(&NSString::from_str(source), Some(&options))
                    .map_err(|e| Error::Compilation(e.to_string()))?;
                libraries.insert(source.to_owned(), lib);
            }
            let lib = libraries
                .get(source)
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
        let p = self.compile_kernel(include_str!("shaders/ops.metal"), name)?;
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
        buffers: &[(&MetalBuffer, usize)],
        params: &[u32; 9],
        grid: [usize; 2],
        tiled: bool,
    ) -> Result<DispatchTiming> {
        for (b, offset) in buffers {
            if *offset > b.len_bytes() {
                return Err(Error::Range("binding offset exceeds allocation".into()));
            }
            if !self.owns(b) {
                return Err(Error::DeviceMismatch);
            }
        }
        if grid.contains(&0) {
            return Ok(DispatchTiming::default());
        }
        debug_assert_eq!(buffers.len(), 3);
        debug_assert!(params[4] <= 2);
        let p = self.builtin(name)?;
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
                    ready: Rc::new(Cell::new(false)),
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
                for (i, (b, offset)) in buffers.iter().enumerate() {
                    encoder.setBuffer_offset_atIndex(Some(&b.raw), *offset, i);
                }
                encoder.setBytes_length_atIndex(
                    NonNull::from(params).cast(),
                    size_of_val(params),
                    3,
                );
            }
            if matches!(name, "rmsnorm" | "softmax") {
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
            } else if name == "gemv" {
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
                .extend(buffers.iter().map(|(b, _)| b.raw.clone()));
            *buffers[2].0.ready.borrow_mut() = Some(submission.ready.clone());
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
                synchronized,
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
