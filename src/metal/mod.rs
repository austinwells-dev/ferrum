//! The only module allowed to interact with Objective-C or raw memory.
#[cfg(feature = "legacy-engine")]
mod legacy;
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
    #[cfg(feature = "legacy-engine")]
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
    #[cfg(feature = "legacy-engine")]
    fn acquire(&mut self, bytes: usize) {
        self.live += bytes;
        self.peak = self.peak.max(self.live);
        self.high_water = self.high_water.max(self.capacity);
    }
}
type WriteRange = (usize, usize, Rc<Cell<Completion>>);
/// (offset, rows, head_dim, theta bits) and its cos/sin table.
pub(crate) type RopeTableEntry = ((usize, usize, usize, u32), crate::Tensor);
pub struct MetalBuffer {
    raw: Retained<ProtocolObject<dyn MTLBuffer>>,
    len: usize,
    owner: Rc<()>,
    ready: RefCell<Option<Rc<Cell<Completion>>>>,
    writes: RefCell<Vec<WriteRange>>,
    arena: Option<Rc<RefCell<Arena>>>,
    resident: Option<Retained<ProtocolObject<dyn MTLResidencySet>>>,
}
impl Drop for MetalBuffer {
    fn drop(&mut self) {
        if let Some(set) = &self.resident {
            set.removeAllocation(ProtocolObject::from_ref(&*self.raw));
            set.commit();
        }
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
pub type Profile = std::collections::BTreeMap<String, ProfileEntry>;
/// An encoding epoch owns every referenced Metal resource through completion.
/// No tensor carrying this epoch may be mapped until successful completion.
/// Default for `MetalDevice::early_commit`: enough work to cover encoding the
/// rest of a decode step, small enough that the GPU starts almost at once.
const EARLY_COMMIT: usize = 32;
struct Submission {
    command: Retained<ProtocolObject<dyn MTLCommandBuffer>>,
    // With a shared encoder, one compute encoder spans the epoch (serial, or
    // concurrent with explicit barriers); otherwise each kernel gets its own.
    encoder: Option<Retained<ProtocolObject<dyn MTLComputeCommandEncoder>>>,
    // Concurrent encoders need explicit ordering: byte ranges touched since the
    // last buffer barrier. Retained resources keep addresses unique per epoch.
    reads: Vec<(usize, usize, usize)>,
    writes: Vec<(usize, usize, usize)>,
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
        self.device.execution_shared_encoder.set(false);
        self.device.rope_table_cache.borrow_mut().take();
    }
}
/// Background residency keep-alive. It holds only the (Send + Sync) command
/// queue and a no-op pipeline: while real work was submitted within `window`,
/// it executes one empty dispatch every `interval`, which re-applies the
/// queue's residency set and keeps weight residency through short idle gaps.
struct KeepAlive {
    stop: std::sync::Arc<std::sync::atomic::AtomicBool>,
    last_use: std::sync::Arc<std::sync::atomic::AtomicU64>,
    started: Instant,
    thread: Option<std::thread::JoinHandle<()>>,
}
impl KeepAlive {
    fn start(
        queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
        pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
        interval: Duration,
        window: Duration,
    ) -> Self {
        use std::sync::{
            Arc,
            atomic::{AtomicBool, AtomicU64, Ordering},
        };
        let stop = Arc::new(AtomicBool::new(false));
        let last_use = Arc::new(AtomicU64::new(0));
        let started = Instant::now();
        let thread = {
            let (stop, last_use) = (stop.clone(), last_use.clone());
            std::thread::spawn(move || {
                while !stop.load(Ordering::Relaxed) {
                    std::thread::park_timeout(interval);
                    if stop.load(Ordering::Relaxed) {
                        break;
                    }
                    let now = started.elapsed().as_millis() as u64;
                    let last = last_use.load(Ordering::Relaxed);
                    if last != 0 && now.saturating_sub(last) < window.as_millis() as u64 {
                        autoreleasepool(|_| {
                            if let Some(command) = queue.commandBuffer()
                                && let Some(encoder) = command.computeCommandEncoder()
                            {
                                encoder.setComputePipelineState(&pipeline);
                                let one = MTLSize {
                                    width: 1,
                                    height: 1,
                                    depth: 1,
                                };
                                encoder.dispatchThreadgroups_threadsPerThreadgroup(one, one);
                                encoder.endEncoding();
                                command.commit();
                            }
                        });
                    }
                }
            })
        };
        Self {
            stop,
            last_use,
            started,
            thread: Some(thread),
        }
    }
    fn touch(&self) {
        let now = self.started.elapsed().as_millis().max(1) as u64;
        self.last_use
            .store(now, std::sync::atomic::Ordering::Relaxed);
    }
}
impl Drop for KeepAlive {
    fn drop(&mut self) {
        self.stop.store(true, std::sync::atomic::Ordering::Relaxed);
        if let Some(thread) = self.thread.take() {
            thread.thread().unpark();
            let _ = thread.join();
        }
    }
}
#[cfg(feature = "legacy-engine")]
struct ProfileStageScope<'a> {
    stage: &'a Cell<Option<&'static str>>,
    previous: Option<&'static str>,
}
#[cfg(feature = "legacy-engine")]
impl Drop for ProfileStageScope<'_> {
    fn drop(&mut self) {
        self.stage.set(self.previous);
    }
}
pub struct MetalDevice {
    profile_stage: Cell<Option<&'static str>>,
    #[cfg(test)]
    fail_after: Cell<Option<usize>>,
    #[cfg(test)]
    fail_completion: Cell<bool>,
    arena: Rc<RefCell<Arena>>,
    native_matmul: Cell<bool>,
    resident_weights: Cell<bool>,
    keep_alive: RefCell<Option<KeepAlive>>,
    residency: Option<Retained<ProtocolObject<dyn MTLResidencySet>>>,
    residency_dirty: Cell<bool>,
    rope_table_cache: RefCell<Option<RopeTableEntry>>,
    shared_encoder: Cell<bool>,
    execution_shared_encoder: Cell<bool>,
    concurrent_dispatch: Cell<bool>,
    #[cfg(feature = "legacy-engine")]
    tuning: legacy::Tuning,
    batching: Cell<bool>,
    batch_limit: Cell<usize>,
    /// Dispatches after which an epoch's first command buffer is committed
    /// without waiting, so the GPU starts while the rest is encoded. 0 is off.
    early_commit: Cell<usize>,
    pending: RefCell<Option<Submission>>,
    /// Committed, not yet awaited parts of the current epoch (in queue order).
    in_flight: RefCell<Vec<Submission>>,
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
    /// Default device: weights stay in a queue residency set and a keep-alive
    /// heartbeat (500 ms, for 3 minutes after the last submission) keeps that
    /// residency applied across short idle gaps. See Phase 7A Experiment 64.
    pub fn new() -> Result<Self> {
        let device = Self::without_keep_alive()?;
        device.set_residency_keep_alive(Some((
            Duration::from_millis(500),
            Duration::from_secs(180),
        )))?;
        Ok(device)
    }
    /// A device with no background keep-alive thread.
    pub fn without_keep_alive() -> Result<Self> {
        let raw = MTLCreateSystemDefaultDevice()
            .ok_or_else(|| Error::Initialization("no default GPU".into()))?;
        if !raw.hasUnifiedMemory() {
            return Err(Error::Initialization("unified memory GPU required".into()));
        }
        let queue = raw
            .newCommandQueue()
            .ok_or_else(|| Error::Initialization("command queue creation failed".into()))?;
        // Model weights join one residency set attached to the queue and kept
        // resident, so a submission after an idle period does not re-wire
        // gigabytes of weight allocations before it can start.
        let residency = {
            let descriptor = MTLResidencySetDescriptor::new();
            // SAFETY: a capacity hint has no memory-safety preconditions.
            unsafe { descriptor.setInitialCapacity(1024) };
            raw.newResidencySetWithDescriptor_error(&descriptor).ok()
        };
        if let Some(set) = &residency {
            queue.addResidencySet(set);
        }
        Ok(Self {
            profile_stage: Cell::new(None),
            #[cfg(test)]
            fail_after: Cell::new(None),
            #[cfg(test)]
            fail_completion: Cell::new(false),
            arena: Rc::new(RefCell::new(Arena::default())),
            native_matmul: Cell::new(raw.supportsFamily(MTLGPUFamily::Apple7)),
            resident_weights: Cell::new(true),
            keep_alive: RefCell::new(None),
            residency,
            residency_dirty: Cell::new(false),
            rope_table_cache: RefCell::new(None),
            shared_encoder: Cell::new(true),
            execution_shared_encoder: Cell::new(false),
            concurrent_dispatch: Cell::new(true),
            #[cfg(feature = "legacy-engine")]
            tuning: legacy::Tuning::default(),
            batching: Cell::new(false),
            batch_limit: Cell::new(1024),
            early_commit: Cell::new(
                std::env::var("FERRUM_EARLY_COMMIT")
                    .ok()
                    .and_then(|v| v.parse().ok())
                    .unwrap_or(EARLY_COMMIT),
            ),
            pending: RefCell::new(None),
            in_flight: RefCell::new(Vec::new()),
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
    /// Keep weight allocations made after this call in the resident queue set.
    pub fn set_resident_weights(&self, enabled: bool) {
        self.resident_weights.set(enabled);
    }
    pub fn resident_weights(&self) -> bool {
        self.resident_weights.get() && self.residency.is_some()
    }
    /// Keep weight residency warm between requests: an empty command buffer is
    /// committed every `interval` while real work ran within `window`. `None`
    /// stops the keep-alive thread.
    pub fn set_residency_keep_alive(&self, schedule: Option<(Duration, Duration)>) -> Result<()> {
        let next = match schedule {
            Some((interval, window)) => {
                let pipeline = self.builtin("keep_alive")?.raw.clone();
                let keep_alive = KeepAlive::start(self.queue.clone(), pipeline, interval, window);
                keep_alive.touch();
                Some(keep_alive)
            }
            None => None,
        };
        *self.keep_alive.borrow_mut() = next;
        Ok(())
    }
    /// Zeroed shared storage for model weights, added to the resident set.
    pub(crate) fn allocate_resident(&self, bytes: usize) -> Result<MetalBuffer> {
        let mut buffer = self.allocate(bytes)?;
        if let (true, Some(set)) = (self.resident_weights.get(), &self.residency) {
            set.addAllocation(ProtocolObject::from_ref(&*buffer.raw));
            self.residency_dirty.set(true);
            buffer.resident = Some(set.clone());
        }
        Ok(buffer)
    }
    /// Encode each command buffer's kernels through one compute encoder when the
    /// execution scope allows it (see `execution_with_shared_encoder`).
    pub fn set_shared_encoder(&self, enabled: bool) -> Result<()> {
        if self.batching.get() {
            return Err(Error::Parameter(
                "cannot change compute encoder policy during execution".into(),
            ));
        }
        self.shared_encoder.set(enabled);
        Ok(())
    }
    pub fn shared_encoder(&self) -> bool {
        self.shared_encoder.get()
    }
    /// Let independent kernels in the shared encoder overlap; dependent
    /// buffer ranges are ordered with memory barriers.
    pub fn set_concurrent_dispatch(&self, enabled: bool) -> Result<()> {
        if self.batching.get() {
            return Err(Error::Parameter(
                "cannot change dispatch concurrency during execution".into(),
            ));
        }
        self.concurrent_dispatch.set(enabled);
        Ok(())
    }
    pub fn concurrent_dispatch(&self) -> bool {
        self.concurrent_dispatch.get()
    }
    pub(crate) fn mpp_projection(&self) -> bool {
        self.native_matmul.get()
            && self.raw.supportsFamily(MTLGPUFamily::Apple10)
            && self.raw.supportsFamily(MTLGPUFamily::Metal4)
    }
    pub fn capabilities(&self) -> Result<serde_json::Value> {
        let pipeline = self.builtin("gemv")?;
        Ok(
            serde_json::json!({"apple_families":self.apple_families(),"metal4":self.raw.supportsFamily(MTLGPUFamily::Metal4),"msl_requested":"3.1","mpp_msl_requested":"4.0","mpp_available":self.mpp_projection(),"mpp_compiled":self.mpp_projection() && self.builtin("project_mpp").is_ok(),"simdgroup_matrix":self.raw.supportsFamily(MTLGPUFamily::Apple7),"native_bfloat_compiled":self.builtin("project_bf16").is_ok(),"thread_execution_width":pipeline.raw.threadExecutionWidth(),"pipeline_max_threads":pipeline.raw.maxTotalThreadsPerThreadgroup(),"device_max_threads":self.raw.maxThreadsPerThreadgroup().width,"recommended_working_set":self.recommended_max_working_set(),"storage":"shared","hazards":"tracked"}),
        )
    }
    /// Maximum kernels per completion boundary in model execution. One is the control.
    pub fn set_batch_limit(&self, limit: usize) -> Result<()> {
        if limit == 0 || self.batching.get() {
            return Err(Error::Parameter("invalid active batch limit".into()));
        }
        self.batch_limit.set(limit);
        Ok(())
    }
    #[cfg(all(test, feature = "legacy-engine"))]
    pub(crate) fn execution(&self) -> Result<Execution<'_>> {
        self.execution_with_shared_encoder(true)
    }
    /// Begin an execution scope. `shared_encoder` lets this scope use the
    /// device's shared/concurrent encoder policy; false keeps one encoder per kernel.
    pub(crate) fn execution_with_shared_encoder(
        &self,
        shared_encoder: bool,
    ) -> Result<Execution<'_>> {
        if self.batching.replace(true) {
            return Err(Error::Dispatch("nested execution".into()));
        }
        self.execution_shared_encoder.set(shared_encoder);
        Ok(Execution { device: self })
    }
    /// End the open submission's encoder and commit it without waiting.
    fn commit_pending(&self) -> Option<Submission> {
        let submission = self.pending.borrow_mut().take()?;
        if let Some(encoder) = &submission.encoder {
            encoder.endEncoding();
        }
        if let Some(keep_alive) = self.keep_alive.borrow().as_ref() {
            keep_alive.touch();
        }
        submission.command.commit();
        Some(submission)
    }
    fn flush(&self) -> Result<()> {
        let mut parts = std::mem::take(&mut *self.in_flight.borrow_mut());
        let start = Instant::now();
        parts.extend(self.commit_pending());
        let Some(last) = parts.last() else {
            return Ok(());
        };
        // One queue runs the parts in order; they share one completion cell.
        last.command.waitUntilCompleted();
        let mut counters = self.counters.get();
        counters.command_buffers += parts.len();
        counters.completion_waits += 1;
        counters.wait += start.elapsed();
        let mut failure = None;
        for submission in &parts {
            submission.command.waitUntilCompleted();
            let seconds = submission.command.GPUEndTime() - submission.command.GPUStartTime();
            if seconds.is_finite() && seconds > 0. {
                counters.gpu += Duration::from_secs_f64(seconds);
            }
            if failure.is_none() && submission.command.status() != MTLCommandBufferStatus::Completed
            {
                failure = Some(
                    submission
                        .command
                        .error()
                        .map(|e| e.to_string())
                        .unwrap_or_else(|| format!("status {:?}", submission.command.status())),
                );
            }
        }
        self.counters.set(counters);
        #[cfg(test)]
        if failure.is_none() && self.fail_completion.replace(false) {
            failure = Some("injected completion failure".into());
        }
        if let Some(error) = failure {
            last.ready.set(Completion::Failed);
            self.arena.borrow_mut().complete();
            return Err(Error::Synchronization(error));
        }
        last.ready.set(Completion::Completed);
        self.arena.borrow_mut().complete();
        // The parts' resources drop only after the wait and status check.
        Ok(())
    }
    /// Complete the current encoding epoch before safe host readback. Model
    /// code uses this at CPU routing boundaries inside an active transaction.
    pub(crate) fn synchronize(&self) -> Result<()> {
        self.flush()
    }
    pub fn set_profiling(&self, enabled: bool) {
        self.profiling.set(enabled);
        self.profile.borrow_mut().clear();
    }
    #[cfg(all(test, feature = "legacy-engine"))]
    pub(crate) fn inject_dispatch_failure_after(&self, dispatch_count: Option<usize>) {
        self.fail_after.set(dispatch_count);
    }
    #[cfg(feature = "legacy-engine")]
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
        let name = if let Some(stage) = self.profile_stage.get() {
            format!("{stage}.{name}")
        } else {
            name.to_owned()
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
    /// Bytes currently allocated on this device (Metal's own accounting).
    pub fn allocated_bytes(&self) -> usize {
        self.raw.currentAllocatedSize()
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
            resident: None,
        })
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
    /// Bounds, ownership, failed-producer and aliasing checks shared by every
    /// dispatch path. Outputs follow the `input_count` inputs in `buffers`.
    fn check_bindings(
        &self,
        name: &str,
        buffers: &[(&MetalBuffer, usize, usize)],
        input_count: usize,
    ) -> Result<()> {
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
        for (buffer, offset, length) in &buffers[..input_count] {
            if buffer.writes.borrow().iter().any(|(start, end, state)| {
                *offset < *end && offset + length > *start && state.get() == Completion::Failed
            }) {
                return Err(Error::Synchronization(
                    "input range belongs to a failed submission".into(),
                ));
            }
        }
        for (output_index, (output, output_offset, output_length)) in
            buffers[input_count..].iter().enumerate()
        {
            for (input, input_offset, input_length) in &buffers[..input_count] {
                if std::ptr::eq(*input, *output)
                    && *input_offset < *output_offset + *output_length
                    && *output_offset < *input_offset + *input_length
                {
                    return Err(Error::Dispatch(format!("input/output alias for {name}")));
                }
            }
            for (other, other_offset, other_length) in
                buffers[input_count..input_count + output_index].iter()
            {
                if std::ptr::eq(*other, *output)
                    && *other_offset < *output_offset + *output_length
                    && *output_offset < *other_offset + *other_length
                {
                    return Err(Error::Dispatch(format!("output/output alias for {name}")));
                }
            }
        }
        Ok(())
    }
    fn hybrid_pipeline(&self, name: &'static str) -> Result<Rc<ComputePipeline>> {
        if let Some(p) = self.builtins.borrow().get(name) {
            return Ok(p.clone());
        }
        // The MLX kernels (Apache-2.0, adapted from Kapitsa) and the image
        // transformer's kernels build on hybrid.metal's types.
        const SOURCE: &str = concat!(
            include_str!("shaders/hybrid.metal"),
            include_str!("shaders/hybrid_mlx.metal"),
            include_str!("shaders/diffusion.metal"),
            include_str!("shaders/diffusion_vae.metal")
        );
        let p = self.compile_kernel_version(SOURCE, name, MTLLanguageVersion::Version4_0)?;
        self.builtins.borrow_mut().insert(name, p.clone());
        Ok(p)
    }
    /// Dispatch a kernel from the hybrid-model library with explicit
    /// threadgroup geometry. `inputs` are read-only; `outputs` are written.
    /// Parameters are one little-endian block bound after the buffers.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn dispatch_hybrid(
        &self,
        name: &'static str,
        inputs: &[(&MetalBuffer, usize, usize)],
        outputs: &[(&MetalBuffer, usize, usize)],
        params: &[u8],
        groups: [usize; 3],
        threads: [usize; 3],
        threadgroup_bytes: usize,
    ) -> Result<()> {
        let buffers: smallvec::SmallVec<[(&MetalBuffer, usize, usize); 12]> =
            inputs.iter().chain(outputs).copied().collect();
        self.check_bindings(name, &buffers, inputs.len())?;
        if groups.contains(&0) {
            return Ok(());
        }
        let p = self.hybrid_pipeline(name)?;
        let per_group: usize = threads.iter().product();
        if p.raw.threadExecutionWidth() != 32
            || per_group == 0
            || p.raw.maxTotalThreadsPerThreadgroup() < per_group
            || threadgroup_bytes > 32 * 1024
            || !threadgroup_bytes.is_multiple_of(16)
        {
            return Err(Error::Dispatch(format!(
                "{name}: unsupported threadgroup configuration"
            )));
        }
        let start = self.profiling.get().then(Instant::now);
        let timing = self.encode_dispatch(&p, &buffers, inputs.len(), params, |encoder| {
            // SAFETY: the length is bounded by the device's 32 KiB threadgroup limit.
            if threadgroup_bytes > 0 {
                unsafe { encoder.setThreadgroupMemoryLength_atIndex(threadgroup_bytes, 0) };
            }
            let size = |v: [usize; 3]| MTLSize {
                width: v[0],
                height: v[1],
                depth: v[2],
            };
            encoder.dispatchThreadgroups_threadsPerThreadgroup(size(groups), size(threads));
            Ok(())
        })?;
        if let Some(start) = start {
            self.record_profile(name, start.elapsed(), Duration::ZERO, 0, &timing);
        }
        Ok(())
    }
    pub(crate) fn owns(&self, buffer: &MetalBuffer) -> bool {
        Rc::ptr_eq(&self.owner, &buffer.owner)
    }
    /// Shared encoding path: open or reuse the epoch's command buffer and
    /// encoder, order hazards, bind buffers and parameters, let `geometry`
    /// dispatch, then record completion ownership of every output range.
    fn encode_dispatch(
        &self,
        p: &ComputePipeline,
        buffers: &[(&MetalBuffer, usize, usize)],
        input_count: usize,
        params: &[u8],
        geometry: impl FnOnce(&ProtocolObject<dyn MTLComputeCommandEncoder>) -> Result<()>,
    ) -> Result<DispatchTiming> {
        autoreleasepool(|_| {
            let start = Instant::now();
            if self.pending.borrow().is_none() {
                if self.residency_dirty.replace(false)
                    && let Some(set) = &self.residency
                {
                    set.commit();
                    set.requestResidency();
                }
                let command = self
                    .queue
                    .commandBuffer()
                    .ok_or_else(|| Error::Dispatch("command buffer creation failed".into()))?;
                // A continuation of a partly committed epoch shares its state.
                let ready = self.in_flight.borrow().last().map_or_else(
                    || Rc::new(Cell::new(Completion::Encoding)),
                    |s| s.ready.clone(),
                );
                *self.pending.borrow_mut() = Some(Submission {
                    command,
                    encoder: None,
                    reads: Vec::new(),
                    writes: Vec::new(),
                    resources: Vec::new(),
                    ready,
                    dispatches: 0,
                });
            }
            let mut pending = self.pending.borrow_mut();
            let submission = pending
                .as_mut()
                .ok_or_else(|| Error::Dispatch("missing submission".into()))?;
            let shared = self.shared_encoder.get() && self.execution_shared_encoder.get();
            let concurrent = shared && self.concurrent_dispatch.get();
            if submission.encoder.is_none() {
                let encoder = if concurrent {
                    submission
                        .command
                        .computeCommandEncoderWithDispatchType(MTLDispatchType::Concurrent)
                } else {
                    submission.command.computeCommandEncoder()
                };
                submission.encoder =
                    Some(encoder.ok_or_else(|| {
                        Error::Dispatch("compute encoder creation failed".into())
                    })?);
            }
            if concurrent {
                let range = |(b, offset, length): &(&MetalBuffer, usize, usize)| {
                    (
                        Retained::as_ptr(&b.raw) as *const () as usize,
                        *offset,
                        offset + length,
                    )
                };
                let overlaps =
                    |set: &[(usize, usize, usize)], (id, start, end): (usize, usize, usize)| {
                        set.iter()
                            .any(|&(other, s, e)| other == id && start < e && s < end)
                    };
                let hazard = buffers[..input_count]
                    .iter()
                    .any(|b| overlaps(&submission.writes, range(b)))
                    || buffers[input_count..].iter().any(|b| {
                        overlaps(&submission.writes, range(b))
                            || overlaps(&submission.reads, range(b))
                    });
                if hazard {
                    if let Some(encoder) = &submission.encoder {
                        encoder.memoryBarrierWithScope(MTLBarrierScope::Buffers);
                    }
                    submission.reads.clear();
                    submission.writes.clear();
                }
                submission
                    .reads
                    .extend(buffers[..input_count].iter().map(range));
                submission
                    .writes
                    .extend(buffers[input_count..].iter().map(range));
            }
            let encoder = submission
                .encoder
                .as_ref()
                .ok_or_else(|| Error::Dispatch("missing compute encoder".into()))?;
            encoder.setComputePipelineState(&p.raw);
            // SAFETY: crate-private callers validate dimensions, dtypes and lengths against the
            // embedded kernel ABI. Buffers stay alive through completion; params is copied by Metal.
            unsafe {
                for (i, (b, offset, _)) in buffers.iter().enumerate() {
                    encoder.setBuffer_offset_atIndex(Some(&b.raw), *offset, i);
                }
                if !params.is_empty() {
                    encoder.setBytes_length_atIndex(
                        NonNull::new(params.as_ptr() as *mut u8)
                            .ok_or_else(|| Error::Dispatch("null parameter block".into()))?
                            .cast(),
                        params.len(),
                        buffers.len(),
                    );
                }
            }
            geometry(encoder)?;
            if !shared {
                encoder.endEncoding();
                submission.encoder = None;
            }
            submission
                .resources
                .extend(buffers.iter().map(|(b, _, _)| b.raw.clone()));
            for (output, offset, length) in &buffers[input_count..] {
                *output.ready.borrow_mut() = Some(submission.ready.clone());
                let mut writes = output.writes.borrow_mut();
                writes.retain(|(_, _, state)| state.get() != Completion::Completed);
                writes.push((*offset, offset + length, submission.ready.clone()));
            }
            submission.dispatches += 1;
            let flush = !self.batching.get() || submission.dispatches >= self.batch_limit.get();
            let early = !flush
                && submission.dispatches == self.early_commit.get()
                && self.in_flight.borrow().is_empty();
            let mut counters = self.counters.get();
            counters.dispatches += 1;
            let submission_time = start.elapsed();
            counters.encode += submission_time;
            self.counters.set(counters);
            drop(pending);
            let before_gpu = counters.gpu;
            if flush {
                self.flush()?;
            } else if early && let Some(part) = self.commit_pending() {
                self.in_flight.borrow_mut().push(part);
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

#[cfg(all(test, feature = "legacy-engine"))]
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

#[cfg(all(test, feature = "legacy-engine"))]
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

#[cfg(all(test, feature = "legacy-engine"))]
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

#[cfg(all(test, feature = "legacy-engine"))]
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
        // Keep every intermediate alive so the live budget, not recycling,
        // bounds the epoch.
        let mut alive = Vec::new();
        for _ in 0..40 {
            let next = d.add(&x, &x).unwrap().tensor;
            alive.push(std::mem::replace(&mut x, next));
        }
        execution.finish().unwrap();
        let after = d.counters();
        assert!(after.command_buffers - before.command_buffers >= 2);
        assert!(x.to_f32().iter().all(|&v| v == 0.01f32 * 2f32.powi(40)));
        drop(alive);
    }
    #[test]
    fn epoch_reuse_orders_rewrites_after_pending_reads() {
        use crate::{DType, Tensor};
        let d = MetalDevice::new().unwrap();
        // Small enough that the whole chain stays in one epoch.
        let n = 1 << 16;
        let a = Tensor::from_f32(&d, [n], DType::F32, &vec![1.0; n]).unwrap();
        // Early commit 0 keeps one command buffer; 32 splits the chain so
        // recycled storage is rewritten across buffers.
        for (concurrent, early) in [(false, 0), (true, 0), (false, 32), (true, 32)] {
            d.set_concurrent_dispatch(concurrent).unwrap();
            d.early_commit.set(early);
            let before = d.counters();
            let execution = d.execution().unwrap();
            // Each step drops the previous tensor right after its last read is
            // encoded, so the next output recycles storage still being read.
            let mut x = d.add(&a, &a).unwrap().tensor;
            let mut sums = Vec::new();
            for _ in 0..64 {
                let y = d.add(&x, &a).unwrap().tensor;
                sums.push(d.add(&y, &y).unwrap().tensor);
                x = y;
            }
            execution.finish().unwrap();
            let after = d.counters();
            assert!(
                after.reused_bytes > before.reused_bytes,
                "concurrent={concurrent}"
            );
            assert_eq!(
                after.command_buffers - before.command_buffers,
                if early == 0 { 1 } else { 2 }
            );
            assert!(x.to_f32().iter().all(|&v| v == 66.0));
            for (i, s) in sums.iter().enumerate() {
                let expected = 2.0 * (3.0 + i as f32);
                assert!(s.to_f32().iter().all(|&v| v == expected), "step {i}");
            }
        }
        d.set_concurrent_dispatch(true).unwrap();
        d.early_commit.set(EARLY_COMMIT);
    }
}

#[cfg(all(test, feature = "legacy-engine"))]
mod custom_output_tests {
    use super::*;
    use crate::{DType, Tensor};

    #[test]
    fn expert_projection_tracks_its_real_output_range() {
        let d = MetalDevice::new().unwrap();
        let input = Tensor::from_f32(&d, [1, 2], DType::F32, &[5., 7.]).unwrap();
        let metadata = Tensor::from_f32(&d, [1, 3], DType::F32, &[0., 0., 1.]).unwrap();
        let weights = Tensor::from_f32(&d, [1, 1, 2], DType::F32, &[2., 3.]).unwrap();

        let execution = d.execution().unwrap();
        let output = d.expert_project(&input, &metadata, &weights, 2, 1).unwrap();
        execution.finish().unwrap();
        assert_eq!(output.to_f32(), vec![31.]);

        let execution = d.execution().unwrap();
        let failed_output = d.expert_project(&input, &metadata, &weights, 2, 1).unwrap();
        d.fail_completion.set(true);
        assert!(execution.finish().is_err());
        assert!(
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| { failed_output.to_f32() }))
                .is_err()
        );
    }
}
