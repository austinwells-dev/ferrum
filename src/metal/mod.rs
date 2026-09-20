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
pub struct MetalBuffer {
    raw: Retained<ProtocolObject<dyn MTLBuffer>>,
    len: usize,
    owner: Rc<()>,
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
}

pub struct MetalDevice {
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
            counters: Cell::new(Counters::default()),
            name: raw.name().to_string(),
            raw,
            queue,
            owner: Rc::new(()),
            libraries: RefCell::new(HashMap::new()),
            pipelines: RefCell::new(HashMap::new()),
            builtins: RefCell::new(HashMap::new()),
        })
    }
    pub fn counters(&self) -> Counters {
        self.counters.get()
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
        })
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
        buffers: &[&MetalBuffer],
        params: &[u32; 9],
        grid: [usize; 2],
        tiled: bool,
    ) -> Result<DispatchTiming> {
        for b in buffers {
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
            let command = self
                .queue
                .commandBuffer()
                .ok_or_else(|| Error::Dispatch("command buffer creation failed".into()))?;
            let encoder = command
                .computeCommandEncoder()
                .ok_or_else(|| Error::Dispatch("compute encoder creation failed".into()))?;
            encoder.setComputePipelineState(&p.raw);
            // SAFETY: crate-private callers validate dimensions, dtypes and lengths against the
            // embedded kernel ABI. Buffers stay alive through completion; params is copied by Metal.
            unsafe {
                for (i, b) in buffers.iter().enumerate() {
                    encoder.setBuffer_offset_atIndex(Some(&b.raw), 0, i);
                }
                encoder.setBytes_length_atIndex(
                    NonNull::from(params).cast(),
                    size_of_val(params),
                    3,
                );
            }
            if tiled {
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
            command.commit();
            let mut counters = self.counters.get();
            counters.dispatches += 1;
            self.counters.set(counters);
            let submission = start.elapsed();
            command.waitUntilCompleted();
            let synchronized = start.elapsed();
            if command.status() != MTLCommandBufferStatus::Completed {
                return Err(Error::Synchronization(
                    command
                        .error()
                        .map(|e| e.to_string())
                        .unwrap_or_else(|| format!("status {:?}", command.status())),
                ));
            }
            let seconds = command.GPUEndTime() - command.GPUStartTime();
            Ok(DispatchTiming {
                submission,
                synchronized,
                gpu: (seconds.is_finite() && seconds > 0.)
                    .then(|| Duration::from_secs_f64(seconds)),
                dispatches: 1,
            })
        })
    }
}
