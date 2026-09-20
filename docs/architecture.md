# Phase 1 architecture

## Modules and dependency direction

```text
CLI / tests / benchmarks
        │
        ├── reference (CPU oracles only)
        │
        ▼
Tensor + operation API ──► MetalDevice ──► objc2 / Apple's Metal
                                │
                                └── embedded ops.metal
```

`src/lib.rs` denies unsafe Rust except in `metal`; tensor, operations, and CPU reference modules additionally forbid it. Objective-C object types are private to the backend. There is one crate, one queue per device context, and no runtime dependency on another ML engine.

## Tensor and layout

`Tensor` contains `Rc<MetalBuffer>`, `Shape`, `Layout`, and `DType`. Fields are private. Shapes cache checked element counts; `[]` denotes a scalar with one element. Layout strides are in elements, row-major, and checked for overflow. Dimensions and strides use `SmallVec<[usize; 4]>`, supporting arbitrary rank with no dimension/stride heap allocation for the usual rank ≤4. Higher ranks spill to a vector.

All current tensors are contiguous and cover their entire logical allocation. Reshape preserves dtype, element count, and storage, rebuilding only contiguous metadata. There is no operation that can construct a slice, nonzero offset, or invalid raw memory view. Zero-sized dimensions are accepted when element and stride calculations are representable; pathological dimensions that overflow intermediate calculations are conservatively rejected.

DType's host/shader ABI is F32=0, F16=1, BF16=2. F16 uses Metal `half` loads/stores; BF16 uses two-byte integer storage and explicit conversion. Arithmetic and matmul/reduction accumulation are F32 for all three. CPU upload uses `half` crate conversions. BF16 stores implement round-to-nearest, ties-to-even and preserve NaN encoding.

## Storage

A `MetalBuffer` owns a retained `MTLBuffer`, logical length, and an `Rc` context identity. The physical Metal buffer length is `max(logical length, 4)` so empty tensors still have valid backing storage. Every allocation is zero-initialized. This simplifies initialized-memory guarantees and K=0/empty behavior, but output zeroing is included in end-to-end benchmark cost.

Storage mode is Shared. Actual CPU base-pointer alignment, physical allocation length, logical byte length, offset (currently always zero), and tensor owner count are observable. Alignment describes this allocation, not a universal allocator guarantee. Metal retains allocation ownership; `Rc` preserves it through tensor clones/reshapes. Storage can outlive the creating `MetalDevice` for CPU inspection. Operations reject tensors from a different context, even if both contexts use the same physical GPU.

Initialization converts directly into a borrowed byte slice of the shared allocation. Readback borrows initialized completed bytes and decodes directly into the returned F32 vector. Scoped closures prevent mapped references from escaping. There is no upload/readback staging vector and no CPU↔GPU blit. No storage mutation is exposed after tensor construction.

Apple's shared-memory guidance requires CPU/GPU accesses to be ordered despite shared physical memory: [MTLStorageMode.shared](https://developer.apple.com/documentation/metal/mtlstoragemode/shared). Phase 1 uses tracked hazards and command completion; managed-memory flush/synchronize calls do not apply to these shared buffers.

## MetalDevice and pipeline cache

`MetalDevice::new` discovers the default GPU, requires unified memory, creates one command queue, and retains device identity. CoreGraphics is linked for command-line discovery. Device name, recommended working set, max buffer length, unified-memory flag, and known Apple families 1–10 are exposed. The working-set recommendation is advisory, not an allocation budget.

The backend caches shader libraries by source text and pipelines by `(source, entry point)`. A separate built-in-name cache avoids copying/hashing shader source in operation hot paths. `warm_up` resolves all seven pipelines. The first compilation may benefit from Metal's persistent driver cache; measured process warm-up is not claimed to be an uncached compiler measurement.

Compilation requests safe math optimizations and precise single-precision math functions using macOS 15+ APIs. The compiler's full NSError description is returned on failure. Missing functions and pipeline creation failures have distinct contextual errors. Current bindings: [objc2-metal 0.3.2](https://docs.rs/objc2-metal/0.3.2/objc2_metal/).

`compile_kernel` returns an opaque cached pipeline; only built-in operations can dispatch. A future custom-kernel API must either be explicitly unsafe with a documented ABI/range contract or validate an expressive binding contract. Compilation alone cannot grant safe arbitrary GPU memory access.

## Kernel ABI and dispatch

Three buffers are bound: input A at index 0, optional input B at 1 (A is reused as an unused placeholder for unary kernels), and a fresh output at 2. Index 3 contains nine packed native-endian `u32` values (36 bytes):

| Index | Meaning |
|---|---|
| 0 | A element count |
| 1 | Reduction width, or matmul M |
| 2 | Matmul K |
| 3 | Matmul N |
| 4 | Dtype discriminant |
| 5 | RoPE position |
| 6 | RoPE head dimension |
| 7 | RMSNorm epsilon, F32 bits |
| 8 | RoPE theta, F32 bits |

Host validation checks context identity, matching dtypes, shapes, parameter domains, allocation sizes, and `u32` indexing before any command is committed. Matmul also guards K's final tile increment against wrap. The trusted shader guards loads/stores at tile edges. Full 16×16 groups are dispatched for matmul so every thread reaches both tile barriers. Other kernels use exact grids and up to 256 threads per group, bounded by pipeline limits. Empty grids return zero-dispatch metrics.

Add/mul/SiLU dispatch one thread per element. RMSNorm and softmax dispatch one thread per row. Softmax subtracts the row maximum before exponentiation. RoPE dispatches one thread per adjacent pair and uses `position * theta^(-2*pair/head_dim)`. Matmul loads two 16×16 threadgroup tiles, accumulates in F32, and stores one result per thread. It uses our own MSL, not MPS or MPSGraph.

## Synchronization and unsafe boundary

Command creation, encoding, commit, wait, and status inspection live in `metal::dispatch`. One public operation submits one command buffer and waits once. Completion is checked before output publication, including the GPU-error path. Inputs/outputs stay alive through the wait, and command buffers retain referenced resources. Autorelease pools bound temporary Objective-C lifetimes during compilation and dispatch.

Tensors and devices are deliberately `!Send` and `!Sync` via `Rc`. Published tensors are immutable, output allocations are fresh, and CPU initialization happens before publication. These invariants prevent safe users from racing CPU access with GPU work. An input may occupy both read-only bindings, but no output aliases an input.

Four unsafe blocks are confined to `src/metal/mod.rs`:

1. Mutable mapped byte slice during initialization: unique allocation access, initialized bounded range, no submitted work.
2. Immutable mapped byte slice during inspection: completed GPU work and a bounded initialized allocation.
3. Allocation zeroing: fresh owned shared memory of the requested physical length.
4. Encoder buffer/scalar binding: validated internal ABI and retained allocations/parameters through command encoding and completion.

The empty `unsafe extern "C"` declaration only links CoreGraphics; it declares no callable functions. No unsafe impl is used. No library/runtime `unwrap` or `expect` is used. CPU reference routines are test utilities with dimensional preconditions; they are not general runtime tensor operations.

MSL bounds, host/shader ABI consistency, Metal's driver, allocation availability, and Objective-C framework behavior remain trust boundaries. Rust's type system alone cannot establish shader memory safety. Apple validation layers provide an additional development check, not a proof.

## Observability and errors

Operations return `Output { tensor, metrics }`. Metrics include operation/output shape/dtype, logical bytes read/written, output allocation bytes, dispatch count, submission duration, synchronized wall duration, and optional GPU timestamps. No logging runs in the library hot path; callers explicitly inspect metrics. Logical bytes count each input payload once, not serial-kernel rereads or matmul tile traffic.

Submission duration starts at command creation and ends immediately after commit. Synchronized duration extends through completion. Both exclude validation, output allocation, and pipeline lookup/compilation. GPU duration is `GPUEndTime - GPUStartTime` after completion and may be unavailable. Criterion measures the wider public call including allocation and result disposal. Thus these numbers answer different questions and must not be substituted for each other.

`Error` distinguishes initialization, shader compilation, missing kernel, pipeline creation, invalid shape/reshape, dtype/context mismatch, allocation, dispatch, synchronization, parameter, and numerical-validation failures. Metal error descriptions are retained; no silent CPU fallback exists.

## Future work: graph execution and Phase 2

No graph executor, model, memory planner, or transformer is implemented. Phase 2 can compose the existing synchronous tensor operations immediately, but should not remove waits in-place. Before asynchronous batching or allocation reuse, introduce submission/completion ownership that keeps buffers alive, prevents premature CPU mapping, and tracks when each write completes. Keep encoding distinct from completion, submit several operations together, and publish readable tensors only behind the appropriate completion token.

A future planner can consume byte size, alignment, storage mode, and liveness to allocate arenas. Nonzero offsets and views will require checked ranges and dtype alignment in both tensor metadata and encoder bindings. Thread-safe execution will require an explicit synchronization design rather than replacing `Rc` with `Arc` or adding unsafe Send/Sync implementations.

RoPE convention and scaling are model-dependent; current adjacent-pair rotation is not interchangeable with split-half variants. F32 long-position phase accuracy and native BF16/SIMD-group matmul acceleration need separate designs and tests. Serial reductions and per-operation allocations are measured optimization targets, not architectural promises.
