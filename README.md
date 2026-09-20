# Ferrum

Ferrum is an independent Rust + Metal machine-learning runtime for Apple Silicon. The long-term goal is native LLM execution without MLX, llama.cpp, PyTorch, Python, MPSGraph, or another inference engine in the execution path.

**Phase 1 implements the runtime foundation only. It does not load or run models.**

## Requirements and commands

- Apple Silicon Mac with a Metal GPU; macOS 15 or later (tested on macOS 27.0, Apple M5).
- Rust 1.96+ and Apple's Command Line Tools / macOS SDK (`xcode-select --install` if missing).
- Metal's runtime shader compiler. Shaders are embedded MSL; no separate shader build command or Python environment is needed.

```sh
cargo run --release -- info
cargo run --release -- smoke
cargo fmt --check
cargo clippy --all-targets --all-features -- -D warnings
cargo test
cargo build --release
cargo bench --bench runtime -- --noplot
```

GPU tests run by default and fail if Metal is unavailable; they are never silently skipped. For per-case numerical errors, use `cargo test --test correctness -- --nocapture --test-threads=1`. To additionally enable Apple's validation layers:

```sh
MTL_DEBUG_LAYER=1 MTL_SHADER_VALIDATION=1 cargo test --test correctness -- --test-threads=1
```

The macOS 27 toolchain used for validation needs unstripped build-time proc-macro dylibs. A local Cargo profile override handles this; see the [build finding](docs/phase1-results.md) for details.

## Implemented

- Metal device discovery, device properties, one retained command queue, owned shared allocations, cached libraries and compute pipelines, completion/error handling, GPU timing.
- Arbitrary-rank contiguous tensors, checked element/byte counts, row-major strides, immutable shared storage, and zero-copy checked reshape. Dimensions and strides up to rank four stay inline.
- F32, F16, and BF16 storage. Kernels compute in F32; BF16 uses explicit load/store conversion with ties-to-even rounding rather than native BF16 instructions.
- Add, multiply, stable SiLU, last-axis weighted RMSNorm, stable last-axis softmax, interleaved adjacent-pair RoPE, and 16×16 tiled rank-two matmul. Every operation supports all three dtypes.
- Deterministic CPU reference comparisons, invalid-input tests, empty tensors, irregular tiles, transformer-like widths, a smoke CLI, and Criterion benchmarks with explicit completion waits.

```rust
use ferrum::{DType, MetalDevice, Tensor};

fn example() -> ferrum::Result<()> {
    let device = MetalDevice::new()?;
    let x = Tensor::from_f32(&device, [2, 2], DType::F32, &[1., 2., 3., 4.])?;
    let result = device.add(&x, &x)?; // complete when returned
    assert_eq!(result.tensor.to_f32(), vec![2., 4., 6., 8.]);
    println!("{:?}", result.metrics.timing);
    Ok(())
}
```

## Architecture and safety

`tensor` owns dtype, shape, layout, and retained allocation metadata. `ops` checks operation contracts and calls the private dispatcher in `metal`. Only `metal` interacts with Objective-C and mapped memory. `reference` contains scalar CPU validation oracles and is not called by the GPU execution path. The compact single-crate layout deliberately keeps related code together rather than creating a crate or module per tiny operation.

The safe tensor API cannot supply raw pointers, forge buffer ranges, mutate published tensor storage, or dispatch arbitrary user shaders. `Rc` makes tensors and execution contexts single-threaded; device identity is checked before dispatch. A result is published only after its command buffer completes. `compile_kernel` supports compiler diagnostics and cached opaque pipelines, but custom dispatch is intentionally not public: a safe arbitrary-shader API could bypass all buffer bounds guarantees.

There are four small unsafe blocks, all in `src/metal/mod.rs`: mapped read, mapped initialization, allocation zeroing, and encoder argument binding. An empty unsafe extern block links CoreGraphics for command-line device discovery. Each block documents its invariants. Safe Rust does **not** prove shader memory accesses or Apple's driver correct; embedded MSL and its host ABI remain an audited trust boundary.

## Unified memory and synchronization

Allocations use `MTLStorageModeShared` and Metal's default tracked hazards. Rust initializes the actual shared allocation directly, with dtype conversion as needed. Kernels access that same allocation; there are no staging buffers, blit uploads, or discrete-device transfers. `to_f32` reads completed shared storage directly into its returned Rust vector. Reshapes only share storage.

Shared memory does not eliminate synchronization: the CPU must not read GPU-written bytes before completion or modify bytes while the GPU reads them. Phase 1 waits once at each public operation boundary. No internal helper submits or waits separately. This deliberately simple synchronous contract establishes a correctness baseline; batching and completion-tracked asynchronous execution are future work.

Storage reports allocation length, logical byte length, zero offset, actual pointer alignment, storage mode, and Rust owner count. It does not implement pooling or a memory planner. See [architecture](docs/architecture.md) for the ABI and future execution constraints.

## Measurement

Each operation returns metrics without emitting hot-path logs: name, output shape, dtype, logical input/output bytes, allocation size, dispatch count, CPU submission duration, synchronized dispatch duration, and optional Metal GPU duration. Byte counts describe tensor payloads, **not measured memory traffic**.

Criterion measures the public synchronous operation, including output allocation/zeroing, metadata, encoding, submission, completion, and result disposal. Input construction and pipeline warm-up are outside its samples. Separate `TIMING` records show medians from 30 instrumented invocations; GPU timestamps can be unavailable. Small tensors are dominated by allocation/submission/synchronization overhead. The serial row reductions are intentionally correctness baselines.

Criterion stores machine-readable results in `target/criterion`. Use `--save-baseline phase1` / `--baseline phase1` for same-machine comparisons. No external engine benchmark integration is included. [Phase 1 results](docs/phase1-results.md) records the actual machine, validation, and measured baseline.

## Current limits

- Apple Silicon macOS only; synchronous, single-threaded contexts; full contiguous allocations; fresh output allocation per operation.
- Kernel indices and element counts fit `u32`; matmul K is capped below `u32::MAX - 16` to avoid tiled-loop overflow. Allocation also obeys the device's reported maximum buffer length.
- Matmul accepts only `[M,K] × [K,N]`; no batching, transpose, or broadcasting. RMSNorm and softmax reduce the last axis, which must be nonempty. Leading zero dimensions are supported. Empty elementwise/matmul outputs skip dispatch; K=0 matmul produces zero.
- RMSNorm/softmax use one GPU thread per row and serial F32 reductions. They are not optimized parallel reductions. Extremely large RMSNorm inputs can overflow F32 sum-of-squares. Finite, representable arithmetic is the numerical validation domain; NaN/infinity semantics for reductions are not promised.
- RoPE rotates adjacent pairs, applies one position to all heads in the call, and supports no partial rotation or scaling variants. F32 phase error grows with position; the test at position 2048 uses a 5e-4 absolute tolerance against an F64 oracle. Very long-context precision has not been established.
- F16/BF16 have reduced storage bandwidth but still use F32 arithmetic. There are no tensor-core/SIMD-group matrix instructions or peak-performance claims.
- No model formats, tokenizer, transformer, attention, KV cache, quantization, graph scheduler, autograd, training, bindings, or service interface.

Phase 2 should preserve the validated tensor/storage contracts and introduce explicit lifetime/completion tracking before allowing asynchronous execution or storage reuse. It has not been started.
