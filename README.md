# Ferrum

Ferrum is an independent Rust + Metal LLM runtime for Apple Silicon. Qwen2.5-0.5B-Instruct BF16 executes entirely through Ferrum; MLX, llama.cpp, PyTorch, Python and MPSGraph are not runtime dependencies.

Phase 4 adds completion-owned command batching, completion-safe storage reuse, checked contiguous views, growing KV storage, direct grouped attention layouts, vectorized decode GEMV, native SIMD-group prefill GEMM, and parallel reductions. On the tested Apple M5, three warm runs measured **511–514 prefill tok/s and 84–88 decode tok/s** for the pinned 21-token Hello workload. See [Phase 4 results](docs/phase4-results.md) for controls, numerical qualifications, memory, external comparisons and remaining bottlenecks.

## Build and run

Requires Apple Silicon macOS (tested on macOS 27, Apple M5), Rust 1.96+, Apple's Command Line Tools/macOS SDK, and Metal's runtime shader compiler. Native BF16 kernels require the supported Metal compiler/device capabilities; `info` reports these. We do not claim validation on older devices.

```sh
cargo build --release
cargo run --release -- info
cargo run --release -- smoke
cargo run --release -- transformer-smoke
cargo run --release -- run --model /path/to/Qwen2.5-0.5B-Instruct \
  --prompt 'Hello!' --max-new-tokens 32 --temperature 0 --warmup
cargo run --release -- profile --model /path/to/Qwen2.5-0.5B-Instruct \
  --prompt 'Hello!' --max-new-tokens 8 --temperature 0 --warmup
```

Supply the official local safetensors/tokenizer/config files, revision `7ae557604adf67be50417f59c2c2f167def9a775`. No runtime downloads occur. `run` streams text and a short timing summary; `profile` additionally emits machine-readable `SUMMARY` and per-operation records. `FERRUM_BATCH_LIMIT=1` isolates kernel timestamps but changes execution; use default batching for throughput. `FERRUM_NATIVE_MATMUL=0` selects the diagnostic matrix fallback.

## Validation

```sh
cargo fmt --check
cargo clippy --all-targets --all-features -- -D warnings
cargo test
MTL_DEBUG_LAYER=1 MTL_SHADER_VALIDATION=1 cargo test --lib \
  --test correctness --test transformer --test runtime_optimization -- --test-threads=1
FERRUM_QWEN_MODEL=/path/to/checkpoint MTL_DEBUG_LAYER=1 MTL_SHADER_VALIDATION=1 \
  cargo test --release --test real_model -- --ignored --test-threads=1
cargo run --release --example qwen_probe -- /path/to/checkpoint --diagnostic-f32
python3 tools/check_phase4_probe.py
cargo bench --bench runtime -- --noplot
```

GPU tests fail if Metal is unavailable. Explicit real-model tests require the local checkpoint. Python is only used for optional measurement/reference tooling. The macOS 27 build profile preserves proc-macro dylibs as explained in [Phase 1 results](docs/phase1-results.md).

## API and safety

The crate provides checked F32/F16/BF16 tensors, tensor operations, safetensors loading, a configurable dense Qwen2-style transformer, Qwen2.5 metadata/tokenizer validation and generation. Public tensor operations remain synchronous. Model forward batches dependent operations and publishes logits/cache only after completion. Tensor handles use single-threaded `Rc`; there is no unsafe `Send`/`Sync`.

```rust
use ferrum::{DType, MetalDevice, Result, Tensor};
fn example() -> Result<()> {
    let device = MetalDevice::new()?;
    let x = Tensor::from_f32(&device, [2, 2], DType::F32, &[1., 2., 3., 4.])?;
    let y = device.add(&x, &x)?;
    assert_eq!(y.tensor.to_f32(), vec![2., 4., 6., 8.]);
    Ok(())
}
```

Four unsafe blocks plus framework linkage remain confined to the Metal backend. Host bounds, completion tracking and retained resources protect safe Rust callers; shader indexing and driver behavior remain an audited trust boundary. See [architecture](docs/architecture.md).

## Limits

One validated production checkpoint, batch one, single-threaded contexts, contiguous views only, no paged/fused attention or graph scheduler. Native GEMM and reductions accumulate in F32, but changed reduction order can change BF16 rounding and near-tie greedy choices. The strict F32 reference diagnostic intentionally retains ordered reductions. Long-context performance and numerical accuracy beyond the recorded tests are not established.

No quantization, GGUF support, other model architectures, training, bindings or server. [Phase 1](docs/phase1-results.md), [Phase 2](docs/phase2-results.md) and [Phase 3](docs/phase3-results.md) remain historical reproducible baselines.
