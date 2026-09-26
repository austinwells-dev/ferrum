# Ferrum

Ferrum is an independent Rust + Metal LLM runtime for Apple Silicon. Its validated model families include dense Qwen2.5, Qwen3, IBM Granite 4, and AllenAI OLMo 2; sparse IBM Granite MoE 3.1; and hybrid LiquidAI LFM2.5-230M and LFM2.5-8B-A1B hybrid MoE. MLX, llama.cpp, PyTorch, Python, and MPSGraph are not runtime dependencies.

Phase 4 adds completion-owned command batching, completion-safe storage reuse, checked contiguous views, growing KV storage, direct grouped attention layouts, vectorized decode GEMV, native SIMD-group prefill GEMM, and parallel reductions. On the tested Apple M5, three warm runs measured **511–514 prefill tok/s and 84–88 decode tok/s** for the pinned 21-token Hello workload. See [Phase 4 results](docs/phase4-results.md) for controls, numerical qualifications, memory, external comparisons and remaining bottlenecks.

Phase 5 added packed GGUF Q4/Q5/Q6/Q8 execution and a pinned MLX affine-Q4 loader. Phase 5.5 performance work remains incomplete; see [its experiment journal](docs/phase5.5-experiments.md). Phase 6 validated shared modern dense, sparse MoE, and hybrid/state execution; see the [architecture journal](docs/phase6-architecture-journal.md).

Phase 7 targets GGUF performance against a matched llama.cpp build. Its journal is the [Phase 7A performance journal](docs/phase7a-performance-journal.md). Phase 7A tuned format-specific GEMV and Metal 4 TensorOps paths. Phase 7B reworked scheduling around those kernels:

- Dense models use one concurrent compute encoder per forward, with range-tracked barriers.
- A wider M=1 attention context kernel is used where the old one underfilled the GPU.
- Sparse-MoE prompt chunks are assembled on the device instead of through host round trips.
- Greedy decoding selects argmax on the GPU and reads back one ID.

Against the Phase 7A checkpoint on an Apple M5, paired runs measured Qwen2.5-0.5B Q4_K_M at +4–8% cached decode and +5–11% complete generation. Qwen3-0.6B Q8_0 measured +2–7% decode and +3–9% complete generation. LFM2.5-8B-A1B Q4_K_M was neutral, within about 1% on decode and generation. No format yet meets the Phase 7 gate of 0.85x llama.cpp decode and 0.80x prefill.

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

Supply an official local checkpoint with its config, tokenizer, and weight files. Validated checkpoints are Qwen2.5-0.5B-Instruct BF16, Qwen3-0.6B and 1.7B BF16, Granite 4.0 350M BF16, OLMo 2 0425 1B F32, Granite 3.1 1B-A400M BF16 MoE, LFM2.5-230M BF16 hybrid, and LFM2.5-8B-A1B BF16 hybrid MoE. The official Qwen3-0.6B Q8_0 GGUF and LFM2.5-8B-A1B Q4_K_M GGUF also pass the packed-weight model tests. The exact revisions and reference results are in the [Phase 6 journal](docs/phase6-architecture-journal.md); earlier Qwen2 quantized variants remain documented in [Phase 5](docs/phase5-closeout.md). Use `--raw` for plain completion prompts; OLMo 2 has no official chat template. No runtime downloads occur. `run` streams text and a short timing summary; `profile` additionally emits machine-readable `SUMMARY` and per-operation records. `FERRUM_BATCH_LIMIT=1` isolates kernel timestamps but changes execution; use default batching for throughput. At `--temperature 0`, `run` selects the argmax on the GPU (`generation::generate_greedy`). `FERRUM_NATIVE_MATMUL=0` selects the diagnostic matrix fallback.

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

The crate provides checked F32/F16/BF16 tensors, tensor operations, safetensors and GGUF loading, a shared transformer with explicit dense and sparse-expert policies, model-specific metadata/weight mapping, packed quantized matrices, and generation. Public tensor operations remain synchronous. Model forward batches dependent operations and publishes logits/cache only after completion. Tensor handles use single-threaded `Rc`; there is no unsafe `Send`/`Sync`.

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

Batch one, single-threaded contexts, contiguous views only, no paged/fused attention or graph scheduler (dense-model kernels may overlap within a command buffer only where no buffer range dependency exists). Native GEMM and reductions accumulate in F32, but changed reduction order can change BF16 rounding and near-tie greedy choices. The strict F32 reference diagnostic intentionally retains ordered reductions. Long-context performance and numerical accuracy beyond the recorded tests are not established.

Hybrid convolutional state is validated for the LFM2.5 schedules in the Phase 6 journal; the tested LFM2 context is capped at 32K and does not establish long-context performance. Training, bindings, and a server are outside the current scope. [Phase 1](docs/phase1-results.md), [Phase 2](docs/phase2-results.md), and [Phase 3](docs/phase3-results.md) remain historical reproducible baselines.
