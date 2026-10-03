# Development guide

## Layout

| Path | Contents |
|---|---|
| `src/metal/` | Metal backend: device, pipeline cache, command batching, residency. The only module allowed to use `unsafe` |
| `src/metal/shaders/` | All GPU kernels (`ops.metal`, `project_mpp.metal`, `hybrid.metal`) |
| `src/tensor/`, `src/ops/`, `src/nn/` | Tensors, operations, attention, KV cache, MoE routing |
| `src/loader/` | safetensors and GGUF parsing |
| `src/model/` | The general transformer engine (Qwen2.5/3, Granite, OLMo 2, LFM2) |
| `src/hybrid/` | The large-model engine for Qwen3.5-family hybrids: memory planner, forward pass, sessions, drafters |
| `src/vision/` | Qwen3-VL image encoder |
| `src/tokenizer/`, `src/reference/` | Tokenizers; CPU reference implementations used as test oracles |
| `src/bin/` | `ferrum-cli`, `ferrum-server`, `ferrum-tui` |
| `tests/`, `benches/`, `examples/` | Integration tests, Criterion benchmarks, probes and measurement harnesses |
| `tools/` | Optional Python scripts that compare against Transformers, MLX and llama.cpp |

## Building

```sh
cargo build --release
```

`.cargo/config.toml` sets `MACOSX_DEPLOYMENT_TARGET=15.0` because the Metal compile options Ferrum uses need macOS 15. The release profile keeps build-time proc-macro dylibs unstripped, because macOS 27's dyld rejects some stripped ones ([Phase 1 results](phase1-results.md) has the details).

## Developer subcommands

```sh
cargo run --release -- info                # Metal device and compiler capabilities
cargo run --release -- smoke               # Quick GPU sanity check against CPU references
cargo run --release -- transformer-smoke

# Generate text from a local checkpoint with the general engine
cargo run --release -- run --model /path/to/Qwen2.5-0.5B-Instruct \
  --prompt 'Hello!' --max-new-tokens 32 --temperature 0 --warmup

# Same, with per-operation timing
cargo run --release -- profile --model /path/to/Qwen2.5-0.5B-Instruct \
  --prompt 'Hello!' --max-new-tokens 8 --temperature 0 --warmup

# Raw generation on the hybrid engine
cargo run --release -- hybrid --model /path/to/model.gguf --prompt 'The capital of France is'
```

- Point `--model` at a checkpoint directory with its config, tokenizer and weights, or at a GGUF file.
- Use `--raw` for plain completion prompts (OLMo 2 has no official chat template).
- `profile` also prints machine-readable `SUMMARY` and per-operation records.
- `FERRUM_BATCH_LIMIT=1` gives cleaner per-kernel timings but changes how work is executed, so leave it unset when you measure throughput.
- `FERRUM_NATIVE_MATMUL=0` switches to a slower diagnostic matrix-multiply fallback.
- Native BF16 kernels need a capable Metal compiler and GPU, and `info` reports whether yours qualifies. Devices older than the M5 have not been validated.

## Testing

```sh
cargo fmt --check
cargo clippy --all-targets --all-features -- -D warnings
cargo test
```

GPU tests need a Mac with Metal. Tests that need real model weights are `#[ignore]`d by default:

```sh
# With Metal's API and shader validation layers
MTL_DEBUG_LAYER=1 MTL_SHADER_VALIDATION=1 cargo test --lib \
  --test correctness --test transformer --test runtime_optimization -- --test-threads=1

# Against a real local checkpoint
FERRUM_QWEN_MODEL=/path/to/checkpoint MTL_DEBUG_LAYER=1 MTL_SHADER_VALIDATION=1 \
  cargo test --release --test real_model -- --ignored --test-threads=1
```

### Probes and benchmarks

```sh
cargo run --release --example qwen_probe -- /path/to/checkpoint --diagnostic-f32
python3 tools/check_phase4_probe.py
cargo bench --bench runtime -- --noplot
```

The `examples/hybrid_*` programs measure the hybrid engine: `hybrid_bench` (prefill and decode), `hybrid_kld` (KL divergence against llama.cpp logits), `hybrid_spec` (speculative decoding), `hybrid_verify_cost` (verify cost against block width) and `hybrid_needle` (long-context retrieval).

Python is only used by the optional comparison tools. To set up the reference environment:

```sh
python3 -m venv .venv-reference
.venv-reference/bin/pip install -r tools/reference-requirements.txt
```

## Using Ferrum as a library

The `ferrum` crate exposes GPU tensors (F32/F16/BF16), tensor operations, safetensors and GGUF loading, quantized matrices, both engines and text generation.

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

From the caller's side, tensor operations are synchronous. Model forward passes batch GPU work internally and only publish results after the GPU has finished.

**Safety.** The crate is `#![deny(unsafe_code)]` except for `src/metal`, which contains six small `unsafe` blocks for Objective-C FFI and shared-memory mapping, each with a written justification. Bounds checks, completion tracking and resource retention keep safe Rust callers safe. The remaining trust boundary is shader indexing and driver behaviour. Tensor handles use `Rc` and are deliberately not `Send`/`Sync`. See [architecture.md](architecture.md).

## Measurement hygiene

Background macOS work (Spotlight indexing, `dasd`) skews GPU benchmarks. `scripts/phase7a_cpu_preflight.py` checks for that before a run. When a known-noisy process won't settle, set `PHASE7A_PREFLIGHT_IGNORE=dasd`.
