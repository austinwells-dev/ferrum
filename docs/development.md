# Development guide

## Layout

| Path | Contents |
|---|---|
| `src/metal/` | Metal backend: device, pipeline cache, command batching, residency. The only module allowed to use `unsafe`. `metal/legacy.rs` holds the general engine's dispatcher and kernel switches |
| `src/metal/shaders/` | All GPU kernels (`hybrid.metal` and `hybrid_mlx.metal` for the hybrid engine; `ops.metal` and `project_mpp.metal` for the general engine, plus the keep-alive kernel) |
| `src/tensor/`, `src/loader/` | Tensors; safetensors and GGUF parsing |
| `src/hybrid/` | The engine the apps run: Qwen3.5-family hybrids (and, experimentally, Flint and Spark): memory planner, forward pass, sessions, drafters |
| `src/diffusion/` | Qwen-Image-2.1 drawing and editing: transformer, VAE decoder and encoder (`vae_encoder.rs`), text-encoder and vision-tower glue, staged pipeline (kernels in `diffusion.metal`, `diffusion_vae.metal`; compiled only for image runs) |
| `src/vision/`, `src/tokenizer/` | Qwen3-VL image encoder (from a GGUF projector or, with DeepStack, from safetensors); tokenizers |
| `src/model/`, `src/ops/`, `src/nn/`, `src/generation.rs`, `src/sampling.rs`, `src/quantization.rs`, `src/reference/` | The general transformer engine (Qwen2.5/3, Granite, OLMo 2, LFM2) and its CPU test oracles. Built only with `--features legacy-engine` |
| `src/bin/` | `ferrum-cli` (also `ferrum-cli image`), `ferrum-server` (image routes in `images.rs`, form uploads in `multipart.rs`), `ferrum-tui` (Image tab in `imagetab.rs`, the chat tool in `imagetool.rs`) |
| `tests/`, `benches/`, `examples/` | Integration tests and Criterion benchmarks (all general-engine, so they need the feature), `examples/hybrid_*` probes for the hybrid engine |
| `tools/` | Optional Python scripts that compare against Transformers, MLX and llama.cpp, `tools/imgref` (diffusers reference dumps for image generation), and `check_links.py` (run by CI) |

## Building

```sh
cargo build --release
cargo build --release --features legacy-engine   # also the general engine
```

The general engine (`ferrum run`, `smoke`, `transformer-smoke`, `profile`, the integration tests in `tests/`, `benches/` and the non-hybrid examples) is behind the `legacy-engine` feature, off by default. The TUI, server and CLI don't use it.

`.cargo/config.toml` sets `MACOSX_DEPLOYMENT_TARGET=15.0` because the Metal compile options Ferrum uses need macOS 15. The release profile keeps build-time proc-macro dylibs unstripped, because macOS 27's dyld rejects some stripped ones ([Phase 1 results](history/phase1-results.md) has the details).

## Developer subcommands

```sh
cargo run --release -- info                # Metal device and compiler capabilities

# General engine only (--features legacy-engine)
cargo run --release --features legacy-engine -- smoke   # Quick GPU sanity check against CPU references
cargo run --release --features legacy-engine -- transformer-smoke

# Generate text from a local checkpoint with the general engine
cargo run --release --features legacy-engine -- run --model /path/to/Qwen2.5-0.5B-Instruct \
  --prompt 'Hello!' --max-new-tokens 32 --temperature 0 --warmup

# Same, with per-operation timing
cargo run --release --features legacy-engine -- profile --model /path/to/Qwen2.5-0.5B-Instruct \
  --prompt 'Hello!' --max-new-tokens 8 --temperature 0 --warmup

# Generation on the hybrid engine: one user turn through the GGUF's chat template,
# or plain completion with --raw
cargo run --release -- hybrid --model /path/to/model.gguf --raw --prompt 'The capital of France is'
```

- Point `--model` at a checkpoint directory with its config, tokenizer and weights, or at a GGUF file.
- Use `--raw` for plain completion prompts (OLMo 2 has no official chat template).
- `profile` also prints machine-readable `SUMMARY` and per-operation records.
- `FERRUM_BATCH_LIMIT=1` gives cleaner per-kernel timings but changes how work is executed, so leave it unset when you measure throughput.
- `FERRUM_NATIVE_MATMUL=0` switches to a slower diagnostic matrix-multiply fallback.
- `FERRUM_KV_CACHE=q8` makes every tool and front end default to the 8-bit K/V cache (an explicit `--kv-cache` or TUI setting wins).
- `FERRUM_EARLY_COMMIT=N` commits each execution's first command buffer after N kernels (default 32) without waiting, so the GPU starts while the CPU encodes the rest of the step. `0` restores one command buffer per step. Worth about 2–3% decode on the 35B MoE; neutral on the dense 27B.
- `FERRUM_HYBRID_LEGACY_ATTENTION=1` restores the Phase 9 attention routing (simdgroup kernels for 2–15 query rows, unstaged TensorOps prompt attention) for A/B runs. It reads only an F16 cache.
- Native BF16 kernels need a capable Metal compiler and GPU, and `info` reports whether yours qualifies. Devices older than the M5 have not been validated.

## Testing

```sh
cargo fmt --check
cargo clippy --all-targets --all-features -- -D warnings
cargo test                              # hybrid engine, apps, Metal backend
cargo test --features legacy-engine     # adds the general engine and tests/
python3 tools/check_links.py            # relative links and #anchors in all markdown
```

The branch flow, the pre-PR checklist and the regression checks for large changes are in [AGENTS.md](../AGENTS.md).

GPU tests need a Mac with Metal. Tests that need real model weights are `#[ignore]`d by default:

```sh
# With Metal's API and shader validation layers
MTL_DEBUG_LAYER=1 MTL_SHADER_VALIDATION=1 cargo test --features legacy-engine --lib \
  --test correctness --test transformer --test runtime_optimization -- --test-threads=1

# Against a real local checkpoint
FERRUM_QWEN_MODEL=/path/to/checkpoint MTL_DEBUG_LAYER=1 MTL_SHADER_VALIDATION=1 \
  cargo test --release --features legacy-engine --test real_model -- --ignored --test-threads=1
```

### Probes and benchmarks

```sh
cargo run --release --features legacy-engine --example qwen_probe -- /path/to/checkpoint --diagnostic-f32
python3 tools/check_phase4_probe.py
cargo bench --features legacy-engine --bench runtime -- --noplot
```

The `examples/hybrid_*` programs measure the hybrid engine: `hybrid_bench` (prefill and decode), `hybrid_kld` (KL divergence against llama.cpp logits), `hybrid_spec` (speculative decoding, `--depth N --filler FILE` for long context, `--sweep 0,4,8,15` for draft-tree budgets), `hybrid_verify_cost` (verify cost against block width), `hybrid_tree_check` (draft-tree verify and path commit against plain decoding and a chain, with oracle trees), `hybrid_tree_cost` (tree verify and commit cost against a chain's), `hybrid_mtp_tree_check` (best-first MTP trees against MTP's own chain, logit for logit), `hybrid_attention` and `hybrid_prefill_attention` (attention kernels alone, per query width and K/V format), `hybrid_plan` (memory plan per K/V format) and `hybrid_needle` (long-context retrieval).

To check that a drafter (or a kernel change) keeps plain decoding's answers, run the benchmark's `≈` answer check headless. It asks the 30 bundled math questions with and without the drafter and fails if the drafter loses more than one net answer:

```sh
FERRUM_CHECK_MODEL=/path/to/model.gguf FERRUM_CHECK_DRAFTER=/path/to/drafter \
  FERRUM_CHECK_DEPTH=4 cargo test --release --bin ferrum-tui -- --ignored --nocapture keeps_its_answers
```

`FERRUM_CHECK_DRAFTER=mtp` uses the model's MTP head, `FERRUM_CHECK_DRAFTER=lookup` prompt lookup, and `FERRUM_CHECK_QUANT=q8_0` loads the drafter in 8 bits. On a dense 27B each pass takes about 15 minutes.

The agent benchmark's tasks can be checked without a model as well. This clones all ten (it needs the network) into a temporary folder and checks, in the sandbox, that each fails as checked out and passes with its real fix:

```sh
cargo test --release --bin ferrum-tui -- --ignored --nocapture tasks_fail_before
```

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
