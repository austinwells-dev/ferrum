# Ferrum

Ferrum is an independent Rust + Metal LLM runtime for Apple Silicon. Its validated model families include dense Qwen2.5, Qwen3, IBM Granite 4, and AllenAI OLMo 2; sparse IBM Granite MoE 3.1; and hybrid LiquidAI LFM2.5-230M and LFM2.5-8B-A1B hybrid MoE. MLX, llama.cpp, PyTorch, Python, and MPSGraph are not runtime dependencies.

Phase 4 adds completion-owned command batching, completion-safe storage reuse, checked contiguous views, growing KV storage, direct grouped attention layouts, vectorized decode GEMV, native SIMD-group prefill GEMM, and parallel reductions. On the tested Apple M5, three warm runs measured **511–514 prefill tok/s and 84–88 decode tok/s** for the pinned 21-token Hello workload. See [Phase 4 results](docs/phase4-results.md) for controls, numerical qualifications, memory, external comparisons and remaining bottlenecks.

Phase 5 added packed GGUF Q4/Q5/Q6/Q8 execution and a pinned MLX affine-Q4 loader. Phase 5.5 performance work remains incomplete; see [its experiment journal](docs/phase5.5-experiments.md). Phase 6 validated shared modern dense, sparse MoE, and hybrid/state execution; see the [architecture journal](docs/phase6-architecture-journal.md).

Phase 7 targets GGUF performance against a matched llama.cpp build. Its journal is the [Phase 7A performance journal](docs/phase7a-performance-journal.md). Phase 7A tuned format-specific GEMV and Metal 4 TensorOps paths. Phase 7B reworked scheduling around those kernels:

- Dense models use one concurrent compute encoder per forward, with range-tracked barriers.
- A wider M=1 attention context kernel is used where the old one underfilled the GPU.
- Sparse-MoE prompt chunks are assembled on the device instead of through host round trips.
- Greedy decoding selects argmax on the GPU and reads back one ID.

Against the Phase 7A checkpoint on an Apple M5, paired runs measured Qwen2.5-0.5B Q4_K_M at +4–8% cached decode and +5–11% complete generation. Qwen3-0.6B Q8_0 measured +2–7% decode and +3–9% complete generation. LFM2.5-8B-A1B Q4_K_M was neutral, within about 1% on decode and generation.

Kernel work after that checkpoint:
- **Factored-scale Q4_K/Q6_K M=1 kernels** (llama.cpp ports, Experiments 54–56) doubled LFM2.5-8B-A1B decode.
- **Block-wise GGUF-to-TensorOps tile decoding** (Experiment 57) raised prefill 1.3–2.5x.
- **Paired TensorOps M tiles, a shared RoPE table, and intra-epoch arena reuse** (Experiments 61–63) lifted prefill further.
- **Weight residency set with a dispatching keep-alive** (Experiments 64 and 69) removed a post-idle re-wiring stall: short-prompt prefill +20–33%.
- **Grouped small-batch expert GEMV** (Experiment 65): LFM2.5 short-prompt prefill +32%.
- **RoPE written straight into KV-cache slots** (Experiment 67), **16-bit Q5_0 loads** (Experiment 68), **K-split M=1 GEMVs** (Experiment 70), **fused Q5_0 SwiGLU** (Experiment 71), and **four-row Q8_0 GEMV** (Experiment 72) together raised Qwen2.5 Q4_K_M sustained decode from 178.8 to 202.6 tok/s (+13%).

Status against the saved llama.cpp rows (decode target 0.85x; prefill target 0.80x at 512/1,024 tokens):

| Model | Decode | 512 / 1,024 prefill | Short / 128 prefill |
|---|---|---|---|
| LFM2.5-8B-A1B Q4_K_M | 0.87–1.01x | 0.85–0.88x | 0.86x / 0.66–0.70x |
| Qwen3-0.6B Q8_0 | 0.80–0.88x | 0.81x / 0.93x | 0.65x / 0.62x |
| Qwen2.5-0.5B Q4_K_M | 0.75–0.86x | 0.80x / 0.90x | 0.67x / 0.65x |

LFM2.5-8B-A1B, the only realistically sized model measured, meets the decode and medium/long prefill targets. The sub-1B Qwen models still fall short at short prompts and long-context decode, where fixed per-kernel latency dominates. Split-context flash decode is implemented but off by default (Experiment 73). Results since Experiment 65 were measured while a system daemon (`dasd`) was pinned, so their llama.cpp ratios are indicative until re-measured on a quiet machine. See the journal's closing summary.

## Phase 8: large hybrid models

Phase 8 runs two current models end to end on a 32 GB Apple M5. Both use the Qwen3.5 hybrid architecture: Gated DeltaNet linear attention interleaved with gated full attention.

- **Swift 1.5 Qwen3.8-27B**, Q4_K_M (dense, 27B)
- **Tiel-Coder 35B-A3B**, UD-IQ4_XS (256-expert MoE with 3B active)

They run on a dedicated engine (`src/hybrid`) with its own kernel library. Phase 9 adds lossless speculative decoding on top (below).

| Same GGUF, same machine (tok/s) | Ferrum | llama.cpp |
|---|---|---|
| Swift pp512 / pp8192 | 171.9 / 162.1 | 149.8 / 134.4 |
| Swift tg @0 / @32K context | 6.65 / 5.85 | 5.98 / 5.47 |
| Tiel pp512 / pp16384 | 867 / 623 | 803 / 559 |
| Tiel tg @0 / @32K context | 42.1 / 29.5 | 35.8 / 26.8 |

Accuracy is measured with KL divergence over llama-perplexity's own chunks. On Swift, Ferrum's logits match llama.cpp's at a mean KLD of 4e-6. Tiel is a routing-sensitive MoE, so Ferrum and llama.cpp were each scored against a high-precision reference, and they sit equally close to it (KLD 0.0106 vs 0.0107).

Memory is planned from GGUF metadata before any weight is loaded. The largest context that fits is chosen automatically: 101K tokens for Swift and the full 262K for Tiel within the 24 GiB working set. Predictions match Metal's own accounting to within a few MiB. See the [Phase 8 plan](docs/phase8-plan.md) and [journal](docs/phase8-journal.md).

```sh
cargo build --release
# Interactive chat
./target/release/ferrum-cli --model /path/to/model.gguf
# OpenAI-compatible server for agents (http://127.0.0.1:8080/v1)
./target/release/ferrum-server --model /path/to/model.gguf --port 8080
```

Both binaries render the chat template embedded in the GGUF. They stream reasoning (as `reasoning_content`) and content separately, parse tool calls, and sample with the model's recommended settings unless told otherwise. The server keeps the conversation state between requests: a follow-up that extends the previous request prefills only the new tokens, and `usage.prompt_tokens_details.cached_tokens` reports how many were reused.

`ferrum-server` follows llama-server's conventions and adds a few of its own:

- **Endpoints.** It serves OpenAI `/v1/chat/completions` and `/v1/completions` and Anthropic `/v1/messages` (with `count_tokens`). It also serves `/health`, `/props`, `/slots`, `/metrics` (Prometheus), `/tokenize`, `/detokenize` and `/apply-template`.
- **Thinking controls.** Requests can pass any of these, and each is mapped onto the template's `enable_thinking` and effort levels:
  - `chat_template_kwargs`
  - `enable_thinking` or `think`
  - `reasoning_effort`, `reasoning: {effort}` or Anthropic `thinking: {type, budget_tokens}`
  - `reasoning_budget` or `thinking_budget_tokens`, which forces `</think>` after N tokens
  - `reasoning_format: none`, which keeps `<think>` inline

  Server defaults come from `--no-think`, `--reasoning-effort`, `--reasoning-budget` and `--chat-template-kwargs`.
- **Streaming.** SSE keep-alives cover long prefills. `return_progress` streams prompt progress, and `stream_options.include_usage` adds a usage chunk. When a client disconnects, generation stops within one step, and whatever it had already prefilled stays cached for a retry. A panic inside the model resets the session instead of taking the server down.
- **Logging.** One line per request covers the method, client and sampling. A progress line is printed every few seconds during long prefills and generations. A summary line gives cached and prefilled tokens, rates, reasoning tokens and the stop reason. Add `-v` to also log request bodies and outputs.
- **Snapshots.** A recurrent snapshot is taken at the prompt's second-to-last token, so an identical retry reuses everything but one token.

## Phase 9: speculative decoding

Phase 9 adds lossless speculative decoding to the hybrid engine. It supports three drafter families:

- **MTP**: the target GGUF's own NextN block.
- **DFlash / DFlash2**: block-diffusion drafters that propose a whole block in one pass.
- **DSpark**: DFlash plus a Markov bias and a confidence head.

Drafters load from Hugging Face checkpoints and are quantized at load. They read the target's hidden states at a few layers.

The target verifies all drafts in one forward pass. Recurrent layers only read their state during verification and record their inputs; the accepted prefix is then replayed into the live state. Greedy output equals plain greedy decoding. Sampled output keeps the target's distribution through rejection sampling.

Small-batch kernels keep that verify cheap. An 8–16-row forward on Swift costs about 1.5× a single-token decode; llama.cpp's costs 2.4–3.8×.

| Same GGUF, 8 chat prompts × 128 tokens, greedy (tok/s) | Ferrum | llama.cpp 9710a32 |
|---|---|---|
| Swift, no speculation | 7.11 | 6.74 |
| Swift + MTP (3 drafts) | 12.60 | 11.17 |
| Swift + `z-lab/Qwen3.8-27B-DFlash2` | **17.78** | 9.87 |
| Swift + `RedHatAI/Qwen3.8-27B-speculator.dspark` | 16.37 | 6.83 |
| Swift + `RadixArk/Qwen3.8-27B-DSpark` | 12.53 | 6.93 |
| Tiel, no speculation | 41.29 | 41.14 |
| Tiel + `jzinno/Ornith-1.5-35B-A3B-DFlash2` (2 drafts) | **57.12** | 50.61 |

Both engines accept drafts at the same rate, to within a point for every drafter. On Swift, DFlash2 decodes 2.5× faster than plain decoding.

On Tiel the gain is smaller. Each verify row routes to its own experts, so MoE targets default to 2 drafts.

Drafter memory is planned before loading, and the auto-fitted context shrinks to make room. Q4_0 draft weights are the default; they cost no measurable acceptance. See the [Phase 9 plan](docs/phase9-plan.md) and [journal](docs/phase9-journal.md).

```sh
# The GGUF's MTP block, or a draft checkpoint directory / HF cache entry
./target/release/ferrum-server --model swift.gguf --draft mtp
./target/release/ferrum-server --model swift.gguf \
  --draft ~/.cache/huggingface/hub/models--z-lab--Qwen3.8-27B-DFlash2
./target/release/ferrum-cli --model tiel.gguf \
  --draft ~/.cache/huggingface/hub/models--jzinno--Ornith-1.5-35B-A3B-DFlash2
```

Tuning flags: `--draft-max N`, `--draft-quant q4_0|q8_0`, `--draft-context N` (context slots for drafts without a sliding window), `--draft-p-min P` (DSpark confidence cut-off) and `--draft-vocab N`. Responses report `draft_n` and `draft_n_accepted` in `timings`, as llama-server does.

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
