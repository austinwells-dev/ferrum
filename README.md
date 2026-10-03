# Ferrum

Ferrum runs large language models locally on Apple Silicon Macs. It is written from scratch in Rust with its own Metal GPU kernels. It does not depend on MLX, llama.cpp, PyTorch, Python or MPSGraph at runtime.

You point it at a model file (GGUF or a Hugging Face checkpoint) and either chat with it in the terminal or run an OpenAI- and Anthropic-compatible server that coding agents and other tools can talk to.

**Highlights**

- Runs 27B–35B models on a 32 GB Mac, faster than llama.cpp on the same GGUF files (see [Performance](#performance)).
- Lossless speculative decoding: up to 2.5× faster generation with identical output.
- Automatically picks the largest context that fits in memory, before loading any weights.
- Drop-in server: OpenAI `/v1/chat/completions`, Anthropic `/v1/messages`, tool calls, reasoning/thinking controls, and conversation caching between requests.

## Quick start

Requirements: an Apple Silicon Mac (tested on an M5 with macOS 27), Rust 1.96+, and Apple's Command Line Tools.

```sh
cargo build --release

# Chat in the terminal
./target/release/ferrum-cli --model /path/to/model.gguf

# Run a server at http://127.0.0.1:8080/v1
./target/release/ferrum-server --model /path/to/model.gguf --port 8080
```

Or use the interactive launcher, which finds your models, lets you tune every setting, save favorite setups (with a DSpark/DFlash drafter) and starts chat or the server for you:

```sh
./target/release/ferrum-tui        # alias it: alias ferrum='/path/to/ferrum/target/release/ferrum-tui'
```

Ferrum never downloads anything. You supply the model files yourself.

## Supported models

**Large hybrid models** (dedicated engine in `src/hybrid`). Both use the Qwen3.5 hybrid architecture, which mixes Gated DeltaNet linear attention with gated full attention:

| Model | Format | Type |
|---|---|---|
| Swift 1.5 Qwen3.8-27B | Q4_K_M GGUF | Dense, 27B |
| Tiel-Coder 35B-A3B | UD-IQ4_XS GGUF | Mixture of experts: 256 experts, 3B active |

**Smaller models** (shared transformer engine):

| Family | Validated checkpoints |
|---|---|
| Qwen2.5 | 0.5B-Instruct (BF16, Q4_K_M GGUF) |
| Qwen3 | 0.6B and 1.7B (BF16), 0.6B Q8_0 GGUF |
| IBM Granite 4 | 4.0 350M (BF16) |
| IBM Granite MoE | 3.1 1B-A400M (BF16) |
| AllenAI OLMo 2 | 0425 1B (F32) |
| LiquidAI LFM2.5 | 230M (BF16), 8B-A1B hybrid MoE (BF16, Q4_K_M GGUF) |

Supported weight formats: safetensors in F32/F16/BF16, GGUF Q4/Q5/Q6/Q8 (including K-quants), and MLX affine Q4. Exact revisions and reference results are in the [Phase 6 journal](docs/phase6-architecture-journal.md); older Qwen2 quantized variants are covered in [Phase 5](docs/phase5-closeout.md).

## Using the CLI and server

Both `ferrum-cli` and `ferrum-server`:

- use the chat template embedded in the GGUF,
- stream the model's reasoning separately from its answer (as `reasoning_content`),
- parse tool calls,
- use the model's recommended sampling settings unless you override them.

### Server features

`ferrum-server` follows llama-server's conventions, so most clients that work with llama.cpp work with Ferrum.

**Endpoints**

- OpenAI: `/v1/chat/completions`, `/v1/completions`
- Anthropic: `/v1/messages` and `count_tokens`
- Utility: `/health`, `/props`, `/slots`, `/metrics` (Prometheus), `/tokenize`, `/detokenize`, `/apply-template`

**Conversation caching.** The server remembers the previous conversation. If the next request extends it, only the new tokens are processed. `usage.prompt_tokens_details.cached_tokens` tells you how many were reused. An identical retry reuses everything except the last token.

**Thinking controls.** Any of these request fields turn reasoning on or off, or set its effort:

| Field | Effect |
|---|---|
| `chat_template_kwargs`, `enable_thinking`, `think` | Enable or disable thinking |
| `reasoning_effort`, `reasoning: {effort}`, Anthropic `thinking: {type, budget_tokens}` | Set effort level |
| `reasoning_budget`, `thinking_budget_tokens` | Force `</think>` after N tokens |
| `reasoning_format: none` | Keep `<think>` inline in the content |

Server-wide defaults: `--no-think`, `--reasoning-effort`, `--reasoning-budget`, `--chat-template-kwargs`.

**Streaming.** Keep-alives prevent timeouts during long prompts. `return_progress` streams prompt-processing progress, and `stream_options.include_usage` adds a final usage chunk. If a client disconnects, generation stops right away and the work already done stays cached for a retry. If the model crashes mid-request, the session resets instead of the server going down.

**Logging.** Each request logs one line on arrival, progress lines every few seconds, and a summary (cached/processed tokens, speeds, reasoning tokens, stop reason). Add `-v` to also log request bodies and outputs.

## Speculative decoding

Speculative decoding makes generation faster without changing the output. A small, fast "drafter" guesses the next few tokens, and the main model checks all the guesses in a single pass. With greedy decoding the output is exactly the same as without speculation; with sampling, the output distribution is unchanged.

Three kinds of drafter are supported:

- **MTP** — the prediction head built into the model's own GGUF. No extra download.
- **DFlash / DFlash2** — separate drafters that propose a whole block of tokens at once.
- **DSpark** — DFlash with extra tricks to predict which guesses will be accepted.

Drafters are loaded from Hugging Face checkpoints and quantized to Q4_0 on load (with no measurable loss in accuracy). Memory for the drafter is planned up front, and the automatic context size shrinks to make room.

```sh
# Use the model's built-in MTP head
./target/release/ferrum-server --model swift.gguf --draft mtp

# Use a separate drafter from your Hugging Face cache
./target/release/ferrum-server --model swift.gguf \
  --draft ~/.cache/huggingface/hub/models--z-lab--Qwen3.8-27B-DFlash2

./target/release/ferrum-cli --model tiel.gguf \
  --draft ~/.cache/huggingface/hub/models--jzinno--Ornith-1.5-35B-A3B-DFlash2
```

Tuning flags:

| Flag | Meaning |
|---|---|
| `--draft-max N` | Maximum tokens drafted per step (MoE models default to 2) |
| `--draft-quant q4_0\|q8_0` | Drafter weight precision (default `q4_0`) |
| `--draft-context N` | Context size for drafters without a sliding window |
| `--draft-p-min P` | DSpark confidence cut-off |
| `--draft-vocab N` | Drafter vocabulary size |

Responses include `draft_n` and `draft_n_accepted` in `timings`, as llama-server does.

## Performance

All numbers are from a 32 GB Apple M5, comparing Ferrum and llama.cpp on the same GGUF file. "pp" is prompt processing (prefill) and "tg" is text generation (decode), both in tokens per second; higher is better.

### Large models

| Benchmark (tok/s) | Ferrum | llama.cpp |
|---|---|---|
| Swift pp512 / pp8192 | 171.9 / 162.1 | 149.8 / 134.4 |
| Swift tg at empty / 32K context | 6.65 / 5.85 | 5.98 / 5.47 |
| Tiel pp512 / pp16384 | 867 / 623 | 803 / 559 |
| Tiel tg at empty / 32K context | 42.1 / 29.5 | 35.8 / 26.8 |

Maximum context chosen automatically within a 24 GiB working set: 101K tokens for Swift, and the full 262K for Tiel. Predicted memory use matches Metal's own accounting to within a few MiB.

### With speculative decoding

8 chat prompts × 128 tokens, greedy, llama.cpp at commit 9710a32:

| Setup (tok/s) | Ferrum | llama.cpp |
|---|---|---|
| Swift, no speculation | 7.11 | 6.74 |
| Swift + MTP (3 drafts) | 12.60 | 11.17 |
| Swift + `z-lab/Qwen3.8-27B-DFlash2` | **17.78** | 9.87 |
| Swift + `RedHatAI/Qwen3.8-27B-speculator.dspark` | 16.37 | 6.83 |
| Swift + `RadixArk/Qwen3.8-27B-DSpark` | 12.53 | 6.93 |
| Tiel, no speculation | 41.29 | 41.14 |
| Tiel + `jzinno/Ornith-1.5-35B-A3B-DFlash2` (2 drafts) | **57.12** | 50.61 |

Both engines accept the same share of drafts, so Ferrum's lead comes from cheaper verification: checking 8–16 tokens at once costs Ferrum about 1.5× a single-token step, versus 2.4–3.8× for llama.cpp. Gains on Tiel are smaller because each drafted token may route to different experts.

### Accuracy

Measured as KL divergence against reference outputs using llama-perplexity's own text chunks (lower is closer):

- **Swift:** Ferrum matches llama.cpp to a mean KLD of 4e-6.
- **Tiel:** both engines are equally close to a high-precision reference (0.0106 for Ferrum vs 0.0107 for llama.cpp).

### Small models

Speed relative to llama.cpp (1.0x = equal). These are mostly limited by fixed per-kernel overhead, so the tiny Qwen models still trail at short prompts:

| Model | Decode | Prefill 512 / 1,024 | Prefill short / 128 |
|---|---|---|---|
| LFM2.5-8B-A1B Q4_K_M | 0.87–1.01x | 0.85–0.88x | 0.86x / 0.66–0.70x |
| Qwen3-0.6B Q8_0 | 0.80–0.88x | 0.81x / 0.93x | 0.65x / 0.62x |
| Qwen2.5-0.5B Q4_K_M | 0.75–0.86x | 0.80x / 0.90x | 0.67x / 0.65x |

Some of these were measured while a background system process (`dasd`) was busy, so treat them as approximate until re-measured.

## Development commands

The top-level binary has a few developer subcommands:

```sh
cargo run --release -- info                # Report Metal device and compiler capabilities
cargo run --release -- smoke               # Quick GPU sanity check
cargo run --release -- transformer-smoke

# Generate text from a local checkpoint
cargo run --release -- run --model /path/to/Qwen2.5-0.5B-Instruct \
  --prompt 'Hello!' --max-new-tokens 32 --temperature 0 --warmup

# Same, with per-operation timing output
cargo run --release -- profile --model /path/to/Qwen2.5-0.5B-Instruct \
  --prompt 'Hello!' --max-new-tokens 8 --temperature 0 --warmup
```

- Point `--model` at a checkpoint directory with its config, tokenizer and weights.
- Use `--raw` for plain completion prompts (OLMo 2 has no official chat template).
- `profile` also prints machine-readable `SUMMARY` and per-operation records.
- `FERRUM_BATCH_LIMIT=1` gives cleaner per-kernel timings but changes execution; leave it unset when measuring throughput.
- `FERRUM_NATIVE_MATMUL=0` switches to a slower diagnostic matrix-multiply fallback.
- Native BF16 kernels need a capable Metal compiler and GPU; `info` reports whether yours qualifies. Devices older than the M5 have not been validated.

## Testing

```sh
cargo fmt --check
cargo clippy --all-targets --all-features -- -D warnings
cargo test

# With Metal's validation layers
MTL_DEBUG_LAYER=1 MTL_SHADER_VALIDATION=1 cargo test --lib \
  --test correctness --test transformer --test runtime_optimization -- --test-threads=1

# Against a real local model
FERRUM_QWEN_MODEL=/path/to/checkpoint MTL_DEBUG_LAYER=1 MTL_SHADER_VALIDATION=1 \
  cargo test --release --test real_model -- --ignored --test-threads=1

# Numerical probe and benchmarks
cargo run --release --example qwen_probe -- /path/to/checkpoint --diagnostic-f32
python3 tools/check_phase4_probe.py
cargo bench --bench runtime -- --noplot
```

GPU tests fail if Metal is unavailable, and the real-model tests need a local checkpoint. Python is only used for optional measurement tools. The macOS 27 build profile keeps proc-macro dylibs around for a reason explained in [Phase 1 results](docs/phase1-results.md).

## Using Ferrum as a library

The `ferrum` crate exposes GPU tensors (F32/F16/BF16), tensor operations, safetensors and GGUF loading, quantized matrices, the transformer, and text generation.

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

Tensor operations are synchronous from the caller's point of view. Model forward passes batch GPU work internally and only publish results once the GPU has finished.

**Safety.** Only four `unsafe` blocks exist, all inside the Metal backend. Bounds checks, completion tracking and resource retention keep safe Rust callers safe; shader indexing and driver behavior are the remaining audited trust boundary. Tensor handles use `Rc` and are not `Send`/`Sync`. See [architecture](docs/architecture.md).

## Limitations

- One request at a time (batch size 1), single-threaded contexts.
- Tensor views must be contiguous. No paged attention or graph scheduler.
- GPU math accumulates in F32, but a different summation order can change BF16 rounding, and occasionally which token wins a near-tie in greedy decoding.
- For the smaller models, long-context speed and accuracy beyond the recorded tests are not established. LFM2 has been tested only up to 32K context.
- No training and no language bindings.

## Project history

Ferrum was built in phases, each with its own write-up:

| Phase | Focus | Docs |
|---|---|---|
| 1–3 | Foundations and baselines | [1](docs/phase1-results.md), [2](docs/phase2-results.md), [3](docs/phase3-results.md) |
| 4 | Command batching, KV cache, fast GEMV/GEMM | [Results](docs/phase4-results.md) |
| 5 / 5.5 | Quantized GGUF and MLX weights | [Closeout](docs/phase5-closeout.md), [experiments](docs/phase5.5-experiments.md) |
| 6 | Dense, MoE and hybrid architectures | [Journal](docs/phase6-architecture-journal.md) |
| 7 | Closing the gap to llama.cpp on GGUF | [Journal](docs/phase7a-performance-journal.md) |
| 8 | Large hybrid models, server | [Plan](docs/phase8-plan.md), [journal](docs/phase8-journal.md) |
| 9 | Speculative decoding | [Plan](docs/phase9-plan.md), [journal](docs/phase9-journal.md) |
