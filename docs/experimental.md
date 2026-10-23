# Experimental: Flint and Spark

> **Experimental.** This page covers Flint and Spark support, which lives on the `experimental` branch only. It is not production-ready. Expect rough edges, and expect the behaviour and the parsing to change.

## What works

- **Loading and generation** in `ferrum-cli hybrid`, `ferrum-server` and `ferrum-tui`, from `unsloth/flint-GGUF`, file `flint-UD-Q3_K_XL.gguf` (18.5 GB).
- **Tool calls.** Flint's `<tool_call>NAME<arg_key>…</arg_key><arg_value>…</arg_value></tool_call>` blocks are parsed into OpenAI `tool_calls`. Argument values are typed by the tool's JSON schema: a `string` parameter keeps its text, anything else is read as JSON. Malformed blocks, and blocks whose name is not a declared tool, stay visible as plain text.
- **Chat template.** The GGUF's Jinja template renders identically to Jinja2 (the reference used by Hugging Face) on the cases in `tests/fixtures/flint/render_cases.json`. Regenerate them with `tests/fixtures/flint/render_cases.py`.
- **Stop tokens.** Generation ends at Flint's end-of-generation ids from the GGUF: `<|end_of_text|>`, `<|user|>`, `<|observation|>` and `<|developer|>`.

## Validation

Measured on a 32 GB M-series Mac against llama.cpp built from `e3797b1` (after Flint merged).

| Check | Result |
|---|---|
| Tokenizer ids vs `llama-tokenize` (two mixed-script samples) | identical |
| Greedy continuation, "The capital of France is" | first three tokens match llama.cpp; token 4 differs (`" \n"` vs `" is"`), likely a near tie |
| KLD vs llama.cpp Metal, 5,115 positions | mean 0.052, median 0.0049, top-1 93.5% |
| llama.cpp CPU vs its own Metal (same file) | mean 0.074, median 0.011, top-1 91.2% |
| Speed (`hybrid_bench`) | prefill 883 tok/s (pp512), 899 tok/s (pp2048); decode 52 tok/s (tg128) |
| Qwen3.5 regression (dense 27B and MoE 35B) | identical generated tokens and per-tensor sums before and after |

Ferrum sits closer to llama.cpp's Metal output than llama.cpp's own CPU backend does, so the KLD gap looks like quantization sensitivity rather than a bug. It is still one quant on one text.

## Known gaps

- One quant (`UD-Q3_K_XL`) is validated. Other Flint files may use quant types the kernels do not cover; the TUI lists the supported set when it refuses a file.
- Flint's template has no thinking switch, so the prompt itself ends in `<think>` (thinking on) or `<think></think>` (off), written the way the template writes past reasoning so later turns reuse the cached prefix. A budget-cut reasoning closes with a bare `</think>`.
- Sliding-window layers keep their K/V in a ring of 1,088 rows (window 513 plus one prompt chunk), so only the 13 full-attention layers grow with the context: about 26 KiB per token plus 76 MiB of rings, against 98 KiB per token before. A session can rewind only while the rings still hold the rewind point's window; further back it recomputes from the start.
- The legacy attention kernels (`FERRUM_LEGACY_ATTENTION`, `FERRUM_SIMD_ATTENTION`) reject Flint's geometry.
- Drafters (MTP, DFlash) are not available for Flint.
- Expert routing implements neither a selection bias on the scores (`exp_probs_b.bias`) nor an `expert_weights_scale` other than 1.0; this GGUF has neither, and a file that does is refused at load.
- Chat `bos_token` and `eos_token` strings come from the vocabulary, but Flint's template does not use them.

## Running it

```sh
hf download unsloth/flint-GGUF flint-UD-Q3_K_XL.gguf --local-dir ~/models
cargo run --release --bin ferrum-tui
```

Pick the model in the Models tab. It shows as runnable once the quant check passes.

# Spark-X2.6-4B

XHToken's `spark2_5` architecture: a 4.4B dense model with attention on all 36 layers (three 512-token sliding-window layers to each global one), fused QKV, a per-head sigmoid output gate, no QK norm, a GEGLU feed-forward, and separate RoPE on sliding-window layers (256 dims at a 10k base, against 64 dims at 5M on global layers). It runs on the Flint path of the hybrid engine (`Variant::Spark`), with the same K/V rings.

## Validation

`XHToken/Spark-X2.6-4B-GGUF`, file `Spark-X2.6-4B-Q4_K_M.gguf` (3.5 GB), against llama.cpp build 11669 on a 32 GB M-series Mac.

| Check | Result |
|---|---|
| Tokenizer ids vs `llama-tokenize` (mixed-script stress file, README, architecture.md; 13.4K tokens) | identical |
| Greedy continuation, "The capital of France is" | identical to `llama-simple` |
| KLD vs llama.cpp Metal, 5,115 positions | mean 0.0012, median 0.00008, top-1 99.35%; PPL 13.303 vs 13.298 |
| llama.cpp CPU vs its own Metal (same file) | mean 0.0040, median 0.0010, top-1 97.9% |
| Speed (`hybrid_bench` vs `llama-bench`, interleaved) | pp512 1161 vs 1110-1165 tok/s; pp2048 1094 vs 1066-1121; tg128 35-38 vs 36-39 |

## Known gaps

- One quant (`Q4_K_M`) is validated. Its embedding and head gates are BF16. `Q8_0` should load but is untested; the all-BF16 file (`Spark-X2.6-4B.gguf`) is refused, since only embeddings, LM heads, routers, head gates and the fused QKV accept BF16.
- Tool calls use Flint's `<arg_key>`/`<arg_value>` format and go through the same parser, untested against this model.
- `reasoning_effort` (`low`/`high`) is passed through to the template; the 1M trained context is capped by the memory plan.
- No trained drafter (MTP, DFlash, DSpark) exists for Spark. Prompt lookup (`--draft lookup`, "Prompt lookup" in the TUI) works and is lossless: 1.13× on the `hybrid_spec` chat prompts (35.8 → 40.4 tok/s). Plain decode already reads weights at about 81% of the M5's measured bandwidth ceiling.
