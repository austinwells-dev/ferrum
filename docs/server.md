# Server and CLI guide

`ferrum-server` and `ferrum-cli` run Qwen3.5-family GGUF models (`qwen35` dense and `qwen35moe` hybrids) on the hybrid engine. Both of them:

- render the chat template embedded in the GGUF, the same way Hugging Face's `apply_chat_template` does,
- stream the model's reasoning separately from its answer (as `reasoning_content`),
- parse XML-style tool calls into structured `tool_calls`,
- use the sampling settings the GGUF recommends unless you override them,
- pick the largest context that fits in GPU memory unless you pass `--context N`.

Run `ferrum-server --help` or `ferrum-cli --help` for every flag.

## Server

```sh
ferrum-server --model model.gguf --port 8080      # http://127.0.0.1:8080/v1
```

`ferrum-server` follows llama-server's conventions, so most clients that work with llama.cpp work with Ferrum. Point an OpenAI SDK at `base_url="http://127.0.0.1:8080/v1"` (any API key works unless you set `--api-key`), or point an Anthropic client at `http://127.0.0.1:8080`.

### Endpoints

| Kind | Paths |
|---|---|
| OpenAI | `/v1/chat/completions`, `/v1/completions`, `/v1/models` |
| Anthropic | `/v1/messages`, `/v1/messages/count_tokens` |
| Utility | `/health`, `/props`, `/slots`, `/metrics` (Prometheus), `/tokenize`, `/detokenize`, `/apply-template` |

The model runs on its own thread behind a FIFO queue, and each connection gets a thread, so `/health` and `/metrics` respond while a request is generating. Requests are processed one at a time.

### Conversation caching

The server keeps the previous conversation. If the next request extends it, only the new tokens are processed, and `usage.prompt_tokens_details.cached_tokens` reports how many were reused. An identical retry reuses everything except the last token.

Hybrid models can't rewind their linear-attention state the way a KV cache can, so the session keeps recurrent-state snapshots (`--snapshots N`, default 2) to restore from when a request diverges.

### Thinking controls

Any of these request fields turn reasoning on or off, or set its effort:

| Field | Effect |
|---|---|
| `chat_template_kwargs`, `enable_thinking`, `think` | Enable or disable thinking |
| `reasoning_effort`, `reasoning: {effort}`, Anthropic `thinking: {type, budget_tokens}` | Set effort level |
| `reasoning_budget`, `thinking_budget_tokens` | Force `</think>` after N tokens |
| `reasoning_format: none` | Keep `<think>` inline in the content |

Server-wide defaults: `--no-think`, `--reasoning-effort`, `--reasoning-budget`, `--reasoning-format`, `--chat-template-kwargs`.

### Streaming

- Keep-alives stop clients from timing out during long prompts.
- `return_progress` streams prompt-processing progress.
- `stream_options.include_usage` adds a final usage chunk.
- If a client disconnects, generation stops right away and the work already done stays cached for a retry.
- If the model fails mid-request, the session resets and the server keeps running.

### Logging

Each request logs one line when it arrives, a progress line every few seconds, and a summary at the end (cached and processed tokens, speeds, reasoning tokens, stop reason). Add `-v` to also log request bodies and outputs.

### Main flags

| Flag | Default | Meaning |
|---|---|---|
| `--host`, `--port` | `127.0.0.1`, `8080` | Bind address |
| `--api-key KEY` | none | Require `Authorization: Bearer KEY` or `x-api-key: KEY` |
| `--alias NAME` | file name | Model name reported to clients |
| `-c`, `--context N\|auto` | `auto` | Context length |
| `--reserve-mib N` | `1024` | GPU memory left free for the system |
| `--chunk N` | `512` | Prompt tokens per forward pass |
| `--max-tokens N` | context | Per-request cap when the client sends none |
| `--temp`, `--top-p`, `--top-k`, `--min-p`, ... | from the GGUF | Sampling defaults |

## CLI

```sh
ferrum-cli --model model.gguf
```

Commands: `/reset`, `/think on|off`, `/image PATH`, `/stats`, `/exit`. To continue a message on the next line, end the line with `\`.

## Speculative decoding

Speculative decoding speeds up generation without changing the output. A small, fast drafter guesses the next few tokens, and the main model checks all of the guesses in one forward pass. With greedy decoding the output is exactly the same as without speculation. With sampling, the output distribution is unchanged.

Ferrum supports three kinds of drafter:

- **MTP.** The multi-token-prediction head built into the model's own GGUF. Nothing extra to download.
- **DFlash / DFlash2.** Separate block-diffusion drafters that propose a whole block of tokens at once.
- **DSpark.** DFlash plus a low-rank Markov bias and a confidence head that predicts which guesses will be accepted.

Drafters are loaded from Hugging Face checkpoints and quantized to Q4_0 when they load, with no measurable loss in accuracy. The memory planner budgets for the drafter up front and shrinks the automatic context to make room.

```sh
# The model's built-in MTP head
ferrum-server --model swift.gguf --draft mtp

# A separate drafter, from a directory or its Hugging Face cache entry
ferrum-server --model swift.gguf \
  --draft ~/.cache/huggingface/hub/models--z-lab--Qwen3.8-27B-DFlash2

ferrum-cli --model tiel.gguf \
  --draft ~/.cache/huggingface/hub/models--jzinno--Ornith-1.5-35B-A3B-DFlash2
```

| Flag | Meaning |
|---|---|
| `--draft mtp\|DIR` | Drafter to use (off by default) |
| `--draft-max N` | Most tokens drafted per step (default: 3 for MTP, the trained block size for dense targets, 2 for MoE targets) |
| `--draft-quant q4_0\|q8_0` | Drafter weight precision (default `q4_0`) |
| `--draft-context N` | Context slots for drafters without a sliding window (default 8192) |
| `--draft-p-min P` | DSpark: stop drafting below this confidence |

Responses include `draft_n` and `draft_n_accepted` in `timings`, as llama-server does. Gains are smaller on MoE models because each drafted token may route to different experts, so verifying a wide block reads more expert weights. The TUI's Benchmark tab finds the best drafter and depth for your machine automatically.

## Vision (images)

The Qwen3.5-family models can read images once you load their vision projector (the `mmproj*.gguf` published alongside the model). If a projector with a matching width sits in the same folder as the model, Ferrum loads it automatically. It takes about 1 GiB of the memory budget, and the auto-fitted context shrinks to make room.

| Flag | Effect |
|---|---|
| `--mmproj FILE\|auto\|none` | Projector to load (default `auto`) |
| `--image-tokens N` | Most context tokens one image may use (default 1024; larger images are scaled down) |
| `--image-min-tokens N` | Scale small images up to at least N tokens (default 64; Qwen-VL reads fine print better at 1024) |

Send images inline, either as OpenAI `{"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}` parts or as Anthropic `{"type": "image", "source": {"type": "base64", ...}}` blocks. Ferrum decodes PNG, JPEG, GIF, WebP and BMP, and it rejects remote URLs rather than fetching them. `/props` reports `modalities.vision`, and sending images to a model without a projector returns a clear 400 error.

Encoded images are cached, and the conversation cache keeps working across turns as long as the same pictures stay in the same places. [vision.md](vision.md) describes how the encoder works.
