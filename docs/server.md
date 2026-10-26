# Server and CLI guide

`ferrum-server` and `ferrum-cli` run Qwen3.5-family GGUF models (`qwen35` dense and `qwen35moe` hybrids) on the hybrid engine, and, experimentally, Flint and Spark ([experimental.md](experimental.md)). Both of them:

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
| Images | `/v1/images/generations` (needs `--image-model`; see [Image generation](#image-generation)) |
| Utility | `/health`, `/props`, `/slots`, `/metrics` (Prometheus), `/tokenize`, `/detokenize`, `/apply-template` |

The model runs on its own thread behind a FIFO queue, and each connection gets a thread, so `/health` and `/metrics` respond while a request is generating. By default requests are processed one at a time.

### Parallel requests

`--parallel N` (1–8, alias `-np`) serves up to N requests at once. Each runs in its own session (a *slot*) with its own K/V cache of the full context, its own recurrent state and its own snapshots; the slots share the weights, the scratch buffers, the drafter and the vision projector. With `--context auto` the planner fits the largest context that N slots hold together, so `--parallel 4` gives each slot roughly a quarter of the single-slot context (the Model line of the startup log shows it). With `--context N` each slot gets N tokens, and loading fails if N slots do not fit.

The slots take turns on the GPU: every running request takes one step in turn (a prompt chunk of `--chunk` tokens, or one decode step), so their streams interleave and a short request is not stuck behind a long one. This does not batch the requests together, so total throughput stays about that of one request, split between the running ones. A request that arrives while every slot is busy waits in the queue.

Speculative decoding runs on slot 0 only; the other slots decode without the drafter. `/slots` lists every slot (its request, phase, cached tokens and cache key), `/props` reports `total_slots`, and `/metrics` adds `ferrum_slots`, with `ferrum_requests_processing` counting the busy ones and `ferrum_cached_tokens` summing all slots.

### Conversation caching

Each slot keeps its previous conversation. If the next request extends it, only the new tokens are processed, and `usage.prompt_tokens_details.cached_tokens` reports how many were reused. An identical retry reuses everything except the last token.

With several slots, a request goes to the free slot that suits its cache best, so concurrent conversations each keep their own:

1. `id_slot: N` (llama-server's field; `-1` means any) pins the request to slot N, waiting for it if it is busy.
2. Otherwise, the free slot whose cache shares the longest prefix with the prompt, provided the shared part is at least half of what that slot holds. A new conversation that only shares the system prompt does not take over another conversation's slot.
3. Otherwise, the free slot that last served the same cache key: `prompt_cache_key` (OpenAI's field) or `session_id`, and for `/v1/messages` `metadata.user_id`.
4. Otherwise, the least recently used free slot, empty ones first.

Hybrid models can't rewind their linear-attention state the way a KV cache can, so each slot keeps recurrent-state snapshots (`--snapshots N`, default 2) to restore from when a request diverges.

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
| `--api-key KEY` | none | Require `Authorization: Bearer KEY` or `x-api-key: KEY`. `FERRUM_API_KEY=KEY` does the same without putting the key in the process list |
| `--cors ORIGIN` | localhost pages only | Let web pages from `ORIGIN` (e.g. `https://chat.example.com`) call the server; repeatable, `*` for any. Requests from other sites' pages get 403, so a site you visit can't use your model. Clients that aren't browsers (curl, SDKs, agents) send no `Origin` and are unaffected |
| `--alias NAME` | file name | Model name reported to clients |
| `-np`, `--parallel N` | `1` | Requests served at once, one slot each (1–8). See [Parallel requests](#parallel-requests) |
| `-c`, `--context N\|auto` | `auto` | Context length of each slot |
| `--kv-cache f16\|q8` | `f16` | K/V cache format (also `-ctk`/`-ctv`). `q8` nearly halves K/V memory, so `auto` fits about twice the context, and decodes faster at long context. See [K/V cache format](#kv-cache-format) |
| `--reserve-mib N` | `1024` | GPU memory left free for the system |
| `--chunk N` | `512` | Prompt tokens per forward pass |
| `--max-tokens N` | context | Per-request cap when the client sends none |
| `--temp`, `--top-p`, `--top-k`, `--min-p`, ... | from the GGUF | Sampling defaults |

### K/V cache format

`--kv-cache q8` stores attention keys and values as 8-bit values with one F16 scale per 32 (llama.cpp's `q8_0` layout), 1.06 bytes per element instead of 2. `FERRUM_KV_CACHE=q8` sets the same default for every front end and tool.

- **Memory.** The cache is the only thing that grows with context, so on a 32 GB Mac `auto` context roughly doubles for the dense 27B. The 35B MoE already fits its full 262K in F16; Q8 leaves about 2.3 GiB of that free (5.0 → 2.7 GiB). The dense 27B goes from 101K to 190K tokens of auto context.
- **Speed.** Decode attention reads about half the bytes, which matters from roughly 16K tokens of context. Prompt attention also gets faster (the kernels stage K/V through threadgroup memory either way).
- **Accuracy.** On Tiel the mean KL divergence against F16 is about 0.009, the same size as the difference between Ferrum and llama.cpp on that model, and perplexity is unchanged. F16 stays the default because it is exact.
- **Speculative decoding** stays correct, but greedy output with a drafter can differ from greedy output without one where the top two tokens are within a few tenths of a logit: a verify of 5 or more rows computes K/V through different matrix kernels than single-token decode, and 8-bit rounding amplifies that last-bit difference.

## CLI

```sh
ferrum-cli --model model.gguf     # --kv-cache q8 as for the server
```

Commands: `/reset`, `/think on|off`, `/image PATH [low|high|auto|N]`, `/detail low|high|auto|N`, `/stats`, `/exit`. To continue a message on the next line, end the line with `\`. A size after `/image PATH` applies to that image; `/detail` sets it for the images that follow (see [Image size per request](#image-size-per-request)).

## Speculative decoding

Speculative decoding speeds up generation without changing the output. A small, fast drafter guesses the next few tokens, and the main model checks all of the guesses in one forward pass. The output is the same as without speculation, with sampling too: every token is drawn with a random number tied to its seed and position, and a guess is kept only when it equals the token the model draws there, so a seeded request gives the same text with or without a drafter (as in Kapitsa). Guesses are kept as often as rejection sampling would keep them.

Ferrum supports four kinds of drafter:

- **MTP.** The multi-token-prediction head built into the model's own GGUF. Nothing extra to download. Dense models only: an MTP block with routed experts (as in the 35B MoE GGUFs) is refused.
- **DFlash / DFlash2.** Separate block-diffusion drafters that propose a whole block of tokens at once.
- **DSpark.** DFlash plus a low-rank Markov bias and a confidence head that predicts which guesses will be accepted.
- **Prompt lookup.** No model: it proposes the tokens that followed the latest earlier occurrence of the last few tokens. It pays off when output repeats its context (code edits, quoted text) and costs nothing when it misses.

Drafters are loaded from Hugging Face checkpoints (or DFlash GGUF files) and quantized to Q4_0 when they load, with no measurable loss in accuracy. The memory planner budgets for the drafter up front and shrinks the automatic context to make room.

**Draft trees.** With `--draft-tree N`, the drafter proposes a tree of up to N guesses instead of a chain, including its second and third choices where it is unsure, and the model checks the whole tree in the same single forward pass, keeping the longest path it agrees with. The MTP head keeps extending its likeliest guesses; DFlash, DFlash2 and DSpark build the tree from the candidates their one drafting pass already scores, at no extra drafting cost. Trees apply to every request, sampled or greedy, and the output is still the same as without speculation. On the dense 27B, `--draft mtp --draft-tree 15` accepts 4.0–4.2 tokens per pass against the 3-draft chain's 2.7–3.0, for 14.2–15.1 tok/s against 12.0–13.2 on a coding and a prose prompt (2.0–2.1× plain decoding); with the `z-lab/Qwen3.8-27B-DFlash2` drafter, trees of 15 reach 17.9–19.8 tok/s against its chain's 16.4–18.7, the fastest setup measured. DSpark trees help prose only (about +6%). Keep N at 15: 16 or more guesses push the check past 16 rows, where it costs about 60% more. On MoE targets keep chains: each guess routes to its own experts, so trees make the check slower than they save ([Phase 11 journal](history/phase11-journal.md)).

**KFlash (`--draft-tree auto`).** Ported from [Kapitsa](https://github.com/nace-ai/kapitsa): DFlash2 drafts a 16-row block (15 guesses deep) over its first 98,304 token ids, the tree ranks guesses by a per-depth calibration of the drafter's probabilities learned from what the model accepted, and each cycle checks as many guesses as are expected to pay for their cost. On the MLX 4-bit 27B with its DFlash2 drafter this gives 25.4 tok/s over 8 prompts (3.3× plain decoding, against 2.7× for the chain). On the Q4_K_XL GGUF, prefer `--draft-block 16 --draft-vocab 98304 --draft-tree 15` (19.3–19.8 tok/s, against 18.2 with the trained 8-row block; `auto` is 3–5% behind there).

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
| `--draft mtp\|lookup\|DIR\|FILE` | Drafter to use (off by default): the MTP head, prompt lookup, a checkpoint directory or its Hugging Face cache entry, or a DFlash GGUF |
| `--draft-max N` | Most tokens drafted per step, and the depth of draft trees (default: 3 for MTP and lookup, 7 for MTP with `--draft-tree`, the trained block size for dense targets, 2 for MoE targets) |
| `--draft-tree N\|auto` | Draft trees of up to N guesses (1–31; 15 recommended) per step, instead of chains. `auto`: KFlash's budget picks each cycle's guesses (up to 31), learning the drafter's calibration and the check's cost as it runs (kept in `~/.cache/ferrum/kflash`); for DFlash2 it also defaults `--draft-block 16` and `--draft-vocab 98304` |
| `--draft-block N` | DFlash/DFlash2: draft N rows per pass (N−1 guesses deep), wider than the drafter's trained block |
| `--draft-vocab N` | Draft among the first N token ids only (the check still uses the full vocabulary) |
| `--draft-tree-search best-first\|siblings` | How MTP grows trees: extending its likeliest guesses (default), or its chain plus each step's next-best guesses as leaves |
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

Encoded images are cached for every slot, by content and size, and the conversation cache keeps working across turns as long as the same pictures, at the same sizes, stay in the same places.

### Image size per request

Each image can ask for its own token budget, and a request can set one for all its images. Fewer tokens encode faster and leave more context; more read fine detail better.

| Where | Field | Effect |
|---|---|---|
| Image part (inside `image_url` or `source`, or on the part) | `detail: "low"` | At most 256 tokens (a 512x512 picture), as OpenAI's low detail |
| | `detail: "high"` | Up to the server's `--image-tokens` |
| | `detail: "auto"` | The server's defaults |
| | `detail: "N"`, `max_tokens: N` | At most N tokens |
| | `min_tokens: N` | Scale small images up to at least N tokens |
| Request body | `image_detail`, `image_max_tokens`, `image_min_tokens` | The same, for every image that doesn't set its own |

A request can lower the cap but never raise it past `--image-tokens`, because the memory plan reserves the encoder's activations for that size. A floor above the cap drops to the cap. Invalid values (`detail: "huge"`, `max_tokens: 0`) return 400.

```json
{"type": "image_url", "image_url": {"url": "data:image/png;base64,...", "detail": "low"}}
{"type": "image", "max_tokens": 512, "source": {"type": "base64", "media_type": "image/png", "data": "..."}}
``` [vision.md](vision.md) describes how the encoder works.

## Image generation

Ferrum draws and edits images with Qwen-Image-2.1 and its Turbo distillation (a Hugging Face directory with `model_index.json`, `transformer/`, `text_encoder/`, `vae/` and `processor/`). The stages run one after another and only one is in memory at a time, so it fits a 32 GB Mac. See [architecture](architecture.md#image-generation) for the design and [experimental.md](experimental.md#image-generation) for what is validated.

```sh
ferrum-cli image -m DIR -p "a red fox in snow" --size 1024x1024 --seed 7 -o fox.png
ferrum-cli image -m DIR --image fox.png -p "give the fox a red scarf" -o scarf.png
ferrum-server --image-model DIR                    # image-only server
ferrum-server --model model.gguf --image-model DIR # chat and images
curl localhost:8080/v1/images/generations -d '{"prompt":"a lighthouse at dusk","size":"512x512","n":1}'
curl localhost:8080/v1/images/edits -F 'image[]=@fox.png' -F 'prompt=give the fox a red scarf'
```

| Flag | Default | Meaning |
|---|---|---|
| `--size WxH` (CLI), `size` (API) | `1024x1024`; for an edit, the last image's shape | Multiples of 32, 256 to 2048. `"auto"` in the API is the default |
| `--image FILE` (CLI), `image` (API) | none | Edit: a condition image (PNG, JPEG, GIF, WebP, BMP). Up to 4; the prompt describes the change, or how to combine them |
| `--resolution N` (CLI), `resolution` (API) | 1024 | Edit: condition images are resized to about N² pixels at their own aspect ratio, sides multiples of 32 (as diffusers does), and an edit without a size comes out at that area too. 256 to 2048 |
| `--seed N` | random, printed | Noise seed. Seeds do not reproduce diffusers' images (different random generator). With `n` above 1, image *i* takes seed + *i* |
| `--steps N` | the checkpoint's schedule | Turbo runs only its own 8 sigmas; the base model takes 2 to 200 (default 40) |
| `--quantize bf16\|q8_0\|q4_0` | `bf16` | Transformer matrices, quantized as they load |
| `--text-quantize bf16\|q8_0\|q4_0` | `bf16` | Text encoder matrices, likewise |
| `--text-file FILE.gguf` (CLI), `--image-text-file` (server) | none | Take the text encoder's language model from a Qwen3-VL GGUF such as Unsloth's `Qwen3-VL-8B-Instruct-UD-Q4_K_XL.gguf` (`--model` still supplies the tokenizer, config and the vision tower edits use). Another architecture or geometry is refused |
| `--transformer-file FILE.gguf` (CLI), `--image-transformer-file` (server) | none | Take the transformer from a GGUF (below) instead of the directory's safetensors |
| `--image-model DIR`, `--image-alias NAME` | none | Server: the checkpoint and the name `/v1/models` reports |
| `--image-quantize`, `--image-text-quantize` | `bf16` | Server: as `--quantize` and `--text-quantize` |

**Edits** go to `/v1/images/edits` as OpenAI's clients send them: `multipart/form-data` with `image` or `image[]` file parts and the other fields as form fields, or a JSON body whose `image` is a base64 string, a `data:` URL or a list of them. Each condition image goes through the text encoder's vision tower (so the prompt can refer to it) and through the VAE's encoder (so the transformer sees its pixels), then the picture is denoised beside them.

**Memory.** Each stage is planned before it loads and checked against the GPU working set minus what this process already holds. With BF16 weights the largest stage is about 14.1 GiB (the text encoder) for a new picture at 512² or 1024², and the transformer's 16.2 GiB for a 1024² edit with one condition image ([edit-parity-2026-10-26.txt](measurements/image-gen/edit-parity-2026-10-26.txt)); `--image-quantize q8_0` and `--image-text-quantize q8_0` store those stages' matrices in about 8.5 bits a weight instead of 16. The check sees only this process: another program's GPU memory (llama.cpp, a second Ferrum) is invisible to it.

**GGUF transformers.** The `unsloth/Qwen-Image-2.1-GGUF` files hold the denoiser only and keep their own encoding, so `--quantize` does not apply. Their Q4_K, Q5_K, Q6_K, Q8_0, Q4_0 and BF16 block matrices load; a file whose blocks use another type (Q2_K, Q3_K, F16, the IQ types) is refused at load with the tensor named. They are the *base* model, which runs 40 steps with the dynamic-shift schedule and no guidance, so give `--model` a directory with the base `Qwen/Qwen-Image-2.1` `model_index.json` and `scheduler/`, and the text encoder, `vae/` and `processor/` (the Turbo ones work). The Q4_K_M transformer stage takes about 4.2 GiB at 512² and 4.9 GiB at 1024² ([gguf-2026-10-26.txt](measurements/image-gen/gguf-2026-10-26.txt)).

The response follows OpenAI's shape with `b64_json` PNGs only; `n` is at most 4, and if one of them fails the request fails. A request that does not fit in memory gets 503 and invalid parameters 400. A client that hangs up cancels the generation when the current denoising step ends (decoding is not interrupted). With no `--model`, the chat routes return 404. Timings are in [retime-2026-10-26.txt](measurements/image-gen/retime-2026-10-26.txt) and [timing-2026-10-26.txt](measurements/image-gen/timing-2026-10-26.txt): a new picture takes about 20 s at 512² and 69 s at 1024² (Q8_0 text encoder); an edit with one image takes about 24 s at 512² and 91 s at 1024².
