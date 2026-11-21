# Server and CLI guide

`ferrum-server` and `ferrum-cli` run Qwen3.5-family GGUF models (`qwen35` dense and `qwen35moe` hybrids) on the hybrid engine, and also Flint, Spark, Inkling 2 Mini, LFM3 and Qwen3.8-Flash-Next ([models.md](models.md)). LFM3-Omni is not supported. Both of them:

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
| OpenAI | `/v1/chat/completions`, `/v1/responses`, `/v1/completions`, `/v1/models` |
| Anthropic | `/v1/messages`, `/v1/messages/count_tokens` |
| Images | `/v1/images/generations` (needs `--image-model`; see [Image generation](#image-generation)) |
| Utility | `/health`, `/props`, `/slots`, `/metrics` (Prometheus), `/tokenize`, `/detokenize`, `/apply-template` |

The model runs on its own thread behind a FIFO queue, and each connection gets a thread, so `/health` and `/metrics` respond while a request is generating. By default requests are processed one at a time. At most 256 connections are open at once; further ones get 503.

### Parallel requests

`--parallel N` (1–8, alias `-np`) serves up to N requests at once. Each runs in its own session (a *slot*) with its own K/V cache of the full context, its own recurrent state and its own snapshots; the slots share the weights, the scratch buffers, the drafter and the vision projector. With `--context auto` the planner fits the largest context that N slots hold together, so `--parallel 4` gives each slot roughly a quarter of the single-slot context (the Model line of the startup log shows it). With `--context N` each slot gets N tokens, and loading fails if N slots do not fit.

The slots take turns on the GPU: every running request takes one step in turn (a prompt chunk of `--chunk` tokens, or one decode step), so their streams interleave and a short request is not stuck behind a long one. This does not batch the requests together, so total throughput stays about that of one request, split between the running ones. A request that arrives while every slot is busy waits in the queue.

Speculative decoding runs on slot 0 only; the other slots decode without the drafter. `/slots` lists every slot (its request, phase, cached tokens and cache key), `/props` reports `total_slots`, and `/metrics` adds `ferrum_slots`, with `ferrum_requests_processing` counting the busy ones and `ferrum_cached_tokens` summing all slots.

### Conversation caching

Each slot keeps its previous conversation. If the next request extends it, only the new tokens are processed, and `usage.prompt_tokens_details.cached_tokens` reports how many were reused. `usage.completion_tokens_details.reasoning_tokens` is the part of `completion_tokens` spent on reasoning. An identical retry reuses everything except the last token.

With several slots, a request goes to the free slot that suits its cache best, so concurrent conversations each keep their own:

1. `id_slot: N` (llama-server's field; `-1` means any) pins the request to slot N, waiting for it if it is busy.
2. Otherwise, the free slot whose cache shares the longest prefix with the prompt, provided the shared part is at least half of what that slot holds. A new conversation that only shares the system prompt does not take over another conversation's slot.
3. Otherwise, the free slot that last served the same cache key: `prompt_cache_key` (OpenAI's field) or `session_id`, and for `/v1/messages` `metadata.user_id`.
4. Otherwise, the least recently used free slot, empty ones first.

Hybrid models can't rewind linear-attention state the way a KV cache can, and a sliding-window layer's ring overwrites its oldest rows, so each slot snapshots both (`--snapshots N`, default 2): the recurrent state and a copy of every ring. A request that diverges restores the deepest snapshot at or before that point. Full-attention K/V rows stay valid. Chat templates that drop a past reply's reasoning diverge a few tokens before the end of the previous prompt; the session also snapshots there. See [architecture](architecture.md#the-hybrid-engine) and [models.md](models.md#sessions).

**Across restarts.** With `--slot-save-path DIR`, SIGINT or SIGTERM (Ctrl-C, `kill`) makes the model thread save every slot that holds a conversation to `DIR/slot-N` before the process ends, and the next start loads them back when the model file, context and K/V format are the same. A request in flight is dropped where it is; what its slot had cached by its last step is saved. A second signal quits at once without saving. Without the flag the same signals just end the process. An example is in [measurements/gpu-share](measurements/gpu-share/README.md#slot-caches-across-a-restart---slot-save-path).

**Idle unload.** With `--idle-unload SECS` the model thread drops the model once no request has run or waited for that long, which returns its memory (weights and every slot's K/V cache) to the system. The server keeps listening and `/health` keeps answering `ok`; `/v1/models` reports `"status": "unloaded"`. The next request loads the model again and waits for the load and warmup, so it starts a few seconds late. Slot caches are lost on unload unless `--slot-save-path` is set, in which case they are saved before the unload and restored after the reload. On a 4B model with a 23 GiB plan the process went from 23 GB to 81 MB while unloaded, and the request that woke it took 5.9 s instead of 1.7 s: [record](measurements/qol-2026-11/idle-unload-2026-11-15.txt).

### Sharing the GPU

A model that runs flat out can make video stutter and the desktop lag, because the window server and video decoders wait for the same GPU. `--gpu-share PCT` (5–100, or `off`) keeps Ferrum's share of the GPU's time at about PCT%: after every prompt piece and every decode step the session sleeps in proportion to how long the step took, and prompts are read 128 tokens at a time instead of `--chunk`, so no single step holds the GPU for long. The text is the same as without it; speeds drop to about that share (on Spark-X2.6-4B, 36.0 tok/s decode becomes 15.9 at 50 and 7.8 at 25: [measurement](measurements/gpu-share/README.md)). Reported speeds include the pauses.

`FERRUM_GPU_SHARE=50` sets the default for `ferrum-server`, `ferrum-cli` and the TUI's chats, agents and server (the TUI's *GPU share* setting wins). Benchmarks and the developer tools ignore it. Whether a share is low enough for a given video on a given model has not been measured: start at 50 and lower it if playback still drops frames.

### Thinking controls

Any of these request fields turn reasoning on or off, or set its effort:

| Field | Effect |
|---|---|
| `chat_template_kwargs`, `enable_thinking`, `think` | Enable or disable thinking |
| `reasoning_effort`, `reasoning: {effort}`, Anthropic `thinking: {type, budget_tokens}` | Set effort level |
| `reasoning_budget`, `thinking_budget_tokens` | Force `</think>` after N tokens |
| `reasoning_format: none` | Keep `<think>` inline in the content |

Server-wide defaults: `--no-think`, `--reasoning-effort`, `--reasoning-budget`, `--reasoning-format`, `--chat-template-kwargs`.

Inkling 2's template knows only an effort level (`none`, `minimal`, `low`, `medium`, `high`, `xhigh`, `max`, or a number up to 0.99; default 0.9). Turning thinking off sends level 0, at which the model may still reason briefly before a tool call. Its reply arrives in the same `reasoning_content`, `content` and `tool_calls` fields as other models'.

### Structured output

`/v1/chat/completions` can hold a reply to JSON. Every token is checked against a grammar before it is chosen, so the content always parses (unless `max_tokens` cuts it short, which reports `finish_reason: "length"`).

| Request field | Reply |
|---|---|
| `response_format: {"type": "json_object"}` | One JSON object, any keys |
| `response_format: {"type": "json_schema", "json_schema": {"schema": {...}}}` | One value matching the schema (OpenAI's form) |
| `response_format: {"type": "json_object", "schema": {...}}`, or a top-level `json_schema: {...}` | The same (llama-server's forms) |

Schemas may use `type` (also a list of types), `properties`, `required`, `additionalProperties`, `items`, `minItems` and `maxItems` (up to 64), `enum`, `const`, `anyOf`, `oneOf`, `nullable` and local `$ref`s (`#/$defs/...`, recursive ones included). Things to know:

- **Property order is the schema's.** An object writes its properties in the order `properties` lists them (optional ones may be left out), and nothing else. A model does not get to reorder them. An object schema without `properties` is free-form.
- **Not enforced, with a warning in the log:** `pattern`, `format`, `minLength`, `maxLength`, `minimum`, `maximum`, `exclusiveMinimum`, `exclusiveMaximum`, `multipleOf`, `uniqueItems`, `minProperties`, `maxProperties`. The reply has the right shape but may break these.
- **Refused with 400:** anything else that changes what is valid (`allOf`, `not`, `if`, `patternProperties`, remote `$ref`s, ...), rather than returning JSON that silently ignores it.
- **Reasoning is not constrained.** With thinking on, the model reasons freely and the grammar starts after `</think>`.
- **Tools win.** A request that also has `tools` is not constrained (a constrained reply could not call a tool); the log says so. A forced `tool_choice` is a separate mechanism (below).
- **Speed.** A constrained reply reads the full logits and does not use the drafter, so speculative decoding does not speed it up. On Spark-X2.6-4B constrained replies ran at 31.9–36.1 tok/s against 37.2 unconstrained ([record](measurements/qol-2026-11/README.md#structured-output)).
- **Whitespace** between JSON tokens is at most one space, or a newline and up to 20 spaces or tabs.
- A token that is only part of a multi-byte character is never chosen inside a constrained reply, so text in scripts the tokenizer splits below the character may come out worse than unconstrained.

A constraint steers the model, it does not make it smarter: a small model held to a tight schema (for example `maxItems` it wants to exceed) can run on inside a string until `max_tokens`. Constraints are not available on `/v1/completions` or `/v1/messages`, and `logprobs` is ignored with a warning. What was checked: [models.md](models.md#structured-output).

**GBNF grammars.** `grammar` takes llama.cpp's grammar text and holds the content to its `root` rule, with the same rules as a schema (reasoning is free, tools win, no drafter). It cannot be combined with `response_format`. Supported: quoted literals with escapes, character classes (ranges, `^` negation), `.`, rule names, groups, `|`, and the repetitions `*`, `+`, `?`, `{m}`, `{m,}`, `{m,n}` (counts up to 256). Refused with 400: token references (`<[id]>`), and left recursion, including a repetition of something that can be empty such as `("a"?)*`. Checked on Spark-X2.6-4B: [record](measurements/qol-2026-11/README.md#gbnf).

**`n`.** `/v1/chat/completions` returns up to 8 choices for `n`, run one after another with seeds `seed`, `seed + 1`, ..., each reusing the cached prompt. `usage` counts the prompt once and adds the completions up. `n` above 1 cannot be streamed.

**`tool_choice`.** `"none"` drops the tools. `"required"`, or a named function (`{"type": "function", "function": {"name": N}}`; on `/v1/messages`, `{"type": "any"}` and `{"type": "tool", "name": N}`), makes the reply call a tool: the server writes the opening of the model's tool-call format where the reply's content starts (after the reasoning when thinking is on, including a reasoning budget's forced close), and the model continues from there. A name that is not in `tools` is refused with 400. Things to know:

- This is enforced for Qwen3.5-family models (dense and MoE) and Flint, whose tool-call formats the server knows how to open. For Spark, LFM3 and Inkling 2 it is not: the request runs, the model decides, and the log says so.
- Only the opening is forced (and for a named function, its name). The arguments are the model's and are not checked against the tool's schema; a call the parser cannot read comes back as content.
- With thinking on and reasoning left inline (`--reasoning-format none`), nothing is forced.
- Checked on a 35B MoE: [record](measurements/qol-2026-11/README.md#tool_choice). Flint's opening has not been run.

**`logit_bias`** on `/v1/chat/completions` is OpenAI's map of token id to a value from -100 to 100, added to that token's logit at every step (reasoning included) after the penalties; llama-server's list form `[[id, bias], ...]` is accepted too, where `false` bans the token. Ids outside the vocabulary are ignored. Like a schema, a biased reply reads the full logits, samples from the 1024 likeliest tokens and does not use the drafter.

### Responses API

`/v1/responses` is OpenAI's Responses API, stateless: every request carries its whole conversation in `input`, and `previous_response_id` is refused with 400 (nothing is stored, so `store` is ignored). It runs the same path as `/v1/chat/completions`, so conversation caching, thinking controls, `tool_choice`, `logit_bias`, `grammar` and this server's sampling fields work the same way.

| Request | Meaning |
|---|---|
| `input` | A string, or items: messages (`role` with `input_text`, `output_text` and `input_image` parts; `developer` is the system role), `function_call`, `function_call_output`; `reasoning` items are dropped |
| `instructions` | The system message |
| `tools`, `tool_choice` | Function tools; other tool types (web search, file search, ...) are ignored with a warning |
| `text.format` | `json_schema` or `json_object`, as [Structured output](#structured-output) |
| `reasoning.effort` | As `reasoning_effort`; `none` turns thinking off |
| `max_output_tokens` | The token limit; a reply cut by it has `status: "incomplete"` |

The reply's `output` holds a `reasoning` item (the thinking, as `reasoning_text` content), a `message` with one `output_text` part, and one `function_call` per tool call. With `stream: true` the events are `response.created`, `response.in_progress`, then for each item `response.output_item.added`, its text as `response.reasoning_text.delta` or `response.output_text.delta`, and `response.output_item.done`, ending with `response.completed` (or `response.incomplete`). Function calls arrive whole after the text, since a call is parsed when the reply ends. Checked on a 35B MoE: [record](measurements/qol-2026-11/README.md#responses-api).

### Several models

With `--models-dir DIR` the server also offers every other chat GGUF in `DIR`, named by its file name without `.gguf` (projector files are skipped, and a file whose tokenizer or chat template cannot be read is left out with a warning). `/v1/models` lists them with `status` `ready` or `unloaded`. One model is in memory at a time:

- A request whose `model` names one of them is for that model. Any other `model` value, or none, means the `--model`.
- When the oldest waiting request is for a model that is not loaded, nothing new starts, the running requests finish, the server unloads the model and loads the other one. The request waits for the load. Clients that alternate between two models therefore reload on every request.
- A switched-to model gets the largest context that fits, its own sibling `mmproj` projector, its GGUF's recommended sampling under the same command-line overrides, and no drafter: `--draft`, `--alias` and `--mmproj` apply to the `--model` only.
- Slot caches do not survive a switch. With `--slot-save-path` they are saved before it, but the directory holds one model's slots at a time: the next model to save replaces them, and a model only loads slots it wrote itself.
- If the model fails to load, its requests get 503 and the server goes back to the `--model`. This path has not been exercised.
- `/tokenize`, `/detokenize`, `/apply-template` and `/v1/messages/count_tokens` take `model` too; `/props` describes the `--model`.

Checked with a 4B and a 35B model: [record](measurements/qol-2026-11/README.md#several-models).

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
| `--api-key KEY` | none | Require `Authorization: Bearer KEY` or `x-api-key: KEY`. `FERRUM_API_KEY=KEY` does the same without putting the key in the process list. A request without the key may carry a body of at most 64 KiB (it gets 401); a larger one is refused before its body is read |
| `--cors ORIGIN` | localhost pages only | Let web pages from `ORIGIN` (e.g. `https://chat.example.com`) call the server; repeatable, `*` for any. Requests from other sites' pages get 403, so a site you visit can't use your model. Clients that aren't browsers (curl, SDKs, agents) send no `Origin` and are unaffected |
| `--alias NAME` | file name | Model name reported to clients |
| `--gpu-share PCT\|off` | `off` | Keep the GPU busy at most PCT% of the time (5–100). See [Sharing the GPU](#sharing-the-gpu) |
| `--slot-save-path DIR` | none | Keep each slot's conversation cache in `DIR/slot-N` across restarts. See [Conversation caching](#conversation-caching) |
| `--models-dir DIR` | none | Offer the other chat GGUFs in `DIR`, loaded when a request's `model` names one. See [Several models](#several-models) |
| `--idle-unload SECS` | 0 (never) | Free the model after `SECS` seconds without a request; the next request loads it again. See [Conversation caching](#conversation-caching) |
| `-np`, `--parallel N` | `1` | Requests served at once, one slot each (1–8). See [Parallel requests](#parallel-requests) |
| `-c`, `--context N\|auto` | `auto` | Context length of each slot |
| `--kv-cache f16\|q8` | `f16` | K/V cache format (also `-ctk`/`-ctv`). `q8` nearly halves K/V memory, so `auto` fits about twice the context, and decodes faster at long context. See [K/V cache format](#kv-cache-format) |
| `--reserve-mib N` | `1024` | GPU memory left free for the system |
| `--chunk N` | `512` | Prompt tokens per forward pass (`2048` for a model whose experts stream from the SSD) |
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

`--prompt TEXT` (`-p`) answers one message and exits, with the reply on stdout and the statistics on stderr; text piped on stdin is appended to it (`cat notes.txt | ferrum-cli -m model.gguf -p "Summarize:"`). `--gpu-share PCT` works as for the server ([Sharing the GPU](#sharing-the-gpu)). `--version` prints the version, as it does for `ferrum-server`.

In a terminal the prompt has line editing and history (↑/↓, kept in `~/.cache/ferrum/cli-history`, or under `$XDG_CACHE_HOME`). Ctrl-C stops the reply being written and keeps the conversation; at the prompt, Ctrl-C or Ctrl-D leaves. Piped input without `--prompt` is read one message per line, as before.

Commands: `/reset`, `/think on|off`, `/image PATH [low|high|auto|N]`, `/detail low|high|auto|N`, `/stats`, `/exit`. To continue a message on the next line, end the line with `\`. A size after `/image PATH` applies to that image; `/detail` sets it for the images that follow (see [Image size per request](#image-size-per-request)).

## Speculative decoding

Speculative decoding speeds up generation without changing the output. A small, fast drafter guesses the next few tokens, and the main model checks all of the guesses in one forward pass. The output is the same as without speculation, with sampling too: every token is drawn with a random number tied to its seed and position, and a guess is kept only when it equals the token the model draws there, so a seeded request gives the same text with or without a drafter (as in Kapitsa). Guesses are kept as often as rejection sampling would keep them.

Ferrum supports four kinds of drafter:

- **MTP.** The multi-token-prediction head built into the model's own GGUF. Nothing extra to download. On the 35B MoE GGUFs the block routes to its own experts; Ferrum loads their Q3_K experts as Q6_K holding the same values (about 0.3 GB more), and on Tiel-Coder 2 drafts reach 63.4 tok/s (1.47× plain decoding) against 64.8 with the `incoai/Qwen3.6-35B-A3B-DFlash2` drafter ([measurement](measurements/mtp-moe/README.md)).
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
| `--draft-max N` | Most tokens drafted per step, and the depth of draft trees (default: 3 for MTP and lookup, 2 for MTP on MoE targets, 7 for MTP with `--draft-tree`, the trained block size for dense targets, 2 for MoE targets) |
| `--draft-tree N\|auto` | Draft trees of up to N guesses (1–31; 15 recommended) per step, instead of chains. `auto`: KFlash's budget picks each cycle's guesses (up to 31), learning the drafter's calibration and the check's cost as it runs (kept in `~/.cache/ferrum/kflash`); for DFlash2 it also defaults `--draft-block 16` and `--draft-vocab 98304` |
| `--draft-block N` | DFlash/DFlash2: draft N rows per pass (N−1 guesses deep), wider than the drafter's trained block |
| `--draft-vocab N` | Draft among the first N token ids only (the check still uses the full vocabulary) |
| `--draft-tree-search best-first\|siblings` | How MTP grows trees: extending its likeliest guesses (default), or its chain plus each step's next-best guesses as leaves |
| `--draft-quant q4_0\|q8_0` | Drafter weight precision (default `q4_0`) |
| `--draft-context N` | Context slots for drafters without a sliding window (default 8192) |
| `--draft-p-min P` | DSpark: stop drafting below this confidence |

Responses include `draft_n` and `draft_n_accepted` in `timings`, as llama-server does. Gains are smaller on MoE models because each drafted token may route to different experts, so verifying a wide block reads more expert weights. The TUI's Benchmark tab finds the best drafter and depth for your machine automatically.

## Vision (images)

The Qwen3.5-family models can read images once you load their vision projector (the `mmproj*.gguf` published alongside the model). If a projector with a matching width sits in the same folder as the model, Ferrum loads it automatically. It takes about 1 GiB of the memory budget, and the auto-fitted context shrinks to make room. LFM3-VL uses the same flags. The pair checked here is `LFM3-VL-2B-Q4_K_M.gguf` with `mmproj-LFM3-VL-2B-Q8_0.gguf` (that repo publishes no Q4_K_M projector). Its template marker is `<|image|>`, and a prompt that contains `<|video|>` is refused. LFM3-Omni's combined projector is not this path. Limits: [models.md](models.md#lfm3-vl-2b). Inkling 2 Mini reads images through `mmproj-F16.gguf` from the same repo ([models.md](models.md#inkling-2-mini)).

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

### Audio

LFM3-Audio-2B hears speech once its audio projector is loaded with the same `--mmproj` flag (auto-picked when its width matches, preferring the file named after the model; with two same-width projectors, a renamed file needs an explicit path). Send WAV clips inline as OpenAI `{"type": "input_audio", "input_audio": {"data": "<base64>", "format": "wav"}}` parts; `data` may also be a `data:audio/wav;base64,...` URL, and an `{"type": "audio_url", "audio_url": {"url": "data:..."}}` part works too. Ferrum reads PCM and float WAV at any sample rate and channel count (it mixes to mono and resamples to 16 kHz), rejects other formats and remote URLs, and refuses a clip longer than 120 s or more than 8 clips in one request. `/props` reports `modalities.audio`. For transcription, put Liquid's instruction in the system message (`"Perform ASR in English."`) and only the audio in the user message. Encoded clips are cached like images, so a conversation keeps its cache across turns. The speech generator in the same projector is not loaded. A projector that also has a vision tower (LFM3-Omni) is not loaded as audio. Limits: [models.md](models.md#lfm3-audio-2b).

## Image generation

Ferrum draws and edits images with Qwen-Image-2.1 and its Turbo distillation (a Hugging Face directory with `model_index.json`, `transformer/`, `text_encoder/`, `vae/` and `processor/`). The stages run one after another and only one is in memory at a time, so it fits a 32 GB Mac. See [architecture](architecture.md#image-generation) for the design and [models.md](models.md#image-generation) for what is validated and what is not.

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
