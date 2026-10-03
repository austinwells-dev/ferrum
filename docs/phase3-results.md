# Phase 3: official Qwen inference

Ferrum generates text from **Qwen/Qwen2.5-0.5B-Instruct entirely through Rust and its own synchronous Metal kernels**, with BF16 checkpoint/activation storage and F32 kernel arithmetic. This phase adds production integration and measurement, not kernel/execution optimization.

Measured 2026-09-20, Apple M5, arm64 macOS 27.0, Rust 1.96.0. Branch: `codex/phase3-real-model`, based directly on immutable Phase 2 `cdf400c0fa3b9160d7340e0325adfc28fb4fcb8e`. The Phase 1 and Phase 2 commits are unchanged. Baseline verification before edits passed all 20 tests, release build, and transformer smoke: 41 intermediate comparisons, maximum error 4.76837158e-7, cached error zero. Initial unsafe audit found four blocks plus framework linkage.

## Checkpoint and reproduction

Official source: [Qwen/Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/tree/7ae557604adf67be50417f59c2c2f167def9a775), immutable revision **7ae557604adf67be50417f59c2c2f167def9a775**, Apache-2.0. No GGUF, conversion, larger variant, or alternate checkpoint was used for production.

The target was absent from the existing local cache. Acquisition used the external HF CLI:

```sh
hf download Qwen/Qwen2.5-0.5B-Instruct \
  config.json model.safetensors tokenizer.json tokenizer_config.json generation_config.json \
  --revision 7ae557604adf67be50417f59c2c2f167def9a775
export FERRUM_QWEN_MODEL="$HOME/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775"
cargo run --release -- run --model "$FERRUM_QWEN_MODEL" \
  --prompt "What is Rust?" --max-new-tokens 32 --temperature 0 --warmup
cargo test --release --test real_model -- --ignored --nocapture
```

Ferrum accepts any local directory containing those five files, never downloads or contacts the Hub, and does not infer a revision from directory names. Users are responsible for supplying the pinned official snapshot. `hf cache verify` verified all five present files; its five-missing-files warning refers to intentionally omitted repository files. See [checksum verification](measurements/phase3/checkpoint-verification.txt). No `special_tokens_map.json` exists in this revision. Tokenizer vocabulary/merges are self-contained in tokenizer.json; separate vocab.json and merges.txt are unnecessary. Weights, caches, reference virtualenv, and Cargo output are ignored by Git.

## Inspected configuration

| Field | Official value / handling |
|---|---|
| architectures / model_type | `Qwen2ForCausalLM` / `qwen2` |
| vocab_size | 151936 embedding/output rows |
| hidden_size | 896 |
| intermediate_size | 4864 |
| num_hidden_layers | 24 |
| num_attention_heads / num_key_value_heads | 14 / 2; GQA ratio 7 |
| head_dim | 896 / 14 = 64; full split-half rotation |
| rope_theta | 1000000 |
| rms_norm_eps | 0.000001 |
| max_position_embeddings | 32768 |
| tie_word_embeddings | true |
| torch_dtype | bfloat16 |
| hidden_act | silu |
| attention_dropout | 0 |
| use_sliding_window | false; sliding_window=32768 and max_window_layers=21 are inactive |
| use_cache | true |
| Q/K/V bias | Present; required by adapter |
| O / gate / up / down / LM bias | Absent; unexpected tensors rejected |
| config BOS / EOS | 151643 / 151645 |
| tokenizer PAD | 151643 |
| generation stop IDs | [151645, 151643] |

`model::qwen::QwenConfig` deserializes model-specific fields and explicitly converts into Phase 2's `ModelConfig`. It checks architecture, dimensions, divisibility, even head size, GQA, finite positive epsilon/theta, context/kernel indexing, BF16 storage, activation, full attention, and supported optional flags. Tied and untied dense configurations have explicit weight contracts; only the tied target was validated as a production model. Unknown fields fail closed, except a short documented metadata whitelist. Non-null rope scaling, partial rotation, other architectures, changed attention bias, MLP bias, and other execution mechanisms are rejected. Tokenizer integration intentionally narrows support to this checkpoint's vocabulary and policy.

## Exact tensor mapping and orientation

The inspected safetensors archive contains **290 BF16 tensors, 494,032,768 elements, 988,065,536 payload bytes** (about 0.920 GiB). Full names/shapes/bytes for all layers are recorded in [tensors.tsv](measurements/phase3/tensors.tsv).

| Official tensor | Phase 2 canonical name | Shape |
|---|---|---|
| model.embed_tokens.weight | embedding.weight | [151936,896] |
| model.layers.L.input_layernorm.weight | layers.L.input_norm.weight | [896] |
| model.layers.L.self_attn.q_proj.weight / bias | layers.L.q.weight / bias | [896,896] / [896] |
| model.layers.L.self_attn.k_proj.weight / bias | layers.L.k.weight / bias | [128,896] / [128] |
| model.layers.L.self_attn.v_proj.weight / bias | layers.L.v.weight / bias | [128,896] / [128] |
| model.layers.L.self_attn.o_proj.weight | layers.L.o.weight | [896,896] |
| model.layers.L.post_attention_layernorm.weight | layers.L.post_norm.weight | [896] |
| model.layers.L.mlp.gate_proj.weight | layers.L.gate.weight | [4864,896] |
| model.layers.L.mlp.up_proj.weight | layers.L.up.weight | [4864,896] |
| model.layers.L.mlp.down_proj.weight | layers.L.down.weight | [896,4864] |
| model.norm.weight | final_norm.weight | [896] |
| No lm_head.weight in tied target | embedding reused as LM weight | [151936,896] |

`L=0..23`. The adapter validates every official name, dtype, shape, device, required Q/K/V bias, and absence of unknown tensors **before transposition dispatch**. Errors include official names and expected/actual shape/dtype. The safe remapping operation shares immutable tensor handles; it copies no payload. Runtime layers use typed fields, with no per-token string lookup.

Checkpoint linear matrices already use Phase 2's `[output,input]`, `Y=XWᵀ` convention. Phase 2's existing `Linear::new` materializes one `[input,output]` transpose for each of 168 layer linears and the tied LM head: **169 transposes**. Norms, biases, and embedding lookup weights are not transposed. No runtime math convention or shader changed.

## Weight and cache memory

| Payload / allocation measure | Bytes |
|---|---:|
| Official source weight payload | 988,065,536 |
| New construction/transposition allocations | 987,922,432 |
| Retained model weights, summed from actual typed tensors | **1,260,334,848** |
| Retained embedding lookup storage | 272,269,312 |
| Retained tied LM transpose storage | 272,269,312 |
| Retained norms + Q/K/V biases | 143,104 |
| Source + transposes concurrently during construction | 1,975,987,968 |
| Source payload + file-read vector during loading, approximately | 1,976,131,072, plus safetensors header |
| Active KV per cached token, all layers | **12,288** |
| Active KV after Hello generation, 30 cached tokens | 368,640 |
| Active KV after Rust demo, 54 cached tokens | 663,552 |
| Full 32768-token KV payload, not preallocated | 402,653,184 |
| Metal recommended working set | **25,769,803,776** (24 GiB) |
| Process maximum RSS, separate one-token run | 3,053,092,864 |
| Process peak memory footprint, same run | 3,055,192,920 |

Retained weight accounting now sums actual tensors instead of reconstructing expected shapes. The extra tied LM transpose is intentional Phase 2 behavior. Source storage is dropped after construction. The loader's file byte vector is dropped before transposition. These payload measures exclude allocator/driver overhead and process libraries. A separate `/usr/bin/time -l target/release/ferrum run ... --max-new-tokens 1` run reported maximum RSS 3,053,092,864 bytes and peak memory footprint 3,055,192,920 bytes, with zero swaps; see [process memory](measurements/phase3/process-memory.txt). These process-lifetime peaks include loading, construction, library/driver/allocator retention and prefill, and must not be equated with live tensor payload. Allocation counters count successful Metal allocations and cumulative physical bytes, not peak live bytes or DRAM traffic. KV appends allocate/copy active history and retain no capacity arena. The generated terminal EOS (or final token at the limit) is not fed back into the cache.

## Tokenizer, template, generation, and sampling

The existing Rust tokenizers wrapper loads the real tokenizer.json. There are **151665 defined token IDs** and 271 unused padded model rows. We validate contiguous defined IDs and special token roundtrips; equality with embedding-row count would incorrectly reject this official model. An undefined generated padded ID produces an error instead of silently disappearing from output.

The official tokenizer config sets `add_bos_token=false`, `bos_token=null`, `eos_token=<|im_end|>`, `pad_token=<|endoftext|>`, no prefix space, no cleanup, and replacement decoding for invalid byte tails. Config BOS=151643 is **not** automatically inserted. No EOS is appended to raw prompts. Generation stops on either 151645 or 151643; PAD is never added for this unbatched path. The tokenizer's model_max_length=131072 does not override the model's stricter 32768 context capacity.

Ferrum implements the pinned template's single system/user, no-tools branch. It validates exact template text against the inspected revision and emits:

```text
<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
Hello!<|im_end|>
<|im_start|>assistant
```

The final newline is included. This is exactly the official template with an **explicit** system message; the upstream fallback system message mentioning Alibaba is therefore not used. `--system` changes the explicit system content. `--raw` passes text without any chat formatting. Tools, conversations with multiple turns, and a universal Jinja engine are outside this CLI. `--tokenizer-diagnostic` reports input, formatted text, count, IDs, and decode roundtrip.

For Hello, the exact 21 prompt IDs are:

```text
[151644,8948,198,2610,525,264,10950,17847,13,151645,198,
 151644,872,198,9707,0,151645,198,151644,77091,198]
```

Generation prefills once, explicitly extracts row `sequence-1` from `[sequence,vocab]` with the existing checked copy kernel, selects a token on CPU, and feeds only that token through `forward_decode` and Phase 2's KV cache. Tests deliberately put larger values in earlier rows to catch incorrect logit selection. Empty prompts, prompt overflow, checked prompt+max-new overflow, invalid EOS/selection IDs, and malformed options return errors. The reservation check conservatively requires `prompt + max_new <= context`, although the final sampled token is not cached. `--max-new-tokens 0` returns without running prefill. Default is 32; EOS is excluded from displayed text.

Sampling was added after greedy bring-up. Default `temperature=0` uses finite-logit argmax with lowest-ID tie breaking. Positive temperature uses stable F64 CPU probabilities, top-k (`0` disables), then nucleus top-p (`1` disables), then a ChaCha8 RNG seeded with `--seed` (default 0). Invalid temperature/top-p fails; top-k above vocabulary simply keeps all candidates. Same model/prompt/settings/seed and runtime produces the same sequence. Tests cover filters, approximate uniform sampling, extremes, and deterministic repeats. Two independent real-model processes with temperature=.7, top-k=20, top-p=.8, seed=42 returned identical ten IDs; see [repeatability](measurements/phase3/sampling-repeatability.txt). Checkpoint generation defaults, including repetition_penalty=1.1, are **not adopted**: Ferrum exposes the specified small sampler and uses generation_config only for stop/BOS/PAD policy. Reference calls likewise do not apply generation processors.

The CLI streams using tokenizers' incremental decoder, which buffers incomplete UTF-8 fragments. At termination it reconciles with full decode and flushes any final incomplete sequence using the tokenizer's replacement policy. A multi-byte `Hello 世界 🌙 café!` stream/full comparison passes. Model/runtime diagnostics go to stderr; stdout contains generated text. No inference engine, Python subprocess, networking, server, or UI is invoked by the production executable.

## Reference validation and investigated disagreement

Validation-only environment: CPU **PyTorch 2.8.0, Transformers 4.57.6**, eager full attention, four CPU threads, local files only, model.eval(), inference_mode(), no sampling processors. [tools/reference_qwen.py](../tools/reference_qwen.py) and its [locked requirements](../tools/reference-requirements.txt) are external diagnostics, never runtime dependencies. Reproduction:

```sh
uv venv .venv-reference
uv pip install --python .venv-reference/bin/python -r tools/reference-requirements.txt
.venv-reference/bin/python tools/reference_qwen.py --model "$FERRUM_QWEN_MODEL" \
  --steps 8 --output /tmp/qwen-reference.json
cargo run --release --example qwen_probe -- "$FERRUM_QWEN_MODEL"
```

The probe is a validation executable; it records prompt IDs, selected logits, top-10 lists, and eight greedy steps. CPU reference logits/tokens are never read by Ferrum inference. All token IDs and formatted prompt bytes match the trusted template.

First token (BF16): **9707, `Hello`**, both engines. Selected logits:

| ID | Ferrum | Transformers BF16 |
|---|---:|---:|
| 0 | 9.875 | 9.875 |
| 1 | 9.625 | 9.500 |
| 13 | 4.9375 | 4.875 |
| 198 | 1.625 | 1.4453125 |
| 9707 | **24.875** | **24.625** |
| 151643 | 2.96875 | 2.859375 |
| 151645 | -5.90625 | -5.750 |

First-token top-10 candidate sets match. Ferrum's first three are `(9707,24.875), (13048,18.75), (4340,17.25)`; reference: `(9707,24.625), (13048,18.5), (4340,17.25)`. Full lists and subsequent steps are retained in [Ferrum probe](measurements/phase3/ferrum-probe.json) and [reference BF16](measurements/phase3/reference-bf16.json).

BF16 tokens agree for the first five: `[9707,0,2585,646,358]` = `Hello! How can I`. At token six, reference ties `help` (1492) and `assist` (7789) at 25.75 and chooses the lower ID. Ferrum has 25.625 versus 25.75, choosing `assist`. This is a real cross-engine continuation difference, not hidden by a tolerance or asserted token equality. Both then generate `you today`; Ferrum's complete response terminates at EOS.

Investigation checked exact names/shapes/biases, tied embedding, template/BOS, final row, split-half RoPE and positions, GQA index groups, epsilon/theta, and rounding. Upstream BF16 and Ferrum differ at several arithmetic boundaries: Ferrum rounds matmul before separate bias addition, multiplies RMSNorm weight before its final BF16 store, and computes RoPE trigonometry/products in F32 before a single BF16 store. Upstream can fuse bias, rounds normalized values before norm weight multiplication, and rounds cos/sin and rotation products to BF16. Reduction order and BF16 rounding perturbations can accumulate through 24 blocks. A diagnostic reference mode matching those three operation boundaries still exhibits some BF16 differences; it is supporting evidence, not the authoritative reference.

To rule out a structural error, **temporary in-memory F32 diagnostic weights** were used only by `qwen_probe --diagnostic-f32`, compared with Transformers float32 loaded from the same BF16 checkpoint. No F32 checkpoint was saved and production loading remains BF16-only. All eight greedy IDs and every ordered top-10 list matched; maximum recorded selected/top-10 logit absolute error was **7.8201294e-5**. See [comparison](measurements/phase3/reference-comparison.txt), [Ferrum F32](measurements/phase3/ferrum-probe-f32.json), [reference F32](measurements/phase3/reference-f32.json). This validates the architecture independently of BF16 near-tie instability. It does not promise cross-engine BF16 token identity.

The real-model test checks finite full vocabulary logits at every generated step, exact first-five BF16 reference tokens, recorded selected/top-10 values within 0.5 logit units, valid IDs, cached step counts/bytes, EOS and decodable output. The 0.5 comparison is an empirical BF16 regression check, not a universal numerical bound; the separate F32 diagnosis and explicit near-tie accounting are essential evidence. Phase 1/2 tolerances were unchanged.

## First unoptimized performance control

The first warmed baseline was captured **before adding opt-in profiling**. No optimization was made before or after it. Warmup runs one full prefill and one cached decode, discarding output; no timings are blended with model load. Driver/compiler caches and filesystem caches may already be warm. Desktop scheduling is uncontrolled, so the first baseline remains the control even when later runs measure less wall time.

| Measurement | First Hello baseline | Rust demo |
|---|---:|---:|
| Prompt tokens | 21 | 23 |
| Config/tokenizer load | 102.589 ms | 105.039 ms |
| Source weight load | 183.712 ms | 190.543 ms |
| Construction / transposes | 279.840 ms | 278.126 ms |
| Prompt formatting/tokenization | 0.128 ms | 0.090 ms |
| Prefill model wall | **800.050 ms** | 750.848 ms |
| Prefill tokens/sec | **26.248** | 30.632 |
| First-token latency, inference start through selection | **800.578 ms** | 751.380 ms |
| Median cached decode wall including final-row extraction | **761.254 ms** | 751.324 ms |
| Median decode tokens/sec, reciprocal of median step time | **1.314** | 1.331 |
| Generated tokens including EOS where present | 10 | 32 |
| Cached decode calls | 9 | 31 |
| Prefill model dispatches / with final-row extraction | 3795 / **3796** | 3795 / 3796 |
| Decode model dispatches / with final-row extraction | 3843 / **3844** | 3843 / 3844 |
| Prefill allocations including row extraction + ID carrier | **3797** | 3797 |
| Prefill cumulative allocated bytes | **48,459,476** | 53,169,372 |
| Allocations per cached decode including row extraction + ID carrier | **3845** | 3845 |
| Decode allocated bytes, first → last | **5,579,524 → 6,731,524** | 5,867,524 → 10,187,524 |
| Active final KV bytes | **368,640** | 663,552 |
| CPU sampling, all generated tokens | 2.744 ms | 8.121 ms |

The final exact Cargo CLI demonstration (without `--warmup`) also produced the same 32 Rust tokens: prefill 810.741 ms / 28.369 tok/s, first-token latency 811.378 ms, and median cached decode 789.555 ms / 1.267 tok/s. It is a separate run, not a replacement for the first warmed baseline.

Each decode grows active KV by 12,288 bytes, but allocation volume grows by 144,000 bytes/step due to the decomposed attention copies and score tensors. Counters exclude CPU vectors/tokenizer/RNG allocations. Prefill wall excludes final-row copy/readback; first-token latency includes it plus argmax. Decode step wall includes copy/readback and excludes CPU sampling, text decoding/printing, previous output destruction, loading, and tokenization. These definitions must be preserved for Phase 4 comparisons. Full per-step wall times/counters are in [first baseline](measurements/phase3/hello-baseline.txt) and [Rust demo](measurements/phase3/rust-demo.txt).

Observed prompt fixtures (greedy BF16, max-new=32):

| User prompt | Generated text | Prompt tokens | Prefill ms / tok/s | Median decode tok/s | Stop |
|---|---|---:|---:|---:|---|
| Hello! | Hello! How can I assist you today? | 21 | 800.050 / 26.248 | 1.314 | EOS |
| What is 2 + 2? | 2 + 2 is equal to 4. | 27 | 705.295 / 38.282 | 1.490 | EOS |
| Write one sentence about the Moon. | The Moon is a natural satellite of Earth and a celestial body that orbits around the planet. | 26 | 792.631 / 32.802 | 1.417 | EOS |
| What is Rust? | Rust is a high-level, statically typed, compiled programming language. It was created by the Rust developers in 2009. Rust is designed to | 23 | 750.848 / 30.632 | 1.331 | 32-token limit |

These are verbatim model outputs, not claims about factual correctness or model intelligence. Plain-text mode with `The capital of France is` (five tokens) continues ` Paris. It is the largest city in` at the eight-token cap. Logs are in the measurement directory. No MLX/llama.cpp comparison was run; equivalent precision benchmarking is deferred to Phase 4. Transformers timings here are validation diagnostics, not an equivalent engine benchmark.

## Profile and Phase 4 priorities

`--profile --warmup` enables aggregate measurements with no tensor retention and no logging inside operations. Timers separately capture operation wall, output allocation/zeroing, command submission, synchronized dispatch, and Metal GPU timestamps. Disabled profiling adds only checks; it does not change math, waits, allocation strategy, or command buffers. ID-carrier allocation and final tensor destruction are outside individual operation timers. JSON records and the [summary](measurements/phase3/profile-summary.json) are generated with `tools/summarize_profile.py`.

Separate Hello profile: 21-token prefill, eight generated tokens, seven measured cached steps. Means across those seven steps (not the initial baseline's medians):

| Category | Calls/decode | Mean operation wall ms | Mean Metal GPU ms |
|---|---:|---:|---:|
| All matmuls (169 weight projections + 672 attention products) | 841 | 174.751 | 45.083 |
| Select-head copies | 1008 | 168.151 | 13.507 |
| Concatenation, including KV append | 360 | 59.562 | 4.789 |
| Score scaling | 336 | 57.282 | 4.862 |
| Softmax | 336 | 56.619 | 6.271 |
| Causal mask | 336 | 54.204 | 3.971 |
| K transposes | 336 | 54.067 | 4.015 |
| RMSNorm | 49 | 18.222 | 10.228 |
| Bias add | 72 | 12.038 | 1.185 |
| Residual add | 48 | 8.497 | 0.709 |
| RoPE | 48 | 8.273 | 0.822 |
| SwiGLU multiply / SiLU | 24 each | 4.565 / 4.553 | 0.764 / 0.504 |
| Head merge swap | 24 | 4.252 | 0.309 |
| Final-row copy | 1 | 0.311 | 0.080 |
| Embedding gather | 1 | 0.185 | 0.054 |

Summed operation wall **685.530 ms**, GPU **97.152 ms**, synchronized dispatch **671.853 ms**, command submission **18.769 ms**, and output allocation/zeroing **10.944 ms** per decode. Synchronized-minus-GPU is **574.701 ms**; this includes submission, scheduling, completion wakeups and other host/driver overhead, not a pure CPU wait measurement. GPU timestamps measure command-buffer duration and are not isolated hardware counters. They were available for every profiled dispatch. The categories overlap: submission is inside synchronized time; allocation is inside operation wall. Do not add them as independent costs.

Layout copies (`select_head`, transpose, concat, swap) total **286.031 ms wall / 22.619 ms GPU**. Much of this cost is their completion boundaries. Attention is decomposed into thousands of these operations plus 672 matmuls; matmul totals above mix attention with projections and do not isolate GEMV compute. KV append accounts for 48 of 360 concatenation calls; its precise time is not separated from context concatenation. Using the concat mean call cost gives roughly 7.9 ms/decode as an estimate, not a directly isolated measurement. Prefill summed operation wall/GPU in this profile was 704.405/127.350 ms.

Ranked Phase 4 recommendations, **none implemented here**:

1. Design completion/lifetime ownership, then remove per-operation waits and reduce command buffers. The 574.7 ms dispatch/completion gap dominates 97.2 ms GPU execution; preserving safe buffer lifetimes is prerequisite.
2. Eliminate repeated head layout copies/concatenation dispatches and batch attention work. Layout operations cost 286.0 ms/decode wall; current attention alone makes 1008 select-head dispatches and 336 K transposes.
3. Profile projection versus attention matmuls separately, then evaluate decode-specialized GEMV and better Metal matmul/native lower-precision arithmetic. Matmul's 45.1 ms is the largest measured GPU category. Avoid assuming a bandwidth bottleneck from payload byte counts.
4. Parallelize reductions where supported by measurements: RMSNorm consumes 10.2 ms GPU across 49 calls; softmax 6.3 ms across 336 short rows. Their relative priorities change with context length.
5. Add safe storage reuse/arenas after the lifetime model. There are 3845 allocations/token but only 10.9 ms measured output allocation/zeroing here; do not rank allocation ahead of the much larger synchronization cost without further evidence. KV append copy volume grows with context and deserves a longer-context control.
6. GPU-resident logit selection/sampling is a later target. CPU greedy selection averaged approximately 0.27 ms/token in the initial baseline; the row copy is one dispatch. Stochastic sorting is more expensive, but still not the current dominant cost.

## Primitive regression investigation

The first Criterion rerun flagged regressions relative to its earlier stored
samples (including +8% for F32 256×512×256 and a noisier large-add result).
We therefore built immutable `cdf400c` in a separate detached worktree and ran
all 13 cases, followed immediately by Phase 3 with profiling disabled. No
performance code was changed in response. These are short, non-interleaved
controls on an active desktop, not proof of zero overhead.

| Case | Phase 2 control µs | Phase 3 control µs |
|---|---:|---:|
| Add 1024 | 195.97 | 183.68 |
| Add 65536 | 201.20 | 201.24 |
| Add 1048576 | 504.77 | 496.91 |
| RMSNorm 1×128 | 219.33 | 236.41 |
| Softmax 1×128 | 224.04 | 223.10 |
| RMSNorm 4×4096 | 849.23 | 883.74 |
| Softmax 4×4096 | 1007.5 | 1014.1 |
| RMSNorm 32×4096 | 912.67 | 926.60 |
| Softmax 32×4096 | 1160.3 | 1135.6 |
| Matmul F32 17×19×23 | 168.79 | 181.48 |
| Matmul F32 128×128×128 | 210.23 | 207.11 |
| Matmul F32 256×512×256 | 339.89 | 330.25 |
| Matmul F16 256×512×256 | 333.62 | 325.25 |

The separate instrumented medians for the apparent small-case regressions are
202.750→204.292 µs (RMSNorm 1×128, GPU 21.5→21.5 µs) and
178.125→178.625 µs (small matmul, GPU 2.5→2.458 µs). Large F32 matmul measured
319.666→317.792 µs instrumented wall and 127.125→126.958 µs GPU. Larger
reduction timings varied in both GPU and host time despite identical shaders.
The investigation did not establish a systematic runtime regression attributable
to the disabled profiler. Residual desktop variance and small overhead remain
measurement limitations; the raw flagged run is preserved rather than discarded.
See [Phase 2 control](measurements/phase3/phase2-control-benchmark.txt) and
[Phase 3 control](measurements/phase3/phase3-control-benchmark.txt).

## Validation, safety, and limitations

Normal tests never download/load ~1 GB. Seven new ordinary suites cover config/rejection, exact name mapping with tiny payloads, errors, final row/argmax, sampling, stopping/cache limits, profiling, and special-token metadata. The ignored real-model test explicitly requires `FERRUM_QWEN_MODEL`; it runs full BF16 inference and reference checks. Synthetic Phase 1/2 tests are retained. Tests exercise actual Metal, not a fallback.

The only changes to primitive/runtime infrastructure are opt-in timing aggregation, actual weight-memory accounting, contextual weight errors, and safe construction-time name remapping. No shader, GEMM, RoPE, attention, normalization math, queue, wait, buffer ownership, or cache strategy was optimized. The primitive benchmark was rerun because profiling touches the public operation wrapper.

Unsafe audit remains **four blocks plus one framework-linkage declaration, all in src/metal/mod.rs**. No new unsafe code, Send/Sync impl, Arc conversion, asynchronous work, or background Metal execution was added. `Rc`, immutable published tensors, and completion-before-return are unchanged. See [audit](measurements/phase3/unsafe-audit.txt).

All required gates passed: **27 ordinary tests**, one intentionally ignored real-model test in ordinary runs, **9 Phase 1 + 8 Phase 2 suites with both Metal validation layers**, and the explicit BF16 real-model test. Formatting, warning-denying Clippy, release build, device info, both smoke commands, all 13 primitive benchmark cases, and the final Cargo CLI demonstration passed. Raw evidence:

- `cargo fmt --check`, `cargo clippy --all-targets --all-features -- -D warnings`, `cargo test`, `cargo build --release`, `info`, `smoke`, `transformer-smoke`: [quality gates](measurements/phase3/quality-gates.txt).
- Both Phase 1 and Phase 2 Metal API/shader validation suites: [Metal validation](measurements/phase3/metal-validation.txt).
- Explicit release real-model integration: [real model](measurements/phase3/real-model-test.txt).
- Primitive regression benchmark: [benchmark](measurements/phase3/primitive-benchmark.txt).
- Final exact Cargo command: [final CLI demonstration](measurements/phase3/final-cli-demo.txt); final repeat of [real-model test](measurements/phase3/real-model-test-final.txt).
- Real CLI fixtures, tokenizer diagnostic and seeded repeatability: [measurement directory](measurements/phase3/).

Known limits: only this dense target is production-validated; exact pinned chat-template policy; single system/user turn; synchronous allocations/copies; no long-context performance or accuracy validation beyond the existing synthetic position-2048/2049 tests; CPU sampling; tensor payload accounting distinct from measured process RSS; BF16 arithmetic can change near-tie greedy choices compared with Transformers. F32 accumulation is intentional despite BF16 storage, and intermediate BF16 stores remain visible numerical boundaries. Small desktop measurements have scheduling/thermal variance. Phase 3 stops after validation and commit; Phase 4 optimization is not part of this result.
