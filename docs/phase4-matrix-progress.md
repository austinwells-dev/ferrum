# Phase 4 broader workload optimization — in progress

The expanded user goal supersedes the short-context completion assessment in `phase4-results.md`. Phase 4 is active: the 100 tok/s decode and 1,000 tok/s approximately 1k-token prefill targets are directions, not stopping criteria. Remaining major bottlenecks still need systematic matrix evaluation and concrete experiments.

## Reproducible matrix

`cargo run --release --example runtime_matrix -- "$MODEL" 3` uses the pinned official Qwen2.5-0.5B BF16 safetensors and production `generation::generate`. Cases: 21-token templated Hello, 128/512/1024-token deterministic prose prefixes, and the short prompt with 128 cached decode calls. Other cases have 16 cached decode calls. EOS stopping is disabled for fixed work, with ordinary greedy selection; model arithmetic is unchanged. Each case warms via a complete generation and discards its cache. Prefill, final-row extraction, selection and cached generation keep their normal timing boundaries; no inference work is moved into prompt construction or reused from warmup.

Raw JSONL includes exact prompt/generated IDs, per-step latency, GPU/encode/wait/allocation durations, physical allocations/new/reused bytes, submissions/waits, transient peak and arena capacity. `tools/summarize_matrix.py` produces median latency and mean counters while retaining peaks. Per-operation profiling requires `FERRUM_MATRIX_PROFILE=1`; `FERRUM_BATCH_LIMIT=1` isolates GPU categories, changing execution and timing overhead. Do not add isolated totals to batched totals.

## Current experiments

| Experiment | Short prefill ms | 128 ms | 512 ms | 1024 ms | 1024 peak transient bytes |
|---|---:|---:|---:|---:|---:|
| h-baseline, previous full logits | 38.23 | 197.87 | 947.76 | 2394.73 | 5,907,677,184 |
| h-last, final-position logits | 32.54 | 156.47 | 778.93 | 2060.00 | 5,371,330,560 |
| h-bounded, 64 MiB epoch budget | 32.49 | 145.72 | 730.41 | see raw log | 147,587,072 |
| h-bounded256, 256 MiB epoch/pool | 35.74 | 164.49 | 771.16 | 2020.56 | 268,435,456 |
| h-fused, scale/mask/softmax | 34.64 | 163.79 | 771.22 | 2000.61 | 267,649,024 |

These runs are sequential on a desktop and short/decode performance drifted during the series; final matched controls and repetition remain necessary. They are not confidence intervals. All raw data are under `measurements/phase4/h-*`. The complete matrix includes decode metrics even where this compact table shows only prefill. In the initial baseline, short decode was ~93 tok/s and 1k decode ~55 tok/s; later series were ~87 and ~51 tok/s. The memory changes do not add a completion boundary to short decode, so thermal/system variation versus regression needs a matched control rather than attribution from this table alone.

### Requested logits only

`forward_prefill_last` computes every transformer hidden position and all KV updates, then selects the final hidden row before LM projection. Production generation requests this API because it only consumes those logits. Existing `forward_prefill`/`forward` continue returning all requested position logits. This is an ordinary output contract optimization, independent of prompt content/length. New tests compare last-position values with full output at existing dtype tolerances and verify exact equality of subsequent cached decode. The existing suite and real-model tests pass; generated IDs across all matrix cases match the baseline exactly.

### Bounded in-flight transient storage

Before allocating another output, the runtime completes the current encoding epoch if live transient bytes plus the requested allocation exceed a fixed budget. This makes retired intermediates eligible for reclamation and reuses the existing completion/failure contracts. It is a soft working-set bound: individually necessary live input/output tensors may exceed it, especially for APIs returning all-position logits. Attention remains quadratic in sequence length, so this does not claim arbitrary-context constant memory.

64 MiB forced 145 submissions at 1k and repeatedly allocated large intermediates, regressing latency. The 256 MiB variant uses 21 submissions at 1k (15 after fusion) and bounds the measured working set without retaining every layer's intermediates. The pool's retention cap is also 256 MiB. Short decode remains one submission. Existing transaction/lifetime tests and real-model Metal API/shader validation passed.

A deterministic largest-first pool trimming experiment (`h-trim-*`) tried retaining part of an over-capacity pool instead of releasing completed excess storage together. It increased large-shape allocation volume and regressed 512/1k latency. Reverted; patch preserved as evidence. The 64 MiB epoch variant is likewise not retained.

### Attention scale/mask/softmax fusion

The new kernel computes causal visibility and scaled scores inside the existing stable parallel softmax. It explicitly rounds scaled values through the original storage dtype before reduction, preserving the separate scale kernel's numerical boundary. This removes two materialized attention-score intermediates and 48 dispatches per model forward. Trace collection and explicit reference-math diagnostics retain staged operations when those intermediate values are requested.

A direct unit test verifies exact staged/fused equality for F32/F16/BF16 with multiple sequence lengths and causal offsets, including a 1k prefix. Existing tests/tolerances are unchanged. All matrix generated IDs match the original candidate. Throughput benefit is modest so far; lower intermediate traffic and completion count are measured, but a matched control is still needed before final acceptance.

## Outstanding work

- Measure long-prefill/long-decode GPU categories; investigate larger GEMM tiles and reuse with matrix-wide evaluation.
- Investigate attention products and decode memory access, further GEMV improvements, KV growth behavior, and useful low-risk fusion.
- Matched controls for provisional memory/fusion choices; reject changes that materially regress the runtime.
- Final complete matrix, numerical/F32/failure/lifetime/Metal validation, and accepted/rejected experiment report.
- Audit the full expanded goal before claiming completion. No quantization, GGUF, alternate model or speculative decoding work.

### Latest category evidence and validation

The isolated 1k prefill profile (`h-fused-isolated.jsonl`) measures transformer projections 952.16 ms, attention scores 451.30 ms and attention context 447.92 ms. Fused softmax is 27.69 ms, RMSNorm 11.08 ms and LM head GEMV 2.15 ms. Thus the next major experiments must address both GEMM and attention products, unlike the short-prompt case. Isolated timing changes scheduling and is attribution evidence rather than normal production latency.

The current candidate passes Metal API/shader validation for the full backend/correctness/transformer/runtime optimization suites and both official real-model tests. The unchanged F32 diagnostic still passes its existing 8e-5 tolerance. Logs: `h-fused-validation.txt`, `h-fused-real-validation.txt`, `h-fused-f32-check.json`, `h-fused-clippy.txt`. This is an intermediate checkpoint, not goal completion.
