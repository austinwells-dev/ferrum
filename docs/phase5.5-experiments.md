# Phase 5.5 Experiments

## Goal and status

Phase 5.5 compares Ferrum’s supported GGUF quantized and MLX affine-Q4 paths with current llama.cpp Metal and native MLX/MLX-LM on the same Apple M5. This phase starts from clean Phase 5 commit `b47f03367672b62b2b5208242dc467c902dc9ff7`. No Phase 6 work is included.

Current status: Phase 5.5 is suspended at a clean performance checkpoint and is not declared performance-complete. The three-pair matched matrices cover Q4_0, Q4_K_M, Q5_K_M, Q6_K, Q8_0, and native MLX affine Q4. F16 MPP attention, the wide-output Q4_0 two-row M=1 GEMV, and the guarded wide/short-K affine-Q4 quad GEMV are retained. The attempted Q5_0/Q5_1 row-sharing candidate was rejected and removed after synchronized GPU profiles showed regressions. The major remaining deficit is quantized M=1 projection/GEMV performance; prefill also remains behind mature reference runtimes. This checkpoint makes no parity claim. Short smoke numbers are sanity checks only and are not reported as benchmark results.

## Host and pinned artifacts

- Host: Apple M5, Mac17,2, 32 GB unified memory, macOS 27, Xcode command line tools 72.
- Ferrum starting revision: `b47f03367672b62b2b5208242dc467c902dc9ff7`.
- llama.cpp source: `ggerganov/llama.cpp`, commit `fee39dd92673ba0c08c8da96040ce53368b35188` (2026-09-23, 0.5.0-dev), built Release with Metal enabled, embedded library enabled, OpenMP and tests disabled.
- MLX source inspected at `ml-explore/mlx` commit `59d600b5e64c238427d0f8d897ab7c682ef4d3d2`.
- MLX-LM source inspected at `ml-explore/mlx-lm` commit `15bcf8b929e5da67aa7f8fe6feda7fb9eda00050`.
- Native runtime: `mlx==0.32.2`, `mlx-metal==0.32.2`, `mlx-lm==0.31.3`.
- GGUF model family: Qwen/Qwen2.5-0.5B-Instruct-GGUF at `9217f5db79a29953eb74d5343926648285ec7e67`; exact artifact hashes and sizes are recorded in `docs/measurements/phase5/model-source.json`.
- Native affine-Q4 model: mlx-community/Qwen2.5-0.5B-Instruct-4bit at `a5339a4131f135d0fdc6a5c8b5bbed2753bbe0f3`; `model.safetensors` SHA-256 `ddffab9cbc7bf6dde941c6724841eeca8981fcfa81ca20ff8efff1396326d153`.

## Matched measurement method

`tools/phase55_matched_matrix.py` reads the prompt token IDs and generation lengths from the saved Phase 5 workload matrices. Each engine receives those IDs directly. Ferrum and llama.cpp receive the same GGUF file; Ferrum and native MLX receive the same affine-Q4 model directory. GGUF and MLX comparisons are separate runs. Each engine/model is loaded once, and each workload is warmed before measurement. Three paired repetitions are interleaved with alternating engine order. Output rows retain the case, repetition, order, exact prompt IDs, generated IDs, timing fields, and available memory/profile counters.

An input workload row may include a `request_options` object. The runner copies it into that case's warmup and measured requests, allowing a fixed option to be held constant while `--ferrum-paired-option` changes a separate candidate/control field. Duplicate rows for the same case must agree on those options.

The workloads are the Phase 5 `short`, `128`, `512`, `1024`, `sustained-128-decode`, and `long-horizon-1601` cases. Their input comes from the existing quantized rows, so prompt lengths, prompt IDs, and requested output lengths remain fixed across runtimes. The tokenizer is bypassed during matched runs.

Ferrum uses its default `FERRUM_BATCH_LIMIT=1024`, which limits kernel dispatches per command buffer. llama.cpp uses `n_ubatch=512`, which limits prompt tokens per microbatch; these controls have different units and are recorded separately. The llama.cpp reference harness uses current upstream Metal, 99 GPU layers, BF16 KV, a 4096-token context, batch size 2048, flash attention, four CPU threads, and greedy argmax. Native MLX uses `mlx_lm.generate_step`, its default 2048-token prefill step, a fresh prompt cache per sample, and greedy argmax.

Ferrum reports prompt prefill separately from decode and full generation. llama.cpp reports prompt processing, first-token latency, decode, and total generation using synchronized C API calls. Native MLX reports prefill chunks from its progress callback and records first-token, decode, and total timings around `generate_step`. MLX-LM schedules one look-ahead token; the harness captures the requested output timing boundary, then synchronizes that unused next-token computation before the following sample so it cannot spill into the next pair. These boundaries are not identical: timing comparisons will name the field and exclude load/warmup. Startup and first warmup are recorded separately. Results will be summarized per case using paired medians and spread, and generated token IDs will be retained to expose any output divergence.

The matched prompt IDs make input work comparable; they do not guarantee identical generations after the first divergent token. Throughput will therefore be treated as runtime performance on the saved workload, alongside generated-token agreement, rather than as a claim that all engines followed the same autoregressive trajectory.

## Files and commands

- Ferrum JSONL runner: `examples/phase55_bench.rs` (built as `target/release/examples/phase55_bench`).
- llama.cpp C API runner: `tools/phase55_llama_matrix.cpp` (built against the pinned external source tree).
- Paired orchestrator: `tools/phase55_matched_matrix.py` (run with the pinned native MLX Python environment).
- Raw measurements and run logs will be stored under `docs/measurements/phase5.5/`.

## Results

### GGUF cross-format summary

These are medians of the paired Ferrum-to-llama.cpp throughput ratios over the six workloads and three pairs per workload. Each workload contributes one value per pair, so the summary weights cases equally. Exact output matches compare complete generated ID sequences within a pair.

| GGUF format | Prefill ratio | Decode ratio | Full generation ratio | Exact output matches |
| --- | ---: | ---: | ---: | ---: |
| Q4_0 | 0.494 | 0.470 | 0.493 | 15 / 18 |
| Q4_K_M | 0.513 | 0.485 | 0.505 | 12 / 18 |
| Q5_K_M | 0.495 | 0.488 | 0.501 | 12 / 18 |
| Q6_K | 0.429 | 0.603 | 0.571 | 15 / 18 |
| Q8_0 | 0.378 | 0.636 | 0.595 | 12 / 18 |

Raw matrices are `q4_0-ferrum-vs-llama-current-after-8rows.jsonl`, `q4_k_m-ferrum-vs-llama-current.jsonl`, `q5_k_m-ferrum-vs-llama-current.jsonl`, `q6_k-ferrum-vs-llama-current.jsonl`, and `q8_0-ferrum-vs-llama-current.jsonl` under `docs/measurements/phase5.5/`.

### MLX affine Q4 vs native MLX-LM baseline

Ferrum and native MLX-LM used the same pinned group-size-64 affine-Q4 model directory. The table shows median throughput in tokens per second as Ferrum / native MLX-LM; output lengths and prompt IDs were identical. Native MLX-LM and the saved artifacts report `mlx==0.32.2` and `mlx-lm==0.31.3`.

| Case | Prefill tok/s | Decode tok/s | Full generation tok/s | Paired generation ratio |
| --- | ---: | ---: | ---: | ---: |
| `short` | 855 / 1,876 | 121 / 316 | 104 / 255 | 0.408 |
| `128` | 2,764 / 6,254 | 122 / 311 | 93.6 / 220.5 | 0.426 |
| `512` | 1,750 / 13,729 | 114 / 300 | 39.0 / 174.3 | 0.221 |
| `1024` | 1,010 / 14,058 | 105 / 289 | 14.5 / 125.0 | 0.115 |
| `sustained-128-decode` | 825 / 1,868 | 117 / 311 | 112.4 / 302.0 | 0.373 |
| `long-horizon-1601` | 1,892 / 12,198 | 102 / 266 | 98.8 / 264.0 | 0.374 |

This is the pre-optimization reference matrix. Across all 18 paired samples, the median Ferrum/native throughput ratios are 0.251 for prefill, 0.381 for aggregate decode, and 0.374 for full generation. All 18 pairs produced exactly the same token IDs. The long-prompt prefill gap is the most severe: Ferrum reaches 12.5% of native MLX-LM at 512 tokens and 7.1% at 1,024 tokens. Native decode aggregate rates are derived from the recorded output-token intervals; the runner now emits this derived field for future runs. Raw rows are in `docs/measurements/phase5.5/mlx-affine4-ferrum-vs-native-mlx-lm.jsonl`.

### F16 attention MPP result

After adding F16 grouped MPP score and context kernels, the full three-pair matrix retained exact token-ID agreement with native MLX-LM in all 18 pairs. The table shows median tokens per second as Ferrum / native MLX-LM and the prefill change within Ferrum versus the baseline above.

| Case | Prefill tok/s | Ferrum prefill change | Full generation tok/s | Paired generation ratio |
| --- | ---: | ---: | ---: | ---: |
| `short` | 844 / 1,888 | -1% | 103.5 / 257.2 | 0.402 (0.400–0.408) |
| `128` | 3,987 / 6,144 | +44% | 100.2 / 221.5 | 0.460 (0.408–0.463) |
| `512` | 5,836 / 13,774 | +234% | 73.1 / 175.6 | 0.416 (0.387–0.420) |
| `1024` | 5,295 / 14,329 | +425% | 48.6 / 126.6 | 0.384 (0.362–0.388) |
| `sustained-128-decode` | 797 / 1,570 | -3% | 111.1 / 272.1 | 0.390 (0.365–0.408) |
| `long-horizon-1601` | 4,284 / 11,258 | +126% | 97.0 / 240.2 | 0.396 (0.360–0.404) |

The paired median full-generation ratio over all cases rose from 0.374 to 0.401. Aggregate decode remained about 0.38 of native MLX-LM, so M=1 quantized projections were the next target. The post-change raw matrix is `docs/measurements/phase5.5/mlx-affine4-ferrum-vs-native-mlx-lm-after-f16-attention-mpp.jsonl`.

### Affine-Q4 eight-row M=1 GEMV

The new kernel maps each four-lane quad to eight output rows. Each lane loads a 16-value activation fragment once and reuses it across those rows, following the shape of MLX's affine `qmv_quad` path. The first broad experiment showed regressions for small-output Q/K/V and long-K down projections, so the retained policy selects the kernel only when `N >= 2048` and `K <= 2048`. For the pinned Qwen model, that selects the wide gate/up projections and LM head, and leaves Q/K/V/down on the existing kernel.

| Case | Prefill tok/s Ferrum / native | Decode tok/s Ferrum / native | Full generation tok/s Ferrum / native | Paired generation ratio | Ferrum generation change vs F16 attention matrix |
| --- | ---: | ---: | ---: | ---: | ---: |
| `short` | 825 / 1,806 | 126 / 294 | 108.7 / 239.8 | 0.455 (0.423–0.455) | +5.0% |
| `128` | 3,926 / 5,836 | 127 / 292 | 102.2 / 206.7 | 0.491 (0.455–0.506) | +1.9% |
| `512` | 5,608 / 13,646 | 120 / 282 | 74.9 / 167.1 | 0.434 (0.423–0.449) | +2.5% |
| `1024` | 5,184 / 13,762 | 110 / 292 | 48.9 / 123.6 | 0.393 (0.367–0.400) | +0.7% |
| `sustained-128-decode` | 861 / 1,814 | 126 / 311 | 119.6 / 300.2 | 0.404 (0.387–0.413) | +7.7% |
| `long-horizon-1601` | 4,694 / 12,296 | 112 / 279 | 108.8 / 277.9 | 0.392 (0.388–0.415) | +12.2% |

Across all 18 paired samples, the final median Ferrum/native ratios are 0.411 for prefill, 0.415 for aggregate decode, and 0.419 for full generation. Ferrum produced exactly the same IDs as the post-F16 baseline and native MLX-LM in all 18 pairs. In a synchronized short-prompt profile, the guarded candidate stages recorded 6.10 ms for up, 5.13 ms for gate, and 0.99 ms for the LM head, versus 6.90 ms, 5.87 ms, and 4.70 ms on the original path; Q/K/V/down used the original path. Profile values are diagnostic samples; the matched matrix is the latency evidence. Raw results are `docs/measurements/phase5.5/mlx-affine4-ferrum-vs-native-mlx-lm-after-affine-quad-policy.jsonl`, and the guarded profile is `docs/measurements/phase5.5/profiles/mlx-affine4-stage-profile-after-quad-policy.jsonl`.

### Upstream kernel survey

At the pinned llama.cpp revision, `ggml/src/ggml-metal/kernels/mul_mv.metal` implements Q4_0, Q5_0, and Q5_1 through `mul_vec_q_n_f32_impl`; `ggml-metal-impl.h` sets four output rows per SIMD group for these formats. The generic kernel loads and rearranges an activation fragment once, then reuses it while accumulating several weight rows. The Q5_K and Q6_K implementations also keep per-row accumulators while reusing activation fragments across rows. These ideas informed the retained Q4_0 candidate and the rejected Q5 candidate documented below.

At the pinned MLX revision, `mlx/backend/metal/quantized.cpp` chooses quantized matrix/vector paths using M, K, N, transpose state, and Apple GPU generation. The `dispatch_qmv` path selects `qmv_quad` for K=64 or 128, and `mlx/backend/metal/kernels/quantized.h`'s affine `qmv_quad_impl` loads an activation fragment once and accumulates up to eight output rows per quadgroup. This is directly relevant to the pinned group-size-64 affine-Q4 decode path. The MLX-LM generation loop uses MLX's lazy evaluation and cache policy; the comparison harness calls its native `generate_step` path and measures that runtime separately.

Ferrum's retained F16 attention, Q4_0 two-row, and affine-Q4 quad kernels are native implementations using Ferrum's existing Metal and Rust interfaces. The work reuses dataflow ideas from upstream; it does not copy upstream source code. No new third-party runtime dependency was added.

### Rejected Q5_0/Q5_1 two-row-per-SIMD candidate

A two-row-per-SIMD M=1 candidate for Q5_0 and Q5_1 passed deterministic numerical checks for both formats, a 131-row output tail, and a five-block K dimension. The synchronized real-model stage profiles showed the candidate was substantially slower in the Q4_K_M and Q5_K_M models, so it was removed before a full latency matrix:

| Model | Baseline Q5 GEMV GPU time across profiled projections | Candidate time | Change |
| --- | ---: | ---: | ---: |
| Q4_K_M (Q5_0) | 5.18 ms | 8.16 ms | +58% |
| Q5_K_M (Q5_1) | 5.27 ms | 8.73 ms | +66% |

These are one-token diagnostic profiles with a synchronized dispatch limit of one, not production throughput measurements. The candidate doubled per-thread accumulators and still decoded a separate weight row; the extra Q5 decode work appears to outweigh the activation-load reuse. Register pressure is a possible contributor but was not measured directly. The candidate tests and production kernels were removed; the four raw before/after profiles are retained in `docs/measurements/phase5.5/profiles/` for traceability.

### Stage profile diagnosis

To attribute GPU time by projection role, the opt-in profile tags label Q/K/V, attention output, gate/up/down, and LM-head operations. Diagnostic runs set Ferrum's dispatch limit to one so every kernel has a synchronized GPU duration. This deliberately changes command-buffer behavior; use these rows to attribute device work, not as latency measurements. The diagnostic input uses the saved `short` and `1024` prompts but limits output to two tokens.

Before the F16 attention MPP change, the 1,024-token affine-Q4 profile recorded 422.4 ms in `attention_scores` and 412.0 ms in `attention_context` across 24 layers. Those F16 operations were using the scalar kernels because the existing MPP dispatch accepted only BF16. By comparison, the Q4_0 BF16 profile at the same prompt length spent 7.1 ms in MPP score products and 8.1 ms in MPP context products. The affine profile also shows about 101 ms total across the three MPP feed-forward projections (`gate_proj`, `up_proj`, and `down_proj`). These profiles are in `docs/measurements/phase5.5/profiles/`.

The initial optimization adds F16 variants of the grouped MPP score and context kernels and dispatches them for F16 inputs on MPP-capable devices. They retain F16 output rounding and share the existing grouped layout. The tiled-attention test now checks BF16 and F16 against the scalar kernels for nonaligned sequence lengths and column tails. After the change, the synchronized 1,024-token diagnostic profile records 7.1 ms for F16 MPP scores and 8.0 ms for F16 MPP context, down from 834.4 ms combined on the scalar path. The remaining profiled work is led by the three MPP feed-forward projections (about 103 ms combined) and attention softmax (about 29 ms). The before/after profiles are retained in the same folder.

### Q4_0 vs current llama.cpp Metal baseline

This corrected run uses Ferrum's default dispatch limit and three interleaved pairs per case. Each cell lists Ferrum / llama.cpp median throughput. The final column gives the median of the three paired full-generation throughput ratios, with the observed min–max in parentheses.

| Case | Prefill tok/s | Decode tok/s | Full generation tok/s | Paired generation ratio |
| --- | ---: | ---: | ---: | ---: |
| `short` | 765 / 1,510 | 110 / 257 | 95.8 / 211.0 | 0.453 (0.446–0.455) |
| `128` | 3,562 / 6,917 | 109 / 247 | 91.3 / 192.1 | 0.475 (0.418–0.480) |
| `512` | 5,151 / 9,643 | 103 / 255 | 65.9 / 140.4 | 0.469 (0.431–0.471) |
| `1024` | 4,802 / 9,011 | 97.8 / 247 | 44.8 / 93.2 | 0.480 (0.455–0.485) |
| `sustained-128-decode` | 755 / 1,457 | 106 / 259 | 102.1 / 238.4 | 0.428 (0.426–0.436) |
| `long-horizon-1601` | 4,373 / 9,514 | 96.0 / 246 | 93.5 / 230.3 | 0.406 (0.405–0.408) |

All 18 pairs used identical saved prompt IDs and requested token counts. Generated IDs matched exactly in 15/18 pairs. The three sustained-decode pairs shared the first 63 output tokens and then diverged; the other cases matched their full generated sequences. This establishes a substantial baseline gap to investigate, especially in decode. Raw rows are in `docs/measurements/phase5.5/q4_0-ferrum-vs-llama-current.jsonl`.

The first Q4_0 and Q4_K_M captures set Ferrum's dispatch limit to 512 based on a mistaken assumption that this matched llama.cpp's 512-token microbatch. Those complete captures are retained under `docs/measurements/phase5.5/rejected/` for traceability and excluded from analysis; corrected runs use Ferrum's default. Do not use Phase 5 llama-bench rows or Phase 5.5 smoke runs as matched comparison results.

### Q4_0 two-row-per-SIMD M=1 GEMV

The baseline Q4_0 M=1 kernel assigned one SIMD group to one output row, repeating activation loads for every row. The candidate assigns two output rows to each SIMD group, reusing each activation value across both rows; four SIMD groups produce up to eight rows per threadgroup. It retains the existing one-row kernel for outputs narrower than 128 rows and handles output tails. The candidate was compared in the full six-workload, three-pair Q4_0 matrix before becoming the default for wide outputs.

| Case | Ferrum prefill tok/s before → after | Ferrum decode tok/s before → after | Decode ratio vs llama.cpp after | Ferrum generation tok/s before → after | Generation ratio vs llama.cpp after |
| --- | ---: | ---: | ---: | ---: | ---: |
| `short` | 765 → 759 | 110 → 126 | 0.498 (0.493–0.501) | 95.8 → 106.0 | 0.507 (0.507–0.518) |
| `128` | 3,562 → 3,635 | 109 → 124 | 0.496 (0.484–0.543) | 91.3 → 98.2 | 0.497 (0.473–0.515) |
| `512` | 5,150 → 5,131 | 103 → 117 | 0.468 (0.465–0.471) | 65.9 → 70.6 | 0.505 (0.463–0.506) |
| `1024` | 4,802 → 4,821 | 97.8 → 108 | 0.439 (0.434–0.439) | 44.8 → 46.6 | 0.501 (0.469–0.502) |
| `sustained-128-decode` | 755 → 630 | 106 → 120 | 0.471 (0.469–0.471) | 102.1 → 114.4 | 0.485 (0.483–0.488) |
| `long-horizon-1601` | 4,372 → 4,385 | 96.0 → 108 | 0.441 (0.439–0.442) | 93.5 → 104.2 | 0.457 (0.454–0.457) |

Decode improved 10.7–14.7% across all six cases; full-generation throughput improved 4.1–12.1%. Prefill remained within measurement variation except for the sustained case, whose prefill rate fell 17% in the candidate run while decode improved 13%; this diagnostic run does not change the prefill implementation. The candidate produced the same Ferrum output IDs as the original kernel in all 18 pairs, including all 129 tokens of each sustained-decode output. Its full output matched llama.cpp in 15/18 pairs, the same agreement count as the baseline. This supports retaining it for M=1 decode. The raw candidate matrix is `docs/measurements/phase5.5/q4_0-ferrum-vs-llama-current-after-8rows.jsonl`; the before/after synchronized profiles are in `docs/measurements/phase5.5/profiles/`.

## Final checkpoint matrices and remaining work

The following are the final paired-median throughput ratios after retaining the validated candidates. Each value is Ferrum / reference; GGUF rows compare against the pinned llama.cpp Metal build and affine-Q4 compares against native MLX-LM. Overall values take the median of paired ratios across six workloads, with workloads weighted equally. Exact matches count complete generated sequences.

| Format | Prefill ratio | Decode ratio | Full generation ratio | Exact output matches |
| --- | ---: | ---: | ---: | ---: |
| Q4_0 | 0.494 | 0.470 | 0.493 | 15 / 18 |
| Q4_K_M | 0.513 | 0.485 | 0.505 | 12 / 18 |
| Q5_K_M | 0.495 | 0.488 | 0.501 | 12 / 18 |
| Q6_K | 0.429 | 0.603 | 0.571 | 15 / 18 |
| Q8_0 | 0.378 | 0.636 | 0.595 | 12 / 18 |
| MLX affine-Q4 | 0.411 | 0.415 | 0.419 | 18 / 18 |

Per-workload matrix, with each cell listing paired-median `prefill / decode / full-generation` ratios:

| Workload | Q4_0 | Q4_K_M | Q5_K_M | Q6_K | Q8_0 | MLX affine-Q4 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| short | 0.490 / 0.498 / 0.507 | 0.514 / 0.496 / 0.510 | 0.515 / 0.504 / 0.518 | 0.423 / 0.621 / 0.587 | 0.378 / 0.658 / 0.603 | 0.457 / 0.431 / 0.455 |
| 128 | 0.518 / 0.496 / 0.497 | 0.515 / 0.512 / 0.527 | 0.499 / 0.514 / 0.527 | 0.408 / 0.628 / 0.588 | 0.389 / 0.670 / 0.607 | 0.636 / 0.433 / 0.491 |
| 512 | 0.531 / 0.468 / 0.505 | 0.539 / 0.482 / 0.512 | 0.520 / 0.490 / 0.511 | 0.458 / 0.608 / 0.538 | 0.429 / 0.631 / 0.539 | 0.410 / 0.427 / 0.434 |
| 1024 | 0.534 / 0.439 / 0.501 | 0.537 / 0.452 / 0.507 | 0.526 / 0.465 / 0.502 | 0.469 / 0.561 / 0.509 | 0.437 / 0.588 / 0.493 | 0.372 / 0.375 / 0.393 |
| sustained-128-decode | 0.433 / 0.471 / 0.485 | 0.474 / 0.489 / 0.503 | 0.451 / 0.485 / 0.500 | 0.389 / 0.604 / 0.609 | 0.379 / 0.649 / 0.648 | 0.463 / 0.412 / 0.404 |
| long-horizon-1601 | 0.450 / 0.441 / 0.457 | 0.467 / 0.456 / 0.471 | 0.460 / 0.459 / 0.473 | 0.427 / 0.560 / 0.572 | 0.353 / 0.585 / 0.596 | 0.358 / 0.403 / 0.392 |

Raw, per-pair rows remain in the JSONL matrices listed above and in the affine-Q4 final matrix `docs/measurements/phase5.5/mlx-affine4-ferrum-vs-native-mlx-lm-after-affine-quad-policy.jsonl`. The benchmark runners, pinned revisions, matched prompts, logs, diagnostic profiles, and upstream kernel survey remain in this checkpoint so later work can reproduce the comparisons without rebuilding the setup.

Phase 5.5 is paused, not performance-complete. The central remaining deficit is quantized M=1 projection/GEMV throughput: final overall decode ratios range from 0.470 to 0.636 against llama.cpp across GGUF formats, and 0.415 against native MLX-LM for affine-Q4. Prefill also remains behind mature references: the final overall ratios range from 0.378 to 0.513 for GGUF and are 0.411 for affine-Q4. The long affine-Q4 cases are lower still (0.372 at 1,024 prompt tokens and 0.358 for the long-horizon workload). These results are far from parity and are recorded as remaining gaps, not completion criteria met.

Recommended continuation point: resume at `src/ops/transformer.rs::project_quantized` and the Metal M=1 quantized GEMV implementations. First profile synchronized per-projection M=1 time for Q6_K and Q8_0 against the retained baseline profiles, then select and validate a format-specific dataflow; their final decode ratios (0.603 and 0.636) are the strongest GGUF results but still leave a substantial gap. Continue with Q4_K_M and Q5_K_M after that, designing a different Q5 unpack/reuse strategy from the rejected Q5_0/Q5_1 two-row candidate. Keep the matched matrix harness as the acceptance measurement. Address prefill in a later, separately measured batch/projection tiling pass. No Phase 6 work is started by this checkpoint.
