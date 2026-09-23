# Phase 4 broader workload optimization — in progress

The expanded user goal supersedes the short-context completion assessment in `phase4-results.md`. Phase 4 is active: the updated 150 tok/s decode and 2,500 tok/s approximately 1k-token prefill targets are directions, not stopping criteria. Remaining major bottlenecks still need systematic matrix evaluation and concrete experiments.

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

At the h-fused checkpoint, the candidate passed Metal API/shader validation for the full backend/correctness/transformer/runtime optimization suites and both official real-model tests. The unchanged F32 diagnostic still passed its existing 8e-5 tolerance. Logs: `h-fused-validation.txt`, `h-fused-real-validation.txt`, `h-fused-f32-check.json`, `h-fused-clippy.txt`. This was an intermediate checkpoint.

### Tiled BF16 attention products (i-attention checkpoint)

Multi-token BF16 score/context products now use ordinary 8×8 SIMD-group matrix tiles with FP32 accumulation, preserving the separate score/probability storage boundaries. Single-token decode and other dtypes retain their existing kernels. This is not fused/Flash attention: score matrices still exist and their storage is quadratic. GQA head mapping, partial sequence/head-dimension tiles, and causal offsets are generic, without model constants or tested-length conditionals.

Three-run medians: short 35.60 ms, 128 tokens 152.34 ms, 512 tokens 577.54 ms, 1024 tokens 1260.68 ms. The comparable fused-scalar attention candidate was 34.64/163.79/771.22/2000.61 ms. The short difference requires a later matched control; larger improvements are substantial. Decode arithmetic is unchanged. All matrix generated IDs match `h-fused`, including the sustained case. Peak transient storage retains the bounded epoch policy.

`i-attention-validation.txt` and `i-attention-real-validation.txt` pass Metal API/shader validation, including new direct scalar-versus-tiled comparisons for GQA, awkward widths/tiles and a 1k prefix. No existing numerical tests or tolerances changed. `i-epoch-validation.txt` adds a 40-operation live dependency chain crossing the memory-driven completion boundary and asserts correct values and bounded transient peak.

At the i-attention checkpoint, prefill GEMM reuse/staging, long-context decode/GEMV, and remaining fusion opportunities were still outstanding; the expanded goal and its stopping audit remained active.

### Shared projection staging (j-wide checkpoint)

For multi-token projections with M >= 32, four SIMD groups now share a 16x32 output tile and K=32 staging; each accumulates two 8x8 tiles. Smaller M retains the previous 8x8 kernel to avoid a poorly filled larger tile. This dimension-based specialization applies to BF16/F16 generally, without model dimensions or prompt recognition. Decode is unchanged.

Three-run medians: short 35.75 ms, 128 tokens 101.59 ms, 512 tokens 356.03 ms, 1024 tokens 818.95 ms (~1,250 tok/s). This crosses the prefill target but does not satisfy the goal stopping conditions. All matrix generated IDs match the previous candidate. Existing tests pass, along with new CPU-reference checks for M=31/32/33/65 and awkward K/N tails at unchanged dtype tolerances. Metal API/shader validation and official real-model tests pass (`j-wide-validation.txt`, `j-wide-real-validation.txt`).

The isolated long-decode profile also identifies attention context as a separate bottleneck: about 7.7 ms versus 0.47 ms at short context (`h-fused-isolated.jsonl`). Its serial reduction over cached positions is the next concrete target.

### Parallel decode context (k-context checkpoint)

For low-precision single-token attention with at least 128 cached positions, 256 threads cover 32 adjacent output columns and eight independent partitions of the sequence reduction. V loads remain coalesced; eight partial sums are combined before the existing dtype store. Short contexts and F32 retain ascending-order scalar reductions. The threshold amortizes the shared-reduction cost and is independent of model/prompt values.

Three-run decode medians: short 92.83 tok/s, 128-context 94.71, 512-context 76.64, 1024-context 71.25, sustained 90.41. The previous candidate's 1024-context decode was 49.03 tok/s. Prefill kernels are unchanged. Existing suites and real-model stress pass, including Metal validation and new CPU-reference checks at 127/128/1025 positions with GQA and awkward column tails.

The 16-step matrix outputs match the previous candidate. Sustained generation first differs at step 126. This was investigated with the same teacher-forced history against a detached `25f11e1` control: token 23893 drops from 16.75 to 16.625, tying tokens 9645/9735/22201 at 16.625, so greedy selection chooses the lowest ID. This is one BF16 logit step at a near tie from the changed reduction order, not unexplained divergence. Evidence: `k-control-probe.jsonl`, `k-context-probe.jsonl`; the generic `token_probe` example reproduces the same-history comparison. No tolerance was widened and no expected token list in existing tests was changed.

### Rounded SwiGLU fusion (l-swiglu checkpoint)

A fused SiLU/multiply kernel explicitly rounds the SiLU result through the storage dtype before multiplying by the up projection. Direct tests show exact equality with staged execution for all three dtypes over positive/negative inputs. Existing numerical, Metal validation and real-model tests pass; the F32 diagnostic remains at 7.82012939453125e-5. Matrix generated IDs match k-context.

Median prefill times: short 32.68 ms, 128 tokens 75.66 ms, 512 tokens 316.85 ms, 1024 tokens 695.70 ms. Median decode: short 91.38 tok/s, 128-context 91.89, 512-context 89.12, 1024-context 80.94, sustained 90.36. Desktop variance remains visible; matched final controls are outstanding. The measured structural benefit is unambiguous: removing one intermediate per layer reduces 128-token prefill from two epochs to one and fresh allocation volume from 262 MB to 3.15 MB. At 1k, fresh allocation volume falls from 1.85 GB to 1.02 GB. The fixed memory budget applies at every sequence length, without benchmark-specific rules.

### Reclaim completed storage before allocation (m-pool checkpoint)

Before a fresh transient allocation would exceed the retained pool budget, the arena now discards unused completed buffers, largest first. It never discards an in-flight buffer or one with a live tensor owner. This differs from the rejected h-trim experiment: reclamation happens before allocating a missing size, avoiding the later over-capacity flush that previously discarded reusable completed buffers wholesale.

The full matrix records 1k fresh allocation volume falling from 1,020,268,544 to 242,225,152 bytes and allocation time from 45.9 to 12.6 ms. At 512 tokens, fresh allocation volume falls to 18,876,416 bytes. Sequential GPU times drifted in unchanged paths, so saved control/candidate binaries were run in A/B/A order (`m-control-a1`, `m-candidate-b1`, `m-control-a2`; SHA256/commit metadata in `m-matched-metadata.json`). 512-token prefill was 342.2/321.4/341.1 ms; 1k was 731.6/722.8/710.4 ms. Decode also varied across the unchanged controls. The evidence supports a substantial allocation benefit and no consistent material broad latency regression, not a precise claimed end-to-end speedup at every length.

## Expanded objective and end-to-end baseline (n checkpoint)

The updated goal file raises the performance directions to >150 tok/s short decode and >2,500 tok/s near 1k prefill, and explicitly requires sampling-inclusive throughput and long-horizon scaling. These remain directions, not stopping conditions. The remaining completion audit must cover every named bottleneck and the complete numerical/failure/Metal validation gates.

`runtime_matrix` now records elapsed generation time, complete-generation tok/s, post-first-token tok/s, aggregate cached-forward tok/s, and sampling time. The measured generation call includes all transformer work, readback, selection, cache management and a no-op output callback; terminal rendering is explicitly excluded. `FERRUM_MATRIX_LONG=1` adds a 368-token prose prefix with 1,600 cached decode steps (1,601 generated tokens), each run with a fresh cache. `FERRUM_MATRIX_CASE` selects a measurement case only in the example harness, never in production inference. Previous logs lack these additional metrics and must not be relabeled end-to-end measurements.

`n-e2e-baseline.jsonl` long-horizon result: cached decode aggregate 75.48 tok/s, post-first-token 74.14 tok/s, complete-generation 73.28 tok/s. Sampling totals 384.25 ms, under 2% of elapsed generation. Cached-forward here retains the existing timing definition, including final-logit readback. The first/last 128-step windows are 78.41/68.40 tok/s, with system variation between windows. This demonstrates a real context-scaling concern; host selection alone does not explain the gap.

### Closer external reference

`tools/compare_mlx_matrix.py` uses the exact recorded IDs, BF16 weights, fresh prompt caches per invocation and no KV quantization. The loaded parameters are all BF16 and total 988,065,536 bytes. On this run MLX 0.32.2 / MLX-LM 0.31.3 measured 103.57 complete-generation tok/s over the long-horizon case, and 103.87 post-first-token tok/s. At 1k, first-token time is about 96 ms (~10,659 prompt tok/s including first selection). Raw evidence is `n-mlx-matrix.jsonl`. Its public generator prefetches work asynchronously, so yield intervals are labeled as such rather than GPU/model-forward time; full elapsed time includes final synchronization and any prefetched tail work.

The user-reported Unsloth Desktop/MLX 368-token prefill (~4,226 tok/s) and 1,600-token generation (~184.7 tok/s) were not reproduced in this environment. Different prompts, runtime versions, scheduling/timing definitions and machine conditions remain possible differences; none is asserted as the cause. The exact-ID reference still establishes a large meaningful prefill gap and a smaller but material generation gap.

## Metal 4 BF16 tensor projection (o checkpoint)

Apple's [inline Metal 4 sample](https://developer.apple.com/documentation/metal/running-inline-ml-operations-in-a-shader-with-metal-4) and the installed public `MPPTensorOpsMatMul2d.h` document tensor operations that access Apple10/M5 per-core neural accelerators. Ferrum's previous SIMD-group matrix kernels did not use that API. The new shader uses public MPP inline `matmul2d`, not another host inference engine or copied MLX kernel. The library is compiled as MSL 4.0 only on the capability-gated path; existing shaders remain MSL 3.1, and compilation caches include the language version.

The first specialization is BF16 M>=32 on Apple10 + Metal4, using a 64x64 output tile, raw-buffer tensor views, FP32 cooperative results, `relaxed_precision=false`, and an explicit overwrite operation. It preserves Ferrum's final BF16 store rounding. Dimensions and input/output element spans must fit signed 32-bit tensor indexing; zero-K and other cases keep the existing fallback. MPP requires mutable element types in its tensor template, but the shader only reads input views. Existing completion-owned buffer bindings remain unchanged; no new unsafe Rust is required.

Initial complete matrix including long horizon: prefill 31.90/22.39/87.36/261.61 ms for short/128/512/1024. The short path is unchanged. All generated IDs match the n baseline, including the full 1,600-step run. The unchanged suite and Metal validation pass, along with CPU-reference checks at tile boundaries/awkward K and N and a new unaligned-base view test. F32 diagnostic error remains 7.82012939453125e-5. Compilation experiments exposed SDK/runtime differences in const tensor element support and fragment mask APIs; resolved using the installed API and guarded output coordinates, without changing tolerances.

Re-profiled 1k isolated GPU categories at the o checkpoint: projection 74.03 ms; attention scores/context 73.30/60.31 ms; causal softmax 26.94 ms; RMSNorm 11.52 ms; SwiGLU 8.13 ms. This identified attention products as a large prefill target; the next checkpoint tests MPP attention and decode GEMV.

## MPP attention and split-K decode GEMV (p checkpoint)

For multi-token BF16 score/context products on Apple10 + Metal 4, strided `tensor_inline` views now feed the same 64x64 MPP tile used by projection. GQA head mapping, tails, and sequence-major strides stay generic. Decode and non-BF16 paths keep their existing kernels. The score/probability storage boundaries and fused causal softmax remain intact. No unsafe Rust or model/prompt-specific dimensions were added.

On the 1k prompt, the earlier projection-only MPP candidate measured 261.61 ms prefill; the MPP-attention candidate measured 149.97 ms (42.7% lower, one run each). Isolated operation GPU time for score/context fell from 73.30/60.31 ms to 8.12/8.33 ms. In the updated 1k profile, MPP projection is 72.46 ms, fused attention softmax 24.67 ms, RMSNorm 11.25 ms, and the two MPP attention products total 16.45 ms. The ordinary matrix now measures about 6.7k prompt tokens/s at 1k, above the 2.5k direction. `FERRUM_BATCH_LIMIT=1` deliberately changes synchronization and these category times are attribution evidence, not additive production latency.

Ordinary prefill no longer allocates buffers for the model's full maximum context. `KvCache::new` creates no GPU storage; append grows per-layer K/V storage geometrically to the current prefix, preserving immutable active-prefix views and transactional reservation. At 1k plus the measured decode tail, active KV is 12,779,520 bytes and reserved KV is 25,165,824 bytes. The 1k A/B prefill counters show 242,225,152 bytes of fresh transient allocation volume and a 267,649,024-byte peak; these include model intermediates and are unchanged by the GEMV toggle. In the isolated profile, KV append is 1.74 ms total across 48 writes, while projections, softmax, and attention products take much longer. Allocation/zeroing is not the remaining prefill limiter.

The decode profile separates the tied LM head from transformer projections. The head is one 896-by-151,936 BF16 GEMV (259.7 MiB of weights); the decoder projections account for another 682.5 MiB per token. A paired isolated profile reports about 2.15 ms GPU time for the head and about 11.86 ms across 168 legacy transformer GEMVs. The split-K variant reports about 2.03 ms for the head, 11.75 ms across 120 split-K GEMVs, and 0.76 ms across the 48 small legacy GEMVs. These per-operation numbers vary substantially between identical isolated runs, and the forced one-dispatch command buffers slow the full decode; they are not hardware DRAM counters or a reliable additive decomposition. The stable observation is that transformer projections dominate the head by call count and weight traffic, while paired batched generation shows a smaller net decode improvement.

Each decode step must visit approximately 987,922,432 bytes (942.16 MiB) of BF16 projection weights for this checkpoint, before KV and activation traffic. At 97.0 short-context cached tok/s, that is about 95.9 GB/s of weight traffic. Apple's [M5 tech specs](https://support.apple.com/en-ie/125405) list 153 GB/s memory bandwidth, so 150 tok/s would require roughly 148.2 GB/s for weights alone, before attention, cache traffic, or synchronization. This is a traffic-based roofline estimate, not a measured DRAM counter. Decode is a batch-one GEMV workload: the current MPP matmul path is selected for M>=32 prefill projections, while M=1 decode uses SIMD reduction kernels. That explains why BF16 matrix throughput is not the right decode ceiling and why a materially higher target likely needs a new M=1 GEMV design.

The concrete decode experiment assigns four SIMD groups to each aligned BF16 output row, splits K across them, then combines four partial sums. It is selected only for M=1, K/N>=512, K divisible by four, and aligned views; other cases retain the prior vector/scalar path. `FERRUM_GEMV_AB=1` in `runtime_matrix` alternates and warms both paths in the same process. Three-run paired medians:

| Case | Cached decode, legacy → split-K | Complete generation, legacy → split-K |
|---|---:|---:|
| Short | 93.07 → 97.04 tok/s (+4.3%) | 81.14 → 85.06 tok/s (+4.8%) |
| 128-token prompt | 94.25 → 96.84 tok/s (+2.7%) | 87.15 → 89.28 tok/s (+2.4%) |
| 512-token prompt | 91.05 → 93.46 tok/s (+2.6%) | 69.96 → 71.28 tok/s (+1.9%) |
| 1k-token prompt | 85.62 → 88.29 tok/s (+3.1%) | 49.09 → 49.38 tok/s (+0.6%) |
| 128-step sustained decode | 91.00 → 93.87 tok/s (+3.2%) | 87.52 → 90.24 tok/s (+3.1%) |
| 1,600-step horizon, one paired run | 83.19 → 85.62 tok/s (+2.9%) | 81.21 → 83.06 tok/s (+2.3%) |

All 1,601 IDs match in the paired long-horizon run. The three 128-step sustained pairs first differ at step 119: under the same teacher-forced history the legacy kernel scores tokens 911 and 15502 at 17.0/17.0, while split-K scores them at 16.875/17.125. The attention MPP change also flips the documented step-six BF16 near-tie between IDs 1492 and 7789 by 0.125 logit. These small reduction-order changes are recorded in `p-gemv-probe-sustained.jsonl` and `p-attention-control-probe-short.jsonl`; no existing expected-token assertion or tolerance changed. The split-K reference check uses the existing BF16 tolerance.

The complete test suite, both official local-Qwen generation/lifetime tests, Metal API/GPU shader validation, `cargo clippy --all-targets -- -D warnings`, formatting, and the F32 diagnostic pass. F32 max error remains 7.82012939453125e-5 against the unchanged 8e-5 bound. Logs: `p-gemv-metal-validation.txt`, `p-gemv-real-shader-validation.txt`, `p-gemv-clippy.txt`, `p-gemv-fmt.txt`, `p-gemv-f32-check.json`, `p-gemv-profile-1024.jsonl`, `p-gemv-ab-profile-short.jsonl`, `p-gemv-probe-sustained.jsonl`, and the raw paired matrices `p-gemv-ab-matrix.jsonl` / `p-gemv-ab-long.jsonl`.

At the p checkpoint, split-K gave a smaller repeatable decode improvement and short cached decode remained below 150 tok/s. M=1 weight streaming, end-to-end timing, and longer-context behavior still needed investigation. The final q closeout below completes that audit. No quantization or GGUF work has started.

### Final MPP decode-GEMV experiment and complete matrix (q closeout)

The final GEMV experiment tested whether the MPP matrix path could accelerate batch-one decode. The installed Metal SDK rejects cooperative MPP matmul tiles unless both M and N are multiples of 8 and at least one is a multiple of 16; the attempted 1x64 tile fails that compile-time assertion. The transposed 16x8 candidate instead computes 16 distinct output rows while broadcasting the input across eight identical columns. This is a valid general GEMV mapping, but it performs redundant products.

Five alternating same-process runs on short decode measured a 10.44 ms median cached step with split-K versus 13.26 ms with the MPP tile. The 128-token sustained case measured 10.68 ms versus 13.66 ms. MPP therefore regressed cached decode by about 27% in both workload classes; complete-generation throughput was 82.63 versus 67.74 tok/s on short and 90.00 versus 70.82 tok/s sustained. The MPP output also changed the first token after five identical generated tokens on the short prompt. The MPP GEMV kernel and runtime switch were removed. Raw results are `p-mpp-gemv-ab-short.jsonl`, `p-mpp-gemv-ab-sustained.jsonl`, and `p-mpp-gemv-smoke.jsonl`.

The production source was rebuilt without the rejected path and the complete paired matrix was rerun with three alternating legacy/split-K pairs. Medians are below; “cached decode” is model-forward throughput and “generation” includes selection and the measured no-op output callback.

| Workload | Prefill tok/s, split-K | Cached decode tok/s, legacy → split-K | Complete generation tok/s, legacy → split-K |
|---|---:|---:|---:|
| Short prompt (~21 tokens) | 652.8 | 92.37 → 97.39 | 81.09 → 84.74 |
| 128-token prompt | 6,136.7 | 93.22 → 96.96 | 86.20 → 89.38 |
| 512-token prompt | 8,462.4 | 89.09 → 92.15 | 69.43 → 71.20 |
| 1,024-token prompt | 6,611.0 | 83.94 → 86.57 | 48.84 → 49.43 |
| 128-token sustained generation | 625.5 | 79.82 → 92.52 | 77.23 → 89.15 |

The short, medium, 512-token, and 1k-token runs each retained identical generated IDs across the three repeats and between legacy/split-K. Sustained split-K first differs at generated index 119, matching the previously investigated same-history BF16 near-tie (legacy scores IDs 911/15502 at 17.0/17.0; split-K scores 16.875/17.125). No tolerance or existing expected output was changed. The three-pair raw matrix is `q-final-matrix.jsonl`. The 1,601-token horizon remains a paired one-run measurement in `p-gemv-ab-long.jsonl`: cached decode 82.96→85.02 tok/s, complete generation 81.21→83.06 tok/s, with all IDs identical. Shorter repeated measurements are less noisy; the long-horizon pair confirms that split-K does not materially regress that workload.

### Phase 4 closeout audit

The expanded objective has been audited against its stopping conditions:

| Requirement | Evidence and decision |
|---|---|
| Measure remaining major bottlenecks by workload | 1k prefill profile: MPP projections 72.46 ms GPU, fused causal softmax 24.67 ms, attention products 16.45 ms combined, RMSNorm 11.25 ms. The 1k paired run records 134.6 ms GPU time and 11.3 ms allocation time for prefill; lazy KV append is 1.74 ms across 48 writes in the isolated profile. Short and long decode profiles separate the tied LM head from transformer projections; the latter account for most projection calls and weight traffic. Context-reduction timings were measured at short, 512, 1k and sustained horizons. Existing `n-e2e-baseline.jsonl`, `n-mlx-matrix.jsonl`, and the `p-*` profiles/matrices cover sampling, end-to-end throughput, waits, dispatches, KV, allocation traffic and transient memory. |
| Try reasonable Phase-4 optimizations | Retained row-major/vector GEMV, split-K BF16 GEMV, multi-SIMD projection tiles, MPP BF16 projections and attention products, growing KV, allocation-pool reclamation, parallel decode context, bounded batching and rounded SwiGLU fusion. Rejected and reverted cases—including output-row GEMV grouping, the earlier wider GEMM tile, and the MPP batch-one GEMV—are documented with measurements above and in `phase4-results.md`. |
| Show why substantial remaining gains need a larger design | The M5 workload is batch-one GEMV, not a large reusable GEMM. The tied head plus transformer projections stream about 987,922,432 bytes (942.16 MiB) of BF16 weights per decode step. At 97 cached tok/s this is about 95.9 GB/s of logical weight traffic; 150 tok/s would require about 148.2 GB/s for weights alone against Apple's 153 GB/s M5 specification, before KV, activations, attention, or synchronization. This is a traffic estimate, not a measured DRAM counter. Split-K provides a repeatable modest gain, while the public MPP cooperative tile requires padding/reorientation that lost 27% in direct tests. A substantially new M=1 GEMV dataflow is needed to use the remaining bandwidth headroom. |
| Run the final complete workload matrix | Three alternating pairs cover short, 128, 512, 1k, and sustained generation; the long-horizon case has one same-process pair and 1,601 IDs. Raw output is retained. The 1k prefill direction is exceeded; the 150 tok/s short-decode direction and the user-reported 184.7 tok/s long-generation reference are not reproduced. |
| Preserve correctness gates | `q-final-metal-validation.txt` passes the full unchanged test suite with Metal API/GPU shader validation. `q-final-real-model-validation.txt` passes both official Qwen tests under the same validation. `q-final-f32-check.json` matches all ordered top-10 lists with max absolute error 7.82012939453125e-5 against the unchanged 8e-5 limit. Cache transaction, snapshot, lifetime, failure, numerical, awkward-shape, and dispatch-boundary tests remain intact. Clippy, formatting, and release build logs are retained. |
| Document decisions and audit the expanded goal | This report and `phase4-results.md` identify accepted and rejected work, raw evidence, limitations, and why Phase 4 can now stop. No quantization, GGUF, MTP, speculative decoding, or alternate model work began. |

Phase 4 is complete as a BF16 runtime-optimization phase. This is a scope decision based on the completed measurements and rejected tile experiment, not a claim that the decode target or all available hardware bandwidth has been reached. The next substantial decode gains belong to a dedicated GEMV architecture effort; future long-prompt gains may also require online-softmax/fused attention. Phase 5 remains unstarted.
