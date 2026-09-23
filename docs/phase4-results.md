# Phase 4 — final measured Rust/Metal runtime report

> The first-pass short-context milestone and its external comparisons are retained below as historical context. The expanded Phase 4 goal is now closed by the final matrix and audit in [matrix optimization progress](phase4-matrix-progress.md#phase-4-closeout-audit).

Measured on Apple M5, macOS 27, Rust 1.96, September 2026. Runtime: independent Rust + Metal, shared storage, batch one, Qwen2.5-0.5B-Instruct BF16. Checkpoint revision `7ae557604adf67be50417f59c2c2f167def9a775`. Immutable Phase 3 control: `7dc2dbdf3721a96d5a44055352084ede71f35ea2`; working branch: `codex/phase4-runtime-optimization`.

**Final expanded-matrix result:** the retained split-K GEMV candidate measures 97.39 cached decode tok/s and 84.74 complete-generation tok/s on the short prompt, versus 92.37 and 81.09 for the matched legacy kernel. At a 1,024-token prompt it measures 6,611 prefill tok/s and 86.57 cached decode tok/s. The final 1k prefill profile is projection-dominated; ordinary prefill no longer allocates full-context KV capacity. The target of 150 short-context decode tok/s remains unmet, and the remaining throughput headroom is documented as a batch-one GEMV design problem. Existing tests and tolerances were preserved. No Ferrum quantization or GGUF implementation was started.

## Reproduction and measurement definitions

```sh
cargo build --release
# MODEL is the local directory for the pinned checkpoint.
target/release/ferrum run --model "$MODEL" --prompt 'Hello!' \
  --max-new-tokens 32 --temperature 0 --warmup
python3 tools/measure_phase4.py "$MODEL" g
# Isolated category timing, not normal throughput:
FERRUM_BATCH_LIMIT=1 target/release/ferrum profile --model "$MODEL" \
  --prompt 'Hello!' --max-new-tokens 8 --temperature 0 --warmup
```

The Qwen chat template produces 21 prompt IDs. Ordinary runs request 32 tokens and stop on EOS at 10 generated tokens. Profile runs request 8 tokens to match the control diagnostic. Decode is the median cached forward latency, excluding text streaming; first-token latency includes prefill and first selection, excluding checkpoint load/construction/tokenization. Each repeated run warms the model before measurement. Times reflect a desktop, not a thermally controlled laboratory. The fresh control was run before modifications; a later clean detached worktree at the same immutable commit provided the memory control.

All raw results are in [measurements/phase4](measurements/phase4). `a-*` is the fresh baseline, `b/c/d-*` incremental optimizations, `e-*` the pre-follow-up candidate, `f-*` follow-up experiments, and `g-*` the final repeat/validation. Rejected experiments are explicitly identified below. Earlier checkpoints' allocation accounting predates the fix that includes retired GPU-owned resources in peak/live counters; do not compare their transient peaks as if identically defined.

## First-pass short-context before/after (historical)

| Metric | Fresh Phase 3 | Final Phase 4 | Interpretation |
|---|---:|---:|---|
| Prefill tok/s | 25.032 | 512.401 | 20.47× |
| Prefill ms | 838.923 | 40.984 | final median of 3 |
| First token ms | 839.746 | 41.285 | 20.34× lower latency |
| Decode tok/s | 1.279 | 87.513 | 68.42× |
| Decode ms | 782.163 | 11.427 | final median of 3 |
| Actual GPU ms/decode, profile | 104.209 | 11.031 | baseline many buffers, final batched |
| Completion wait ms/decode | 728.084 | 11.296 | includes GPU execution |
| Wait beyond GPU ms/decode | 623.875 | 0.264 | difference of measured durations |
| Dispatches/decode | 3,844 | 603 | 6.37× fewer |
| Command buffers / waits per decode | 3,844 / 3,844 | 1 / 1 | completion boundary reduction |
| Physical allocations/decode | 3,845 | 1 | four-byte token ID carrier |
| Allocation/zero time ms/decode | 9.410 | 0.0042 | actual measured allocation path |
| Layout/copy wall ms/decode | 309.95 | 0 ordinary layout kernels | includes old concat; growth is an exception |
| Projection + attention product GPU ms/decode | 45.304 | 10.575 | isolated categories; see qualification below |
| RMSNorm GPU ms/decode | 10.381 | 0.838 | isolated category |
| Softmax GPU ms/decode | 6.980 | 0.328 | isolated category |
| Unique retained weight payload bytes | 1,260,334,848 | 988,065,536 | no transposed tied LM copy |
| Process peak RSS bytes | 3,053,256,704 | 2,123,481,088 | separate time -l memory control; final median |

Sources: `a-baseline.txt`, `a-profile-summary.json`, `a-memory-control.txt`, `g-summary.json`, `g-profile.txt`, `f-kernel-summary.json`. Kernel attribution uses an isolated one-operation-per-command-buffer run; its sums are **not additive with the batched total**. The old matmul bucket included attention products, so the comparison includes final GEMVs and both attention products. Isolated timing is noisy: a second category run (`g-kernel-summary.json`) measured larger totals. Both identify matrix projections as the dominant category, but neither is a hardware-counter decomposition of normal batched execution.

First-pass final-candidate three runs (`g-summary.json`):

| Run | Prefill tok/s | First token ms | Decode tok/s | Peak RSS bytes |
|---|---:|---:|---:|---:|
| 1 | 511.330 | 41.370 | 88.038 | 2,123,481,088 |
| 2 | 512.401 | 41.285 | 84.015 | 2,124,333,056 |
| 3 | 513.646 | 41.189 | 87.513 | 2,123,399,168 |

That first-pass candidate's profiled prefill was 39.931 ms GPU, 0.442 ms allocation/initialization, 0.229 ms encode, and 40.213 ms wait. Previously allocation/initialization alone was 18.608 ms (`e-profile.txt`); the remainder of the observed prefill gain includes run variation. Normal decode used 0.211 ms encode and 11.031 ms GPU on average. Sampling took 1.822 ms total for eight output tokens.

First-pass memory accounting: 988,065,536 retained weight bytes, 3,145,728 KV reserved bytes (344,064 active after the eight-token profile), 53,118,976 bytes retained transient arena capacity/high-water. Prefill reuses 51,970,048 transient bytes; decode reuses 2,721,792 bytes/token, also its measured peak transient live storage. Prefill has 49 fresh allocations totaling 3,145,812 bytes (48 KV allocations plus IDs); decode has one four-byte allocation. Accounted persistent weight + KV + arena capacity is 1,044,330,240 bytes. This is not total process memory: CPU loading, allocator, driver, pipeline and other overhead are separate. Peak RSS is not steady-state GPU allocation. Geometric growth increases physical KV and causes occasional copies/allocations outside this short workload.

## Optimization checkpoints and decisions

| Stage | Prefill ms | Decode ms | Change / evidence |
|---|---:|---:|---|
| A fresh unchanged Phase 3 | 838.923 | 782.163 | baseline generation |
| B batching limit 1024 | 92.880 | 57.644 | original arithmetic/layout, `b-limit-1024.txt` |
| C bucket pool | — | 53.355 | `c-pool-buckets.txt` |
| C direct attention layouts | 108.229 | 43.488 | full-capacity KV allocation still present |
| D row-major GEMV | — | 25.237 | `d-gemv-profile.txt` |
| D parallel RMSNorm | — | 14.682 | midpoint safeguard retained |
| D parallel softmax | — | 13.977 | stable parallel reduction |
| D native SIMD-group GEMM | 64.686 | 13.853 | `d-native-profile.txt` |
| E repeated prior candidate | 64.726 | 14.676 | medians, `e-summary.json` |
| F growing KV | 38.053 | 13.736 | `f-kv-profile.txt` |
| F vector GEMV | 42.578 | 10.883 | first experiment, `f-vector-profile.txt` |
| G earlier repeated candidate | 40.984 | 11.427 | medians, `g-summary.json` |

These are incremental measurements, not all repeated confidence estimates. Batch limits 1/64/256/1024/8192 gave decode 791.885/68.697/59.996/57.644/58.141 ms, supporting 1024 rather than arbitrary unbounded batching. A linear scan free-list pool regressed to 64 ms; bounded size buckets were retained instead.

### Completion, reuse and copies

See [current architecture](architecture.md#phase-4-current-execution-and-storage-model) for the contracts. Resources remain owned until completion; pending tensors cannot be mapped or reused. Writes have range-specific completion states. The cache is staged and committed only after successful GPU completion, including late-error handling. Failed suffixes cannot poison published prefixes, and retries/branches never overwrite reserved suffixes.

| Old layout work | Shapes / reason | Final path |
|---|---|---|
| Q/K/V head selection and K transpose | `[S,H,D]` to per-head matrices and `[D,T]` | direct grouped indexing of sequence-major storage |
| Per-head context stacking/merge | individual `[S,D]` to `[S,Q,D]` | context kernel writes final layout |
| Cache concatenation | `[P,KV,D]` + `[S,KV,D]` | suffix write, copy only at geometric growth or branch |
| Weight transpose | `[N,K]` to `[K,N]` | row-major projection, tied storage shared |
| Final logits extraction | `[S,V]` to last row | checked contiguous view |

The direct layout change removes 1,680 head/layout materializations per ordinary decode; capacity-backed KV removes 48 ordinary history concatenations. General copy APIs remain available for tests/other callers. Logical bytes depend on current sequence length, not a fixed model constant. There is no broad unchecked arbitrary-stride API.

### Additional GEMV pass: why throughput was low

The previous kernel loaded/converts one scalar at a time, with one accumulation chain per lane. The retained vector kernel uses aligned native four-element loads and four independent F32 chains. This reduces loop/address calculations and exposes more instruction-level parallelism. The improvement demonstrates instruction/load organization mattered; timings alone do not distinguish DRAM, cache, occupancy and issue limits conclusively.

Standalone BF16 projection GPU medians (nine samples after warmup; dimensions are benchmark cases, not hard-coded runtime dispatch rules):

| Projection `[N,K]`, M=1 | Scalar ms | Vector ms |
|---|---:|---:|
| Q `[896,896]` | 0.0310 | 0.0345 |
| KV `[128,896]` | 0.0193 | 0.0086 |
| Gate `[4864,896]` | 0.1329 | 0.0715 |
| Down `[896,4864]` | 0.0982 | 0.0729 |
| LM head `[151936,896]` | 3.1040 | 2.2011 |

Source: scalar M=1 cases in `f-gemm-projections.jsonl` (that experiment only changed GEMM), versus `f-vector-projections.jsonl`. Small Q regressed within a noisy small-kernel regime; the large kernels and whole-model decode improved enough to keep the generic vector path. Awkward K/alignment uses the scalar fallback. The unchanged numerical suite plus a new tail/alignment test covers both paths.

The LM head's 272.27 MB weight payload divided by 2.2011 ms is **123.7 GB/s effective logical bandwidth**, versus 87.7 GB/s before. Apple specifies [153 GB/s for M5](https://www.apple.com/macbook-pro/specs/); that proxy is about 81% of the specification. It is not measured DRAM traffic or proof of saturation. Decode arithmetic intensity is approximately one FLOP per BF16 weight byte, so bandwidth/issue efficiency is more relevant than peak BF16 matrix FLOPS. A theoretical one-read 988 MB / 153 GB/s floor is about 6.46 ms before activations, other operations and practical inefficiencies; actual batched GPU time remains 11.03 ms.

Isolated LM head: 2.235 ms; 168 transformer GEMVs: 7.277 ms. The transformer projections collectively cost more than the LM head. Remaining opportunities include better GEMV scheduling/coalescing and reducing the many small operation/encoder costs. It would be incorrect to say memory bandwidth alone has been proven to dominate every projection.

Rejected concrete experiment: four output rows per SIMD group sharing activation loads improved gate locally but regressed other shapes and full decode to 17.094 ms. Removed from production; source/logs retained in `f-gemv-rows4-rejected.metal` and `f-gemv-*`.

### Prefill investigation and rejected GEMM experiment

`f-kernel-summary.json` measured transformer GEMM 25.512 ms plus LM GEMM 7.896 ms. Attention scores/context together were 0.885 ms; RMSNorm 3.140 ms, softmax 0.289 ms. The repeat isolated run measured matrix projections 37.052 of 51.164 ms GPU total. Despite run-to-run category variation, matrix work dominates this short prompt. After growing KV, normal allocation/zeroing is only 0.442 ms. Attention is not the current short-prompt bottleneck; long-context attention is unmeasured.

Native SIMD-group instructions do not by themselves make a high-throughput GEMM: the retained kernel stages an 8×8 tile, barriers every eight K elements, one SIMD group per output tile, transposed weight gathering and repeated activation loads. M=21 leaves partial tiles. These are concrete reuse/staging limitations, not a claim that Apple's BF16 hardware ceiling is low.

A tested 8×32, four-SIMD output tile with K=32 staging/coalesced weight loading reduced Q 0.288→0.0868 ms and LM 7.830→6.413 ms, but KV 0.0415→0.096 ms and gate 0.246→0.332 ms regressed. Whole prefill 38.053→39.07 ms did not improve, so it was reverted (`f-gemm-rejected.patch`, `f-gemm-*`). Future work needs a more systematic shape-dependent GEMM design; fused/tiled attention may matter for long prompts, but this measurement does not justify calling it the dominant short-prompt fix.

## Numerical and lifetime validation

Final `g-tests.txt`, `g-validation.txt`, `g-real-validation.txt`, `g-clippy.txt`, `g-fmt.txt`, `g-smoke.txt` and `g-transformer-smoke.txt` record passing checks. All existing numerical, F32 diagnostic, transaction, lifetime and Metal validation tests were kept unchanged during the additional pass. New `runtime_optimization` tests verify geometric allocation boundaries over 600 appends, immutable snapshots, and vector GEMV tail/misaligned-view fallback. The real-model stress generates 128 tokens with EOS suppressed, resets and reinvokes; Metal API/shader validation passes.

The strict F32 diagnostic matches eight token IDs and ordered top-10 lists; maximum selected/top-10 absolute difference is **7.82012939453125e-5**, below the existing **8e-5** tolerance (`g-f32-check.json`). F32 projection and explicit diagnostic reductions preserve ordered arithmetic. This is a deliberate reference mode, not evidence that arbitrary parallel F32 reductions have identical rounding.

BF16 real-model selected-logit/cached-generation tests pass unchanged. Straight parallel RMSNorm previously crossed the existing test's tolerance; its midpoint sensitivity safeguard restored it. Parallel low-precision reductions are not bitwise equivalent to the baseline. Hello retains the documented `help`/`assist` near-tie variation. An additional Rust prompt changes at zero-based step 11: Phase 3 chose ` programming` (19.0 vs 18.75 for ` language`); final teacher-forced logits tie at 18.875, selecting lower-ID ` language`. At teacher-forced step 27, baseline `.`/` and` tie at 20.5; final has 20.375/20.5. These are recorded reduction-order/BF16 rounding-sensitive choices, not hidden token equivalence. See `rust-phase3-logits.json`, `g-rust-teacher-logits.json`, and `g-rust-demo.txt`. Generated prose is a behavior sanity check, not factual validation.

Four unsafe blocks plus framework linkage remain solely in `src/metal/mod.rs` (`g-unsafe-audit.txt`): mapping, initialization, zeroing, encoder binding. No unsafe Send/Sync, model, tokenizer, sampling or cache code was introduced. Metal validation and tests support, but cannot formally prove, shader/driver correctness. Larger Qwen2.5 checkpoints were not tested; the final long-horizon run covers 1,601 generated tokens and is recorded in the matrix progress report.

## External engines, matched as closely as practical

These early comparisons were performed before the expanded optimization request; the final runtime matrix and later exact-ID MLX long-horizon comparison are in the matrix progress report. These rows use the same pinned model, 21 exact prompt IDs, greedy selection, ten output tokens, and batch one. No quantized weights were used.

| Runtime | Prefill tok/s | Decode tok/s | First token ms | Memory |
|---|---:|---:|---:|---|
| Ferrum earlier short-context candidate | 511–514 | 84–88 | 41.19–41.37 | accounted retained storage 1.044 GB; process peak 2.123–2.124 GB |
| MLX 0.32.2 / MLX-LM 0.31.3 | 681–736, includes first selection | 86–109 | 28.53–30.84 | active 992.56 MB + cache 0.637 MB; peak active ~1.009 GB |
| llama.cpp 10330 (`687e77892`), warmed | 1,074–1,276 | 113.16–113.75 | not directly instrumented | steady allocator total unavailable; process peak 1.503–1.504 GB |

MLX loads the exact BF16 safetensors. Its memory counters describe MLX allocations, not RSS. llama.cpp uses upstream BF16 conversion of the same checkpoint for an external reference only: some norms/bias are F32 and KV is F16; Ferrum is BF16 throughout that path. This is an unavoidable precision/layout difference and no claim of exact numerical identity is made. llama.cpp reports prefill eval time 16.46–19.56 ms, not true first-token latency. Its cold first run was 111.17 ms; warm results above are separate. Initial `llama-1..3` runs accidentally trimmed the prompt newline (20 tokens) and are excluded; only `llama-exact-*` using `-bf` have verified 21 IDs. Source logs and `tools/compare_mlx.py` preserve details. These measurements show remaining gaps, not universal engine rankings.

## Final expanded-goal closeout

The last clean-source matrix used the official Qwen2.5-0.5B-Instruct BF16 checkpoint with three alternating legacy/split-K pairs for every ordinary workload. It measures cached model-forward throughput separately from complete-generation throughput, which includes sampling and the no-op output callback.

| Workload | Prefill tok/s | Cached decode tok/s, legacy → split-K | Complete generation tok/s, legacy → split-K |
|---|---:|---:|---:|
| Short prompt (~21 tokens) | 652.8 | 92.37 → 97.39 | 81.09 → 84.74 |
| 128-token prompt | 6,136.7 | 93.22 → 96.96 | 86.20 → 89.38 |
| 512-token prompt | 8,462.4 | 89.09 → 92.15 | 69.43 → 71.20 |
| 1,024-token prompt | 6,611.0 | 83.94 → 86.57 | 48.84 → 49.43 |
| 128-token sustained generation | 625.5 | 79.82 → 92.52 | 77.23 → 89.15 |
| 1,601-token horizon, one paired run | — | 82.96 → 85.02 | 81.21 → 83.06 |

The short, 128, 512 and 1k runs produced identical token IDs across three repeats and between the two GEMV modes. Sustained split-K first diverges at generated index 119; same-history logits show the documented BF16 near-tie between IDs 911 and 15502, with no tolerance change. The 1,601-token pair produces identical IDs. Raw clean-source matrix: `q-final-matrix.jsonl`; long-horizon raw pair: `p-gemv-ab-long.jsonl`.

The major remaining decode bottleneck is batch-one BF16 GEMV weight streaming. The separate profile attributes more projection work to transformer GEMVs than to the tied LM head. The model reads roughly 987.9 MB of BF16 weights per decode step (942.16 MiB). At 97 cached tok/s that represents about 95.9 GB/s of logical weight traffic. The [M5 specification](https://support.apple.com/en-ie/125405) lists 153 GB/s of unified-memory bandwidth; reaching 150 tok/s would require approximately 148.2 GB/s for weights alone, before KV, activations, attention and synchronization. This is a traffic-based estimate, not a DRAM counter. The public MPP cooperative matmul API's minimum tile constraints do not fit an M=1 GEMV naturally: the transposed 16x8 candidate repeated output columns and ran about 27% slower than split-K at both short and sustained context. It was removed. A materially faster path needs a new M=1 GEMV dataflow; split-K's repeatable gain is a modest 3–5% in ordinary cases.

For 1k prefill, actual GPU time is about 135 ms, versus about 11 ms of allocation work. The isolated GPU profile is led by MPP transformer projections (72.46 ms), fused causal softmax (24.67 ms), and MPP attention products (16.45 ms together). Lazy, geometric KV allocation reserves storage only for the current prefix; at 1k plus the decode tail, active KV is 12.78 MB and reserved KV is 25.17 MB. Prefill exceeds the >2,500 tok/s direction, but short decode remains below 150 tok/s and the user-reported 184.7 tok/s long-generation figure was not reproduced. A fused/online-softmax attention design may be worthwhile for longer prefills, but current 1k evidence does not make it the leading operation.

Final gates passed on the clean production candidate. `q-final-metal-validation.txt` records the unchanged full suite under Metal API and GPU shader validation; `q-final-real-model-validation.txt` records both ignored official-model generation/lifetime tests under the same validation. `q-final-f32-check.json` retains the exact eight-step diagnostic result: ordered top-10 matches and maximum absolute error **7.82012939453125e-5** under the unchanged **8e-5** bound. `q-final-clippy.txt`, `q-final-fmt.txt`, and `q-final-build.txt` record clean static checks, formatting, and release build. No existing test or numerical tolerance was removed or weakened.

The expanded Phase 4 goal is closed because relevant bottlenecks were measured across the workload matrix, practical Phase-4-sized optimizations were attempted, useful changes were retained, and the new MPP batch-one design was rejected with direct performance evidence. The substantial remaining decode gain requires a new GEMV architecture; further long-prefill gains may require online-softmax/fused attention. This is not a claim that the 150 tok/s direction or all plausible M5 bandwidth has been reached. **Phase 4 ends at BF16 runtime optimization. Phase 5, quantization, GGUF, MTP and speculative decoding have not begun.**
