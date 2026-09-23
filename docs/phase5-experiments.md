# Phase 5 experiment journal

This journal starts from the finalized Phase 4 checkpoint. Phase 4 reports and raw measurements remain historical records and are not edited by Phase 5.

## Status

Phase 5 is active. The strict GGUF/Qwen2 adapter now retains packed Q8_0, Q4_0, Q5_0, Q5_1, Q4_K, Q5_K, and Q6_K weights and executes direct Metal GEMV, GEMM, MPP prefill, and embedding gather paths. Official Qwen2.5-0.5B GGUF files have completed real generation, repeated paired matrices, and fixed-stream prefill and cached-decode quality probes. MLX interchange, held-out corpus evaluation, broader model coverage, and further optimization remain outstanding.

## Baseline

- Clean starting revision: `16772074dda887ac66aadb7a16408d2288d2e842` (`phase 4: long-context optimization checkpoint`), identical to `codex/phase4-runtime-optimization` and its origin ref. The starting worktree was clean.
- Phase 4 reference: official Qwen2.5-0.5B-Instruct, BF16 safetensors, Ferrum's existing dense Metal path. It remains the correctness fallback and quality reference.
- Final paired matrix: [`phase4-matrix-progress.md`](phase4-matrix-progress.md#final-mpp-decode-gemv-experiment-and-complete-matrix-q-closeout); raw rows are in [`q-final-matrix.jsonl`](measurements/phase4/q-final-matrix.jsonl). The short case used 21 prompt tokens and recorded 652.8 prefill tok/s, 97.39 cached decode tok/s, and 84.74 complete-generation tok/s on the split-K path. Three alternating pairs cover short, 128, 512, and 1,024-token prompts plus 128 generated tokens; the 1,601-token horizon has one paired run.
- The Qwen2.5-0.5B dimensions and shapes are in the report and projection journals: hidden/K 896, intermediate 4,864, vocabulary 151,936, 24 layers, Q projection `[896,896]`, K/V `[128,896]`, MLP gate/up `[4864,896]`, down `[896,4864]`, and tied embedding/output `[151936,896]`.
- Phase 4 reports about 987.9 MB of BF16 weight payload per decode step and 988,065,536 retained weight bytes. The short end-to-end matrix is a runtime baseline, not a bandwidth-saturation claim; its logical-traffic calculation is not a DRAM counter.
- Existing raw profiles, process-memory runs, quality checks, and rejected experiments remain under [`measurements/phase4`](measurements/phase4). Do not overwrite or reinterpret these as Phase 5 results.

## Initial representation decision

### Dense fallback

Keep `Tensor` and the current `DType::{F32,F16,BF16}` behavior for activations, norms, cache, existing safetensors models, and the Phase 4 reference. A quantized weight is not assigned a fake dense `DType`, and existing BF16 projection dispatch remains available unchanged.

### Packed weight

Add a distinct packed matrix weight with logical `[out,in]` dimensions, an explicit upstream format tag, and owned Metal storage whose byte length is computed from that format. Supported GGML blocks include Q8_0 (32 values/34 bytes), Q4_0 (32/18), Q5_0 (32/22), Q5_1 (32/24), Q4_K (256/144), Q5_K (256/176), and Q6_K (256/210). Keep these bytes packed for the lifetime of the model. Direct GEMV reads packed blocks; the MPP GEMM expands only a bounded 64-by-128 BF16 threadgroup tile immediately before accumulation. No complete dequantized matrix is stored. Quantized embeddings use direct row gather/decode.

Keep the packed storage abstraction separate from ordinary tensor arithmetic. Its Metal binding must carry its byte length and lifetime through the same submission/resource-retention mechanism as `Tensor`; it must not be mappable or reusable before successful completion. It introduces no unsafe `Send`/`Sync` implementation and no unsafe code outside `src/metal`.

### GGUF index and model adapter

Parse GGUF independently of Qwen naming. Read the versioned header, typed metadata, tensor descriptors, alignment and relative tensor offsets with checked arithmetic and explicit size bounds. Index dimensions in GGUF order; expose a runtime `[out,in]` matrix by reversing the descriptor dimensions while preserving row-major payload order. Validate required tensor byte lengths from the exact GGML type and block geometry, file bounds, duplicate names, duplicate/overlapping tensor ranges, dimensions, metadata types, and required architecture fields before allocating runtime weights. Unknown or unsupported types fail with the tensor name and type code rather than being interpreted as another format.

Accept GGUF v2/v3 little-endian indices and Qwen2 architecture metadata, with dense F32/F16/BF16 plus Q8_0, Q4_0, Q5_0, Q5_1, Q4_K, Q5_K, and Q6_K tensor loading. Stream tensor payloads into packed or dense Metal storage one tensor at a time so a complete second CPU copy of a large model is not retained. The Qwen adapter maps standard GGUF names and metadata into the current strict model contract. Tokenizer metadata is checked against Ferrum's Qwen tokenizer behavior; BF16 and GGUF prompt token IDs match on both benchmark prompts and the quality corpus.

The parser follows the upstream [GGUF specification](https://github.com/ggml-org/ggml/blob/master/docs/gguf.md) and [ggml block declarations](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-common.h). Quantization type IDs and layouts come from GGML; Ferrum will not define private on-disk variants.

### Execution split

- M=1 decode: dedicated GEMVs for Q8_0, Q4_0, Q5_0, Q5_1, Q4_K, Q5_K, and Q6_K read packed rows and write output directly without a whole-weight intermediate. Paired lane loads share nibbles/bit planes and K-block metadata; each format keeps its own decoder where layouts differ. The vocabulary projection and transformer projections are exercised by production generation.
- M>1 prefill: use format-specific packed GEMM/MPP paths for BF16 activation batches when K/tile shapes and Metal 4 support allow. Quant data is decoded into bounded threadgroup tiles rather than whole weights. Keep direct SIMD GEMM as the fallback for unsupported shapes and devices. Apple's [Metal 4 TensorOps guide](https://developer.apple.com/download/files/Metal-Performance-Primitives-Programming-Guide.pdf) documents threadgroup-backed tensor operands and multiply-accumulate cooperative tensors.
- Direct packed embedding gather keeps all supported quantized embeddings from forcing a whole-model dense copy.
- Q8_0 was the first correctness bridge; Q4_0, Q5_0/Q5_1, Q4_K/Q5_K, and Q6_K now execute through official real-model files. Keep each format's exact GGML block geometry and metadata. Do not collapse distinct GGML schemes into one decoder when that costs correctness or speed.
- MLX `quantized` weights use their own group-size/scale representation. Ferrum now has a strict explicit importer/repacker for the pinned Qwen2 affine-Q4, group-64 repository below; it does not add MLX as a runtime dependency. This is a scoped repository contract, not general MLX model support.

## Measurement definitions and acceptance gates

Use the production Ferrum generation path and the existing Phase 4 workload/matrix conventions. Retain raw JSONL and logs for each run. Record model/load time separately from timed generation. Report prefill tok/s, cached decode tok/s, complete-generation tok/s, first-token latency, GPU/encode/wait time, dispatches, command buffers, allocations and allocation time, active/retained KV, transient peak, retained model bytes, and process memory where available.

Cover short, 128, 512, approximately 1k, and long-horizon workloads. Use repeated paired/alternating A/B runs where the baseline path can be selected in the same build/process; retain exact prompt IDs, generated IDs, and model revisions. Compare Ferrum Q8_0 to Phase 4 BF16 on the same model and prompt. For quality, retain teacher-forced logits and report max/mean absolute error, cosine similarity, top-k agreement, and perplexity on a fixed token corpus where practical; token/prose checks are secondary. Compare MLX/llama.cpp only when weights, tokenizer, prompt IDs, sampling, and cache precision make the comparison meaningful, documenting unavoidable layout and precision differences.

Do not attribute traffic estimates to DRAM bandwidth without hardware counters. Keep before/after measures and rejected experiment artifacts. Change kernel organization only after the first end-to-end correctness and performance baseline exists.

## Format support, experiments, quality, and memory

The journal began before Phase 5 runtime changes; the current checkpoint status is below. Q8_0, Q4_0, Q5_0, Q5_1, Q4_K, Q5_K, and Q6_K are implemented and validated on official GGUF model files. A narrowly scoped MLX affine-Q4 Qwen2 importer and Metal path are now implemented and measured; see the MLX milestone below.

| Experiment | Status | Evidence / decision |
|---|---|---|
| Phase 4 BF16 baseline | Accepted reference | `q-final-matrix.jsonl`, `phase4-results.md`, `phase4-matrix-progress.md` |
| GGUF v2/v3 reader and Qwen2 adapter | Accepted for Q8_0 checkpoint | Full suite passes; official tokenizer IDs match; malformed/unknown/overlap cases reject explicitly |
| Q8_0 direct M=1 GEMV | Accepted | `q8_0_gemv`, official Qwen generation matrix; 168 quantized GEMV calls per decode step; Metal API/GPU validation passed |
| Q4_0 direct M=1 GEMV and packed embedding gather | Accepted | Official Qwen Q4_0 generation exercises transformer GEMVs and Q4_0 embedding; the upstream output matrix is Q8_0 |
| Q8_0 direct M>1 GEMM, one SIMD group per channel and four prompt rows | Rejected for primary prefill | `q8-matrix-after-tile4.jsonl`: 512-token median was 206 tok/s versus 7,318 BF16; the raw run remains for comparison. Four independent K chains raised the 128-token result to 253 tok/s, still far behind |
| Q8_0 MPP prefill with on-the-fly 64x128 tile decode | Accepted for supported shapes | `q8-matrix-after-mpp.jsonl`; 2,392 / 3,675 / 3,510 tok/s at 128 / 512 / 1,024 prompts. The direct SIMD path remains the fallback |
| Q4_0 MPP scalar-per-value tile expansion | Rejected optimization | `q4_0-matrix-before-nibble.jsonl`; 512-token median prefill was 4,044 tok/s |
| Q4_0 paired-nibble MPP tile expansion | Accepted optimization | `q4_0-profile-paired-512.jsonl` and final `q4_0-matrix.jsonl`; 512-token prefill rose to 5,087 tok/s (+25.8%) and first-token median fell from 127.3 to 101.0 ms. Generated IDs matched the earlier kernel for all 36 runs |
| Q6_K direct GEMV/GEMM and bounded MPP tiles | Accepted | Official Q6_K GGUF generation, 448-target quality, multi-superblock and signed-scale tests, and Metal validation passed |
| Q6_K MPP scalar-per-value tile expansion | Rejected optimization | `q6_k-matrix-before-pair.jsonl`; 512-token median prefill was 3,887 tok/s and phase GPU time 128.3 ms |
| Q6_K paired low-plane tile expansion | Accepted optimization | `q6_k-paired-512.jsonl` and final `q6_k-matrix.jsonl`; 512-token prefill rose to 4,214 tok/s (+8.4%) and first-token median fell from 132.0 to 121.8 ms. All 36 generated sequences matched the pre-optimization Q6_K kernel |
| Q5_0/Q5_1 direct GEMV, GEMM, embedding and Qwen GGUF loading | Accepted for supported shapes | Official Q4_K_M and Q5_K_M repositories exercise Q5_0 and Q5_1 tensors respectively; exact high-bit planes, scale/min metadata, odd rows, tails, and unsupported partial blocks covered by tests |
| Q4_K/Q5_K direct GEMV, GEMM, embedding and bounded MPP | Accepted for supported shapes | Official Q4_K_M and Q5_K_M generation, fixed-stream quality, and Metal API/GPU validation pass; 256-value GGML superblock layouts are kept packed |
| Q5_0/Q5_1 paired-byte GEMV and Q4_K/Q5_K shared-byte/superblock GEMV | Accepted provisionally | [Focused Q4_K_M](measurements/phase5/q4_k_m-gemv-paired.jsonl.gz) and [Q5_K_M](measurements/phase5/q5_k_m-gemv-paired.jsonl.gz) alternating A/B increased decode by 14.6% and 11.4%; full matrices retain before/after raw rows and expose output-token drift from the changed reduction order |
| Pinned MLX affine-Q4/group-64 Qwen2 importer, direct Metal GEMV/GEMM, and bounded MPP GEMM | Accepted for the documented checkpoint | The strict safetensors importer retains 277,853,184 packed bytes and 143,104 dense bytes; native MLX/Ferrum logits match at cosine 0.99999, with exact greedy IDs on the measured generation matrix. Coverage is limited to the repository/config contract recorded below |
| MLX Q4 512-token direct SIMD GEMM vs Metal 4 MPP | Direct path rejected for M=512 on Apple M5 | The in-process alternating comparison measured 216 vs 1,583 prefill tok/s and 2.365 s vs 0.320 s phase GPU time. Keep direct SIMD GEMM as the portable fallback; MPP is the selected M5 route |
| Phase 5 paired matrix summarizer | Accepted | `tools/summarize_phase5_matrix.py` reports each model separately and paired A/B deltas; it avoids the old cross-model aggregation in the Phase 4 helper |
| Dispatch threshold 16 prompt rows | Accepted provisionally | On the 21-token short prompt, Q8_0 first-token median improved from 113.0 ms (direct path) to 42.6 ms (MPP), versus 36.0 ms BF16; `q8-mpp-short.jsonl` |
| Q8_0 four-chain GEMV reduction | Accepted provisionally | Cached decode in the full paired matrix is 19–24% faster than BF16 for this model/run; individual before/after result is in `q8-matrix-after-tile4.jsonl`. Keep measuring by shape |

## Validation status and remaining bottlenecks

Phase 4 validation remains recorded in its own journal. For this Q8_0 checkpoint, `cargo test --all-targets` passed (57 tests, two official-model tests remain ignored by their existing environment gate), `cargo clippy --all-targets -- -D warnings` passed, and `cargo fmt --all` passed. With `MTL_DEBUG_LAYER=1 MTL_SHADER_VALIDATION=1`, all Q8-specific Metal tests passed, including Q8_0 GEMV/GEMM tails, embedding gather, and the tiled MPP GEMM. The full paired matrix also exercised official Qwen Q8_0 generation through 1,601 generated tokens.

The teacher-forced fixed corpus repeated 16 times produced 449 positions / 448 targets: mean absolute logit error 0.3048, RMSE 0.3903, cosine similarity 0.99323, top-1 agreement 99.55%, top-5 overlap 86.01%, and perplexity 1.50838 versus BF16 1.50983 (ratio 0.99904). This repeated short text is a reproducible smoke corpus, not a broad language-quality evaluation; a standard held-out corpus is still required before accepting lower-bit formats.

The full matrix used three alternating model pairs per workload and the same tokenizer IDs. Medians are tok/s; first-token values are milliseconds. Raw rows retain exact token IDs, command/counter data, and model revisions in [`q8-matrix-after-mpp.jsonl`](measurements/phase5/q8-matrix-after-mpp.jsonl).

| Workload | BF16 prefill | Q8_0 prefill | BF16 cached decode | Q8_0 cached decode | BF16 complete generation | Q8_0 complete generation | BF16 / Q8_0 first token |
|---|---:|---:|---:|---:|---:|---:|---:|
| short (21 prompt, 17 generated) | 588 | 496 | 87 | 104 | 76 | 85 | 36 / 43 ms |
| 128 prompt, 17 generated | 5,745 | 2,392 | 89 | 107 | 82 | 81 | 23 / 54 ms |
| 512 prompt, 17 generated | 7,591 | 3,675 | 83 | 101 | 64 | 56 | 68 / 140 ms |
| 1,024 prompt, 17 generated | 5,653 | 3,510 | 73 | 88 | 41 | 35 | 182 / 292 ms |
| short prompt, 129 generated | 442 | 445 | 81 | 100 | 78 | 95 | 48 / 48 ms |
| 368 prompt, 1,601 generated | 6,301 | 3,031 | 77 | 91 | 75 | 88 | 59 / 122 ms |

The 128-token profiled run records 168 `q8_0_gemm_mpp` calls for prefill and 168 `q8_0_gemv` calls per cached decode step. Per-operation Metal GPU counter samples were unavailable in that profile; whole-phase GPU, encode, wait, dispatch, command-buffer, allocation, and allocation-time counters are recorded in the matrix JSONL. The process peak in the paired run was 2,248,605,696 bytes while both BF16 and Q8 models were loaded. Q8 retained weights were 524,976,896 bytes (524,833,792 packed); BF16 retained weights were 988,065,536 bytes. Q8's measured transient peaks matched BF16 at each workload (about 37 MB short and 244–268 MB for the longer prefills). No DRAM-bandwidth claim is made from logical traffic.

### Q4_0 real-model milestone

The pinned official Qwen GGUF Q4_0 file, size, and SHA-256 are recorded in [`model-source.json`](measurements/phase5/model-source.json). It contains 169 Q4_0 tensors and a separate Q8_0 output matrix, so the input embedding and output projection are not a physical tied-weight alias. Ferrum retains the quantized payloads and reports the physical tensor count. The complete paired matrix has three alternating model pairs for each workload; the original scalar-per-value tile run and its paired optimization experiment are both retained.

| Workload | BF16 prefill | Q4_0 prefill | BF16 cached decode | Q4_0 cached decode | BF16 complete generation | Q4_0 complete generation | BF16 / Q4_0 first token |
|---|---:|---:|---:|---:|---:|---:|---:|
| short (21 prompt, 17 generated) | 613 | 738 | 87 | 102 | 77 | 89 | 35 / 29 ms |
| 128 prompt, 17 generated | 5,578 | 3,341 | 85 | 99 | 79 | 83 | 23 / 39 ms |
| 512 prompt, 17 generated | 8,215 | 5,087 | 90 | 99 | 70 | 63 | 63 / 101 ms |
| 1,024 prompt, 17 generated | 6,237 | 4,466 | 75 | 89 | 45 | 41 | 165 / 230 ms |
| short prompt, 129 generated | 464 | 609 | 82 | 93 | 79 | 90 | 46 / 35 ms |
| 368 prompt, 1,601 generated | 6,537 | 4,494 | 81 | 93 | 79 | 90 | 57 / 82 ms |

All throughput values are tok/s medians across three alternating pairs; raw rows contain per-token decode timings, prompt/generated IDs, KV use, and phase counters. The final matrix process peak RSS was 2,243,559,424 bytes with both BF16 and Q4_0 models loaded. Q4_0 retained 422,639,360 weight bytes, of which 422,496,256 were packed, versus 988,065,536 BF16 bytes. At 512 prompt tokens, the optimized paired-nibble run recorded 97.8 ms phase GPU time vs 57.6 ms BF16, with the same 531 dispatches, five command buffers, 58 allocations, 18.9 MB allocated in 1.39 ms, and 268,042,240-byte transient peak. This lowers Q4_0 prefill time but does not yet match dense BF16 latency.

The 449-position / 448-target teacher-forced quality probe produced mean absolute logit error 0.7074, RMSE 0.9037, cosine similarity 0.96486, top-1 agreement 98.22%, top-5 overlap 74.12%, and perplexity 1.52762 vs BF16 1.50983 (ratio 1.01178). BF16 and GGUF tokenizers returned identical IDs. The repeated short corpus is a reproducible smoke test rather than broad quality validation; a held-out evaluation remains required.

The five Q4_0 MPP/GEMV/embedding and signed-scale tests passed with `MTL_DEBUG_LAYER=1 MTL_SHADER_VALIDATION=1`. `cargo test --all-targets` passed 63 tests at the Q4_0 checkpoint; the two existing real-model tests remain ignored behind their existing environment gate. `cargo clippy --all-targets -- -D warnings`, formatting, diff check, and all-target check passed. The generation matrix used the official local BF16 and Q4_0 models.

### Q6_K real-model milestone

The pinned Qwen Q6_K source, file size, and SHA-256 are recorded in [`model-source.json`](measurements/phase5/model-source.json). The official file mixes Q6_K down-projection matrices with Q8_0 tensors. Q6_K blocks keep 256 weights as `ql[128]`, `qh[64]`, signed `scales[16]`, and `f16 d` (210 bytes). Ferrum stores that layout in packed GPU storage; the MPP tile decodes paired values sharing each ql byte. The original and optimized three-pair matrices are retained.

| Workload | BF16 prefill | Q6_K prefill | BF16 cached decode | Q6_K cached decode | BF16 complete generation | Q6_K complete generation | BF16 / Q6_K first token |
|---|---:|---:|---:|---:|---:|---:|---:|
| short (21 prompt, 17 generated) | 656 | 608 | 97 | 114 | 84 | 95 | 32 / 35 ms |
| 128 prompt, 17 generated | 6,168 | 2,884 | 97 | 114 | 90 | 90 | 21 / 45 ms |
| 512 prompt, 17 generated | 8,377 | 4,214 | 92 | 109 | 71 | 62 | 61 / 122 ms |
| 1,024 prompt, 17 generated | 6,645 | 4,114 | 86 | 99 | 49 | 41 | 154 / 249 ms |
| short prompt, 129 generated | 500 | 524 | 94 | 110 | 90 | 105 | 42 / 40 ms |
| 368 prompt, 1,601 generated | 6,438 | 3,600 | 86 | 97 | 84 | 94 | 58 / 103 ms |

Medians are tok/s across three alternating pairs; the raw matrix includes token IDs, per-token decode times, KV use, and counters. Q6_K retained 499,645,184 weight bytes, of which 499,502,080 were packed, versus 988,065,536 BF16 bytes. The process peak RSS was 2,242,265,088 bytes with both models loaded. At the 512-token prompt, the optimized run used 117.98 ms phase GPU time vs 57.83 ms BF16, 531 dispatches, five command buffers, 58 allocations, 1.37 ms allocation time, and a 268,042,240-byte transient peak. The paired-plane change reduced Q6_K GPU time from 128.32 to 116.55 ms on the focused three-pair run, with unchanged dispatch and memory counters.

The 449-position / 448-target teacher-forced probe measured mean absolute logit error 0.3181, RMSE 0.4067, cosine similarity 0.99265, top-1 agreement 99.55%, top-5 overlap 85.92%, and perplexity 1.51473 vs BF16 1.50983 (ratio 1.00324). Token IDs matched BF16 on the short, 128, 512, 1,024, and 1,601-token cases. The sustained 129-token decode diverged on 106 generated positions, showing autoregressive drift despite the strong teacher-forced score. Generated IDs matched exactly between the original and paired-plane Q6_K kernels across all 36 runs.

All four Q6_K Metal kernel tests passed with `MTL_DEBUG_LAYER=1 MTL_SHADER_VALIDATION=1`, including signed per-group scales, multi-superblock decode, embedding gather, and MPP tails. `cargo test --all-targets` passed 68 tests; two pre-existing real-model tests remain environment-gated and ignored. `cargo clippy --all-targets -- -D warnings`, formatting, diff check, and all-target check passed.

### Q5_0/Q5_1 and Q4_K/Q5_K real-model milestone

The pinned official Q4_K_M and Q5_K_M files, byte sizes, revisions, and SHA-256 hashes are recorded in [`model-source.json`](measurements/phase5/model-source.json). Q4_K_M includes Q5_0, Q4_K, Q6_K, and Q8_0 tensors; Q5_K_M includes Q5_1, Q5_K, Q6_K, and Q8_0. GGML type IDs and byte layouts follow the upstream declarations: Q5_0 type 6 (32 values/22 bytes), Q5_1 type 7 (32/24), Q4_K type 12 (256/144), and Q5_K type 13 (256/176). Q5 legacy blocks read the high-bit plane separately; K blocks unpack their 6-bit group scales/minima and format-specific high bits. Packed source bytes stay resident for the model lifetime. The MPP kernels decode only a bounded tile into threadgroup memory.

The complete production matrices use three alternating BF16/quantized pairs for short 21-token, 128-, 512-, 1,024-, sustained 129-token decode, and 368-token plus 1,601-token horizon workloads. Rates are medians in tokens/s; first-token numbers are milliseconds. The paired deltas are quantized versus BF16 within each matrix; compressed raw JSONL preserves each run's IDs, counters, timing, and KV information: [Q4_K_M optimized](measurements/phase5/q4_k_m-matrix.jsonl.gz), [Q4_K_M before GEMV tuning](measurements/phase5/q4_k_m-matrix-before-gemv.jsonl.gz), [Q5_K_M optimized](measurements/phase5/q5_k_m-matrix.jsonl.gz), and [Q5_K_M before GEMV tuning](measurements/phase5/q5_k_m-matrix-before-gemv.jsonl.gz). Focused A/B summaries and the single-pass [Q4_K_M memory run](measurements/phase5/q4_k_m-memory-matrix.jsonl.gz) and [Q5_K_M memory run](measurements/phase5/q5_k_m-memory-matrix.jsonl.gz) are also retained.

| Workload | Q4_K_M prefill BF16 → Q4 | cached decode BF16 → Q4 | complete generation BF16 → Q4 | first token BF16 → Q4 | Q5_K_M prefill BF16 → Q5 | cached decode BF16 → Q5 | complete generation BF16 → Q5 | first token BF16 → Q5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| short (21 prompt, 17 generated) | 659 → 710 (+8%) | 96 → 110 (+15%) | 84 → 95 (+14%) | 32 → 30 (-8%) ms | 656 → 673 (+3%) | 96 → 105 (+10%) | 84 → 91 (+8%) | 32 → 32 (-3%) ms |
| 128 prompt, 17 generated | 6,159 → 3,374 (-45%) | 98 → 112 (+14%) | 90 → 92 (+2%) | 21 → 38 (+82%) ms | 6,132 → 3,225 (-47%) | 97 → 107 (+10%) | 90 → 88 (-2%) | 21 → 40 (+89%) ms |
| 512 prompt, 17 generated | 8,440 → 4,697 (-44%) | 92 → 106 (+15%) | 70 → 64 (-9%) | 61 → 109 (+79%) ms | 8,451 → 4,553 (-46%) | 92 → 103 (+12%) | 71 → 62 (-12%) | 61 → 113 (+84%) ms |
| 1,024 prompt, 17 generated | 6,674 → 4,574 (-32%) | 88 → 98 (+12%) | 50 → 43 (-13%) | 154 → 224 (+47%) ms | 6,609 → 4,402 (-34%) | 88 → 97 (+11%) | 49 → 42 (-14%) | 155 → 233 (+51%) ms |
| short prompt, 129 generated | 465 → 573 (+20%) | 94 → 108 (+15%) | 89 → 103 (+15%) | 45 → 37 (-17%) ms | 474 → 546 (+14%) | 94 → 104 (+10%) | 88 → 99 (+12%) | 45 → 39 (-13%) ms |
| 368 prompt, 1,601 generated | 6,540 → 4,135 (-35%) | 86 → 98 (+15%) | 83 → 95 (+14%) | 57 → 89 (+53%) ms | 6,469 → 3,999 (-38%) | 86 → 94 (+10%) | 84 → 92 (+10%) | 57 → 92 (+60%) ms |

These results show the decode benefit and the prefill cost separately: quantized decode is consistently faster, while larger-prompt prefill and first-token latency remain slower than Phase 4 BF16. Complete generation improves on the sustained/long workloads, but regresses for 512- and 1,024-token prompts. Q4_K_M retained 485,309,184 weight bytes (485,166,080 packed); Q5_K_M retained 516,095,744 (515,952,640 packed), compared with 988,065,536 BF16 weight bytes. `/usr/bin/time -l` peak process footprint was 2,219,854,032 bytes for BF16+Q4_K_M and 2,216,888,528 for BF16+Q5_K_M. The matrices record about 37 MB transient peak on short input, 244–268 MB on prompt prefill, and 2.3–5.4 MB per decode phase. At the profiled 512-token case each phase used one command buffer and one completion wait, with 531 dispatches; decode allocated one 4-byte result while reusing the bounded arena. Per-operation GPU timestamp samples were unavailable. No DRAM bandwidth conclusion is inferred from logical bytes.

The 449-position / 448-target prefill quality probe for Q4_K_M measured mean absolute logit error 0.42993, RMSE 0.54726, cosine 0.98682, top-1 agreement 99.11%, top-5 overlap 81.25%, and perplexity 1.51113 versus BF16 1.50983 (ratio 1.00086). For Q5_K_M it measured 0.38387 mean error, 0.49211 RMSE, 0.98939 cosine, 99.33% top-1, 82.32% top-5 overlap, and perplexity ratio 1.00389. Cached decode was measured on the same 448 fixed next-token targets: Q4_K_M mean error 0.43394, RMSE 0.55216, cosine 0.98659, top-1 99.55%, top-5 overlap 81.12%, and perplexity ratio 1.00573; Q5_K_M 0.38481, 0.49344, 0.98939, 99.33%, 82.50%, and 1.00623 respectively. These are reproducible short-corpus probes, not held-out language-quality evaluations.

The paired matrix's exact generated IDs match BF16 in five of six Q4_K_M workloads and four of six Q5_K_M workloads. The sustained 129-token generation differs for both, and the short Q5_K_M sample differs in nine of 17 tokens. The optimized GEMV reduction order also changes autoregressive IDs relative to the prior scalar-reduction quant kernels: first divergence was token 38 with 90/129 differing for Q4_K_M and token 5 with 121/129 for Q5_K_M. Teacher-forced decode logits remain close to BF16 as measured above, but that does not erase cumulative sampling drift. Keep the paired-byte kernels provisionally for their 9–15% cached-decode gain, retain both raw before/after matrices, and revisit numeric reduction order if additional quality data shows the drift is unacceptable. No expected-token assertions were weakened.

`llama-bench` 0.19.0 was also run three times per workload against the same Q4_K_M/Q5_K_M GGUF files on Apple M5, with `-ngl 99 -ctk bf16 -ctv bf16 -fa on`. Its standalone `pp512`/`tg128` rates were 8,981/224 tok/s for Q4_K_M and 8,752/213 for Q5_K_M. Composite `-pg` rates were 423, 1,542, 3,992, 5,257, and 269 tok/s (Q4_K_M) and 407, 1,486, 3,856, 5,097, and 257 (Q5_K_M) for 21/17, 128/17, 512/17, 1,024/17, and 368/1,601 tokens. Raw rows and stderr are in [`llama-q4_k_m.jsonl`](measurements/phase5/llama-q4_k_m.jsonl), [`llama-q4_k_m-time.txt`](measurements/phase5/llama-q4_k_m-time.txt), [`llama-q5_k_m.jsonl`](measurements/phase5/llama-q5_k_m.jsonl), and [`llama-q5_k_m-time.txt`](measurements/phase5/llama-q5_k_m-time.txt). This is directional only: llama-bench uses synthetic token inputs and its own llama.cpp Metal/TensorOps scheduling, so the composite rate and Ferrum's production corpus/workloop rates are not directly paired. The comparison nevertheless shows that Ferrum still has substantial prefill and decode headroom.

The final Q5_0/Q5_1/Q4_K/Q5_K test pass runs GGUF malformed/truncated payload checks, unsupported type rejection, alignment, metadata and scale/minimum decoding, odd output tails, direct GEMV/GEMM, MPP tiles, and packed embedding gather under Metal API and GPU shader validation. The same source code paths have run short and long-horizon generation using the pinned official files. Raw measurement artifacts also include original and paired-GEMV matrix data, per-case counters, llama.cpp results, process peak-footprint logs, and quality JSON. MLX 0.32.2 and MLX-LM 0.31.3 were used only from an external comparison venv; Ferrum has no runtime or build dependency on them.

### MLX affine-Q4 interoperability milestone

Ferrum accepts the pinned [`mlx-community/Qwen2.5-0.5B-Instruct-4bit`](measurements/phase5/model-source.json) Qwen2 repository at revision `a5339a4131f135d0fdc6a5c8b5bbed2753bbe0f3`. The safetensors importer validates the complete tensor index, names, shapes, dtypes, offsets, and finite F16 metadata before model construction. It streams one tensor at a time and repacks MLX's U32 low-nibble-first weights plus separate F16 scale/bias arrays into explicit 64-value, 36-byte affine blocks. It does not materialize dense matrices. The implemented support is deliberately limited to the pinned Qwen2 architecture, affine mode, 4 bits, group size 64, tied embeddings, and this single-file safetensors layout. Other MLX quantization schemes and sharded layouts reject explicitly.

The layout and kernel decisions were checked against MLX's primary implementation: [MLX-LM model loading/config handling](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/utils.py), [MLX quantized layers](https://github.com/ml-explore/mlx/blob/main/python/mlx/nn/layers/quantized.py), and [MLX's quantized Metal kernels](https://github.com/ml-explore/mlx/blob/main/mlx/backend/metal/kernels/quantized.h). Ferrum contains its own Rust loader and Metal kernels and does not call these libraries at runtime.

The quantized path uses direct M=1 GEMV, direct SIMD GEMM, Metal 4 MPP GEMM for supported prefill shapes, and packed embedding gather. The loader also handles this repository's missing `generation_config.json` and narrowly validated chat-template spelling variant without loosening the dense Qwen tokenizer contract. A real Ferrum CLI generation and a full-model generation under Metal API/GPU shader validation both succeeded. The model retains 277,853,184 packed bytes plus 143,104 dense bytes (277,996,288 bytes total); its safetensors file is 278,064,920 bytes. These totals include small dense norms and other non-quantized tensors, but no persistent dense copy of quantized matrices.

The 449-position / 448-target fixed-corpus quality probe against Phase 4 BF16 reported prefill mean absolute logit error 0.89448, RMSE 1.14641, cosine 0.94003, top-1 agreement 97.55%, top-5 overlap 71.14%, and perplexity 1.52485 vs 1.50983 (ratio 1.00995). Cached-decode cosine was 0.93953, top-1 agreement 97.54%, and perplexity ratio 1.01250. This is the same repeated short smoke corpus used above, not a held-out evaluation.

Against native MLX 0.32.2 / MLX-LM 0.31.3 using the same checkpoint, prompt IDs, F16 activations, and F16 KV cache, Ferrum's logits reached 0.9999895 prefill cosine and 0.9999842 cached-decode cosine. Greedy next-token IDs matched at every compared position; top-5 overlap was 99.51%. Native MLX generation rows also matched Ferrum's generated IDs on all six measured cases. The native MLX timing run followed the Ferrum run rather than being cross-framework interleaved, so this performance comparison is directional.

The three-pair production matrix, alternating BF16 and quantized order within each workload, used the same prompt IDs. Medians are tok/s except first-token latency:

| Workload | BF16 / MLX-Q4 prefill | BF16 / MLX-Q4 cached decode | BF16 / MLX-Q4 complete generation | BF16 / MLX-Q4 first token |
|---|---:|---:|---:|---:|
| short (21 prompt, 17 generated) | 659 / 849 | 96 / 119 | 84 / 105 | 32 / 25 ms |
| 128 prompt, 17 generated | 6,122 / 2,861 | 97 / 120 | 90 / 94 | 21 / 45 ms |
| 512 prompt, 17 generated | 8,418 / 1,708 | 93 / 115 | 71 / 38 | 61 / 300 ms |
| 1,024 prompt, 17 generated | 6,261 / 994 | 87 / 106 | 48 / 14 | 164 / 1,031 ms |
| short prompt, 129 generated | 508 / 775 | 94 / 117 | 89 / 112 | 42 / 27 ms |
| 368 prompt, 1,601 generated | 6,295 / 1,888 | 78 / 101 | 76 / 95 | 59 / 195 ms |

Quantized decode improves across these cases, but larger-prompt prefill remains substantially slower than Phase 4 BF16 and dominates complete generation at 512 and 1,024 prompt tokens. The matrices record 531 dispatches, five command buffers, 58 allocations, and a 268,042,240-byte transient peak for the 512-token prefill; allocation volume/time is similar between the paths. Per-operation GPU timestamp samples were unavailable. The short-case process memory capture with both reference and MLX-Q4 models resident recorded 2,505,965,568 bytes maximum resident set size and 2,511,243,736 bytes peak footprint in [`mlx-affine4-memory-short.stderr.txt`](measurements/phase5/mlx-affine4-memory-short.stderr.txt); the six timed rows are in [`mlx-affine4-memory-short.jsonl`](measurements/phase5/mlx-affine4-memory-short.jsonl).

The new in-process paired kernel experiment alternated MPP and direct SIMD projection execution three times at a 512-token prompt. MPP measured 1,583 prefill tok/s, 323 ms first-token latency, and 0.320 s phase GPU time; direct SIMD measured 216 tok/s, 2,372 ms, and 2.365 s. Dispatches, command buffers, allocation count/bytes, and transient peak were unchanged; all six paired greedy sequences matched. Direct SIMD GEMM is therefore a correctness/portability fallback, not the preferred M5 prefill route. An older file named [`mlx-affine4-vs-bf16-512-no-mpp.jsonl`](measurements/phase5/mlx-affine4-vs-bf16-512-no-mpp.jsonl) is preserved, but the matrix runner at that point did not apply the requested native-matmul override; it was actually the production-default path and must not be interpreted as a direct-vs-MPP comparison. The corrected alternating experiment is [`mlx-affine4-512-mpp-direct-ab.jsonl`](measurements/phase5/mlx-affine4-512-mpp-direct-ab.jsonl).

Raw artifacts include the [Ferrum/BF16 matrix](measurements/phase5/mlx-affine4-vs-bf16-matrix.jsonl), [native MLX matrix](measurements/phase5/mlx-affine4-native-matrix.jsonl), [BF16 quality probe](measurements/phase5/mlx-affine4-quality.json), [native MLX logit parity](measurements/phase5/mlx-affine4-vs-native-mlx.json), corrected [MPP/direct kernel A/B](measurements/phase5/mlx-affine4-512-mpp-direct-ab.jsonl), and the retained but invalidated earlier [no-MPP-labelled run](measurements/phase5/mlx-affine4-vs-bf16-512-no-mpp.jsonl). The native MLX script and logit comparison script are in `tools/phase5_mlx_matrix.py` and `tools/compare_mlx_quant_logits.py`.

## Validation status and remaining work

Phase 4 remains frozen and its reports/history remain untouched. On the current Phase 5 checkpoint, `cargo test --all-targets` passed 90 tests with 0 failures; the two pre-existing official-model lifetime tests remain ignored behind their existing environment gate. `cargo clippy --all-targets -- -D warnings`, `cargo fmt --all -- --check`, `cargo check --all-targets`, and `git diff --check` passed. Q5_0/Q5_1, Q4_K/Q5_K, and MLX affine-Q4 kernels have passed Metal API/GPU shader validation; the three MLX tests cover direct GEMV/GEMM, embedding gather, signed scales, tails, and MPP K/batch/output tails. The real pinned MLX model also generated through Ferrum's CLI with both Metal validation layers enabled. Current raw logs are `validation-cargo-test-all-targets-mlx.txt`, `validation-cargo-clippy-mlx.txt`, `validation-metal-mlx-affine4.txt`, and `validation-mlx-real-generation.txt` under [`measurements/phase5`](measurements/phase5).

Remaining Phase 5 work: evaluate a held-out quality corpus; broaden the narrowly scoped MLX importer only where a concrete repository and format contract warrants it; optimize long-prompt quantized prefill, vocabulary LM-head execution, and projection GEMVs by shape; investigate output-token drift after paired GEMV reductions; extend real-model validation beyond this Qwen family; and retain further accepted/rejected experiments. Ferrum remains slower than llama.cpp on the existing Q4_K_M/Q5_K_M decode comparison. Current shader profiling does not expose per-operation GPU timestamps, so use whole-phase GPU timing and validated Metal profiling tools instead of treating encode wall time as kernel time.
