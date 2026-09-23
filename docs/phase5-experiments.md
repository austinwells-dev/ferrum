# Phase 5 experiment journal

This journal starts from the finalized Phase 4 checkpoint. Phase 4 reports and raw measurements remain historical records and are not edited by Phase 5.

## Status

Phase 5 is closed as a quantization, model-format, correctness, and interoperability milestone. The strict GGUF/Qwen2 adapter retains packed Q8_0, Q4_0, Q5_0, Q5_1, Q4_K, Q5_K, and Q6_K weights and executes native Metal GEMV, GEMM, MPP prefill, and embedding-gather paths. One pinned MLX affine-Q4/group-64 Qwen2 repository is supported through an explicit importer. Real-model generation, paired performance matrices, quality probes, memory measurements, lifetime/cache checks, and Metal validation are recorded below. Further performance optimization is assigned to Phase 5.5. See the concise [Phase 5 closeout](phase5-closeout.md).

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

## Final interoperability contract

The GGUF reader accepts version 2 and 3 little-endian files and validates the complete index, metadata, alignment, block geometry, offsets, and tensor ranges. The inference adapter currently supports the Qwen2 architecture contract (real-model validation uses Qwen2.5-0.5B-Instruct); GGUF parsing is reusable, but this does not claim support for every GGUF architecture, tokenizer, or multi-file packaging scheme.

| Runtime tensor type | GGML type ID | Block geometry | Runtime paths | Real-model coverage |
|---|---:|---:|---|---|
| F32 / F16 / BF16 | 0 / 1 / 30 | dense | Existing dense fallback and mixed GGUF tensors | Qwen2.5-0.5B GGUF variants |
| Q4_0 | 2 | 32 values / 18 bytes | Direct GEMV, SIMD GEMM, MPP GEMM, packed embedding gather | Official Q4_0 file |
| Q5_0 | 6 | 32 / 22 | Direct GEMV, SIMD GEMM, MPP GEMM, packed embedding gather | Mixed Q4_K_M file |
| Q5_1 | 7 | 32 / 24 | Direct GEMV, SIMD GEMM, MPP GEMM, packed embedding gather | Mixed Q5_K_M file |
| Q8_0 | 8 | 32 / 34 | Direct GEMV, SIMD/MPP GEMM, packed embedding gather | Official Q8_0 and mixed K-quant files |
| Q4_K | 12 | 256 / 144 | Direct GEMV, SIMD/MPP GEMM, packed embedding gather | Mixed Q4_K_M file |
| Q5_K | 13 | 256 / 176 | Direct GEMV, SIMD/MPP GEMM, packed embedding gather | Mixed Q5_K_M file |
| Q6_K | 14 | 256 / 210 | Direct GEMV, SIMD/MPP GEMM, packed embedding gather | Official Q6_K and mixed K-quant files |

The GGUF indexer also knows block geometry for Q4_1 (type 3), Q8_1 (9), Q2_K (10), Q3_K (11), and Q8_K (15) so it can validate their tensor ranges. The Ferrum Qwen2 loader does **not** load those types for execution. Every other quantized or unknown GGML type, including IQ families and removed/private encodings, fails explicitly; no type is reinterpreted as a supported format. Dense F32/F16/BF16 are supported as listed above.

The MLX importer is a separate, explicit safetensors path. Its tested scope is `mlx-community/Qwen2.5-0.5B-Instruct-4bit`, revision `a5339a4131f135d0fdc6a5c8b5bbed2753bbe0f3`: Qwen2, affine 4-bit weights, group size 64, tied embeddings, and one `model.safetensors` file. It repacks to a 64-value/36-byte Ferrum packed block. Other MLX quantization modes, group sizes, architectures, untied embeddings, and sharded layouts are unsupported and rejected. MLX is not a Ferrum runtime or build dependency.

## Measurement definitions and acceptance gates

Use the production Ferrum generation path and the existing Phase 4 workload/matrix conventions. Retain raw JSONL and logs for each run. Record model/load time separately from timed generation. Report prefill tok/s, cached decode tok/s, complete-generation tok/s, first-token latency, GPU/encode/wait time, dispatches, command buffers, allocations and allocation time, active/retained KV, transient peak, retained model bytes, and process memory where available.

Cover short, 128, 512, approximately 1k, and long-horizon workloads. Use repeated paired/alternating A/B runs where the baseline path can be selected in the same build/process; retain exact prompt IDs, generated IDs, and model revisions. Compare Ferrum Q8_0 to Phase 4 BF16 on the same model and prompt. For quality, retain teacher-forced logits and report max/mean absolute error, cosine similarity, top-k agreement, and perplexity on a fixed token corpus where practical; token/prose checks are secondary. Compare MLX/llama.cpp only when weights, tokenizer, prompt IDs, sampling, and cache precision make the comparison meaningful, documenting unavoidable layout and precision differences.

Do not attribute traffic estimates to DRAM bandwidth without hardware counters. Keep before/after measures and rejected experiment artifacts. Change kernel organization only after the first end-to-end correctness and performance baseline exists.

## Format support, experiments, quality, and memory

The experiment table preserves accepted and rejected decisions across Phase 5. Q8_0, Q4_0, Q5_0, Q5_1, Q4_K, Q5_K, and Q6_K are implemented and validated on official GGUF model files. A narrowly scoped MLX affine-Q4 Qwen2 importer and Metal path are also implemented and measured; see the MLX milestone below.

| Experiment | Status | Evidence / decision |
|---|---|---|
| Phase 4 BF16 baseline | Accepted reference | `q-final-matrix.jsonl`, `phase4-results.md`, `phase4-matrix-progress.md` |
| GGUF v2/v3 reader and Qwen2 adapter | Accepted for the documented Qwen2 contract | Full suite passes; official tokenizer IDs match; malformed/unknown/overlap cases reject explicitly |
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
| MLX Q4 MPP K=64 tile for 512<=M<=1,024 | Accepted provisionally | Paired MPP K=128/K=64 A/B was neutral at M=128, +2.0% at M=512, and +0.5% at M=1,024; all measured greedy sequences matched. Keep K=128 outside the measured interval |
| Phase 5 paired matrix summarizer | Accepted | `tools/summarize_phase5_matrix.py` reports each model separately and paired A/B deltas; it avoids the old cross-model aggregation in the Phase 4 helper |
| Dispatch threshold 16 prompt rows | Accepted provisionally | On the 21-token short prompt, Q8_0 first-token median improved from 113.0 ms (direct path) to 42.6 ms (MPP), versus 36.0 ms BF16; `q8-mpp-short.jsonl` |
| Q8_0 four-chain GEMV reduction | Accepted provisionally | Cached decode in the full paired matrix is 19–24% faster than BF16 for this model/run; individual before/after result is in `q8-matrix-after-tile4.jsonl`. Further shape-specific work belongs to Phase 5.5 |
| Q8_0 8-output-row M=1 GEMV with shared activation loads | Accepted for aligned `N>=128` shapes | Three alternating 129-token pairs measured median cached decode at 111.35 → 120.28 tok/s (+7.92%) versus the original 4-row tile; generated IDs matched exactly in every pair. Full post-change matrix matches the prior Q8 IDs in all 18 workload/pair runs. Raw A/B and matrix: [`q8_0-129decode-gemv-4rows-vs-8rows.jsonl`](measurements/phase5/q8_0-129decode-gemv-4rows-vs-8rows.jsonl), [`q8-matrix-after-8row-gemv.jsonl`](measurements/phase5/q8-matrix-after-8row-gemv.jsonl) |

## Validation status and remaining bottlenecks

Phase 4 validation remains recorded in its own journal. At the initial Q8_0 checkpoint, `cargo test --all-targets` passed (57 tests); the two existing official-model tests were then environment-gated. The final Phase 5 validation below reruns the full suite, enables those real-model tests, and validates the final Q8 kernel with Metal API and GPU shader validation. The full Q8 matrix exercises official Qwen generation through 1,601 generated tokens.

The teacher-forced fixed corpus repeated 16 times produced 449 positions / 448 targets: mean absolute logit error 0.3048, RMSE 0.3903, cosine similarity 0.99323, top-1 agreement 99.55%, top-5 overlap 86.01%, and perplexity 1.50838 versus BF16 1.50983 (ratio 0.99904). This repeated short text is a reproducible smoke corpus, not a broad language-quality evaluation; this evidence limit is explicit in the closeout.

The original Q8 matrix used three alternating model pairs per workload and the same tokenizer IDs; it remains in [`q8-matrix-after-mpp.jsonl`](measurements/phase5/q8-matrix-after-mpp.jsonl). The final matrix reran all workloads after enabling the accepted 8-row M=1 kernel for aligned output dimensions of at least 128. Medians are tok/s; first-token values are milliseconds. Raw rows retain exact token IDs, command/counter data, and model revisions in [`q8-matrix-after-8row-gemv.jsonl`](measurements/phase5/q8-matrix-after-8row-gemv.jsonl).

| Workload | BF16 prefill | Q8_0 prefill | BF16 cached decode | Q8_0 cached decode | BF16 complete generation | Q8_0 complete generation | BF16 / Q8_0 first token |
|---|---:|---:|---:|---:|---:|---:|---:|
| short (21 prompt, 17 generated) | 656 | 541 | 95 | 123 | 83 | 99 | 32 / 39 ms |
| 128 prompt, 17 generated | 6,089 | 2,632 | 97 | 125 | 90 | 96 | 21 / 49 ms |
| 512 prompt, 17 generated | 8,406 | 4,047 | 92 | 118 | 70 | 64 | 61 / 127 ms |
| 1,024 prompt, 17 generated | 6,602 | 3,885 | 87 | 109 | 49 | 41 | 155 / 264 ms |
| short prompt, 129 generated | 492 | 453 | 93 | 121 | 89 | 114 | 43 / 47 ms |
| 368 prompt, 1,601 generated | 6,468 | 3,284 | 79 | 105 | 76 | 102 | 57 / 112 ms |

The profiled runs record 168 Q8 quantized projection calls per cached decode step; the focused 8-row test reduced whole-decode phase GPU time from 8.305 to 7.619 ms with unchanged 531 dispatches and one command buffer. Per-operation Metal GPU counter samples were unavailable; whole-phase GPU, encode, wait, dispatch, command-buffer, allocation, and allocation-time counters are retained in the raw matrices. The process peak in the earlier paired run was 2,248,605,696 bytes while both BF16 and Q8 models were loaded. Q8 retained weights were 524,976,896 bytes (524,833,792 packed); BF16 retained weights were 988,065,536 bytes. Q8's measured transient peaks matched BF16 at each workload (about 37 MB short and 244–268 MB for the longer prefills). No DRAM-bandwidth claim is made from logical traffic.

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

The 449-position / 448-target teacher-forced quality probe produced mean absolute logit error 0.7074, RMSE 0.9037, cosine similarity 0.96486, top-1 agreement 98.22%, top-5 overlap 74.12%, and perplexity 1.52762 vs BF16 1.50983 (ratio 1.01178). BF16 and GGUF tokenizers returned identical IDs. The repeated short corpus is a reproducible smoke test rather than broad quality validation; see the closeout for this limitation.

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

At the Q6_K checkpoint all four Q6_K Metal kernel tests passed with `MTL_DEBUG_LAYER=1 MTL_SHADER_VALIDATION=1`, including signed per-group scales, multi-superblock decode, embedding gather, and MPP tails. The then-current `cargo test --all-targets` run passed 68 tests; the two existing real-model tests are now explicitly run and recorded in final validation below. Clippy, formatting, diff check, and all-target check passed at that checkpoint as well.

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
| 512 prompt, 17 generated | 8,383 / 1,749 | 93 / 115 | 71 / 39 | 61 / 293 ms |
| 1,024 prompt, 17 generated | 6,254 / 1,010 | 88 / 106 | 48 / 15 | 164 / 1,014 ms |
| short prompt, 129 generated | 508 / 775 | 94 / 117 | 89 / 112 | 42 / 27 ms |
| 368 prompt, 1,601 generated | 6,295 / 1,888 | 78 / 101 | 76 / 95 | 59 / 195 ms |

Quantized decode improves across these cases, but larger-prompt prefill remains substantially slower than Phase 4 BF16 and dominates complete generation at 512 and 1,024 prompt tokens. The matrices record 531 dispatches, five command buffers, 58 allocations, and a 268,042,240-byte transient peak for the 512-token prefill; allocation volume/time is similar between the paths. Per-operation GPU timestamp samples were unavailable. The short-case process memory capture with both reference and MLX-Q4 models resident recorded 2,505,965,568 bytes maximum resident set size and 2,511,243,736 bytes peak footprint in [`mlx-affine4-memory-short.stderr.txt`](measurements/phase5/mlx-affine4-memory-short.stderr.txt); the six timed rows are in [`mlx-affine4-memory-short.jsonl`](measurements/phase5/mlx-affine4-memory-short.jsonl).

The new in-process paired kernel experiment alternated MPP and direct SIMD projection execution three times at a 512-token prompt. MPP measured 1,583 prefill tok/s, 323 ms first-token latency, and 0.320 s phase GPU time; direct SIMD measured 216 tok/s, 2,372 ms, and 2.365 s. Dispatches, command buffers, allocation count/bytes, and transient peak were unchanged; all six paired greedy sequences matched. Direct SIMD GEMM is therefore a correctness/portability fallback, not the preferred M5 prefill route. An older file named [`mlx-affine4-vs-bf16-512-no-mpp.jsonl`](measurements/phase5/mlx-affine4-vs-bf16-512-no-mpp.jsonl) is preserved, but the matrix runner at that point did not apply the requested native-matmul override; it was actually the production-default path and must not be interpreted as a direct-vs-MPP comparison. The corrected alternating experiment is [`mlx-affine4-512-mpp-direct-ab.jsonl`](measurements/phase5/mlx-affine4-512-mpp-direct-ab.jsonl).

The MPP tile-width experiment was motivated by MLX's quantized NAX path using K=64, while Apple's MPP guide calls K=128 a good M5 starting point for general cooperative GEMM; the paths are different, so Ferrum measured both directly. Alternating three-pair tests found MPP K=64 neutral at 128 prompt tokens, +2.0% prefill throughput at 512, and +0.5% at 1,024. Phase GPU time moved from 295.1 to 289.1 ms at 512 and 1,001.8 to 997.0 ms at 1,024; all generated IDs matched between tile widths. The shape policy now selects K=64 only for 512<=M<=1,024 and keeps K=128 outside that measured interval. Focused post-policy BF16 comparisons are retained for [512](measurements/phase5/mlx-affine4-512-shape-k64-matrix.jsonl) and [1,024](measurements/phase5/mlx-affine4-1024-shape-k64-matrix.jsonl); raw alternating tile A/B records are [128](measurements/phase5/mlx-affine4-128-mpp-k128-k64-ab.jsonl), [512](measurements/phase5/mlx-affine4-512-mpp-k128-k64-ab.jsonl), and [1,024](measurements/phase5/mlx-affine4-1024-mpp-k128-k64-ab.jsonl). This is a small, provisional improvement and does not close the prefill gap to BF16 or native MLX. See [Apple's MPP programming guide](https://developer.apple.com/download/files/Metal-Performance-Primitives-Programming-Guide.pdf) and [MLX's quantized dispatch](https://github.com/ml-explore/mlx/blob/main/mlx/backend/metal/quantized.cpp).

Raw artifacts include the [Ferrum/BF16 matrix](measurements/phase5/mlx-affine4-vs-bf16-matrix.jsonl), [native MLX matrix](measurements/phase5/mlx-affine4-native-matrix.jsonl), [BF16 quality probe](measurements/phase5/mlx-affine4-quality.json), [native MLX logit parity](measurements/phase5/mlx-affine4-vs-native-mlx.json), corrected [MPP/direct kernel A/B](measurements/phase5/mlx-affine4-512-mpp-direct-ab.jsonl), [MPP K-tile A/B matrices](measurements/phase5/mlx-affine4-128-mpp-k128-k64-ab.jsonl), post-policy production matrices for [512](measurements/phase5/mlx-affine4-512-shape-k64-matrix.jsonl) and [1,024](measurements/phase5/mlx-affine4-1024-shape-k64-matrix.jsonl), and the retained but invalidated earlier [no-MPP-labelled run](measurements/phase5/mlx-affine4-vs-bf16-512-no-mpp.jsonl). The native MLX script and logit comparison script are in `tools/phase5_mlx_matrix.py` and `tools/compare_mlx_quant_logits.py`.

## Phase 5 closeout

Phase 5 is complete as a feature, correctness, and interoperability milestone. Phase 4 remains unchanged and is still the dense BF16 reference/fallback. Quantized tensors use owned packed storage for the model lifetime; direct GEMV and embedding kernels consume the packed representation, while GEMM paths decode only bounded threadgroup tiles. No complete quantized model or persistent dequantized matrix copy is created. GPU resources remain retained through command completion, and the final validation includes cache staging/publication, failures, branches, snapshots, and the 128-step lifetime stress.

### Final quality measurements

These are teacher-forced comparisons against the Phase 4 BF16 checkpoint on the repeated fixed smoke corpus (449 logits positions / 448 target tokens). They measure approximation error; they are not a held-out language benchmark. Q4_K_M and Q5_K_M are mixed-quantization GGUFs, so their whole-model scores do not isolate one tensor type. Perplexity ratios use the corresponding BF16 result.

| Loaded variant | Mean abs. logit error | RMSE | Cosine | Top-1 agreement | Top-5 overlap | Perplexity ratio |
|---|---:|---:|---:|---:|---:|---:|
| GGUF Q8_0 | 0.30484 | 0.39034 | 0.99323 | 99.55% | 86.01% | 0.99904 |
| GGUF Q4_0 | 0.70736 | 0.90369 | 0.96486 | 98.22% | 74.12% | 1.01178 |
| GGUF Q6_K | 0.31811 | 0.40665 | 0.99265 | 99.55% | 85.92% | 1.00324 |
| GGUF Q4_K_M | 0.42993 | 0.54726 | 0.98682 | 99.11% | 81.25% | 1.00086 |
| GGUF Q5_K_M | 0.38387 | 0.49211 | 0.98939 | 99.33% | 82.32% | 1.00389 |
| Pinned MLX affine-Q4 | 0.89448 | 1.14641 | 0.94003 | 97.55% | 71.14% | 1.00995 |

The BF16 corpus perplexity is 1.50983. Cached-decode probes are also retained for Q4_K_M, Q5_K_M, and MLX affine-Q4. MLX-versus-BF16 cached-decode cosine is 0.93953 with 97.54% top-1 agreement and a 1.01250 perplexity ratio. The complete JSON artifacts retain max error, token IDs, exact corpus, and both prefill/decode metrics. Quantization can change autoregressive output: for example, Q8 differs from BF16 on the short and sustained-decode samples, and the paired-byte Q4_K_M/Q5_K_M reduction kernels have additional measured sequence drift recorded above. No token expectation or numerical tolerance was weakened.

### Retained weights and memory

The following retained sizes are measured from Ferrum model storage. “Packed” excludes the small dense norms and other non-quantized tensors. Process peaks were captured with both the BF16 reference and quantized model resident; these include the runtime and allocator and are not the quantized model size alone.

| Variant | Retained weights | Packed quantized bytes | Paired-process peak RSS |
|---|---:|---:|---:|
| BF16 reference | 988,065,536 | — | — |
| GGUF Q8_0 | 524,976,896 | 524,833,792 | 2,248,605,696 |
| GGUF Q4_0 | 422,639,360 | 422,496,256 | 2,243,559,424 |
| GGUF Q6_K | 499,645,184 | 499,502,080 | 2,242,265,088 |
| GGUF Q4_K_M | 485,309,184 | 485,166,080 | 2,219,854,032 |
| GGUF Q5_K_M | 516,095,744 | 515,952,640 | 2,216,888,528 |
| Pinned MLX affine-Q4 | 277,996,288 | 277,853,184 | 2,505,965,568 |

The Qwen 512-token prefill transient peak is 268,042,240 bytes for the Q4_0, Q6_K, Q4_K_M, Q5_K_M, and MLX runs, and 244–268 MB across Q8 workloads. Short-run transients are about 37 MB; decode phases report roughly 2.3–5.4 MB. Streaming loaders retain tensor data one tensor at a time. Runtime measurements and storage accounting show the expected packed model sizes and bounded temporary tiles; they show no whole-model BF16/F16 expansion. No bandwidth saturation is inferred from logical byte counts.

### Final production performance matrix

The full six-workload matrices earlier in this journal contain paired BF16 baselines, first-token latency, complete-generation rates, per-token decode time, counters, KV use, and exact IDs. The compact view below gives the most useful cross-variant checkpoints: 512-token prefill, the 129-token sustained decode case, and the 1,601-token generation horizon. Rates are medians in tok/s from each variant's three-pair production matrix. MLX's 512-token value is from the post-policy K=64 matrix; its sustained/long values are from the paired production matrix. Compare against each matrix's paired BF16 rows, since clock conditions varied between separate variant runs.

| Variant | Prefill at M=512 | Cached decode, 129 generated | Complete generation, 129 generated | Complete generation, 1,601 generated |
|---|---:|---:|---:|---:|
| GGUF Q8_0 (final 8-row GEMV) | 4,047 | 121 | 114 | 102 |
| GGUF Q4_0 | 5,087 | 93 | 90 | 90 |
| GGUF Q6_K | 4,214 | 110 | 105 | 94 |
| GGUF Q4_K_M | 4,697 | 108 | 103 | 95 |
| GGUF Q5_K_M | 4,553 | 103 | 99 | 92 |
| Pinned MLX affine-Q4 | 1,749 | 117 | 112 | 95 |

The completed Q8 4-row/8-row A/B measured 111.35 → 120.28 cached decode tok/s over three alternating 129-token pairs (+7.92% median); all three pairs generated identical IDs. The full final matrix retained exact Q8 IDs relative to the previous kernel in all 18 workload/pair comparisons. The 8-row route is the retained shape policy for aligned output counts `N>=128`; smaller and irregular rows use the original kernel. The raw A/B, post-change full matrix, and Metal-validation real-generation output are linked in the Q8 section and [`measurements/phase5`](measurements/phase5).

Across the paired per-variant matrices, quantized cached decode improves over BF16 on most measured shapes, while quantized prefill and first-token latency trail BF16 for longer prompts; full-generation impact depends on how much decode offsets prefill. Ferrum also trails llama.cpp on the measured Q4_K_M/Q5_K_M runs. These are recorded performance limits, not blockers for this closeout.

### External interoperability and comparison limits

For the pinned MLX repository, Ferrum and native MLX used the same weights revision, prompt token IDs, F16 activations, and F16 KV cache. Prefill/decode logit cosine was 0.9999895/0.9999842, greedy IDs matched, top-5 overlap was 99.51%, and the six generation cases matched. Timing was run sequentially, not interleaved, so it is directional rather than a matched performance comparison. Ferrum has no MLX runtime dependency.

`llama-bench` 0.19.0 measured Q4_K_M and Q5_K_M with full Metal offload, BF16 KV, and flash attention. It reports synthetic `pp512`/`tg128` inputs and llama.cpp's own scheduling, so its throughput cannot be directly paired with Ferrum's tokenizer prompts and production generation loop. These runs establish a useful external reference and show remaining headroom; Phase 5 does not require Ferrum to match either engine. The exact commands, rates, and raw logs remain linked in the Q5/K-quant section.

### Final validation

| Gate | Result | Evidence |
|---|---|---|
| Full tests with Metal API and GPU shader validation | 90 passed, 0 failed; two pre-existing local-checkpoint tests were explicitly enabled and run separately | [`validation-final-cargo-test-all-targets.txt`](measurements/phase5/validation-final-cargo-test-all-targets.txt) |
| Phase 4 BF16 cached-generation and 128-step lifetime/cache stress | Both real-model tests passed; snapshot re-forward equality, bounded waits, and cache byte counts checked | [`validation-final-real-model-lifetime.txt`](measurements/phase5/validation-final-real-model-lifetime.txt) |
| Q8 final kernel correctness and real generation | Odd output tails, MPP/GEMM paths, aligned 8-row route, and official Qwen short generation passed with both Metal validation layers enabled | [`validation-metal-q8-gemv-8rows.txt`](measurements/phase5/validation-metal-q8-gemv-8rows.txt), [`validation-metal-q8-real-short.jsonl`](measurements/phase5/validation-metal-q8-real-short.jsonl) |
| Q4/Q5/Q6_K and MLX Metal kernels | Direct GEMV/GEMM, MPP, metadata/scales, embedding, tails, malformed inputs and real pinned-model generation have retained validation logs | [`validation-metal-q5-qk.txt`](measurements/phase5/validation-metal-q5-qk.txt), [`validation-metal-mlx-affine4.txt`](measurements/phase5/validation-metal-mlx-affine4.txt), [`validation-mlx-real-generation.txt`](measurements/phase5/validation-mlx-real-generation.txt) |
| Formatting, all-target check, Clippy, release build | All passed | [`validation-final-format.txt`](measurements/phase5/validation-final-format.txt), [`validation-final-cargo-check-all-targets.txt`](measurements/phase5/validation-final-cargo-check-all-targets.txt), [`validation-final-cargo-clippy.txt`](measurements/phase5/validation-final-cargo-clippy.txt), [`validation-final-cargo-build-release-all-targets.txt`](measurements/phase5/validation-final-cargo-build-release-all-targets.txt) |
| Phase 4 history and fallback | Preserved; no Phase 4 report or baseline was rewritten | Phase 4 reports and `q-final-matrix.jsonl` unchanged |

### Phase 5.5 performance handoff

Phase 5.5 owns the following work. Phase 5 performs no further optimization experiments.

- Optimize quantized M=1 GEMV and the large vocabulary LM head.
- Develop shape-specific Q/K/V and MLP decode kernels, with representative shape measurements.
- Improve quantized MPP/GEMM prefill and fuse dequantization with matrix multiplication where it helps.
- Revisit tile geometry, dispatch policy, scale/metadata locality, and activation/weight reuse.
- Run genuinely matched MLX and llama.cpp performance comparisons with identical prompts, cache precision, sampling, and timed-region definitions where possible.
- Tune Q8, Q6, Q5, and Q4 formats broadly; retain accepted and rejected results and quality checks for each change.
- Add a held-out language corpus and broaden architecture/repository coverage only when a concrete model-format contract is selected.

Shader profiling currently lacks per-operation GPU timestamps. Phase 5.5 should use validated GPU profiling/counters for kernel attribution; encode wall time is not GPU execution time, and logical traffic is not a DRAM counter.
