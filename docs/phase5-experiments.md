# Phase 5 experiment journal

This journal starts from the finalized Phase 4 checkpoint. Phase 4 reports and raw measurements remain historical records and are not edited by Phase 5.

## Status

Phase 5 is active. The strict GGUF/Qwen2 adapter, packed Q8_0 storage, direct Q8_0 decode GEMV, tiled Q8_0 prefill GEMM, and real-model generation path are implemented. Q8_0 has a three-pair production matrix, a 448-target teacher-forced quality probe, and Metal API/GPU shader validation. This is the first validated runtime milestone; Q4/Q5/Q6 formats, MLX interchange, broader quality data, and remaining optimization are still outstanding.

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

Add a distinct packed matrix weight with logical `[out,in]` dimensions, an explicit upstream format tag, and owned Metal storage whose byte length is computed from that format. The current format is GGUF `Q8_0`: 32 values per block, 34 bytes per block (`f16 d` followed by 32 signed quant bytes), decoded as `d * q`. Keep these bytes packed for the lifetime of the model. The GEMV reads packed blocks directly; the MPP GEMM expands only one 64-by-128 BF16 tile (16 KiB) into threadgroup memory immediately before accumulation. No complete dequantized matrix is stored. Quantized embeddings use direct row gather/decode.

Keep the packed storage abstraction separate from ordinary tensor arithmetic. Its Metal binding must carry its byte length and lifetime through the same submission/resource-retention mechanism as `Tensor`; it must not be mappable or reusable before successful completion. It introduces no unsafe `Send`/`Sync` implementation and no unsafe code outside `src/metal`.

### GGUF index and model adapter

Parse GGUF independently of Qwen naming. Read the versioned header, typed metadata, tensor descriptors, alignment and relative tensor offsets with checked arithmetic and explicit size bounds. Index dimensions in GGUF order; expose a runtime `[out,in]` matrix by reversing the descriptor dimensions while preserving row-major payload order. Validate required tensor byte lengths from the exact GGML type and block geometry, file bounds, duplicate names, duplicate/overlapping tensor ranges, dimensions, metadata types, and required architecture fields before allocating runtime weights. Unknown or unsupported types fail with the tensor name and type code rather than being interpreted as another format.

Initially accept GGUF v2/v3 little-endian indices and Qwen2 architecture metadata, with dense F32/F16/BF16 plus Q8_0 tensor loading. Stream tensor payloads into packed or dense Metal storage one tensor at a time so a complete second CPU copy of a large model is not retained. The Qwen adapter maps standard GGUF names and metadata into the current strict model contract. Tokenizer metadata is checked against Ferrum's Qwen tokenizer behavior; BF16 and GGUF prompt token IDs match on both benchmark prompts and the quality corpus.

The parser follows the upstream [GGUF specification](https://github.com/ggml-org/ggml/blob/master/docs/gguf.md) and [ggml block declarations](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-common.h). Quantization type IDs and layouts come from GGML; Ferrum will not define private on-disk variants.

### Execution split

- M=1 decode: a dedicated Q8_0 Metal GEMV reads coalesced packed rows, uses four independent reduction chains per SIMD lane, and writes output directly without a whole-weight intermediate. The tied vocabulary projection and transformer projections are both exercised by production generation.
- M>1 prefill: use the Q8_0 MPP GEMM for BF16 activation batches of at least 16 rows when K is a multiple of 128 and Metal 4 is available. It decodes Q8_0 blocks directly into a bounded threadgroup tile, then accumulates 64x64 output tiles over K. Keep the direct SIMD GEMM for unsupported shapes and devices. Apple's [Metal 4 TensorOps guide](https://developer.apple.com/download/files/Metal-Performance-Primitives-Programming-Guide.pdf) documents threadgroup-backed tensor operands and multiply-accumulate cooperative tensors.
- Direct packed embedding gather keeps quantized Q8_0 embeddings from forcing a whole-model dense copy.
- Q8_0 is the first correctness bridge. After it is correct and measured, evaluate upstream `Q6_K`, `Q5_K`, and `Q4_K` (and common legacy `Q4_0`/`Q4_1` where useful), preserving each format's real block geometry and metadata. Do not collapse distinct GGML schemes into one decoder when that costs correctness or speed.
- MLX `quantized` weights use their own group-size/scale representation. Interoperability will use an explicit importer/repacker and must not add MLX as a runtime dependency.

## Measurement definitions and acceptance gates

Use the production Ferrum generation path and the existing Phase 4 workload/matrix conventions. Retain raw JSONL and logs for each run. Record model/load time separately from timed generation. Report prefill tok/s, cached decode tok/s, complete-generation tok/s, first-token latency, GPU/encode/wait time, dispatches, command buffers, allocations and allocation time, active/retained KV, transient peak, retained model bytes, and process memory where available.

Cover short, 128, 512, approximately 1k, and long-horizon workloads. Use repeated paired/alternating A/B runs where the baseline path can be selected in the same build/process; retain exact prompt IDs, generated IDs, and model revisions. Compare Ferrum Q8_0 to Phase 4 BF16 on the same model and prompt. For quality, retain teacher-forced logits and report max/mean absolute error, cosine similarity, top-k agreement, and perplexity on a fixed token corpus where practical; token/prose checks are secondary. Compare MLX/llama.cpp only when weights, tokenizer, prompt IDs, sampling, and cache precision make the comparison meaningful, documenting unavoidable layout and precision differences.

Do not attribute traffic estimates to DRAM bandwidth without hardware counters. Keep before/after measures and rejected experiment artifacts. Change kernel organization only after the first end-to-end correctness and performance baseline exists.

## Format support, experiments, quality, and memory

The journal began before Phase 5 runtime changes; current checkpoint status is below. Q8_0 is implemented and validated end-to-end. Q6_K, Q5_K, Q4_K/Q4_0/Q4_1, Q8_1, and MLX compatibility remain planned, not yet implemented.

| Experiment | Status | Evidence / decision |
|---|---|---|
| Phase 4 BF16 baseline | Accepted reference | `q-final-matrix.jsonl`, `phase4-results.md`, `phase4-matrix-progress.md` |
| GGUF v2/v3 reader and Qwen2 adapter | Accepted for Q8_0 checkpoint | Full suite passes; official tokenizer IDs match; malformed/unknown/overlap cases reject explicitly |
| Q8_0 direct M=1 GEMV | Accepted | `q8_0_gemv`, official Qwen generation matrix; 168 quantized GEMV calls per decode step; Metal API/GPU validation passed |
| Q8_0 direct M>1 GEMM, one SIMD group per channel and four prompt rows | Rejected for primary prefill | `q8-matrix-after-tile4.jsonl`: 512-token median was 206 tok/s versus 7,318 BF16; the raw run remains for comparison. Four independent K chains raised the 128-token result to 253 tok/s, still far behind |
| Q8_0 MPP prefill with on-the-fly 64x128 tile decode | Accepted for supported shapes | `q8-matrix-after-mpp.jsonl`; 2,392 / 3,675 / 3,510 tok/s at 128 / 512 / 1,024 prompts. The direct SIMD path remains the fallback |
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

Remaining limits: only GGUF Q8_0 executes quantized today; Q4/Q5/Q6 kernels and model-format compatibility are not implemented. MPP Q8_0 prefill is still 1.6–2.4x slower than BF16 for longer prompts, and short first-token latency remains slightly higher. The tied vocabulary GEMV and transformer projection GEMVs need per-shape tuning. MLX and llama.cpp comparisons, a standard held-out quality corpus, and final real-model validation for every retained format remain outstanding.
