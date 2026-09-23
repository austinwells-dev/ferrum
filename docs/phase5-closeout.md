# Ferrum Phase 5 closeout

**Status: complete.** Phase 5 delivers native packed quantized inference and a bounded interoperability contract. The Phase 4 BF16 runtime and its historical reports remain intact. Phase 5.5 owns further performance work; Phase 5 ends without requiring parity with MLX or llama.cpp.

The detailed experiment history, six-workload matrices, quality and memory measurements, accepted/rejected experiments, external comparisons, and artifact links are in [`phase5-experiments.md`](phase5-experiments.md).

## Supported formats

The GGUF v2/v3 reader validates common metadata and tensor layouts; the runtime adapter currently constructs Qwen2 models, validated on official Qwen2.5-0.5B-Instruct checkpoints.

| GGML type | ID | Block | Metal support |
|---|---:|---:|---|
| Q4_0 | 2 | 32 values / 18 bytes | GEMV, GEMM/MPP, packed embedding gather |
| Q5_0 | 6 | 32 / 22 | GEMV, GEMM/MPP, packed embedding gather |
| Q5_1 | 7 | 32 / 24 | GEMV, GEMM/MPP, packed embedding gather |
| Q8_0 | 8 | 32 / 34 | GEMV, GEMM/MPP, packed embedding gather |
| Q4_K | 12 | 256 / 144 | GEMV, GEMM/MPP, packed embedding gather |
| Q5_K | 13 | 256 / 176 | GEMV, GEMM/MPP, packed embedding gather |
| Q6_K | 14 | 256 / 210 | GEMV, GEMM/MPP, packed embedding gather |

GGUF dense F32, F16, and BF16 tensors are accepted. Q4_1, Q8_1, Q2_K, Q3_K, and Q8_K have index geometry for safe range validation but are not runtime-loadable. Other unknown/quantized GGML types, including IQ families, reject explicitly.

MLX support is limited to the pinned [`mlx-community/Qwen2.5-0.5B-Instruct-4bit`](measurements/phase5/model-source.json) revision `a5339a4131f135d0fdc6a5c8b5bbed2753bbe0f3`: Qwen2, affine Q4, group size 64, tied embeddings, one safetensors file. Ferrum repacks to its own 64-value packed layout and has no MLX runtime dependency. This is not general MLX repository support.

## Correctness and quality

Quantized matrices remain packed in persistent storage. GEMV and embedding gather decode directly from packed weights; GEMM decodes bounded tiles next to accumulation. No full-model BF16/F16 quantized copy is retained. Loaders stream tensor payloads and reject malformed ranges, unsupported types, incompatible layouts, and invalid metadata explicitly.

Each tested variant has a fixed-stream teacher-forced comparison to Phase 4 BF16. Across the 449-position / 448-target smoke corpus, whole-model logit cosine ranges from 0.94003 (MLX affine-Q4) to 0.99323 (GGUF Q8_0); top-1 agreement ranges from 97.55% to 99.55%, and perplexity ratios range from 0.99904 to 1.01178. These are reproducible format checks, not held-out language evaluation. Token drift is recorded; no expected-token assertions or tolerances were weakened.

Retained quantized weight bytes are 422,496,256 for Q4_0; 499,502,080 for Q6_K; 524,833,792 for Q8_0; 485,166,080 for Q4_K_M; 515,952,640 for Q5_K_M; and 277,853,184 for pinned MLX affine-Q4. Dense norms and other non-quantized values are counted separately. Paired process peak-RSS and bounded transient measurements are in the experiment journal.

## Performance record

Production matrices cover short, 128, 512, 1,024, sustained 129-token decode, and 1,601-token horizon workloads. They include paired Phase 4 BF16 results, prefill/cached-decode/complete-generation throughput, first-token latency, timing counters, dispatch and allocation counts, KV use, and generated IDs. Full results and external comparison limitations are in [`phase5-experiments.md`](phase5-experiments.md#final-production-performance-matrix).

The last Phase 5 experiment, completed before closeout, accepted an 8-output-row Q8_0 GEMV route for aligned output counts at least 128: three alternating sustained-decode pairs improved median cached decode 7.92%, with exact IDs in each pair. The final Q8 matrix preserved prior-kernel IDs in all 18 workload/pair runs. The A/B runner override was removed; raw measurements remain.

## Final validation

- `MTL_DEBUG_LAYER=1 MTL_SHADER_VALIDATION=1 cargo test --all-targets`: 90 passed, 0 failed. The two environment-gated official-model tests were then run explicitly; both passed, including 128-step lifetime/cache stress.
- Final Q8 odd-tail/aligned-tile tests and official Qwen short generation passed with Metal API and GPU shader validation enabled. GGUF Q4_0/Q5/Q6_K and pinned MLX kernels and real-model generation have retained Metal-validation logs: [`q4_0-metal-validation.txt`](measurements/phase5/q4_0-metal-validation.txt), [`validation-metal-q5-qk.txt`](measurements/phase5/validation-metal-q5-qk.txt), [`q6_k-metal-validation.txt`](measurements/phase5/q6_k-metal-validation.txt), [`validation-metal-mlx-affine4.txt`](measurements/phase5/validation-metal-mlx-affine4.txt), and [`validation-mlx-real-generation.txt`](measurements/phase5/validation-mlx-real-generation.txt).
- `cargo fmt --all -- --check`, `cargo check --all-targets`, `cargo clippy --all-targets -- -D warnings`, and `cargo build --release --all-targets` passed.
- Cache ordering, failure behavior, snapshots/branches, malformed GGUF inputs, unsupported type rejection, quantized tails/metadata, and Phase 4 BF16 fallback are covered by the final suite and recorded tests.

## Phase 5.5 handoff

Optimize quantized M=1 GEMV and the LM head; specialize Q/K/V and MLP decode shapes; improve quantized MPP/GEMM prefill and dequantization/matmul fusion; tune tile geometry, dispatch thresholds, and scale/metadata locality; run matched MLX and llama.cpp performance comparisons; and continue broad Q8/Q6/Q5/Q4 tuning with paired measurements and quality checks. Per-operation GPU timestamp attribution and DRAM counters were unavailable in Phase 5, so Phase 5.5 should use validated profiling tools and avoid inferring hardware bandwidth from logical weight traffic.
