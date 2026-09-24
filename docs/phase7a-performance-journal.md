# Phase 7A GGUF performance journal

## Scope and gate

Phase 7A starts on `codex/phase7-gguf-performance`, from the verified Phase 6 checkpoint `66d721d9bfab04339ad170a426030c847f2d2028`. The first campaign establishes matched small-model baselines and tests general GGUF kernel improvements. The larger Qwen3.8 and Qwen3.6 targets remain behind the performance gate.

The gate is at least 0.85x llama.cpp decode throughput and 0.80x prefill throughput for medium and long prompts, with no routine workload below 0.70x without an isolated explanation. No measured architecture/format currently clears it.

## Machine and reference runtime

- Machine: Apple M5, Mac17,2, 32 GiB unified memory; macOS 27.0; Metal 4 / Apple GPU family 10.
- Ferrum: release build from this branch, native Rust and Metal kernels.
- Reference: official `ggml-org/llama.cpp` at commit `9710a32175b3b8f04636aaac4aa3cc28651b505d` (2026-09-24), built with Metal Release, embedded Metal library, OpenMP off, tests/examples/server off. The paired C API runner is built from `tools/phase55_llama_matrix.cpp`.
- Both runtimes use the same GGUF bytes and exact prompt token IDs per pair, greedy argmax, BF16 KV, context 4096, batch 2048, microbatch 512, four CPU threads, and all model layers on Metal. Model loading and one two-token warmup per workload are outside measured generation. Three interleaved pairs were recorded for each workload.
- Prefill, first-token latency, cached decode, full-generation throughput, output IDs, KV allocation, and process RSS are retained separately in the JSONL records. Prompt tokenization is outside the timed path.

## Artifacts and workload coverage

| Model / format | Pinned artifact | SHA-256 | Workload file | Matched matrix |
| --- | --- | --- | --- | --- |
| Qwen2.5-0.5B Q4_0 | Phase 5 pinned `Q4_0.gguf` | `7671c0c304e6ce5a7fc577bcb12aba01e2c155cc2efd29b2213c95b18edaf6ed` | `qwen2.5-q4k-workloads.jsonl` | `qwen2.5-q4_0-current.jsonl` |
| Qwen2.5-0.5B Q4_K_M | Phase 5 pinned `Q4_K_M.gguf` | `74a4da8c9fdbcd15bd1f6d01d621410d31c6fc00986f5eb687824e7b93d7a9db` | `qwen2.5-q4k-workloads.jsonl` | Initial: `qwen2.5-q4_k_m-current.jsonl`; same-period baseline: `qwen2.5-q4_k_m-baseline-recheck.jsonl` |
| Qwen2.5-0.5B Q5_K_M | Phase 5 pinned `Q5_K_M.gguf` | `041474553fcabfc2a2d67903f9d2c2e50bd92528e670da4f33b5d0ce6e59fd55` | `qwen2.5-q4k-workloads.jsonl` | `qwen2.5-q5_k_m-current.jsonl` |
| Qwen2.5-0.5B Q6_K | Phase 5 pinned `Q6_K.gguf` | `2f82233630c349ccf6b8daccf48f9a7865713d9f08a2eadfa456cebe9b97c7f5` | `qwen2.5-q4k-workloads.jsonl` | `qwen2.5-q6_k-current.jsonl` |
| Qwen2.5-0.5B Q8_0 | Phase 5 pinned `Q8_0.gguf` | `ca59ca7f13d0e15a8cfa77bd17e65d24f6844b554a7b6c12e07a5f89ff76844e` | `qwen2.5-q4k-workloads.jsonl` | `qwen2.5-q8_0-current.jsonl` |
| Qwen3-0.6B Q8_0 | `Qwen/Qwen3-0.6B-GGUF`, revision `23749fefcc72300e3a2ad315e1317431b06b590a` | `9465e63a22add5354d9bb4b99e90117043c7124007664907259bd16d043bb031` | `qwen3-0.6b-q8_0-workloads.jsonl` | `qwen3-0.6b-q8_0-current.jsonl` |
| LFM2.5-8B-A1B Q4_K_M | `LiquidAI/LFM2.5-8B-A1B-GGUF`, revision `49c14831707011e64d70b2ebd8462ba08d608434` | `4923ec14f06b968b74d663e5949867d2d9c3bf13a20b8be1a9f9af39989b2bb0` | `lfm2.5-8b-a1b-q4_k_m-workloads.jsonl` | `lfm2.5-8b-a1b-q4_k_m-current.jsonl` |

Each baseline matrix has 30 raw rows (15 paired samples) across short, 128-, 512-, and 1,024-token prefill, plus a 129-token sustained-decode workload. Short Qwen3 and LFM2 prompts reproduce the saved Phase 6 reference IDs exactly using the official tokenizer. Long prompts are tokenized once with the matching official tokenizer and then passed as fixed IDs. The 1,601-token generation case from Phase 5.5 is not included in this first campaign.

The table below summarizes each format's initial baseline. Q4_K_M was remeasured later with a fresh three-pair baseline immediately before its candidate matrix; those same-period files are used for the experiment decision because the reference's absolute throughput shifted between sessions.

## Baseline results

Each ratio is the median of the 15 paired Ferrum/reference throughput ratios across five equally weighted workloads. Prefill columns show the paired median on that exact prompt length. Exact outputs count complete sequence matches across 15 pairs.

| Model / format | Prefill 128 | Prefill 512 | Prefill 1,024 | Decode | Full generation | Exact outputs |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen2.5 Q4_0 | 0.496 | 0.538 | 0.531 | 0.461 | 0.496 | 12 / 15 |
| Qwen2.5 Q4_K_M | 0.483 | 0.535 | 0.550 | 0.520 | 0.514 | 9 / 15 |
| Qwen2.5 Q5_K_M | 0.500 | 0.504 | 0.525 | 0.542 | 0.505 | 9 / 15 |
| Qwen2.5 Q6_K | 0.409 | 0.456 | 0.474 | 0.647 | 0.583 | 12 / 15 |
| Qwen2.5 Q8_0 | 0.386 | 0.421 | 0.442 | 0.688 | 0.596 | 9 / 15 |
| Qwen3-0.6B Q8_0 | 0.314 | 0.403 | 0.479 | 0.658 | 0.561 | 12 / 15 |
| LFM2.5-8B-A1B Q4_K_M | 0.020 | 0.014 | 0.014 | 0.196 | 0.049 | 9 / 15 |

Every decode ratio is below 0.70 except Qwen2.5 Q8_0 at 0.688; each medium/long prefill ratio is below 0.80. The gate is not met, so large Qwen model performance runs remain deferred. The 8B hybrid model's sampled maximum process RSS was 5.29 GiB for Ferrum and 4.98 GiB for llama.cpp; the model and matched workloads fit comfortably in the available memory. Qwen Q4_K_M per-case decode ratios ranged from 0.446 to 0.565. The full raw rows retain first-token timings, all generated IDs, KV bytes, and per-process memory observations.

Output differences are format/workload dependent. The five Qwen2.5 matrices produced exact IDs in 9–12 of 15 pairs; Qwen3 did so in 12 of 15. LFM2 matched all short and sustained sequences and the 1,024-token case, while its 128- and 512-token cases diverged after 9 and 8 generated IDs respectively. Divergences remain visible in raw rows and are not hidden by throughput summaries.

## Upstream Metal review

The pinned current upstream `ggml-metal` implementation is in `ggml/src/ggml-metal/kernels/mul_mv.metal`, with row/SIMD-group defaults in `ggml/src/ggml-metal/ggml-metal-impl.h`. For Q4_K it currently uses `N_R0_Q4_K=2` and `N_SG_Q4_K=2`; its M=1 kernel loads activation fragments and reuses them while it accumulates adjacent weight rows. An M5-specific tuning discussion in [llama.cpp issue 19303](https://github.com/ggml-org/llama.cpp/issues/19303) reports a larger Q4_K row tile experiment, but the current upstream default remains two rows. This is an unmerged result, not an upstream production guarantee.

Phase 5.5 already retained Q4_0 and Q8_0 multirow M=1 kernels, and rejected a Q5_0/Q5_1 two-row candidate after synchronized profiles showed regressions. The current bottleneck evidence points to quantized projections for decode and quantized matrix tiling for prefill. The Q4_K candidate below is a native Ferrum implementation with its own measured shape guard.

## Experiment 1: Q4_K two-row reuse for M=1

Status: accepted for wide Q4_K M=1 projections; the matched matrix showed a small consistent decode improvement with unchanged outputs.

The candidate assigns each 32-lane SIMD group two adjacent output rows. It loads the eight activation values for a Q4_K superblock once and reuses them across both rows. Four SIMD groups cover up to eight output rows per threadgroup. A tail guard handles arbitrary output-row counts, and production selection is limited to M=1 Q4_K projections with at least 128 output rows.

The test `q4_k_eight_row_gemv_reuses_activations_and_covers_output_tail` compares a 133-row output against a scalar reference under Metal API Validation and GPU Shader Validation; it passed. In isolated two-token diagnostic requests with synchronized dispatches, the 12 cached-decode Q4_K down-projection calls fell from 0.822 ms to 0.678 ms. This is operation attribution, not the production latency result.

The same-period three-pair baseline recheck and candidate matrices use the same prompts, generation lengths, build, and reference process options. Across all 15 paired Ferrum runs, the candidate's decode throughput was 1.019x the original kernel (range 1.011–1.035x), and full-generation throughput was 1.008x (range 1.002–1.038x). Prefill was effectively unchanged at 1.001x. Generated IDs were identical between the two Ferrum paths in all 15 pairs, with 9/15 exact matches against llama.cpp in both matrices. The Ferrum/reference median decode ratio rose from 0.480 to 0.494; generation rose from 0.506 to 0.510. Because the measured gain is small, it remains a narrow Q4_K shape rule and does not change the overall gate status. The production default is enabled for `N >= 128`; `FERRUM_Q4_K_GEMV_8ROWS=false` selects the original kernel for controlled comparisons.

## Experiment 2: Q6_K two-row reuse for M=1

Status: accepted for wide Q6_K M=1 projections; the matched matrix showed a consistent decode improvement with unchanged outputs.

The candidate uses the same two-output-row-per-SIMD-group mapping and four-group, eight-row tile for Q6_K. It reuses the Q6_K activation fragment across adjacent rows and handles output tails. Selection is limited to M=1 projections with at least 128 output rows. The focused 133-row tail test passed with Metal API Validation and GPU Shader Validation enabled. The upstream reference keeps `N_R0_Q6_K=2`; the Ferrum candidate uses two rows per SIMD group and eight rows per threadgroup, a separate M5-specific shape selected from this paired measurement.

The A/B used the same-period baseline recheck and candidate matrices, three interleaved pairs per workload. Candidate decode throughput was 1.030x the original kernel at the median across 15 pairs (range 1.018–1.043x); per-workload medians ranged from 1.025x to 1.036x. Prefill was effectively unchanged (median 0.999x), and full-generation throughput improved by roughly 1–3%. Generated IDs matched the original Ferrum path in all 15 pairs. The Ferrum/reference decode medians after the change remain below the Phase 7 gate, so this kernel improves Q6_K M=1 without clearing the model-level gate. The production default is enabled for `N >= 128`; `FERRUM_Q6_K_GEMV_8ROWS=false` selects the original kernel for controlled comparisons.

## Raw evidence

All workload definitions, paired JSONL matrices, and complete runner logs are retained in `docs/measurements/phase7a/`, including the Q6_K baseline recheck and `qwen2.5-q6_k-q6k8rows-candidate.jsonl`. The directory also includes the earlier Q4_K_M profile attempt that used the default batching limit and therefore reported no GPU samples; it is not used for operation attribution. Isolated baseline and candidate profiles are retained under `docs/measurements/phase7a/profiles/`. The first profile command also contains a logged incorrect model path; the corrected capture followed it.

## M5 GPU Neural Accelerator and Metal 4 TensorOps audit

Ferrum's `mpp::tensor_ops` kernels are inline MSL operations on the M5 GPU and use the GPU's per-core Neural Accelerators. This is distinct from the standalone Apple Neural Engine and from Core ML/Core AI model execution. Apple documents the MSL 4 TensorOps path and its M5 hardware use in [Running inline ML operations in a shader with Metal 4](https://developer.apple.com/documentation/metal/running-inline-ml-operations-in-a-shader-with-metal-4) and [WWDC26: Optimize custom machine learning operations with Metal tensors](https://developer.apple.com/videos/play/wwdc2026/330/).

### Current coverage and remaining conventional paths

| Workload | Current M5 path | Remaining conventional MSL / candidate boundary |
| --- | --- | --- |
| Dense BF16 projections | `project_mpp` for eligible batched shapes; other batch sizes use native matrix or direct kernels | The large dense prefill path is already TensorOps-backed. M=1 GEMV stays on direct shaders. |
| GGUF Q4_0, Q5_0, Q5_1, Q4_K, Q5_K, Q6_K, and Q8_0 projections | MPP GEMM is selected for BF16 activations when M>=16, K is divisible by 128, and the device supports Metal 4 / Apple GPU family 10. Existing quantized MPP kernels unpack the custom GGUF blocks into BF16 threadgroup tiles before `matmul2d`. | Batched M=2..15 and unsupported dtype/K shapes still use conventional `q*_gemm` MSL. M=1 uses format-specific direct `q*_gemv` kernels, including the measured Q4_K/Q6_K multirow cases retained above. MLX affine-Q4 is a separate F16 path and is outside this GGUF campaign. |
| Quantized MoE expert projections | BF16 prefill batches that average at least 8 assignments per expert use expert grouping plus Q4_K/Q5_K/Q6_K MPP GEMM; Q4_K and Q6_K use a 32x64x64 `matmul2d` tile, while Q5_K keeps K=128 pending a model-level measurement. One-token routing remains a separate GPU routing optimization. | Smaller batches, M=1, non-BF16 activations, and unsupported alignment continue through conventional SIMD expert kernels. MPP stages exact GGUF values into BF16; it does not reinterpret GGUF blocks as Metal's native block-scaled int4/int8 tensors. |
| Prefill attention | BF16/F16 attention score and context products use TensorOps when query length exceeds one. | `attention_mask` and `attention_softmax` remain conventional kernels between those products. A fused FlashAttention-style TensorOps kernel using cooperative results and row reductions is a possible longer-prefill experiment. Decode's one-query context stays on its direct kernel. |

The dispatch rules are visible in `src/ops/transformer.rs`; the current quantized MPP unpack/stage/multiply path is in `src/metal/shaders/project_mpp.metal`; quantized expert projection is in `src/metal/shaders/ops.metal`. Norm, RoPE, activation, and elementwise kernels are conventional MSL but are not matrix contractions that should be moved to `matmul2d` by default.

Profile attribution shows why the remaining conventional shapes need separate treatment. In Qwen2.5 Q4_K M=1 decode, the leading projection costs were `up_proj.q5_0_gemv` at 1.812 ms, `lm_head.q8_0_gemv_8rows` at 1.650 ms, and `gate_proj.q5_0_gemv` at 1.332 ms. These one-row GEMVs are not TensorOps candidates for this dispatch; direct shaders remain selected. In the LFM2.5 1,024-token flushed-dispatch diagnostic, no conventional expert projection fallback appeared: grouped MPP expert products accounted for about 1.126 s (Q4_K input), 0.316 s (Q4_K output), and 0.215 s (Q6_K output). Conventional `attention_softmax` was about 17 ms in that capture, while attention score and context products already used TensorOps. These are diagnostic GPU totals rather than end-to-end timings.

### Native quantized API constraints

The installed Xcode 27 SDK's `matmul2d` declarations include BF16/half multiplied by native signed or unsigned 4-bit and 8-bit operands, plus cooperative tensors as inputs and destinations. Metal tensor data types expose signed/unsigned int4 from macOS 26.4; macOS 27 adds int2, FP4, and FP8 types. macOS 27 multi-plane tensors support an auxiliary scale plane encoded only as FP8 UE8M0, with the first-axis block factor fixed at 32. This is a power-of-two scale representation, not a general FP16 scale. Cooperative tensor operands have scope and layout constraints, and the API provides compatibility checks before reusing them as inputs. The Q4_K input experiment used a single-SIMD-group operation and a 32x32 output tile. See the [Metal Performance Primitives programming guide](https://developer.apple.com/download/files/Metal-Performance-Primitives-Programming-Guide.pdf), the [current MTLTensor data-type API](https://developer.apple.com/documentation/metal/mtltensordatatype), and the WWDC26 session linked above.

Those native types are not a lossless, zero-copy view of Ferrum's common GGUF blocks. Q4_0 stores an arbitrary FP16 scale per 32 values and packs the low 16 logical values separately from the high 16; its nibble order must be rearranged for a native int4 operand. Q8_0 stores an arbitrary FP16 scale per 32 signed bytes. Q4_K stores a superblock FP16 scale/minimum plus per-32-value 6-bit scale/minimum fields and packed nibbles. The E8M0 plane cannot represent those scales exactly, and Q4_K also has a non-native affine offset layout. Re-encoding to E8M0 would change weights and was not used. Cooperative input tensors can preserve custom GGUF math, but still require custom unpacking and conversion.

### Rejected cooperative-input experiment

An opt-in Q4_K MPP variant decoded the exact existing BF16-rounded Q4_K values directly into an MPP cooperative right-input tensor, avoiding the existing threadgroup BF16 staging. The SDK requires cooperative inputs to use a single SIMD group, so the experiment used a 32x32 output tile and a 128-element K loop. Its correctness check used M=35, N=65, K=512 to cover both output tails; it passed with Metal API Validation and GPU Shader Validation enabled.

The model-level A/B used the pinned LFM2.5-8B-A1B Q4_K_M artifact, three pairs for each of five workloads, with the same router and GEMV settings in both runs. The candidate/control paired medians were 1.001x for prefill throughput (per-case: short 1.000x, 128 tokens 0.986x, 512 tokens 1.003x, 1,024 tokens 1.001x), 1.003x for decode, and 1.003x for full generation. All 15 Ferrum output sequences matched the control; each path matched llama.cpp in 9/15 sequences. Ferrum/reference median prefill was 0.0198x for control and 0.0197x for candidate. This is no measurable win and includes a small 128-token regression, so the experiment was removed from the runtime.

Raw results and logs are `lfm2.5-8b-a1b-q4_k_m-tensorops-baseline.jsonl`, `lfm2.5-8b-a1b-q4_k_m-tensorops-candidate.jsonl`, their `*-run.log` files, and `lfm2.5-8b-a1b-q4_k_m-tensorops-correctness.log` in `docs/measurements/phase7a/`. The M=1 GEMV kernels remain direct shaders; no TensorOps variant is enabled there.

The Q4_K cooperative-input control and candidate used the same temporary release build while the unrelated Q8_0 MPP K tile was set to 32; the production Q8_0 tile was restored to 64 before later matrices. Their paired relative result is still a valid Q4_K comparison, but the absolute Ferrum/reference prefill rates in those two files should not be read as current-production rates.

## Experiment 3: GPU MoE routing for one-token decode

Status: retained for one-token steps. The GPU route kernel preserves the CPU route selection and weights while removing the per-layer GPU-to-CPU router readback boundary. `MetalDevice::use_moe_gpu_routing` limits it to `top_k <= 16` and `token_count == 1`; prompt prefill continues to use the CPU route path.

The current-build A/B used three pairs for each of five LFM2.5-8B-A1B Q4_K_M workloads with all Q4_K/Q6_K GEMV settings held constant. Candidate/control decode medians by case were short 1.078x, 128 tokens 1.066x, 512 tokens 1.078x, 1,024 tokens 1.083x, and sustained decode 1.078x. The paired median across all 15 samples was 1.078x for decode and 1.016x for full generation. Prefill was unchanged at a 1.002x overall median; the 128-, 512-, and 1,024-token cases were 1.004x, 1.001x, and 1.002x respectively.

All 15 Ferrum sequences matched the CPU-routing path exactly. Both paths matched llama.cpp in 9/15 cases. First decode command buffers fell from 23 to 1 in every case; dispatches rose from 407 to 429 on the short/128-token cases and 419 to 441 on the longer-prefill cases because the GPU route shader adds one dispatch per sparse layer while eliminating the CPU synchronization boundary. The remaining LFM2.5 gap is substantial: Ferrum/reference median decode is 0.225x and medium/long prefill remains about 0.020x, so this improvement does not clear the campaign gate.

The paired files are `lfm2.5-8b-a1b-gpu-route-guard-baseline.jsonl` and `lfm2.5-8b-a1b-gpu-route-guard-candidate.jsonl`, with their runner logs in `docs/measurements/phase7a/`. An earlier run was interrupted during reference warmup and recorded no rows; its log is retained as `lfm2.5-8b-a1b-gpu-route-guard-interrupted-run.log` and is excluded from these results.

## Experiment 4: Native signed-int8 TensorOps for Q8_0 prefill

Status: rejected; the experimental kernel and dispatch control were removed after the paired run. The current SDK's `matmul2d` supports BF16 x signed-int8 with FP32 output. Q8_0's signed payload has an arbitrary FP16 scale for every 32 values, while Metal's native quantized tensor scale plane uses E8M0. The prototype therefore viewed each payload block as a native int8 TensorOps input, applied its stored FP16 scale to that 32-wide partial product, and accumulated the result. This preserves the stored scales without converting them to E8M0, but it needs four `matmul2d` calls for each existing K=128 tile.

The candidate passed its M=35, N=64, K=512 reference check with Metal API Validation and GPU Shader Validation enabled. A 1024-token candidate profile confirmed it ran for Qwen3 Q8_0 `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, and `down_proj`; token-by-token decode continued to use `q8_0_gemv_8rows`.

The release A/B used the pinned Qwen3-0.6B Q8_0 artifact, three interleaved control/candidate pairs for each of the short, 128-, 512-, and 1,024-token prompts plus sustained decode. Candidate/control prefill-throughput medians by prompt length were 1.304x (short, 21 tokens), 1.067x (128), 0.836x (512), and 0.770x (1,024). The 512- and 1,024-token regressions appeared in all three pairs. Median cached decode was 0.999x and median full-generation throughput was 1.012x across the 15 samples. Ferrum's complete generated IDs matched control in 12/15 pairs; the only differences were in the 129-token sustained-decode workload. These results do not justify the consistent medium/long prefill loss, so the path remains removed. The existing direct M=1 GEMV and BF16-staged K=128 TensorOps kernels remain selected.

Raw A/B, candidate profile, validation output, and runner logs are `qwen3-0.6b-q8_0-native-i8-ab.jsonl`, `qwen3-0.6b-q8_0-native-i8-profile.jsonl`, `qwen3-0.6b-q8_0-native-i8-correctness.log`, and their `*-run.log` files in `docs/measurements/phase7a/`.

After deleting the rejected prototype, the release benchmark rebuilt successfully. The retained quantized MPP kernels passed six focused correctness tests and the GPU router passed its focused ordering/weight test with both Metal API Validation and GPU Shader Validation enabled; logs are `mpp-gemm-validation.log` and `moe-router-validation.log` in the same measurement directory.

## Experiment 5: Grouped TensorOps for quantized MoE prefill

Status: retained for sufficiently large BF16 expert batches. This ports the applicable scheduling idea from the pinned llama.cpp Metal backend at revision `9710a32175b3b8f04636aaac4aa3cc28651b505d`: `kernel_mul_mm_id_map0` groups routed rows by expert, and `kernel_mul_mm_id` feeds per-expert row tiles to MPP. Ferrum implements that pattern with its own route metadata, buffers, and output order; no llama.cpp source was copied.

The GPU first counts assignments and builds 32-row-padded expert offsets, then atomically compacts each assignment's BF16 input row into its expert segment. A Q4_K, Q5_K, or Q6_K Metal 4 kernel decodes the original GGUF blocks into a BF16 threadgroup tile and runs `matmul2d` with M=32, N=64, K=128. It scatters the result back to the original assignment index, so the existing activation and combine kernels keep their input contract. The initial dispatch threshold was 24 average routes per expert, with BF16 activations, K divisible by 128, and at least 64 output rows; otherwise the existing direct kernel stayed selected. M=1 GEMV never enters this path.

The current-build operation profile used the pinned LFM2.5-8B-A1B Q4_K_M model with a 1,024-token prompt. With per-dispatch GPU timing enabled only for attribution, conventional expert projection accounted for 35.70s of the 36.50s prefill: 24.21s in Q4_K input projections, 6.61s in Q4_K output projections, and 4.88s in Q6_K output projections. Under the grouped candidate, the expert operation totals (including count and compaction) were 0.90s, 0.26s, and 0.17s respectively; smaller expert chunks continued to use the direct fallback. This isolated timing mode flushes every dispatch and is diagnostic only.

The release A/B used the same model, prompt IDs, greedy decoding, and one warmed Ferrum process for three paired runs of each of five workloads. Control/candidate order alternated by pair. Each regular case generated 17 tokens; the sustained-decode case generated 129. Candidate/control paired median prefill-throughput ratios were short 0.968x, 128 tokens 0.997x, 512 tokens 4.246x, 1,024 tokens 4.297x, and sustained decode 0.990x. The direct path remained selected for the short and 128-token prompts and for cached M=1 decode. Full-generation ratios were 3.720x at 512 tokens and 4.017x at 1,024 tokens. All 15 complete candidate sequences matched their control sequences exactly.

At the median, prefill GPU time fell from 18.19s to 4.20s for the 512-token prompt and from 34.57s to 8.30s for the 1,024-token prompt. The candidate adds two dispatches per expert projection call for segment counting and compaction: total prefill dispatches rose from 671 to 847 at 512 tokens and 935 to 1,287 at 1,024 tokens. The 1,024-token transient peak remained 240.03 MiB and arena high-water remained 255.87 MiB in both paths; cumulative allocated bytes rose from 805.25 MiB to 901.25 MiB. These runner records do not include process RSS.

The correctness tests exercised uneven expert counts, assignment-order restoration, row and output tails, and Q4_K/Q5_K/Q6_K values at M=197, N=65, K=512. Both tests passed with Metal API Validation and GPU Shader Validation enabled; output is `lfm2.5-8b-a1b-expert-tensorops-validation.log`. The full A/B and request matrix are `lfm2.5-8b-a1b-expert-tensorops-ab.jsonl` and `lfm2.5-8b-a1b-expert-tensorops-ab-workloads.jsonl`; per-operation control and candidate captures are under `docs/measurements/phase7a/profiles/` as `lfm2.5-8b-a1b-isolated-profile.jsonl` and `lfm2.5-8b-a1b-expert-tensorops-profile.jsonl`.

At the initial 24-route threshold this was a large LFM prefill improvement, but the short and 128-token cases did not improve. Experiment 6 measures lower route thresholds; the remaining LFM2.5/reference ratio and general small-model gate are still below target.

## Experiment 6: Lower grouped-expert TensorOps route threshold

Status: retained with a default of 8 average assignments per expert. The threshold is exposed through the device setter, the benchmark request field `moe_expert_tensorops_min_routes_per_expert`, and `FERRUM_MOE_EXPERT_TENSOROPS_MIN_ROUTES_PER_EXPERT`. Dispatch remains based on input dtype, dimensions, MPP support, and average route density; it does not depend on model identity. The M=1 GEMV path is unchanged.

Three interleaved sweeps (24 vs 12, 12 vs 8, and a direct 24 vs 8 confirmation) used the pinned LFM2.5-8B-A1B Q4_K_M artifact, the same five prompts and greedy token counts, and one warmed Ferrum process. Candidate/control order alternated within each of three pairs per workload. The direct 24-vs-8 paired medians were:

| Workload | Prefill throughput 8/24 | Cached decode 8/24 | Full generation 8/24 | Candidate IDs matching saved llama.cpp |
| --- | ---: | ---: | ---: | ---: |
| Short | 1.002x | 0.999x | 0.999x | 3 / 3 |
| 128-token prompt | 12.125x | 1.025x | 4.664x | 0 / 3 |
| 512-token prompt | 3.536x | 1.009x | 2.545x | 3 / 3 |
| 1,024-token prompt | 3.717x | 1.000x | 3.013x | 3 / 3 |
| Sustained decode | 0.998x | 1.000x | 1.001x | 3 / 3 |

At 128 tokens, threshold 8 changes the generated sequence relative to threshold 24, but both first diverge from the saved llama.cpp sequence at generated token 9. At 512 tokens, threshold 8 matches llama.cpp in all three runs; threshold 24 diverges after token 8. Short, 1,024-token, and sustained-decode sequences match both control and reference. The 128-token and 512-token changes arise from selecting BF16 TensorOps for smaller expert batches; no numerical tolerance was changed.

Median prefill time fell from 4.515s to 0.372s at 128 tokens, 4.253s to 1.203s at 512, and 8.498s to 2.291s at 1,024. The candidate adds 88 dispatches on each of those prompts because the segment-count and compaction passes now run for additional expert projections. Transient peak, arena high-water, and cumulative allocated bytes were unchanged in each case. At 1,024 tokens the transient peak was 240.03 MiB, arena high-water 255.87 MiB, and cumulative allocated bytes 901.25 MiB for both variants. Post-request RSS samples were also unchanged at approximately 5.30 GiB; these are samples, not peak-RSS measurements.

The expanded correctness tests cover uneven 8- and 12-route groups with output tails for Q4_K/Q5_K/Q6_K at M=25 or 37, E=3, N=65, K=512. All six TensorOps expert tests passed with Metal API Validation and GPU Shader Validation enabled, within their existing elementwise tolerances. The validation log is `lfm2.5-8b-a1b-expert-tensorops-threshold-8-validation.log`. Raw paired results and runner logs for all three threshold sweeps are `lfm2.5-8b-a1b-expert-tensorops-threshold-{24-vs-12,12-vs-8,24-vs-8}.{jsonl,run.log}` in `docs/measurements/phase7a/`.

This reduces a major LFM prefill bottleneck without changing cached decode or transient-memory use. The model remains far below the campaign's llama.cpp prefill and decode gate; no large Qwen target is unlocked.

A fresh matched Ferrum-versus-llama.cpp matrix used the default threshold of 8 and the pinned llama.cpp revision above, with three interleaved pairs and the same GGUF and token IDs. Ferrum/reference median prefill throughput was 0.247x at 128 tokens, 0.211x at 512, and 0.225x at 1,024. Median sustained decode was 0.224x; the five-workload median decode ratio was 0.225x and full-generation ratio 0.219x. Exact generated IDs matched llama.cpp in 12/15 runs: 3/3 short, 0/3 at 128, 3/3 at 512, 3/3 at 1,024, and 3/3 sustained decode. This improves LFM materially but leaves the general gate closed.

The current matched file and runner output are `lfm2.5-8b-a1b-threshold8-current.jsonl` and `lfm2.5-8b-a1b-threshold8-current-run.log`. Post-request RSS samples were 5.24–5.31 GiB for Ferrum and 4.99 GiB for llama.cpp; the runner did not capture peak RSS. A separate flushed-dispatch diagnostic profile at 1,024 tokens attributed about 1.66s of sampled GPU time to the remaining expert MPP projections: 1.13s Q4_K input, 0.316s Q4_K output, and 0.215s Q6_K output. No conventional expert projection fallback appeared in that capture. The profile is `lfm2.5-8b-a1b-threshold8-profile.jsonl` under `docs/measurements/phase7a/profiles/`; it identifies expert MPP weight staging and matmul as the next bottleneck to investigate, not production timing.

## Experiment 7: M16 versus M32 grouped-expert TensorOps tiles

Status: rejected; M32 remains selected. The temporary M16 implementation changed only the expert `matmul2d` M tile and grid height for grouped Q4_K/Q5_K/Q6_K prefill. It kept the 64-column by 128-K tile, packed GGUF weights, quantization, output order, dispatch threshold, and M=1 GEMV path constant.

The current release build ran three paired M32/M16 comparisons for each of the same five LFM2.5-8B-A1B Q4_K_M workloads. Pair order alternated in one warmed Ferrum process. All 15 generated sequences matched exactly between tile sizes. Candidate/control paired median ratios were:

| Workload | Prefill throughput M16/M32 | Cached decode M16/M32 | Full generation M16/M32 | Exact IDs |
| --- | ---: | ---: | ---: | ---: |
| Short | 0.999x | 0.998x | 0.997x | 3 / 3 |
| 128-token prompt | 0.780x | 1.001x | 0.915x | 3 / 3 |
| 512-token prompt | 0.702x | 0.993x | 0.796x | 3 / 3 |
| 1,024-token prompt | 0.689x | 1.000x | 0.749x | 3 / 3 |
| Sustained decode | 0.999x | 1.000x | 1.000x | 3 / 3 |

Across all 15 pairs, median prefill, cached-decode, and complete-generation ratios were 0.780x, 1.000x, and 0.915x. The M16 candidate regressed on every medium/long prefill; its largest loss was 31.1% at 1,024 tokens. It left cached decode effectively unchanged. Dispatch counts were equal between variants, and transient peaks were identical per prompt length: 4.05, 34.04, 160.05, and 240.03 MiB. Sampled post-request RSS also stayed within 5.23–5.31 GiB for both variants. These results do not support an adaptive small-route exception, so the M16 code and benchmark control were removed.

The M16 Q4_K/Q5_K/Q6_K correctness variants covered uneven route groups and output tails. All six expert TensorOps tests passed with Metal API Validation and GPU Shader Validation enabled; the output is `lfm2.5-8b-a1b-expert-tensorops-m16-validation.log`. Raw paired samples and the complete runner output are `lfm2.5-8b-a1b-expert-tensorops-m32-vs-m16.jsonl` and `lfm2.5-8b-a1b-expert-tensorops-m32-vs-m16.run.log` in `docs/measurements/phase7a/`.

## Experiment 8: M64 versus M32 grouped-expert TensorOps tiles

Status: rejected; M32 remains selected. This tested the complementary hypothesis that a 64-row M tile would reuse each dequantized Q4_K/Q5_K/Q6_K weight tile across more routed rows. Expert segments were aligned to the selected M tile, so padding and transient allocation costs were included. For the 32-expert LFM2.5 model, average route density is about 16, 64, and 128 routes per expert at the 128-, 512-, and 1,024-token prompts respectively.

Three paired M32/M64 runs were collected for each of the same five workloads in one warmed Ferrum process, with alternating order. M64 improved median prefill throughput by 7.8%, 6.2%, and 11.9% at 128, 512, and 1,024 tokens. However, its generated sequences matched M32 in only 6/15 pairs: 3/3 short, 0/3 at 128, 0/3 at 512, 0/3 at 1,024, and 3/3 sustained decode. Against the saved llama.cpp rows from the same prompt matrix, M32 matched 12/15 while M64 matched 6/15. M64 diverged from llama.cpp by token 5 at all 512-token runs, and by tokens 1, 3, or 5 at 1,024 tokens; the M32 control matched all six of those medium/long reference sequences. The throughput improvement does not justify this output regression, so no high-route-density dispatch exception was retained.

Transient peak memory rose from 34.04 to 49.82 MiB at 128 tokens, with no change at 512 or 1,024 tokens. Cumulative allocated bytes at 1,024 tokens increased from 885.25 to 901.25 MiB; sampled process RSS remained about 5.30 GiB. The M64 correctness path passed the same six focused tests under Metal API Validation and GPU Shader Validation. Its validation output is `lfm2.5-8b-a1b-expert-tensorops-m64-validation.log`; raw paired samples and runner output are `lfm2.5-8b-a1b-expert-tensorops-m32-vs-m64.jsonl` and `lfm2.5-8b-a1b-expert-tensorops-m32-vs-m64.run.log` in `docs/measurements/phase7a/`.

After removing M64, the six production M32 expert TensorOps tests also passed with both validation layers enabled; the current-path output is `lfm2.5-8b-a1b-expert-tensorops-m32-current-validation.log`.

## Experiment 9: Native signed-int4 TensorOps for Q4_0 prefill

Status: rejected; the temporary kernel, dispatch control, and benchmark option were removed. The experiment tested whether Metal's native signed-int4 `matmul2d` operand could avoid the existing Q4_0 MPP path's BF16 threadgroup staging. It reordered Q4_0's split nibble halves into a packed logical int4 tile, retained each block's FP16 scale in the surrounding shader, and used M=32, N=64, K=32 TensorOps calls (four calls per existing K=128 tile). The current M=1 direct GEMV dispatch was not changed.

The native int4 path passed a Q4_0 reference check at M=35, N=65, K=512 with Metal API Validation and GPU Shader Validation enabled. The release A/B used the pinned Qwen2.5-0.5B Q4_0 artifact (SHA-256 `7671c0c304e6ce5a7fc577bcb12aba01e2c155cc2efd29b2213c95b18edaf6ed`), three interleaved control/candidate pairs for each of the short, 128-, 512-, and 1,024-token prompts plus sustained decode. Candidate/control paired median throughput ratios were:

| Workload | Prefill | Cached decode | Full generation | Exact generated IDs |
| --- | ---: | ---: | ---: | ---: |
| Short | 0.825x | 1.001x | 0.968x | 3 / 3 |
| 128-token prompt | 0.658x | 0.993x | 0.901x | 3 / 3 |
| 512-token prompt | 0.580x | 0.992x | 0.768x | 3 / 3 |
| 1,024-token prompt | 0.601x | 1.016x | 0.721x | 3 / 3 |
| Sustained decode | 0.826x | 1.005x | 0.998x | 0 / 3 |

All matched pairs had identical transient prefill peaks. On sustained decode, control and candidate first differed at generated token 64 in each pair. Cached decode was flat, and the 128–1,024-token prefill cases consistently regressed, so the native int4 path was not retained. This supports keeping one-token projections on direct GEMV shaders and the current BF16-staged TensorOps path for eligible batched shapes.

Raw A/B rows, the complete runner output, and correctness output are `qwen2.5-q4_0-native-int4-ab.jsonl`, `qwen2.5-q4_0-native-int4-ab.run.log`, and `qwen2.5-q4_0-native-int4-validation.log` in `docs/measurements/phase7a/`.

## Experiment 10: Four-row-per-SIMD Q5_0 M=1 GEMV

Status: retained as the default for Q5_0 M=1 projections with at least 128 output rows. The direct MSL kernel follows llama.cpp's Q5_0 row organization: four adjacent output rows per SIMD group, two SIMD groups per 64-thread threadgroup. Each SIMD group loads the activation fragment once and reuses it across those four rows. Dispatch remains format-, batch-, and shape-based; this is not a TensorOps path and does not use model identity.

The A/B used the pinned Qwen2.5-0.5B Q4_K_M artifact and the same llama.cpp revision, prompt IDs, greedy policy, and three interleaved control/candidate pairs for each of the five workloads. The candidate improved paired-median cached-decode throughput by 9.8–12.4% across the five cases. Absolute Ferrum control/candidate decode medians were 111.5→122.8 tok/s (short), 112.1→123.3 (128-token prompt), 107.7→119.3 (512), 98.8→110.7 (1,024), and 106.7→118.7 (sustained decode). Complete-generation throughput rose from 96.2→103.4, 92.4→100.7, 63.8→68.4, 41.7→44.2, and 102.6→113.9 tok/s respectively. Prefill stayed near the control rates; first-token latency changed from 30.8→31.0, 39.7→38.8, 113.0→112.2, 238.7→236.3, and 31.7→31.4 ms.

The candidate and control produced identical IDs for all three runs of the short, 128-, 512-, and 1,024-token workloads. Sustained decode diverged from control at generated token 39 in all three pairs; it matched llama.cpp through token 14 and first diverged at token 15. This output divergence is retained in the raw records; the numerical tolerance was not changed. The Q5_0 scalar-reference test passed for an output-tail shape (N=131, K=512) and a production-sized wide projection (N=4,864, K=896), with Metal API Validation and GPU Shader Validation enabled. The same test verifies the N>=128 dispatch boundary. Its log is `qwen2.5-q5_0-n4-gemv-validation.log`.

The release A/B records are `qwen2.5-q4_k_m-q5n4-ab.jsonl` and `qwen2.5-q4_k_m-q5n4-ab.run.log`. The full absolute Ferrum/control, candidate, and llama.cpp rates, their ratios, and first-token latencies are in the JSONL-derived matrix below. The sustained decode gain leaves Qwen2.5 Q4_K_M at 0.574x of llama.cpp decode throughput; this kernel does not change the Phase 7 gate status.

## Absolute performance record (JSONL-derived)

This section applies the reporting format requested for Phase 7: absolute Ferrum and llama.cpp throughput is shown beside each ratio, and first-token latency is shown in milliseconds. Values are computed from the existing JSONL files; no benchmark was rerun just to change presentation. Throughput and latency values are medians across the three runs for each condition and workload. Baseline ratios are medians of matched per-pair ratios. Internal candidate/control ratios use the recorded pair ordering. Candidate/llama.cpp ratios use matched pairs when the llama.cpp rows share the candidate A/B matrix; otherwise they are ratios of the displayed condition medians from the named reference matrix. `Prefill`, `cached decode`, and `complete generation` are kept distinct.

### Matched baseline and current production matrices

| Model / workload | Prefill tok/s: Ferrum / llama.cpp (ratio) | Cached decode tok/s: Ferrum / llama.cpp (ratio) | Complete-generation tok/s: Ferrum / llama.cpp (ratio) | First-token latency ms: Ferrum / llama.cpp |
|---|---:|---:|---:|---:|
| Qwen2.5 Q4_0 / Short | 758.6 / 1450.8 (0.523x) | 114.7 / 263.6 (0.447x) | 99.9 / 199.5 (0.496x) | 28.0 / 14.7 |
| Qwen2.5 Q4_0 / 128 prompt | 3385.4 / 6767.5 (0.496x) | 115.1 / 215.0 (0.548x) | 96.3 / 182.9 (0.526x) | 38.1 / 19.1 |
| Qwen2.5 Q4_0 / 512 prompt | 4882.8 / 9068.0 (0.538x) | 111.5 / 237.5 (0.470x) | 67.3 / 133.2 (0.507x) | 105.2 / 56.7 |
| Qwen2.5 Q4_0 / 1,024 prompt | 4578.6 / 8612.9 (0.531x) | 103.3 / 233.1 (0.447x) | 44.1 / 88.7 (0.499x) | 224.0 / 119.1 |
| Qwen2.5 Q4_0 / Sustained decode | 656.8 / 1347.3 (0.487x) | 113.8 / 272.2 (0.420x) | 109.8 / 224.9 (0.488x) | 32.3 / 15.8 |
| Qwen2.5 Q4_K_M initial / Short | 693.9 / 1267.0 (0.548x) | 108.6 / 194.3 (0.559x) | 93.8 / 179.1 (0.524x) | 30.6 / 16.8 |
| Qwen2.5 Q4_K_M initial / 128 prompt | 3235.1 / 6399.1 (0.483x) | 109.3 / 193.6 (0.564x) | 89.5 / 171.9 (0.526x) | 39.9 / 20.2 |
| Qwen2.5 Q4_K_M initial / 512 prompt | 4554.2 / 8535.8 (0.535x) | 103.7 / 192.9 (0.536x) | 62.3 / 122.9 (0.509x) | 112.8 / 60.2 |
| Qwen2.5 Q4_K_M initial / 1,024 prompt | 4447.2 / 8038.8 (0.550x) | 97.7 / 208.1 (0.469x) | 42.5 / 81.8 (0.517x) | 230.6 / 127.6 |
| Qwen2.5 Q4_K_M initial / Sustained decode | 633.4 / 1257.4 (0.512x) | 105.7 / 235.3 (0.448x) | 101.3 / 200.4 (0.506x) | 33.5 / 17.0 |
| Qwen2.5 Q5_K_M / Short | 662.6 / 1255.2 (0.525x) | 103.5 / 188.2 (0.551x) | 89.4 / 173.4 (0.517x) | 32.0 / 17.0 |
| Qwen2.5 Q5_K_M / 128 prompt | 3106.6 / 6289.9 (0.500x) | 104.8 / 188.8 (0.555x) | 84.7 / 167.3 (0.519x) | 41.5 / 20.6 |
| Qwen2.5 Q5_K_M / 512 prompt | 4416.9 / 8575.7 (0.504x) | 101.6 / 186.9 (0.542x) | 60.9 / 119.9 (0.500x) | 116.2 / 59.9 |
| Qwen2.5 Q5_K_M / 1,024 prompt | 4278.0 / 8144.6 (0.525x) | 95.4 / 183.5 (0.519x) | 41.1 / 80.5 (0.504x) | 239.7 / 126.0 |
| Qwen2.5 Q5_K_M / Sustained decode | 594.5 / 1266.2 (0.459x) | 102.5 / 199.4 (0.514x) | 98.1 / 194.3 (0.505x) | 35.6 / 16.9 |
| Qwen2.5 Q6_K initial / Short | 585.0 / 1381.2 (0.424x) | 115.5 / 172.8 (0.669x) | 96.3 / 162.2 (0.597x) | 36.2 / 15.5 |
| Qwen2.5 Q6_K initial / 128 prompt | 2739.7 / 6532.1 (0.409x) | 115.8 / 173.5 (0.666x) | 90.7 / 155.6 (0.583x) | 47.0 / 19.8 |
| Qwen2.5 Q6_K initial / 512 prompt | 4027.6 / 8992.5 (0.456x) | 110.8 / 178.1 (0.626x) | 61.3 / 115.9 (0.536x) | 127.5 / 57.2 |
| Qwen2.5 Q6_K initial / 1,024 prompt | 3961.4 / 8320.6 (0.474x) | 103.7 / 170.7 (0.607x) | 40.8 / 79.4 (0.513x) | 258.8 / 123.3 |
| Qwen2.5 Q6_K initial / Sustained decode | 548.0 / 1298.5 (0.423x) | 112.7 / 173.5 (0.648x) | 108.1 / 176.0 (0.616x) | 38.6 / 16.4 |
| Qwen2.5 Q8_0 / Short | 520.2 / 1418.6 (0.367x) | 118.3 / 171.3 (0.691x) | 95.6 / 159.7 (0.598x) | 40.7 / 15.1 |
| Qwen2.5 Q8_0 / 128 prompt | 2522.5 / 6605.4 (0.386x) | 119.1 / 169.4 (0.707x) | 91.1 / 151.6 (0.601x) | 51.1 / 19.7 |
| Qwen2.5 Q8_0 / 512 prompt | 3756.5 / 8914.4 (0.421x) | 114.9 / 167.9 (0.685x) | 60.8 / 111.9 (0.542x) | 136.6 / 57.7 |
| Qwen2.5 Q8_0 / 1,024 prompt | 3735.6 / 8453.4 (0.442x) | 107.2 / 166.9 (0.642x) | 39.6 / 79.2 (0.500x) | 274.5 / 121.4 |
| Qwen2.5 Q8_0 / Sustained decode | 480.7 / 1360.9 (0.351x) | 115.4 / 168.7 (0.684x) | 110.7 / 170.2 (0.651x) | 44.0 / 15.7 |
| Qwen3-0.6B Q8_0 / Short | 407.9 / 1152.5 (0.355x) | 97.9 / 145.3 (0.674x) | 77.6 / 133.0 (0.581x) | 51.8 / 18.5 |
| Qwen3-0.6B Q8_0 / 128 prompt | 1795.6 / 5717.6 (0.314x) | 97.9 / 144.4 (0.677x) | 72.0 / 126.8 (0.561x) | 71.6 / 22.6 |
| Qwen3-0.6B Q8_0 / 512 prompt | 2618.5 / 6509.4 (0.403x) | 90.5 / 137.8 (0.658x) | 43.9 / 87.3 (0.504x) | 195.9 / 78.9 |
| Qwen3-0.6B Q8_0 / 1,024 prompt | 2748.1 / 5757.1 (0.479x) | 80.3 / 128.3 (0.626x) | 28.6 / 56.0 (0.510x) | 373.0 / 178.1 |
| Qwen3-0.6B Q8_0 / Sustained decode | 368.8 / 1127.0 (0.328x) | 95.4 / 146.1 (0.653x) | 90.3 / 145.4 (0.620x) | 57.2 / 18.9 |
| LFM2.5-8B-A1B Q4_K_M initial / Short | 22.4 / 217.3 (0.103x) | 18.6 / 93.3 (0.200x) | 12.5 / 74.5 (0.168x) | 491.8 / 50.8 |
| LFM2.5-8B-A1B Q4_K_M initial / 128 prompt | 28.2 / 1414.5 (0.020x) | 18.4 / 94.0 (0.195x) | 3.1 / 63.7 (0.049x) | 4542.3 / 90.7 |
| LFM2.5-8B-A1B Q4_K_M initial / 512 prompt | 28.4 / 1970.5 (0.014x) | 18.4 / 93.9 (0.196x) | 0.9 / 39.0 (0.023x) | 18033.8 / 260.1 |
| LFM2.5-8B-A1B Q4_K_M initial / 1,024 prompt | 29.6 / 2168.7 (0.014x) | 19.6 / 100.2 (0.196x) | 0.5 / 26.8 (0.018x) | 34551.9 / 472.4 |
| LFM2.5-8B-A1B Q4_K_M initial / Sustained decode | 24.1 / 233.2 (0.103x) | 20.0 / 101.9 (0.196x) | 18.7 / 96.5 (0.194x) | 457.0 / 47.4 |
| LFM2.5-8B-A1B Q4_K_M threshold-8 current / Short | 22.5 / 216.6 (0.104x) | 21.1 / 94.7 (0.223x) | 13.5 / 74.8 (0.181x) | 490.1 / 51.0 |
| LFM2.5-8B-A1B Q4_K_M threshold-8 current / 128 prompt | 329.8 / 1334.2 (0.247x) | 20.2 / 88.3 (0.229x) | 14.4 / 60.6 (0.238x) | 388.4 / 96.2 |
| LFM2.5-8B-A1B Q4_K_M threshold-8 current / 512 prompt | 414.7 / 1973.6 (0.211x) | 20.8 / 94.7 (0.219x) | 8.5 / 39.2 (0.216x) | 1234.9 / 259.7 |
| LFM2.5-8B-A1B Q4_K_M threshold-8 current / 1,024 prompt | 442.4 / 1986.7 (0.225x) | 20.7 / 93.6 (0.221x) | 5.5 / 24.6 (0.224x) | 2314.8 / 515.7 |
| LFM2.5-8B-A1B Q4_K_M threshold-8 current / Sustained decode | 22.1 / 216.5 (0.103x) | 21.0 / 94.4 (0.222x) | 19.5 / 89.0 (0.219x) | 497.1 / 51.0 |

### Internal candidate/control matrices with reference context

Each throughput cell reports Ferrum control → candidate and the llama.cpp value, followed by candidate/control and candidate/llama.cpp ratios. For experiments whose A/B JSONL did not contain llama.cpp rows, the matching reference matrix is named in the subsection and uses the same artifact and workload definitions. This keeps the internal kernel decision and the engine comparison visible together.


#### Qwen2.5 Q4_K_M: Q4_K M=1 row-reuse GEMV

Sources: control `qwen2.5-q4_k_m-baseline-recheck.jsonl`, candidate `qwen2.5-q4_k_m-q4k8rows-current.jsonl`, llama.cpp reference `qwen2.5-q4_k_m-q4k8rows-current.jsonl`.

| Workload | Prefill tok/s: control → candidate / llama.cpp (candidate/control; candidate/llama) | Cached decode tok/s: control → candidate / llama.cpp (candidate/control; candidate/llama) | Complete-generation tok/s: control → candidate / llama.cpp (candidate/control; candidate/llama) | First-token latency ms: control → candidate / llama.cpp |
|---|---:|---:|---:|---:|
| Short | 749.8 → 748.6 / 1441.8 (0.999x; 0.518x) | 117.9 → 120.0 / 228.7 (1.017x; 0.520x) | 101.3 → 102.2 / 194.4 (1.008x; 0.526x) | 28.3 → 28.4 / 14.8 |
| 128 prompt | 3476.2 → 3501.1 / 6943.7 (1.003x; 0.510x) | 119.1 → 121.2 / 236.0 (1.018x; 0.518x) | 96.9 → 98.5 / 182.7 (1.015x; 0.539x) | 37.3 → 37.0 / 18.8 |
| 512 prompt | 4951.6 → 4948.3 / 9377.7 (0.995x; 0.528x) | 113.5 → 115.5 / 233.8 (1.019x; 0.494x) | 67.8 → 68.6 / 134.3 (1.008x; 0.510x) | 103.7 → 103.8 / 54.8 |
| 1,024 prompt | 4712.6 → 4700.0 / 8790.6 (1.001x; 0.535x) | 104.5 → 105.8 / 226.7 (1.015x; 0.467x) | 45.2 → 45.4 / 89.4 (1.005x; 0.508x) | 217.6 → 218.2 / 116.8 |
| Sustained decode | 649.2 → 661.6 / 1383.0 (1.033x; 0.478x) | 113.7 → 116.7 / 243.2 (1.023x; 0.479x) | 107.6 → 111.1 / 220.8 (1.030x; 0.503x) | 32.6 → 32.0 / 15.4 |

#### Qwen2.5 Q6_K: Q6_K M=1 row-reuse GEMV

Sources: control `qwen2.5-q6_k-baseline-recheck.jsonl`, candidate `qwen2.5-q6_k-q6k8rows-candidate.jsonl`, llama.cpp reference `qwen2.5-q6_k-q6k8rows-candidate.jsonl`.

| Workload | Prefill tok/s: control → candidate / llama.cpp (candidate/control; candidate/llama) | Cached decode tok/s: control → candidate / llama.cpp (candidate/control; candidate/llama) | Complete-generation tok/s: control → candidate / llama.cpp (candidate/control; candidate/llama) | First-token latency ms: control → candidate / llama.cpp |
|---|---:|---:|---:|---:|
| Short | 635.0 → 636.9 / 1450.9 (0.998x; 0.441x) | 126.0 → 129.6 / 200.3 (1.025x; 0.645x) | 103.7 → 106.1 / 174.6 (1.023x; 0.602x) | 33.4 → 33.3 / 14.7 |
| 128 prompt | 3013.8 → 3007.1 / 7071.3 (0.998x; 0.425x) | 124.9 → 129.2 / 201.2 (1.031x; 0.642x) | 97.1 → 99.1 / 166.2 (1.018x; 0.596x) | 42.8 → 42.9 / 18.4 |
| 512 prompt | 4428.8 → 4395.0 / 9583.9 (1.000x; 0.459x) | 121.0 → 124.7 / 199.0 (1.028x; 0.628x) | 67.2 → 68.0 / 123.8 (1.015x; 0.551x) | 115.9 → 116.8 / 53.7 |
| 1,024 prompt | 4291.1 → 4293.1 / 9092.6 (1.000x; 0.472x) | 111.4 → 115.5 / 199.1 (1.036x; 0.582x) | 44.0 → 44.4 / 86.7 (1.011x; 0.511x) | 239.0 → 238.9 / 112.9 |
| Sustained decode | 574.7 → 568.2 / 1406.6 (0.989x; 0.404x) | 121.5 → 125.2 / 207.9 (1.030x; 0.602x) | 115.9 → 118.9 / 193.7 (1.027x; 0.615x) | 36.8 → 37.3 / 15.2 |

#### LFM2.5: GPU-resident one-token route selection

Sources: control `lfm2.5-8b-a1b-gpu-route-guard-baseline.jsonl`, candidate `lfm2.5-8b-a1b-gpu-route-guard-candidate.jsonl`, llama.cpp reference `lfm2.5-8b-a1b-gpu-route-guard-candidate.jsonl`.

| Workload | Prefill tok/s: control → candidate / llama.cpp (candidate/control; candidate/llama) | Cached decode tok/s: control → candidate / llama.cpp (candidate/control; candidate/llama) | Complete-generation tok/s: control → candidate / llama.cpp (candidate/control; candidate/llama) | First-token latency ms: control → candidate / llama.cpp |
|---|---:|---:|---:|---:|
| Short | 22.4 → 22.4 / 212.0 (0.994x; 0.106x) | 19.5 → 21.0 / 93.7 (1.078x; 0.224x) | 12.9 → 13.6 / 74.8 (1.051x; 0.182x) | 490.3 → 492.1 / 52.1 |
| 128 prompt | 27.8 → 28.0 / 1385.6 (1.004x; 0.020x) | 19.3 → 20.3 / 93.9 (1.066x; 0.222x) | 3.1 → 3.2 / 63.6 (1.016x; 0.050x) | 4597.8 → 4579.1 / 92.6 |
| 512 prompt | 28.0 → 28.0 / 1972.8 (1.001x; 0.014x) | 19.4 → 20.9 / 93.5 (1.078x; 0.223x) | 0.9 → 0.9 / 38.9 (1.005x; 0.023x) | 18271.5 → 18260.2 / 259.8 |
| 1,024 prompt | 27.9 → 28.0 / 1971.4 (1.002x; 0.014x) | 18.8 → 20.6 / 88.6 (1.083x; 0.225x) | 0.5 → 0.5 / 24.1 (1.004x; 0.019x) | 36667.8 → 36599.2 / 519.7 |
| Sustained decode | 21.3 → 21.3 / 214.8 (1.012x; 0.100x) | 19.4 → 20.9 / 91.1 (1.078x; 0.230x) | 17.9 → 19.3 / 85.8 (1.077x; 0.226x) | 517.1 → 516.5 / 51.4 |

#### Qwen3 Q8_0: native signed-int8 TensorOps

Sources: control `qwen3-0.6b-q8_0-native-i8-ab.jsonl`, candidate `qwen3-0.6b-q8_0-native-i8-ab.jsonl`, llama.cpp reference `qwen3-0.6b-q8_0-current.jsonl`.

| Workload | Prefill tok/s: control → candidate / llama.cpp (candidate/control; candidate/llama) | Cached decode tok/s: control → candidate / llama.cpp (candidate/control; candidate/llama) | Complete-generation tok/s: control → candidate / llama.cpp (candidate/control; candidate/llama) | First-token latency ms: control → candidate / llama.cpp |
|---|---:|---:|---:|---:|
| Short | 400.1 → 517.9 / 1152.5 (1.304x; 0.449x) | 99.1 → 98.1 / 145.3 (0.999x; 0.675x) | 77.9 → 82.2 / 133.0 (1.057x; 0.618x) | 52.8 → 40.8 / 18.5 |
| 128 prompt | 1811.4 → 1950.2 / 5717.6 (1.067x; 0.341x) | 98.1 → 101.1 / 144.4 (1.024x; 0.700x) | 71.7 → 74.7 / 126.8 (1.039x; 0.590x) | 71.0 → 65.9 / 22.6 |
| 512 prompt | 2615.7 → 2140.0 / 6509.4 (0.836x; 0.329x) | 91.0 → 90.6 / 137.8 (0.996x; 0.657x) | 44.1 → 38.9 / 87.3 (0.901x; 0.446x) | 196.0 → 239.6 / 78.9 |
| 1,024 prompt | 2715.8 → 2077.0 / 5757.1 (0.770x; 0.361x) | 81.1 → 80.5 / 128.3 (0.992x; 0.627x) | 28.6 → 23.9 / 56.0 (0.838x; 0.427x) | 377.4 → 493.4 / 178.1 |
| Sustained decode | 378.7 → 457.9 / 1127.0 (1.171x; 0.406x) | 95.7 → 95.4 / 146.1 (0.998x; 0.653x) | 90.3 → 91.0 / 145.4 (1.015x; 0.626x) | 55.8 → 46.2 / 18.9 |


#### LFM2.5: cooperative-input Q4_K TensorOps

Sources: control `lfm2.5-8b-a1b-q4_k_m-tensorops-baseline.jsonl`, candidate `lfm2.5-8b-a1b-q4_k_m-tensorops-candidate.jsonl`, llama.cpp reference `lfm2.5-8b-a1b-q4_k_m-tensorops-candidate.jsonl`.

| Workload | Prefill tok/s: control → candidate / llama.cpp (candidate/control; candidate/llama) | Cached decode tok/s: control → candidate / llama.cpp (candidate/control; candidate/llama) | Complete-generation tok/s: control → candidate / llama.cpp (candidate/control; candidate/llama) | First-token latency ms: control → candidate / llama.cpp |
|---|---:|---:|---:|---:|
| Short | 24.3 → 24.3 / 236.4 (1.000x; 0.103x) | 22.9 → 22.9 / 100.3 (0.997x; 0.228x) | 14.7 → 14.7 / 80.3 (0.999x; 0.183x) | 453.0 → 453.7 / 46.7 |
| 128 prompt | 30.6 → 30.2 / 1531.4 (0.986x; 0.020x) | 22.7 → 22.7 / 102.0 (1.003x; 0.223x) | 3.5 → 3.4 / 69.4 (0.988x; 0.049x) | 4187.3 → 4239.9 / 83.8 |
| 512 prompt | 30.2 → 30.3 / 2138.4 (1.003x; 0.014x) | 22.6 → 22.7 / 101.6 (1.001x; 0.224x) | 1.0 → 1.0 / 42.4 (1.003x; 0.023x) | 16938.5 → 16884.1 / 239.7 |
| 1,024 prompt | 30.2 → 30.3 / 2150.9 (1.001x; 0.014x) | 22.3 → 22.4 / 101.0 (1.002x; 0.222x) | 0.5 → 0.5 / 26.6 (1.004x; 0.019x) | 33881.0 → 33809.3 / 476.2 |
| Sustained decode | 22.8 → 24.2 / 233.1 (1.057x; 0.104x) | 22.4 → 22.7 / 101.2 (1.016x; 0.225x) | 20.4 → 21.1 / 96.2 (1.029x; 0.219x) | 482.9 → 455.4 / 47.4 |

#### LFM2.5: grouped expert TensorOps at threshold 24

Sources: control `lfm2.5-8b-a1b-expert-tensorops-ab.jsonl`, candidate `lfm2.5-8b-a1b-expert-tensorops-ab.jsonl`, llama.cpp reference `lfm2.5-8b-a1b-q4_k_m-current.jsonl`.

| Workload | Prefill tok/s: control → candidate / llama.cpp (candidate/control; candidate/llama) | Cached decode tok/s: control → candidate / llama.cpp (candidate/control; candidate/llama) | Complete-generation tok/s: control → candidate / llama.cpp (candidate/control; candidate/llama) | First-token latency ms: control → candidate / llama.cpp |
|---|---:|---:|---:|---:|
| Short | 22.4 → 21.7 / 217.3 (0.968x; 0.100x) | 21.1 → 21.0 / 93.3 (0.995x; 0.225x) | 13.6 → 13.4 / 74.5 (0.986x; 0.179x) | 491.9 → 508.1 / 50.8 |
| 128 prompt | 28.0 → 27.9 / 1414.5 (0.997x; 0.020x) | 20.9 → 20.1 / 94.0 (0.962x; 0.213x) | 3.2 → 3.1 / 63.7 (0.992x; 0.049x) | 4579.0 → 4594.1 / 90.7 |
| 512 prompt | 28.0 → 118.7 / 1970.5 (4.246x; 0.060x) | 20.8 → 20.7 / 93.9 (0.999x; 0.220x) | 0.9 → 3.3 / 39.0 (3.720x; 0.085x) | 18316.2 → 4314.0 / 260.1 |
| 1,024 prompt | 29.4 → 119.4 / 2168.7 (4.297x; 0.055x) | 20.5 → 20.7 / 100.2 (1.006x; 0.206x) | 0.5 → 1.8 / 26.8 (4.017x; 0.068x) | 34858.5 → 8579.6 / 472.4 |
| Sustained decode | 23.7 → 23.4 / 233.2 (0.990x; 0.100x) | 22.3 → 21.6 / 101.9 (1.002x; 0.212x) | 20.4 → 20.0 / 96.5 (1.002x; 0.207x) | 464.2 → 470.0 / 47.4 |


#### LFM2.5: grouped expert threshold 8 vs 24

Sources: control `lfm2.5-8b-a1b-expert-tensorops-threshold-24-vs-8.jsonl`, candidate `lfm2.5-8b-a1b-expert-tensorops-threshold-24-vs-8.jsonl`, llama.cpp reference `lfm2.5-8b-a1b-threshold8-current.jsonl`.

| Workload | Prefill tok/s: control → candidate / llama.cpp (candidate/control; candidate/llama) | Cached decode tok/s: control → candidate / llama.cpp (candidate/control; candidate/llama) | Complete-generation tok/s: control → candidate / llama.cpp (candidate/control; candidate/llama) | First-token latency ms: control → candidate / llama.cpp |
|---|---:|---:|---:|---:|
| Short | 22.4 → 22.5 / 216.6 (1.002x; 0.104x) | 21.1 → 21.1 / 94.7 (1.001x; 0.223x) | 13.6 → 13.6 / 74.8 (0.999x; 0.182x) | 491.7 → 490.2 / 51.0 |
| 128 prompt | 28.3 → 343.8 / 1334.2 (12.125x; 0.258x) | 20.9 → 21.0 / 88.3 (1.004x; 0.238x) | 3.2 → 15.0 / 60.6 (4.664x; 0.247x) | 4515.6 → 372.6 / 96.2 |
| 512 prompt | 120.4 → 425.5 / 1973.6 (3.536x; 0.216x) | 20.9 → 20.6 / 94.7 (0.988x; 0.218x) | 3.4 → 8.6 / 39.2 (2.545x; 0.219x) | 4253.5 → 1203.5 / 259.7 |
| 1,024 prompt | 120.5 → 447.0 / 1986.7 (3.717x; 0.225x) | 20.7 → 20.7 / 93.6 (0.999x; 0.221x) | 1.8 → 5.5 / 24.6 (3.013x; 0.224x) | 8498.6 → 2291.4 / 515.7 |
| Sustained decode | 22.4 → 22.4 / 216.5 (0.998x; 0.103x) | 21.0 → 20.9 / 94.4 (0.998x; 0.222x) | 19.5 → 19.5 / 89.0 (1.000x; 0.219x) | 490.8 → 491.4 / 51.0 |


#### LFM2.5: expert TensorOps M16 vs M32 tile

Sources: control `lfm2.5-8b-a1b-expert-tensorops-m32-vs-m16.jsonl`, candidate `lfm2.5-8b-a1b-expert-tensorops-m32-vs-m16.jsonl`, llama.cpp reference `lfm2.5-8b-a1b-threshold8-current.jsonl`.

| Workload | Prefill tok/s: control → candidate / llama.cpp (candidate/control; candidate/llama) | Cached decode tok/s: control → candidate / llama.cpp (candidate/control; candidate/llama) | Complete-generation tok/s: control → candidate / llama.cpp (candidate/control; candidate/llama) | First-token latency ms: control → candidate / llama.cpp |
|---|---:|---:|---:|---:|
| Short | 22.4 → 22.4 / 216.6 (0.999x; 0.103x) | 21.1 → 21.1 / 94.7 (1.000x; 0.223x) | 13.6 → 13.6 / 74.8 (0.997x; 0.181x) | 491.2 → 491.5 / 51.0 |
| 128 prompt | 351.7 → 273.0 / 1334.2 (0.780x; 0.205x) | 21.0 → 21.0 / 88.3 (1.001x; 0.238x) | 15.1 → 13.8 / 60.6 (0.915x; 0.228x) | 364.2 → 469.2 / 96.2 |
| 512 prompt | 431.4 → 303.0 / 1973.6 (0.702x; 0.154x) | 20.8 → 20.8 / 94.7 (1.002x; 0.220x) | 8.6 → 6.9 / 39.2 (0.796x; 0.176x) | 1187.1 → 1689.9 / 259.7 |
| 1,024 prompt | 454.7 → 312.7 / 1986.7 (0.689x; 0.157x) | 20.6 → 20.7 / 93.6 (1.004x; 0.221x) | 5.6 → 4.2 / 24.6 (0.749x; 0.170x) | 2252.4 → 3275.1 / 515.7 |
| Sustained decode | 22.4 → 22.4 / 216.5 (0.999x; 0.103x) | 20.9 → 21.0 / 94.4 (1.000x; 0.222x) | 19.5 → 19.5 / 89.0 (1.000x; 0.219x) | 490.9 → 491.5 / 51.0 |


#### LFM2.5: expert TensorOps M64 vs M32 tile

Sources: control `lfm2.5-8b-a1b-expert-tensorops-m32-vs-m64.jsonl`, candidate `lfm2.5-8b-a1b-expert-tensorops-m32-vs-m64.jsonl`, llama.cpp reference `lfm2.5-8b-a1b-threshold8-current.jsonl`.

| Workload | Prefill tok/s: control → candidate / llama.cpp (candidate/control; candidate/llama) | Cached decode tok/s: control → candidate / llama.cpp (candidate/control; candidate/llama) | Complete-generation tok/s: control → candidate / llama.cpp (candidate/control; candidate/llama) | First-token latency ms: control → candidate / llama.cpp |
|---|---:|---:|---:|---:|
| Short | 23.8 → 23.8 / 216.6 (1.001x; 0.110x) | 22.5 → 22.5 / 94.7 (1.002x; 0.237x) | 14.4 → 14.5 / 74.8 (1.004x; 0.193x) | 462.3 → 461.8 / 51.0 |
| 128 prompt | 365.7 → 393.4 / 1334.2 (1.078x; 0.295x) | 22.4 → 22.4 / 88.3 (0.998x; 0.254x) | 15.9 → 16.3 / 60.6 (1.030x; 0.269x) | 350.3 → 325.7 / 96.2 |
| 512 prompt | 446.2 → 473.7 / 1973.6 (1.062x; 0.240x) | 22.2 → 22.3 / 94.7 (1.005x; 0.235x) | 9.1 → 9.4 / 39.2 (1.037x; 0.241x) | 1147.7 → 1081.0 / 259.7 |
| 1,024 prompt | 473.2 → 527.3 / 1986.7 (1.119x; 0.265x) | 22.0 → 22.1 / 93.6 (1.001x; 0.236x) | 5.8 → 6.3 / 24.6 (1.085x; 0.258x) | 2164.5 → 1942.3 / 515.7 |
| Sustained decode | 22.2 → 22.9 / 216.5 (1.044x; 0.106x) | 21.9 → 22.3 / 94.4 (1.023x; 0.237x) | 20.0 → 20.3 / 89.0 (1.036x; 0.228x) | 495.7 → 480.0 / 51.0 |


#### Qwen2.5 Q4_0: native signed-int4 TensorOps

Sources: control `qwen2.5-q4_0-native-int4-ab.jsonl`, candidate `qwen2.5-q4_0-native-int4-ab.jsonl`, llama.cpp reference `qwen2.5-q4_0-current.jsonl`.

| Workload | Prefill tok/s: control → candidate / llama.cpp (candidate/control; candidate/llama) | Cached decode tok/s: control → candidate / llama.cpp (candidate/control; candidate/llama) | Complete-generation tok/s: control → candidate / llama.cpp (candidate/control; candidate/llama) | First-token latency ms: control → candidate / llama.cpp |
|---|---:|---:|---:|---:|
| Short | 793.3 → 652.6 / 1450.8 (0.825x; 0.450x) | 127.7 → 127.7 / 263.6 (1.001x; 0.485x) | 108.9 → 105.6 / 199.5 (0.968x; 0.529x) | 26.8 → 32.5 / 14.7 |
| 128 prompt | 3714.9 → 2460.7 / 6767.5 (0.658x; 0.364x) | 128.4 → 127.5 / 215.0 (0.993x; 0.593x) | 104.2 → 93.8 / 182.9 (0.901x; 0.513x) | 34.8 → 52.3 / 19.1 |
| 512 prompt | 5155.3 → 2991.7 / 9068.0 (0.580x; 0.330x) | 121.9 → 120.9 / 237.5 (0.992x; 0.509x) | 71.1 → 54.8 / 133.2 (0.768x; 0.411x) | 99.6 → 171.5 / 56.7 |
| 1,024 prompt | 4951.1 → 2969.4 / 8612.9 (0.601x; 0.345x) | 112.7 → 113.9 / 233.1 (1.016x; 0.489x) | 47.9 → 34.5 / 88.7 (0.721x; 0.389x) | 207.2 → 345.2 / 119.1 |
| Sustained decode | 785.6 → 649.0 / 1347.3 (0.826x; 0.482x) | 123.4 → 124.0 / 272.2 (1.005x; 0.455x) | 117.9 → 117.7 / 224.9 (0.998x; 0.523x) | 27.1 → 32.7 / 15.8 |


#### Qwen2.5 Q4_K_M: Q5_0 N4 direct GEMV

Sources: control `qwen2.5-q4_k_m-q5n4-ab.jsonl`, candidate `qwen2.5-q4_k_m-q5n4-ab.jsonl`, llama.cpp reference `qwen2.5-q4_k_m-q5n4-ab.jsonl`.

| Workload | Prefill tok/s: control → candidate / llama.cpp (candidate/control; candidate/llama) | Cached decode tok/s: control → candidate / llama.cpp (candidate/control; candidate/llama) | Complete-generation tok/s: control → candidate / llama.cpp (candidate/control; candidate/llama) | First-token latency ms: control → candidate / llama.cpp |
|---|---:|---:|---:|---:|
| Short | 689.1 → 684.8 / 1312.6 (1.004x; 0.522x) | 111.5 → 122.8 / 188.5 (1.112x; 0.644x) | 96.2 → 103.4 / 169.3 (1.089x; 0.609x) | 30.8 → 31.0 / 16.3 |
| 128 prompt | 3253.4 → 3323.7 / 6307.1 (1.020x; 0.525x) | 112.1 → 123.3 / 212.8 (1.098x; 0.583x) | 92.4 → 100.7 / 170.1 (1.085x; 0.592x) | 39.7 → 38.8 / 20.5 |
| 512 prompt | 4544.3 → 4579.5 / 8625.6 (1.009x; 0.532x) | 107.7 → 119.3 / 194.8 (1.112x; 0.610x) | 63.8 → 68.4 / 122.3 (1.073x; 0.559x) | 113.0 → 112.2 / 59.6 |
| 1,024 prompt | 4295.2 → 4340.1 / 8055.4 (1.010x; 0.539x) | 98.8 → 110.7 / 189.5 (1.124x; 0.564x) | 41.7 → 44.2 / 81.6 (1.056x; 0.541x) | 238.7 → 236.3 / 127.4 |
| Sustained decode | 671.3 → 674.7 / 1247.2 (0.982x; 0.541x) | 106.7 → 118.7 / 208.3 (1.120x; 0.574x) | 102.6 → 113.9 / 198.2 (1.110x; 0.573x) | 31.7 → 31.4 / 17.1 |

The Q4_K cooperative-input experiment was run while the unrelated Q8_0 MPP K tile was set to 32; its own candidate/control ratio remains valid, while its absolute engine rates should not be treated as current-production rates. The M=16 and M=64 rows are rejected candidates; their llama.cpp column is the current threshold-8 reference matrix, so candidate/llama ratios are ratios of condition medians rather than interleaved timings.


## Experiment 11: Q8_0 sixteen-row M=1 GEMV

Status: rejected; the temporary direct-MSL kernel and dispatch were removed. It did not improve either the all-layer Qwen3 Q8_0 workload or the Qwen2.5 Q4_K_M vocabulary-head workload.

The candidate assigned four output rows to each SIMD group and sixteen outputs to a 128-thread threadgroup, reusing each activation fragment across four rows. The retained Q8_0 M=1 path assigns two rows per SIMD group and eight outputs per threadgroup. This is a conventional MSL GEMV experiment, not Metal 4 TensorOps and not the standalone Apple Neural Engine/Core ML.

The paired matrix used five workloads (short, 128-, 512-, and 1,024-token prompts, plus 128-token sustained decode), three interleaved repetitions, and the normal release settings. Control and candidate ran in one Ferrum process; llama.cpp used the same artifact and prompt IDs. Ferrum's dispatch batch limit remained at its normal 1,024 setting. The pinned llama.cpp revision is `9710a32175b3b8f04636aaac4aa3cc28651b505d`.

Artifacts and raw results:

- Qwen3-0.6B Q8_0, SHA-256 `9465e63a22add5354d9bb4b99e90117043c7124007664907259bd16d043bb031`: `qwen3-0.6b-q8_0-q8n16-ab.jsonl` and `qwen3-0.6b-q8_0-q8n16-ab.run.log`.
- Qwen2.5-0.5B Q4_K_M, SHA-256 `74a4da8c9fdbcd15bd1f6d01d621410d31c6fc00986f5eb687824e7b93d7a9db`: `qwen2.5-q4_k_m-q8n16-ab.jsonl` and `qwen2.5-q4_k_m-q8n16-ab.run.log`.

Each numeric cell gives Ferrum control → candidate absolute values / llama.cpp absolute values, followed by `[control/llama → candidate/llama; candidate/control]`. Ratios are medians of the per-pair ratios; throughput is in tok/s and first-token latency is in ms. For the latency ratios, below 1.0 means lower latency. `C=K` and `K=L` count full generated-ID matches across the three pairs for that workload.

### Qwen3-0.6B Q8_0

| Workload | Prefill tok/s | Cached decode tok/s | Complete generation tok/s | First token ms | IDs |
|---|---:|---:|---:|---:|---|
| Short | 409.7→414.8 / 1,180.6 [0.348→0.351; 1.022] | 97.8→93.4 / 144.4 [0.678→0.647; 0.955] | 77.1→75.5 / 129.3 [0.597→0.583; 0.971] | 51.6→50.9 / 18.1 [2.854→2.821; 0.978] | C=K 3/3; K=L 3/3 |
| 128 | 1,872.4→1,844.1 / 5,653.5 [0.332→0.326; 0.992] | 97.3→93.3 / 143.7 [0.677→0.649; 0.957] | 71.9→69.4 / 124.0 [0.577→0.560; 0.970] | 68.7→69.7 / 22.9 [2.995→3.044; 1.008] | C=K 3/3; K=L 0/3 |
| 512 | 2,635.0→2,642.0 / 6,504.9 [0.401→0.406; 1.010] | 89.2→85.9 / 134.7 [0.662→0.633; 0.959] | 44.3→43.4 / 84.9 [0.514→0.511; 0.978] | 194.6→194.1 / 79.0 [2.490→2.459; 0.990] | C=K 3/3; K=L 3/3 |
| 1,024 | 2,760.3→2,786.1 / 5,786.6 [0.477→0.483; 1.009] | 80.0→76.3 / 125.8 [0.634→0.606; 0.955] | 28.8→28.6 / 55.2 [0.522→0.518; 0.993] | 371.3→367.9 / 177.2 [2.097→2.070; 0.991] | C=K 3/3; K=L 3/3 |
| Sustained decode | 398.4→402.9 / 1,159.9 [0.344→0.347; 1.009] | 95.6→90.3 / 145.7 [0.655→0.620; 0.944] | 90.4→85.6 / 139.1 [0.647→0.616; 0.950] | 53.0→52.5 / 18.3 [2.886→2.863; 0.992] | C=K 3/3; K=L 0/3 |

### Qwen2.5-0.5B Q4_K_M

| Workload | Prefill tok/s | Cached decode tok/s | Complete generation tok/s | First token ms | IDs |
|---|---:|---:|---:|---:|---|
| Short | 711.6→717.8 / 1,329.1 [0.537→0.527; 1.016] | 113.1→112.7 / 207.1 [0.538→0.537; 0.999] | 97.4→97.4 / 176.0 [0.553→0.557; 1.002] | 29.8→29.6 / 16.0 [1.855→1.892; 0.984] | C=K 3/3; K=L 0/3 |
| 128 | 3,260.5→3,269.7 / 6,233.1 [0.519→0.534; 0.998] | 113.7→113.9 / 201.3 [0.564→0.562; 1.002] | 92.4→92.0 / 164.9 [0.559→0.558; 0.996] | 39.6→39.5 / 20.8 [1.919→1.856; 1.001] | C=K 3/3; K=L 3/3 |
| 512 | 4,607.5→4,616.4 / 8,414.4 [0.548→0.550; 1.002] | 108.1→107.7 / 201.6 [0.536→0.535; 0.995] | 64.1→64.2 / 118.7 [0.533→0.544; 1.005] | 111.5→111.2 / 61.1 [1.821→1.818; 0.998] | C=K 3/3; K=L 3/3 |
| 1,024 | 4,347.8→4,434.6 / 8,054.1 [0.542→0.552; 1.020] | 98.2→101.0 / 200.3 [0.490→0.498; 1.009] | 42.1→42.9 / 80.7 [0.522→0.532; 1.018] | 235.9→231.2 / 127.4 [1.845→1.809; 0.980] | C=K 3/3; K=L 3/3 |
| Sustained decode | 690.8→693.9 / 1,264.3 [0.547→0.546; 1.005] | 110.4→109.7 / 206.9 [0.534→0.527; 0.994] | 105.7→105.0 / 195.7 [0.541→0.536; 0.993] | 30.7→30.6 / 16.9 [1.820→1.824; 0.996] | C=K 3/3; K=L 0/3 |

The candidate/control cached-decode ratios for Qwen3 ranged from 0.944 to 0.959; full-generation ratios ranged from 0.950 to 0.993. In Qwen2.5, decode ratios ranged from 0.994 to 1.009 and did not establish a repeatable gain for its Q8_0 vocabulary head. Candidate and control produced identical Ferrum token IDs in all 30 workload pairs. With no meaningful cross-shape win, the candidate kernel, selector, benchmark toggle, and temporary tail test were removed; the raw measurements remain as rejected-experiment evidence.


## Experiment 12: Q8_0 MPP K=64 tile for long prefill

Status: retained for Q8_0 MPP projection batches with `M >= 512`; smaller batches continue using K=128.

Q8_0 prompt GEMM already dispatches to Metal 4 MPP `matmul2d` for BF16 inputs when `M >= 16` and K is divisible by 128. The existing kernel stages a 64x128 BF16-dequantized weight tile (16 KiB) and reuses it across 64 prompt rows. This candidate stages a 64x64 tile (8 KiB), invokes the same cooperative `matmul2d` operation twice as often along K, and keeps the quantized weights packed in model storage. M=1 remains on the direct Q8_0 GEMV shader; this is MPP on the M5 GPU, not the standalone Neural Engine or Core ML.

The final shape guard selects K=64 at `M >= 512`, matching the shapes where both model families showed repeatable gains. Three interleaved pairs were run for each of five workloads on each model. The K=64 code path was validated at M=515, N=65, K=256, exercising M and N tails and multiple K tiles; the focused test passed with Metal API Validation and GPU Shader Validation enabled. Candidate/control generated IDs matched in all 30 pairs.

The matrices report Ferrum control → candidate absolute values / llama.cpp absolute values, followed by `[control/llama → candidate/llama; candidate/control]`. Ratios are medians of paired ratios; throughput is tok/s and first-token latency is ms. `K=64` marks rows where the final production selector uses the candidate; `K=128` rows use the same established path for both variants and their small differences are run variance.

### Qwen3-0.6B Q8_0

Artifact SHA-256: `9465e63a22add5354d9bb4b99e90117043c7124007664907259bd16d043bb031`. Final-dispatch files: `qwen3-0.6b-q8_0-q8k64-m512-ab.jsonl` and `.run.log`. The earlier `q8k64-ab` pair files preserve the forced K=64 shape exploration used to choose the M threshold.

| Workload | Tile | Prefill tok/s | Cached decode tok/s | Complete generation tok/s | First token ms | IDs |
|---|---|---:|---:|---:|---:|---|
| Short | K=128 | 438.9→439.4 / 1,273.3 [0.345→0.345; 1.001] | 106.2→106.6 / 164.5 [0.642→0.647; 1.004] | 84.0→84.2 / 145.2 [0.576→0.579; 1.003] | 48.1→48.1 / 16.8 [2.872→2.872; 0.999] | C=K 3/3; K=L 3/3 |
| 128 | K=128 | 1,975.6→1,955.0 / 6,116.9 [0.323→0.320; 0.984] | 107.2→106.0 / 164.2 [0.654→0.649; 0.994] | 78.0→77.4 / 139.6 [0.561→0.555; 0.989] | 65.1→65.8 / 21.2 [3.068→3.110; 1.016] | C=K 3/3; K=L 0/3 |
| 512 | K=64 | 2,837.5→3,082.8 / 7,109.9 [0.399→0.434; 1.086] | 98.8→97.4 / 156.2 [0.632→0.625; 0.986] | 47.9→49.8 / 95.6 [0.501→0.521; 1.042] | 180.8→166.4 / 72.3 [2.501→2.302; 0.920] | C=K 3/3; K=L 3/3 |
| 1,024 | K=64 | 3,019.3→3,247.0 / 6,256.3 [0.480→0.520; 1.083] | 87.3→87.5 / 143.1 [0.610→0.614; 0.997] | 31.4→32.9 / 60.9 [0.515→0.542; 1.053] | 339.5→315.7 / 163.9 [2.080→1.921; 0.923] | C=K 3/3; K=L 3/3 |
| Sustained decode | K=128 | 431.2→428.9 / 1,234.5 [0.349→0.349; 0.995] | 103.7→103.8 / 164.3 [0.630→0.630; 1.000] | 97.9→97.9 / 157.3 [0.622→0.622; 0.999] | 49.0→49.3 / 17.3 [2.840→2.841; 1.006] | C=K 3/3; K=L 0/3 |

### Qwen2.5-0.5B Q8_0

Artifact SHA-256: `ca59ca7f13d0e15a8cfa77bd17e65d24f6844b554a7b6c12e07a5f89ff76844e`. Final-dispatch files: `qwen2.5-q8_0-q8k64-m512-ab.jsonl` and `.run.log`. The earlier `q8k64-ab` pair files preserve the forced K=64 shape exploration used to choose the M threshold.

| Workload | Tile | Prefill tok/s | Cached decode tok/s | Complete generation tok/s | First token ms | IDs |
|---|---|---:|---:|---:|---:|---|
| Short | K=128 | 558.1→557.4 / 1,465.6 [0.384→0.379; 0.999] | 131.2→132.0 / 199.4 [0.661→0.663; 0.996] | 104.2→104.3 / 173.6 [0.601→0.602; 0.995] | 37.9→38.0 / 14.6 [2.580→2.613; 1.002] | C=K 3/3; K=L 0/3 |
| 128 | K=128 | 2,715.1→2,718.2 / 7,008.9 [0.387→0.387; 1.001] | 134.1→134.0 / 197.8 [0.675→0.678; 0.990] | 100.0→100.1 / 165.2 [0.600→0.606; 1.000] | 47.5→47.4 / 18.5 [2.561→2.566; 0.999] | C=K 3/3; K=L 3/3 |
| 512 | K=64 | 4,184.2→4,499.4 / 9,780.8 [0.426→0.460; 1.072] | 127.1→128.4 / 198.2 [0.647→0.645; 1.011] | 66.9→69.5 / 124.5 [0.537→0.558; 1.038] | 122.7→114.1 / 52.6 [2.341→2.169; 0.933] | C=K 3/3; K=L 3/3 |
| 1,024 | K=64 | 4,029.2→4,403.0 / 9,172.6 [0.439→0.479; 1.093] | 117.2→117.8 / 195.1 [0.597→0.600; 1.005] | 42.8→45.4 / 86.5 [0.493→0.525; 1.058] | 254.4→232.9 / 111.9 [2.278→2.085; 0.915] | C=K 3/3; K=L 3/3 |
| Sustained decode | K=128 | 552.1→547.7 / 1,429.8 [0.388→0.383; 1.003] | 128.3→128.3 / 199.5 [0.644→0.644; 1.000] | 120.8→120.8 / 188.9 [0.638→0.640; 1.000] | 38.4→38.7 / 14.9 [2.559→2.591; 0.997] | C=K 3/3; K=L 0/3 |

At 512 and 1,024 prompt tokens, all three paired runs improved prefill for both models: Qwen3 median candidate/control ratios were 1.086 and 1.083; Qwen2.5 ratios were 1.072 and 1.093. First-token latency and complete-generation throughput improved in these long-prefill workloads; cached-decode kernels were unchanged. Prefill ratios versus llama.cpp rose from 0.399→0.434 and 0.480→0.520 for Qwen3, and 0.426→0.460 and 0.439→0.479 for Qwen2.5. The general Phase 7 gate remains unmet.


## Experiment 13: Q4_K MPP K=64 tile for 1,024-row prefill

Status: retained for Q4_K Metal 4 MPP projections with `M >= 1024`; smaller batches keep the existing K=128 tile.

Q4_K GGUF blocks contain a superblock scale/minimum and per-32-value 6-bit scale/minimum fields, so their packed payload is not a zero-copy native int4 TensorOps operand. This candidate keeps the exact existing BF16 dequantization and Metal 4 `matmul2d` path, but stages a 64x64 weight tile (8 KiB) instead of a 64x128 tile (16 KiB). The K=64 kernel makes two MPP calls per old K=128 tile. M=1 decode remains on the direct Q4_K GEMV shader.

The correctness test used M=1,025, N=65, K=256, covering batch and output tails plus multiple K tiles. It compared the candidate against a scalar reference using the BF16-rounded inputs and weights. The focused test passed with Metal API Validation and GPU Shader Validation enabled. The production selector only enables K=64 at M>=1,024; the ordinary Q4_K projection dispatch remains unchanged at smaller M.

The Qwen2.5-0.5B Q4_K_M release A/B used the pinned GGUF and llama.cpp reference, five workloads, and three interleaved control/candidate/reference pairs. The LFM2.5-8B-A1B Q4_K_M cross-check used the pinned artifact and the same 1,024-token prompt IDs for three pairs. Ferrum outputs matched control in all 18 pairs; LFM matched llama.cpp in all three pairs for this case. In Qwen2.5 at M=1,024, all three pairs improved prefill, complete-generation throughput, and first-token latency. The LFM median also improved across the three pairs, with one pair slightly slower; its paired prefill ratios ranged 0.984–1.051x and full-generation ratios 0.987–1.038x. The small decode changes are measurement noise because the decode kernel was untouched.

Each throughput entry shows Ferrum control → K=64 candidate / llama.cpp in tok/s, followed by `[control/llama → candidate/llama; candidate/control]`. Ratios are medians of paired ratios. First-token latency is shown as control → candidate / llama.cpp in milliseconds. `K=128` rows use the same established kernel in both variants under the final shape guard.

### Qwen2.5-0.5B Q4_K_M

Artifact SHA-256: `74a4da8c9fdbcd15bd1f6d01d621410d31c6fc00986f5eb687824e7b93d7a9db`. Final-dispatch matrix: `qwen2.5-q4_k_m-q4kk64-m1024-ab.jsonl` and `.run.log`. The earlier `q4kk64-ab` files preserve the shape sweep that led to the M>=1,024 threshold.

| Workload | Tile | Prefill tok/s | Cached decode tok/s | Complete generation tok/s | First-token latency ms | IDs |
|---|---|---:|---:|---:|---:|---|
| Short | K=128 | 741.1→739.8 / 1,406.8 [0.538→0.522; 1.004] | 122.1→122.2 / 235.2 [0.515→0.518; 1.002] | 104.4→103.4 / 196.4 [0.532→0.529; 0.991] | 28.6→28.7 / 15.2 | C=K 3/3; K=L 0/3 |
| 128 | K=128 | 3,534.9→3,526.1 / 6,919.6 [0.513→0.510; 0.993] | 124.0→123.5 / 241.2 [0.516→0.512; 0.993] | 100.5→100.3 / 189.3 [0.533→0.529; 0.998] | 36.5→36.6 / 18.8 | C=K 3/3; K=L 3/3 |
| 512 | K=128 | 4,952.0→4,988.7 / 9,354.2 [0.529→0.534; 1.007] | 116.5→117.2 / 233.6 [0.499→0.498; 0.999] | 69.1→69.4 / 134.2 [0.513→0.517; 1.007] | 103.7→103.0 / 55.0 | C=K 3/3; K=L 3/3 |
| 1,024 | K=64 | 4,742.1→4,830.9 / 8,752.9 [0.539→0.546; 1.014] | 107.9→108.3 / 231.6 [0.466→0.468; 1.005] | 45.8→46.3 / 89.4 [0.513→0.517; 1.008] | 216.3→212.3 / 117.2 | C=K 3/3; K=L 3/3 |
| Sustained decode | K=128 | 720.8→710.6 / 1,385.6 [0.525→0.513; 0.986] | 119.0→119.2 / 244.2 [0.485→0.488; 1.003] | 113.6→113.7 / 222.2 [0.511→0.512; 1.001] | 29.4→29.9 / 15.4 | C=K 3/3; K=L 0/3 |

### LFM2.5-8B-A1B Q4_K_M

Artifact SHA-256: `4923ec14f06b968b74d663e5949867d2d9c3bf13a20b8be1a9f9af39989b2bb0`. The one-case workload is `lfm2.5-8b-a1b-q4_k_m-1024-workload.jsonl`; paired results and runner output are `lfm2.5-8b-a1b-q4_k_m-q4kk64-m1024-ab.jsonl` and `.run.log`.

| Workload | Tile | Prefill tok/s | Cached decode tok/s | Complete generation tok/s | First-token latency ms | IDs |
|---|---|---:|---:|---:|---:|---|
| 1,024 | K=64 | 477.3→494.4 / 2,160.6 [0.220→0.228; 1.036] | 22.4→22.4 / 102.0 [0.220→0.220; 1.002] | 5.9→6.1 / 26.8 [0.220→0.225; 1.025] | 2,145.7→2,071.6 / 474.2 | C=K 3/3; K=L 3/3 |

The K=64 tile is retained as a shape-specific prefill win: Qwen2.5's 1,024-token prefill rose from 4,742.1 to 4,830.9 tok/s and first-token latency fell from 216.3 to 212.3 ms; LFM2.5's prefill rose from 477.3 to 494.4 tok/s and first-token latency fell from 2,145.7 to 2,071.6 ms. Cached decode was effectively flat. The absolute llama.cpp rates and Ferrum/reference ratios remain visible above; this change does not clear the Phase 7 gate.


## Experiment 14: K=64 TensorOps tiles for quantized MoE experts

Status: retained for grouped Q4_K and Q6_K expert MPP projections. Q5_K experts retain K=128 until a real-model matrix exercises that format.

The LFM2.5-8B-A1B 1,024-token flushed-dispatch profile attributed 1.66 s of sampled GPU work to grouped expert MPP projections: Q4_K input 1.126 s, Q4_K output 0.316 s, and Q6_K output 0.215 s. These are diagnostic totals, not normal end-to-end timings. The candidate halves the staged BF16 weight tile from 64x128 to 64x64 (16 KiB to 8 KiB), while retaining the M=32, N=64 tile and exact GGUF dequantization. K=64 applies only after the existing route-density, dtype, alignment, and MPP eligibility checks; the conventional small-batch and M=1 expert paths are unchanged.

The correctness tests exercise Q4_K, Q5_K, and Q6_K expert TensorOps over uneven expert groups, 65 output rows, and 512 input features. The production K=64 variants for Q4_K and Q6_K passed scalar-reference comparisons with the routing order restored. All eight focused expert TensorOps tests passed with Metal API Validation and GPU Shader Validation enabled; the validation log is `lfm2.5-8b-a1b-expert-k64-suite-validation.log`.

The LFM2.5-8B-A1B Q4_K_M release A/B used the pinned model, five workloads, three interleaved pairs, and the same llama.cpp reference. Ferrum output IDs matched control in all 15 pairs. The candidate improved prefill at 128, 512, and 1,024 tokens while leaving cached decode effectively flat. At 512 tokens it raised Ferrum prefill from 458.1 to 476.2 tok/s; at 1,024 tokens, from 490.6 to 497.4 tok/s. Candidate/reference ratios are included with the absolute rates below. The 1,024-token flushed-dispatch profile reduced the sampled expert MPP GPU total from 1.531 s to 1.465 s (4.3%); this isolated profile is diagnostic only.

Raw A/B rows and runner output are `lfm2.5-8b-a1b-expert-q4k64-ab.jsonl` and `.run.log`. The GPU-timing attribution capture is `lfm2.5-8b-a1b-expert-q4k64-profile.jsonl` and `.run.log`; the five-case prompts are from `lfm2.5-8b-a1b-q4_k_m-workloads.jsonl`.

Each throughput entry shows Ferrum control → K=64 candidate / llama.cpp in tok/s, followed by `[control/llama → candidate/llama; candidate/control]`. Ratios are medians of paired ratios. First-token latency is shown as control → candidate / llama.cpp in milliseconds.

| Workload | Prefill tok/s | Cached decode tok/s | Complete generation tok/s | First-token latency ms | IDs |
|---|---:|---:|---:|---:|---|
| Short | 24.2→24.2 / 235.1 [0.103→0.103; 1.000] | 23.0→22.9 / 102.9 [0.222→0.222; 0.998] | 14.7→14.7 / 82.0 [0.179→0.180; 1.001] | 454.0→454.2 / 47.0 | C=K 3/3; K=L 3/3 |
| 128 prompt | 380.3→389.4 / 1,537.1 [0.247→0.252; 1.022] | 22.8→22.8 / 103.1 [0.222→0.221; 1.000] | 16.3→16.4 / 70.0 [0.233→0.235; 1.005] | 336.9→329.0 / 83.5 | C=K 3/3; K=L 0/3 |
| 512 prompt | 458.1→476.2 / 2,157.7 [0.212→0.221; 1.040] | 22.4→22.5 / 102.6 [0.220→0.220; 1.002] | 9.3→9.5 / 42.7 [0.216→0.223; 1.026] | 1,118.0→1,075.4 / 237.5 | C=K 3/3; K=L 3/3 |
| 1,024 prompt | 490.6→497.4 / 2,155.6 [0.228→0.231; 1.013] | 22.5→22.5 / 101.3 [0.222→0.222; 0.999] | 6.0→6.1 / 26.6 [0.226→0.228; 1.011] | 2,087.4→2,059.2 / 475.3 | C=K 3/3; K=L 3/3 |
| Sustained decode | 21.8→24.3 / 234.2 [0.093→0.104; 1.115] | 22.7→22.8 / 102.2 [0.223→0.223; 1.000] | 20.9→21.1 / 96.7 [0.218→0.217; 1.011] | 504.8→452.9 / 47.2 | C=K 3/3; K=L 3/3 |

The result supports K=64 for the Q4_K/Q6_K grouped expert path, not a blanket TensorOps dispatch. Q5_K remains on K=128 because no available end-to-end MoE artifact exercises Q5_K expert weights. The LFM2.5 engine gap remains architecture-specific and substantial; no large Qwen target is unlocked.
