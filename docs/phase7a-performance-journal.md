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
| Quantized MoE expert projections | BF16 prefill batches that average at least 8 assignments per expert use expert grouping plus Q4_K/Q5_K/Q6_K MPP GEMM; one-token routing remains a separate GPU routing optimization. | Smaller batches, M=1, non-BF16 activations, and unsupported alignment continue through conventional SIMD expert kernels. MPP stages exact GGUF values into BF16 and uses a 32x64x128 `matmul2d` tile; it does not reinterpret GGUF blocks as Metal's native block-scaled int4/int8 tensors. |
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
