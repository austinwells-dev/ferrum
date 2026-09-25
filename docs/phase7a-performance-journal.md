# Phase 7A GGUF performance journal

## Scope and gate

Phase 7A starts on `codex/phase7-gguf-performance`, from the verified Phase 6 checkpoint `66d721d9bfab04339ad170a426030c847f2d2028`. The first campaign establishes matched small-model baselines and tests general GGUF kernel improvements. The larger Qwen3.8 and Qwen3.6 targets remain behind the performance gate.

The gate is at least 0.85x llama.cpp decode throughput and 0.80x prefill throughput for medium and long prompts, with no routine workload below 0.70x without an isolated explanation. No measured architecture/format currently clears it.

Reporting rule: every important Ferrum-versus-llama.cpp result reports absolute Ferrum and llama.cpp prefill, cached-decode, and complete-generation tok/s beside their ratios, with first-token latency for both runtimes where available. Internal candidate/control results report the absolute before/after rates, llama.cpp context when available, ratios, and first-token latency. Ratios support the gate but do not replace throughput values. Use existing raw JSONL for journal updates; do not rerun a benchmark only to reformat its results.

## Machine and reference runtime

- Machine: Apple M5, Mac17,2, 32 GiB unified memory; macOS 27.0; Metal 4 / Apple GPU family 10.
- Ferrum: release build from this branch, native Rust and Metal kernels.
- Historical reference for the initial matrices and earlier experiments: official `ggml-org/llama.cpp` at commit `9710a32175b3b8f04636aaac4aa3cc28651b505d`. These measurements remain useful for those paired experiments but are not the current-upstream gate baseline. The refreshed official upstream `origin/HEAD`, fetched on 2026-09-24, is `84e76d8a23162eca70490da131945ebec1f09bf4`; its Metal Release build uses an embedded Metal library, OpenMP off, and tests/examples/server off. `tools/phase55_llama_matrix.cpp` records `llama_commit` in every refreshed reference row.
- Both runtimes use the same GGUF bytes and exact prompt token IDs per pair, greedy argmax, BF16 KV, context 4096, batch 2048, microbatch 512, four CPU threads, and all model layers on Metal. Model loading and one two-token warmup per workload are outside measured generation. Historical baseline matrices have three interleaved pairs per workload; the current-upstream refresh has three pairs except the Q5_1 tile A/B, which has five.
- Prefill, first-token latency, cached decode, full-generation throughput, output IDs, KV allocation, and process RSS are retained separately in the JSONL records. Prompt tokenization is outside the timed path.

## Artifacts and workload coverage

| Model / format | Pinned artifact | SHA-256 | Workload file | Matched matrix |
| --- | --- | --- | --- | --- |
| Qwen2.5-0.5B Q4_0 | Phase 5 pinned `Q4_0.gguf` | `7671c0c304e6ce5a7fc577bcb12aba01e2c155cc2efd29b2213c95b18edaf6ed` | `qwen2.5-q4k-workloads.jsonl` | Historical: `qwen2.5-q4_0-current.jsonl`; refreshed: `qwen2.5-q4_0-llama84e76d8-current.jsonl` |
| Qwen2.5-0.5B Q4_K_M | Phase 5 pinned `Q4_K_M.gguf` | `74a4da8c9fdbcd15bd1f6d01d621410d31c6fc00986f5eb687824e7b93d7a9db` | `qwen2.5-q4k-workloads.jsonl` | Historical: `qwen2.5-q4_k_m-current.jsonl` and `qwen2.5-q4_k_m-baseline-recheck.jsonl`; refreshed: `qwen2.5-q4_k_m-llama84e76d8-current.jsonl` |
| Qwen2.5-0.5B Q5_K_M | Phase 5 pinned `Q5_K_M.gguf` | `041474553fcabfc2a2d67903f9d2c2e50bd92528e670da4f33b5d0ce6e59fd55` | `qwen2.5-q4k-workloads.jsonl` | Historical: `qwen2.5-q5_k_m-current.jsonl`; refreshed candidates: `qwen2.5-q5_k_m-q5_1k64-m512-ab.jsonl`, `qwen2.5-q5_k_m-q5kk64-m1024-ab.jsonl` |
| Qwen2.5-0.5B Q6_K | Phase 5 pinned `Q6_K.gguf` | `2f82233630c349ccf6b8daccf48f9a7865713d9f08a2eadfa456cebe9b97c7f5` | `qwen2.5-q4k-workloads.jsonl` | Historical: `qwen2.5-q6_k-current.jsonl`; refreshed: `qwen2.5-q6_k-llama84e76d8-current.jsonl` |
| Qwen2.5-0.5B Q8_0 | Phase 5 pinned `Q8_0.gguf` | `ca59ca7f13d0e15a8cfa77bd17e65d24f6844b554a7b6c12e07a5f89ff76844e` | `qwen2.5-q4k-workloads.jsonl` | Historical: `qwen2.5-q8_0-current.jsonl`; refreshed: `qwen2.5-q8_0-llama84e76d8-current.jsonl` |
| Qwen3-0.6B Q8_0 | `Qwen/Qwen3-0.6B-GGUF`, revision `23749fefcc72300e3a2ad315e1317431b06b590a` | `9465e63a22add5354d9bb4b99e90117043c7124007664907259bd16d043bb031` | `qwen3-0.6b-q8_0-workloads.jsonl` | Historical: `qwen3-0.6b-q8_0-current.jsonl`; refreshed: `qwen3-0.6b-q8_0-llama84e76d8-current.jsonl` |
| LFM2.5-8B-A1B Q4_K_M | `LiquidAI/LFM2.5-8B-A1B-GGUF`, revision `49c14831707011e64d70b2ebd8462ba08d608434` | `4923ec14f06b968b74d663e5949867d2d9c3bf13a20b8be1a9f9af39989b2bb0` | `lfm2.5-8b-a1b-q4_k_m-workloads.jsonl` | Historical: `lfm2.5-8b-a1b-q4_k_m-current.jsonl`; refreshed: `lfm2.5-8b-a1b-q4_k_m-llama84e76d8-current.jsonl` |

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

This gate-ratio summary is an index; absolute Ferrum and llama.cpp tok/s plus first-token latency for each workload are tabulated in the current-upstream refresh and the JSONL-derived absolute performance record below.

Output differences are format/workload dependent. The five Qwen2.5 matrices produced exact IDs in 9–12 of 15 pairs; Qwen3 did so in 12 of 15. LFM2 matched all short and sustained sequences and the 1,024-token case, while its 128- and 512-token cases diverged after 9 and 8 generated IDs respectively. Divergences remain visible in raw rows and are not hidden by throughput summaries.

### Current-upstream refresh: llama.cpp `84e76d8a`

The following refresh uses the same artifacts, prompt IDs, generation lengths, greedy policy, BF16 KV, and runner settings with official upstream commit `84e76d8a23162eca70490da131945ebec1f09bf4`. Each Q4_0, Q4_K_M, Q6_K, Q8_0, Qwen3, and LFM matrix has three interleaved pairs per workload; Q5_K_M uses five interleaved control/candidate/reference pairs. This is the earlier refresh snapshot before Experiments 20–22. Ferrum and llama.cpp values are condition medians. Ratios are medians of matched per-pair rates. First-token latency is reported in milliseconds. Exact IDs count full sequence matches per pair. Its Q5_K_M candidate uses the measured Q5_1 MPP K=64 tile at M>=512 and the established Q5_K K=128 path.

| Model / workload | Prefill tok/s Ferrum / llama.cpp (ratio) | Cached decode tok/s Ferrum / llama.cpp (ratio) | Complete-generation tok/s Ferrum / llama.cpp (ratio) | First-token latency ms Ferrum / llama.cpp | Exact IDs |
|---|---:|---:|---:|---:|---:|
| Qwen2.5 Q4_0 / Short | 797.8 / 1,531.8 (0.521x) | 129.0 / 278.8 (0.463x) | 110.2 / 220.0 (0.501x) | 26.7 / 13.9 | 3/3 |
| Qwen2.5 Q4_0 / 128-token prompt | 3,702.4 / 7,342.7 (0.504x) | 129.1 / 260.0 (0.494x) | 102.5 / 203.1 (0.517x) | 34.9 / 17.7 | 3/3 |
| Qwen2.5 Q4_0 / 512-token prompt | 5,335.9 / 9,985.4 (0.534x) | 122.1 / 270.1 (0.454x) | 73.3 / 146.1 (0.499x) | 96.3 / 51.5 | 3/3 |
| Qwen2.5 Q4_0 / 1,024-token prompt | 4,990.8 / 9,369.1 (0.532x) | 113.3 / 264.9 (0.425x) | 48.2 / 97.3 (0.496x) | 205.5 / 109.6 | 3/3 |
| Qwen2.5 Q4_0 / Sustained decode | 768.3 / 1,494.4 (0.514x) | 125.2 / 277.9 (0.450x) | 119.4 / 249.2 (0.480x) | 27.6 / 14.3 | 0/3 |
| Qwen2.5 Q4_K_M / Short | 747.2 / 1,396.5 (0.533x) | 121.5 / 234.6 (0.518x) | 103.9 / 195.7 (0.531x) | 28.4 / 15.3 | 0/3 |
| Qwen2.5 Q4_K_M / 128-token prompt | 3,491.6 / 6,893.8 (0.508x) | 122.8 / 234.8 (0.523x) | 98.6 / 186.2 (0.530x) | 37.1 / 18.8 | 3/3 |
| Qwen2.5 Q4_K_M / 512-token prompt | 4,974.1 / 9,344.6 (0.532x) | 116.7 / 242.5 (0.480x) | 69.1 / 134.3 (0.514x) | 103.3 / 55.1 | 3/3 |
| Qwen2.5 Q4_K_M / 1,024-token prompt | 4,769.5 / 8,773.9 (0.542x) | 107.5 / 232.0 (0.462x) | 45.9 / 89.6 (0.511x) | 215.0 / 117.0 | 3/3 |
| Qwen2.5 Q4_K_M / Sustained decode | 660.6 / 1,384.2 (0.477x) | 118.5 / 244.4 (0.485x) | 112.5 / 222.9 (0.507x) | 32.1 / 15.4 | 0/3 |
| Qwen2.5 Q5_K_M / Short | 703.5 / 1,385.5 (0.510x) | 114.8 / 223.8 (0.511x) | 97.9 / 187.9 (0.518x) | 30.2 / 15.5 | 0/5 |
| Qwen2.5 Q5_K_M / 128-token prompt | 3,361.4 / 6,816.3 (0.487x) | 116.7 / 224.6 (0.515x) | 94.7 / 180.8 (0.515x) | 38.4 / 19.0 | 5/5 |
| Qwen2.5 Q5_K_M / 512-token prompt | 5,005.9 / 9,392.3 (0.533x) | 111.6 / 222.0 (0.502x) | 67.8 / 130.8 (0.517x) | 102.6 / 54.8 | 5/5 |
| Qwen2.5 Q5_K_M / 1,024-token prompt | 4,841.0 / 8,794.4 (0.551x) | 104.0 / 217.6 (0.476x) | 45.7 / 88.2 (0.517x) | 211.8 / 116.7 | 5/5 |
| Qwen2.5 Q5_K_M / Sustained decode | 689.1 / 1,376.8 (0.499x) | 111.9 / 229.3 (0.488x) | 107.1 / 211.9 (0.506x) | 30.8 / 15.5 | 0/5 |
| Qwen2.5 Q6_K / Short | 632.5 / 1,481.1 (0.427x) | 130.5 / 205.2 (0.634x) | 106.5 / 177.4 (0.599x) | 33.5 / 14.4 | 3/3 |
| Qwen2.5 Q6_K / 128-token prompt | 3,003.0 / 7,009.7 (0.428x) | 129.6 / 200.7 (0.650x) | 99.2 / 167.3 (0.597x) | 43.2 / 18.5 | 3/3 |
| Qwen2.5 Q6_K / 512-token prompt | 4,693.0 / 9,669.1 (0.485x) | 125.2 / 202.5 (0.618x) | 70.2 / 125.3 (0.559x) | 109.4 / 53.2 | 3/3 |
| Qwen2.5 Q6_K / 1,024-token prompt | 4,609.9 / 9,001.5 (0.515x) | 115.5 / 197.4 (0.585x) | 46.3 / 85.9 (0.544x) | 222.5 / 114.0 | 3/3 |
| Qwen2.5 Q6_K / Sustained decode | 575.9 / 1,456.2 (0.395x) | 125.6 / 205.9 (0.609x) | 118.9 / 194.2 (0.612x) | 36.8 / 14.7 | 0/3 |
| Qwen2.5 Q8_0 / Short | 558.8 / 1,449.2 (0.386x) | 131.7 / 197.2 (0.671x) | 104.1 / 172.8 (0.604x) | 37.9 / 14.7 | 0/3 |
| Qwen2.5 Q8_0 / 128-token prompt | 2,674.1 / 7,010.2 (0.380x) | 132.3 / 197.0 (0.679x) | 98.2 / 162.4 (0.608x) | 48.3 / 18.5 | 3/3 |
| Qwen2.5 Q8_0 / 512-token prompt | 4,463.8 / 9,734.6 (0.459x) | 127.8 / 196.8 (0.649x) | 68.4 / 123.4 (0.559x) | 115.1 / 52.8 | 3/3 |
| Qwen2.5 Q8_0 / 1,024-token prompt | 4,401.5 / 9,168.2 (0.480x) | 117.5 / 192.5 (0.611x) | 45.3 / 86.0 (0.528x) | 233.0 / 111.9 | 3/3 |
| Qwen2.5 Q8_0 / Sustained decode | 516.2 / 1,423.2 (0.363x) | 128.1 / 198.6 (0.646x) | 120.3 / 187.2 (0.643x) | 41.0 / 15.0 | 0/3 |
| Qwen3-0.6B Q8_0 / Short | 436.4 / 1,287.2 (0.341x) | 106.6 / 164.1 (0.650x) | 83.9 / 145.1 (0.578x) | 48.4 / 16.6 | 3/3 |
| Qwen3-0.6B Q8_0 / 128-token prompt | 1,902.2 / 5,959.3 (0.321x) | 104.7 / 160.8 (0.651x) | 74.2 / 136.0 (0.550x) | 67.7 / 21.7 | 0/3 |
| Qwen3-0.6B Q8_0 / 512-token prompt | 3,070.0 / 7,130.7 (0.431x) | 98.3 / 154.5 (0.636x) | 49.7 / 94.8 (0.523x) | 167.1 / 72.1 | 3/3 |
| Qwen3-0.6B Q8_0 / 1,024-token prompt | 3,239.6 / 6,254.3 (0.519x) | 87.8 / 141.9 (0.616x) | 32.8 / 60.7 (0.541x) | 316.4 / 164.0 | 3/3 |
| Qwen3-0.6B Q8_0 / Sustained decode | 399.8 / 1,240.9 (0.322x) | 103.4 / 162.9 (0.634x) | 97.2 / 156.1 (0.622x) | 52.9 / 17.2 | 0/3 |
| LFM2.5-8B-A1B Q4_K_M / Short | 24.3 / 236.6 (0.103x) | 22.9 / 101.3 (0.226x) | 14.7 / 81.0 (0.181x) | 452.4 / 46.7 | 3/3 |
| LFM2.5-8B-A1B Q4_K_M / 128-token prompt | 386.9 / 1,547.3 (0.251x) | 22.8 / 104.7 (0.218x) | 16.4 / 70.9 (0.232x) | 331.1 / 82.9 | 0/3 |
| LFM2.5-8B-A1B Q4_K_M / 512-token prompt | 475.6 / 2,156.5 (0.221x) | 22.5 / 102.8 (0.219x) | 9.5 / 42.9 (0.220x) | 1,076.9 / 237.6 | 3/3 |
| LFM2.5-8B-A1B Q4_K_M / 1,024-token prompt | 491.6 / 2,176.6 (0.229x) | 22.5 / 102.6 (0.220x) | 6.0 / 26.9 (0.226x) | 2,083.4 / 470.7 | 3/3 |
| LFM2.5-8B-A1B Q4_K_M / Sustained decode | 24.3 / 231.4 (0.105x) | 22.7 / 103.9 (0.219x) | 21.1 / 98.3 (0.215x) | 453.6 / 47.8 | 3/3 |

Across this refresh snapshot, no representative format meets the medium/long prefill or sustained-decode threshold. Qwen2.5 Q6_K has the strongest sustained-decode ratio at 0.609x (125.6 / 205.9 tok/s); the Q5_K_M 1,024-token row in this snapshot is 0.551x prefill (4,841.0 / 8,794.4 tok/s), 0.476x cached decode (104.0 / 217.6 tok/s), and 0.517x complete generation (45.7 / 88.2 tok/s), with 211.8 / 116.7 ms first-token latency. Experiment 22 later measures the Q5_K K=64 prefill candidate with absolute before/after and llama.cpp values. All results remain below their Phase 7 thresholds, so the large Qwen targets remain locked.

## Upstream Metal review

The refreshed official upstream `84e76d8a` Metal source keeps its Q4_K M=1 defaults at `N_R0_Q4_K=2` and `N_SG_Q4_K=2` in `ggml-metal/ggml-metal-impl.h`; `kernel_mul_mv_q4_K_f32_impl` in `ggml-metal/kernels/mul_mv.metal` reuses activation fragments while accumulating adjacent output rows. An M5-specific tuning discussion in [llama.cpp issue 19303](https://github.com/ggml-org/llama.cpp/issues/19303) reports a larger Q4_K row-tile experiment, but that remains an unmerged result. The refreshed upstream also has a multi-kernel `flash_attn_ext` implementation in `ggml-metal/kernels/fa.metal` and its Metal host scheduler. Ferrum's prefill attention score/context products already use MPP, but its mask and softmax remain separate conventional MSL kernels; fusing the score product, mask, row softmax, and context product with TensorOps cooperative intermediates is still a plausible prefill candidate. A flushed Qwen Q5 profile attributed 28.51 ms to 24 `attention_softmax` calls at 1,024 rows; this is diagnostic GPU time, not end-to-end latency.

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
| GGUF Q4_0, Q5_0, Q5_1, Q4_K, Q5_K, Q6_K, and Q8_0 projections | MPP GEMM uses measured format-specific row boundaries for BF16 activations and K divisible by 128: Q4_0/Q4_K M>=4, Q5_K/Q6_K M>=8, Q8_0 M>=12, and Q5_0/Q5_1 M>=16. Quantized MPP kernels unpack custom GGUF blocks into bounded BF16 tiles before `matmul2d`; Q4_K and Q5_K use measured 64-wide K tiles at M>=1024, and Q4_K also uses the measured 128-row M tile there. | Rows below each format's boundary and unsupported dtype/K shapes still use conventional `q*_gemm` MSL. M=1 uses format-specific direct `q*_gemv` kernels, including measured Q5_0, Q5_1, and Q5_K row-reuse variants; it never enters MPP. MLX affine-Q4 is a separate F16 path and is outside this GGUF campaign. |
| Quantized MoE expert projections | BF16 prefill batches that average at least 8 assignments per expert use expert grouping plus Q4_K/Q5_K/Q6_K MPP GEMM; Q4_K and Q6_K use a 32x64x64 `matmul2d` tile, while Q5_K keeps K=128 pending a model-level measurement. One-token routing remains a separate GPU routing optimization. | Sparse assignments and unsupported dtypes/alignment use direct MSL. Q4_K with at least 128 output rows and K divisible by 256 now uses a measured two-output-row-per-SIMD fallback; Q5_K/Q6_K still use the scalar expert kernel. MPP stages exact GGUF values into BF16; it does not reinterpret GGUF blocks as Metal's native block-scaled int4/int8 tensors. |
| Prefill attention | BF16/F16 attention score and context products use TensorOps when query length exceeds one; production scale/causal-mask/row softmax uses one conventional MSL `attention_softmax` kernel. | The softmax reduction remains MSL. Its retained prefix variant skips masked suffix reductions only for full-prefill shapes with M>=256. A fused FlashAttention-style TensorOps kernel using cooperative results and row reductions remains a larger prefill candidate. Decode's one-query context stays on its direct kernel. |

The dispatch rules are visible in `src/ops/transformer.rs`; the current quantized MPP unpack/stage/multiply path is in `src/metal/shaders/project_mpp.metal`; quantized expert projection is in `src/metal/shaders/ops.metal`. Norm, RoPE, activation, and elementwise kernels are conventional MSL but are not matrix contractions that should be moved to `matmul2d` by default.

Profile attribution shows why the remaining conventional shapes need separate treatment. In Qwen2.5 Q4_K M=1 decode, the leading projection costs were `up_proj.q5_0_gemv` at 1.812 ms, `lm_head.q8_0_gemv_8rows` at 1.650 ms, and `gate_proj.q5_0_gemv` at 1.332 ms. These one-row GEMVs are not TensorOps candidates for this dispatch; direct shaders remain selected. In the LFM2.5 1,024-token flushed-dispatch diagnostic, no conventional expert projection fallback appeared: grouped MPP expert products accounted for about 1.126 s (Q4_K input), 0.316 s (Q4_K output), and 0.215 s (Q6_K output). Conventional `attention_softmax` was about 17 ms in that capture, while attention score and context products already used TensorOps. These are diagnostic GPU totals rather than end-to-end timings.

### Native quantized API constraints

The current Xcode 27.0 SDK header `Metal.framework/Headers/MTLTensor.h` exposes signed/unsigned int8 tensors in macOS 26, int4/uint4 in macOS 26.4, and int2/uint2 plus FP4 and FP8 tensor formats in macOS 27. The installed `MPPTensorOpsMatMul2d.h` documents native `int4b_format`/`uint4b_format` and int8/uint8 matmul operands with half or BF16 activations, plus the newer int2 and FP4/FP8 combinations. Cooperative tensors are supported as matmul inputs through `get_left_input_cooperative_tensor` and `get_right_input_cooperative_tensor`; callers can query layout compatibility before reusing an intermediate. Cooperative storage is distributed across participating threads; it can remove a threadgroup-memory round trip, but layout is chosen by the operation and all required threads must participate. For custom-format dequantization, Apple's recommended options are a threadgroup inline tensor or dequantizing into a cooperative tensor. The current scales plane accepts only FP8 UE8M0 and has block factors `[32, 1]`, so it cannot encode arbitrary GGUF FP16 scales/minima exactly. Int4/FP4/FP8 formats also carry stricter row and 128-byte buffer/slice alignment requirements in the installed SDK. Our cooperative-input prototype used the compiler-accepted single-SIMD-group mode and a 32x32 output tile. Sources: [Metal Performance Primitives programming guide](https://developer.apple.com/download/files/Metal-Performance-Primitives-Programming-Guide.pdf), [MTLTensorDataType API](https://developer.apple.com/documentation/metal/mtltensordatatype), [Running inline ML operations in a shader with Metal 4](https://developer.apple.com/documentation/metal/running-inline-ml-operations-in-a-shader-with-metal-4), and [WWDC26 Metal tensors session](https://developer.apple.com/videos/play/wwdc2026/330/).

Those native types are not a lossless, zero-copy view of Ferrum's common GGUF blocks. Q4_0 stores an arbitrary FP16 scale per 32 values and packs the low 16 logical values separately from the high 16; its nibble order must be rearranged for a native int4 operand. Q8_0 stores an arbitrary FP16 scale per 32 signed bytes. Q4_K stores a superblock FP16 scale/minimum plus per-32-value 6-bit scale/minimum fields and packed nibbles. The E8M0 plane cannot represent those scales exactly, and Q4_K also has a non-native affine offset layout. Re-encoding to E8M0 would change weights and was not used. Cooperative input tensors can preserve custom GGUF math, but still require custom unpacking and conversion.

### Rejected cooperative-input experiment

An opt-in Q4_K MPP variant decoded the exact existing BF16-rounded Q4_K values directly into an MPP cooperative right-input tensor, avoiding the existing threadgroup BF16 staging. The SDK requires cooperative inputs to use a single SIMD group, so the experiment used a 32x32 output tile and a 128-element K loop. Its correctness check used M=35, N=65, K=512 to cover both output tails; it passed with Metal API Validation and GPU Shader Validation enabled.

The model-level A/B used the pinned LFM2.5-8B-A1B Q4_K_M artifact, three pairs for each of five workloads, with the same router and GEMV settings in both runs. The table reports condition-median absolute values and paired-ratio medians. Each throughput cell is Ferrum control -> cooperative-input candidate / llama.cpp-control -> llama.cpp-candidate in tok/s, followed by `[control/llama -> candidate/llama; candidate/control]`. First-token latency is in milliseconds in the same engine order.

| Workload | Prefill tok/s C->K / L-C->L-K `[C/L->K/L; K/C]` | Cached decode tok/s C->K / L-C->L-K `[C/L->K/L; K/C]` | Complete generation tok/s C->K / L-C->L-K `[C/L->K/L; K/C]` | First-token ms C->K / L-C->L-K | IDs C=K; K=L; C=L |
|---|---:|---:|---:|---:|---|
| Short | 24.30->24.26 / 235.15->236.44 `[0.1032->0.1026; 1.000]` | 22.91->22.86 / 101.96->100.26 `[0.2247->0.2280; 0.997]` | 14.72->14.71 / 81.50->80.25 `[0.1807->0.1829; 0.999]` | 453.0->453.7 / 46.95->46.74 | 3/3; 3/3; 3/3 |
| 128-token prompt | 30.57->30.19 / 1,541.75->1,531.41 `[0.0198->0.0197; 0.986]` | 22.69->22.69 / 101.72->102.04 `[0.2223->0.2232; 1.003]` | 3.463->3.425 / 69.508->69.423 `[0.0497->0.0493; 0.988]` | 4,187.3->4,239.9 / 83.24->83.81 | 3/3; 0/3; 0/3 |
| 512-token prompt | 30.23->30.32 / 2,139.72->2,138.45 `[0.0143->0.0142; 1.003]` | 22.61->22.71 / 101.74->101.57 `[0.2222->0.2238; 1.001]` | 0.963->0.965 / 42.223->42.367 `[0.0230->0.0228; 1.003]` | 16,938.5->16,884.1 / 239.49->239.65 | 3/3; 0/3; 0/3 |
| 1,024-token prompt | 30.22->30.29 / 2,167.53->2,150.91 `[0.0139->0.0141; 1.001]` | 22.34->22.37 / 100.71->100.95 `[0.2217->0.2216; 1.002]` | 0.490->0.492 / 26.733->26.570 `[0.0183->0.0185; 1.004]` | 33,881.0->33,809.3 / 472.63->476.19 | 3/3; 3/3; 3/3 |
| Sustained decode | 22.80->24.17 / 228.01->233.11 `[0.1009->0.1038; 1.057]` | 22.38->22.74 / 93.51->101.19 `[0.2394->0.2247; 1.016]` | 20.38->21.09 / 87.95->96.16 `[0.2347->0.2191; 1.029]` | 482.9->455.4 / 48.48->47.40 | 3/3; 3/3; 3/3 |

Across the 15 pairs, Ferrum control and candidate matched all output IDs. Each path matched llama.cpp in 9/15 pairs. Across the five workload medians the candidate/control paired ratios were 1.001x prefill, 1.003x cached decode, and 1.003x complete generation; the per-workload results show that these aggregates conceal the small 128-token regression and a sustained-decode difference. This is no measurable improvement, so the prototype was removed from the runtime. The reference rates above are from the saved JSONL; they are not current-production rates because the temporary build also used Q8_0 MPP K=32.

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

| Workload | Prefill tok/s 24->8 (8/24) | Cached decode tok/s 24->8 (8/24) | Complete generation tok/s 24->8 (8/24) | First-token ms 24->8 | Candidate IDs matching saved llama.cpp |
| --- | ---: | ---: | ---: | ---: | ---: |
| Short | 22.39->22.45 tok/s (1.002x) | 21.11->21.11 tok/s (0.999x) | 13.59->13.61 tok/s (0.999x) | 491.65->490.23 ms | 3 / 3 |
| 128-token prompt | 28.35->343.80 tok/s (12.125x) | 20.94->21.03 tok/s (1.025x) | 3.21->14.96 tok/s (4.664x) | 4,515.61->372.62 ms | 0 / 3 |
| 512-token prompt | 120.38->425.54 tok/s (3.536x) | 20.87->20.62 tok/s (1.009x) | 3.37->8.59 tok/s (2.545x) | 4,253.52->1,203.48 ms | 3 / 3 |
| 1,024-token prompt | 120.49->446.95 tok/s (3.717x) | 20.68->20.66 tok/s (1.000x) | 1.83->5.50 tok/s (3.013x) | 8,498.59->2,291.36 ms | 3 / 3 |
| Sustained decode | 22.42->22.40 tok/s (0.998x) | 20.97->20.94 tok/s (1.000x) | 19.50->19.50 tok/s (1.001x) | 490.84->491.36 ms | 3 / 3 |

Rates and latency are condition medians from the saved JSONL; the ratios are medians of matched candidate/control pairs. Values are threshold 24 -> threshold 8, with rates in tok/s and first-token latency in ms. The independent matched llama.cpp matrix below carries Ferrum and llama.cpp absolute rates and their ratios.

At 128 tokens, threshold 8 changes the generated sequence relative to threshold 24, but both first diverge from the saved llama.cpp sequence at generated token 9. At 512 tokens, threshold 8 matches llama.cpp in all three runs; threshold 24 diverges after token 8. Short, 1,024-token, and sustained-decode sequences match both control and reference. The 128-token and 512-token changes arise from selecting BF16 TensorOps for smaller expert batches; no numerical tolerance was changed.

Median prefill time fell from 4.515s to 0.372s at 128 tokens, 4.253s to 1.203s at 512, and 8.498s to 2.291s at 1,024. The candidate adds 88 dispatches on each of those prompts because the segment-count and compaction passes now run for additional expert projections. Transient peak, arena high-water, and cumulative allocated bytes were unchanged in each case. At 1,024 tokens the transient peak was 240.03 MiB, arena high-water 255.87 MiB, and cumulative allocated bytes 901.25 MiB for both variants. Post-request RSS samples were also unchanged at approximately 5.30 GiB; these are samples, not peak-RSS measurements.

The expanded correctness tests cover uneven 8- and 12-route groups with output tails for Q4_K/Q5_K/Q6_K at M=25 or 37, E=3, N=65, K=512. All six TensorOps expert tests passed with Metal API Validation and GPU Shader Validation enabled, within their existing elementwise tolerances. The validation log is `lfm2.5-8b-a1b-expert-tensorops-threshold-8-validation.log`. Raw paired results and runner logs for all three threshold sweeps are `lfm2.5-8b-a1b-expert-tensorops-threshold-{24-vs-12,12-vs-8,24-vs-8}.{jsonl,run.log}` in `docs/measurements/phase7a/`.

This reduces a major LFM prefill bottleneck without changing cached decode or transient-memory use. The model remains far below the campaign's llama.cpp prefill and decode gate; no large Qwen target is unlocked.

A fresh matched Ferrum-versus-llama.cpp matrix used the default threshold of 8 and the pinned llama.cpp revision above, with three interleaved pairs and the same GGUF and token IDs. Ferrum/reference median prefill throughput was 0.247x at 128 tokens, 0.211x at 512, and 0.225x at 1,024. Median sustained decode was 0.224x; the five-workload median decode ratio was 0.225x and full-generation ratio 0.219x. Exact generated IDs matched llama.cpp in 12/15 runs: 3/3 short, 0/3 at 128, 3/3 at 512, 3/3 at 1,024, and 3/3 sustained decode. This improves LFM materially but leaves the general gate closed.

The current matched file and runner output are `lfm2.5-8b-a1b-threshold8-current.jsonl` and `lfm2.5-8b-a1b-threshold8-current-run.log`. Post-request RSS samples were 5.24–5.31 GiB for Ferrum and 4.99 GiB for llama.cpp; the runner did not capture peak RSS. A separate flushed-dispatch diagnostic profile at 1,024 tokens attributed about 1.66s of sampled GPU time to the remaining expert MPP projections: 1.13s Q4_K input, 0.316s Q4_K output, and 0.215s Q6_K output. No conventional expert projection fallback appeared in that capture. The profile is `lfm2.5-8b-a1b-threshold8-profile.jsonl` under `docs/measurements/phase7a/profiles/`; it identifies expert MPP weight staging and matmul as the next bottleneck to investigate, not production timing.

The saved current matrix reports absolute performance as well as ratios. Values are condition medians; ratios are medians of matched pair rates. Throughput is Ferrum / llama.cpp in tok/s and first-token latency is Ferrum / llama.cpp in milliseconds.

| Workload | Prefill tok/s | Cached decode tok/s | Complete generation tok/s | First-token ms | Exact IDs |
|---|---:|---:|---:|---:|---:|
| Short | 22.46 / 216.62 (0.104x) | 21.07 / 94.66 (0.223x) | 13.54 / 74.80 (0.181x) | 490.13 / 50.99 | 3/3 |
| 128-token prompt | 329.82 / 1,334.16 (0.247x) | 20.21 / 88.31 (0.229x) | 14.37 / 60.55 (0.238x) | 388.42 / 96.17 | 0/3 |
| 512-token prompt | 414.71 / 1,973.58 (0.211x) | 20.77 / 94.68 (0.220x) | 8.47 / 39.16 (0.216x) | 1,234.87 / 259.68 | 3/3 |
| 1,024-token prompt | 442.42 / 1,986.65 (0.225x) | 20.68 / 93.58 (0.221x) | 5.48 / 24.59 (0.224x) | 2,314.84 / 515.68 | 3/3 |
| Sustained decode | 22.14 / 216.51 (0.103x) | 20.96 / 94.36 (0.222x) | 19.47 / 89.00 (0.219x) | 497.13 / 51.04 | 3/3 |

## Experiment 7: M16 versus M32 grouped-expert TensorOps tiles

Status: rejected; M32 remains selected. The temporary M16 implementation changed only the expert `matmul2d` M tile and grid height for grouped Q4_K/Q5_K/Q6_K prefill. It kept the 64-column by 128-K tile, packed GGUF weights, quantization, output order, dispatch threshold, and M=1 GEMV path constant.

The current release build ran three paired M32/M16 comparisons for each of the same five LFM2.5-8B-A1B Q4_K_M workloads. Pair order alternated in one warmed Ferrum process. All 15 generated sequences matched exactly between tile sizes. Candidate/control paired median ratios were:

| Workload | Prefill tok/s M32->M16 (M16/M32) | Cached decode tok/s M32->M16 (M16/M32) | Full generation tok/s M32->M16 (M16/M32) | First-token ms M32->M16 | Exact IDs |
| --- | ---: | ---: | ---: | ---: | ---: |
| Short | 22.41->22.39 (0.999x) | 21.08->21.11 (0.998x) | 13.60->13.56 (0.997x) | 491.18->491.55 | 3 / 3 |
| 128-token prompt | 351.74->272.95 (0.780x) | 20.98->20.98 (1.001x) | 15.09->13.81 (0.915x) | 364.18->469.23 | 3 / 3 |
| 512-token prompt | 431.42->303.02 (0.702x) | 20.76->20.80 (0.993x) | 8.64->6.88 (0.796x) | 1,187.09->1,689.91 | 3 / 3 |
| 1,024-token prompt | 454.69->312.69 (0.689x) | 20.64->20.71 (1.000x) | 5.59->4.19 (0.749x) | 2,252.36->3,275.09 | 3 / 3 |
| Sustained decode | 22.42->22.39 (0.999x) | 20.90->20.95 (1.000x) | 19.47->19.50 (1.000x) | 490.89->491.49 | 3 / 3 |

Absolute rates and latency are condition medians from the saved JSONL. Ratios are medians of matched candidate/control pairs. The internal A/B file has no llama.cpp runs; the independent matched matrix above provides Ferrum/reference absolute rates and ratios.

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

| Workload | Prefill tok/s control->candidate (candidate/control) | Cached decode tok/s control->candidate (candidate/control) | Full generation tok/s control->candidate (candidate/control) | First-token ms control->candidate | Exact generated IDs |
| --- | ---: | ---: | ---: | ---: | ---: |
| Short | 793.26->652.58 (0.825x) | 127.71->127.75 (1.001x) | 108.89->105.57 (0.968x) | 26.79->32.50 | 3 / 3 |
| 128-token prompt | 3,714.86->2,460.71 (0.658x) | 128.45->127.48 (0.993x) | 104.23->93.80 (0.901x) | 34.77->52.33 | 3 / 3 |
| 512-token prompt | 5,155.34->2,991.75 (0.580x) | 121.85->120.92 (0.992x) | 71.12->54.79 (0.768x) | 99.63->171.49 | 3 / 3 |
| 1,024-token prompt | 4,951.12->2,969.38 (0.601x) | 112.73->113.89 (1.016x) | 47.89->34.51 (0.721x) | 207.15->345.22 | 3 / 3 |
| Sustained decode | 785.55->649.02 (0.826x) | 123.40->123.96 (1.005x) | 117.89->117.67 (0.998x) | 27.05->32.68 | 0 / 3 |

Absolute rates and latency are condition medians read from the internal A/B JSONL; ratios are medians of matched candidate/control pairs. The internal file has no llama.cpp runs, so the separate matched reference matrix remains the source for llama.cpp rates and Ferrum/reference ratios.

All matched pairs had identical transient prefill peaks. On sustained decode, control and candidate first differed at generated token 64 in each pair. Cached decode was flat, and the 128–1,024-token prefill cases consistently regressed, so the native int4 path was not retained. This supports keeping one-token projections on direct GEMV shaders and the current BF16-staged TensorOps path for eligible batched shapes.

Raw A/B rows, the complete runner output, and correctness output are `qwen2.5-q4_0-native-int4-ab.jsonl`, `qwen2.5-q4_0-native-int4-ab.run.log`, and `qwen2.5-q4_0-native-int4-validation.log` in `docs/measurements/phase7a/`.

### Absolute rates for archived TensorOps control/candidate runs

These values are condition medians read from the saved JSONL; none of these matrices was rerun for this journal update. The cited internal A/B files contain Ferrum control/candidate samples but no llama.cpp samples, so their existing ratios remain in the experiment text above and the independent matched reference matrices are reported separately. Throughput is tok/s; first-token latency is milliseconds.

| Experiment / workload | Prefill tok/s control->candidate | Cached decode tok/s control->candidate | Complete generation tok/s control->candidate | First-token ms control->candidate |
|---|---:|---:|---:|---:|
| Native int8, short | 400.06->517.93 | 99.07->98.12 | 77.86->82.15 | 52.80->40.85 |
| Native int8, 128 | 1,811.37->1,950.16 | 98.05->101.10 | 71.69->74.74 | 71.00->65.94 |
| Native int8, 512 | 2,615.74->2,140.01 | 90.96->90.59 | 44.06->38.89 | 196.04->239.59 |
| Native int8, 1,024 | 2,715.79->2,076.97 | 81.11->80.48 | 28.59->23.91 | 377.39->493.35 |
| Native int8, sustained decode | 378.73->457.88 | 95.67->95.40 | 90.34->90.99 | 55.76->46.20 |
| Grouped expert TensorOps, short | 22.38->21.66 | 21.09->20.99 | 13.56->13.37 | 491.88->508.06 |
| Grouped expert TensorOps, 128 | 27.96->27.86 | 20.85->20.07 | 3.17->3.14 | 4,579.01->4,594.08 |
| Grouped expert TensorOps, 512 | 27.95->118.69 | 20.82->20.68 | 0.89->3.31 | 18,316.16->4,314.01 |
| Grouped expert TensorOps, 1,024 | 29.38->119.36 | 20.55->20.67 | 0.48->1.81 | 34,858.51->8,579.60 |
| Grouped expert TensorOps, sustained decode | 23.71->23.42 | 22.35->21.61 | 20.38->19.96 | 464.24->470.00 |
| Expert threshold 24->8, short | 22.39->22.45 | 21.11->21.11 | 13.59->13.61 | 491.65->490.23 |
| Expert threshold 24->8, 128 | 28.35->343.80 | 20.94->21.03 | 3.21->14.96 | 4,515.61->372.62 |
| Expert threshold 24->8, 512 | 120.38->425.54 | 20.87->20.62 | 3.37->8.59 | 4,253.52->1,203.48 |
| Expert threshold 24->8, 1,024 | 120.49->446.95 | 20.68->20.66 | 1.83->5.50 | 8,498.59->2,291.36 |
| Expert threshold 24->8, sustained decode | 22.42->22.40 | 20.97->20.94 | 19.50->19.50 | 490.84->491.36 |
| Expert tile M32->M16, short | 22.41->22.39 | 21.08->21.11 | 13.60->13.56 | 491.18->491.55 |
| Expert tile M32->M16, 128 | 351.74->272.95 | 20.98->20.98 | 15.09->13.81 | 364.18->469.23 |
| Expert tile M32->M16, 512 | 431.42->303.02 | 20.76->20.80 | 8.64->6.88 | 1,187.09->1,689.91 |
| Expert tile M32->M16, 1,024 | 454.69->312.69 | 20.64->20.71 | 5.59->4.19 | 2,252.36->3,275.09 |
| Expert tile M32->M16, sustained decode | 22.42->22.39 | 20.90->20.95 | 19.47->19.50 | 490.89->491.49 |
| Expert tile M32->M64, short | 23.81->23.84 | 22.49->22.45 | 14.45->14.45 | 462.27->461.77 |
| Expert tile M32->M64, 128 | 365.74->393.37 | 22.41->22.41 | 15.93->16.30 | 350.34->325.68 |
| Expert tile M32->M64, 512 | 446.21->473.75 | 22.16->22.26 | 9.08->9.43 | 1,147.73->1,081.01 |
| Expert tile M32->M64, 1,024 | 473.17->527.28 | 22.03->22.06 | 5.85->6.34 | 2,164.48->1,942.33 |
| Expert tile M32->M64, sustained decode | 22.20->22.93 | 21.86->22.34 | 19.96->20.30 | 495.69->480.00 |
| Native int4, short | 793.26->652.58 | 127.71->127.75 | 108.89->105.57 | 26.79->32.50 |
| Native int4, 128 | 3,714.86->2,460.71 | 128.45->127.48 | 104.23->93.80 | 34.77->52.33 |
| Native int4, 512 | 5,155.34->2,991.75 | 121.85->120.92 | 71.12->54.79 | 99.63->171.49 |
| Native int4, 1,024 | 4,951.12->2,969.38 | 112.73->113.89 | 47.89->34.51 | 207.15->345.22 |
| Native int4, sustained decode | 785.55->649.02 | 123.40->123.96 | 117.89->117.67 | 27.05->32.68 |

## Experiment 10: Four-row-per-SIMD Q5_0 M=1 GEMV

Status: retained as the default for Q5_0 M=1 projections with at least 128 output rows. The direct MSL kernel follows llama.cpp's Q5_0 row organization: four adjacent output rows per SIMD group, two SIMD groups per 64-thread threadgroup. Each SIMD group loads the activation fragment once and reuses it across those four rows. Dispatch remains format-, batch-, and shape-based; this is not a TensorOps path and does not use model identity.

The A/B used the pinned Qwen2.5-0.5B Q4_K_M artifact and the same llama.cpp revision, prompt IDs, greedy policy, and three interleaved control/candidate pairs for each of the five workloads. The candidate improved paired-median cached-decode throughput by 9.8–12.4% across the five cases. Absolute Ferrum control/candidate decode medians were 111.5→122.8 tok/s (short), 112.1→123.3 (128-token prompt), 107.7→119.3 (512), 98.8→110.7 (1,024), and 106.7→118.7 (sustained decode). Complete-generation throughput rose from 96.2→103.4, 92.4→100.7, 63.8→68.4, 41.7→44.2, and 102.6→113.9 tok/s respectively. Prefill stayed near the control rates; first-token latency changed from 30.8→31.0, 39.7→38.8, 113.0→112.2, 238.7→236.3, and 31.7→31.4 ms.

The candidate and control produced identical IDs for all three runs of the short, 128-, 512-, and 1,024-token workloads. Sustained decode diverged from control at generated token 39 in all three pairs; it matched llama.cpp through token 14 and first diverged at token 15. This output divergence is retained in the raw records; the numerical tolerance was not changed. The Q5_0 scalar-reference test passed for an output-tail shape (N=131, K=512) and a production-sized wide projection (N=4,864, K=896), with Metal API Validation and GPU Shader Validation enabled. The same test verifies the N>=128 dispatch boundary. Its log is `qwen2.5-q5_0-n4-gemv-validation.log`.

The release A/B records are `qwen2.5-q4_k_m-q5n4-ab.jsonl` and `qwen2.5-q4_k_m-q5n4-ab.run.log`. The full absolute Ferrum/control, candidate, and llama.cpp rates, their ratios, and first-token latencies are in the JSONL-derived matrix below. The sustained decode gain leaves Qwen2.5 Q4_K_M at 0.574x of llama.cpp decode throughput; this kernel does not change the Phase 7 gate status.

## Absolute performance record (JSONL-derived)

This section applies the reporting format requested for Phase 7: absolute Ferrum and llama.cpp throughput is shown beside each ratio, and first-token latency is shown in milliseconds. Values are computed from the recorded JSONL files; no benchmark was rerun just to change presentation. Throughput and latency values are medians across the repetitions stated for each experiment. Ratios are medians of matched per-pair ratios; internal candidate/control ratios use recorded pair IDs. Candidate/llama.cpp ratios use matched pairs when the llama.cpp rows share the candidate A/B matrix; otherwise they are ratios of displayed condition medians from the named reference matrix. `Prefill`, `cached decode`, and `complete generation` are kept distinct.

### Historical matched baselines (llama.cpp `9710a321`)

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

Before Experiment 18 changed the measured row handoff, Q8_0 prompt GEMM dispatched to Metal 4 MPP `matmul2d` for BF16 inputs when `M >= 16` and K was divisible by 128. The existing kernel stages a 64x128 BF16-dequantized weight tile (16 KiB) and reuses it across 64 prompt rows. This candidate stages a 64x64 tile (8 KiB), invokes the same cooperative `matmul2d` operation twice as often along K, and keeps the quantized weights packed in model storage. M=1 remains on the direct Q8_0 GEMV shader; this is MPP on the M5 GPU, not the standalone Neural Engine or Core ML.

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


## Experiment 15: Q6_K sixteen-row M=1 direct GEMV

Status: rejected; the temporary direct-MSL kernel, selector, correctness test, and benchmark toggle were removed. The existing Q6_K M=1 GEMV remains in production. This experiment tested activation reuse across four output rows per SIMD group, for 16 output rows per threadgroup. It is a conventional direct shader, not Metal 4 TensorOps and not the standalone Apple Neural Engine/Core ML. Since this is M=1, the experiment deliberately evaluated a direct GEMV rather than forcing a cooperative-tensor path onto a one-row workload.

The focused correctness case used an uneven N=133 output tail and K=512 and passed with Metal API Validation and GPU Shader Validation enabled; its log is `qwen2.5-q6_k-16rows-validation.log`. The Qwen2.5-0.5B Q6_K artifact SHA-256 is `2f82233630c349ccf6b8daccf48f9a7865713d9f08a2eadfa456cebe9b97c7f5`. The three-pair exploratory matrix is `qwen2.5-q6_k-q6k16rows-ab.jsonl` and `.run.log`; the five-pair decision matrix is `qwen2.5-q6_k-q6k16rows-ab-5pair.jsonl` and `.run.log`. The decision matrix has five interleaved control/candidate/llama.cpp pairs for each of short, 128-, 512-, and 1,024-token prompts and sustained 128-token decode.

Each throughput cell shows Ferrum control → 16-row candidate / llama.cpp in tok/s, followed by `[control/llama → candidate/llama; candidate/control]`. Ratios are medians of paired ratios; first-token latency is in milliseconds. `C=K` and `K=L` count full generated-ID matches in the five pairs.

| Workload | Prefill tok/s | Cached decode tok/s | Complete-generation tok/s | First-token latency ms | IDs |
|---|---:|---:|---:|---:|---|
| Short | 638.4→635.3 / 1,456.8 [0.436→0.436; 0.992] | 129.9→130.1 / 203.3 [0.642→0.640; 0.996] | 106.2→106.5 / 176.8 [0.601→0.602; 1.001] | 33.2→33.5 / 14.7 | C=K 5/5; K=L 5/5 |
| 128 prompt | 3,018.8→2,995.5 / 7,114.3 [0.425→0.421; 1.001] | 129.5→131.4 / 203.1 [0.641→0.647; 1.008] | 100.4→100.9 / 170.8 [0.588→0.592; 1.008] | 42.7→43.0 / 18.2 | C=K 5/5; K=L 5/5 |
| 512 prompt | 4,687.4→4,704.5 / 9,629.0 [0.485→0.490; 1.010] | 124.9→124.9 / 202.4 [0.620→0.618; 0.999] | 69.9→70.3 / 125.6 [0.558→0.561; 1.007] | 109.6→109.2 / 53.4 | C=K 5/5; K=L 5/5 |
| 1,024 prompt | 4,592.7→4,595.5 / 9,051.4 [0.509→0.508; 1.002] | 115.4→115.6 / 198.0 [0.580→0.581; 1.002] | 46.2→46.2 / 86.2 [0.536→0.536; 1.004] | 223.3→223.2 / 113.4 | C=K 5/5; K=L 5/5 |
| Sustained decode | 628.5→627.4 / 1,447.5 [0.436→0.433; 0.988] | 125.4→126.6 / 210.7 [0.594→0.601; 1.011] | 119.1→120.1 / 194.7 [0.612→0.616; 1.010] | 33.8→33.8 / 14.8 | C=K 5/5; K=L 0/5 |

Across all 25 paired workload runs, the median candidate/control ratios were 1.002 for prefill, 1.006 for cached decode, and 1.006 for complete generation. Individual paired ratios varied from 0.776 to 1.141 for prefill, 0.988 to 1.028 for decode, and 0.938 to 1.065 for generation; one 128-prompt prefill timing was a clear outlier. The small median decode/generation changes and flat first-token latency do not establish a repeatable improvement. Ferrum control and candidate produced identical token IDs in all 25 pairs. The 16-row shader was therefore removed, with its raw matrices and correctness log retained as rejected-experiment evidence.


## Experiment 16: Q6_K MPP K=64 tile for long prefill

Status: rejected; Q6_K projections retain the K=128 Metal 4 MPP kernel. The candidate halved the staged BF16 weight tile from 64x128 (16 KiB) to 64x64 (8 KiB), with twice as many `matmul2d` calls along K. Unlike native int4/int8 TensorOps operands, Q6_K's packed values and per-subblock scales require dequantization into the bounded BF16 tile. This was MPP/TensorOps work on the M5 GPU's Neural Accelerators, not the standalone Apple Neural Engine or Core ML. M=1 remained on its direct GEMV path.

The focused reference case used M=515, N=65, K=256, covering an awkward output dimension and batch tail across multiple K64 tiles. It passed scalar-reference comparison with Metal API Validation and GPU Shader Validation enabled; the log is `qwen2.5-q6_k-q6k64-validation.log`. The Qwen2.5-0.5B Q6_K five-pair matrix used the same artifact, prompt IDs, generation settings, and llama.cpp reference as the Q6_K baseline. The matrix and runner output are `qwen2.5-q6_k-q6k64-m512-ab.jsonl` and `.run.log`. Candidate K64 was selected only when the projection batch had at least 512 rows; short and 128-row prompts therefore remained on K=128 in both variants.

Each throughput cell shows Ferrum control → K64 candidate / llama.cpp in tok/s, followed by `[control/llama → candidate/llama; candidate/control]`. Ratios are medians of paired ratios; first-token latency is in milliseconds. `C=K` and `K=L` count full generated-ID matches across five pairs.

| Workload | Prefill tok/s | Cached decode tok/s | Complete-generation tok/s | First-token latency ms | IDs |
|---|---:|---:|---:|---:|---|
| Short | 639.7→632.6 / 1,481.1 [0.429→0.428; 0.996] | 129.1→128.6 / 202.4 [0.638→0.639; 1.001] | 106.1→105.5 / 174.4 [0.608→0.601; 0.997] | 33.1→33.6 / 14.4 | C=K 5/5; K=L 5/5 |
| 128 prompt | 3,020.4→2,992.9 / 7,051.3 [0.430→0.423; 0.986] | 130.8→129.6 / 202.2 [0.647→0.637; 0.985] | 100.5→100.2 / 170.1 [0.590→0.585; 0.997] | 42.7→43.1 / 18.4 | C=K 5/5; K=L 5/5 |
| 512 prompt | 4,693.9→4,363.5 / 9,659.6 [0.486→0.452; 0.928] | 125.3→124.0 / 200.5 [0.625→0.620; 0.990] | 70.0→67.8 / 125.5 [0.561→0.540; 0.963] | 109.4→117.6 / 53.2 | C=K 5/5; K=L 5/5 |
| 1,024 prompt | 4,588.6→4,377.7 / 9,094.7 [0.504→0.482; 0.954] | 114.9→115.2 / 200.3 [0.573→0.576; 0.999] | 46.2→44.9 / 86.9 [0.532→0.517; 0.969] | 223.5→234.2 / 112.8 | C=K 5/5; K=L 5/5 |
| Sustained decode | 620.7→626.2 / 1,424.6 [0.430→0.440; 1.000] | 125.8→125.5 / 209.4 [0.601→0.598; 0.997] | 119.3→119.1 / 194.7 [0.613→0.613; 1.000] | 34.2→33.8 / 15.0 | C=K 5/5; K=L 0/5 |

Across all 25 paired runs, the median candidate/control ratios were 0.987 for prefill, 0.997 for cached decode, and 0.997 for complete generation. At the targeted 512- and 1,024-row prefills, absolute throughput fell by 330.4 and 210.9 tok/s, respectively; first-token latency increased by 8.2 and 10.7 ms. Ferrum candidate IDs matched control in all 25 pairs. The lower K tile did not repay the extra MPP calls, so the kernel, selector, test, and benchmark toggle were removed; the correctness log and raw timing matrix remain as rejected-experiment evidence.

## Experiment 17: Q5_1 MPP K=64 tile for long prefill

Status: retained for quantized Q5_1 projection batches with `M >= 512`; smaller batches keep the established K=128 tile. The profile-first target was Qwen2.5-0.5B Q5_K_M at 1,024 prompt rows. Its flushed-dispatch profile attributed 44.52 ms to `gate_proj.q5_1_gemm_mpp` (24 calls), 42.49 ms to `up_proj.q5_1_gemm_mpp` (24 calls), 33.14 ms to `down_proj.q5_k_gemm_mpp` (12 calls), 28.51 ms to `attention_softmax` (24 calls), and 22.31 ms to `down_proj.q6_k_gemm_mpp` (12 calls). The profile is diagnostic, not production timing.

The candidate decodes the existing packed Q5_1 blocks exactly into a 64x64 BF16 tile, then uses Metal 4 `matmul2d`. It halves the staged tile footprint from 16 KiB to 8 KiB versus K=128 and doubles the matmul calls per full K span. This is still MPP/TensorOps on the M5 GPU; it is not native int4 input and does not use the standalone Apple Neural Engine or Core ML. The Q5_1 M=1 GEMV selector is unchanged.

The output-tail and K-tile correctness case used M=515, N=65, K=256 and passed scalar-reference comparison. Validation ran with Metal API Validation and GPU Shader Validation enabled; its log is `qwen2.5-q5_k_m-q5_1k64-validation.log`. The profile and its workload are `profiles/qwen2.5-q5_k_m-profile.jsonl`, `profiles/qwen2.5-q5_k_m-profile.run.log`, and `profiles/qwen2.5-q5_k_m-profile-workloads.jsonl`.

The five-pair release A/B used the pinned Qwen2.5 Q5_K_M artifact, matched prompts, the normal production dispatch settings, and current official llama.cpp `84e76d8a23162eca70490da131945ebec1f09bf4`. For short and 128-row prompts, K=64 is not selected. At both selected long-prompt shapes all five pairs improved prefill and complete-generation throughput; cached decode stayed flat. Control/candidate Ferrum IDs matched in all 25 pairs. The raw matrix includes control, candidate, and matched llama.cpp rows; every reference row records its commit. See `qwen2.5-q5_k_m-q5_1k64-m512-ab.jsonl` and `.run.log`.

| Workload | Prefill tok/s: control → candidate / llama.cpp [C/L → K/L; K/C] | Cached decode tok/s: control → candidate / llama.cpp [C/L → K/L; K/C] | Complete-generation tok/s: control → candidate / llama.cpp [C/L → K/L; K/C] | First-token latency ms: control → candidate / llama.cpp | IDs: C=K; K=L |
|---|---:|---:|---:|---:|---:|
| Short | 708.7 → 703.5 / 1,385.5 [0.509→0.510; 1.002] | 115.0 → 114.8 / 223.8 [0.511→0.511; 0.998] | 98.4 → 97.9 / 187.9 [0.523→0.518; 0.996] | 30.0 → 30.2 / 15.5 | 5/5; 0/5 |
| 128 prompt | 3,385.7 → 3,361.4 / 6,816.3 [0.493→0.487; 0.995] | 116.4 → 116.7 / 224.6 [0.513→0.515; 1.000] | 94.7 → 94.7 / 180.8 [0.524→0.515; 0.999] | 38.1 → 38.4 / 19.0 | 5/5; 5/5 |
| 512 prompt | 4,805.6 → 5,005.9 / 9,392.3 [0.511→0.533; 1.042] | 112.0 → 111.6 / 222.0 [0.499→0.502; 0.995] | 66.4 → 67.8 / 130.8 [0.506→0.517; 1.018] | 106.9 → 102.6 / 54.8 | 5/5; 5/5 |
| 1,024 prompt | 4,637.5 → 4,841.0 / 8,794.4 [0.526→0.551; 1.047] | 103.8 → 104.0 / 217.6 [0.477→0.476; 0.998] | 44.6 → 45.7 / 88.2 [0.504→0.517; 1.024] | 221.1 → 211.8 / 116.7 | 5/5; 5/5 |
| Sustained decode | 695.4 → 689.1 / 1,376.8 [0.508→0.499; 0.987] | 111.8 → 111.9 / 229.3 [0.488→0.488; 1.001] | 107.0 → 107.1 / 211.9 [0.504→0.506; 0.999] | 30.5 → 30.8 / 15.5 | 5/5; 0/5 |

At the selected 512-row shape, median prefill increased 200.3 tok/s (4.2%) and complete generation increased 1.4 tok/s (1.8%); first-token latency fell 4.3 ms. At 1,024 rows, prefill increased 203.5 tok/s (4.7%) and complete generation increased 1.1 tok/s (2.4%); first-token latency fell 9.3 ms. Cached-decode throughput changed by less than 0.3%. Across all 25 paired cases, median candidate/control ratios were 1.0045 for prefill, 0.9988 for cached decode, and 1.0009 for complete generation because only the two long-prefill shapes dispatch K=64. This is a retained shape-specific TensorOps win, not a gate-level improvement; current-upstream Qwen2.5 Q5_K_M remains at 0.533x prefill and 0.502x cached decode at 512 rows, and 0.551x prefill and 0.476x decode at 1,024 rows.


## Experiment 18: Format-specific small-batch GGUF MPP handoff

Status: retained with measured MPP boundaries of M>=4 for Q4_0 and Q4_K, M>=8 for Q5_K and Q6_K, and M>=12 for Q8_0. Q5_0 and Q5_1 remain at M>=16 because their small-batch production crossover has not been measured; MLX affine-Q4 also remains at M>=16. M=1 always uses the direct GEMV shader, including when the benchmark override is set below 16.

This experiment targeted the projection GEMMs that still dispatched to conventional MSL at M=2..15. The candidate reuses the existing Metal 4 MPP projection kernels and their bounded custom dequantization into BF16 tiles; it does not add a new native-int4/int8 TensorOps operand or alter GGUF block decoding. MPP eligibility still requires BF16 activations, K aligned to 128, and an MPP-capable Metal 4 GPU. The Q8_0 long-prefill K=64 choice remains separately guarded at M>=512.

The boundary search interleaved five control/candidate/llama.cpp pairs per shape on the M5. Each engine loaded the same GGUF and received the same exact prompt token IDs with 16 generated tokens; llama.cpp rows identify current upstream `84e76d8a23162eca70490da131945ebec1f09bf4`. Small-M workloads use deliberately truncated prompt-token prefixes to probe projection row counts. These runs compare dispatch and throughput, not language quality. Output-ID match counts remain visible, and the separate scalar-reference/tail tests ran with Metal API Validation and GPU Shader Validation enabled. Do not interpret a changed token sequence from these short prefixes as a quality result.

At M=4, five-pair A/Bs retained the MPP path for Q4_0 and Q4_K: all five pairs improved prefill, cached decode, and complete-generation rates, while first-token latency fell. Q5_K and Q6_K regressed at M=4 and remain on MSL through M=7. Each cell below shows Ferrum control -> candidate / llama.cpp in tok/s, followed by `[control/llama -> candidate/llama; candidate/control]`; latency is control -> candidate / llama.cpp in ms. Ratios are medians of matched per-pair ratios; throughput and latency are medians of each condition.

| Model | Prefill tok/s C->K / llama [C/L->K/L; K/C] | Cached decode tok/s C->K / llama [C/L->K/L; K/C] | Complete generation tok/s C->K / llama [C/L->K/L; K/C] | First-token ms C->K / llama | C=K |
|---|---:|---:|---:|---:|---:|
| Qwen2.5 Q4_0 | 157.9->163.2 / 769.6 [0.205->0.211; 1.028] | 130.4->133.1 / 263.6 [0.499->0.505; 1.015] | 110.8->113.2 / 241.5 [0.463->0.469; 1.020] | 25.8->24.8 / 5.5 | 0/5 |
| Qwen2.5 Q4_K_M | 147.5->150.1 / 670.0 [0.219->0.222; 1.017] | 124.6->125.5 / 237.0 [0.526->0.533; 1.008] | 105.6->106.5 / 218.3 [0.484->0.490; 1.012] | 27.4->27.0 / 6.2 | 0/5 |
| Qwen2.5 Q5_K_M | 149.0->143.2 / 645.3 [0.230->0.222; 0.965] | 116.6->117.2 / 229.1 [0.514->0.514; 1.006] | 100.5->100.1 / 211.0 [0.479->0.476; 0.996] | 27.2->28.3 / 6.4 | 0/5 |
| Qwen2.5 Q6_K | 168.3->128.3 / 628.8 [0.268->0.203; 0.764] | 131.9->133.7 / 205.2 [0.644->0.656; 1.020] | 113.3->108.5 / 191.2 [0.590->0.570; 0.964] | 24.1->31.5 / 6.6 | 0/5 |

For Q4_0 at M=4, median prefill rose 5.3 tok/s, cached decode rose 2.8 tok/s, complete generation rose 2.4 tok/s, and first-token latency fell 1.0 ms. For Q4_K_M, the corresponding changes were +2.6, +0.9, and +0.9 tok/s, with 0.4 ms lower first-token latency. Every paired rate improved for these two formats. Q5_K_M's prefill fell 5.8 tok/s and first-token latency rose 1.1 ms; Q6_K's prefill fell 40.0 tok/s and first-token latency rose 7.4 ms. This is why the cutoff is per format rather than one generic threshold.

The small cached-decode differences are not treated as a decode-kernel improvement: this change only dispatches prefill projections, and decode continues to use the same direct GEMV path. The retained M=4 decision is supported by the repeated prefill and end-to-end generation results and lower first-token latency.

At M=8, 12, and 15, the measured MPP wins support the M>=8 boundary for Q5_K/Q6_K and the already-selected M>=4 boundary for Q4_0/Q4_K. The same table reports absolute Ferrum and llama.cpp values for every metric alongside both ratios.

| Model / M | Prefill tok/s C->K / llama [C/L->K/L; K/C] | Cached decode tok/s C->K / llama [C/L->K/L; K/C] | Complete generation tok/s C->K / llama [C/L->K/L; K/C] | First-token ms C->K / llama | C=K |
|---|---:|---:|---:|---:|---:|
| Qwen2.5 Q4_0 / 8 | 176.5->314.0 / 1,080.4 [0.164->0.290; 1.777] | 130.0->133.1 / 265.9 [0.489->0.501; 1.026] | 96.9->112.5 / 237.0 [0.410->0.475; 1.160] | 45.6->25.8 / 7.7 | 0/5 |
| Qwen2.5 Q4_0 / 12 | 183.0->463.9 / 857.0 [0.215->0.542; 2.521] | 129.8->130.0 / 268.1 [0.486->0.484; 0.997] | 86.6->110.3 / 217.0 [0.400->0.508; 1.273] | 65.9->26.2 / 14.3 | 5/5 |
| Qwen2.5 Q4_0 / 15 | 185.4->577.0 / 1,027.6 [0.181->0.559; 3.101] | 129.2->130.1 / 268.7 [0.481->0.483; 1.003] | 79.7->109.8 / 214.8 [0.372->0.512; 1.380] | 81.2->26.3 / 14.9 | 0/5 |
| Qwen2.5 Q4_K_M / 8 | 168.2->292.9 / 895.0 [0.183->0.323; 1.737] | 123.9->124.2 / 238.4 [0.519->0.520; 1.002] | 92.7->105.1 / 210.7 [0.439->0.498; 1.134] | 47.9->27.6 / 9.2 | 0/5 |
| Qwen2.5 Q4_K_M / 12 | 175.2->428.1 / 798.9 [0.220->0.537; 2.444] | 122.7->123.1 / 237.8 [0.516->0.520; 1.003] | 82.2->104.1 / 194.4 [0.421->0.538; 1.267] | 68.8->28.3 / 15.3 | 0/5 |
| Qwen2.5 Q4_K_M / 15 | 175.4->534.2 / 980.5 [0.179->0.542; 3.035] | 122.0->123.6 / 237.5 [0.514->0.520; 1.009] | 75.4->104.3 / 194.0 [0.389->0.536; 1.380] | 85.8->28.4 / 15.6 | 5/5 |
| Qwen2.5 Q5_K_M / 8 | 169.2->274.8 / 899.2 [0.188->0.303; 1.622] | 117.9->116.5 / 226.6 [0.520->0.512; 0.988] | 89.6->98.8 / 202.8 [0.442->0.487; 1.102] | 47.6->29.4 / 9.2 | 0/5 |
| Qwen2.5 Q5_K_M / 12 | 177.0->410.9 / 775.1 [0.227->0.527; 2.322] | 116.9->115.7 / 228.1 [0.514->0.508; 0.991] | 80.0->98.3 / 188.0 [0.425->0.523; 1.229] | 68.1->29.5 / 15.8 | 0/5 |
| Qwen2.5 Q5_K_M / 15 | 177.2->508.9 / 967.4 [0.183->0.524; 2.860] | 116.9->116.1 / 227.8 [0.513->0.509; 0.995] | 73.7->98.4 / 187.5 [0.393->0.524; 1.333] | 85.0->29.8 / 15.7 | 5/5 |
| Qwen2.5 Q6_K / 8 | 196.6->251.1 / 1,018.2 [0.193->0.247; 1.277] | 130.0->130.9 / 206.9 [0.629->0.633; 1.008] | 99.9->106.3 / 188.4 [0.532->0.565; 1.063] | 41.0->32.2 / 8.1 | 0/5 |
| Qwen2.5 Q6_K / 12 | 208.5->366.1 / 807.1 [0.257->0.453; 1.755] | 131.6->128.9 / 206.3 [0.637->0.625; 0.981] | 91.2->104.6 / 175.3 [0.521->0.597; 1.151] | 57.9->33.1 / 15.1 | 5/5 |
| Qwen2.5 Q6_K / 15 | 210.2->458.0 / 1,026.2 [0.205->0.444; 2.181] | 131.2->130.8 / 207.2 [0.633->0.629; 0.993] | 84.4->105.5 / 175.3 [0.481->0.602; 1.251] | 71.7->33.1 / 14.9 | 5/5 |

Q8_0 did not cross over at M=8: an earlier M>=8 candidate was effectively flat on Qwen2.5 and slower on Qwen3. A fresh five-pair control16/candidate12 run confirms M=8 stays direct and M=12 benefits on both model families. The table includes M=8 as a neighboring-shape control; for M=12 and 15 the candidate selects MPP.

| Model / M | Prefill tok/s C->K / llama [C/L->K/L; K/C] | Cached decode tok/s C->K / llama [C/L->K/L; K/C] | Complete generation tok/s C->K / llama [C/L->K/L; K/C] | First-token ms C->K / llama | C=K |
|---|---:|---:|---:|---:|---:|
| Qwen2.5 Q8_0 / 8 | 216.5->216.6 / 1,050.6 [0.205->0.207; 0.994] | 132.9->133.0 / 200.1 [0.665->0.665; 1.002] | 104.1->104.2 / 184.8 [0.563->0.564; 1.001] | 37.3->37.2 / 7.9 | 5/5 |
| Qwen2.5 Q8_0 / 12 | 230.3->321.8 / 828.6 [0.278->0.388; 1.400] | 133.6->132.4 / 199.2 [0.671->0.668; 0.991] | 95.1->103.7 / 170.7 [0.557->0.609; 1.091] | 52.4->37.6 / 14.7 | 0/5 |
| Qwen2.5 Q8_0 / 15 | 233.5->400.3 / 1,016.8 [0.230->0.395; 1.721] | 133.7->134.3 / 199.6 [0.670->0.675; 1.007] | 88.5->104.5 / 170.4 [0.520->0.615; 1.182] | 64.6->37.8 / 15.0 | 5/5 |
| Qwen3-0.6B Q8_0 / 8 | 170.9->169.2 / 909.6 [0.187->0.183; 0.988] | 109.9->109.7 / 166.9 [0.658->0.657; 0.996] | 85.4->84.8 / 156.0 [0.547->0.543; 0.993] | 47.1->47.6 / 9.1 | 5/5 |
| Qwen3-0.6B Q8_0 / 12 | 181.2->250.8 / 688.6 [0.260->0.364; 1.397] | 109.4->108.8 / 167.3 [0.655->0.650; 0.995] | 77.2->84.4 / 144.4 [0.535->0.585; 1.094] | 66.5->48.2 / 17.7 | 5/5 |
| Qwen3-0.6B Q8_0 / 15 | 186.3->312.6 / 838.5 [0.221->0.373; 1.687] | 108.5->108.4 / 166.8 [0.651->0.651; 0.999] | 71.8->84.1 / 143.1 [0.501->0.588; 1.174] | 80.8->48.3 / 18.0 | 0/5 |

The earlier Q8_0 M>=8 candidate was rejected at M=8. These rows show the actual rates; Qwen2.5 was effectively flat, while Qwen3 was slower across prefill, decode, and generation.

| Model / M | Prefill tok/s C->K / llama [C/L->K/L; K/C] | Cached decode tok/s C->K / llama [C/L->K/L; K/C] | Complete generation tok/s C->K / llama [C/L->K/L; K/C] | First-token ms C->K / llama | C=K |
|---|---:|---:|---:|---:|---:|
| Qwen2.5 Q8_0 / 8, earlier cutoff | 216.9->217.7 / 1,083.6 [0.200->0.197; 1.008] | 133.0->134.2 / 199.7 [0.667->0.671; 1.005] | 104.4->105.0 / 185.4 [0.564->0.566; 1.006] | 37.2->37.1 / 7.6 | 5/5 |
| Qwen3-0.6B Q8_0 / 8, earlier cutoff | 171.3->168.7 / 877.0 [0.192->0.194; 0.982] | 109.8->108.2 / 166.5 [0.660->0.651; 0.986] | 85.1->84.1 / 155.4 [0.549->0.541; 0.985] | 47.0->47.7 / 9.4 | 5/5 |

The lower M=2 boundary was rejected even for Q4_0: prefill fell from 116.6 to 83.9 tok/s, cached decode from 130.9 to 130.0 tok/s, and complete generation from 117.6 to 111.7 tok/s; first-token latency rose from 17.5 to 24.1 ms. The corresponding llama.cpp values were 425.0, 269.6, and 250.0 tok/s, with 5.0 ms first-token latency. Ratios were 0.275->0.198x prefill versus llama.cpp (0.723x candidate/control), 0.485->0.484x cached decode (0.998x), and 0.470->0.449x complete generation (0.952x). Thus M=2 and M=3 retain conventional MSL. At M=4, Q4_0 and Q4_K use the measured MPP path, while Q5_K and Q6_K remain on MSL.

The four-format M=4 matrices are `qwen2.5-q4_0-small-batch-mpp-m4-ab.jsonl`, `qwen2.5-q4_k_m-small-batch-mpp-m4-ab.jsonl`, `qwen2.5-q5_k_m-small-batch-mpp-m4-ab.jsonl`, and `qwen2.5-q6_k-small-batch-mpp-m4-ab.jsonl`, with matching `-m4-workload.jsonl` files and `.run.log` outputs. The eight-shape threshold-8 matrices are the corresponding `qwen2.5-*-small-batch-mpp-m8-ab.jsonl` files plus `qwen3-0.6b-q8_0-small-batch-mpp-m8-ab.jsonl`; the Q8_0 threshold-12 refreshes are `qwen2.5-q8_0-small-batch-mpp-m12-ab.jsonl` and `qwen3-0.6b-q8_0-small-batch-mpp-m12-ab.jsonl`. The M=2 threshold-2 exploratory matrix is `qwen2.5-q4_0-small-batch-mpp-ab.jsonl`. The validation log `small-batch-mpp-validation.log` records nine passing MPP tests with Metal API and GPU Shader Validation enabled, including row/output tails and the Q8_0 M=11 direct-MSL / M=12 MPP boundary. M=1 remains covered by the existing direct GEMV tests and the explicit Q4_0 override check.

The production change is limited to measured dispatch boundaries; the small-batch A/Bs do not close the general Phase 7 gate. In particular, Qwen2.5 Q5_K_M at M=12 reaches 0.527x llama.cpp prefill and 0.523x complete generation, while its cached decode is 0.508x; Qwen2.5 Q6_K at M=12 reaches 0.453x prefill, 0.597x decode, and 0.597x complete generation. These short-prompt improvements do not change the larger-model lock.


## Experiment 19: Prefix-bounded causal attention softmax

Status: retained as the default MSL softmax kernel for full-prefill attention shapes with M>=256. `attention_softmax_prefix` loops only over the causal prefix when computing row maximum, normalization sum, and valid probabilities, then writes exact zeros to the masked suffix. It preserves the existing score scaling and storage-rounding order. The score and context matrix products remain on their existing Metal 4 MPP/TensorOps paths; this row reduction is conventional MSL work and is not routed through `matmul2d` or the standalone Apple Neural Engine/Core ML.

The profile-first target was the 1,024-token Qwen2.5 Q5_K_M prefill, whose flushed-dispatch profile attributed 28.51 ms to 24 `attention_softmax` calls. A first M>=64 probe showed no repeatable M=128 improvement, so the retained selector requires M>=256 and `M == context_width` (zero cached offset). That rejected M=128 probe measured prefill at 3,361.7->3,396.2 / 6,983.5 tok/s `[0.482->0.487x llama.cpp; 0.998x candidate/control]`, cached decode at 116.4->116.7 / 230.6 tok/s `[0.505->0.508; 1.003]`, complete generation at 94.6->95.0 / 184.7 tok/s `[0.512->0.514; 1.001]`, and first-token latency at 38.4->38.0 / 18.6 ms; Ferrum IDs matched in 5/5 pairs. M=1, short batches, 128-token prefill, and cached decode keep the existing full-scan softmax. The explicit `attention_softmax_prefix` request field remains available for paired control/candidate runs; new Metal devices enable the shape-guarded path by default.

The correctness test compares the prefix path against the established kernel for F32, F16, and BF16 values, including M=256 tails, with exact output equality. Metal API Validation and GPU Shader Validation were enabled; the log is `attention-prefix-validation.log`. Five-pair end-to-end matrices use current upstream llama.cpp `84e76d8a23162eca70490da131945ebec1f09bf4`, the same GGUF and prompt IDs for each engine, 17 generated tokens for short/128/512/1,024 prompts, and a 129-token sustained-decode case. Values below show Ferrum control -> prefix candidate / llama.cpp in tok/s, followed by `[control/llama -> candidate/llama; candidate/control]`; first-token latency is control -> candidate / llama.cpp in milliseconds. Ratios are medians of matched per-pair ratios; rates and latency are medians of each condition.

| Model / workload | Prefill tok/s C->K / llama [C/L->K/L; K/C] | Cached decode tok/s C->K / llama [C/L->K/L; K/C] | Complete generation tok/s C->K / llama [C/L->K/L; K/C] | First-token ms C->K / llama | C=K |
|---|---:|---:|---:|---:|---:|
| Qwen2.5 Q5_K_M / Short M=21 (fallback) | 707.5->711.1 / 1,415.9 [0.498->0.502; 1.007] | 115.9->114.7 / 232.8 [0.497->0.496; 0.998] | 98.6->98.1 / 193.2 [0.510->0.510; 0.998] | 30.0->29.8 / 15.1 | 5/5 |
| Qwen2.5 Q5_K_M / 128 (fallback) | 3,391.1->3,373.0 / 6,956.3 [0.489->0.486; 0.994] | 116.7->117.2 / 231.9 [0.504->0.504; 1.004] | 95.0->94.8 / 185.4 [0.511->0.512; 0.999] | 38.0->38.2 / 18.7 | 5/5 |
| Qwen2.5 Q5_K_M / 512 | 4,955.3->5,076.4 / 9,359.2 [0.533->0.543; 1.025] | 111.2->111.4 / 231.4 [0.481->0.482; 1.001] | 67.8->68.3 / 132.6 [0.510->0.516; 1.010] | 103.7->101.2 / 55.0 | 5/5 |
| Qwen2.5 Q5_K_M / 1,024 | 4,847.5->4,966.9 / 8,812.0 [0.548->0.563; 1.025] | 103.2->103.3 / 226.0 [0.456->0.458; 1.004] | 45.9->46.5 / 88.9 [0.518->0.524; 1.012] | 211.6->206.5 / 116.4 | 5/5 |
| Qwen2.5 Q5_K_M / Sustained decode, prompt M=21 (fallback) | 694.9->696.5 / 1,376.5 [0.504->0.508; 0.993] | 112.5->112.7 / 234.1 [0.481->0.481; 1.003] | 107.7->107.8 / 217.0 [0.496->0.497; 1.001] | 30.5->30.5 / 15.5 | 5/5 |
| Qwen3-0.6B Q8_0 / Short M=21 (fallback) | 434.1->436.0 / 1,255.3 [0.347->0.341; 1.003] | 106.1->106.9 / 166.7 [0.636->0.641; 1.000] | 83.5->84.1 / 144.9 [0.573->0.576; 1.001] | 48.7->48.5 / 17.0 | 5/5 |
| Qwen3-0.6B Q8_0 / 128 (fallback) | 1,952.0->1,951.0 / 6,078.3 [0.321->0.322; 0.999] | 107.2->107.6 / 165.5 [0.649->0.652; 1.002] | 77.6->77.7 / 139.3 [0.558->0.558; 1.002] | 65.9->65.9 / 21.3 | 5/5 |
| Qwen3-0.6B Q8_0 / 512 | 3,071.0->3,123.5 / 7,099.5 [0.432->0.439; 1.019] | 93.5->93.6 / 156.8 [0.595->0.597; 1.000] | 49.6->50.1 / 95.3 [0.520->0.526; 1.012] | 167.1->164.2 / 72.4 | 5/5 |
| Qwen3-0.6B Q8_0 / 1,024 | 3,239.5->3,332.6 / 6,266.8 [0.519->0.531; 1.029] | 80.1->80.2 / 144.1 [0.554->0.557; 1.003] | 32.7->33.2 / 60.8 [0.537->0.545; 1.016] | 316.4->307.6 / 163.7 | 5/5 |
| Qwen3-0.6B Q8_0 / Sustained decode, prompt M=21 (fallback) | 422.6->422.4 / 1,237.0 [0.344->0.342; 0.994] | 103.7->103.3 / 165.1 [0.628->0.628; 1.002] | 97.9->97.6 / 156.2 [0.627->0.627; 1.000] | 50.0->50.0 / 17.3 | 5/5 |

At M=512 and M=1,024, all five paired prefill rates improved for both architectures, and all candidate/control token IDs matched. Qwen2.5 prefill increased by 121.1 and 119.4 tok/s, with first-token latency falling by 2.5 and 5.1 ms. Qwen3 prefill increased by 52.5 and 93.1 tok/s, with first-token latency falling by 2.9 and 8.8 ms. Cached-decode kernels were unchanged; their measured rates stayed effectively flat. The M=21 and M=128 rows demonstrate the neighboring-shape guard: no prefix kernel is dispatched there.

The final A/B matrices and logs are `qwen2.5-q5_k_m-attention-prefix-m256-ab.jsonl` / `.run.log` and `qwen3-0.6b-q8_0-attention-prefix-m256-ab.jsonl` / `.run.log`. The earlier M>=64 Qwen2.5 probe is preserved as `qwen2.5-q5_k_m-attention-prefix-ab.jsonl` / `.run.log`; it informed the M>=256 floor and is not the retained dispatch. This optimization is a measured long-prefill improvement, not a gate-level result: Qwen2.5 Q5_K_M at 1,024 tokens reaches 0.563x llama.cpp prefill, 0.458x cached decode, and 0.524x complete generation; Qwen3 Q8_0 reaches 0.531x prefill, 0.557x decode, and 0.545x complete generation.


## Experiment 20: Four-row Q5_1 M=1 GEMV

Status: retained by default for Q5_1 direct GEMV projections with at least 128 output rows. The new MSL kernel assigns four adjacent output rows to each SIMD group and reuses each activation fragment across those rows. It keeps M=1 on a direct shader; it does not use Metal 4 TensorOps, the standalone Apple Neural Engine, or Core ML. A device setter and the `q5_1_gemv_n4` benchmark request field select the original kernel for controls.

The Qwen2.5 Q5_K_M 1,024-token profile attributed 1.564 ms per decode step to 24 `gate_proj.q5_1_gemv` calls and 1.413 ms to 24 `up_proj.q5_1_gemv` calls. These are flushed-dispatch attribution values, not normal generation timings. This made Q5_1 one of the largest remaining direct projection costs in the M=1 path.

The scalar-reference correctness test covers N=131, K=512 and N=4,864, K=896, including an output tail and a wide production projection. It verifies the N>=128 selector and passes with Metal API Validation and GPU Shader Validation enabled; the focused log is `qwen2.5-q5_1-n4-gemv-validation.log`. After the default selector was enabled, the complete library suite passed all 84 tests with both validation layers enabled; output is `q5_1-gemv-current-library-validation.log`.

The release A/B used the pinned Qwen2.5-0.5B Q5_K_M artifact (SHA-256 `041474553fcabfc2a2d67903f9d2c2e50bd92528e670da4f33b5d0ce6e59fd55`; tokenizer revision `9217f5db79a29953eb74d5343926648285ec7e67`), exact shared prompt token IDs, five interleaved control/candidate/llama.cpp pairs, 17 generated tokens for short/128/512/1,024 prompts, and 129 tokens for sustained decode. The llama.cpp source revision is `84e76d8a23162eca70490da131945ebec1f09bf4`. Candidate and control differ only in `q5_1_gemv_n4`; all MPP settings and other direct GEMV selectors are identical. Each throughput value below is the Ferrum control -> candidate / llama.cpp median in tok/s, followed by `[control/llama -> candidate/llama; candidate/control]` using paired ratios. First-token latency is control -> candidate / llama.cpp in milliseconds.

| Workload | Prefill tok/s C->K / llama [C/L->K/L; K/C] | Cached decode tok/s C->K / llama [C/L->K/L; K/C] | Complete generation tok/s C->K / llama [C/L->K/L; K/C] | First-token ms C->K / llama | IDs C=K; K=L |
|---|---:|---:|---:|---:|---:|
| Short, M=21 | 707.4->706.2 / 1,405.7 [0.502->0.502; 1.000] | 114.4->132.9 / 224.1 [0.509->0.585; 1.162] | 98.3->110.2 / 191.7 [0.513->0.573; 1.122] | 30.0->30.1 / 15.2 | 0/5; 5/5 |
| 128 prompt | 3,367.4->3,369.5 / 6,898.9 [0.487->0.484; 0.995] | 115.8->134.6 / 222.4 [0.523->0.603; 1.152] | 93.9->105.2 / 181.5 [0.517->0.580; 1.110] | 38.3->38.3 / 18.8 | 5/5; 5/5 |
| 512 prompt | 5,050.5->5,071.1 / 9,310.3 [0.543->0.548; 1.006] | 111.4->128.1 / 227.8 [0.486->0.560; 1.151] | 68.0->73.5 / 131.1 [0.519->0.560; 1.081] | 101.7->101.3 / 55.2 | 5/5; 5/5 |
| 1,024 prompt | 4,812.7->4,965.3 / 8,804.2 [0.557->0.564; 1.000] | 103.8->118.0 / 222.2 [0.467->0.532; 1.138] | 45.0->48.8 / 88.5 [0.523->0.552; 1.054] | 213.1->206.6 / 116.6 | 5/5; 5/5 |
| Sustained decode, prompt M=21 | 691.8->693.1 / 1,367.7 [0.506->0.507; 0.999] | 111.8->128.4 / 235.7 [0.475->0.544; 1.148] | 107.3->122.1 / 215.4 [0.498->0.565; 1.138] | 30.7->30.6 / 15.6 | 0/5; 0/5 |

At the five workloads, cached decode rose by 14.2–18.8 tok/s (13.8–16.2% by paired ratios), while first-token latency stayed within 0.1 ms except for the 1,024-token prompt, where it fell 6.5 ms. Prefill was unchanged within measurement variation; the kernel only changes M=1 projections. Complete-generation throughput rose by 3.8–14.8 tok/s. The candidate matched control and llama.cpp in all five pairs at 128, 512, and 1,024 prompt tokens. For the short prompt, control first diverged from llama.cpp at generated token 6 while the candidate matched llama.cpp for all 17 tokens. In the 129-token sustained-decode workload, candidate and llama.cpp first diverged at token 42; control first diverged at token 6. The raw ID sequences remain in the paired file so these floating-point accumulation-order differences are explicit.

The five-pair matrix and runner output are `qwen2.5-q5_k_m-q5_1-n4-ab.jsonl` and `.run.log`. This improves the measured direct decode bottleneck without dispatching M=1 through TensorOps; Qwen2.5 Q5_K_M sustained decode rises from 111.8 to 128.4 tok/s against 235.7 tok/s for llama.cpp, so the Phase 7 decode gate remains unmet.


## Experiment 21: Two-row Q5_K M=1 GEMV

Status: retained by default for Q5_K direct GEMV projections with at least 128 output rows. Each SIMD group computes two adjacent output rows and reuses the loaded activations across both; four SIMD groups process eight output rows per threadgroup. This remains direct MSL for M=1, without TensorOps or Apple Neural Engine/Core ML execution.

The Qwen2.5 Q5_K_M 1,024-token decode profile attributed 0.830 ms per step to 12 `down_proj.q5_k_gemv` calls. The candidate keeps Q5_K superblocks packed in place, loads each activation value once for a pair of output rows, and applies the same eight 32-value scale/minimum groups as the existing Q5_K dot product. Its selector remains limited to M=1 and N>=128.

The scalar-reference test uses N=133, K=512, covering the output tail and two Q5_K superblocks. It verifies the default selector and its override, and passes with Metal API Validation and GPU Shader Validation enabled; output is `qwen2.5-q5_k-8rows-gemv-validation.log`.

The Qwen2.5-0.5B Q5_K_M five-pair A/B uses the same artifact, prompts, token IDs, generation lengths, and current llama.cpp revision as Experiment 20. Ferrum control and candidate differ only in `q5_k_gemv_8rows`; all 25 candidate/control output sequences matched. Cached decode improved in 24/25 pairs. The table reports control -> candidate / llama.cpp medians in tok/s, followed by `[control/llama -> candidate/llama; candidate/control]` from paired ratios. First-token latency is in milliseconds.

| Workload | Prefill tok/s C->K / llama [C/L->K/L; K/C] | Cached decode tok/s C->K / llama [C/L->K/L; K/C] | Complete generation tok/s C->K / llama [C/L->K/L; K/C] | First-token ms C->K / llama | IDs C=K; K=L |
|---|---:|---:|---:|---:|---:|
| Short, M=21 | 704.2->708.3 / 1,424.6 [0.498->0.504; 1.014] | 134.0->137.4 / 231.2 [0.585->0.594; 1.027] | 110.9->113.2 / 193.0 [0.575->0.586; 1.021] | 30.1->29.9 / 15.0 | 5/5; 5/5 |
| 128 prompt | 3,374.8->3,376.9 / 6,888.1 [0.491->0.491; 1.001] | 134.9->137.9 / 228.5 [0.595->0.606; 1.025] | 106.0->108.0 / 185.3 [0.573->0.584; 1.018] | 38.2->38.2 / 18.8 | 5/5; 5/5 |
| 512 prompt | 5,080.9->5,086.7 / 9,343.6 [0.542->0.544; 1.002] | 128.4->131.1 / 224.3 [0.572->0.584; 1.022] | 73.5->74.7 / 132.7 [0.557->0.565; 1.014] | 101.1->101.0 / 55.0 | 5/5; 5/5 |
| 1,024 prompt | 4,973.1->5,012.1 / 8,797.9 [0.563->0.567; 1.008] | 118.7->121.1 / 227.2 [0.517->0.534; 1.022] | 49.0->49.7 / 88.8 [0.550->0.558; 1.015] | 206.2->204.6 / 116.6 | 5/5; 5/5 |
| Sustained decode, prompt M=21 | 701.3->697.0 / 1,373.8 [0.504->0.512; 0.982] | 130.2->132.8 / 235.0 [0.552->0.567; 1.019] | 123.4->125.9 / 216.2 [0.570->0.581; 1.022] | 30.3->30.4 / 15.6 | 5/5; 0/5 |

The current-production matrix kept the Q5_1 four-row GEMV candidate enabled in both Ferrum variants. Cached decode rose by 2.4–3.4 tok/s by workload, with paired candidate/control ratios of 1.019–1.027x; decode improved in four of five short-prompt pairs and in all five pairs for the other workloads. Complete-generation throughput rose by 0.7–2.5 tok/s and first-token latency stayed within 1.6 ms. Prefill paired ratios were 0.982–1.014x; the Q5_K M=1 path does not change prompt GEMMs. Candidate/control IDs matched in all 25 pairs, and candidate/llama.cpp IDs matched for all five pairs at short, 128, 512, and 1,024 prompt tokens. Sustained-decode sequences first diverged from llama.cpp at generated token 43 in all five pairs.

The five-pair current-production matrix and runner output are `qwen2.5-q5_k_m-q5_k-8rows-current-ab.jsonl` and `.run.log`; all Ferrum rows record `q5_1_gemv_n4=true`. A preliminary matrix with that optimization disabled in both variants is preserved as `qwen2.5-q5_k_m-q5_k-8rows-ab.jsonl` and `.run.log`. Sustained decode with both retained direct-shader candidates rises from 130.2 to 132.8 tok/s versus 235.0 tok/s for llama.cpp, so the Phase 7 decode gate remains unmet.


## Experiment 22: Q5_K MPP K=64 tile for long prefill

Status: retained for Q5_K projection MPP at `M >= 1024`; smaller batches continue to use K=128. The profile-first target was the Qwen2.5-0.5B Q5_K_M 1,024-token prefill, where the flushed-dispatch profile attributed 33.14 ms to 12 `down_proj.q5_k_gemm_mpp` calls. The candidate keeps the GGUF Q5_K decode exact, stages a 64x64 BF16 tile instead of 64x128, and uses the existing Metal 4 `matmul2d` operation. M=1 remains on direct Q5_K GEMV; grouped Q5_K MoE experts remain on their existing K=128 path because a real-model expert A/B has not exercised that format.

The scalar-reference case uses M=1025, N=65, K=256, covering the M selector boundary, output tail, and four K=64 tiles. It passed with Metal API Validation and GPU Shader Validation enabled. The focused log is `qwen2.5-q5_k_m-q5kk64-validation.log`; the full library suite passed all 86 tests with both validation layers enabled in `q5_k_mpp_k64-current-library-validation.log`. The release benchmark was rebuilt after the selector change.

The Qwen2.5-0.5B Q5_K_M five-pair A/B used the pinned artifact (SHA-256 `041474553fcabfc2a2d67903f9d2c2e50bd92528e670da4f33b5d0ce6e59fd55`), shared prompt token IDs, and official llama.cpp `84e76d8a23162eca70490da131945ebec1f09bf4`. The A/B changed only the `q5_k_mpp_tile_k64` control; Q5_1 K=64 remained enabled in both Ferrum variants. Ratios are medians of matched pair rates, and throughput and latency are condition medians. Every row shows absolute rates as well as ratios: `[control/llama -> candidate/llama; candidate/control]`.

| Workload | Prefill tok/s C->K / llama [C/L->K/L; K/C] | Cached decode tok/s C->K / llama [C/L->K/L; K/C] | Complete generation tok/s C->K / llama [C/L->K/L; K/C] | First-token ms C->K / llama | IDs C=K; K=L |
|---|---:|---:|---:|---:|---:|
| Short, M=21 | 701.3->700.0 / 1,377.4 [0.509->0.496; 1.003] | 133.6->133.3 / 221.2 [0.603->0.596; 0.998] | 110.4->110.1 / 186.8 [0.591->0.581; 0.996] | 30.2->30.3 / 15.5 | 5/5; 5/5 |
| 128 prompt | 3,390.3->3,366.1 / 6,848.9 [0.495->0.489; 0.992] | 134.3->134.0 / 228.0 [0.592->0.585; 0.995] | 105.5->104.9 / 182.1 [0.580->0.578; 0.999] | 38.1->38.4 / 19.0 | 5/5; 5/5 |
| 512 prompt, K=128 fallback | 5,064.0->5,064.8 / 9,293.4 [0.542->0.547; 1.000] | 127.5->127.9 / 222.0 [0.574->0.576; 1.003] | 73.3->73.5 / 130.8 [0.562->0.563; 1.002] | 101.4->101.4 / 55.3 | 5/5; 5/5 |
| 1,024 prompt, K=64 | 4,967.9->5,041.4 / 8,806.3 [0.563->0.572; 1.020] | 117.7->117.6 / 218.8 [0.538->0.537; 0.998] | 48.9->49.3 / 88.0 [0.554->0.560; 1.012] | 206.5->203.4 / 116.5 | 5/5; 5/5 |
| Sustained decode, prompt M=21 | 688.1->691.2 / 1,370.2 [0.502->0.504; 0.991] | 129.0->128.8 / 231.9 [0.557->0.557; 0.998] | 122.4->122.5 / 212.2 [0.577->0.576; 1.001] | 30.8->30.7 / 15.6 | 5/5; 0/5 |

At the selected 1,024-row shape, prefill rose by 73.5 tok/s (4,967.9->5,041.4), complete generation rose by 0.4 tok/s (48.9->49.3), cached decode changed by -0.1 tok/s (117.7->117.6), and first-token latency fell 3.1 ms (206.5->203.4). The paired candidate/control ratios were 1.020x, 1.012x, and 0.998x for those three throughput metrics. The five-pair exploratory matrix that forced K=64 at M>=512 showed no repeatable benefit at 512 rows, so the production selector stays at M>=1024. Candidate and control matched all generated IDs in all 25 pairs; the Phase 7 prefill/decode gates remain unmet.

The final-selector matrix and runner output are `qwen2.5-q5_k_m-q5kk64-m1024-ab.jsonl` and `.run.log`. The forced M>=512 exploration is preserved in `qwen2.5-q5_k_m-q5kk64-m512-ab.jsonl` and `.run.log`; no benchmark was rerun to prepare this journal summary.


## Experiment 23: Q8_0 K-split M=1 GEMV

Status: retained by default for Q8_0 M=1 projections with N>=128 and K>=768. This is a direct MSL GEMV selected by quant format and shape; it does not invoke Metal 4 TensorOps or the standalone Apple Neural Engine/Core ML. M=1 has no matrix tile to feed to TensorOps, and the measured candidate is faster than the existing direct shader on models with Q8_0 throughout their projections.

The Qwen2.5 Q5_K_M decode profile attributed 23.69 ms across 16 samples to the Q8_0 vocabulary-head GEMV, about 1.48 ms per sample. Current llama.cpp uses two output rows per tile and splits K over four SIMD groups (N_R0_Q8_0=2, N_SG_Q8_0=4). Ferrum's candidate adopts that scheduling shape: each lane processes eight adjacent quantized values, four SIMD groups reduce separate K regions, then a threadgroup reduction combines them for two output rows. Outside the candidate's N and K bounds, dispatch falls back to the existing direct Q8_0 GEMV selection, using the eight-row path only where N is divisible by eight. No dispatch rule depends on model identity.

The test uses N=129 and K=1056 to cover an odd output tail and a long K reduction, checks the selector boundary and scalar-reference values, and verifies the control path. It passed with Metal API Validation and GPU Shader Validation enabled; the focused log is q8_0_gemv_k_split-validation.log. The complete library suite also passed all 86 tests with both validation layers enabled; output is q8_0_gemv_k_split-library-validation.log.

Three five-pair release A/B matrices used exact prompt IDs, the same GGUF artifact within each comparison, and current official llama.cpp 84e76d8a23162eca70490da131945ebec1f09bf4: Qwen3-0.6B Q8_0, Qwen2.5-0.5B Q8_0, and Qwen2.5-0.5B Q5_K_M. The last model isolates the Q8_0 vocabulary head while its other projections use the model's other quant formats. Ratios are medians of matched pair rates; absolute throughput and first-token latency are condition medians. Each throughput cell shows Ferrum control -> K-split / llama.cpp, followed by [control/llama -> K-split/llama; K-split/control]. Latencies are milliseconds; IDs count exact complete generated sequences as control=K-split; K-split=llama.cpp; control=llama.cpp.

### Qwen3-0.6B Q8_0

| Workload | Prefill tok/s C->K / llama [C/L->K/L; K/C] | Cached decode tok/s C->K / llama [C/L->K/L; K/C] | Complete generation tok/s C->K / llama [C/L->K/L; K/C] | First-token ms C->K / llama | IDs C=K; K=L; C=L |
|---|---:|---:|---:|---:|---:|
| Short | 438.3->440.8 / 1,241.6 [0.353->0.355; 1.003] | 106.9->113.6 / 161.1 [0.656->0.705; 1.067] | 84.2->88.2 / 140.9 [0.595->0.622; 1.053] | 48.21->47.93 / 17.17 | 5/5; 5/5; 5/5 |
| 128 prompt | 1,976.7->1,968.2 / 6,054.1 [0.325->0.322; 0.995] | 107.4->114.5 / 162.1 [0.663->0.706; 1.068] | 78.1->81.3 / 138.2 [0.566->0.588; 1.041] | 65.05->65.33 / 21.39 | 0/5; 0/5; 0/5 |
| 512 prompt | 3,120.7->3,122.9 / 7,086.1 [0.440->0.442; 1.001] | 99.2->105.4 / 153.5 [0.645->0.684; 1.062] | 50.4->51.7 / 94.7 [0.531->0.546; 1.028] | 164.38->164.26 / 72.50 | 5/5; 5/5; 5/5 |
| 1,024 prompt | 3,337.6->3,336.7 / 6,250.6 [0.535->0.533; 1.004] | 88.9->93.4 / 141.9 [0.625->0.659; 1.053] | 33.5->34.1 / 60.7 [0.551->0.562; 1.021] | 307.13->307.19 / 164.06 | 5/5; 5/5; 5/5 |
| Sustained decode | 424.9->430.3 / 1,218.0 [0.352->0.350; 1.004] | 104.0->111.0 / 163.3 [0.637->0.679; 1.067] | 98.6->104.8 / 156.1 [0.631->0.670; 1.063] | 49.79->49.10 / 17.49 | 0/5; 0/5; 0/5 |

### Qwen2.5-0.5B Q8_0

| Workload | Prefill tok/s C->K / llama [C/L->K/L; K/C] | Cached decode tok/s C->K / llama [C/L->K/L; K/C] | Complete generation tok/s C->K / llama [C/L->K/L; K/C] | First-token ms C->K / llama | IDs C=K; K=L; C=L |
|---|---:|---:|---:|---:|---:|
| Short | 558.8->555.2 / 1,490.8 [0.374->0.371; 1.001] | 131.8->135.8 / 197.8 [0.670->0.685; 1.026] | 104.3->106.5 / 171.9 [0.610->0.614; 1.017] | 37.88->38.12 / 14.35 | 0/5; 5/5; 0/5 |
| 128 prompt | 2,736.8->2,731.2 / 7,047.2 [0.388->0.388; 0.999] | 133.1->135.7 / 196.9 [0.677->0.691; 1.020] | 99.9->101.5 / 166.1 [0.602->0.610; 1.015] | 47.07->47.16 / 18.41 | 5/5; 5/5; 5/5 |
| 512 prompt | 4,557.9->4,567.4 / 9,736.6 [0.468->0.470; 1.006] | 127.8->130.1 / 196.1 [0.655->0.664; 1.017] | 69.8->70.7 / 123.9 [0.566->0.571; 1.012] | 112.65->112.42 / 52.82 | 5/5; 5/5; 5/5 |
| 1,024 prompt | 4,530.8->4,503.5 / 9,186.0 [0.493->0.490; 0.995] | 117.8->119.6 / 193.1 [0.608->0.621; 1.015] | 46.3->46.3 / 86.2 [0.537->0.537; 1.000] | 226.33->227.72 / 111.70 | 5/5; 5/5; 5/5 |
| Sustained decode | 551.4->551.7 / 1,443.7 [0.388->0.385; 0.982] | 129.1->132.1 / 199.9 [0.643->0.661; 1.028] | 121.9->124.4 / 189.4 [0.644->0.657; 1.021] | 38.38->38.36 / 14.80 | 0/5; 0/5; 0/5 |

### Qwen2.5-0.5B Q5_K_M, with Q8_0 vocabulary head

| Workload | Prefill tok/s C->K / llama [C/L->K/L; K/C] | Cached decode tok/s C->K / llama [C/L->K/L; K/C] | Complete generation tok/s C->K / llama [C/L->K/L; K/C] | First-token ms C->K / llama | IDs C=K; K=L; C=L |
|---|---:|---:|---:|---:|---:|
| Short | 708.4->709.3 / 1,406.3 [0.507->0.504; 0.995] | 133.9->134.1 / 226.1 [0.592->0.595; 1.004] | 110.8->111.0 / 192.6 [0.577->0.579; 0.998] | 29.93->29.90 / 15.17 | 0/5; 0/5; 5/5 |
| 128 prompt | 3,356.1->3,321.5 / 6,914.5 [0.485->0.480; 0.989] | 134.2->134.2 / 229.1 [0.585->0.586; 1.002] | 106.1->105.2 / 185.5 [0.571->0.571; 0.996] | 38.44->38.83 / 18.75 | 5/5; 5/5; 5/5 |
| 512 prompt | 5,030.8->5,079.1 / 9,334.2 [0.541->0.544; 1.007] | 128.8->127.3 / 224.5 [0.576->0.555; 0.988] | 73.5->73.3 / 131.4 [0.556->0.554; 1.001] | 102.10->101.13 / 55.08 | 5/5; 5/5; 5/5 |
| 1,024 prompt | 5,010.3->4,986.5 / 8,823.0 [0.565->0.566; 1.004] | 118.1->118.6 / 222.0 [0.532->0.533; 1.002] | 49.1->49.2 / 88.9 [0.552->0.555; 1.003] | 204.67->205.68 / 116.29 | 5/5; 5/5; 5/5 |
| Sustained decode | 691.6->703.1 / 1,381.5 [0.501->0.507; 1.018] | 129.4->130.2 / 234.8 [0.552->0.553; 1.002] | 123.1->123.4 / 216.1 [0.570->0.571; 1.003] | 30.70->30.16 / 15.46 | 0/5; 0/5; 0/5 |

Across all 25 matched pairs per model, the median cached-decode candidate/control ratios were 1.064x for Qwen3 Q8_0, 1.021x for Qwen2.5 Q8_0, and 1.002x for Qwen2.5 Q5_K_M. Their corresponding absolute condition medians were 99.2->105.4, 127.8->130.1, and 128.8->127.3 tok/s on the 512-token workload; this last row is a small head-only-model regression, while the whole-matrix Q5_K_M decode result is effectively flat. Median complete-generation candidate/control ratios were 1.042x, 1.015x, and 1.001x. Prefill and first-token latency were effectively unchanged because this M=1 kernel is used during decode. Output IDs changed in some cases: across 25 pairs, control/candidate matched 15/25 for each model; candidate/llama.cpp matched 15/25, 20/25, and 15/25 respectively. The per-workload counts above preserve where those divergences occurred.

The wider Q8_0 models show repeatable decode wins: all 25 Qwen3 pairs improved cached decode (paired ratios ranged 1.045-1.081x), as did all 25 Qwen2.5 Q8_0 pairs (1.003-1.040x). The Q5_K_M model, where the Q8_0 path is chiefly the vocabulary head, is flat and does not establish a standalone end-to-end win. The shape policy is format-and-shape based rather than model specific. This is a retained M5 direct-GEMV win with a narrower measured benefit than the prefill TensorOps changes; it does not change Phase 7 gate status.

Raw A/B matrices and runner logs are qwen3-0.6b-q8_0-gemv-k-split-ab.jsonl and .run.log, qwen2.5-q8_0-gemv-k-split-ab.jsonl and .run.log, and qwen2.5-q5_k_m-q8_0-head-k-split-ab.jsonl and .run.log. The runner now accepts source matrices that identify their model with source_matrix_case_model when they do not have a model field; the fix is in tools/phase55_matched_matrix.py. No benchmark was rerun for journal formatting.


## Experiment 24: Q4_K sparse-expert direct row reuse

Status: retained by default for Q4_K expert fallbacks with at least 128 output rows and K divisible by 256. This path is selected only after the existing MPP eligibility check; grouped TensorOps prefill remains unchanged. One-token routed assignments stay on direct MSL rather than being forced through `matmul2d`.

The existing Q4_K expert fallback assigned one output row to each SIMD group and loaded the routed activation for every row. The new kernel assigns two adjacent output rows to each SIMD group, shares each activation load, and accumulates each row in the same increasing-column order as the original shader. Four SIMD groups cover eight output rows per threadgroup. Dispatch depends on quant format and shape, never model identity.

The focused correctness case uses 3 experts, 4 assignment rows, 133 output rows per expert, and K=512. It checks the output tail, assignment metadata, scalar reference values, and bit-for-bit equality against the original direct shader. It passed with Metal API Validation and GPU Shader Validation enabled in `lfm2.5-8b-a1b-expert-q4k8rows-validation.log`.

The A/B used the pinned LFM2.5-8B-A1B Q4_K_M artifact (SHA-256 `4923ec14f06b968b74d663e5949867d2d9c3bf13a20b8be1a9f9af39989b2bb0`), five interleaved pairs per workload, shared prompt IDs, greedy decoding, and llama.cpp `84e76d8a23162eca70490da131945ebec1f09bf4`. Every throughput cell reports absolute control, candidate, and llama.cpp tok/s; brackets show `[control/llama -> candidate/llama; candidate/control]`. Latencies are milliseconds. ID counts are `control=candidate; candidate=llama.cpp; control=llama.cpp`.

| Workload | Prefill tok/s C->K / llama [C/L->K/L; K/C] | Cached decode tok/s C->K / llama [C/L->K/L; K/C] | Complete generation tok/s C->K / llama [C/L->K/L; K/C] | First-token ms C->K / llama | IDs C=K; K=L; C=L |
|---|---:|---:|---:|---:|---:|
| Short | 28.3->37.1 / 236.0 [0.120->0.158; 1.308] | 22.9->28.3 / 103.2 [0.220->0.274; 1.236] | 15.5->19.6 / 82.2 [0.189->0.237; 1.263] | 389.1->297.1 / 46.8 | 5/5; 0/5; 0/5 |
| 128-token prompt | 394.4->394.5 / 1,539.8 [0.256->0.257; 1.000] | 22.8->28.2 / 103.1 [0.221->0.273; 1.236] | 16.5->19.0 / 70.2 [0.235->0.270; 1.152] | 324.8->324.8 / 83.3 | 5/5; 0/5; 0/5 |
| 512-token prompt | 473.1->482.2 / 2,148.8 [0.221->0.224; 1.020] | 22.5->27.8 / 101.4 [0.222->0.274; 1.233] | 9.5->10.4 / 42.3 [0.223->0.245; 1.091] | 1,082.5->1,062.0 / 238.5 | 5/5; 5/5; 5/5 |
| 1,024-token prompt | 506.3->504.0 / 2,166.6 [0.233->0.233; 0.998] | 22.4->27.6 / 100.9 [0.222->0.274; 1.232] | 6.2->6.5 / 26.7 [0.230->0.242; 1.049] | 2,022.9->2,031.9 / 472.8 | 5/5; 5/5; 5/5 |
| Sustained decode | 25.3->37.0 / 233.1 [0.108->0.158; 1.465] | 22.7->28.0 / 102.6 [0.222->0.273; 1.234] | 21.2->26.3 / 97.1 [0.219->0.271; 1.246] | 435.4->297.9 / 47.4 | 5/5; 0/5; 0/5 |

Control and candidate generated identical sequences in all 25 pairs. The candidate matched llama.cpp in 10/25 pairs, the same count as control. Across the 25 pairs, the median cached-decode ratio was 1.234x (22.7->28.0 tok/s condition medians); the candidate/reference decode ratio was 0.273x (28.0/102.6 tok/s). Complete-generation median paired ratio was 1.152x. Short-prompt first-token latency fell from 389.1 to 297.1 ms, while 128-token latency was unchanged and the longer cases stayed within measurement variation. The Phase 7 gate remains unmet.

An earlier pairwise-dot variant changed accumulation order and was rejected despite higher throughput because it matched control output IDs in only 5/25 pairs. Its absolute measurements and ratios are retained here as exploratory data, not production results:

| Workload | Prefill tok/s C->K / llama [C/L->K/L; K/C] | Cached decode tok/s C->K / llama [C/L->K/L; K/C] | Complete generation tok/s C->K / llama [C/L->K/L; K/C] | First-token ms C->K / llama | IDs C=K; K=L; C=L |
|---|---:|---:|---:|---:|---:|
| Short | 28.3->62.9 / 233.9 [0.121->0.269; 2.221] | 22.9->41.5 / 102.5 [0.223->0.403; 1.811] | 15.6->30.1 / 82.1 [0.190->0.366; 1.932] | 388.4->175.2 / 47.2 | 0/5; 5/5; 0/5 |
| 128-token prompt | 398.0->393.6 / 1,541.0 [0.259->0.257; 0.993] | 22.8->41.2 / 103.3 [0.221->0.400; 1.805] | 16.6->23.8 / 70.3 [0.236->0.339; 1.435] | 321.9->325.5 / 83.3 | 0/5; 0/5; 0/5 |
| 512-token prompt | 477.5->482.9 / 2,152.7 [0.221->0.224; 1.017] | 22.5->40.9 / 102.2 [0.220->0.400; 1.819] | 9.4->11.7 / 42.7 [0.222->0.273; 1.242] | 1,072.6->1,060.6 / 238.0 | 0/5; 0/5; 5/5 |
| 1,024-token prompt | 508.4->503.1 / 2,176.0 [0.234->0.233; 0.988] | 22.5->40.6 / 101.4 [0.221->0.402; 1.807] | 6.2->6.9 / 26.8 [0.231->0.260; 1.121] | 2,014.4->2,035.6 / 470.8 | 5/5; 5/5; 5/5 |
| Sustained decode | 24.8->62.6 / 234.1 [0.106->0.267; 2.535] | 22.7->41.2 / 102.5 [0.222->0.402; 1.813] | 21.1->39.0 / 97.1 [0.217->0.401; 1.843] | 443.5->176.0 / 47.2 | 0/5; 0/5; 0/5 |

Raw measurements are `lfm2.5-8b-a1b-expert-q4k8rows-ab.jsonl` and `lfm2.5-8b-a1b-expert-q4k8rows-ab.run.log`; the rejected-order run is `lfm2.5-8b-a1b-expert-q4k8rows-order-v1-ab.jsonl` and `.run.log`. The kernel and dispatch are in `src/metal/shaders/ops.metal`, `src/metal/mod.rs`, and `src/ops/mod.rs`; the focused reference test is in `src/ops/transformer.rs`.

## Experiment 25: Q4_K MPP M=128 tile for long prefill

Status: retained as the default for Q4_K MPP projections with at least 1,024 prompt rows on M5. MPP dispatch already requires Apple GPU family 10 and Metal 4; the shape rule uses quantization format and batch rows only. M=1 GEMV and shorter batches keep their existing direct or K=128/K=64 paths.

The candidate keeps Q4_K weights packed and uses the existing exact BF16 dequantization plus `matmul2d` TensorOps operation. It changes the MPP tile from 64x64x64 to 128x64x64, so each 8 KiB dequantized Q4_K tile is reused across twice as many prompt rows. K remains 64. This does not reinterpret Q4_K as Metal's native signed-int4 format; Q4_K scale/minimum metadata and nibble arrangement do not form a zero-copy native int4 tensor.

The correctness test covers M=1,025, N=65, K=256, row and output-column tails, and multiple K tiles against the existing BF16-rounded scalar reference. It passed with Metal API Validation and GPU Shader Validation enabled. The log is `q4_k_mpp_m128-validation.log` in `docs/measurements/phase7a/`.

The release A/B used current official llama.cpp `84e76d8a23162eca70490da131945ebec1f09bf4`, the same GGUF bytes and prompt token IDs per engine, greedy decoding, BF16 KV, five paired runs per Qwen workload, and five paired runs for the LFM 1,024-token workload. Ferrum control uses the established M=64/K=64 tile; candidate uses M=128/K=64. Ratios are medians of matched per-pair rates; throughput and latency are condition medians. Every rate below reports absolute Ferrum control -> candidate / llama.cpp tok/s, followed by `[control/llama -> candidate/llama; candidate/control]`; first-token latency is in milliseconds.

### Qwen2.5-0.5B Q4_K_M

Artifact SHA-256: `74a4da8c9fdbcd15bd1f6d01d621410d31c6fc00986f5eb687824e7b93d7a9db`.

| Workload | Prefill tok/s C->M128 / llama `[C/L->M128/L; M128/C]` | Cached decode tok/s C->M128 / llama `[C/L->M128/L; M128/C]` | Complete generation tok/s C->M128 / llama `[C/L->M128/L; M128/C]` | First-token ms C->M128 / llama | IDs C=M128; M128=L; C=L |
|---|---:|---:|---:|---:|---|
| Short | 741.91->743.54 / 1,395.25 `[0.5317->0.5273; 0.9961]` | 138.00->138.35 / 237.39 `[0.5750->0.5773; 1.0052]` | 114.96->114.28 / 197.53 `[0.5795->0.5783; 0.9940]` | 28.60->28.55 / 15.29 | 5/5; 0/5; 0/5 |
| 128-token prompt | 3,537.64->3,532.64 / 6,902.97 `[0.5101->0.5042; 1.0044]` | 138.49->139.23 / 240.49 `[0.5810->0.5789; 1.0055]` | 109.41->109.28 / 187.63 `[0.5829->0.5778; 0.9988]` | 36.48->36.53 / 18.80 | 5/5; 5/5; 5/5 |
| 512-token prompt | 5,020.58->5,048.77 / 9,361.91 `[0.5373->0.5392; 1.0070]` | 132.74->132.42 / 231.31 `[0.5717->0.5708; 0.9984]` | 74.47->74.88 / 132.87 `[0.5559->0.5631; 1.0058]` | 102.32->101.71 / 54.93 | 5/5; 5/5; 5/5 |
| 1,024-token prompt | 4,916.20->5,098.74 / 8,803.12 `[0.5604->0.5793; 1.0318]` | 122.01->121.85 / 234.73 `[0.5148->0.5181; 1.0026]` | 49.04->50.14 / 89.48 `[0.5467->0.5609; 1.0225]` | 208.64->201.13 / 116.58 | 5/5; 5/5; 5/5 |
| Sustained decode | 740.20->735.69 / 1,374.66 `[0.5402->0.5352; 0.9907]` | 133.42->132.96 / 245.63 `[0.5423->0.5433; 1.0013]` | 126.84->126.99 / 222.98 `[0.5683->0.5695; 1.0016]` | 28.67->28.84 / 15.53 | 5/5; 0/5; 0/5 |

### LFM2.5-8B-A1B Q4_K_M

Artifact SHA-256: `4923ec14f06b968b74d663e5949867d2d9c3bf13a20b8be1a9f9af39989b2bb0`.

| Workload | Prefill tok/s C->M128 / llama `[C/L->M128/L; M128/C]` | Cached decode tok/s C->M128 / llama `[C/L->M128/L; M128/C]` | Complete generation tok/s C->M128 / llama `[C/L->M128/L; M128/C]` | First-token ms C->M128 / llama | IDs C=M128; M128=L; C=L |
|---|---:|---:|---:|---:|---|
| 1,024-token prompt | 508.17->515.13 / 2,173.91 `[0.2336->0.2375; 1.0163]` | 27.59->27.63 / 102.77 `[0.2687->0.2685; 1.0018]` | 6.510->6.587 / 26.897 `[0.2421->0.2449; 1.0117]` | 2,015.34->1,988.15 / 471.25 | 5/5; 5/5; 5/5 |

The 1,024-row prefill improved on both architectures: Qwen2.5 rose from 4,916.20 to 5,098.74 tok/s and LFM2.5 from 508.17 to 515.13 tok/s. Complete-generation throughput rose from 49.04 to 50.14 tok/s and from 6.510 to 6.587 tok/s, respectively. First-token latency fell from 208.64 to 201.13 ms on Qwen2.5 and from 2,015.34 to 1,988.15 ms on LFM2.5. Cached decode stayed effectively flat. Ferrum control and candidate generated identical IDs in all 30 pairs; the LFM candidate matched llama.cpp in all five pairs. At 1,024 rows, transient prefill peak was unchanged at 255.25 MiB for Qwen2.5 and 240.03 MiB for LFM2.5. The Qwen2.5 and LFM2.5 raw A/Bs and logs are `qwen2.5-q4_k_m-mpp-m128-vs-m64.jsonl` / `.run.log` and `lfm2.5-8b-a1b-q4_k_m-mpp-m128-vs-m64.jsonl` / `.run.log`. This is a retained M5 prefill win; the matched llama.cpp gaps remain visible above and the general performance gate remains unmet.

## Experiment 26: Q6_K sparse-expert direct row reuse

Status: retained by default for Q6_K expert fallbacks with at least 128 output rows and K at least 256 and divisible by 256. The selector runs only after the existing grouped MPP/TensorOps eligibility check; sufficiently grouped prefill remains on TensorOps. Low-route-count expert work, including M=1, stays on direct MSL. Dispatch depends on Q6_K format and shape, not model identity.

The previous direct fallback assigned one output row to each SIMD group and reloaded the activation for each row. This kernel shares each activation fragment across two adjacent output rows, keeps each row's per-lane accumulation in increasing K order, and covers eight output rows per 128-thread group. It uses the existing Q6_K block loader and does not route M=1 work through TensorOps. The Q6 test uses 3 experts, 4 assignment rows, 133 output rows, and K=512; it checks the tail, assignment metadata, scalar reference, and exact equality to the original direct kernel. All 15 focused Q/K quantized tests passed with Metal API Validation and GPU Shader Validation enabled; the log is `lfm2.5-8b-a1b-expert-q6k8rows-validation.log`.

The real-model A/B used the pinned LFM2.5-8B-A1B Q4_K_M GGUF (SHA-256 `4923ec14f06b968b74d663e5949867d2d9c3bf13a20b8be1a9f9af39989b2bb0`), the current official llama.cpp reference `84e76d8a23162eca70490da131945ebec1f09bf4`, fixed shared prompt IDs, greedy decoding, BF16 KV, and five interleaved pairs per workload. The only Ferrum option changed was `q6_k_expert_project_8rows=false` for control and `true` for candidate. Absolute rates below are condition medians; ratios in brackets are medians of matched per-pair rates. Every rate includes absolute control, candidate, and llama.cpp tok/s, followed by `[control/llama -> candidate/llama; candidate/control]`; first-token latency is milliseconds.

| Workload | Prefill tok/s C->K / llama `[C/L->K/L; K/C]` | Cached decode tok/s C->K / llama `[C/L->K/L; K/C]` | Complete generation tok/s C->K / llama `[C/L->K/L; K/C]` | First-token ms C->K / llama | IDs C=K; K=L; C=L |
|---|---:|---:|---:|---:|---:|
| Short | 37.0->40.2 / 235.0 `[0.158->0.170; 1.087]` | 28.3->30.0 / 103.0 `[0.275->0.291; 1.060]` | 19.6->20.9 / 82.5 `[0.238->0.254; 1.066]` | 297.5->273.6 / 47.0 | 5/5; 0/5; 0/5 |
| 128-token prompt | 397.9->397.7 / 1,531.4 `[0.260->0.260; 1.000]` | 28.2->29.9 / 103.4 `[0.273->0.290; 1.060]` | 19.0->19.7 / 70.2 `[0.272->0.281; 1.035]` | 322.0->322.2 / 83.8 | 5/5; 0/5; 0/5 |
| 512-token prompt | 472.0->482.4 / 2,145.9 `[0.220->0.225; 1.014]` | 27.7->29.4 / 102.9 `[0.270->0.287; 1.062]` | 10.2->10.6 / 42.7 `[0.241->0.248; 1.029]` | 1,085.0->1,061.5 / 238.8 | 5/5; 5/5; 5/5 |
| 1,024-token prompt | 516.1->516.7 / 2,167.8 `[0.238->0.238; 1.001]` | 27.6->29.3 / 101.0 `[0.274->0.290; 1.063]` | 6.6->6.7 / 26.8 `[0.247->0.250; 1.014]` | 1,984.4->1,982.1 / 472.6 | 5/5; 5/5; 5/5 |
| Sustained decode | 31.7->40.1 / 233.2 `[0.135->0.171; 1.266]` | 28.0->29.7 / 102.5 `[0.274->0.290; 1.061]` | 26.1->27.9 / 97.0 `[0.271->0.289; 1.076]` | 347.8->274.9 / 47.4 | 5/5; 0/5; 0/5 |

Control and candidate generated identical IDs in all 25 pairs. The candidate's cached decode improved in every workload, with paired candidate/control ratios of 1.060-1.063x; the actual throughput rose from 27.6-28.3 to 29.3-30.0 tok/s in the four fixed-prompt cases and from 28.0 to 29.7 tok/s in sustained decode. Complete-generation throughput rose from 1.4% to 7.6% by paired ratio, with actual before/after rates shown above. At the same time, first-token latency changed from 297.5 to 273.6 ms for the short prompt, 322.0 to 322.2 ms at 128 tokens, 1,085.0 to 1,061.5 ms at 512, 1,984.4 to 1,982.1 ms at 1,024, and 347.8 to 274.9 ms for sustained decode. Ferrum RSS medians were 5.24-5.31 GiB for both variants versus 4.99 GiB for llama.cpp; Ferrum transient prefill peaks were unchanged between control and candidate at 4.05, 34.04, 160.05, 240.03, and 4.05 MiB respectively. The decode gap to llama.cpp remains substantial, so this measured fallback improvement does not meet or change the general gate.

Raw measurements and the run log are `lfm2.5-8b-a1b-expert-q6k8rows-ab.jsonl` and `.run.log` in `docs/measurements/phase7a/`. The implementation is in `src/metal/shaders/ops.metal`, `src/metal/mod.rs`, and `src/ops/mod.rs`; the correctness test is in `src/ops/transformer.rs`.

## Experiment 27: Register reuse in full-prefill causal softmax

Status: retained by default for full-prefill causal rows with width 256 through 1,024. The existing prefix-bounded reduction still handles shorter queries and widths above 1,024. The candidate keeps four scaled scores per thread in registers, computes each visible score's exponential once, and reuses it for the row sum and output. It preserves the existing staged input rounding, reduction order, and masked-suffix zeros. This operation is a row reduction, so it remains conventional MSL; Metal 4 TensorOps `matmul2d` is not the appropriate primitive. Decode and one-query attention remain on the existing path.

The validation test compares the candidate bit-for-bit to the existing prefix kernel for F32, F16, and BF16 at widths 256, 512, 768, and 1,024, including masked suffixes. The three attention softmax tests passed with Metal API Validation and GPU Shader Validation enabled; see `qwen2.5-q5_k_m-attention-prefix-reuse-validation.log`.

The release A/B used the pinned Qwen2.5-0.5B Q5_K_M artifact (SHA-256 `041474553fcabfc2a2d67903f9d2c2e50bd92528e670da4f33b5d0ce6e59fd55`), five interleaved control/candidate/llama.cpp pairs per workload, fixed prompt IDs, greedy decoding, BF16 KV, and llama.cpp `84e76d8a23162eca70490da131945ebec1f09bf4`. The only changed Ferrum option was `attention_softmax_prefix_reuse=false` for control and `true` for candidate. Absolute rates below are condition medians; ratios are medians of matched per-pair rates. Each cell reports Ferrum control, candidate, and llama.cpp tok/s followed by `[control/llama -> candidate/llama; candidate/control]`. First-token latency is in milliseconds.

| Workload | Prefill tok/s C->K / llama `[C/L->K/L; K/C]` | Cached decode tok/s C->K / llama `[C/L->K/L; K/C]` | Complete generation tok/s C->K / llama `[C/L->K/L; K/C]` | First-token ms C->K / llama | IDs C=K; K=L; C=L |
|---|---:|---:|---:|---:|---:|
| Short | 702.37->705.65 / 1,383.86 `[0.5058->0.5147; 1.0073]` | 134.38->133.78 / 221.54 `[0.6053->0.6047; 0.9934]` | 111.21->110.99 / 188.81 `[0.5855->0.5864; 0.9967]` | 30.20->30.06 / 15.42 | 5/5; 0/5; 0/5 |
| 128-token prompt | 3,348.44->3,392.04 / 6,945.01 `[0.4876->0.4905; 1.0020]` | 134.13->134.58 / 222.61 `[0.6026->0.6002; 1.0066]` | 105.28->105.79 / 183.01 `[0.5759->0.5782; 1.0045]` | 38.53->38.04 / 18.68 | 5/5; 5/5; 5/5 |
| 512-token prompt | 5,064.37->5,111.20 / 9,322.56 `[0.5428->0.5483; 1.0126]` | 127.24->126.98 / 220.92 `[0.5699->0.5737; 0.9961]` | 73.44->73.60 / 130.47 `[0.5627->0.5648; 1.0037]` | 101.40->100.49 / 55.16 | 5/5; 5/5; 5/5 |
| 1,024-token prompt | 5,044.79->5,095.05 / 8,823.80 `[0.5727->0.5796; 1.0089]` | 119.26->118.75 / 222.44 `[0.5362->0.5339; 0.9905]` | 49.45->49.53 / 88.21 `[0.5607->0.5629; 1.0038]` | 203.34->201.31 / 116.30 | 5/5; 5/5; 5/5 |
| Sustained decode | 694.22->698.72 / 1,375.69 `[0.5087->0.5079; 0.9848]` | 129.68->129.88 / 233.40 `[0.5564->0.5574; 1.0029]` | 123.30->123.21 / 213.29 `[0.5781->0.5772; 1.0001]` | 30.55->30.35 / 15.49 | 5/5; 0/5; 0/5 |

The candidate improved long-prefill throughput from 5,064.37 to 5,111.20 tok/s at 512 tokens and from 5,044.79 to 5,095.05 tok/s at 1,024 tokens. Paired ratios were 1.0126x and 1.0089x; all five 512-token pairs and four of five 1,024-token pairs favored the candidate. Cached decode stayed effectively flat because this dispatch is not selected for decode. Control and candidate generated identical IDs in all 25 pairs. First-token latency moved from 101.40 to 100.49 ms at 512 tokens and from 203.34 to 201.31 ms at 1,024 tokens.

A separate flushed-dispatch diagnostic profile ran one 1,024-token prefill with `FERRUM_BATCH_LIMIT=1`. Across 24 attention-softmax calls, GPU time fell from 17.173 to 13.806 ms; total sampled prefill GPU time fell from 185.117 to 181.229 ms. This is attribution data, not the end-to-end timing used above. The profile, runner log, and workload are `profiles/qwen2.5-q5_k_m-attention-prefix-reuse-1024-flush-profile.jsonl`, `.run.log`, and `profiles/qwen2.5-q5_k_m-attention-prefix-reuse-1024-flush-workload.jsonl`.

Ferrum process RSS was unchanged between variants at 0.969-0.978 GiB; transient prefill peaks were also identical at 35.31, 232.75, 255.62, 255.25, and 35.31 MiB across the five workloads. The matched llama.cpp ratios remain well below the Phase 7 gate, so large-target work stays deferred. Raw A/B rows and runner output are `qwen2.5-q5_k_m-attention-prefix-reuse-ab.jsonl` and `.run.log` in `docs/measurements/phase7a/`.

## Experiment 28: Sixteen-row Q5_K M=1 GEMV (rejected)

Status: rejected and removed from the shipping source. The experiment assigned four adjacent Q5_K outputs to each SIMD group, sharing activation loads across 16 output rows per threadgroup. It remained a direct MSL M=1 kernel; Metal 4 TensorOps and the standalone Apple Neural Engine/Core ML were not applicable to this GEMV shape. The retained eight-row direct GEMV remains the production path for eligible Q5_K projections.

A short-prompt decode profile (21 input tokens, 17 generated) attributed 0.883 ms of sampled GPU time per decode step to 12 `down_proj.q5_k_gemv` calls. That diagnostic explicitly had `q5_k_gemv_8rows=false`, so it profiled the older fallback rather than the production eight-row selector; it motivated a shape experiment but is not used as the A/B baseline. The experimental kernel passed a focused scalar-reference and output-tail check at N=133, K=512 before the performance result was evaluated.

The Qwen2.5-0.5B Q5_K_M release A/B used the pinned artifact (SHA-256 `041474553fcabfc2a2d67903f9d2c2e50bd92528e670da4f33b5d0ce6e59fd55`), five interleaved control/candidate/llama.cpp pairs for each workload, the same prompt IDs and generation lengths, greedy decoding, BF16 KV, and llama.cpp `84e76d8a23162eca70490da131945ebec1f09bf4`. Both Ferrum variants enabled the retained `q5_k_gemv_8rows`; only `q5_k_gemv_16rows` changed from false (control) to true (candidate). Throughput is the condition median, ratios are medians of matched per-pair rates, and first-token latency is milliseconds. Every workload row reports absolute rates alongside `[control/llama -> candidate/llama; candidate/control]` ratios.

| Workload | Prefill tok/s C (8-row)->K (16-row) / llama `[C/L->K/L; K/C]` | Cached decode tok/s C->K / llama `[C/L->K/L; K/C]` | Complete generation tok/s C->K / llama `[C/L->K/L; K/C]` | First-token ms C->K / llama | IDs C=K; K=L; C=L |
|---|---:|---:|---:|---:|---:|
| Short, M=21 | 703.8->712.2 / 1,412.6 `[0.502->0.504; 1.008]` | 137.9->138.1 / 222.9 `[0.616->0.619; 1.003]` | 113.3->113.3 / 190.0 `[0.595->0.597; 1.005]` | 30.25->29.79 / 15.13 | 5/5; 0/5; 0/5 |
| 128-token prompt | 3,389.0->3,358.2 / 6,951.6 `[0.487->0.481; 0.988]` | 138.8->138.5 / 227.8 `[0.609->0.612; 0.998]` | 108.0->107.7 / 182.1 `[0.593->0.590; 0.997]` | 38.07->38.42 / 18.65 | 5/5; 5/5; 5/5 |
| 512-token prompt | 5,053.1->5,076.0 / 9,313.0 `[0.538->0.543; 1.006]` | 130.1->130.4 / 226.2 `[0.573->0.571; 1.001]` | 74.0->74.2 / 130.6 `[0.565->0.567; 1.008]` | 101.67->101.19 / 55.22 | 5/5; 5/5; 5/5 |
| 1,024-token prompt | 5,040.8->5,091.3 / 8,748.7 `[0.574->0.580; 1.010]` | 120.9->121.4 / 219.1 `[0.551->0.554; 1.000]` | 49.8->50.1 / 88.0 `[0.564->0.568; 1.006]` | 203.47->201.43 / 117.27 | 5/5; 5/5; 5/5 |
| Sustained decode, prompt M=21 | 695.5->699.7 / 1,364.0 `[0.507->0.515; 1.001]` | 132.3->132.9 / 233.9 `[0.568->0.568; 1.004]` | 125.7->125.9 / 213.6 `[0.589->0.590; 1.003]` | 30.61->30.32 / 15.66 | 5/5; 0/5; 0/5 |

The 16-row candidate matched the eight-row control's output IDs in all 25 pairs. Cached decode moved only 0.998-1.004x by workload, with the actual rates changing by -0.3 to +0.6 tok/s; the candidate beat control in 15 of 25 individual comparisons. Complete-generation rates likewise changed by less than 0.6 tok/s per workload, and first-token latency did not move consistently. These are within-run variation rather than a stable gain, so the sixteen-row kernel and its selector were removed. M=1 remains on direct MSL, with no TensorOps route. The Phase 7 decode gate remains unmet.

Raw measurements and the run log are `qwen2.5-q5_k_m-q5_k-16rows-ab.jsonl` and `.run.log`; the source workload is `qwen2.5-q5_k_m-q5_k-16rows-workloads.jsonl`. The profile-first diagnostic and its workload are `profiles/qwen2.5-q5_k_m-current-decode-profile.jsonl`, `.run.log`, and `profiles/qwen2.5-q5_k_m-current-decode-profile-workload.jsonl` in `docs/measurements/phase7a/`.
