# Phase 7A GGUF performance journal

## Scope and gate

Phase 7A starts on `codex/phase7-gguf-performance`, from the verified Phase 6 checkpoint `66d721d9bfab04339ad170a426030c847f2d2028`. The first campaign establishes matched small-model baselines and tests general GGUF kernel improvements. The larger Qwen3.8 and Qwen3.6 targets remain behind the performance gate.

The gate is at least 0.85x llama.cpp decode throughput and 0.80x prefill throughput for medium and long prompts, with no routine workload below 0.70x without an isolated explanation. No measured architecture/format currently clears it.

Reporting rule: every important Ferrum-versus-llama.cpp result reports absolute Ferrum and llama.cpp prefill, cached-decode, and complete-generation tok/s beside their ratios, with first-token latency for both runtimes where available. Internal candidate/control results report the absolute before/after rates, llama.cpp context when available, ratios, and first-token latency. Ratios support the gate but do not replace throughput values. Use existing raw JSONL for journal updates; do not rerun a benchmark only to reformat its results.

Host-load preflight: run every benchmark command through `python3 scripts/phase7a_cpu_preflight.py --wait -- <benchmark command>`. The guard samples process CPU-time deltas across three snapshots two seconds apart and waits while any one process exceeds 80% of one core or total sampled CPU exceeds 220% of one core. The `phase55_bench` executable checks before model loading; the shared `tools/phase55_matched_matrix.py` runner and experiment-specific runners check before inference requests, including warmups. Do not start or report a benchmark if the preflight is blocked. On 2026-09-25 an LFM Q4_K expert K-split matrix was interrupted with a separate `engine_sim` Python process near 100% of one core; its incomplete 94-row capture is invalidated and retained only as `lfm2.5-8b-a1b-expert-q4k16rows-ksplit2-ab.invalidated-cpu-load.jsonl` and the matching `.run.log`. It is excluded from performance analysis.

## Machine and reference runtime

- Machine: Apple M5, Mac17,2, 32 GiB unified memory; macOS 27.0; Metal 4 / Apple GPU family 10.
- Ferrum: release build from this branch, native Rust and Metal kernels.
- Historical reference for the initial matrices and earlier experiments: official `ggml-org/llama.cpp` at commit `9710a32175b3b8f04636aaac4aa3cc28651b505d`. The 2026-09-24 upstream refresh at `84e76d8a23162eca70490da131945ebec1f09bf4` is now also historical. The latest official `master` fetched on 2026-09-25 is `1ab7e5ad2d4e7295c94c3b966a3e0b70fa365865`; its Metal Release build used an embedded Metal library, OpenMP off, and optional tools/tests/server off. Its build log is `llama.cpp-1ab7e5a-current-upstream-build.log`. `tools/phase55_llama_matrix.cpp` records `llama_commit` in every current reference row.
- Both runtimes use the same GGUF bytes and exact prompt token IDs per pair, greedy argmax, BF16 KV, context 4096, batch 2048, microbatch 512, four CPU threads, and all model layers on Metal. Model loading and one two-token warmup per workload are outside measured generation. Historical baseline matrices have three interleaved pairs per workload; the 2026-09-24 `84e76d8` refresh and 2026-09-25 `1ab7e5a` refresh have five interleaved pairs per workload.
- Prefill, first-token latency, cached decode, full-generation throughput, output IDs, KV allocation, and process RSS are retained separately in the JSONL records. Prompt tokenization is outside the timed path.

## Artifacts and workload coverage

| Model / format | Pinned artifact | SHA-256 | Workload file | Historical / 84e76d8 matrices | Current upstream matrix (1ab7e5a) |
| --- | --- | --- | --- | --- | --- |
| Qwen2.5-0.5B Q4_0 | Phase 5 pinned `Q4_0.gguf` | `7671c0c304e6ce5a7fc577bcb12aba01e2c155cc2efd29b2213c95b18edaf6ed` | `qwen2.5-q4k-workloads.jsonl` | Historical: `qwen2.5-q4_0-current.jsonl`; 84e76d8: `qwen2.5-q4_0-llama84e76d8-current.jsonl` | 1ab7e5a: `qwen2.5-q4_0-llama1ab7e5a-current.jsonl` |
| Qwen2.5-0.5B Q4_K_M | Phase 5 pinned `Q4_K_M.gguf` | `74a4da8c9fdbcd15bd1f6d01d621410d31c6fc00986f5eb687824e7b93d7a9db` | `qwen2.5-q4k-workloads.jsonl` | Historical: `qwen2.5-q4_k_m-current.jsonl` and `qwen2.5-q4_k_m-baseline-recheck.jsonl`; 84e76d8: `qwen2.5-q4_k_m-llama84e76d8-current.jsonl` | 1ab7e5a: `qwen2.5-q4_k_m-llama1ab7e5a-current.jsonl` |
| Qwen2.5-0.5B Q5_K_M | Phase 5 pinned `Q5_K_M.gguf` | `041474553fcabfc2a2d67903f9d2c2e50bd92528e670da4f33b5d0ce6e59fd55` | `qwen2.5-q4k-workloads.jsonl` | Historical: `qwen2.5-q5_k_m-current.jsonl`; 84e76d8 experiment matrices: `qwen2.5-q5_k_m-q5_1k64-m512-ab.jsonl`, `qwen2.5-q5_k_m-q5kk64-m1024-ab.jsonl` | 1ab7e5a: `qwen2.5-q5_k_m-llama1ab7e5a-current.jsonl` |
| Qwen2.5-0.5B Q6_K | Phase 5 pinned `Q6_K.gguf` | `2f82233630c349ccf6b8daccf48f9a7865713d9f08a2eadfa456cebe9b97c7f5` | `qwen2.5-q4k-workloads.jsonl` | Historical: `qwen2.5-q6_k-current.jsonl`; 84e76d8: `qwen2.5-q6_k-llama84e76d8-current.jsonl` | 1ab7e5a: `qwen2.5-q6_k-llama1ab7e5a-current.jsonl` |
| Qwen2.5-0.5B Q8_0 | Phase 5 pinned `Q8_0.gguf` | `ca59ca7f13d0e15a8cfa77bd17e65d24f6844b554a7b6c12e07a5f89ff76844e` | `qwen2.5-q4k-workloads.jsonl` | Historical: `qwen2.5-q8_0-current.jsonl`; 84e76d8: `qwen2.5-q8_0-llama84e76d8-current.jsonl` | 1ab7e5a: `qwen2.5-q8_0-llama1ab7e5a-current.jsonl` |
| Qwen3-0.6B Q8_0 | `Qwen/Qwen3-0.6B-GGUF`, revision `23749fefcc72300e3a2ad315e1317431b06b590a` | `9465e63a22add5354d9bb4b99e90117043c7124007664907259bd16d043bb031` | `qwen3-0.6b-q8_0-workloads.jsonl` | Historical: `qwen3-0.6b-q8_0-current.jsonl`; 84e76d8: `qwen3-0.6b-q8_0-llama84e76d8-current.jsonl` | 1ab7e5a: `qwen3-0.6b-q8_0-llama1ab7e5a-current.jsonl` |
| LFM2.5-8B-A1B Q4_K_M | `LiquidAI/LFM2.5-8B-A1B-GGUF`, revision `49c14831707011e64d70b2ebd8462ba08d608434` | `4923ec14f06b968b74d663e5949867d2d9c3bf13a20b8be1a9f9af39989b2bb0` | `lfm2.5-8b-a1b-q4_k_m-workloads.jsonl` | Historical: `lfm2.5-8b-a1b-q4_k_m-current.jsonl`; 84e76d8: `lfm2.5-8b-a1b-q4_k_m-llama84e76d8-current.jsonl` | 1ab7e5a: `lfm2.5-8b-a1b-llama1ab7e5a-current.jsonl` |

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

This gate-ratio summary is an index; absolute Ferrum and llama.cpp tok/s plus first-token latency for each workload are tabulated in the upstream refreshes below.

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

### Current-upstream refresh: llama.cpp `1ab7e5ad`

This 2026-09-25 refresh uses official `ggml-org/llama.cpp` commit `1ab7e5ad2d4e7295c94c3b966a3e0b70fa365865`, built for Metal Release on the same M5 with an embedded Metal library and optional OpenMP, tests, examples, server, and tools disabled. The complete build output is `docs/measurements/phase7a/llama.cpp-1ab7e5a-current-upstream-build.log`. Source inspection found the Q4_K M=1 tuning unchanged from the prior refresh: `N_R0_Q4_K=2` and `N_SG_Q4_K=2`; the relevant direct MSL kernel had no intervening diff.

All seven pinned GGUF artifacts and hashes in the table above were independently reverified. Each matrix has five interleaved pairs for each workload and 50 JSONL rows (two runtimes × five workloads × five pairs). Prompt IDs, generation lengths, greedy decoding, BF16 KV, and runner options match within each pair. Ferrum and llama.cpp rates below are condition medians; ratios are medians of the five matched per-pair throughput ratios. First-token latency is the condition median in milliseconds. Exact IDs count full output-sequence matches across the five pairs. Rates are tok/s.

| Model / workload | Prefill tok/s Ferrum / llama.cpp (ratio) | Cached decode tok/s Ferrum / llama.cpp (ratio) | Complete-generation tok/s Ferrum / llama.cpp (ratio) | First-token ms Ferrum / llama.cpp | Exact IDs |
|---|---:|---:|---:|---:|---:|
| Qwen2.5 Q4_0 / Short | 804.0 / 1,550.7 (0.518x) | 130.2 / 269.3 (0.484x) | 110.5 / 217.2 (0.511x) | 26.4 / 13.8 | 5/5 |
| Qwen2.5 Q4_0 / 128-token prompt | 3,756.8 / 7,401.1 (0.505x) | 128.8 / 273.2 (0.474x) | 105.3 / 209.4 (0.501x) | 34.4 / 17.6 | 5/5 |
| Qwen2.5 Q4_0 / 512-token prompt | 5,451.4 / 9,925.7 (0.549x) | 122.8 / 268.8 (0.457x) | 74.4 / 146.3 (0.508x) | 94.3 / 51.8 | 5/5 |
| Qwen2.5 Q4_0 / 1,024-token prompt | 5,306.8 / 9,393.7 (0.565x) | 114.4 / 264.8 (0.432x) | 50.2 / 96.5 (0.517x) | 193.3 / 109.3 | 5/5 |
| Qwen2.5 Q4_0 / Sustained decode | 779.4 / 1,513.0 (0.507x) | 125.8 / 276.2 (0.457x) | 120.1 / 248.1 (0.485x) | 27.2 / 14.1 | 0/5 |
| Qwen2.5 Q4_K_M / Short | 742.6 / 1,406.7 (0.526x) | 138.3 / 238.8 (0.578x) | 114.3 / 197.5 (0.579x) | 28.6 / 15.2 | 0/5 |
| Qwen2.5 Q4_K_M / 128-token prompt | 3,532.0 / 6,989.8 (0.504x) | 138.2 / 230.7 (0.593x) | 109.1 / 187.6 (0.582x) | 36.5 / 18.6 | 5/5 |
| Qwen2.5 Q4_K_M / 512-token prompt | 5,065.8 / 9,336.9 (0.543x) | 131.2 / 233.5 (0.562x) | 74.9 / 133.5 (0.560x) | 101.4 / 55.1 | 5/5 |
| Qwen2.5 Q4_K_M / 1,024-token prompt | 5,220.3 / 8,844.2 (0.592x) | 121.2 / 236.5 (0.513x) | 50.8 / 90.0 (0.567x) | 196.5 / 116.0 | 5/5 |
| Qwen2.5 Q4_K_M / Sustained decode | 737.3 / 1,385.7 (0.534x) | 132.4 / 244.3 (0.543x) | 126.0 / 221.8 (0.569x) | 28.8 / 15.4 | 0/5 |
| Qwen2.5 Q5_K_M / Short | 707.4 / 1,432.7 (0.496x) | 133.5 / 226.2 (0.590x) | 110.4 / 189.8 (0.584x) | 30.0 / 14.9 | 0/5 |
| Qwen2.5 Q5_K_M / 128-token prompt | 3,363.7 / 6,900.3 (0.487x) | 132.9 / 221.0 (0.601x) | 104.9 / 180.7 (0.578x) | 38.4 / 18.8 | 5/5 |
| Qwen2.5 Q5_K_M / 512-token prompt | 5,081.0 / 9,306.5 (0.546x) | 126.4 / 227.6 (0.555x) | 73.1 / 130.5 (0.560x) | 101.1 / 55.3 | 5/5 |
| Qwen2.5 Q5_K_M / 1,024-token prompt | 5,109.7 / 8,844.5 (0.581x) | 118.1 / 217.3 (0.539x) | 49.7 / 88.2 (0.563x) | 200.8 / 116.0 | 5/5 |
| Qwen2.5 Q5_K_M / Sustained decode | 691.2 / 1,382.2 (0.495x) | 129.1 / 231.5 (0.558x) | 122.2 / 210.6 (0.578x) | 30.7 / 15.4 | 0/5 |
| Qwen2.5 Q6_K / Short | 629.9 / 1,478.6 (0.430x) | 132.1 / 199.1 (0.658x) | 107.3 / 173.9 (0.619x) | 33.7 / 14.5 | 5/5 |
| Qwen2.5 Q6_K / 128-token prompt | 2,984.5 / 7,068.9 (0.423x) | 134.6 / 201.8 (0.667x) | 102.4 / 168.2 (0.607x) | 43.2 / 18.4 | 5/5 |
| Qwen2.5 Q6_K / 512-token prompt | 4,779.1 / 9,643.1 (0.499x) | 127.2 / 203.2 (0.624x) | 71.1 / 125.0 (0.571x) | 107.5 / 53.4 | 5/5 |
| Qwen2.5 Q6_K / 1,024-token prompt | 4,778.7 / 9,077.0 (0.526x) | 116.8 / 197.2 (0.594x) | 47.6 / 86.3 (0.551x) | 214.6 / 113.1 | 5/5 |
| Qwen2.5 Q6_K / Sustained decode | 575.6 / 1,451.0 (0.397x) | 128.2 / 202.7 (0.631x) | 120.9 / 191.2 (0.632x) | 36.8 / 14.7 | 0/5 |
| Qwen2.5 Q8_0 / Short | 554.1 / 1,446.1 (0.382x) | 134.3 / 198.5 (0.674x) | 105.3 / 173.5 (0.607x) | 38.2 / 14.8 | 5/5 |
| Qwen2.5 Q8_0 / 128-token prompt | 2,707.6 / 7,074.1 (0.386x) | 135.8 / 197.8 (0.684x) | 100.4 / 165.5 (0.601x) | 47.6 / 18.3 | 5/5 |
| Qwen2.5 Q8_0 / 512-token prompt | 4,565.4 / 9,772.5 (0.466x) | 128.3 / 198.8 (0.645x) | 70.2 / 124.2 (0.564x) | 112.5 / 52.6 | 5/5 |
| Qwen2.5 Q8_0 / 1,024-token prompt | 4,592.1 / 9,132.3 (0.504x) | 118.9 / 195.3 (0.608x) | 46.8 / 86.1 (0.545x) | 223.3 / 112.4 | 5/5 |
| Qwen2.5 Q8_0 / Sustained decode | 511.5 / 1,449.3 (0.353x) | 130.9 / 199.9 (0.656x) | 122.8 / 189.4 (0.648x) | 41.4 / 14.7 | 0/5 |
| Qwen3-0.6B Q8_0 / Short | 435.8 / 1,258.3 (0.347x) | 113.2 / 164.5 (0.690x) | 87.6 / 143.8 (0.608x) | 48.5 / 17.0 | 5/5 |
| Qwen3-0.6B Q8_0 / 128-token prompt | 1,970.9 / 6,001.2 (0.323x) | 113.5 / 163.8 (0.696x) | 80.9 / 138.8 (0.580x) | 65.3 / 21.6 | 0/5 |
| Qwen3-0.6B Q8_0 / 512-token prompt | 3,114.7 / 7,099.6 (0.437x) | 104.0 / 154.9 (0.673x) | 51.1 / 95.2 (0.538x) | 164.7 / 72.4 | 5/5 |
| Qwen3-0.6B Q8_0 / 1,024-token prompt | 3,359.8 / 6,283.2 (0.536x) | 92.1 / 143.3 (0.641x) | 34.2 / 61.1 (0.561x) | 305.1 / 163.2 | 5/5 |
| Qwen3-0.6B Q8_0 / Sustained decode | 391.0 / 1,241.8 (0.312x) | 110.3 / 164.7 (0.669x) | 103.4 / 157.4 (0.657x) | 54.1 / 17.2 | 0/5 |
| LFM2.5-8B-A1B Q4_K_M / Short | 42.9 / 215.6 (0.191x) | 30.8 / 86.4 (0.337x) | 21.5 / 72.5 (0.292x) | 257.0 / 51.2 | 0/5 |
| LFM2.5-8B-A1B Q4_K_M / 128-token prompt | 392.5 / 1,511.1 (0.260x) | 31.3 / 98.2 (0.320x) | 20.2 / 67.5 (0.300x) | 326.4 / 84.9 | 0/5 |
| LFM2.5-8B-A1B Q4_K_M / 512-token prompt | 470.5 / 2,108.3 (0.222x) | 30.6 / 97.5 (0.314x) | 10.5 / 41.2 (0.253x) | 1,088.6 / 243.1 | 5/5 |
| LFM2.5-8B-A1B Q4_K_M / 1,024-token prompt | 481.0 / 2,141.5 (0.225x) | 29.5 / 96.4 (0.306x) | 6.3 / 25.8 (0.245x) | 2,129.0 / 478.4 | 5/5 |
| LFM2.5-8B-A1B Q4_K_M / Sustained decode | 38.9 / 233.7 (0.167x) | 31.1 / 101.2 (0.308x) | 28.2 / 95.7 (0.296x) | 282.7 / 47.3 | 0/5 |

No format/architecture meets the gate. The strongest sustained decode in this refresh is Qwen3 Q8_0 at 0.669x (110.3 / 164.7 tok/s); Qwen2.5 Q8_0 reaches 0.656x (130.9 / 199.9 tok/s). The strongest 1,024-token prefill ratio is Q4_K_M at 0.592x (5,220.3 / 8,844.2 tok/s), still below 0.80x. Large Qwen tuning stays locked. A fresh flushed LFM M=1 diagnostic profile measured 34.41 prefill, 9.05 cached-decode, and 7.87 complete-generation tok/s with 319.92 ms first-token latency. It attributed 246.68 ms across 352 dispatches to the retained sparse-expert Q4_K 16-row path and 60.46 ms across 576 dispatches to dense `q4_k_gemv_8rows`. The latter is the next direct-MSL candidate. This per-dispatch-flushed profile is diagnostic and is not comparable to production throughput.

## Upstream Metal review

The refreshed official upstream `84e76d8a` Metal source keeps its Q4_K M=1 defaults at `N_R0_Q4_K=2` and `N_SG_Q4_K=2` in `ggml-metal/ggml-metal-impl.h`; `kernel_mul_mv_q4_K_f32_impl` in `ggml-metal/kernels/mul_mv.metal` reuses activation fragments while accumulating adjacent output rows. An M5-specific tuning discussion in [llama.cpp issue 19303](https://github.com/ggml-org/llama.cpp/issues/19303) reports a larger Q4_K row-tile experiment, but that remains an unmerged result. The refreshed upstream also has a multi-kernel `flash_attn_ext` implementation in `ggml-metal/kernels/fa.metal` and its Metal host scheduler. Ferrum's prefill attention score/context products already use MPP, but its mask and softmax remain separate conventional MSL kernels; fusing the score product, mask, row softmax, and context product with TensorOps cooperative intermediates is still a plausible prefill candidate. A flushed Qwen Q5 profile attributed 28.51 ms to 24 `attention_softmax` calls at 1,024 rows; this is diagnostic GPU time, not end-to-end latency.

Phase 5.5 already retained Q4_0 and Q8_0 multirow M=1 kernels, and rejected a Q5_0/Q5_1 two-row candidate after synchronized profiles showed regressions. The current bottleneck evidence points to quantized projections for decode and quantized matrix tiling for prefill. The Q4_K candidate below is a native Ferrum implementation with its own measured shape guard.

## Experiment 1: Q4_K two-row reuse for M=1

Status: accepted for wide Q4_K M=1 projections; the matched matrix showed a small consistent decode improvement with unchanged outputs.

The candidate assigns each 32-lane SIMD group two adjacent output rows. It loads the eight activation values for a Q4_K superblock once and reuses them across both rows. Four SIMD groups cover up to eight output rows per threadgroup. A tail guard handles arbitrary output-row counts, and production selection is limited to M=1 Q4_K projections with at least 128 output rows.

The test `q4_k_eight_row_gemv_reuses_activations_and_covers_output_tail` compares a 133-row output against a scalar reference under Metal API Validation and GPU Shader Validation; it passed. In isolated two-token diagnostic requests with synchronized dispatches, the 12 cached-decode Q4_K down-projection calls fell from 0.822 ms to 0.678 ms. This is operation attribution, not the production latency result.

The same-period three-pair baseline recheck and candidate matrices use the same prompts, generation lengths, build, and reference process options. Across all 15 paired Ferrum runs, the candidate's decode throughput was 1.019x the original kernel (range 1.011–1.035x), and full-generation throughput was 1.008x (range 1.002–1.038x). Prefill was effectively unchanged at 1.001x. Generated IDs were identical between the two Ferrum paths in all 15 pairs, with 9/15 exact matches against llama.cpp in both matrices. The Ferrum/reference median decode ratio rose from 0.480 to 0.494; generation rose from 0.506 to 0.510. Because the measured gain is small, it remains a narrow Q4_K shape rule and does not change the overall gate status. The production default is enabled for `N >= 128`; `FERRUM_Q4_K_GEMV_8ROWS=false` selects the original kernel for controlled comparisons.

The table reports condition-median absolute rates and first-token latency; bracketed ratios are medians of matched pairs in the saved JSONL. Rate cells are Ferrum control -> 8-row candidate / llama.cpp tok/s, followed by `[control/llama -> candidate/llama; candidate/control]`.

| Workload | Prefill tok/s C -> 8 / llama | Cached decode tok/s C -> 8 / llama | Complete generation tok/s C -> 8 / llama | First-token ms C -> 8 / llama | IDs C=8; 8=L |
|---|---:|---:|---:|---:|---:|
| Short | 749.75 -> 748.59 / 1,441.79 `[0.519 -> 0.518; 0.999]` | 117.87 -> 119.99 / 228.75 `[0.480 -> 0.520; 1.017]` | 101.33 -> 102.21 / 194.44 `[0.511 -> 0.526; 1.008]` | 28.3 -> 28.4 / 14.8 | 3/3; 0/3 |
| 128-token prompt | 3,476.22 -> 3,501.10 / 6,943.69 `[0.516 -> 0.510; 1.003]` | 119.11 -> 121.21 / 236.00 `[0.522 -> 0.518; 1.018]` | 96.91 -> 98.50 / 182.72 `[0.529 -> 0.539; 1.015]` | 37.3 -> 37.0 / 18.8 | 3/3; 3/3 |
| 512-token prompt | 4,951.64 -> 4,948.33 / 9,377.70 `[0.532 -> 0.528; 0.995]` | 113.46 -> 115.46 / 233.77 `[0.493 -> 0.494; 1.019]` | 67.77 -> 68.58 / 134.33 `[0.506 -> 0.510; 1.008]` | 103.7 -> 103.8 / 54.8 | 3/3; 3/3 |
| 1,024-token prompt | 4,712.56 -> 4,699.98 / 8,790.56 `[0.539 -> 0.535; 1.001]` | 104.48 -> 105.77 / 226.72 `[0.461 -> 0.467; 1.015]` | 45.15 -> 45.39 / 89.42 `[0.506 -> 0.508; 1.005]` | 217.6 -> 218.2 / 116.8 | 3/3; 3/3 |
| Sustained decode | 649.23 -> 661.58 / 1,383.01 `[0.471 -> 0.478; 1.033]` | 113.74 -> 116.66 / 243.17 `[0.476 -> 0.479; 1.023]` | 107.60 -> 111.12 / 220.81 `[0.497 -> 0.503; 1.030]` | 32.6 -> 32.0 / 15.4 | 3/3; 0/3 |

## Experiment 2: Q6_K two-row reuse for M=1

Status: accepted for wide Q6_K M=1 projections; the matched matrix showed a consistent decode improvement with unchanged outputs.

The candidate uses the same two-output-row-per-SIMD-group mapping and four-group, eight-row tile for Q6_K. It reuses the Q6_K activation fragment across adjacent rows and handles output tails. Selection is limited to M=1 projections with at least 128 output rows. The focused 133-row tail test passed with Metal API Validation and GPU Shader Validation enabled. The upstream reference keeps `N_R0_Q6_K=2`; the Ferrum candidate uses two rows per SIMD group and eight rows per threadgroup, a separate M5-specific shape selected from this paired measurement.

The A/B used the same-period baseline recheck and candidate matrices, three interleaved pairs per workload. Candidate decode throughput was 1.030x the original kernel at the median across 15 pairs (range 1.018–1.043x); per-workload medians ranged from 1.025x to 1.036x. Prefill was effectively unchanged (median 0.999x), and full-generation throughput improved by roughly 1–3%. Generated IDs matched the original Ferrum path in all 15 pairs. The Ferrum/reference decode medians after the change remain below the Phase 7 gate, so this kernel improves Q6_K M=1 without clearing the model-level gate. The production default is enabled for `N >= 128`; `FERRUM_Q6_K_GEMV_8ROWS=false` selects the original kernel for controlled comparisons.

The table reports condition-median absolute rates and first-token latency; bracketed ratios are medians of matched pairs in the saved JSONL. Rate cells are Ferrum control -> 8-row candidate / llama.cpp tok/s, followed by `[control/llama -> candidate/llama; candidate/control]`.

| Workload | Prefill tok/s C -> 8 / llama | Cached decode tok/s C -> 8 / llama | Complete generation tok/s C -> 8 / llama | First-token ms C -> 8 / llama | IDs C=8; 8=L |
|---|---:|---:|---:|---:|---:|
| Short | 635.04 -> 636.90 / 1,450.93 `[0.431 -> 0.441; 0.998]` | 126.02 -> 129.60 / 200.27 `[0.624 -> 0.645; 1.025]` | 103.69 -> 106.07 / 174.59 `[0.587 -> 0.602; 1.023]` | 33.4 -> 33.3 / 14.7 | 3/3; 3/3 |
| 128-token prompt | 3,013.76 -> 3,007.08 / 7,071.29 `[0.420 -> 0.425; 0.998]` | 124.94 -> 129.24 / 201.21 `[0.618 -> 0.642; 1.031]` | 97.11 -> 99.13 / 166.20 `[0.579 -> 0.596; 1.018]` | 42.8 -> 42.9 / 18.4 | 3/3; 3/3 |
| 512-token prompt | 4,428.82 -> 4,395.02 / 9,583.89 `[0.461 -> 0.459; 1.000]` | 120.99 -> 124.69 / 199.01 `[0.606 -> 0.628; 1.028]` | 67.18 -> 68.02 / 123.77 `[0.538 -> 0.551; 1.015]` | 115.9 -> 116.8 / 53.7 | 3/3; 3/3 |
| 1,024-token prompt | 4,291.08 -> 4,293.15 / 9,092.63 `[0.470 -> 0.472; 1.000]` | 111.45 -> 115.52 / 199.12 `[0.571 -> 0.582; 1.036]` | 43.97 -> 44.38 / 86.75 `[0.508 -> 0.511; 1.011]` | 239.0 -> 238.9 / 112.9 | 3/3; 3/3 |
| Sustained decode | 574.73 -> 568.21 / 1,406.63 `[0.413 -> 0.404; 0.989]` | 121.48 -> 125.24 / 207.88 `[0.598 -> 0.602; 1.030]` | 115.86 -> 118.95 / 193.75 `[0.600 -> 0.615; 1.027]` | 36.8 -> 37.3 / 15.2 | 3/3; 0/3 |

## Raw evidence

All workload definitions, paired JSONL matrices, and complete runner logs are retained in `docs/measurements/phase7a/`, including the Q6_K baseline recheck and `qwen2.5-q6_k-q6k8rows-candidate.jsonl`. The directory also includes the earlier Q4_K_M profile attempt that used the default batching limit and therefore reported no GPU samples; it is not used for operation attribution. Isolated baseline and candidate profiles are retained under `docs/measurements/phase7a/profiles/`. The first profile command also contains a logged incorrect model path; the corrected capture followed it.

## M5 GPU Neural Accelerator and Metal 4 TensorOps audit

Ferrum's `mpp::tensor_ops` kernels are inline MSL operations on the M5 GPU and use the GPU's per-core Neural Accelerators. This is distinct from the standalone Apple Neural Engine and from Core ML/Core AI model execution. Apple documents the MSL 4 TensorOps path and its M5 hardware use in [Running inline ML operations in a shader with Metal 4](https://developer.apple.com/documentation/metal/running-inline-ml-operations-in-a-shader-with-metal-4) and [WWDC26: Optimize custom machine learning operations with Metal tensors](https://developer.apple.com/videos/play/wwdc2026/330/).

### Current coverage and remaining conventional paths

| Workload | Current M5 path | Remaining conventional MSL / candidate boundary |
| --- | --- | --- |
| Dense BF16 projections | `project_mpp` for eligible batched shapes; other batch sizes use native matrix or direct kernels | The large dense prefill path is already TensorOps-backed. M=1 GEMV stays on direct shaders. |
| GGUF Q4_0, Q5_0, Q5_1, Q4_K, Q5_K, Q6_K, and Q8_0 projections | MPP GEMM uses measured format-specific row boundaries for BF16 activations and K divisible by 128: Q4_0/Q4_K M>=4, Q5_1/Q5_K/Q6_K M>=8, Q8_0 M>=12, and Q5_0 M>=16. Quantized MPP kernels unpack custom GGUF blocks into bounded BF16 tiles before `matmul2d`; Q4_K and Q5_K use measured 64-wide K tiles at M>=1024, and Q4_K also uses the measured 128-row M tile there. | Rows below each format's boundary and unsupported dtype/K shapes still use conventional `q*_gemm` MSL. M=1 uses format-specific direct `q*_gemv` kernels, including measured Q5_0, Q5_1, and Q5_K row-reuse variants; it never enters MPP. MLX affine-Q4 is a separate F16 path and is outside this GGUF campaign. |
| Quantized MoE expert projections | BF16 prefill batches that average at least 8 assignments per expert use expert grouping plus Q4_K/Q5_K/Q6_K MPP GEMM; Q4_K and Q6_K use a 32x64x64 `matmul2d` tile, while Q5_K keeps K=128 pending a model-level measurement. One-token routing remains a separate GPU routing optimization. | Sparse assignments and unsupported dtypes/alignment use direct MSL. Q4_K with at least 128 output rows and K divisible by 256 uses the measured 16-row-per-threadgroup fallback with block/pair activation reuse. Q6_K with at least 128 output rows and K>=256 divisible by 256 uses its measured 8-row fallback; Q5_K remains on its scalar shader. MPP stages exact GGUF values into BF16; it does not reinterpret GGUF blocks as Metal's native block-scaled int4/int8 tensors. |
| Prefill attention | BF16/F16 attention score and context products use TensorOps when query length exceeds one; production scale/causal-mask/row softmax uses one conventional MSL `attention_softmax` kernel. | The softmax reduction remains MSL. Its retained prefix variant skips masked suffix reductions only for full-prefill shapes with M>=256. A fused FlashAttention-style TensorOps kernel using cooperative results and row reductions remains a larger prefill candidate. Decode's one-query context stays on its direct kernel. |

The dispatch rules are visible in `src/ops/transformer.rs`; the current quantized MPP unpack/stage/multiply path is in `src/metal/shaders/project_mpp.metal`; quantized expert projection is in `src/metal/shaders/ops.metal`. Norm, RoPE, activation, and elementwise kernels are conventional MSL but are not matrix contractions that should be moved to `matmul2d` by default.

Profile attribution shows why the remaining conventional shapes need separate treatment. In Qwen2.5 Q4_K M=1 decode, the leading projection costs were `up_proj.q5_0_gemv` at 1.812 ms, `lm_head.q8_0_gemv_8rows` at 1.650 ms, and `gate_proj.q5_0_gemv` at 1.332 ms. These one-row GEMVs are not TensorOps candidates for this dispatch; direct shaders remain selected. In the LFM2.5 1,024-token flushed-dispatch diagnostic, no conventional expert projection fallback appeared: grouped MPP expert products accounted for about 1.126 s (Q4_K input), 0.316 s (Q4_K output), and 0.215 s (Q6_K output). Conventional `attention_softmax` was about 17 ms in that capture, while attention score and context products already used TensorOps. These are diagnostic GPU totals rather than end-to-end timings.

A follow-up isolated LFM2.5 M=1 capture used the saved 11-token prompt and `FERRUM_BATCH_LIMIT=1` so per-operation timings were available. This flushes every dispatch and is not a production throughput comparison: the run measured 31.90 prefill tok/s, 8.88 median cached-decode tok/s (8.45 aggregate), 7.58 complete-generation tok/s, and 345.12 ms first-token latency. The decode profile attributes 295.13 ms across 352 calls to direct-MSL `moe.input_projection.expert_project_q4_k_8rows`, 82.76 ms across 192 calls to `moe.output_projection.expert_project_q4_k_8rows`, 42.67 ms across 160 calls to `moe.output_projection.expert_project_q6_k_8rows`, 64.32 ms across 576 calls to `q4_k_gemv_8rows`, and 41.25 ms across 16 calls to `lm_head.q6_k_gemv_8rows`. These sparse M=1 expert rows remain a poor fit for the 32-row TensorOps tile; the existing direct path stays selected. The workload and raw profile are `profiles/lfm2.5-8b-a1b-m1-isolated-profile-workload.jsonl` and `profiles/lfm2.5-8b-a1b-m1-isolated-profile.jsonl` under `docs/measurements/phase7a/`.

### Native quantized API constraints

The installed Xcode 27.0 SDK (`MacOSX27.0.sdk`) confirms the current API surface. `MTLTensorDataTypeInt8`/`UInt8` are available with the macOS 26 tensor API; `Int4`/`UInt4` arrive in macOS 26.4; macOS 27 adds `Int2`/`UInt2`, FP4 E2M1, FP8 E4M3/E5M2, and the UE8M0 block-scale type. The installed `MPPTensorOpsMatMul2d.h` explicitly lists native `int8_t`/`uint8_t` and `int4b_format`/`uint4b_format` operand combinations with half or BF16 activations; it also lists integer accumulations and FP32 destinations for supported combinations. The `get_left_input_cooperative_tensor` and `get_right_input_cooperative_tensor` methods can produce operation-compatible cooperative input layouts from values already held in cooperative tensors, avoiding an intermediate threadgroup-memory round trip. Cooperative element ownership is chosen by the operation, so every participating thread must execute the operation and the existing cooperative value must match the required layout. Apple's MPP guide describes cooperative tensors as a fusion tool and notes that threadgroup tensors remain an option when custom thread ownership is needed. The current scales plane accepts only FP8 UE8M0 with block factors `[32, 1]`; it cannot encode arbitrary GGUF FP16 scales/minima exactly. Native 4-bit and FP8 tensor rows also have stricter row, stride, and 128-byte buffer-alignment rules in the installed feature tables. Our cooperative-input prototype used the compiler-accepted single-SIMD-group mode and a 32x32 output tile. Sources: [Metal Performance Primitives programming guide](https://developer.apple.com/download/files/Metal-Performance-Primitives-Programming-Guide.pdf), [MTLTensorDataType API](https://developer.apple.com/documentation/metal/mtltensordatatype), [Metal feature-set tables](https://developer.apple.com/metal/Metal-Feature-Set-Tables.pdf), [Running inline ML operations in a shader with Metal 4](https://developer.apple.com/documentation/metal/running-inline-ml-operations-in-a-shader-with-metal-4), and [WWDC26 Metal tensors session](https://developer.apple.com/videos/play/wwdc2026/330/).

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

Absolute rates and first-token latency below are condition medians; bracketed ratios are medians of matched pairs. Rate cells are CPU-route control -> GPU-route candidate / llama.cpp tok/s, followed by `[control/llama -> candidate/llama; candidate/control]`.

| Workload | Prefill tok/s C -> GPU / llama | Cached decode tok/s C -> GPU / llama | Complete generation tok/s C -> GPU / llama | First-token ms C -> GPU / llama | IDs C=GPU; GPU=L |
|---|---:|---:|---:|---:|---:|
| Short | 22.45 -> 22.37 / 211.95 `[0.103 -> 0.106; 0.994]` | 19.49 -> 21.03 / 93.74 `[0.207 -> 0.224; 1.078]` | 12.90 -> 13.58 / 74.75 `[0.173 -> 0.182; 1.051]` | 490.3 -> 492.1 / 52.1 | 3/3; 3/3 |
| 128-token prompt | 27.84 -> 27.96 / 1,385.62 `[0.020 -> 0.020; 1.004]` | 19.28 -> 20.30 / 93.93 `[0.207 -> 0.222; 1.066]` | 3.11 -> 3.17 / 63.61 `[0.049 -> 0.050; 1.016]` | 4,597.8 -> 4,579.1 / 92.6 | 3/3; 0/3 |
| 512-token prompt | 28.02 -> 28.04 / 1,972.85 `[0.014 -> 0.014; 1.001]` | 19.42 -> 20.91 / 93.53 `[0.207 -> 0.223; 1.078]` | 0.89 -> 0.89 / 38.93 `[0.023 -> 0.023; 1.005]` | 18,271.5 -> 18,260.2 / 259.8 | 3/3; 0/3 |
| 1,024-token prompt | 27.93 -> 27.98 / 1,971.43 `[0.015 -> 0.014; 1.002]` | 18.80 -> 20.61 / 88.57 `[0.216 -> 0.225; 1.083]` | 0.45 -> 0.45 / 24.06 `[0.020 -> 0.019; 1.004]` | 36,667.8 -> 36,599.2 / 519.7 | 3/3; 3/3 |
| Sustained decode | 21.28 -> 21.31 / 214.82 `[0.100 -> 0.100; 1.012]` | 19.35 -> 20.91 / 91.11 `[0.215 -> 0.230; 1.078]` | 17.92 -> 19.29 / 85.78 `[0.208 -> 0.226; 1.077]` | 517.1 -> 516.5 / 51.4 | 3/3; 3/3 |

All 15 Ferrum sequences matched the CPU-routing path exactly. Both paths matched llama.cpp in 9/15 cases. First decode command buffers fell from 23 to 1 in every case; dispatches rose from 407 to 429 on the short/128-token cases and 419 to 441 on the longer-prefill cases because the GPU route shader adds one dispatch per sparse layer while eliminating the CPU synchronization boundary. The remaining LFM2.5 gap is substantial: Ferrum/reference median decode is 0.225x and medium/long prefill remains about 0.020x, so this improvement does not clear the campaign gate.

The paired files are `lfm2.5-8b-a1b-gpu-route-guard-baseline.jsonl` and `lfm2.5-8b-a1b-gpu-route-guard-candidate.jsonl`, with their runner logs in `docs/measurements/phase7a/`. An earlier run was interrupted during reference warmup and recorded no rows; its log is retained as `lfm2.5-8b-a1b-gpu-route-guard-interrupted-run.log` and is excluded from these results.

## Experiment 4: Native signed-int8 TensorOps for Q8_0 prefill

Status: rejected; the experimental kernel and dispatch control were removed after the paired run. The current SDK's `matmul2d` supports BF16 x signed-int8 with FP32 output. Q8_0's signed payload has an arbitrary FP16 scale for every 32 values, while Metal's native quantized tensor scale plane uses E8M0. The prototype therefore viewed each payload block as a native int8 TensorOps input, applied its stored FP16 scale to that 32-wide partial product, and accumulated the result. This preserves the stored scales without converting them to E8M0, but it needs four `matmul2d` calls for each existing K=128 tile.

The candidate passed its M=35, N=64, K=512 reference check with Metal API Validation and GPU Shader Validation enabled. A 1024-token candidate profile confirmed it ran for Qwen3 Q8_0 `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, and `down_proj`; token-by-token decode continued to use `q8_0_gemv_8rows`.

The release A/B used the pinned Qwen3-0.6B Q8_0 artifact, three interleaved control/candidate pairs for each of the short, 128-, 512-, and 1,024-token prompts plus sustained decode. Candidate/control prefill-throughput medians by prompt length were 1.304x (short, 21 tokens), 1.067x (128), 0.836x (512), and 0.770x (1,024). The 512- and 1,024-token regressions appeared in all three pairs. Median cached decode was 0.999x and median full-generation throughput was 1.012x across the 15 samples. Ferrum's complete generated IDs matched control in 12/15 pairs; the only differences were in the 129-token sustained-decode workload. These results do not justify the consistent medium/long prefill loss, so the path remains removed. The existing direct M=1 GEMV and BF16-staged K=128 TensorOps kernels remain selected.

The internal candidate/control matrix has no llama.cpp rows. Its absolute condition-median rates and first-token latency are shown below; the adjacent archived-runs table later in this journal has the same raw-JSONL values. Rates are tok/s and latency is milliseconds.

| Workload | Prefill tok/s control -> int8 | Cached decode tok/s control -> int8 | Complete generation tok/s control -> int8 | First-token ms control -> int8 |
|---|---:|---:|---:|---:|
| Short | 400.06 -> 517.93 | 99.07 -> 98.12 | 77.86 -> 82.15 | 52.80 -> 40.85 |
| 128-token prompt | 1,811.37 -> 1,950.16 | 98.05 -> 101.10 | 71.69 -> 74.74 | 71.00 -> 65.94 |
| 512-token prompt | 2,615.74 -> 2,140.01 | 90.96 -> 90.59 | 44.06 -> 38.89 | 196.04 -> 239.59 |
| 1,024-token prompt | 2,715.79 -> 2,076.97 | 81.11 -> 80.48 | 28.59 -> 23.91 | 377.39 -> 493.35 |
| Sustained decode | 378.73 -> 457.88 | 95.67 -> 95.40 | 90.34 -> 90.99 | 55.76 -> 46.20 |

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

The internal A/B's absolute rates and first-token latencies are condition medians from its saved JSONL. The ratios above are medians of matched pairs; rates below are tok/s and latency is milliseconds.

| Workload | Prefill tok/s control -> grouped TensorOps | Cached decode tok/s control -> grouped TensorOps | Complete generation tok/s control -> grouped TensorOps | First-token ms control -> grouped TensorOps |
|---|---:|---:|---:|---:|
| Short | 22.38 -> 21.66 | 21.09 -> 20.99 | 13.56 -> 13.37 | 491.88 -> 508.06 |
| 128-token prompt | 27.96 -> 27.86 | 20.85 -> 20.07 | 3.17 -> 3.14 | 4,579.01 -> 4,594.08 |
| 512-token prompt | 27.95 -> 118.69 | 20.82 -> 20.68 | 0.89 -> 3.31 | 18,316.16 -> 4,314.01 |
| 1,024-token prompt | 29.38 -> 119.36 | 20.55 -> 20.67 | 0.48 -> 1.81 | 34,858.51 -> 8,579.60 |
| Sustained decode | 23.71 -> 23.42 | 22.35 -> 21.61 | 20.38 -> 19.96 | 464.24 -> 470.00 |

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

## Experiment 29: Eight-row-per-SIMD Q5_1 M=1 GEMV (rejected)

Status: rejected and removed from the shipping source. The experiment extended the retained four-row-per-SIMD Q5_1 kernel to eight output rows per SIMD group, reusing each loaded activation fragment across twice as many rows. It remains conventional MSL for M=1; `matmul2d` TensorOps and the standalone Apple Neural Engine/Core ML are not used for this shape.

The short-prompt Qwen2.5 Q5_K_M decode profile attributed 1.073 ms per step to 24 `gate_proj.q5_1_gemv_n4` calls and 1.009 ms to 24 `up_proj.q5_1_gemv_n4` calls. The A/B used the pinned Qwen2.5-0.5B Q5_K_M artifact (SHA-256 `041474553fcabfc2a2d67903f9d2c2e50bd92528e670da4f33b5d0ce6e59fd55`), five interleaved control/candidate/llama.cpp pairs per workload, shared prompt IDs and generation lengths, greedy decoding, BF16 KV, and llama.cpp `84e76d8a23162eca70490da131945ebec1f09bf4`. Both Ferrum variants held the retained `q5_1_gemv_n4` and `q5_k_gemv_8rows` selectors enabled; only `q5_1_gemv_n8` changed from false (control) to true (candidate). The candidate passed the scalar-reference/tail check at N=131, K=512 and N=4,864, K=896 with Metal API and GPU Shader Validation enabled; the log is `q5_1-n8-gemv-validation.log`.

Throughput below is the condition median; ratios are medians of matched per-pair rates. Absolute tok/s for Ferrum control, candidate, and llama.cpp appear before `[control/llama -> candidate/llama; candidate/control]`. First-token latency is milliseconds.

| Workload | Prefill tok/s C (4-row)->K (8-row) / llama `[C/L->K/L; K/C]` | Cached decode tok/s C->K / llama `[C/L->K/L; K/C]` | Complete generation tok/s C->K / llama `[C/L->K/L; K/C]` | First-token ms C->K / llama | IDs C=K; K=L; C=L |
|---|---:|---:|---:|---:|---:|
| Short, M=21 | 705.9->715.5 / 1,418.8 `[0.498->0.501; 1.007]` | 137.5->132.6 / 222.1 `[0.613->0.590; 0.965]` | 113.4->109.9 / 189.8 `[0.596->0.582; 0.969]` | 30.05->29.66 / 15.10 | 5/5; 0/5; 0/5 |
| 128-token prompt | 3,347.3->3,337.1 / 6,835.0 `[0.490->0.488; 0.992]` | 137.9->134.2 / 221.5 `[0.618->0.606; 0.973]` | 107.2->104.9 / 181.8 `[0.591->0.575; 0.974]` | 38.56->38.66 / 18.96 | 5/5; 5/5; 5/5 |
| 512-token prompt | 5,021.1->5,055.5 / 9,169.6 `[0.551->0.551; 1.007]` | 129.1->123.7 / 211.9 `[0.608->0.583; 0.963]` | 73.5->72.3 / 126.2 `[0.580->0.572; 0.986]` | 102.36->101.60 / 56.06 | 5/5; 5/5; 5/5 |
| 1,024-token prompt | 5,126.5->5,114.5 / 8,828.6 `[0.580->0.578; 0.999]` | 121.9->117.1 / 218.1 `[0.560->0.538; 0.960]` | 50.3->49.6 / 88.4 `[0.571->0.561; 0.984]` | 200.07->200.55 / 116.22 | 5/5; 5/5; 5/5 |
| Sustained decode, prompt M=21 | 693.5->693.9 / 1,372.6 `[0.505->0.506; 0.999]` | 133.1->128.4 / 225.3 `[0.591->0.570; 0.968]` | 126.1->122.0 / 210.0 `[0.600->0.581; 0.967]` | 30.58->30.56 / 15.55 | 5/5; 0/5; 0/5 |

The eight-row variant reduced cached decode by 3.7-5.4 tok/s across the five workloads; every workload's paired candidate/control ratio was below 0.974x. Complete-generation rates also fell in all five cases. Prefill stayed within measurement variation, as this kernel is selected only for M=1. Control and candidate IDs matched in all 25 pairs. Because the candidate was consistently slower, it and its selector were removed; production keeps the four-row-per-SIMD direct shader. M=1 remains outside TensorOps. The general Phase 7 performance gate remains unmet.

Raw results and the run log are `qwen2.5-q5_k_m-q5_1-n8-ab.jsonl` and `.run.log`; the source matrix is `qwen2.5-q5_k_m-q5_1-n8-workloads.jsonl` in `docs/measurements/phase7a/`.

## Experiment 30: Shape-gated GPU MoE routing for prompt prefill

Status: retained by default for MoE routing with `top_k <= 16` when the prompt chunk has `M <= 32` or `M >= 512` tokens. M=1 continues to use the existing GPU router and direct MSL expert GEMV. The retained change moves router top-k selection and routing metadata onto Metal for measured prompt shapes; it is a conventional GPU routing kernel and synchronization change, not Metal 4 TensorOps or the standalone Apple Neural Engine/Core ML. Shapes 33–511 stay on the CPU routing path because the measured speedups there were absent or small while temporary memory rose sharply.

The LFM2.5-8B-A1B Q4_K_M final comparison used the same GGUF in Ferrum and llama.cpp, fixed official-tokenizer IDs, greedy decoding, BF16 KV, three interleaved pairs per workload, and the verified llama.cpp build at `84e76d8a23162eca70490da131945ebec1f09bf4`. Its build cache records that exact commit, and each reference JSONL row now records `llama_commit`. The first all-shape route sweep used a runner that did not preserve its executable path or commit in JSONL; its Ferrum-only control/candidate measurements are retained below, while its llama.cpp values are excluded. The final current-upstream matrix reran all eight workloads with the production shape selector active. Ferrum used `FERRUM_MOE_GPU_ROUTING=true`; its candidate set `moe_gpu_routing_prefill=true`, while control set it to false. The selector routes only M<=32 and M>=512. Values are condition medians; ratios are medians of matched per-pair rates. Each throughput cell lists Ferrum control -> candidate / llama.cpp and `[control/llama -> candidate/llama; candidate/control]`. First-token latency is absolute milliseconds for Ferrum control, candidate, and llama.cpp.

### Verified current-upstream production-selector A/B

| Workload | GPU route in candidate? | Prefill tok/s C->K / llama `[C/L->K/L; K/C]` | Cached decode tok/s C->K / llama `[C/L->K/L; K/C]` | Complete generation tok/s C->K / llama `[C/L->K/L; K/C]` | First-token ms C->K / llama | IDs C=K; K=L |
|---|:---:|---:|---:|---:|---:|---:|
| M=11, 17 generated | yes | 40.1->40.8 / 235.0 `[0.170->0.173; 1.018]` | 30.0->30.0 / 103.2 `[0.291->0.291; 1.000]` | 20.9->21.0 / 82.6 `[0.254->0.255; 1.005]` | 274.7->269.8 / 47.0 | 3/3; 0/3 |
| M=32, 17 generated | yes | 43.9->44.2 / 583.2 `[0.075->0.076; 1.002]` | 29.6->29.5 / 102.8 `[0.288->0.287; 0.999]` | 13.3->13.4 / 79.1 `[0.169->0.170; 1.002]` | 729.9->725.0 / 55.1 | 3/3; 0/3 |
| M=64, 17 generated | no | 260.3->261.6 / 1,028.5 `[0.252->0.254; 1.006]` | 29.8->29.8 / 102.7 `[0.291->0.291; 1.001]` | 21.6->21.6 / 76.4 `[0.283->0.283; 1.002]` | 246.1->245.0 / 62.5 | 3/3; 0/3 |
| M=128, 17 generated | no | 394.7->396.0 / 1,534.3 `[0.258->0.258; 1.000]` | 29.8->30.0 / 101.7 `[0.293->0.294; 1.004]` | 19.7->19.7 / 69.4 `[0.284->0.284; 1.001]` | 324.6->323.4 / 83.6 | 3/3; 0/3 |
| M=256, 17 generated | no | 170.7->171.1 / 1,946.1 `[0.088->0.088; 1.002]` | 29.8->29.8 / 102.5 `[0.290->0.291; 0.999]` | 8.3->8.3 / 58.1 `[0.143->0.143; 1.004]` | 1,499.6->1,496.3 / 131.8 | 3/3; 0/3 |
| M=512, 17 generated | yes | 472.2->490.2 / 2,150.9 `[0.220->0.228; 1.037]` | 29.5->29.5 / 103.0 `[0.286->0.286; 1.002]` | 10.4->10.7 / 42.8 `[0.243->0.249; 1.023]` | 1,084.6->1,044.9 / 238.3 | 3/3; 3/3 |
| M=1,024, 17 generated | yes | 519.2->522.8 / 2,161.3 `[0.241->0.241; 1.010]` | 29.3->29.2 / 102.1 `[0.287->0.286; 0.998]` | 6.7->6.7 / 26.8 `[0.250->0.250; 1.007]` | 1,972.4->1,959.1 / 474.0 | 3/3; 3/3 |
| Sustained decode, M=11, 129 generated | yes | 34.0->40.3 / 234.1 `[0.145->0.172; 1.186]` | 29.7->29.7 / 102.7 `[0.289->0.289; 1.001]` | 27.7->27.8 / 97.3 `[0.285->0.286; 1.005]` | 323.8->273.1 / 47.2 | 3/3; 0/3 |

The selected M=512 route raised absolute prefill from 472.2 to 490.2 tok/s versus 2,150.9 tok/s for llama.cpp (0.220x to 0.228x) and reduced first-token latency from 1,084.6 to 1,044.9 ms. At M=1,024, prefill moved from 519.2 to 522.8 tok/s versus 2,161.3 tok/s (0.241x to 0.241x at three decimals; 1.010x candidate/control), and first-token latency fell from 1,972.4 to 1,959.1 ms. The short M=11 case moved from 40.1 to 40.8 tok/s versus 235.0 tok/s; sustained prefill moved from 34.0 to 40.3 tok/s versus 234.1 tok/s, with first-token latency changing from 323.8 to 273.1 ms. Cached decode remained effectively flat. Ferrum control/candidate IDs matched in all 24 pairs; candidate/llama.cpp matched in 6/24, with all matches at 512 and 1,024 prompt tokens. Rows M=64–256 are control/candidate observations with CPU routing in both variants; their small deltas are run variance, not a selected route win.

The route reduced prefill command buffers from 23 to 1 at M=11 and M=32, from 133 to 67 at M=512, and from 222 to 112 at M=1,024. Candidate transient peaks at those shapes were 50.8, 101.6, 197.1, and 240.0 MiB; Ferrum process RSS stayed around 5.3 GiB. At M=512, all three candidate prefill pairs improved; at M=1,024, two of three improved. Granite's independent eight-pair internal check showed all eight pairs improve at both long sizes.

### Full-route LFM shape sweep (internal Ferrum comparison)

This first sweep temporarily enabled GPU routing for every multi-token shape to measure the crossover. It used the same pinned LFM GGUF in Ferrum control and candidate, three interleaved pairs per case. Its llama.cpp JSONL rows are omitted here because the run did not preserve the runner commit. Rates are Ferrum condition medians; ratios are medians of matched candidate/control rates. First-token latency shows both absolute values in milliseconds.

| Workload | Prefill tok/s C->K (K/C) | Cached decode tok/s C->K (K/C) | Complete generation tok/s C->K (K/C) | First-token ms C->K | IDs C=K |
|---|---:|---:|---:|---:|---:|
| M=11, 17 generated | 40.1->40.4 (1.011x) | 30.0->30.0 (0.999x) | 21.0->21.0 (1.005x) | 274.6->272.2 | 3/3 |
| M=32, 17 generated | 43.9->44.3 (1.011x) | 29.7->29.7 (1.001x) | 13.4->13.5 (1.005x) | 728.5->721.9 | 3/3 |
| M=64, 17 generated | 262.2->262.9 (0.999x) | 29.7->29.7 (1.002x) | 21.7->21.6 (0.998x) | 244.4->243.7 | 3/3 |
| M=128, 17 generated | 397.2->397.3 (0.995x) | 29.9->29.6 (0.991x) | 19.7->19.7 (0.998x) | 322.5->322.5 | 3/3 |
| M=256, 17 generated | 170.6->173.1 (1.015x) | 30.0->29.9 (1.000x) | 8.3->8.4 (1.009x) | 1,500.5->1,479.0 | 3/3 |
| M=512, 17 generated | 474.6->491.3 (1.029x) | 29.5->29.6 (1.005x) | 10.5->10.7 (1.021x) | 1,079.1->1,042.5 | 3/3 |
| M=1,024, 17 generated | 515.6->531.6 (1.029x) | 29.3->29.2 (0.998x) | 6.7->6.8 (1.020x) | 1,986.6->1,926.7 | 3/3 |
| Sustained decode, M=11, 129 generated | 33.2->40.1 (1.208x) | 29.7->29.7 (0.999x) | 27.7->28.0 (1.012x) | 331.6->274.6 | 3/3 |

This broad route experiment reduced command buffers from 23 to 1 at M=11/32, 23 to 3 at M=64, 23 to 4 at M=128, 89 to 45 at M=256, 133 to 67 at M=512, and 222 to 112 at M=1,024. Candidate transient peaks were 50.8, 101.6, 256.0, 254.5, 109.1, 197.1, and 240.0 MiB respectively for M=11 through 1,024. The 64- and 128-token shapes had no repeatable throughput win and peaked near the 256 MiB arena limit. M=256 improved modestly in LFM but peaked at 253.0 MiB in the Granite check below. These measurements support skipping the full middle range rather than retaining a small inconsistent exception.

For a second architecture check, Granite 3.1 1B-A400M Instruct used the official BF16 safetensors snapshot at revision `0da7a48b0276d500ce5922fd2b33944091fc6c09` (model.safetensors SHA-256 `ac02591061f1344027a7e7b11dbb4143f75f166c47dc09b742f5de3ab1dde1d1`). These are internal Ferrum control/candidate measurements only; they are not reported as a GGUF or a llama.cpp comparison. The separate bartowski Q4_K_M GGUF inspected for this architecture is pinned at revision `940d2e1f9f65330615c7c8e980e6c5ac73d3360c` (SHA-256 `3a2ec1c2a78cb29d901e29bbf5162dcd03381e13803d2cbdcff838d4d08142eb`), but Ferrum's matched runner does not yet load Granite MoE GGUF. The official tokenizer IDs used for these shape prompts match the GGUF token vocabulary. Eight interleaved internal pairs per shape combine the first three-pair run and a separate five-pair run. Rates are condition medians; candidate/control ratios are medians of paired rates. First-token values are absolute milliseconds; the delta is the median paired candidate-minus-control latency.

| Prompt M | Prefill tok/s C->K (K/C; K wins) | Cached decode tok/s C->K (K/C) | Complete generation tok/s C->K (K/C) | First-token ms C->K (paired delta) | IDs C=K | Prefill buffers C->K | Transient peak MiB C->K |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 11 | 139.57->145.28 (1.047x; 8/8) | 79.66->80.15 (1.002x) | 60.56->61.18 (1.011x) | 78.93->75.83 (-3.56 ms) | 8/8 | 25->1 | 1.27->30.98 |
| 32 | 163.25->166.20 (1.019x; 8/8) | 79.62->79.59 (1.003x) | 42.57->42.95 (1.009x) | 196.14->192.63 (-3.69 ms) | 8/8 | 25->1 | 2.60->62.83 |
| 64 | 169.01->170.79 (1.011x; 6/8) | 77.33->77.25 (0.999x) | 28.95->29.14 (1.006x) | 378.77->374.83 (-4.23 ms) | 8/8 | 25->1 | 5.32->128.41 |
| 128 | 172.37->172.78 (1.002x; 5/8) | 79.54->79.30 (0.999x) | 17.99->18.01 (1.001x) | 742.72->740.96 (-1.48 ms) | 8/8 | 25->2 | 11.13->254.16 |
| 256 | 173.16->174.21 (1.006x; 6/8) | 78.38->78.59 (1.002x) | 10.06->10.14 (1.007x) | 1,478.49->1,469.61 (-8.52 ms) | 8/8 | 25->3 | 24.27->252.97 |
| 512 | 170.54->171.73 (1.006x; 8/8) | 76.56->76.87 (1.003x) | 5.27->5.30 (1.006x) | 3,002.40->2,981.60 (-17.84 ms) | 8/8 | 97->49 | 35.03->56.56 |
| 1,024 | 171.10->171.84 (1.005x; 8/8) | 72.51->72.43 (1.002x) | 2.73->2.74 (1.005x) | 5,985.08->5,959.02 (-28.79 ms) | 8/8 | 193->97 | 87.03->116.06 |

Granite corroborated the LFM shape selection: prefill improved in all eight pairs at M=11, 32, 512, and 1,024. The long-prompt absolute rates were 170.54->171.73 tok/s at M=512 and 171.10->171.84 tok/s at M=1,024, with first-token latency falling by median paired deltas of 17.84 and 28.79 ms. At M=64 and 128, memory rose to 128.41 and 254.16 MiB without a repeatable throughput gain. M=256 improved in six of eight pairs but peaked at 252.97 MiB, so the selector excludes it. All eight candidate outputs matched their controls at every shape. The production policy is therefore `M <= 32 || M >= 512`, with M=1 unchanged. The selector is shape-based and model-agnostic; it does not depend on expert count or model name.

The runner's `moe_gpu_routing_prefill` field retains the paired control for this shape test; its normal Metal-device default now selects the measured ranges. The focused selector and GPU route ordering/weight tests passed with Metal API Validation and GPU Shader Validation enabled. The verified current-upstream LFM A/B and complete eight-shape source matrix are `lfm2.5-8b-a1b-gpu-route-prefill-current84e76d8-ab.jsonl` and `lfm2.5-8b-a1b-gpu-route-prefill-current84e76d8-workloads.jsonl`, with the runner output in `.run.log`. The earlier full-route Ferrum internal sweep is preserved as `lfm2.5-8b-a1b-gpu-route-prefill-ab.jsonl` and `lfm2.5-8b-a1b-gpu-route-prefill-shape-ab.jsonl`, with their source matrices and logs in `docs/measurements/phase7a/`; their llama.cpp rows are excluded because that run did not record the executable commit. Granite internal A/B files are `granite-3.1-1b-a400m-bf16-gpu-route-prefill-shape-ab.jsonl` and `granite-3.1-1b-a400m-bf16-gpu-route-prefill-shape-more-ab.jsonl`; all source workload matrices and logs are in the same directory. This improvement does not clear the overall GGUF performance gate, and large Qwen targets remain locked.

## Experiment 31: Cooperative TensorOps fusion for short-prefill attention (rejected before A/B)

The candidate fused a complete causal attention tile for BF16 full-prefill shapes with 2–32 rows: TensorOps QK, cooperative row max/sum and softmax, then a cooperative tensor as the left input to the value product. The intended selector excluded decode and M=1. The prototype preserved the existing score, scale, and probability BF16 rounding points and included GQA head mapping.

The current Xcode 27.0/macOS 27.0 SDK compiled the corrected kernel, but the first M=2 GQA dispatch did not complete under Metal API Validation and GPU Shader Validation. After more than two minutes the host was still blocked in `MTLCommandBuffer.waitUntilCompleted`; the run was interrupted. No throughput or latency sample was produced, so there is no A/B result or performance claim. The experimental host path and shader were removed rather than retained without completed validation. The verified conventional MSL softmax and the existing MPP score/context products remain selected; the existing attention-prefix softmax optimization remains unchanged. M=1 still uses its existing direct path.

The validation attempt and outcome are recorded in `docs/measurements/phase7a/attention-fused-short-mpp-validation.log`. No model benchmark was launched. The relevant current APIs remain documented above: cooperative inputs require a compatible fixed inner dimension, and native quantized TensorOps formats do not directly preserve GGUF's arbitrary per-block scales/minima.

## Experiment 32: N=128 output tile for grouped Q4_K/Q6_K expert TensorOps (rejected)

Status: rejected and removed. This compared N=64 with N=128 for the existing K=64 grouped Q4_K/Q6_K expert `matmul2d` kernels. The candidate doubled each threadgroup's dequantized BF16 weight tile from 8 KiB to 16 KiB, aiming to reuse each staged expert tile across more output columns. It kept the K tile, M=32 tile, quantization, dispatch-density threshold, output conversion, and expert ordering fixed. Sparse batches and M=1 direct MSL GEMV remained on their existing paths.

Correctness used 197 assignments, three experts, 65 output rows per expert, and K=512. Both N=64 control and N=128 candidate matched the scalar reference for Q4_K and Q6_K, including the partial expert-row tail. The focused tests passed with Metal API Validation and GPU Shader Validation enabled. The release benchmark then ran five interleaved Ferrum control/candidate pairs for the same short, 128-, 512-, 1,024-token, and sustained-decode workloads. It used the pinned LiquidAI Q4_K_M GGUF (revision `49c14831707011e64d70b2ebd8462ba08d608434`, SHA-256 `4923ec14f06b968b74d663e5949867d2d9c3bf13a20b8be1a9f9af39989b2bb0`) and the saved current-upstream llama.cpp matrix at `84e76d8a23162eca70490da131945ebec1f09bf4`. The saved llama.cpp rows use the same prompt IDs and generated-token counts; llama.cpp was not rerun just to repeat its reference numbers.

Cells show the absolute median tok/s for Ferrum control -> candidate / llama.cpp, followed by the median paired candidate/control ratio and the candidate/llama.cpp ratio. First-token latency lists absolute milliseconds for control -> candidate / llama.cpp. The saved reference has three llama.cpp pairs per workload; the internal candidate/control run has five.

| Workload | Prefill tok/s C -> K / llama (K/C; K/llama) | Cached decode tok/s C -> K / llama (K/C; K/llama) | Complete generation tok/s C -> K / llama (K/C; K/llama) | First-token ms C -> K / llama | IDs C=K; K matches llama |
|---|---:|---:|---:|---:|---:|
| Short, M=11 | 40.10 -> 40.16 / 216.62 (1.002; 0.185) | 29.95 -> 29.99 / 94.66 (1.002; 0.317) | 20.92 -> 20.98 / 74.80 (1.003; 0.281) | 274.63 -> 274.21 / 50.99 | 5/5; 0/5 |
| 128-token prompt | 394.58 -> 339.65 / 1,334.16 (0.861; 0.255) | 29.93 -> 29.92 / 88.31 (1.000; 0.339) | 19.65 -> 18.57 / 60.55 (0.944; 0.307) | 324.67 -> 377.15 / 96.17 | 5/5; 0/5 |
| 512-token prompt | 480.56 -> 416.42 / 1,973.58 (0.863; 0.211) | 29.38 -> 29.33 / 94.68 (0.999; 0.310) | 10.53 -> 9.56 / 39.16 (0.904; 0.244) | 1,065.73 -> 1,229.82 / 259.68 | 5/5; 5/5 |
| 1,024-token prompt | 524.04 -> 455.44 / 1,986.65 (0.870; 0.229) | 29.33 -> 29.35 / 93.58 (0.999; 0.314) | 6.76 -> 6.05 / 24.59 (0.896; 0.246) | 1,954.35 -> 2,248.67 / 515.68 | 5/5; 5/5 |
| Sustained decode, M=11 prompt | 40.10 -> 40.08 / 216.51 (0.998; 0.185) | 29.74 -> 29.76 / 94.36 (1.001; 0.315) | 28.00 -> 28.00 / 89.00 (1.001; 0.315) | 274.62 -> 274.75 / 51.04 | 5/5; 0/5 |

N=128 reduced medium/long prefill from 394.58 to 339.65 tok/s at 128 tokens, 480.56 to 416.42 tok/s at 512, and 524.04 to 455.44 tok/s at 1,024. Those are 13–14% regressions, with first-token latency increasing from 324.67 to 377.15 ms, 1,065.73 to 1,229.82 ms, and 1,954.35 to 2,248.67 ms respectively. Complete-generation throughput fell by 5.6–10.4%; cached decode was flat. Control and candidate output IDs matched in all 25 comparisons. Prefill transient peaks were identical per shape: 34.04 MiB at 128, 160.05 MiB at 512, and 240.03 MiB at 1,024. Larger threadgroup staging may reduce occupancy, but this run did not isolate occupancy as the cause. The N=128 kernels and selector were removed; production keeps the measured N=64 grouped expert path and direct MSL M=1 path.

Raw internal results and full runner output are `lfm2.5-8b-a1b-expert-mpp-n128-ab.jsonl` and `lfm2.5-8b-a1b-expert-mpp-n128-ab.run.log`. Workloads are `lfm2.5-8b-a1b-q4_k_m-workloads.jsonl`; the llama.cpp reference is `lfm2.5-8b-a1b-threshold8-current.jsonl`. This experiment provides no performance win and does not change the Phase 7 gate.

## Experiment 33: Q5_1 projection TensorOps boundary at M=8 (retained)

Status: retained for eligible Q5_1 GGUF projections. Lowering only the Q5_1 MPP row boundary from 16 to 8 raised prefill and complete-generation throughput on both tested architectures for M=8, 12, and 15. At M=16 and above, the control and candidate already select MPP; M=1 remains on the direct MSL GEMV path. The selector depends on format, shape, dtype, K alignment, and MPP capability, never model identity.

The Qwen2.5 test used the pinned Q5_K_M GGUF in the artifact table above (SHA-256 `041474553fcabfc2a2d67903f9d2c2e50bd92528e670da4f33b5d0ce6e59fd55`), five interleaved control/candidate/llama.cpp pairs per shape, and the exact matched Qwen ChatML token IDs. Control used the automatic M=16 boundary from the pre-change binary; candidate set `gguf_mpp_q5_1_min_rows=8`. Prompts were M=8 (empty-user shape probe), 12 (“Can you help?”), 15 (“What causes tides on Earth?”), 16, and 21 tokens; each generated 16 tokens.

For a second architecture, Qwen3-0.6B used official safetensors at revision `c1899de289a04d12100db370d81485cdf75e47ca` (model.safetensors SHA-256 `f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b`). A community Q5_K_M GGUF was rejected because Ferrum found it missing `tokenizer.ggml.bos_token_id`; the failed load is recorded in `profiles/qwen3-0.6b-q5_k_m-q5_1-format-profile.run.log`. The diagnostic GGUF was instead converted from official safetensors with the pinned llama.cpp converter and quantized with `llama-quantize --pure ... Q5_1`; it is not represented as an official published Q5_K_M artifact. The ordinary Q5_K_M quantization mix uses Q5_K/Q6_K projection tensors and does not exercise the Q5_1 boundary. The diagnostic GGUF SHA-256 is `6b17d9409c78626f516601aac473a69ba1559c19bb55c4c2f55d331d9149e6d5`. The five-pair matched run used explicit M=16 control and M=8 candidate boundaries, the Qwen3 official tokenizer, and llama.cpp `84e76d8a23162eca70490da131945ebec1f09bf4`. It covered M=8, 12, 15, 16, 128, 512, 1,024, plus a 21-token prompt with 129 generated tokens for sustained decode. All Ferrum/llama.cpp runs loaded the same derived GGUF bytes; prompt token IDs matched exactly.

Rates below are condition medians in tok/s. Bracketed ratios are medians of matched per-pair rates, written `[control/llama.cpp -> candidate/llama.cpp; candidate/control]`; first-token values are absolute milliseconds. ID counts are `control=candidate; candidate=llama.cpp; control=llama.cpp`.

| Qwen2.5 workload | Prefill tok/s C -> K / llama `[C/L -> K/L; K/C]` | Cached decode tok/s C -> K / llama `[C/L -> K/L; K/C]` | Complete generation tok/s C -> K / llama `[C/L -> K/L; K/C]` | First-token ms C -> K / llama | IDs C=K; K=L; C=L |
|---|---:|---:|---:|---:|---:|
| M=8, empty-user shape probe | 187.64 -> 287.96 / 886.91 `[0.212 -> 0.323; 1.538]` | 135.65 -> 134.83 / 218.81 `[0.617 -> 0.615; 1.008]` | 101.74 -> 111.96 / 198.05 `[0.512 -> 0.565; 1.100]` | 42.95 -> 28.09 / 9.28 | 0/5; 5/5; 0/5 |
| M=12 | 207.41 -> 411.41 / 792.65 `[0.264 -> 0.519; 1.985]` | 132.75 -> 134.48 / 221.77 `[0.594 -> 0.604; 1.016]` | 91.67 -> 110.62 / 186.84 `[0.493 -> 0.592; 1.205]` | 58.15 -> 29.48 / 15.40 | 0/5; 0/5; 0/5 |
| M=15 | 216.78 -> 508.78 / 961.07 `[0.225 -> 0.529; 2.347]` | 135.00 -> 135.87 / 229.45 `[0.589 -> 0.593; 1.004]` | 86.84 -> 111.17 / 185.78 `[0.467 -> 0.598; 1.281]` | 69.50 -> 29.78 / 15.85 | 5/5; 5/5; 5/5 |
| M=16 | 545.46 -> 541.88 / 1,010.82 `[0.538 -> 0.536; 0.991]` | 134.06 -> 133.88 / 224.92 `[0.596 -> 0.594; 0.997]` | 110.42 -> 109.68 / 185.70 `[0.596 -> 0.592; 0.993]` | 29.65 -> 29.83 / 16.07 | 5/5; 5/5; 5/5 |
| M=21 | 709.45 -> 701.54 / 1,385.52 `[0.511 -> 0.506; 0.991]` | 134.38 -> 133.76 / 231.21 `[0.587 -> 0.588; 0.995]` | 110.20 -> 109.72 / 187.86 `[0.586 -> 0.586; 0.996]` | 29.90 -> 30.24 / 15.43 | 5/5; 5/5; 5/5 |

| Qwen3 workload | Prefill tok/s C -> K / llama `[C/L -> K/L; K/C]` | Cached decode tok/s C -> K / llama `[C/L -> K/L; K/C]` | Complete generation tok/s C -> K / llama `[C/L -> K/L; K/C]` | First-token ms C -> K / llama | IDs C=K; K=L; C=L |
|---|---:|---:|---:|---:|---:|
| M=8, empty-user shape probe | 139.95 -> 257.52 / 760.47 `[0.183 -> 0.341; 1.841]` | 133.35 -> 132.43 / 200.03 `[0.667 -> 0.665; 0.996]` | 92.20 -> 108.01 / 179.51 `[0.511 -> 0.604; 1.171]` | 57.46 -> 31.37 / 10.76 | 5/5; 5/5; 5/5 |
| M=12 | 148.58 -> 377.82 / 680.77 `[0.219 -> 0.550; 2.546]` | 131.29 -> 132.62 / 200.26 `[0.656 -> 0.661; 1.006]` | 80.59 -> 107.31 / 166.86 `[0.483 -> 0.641; 1.330]` | 81.10 -> 32.07 / 17.88 | 5/5; 5/5; 5/5 |
| M=15 | 149.82 -> 469.83 / 831.63 `[0.179 -> 0.566; 3.157]` | 131.54 -> 132.57 / 199.71 `[0.659 -> 0.662; 1.005]` | 73.77 -> 107.85 / 165.28 `[0.445 -> 0.652; 1.462]` | 100.44 -> 32.23 / 18.28 | 5/5; 5/5; 5/5 |
| M=16 | 501.51 -> 501.85 / 878.97 `[0.570 -> 0.570; 1.004]` | 132.22 -> 131.66 / 199.72 `[0.664 -> 0.658; 0.996]` | 107.11 -> 106.96 / 166.51 `[0.643 -> 0.643; 0.998]` | 32.20 -> 32.18 / 18.44 | 5/5; 5/5; 5/5 |
| M=21, sustained 129-token generation | 629.88 -> 639.41 / 1,217.92 `[0.514 -> 0.525; 1.021]` | 126.68 -> 126.53 / 202.16 `[0.626 -> 0.625; 0.999]` | 119.78 -> 119.53 / 189.65 `[0.631 -> 0.629; 1.001]` | 33.64 -> 33.14 / 17.48 | 5/5; 0/5; 0/5 |
| M=128 | 2,794.15 -> 2,755.03 / 5,963.27 `[0.470 -> 0.462; 0.978]` | 130.45 -> 131.64 / 198.23 `[0.658 -> 0.659; 1.006]` | 97.53 -> 97.18 / 159.81 `[0.611 -> 0.609; 0.998]` | 46.13 -> 46.78 / 21.71 | 5/5; 5/5; 5/5 |
| M=512 | 4,108.76 -> 4,104.80 / 6,872.11 `[0.594 -> 0.598; 1.001]` | 117.66 -> 117.69 / 187.15 `[0.630 -> 0.628; 1.000]` | 60.57 -> 60.60 / 101.28 `[0.597 -> 0.597; 1.000]` | 124.92 -> 125.07 / 74.76 | 5/5; 5/5; 5/5 |
| M=1,024 | 4,227.52 -> 4,247.02 / 6,068.61 `[0.697 -> 0.699; 0.999]` | 102.11 -> 101.92 / 168.15 `[0.607 -> 0.607; 1.001]` | 39.36 -> 39.30 / 61.23 `[0.643 -> 0.643; 0.998]` | 242.52 -> 241.48 / 169.02 | 5/5; 5/5; 5/5 |

At M=8, 12, and 15, the Qwen2.5 candidate won prefill and complete-generation throughput in all five pairs. Its absolute prefill rose by 100.32, 204.00, and 292.00 tok/s; complete generation rose by 10.22, 18.95, and 24.33 tok/s. First-token latency fell by 14.86, 28.67, and 39.72 ms. Qwen3 independently showed the same boundary: all five candidate pairs won prefill and complete generation at M=8, 12, and 15. Its absolute prefill rose by 117.57, 229.24, and 320.01 tok/s; complete generation rose by 15.81, 26.72, and 34.08 tok/s. First-token latency fell by 26.09, 49.03, and 68.21 ms. At M=16 and above the variants select the same projection route; the observed small deltas are measurement noise. Cached decode remains effectively flat because M=1 keeps its direct GEMV kernels.

Output IDs matched between Qwen2.5 control and candidate in 15/25 pairs: both variants diverged on M=8 and M=12, while all M=15, 16, and 21 pairs matched. Candidate/reference matches were 20/25; the M=12 case had no exact reference matches. Qwen3 control/candidate outputs matched in all 40 pairs; candidate/reference matched in 35/40, with all five differences on the 129-token sustained-decode case. These differences remain visible in raw IDs and do not change the tested numerical tolerances. Qwen3 Ferrum RSS peaked at 1.055 GiB and transient prefill storage at 256 MiB; control and candidate had the same transient peak per shape, with no material memory increase.

The Qwen3 M=1 profile selected direct `lm_head.q5_1_gemv_n4`; M=1 stays outside TensorOps. At M=1,024 the profile selected Q5_1 MPP GEMM for Q/K/V/O and gate/up/down projections (28 calls per projection, one per layer), confirming this architecture exercises the quantized prefill path. The API-validation coverage for the retained MPP tail and dispatch cases is in `q5_1-mpp-boundary-validation.log`. The Qwen2.5 boundary A/B is `qwen2.5-q5_k_m-q5_1-mpp-boundary-chat-ab.jsonl`; its mixed Q5_0/Q5_1 exploratory sweeps remain preserved as `qwen2.5-q5_k_m-q5-nonk-mpp-boundary-ab.jsonl` and `qwen2.5-q5_k_m-q5-nonk-mpp-chat-ab.jsonl`. The cross-architecture A/B, profile, exact workload IDs, and runner logs are `qwen3-0.6b-q5_1-mpp-boundary-chat-ab.jsonl`, `profiles/qwen3-0.6b-q5_1-mpp-profile.jsonl`, `qwen3-0.6b-q5_1-mpp-boundary-chat-workloads.jsonl`, and their `.run.log` files in `docs/measurements/phase7a/`. These short-shape gains do not clear the overall Phase 7 gate; large-target tuning remains deferred.


## Experiment 34: Sixteen-row Q4_K sparse-expert M=1 projection (retained)

Status: retained by default for Q4_K expert fallbacks with at least 128 output rows and K at least 256 and divisible by 256. The selector runs after grouped expert MPP/TensorOps eligibility, so routed prefill batches that qualify for MPP are unchanged. Sparse expert GEMV remains conventional MSL: a one-token assignment batch does not provide a useful TensorOps tile. The candidate assigns four adjacent output rows to each SIMD group, reuses each activation load across those rows, and preserves the current increasing-K accumulation order for every output. Dispatch uses only quant format and shape.

The focused correctness test covers 3 experts, 4 assignments, 133 output rows per expert, K=512, and the partial output-row tail. It compares the scalar fallback, retained 8-row kernel, and 16-row candidate bit-for-bit. It also checks the 128-row minimum and K-alignment selector boundaries. The test passed with Metal API Validation and GPU Shader Validation enabled; see `lfm2.5-8b-a1b-expert-q4k16rows-validation.log`.

The release A/B used the pinned LFM2.5-8B-A1B Q4_K_M artifact (revision `49c14831707011e64d70b2ebd8462ba08d608434`, SHA-256 `4923ec14f06b968b74d663e5949867d2d9c3bf13a20b8be1a9f9af39989b2bb0`), five interleaved control/candidate pairs for each of the five saved workloads, and exact shared prompt IDs. The only changed request field was `q4_k_expert_project_16rows`; control selected the existing 8-row direct kernel, candidate selected 16 rows. The saved llama.cpp comparison is from `lfm2.5-8b-a1b-expert-q4k8rows-ab.jsonl` at official revision `84e76d8a23162eca70490da131945ebec1f09bf4`; its five reference rows per workload use the same prompt IDs and generation counts. Ferrum control/candidate ratios are medians of paired rates. Because the llama.cpp rows are from that saved reference matrix rather than this internal A/B process, Ferrum/llama.cpp ratios below use displayed condition medians. All rates are tok/s and first-token latency is in milliseconds.

| Workload | Prefill tok/s C -> 16 / llama `[C/L -> 16/L; 16/C]` | Cached decode tok/s C -> 16 / llama `[C/L -> 16/L; 16/C]` | Complete generation tok/s C -> 16 / llama `[C/L -> 16/L; 16/C]` | First-token ms C -> 16 / llama | IDs C=16; 16=L |
|---|---:|---:|---:|---:|---:|
| Short, M=11 | 40.13 -> 41.64 / 235.96 `[0.170 -> 0.176; 1.047]` | 29.54 -> 31.52 / 103.21 `[0.286 -> 0.305; 1.067]` | 20.60 -> 21.69 / 82.24 `[0.250 -> 0.264; 1.067]` | 274.39 -> 264.48 / 46.82 | 5/5; 0/5 |
| 128-token prompt | 382.60 -> 393.80 / 1,539.81 `[0.248 -> 0.256; 1.035]` | 29.63 -> 31.34 / 103.10 `[0.287 -> 0.304; 1.054]` | 19.33 -> 20.26 / 70.16 `[0.275 -> 0.289; 1.036]` | 334.86 -> 325.34 / 83.33 | 5/5; 0/5 |
| 512-token prompt | 438.02 -> 441.92 / 2,148.83 `[0.204 -> 0.206; 0.998]` | 26.29 -> 29.19 / 101.36 `[0.259 -> 0.288; 1.060]` | 9.80 -> 10.01 / 42.28 `[0.232 -> 0.237; 1.014]` | 1,169.18 -> 1,158.87 / 238.47 | 5/5; 5/5 |
| 1,024-token prompt | 512.52 -> 522.65 / 2,166.65 `[0.237 -> 0.241; 1.001]` | 29.30 -> 30.83 / 100.87 `[0.290 -> 0.306; 1.052]` | 6.61 -> 6.81 / 26.72 `[0.247 -> 0.255; 1.013]` | 1,998.28 -> 1,959.53 / 472.85 | 5/5; 5/5 |
| Sustained decode, M=11 prompt | 40.07 -> 43.69 / 233.08 `[0.172 -> 0.187; 1.091]` | 29.63 -> 31.29 / 102.59 `[0.289 -> 0.305; 1.056]` | 27.83 -> 29.49 / 97.07 `[0.287 -> 0.304; 1.059]` | 274.83 -> 252.06 / 47.41 | 5/5; 0/5 |

The candidate improved cached decode in every workload: absolute condition medians rose by 1.53–2.90 tok/s, and the paired candidate/control ratios ranged from 1.052x to 1.067x. Complete-generation throughput rose by 0.21–1.66 tok/s; control and candidate produced identical IDs in all 25 pairs. Prefill was effectively flat at the 512- and 1,024-token shapes, where grouped MPP already handles eligible routed batches. First-token latency fell by 9.9 ms on the short prompt and by 38.7 ms in sustained decode; longer-prompt deltas were smaller than the total prefill latency. Process RSS and transient prefill peak were identical between variants (RSS 5.26–5.31 GiB; transient peak 4.05–240.03 MiB depending on prompt). Against the saved current-upstream reference, candidate cached decode is 0.288x at 512 tokens and 0.306x at 1,024 tokens, so this local M=1 improvement does not clear the LFM or Phase 7 gate. The remaining LFM gap includes grouped expert MPP work and other M=1 projections; this experiment changes neither.

Raw internal measurements and runner output are `lfm2.5-8b-a1b-expert-q4k16rows-ab.jsonl` and `.run.log`; the saved llama.cpp rows are in `lfm2.5-8b-a1b-expert-q4k8rows-ab.jsonl`. The workload is `lfm2.5-8b-a1b-q4_k_m-workloads.jsonl`. The kernel, selector, and request toggle are in `src/metal/shaders/ops.metal`, `src/metal/mod.rs`, `src/ops/mod.rs`, and `examples/phase55_bench.rs`.

## Experiment 35: Dense Q4_K sixteen-row M=1 GEMV (rejected)

Status: rejected and removed. This was a conventional direct-MSL M=1 GEMV candidate, not a TensorOps kernel. Four adjacent output rows shared each SIMD group's activation loads, with a 16-row threadgroup tile. M=1 remained on direct shaders; it was not sent through `matmul2d`.

The first five-pair A/B enabled the candidate for dense Q4_K output widths `N >= 128` and ran the pinned Qwen2.5-0.5B Q4_K_M and LFM2.5-8B-A1B Q4_K_M artifacts against current official llama.cpp `1ab7e5ad2d4e7295c94c3b966a3e0b70fa365865`. It changed only `q4_k_gemv_16rows`; sparse expert projection settings were held fixed. Qwen's Q4_K down projections have N=896. LFM has dense Q4_K output widths including N=512, 2,048, 6,144, and 7,168; the separate sparse-expert kernel was not part of this candidate. Absolute throughput and latency are condition medians. Ratios in brackets are matched-pair medians. Each rate cell is Ferrum control -> candidate / llama.cpp tok/s, followed by `[control/llama -> candidate/llama; candidate/control]`.

### Qwen2.5-0.5B Q4_K_M, broad `N >= 128` candidate

| Workload | Prefill tok/s C -> 16 / llama | Cached decode tok/s C -> 16 / llama | Complete generation tok/s C -> 16 / llama | First-token ms C -> 16 / llama | IDs C=16; 16=L; C=L |
|---|---:|---:|---:|---:|---:|
| Short | 722.99 -> 684.01 / 1,325.96 `[0.537 -> 0.501; 0.964]` | 132.02 -> 122.44 / 198.31 `[0.624 -> 0.617; 0.990]` | 105.58 -> 104.03 / 172.48 `[0.609 -> 0.603; 0.991]` | 29.3 -> 31.1 / 16.1 | 5/5; 0/5; 0/5 |
| 128-token prompt | 3,218.70 -> 3,232.94 / 6,160.83 `[0.495 -> 0.536; 1.036]` | 124.55 -> 123.98 / 202.25 `[0.609 -> 0.613; 1.017]` | 99.67 -> 101.69 / 169.76 `[0.591 -> 0.592; 1.008]` | 40.1 -> 39.9 / 21.0 | 5/5; 5/5; 5/5 |
| 512-token prompt | 4,660.94 -> 4,657.46 / 8,551.79 `[0.542 -> 0.545; 0.998]` | 120.18 -> 119.91 / 192.16 `[0.623 -> 0.619; 1.008]` | 68.90 -> 68.76 / 121.65 `[0.567 -> 0.566; 0.998]` | 110.2 -> 110.2 / 60.1 | 5/5; 5/5; 5/5 |
| 1,024-token prompt | 4,753.58 -> 4,771.07 / 8,007.22 `[0.592 -> 0.596; 1.008]` | 110.19 -> 109.96 / 200.69 `[0.548 -> 0.549; 0.999]` | 46.59 -> 46.57 / 81.47 `[0.572 -> 0.570; 1.006]` | 215.7 -> 214.9 / 128.1 | 5/5; 5/5; 5/5 |
| Sustained decode | 678.65 -> 678.97 / 1,289.27 `[0.524 -> 0.543; 1.000]` | 120.56 -> 120.41 / 214.69 `[0.562 -> 0.561; 1.003]` | 115.44 -> 115.91 / 197.58 `[0.584 -> 0.590; 1.005]` | 31.2 -> 31.2 / 16.5 | 5/5; 0/5; 0/5 |

### LFM2.5-8B-A1B Q4_K_M, broad `N >= 128` candidate

| Workload | Prefill tok/s C -> 16 / llama | Cached decode tok/s C -> 16 / llama | Complete generation tok/s C -> 16 / llama | First-token ms C -> 16 / llama | IDs C=16; 16=L; C=L |
|---|---:|---:|---:|---:|---:|
| Short | 43.62 -> 43.84 / 235.16 `[0.186 -> 0.186; 1.005]` | 31.69 -> 32.32 / 103.22 `[0.306 -> 0.313; 1.021]` | 22.38 -> 22.68 / 82.64 `[0.271 -> 0.275; 1.013]` | 252.4 -> 251.2 / 47.0 | 5/5; 0/5; 0/5 |
| 128-token prompt | 395.97 -> 395.29 / 1,539.16 `[0.257 -> 0.257; 1.005]` | 31.60 -> 32.16 / 102.36 `[0.309 -> 0.315; 1.021]` | 20.38 -> 20.65 / 69.69 `[0.293 -> 0.297; 1.014]` | 323.5 -> 324.1 / 83.4 | 5/5; 0/5; 0/5 |
| 512-token prompt | 474.88 -> 483.74 / 2,151.24 `[0.221 -> 0.225; 1.014]` | 30.95 -> 31.56 / 101.52 `[0.306 -> 0.312; 1.020]` | 10.65 -> 10.84 / 42.50 `[0.250 -> 0.255; 1.014]` | 1,078.4 -> 1,058.7 / 238.2 | 5/5; 5/5; 5/5 |
| 1,024-token prompt | 520.06 -> 513.58 / 2,169.34 `[0.240 -> 0.237; 0.987]` | 30.94 -> 31.46 / 101.59 `[0.305 -> 0.310; 1.017]` | 6.80 -> 6.74 / 26.82 `[0.254 -> 0.251; 0.991]` | 1,969.3 -> 1,994.1 / 472.3 | 5/5; 5/5; 5/5 |
| Sustained decode | 37.84 -> 43.46 / 233.97 `[0.161 -> 0.186; 1.155]` | 31.42 -> 31.99 / 102.18 `[0.307 -> 0.313; 1.018]` | 29.35 -> 30.14 / 96.68 `[0.303 -> 0.311; 1.027]` | 291.0 -> 253.4 / 47.3 | 5/5; 0/5; 0/5 |

The broad candidate improved LFM cached decode in all five workload medians by 0.52-0.65 tok/s, but Qwen was neutral or slower in four of five workload medians; its short cached decode moved from 132.02 to 122.44 tok/s and first-token latency from 29.3 to 31.1 ms. To test whether the LFM gains belonged to wider output shapes, a second five-pair A/B enabled the candidate only for dense Q4_K `N >= 1,024`. This selector was tested for that run only and was not retained. It used the same LFM2.5 artifact and llama.cpp commit. The table again gives condition-median tok/s and latency, with matched-pair ratios.

| Workload | Prefill tok/s C -> 16 / llama | Cached decode tok/s C -> 16 / llama | Complete generation tok/s C -> 16 / llama | First-token ms C -> 16 / llama | IDs C=16; 16=L; C=L |
|---|---:|---:|---:|---:|---:|
| Short | 42.18 -> 42.18 / 225.75 `[0.187 -> 0.186; 0.998]` | 30.55 -> 31.05 / 95.37 `[0.321 -> 0.326; 1.013]` | 21.57 -> 21.74 / 77.27 `[0.279 -> 0.282; 1.010]` | 261.1 -> 261.1 / 48.9 | 5/5; 0/5; 0/5 |
| 128-token prompt | 372.79 -> 370.77 / 1,385.71 `[0.260 -> 0.252; 0.971]` | 30.09 -> 30.42 / 90.14 `[0.317 -> 0.323; 1.015]` | 19.27 -> 18.70 / 60.90 `[0.299 -> 0.301; 0.997]` | 343.6 -> 345.5 / 92.6 | 5/5; 0/5; 0/5 |
| 512-token prompt | 436.40 -> 460.80 / 2,050.11 `[0.212 -> 0.224; 1.056]` | 29.57 -> 29.80 / 95.35 `[0.310 -> 0.312; 1.015]` | 9.92 -> 10.36 / 40.29 `[0.245 -> 0.256; 1.045]` | 1,173.5 -> 1,111.4 / 250.0 | 5/5; 5/5; 5/5 |
| 1,024-token prompt | 499.58 -> 486.02 / 2,009.21 `[0.249 -> 0.243; 0.973]` | 29.83 -> 28.95 / 92.33 `[0.323 -> 0.328; 0.969]` | 6.50 -> 6.30 / 24.86 `[0.262 -> 0.259; 0.969]` | 2,050.0 -> 2,107.2 / 509.9 | 5/5; 5/5; 5/5 |
| Sustained decode | 33.54 -> 41.06 / 223.95 `[0.150 -> 0.183; 1.256]` | 30.19 -> 30.69 / 94.33 `[0.320 -> 0.325; 1.017]` | 27.84 -> 28.35 / 88.74 `[0.316 -> 0.323; 1.030]` | 328.3 -> 268.2 / 49.3 | 5/5; 0/5; 0/5 |

The narrower candidate gained cached decode on four cases but fell from 29.83 to 28.95 tok/s on the 1,024-token prompt (0.969x paired median); complete generation fell from 6.50 to 6.30 tok/s and first-token latency rose from 2,050.0 to 2,107.2 ms. Control and candidate output IDs matched in all 25 LFM pairs, and sampled process memory was unchanged between variants. This was not a repeatable all-workload win, so the 16-row dense kernel, dispatch toggle, pipeline, and benchmark option were removed. Dense Q4_K M=1 stays on the established 8-row direct MSL path. The retained sparse-expert Q4_K 16-row kernel in Experiment 34 is a separate path with its own measured result. No M=1 case was routed through TensorOps.

Raw rows and run logs for the broad Qwen, broad LFM, and narrowed LFM A/Bs are `qwen2.5-q4_k_m-q4k16gemv-current-ab.{jsonl,run.log}`, `lfm2.5-8b-a1b-q4k16gemv-current-ab.{jsonl,run.log}`, and `lfm2.5-8b-a1b-q4k16gemv-n1024-current-ab.{jsonl,run.log}` in `docs/measurements/phase7a/`. The correctness result for the temporary 1,025-row candidate is from the focused run with Metal API Validation and GPU Shader Validation enabled; the retained code was removed after the A/B disposition.

## Experiment 36: N=32 output tiles for grouped Q4_K/Q6_K expert TensorOps (rejected)

Status: rejected and removed. The candidate kept the accepted M=32 and K=64 expert TensorOps tiles and changed N from 64 to 32 for grouped Q4_K/Q6_K projections. This tested whether halving each output tile and dequantized weight work per threadgroup could help expert prefill. The A/B changed only this tile width; Q5_K stayed on its existing K=128 TensorOps kernel, and sparse/M=1 expert work stayed on direct MSL.

Correctness used the existing 197-assignment, three-expert, N=65, K=512 Q4_K and Q6_K cases, which exercise an output tail across the N=32 and N=64 grid widths. Both candidates matched the scalar reference with Metal API Validation and GPU Shader Validation enabled. The five-pair Ferrum A/B used the pinned LFM2.5-8B-A1B Q4_K_M artifact and its saved five-case workload. The saved llama.cpp reference matrix uses the same prompt IDs and generation lengths at official revision `1ab7e5ad2d4e7295c94c3b966a3e0b70fa365865`; its JSONL was reused rather than rerunning llama.cpp.

The short M=11 and sustained-decode cases do not reach this model's expert TensorOps selector threshold of 256 assignment rows (32 experts x 8 routes); they are neighboring-shape controls. The N=32 setting affects only the 128-, 512-, and 1,024-token prefill cases.

Rates are condition medians in tok/s. The first two ratios in each rate cell are condition-median Ferrum/llama.cpp ratios; the final ratio is the median of the five paired N=32/N=64 rates. First-token latency is in milliseconds. `N64=N32` counts exact generated-sequence matches across five pairs.

| Workload | Prefill tok/s N64 -> N32 / llama `[N64/L -> N32/L; N32/N64]` | Cached decode tok/s N64 -> N32 / llama `[N64/L -> N32/L; N32/N64]` | Complete generation tok/s N64 -> N32 / llama `[N64/L -> N32/L; N32/N64]` | First-token ms N64 -> N32 / llama | N64=N32 |
|---|---:|---:|---:|---:|---:|
| Short, M=11 | 41.16 -> 36.11 / 215.55 `[0.191 -> 0.168; 0.988]` | 30.38 -> 26.03 / 86.42 `[0.352 -> 0.301; 0.978]` | 21.42 -> 18.42 / 72.48 `[0.295 -> 0.254; 0.988]` | 267.6 -> 304.9 / 51.2 | 5/5 |
| 128-token prompt | 381.54 -> 378.33 / 1,511.11 `[0.253 -> 0.250; 0.992]` | 30.43 -> 30.44 / 98.15 `[0.310 -> 0.310; 1.003]` | 19.65 -> 19.62 / 67.51 `[0.291 -> 0.291; 0.997]` | 335.8 -> 338.6 / 84.9 | 5/5 |
| 512-token prompt | 466.38 -> 461.56 / 2,108.34 `[0.221 -> 0.219; 0.990]` | 29.82 -> 29.85 / 97.51 `[0.306 -> 0.306; 1.003]` | 10.37 -> 10.30 / 41.23 `[0.252 -> 0.250; 0.991]` | 1,098.1 -> 1,109.6 / 243.1 | 5/5 |
| 1,024-token prompt | 498.33 -> 491.55 / 2,141.52 `[0.233 -> 0.230; 0.988]` | 29.85 -> 29.83 / 96.43 `[0.310 -> 0.309; 1.000]` | 6.50 -> 6.43 / 25.77 `[0.252 -> 0.250; 0.989]` | 2,055.1 -> 2,083.5 / 478.4 | 5/5 |
| Sustained decode, M=11 prompt | 41.91 -> 42.04 / 233.68 `[0.179 -> 0.180; 1.002]` | 30.25 -> 30.28 / 101.18 `[0.299 -> 0.299; 1.001]` | 28.54 -> 28.54 / 95.70 `[0.298 -> 0.298; 1.000]` | 262.7 -> 262.0 / 47.3 | 5/5 |

N=32 lowered the displayed condition-median prefill from 381.54 to 378.33 tok/s at 128, from 466.38 to 461.56 tok/s at 512, and from 498.33 to 491.55 tok/s at 1,024; first-token latency rose from 335.8 to 338.6 ms, 1,098.1 to 1,109.6 ms, and 2,055.1 to 2,083.5 ms respectively. The paired prefill ratios were 0.992x (range 0.991–0.998), 0.990x (0.981–1.007), and 0.988x (0.984–0.994). Cached decode was effectively flat and complete-generation rates were slightly lower on those cases. The short case had larger between-pair variation: its condition-median prefill was 41.16 -> 36.11 tok/s, but the paired ratio was 0.988x (0.870–0.998); its displayed first-token medians were 267.6 -> 304.9 ms, while the median paired latency increase was 3.8 ms (0.6–39.8 ms). All 25 control/candidate sequences matched exactly, and transient prefill peaks were unchanged at 4.05, 34.04, 160.05, and 240.03 MiB for the short, 128-, 512-, and 1,024-token workloads. No measured workload supports retaining N=32; production keeps the N=64 expert TensorOps tile.

The temporary tile selector and kernel branch were removed. Raw paired rows, the run log, and the validation output are `lfm2.5-8b-a1b-expert-mpp-n32-vs-n64.jsonl`, `lfm2.5-8b-a1b-expert-mpp-n32-vs-n64.run.log`, and `lfm2.5-8b-a1b-expert-mpp-n32-validation.log` in `docs/measurements/phase7a/`.

## Experiment 37: Block-and-pair traversal for direct Q4_K expert GEMV (retained)

Status: retained by default for the existing Q4_K 16-row expert fallback shape family (`N >= 128`, `K >= 256`, `K` divisible by 256), after grouped expert TensorOps eligibility. This remains a conventional direct-MSL path for sparse/M=1 routed work; it does not send M=1 GEMV to TensorOps or Core ML. The kernel walks each 256-value GGUF block as four 64-value pairs, reuses the loaded activation pair across four output rows per SIMD group, and reuses row/block addresses rather than recomputing Q4_K group addressing for each scalar K value. Products are accumulated separately in increasing K order. Dispatch remains format- and shape-based.

The focused Q4_K expert test covers three experts, four assignments, 133 output rows per expert, and K=256, 512, 1,024, and 4,096. It checks the output-row tail and compares the pair traversal bit-for-bit with the existing direct scalar path, in addition to the retained 8-row and 16-row kernels. It passed with Metal API Validation and GPU Shader Validation enabled; output is `lfm2.5-8b-a1b-expert-q4k16pairs-validation.log`.

The release A/B used the pinned LFM2.5-8B-A1B Q4_K_M artifact (SHA-256 `4923ec14f06b968b74d663e5949867d2d9c3bf13a20b8be1a9f9af39989b2bb0`), five interleaved pairs for each of the saved five workloads, exact shared prompt IDs, and the retained 16-row kernel as control. The only changed request field was `q4_k_expert_project_16rows_pairs` (`false` for control, `true` for candidate); both variants had the existing `q4_k_expert_project_16rows` path enabled. The current official llama.cpp reference rows were reused from `lfm2.5-8b-a1b-llama1ab7e5a-current.jsonl` at commit `1ab7e5ad2d4e7295c94c3b966a3e0b70fa365865`; no reference benchmark was rerun. All 25 control/candidate pairs and all 25 llama.cpp references used matching prompt IDs and generation lengths.

Rates are condition medians in tok/s. The first two ratios are condition-median Ferrum/llama.cpp ratios; the last is the median paired candidate/control rate ratio. First-token latency is milliseconds. ID counts are `control=candidate; candidate=llama.cpp; control=llama.cpp` out of five pairs.

| Workload | Prefill tok/s C->N / llama `[C/L->N/L; N/C]` | Cached decode tok/s C->N / llama `[C/L->N/L; N/C]` | Complete generation tok/s C->N / llama `[C/L->N/L; N/C]` | First-token ms C->N / llama | IDs C=N; N=L; C=L |
|---|---:|---:|---:|---:|---:|
| Short | 42.16->47.27 / 215.55 `[0.196->0.219; 1.122]` | 30.59->33.19 / 86.42 `[0.354->0.384; 1.086]` | 21.55->23.68 / 72.48 `[0.297->0.327; 1.097]` | 261.21->232.98 / 51.24 | 0/5; 5/5; 0/5 |
| 128-token prompt | 381.63->383.52 / 1,511.11 `[0.253->0.254; 1.004]` | 30.43->33.41 / 98.15 `[0.310->0.340; 1.095]` | 19.65->20.81 / 67.50 `[0.291->0.308; 1.057]` | 335.69->334.04 / 84.92 | 0/5; 0/5; 0/5 |
| 512-token prompt | 449.84->464.16 / 2,108.34 `[0.213->0.220; 1.030]` | 29.12->32.39 / 97.51 `[0.299->0.332; 1.093]` | 9.64->10.57 / 41.23 `[0.234->0.256; 1.080]` | 1,138.47->1,103.36 / 243.07 | 0/5; 0/5; 5/5 |
| 1,024-token prompt | 448.01->445.80 / 2,141.52 `[0.209->0.208; 1.008]` | 26.62->29.40 / 96.43 `[0.276->0.305; 1.100]` | 5.85->5.93 / 25.77 `[0.227->0.230; 1.025]` | 2,285.95->2,297.28 / 478.38 | 5/5; 5/5; 5/5 |
| Sustained decode, M=11 prompt | 42.28->46.66 / 233.68 `[0.181->0.200; 1.115]` | 30.08->32.70 / 101.18 `[0.297->0.323; 1.088]` | 28.34->30.77 / 95.70 `[0.296->0.322; 1.087]` | 260.49->236.08 / 47.29 | 0/5; 0/5; 0/5 |

Cached decode improved in all 25 pairs: absolute condition medians rose by 2.59-3.27 tok/s, with paired candidate/control ratios of 1.086-1.100x. Complete-generation throughput also rose in every workload; it moved from 5.85 to 5.93 tok/s on the 1,024-token prompt because the longer prefill dominates that measurement. First-token latency fell from 261.21 to 232.98 ms on the short prompt and from 260.49 to 236.08 ms on sustained decode; the 512-token result fell from 1,138.47 to 1,103.36 ms. The candidate's absolute cached-decode rates remain below 0.40x llama.cpp, so this local win does not meet the LFM or campaign gate.

Generated IDs changed between the Ferrum variants on the short, 128-token, 512-token, and sustained-decode workloads; the first mismatch was at generated-token index 7, 9, 8, and 7 respectively. All five 1,024-token outputs stayed identical. Candidate and llama.cpp matched on the short and 1,024-token cases (5/5 each); the raw IDs preserve all other divergences. The numerical test uses the unchanged tolerance and also checks exact equality to the scalar kernel for the tested Q4_K block and tail. Process RSS and transient peaks did not change: RSS was 5.22-5.29 GiB per workload, with identical control/candidate transient prefill peaks of 4.05, 34.04, 160.05, and 240.03 MiB at the short, 128-, 512-, and 1,024-token shapes.

Raw A/B rows and runner output are `lfm2.5-8b-a1b-expert-q4k16pairs-ab.jsonl` and `.run.log` in `docs/measurements/phase7a/`. The production selector defaults to the measured pair traversal and remains disabled in paired controls through the request field. Dense Q4_K M=1 GEMV keeps its separate measured 8-row path.

## Experiment 38: Sixteen-row Q6_K sparse-expert GEMV (rejected)

Status: rejected and removed. This direct-MSL candidate extended the retained Q6_K expert fallback from two output rows to four output rows per SIMD group, sharing each loaded activation across sixteen output rows per threadgroup. It remained below the grouped TensorOps eligibility threshold and did not use `matmul2d` or Core ML. The selector, kernel, benchmark field, and temporary test were removed after the paired results; production keeps the measured 8-row Q6_K expert kernel.

Correctness used three experts, four assignments, 133 output rows per expert, and K=256, 512, 1,024, and 4,096. The candidate matched both the scalar fallback and retained 8-row kernel bit-for-bit, covered the output tail, and stayed within the existing numerical tolerance. The focused test passed with Metal API Validation and GPU Shader Validation enabled; see `lfm2.5-8b-a1b-expert-q6k16rows-validation.log`.

The final A/B used ten interleaved control/candidate pairs per workload on the pinned LFM2.5-8B-A1B Q4_K_M artifact (SHA-256 `4923ec14f06b968b74d663e5949867d2d9c3bf13a20b8be1a9f9af39989b2bb0`). Control selected the retained 8-row Q6_K fallback; candidate selected the temporary 16-row kernel. Prompt IDs and generation lengths matched. The five saved llama.cpp reference rows per workload were reused from `lfm2.5-8b-a1b-llama1ab7e5a-current.jsonl`, official commit `1ab7e5ad2d4e7295c94c3b966a3e0b70fa365865`; llama.cpp was not rerun. Rates are condition medians in tok/s. The first two ratios are condition-median Ferrum/llama.cpp ratios; the final ratio is the median of the ten paired candidate/control rates. First-token latency is milliseconds. ID counts are `control=candidate` out of ten and `candidate=llama.cpp; control=llama.cpp` against the five saved references.

| Workload | Prefill tok/s C->N / llama `[C/L->N/L; N/C]` | Cached decode tok/s C->N / llama `[C/L->N/L; N/C]` | Complete generation tok/s C->N / llama `[C/L->N/L; N/C]` | First-token ms C->N / llama | IDs C=N; N=L; C=L |
|---|---:|---:|---:|---:|---:|
| Short | 46.49->47.18 / 215.55 `[0.216->0.219; 1.014]` | 32.68->33.02 / 86.42 `[0.378->0.382; 1.008]` | 23.32->23.55 / 72.48 `[0.322->0.325; 1.008]` | 236.91->233.43 / 51.24 | 10/10; 5/5; 5/5 |
| 128-token prompt | 377.41->372.27 / 1,511.11 `[0.250->0.246; 1.012]` | 32.83->32.29 / 98.15 `[0.334->0.329; 0.987]` | 20.31->20.11 / 67.50 `[0.301->0.298; 1.004]` | 339.46->344.13 / 84.92 | 10/10; 0/5; 0/5 |
| 512-token prompt | 455.61->456.24 / 2,108.34 `[0.216->0.216; 1.004]` | 32.36->32.51 / 97.51 `[0.332->0.333; 1.005]` | 10.49->10.48 / 41.23 `[0.254->0.254; 1.005]` | 1,124.09->1,122.56 / 243.07 | 10/10; 0/5; 0/5 |
| 1,024-token prompt | 495.91->493.19 / 2,141.52 `[0.232->0.230; 0.994]` | 32.37->32.47 / 96.43 `[0.336->0.337; 1.003]` | 6.58->6.55 / 25.77 `[0.255->0.254; 0.997]` | 2,065.24->2,076.61 / 478.38 | 10/10; 5/5; 5/5 |
| Sustained decode, M=11 prompt | 47.16->47.80 / 233.68 `[0.202->0.205; 1.013]` | 32.99->33.21 / 101.18 `[0.326->0.328; 1.007]` | 31.16->31.40 / 95.70 `[0.326->0.328; 1.007]` | 233.55->230.42 / 47.29 | 10/10; 0/5; 0/5 |

The candidate kept identical IDs in all 50 control/candidate comparisons and did not change memory: RSS and transient peaks were equal per workload (RSS 5.24-5.31 GiB; transient prefill peak 4.05, 34.04, 160.05, or 240.03 MiB). The small decode gains were inconsistent across shapes: the 128-token candidate fell from 32.83 to 32.29 tok/s (0.987x paired median), while the 1,024-token full-generation rate fell from 6.58 to 6.55 tok/s and first-token latency rose from 2,065.24 to 2,076.61 ms. At 128 tokens, the paired prefill, decode, and generation ratios ranged from 0.937-1.185x, 0.892-1.138x, and 0.974-1.122x; the condition medians and paired-ratio medians point in different directions for prefill and generation, so the A/B does not establish a stable win there. The short and sustained cases improved by under 1.4%, and the 512-token decode gain was 0.5%. The ten-pair evidence does not establish a robust all-shape benefit worth another Q6_K dispatch policy, so the 16-row kernel was removed. Candidate/reference decode remains only 0.328x on sustained generation.

The preliminary five-pair A/B and the ten-pair follow-up are preserved as `lfm2.5-8b-a1b-expert-q6k16rows-ab.jsonl` / `.run.log` and `lfm2.5-8b-a1b-expert-q6k16rows-10pairs-ab.jsonl` / `.run.log` in `docs/measurements/phase7a/`.

## Experiment 39: Sixteen-row dense Q4_K M=1 GEMV (rejected)

Status: rejected and removed. This direct-MSL candidate grouped four output rows per SIMD group and reused two 32-value activation spans while walking each Q4_K superblock as four 64-value pairs. The M=1 path stayed outside TensorOps. The trial changed only dense Q4_K GEMV; the retained sparse-expert pair traversal and all TensorOps settings were held fixed.

The focused check used N=133 with K=256, 512, 1,024, and 4,096, covering K-block counts and the output-row tail. It compared the candidate to the retained 8-row shader with the existing `2.0e-4` tolerance, and also to the scalar reference for K through 1,024. A first accumulation ordering exceeded the unchanged tolerance by 0.000016 at K=4,096; the candidate was changed to preserve the 8-row shader's per-pair accumulation grouping and then passed with Metal API Validation and GPU Shader Validation enabled. No tolerance was changed. The final validation output is `lfm2.5-8b-a1b-q4k16gemv-pairs-validation.log`.

The release A/B used ten interleaved control/candidate pairs for each of the five LFM2.5-8B-A1B Q4_K_M workloads, the pinned artifact (SHA-256 `4923ec14f06b968b74d663e5949867d2d9c3bf13a20b8be1a9f9af39989b2bb0`), and exact matching prompt IDs. Control used the retained dense 8-row Q4_K kernel; candidate enabled the temporary 16-row pair kernel. The five llama.cpp rows per workload were reused from `lfm2.5-8b-a1b-llama1ab7e5a-current.jsonl`, official revision `1ab7e5ad2d4e7295c94c3b966a3e0b70fa365865`; no llama.cpp rerun was needed. The llama.cpp ratios below compare condition medians from the saved reference with each Ferrum condition median; the candidate/control ratio is the median of the ten interleaved paired ratios. Rates are tok/s and latency is milliseconds.

| Workload | Prefill tok/s C->N / llama `[C/L->N/L; N/C]` | Cached decode tok/s C->N / llama `[C/L->N/L; N/C]` | Complete generation tok/s C->N / llama `[C/L->N/L; N/C]` | First-token ms C->N / llama | IDs C=N; N=L; C=L |
|---|---:|---:|---:|---:|---:|
| Short | 47.28->47.18 / 215.55 `[0.219->0.219; 1.002]` | 33.22->28.06 / 86.42 `[0.384->0.325; 0.844]` | 23.66->21.09 / 72.48 `[0.326->0.291; 0.892]` | 232.94->233.41 / 51.24 | 10/10; 5/5; 5/5 |
| 128-token prompt | 381.00->381.26 / 1,511.11 `[0.252->0.252; 1.002]` | 33.38->27.89 / 98.15 `[0.340->0.284; 0.835]` | 20.73->18.65 / 67.50 `[0.307->0.276; 0.895]` | 336.24->336.01 / 84.92 | 10/10; 0/5; 0/5 |
| 512-token prompt | 452.83->445.17 / 2,108.34 `[0.215->0.211; 0.999]` | 32.01->26.72 / 97.51 `[0.328->0.274; 0.843]` | 10.39->9.74 / 41.23 `[0.252->0.236; 0.945]` | 1,130.96->1,150.39 / 243.07 | 10/10; 0/5; 0/5 |
| 1,024-token prompt | 484.96->485.58 / 2,141.52 `[0.226->0.227; 0.996]` | 31.32->26.73 / 96.43 `[0.325->0.277; 0.847]` | 6.43->6.22 / 25.77 `[0.250->0.241; 0.963]` | 2,111.81->2,109.16 / 478.38 | 10/10; 5/5; 5/5 |
| Sustained decode, M=11 prompt | 46.41->46.68 / 233.68 `[0.199->0.200; 1.000]` | 32.45->27.43 / 101.18 `[0.321->0.271; 0.846]` | 30.62->26.16 / 95.70 `[0.320->0.273; 0.854]` | 237.30->235.95 / 47.29 | 10/10; 0/5; 0/5 |

The candidate matched control IDs in all 50 pairs. It reduced cached decode from 31.32-33.38 to 26.72-28.06 tok/s on the four prompt cases and from 32.45 to 27.43 tok/s in sustained decode; paired candidate/control decode ratios ranged from 0.835x to 0.847x. Complete-generation throughput fell on every case. Prefill stayed effectively flat because the candidate affects only M=1. First-token latency changed by less than 2 ms on four workloads and was 19.43 ms higher at 512 tokens. Candidate and control had equal median RSS per workload, equal KV reservations and transient prefill peaks, with one sustained-case RSS difference of only 8 KiB. The slower decode makes the 16-row dense schedule a clear loss, so its shader, selector, request option, and test were removed. Production keeps the direct 8-row Q4_K M=1 kernel.

The 100 raw Ferrum rows, runner output, and final focused validation output are `lfm2.5-8b-a1b-q4k16gemv-pairs-ab.jsonl`, `lfm2.5-8b-a1b-q4k16gemv-pairs-ab.run.log`, and `lfm2.5-8b-a1b-q4k16gemv-pairs-validation.log` in `docs/measurements/phase7a/`.

## Experiment 40: Thirty-two-row Q4_K sparse-expert M=1 GEMV (rejected)

Status: rejected and removed. This direct-MSL candidate assigned eight adjacent output rows to each SIMD group and covered 32 rows per threadgroup while sharing each loaded activation pair. It tests the sparse expert projection path only; it does not route M=1 work through TensorOps. Production retains the measured 16-row Q4_K expert pair traversal.

The focused correctness check used 133 output rows per expert and K=256, 512, 1,024, and 4,096. The candidate matched the scalar control through all K sizes and the output-row tail with the existing tolerance. Metal API Validation and GPU Shader Validation were enabled. The validation log is `lfm2.5-8b-a1b-expert-q4k32rows-validation.log`.

The release A/B used ten interleaved control/candidate pairs per workload on the pinned LFM2.5-8B-A1B Q4_K_M artifact (SHA-256 `4923ec14f06b968b74d663e5949867d2d9c3bf13a20b8be1a9f9af39989b2bb0`) with matching prompts and generation settings. Control used the retained 16-row pair traversal; candidate enabled the temporary 32-row kernel. The five llama.cpp rows per workload were reused from `lfm2.5-8b-a1b-llama1ab7e5a-current.jsonl`, official commit `1ab7e5ad2d4e7295c94c3b966a3e0b70fa365865`. No llama.cpp rerun was needed. The llama.cpp ratios compare each Ferrum condition median to the saved llama.cpp condition median; candidate/control ratios are medians of ten paired rates. Rates are tok/s and latency is milliseconds. ID counts are `control=candidate` out of ten and `candidate=llama.cpp; control=llama.cpp` against the five saved references.

| Workload | Prefill tok/s C->N / llama `[C/L->N/L; N/C]` | Cached decode tok/s C->N / llama `[C/L->N/L; N/C]` | Complete generation tok/s C->N / llama `[C/L->N/L; N/C]` | First-token ms C->N / llama | IDs C=N; N=L; C=L |
|---|---:|---:|---:|---:|---:|
| Short | 46.56->45.97 / 215.55 `[0.216->0.213; 0.988]` | 32.70->31.30 / 86.42 `[0.378->0.362; 0.957]` | 23.34->22.53 / 72.48 `[0.322->0.311; 0.965]` | 236.53->239.59 / 51.24 | 10/10; 5/5; 5/5 |
| 128-token prompt | 355.78->342.19 / 1,511.11 `[0.235->0.226; 0.991]` | 31.11->29.73 / 98.15 `[0.317->0.303; 0.954]` | 19.30->18.43 / 67.50 `[0.286->0.273; 0.970]` | 360.12->374.43 / 84.92 | 10/10; 0/5; 0/5 |
| 512-token prompt | 435.22->432.21 / 2,108.34 `[0.206->0.205; 1.005]` | 30.93->28.86 / 97.51 `[0.317->0.296; 0.973]` | 9.91->9.70 / 41.23 `[0.240->0.235; 0.993]` | 1,176.69->1,184.93 / 243.07 | 10/10; 0/5; 0/5 |
| 1,024-token prompt | 491.01->492.36 / 2,141.52 `[0.229->0.230; 1.000]` | 31.77->30.41 / 96.43 `[0.329->0.315; 0.958]` | 6.51->6.48 / 25.77 `[0.253->0.251; 0.993]` | 2,085.79->2,080.05 / 478.38 | 10/10; 5/5; 5/5 |
| Sustained decode, M=11 prompt | 46.49->45.80 / 233.68 `[0.199->0.196; 0.986]` | 32.47->31.10 / 101.18 `[0.321->0.307; 0.958]` | 30.67->29.40 / 95.70 `[0.320->0.307; 0.959]` | 236.90->240.48 / 47.29 | 10/10; 0/5; 0/5 |

The 32-row schedule reduced cached decode on every workload: from 30.93-32.70 to 28.86-31.30 tok/s for the prompt cases, and from 32.47 to 31.10 tok/s on sustained decode. Paired decode ratios were 0.954-0.973x. Complete-generation throughput also fell in all five workloads. Prefill was lower in four cases and improved by only 1.35 tok/s on the 1,024-token prompt. First-token latency increased by 3.06-14.31 ms in four cases, with a 5.74 ms decrease on the 1,024-token prompt. Outputs matched control in all 50 pairs. Median KV reservation and transient peaks were equal; median RSS was equal except for a 16 KiB decrease at 1,024 tokens. The measured decode loss makes this schedule unsuitable, so its shader, selector, benchmark option, and test were removed.

After retaining the 16-row sparse-expert traversal, a flushed `FERRUM_BATCH_LIMIT=1` profile was captured on the saved 11-token LFM2.5 workload. This diagnostic run measured 35.99 prefill tok/s, 8.96 cached-decode tok/s, 7.84 complete-generation tok/s, and 305.96 ms first-token latency. Because the limit flushes every dispatch, these rates are diagnostic and are not production-throughput comparisons. Across its 17 generated tokens, profile GPU time was 224.74 ms across 352 calls in Q4_K 16-row-pair input projection, 68.98 ms across 576 calls in dense Q4_K 8-row GEMV, and 62.61 ms across 192 calls in Q4_K 16-row-pair output projection; Q6_K 8-row expert output projection used 43.57 ms across 160 calls. The saved profile is `profiles/lfm2.5-8b-a1b-m1-post-q4k16pairs-profile.jsonl`, with its runner log beside it.

The 100 Ferrum A/B rows, runner output, validation output, and post-update profile are `lfm2.5-8b-a1b-expert-q4k32rows-ab.jsonl`, `lfm2.5-8b-a1b-expert-q4k32rows-ab.run.log`, `lfm2.5-8b-a1b-expert-q4k32rows-validation.log`, and the `profiles/lfm2.5-8b-a1b-m1-post-q4k16pairs-profile.*` files under `docs/measurements/phase7a/`.

## Experiment 41: Twenty-four-row Q4_K sparse-expert M=1 GEMV (rejected)

Status: rejected and removed. This direct-MSL experiment assigned six output rows to each SIMD group and covered 24 rows per threadgroup, between the retained 16-row schedule and the rejected 32-row schedule. It shared each Q4_K activation pair across six rows and did not use TensorOps.

Correctness used 133 output rows per expert with K=256, 512, 1,024, and 4,096. The candidate matched the scalar control through each K size and the output tail. The focused test passed with Metal API Validation and GPU Shader Validation enabled; see `lfm2.5-8b-a1b-expert-q4k24rows-validation.log`.

The release A/B used ten interleaved control/candidate pairs per workload on the pinned LFM2.5-8B-A1B Q4_K_M artifact (SHA-256 `4923ec14f06b968b74d663e5949867d2d9c3bf13a20b8be1a9f9af39989b2bb0`). Control used the retained 16-row pair traversal; candidate enabled the temporary 24-row kernel. The five llama.cpp rows per workload were reused from `lfm2.5-8b-a1b-llama1ab7e5a-current.jsonl`, official commit `1ab7e5ad2d4e7295c94c3b966a3e0b70fa365865`. Rates are condition medians in tok/s. Ferrum/llama.cpp ratios compare condition medians; candidate/control ratios are medians of the ten paired rates. First-token latency is milliseconds. ID counts are `control=candidate` out of ten and `candidate=llama.cpp; control=llama.cpp` against the five saved references.

| Workload | Prefill tok/s C->N / llama `[C/L->N/L; N/C]` | Cached decode tok/s C->N / llama `[C/L->N/L; N/C]` | Complete generation tok/s C->N / llama `[C/L->N/L; N/C]` | First-token ms C->N / llama | IDs C=N; N=L; C=L |
|---|---:|---:|---:|---:|---:|
| Short | 46.42->46.38 / 215.55 `[0.215->0.215; 0.997]` | 32.75->32.45 / 86.42 `[0.379->0.376; 0.992]` | 23.31->23.10 / 72.48 `[0.322->0.319; 0.993]` | 237.24->237.44 / 51.24 | 10/10; 5/5; 5/5 |
| 128-token prompt | 374.70->374.02 / 1,511.11 `[0.248->0.248; 1.002]` | 32.84->32.49 / 98.15 `[0.335->0.331; 0.993]` | 20.44->20.29 / 67.50 `[0.303->0.301; 0.994]` | 341.91->342.60 / 84.92 | 10/10; 0/5; 0/5 |
| 512-token prompt | 456.75->458.41 / 2,108.34 `[0.217->0.217; 1.004]` | 32.11->31.79 / 97.51 `[0.329->0.326; 0.990]` | 10.49->10.48 / 41.23 `[0.254->0.254; 0.999]` | 1,121.25->1,117.21 / 243.07 | 10/10; 0/5; 0/5 |
| 1,024-token prompt | 495.87->495.51 / 2,141.52 `[0.232->0.231; 0.999]` | 31.80->31.63 / 96.43 `[0.330->0.328; 0.994]` | 6.57->6.55 / 25.77 `[0.255->0.254; 0.999]` | 2,065.34->2,066.86 / 478.38 | 10/10; 5/5; 5/5 |
| Sustained decode, M=11 prompt | 46.29->46.30 / 233.68 `[0.198->0.198; 0.999]` | 32.47->32.17 / 101.18 `[0.321->0.318; 0.991]` | 30.67->30.40 / 95.70 `[0.320->0.318; 0.992]` | 237.91->237.84 / 47.29 | 10/10; 0/5; 0/5 |

The candidate left prefill effectively flat and lowered cached decode by 0.30-0.35 tok/s on every workload. Complete-generation throughput also fell in all five cases, by 0.01-0.27 tok/s. First-token changes were small and mixed. Control and candidate matched output IDs in all 50 comparisons; RSS, transient peaks, and KV reservations were unchanged. The small, consistent decode loss does not justify a new dispatch shape, so the 24-row kernel, selector, benchmark option, and test were removed. Production keeps the 16-row Q4_K expert traversal.

The 100 raw Ferrum rows, runner output, and focused validation output are `lfm2.5-8b-a1b-expert-q4k24rows-ab.jsonl`, `lfm2.5-8b-a1b-expert-q4k24rows-ab.run.log`, and `lfm2.5-8b-a1b-expert-q4k24rows-validation.log` in `docs/measurements/phase7a/`.

## Experiment 42: Q4_K dense MPP M=256 tile (rejected)

Status: rejected and removed. This completed independent dense-prefill check doubled the retained Q4_K MPP row tile from M=128 to M=256 while keeping K=64 and N=64 fixed. Its selector applied only at M>=1,024 prompt rows; smaller prompts and M=1 kept their existing paths. It is separate from sparse-expert M=1 GEMV.

The shader passed its focused M=1,025, N=65, K=256 reference check with Metal API Validation and GPU Shader Validation enabled. The A/B used ten interleaved Ferrum control/candidate pairs per workload on the pinned Qwen2.5-0.5B and LFM2.5-8B-A1B Q4_K_M artifacts. Candidate used M=256; control used retained M=128. Five saved llama.cpp references per workload were reused from `qwen2.5-q4_k_m-llama1ab7e5a-current.jsonl` and `lfm2.5-8b-a1b-llama1ab7e5a-current.jsonl`, both pinned to official commit `1ab7e5ad2d4e7295c94c3b966a3e0b70fa365865`; llama.cpp was not rerun. Values are condition medians. Each rate cell lists Ferrum control -> candidate / llama.cpp in tok/s and `[control/llama -> candidate/llama; median paired candidate/control]`. First-token latency is absolute milliseconds. `IDs` shows control=candidate out of ten, then candidate=llama.cpp and control=llama.cpp against five saved reference rows.

| Model / workload | Prefill tok/s C->N / L `[C/L->N/L;N/C]` | Cached decode tok/s C->N / L `[C/L->N/L;N/C]` | Complete generation tok/s C->N / L `[C/L->N/L;N/C]` | First-token ms C->N / L | IDs C=N; N=L; C=L |
|---|---:|---:|---:|---:|---:|
| Qwen2.5 short | 695.0->693.6 / 1,406.7 `[0.494->0.493;1.004]` | 120.0->127.7 / 238.8 `[0.502->0.535;1.001]` | 101.3->106.0 / 197.5 `[0.513->0.537;1.004]` | 30.6->30.7 / 15.2 | 10/10; 0/5; 0/5 |
| Qwen2.5 M=128 | 3,345.2->3,334.3 / 6,989.8 `[0.479->0.477;0.997]` | 129.5->129.6 / 230.7 `[0.561->0.562;1.000]` | 102.7->102.3 / 187.6 `[0.547->0.545;0.992]` | 38.7->38.7 / 18.6 | 10/10; 5/5; 5/5 |
| Qwen2.5 M=512 | 4,742.2->4,784.0 / 9,336.9 `[0.508->0.512;1.008]` | 124.5->123.9 / 233.5 `[0.533->0.531;0.996]` | 70.3->70.1 / 133.5 `[0.527->0.526;1.004]` | 108.3->107.3 / 55.1 | 10/10; 5/5; 5/5 |
| Qwen2.5 M=1,024 | 4,858.7->4,580.2 / 8,844.2 `[0.549->0.518;0.943]` | 112.9->113.6 / 236.5 `[0.477->0.480;1.009]` | 47.3->45.9 / 90.0 `[0.526->0.510;0.966]` | 211.1->223.9 / 116.0 | 10/10; 5/5; 5/5 |
| Qwen2.5 sustained decode | 693.7->698.7 / 1,385.7 `[0.501->0.504;1.000]` | 124.3->125.0 / 244.3 `[0.509->0.512;1.005]` | 118.5->119.0 / 221.8 `[0.534->0.537;1.006]` | 30.6->30.4 / 15.4 | 10/10; 0/5; 0/5 |
| LFM2.5 short | 47.4->47.2 / 215.6 `[0.220->0.219;0.998]` | 33.0->33.2 / 86.4 `[0.382->0.384;1.005]` | 23.6->23.7 / 72.5 `[0.326->0.326;1.002]` | 232.5->233.2 / 51.2 | 10/10; 5/5; 5/5 |
| LFM2.5 M=128 | 382.3->382.5 / 1,511.1 `[0.253->0.253;1.000]` | 33.3->33.3 / 98.2 `[0.340->0.339;1.001]` | 20.7->20.8 / 67.5 `[0.307->0.308;1.002]` | 335.1->334.9 / 84.9 | 10/10; 0/5; 0/5 |
| LFM2.5 M=512 | 466.2->466.3 / 2,108.3 `[0.221->0.221;1.002]` | 32.7->32.6 / 97.5 `[0.335->0.335;0.999]` | 10.7->10.7 / 41.2 `[0.259->0.259;1.002]` | 1,098.6->1,098.2 / 243.1 | 10/10; 0/5; 0/5 |
| LFM2.5 M=1,024 | 468.9->449.4 / 2,141.5 `[0.219->0.210;0.959]` | 30.2->30.1 / 96.4 `[0.313->0.313;1.003]` | 6.2->6.0 / 25.8 `[0.241->0.233;0.967]` | 2,184.1->2,278.9 / 478.4 | 10/10; 5/5; 5/5 |
| LFM2.5 sustained decode | 45.6->45.1 / 233.7 `[0.195->0.193;0.992]` | 32.2->31.5 / 101.2 `[0.319->0.311;0.983]` | 30.3->29.4 / 95.7 `[0.317->0.308;0.980]` | 241.3->244.1 / 47.3 | 10/10; 0/5; 0/5 |

At the only selected shape, M=1,024, the candidate reduced Qwen2.5 prefill from 4,858.7 to 4,580.2 tok/s and LFM2.5 from 468.9 to 449.4 tok/s; first-token latency rose by 12.8 ms and 94.8 ms. Complete-generation throughput also fell, while cached decode was nearly unchanged because this tile is used for prefill only. The candidate/control generated IDs matched in all 100 pairs, and transient prefill peaks were unchanged at 3.63 MiB for Qwen2.5 and 5.45 MiB for LFM2.5 at M=1,024. The measured regression rejects the M=256 tile; production keeps M=128. Raw A/Bs, runner logs, and validation are `qwen2.5-q4_k_m-mpp-m256-vs-m128-ab.jsonl`, `lfm2.5-8b-a1b-q4_k_m-mpp-m256-vs-m128-ab.jsonl`, their matching `.run.log` files, and `q4_k_mpp_m256-validation.log` in `docs/measurements/phase7a/`.

## Experiment 43: Q4_K sparse-expert 16-row 2-way K split (rejected)

Status: rejected and removed. This structural trial kept the retained 16-row/block-and-pair output tile and divided each SIMD group's Q4_K K traversal between two SIMD groups, then reduced their partial sums in the threadgroup. The production schedule and 16-row coverage stayed fixed; there was no row-count sweep. The shape gate was K>=1,024, covering the LFM expert projection K dimensions 1,792 and 2,048. The design differed from experiments 40 and 41, which changed output-row coverage and regressed through higher live row state.

The focused Metal API Validation and GPU Shader Validation check passed for K=256, 512, 1,024, 1,792, 2,048, and 4,096 with the unchanged tolerance. The initial 94-row A/B was invalidated after an unrelated `engine_sim` process used about one core; a separate zero-row attempt had a JSON protocol issue. Neither capture informs this decision. The final 100-row A/B had ten interleaved control/candidate pairs for each of five LFM workloads. CPU preflight ran before the matrix and every warmup/measured inference request. It compared the retained 16-row pair shader against the temporary 2-way K shader on the pinned LFM2.5-8B-A1B Q4_K_M artifact (SHA-256 `4923ec14f06b968b74d663e5949867d2d9c3bf13a20b8be1a9f9af39989b2bb0`). Five saved llama.cpp references per workload were reused from `lfm2.5-8b-a1b-llama1ab7e5a-current.jsonl`, official revision `1ab7e5ad2d4e7295c94c3b966a3e0b70fa365865`. Values are condition medians. Ferrum/llama.cpp ratios compare condition medians; candidate/control ratios are medians of ten paired rates. Rates are tok/s and latency is milliseconds.

| Workload | Prefill tok/s C->K / L `[C/L->K/L; K/C]` | Cached decode tok/s C->K / L `[C/L->K/L; K/C]` | Complete generation tok/s C->K / L `[C/L->K/L; K/C]` | First-token ms C->K / L |
|---|---:|---:|---:|---:|
| Short | 33.81->33.44 / 215.55 `[0.157->0.155; 0.985]` | 32.60->32.56 / 86.42 `[0.377->0.377; 1.000]` | 20.63->20.66 / 72.48 `[0.285->0.285; 1.006]` | 325.6->329.3 / 51.2 |
| 128-token prompt | 299.41->294.82 / 1,511.11 `[0.198->0.195; 0.988]` | 32.71->31.86 / 98.15 `[0.333->0.325; 0.989]` | 18.41->18.03 / 67.50 `[0.273->0.267; 0.979]` | 427.8->434.4 / 84.9 |
| 512-token prompt | 428.36->428.69 / 2,108.34 `[0.203->0.203; 1.002]` | 32.07->32.03 / 97.51 `[0.329->0.328; 0.998]` | 10.02->10.01 / 41.23 `[0.243->0.243; 1.001]` | 1,195.6->1,194.6 / 243.1 |
| 1,024-token prompt | 486.76->487.13 / 2,141.52 `[0.227->0.227; 0.999]` | 31.86->31.80 / 96.43 `[0.330->0.330; 0.997]` | 6.48->6.49 / 25.77 `[0.252->0.252; 1.000]` | 2,104.0->2,102.4 / 478.4 |
| Sustained decode, M=11 prompt | 34.85->34.14 / 233.68 `[0.149->0.146; 0.982]` | 32.54->32.29 / 101.18 `[0.322->0.319; 0.993]` | 30.18->29.90 / 95.70 `[0.315->0.312; 0.991]` | 315.9->322.5 / 47.3 |

There was no clear reduction in serial K cost: cached decode lost 0.27 tok/s on sustained generation (32.54->32.29, 0.993x paired median) and 0.85 tok/s on the 128-token case (32.71->31.86, 0.989x). The longer prefill cases were effectively flat, with no material prefill gain. Generated IDs matched control in 30/50 complete runs; the focused numerical check remained within the existing tolerance. This does not meet the preferred 5% end-to-end bar, so the 2-way split kernel, selector, test-only option, and custom runner were removed; production retains the 16-row block-and-pair schedule. No higher split factor will be tried without new profiling evidence that K serialization remains limiting.

The valid A/B rows and runner output are `lfm2.5-8b-a1b-expert-q4k16rows-ksplit2-ab.jsonl` and `.run.log`. The two invalid captures are retained separately with `.invalidated-cpu-load` and `.invalidated-runner-protocol` suffixes. The retained LFM expert shapes and validation coverage are recorded in `src/ops/transformer.rs`.

## Experiment 44: M=1 shared-token expert input (rejected)

Before coding, the flushed LFM diagnostic profile identifies 224.74 ms across 352 calls in Q4_K expert input projection as the largest remaining decode operation. It also attributes 5.56 ms across 352 calls to `expert_assign`, which copies a token activation into one row per selected expert; this is 0.91% of the diagnostic's 611.07 ms summed GPU time. The diagnostic flushes each dispatch and therefore overstates production wait overhead. The dispatch-count ceiling for removing this copy is 22 per decoded token in the 429-dispatch trace, while the measured kernel-time ceiling is under 1% of GPU work. This is a low-cost structural check rather than an expected 5% win.

Current upstream llama.cpp master at `4b1a27fa0eb875bbca4f6cfe936e3d65adc685c0` (2026-09-25) was inspected before implementation. Its Metal Q4_K mat-vec constants are `N_R0_Q4_K=2` and `N_SG_Q4_K=2`. The `kernel_mul_mv_id` M=1 wrapper loads each expert ID and token slot from the GPU ID tensor, selects the corresponding expert weight base and activation, then calls the quantized mat-vec body. The host dispatches the whole `MUL_MAT_ID` assignment/token grid in one launch. Ferrum already dispatches all selected assignments together for each expert projection, but currently inserts `expert_assign` and an assignment-major activation buffer before input projection. This experiment tests removing that copy by indexing the shared input with the token index already present in GPU routing metadata; it does not change row coverage, K traversal, expert order, or weights. It is structurally distinct from Experiment 43's split-K arithmetic change. Sources: [`ggml-metal-impl.h`](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-metal/ggml-metal-impl.h#L56-L57), [`mul_mv.metal`](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-metal/kernels/mul_mv.metal#L1511-L1629) and [the M=1 ID wrapper](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-metal/kernels/mul_mv.metal#L3295-L3356), [`ggml-metal-ops.cpp`](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-metal/ggml-metal-ops.cpp#L2676-L2865).

The focused shared-input projection test passed with Metal API Validation and GPU Shader Validation enabled, using Q4_K, K=2,048, 133 output rows, two token inputs, and shuffled GPU route metadata. The release A/B then ran ten interleaved control/shared-input pairs for each of five LFM workloads. The CPU preflight passed 114 checks with no blocked sample; the raw file contains 100 complete rows. The control is the retained production path with `expert_assign`; the candidate reads the token activation directly using the GPU token index. Saved llama.cpp references are reused from `lfm2.5-8b-a1b-llama1ab7e5a-current.jsonl`, official revision `1ab7e5ad2d4e7295c94c3b966a3e0b70fa365865`. Values are condition medians. Ferrum/llama.cpp ratios compare condition medians; candidate/control ratios are medians of ten paired rates. Throughput is tok/s and latency is milliseconds.

| Workload | Prefill tok/s C->S / L `[C/L->S/L; S/C]` | Cached decode tok/s C->S / L `[C/L->S/L; S/C]` | Complete generation tok/s C->S / L `[C/L->S/L; S/C]` | First-token ms C->S / L | Decode dispatches/token C->S | IDs C=S; S=L; C=L |
|---|---:|---:|---:|---:|---:|---:|
| Short | 33.07->32.96 / 215.55 `[0.153->0.153; 0.997]` | 33.12->33.18 / 86.42 `[0.383->0.384; 1.002]` | 20.77->20.76 / 72.48 `[0.287->0.286; 0.999]` | 332.96->334.00 / 51.24 | 429->407 | 10/10; 5/5; 5/5 |
| 128-token prompt | 295.14->294.95 / 1,511.11 `[0.195->0.195; 0.998]` | 33.37->33.39 / 98.15 `[0.340->0.340; 1.001]` | 18.55->18.55 / 67.50 `[0.275->0.275; 0.999]` | 433.98->434.26 / 84.92 | 429->407 | 10/10; 0/5; 0/5 |
| 512-token prompt | 428.85->428.62 / 2,108.34 `[0.203->0.203; 1.001]` | 32.61->32.65 / 97.51 `[0.334->0.335; 1.004]` | 10.06->10.08 / 41.23 `[0.244->0.244; 1.003]` | 1,194.19->1,194.81 / 243.07 | 429->407 (441->419 first token) | 10/10; 0/5; 0/5 |
| 1,024-token prompt | 488.99->488.21 / 2,141.52 `[0.228->0.228; 1.000]` | 32.27->32.44 / 96.43 `[0.335->0.336; 1.006]` | 6.52->6.52 / 25.77 `[0.253->0.253; 1.001]` | 2,094.39->2,097.72 / 478.38 | 429->407 (441->419 first token) | 10/10; 5/5; 5/5 |
| Sustained decode, M=11 prompt | 33.36->33.17 / 233.68 `[0.143->0.142; 0.994]` | 32.99->33.03 / 101.18 `[0.326->0.326; 1.001]` | 30.49->30.49 / 95.70 `[0.319->0.319; 1.001]` | 330.06->331.91 / 47.29 | 429->407 | 10/10; 0/5; 0/5 |

The output IDs were identical between control and candidate in all 50 pairs. Decode dispatches fell by 22/token, while the median per-token GPU time was effectively unchanged at 29.718 ms for control and 29.694 ms for the candidate. CPU encode time changed from 0.166 to 0.158 ms/token and wait time from 29.933 to 29.915 ms/token. The strongest paired cached-decode result was 1.006x on the 1,024-token case; sustained cached decode was 1.001x, and sustained complete generation was 1.001x. This does not approach the 5% retention bar, and the tiny copy path was not material to total work. The shared-input selector, API toggle, shader branches, benchmark option, and focused test were removed. Production retains the existing activation assignment copy and grouped projection dispatches. The result changes the next priority to whole-token runtime/scheduling trace; it does not justify another expert tile or split-factor variant.

Raw A/B rows and runner output are `lfm2.5-8b-a1b-expert-shared-token-input-ab.jsonl` and `.run.log` under `docs/measurements/phase7a/`.

## Phase 7B: runtime scheduling and host-boundary pass (Experiments 45–53)

This pass followed Experiment 44's conclusion that the next priority was whole-token scheduling rather than another expert tile. It measured the runtime around the kernels (encoders, barriers, host readbacks, and small serial kernels) before changing GPU arithmetic. Every change was measured against the prior production path with interleaved control/candidate pairs, through the Phase 7A CPU preflight, on the same pinned artifacts and workload files as the 1ab7e5a refresh.

**Retention rule for this pass.** Changes were retained when every workload showed a consistent paired improvement (or was neutral) with identical control/candidate output IDs, and removed otherwise. Several retained changes are individually below the campaign's 5% bar; this is a deliberate departure, recorded here so it is not mistaken for meeting that bar. Their combined effect is measured directly against `b5529c7` in the combined result below.

Measurement notes:

- The in-process A/Bs used `tools/phase55_matched_matrix.py --ferrum-paired-option`. A/Bs that need a different process (a load-time setting, or the pristine `b5529c7` build) used the new `tools/phase7a_binary_ab.py`, which interleaves two `phase55_bench` processes with per-side request fields or environment. Journal tables come from the new `tools/phase7a_ab_table.py`.
- During these sessions the desktop UI (WindowServer and an Electron GPU process) competed for the GPU. Short-prompt prefill ran about 490–510 tok/s in both control and candidate, versus 742.6 tok/s in the stored 1ab7e5a Qwen2.5 Q4_K_M matrix. Paired candidate/control ratios are unaffected. Cross-session ratios against the saved llama.cpp rows understate Ferrum, and llama.cpp was not rerun.
- Several short isolation A/Bs discarded runner stderr, so only their JSONL is retained. This is noted per file below.

Correction to an earlier review estimate: an Instruments trace (`docs/measurements/phase7a/traces/qwen25-dense-ferrum.summary.json`) showed 2.1 ms of CPU encoding per Qwen2.5 decode token. Untraced benchmark counters show about 0.18 ms, so the trace inflates encoding. The real decode gap is GPU time spread across hundreds of small dependent kernels, which is what this pass targets.

A flushed per-operation profile (`FERRUM_BATCH_LIMIT=1`) of a Qwen2.5 Q4_K_M decode token at a 1,024-token context (8.31 ms summed GPU) showed four things:

- Decode attention was 17.6% of GPU time: 0.92 ms `attention_context_decode` plus 0.55 ms scalar `attention_scores`, across 24 layers.
- The lm_head Q8_0 GEMV runs at about 112 GB/s, close to peak.
- The "Q4_K_M" artifact runs most projections as Q5_0, because 896 is not divisible by 256.
- Host logits handling (a 151,936-entry f32 readback plus CPU argmax and finite scan) cost about 0.5 ms per token outside the GPU.

The corresponding LFM2.5-8B-A1B profile (26.85 ms) is dominated by expert GEMVs. `rmsnorm` is 7.1% there (Experiment 53).

## Experiment 45: Benchmark harness defaults (fixed)

When a request omitted `q5_k_gemv_8rows` or `moe_gpu_routing_prefill`, `examples/phase55_bench.rs` turned the setting **off**, although both are production defaults (retained in Experiments 21 and 30). When `q4_k_mpp_tile_m128` was omitted, the harness reported `true` without resetting the device, so a reused device could keep an earlier `false`. Every Ferrum row in the seven `*-llama1ab7e5a-current.jsonl` matrices records `q5_k_gemv_8rows: false` and `moe_gpu_routing_prefill: false`. The current-upstream refresh therefore ran with those two production paths disabled, which affects Qwen2.5 Q5_K_M decode and LFM prompt routing. Omitted fields now select and record the production default. New request fields in this pass follow the same rule. The b5529c7 control in the combined result passes both settings explicitly, so it measures the old production path.

## Experiment 46: One compute encoder per command buffer (retained for dense models)

Status: retained with a per-model policy (see Experiment 47). Previously every dispatch created and ended its own `MTLComputeCommandEncoder`, which meant 531 encoders per Qwen2.5 decode token. The candidate keeps one serial encoder in the pending submission and ends it at commit. Serial dispatch order preserves every producer/consumer dependency. Toggle: `MetalDevice::set_shared_encoder`, bench field `shared_encoder`.

Qwen2.5 Q4_K_M, five pairs, control = per-kernel encoders. The session was loaded (control short-prompt prefill 430.7 tok/s). Raw: `qwen2.5-q4_k_m-shared-encoder-ab.jsonl` and `.run.log`.

| Workload | Prefill tok/s C->S `[S/C]` | Cached decode tok/s C->S `[S/C]` | Complete generation tok/s C->S `[S/C]` | First-token ms C->S | IDs C=S |
|---|---:|---:|---:|---:|---:|
| Short | 430.7->457.2 `[1.015]` | 104.6->111.0 `[1.064]` | 82.4->87.8 `[1.047]` | 49.2->46.4 | 5/5 |
| 128-token prompt | 2,192.4->2,128.9 `[0.987]` | 111.6->110.7 `[1.016]` | 82.1->80.8 `[1.005]` | 58.9->60.6 | 5/5 |
| 512-token prompt | 4,198.4->4,005.8 `[0.995]` | 133.2->126.2 `[0.999]` | 68.7->66.9 `[0.988]` | 122.3->128.4 | 5/5 |
| 1,024-token prompt | 4,740.1->4,754.6 `[1.003]` | 124.6->126.3 `[1.014]` | 48.1->48.7 `[1.006]` | 216.4->215.7 | 5/5 |
| Sustained decode | 439.9->472.3 `[1.070]` | 135.8->137.0 `[1.011]` | 126.8->128.2 `[1.015]` | 48.2->44.9 | 5/5 |

CPU encode time fell about 40% (0.20 -> 0.12 ms/token on long prompts). GPU time per token was unchanged. On its own this is a ~1% effect in a noisy session. It is retained because it is required for Experiment 47.

## Experiment 47: Concurrent dispatch with range-tracked barriers (retained for dense models)

Status: retained for models without sparse-MoE layers. The shared encoder is created with `MTLDispatchTypeConcurrent`. The submission records the byte ranges (buffer identity, offset, length) read and written since the last barrier. Before each dispatch, a read-after-write, write-after-read, or write-after-write overlap inserts `memoryBarrierWithScope(Buffers)` and clears the sets. Independent kernels, such as the q/k/v and gate/up projections that read the same activation, can then overlap. Buffers retained by the epoch keep their addresses unique until completion, and the arena only reuses completed storage. Toggle: `set_concurrent_dispatch`, bench field `concurrent_dispatch`.

Validation covered all unit tests with Metal API Validation and GPU Shader Validation enabled, plus every local model parity test with concurrency forced on for all models: Qwen2.5 BF16 (including the 128-token lifetime stress), Qwen3 0.6B/1.7B BF16 and Q8_0 GGUF, Granite 4, Granite MoE, OLMo 2, LFM2.5-230M, LFM2.5-8B-A1B BF16, and Q4_K_M GGUF.

Qwen2.5 Q4_K_M, five pairs, control = shared serial encoder. Raw: `qwen2.5-q4_k_m-concurrent-dispatch-ab.jsonl` and `.run.log`.

| Workload | Prefill tok/s C->N `[N/C]` | Cached decode tok/s C->N `[N/C]` | Complete generation tok/s C->N `[N/C]` | First-token ms C->N | IDs C=N |
|---|---:|---:|---:|---:|---:|
| Short | 498.2->506.6 `[1.020]` | 139.5->139.5 `[0.999]` | 104.1->105.5 `[1.015]` | 42.7->41.9 | 5/5 |
| 128-token prompt | 2,479.3->2,515.5 `[1.015]` | 140.1->139.9 `[1.010]` | 99.0->96.7 `[1.009]` | 52.1->51.4 | 5/5 |
| 512-token prompt | 4,419.8->4,427.1 `[1.006]` | 137.3->138.8 `[1.006]` | 71.2->71.5 `[1.012]` | 116.3->116.1 | 5/5 |
| 1,024-token prompt | 4,773.2->4,865.4 `[1.020]` | 125.7->127.1 `[1.017]` | 48.8->49.5 `[1.013]` | 214.9->210.8 | 5/5 |
| Sustained decode | 499.8->508.5 `[1.020]` | 135.1->138.5 `[1.025]` | 125.7->128.0 `[1.017]` | 42.5->41.7 | 5/5 |

GPU time per decode token fell a consistent ~3% (e.g. sustained 6.71 -> 6.47 ms). Barrier tracking adds about 0.08 ms/token of CPU encode. On Qwen3 Q8_0 at a 1,024-token context the effect was +0.8% decode and +2.5% prefill (`qwen3-0.6b-q8_0-1024-concurrent-dispatch-ab.jsonl`, JSONL only).

**Sparse-MoE policy.** A first combined run against `b5529c7` showed LFM2.5-8B-A1B short prefill 14% slower. Isolation (JSONL only) showed that the shared serial encoder alone costs LFM short prefill 6.1% (`lfm2.5-8b-a1b-short-shared-encoder-ab.jsonl`). Concurrency on top of it recovers 1.4% (`lfm2.5-8b-a1b-short-concurrent-ab.jsonl`). GPU execution time was identical in every mode (about 222 ms). The loss is non-GPU time between commit and completion, 37 ms with per-kernel encoders versus 53–57 ms with one encoder. A full LFM A/B of per-kernel encoders versus shared+concurrent (`lfm2.5-8b-a1b-encoder-mode-ab.jsonl`, four pairs, JSONL only) was net negative on every workload:

| Workload | Prefill tok/s K->C `[C/K]` | Cached decode tok/s K->C `[C/K]` | Complete generation tok/s K->C `[C/K]` | First-token ms K->C | IDs |
|---|---:|---:|---:|---:|---:|
| Short | 42.1->38.5 `[0.922]` | 35.5->35.4 `[0.998]` | 23.9->23.1 `[0.969]` | 261.6->285.4 | 4/4 |
| 128-token prompt | 315.2->311.9 `[0.982]` | 35.0->34.2 `[0.977]` | 19.7->19.4 `[0.982]` | 406.1->410.4 | 4/4 |
| 512-token prompt | 452.2->452.1 `[0.979]` | 34.0->33.1 `[0.970]` | 10.6->10.6 `[0.974]` | 1,132.2->1,132.5 | 4/4 |
| 1,024-token prompt | 529.3->528.9 `[0.992]` | 34.6->34.5 `[0.998]` | 7.0->7.0 `[0.991]` | 1,934.8->1,936.1 | 4/4 |
| Sustained decode | 41.8->37.6 `[0.899]` | 35.5->35.4 `[0.998]` | 33.4->33.0 `[0.990]` | 263.0->292.9 | 4/4 |

`Transformer` therefore opens its execution scope with `execution_with_shared_encoder(!has_sparse_moe)`. Dense models use the shared concurrent encoder, and models with sparse-MoE layers keep one encoder per kernel, which is the pre-7B behavior. A residency set was tested as an explanation for the commit gap and did not remove it (Experiment 52). The cause of the MoE gap is not established.

## Experiment 48: Wide M=1 attention context kernel (retained, shape-gated); GQA scores kernel (rejected)

Status: context kernel retained. The companion scores kernel was rejected and removed.

Experiment 48 tested two kernels. `attention_scores_decode` used one SIMD group per cached K row and computed every query head in its GQA group from one K load. `attention_context_decode_wide` uses one 1,024-thread group per (query head, 32-column tile), with 32 SIMD partitions over positions and a fixed-order partial reduction. Both accumulate in f32 and keep the existing bf16 storage-rounding boundaries; only the summation order changes, as in Experiment 23.

Rejected intermediate designs, from flushed per-op profiles at Qwen2.5 1,024 context:

- A GQA-sharing context kernel with four threadgroups per layer took 64.6 µs, against 38.2 µs for the control.
- A scores kernel with query heads staged in 16 KB of threadgroup memory took 26–28 µs, against 22.8 µs for the control, because the staging capped occupancy.

The final scores kernel was still slower than the scalar control on both models (27.7 vs 22.8 µs on Qwen2.5 and 63.2 vs 49.4 µs on Qwen3; profiles `qwen3-0.6b-q8_0-decode-attention-v1-op-profile-{control,candidate}.json`). A Qwen3 1,024-context A/B with both kernels showed −2.9% decode (`qwen3-0.6b-q8_0-decode-attention-v1-1024-ab.jsonl`). The scores kernel was removed.

The context kernel cut Qwen2.5 per-layer context time from 38.2 to 22.7 µs. It also replaces the fully scalar `attention_context` path that decode previously used below 128 positions. Ungated, it cost LFM 1.1% decode at 1,024 context (`lfm2.5-8b-a1b-1024-context-wide-ungated-ab.jsonl`). The selector is therefore M=1 with either fewer than 128 positions, or fewer than 64 (head, column-tile) groups, where the old 256-thread kernel underfills the GPU. The groups are 28 for Qwen2.5 and 64 for Qwen3 and LFM2.5. Toggle: `set_attention_context_decode_wide`, bench field `attention_context_decode_wide`. Test: `wide_decode_context_matches_existing_kernels`, covering GQA groups 7, 2, 1, and 10, widths 33, 64, 80, and 128, and F32/F16/BF16, also run under shader validation.

Gated A/B, control = previous context selection. Raw: `qwen2.5-q4_k_m-context-decode-wide-ab.jsonl`, `lfm2.5-8b-a1b-context-decode-wide-ab.jsonl` (JSONL only).

| Model / workload | Prefill tok/s C->W `[W/C]` | Cached decode tok/s C->W `[W/C]` | Complete generation tok/s C->W `[W/C]` | First-token ms C->W | IDs C=W |
|---|---:|---:|---:|---:|---:|
| Qwen2.5 / 1,024-token prompt | 4,845.6->4,839.4 `[0.999]` | 127.9->133.8 `[1.046]` | 50.2->50.8 `[1.013]` | 211.3->211.6 | 5/5 |
| Qwen2.5 / Sustained decode | 492.0->488.4 `[0.996]` | 139.5->149.2 `[1.070]` | 134.5->142.8 `[1.062]` | 42.7->43.0 | 0/5 |
| LFM2.5 / 1,024-token prompt | 543.6->551.2 `[1.014]` | 34.9->34.8 `[0.997]` | 7.2->7.3 `[1.011]` | 1,883.8->1,857.6 | 4/4 |
| LFM2.5 / Sustained decode | 42.7->43.4 `[1.015]` | 35.3->35.4 `[1.003]` | 33.2->33.4 `[1.004]` | 257.6->253.8 | 0/4 |

The 128-token sustained sequences change because the summation order changes. They do not change fidelity. Against the saved llama.cpp sustained reference, control and candidate first diverge at the same generated index in every pair (index 14 for Qwen2.5, index 29 for LFM2.5), so the new kernel only alters the trajectory after both have already left the reference.

## Experiment 49: MoE chunk outputs assembled on Metal (retained)

When routing temporaries force more than one chunk (about 200 tokens per chunk for LFM2.5), `SparseMoe::forward` previously synchronized, read each chunk's combined output to host f32, and re-uploaded the concatenation. That added one completion wait per chunk per MoE layer. The candidate allocates the `[tokens, hidden]` output once and writes each chunk into its row range with the existing append-copy kernel. It adds no host boundary, and routing temporaries stay bounded by the 256 MiB arena flush. Toggle: `set_moe_chunk_device_copy`, bench field `moe_chunk_device_copy`. Test: `chunked_outputs_assemble_on_device_like_host_readback` forces two chunks and requires bit-identical output. LFM2.5 prefill command buffers fell from 67 to 13 at 512 tokens and from 112 to 26 at 1,024. Raw: `lfm2.5-8b-a1b-moe-chunk-device-copy-ab.jsonl` and `.run.log`.

| Workload | Prefill tok/s C->D `[D/C]` | Cached decode tok/s C->D `[D/C]` | Complete generation tok/s C->D `[D/C]` | First-token ms C->D | IDs C=D |
|---|---:|---:|---:|---:|---:|
| 512-token prompt | 475.4->483.8 `[1.006]` | 35.0->35.0 `[1.000]` | 11.0->11.2 `[1.004]` | 1,077.3->1,058.6 | 3/3 |
| 1,024-token prompt | 534.2->541.2 `[1.007]` | 34.4->34.5 `[1.000]` | 7.1->7.2 `[1.005]` | 1,917.0->1,892.3 | 3/3 |

## Experiment 50: Fused gate+up projection (rejected)

The candidate stacked same-format quantized gate and up rows into one `[2I, K]` matrix at load time. GGUF blocks never span rows, so this is an exact byte concatenation. One projection then fed the existing `expert_silu_mul` kernel. A unit test confirmed bit-identical output for M=1 and M=37. Decode was flat, because concurrent dispatch already overlaps the two projections. Prefill at 512 and 1,024 tokens was 1.7% slower in every pair. All code was removed, and QKV fusion was not attempted for the same reason. Raw: `qwen2.5-q4_k_m-fused-gate-up-ab.jsonl` and `.run.log`.

| Workload | Prefill tok/s C->F `[F/C]` | Cached decode tok/s C->F `[F/C]` | Complete generation tok/s C->F `[F/C]` | IDs |
|---|---:|---:|---:|---:|
| Short | 506.3->510.3 `[1.010]` | 147.2->146.5 `[1.001]` | 108.4->108.0 `[1.003]` | 5/5 |
| 128-token prompt | 2,499.8->2,490.6 `[0.999]` | 144.3->144.8 `[1.000]` | 102.1->101.5 `[0.994]` | 5/5 |
| 512-token prompt | 4,441.8->4,370.8 `[0.983]` | 140.3->140.4 `[1.001]` | 72.1->71.2 `[0.986]` | 5/5 |
| 1,024-token prompt | 4,842.2->4,781.5 `[0.983]` | 129.1->129.3 `[1.002]` | 49.8->49.4 `[0.991]` | 5/5 |
| Sustained decode | 505.8->505.6 `[0.991]` | 150.3->150.4 `[1.002]` | 139.0->139.2 `[1.001]` | 5/5 |

## Experiment 51: On-device greedy argmax (retained)

`argmax_rows` is a 1,024-thread row reduction. It selects the first maximal index (matching the host `generation::argmax` tie rule) and sets a non-finite flag, so the host rejects non-finite logits exactly as before. The model encodes it in the same submission as the forward pass, after logits scaling. `generation::generate_greedy` reads back 8 bytes per token instead of the full vocabulary row. The bench uses it by default (field `device_argmax`). `ferrum run` uses it at temperature 0, and `generate(select)` is unchanged for sampling.

Tests:

- `argmax_rows_matches_host_ties_tails_and_nonfinite_flags` covers widths 1 to 151,936, deliberate ties, all dtypes, NaN, and infinity.
- `real_model` now asserts that `generate_greedy` produces the same tokens, stop reason, and KV bytes as `generate(argmax)` on Qwen2.5 BF16.

A first 256-thread version added 0.11 ms of GPU time per token, so its decode rate was 0.5% lower (`qwen2.5-q4_k_m-device-argmax-256-ab.jsonl`). The retained 1,024-thread version:

| Workload | Prefill tok/s C->A `[A/C]` | Cached decode tok/s C->A `[A/C]` | Complete generation tok/s C->A `[A/C]` | First-token ms C->A | IDs C=A |
|---|---:|---:|---:|---:|---:|
| Qwen2.5 / Short | 503.5->508.8 `[1.011]` | 146.0->145.7 `[1.004]` | 107.9->112.6 `[1.044]` | 42.1->41.3 | 5/5 |
| Qwen2.5 / Sustained decode | 505.9->506.3 `[1.002]` | 150.1->151.1 `[1.006]` | 138.9->145.0 `[1.044]` | 41.9->41.5 | 5/5 |
| LFM2.5 / 1,024-token prompt | 533.1->540.3 `[1.004]` | 33.7->34.4 `[1.009]` | 7.0->7.2 `[1.005]` | 1,920.9->1,895.4 | 4/4 |

Raw: `qwen2.5-q4_k_m-device-argmax-ab.jsonl` and `.run.log`, `lfm2.5-8b-a1b-1024-device-argmax-ab.jsonl` (JSONL only). Most of the gain is the removed host argmax and finite scan (about 0.25 ms/token on Qwen's 151,936-entry vocabulary). The cached-decode timer previously also included the full-row f32 conversion.

## Experiment 52: Queue residency set for weights (rejected)

The Instruments traces showed about 5x more driver wiring events for Ferrum than for llama.cpp. The candidate added every weight allocation (`PackedStorage` and `Tensor::from_reader`) to one `MTLResidencySet` attached to the queue. Two A/Bs varied residency at load time with an environment toggle.

- **Qwen2.5 Q4_K_M, five pairs:** no consistent effect. Short decode was −4.1% with a noisy range, and the other workloads were within ±0.7%.
- **LFM2.5, four pairs:** short prefill and decode were −1% and −1.8%, and the 128-token prompt was +6% with pairs spanning 0.95–1.11.

The per-token difference between completion wait and GPU time was unchanged (about 0.28 ms for Qwen, and 67 ms on the short LFM prefill). Residency setup is not the steady-state cost, and the code was removed. Raw: `qwen2.5-q4_k_m-residency-set-ab.jsonl` and `.run.log`, `lfm2.5-8b-a1b-residency-set-ab.jsonl` (JSONL only).

## Experiment 53: `rmsnorm` exactness fallback (unrolled loads rejected; cost measured, decision pending)

In the LFM2.5 decode profile, `rmsnorm` takes 32.4 µs per call across 61 calls, or 1.98 ms (7.4%) of the 26.9 ms token. For bf16, when any output lands within 8 ulps of a rounding midpoint (about 40% of 2,048-wide rows), thread 0 recomputes the legacy ascending sum over the whole row, so the result matches the established path exactly. A candidate issued the serial sum's loads 16 at a time without changing the accumulation order.

Byte comparison of 3,392 bf16 rows (8.8 MB, widths 64 to 4,864) against the `b5529c7` shader was identical. The profile, however, was unchanged at 32.4 µs, because the 2,048-step dependent add chain is the cost, not the loads. The candidate was removed.

For reference, disabling the fallback (a measurement only, not committed) cuts `rmsnorm` from 1.98 to 0.32 ms per LFM token, about 6% of decode GPU time (`lfm2.5-8b-a1b-rmsnorm-no-fallback-op-profile.json`). The trade-off is occasional bf16 rounding changes relative to the established path, so it is left as an explicit decision rather than an optimization.

Also measured and not pursued: folding `lfm2_split3` into `lfm2_short_conv` (0.2% of the LFM decode token), and CPU router sorting and trace-name formatting (below measurement resolution).

## Phase 7B combined result versus `b5529c7`

The control is a pristine `b5529c7` build with `q5_k_gemv_8rows` and `moe_gpu_routing_prefill` passed explicitly, so it runs the old production path. The candidate is the Phase 7B tree. Both use `tools/phase7a_binary_ab.py` with five interleaved pairs per workload. Saved llama.cpp references come from the 1ab7e5a matrices, official commit `1ab7e5ad2d4e7295c94c3b966a3e0b70fa365865`, and were not rerun. Each cell lists Ferrum control -> candidate / llama.cpp in tok/s, then `[control/llama -> candidate/llama; median paired candidate/control]`. First-token latency is in milliseconds. `IDs` gives candidate=control, then candidate=llama.cpp, then control=llama.cpp.

Short-prompt cross-session ratios are depressed by this session's GPU contention; see the measurement notes above.

| Model / workload | Prefill tok/s C->N / L | Cached decode tok/s C->N / L | Complete generation tok/s C->N / L | First-token ms C->N / L | IDs |
|---|---:|---:|---:|---:|---:|
| Qwen2.5 Q4_K_M / Short | 492.0->508.9 / 1,406.7 `[0.350->0.362; 1.034]` | 137.2->144.9 / 238.8 `[0.575->0.607; 1.055]` | 103.4->112.0 / 197.5 `[0.523->0.567; 1.082]` | 43.2->41.3 / 15.2 | 5/5; 0/5; 0/5 |
| Qwen2.5 Q4_K_M / 128-token prompt | 2,448.5->2,515.2 / 6,989.8 `[0.350->0.360; 1.033]` | 139.6->145.7 / 230.7 `[0.605->0.631; 1.042]` | 98.6->105.3 / 187.6 `[0.526->0.561; 1.073]` | 52.8->50.9 / 18.6 | 5/5; 5/5; 5/5 |
| Qwen2.5 Q4_K_M / 512-token prompt | 4,397.9->4,463.1 / 9,336.9 `[0.471->0.478; 1.013]` | 136.0->142.4 / 233.5 `[0.583->0.610; 1.044]` | 70.8->73.6 / 133.5 `[0.530->0.551; 1.046]` | 116.8->114.7 / 55.1 | 5/5; 5/5; 5/5 |
| Qwen2.5 Q4_K_M / 1,024-token prompt | 4,786.4->4,807.3 / 8,844.2 `[0.541->0.544; 1.016]` | 124.1->133.5 / 236.5 `[0.525->0.565; 1.079]` | 48.7->50.7 / 90.0 `[0.541->0.563; 1.047]` | 214.3->213.0 / 116.0 | 5/5; 5/5; 5/5 |
| Qwen2.5 Q4_K_M / Sustained decode | 483.5->489.0 / 1,385.7 `[0.349->0.353; 1.008]` | 133.8->142.9 / 244.3 `[0.548->0.585; 1.075]` | 124.2->137.5 / 221.8 `[0.560->0.620; 1.107]` | 43.9->42.9 / 15.4 | 0/5; 0/5; 0/5 |
| Qwen3 Q8_0 / Short | 336.8->363.6 / 1,258.3 `[0.268->0.289; 1.075]` | 115.9->120.2 / 164.5 `[0.705->0.730; 1.035]` | 82.7->89.2 / 143.8 `[0.575->0.620; 1.076]` | 62.9->57.8 / 17.0 | 5/5; 5/5; 5/5 |
| Qwen3 Q8_0 / 128-token prompt | 1,590.5->1,710.0 / 6,001.2 `[0.265->0.285; 1.076]` | 116.5->118.4 / 163.8 `[0.711->0.723; 1.019]` | 75.9->80.8 / 138.8 `[0.547->0.582; 1.064]` | 81.0->74.9 / 21.6 | 5/5; 0/5; 0/5 |
| Qwen3 Q8_0 / 512-token prompt | 2,854.6->2,957.3 / 7,099.6 `[0.402->0.417; 1.034]` | 107.6->110.2 / 154.9 `[0.695->0.712; 1.024]` | 50.0->52.0 / 95.2 `[0.525->0.546; 1.040]` | 179.7->173.1 / 72.4 | 5/5; 5/5; 5/5 |
| Qwen3 Q8_0 / 1,024-token prompt | 3,277.7->3,349.5 / 6,283.2 `[0.522->0.533; 1.024]` | 94.6->96.5 / 143.3 `[0.660->0.673; 1.019]` | 34.0->35.1 / 61.1 `[0.556->0.574; 1.027]` | 312.7->305.7 / 163.2 | 5/5; 5/5; 5/5 |
| Qwen3 Q8_0 / Sustained decode | 337.8->359.6 / 1,241.8 `[0.272->0.290; 1.063]` | 113.8->121.6 / 164.7 `[0.691->0.738; 1.068]` | 105.9->116.0 / 157.4 `[0.673->0.737; 1.094]` | 62.6->58.4 / 17.2 | 5/5; 0/5; 0/5 |
| LFM2.5-8B-A1B Q4_K_M / Short | 41.7->41.6 / 215.6 `[0.193->0.193; 0.997]` | 35.5->35.4 / 86.4 `[0.410->0.409; 1.000]` | 23.7->23.8 / 72.5 `[0.327->0.328; 1.003]` | 264.3->264.5 / 51.2 | 5/5; 5/5; 5/5 |
| LFM2.5-8B-A1B Q4_K_M / 128-token prompt | 322.4->324.5 / 1,511.1 `[0.213->0.215; 1.000]` | 35.8->35.8 / 98.2 `[0.364->0.364; 0.999]` | 20.1->20.2 / 67.5 `[0.297->0.299; 1.005]` | 397.3->394.5 / 84.9 | 5/5; 0/5; 0/5 |
| LFM2.5-8B-A1B Q4_K_M / 512-token prompt | 470.9->473.1 / 2,108.3 `[0.223->0.224; 1.005]` | 35.1->35.1 / 97.5 `[0.360->0.360; 1.001]` | 11.0->11.1 / 41.2 `[0.267->0.268; 1.005]` | 1,087.7->1,082.2 / 243.1 | 5/5; 0/5; 0/5 |
| LFM2.5-8B-A1B Q4_K_M / 1,024-token prompt | 527.5->536.7 / 2,141.5 `[0.246->0.251; 1.014]` | 35.0->35.0 / 96.4 `[0.363->0.363; 1.001]` | 7.0->7.1 / 25.8 `[0.272->0.276; 1.013]` | 1,941.3->1,908.1 / 478.4 | 5/5; 5/5; 5/5 |
| LFM2.5-8B-A1B Q4_K_M / Sustained decode | 42.5->41.8 / 233.7 `[0.182->0.179; 0.980]` | 35.4->35.5 / 101.2 `[0.350->0.351; 1.003]` | 33.1->33.4 / 95.7 `[0.346->0.349; 1.008]` | 259.1->263.2 / 47.3 | 0/5; 0/5; 0/5 |

Qwen2.5 Q4_K_M improved on every workload:

- Cached decode: +4.2% to +7.9%.
- Complete generation: +4.6% to +10.7%.
- Prefill: +0.8% to +3.4%.

Qwen3 Q8_0 also improved on every workload:

- Cached decode: +1.9% to +6.8%.
- Complete generation: +2.7% to +9.4%.
- Prefill: +2.4% to +7.6%.

LFM2.5-8B-A1B, which keeps per-kernel encoders, is essentially neutral:

- Cached decode: −0.1% to +0.3%.
- Complete generation: +0.3% to +1.3%.
- Prefill: +0.5% to +1.4% for the 512- and 1,024-token prompts, from the MoE chunk and argmax changes. The short-prompt and sustained medians were −0.3% and −2.0%, with pairs spanning both sides of 1.0.

Candidate and control IDs matched in every pair except the sustained-decode sequences. There, candidate and control diverge from llama.cpp at the same index, as described under Experiment 48; Qwen3's sustained IDs matched in all five pairs. The Qwen2.5 run predates the Experiment 48 gate, which does not change Qwen2.5's path (28 groups), and the Qwen3 and LFM runs use the final gated tree.

Raw rows and runner logs are `qwen2.5-q4_k_m-phase7b-vs-b5529c7.jsonl`, `qwen3-0.6b-q8_0-phase7b-vs-b5529c7.jsonl`, `lfm2.5-8b-a1b-phase7b-vs-b5529c7.jsonl`, and the matching `.run.log` files.

Validation for the retained tree:

- `cargo test --release -- --include-ignored` passes 153 tests with every local model checkpoint.
- The library suite passes under Metal API Validation and GPU Shader Validation.
- `cargo clippy --all-targets` reports only the four warnings already present at `b5529c7`.

The three MoE parity tests (Granite MoE, LFM2.5 BF16, LFM2.5 GGUF) previously failed at `b5529c7` on `assert!(stats.active_experts > 0)`, which GPU routing (Experiment 30) no longer satisfies. That assertion now applies only when active experts are counted, so the later cache-branch replay checks in those tests run again.

No measured format meets the Phase 7 gate. The large Qwen targets remain locked.

## Experiment 54: Factored-scale Q4_K M=1 kernels (retained)

Status: retained for dense M=1 Q4_K GEMV and sparse-expert Q4_K projection whenever K is a multiple of 256. Toggle: `set_q4_k_factored`, bench field `q4_k_factored`.

A comparison of bandwidths located the problem. In the LFM2.5 decode profile, the retained expert kernel (`expert_project_q4_k_16rows_pairs`) read 16.5 MB per layer (4 experts × 3,584 rows × 2,048 columns of Q4_K) in 530 µs, about 31 GB/s. The dense Q4_K GEMV in the same model reached about 80 GB/s. The Q4_K kernels were compute-bound. Every lane decoded 6-bit scales and minimums per weight, formed `d*scale*q - dmin*min` per weight, and loaded one quantized byte per lane.

The candidate ports the algorithm of llama.cpp's `kernel_mul_mv_q4_K` (MIT; credited in the shader) as `q4_k_factored_rows<R>`:

- Lanes split into four block streams of eight.
- Each lane owns 32 values of a block and reads packed nibbles as 16-bit words.
- It decodes the block's eight scales and minimums once with the `kmask` bit tricks.
- It applies `d*scale` and `dmin*min` per 32-value group, using a per-group activation sum for the minimum term.

This leaves about one FMA per weight. The f32 summation order changes, as in Experiment 23. It is used by `q4_k_gemv_factored` and `expert_project_q4_k_factored`, both at four rows per SIMD group (Experiment 56).

Tests:

- `q4_k_factored_gemv_matches_reference_across_blocks_and_row_tails` covers N = 1, 7, 130, 133, and 9, with K from 256 to 4,864.
- The existing expert row-reuse test checks the factored projection against the scalar reference and the old kernel, within bf16 rounding.
- Both pass under Metal API and GPU Shader Validation.
- All 155 model/unit tests pass, including the LFM2.5 Q4_K_M GGUF parity test against its llama.cpp reference.

Profile effect for LFM2.5, per layer: expert input 530 -> 142 µs (about 116 GB/s), Q4_K expert output 273 -> 92 µs, and dense Q4_K GEMV 88 -> 62 µs. In-process A/B, five pairs, control = previous Q4_K kernels. Raw: `lfm2.5-8b-a1b-q4_k-factored-ab.jsonl`, `qwen2.5-q4_k_m-q4_k-factored-ab.jsonl`, and matching `.run.log` files. This A/B used the two-row dense variant.

| Model / workload | Prefill tok/s C->F `[F/C]` | Cached decode tok/s C->F `[F/C]` | Complete generation tok/s C->F `[F/C]` | First-token ms C->F | IDs C=F |
|---|---:|---:|---:|---:|---:|
| LFM2.5 / Short | 41.1->76.6 `[1.875]` | 35.5->62.9 `[1.774]` | 23.7->42.8 `[1.811]` | 267.5->143.7 | 5/5 |
| LFM2.5 / 128-token prompt | 321.2->319.2 `[0.995]` | 35.7->63.1 `[1.767]` | 20.1->25.9 `[1.290]` | 398.6->401.0 | 5/5 |
| LFM2.5 / 512-token prompt | 473.7->470.8 `[0.989]` | 35.0->62.4 `[1.785]` | 11.0->12.6 `[1.140]` | 1,080.8->1,087.5 | 0/5 |
| LFM2.5 / 1,024-token prompt | 531.5->529.1 `[0.997]` | 35.0->61.9 `[1.772]` | 7.0->7.7 `[1.091]` | 1,926.6->1,935.2 | 5/5 |
| LFM2.5 / Sustained decode | 41.6->74.9 `[1.809]` | 35.5->62.6 `[1.767]` | 33.3->59.0 `[1.773]` | 264.4->146.9 | 0/5 |
| Qwen2.5 Q4_K_M / Short | 486.7->486.6 `[1.004]` | 144.2->148.0 `[1.032]` | 110.1->111.8 `[1.013]` | 43.1->43.2 | 5/5 |
| Qwen2.5 Q4_K_M / 128-token prompt | 2,422.6->2,372.7 `[0.984]` | 144.6->147.7 `[1.017]` | 103.9->104.1 `[1.014]` | 52.8->53.9 | 5/5 |
| Qwen2.5 Q4_K_M / 512-token prompt | 4,332.8->4,335.4 `[0.991]` | 142.5->145.6 `[1.016]` | 73.0->73.4 `[1.007]` | 118.2->118.1 | 5/5 |
| Qwen2.5 Q4_K_M / 1,024-token prompt | 4,801.9->4,800.0 `[1.000]` | 133.3->135.7 `[1.018]` | 50.6->50.9 `[1.006]` | 213.3->213.3 | 5/5 |
| Qwen2.5 Q4_K_M / Sustained decode | 489.2->475.1 `[0.973]` | 148.9->151.9 `[1.020]` | 142.2->144.9 `[1.019]` | 42.9->44.2 | 0/5 |

Short prompts (M=11) and 128-token routes take the direct expert kernel in prefill too, which is why short prefill nearly doubles.

**Fidelity.** Output changed on the LFM 512-token and sustained cases and the Qwen2.5 sustained case. Against the saved llama.cpp sequences:

- The LFM candidate now matches llama.cpp exactly on the 512-token case (control diverged at generated index 8).
- It also matches on the full 128-token sustained sequence (control diverged at index 29), since its arithmetic now follows llama.cpp's.
- On Qwen2.5 sustained, control and candidate both leave llama.cpp at index 14, and differ from each other only at index 82.

## Experiment 55: Factored-scale Q6_K M=1 kernels (retained)

Status: retained for dense M=1 Q6_K GEMV (including LFM2.5's lm_head) and sparse-expert Q6_K projection. Toggle: `set_q6_k_factored`, bench field `q6_k_factored`.

This is the same approach as Experiment 54, porting llama.cpp's `kernel_mul_mv_q6_K` as `q6_k_factored_rows<R>`:

- Sixteen lanes cover one 256-value block, with two block streams per SIMD group.
- Each lane owns four adjacent values in each of four 32-value slices.
- The signed slice scale and block d are applied once per slice sum.

Test: `q6_k_factored_gemv_matches_reference_across_blocks_and_row_tails`, plus a factored check in the Q6_K expert test. The pinned Qwen2.5 Q6_K artifact was reverified: SHA-256 `2f82233630c349ccf6b8daccf48f9a7865713d9f08a2eadfa456cebe9b97c7f5`.

Profile effect on LFM2.5: Q6_K expert output 225 -> 135 µs per layer, and lm_head 2.34 -> 1.97 ms. The in-process A/B below used the two-row variants on top of Experiment 54. Raw: `lfm2.5-8b-a1b-q6_k-factored-ab.jsonl`, `qwen2.5-q6_k-q6_k-factored-ab.jsonl`, and `.run.log` files.

| Model / workload | Prefill tok/s C->F `[F/C]` | Cached decode tok/s C->F `[F/C]` | Complete generation tok/s C->F `[F/C]` | First-token ms C->F | IDs C=F |
|---|---:|---:|---:|---:|---:|
| LFM2.5 / Short | 74.3->80.3 `[1.074]` | 63.2->69.5 `[1.101]` | 42.3->46.3 `[1.092]` | 148.0->136.9 | 5/5 |
| LFM2.5 / 128-token prompt | 313.0->315.1 `[1.007]` | 63.4->68.7 `[1.085]` | 25.7->26.5 `[1.030]` | 409.0->406.2 | 0/5 |
| LFM2.5 / 512-token prompt | 469.3->467.9 `[1.002]` | 62.7->67.3 `[1.077]` | 12.6->12.8 `[1.016]` | 1,091.0->1,094.3 | 5/5 |
| LFM2.5 / 1,024-token prompt | 525.3->529.8 `[1.003]` | 62.1->66.1 `[1.065]` | 7.6->7.7 `[1.011]` | 1,949.3->1,933.0 | 5/5 |
| LFM2.5 / Sustained decode | 73.6->79.5 `[1.080]` | 62.8->69.3 `[1.103]` | 59.1->64.9 `[1.097]` | 149.4->138.3 | 5/5 |
| Qwen2.5 Q6_K / Short | 444.7->436.6 `[0.983]` | 136.2->138.3 `[1.012]` | 103.0->103.8 `[1.009]` | 47.2->48.1 | 5/5 |
| Qwen2.5 Q6_K / 128-token prompt | 2,155.6->2,147.2 `[1.036]` | 134.2->137.4 `[1.023]` | 95.6->96.7 `[1.025]` | 59.4->59.6 | 5/5 |
| Qwen2.5 Q6_K / 512-token prompt | 4,161.6->4,181.9 `[1.021]` | 134.7->137.1 `[1.013]` | 69.1->70.1 `[1.022]` | 123.0->122.4 | 5/5 |
| Qwen2.5 Q6_K / 1,024-token prompt | 4,456.0->4,388.0 `[1.000]` | 128.5->130.9 `[1.021]` | 47.5->47.4 `[1.006]` | 229.8->233.4 | 5/5 |
| Qwen2.5 Q6_K / Sustained decode | 460.4->451.6 `[0.907]` | 141.6->144.3 `[1.018]` | 135.5->137.9 `[1.014]` | 45.6->46.5 | 5/5 |

Qwen2.5 Q6_K short and sustained prefill samples were bimodal, at about 450 or about 502 tok/s, in both control and candidate (GPU contention). Short prefill runs only one M=1 GEMV, the lm_head, which is faster, so those medians are not attributed to the kernel. Output IDs matched control in 45 of 50 pairs. The 128-token LFM case changed at the index where both variants already leave llama.cpp (index 9).

## Experiment 56: Rows per SIMD group for factored kernels (four retained)

Flushed per-operation profiles (`*-short-decode-op-profile-*-factored.json`) compared rows per SIMD group.

- **One row per SIMD group** (tried for narrow outputs, to double threadgroups) was slower on the Qwen2.5 down projections: Q6_K 49.7 -> 61.5 µs, Q4_K 44.7 -> 56.8 µs. It was removed.
- **Four rows per SIMD group** (sixteen per threadgroup) was faster everywhere:
  - Qwen2.5 down projections: Q6_K 49.7 -> 43.6 µs, Q4_K 44.7 -> 34.8 µs.
  - LFM2.5 dense Q4_K: 61.4 -> 46.3 µs.
  - LFM2.5 lm_head: 1.96 -> 1.62 ms (about 133 GB/s).
  - LFM2.5 Q6_K expert output: 135 -> 104 µs.
  - Summed LFM2.5 decode GPU time: 14.06 -> 12.67 ms.
- **Eight rows** for the Q4_K expert was neutral: input 142.7 -> 147.0 µs and output 93.3 -> 82.1 µs, with the same 12.67 ms total. It was not retained.

Dense and expert factored kernels therefore use four rows per SIMD group.

## Factored-kernel combined result versus `169bd48`

The control is the pushed Phase 7B checkpoint `169bd48`. The candidate has Experiments 54–56. Five interleaved pairs per workload, via `tools/phase7a_binary_ab.py`. llama.cpp references are the saved 1ab7e5a rows (official `1ab7e5ad2d4e7295c94c3b966a3e0b70fa365865`), not rerun. Cells use the combined-result format above. Raw: `*-factored-vs-169bd48.jsonl` and `.run.log`.

| Model / workload | Prefill tok/s C->N / L | Cached decode tok/s C->N / L | Complete generation tok/s C->N / L | First-token ms C->N / L | IDs |
|---|---:|---:|---:|---:|---:|
| LFM2.5-8B-A1B Q4_K_M / Short | 44.2->91.3 / 215.6 `[0.205->0.423; 2.088]` | 35.5->77.7 / 86.4 `[0.411->0.899; 2.188]` | 24.4->52.1 / 72.5 `[0.336->0.719; 2.147]` | 248.8->120.5 / 51.2 | 5/5; 5/5; 5/5 |
| LFM2.5-8B-A1B Q4_K_M / 128-token prompt | 339.5->337.9 / 1,511.1 `[0.225->0.224; 0.995]` | 35.8->76.7 / 98.2 `[0.364->0.781; 2.146]` | 20.6->29.0 / 67.5 `[0.305->0.429; 1.403]` | 377.0->378.8 / 84.9 | 0/5; 0/5; 0/5 |
| LFM2.5-8B-A1B Q4_K_M / 512-token prompt | 487.5->484.1 / 2,108.3 `[0.231->0.230; 0.992]` | 35.1->74.8 / 97.5 `[0.360->0.767; 2.135]` | 11.3->13.4 / 41.2 `[0.273->0.324; 1.183]` | 1,050.2->1,057.7 / 243.1 | 0/5; 5/5; 0/5 |
| LFM2.5-8B-A1B Q4_K_M / 1,024-token prompt | 549.3->541.3 / 2,141.5 `[0.256->0.253; 0.988]` | 35.0->73.5 / 96.4 `[0.363->0.762; 2.100]` | 7.3->8.0 / 25.8 `[0.282->0.311; 1.103]` | 1,864.3->1,891.6 / 478.4 | 5/5; 5/5; 5/5 |
| LFM2.5-8B-A1B Q4_K_M / Sustained decode | 43.6->89.4 / 233.7 `[0.187->0.383; 2.073]` | 35.5->77.4 / 101.2 `[0.351->0.765; 2.181]` | 33.5->72.6 / 95.7 `[0.350->0.758; 2.169]` | 252.1->123.0 / 47.3 | 0/5; 5/5; 0/5 |
| Qwen2.5 Q4_K_M / Short | 485.7->492.1 / 1,406.7 `[0.345->0.350; 1.011]` | 141.7->150.8 / 238.8 `[0.593->0.631; 1.063]` | 108.9->114.5 / 197.5 `[0.552->0.580; 1.051]` | 43.2->42.7 / 15.2 | 5/5; 0/5; 0/5 |
| Qwen2.5 Q4_K_M / 128-token prompt | 2,400.2->2,430.4 / 6,989.8 `[0.343->0.348; 1.010]` | 141.8->149.1 / 230.7 `[0.614->0.646; 1.051]` | 102.3->106.8 / 187.6 `[0.545->0.569; 1.044]` | 53.3->52.7 / 18.6 | 5/5; 5/5; 5/5 |
| Qwen2.5 Q4_K_M / 512-token prompt | 4,377.2->4,355.7 / 9,336.9 `[0.469->0.467; 0.995]` | 139.5->149.6 / 233.5 `[0.598->0.641; 1.070]` | 72.4->74.2 / 133.5 `[0.543->0.556; 1.026]` | 117.0->117.5 / 55.1 | 5/5; 5/5; 5/5 |
| Qwen2.5 Q4_K_M / 1,024-token prompt | 4,786.7->4,768.9 / 8,844.2 `[0.541->0.539; 0.999]` | 134.0->141.3 / 236.5 `[0.567->0.597; 1.054]` | 50.6->51.4 / 90.0 `[0.562->0.571; 1.017]` | 213.9->214.7 / 116.0 | 5/5; 5/5; 5/5 |
| Qwen2.5 Q4_K_M / Sustained decode | 491.0->488.6 / 1,385.7 `[0.354->0.353; 0.989]` | 148.7->157.8 / 244.3 `[0.609->0.646; 1.061]` | 142.1->150.9 / 221.8 `[0.641->0.680; 1.061]` | 42.8->43.0 / 15.4 | 0/5; 0/5; 0/5 |
| Qwen2.5 Q6_K / Short | 451.3->453.4 / 1,478.6 `[0.305->0.307; 1.005]` | 136.3->141.1 / 199.1 `[0.685->0.709; 1.034]` | 103.7->106.7 / 173.9 `[0.596->0.613; 1.028]` | 46.5->46.3 / 14.5 | 5/5; 5/5; 5/5 |
| Qwen2.5 Q6_K / 128-token prompt | 2,242.2->2,228.6 / 7,068.9 `[0.317->0.315; 0.994]` | 135.0->141.0 / 201.8 `[0.669->0.699; 1.045]` | 97.0->99.3 / 168.2 `[0.577->0.591; 1.023]` | 57.1->57.4 / 18.4 | 5/5; 5/5; 5/5 |
| Qwen2.5 Q6_K / 512-token prompt | 4,238.7->4,245.0 / 9,643.1 `[0.440->0.440; 1.006]` | 134.5->140.6 / 203.2 `[0.662->0.692; 1.046]` | 69.8->71.5 / 125.0 `[0.558->0.572; 1.026]` | 120.8->120.6 / 53.4 | 5/5; 5/5; 5/5 |
| Qwen2.5 Q6_K / 1,024-token prompt | 4,506.4->4,515.9 / 9,077.0 `[0.496->0.498; 1.004]` | 128.5->134.2 / 197.2 `[0.652->0.680; 1.047]` | 47.9->48.7 / 86.3 `[0.555->0.564; 1.020]` | 227.2->226.8 / 113.1 | 5/5; 5/5; 5/5 |
| Qwen2.5 Q6_K / Sustained decode | 448.9->452.0 / 1,451.0 `[0.309->0.311; 0.995]` | 141.8->148.4 / 202.7 `[0.699->0.732; 1.047]` | 135.6->141.3 / 191.2 `[0.709->0.739; 1.042]` | 46.8->46.5 / 14.7 | 5/5; 0/5; 0/5 |

**LFM2.5-8B-A1B Q4_K_M:**

- Cached decode rose from 35.5 to 77.7 tok/s on the short workload, ×2.10–×2.19 across all five workloads. Against llama.cpp this moves decode from about 0.36x to 0.76–0.90x.
- Short prefill and sustained generation roughly doubled.
- 128-, 512-, and 1,024-token prefill medians were 0.5–1.2% lower. Those prompts run TensorOps GEMM paths that this round did not change, so the small decrease is recorded, not explained.

**Qwen2.5:**

- Q4_K_M decode improved 5.1–7.0% (its Q4_K and Q6_K down projections).
- Q6_K decode improved 3.4–4.7%.
- Prefill changed by ±1%.

Qwen3-0.6B Q8_0 uses none of these kernels.

The LFM2.5 short and 1,024-token decode ratios (0.899x and 0.762x) are the closest Ferrum has come to the 0.85x decode gate. Medium and long prefill (0.22–0.25x) remain far from the 0.80x prefill gate, so the gate is still not met.

With the factored kernels in place, the remaining LFM2.5 decode profile (12.67 ms summed) is:

- Q4_K expert input: 3.14 ms.
- `rmsnorm`: 2.40 ms, dominated by the exactness fallback of Experiment 53.
- Dense Q4_K: 1.67 ms.
- lm_head: 1.61 ms.
- Expert outputs: 2.16 ms.

The `rmsnorm` fallback is now about 19% of the summed LFM2.5 decode profile and remains an open design decision.

## Experiment 57: Block-wise TensorOps weight-tile decoding (retained)

Status: retained for the dense Q4_K (K=128, K=64, K=64/M=128), Q6_K, Q8_0 (K=128, K=64), and Q5_0 TensorOps GEMMs, and for the Q4_K/Q6_K grouped-expert TensorOps projections. Toggle: `set_mpp_fast_dequant`, bench field `mpp_fast_dequant`. Output is **bit-identical** to the previous decoders.

A flushed profile of LFM2.5's 512-token prefill (`lfm2.5-8b-a1b-512-prefill-op-profile-before-fast-dequant.json`, 947.5 ms summed GPU) attributed 82% to the grouped-expert TensorOps projections. The expert Q4_K input projection ran at about 1.2 TFLOPS (66 calls at 8.28 ms), against about 6.2 TFLOPS for the dense Q4_K TensorOps GEMM in the same model.

The TensorOps kernels spend most of their time dequantizing GGUF weights into bf16 threadgroup tiles, and each decoded tile is reused by only 32 (expert) or 64 (dense) prompt rows. The existing decoders worked on one or two values per iteration. Each iteration reloaded the block's half-precision scale (plus, for K-quants, the 6-bit scale and minimum decode, and for Q5_0 the high-bit bytes). The expert decoder also branched on format per value.

The new decoders give each work item a whole unit, decode its parameters once, and emit the values in a short loop:

- **Q4_K and Q6_K:** 16 packed bytes of one row's 64-value chunk, which is two 32-value groups sharing one scale/minimum (Q4_K) or signed scale (Q6_K) per group.
- **Q8_0 and Q5_0:** one whole 32-value block, with Q5_0's high-bit word assembled once.

Each value keeps the exact per-value formula and operand order: `(d*scale)*q-(dmin*min)`, `(d*scale)*q`, `scale*q`, and `d*q`.

Tests: `q4_k_tensorops_fast_dequant_is_bit_identical` (all three dense Q4_K tiles plus the expert path at K=128 and K=64, uneven segments, and row tails), `q6_k_tensorops_fast_dequant_is_bit_identical`, `q8_0_tensorops_fast_dequant_is_bit_identical`, and `q5_0_tensorops_fast_dequant_is_bit_identical`. They require identical output bits between the old and new decoders and pass under Metal API and GPU Shader Validation. The full 159-test suite, including every local model parity test, passes.

Flushed LFM2.5 512-token prefill profile: summed GPU 986.5 -> 314.1 ms. Expert Q4_K input 8.76 -> 2.12 ms per call, expert Q6_K output 2.98 -> 1.52 ms, expert Q4_K output 4.52 -> 1.11 ms, and dense Q4_K GEMM 2.10 -> 0.82 ms. Qwen2.5 Q6_K dense Q6_K GEMM went from 865 to 496 µs.

**Rejected alongside:** a flattened expert-tile grid indexed the 32-row-padded segment space instead of launching `experts x ceil(assignments/32)` tiles, most of which exit immediately. It changed nothing measurable (summed 960.3 -> 971.1 ms without the new decoder, 364.2 -> 371.5 ms with it) and was removed.

In-process A/B, five interleaved pairs per workload, control = previous decoders. Cells list control->candidate / llama.cpp and `[control/llama -> candidate/llama; median paired candidate/control]`, using the saved 1ab7e5a references. Raw: `*-mpp-fast-dequant-ab.jsonl` and `.run.log`.

| Model / workload | Prefill tok/s C->N / L | Cached decode tok/s C->N / L | Complete generation tok/s C->N / L | First-token ms C->N / L | IDs |
|---|---:|---:|---:|---:|---:|
| LFM2.5-8B-A1B Q4_K_M / Short | 90.8->103.8 / 215.6 `[0.421->0.482; 1.155]` | 77.4->77.6 / 86.4 `[0.895->0.898; 1.001]` | 51.8->54.4 / 72.5 `[0.714->0.751; 1.052]` | 121.1->105.9 / 51.2 | 5/5; 5/5; 5/5 |
| LFM2.5-8B-A1B Q4_K_M / 128-token prompt | 332.8->714.5 / 1,511.1 `[0.220->0.473; 2.193]` | 76.2->76.3 / 98.2 `[0.776->0.778; 1.004]` | 28.6->43.8 / 67.5 `[0.424->0.649; 1.541]` | 384.6->179.1 / 84.9 | 5/5; 0/5; 0/5 |
| LFM2.5-8B-A1B Q4_K_M / 512-token prompt | 465.0->1,176.6 / 2,108.3 `[0.221->0.558; 2.531]` | 74.3->74.5 / 97.5 `[0.762->0.764; 1.002]` | 12.9->26.1 / 41.2 `[0.313->0.634; 2.024]` | 1,101.1->435.2 / 243.1 | 5/5; 5/5; 5/5 |
| LFM2.5-8B-A1B Q4_K_M / 1,024-token prompt | 521.5->1,310.7 / 2,141.5 `[0.244->0.612; 2.503]` | 73.0->72.9 / 96.4 `[0.757->0.756; 0.998]` | 7.7->17.0 / 25.8 `[0.301->0.659; 2.190]` | 1,963.6->781.3 / 478.4 | 5/5; 5/5; 5/5 |
| LFM2.5-8B-A1B Q4_K_M / Sustained decode | 93.9->96.4 / 233.7 `[0.402->0.412; 1.017]` | 77.0->77.1 / 101.2 `[0.761->0.762; 1.002]` | 72.4->72.6 / 95.7 `[0.756->0.758; 1.002]` | 117.1->114.1 / 47.3 | 5/5; 5/5; 5/5 |
| Qwen2.5 Q4_K_M / Short | 487.5->690.7 / 1,406.7 `[0.347->0.491; 1.416]` | 154.7->153.0 / 238.8 `[0.648->0.641; 0.986]` | 115.8->126.3 / 197.5 `[0.586->0.640; 1.092]` | 43.1->30.4 / 15.2 | 5/5; 0/5; 0/5 |
| Qwen2.5 Q4_K_M / 128-token prompt | 2,434.8->3,360.8 / 6,989.8 `[0.348->0.481; 1.378]` | 151.4->152.4 / 230.7 `[0.656->0.660; 1.009]` | 106.7->118.3 / 187.6 `[0.569->0.631; 1.104]` | 52.6->38.1 / 18.6 | 5/5; 5/5; 5/5 |
| Qwen2.5 Q4_K_M / 512-token prompt | 4,358.5->6,186.8 / 9,336.9 `[0.467->0.663; 1.415]` | 151.5->151.4 / 233.5 `[0.649->0.648; 1.002]` | 74.3->88.7 / 133.5 `[0.557->0.665; 1.185]` | 117.5->82.8 / 55.1 | 5/5; 5/5; 5/5 |
| Qwen2.5 Q4_K_M / 1,024-token prompt | 4,813.1->6,344.2 / 8,844.2 `[0.544->0.717; 1.316]` | 140.6->141.1 / 236.5 `[0.595->0.597; 1.003]` | 51.7->61.3 / 90.0 `[0.574->0.681; 1.182]` | 212.8->161.4 / 116.0 | 5/5; 5/5; 5/5 |
| Qwen2.5 Q4_K_M / Sustained decode | 486.6->680.0 / 1,385.7 `[0.351->0.491; 1.390]` | 158.4->158.7 / 244.3 `[0.648->0.650; 1.003]` | 151.4->154.1 / 221.8 `[0.683->0.695; 1.017]` | 43.2->30.9 / 15.4 | 5/5; 0/5; 0/5 |
| Qwen2.5 Q6_K / Short | 454.5->683.4 / 1,478.6 `[0.307->0.462; 1.504]` | 142.2->141.5 / 199.1 `[0.714->0.710; 0.992]` | 107.5->119.3 / 173.9 `[0.618->0.686; 1.113]` | 46.2->30.7 / 14.5 | 5/5; 5/5; 5/5 |
| Qwen2.5 Q6_K / 128-token prompt | 2,248.3->3,298.3 / 7,068.9 `[0.318->0.467; 1.467]` | 142.6->141.3 / 201.8 `[0.707->0.700; 0.989]` | 100.5->111.3 / 168.2 `[0.598->0.662; 1.123]` | 56.9->38.8 / 18.4 | 5/5; 5/5; 5/5 |
| Qwen2.5 Q6_K / 512-token prompt | 4,228.0->6,305.8 / 9,643.1 `[0.438->0.654; 1.486]` | 141.9->140.6 / 203.2 `[0.699->0.692; 0.993]` | 71.8->85.8 / 125.0 `[0.574->0.686; 1.188]` | 121.1->81.2 / 53.4 | 5/5; 5/5; 5/5 |
| Qwen2.5 Q6_K / 1,024-token prompt | 4,504.7->6,325.2 / 9,077.0 `[0.496->0.697; 1.404]` | 133.6->133.3 / 197.2 `[0.677->0.676; 0.997]` | 48.5->59.7 / 86.3 `[0.562->0.691; 1.229]` | 227.3->161.9 / 113.1 | 5/5; 5/5; 5/5 |
| Qwen2.5 Q6_K / Sustained decode | 455.1->678.6 / 1,451.0 `[0.314->0.468; 1.491]` | 147.7->147.0 / 202.7 `[0.729->0.725; 0.995]` | 140.8->142.4 / 191.2 `[0.736->0.745; 1.015]` | 46.1->30.9 / 14.7 | 5/5; 0/5; 0/5 |
| Qwen3 Q8_0 / Short | 357.4->600.2 / 1,258.3 `[0.284->0.477; 1.672]` | 120.3->119.9 / 164.5 `[0.731->0.729; 0.994]` | 88.8->100.7 / 143.8 `[0.618->0.700; 1.133]` | 58.8->35.0 / 17.0 | 5/5; 5/5; 5/5 |
| Qwen3 Q8_0 / 128-token prompt | 1,694.6->2,764.2 / 6,001.2 `[0.282->0.461; 1.631]` | 119.4->116.8 / 163.8 `[0.729->0.713; 0.983]` | 80.5->93.2 / 138.8 `[0.580->0.671; 1.156]` | 75.5->46.3 / 21.6 | 5/5; 0/5; 0/5 |
| Qwen3 Q8_0 / 512-token prompt | 2,932.7->4,584.9 / 7,099.6 `[0.413->0.646; 1.559]` | 109.8->110.0 / 154.9 `[0.709->0.710; 0.991]` | 51.5->62.5 / 95.2 `[0.541->0.657; 1.222]` | 174.6->111.7 / 72.4 | 5/5; 5/5; 5/5 |
| Qwen3 Q8_0 / 1,024-token prompt | 3,343.9->5,041.6 / 6,283.2 `[0.532->0.802; 1.508]` | 96.5->96.4 / 143.3 `[0.674->0.673; 0.998]` | 35.0->44.5 / 61.1 `[0.573->0.727; 1.267]` | 306.2->203.1 / 163.2 | 5/5; 5/5; 5/5 |
| Qwen3 Q8_0 / Sustained decode | 356.6->592.9 / 1,241.8 `[0.287->0.477; 1.659]` | 121.9->121.1 / 164.7 `[0.740->0.735; 0.995]` | 116.2->118.2 / 157.4 `[0.738->0.751; 1.020]` | 58.9->35.4 / 17.2 | 5/5; 0/5; 0/5 |

Prefill rises 2.19–2.53x for LFM2.5 at 128–1,024 tokens, 1.32–1.42x for Qwen2.5 Q4_K_M, 1.40–1.50x for Qwen2.5 Q6_K, and 1.51–1.67x for Qwen3 Q8_0. Cached decode is unchanged, since it uses M=1 GEMV. Every pair produced identical IDs. First-token latency on the LFM2.5 1,024-token prompt fell from 1,964 to 781 ms. Against llama.cpp, 512- and 1,024-token prefill is now 0.56–0.61x for LFM2.5 (0.47x at 128 tokens), 0.65–0.72x for Qwen2.5 Q4_K_M and Q6_K, and 0.65–0.80x for Qwen3. Qwen3's 1,024-token prompt reaches 0.802x, the first workload at the 0.80x prefill threshold; the others remain below it, and no format meets the full gate.

## Experiment 58: Paired expert TensorOps tiles and larger quantized-MoE chunks (retained)

Status: retained. Toggles: `set_moe_expert_tile_pairs` (bench field `moe_expert_tile_pairs`) and `set_moe_routing_temporary_limit` (bench field `moe_routing_temporary_mib`). The default routing limit is 128 MiB for quantized experts. Dense (BF16/F16/F32) experts keep the documented 16 MiB Phase 6 bound.

After Experiment 57, the grouped Q4_K expert input projection ran at about 4.7 TFLOPS, against about 15.6 TFLOPS for the dense Q4_K TensorOps GEMM. Each expert threadgroup still decodes a full weight tile for only 32 routed rows. Experiment 8 showed that a larger `matmul2d` M tile (M=64) changes outputs, so the descriptor stays 32x64.

**Paired tiles.** Each threadgroup instead owns two consecutive 32-row tiles of one expert, decodes each weight tile once, and runs the identical 32x64 product on both row blocks. A second tile is skipped when the expert has at most 32 routes. Tests extend the Q4_K and Q6_K expert bit-identity checks to every combination of the fast decoder and paired tiles (K=64 and K=128, uneven segments, row tails), with identical output bits. They pass under Metal API and GPU Shader Validation.

**Larger chunks.** Pairing only helps experts with more than 32 routes in a chunk. Under the 16 MiB routing bound, LFM2.5 prompts split into chunks of about 200 tokens, which is about 21 routes per expert. Raising the bound for quantized experts gives each expert more rows per TensorOps pass. The bound applies only to quantized experts: dense experts do not use TensorOps and gain nothing, and the Granite MoE parity test keeps asserting the 16 MiB peak on its 256+-token prompt.

Flushed prefill profiles (summed GPU ms, LFM2.5):

| Configuration | 512 tokens | 1,024 tokens |
|---|---:|---:|
| Single tiles, 16 MiB | 327.4 | 599.5 |
| Paired tiles, 16 MiB | 309.1 | 556.6 |
| Paired tiles, 64 MiB | 263.3 | 519.0 |
| Paired tiles, 128 MiB | 262.3 | 495.5 |

End-to-end in-process A/B: five interleaved pairs, control = single tiles with 16 MiB, candidate = paired tiles with 128 MiB. Raw: `lfm2.5-8b-a1b-expert-tile-pairs-chunk128-ab.jsonl` and `.run.log`.

| Model / workload | Prefill tok/s C->N / L | Cached decode tok/s C->N / L | Complete generation tok/s C->N / L | First-token ms C->N / L | IDs |
|---|---:|---:|---:|---:|---:|
| LFM2.5-8B-A1B Q4_K_M / Short | 104.6->100.8 / 215.6 `[0.485->0.468; 0.959]` | 77.8->77.5 / 86.4 `[0.900->0.897; 0.996]` | 54.8->53.8 / 72.5 `[0.756->0.742; 0.981]` | 105.1->109.1 / 51.2 | 5/5; 5/5; 5/5 |
| LFM2.5-8B-A1B Q4_K_M / 128-token prompt | 716.4->727.0 / 1,511.1 `[0.474->0.481; 1.011]` | 75.9->76.3 / 98.2 `[0.774->0.777; 1.000]` | 43.7->44.2 / 67.5 `[0.648->0.655; 1.007]` | 178.7->176.1 / 84.9 | 5/5; 0/5; 0/5 |
| LFM2.5-8B-A1B Q4_K_M / 512-token prompt | 1,159.3->1,407.0 / 2,108.3 `[0.550->0.667; 1.221]` | 74.5->74.5 / 97.5 `[0.764->0.764; 1.000]` | 25.8->29.4 / 41.2 `[0.626->0.712; 1.142]` | 441.7->363.9 / 243.1 | 5/5; 5/5; 5/5 |
| LFM2.5-8B-A1B Q4_K_M / 1,024-token prompt | 1,272.4->1,467.8 / 2,141.5 `[0.594->0.685; 1.155]` | 73.1->72.8 / 96.4 `[0.758->0.755; 0.996]` | 16.6->18.5 / 25.8 `[0.644->0.719; 1.120]` | 804.8->697.6 / 478.4 | 5/5; 5/5; 5/5 |
| LFM2.5-8B-A1B Q4_K_M / Sustained decode | 101.5->104.0 / 233.7 `[0.434->0.445; 1.005]` | 76.6->76.6 / 101.2 `[0.757->0.757; 1.000]` | 72.5->72.6 / 95.7 `[0.758->0.759; 1.000]` | 108.4->105.7 / 47.3 | 5/5; 5/5; 5/5 |

**Results:**

- 512- and 1,024-token prefill rose 22.1% and 15.5%. Against llama.cpp this is 0.550 -> 0.667x and 0.594 -> 0.685x.
- First-token latency fell from 441.7 to 363.9 ms and from 804.8 to 697.6 ms.
- The 11-token short prompt runs neither change: 44 routes is below the 256-route TensorOps threshold, and it fits in one chunk either way. Its 0.959 median ratio comes from overlapping samples (control 97.7–111.8 tok/s, candidate 97.5–106.6 tok/s).
- Output IDs matched control in every pair.
- Median transient prefill peaks were unchanged or lower (512: 255.9 -> 254.5 MiB; 1,024: 255.4 -> 252.6 MiB), because the 256 MiB arena flush already dominates. Sampled RSS stayed at about 5.3 GiB.

## Experiment 59: Parallel RMSNorm without the BF16 midpoint fallback (retained by decision)

Status: retained, as an explicit trade-off. The `rmsnorm` kernel previously checked every BF16 output. If any value fell within eight F32 low-bit units of a BF16 rounding midpoint, thread 0 recomputed the row's sum of squares serially in ascending order, so the result matched the original ordered reduction exactly. That fires on about 40% of 2,048-wide rows, and its dependent 2,048-step add chain cost 32–39 µs per call (Experiment 53). Before this change, that was 2.4 of 12.7 ms of summed LFM2.5 decode GPU time, about 10% of Qwen2.5's, and 6% of LFM2.5's 512-token prefill. The fallback is removed; every row uses the 256-thread parallel sum of squares.

**Effect on reference tests.** The GGUF parity tests (Qwen3 Q8_0, LFM2.5 Q4_K_M against llama.cpp) are unchanged. Three BF16 Transformers-reference tests changed:

- **Granite 3.1 MoE, step 5, and LFM2.5-8B-A1B BF16, step 7.** Ferrum now produces an exact BF16 logit tie between the reference token and another token (27.5 = 27.5 and 36.75 = 36.75), and argmax selects the lower ID. The reference's own top-two margins there are 0.25, two and one BF16 steps. The tests now accept a different token only when the reference's top two are within two BF16 steps and Ferrum scores the reference token within one step of its choice. They then continue on the reference token, so every later step and the cache replay are still checked.
- **Qwen2.5-0.5B BF16.** One selected logit at step 2 deviated 0.516 from the reference (next largest 0.422), and the test's empirical 0.5 bound became 0.55. Greedy tokens, text, and EOS still match.

Validation: all 159 tests, and the library suite under Metal API and GPU Shader Validation.

A/B: five interleaved pairs per workload, control = build of `9065ead` (with the fallback), via `tools/phase7a_binary_ab.py`. The llama.cpp references are the saved 1ab7e5a rows. Raw: `*-rmsnorm-parallel-vs-9065ead.jsonl` and `.run.log`.

| Model / workload | Prefill tok/s C->N / L | Cached decode tok/s C->N / L | Complete generation tok/s C->N / L | First-token ms C->N / L | IDs |
|---|---:|---:|---:|---:|---:|
| LFM2.5-8B-A1B Q4_K_M / Short | 96.6->106.0 / 215.6 `[0.448->0.492; 1.075]` | 77.1->89.2 / 86.4 `[0.893->1.033; 1.160]` | 53.1->59.9 / 72.5 `[0.733->0.826; 1.102]` | 113.9->103.8 / 51.2 | 5/5; 5/5; 5/5 |
| LFM2.5-8B-A1B Q4_K_M / 128-token prompt | 709.0->752.1 / 1,511.1 `[0.469->0.498; 1.063]` | 75.1->89.5 / 98.2 `[0.766->0.912; 1.183]` | 43.2->46.9 / 67.5 `[0.640->0.695; 1.099]` | 180.5->170.2 / 84.9 | 0/5; 0/5; 0/5 |
| LFM2.5-8B-A1B Q4_K_M / 512-token prompt | 1,325.7->1,372.9 / 2,108.3 `[0.629->0.651; 1.024]` | 70.1->82.7 / 97.5 `[0.719->0.848; 1.174]` | 27.7->30.3 / 41.2 `[0.672->0.735; 1.082]` | 386.2->372.9 / 243.1 | 5/5; 5/5; 5/5 |
| LFM2.5-8B-A1B Q4_K_M / 1,024-token prompt | 1,403.0->1,442.3 / 2,141.5 `[0.655->0.673; 1.017]` | 69.4->80.6 / 96.4 `[0.719->0.836; 1.160]` | 17.8->18.6 / 25.8 `[0.690->0.720; 1.047]` | 729.9->710.0 / 478.4 | 5/5; 5/5; 5/5 |
| LFM2.5-8B-A1B Q4_K_M / Sustained decode | 95.2->108.1 / 233.7 `[0.407->0.462; 1.104]` | 72.4->85.4 / 101.2 `[0.716->0.844; 1.167]` | 68.3->80.0 / 95.7 `[0.713->0.836; 1.153]` | 115.5->101.8 / 47.3 | 0/5; 0/5; 5/5 |
| Qwen2.5 Q4_K_M / Short | 641.8->710.6 / 1,406.7 `[0.456->0.505; 1.110]` | 148.4->155.5 / 238.8 `[0.622->0.651; 1.052]` | 121.2->129.1 / 197.5 `[0.614->0.654; 1.061]` | 32.7->29.6 / 15.2 | 5/5; 0/5; 0/5 |
| Qwen2.5 Q4_K_M / 128-token prompt | 3,173.2->3,413.6 / 6,989.8 `[0.454->0.488; 1.043]` | 146.1->154.7 / 230.7 `[0.633->0.671; 1.056]` | 113.2->121.3 / 187.6 `[0.603->0.647; 1.048]` | 40.3->37.5 / 18.6 | 5/5; 5/5; 5/5 |
| Qwen2.5 Q4_K_M / 512-token prompt | 5,985.9->6,058.5 / 9,336.9 `[0.641->0.649; 1.018]` | 143.2->153.1 / 233.5 `[0.614->0.656; 1.060]` | 84.8->88.4 / 133.5 `[0.635->0.662; 1.042]` | 85.5->84.5 / 55.1 | 5/5; 5/5; 5/5 |
| Qwen2.5 Q4_K_M / 1,024-token prompt | 5,986.3->6,266.6 / 8,844.2 `[0.677->0.709; 1.027]` | 133.3->142.6 / 236.5 `[0.564->0.603; 1.067]` | 57.7->60.8 / 90.0 `[0.641->0.676; 1.042]` | 171.1->163.4 / 116.0 | 5/5; 5/5; 5/5 |
| Qwen2.5 Q4_K_M / Sustained decode | 654.4->732.8 / 1,385.7 `[0.472->0.529; 1.144]` | 151.6->161.7 / 244.3 `[0.620->0.662; 1.080]` | 147.5->158.2 / 221.8 `[0.665->0.713; 1.087]` | 32.1->28.7 / 15.4 | 0/5; 0/5; 0/5 |
| Qwen3 Q8_0 / Short | 563.4->644.8 / 1,258.3 `[0.448->0.512; 1.165]` | 114.4->127.2 / 164.5 `[0.695->0.773; 1.112]` | 95.6->107.7 / 143.8 `[0.665->0.749; 1.121]` | 37.3->32.6 / 17.0 | 5/5; 5/5; 5/5 |
| Qwen3 Q8_0 / 128-token prompt | 2,644.9->2,821.7 / 6,001.2 `[0.441->0.470; 1.092]` | 110.1->123.7 / 163.8 `[0.673->0.755; 1.123]` | 86.5->96.1 / 138.8 `[0.623->0.692; 1.113]` | 48.4->45.4 / 21.6 | 5/5; 0/5; 0/5 |
| Qwen3 Q8_0 / 512-token prompt | 4,403.6->4,793.2 / 7,099.6 `[0.620->0.675; 1.077]` | 103.9->115.6 / 154.9 `[0.671->0.746; 1.112]` | 60.8->66.8 / 95.2 `[0.639->0.702; 1.090]` | 116.3->106.8 / 72.4 | 5/5; 5/5; 5/5 |
| Qwen3 Q8_0 / 1,024-token prompt | 4,748.0->5,142.9 / 6,283.2 `[0.756->0.819; 1.085]` | 91.3->100.9 / 143.3 `[0.637->0.704; 1.106]` | 41.9->45.5 / 61.1 `[0.686->0.745; 1.096]` | 215.7->199.1 / 163.2 | 5/5; 5/5; 5/5 |
| Qwen3 Q8_0 / Sustained decode | 561.5->653.2 / 1,241.8 `[0.452->0.526; 1.168]` | 115.7->128.5 / 164.7 `[0.703->0.780; 1.112]` | 112.7->125.6 / 157.4 `[0.716->0.798; 1.115]` | 37.4->32.1 / 17.2 | 0/5; 0/5; 0/5 |

[exited with code 0]

Cached decode rose 16–18% for LFM2.5, 5–8% for Qwen2.5 Q4_K_M, and 11–12% for Qwen3 Q8_0. Prefill rose 2–17%.

LFM2.5 short-prompt decode reached 89.2 tok/s against llama.cpp's 86.4 (1.033x), the first workload where Ferrum decodes faster than the pinned llama.cpp build. LFM2.5 decode is now 0.84–1.03x across workloads, and Qwen3's 1,024-token prefill is 0.819x. No format yet meets every threshold of the gate: LFM2.5's 512-token, 1,024-token, and sustained decode are 0.836–0.848x, and prefill is below 0.80x elsewhere.

**Fidelity cost.** Against llama.cpp's saved sequences, LFM2.5 sustained decode previously matched all 129 tokens and now diverges at generated index 46. The LFM2.5 128-token case diverges at index 8 instead of 9. Every other workload diverges at the same index as control or not at all.

## Experiment 60: Four-wide BF16 loads in the M=1 attention score kernel (retained)

Status: retained. Toggle: `set_attention_scores_vector`, bench field `attention_scores_vector`. Output is bit-identical.

After Experiment 59, LFM2.5 decode reached 1.03x llama.cpp at short context but only 0.836–0.848x at the long-context workloads. The flushed 1,024-context profile (`lfm2.5-8b-a1b-1024-decode-op-profile.json`) attributed 44.3 µs per layer to the scalar `attention_scores` kernel, which serves M=1 since Experiment 48's GQA scores kernel was rejected. That kernel has one thread per (head, position), each loading its query and key rows one BF16 element at a time through the generic `load`.

The candidate reads both rows as `ushort4` when the tensor is BF16, the head dimension is divisible by four, and both view offsets are 8-byte aligned. It accumulates the four products in the same ascending order as the scalar loop; other cases take the scalar loop. Test `vector_bf16_attention_scores_are_bit_identical` covers GQA groups 4, 7, 1, and 2, head dimensions 64, 80, 128, and 33 (scalar fallback), aligned and row-offset key views, and an unaligned query view. It requires identical output bits and passes under Metal API and GPU Shader Validation. All 160 tests pass.

Flushed profile at 1,024 context: `attention_scores` 44.8 -> 13.6 µs per LFM2.5 attention layer. In-process A/B, five pairs, 1,024-token and sustained workloads. Raw: `*-attention-scores-vector-ab.jsonl` (with `.run.log`).

| Model / workload | Prefill tok/s C->V `[V/C]` | Cached decode tok/s C->V `[V/C]` | Complete generation tok/s C->V `[V/C]` | First-token ms C->V | IDs C=V |
|---|---:|---:|---:|---:|---:|
| LFM2.5 / 1,024-token prompt | 1,492.4->1,471.4 `[0.986]` | 82.3->84.2 `[1.023]` | 19.3->19.1 `[0.993]` | 686.2->695.9 | 5/5 |
| LFM2.5 / Sustained decode | 105.6->104.8 `[1.013]` | 85.9->86.0 `[1.005]` | 80.9->80.7 `[1.005]` | 104.2->105.0 | 5/5 |
| Qwen2.5 Q4_K_M / 1,024-token prompt | 6,389.3->6,415.9 `[1.005]` | 143.0->151.9 `[1.063]` | 62.0->63.8 `[1.029]` | 160.3->159.6 | 5/5 |
| Qwen2.5 Q4_K_M / Sustained decode | 743.7->710.3 `[0.982]` | 162.6->166.9 `[1.033]` | 160.3->163.7 `[1.032]` | 28.2->29.6 | 5/5 |
| Qwen3 Q8_0 / 1,024-token prompt | 5,224.8->5,200.6 `[0.998]` | 100.5->109.3 `[1.083]` | 46.0->47.5 `[1.034]` | 196.0->196.9 | 5/5 |
| Qwen3 Q8_0 / Sustained decode | 648.2->639.0 `[0.986]` | 134.0->135.5 `[1.013]` | 129.5->132.3 `[1.028]` | 32.4->32.9 | 5/5 |

Cached decode improved at 1,024 context by 2.3% (LFM2.5), 6.3% (Qwen2.5), and 8.3% (Qwen3), and on sustained decode by 0.5–3.3%. Prefill uses the TensorOps score path, so its ±1–2% changes are noise. Every pair produced identical IDs.

Against the saved llama.cpp rows, LFM2.5 cached decode is now 84.2 / 96.4 tok/s (0.873x) at 1,024 context and 86.0 / 101.2 tok/s (0.850x) sustained. Together with short context (1.03x after Experiment 59), LFM2.5 meets the 0.85x decode threshold on the measured short, 1,024-token, and sustained workloads; the 512-token workload was not rerun here. Its 512- and 1,024-token prefill (0.65–0.69x) remains below the 0.80x prefill threshold, so the gate is not met.

## Experiment 61: Paired M tiles for dense GGUF TensorOps GEMMs (retained, M>=512 and N>=512)

Status: retained. Toggle: `set_dense_mpp_tile_pairs`, bench field `dense_mpp_tile_pairs`. Output is bit-identical.

This is the dense counterpart of Experiment 58. Each threadgroup of the eleven dense GGUF TensorOps kernels (Q4_0, Q5_0, Q5_1 K128/K64, Q4_K K128/K64, Q5_K K128/K64, Q6_K, Q8_0 K128/K64) can own two consecutive 64-row M tiles. It reuses every dequantized weight tile for both, running the identical 64x64 product on each (flag bit 1).

The 512-token prefill profiles taken after Experiment 60 (`*-512-prefill-op-profile-after-exp60.json`) show the GEMMs at about 60% of Qwen2.5/Qwen3 prefill. A first Qwen2.5 Q4_K_M profile gave two results:

- Q5_0 gate/up improved from 480–497 to 422–429 µs per call.
- The N=128 k/v projections slowed from 27 to 39–58 µs, because too few threadgroups remained. Output width is therefore gated at N>=512.

The full A/B below, five pairs with the N gate only, also showed 128-token prompts losing 1.3–3.7% while 512- and 1,024-token prompts gained 3.8–8.1%. Pairing is therefore also gated at M>=512 prompt rows.

Tests: `*_tensorops_paired_tiles_are_bit_identical` for every paired kernel (N=517 with a column tail, M=600 or 1,025 to reach each tile variant), and the Q5_0 fast-dequant/pairs combination test. They pass under Metal API and GPU Shader Validation.

Raw: `*-dense-mpp-tile-pairs-ab.jsonl` and `.run.log`. This A/B predates the M gate; with the gate, the short and 128-token rows run the control path.

| Model / workload | Prefill tok/s C->P `[P/C]` | Cached decode tok/s C->P `[P/C]` | Complete generation tok/s C->P `[P/C]` | First-token ms C->P | IDs C=P |
|---|---:|---:|---:|---:|---:|
| Qwen2.5 Q4_K_M / Short | 730.6->726.5 `[0.985]` | 165.5->163.6 `[0.987]` | 135.8->134.6 `[0.989]` | 28.7->28.9 | 5/5 |
| Qwen2.5 Q4_K_M / 128-token prompt | 3,539.0->3,407.1 `[0.971]` | 162.5->164.8 `[1.022]` | 126.7->126.1 `[0.998]` | 36.2->37.6 | 5/5 |
| Qwen2.5 Q4_K_M / 512-token prompt | 6,314.7->6,549.0 `[1.038]` | 161.1->162.2 `[1.007]` | 92.7->94.3 `[1.014]` | 81.1->78.2 | 5/5 |
| Qwen2.5 Q4_K_M / 1,024-token prompt | 6,331.2->6,688.4 `[1.065]` | 155.6->155.3 `[0.999]` | 63.7->65.8 `[1.039]` | 161.7->153.1 | 5/5 |
| Qwen2.5 Q4_K_M / Sustained decode | 725.2->732.9 `[1.011]` | 170.1->170.0 `[1.001]` | 165.8->165.8 `[1.002]` | 29.0->28.7 | 5/5 |
| Qwen3 Q8_0 / Short | 648.1->648.0 `[0.984]` | 136.6->136.0 `[0.994]` | 113.4->113.4 `[0.996]` | 32.4->32.4 | 5/5 |
| Qwen3 Q8_0 / 128-token prompt | 2,963.0->2,925.8 `[0.987]` | 133.5->134.9 `[1.009]` | 104.6->105.0 `[1.010]` | 43.2->43.8 | 5/5 |
| Qwen3 Q8_0 / 512-token prompt | 4,685.3->5,028.0 `[1.070]` | 124.5->124.3 `[1.000]` | 68.7->70.7 `[1.026]` | 109.3->101.8 | 5/5 |
| Qwen3 Q8_0 / 1,024-token prompt | 5,060.5->5,441.7 `[1.074]` | 111.2->111.2 `[1.001]` | 47.1->49.0 `[1.041]` | 202.4->188.2 | 5/5 |
| Qwen3 Q8_0 / Sustained decode | 656.2->646.1 `[0.975]` | 138.1->138.8 `[1.006]` | 134.6->135.4 `[1.005]` | 32.0->32.5 | 5/5 |
| Qwen2.5 Q6_K / Short | 741.6->732.4 `[0.992]` | 154.9->154.4 `[1.002]` | 129.0->128.8 `[0.998]` | 28.3->28.7 | 5/5 |
| Qwen2.5 Q6_K / 128-token prompt | 3,511.7->3,416.0 `[0.963]` | 154.1->152.6 `[0.987]` | 121.3->119.6 `[0.984]` | 36.5->37.5 | 5/5 |
| Qwen2.5 Q6_K / 512-token prompt | 6,255.5->6,640.7 `[1.063]` | 151.6->150.8 `[0.990]` | 89.1->91.2 `[1.022]` | 81.8->77.1 | 5/5 |
| Qwen2.5 Q6_K / 1,024-token prompt | 6,196.8->6,696.2 `[1.081]` | 146.1->145.6 `[0.997]` | 61.3->64.1 `[1.045]` | 165.2->152.9 | 5/5 |
| Qwen2.5 Q6_K / Sustained decode | 729.2->734.9 `[0.999]` | 159.0->159.2 `[1.001]` | 154.8->155.0 `[0.999]` | 28.8->28.6 | 5/5 |
| LFM2.5 / Short | 99.6->99.5 `[0.999]` | 86.7->86.4 `[1.002]` | 57.9->58.1 `[1.005]` | 110.5->110.6 | 5/5 |
| LFM2.5 / 128-token prompt | 713.9->732.8 `[1.018]` | 88.3->86.7 `[0.978]` | 47.4->47.6 `[1.007]` | 179.3->174.7 | 5/5 |
| LFM2.5 / 512-token prompt | 1,369.6->1,424.1 `[1.040]` | 85.1->85.4 `[0.999]` | 30.3->31.0 `[1.027]` | 373.8->359.5 | 5/5 |
| LFM2.5 / 1,024-token prompt | 1,455.4->1,469.6 `[1.010]` | 84.3->83.8 `[0.995]` | 19.0->19.1 `[0.998]` | 703.6->696.8 | 5/5 |
| LFM2.5 / Sustained decode | 101.6->102.2 `[0.996]` | 87.4->87.2 `[0.997]` | 82.3->81.9 `[0.997]` | 108.2->107.7 | 5/5 |

At 512 and 1,024 tokens, prefill rose 3.8–6.5% (Qwen2.5 Q4_K_M), 7.0–7.4% (Qwen3 Q8_0), 6.3–8.1% (Qwen2.5 Q6_K), and 1.0–4.0% (LFM2.5 dense layers). Every pair produced identical IDs.

## Experiment 62: Per-execution RoPE cos/sin table (retained)

Status: retained. Toggle: `set_rope_table`, bench field `rope_table`. Output is bit-identical.

`rope_split` computed `pow`, `cos`, and `sin` for every rotated pair. Every q and k rotation in every layer therefore recomputed the same angle table: 56 calls, 4.3 ms of Qwen3's 512-token prefill profile.

Inside a batched execution, the first rotation for a given (offset, rows, head dimension, theta) now dispatches `rope_table`, using `rope_split`'s exact angle expression. Every rotation then reads cos/sin from that table through `rope_split_table`. The table lives until the execution ends. Test: `rope_table_matches_per_element_rotation_bitwise` (F32/F16/BF16, several shapes, nonzero offsets, and shared q/k use), which passes under validation.

In-process A/B, five pairs. Raw: `*-rope-table-ab.jsonl` and `.run.log`.

| Model / workload | Prefill tok/s C->R `[R/C]` | Cached decode tok/s C->R `[R/C]` | Complete generation tok/s C->R `[R/C]` | First-token ms C->R | IDs C=R |
|---|---:|---:|---:|---:|---:|
| Qwen3 Q8_0 / Short | 653.7->649.0 `[0.980]` | 136.6->137.8 `[1.009]` | 114.1->114.4 `[0.999]` | 32.1->32.4 | 5/5 |
| Qwen3 Q8_0 / 128-token prompt | 2,962.2->2,975.7 `[1.005]` | 133.3->135.6 `[1.013]` | 104.6->105.8 `[1.006]` | 43.2->43.0 | 5/5 |
| Qwen3 Q8_0 / 512-token prompt | 5,038.6->5,100.9 `[1.022]` | 124.6->125.8 `[1.011]` | 70.7->71.6 `[1.019]` | 101.6->100.4 | 5/5 |
| Qwen3 Q8_0 / 1,024-token prompt | 5,464.1->5,565.1 `[1.019]` | 111.2->111.9 `[1.007]` | 49.2->49.8 `[1.013]` | 187.4->184.0 | 5/5 |
| Qwen3 Q8_0 / Sustained decode | 645.8->649.6 `[0.975]` | 137.9->139.5 `[1.011]` | 134.6->135.9 `[1.009]` | 32.5->32.3 | 5/5 |
| Qwen2.5 Q4_K_M / Short | 718.7->722.7 `[1.002]` | 164.0->165.0 `[1.007]` | 133.3->135.6 `[1.016]` | 29.2->29.1 | 5/5 |
| Qwen2.5 Q4_K_M / 128-token prompt | 3,504.4->3,530.9 `[1.001]` | 164.1->163.9 `[1.002]` | 126.8->126.9 `[0.999]` | 36.5->36.3 | 5/5 |
| Qwen2.5 Q4_K_M / 512-token prompt | 6,495.8->6,545.0 `[0.997]` | 161.1->160.4 `[1.001]` | 93.8->93.4 `[1.003]` | 78.8->78.2 | 5/5 |
| Qwen2.5 Q4_K_M / 1,024-token prompt | 6,691.5->6,732.2 `[1.004]` | 155.4->155.1 `[0.998]` | 65.9->66.0 `[1.001]` | 153.0->152.1 | 5/5 |
| Qwen2.5 Q4_K_M / Sustained decode | 727.2->723.0 `[1.016]` | 169.0->170.4 `[1.007]` | 164.8->165.9 `[1.007]` | 28.9->29.0 | 5/5 |

Qwen3 (head dimension 128, 28 layers): 512- and 1,024-token prefill +1.9–2.2%, and cached decode +0.7–1.3%. Qwen2.5 (head dimension 64) is within noise. All 166 tests pass.

Against the saved llama.cpp rows, the current tree's A/B medians give 1,024-token prefill of 6,732 / 8,844 tok/s (0.76x) for Qwen2.5 Q4_K_M, 5,565 / 6,283 (0.89x) for Qwen3, and 6,696 / 9,077 (0.74x) for Qwen2.5 Q6_K. At 512 tokens it is 0.70x, 0.72x, and 0.69x.

## Experiment 63: Transient arena reuse within the encoding epoch (retained)

Status: retained. Toggle: `set_arena_epoch_reuse`, bench field `arena_epoch_reuse`. Outputs are unchanged.

**Diagnosis.** Batched prefill counters from Experiments 61–62 show wall time well above GPU time. At 1,024 tokens:

| Model | Prefill | GPU | Command buffers |
|---|---:|---:|---:|
| Qwen2.5 | 152.1 ms | 123.4 ms | 14 |
| LFM2.5 | 696.8 ms | 501.4 ms | 23 |

The transient peak sat at the 256 MiB live budget in every long prompt. Allocation counters explained most of the gap: Qwen2.5 allocated 262 MiB of fresh buffers (13.8 ms) at 512 tokens and 496 MiB (26.6 ms) at 1,024; LFM2.5 allocated 2.1 GiB (110 ms) at 1,024.

The arena recycled a buffer only after its command buffer had *completed*. Within one long forward epoch, every intermediate therefore needed fresh, zero-filled storage until the budget forced a flush and a completion wait.

**Change.** `allocate_output` now also reuses storage retired earlier in the *current* epoch (same completion flag as the open submission). This is safe under the runtime's ordering rules:

- The retired tensor has no owner, so no later dispatch reads its old contents.
- Every earlier reader is ordered before the new writer: the serial encoder executes in order, the concurrent encoder inserts a write-after-read buffer barrier from its range tracking (Experiment 47), and Metal's tracked hazards order separate encoders.

A reused pending buffer stays counted as live until its epoch completes, so live-byte accounting is unchanged.

**Tests.**

- `epoch_reuse_orders_rewrites_after_pending_reads` runs a 64-step chain under both serial and concurrent encoders, dropping each tensor right after its last read is encoded. It checks every value and that reuse occurred within one command buffer.
- `bounded_epochs_preserve_live_dependency_chain` now keeps its intermediates alive, so it still exercises budget-bounded epochs.
- All 108 library tests pass under Metal API and GPU Shader Validation, and all 167 tests pass.

A/B: five interleaved pairs, control = completed-only reuse. The llama.cpp references are the saved 1ab7e5a rows. Raw: `*-arena-epoch-reuse-ab.jsonl` and `.run.log`.

| Model / workload | Prefill tok/s C->N / L | Cached decode tok/s C->N / L | Complete generation tok/s C->N / L | First-token ms C->N / L | IDs |
|---|---:|---:|---:|---:|---:|
| LFM2.5-8B-A1B Q4_K_M / Short | 107.0->104.4 / 215.6 `[0.497->0.484; 0.999]` | 86.4->87.4 / 86.4 `[0.999->1.011; 1.015]` | 59.2->59.1 / 72.5 `[0.816->0.815; 1.008]` | 102.8->105.3 / 51.2 | 5/5; 5/5; 5/5 |
| LFM2.5-8B-A1B Q4_K_M / 128-token prompt | 750.7->775.8 / 1,511.1 `[0.497->0.513; 1.021]` | 87.1->87.5 / 98.2 `[0.888->0.892; 1.003]` | 47.9->48.8 / 67.5 `[0.710->0.724; 1.011]` | 170.5->165.0 / 84.9 | 5/5; 0/5; 0/5 |
| LFM2.5-8B-A1B Q4_K_M / 512-token prompt | 1,463.0->1,712.5 / 2,108.3 `[0.694->0.812; 1.176]` | 86.0->86.6 / 97.5 `[0.882->0.889; 1.007]` | 31.6->35.0 / 41.2 `[0.766->0.850; 1.113]` | 350.0->299.0 / 243.1 | 5/5; 5/5; 5/5 |
| LFM2.5-8B-A1B Q4_K_M / 1,024-token prompt | 1,489.6->1,729.9 / 2,141.5 `[0.696->0.808; 1.167]` | 84.3->85.5 / 96.4 `[0.874->0.886; 1.010]` | 19.3->21.9 / 25.8 `[0.750->0.850; 1.131]` | 687.4->592.0 / 478.4 | 5/5; 5/5; 5/5 |
| LFM2.5-8B-A1B Q4_K_M / Sustained decode | 106.0->109.7 / 233.7 `[0.454->0.470; 1.032]` | 87.3->88.3 / 101.2 `[0.863->0.873; 1.009]` | 82.1->83.3 / 95.7 `[0.858->0.870; 1.013]` | 103.7->100.2 / 47.3 | 5/5; 0/5; 0/5 |
| Qwen2.5 Q4_K_M / Short | 729.5->741.7 / 1,406.7 `[0.519->0.527; 1.010]` | 165.8->166.6 / 238.8 `[0.694->0.698; 1.004]` | 135.8->137.4 / 197.5 `[0.688->0.696; 1.014]` | 28.8->28.3 / 15.2 | 5/5; 0/5; 0/5 |
| Qwen2.5 Q4_K_M / 128-token prompt | 3,478.3->3,777.4 / 6,989.8 `[0.498->0.540; 1.086]` | 164.6->165.3 / 230.7 `[0.713->0.716; 1.005]` | 126.5->131.6 / 187.6 `[0.675->0.702; 1.034]` | 36.8->33.9 / 18.6 | 5/5; 5/5; 5/5 |
| Qwen2.5 Q4_K_M / 512-token prompt | 6,518.3->6,989.0 / 9,336.9 `[0.698->0.749; 1.077]` | 162.9->161.2 / 233.5 `[0.698->0.691; 0.986]` | 94.4->97.7 / 133.5 `[0.707->0.732; 1.039]` | 78.5->73.3 / 55.1 | 5/5; 5/5; 5/5 |
| Qwen2.5 Q4_K_M / 1,024-token prompt | 6,716.5->7,822.4 / 8,844.2 `[0.759->0.884; 1.185]` | 155.9->156.6 / 236.5 `[0.659->0.662; 1.006]` | 66.2->72.4 / 90.0 `[0.735->0.804; 1.104]` | 152.5->130.9 / 116.0 | 5/5; 5/5; 5/5 |
| Qwen2.5 Q4_K_M / Sustained decode | 720.9->745.0 / 1,385.7 `[0.520->0.538; 1.027]` | 170.7->171.6 / 244.3 `[0.699->0.702; 1.000]` | 166.3->167.2 / 221.8 `[0.750->0.754; 1.003]` | 29.1->28.2 / 15.4 | 5/5; 0/5; 0/5 |
| Qwen3 Q8_0 / Short | 643.7->670.4 / 1,258.3 `[0.512->0.533; 1.038]` | 137.1->140.0 / 164.5 `[0.833->0.851; 1.022]` | 113.5->116.8 / 143.8 `[0.790->0.812; 1.029]` | 32.6->31.3 / 17.0 | 5/5; 5/5; 5/5 |
| Qwen3 Q8_0 / 128-token prompt | 2,996.4->3,184.8 / 6,001.2 `[0.499->0.531; 1.071]` | 135.2->137.4 / 163.8 `[0.825->0.839; 1.017]` | 105.5->109.0 / 138.8 `[0.761->0.786; 1.052]` | 42.7->40.2 / 21.6 | 5/5; 0/5; 0/5 |
| Qwen3 Q8_0 / 512-token prompt | 5,109.9->5,568.8 / 7,099.6 `[0.720->0.784; 1.092]` | 125.6->127.1 / 154.9 `[0.811->0.821; 1.012]` | 71.4->74.8 / 95.2 `[0.750->0.786; 1.045]` | 100.2->91.9 / 72.4 | 5/5; 5/5; 5/5 |
| Qwen3 Q8_0 / 1,024-token prompt | 5,555.2->6,040.1 / 6,283.2 `[0.884->0.961; 1.087]` | 111.8->113.4 / 143.3 `[0.780->0.792; 1.018]` | 49.7->52.7 / 61.1 `[0.814->0.862; 1.059]` | 184.3->169.5 / 163.2 | 5/5; 5/5; 5/5 |
| Qwen3 Q8_0 / Sustained decode | 647.5->679.6 / 1,241.8 `[0.521->0.547; 1.082]` | 139.8->141.5 / 164.7 `[0.849->0.859; 1.012]` | 136.7->138.6 / 157.4 `[0.869->0.881; 1.015]` | 32.4->30.9 / 17.2 | 5/5; 0/5; 0/5 |
| Qwen2.5 Q6_K / Short | 727.9->752.3 / 1,478.6 `[0.492->0.509; 1.029]` | 156.7->159.3 / 199.1 `[0.787->0.800; 1.028]` | 129.5->133.5 / 173.9 `[0.744->0.768; 1.036]` | 28.8->27.9 / 14.5 | 5/5; 0/5; 0/5 |
| Qwen2.5 Q6_K / 128-token prompt | 3,531.1->3,769.8 / 7,068.9 `[0.500->0.533; 1.057]` | 154.7->156.7 / 201.8 `[0.766->0.776; 1.004]` | 122.0->124.9 / 168.2 `[0.726->0.742; 1.022]` | 36.3->34.0 / 18.4 | 5/5; 5/5; 5/5 |
| Qwen2.5 Q6_K / 512-token prompt | 6,656.1->7,084.8 / 9,643.1 `[0.690->0.735; 1.073]` | 152.4->153.1 / 203.2 `[0.750->0.754; 1.001]` | 91.7->95.4 / 125.0 `[0.734->0.763; 1.043]` | 76.9->72.3 / 53.4 | 5/5; 5/5; 5/5 |
| Qwen2.5 Q6_K / 1,024-token prompt | 6,780.9->7,826.7 / 9,077.0 `[0.747->0.862; 1.166]` | 147.0->148.1 / 197.2 `[0.746->0.751; 1.006]` | 64.6->70.7 / 86.3 `[0.749->0.819; 1.106]` | 151.0->130.8 / 113.1 | 5/5; 5/5; 5/5 |
| Qwen2.5 Q6_K / Sustained decode | 726.9->749.0 / 1,451.0 `[0.501->0.516; 1.041]` | 159.7->160.8 / 202.7 `[0.788->0.793; 1.007]` | 155.5->156.9 / 191.2 `[0.813->0.821; 1.011]` | 28.9->28.0 / 14.7 | 5/5; 0/5; 0/5 |


**Results.** Prefill rose at 512 and 1,024 tokens:

| Model | 512 tokens | 1,024 tokens |
|---|---:|---:|
| LFM2.5 | +17.6% | +16.7% |
| Qwen2.5 Q4_K_M | +7.7% | +18.5% |
| Qwen3 | +9.2% | +8.7% |
| Qwen2.5 Q6_K | +7.3% | +16.6% |

Decode changed by 0–3%, and IDs matched in every pair.

**Gate status.**

- **LFM2.5-8B-A1B Q4_K_M** meets the decode and medium/long prefill thresholds against the saved llama.cpp rows: prefill 0.812x (512) and 0.808x (1,024), cached decode 0.873–1.011x. Its short (11-token) and 128-token prefill are 0.48x and 0.51x. Those remain the routine-workload exception that keeps the gate open.
- **At 1,024 tokens**, Qwen2.5 Q4_K_M (0.884x), Qwen3 (0.961x), and Qwen2.5 Q6_K (0.862x) meet the prefill threshold.
- **At 512 tokens**, the same three models are 0.735–0.784x.
- **Qwen decode** remains 0.66–0.85x.
