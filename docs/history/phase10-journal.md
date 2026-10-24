# Phase 10 journal — long-context attention

> **History.** Written as this phase's work happened and not updated since, so it may describe code or limits that have changed. The current docs start at [docs/README.md](../README.md).

Hardware: Apple M5 (10-core GPU), 32 GB unified memory, macOS 27. Target:
Tiel-Coder 35B-A3B IQ4_XS (GGUF name `Ornith-1.5-35B`; 40 trunk layers, 10 of
them gated attention with 16 query / 2 KV heads × 256). Raw logs are in
[`measurements/phase10/`](../measurements/phase10/).

The starting complaint: TUI generation speed on Tiel fell from 40–50 tok/s to
about 20 tok/s past 32K tokens of context. Two things grow with context. Every
attention layer reads the whole K/V cache per step (20 KiB per cached token in
F16), and a speculative verify pays that read through a slower kernel.

## Experiment 1 — where verify attention went

`examples/hybrid_attention.rs` times one attention layer for `m` query tokens
against `keys` cached positions. Before (Phase 9 routing):

| keys | m = 1 (decode) | m = 3 (2 drafts) | m = 5 and 9 |
|---|---|---|---|
| 8K | 212 µs | 770 µs | 7.44 ms |
| 32K | 745 µs (90 GB/s) | 2.72 ms (25 GB/s) | 29.7 ms (2.3 GB/s) |
| 128K | 2.58 ms | 10.6 ms | 119 ms |

Two separate problems:

- Rows 2–15 used the simdgroup-matrix flash kernel. Each SIMD group loads K/V
  from device memory as transposed, strided 8×8 tiles, which reaches about
  25 GB/s.
- That kernel only split keys across threadgroups when `m × heads ≤ 64`. With
  16 query heads, any verify of 5 or more rows ran 32K keys on 2–6
  threadgroups: a 40× cliff at 4 drafts.

## Experiment 2 — staged simdgroup kernel (rejected)

The first replacement staged each 32-key K block, then V block, into
threadgroup memory with coalesced loads shared by up to eight SIMD groups,
keeping Q in registers. It removed the cliff (m = 5 at 32K: 29.7 → 3.4 ms) but
helped little at m = 3 (2.72 → 2.18 ms). Sweeping SIMD groups (2/4/8) and split
targets (64–1024) moved nothing beyond noise: time scaled with rows, i.e.
simdgroup-matrix throughput (~0.7 TFLOP/s) was the limit. Removed.

## Experiment 3 — TensorOps verify kernel (kept)

Prompt attention already ran on TensorOps (`matmul2d`) at ~3.9 TFLOP/s, so
verify moved there too. A tiny kernel copies Q to KV-head-major rows
(`[kv_head][t·group + g][256]`), so each KV head's (token, head) rows form one
strided matrix. One threadgroup takes 32 of those rows and one key split; S and
O run as `matmul2d` on 64-key blocks read straight from the cache, with the
online softmax in threadgroup memory and the existing split reduction.

| 32K keys | m = 2 | 3 | 5 | 9 | 15 |
|---|---|---|---|---|---|
| 64-row tile | 677 µs | 705 | 1152 | 1870 | 1964 |
| 32-row tile | — | 637 | 683 | 883 | 1036 |

32-row tiles beat 64-row ones even when a second tile re-reads the keys (they
come from cache). Split target swept 16–1024: 16–64 within noise, larger loses
up to 15%; 32 is kept. Verify of 3 rows now costs what one-row decode does.

## Experiment 4 — Q8 K/V cache

`KvCache::Q8` stores K and V as int8 with one F16 scale per 32 values
(llama.cpp's `q8_0` rounding), 1.0625 bytes per element. A cache tensor holds
all rows' values, then all scales, so loads stay aligned. Writes quantize in
the attention-prep and V-store kernels (one 32-value block per SIMD group,
`simd_max` for the scale). Every attention kernel has a Q8 variant:

- decode dots each 32-value block unscaled and applies its scale once;
- verify and prompt attention dequantize 64-key × 128-dim halves into
  threadgroup memory (aliasing the score tile) before `matmul2d`, with 16-byte
  loads per thread (8-byte loads were 20% slower).

The planner, MTP cache, CLI/server (`--kv-cache`, `-ctk`/`-ctv`), TUI setting
and `FERRUM_KV_CACHE` all carry the format. F16 stays the default.

| Model | K/V | KiB/token | auto context | K/V at auto |
|---|---|---|---|---|
| Swift 27B (dense) | F16 | 64.0 | 101,120 | 6.17 GiB |
| Swift 27B | Q8 | 34.0 | 190,208 | 6.17 GiB |
| Tiel 35B-A3B | F16 | 20.0 | 262,144 | 5.00 GiB |
| Tiel 35B-A3B | Q8 | 10.6 | 262,144 | 2.66 GiB |

Attention alone, Q8 vs F16: decode 575 vs 768 µs at 32K, 1.92 vs 2.69 ms at
128K.

**Accuracy** (`hybrid_kld`, Ferrum docs and sources, llama.cpp b11146 base):

| Run | vs | mean KLD | same top-1 | PPL |
|---|---|---|---|---|
| F16, 512 ctx | llama.cpp | 0.0083 | 95.8% | 17.362 (llama.cpp 17.323) |
| Q8, 512 ctx | llama.cpp | 0.0094 | 95.2% | 17.368 |
| Q8, 512 ctx | Ferrum F16 | 0.0086 | 95.5% | |
| F16, 8K ctx | llama.cpp | 0.0089 | 95.0% | 18.284 (llama.cpp 18.259) |
| Q8, 8K ctx | llama.cpp | 0.0095 | 95.1% | 18.245 |
| Q8, 8K ctx | Ferrum F16 | 0.0095 | 94.9% | |

Ferrum F16 scored against its own written logits gives 0.000000, so the Q8
number is real: about the size of this model's known sensitivity to F16
rounding (Phase 8), with no perplexity cost.

**Speculation.** `hybrid_spec_check` with Q8 and blocks of 5+ rows showed
verify-vs-decode logit gaps of 0.47 and a greedy divergence at a 0.16 margin.
With `FERRUM_HYBRID_EXACT=1` (same projection kernels for any row count) the
gap is 0.021 and output identical. The cause is not attention: from 5 rows the
projections switch from GEMV to TensorOps tiles, K/V differ in the last bit,
and 8-bit rounding amplifies that. Verification stays correct; greedy output
with a drafter can differ from greedy without one at close calls.

## Experiment 5 — staged prompt attention

The Q8 prompt kernel, despite dequantizing, ran faster than the F16 one that
reads the cache as a device tensor. Staging F16 the same way:

| 512-row chunk at | direct F16 | staged F16 | staged Q8 |
|---|---|---|---|
| 32K keys | 69–70 ms (3.9 TFLOP/s) | 60 ms (4.6) | 54–56 ms (5.0) |
| 128K keys | 281–300 ms | 236–248 ms | 218–229 ms |

Staged is the default for prompt chunks. For verify widths staging loses
(m = 9: 0.85 → 1.43 ms), so F16 verify keeps direct loads.

## End to end (`hybrid_bench`, mean of two alternating runs)

| | main F16 | branch F16 | branch Q8 |
|---|---|---|---|
| tg @ 0 | 40.1 | 40.1 | 40.0 |
| tg @ 32K | 31.2 | 31.3 | 33.2 |
| tg @ 64K | 26.9 | 25.9 | 28.9 |
| pp512 @ 32K | 389 | 406 | 435 |

Plain decode is unchanged in F16 (the decode kernel is the same; ±5% run noise)
and 7–9% faster with Q8 at depth.

## Speculative decoding at depth

`hybrid_spec` with the DFlash2 drafter (`jzinno/Ornith-1.5-35B-A3B-DFlash2`;
Tiel's NextN block has routed experts, so MTP doesn't run on it), 2 prompts ×
192 tokens, greedy, after `--depth` tokens of Ferrum's docs in the prompt.

| | main F16 | branch F16 | branch Q8 |
|---|---|---|---|
| depth 0, 2 drafts | 59.4 tok/s (1.50×) | 62.5 (1.59×) | 63.6 (1.59×) |
| depth 32K, 2 drafts | 32.2 (1.05×), verify 59.5 ms | **44.8 (1.45×)**, verify 37.5 ms | 45.5 (1.39×), verify 36.0 ms |
| depth 32K, 8 drafts | 9.3 (0.30×), verify 375 ms | 35.6 (1.17×), verify 79 ms | 32.8 (1.01×), verify 81 ms |

At 32K the drafter's gain was almost gone on main (1.05×); it is back to 1.45×.
The 8-draft row is the Experiment 1 cliff; it is gone, but 2 drafts stays the
better MoE default (each verify row still routes to its own experts). Greedy
mismatches: main 0/2 and 2/2, branch F16 0/2 and 1/2, all at top-2 margins
0.04–0.11 (the existing single- vs multi-row rounding); Q8 1/2 and 2/2 at
margins 0.08–0.11, as explained in Experiment 4.

## Swift 27B (dense) check

Same tools on Swift-1.5-Qwen3.8-27B Q4_K_M (log: `measurements/phase10/swift.txt`),
DFlash2 drafter `z-lab/Qwen3.8-27B-DFlash2` (16 drafts, so verify is a 17-row
prompt-attention chunk), 2 prompts × 96 tokens.

| | main F16 | branch F16 | branch Q8 |
|---|---|---|---|
| KLD vs llama.cpp (512 ctx) | 0.000156, 99.46% | 0.000156, 99.46% | 0.000190, 99.41% |
| tg @ 0 / @ 16K | 6.87 / 6.46 | 7.00 / 6.42 | 7.02 / 6.46 |
| pp512 @ 16K | 138.7 | 138.8 | 142.5 |
| spec, depth 0 | 21.2 tok/s (3.05×) | 21.4 (3.09×) | 21.4 (3.10×) |
| spec, depth 16K | 8.85 (1.37×), verify 450 ms | **16.3 (2.54×)**, verify 218 ms | 17.1 (2.66×), verify 218 ms |

All greedy speculative outputs were identical to plain greedy. Verify-vs-decode
logit gaps (`hybrid_spec_check`, blocks 3/8/16): F16 0.0006/0.011/0.012 (main
0.012 at 16), Q8 0.006/0.036/0.029. Plain decode on the dense model is weight
bound, so attention changes barely show there; the gain is in the 17-row
verify, now on the staged prompt kernel.

## Revert and A/B

`FERRUM_HYBRID_LEGACY_ATTENTION=1` restores the Phase 9 routing (simdgroup
kernels for 2–15 rows, unstaged prompt attention) at run time; it reads only an
F16 cache.
