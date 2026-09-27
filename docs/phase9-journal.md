# Phase 9 journal — speculative decoding

Hardware: Apple M5 (10-core GPU), 32 GB unified memory, 24 GiB GPU working set,
macOS 27. Targets and design are in the [Phase 9 plan](phase9-plan.md).

## Draft checkpoints

| Drafter | Revision | Config highlights |
|---|---|---|
| `z-lab/Qwen3.8-27B-DFlash2` | HF main (mirror of `incoai/…`) | 5 layers, 32 heads × 128, block 8, conv kernel 2 / group 16, selector rank 256 top-16, aux 5/19/33/47/61, window 2048, mask 248070 |
| `RadixArk/Qwen3.8-27B-DSpark` | 85ef153b | 5 layers, 40 heads × 128, block 7, aux 4/16/28/40/52, YaRN ×32 from 8192, no window, Markov rank 256, confidence head, mask 248077 |
| `RedHatAI/Qwen3.8-27B-speculator.dspark` | e463040a | 5 layers, 20 heads × 256, 8 drafts, aux (input of) 4/12/…/60, window 2048, Markov rank 256, confidence head, mask 248077 |
| `jzinno/Ornith-1.5-35B-A3B-DFlash2` | HF main | 6 layers, 32 heads × 128, hidden 2048, block 16, aux 1/6/11/16/22/27/32/37, window 4096, mask 248077 |

Swift's MTP block is `blk.64` of the target GGUF (Q4_0 projections, F32 norms).
Reference implementations: the HF `dflash.py`/`dspark.py` shipped with the
RadixArk checkpoint, and llama.cpp 9710a32 (`src/models/dflash.cpp`,
`src/models/qwen35.cpp` `graph_mtp`, `common/speculative.cpp`,
`conversion/qwen.py`).

## Experiment 1 — verify cost versus block width

A speculative step verifies `k` drafts in one target forward of `k + 1` rows,
so its value depends on how that forward scales with rows.
`examples/hybrid_verify_cost.rs` times one forward of `m` rows after 1,024
cached positions (recurrent state restored between runs; median of 3–5).

**Before** (Phase 8 kernels: per-row GEMV up to 4 rows, 128-row TensorOps GEMM
above):

| rows | Swift ms | × 1 row | Tiel ms | × 1 row |
|---|---|---|---|---|
| 1 | 140.6 | 1.00 | 24.4 | 1.00 |
| 2 | 147.4 | 1.05 | 29.9 | 1.23 |
| 4 | 273.9 | 1.95 | 42.0 | 1.72 |
| 5 | 498.7 | 3.55 | 75.2 | 3.08 |
| 8 | 499.1 | 3.55 | 85.8 | 3.52 |
| 16 | 499.4 | 3.55 | 97.3 | 3.99 |

The GEMV kernels stream the weights once per activation row, and the GEMM
costs the same ~500 ms from 5 to 128 rows on Swift: its time is the
dequantization of weight tiles into threadgroup memory (two barriers per
32-wide K step), not the multiply. A block-8 verify at 3.5× would erase most of
the gain from speculation.

Candidates, measured on Swift (ms at 4 / 8 / 16 rows):

| Kernel | 4 | 8 | 16 |
|---|---|---|---|
| Batched GEMV, FP32 dots, 4/8/16 activation rows per threadgroup | 196 | 360 | 1,856 |
| TensorOps, 16-row × 64-weight-row tile, K step 64 | 207 | 208 | 216 |
| same, K step 128 | 231 | 240 | 245 |
| same, K step 32 | 224 | 224 | 234 |
| 8-row tile, K step 64 | 201 | 211 | 337 |
| 32-row tile, K step 128 | 266 | 268 | 268 |
| 16×64 with double-buffered tiles and vector stores | 220 | 226 | 230 |

The batched GEMV is compute-bound past 4 rows (FP32 dot products plus
re-reading activation fragments per output row); the matrix units have to do
the multiply. Kept: the batched GEMV for 2–3 rows and the 16×64 TensorOps tile
for 4–32 rows; above 32 rows the 128-row GEMM is faster again (64 rows: 682 ms
narrow tiles vs 557 ms).

**After:**

| rows | Swift ms | × 1 row | Tiel ms | × 1 row |
|---|---|---|---|---|
| 1 | 151.5 | 1.00 | 24.3 | 1.00 |
| 2 | 162.0 | 1.07 | 29.2 | 1.20 |
| 3 | 183.3 | 1.21 | 33.2 | 1.37 |
| 4 | 216.8 | 1.43 | 42.3 | 1.74 |
| 8 | 221.7 | 1.46 | 71.5 | 2.94 |
| 16 | 216.1 | 1.43 | 81.8 | 3.37 |
| 17 | 334.8 | 2.21 | 88.7 | 3.65 |

(Swift's single-row time drifted 136–151 ms across runs from thermal state.)
Tiel's wide verify stays expensive for a structural reason: 16 rows route to
many distinct experts, each of whose weights must be read. MoE targets will
prefer shorter blocks.

**Accuracy.** `hybrid_kernels MODEL m 1 --check` compares every distinct
projection at `m` rows against `m` single-row GEMVs. All formats agree to
3–6 × 10⁻⁴ of the output scale, except Q4_K through the TensorOps tiles at
5–7 × 10⁻³. The Q4_K (and Q4_0) dequantizer computed `d / 16` in half
precision; for typical super-block scales that is subnormal and loses
mantissa bits. Dividing in float brings Q4_K to 4–6 × 10⁻⁴ at every width,
including the 64-row prompt GEMM that Phase 8 already used.
