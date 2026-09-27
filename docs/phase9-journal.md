# Phase 9 journal — speculative decoding

Hardware: Apple M5 (10-core GPU), 32 GB unified memory, 24 GiB GPU working set,
macOS 27. Targets and design are in the [Phase 9 plan](phase9-plan.md).

## Draft checkpoints

| Drafter | Revision | Config highlights |
|---|---|---|
| `z-lab/Qwen3.8-27B-DFlash2` | 50307d4c (mirror of `incoai/…`) | 5 layers, 32 heads × 128, block 8, conv kernel 2 / group 16, selector rank 256 top-16, aux 5/19/33/47/61, window 2048, mask 248070 |
| `RadixArk/Qwen3.8-27B-DSpark` | 85ef153b | 5 layers, 40 heads × 128, block 7, aux 4/16/28/40/52, YaRN ×32 from 8192, no window, Markov rank 256, confidence head, mask 248077 |
| `RedHatAI/Qwen3.8-27B-speculator.dspark` | e463040a | 5 layers, 20 heads × 256, 8 drafts, aux (input of) 4/12/…/60, window 2048, Markov rank 256, confidence head, mask 248077 |
| `jzinno/Ornith-1.5-35B-A3B-DFlash2` | 9b4852c0 | 6 layers, 32 heads × 128, hidden 2048, block 16, aux 1/6/11/16/22/27/32/37, window 4096, mask 248077 |

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

## Experiment 2 — verify, commit and replay

`HybridModel::verify` runs `[anchor, d1..dk]` as one forward and returns every
row's argmax (or per-row sampling candidates, each row with its own penalty
window) without advancing the state:

- Attention layers write K/V rows past the cached length; rejected rows are
  overwritten by the next forward.
- Gated DeltaNet layers read their conv and delta-rule state but do not write
  it (a `write_state` flag on both kernels), and record the pre-conv qkv, g
  and β rows of every verify row on a tape (~34 MB on Swift for 17 rows).
- `commit(n)` replays the first `n` tape rows through the conv and
  delta-rule kernels into the live state and advances the length.

A per-position state copy would cost ~150 MB per row on Swift (48 layers of
48×128×128 F32); the replay costs one extra read and write of the state per
step. Because verify never writes recurrent state, a failed verify leaves the
session valid.

`examples/hybrid_spec_check.rs` drives verify/commit with oracle drafts (the
greedy reference, every third block corrupted at a moving position, so full,
partial and zero acceptance all occur):

| | Swift (block 7) | Tiel (block 8) |
|---|---|---|
| Speculative output vs greedy decode | identical over 64 tokens | identical until a near-tie (top-2 margin 0.0012) at token 59 |
| Logits after `commit(n)` vs plain forwards of the same n tokens | identical | identical |
| Verify rows vs single-row decode, max \|Δlogit\| / max KL | 0.0095 / 2.9e-7 (8 rows) | 0.36 / 4.8e-4 (4–16 rows) |

Tiel's larger difference is its routing sensitivity (Phase 8: a near-tied
expert flips under F16 rounding; llama.cpp's CPU and Metal backends differ by
KLD 0.02 on it). Multi-row verify versus single-row decode on Tiel is 20×
closer than Ferrum versus llama.cpp. So on Tiel, greedy speculative output can
leave plain greedy output at a near-tied token, with both being faithful
evaluations of the model.

## Experiment 3 — MTP (Swift)

The NextN block (`blk.64`: Q4_0 projections) runs on the trunk's kernels with
its own single-layer K/V cache. MTP position p reads the embedding of token p
and the target's output-normed hidden state at p−1 (llama.cpp's `h_nextn`).
Committed rows are staged and run together with the anchor row; the remaining
draft steps chain on the block's own hidden state and argmax on the GPU, so a
whole draft is one submission. The block's weights are 313 MiB.

Swift, 3 chat prompts × 128 tokens, greedy, thinking on (plain decode
7.15 tok/s; every speculative output identical to plain decode):

| drafts | tok/s | acceptance length | draft ms/step | verify ms/step | accepted at 1/2/3 |
|---|---|---|---|---|---|
| 2 | 11.70 | 2.30 | 22.8 | 170.5 | 74% 56% |
| 3 | 11.78 | 2.70 | 32.7 | 193.7 | 75% 54% 42% |
| 5 | 11.92 | 3.06 | 53.4 | 200.9 | 73% 52% 37% 24% 21% |
| 7 | 11.68 | 3.23 | 73.0 | 200.0 | 71% 48% 33% … 13% |

Each draft step costs ~10 ms, most of it the full-vocabulary LM head (1 GB of
Q6_K), which cancels the extra acceptance beyond 3 drafts.

## Experiment 4 — DFlash, DFlash2 and DSpark

`src/hybrid/draft.rs` loads SpecForge and speculators checkpoints (BF16
safetensors quantized to Q8_0 at load, rows in parallel). The target captures
its residual stream at the drafter's aux layers during every prefill chunk,
plain step and verify. Each committed position is fused (`fc`,
`hidden_norm`) and projected to K/V for every draft layer, then stored in a
ring of `window` slots tagged with its position. A draft step embeds
`[anchor, mask, …]` with the target's embedding and runs the draft layers
with non-causal attention over the valid context slots (tag below the block,
inside the window) plus the block, then reads drafts through the target's LM
head:

- **DFlash**: mask slots predict their own position (`block − 1` drafts).
- **DFlash2**: each layer wraps attention and the MLP in two-tap grouped
  causal convolutions whose coefficients are a base kernel plus a per-token
  projection. The draft keeps the top 16 candidates per position; a selector
  walks one path through them, scoring successor codebook ⋅ (predecessor
  codebook ⊙ gate(hidden)) + logit (codebooks stay in host memory, 2 × 127
  MB BF16). The base-kernel layout `[side][tap][channel]` was confirmed from
  jzinno's checkpoint, whose convolutions were initialized as exact identity
  taps.
- **DSpark**: slot i predicts position anchor + i + 1; a rank-256 Markov bias
  conditioned on the previous (drafted) token is added row by row on the GPU,
  chained through argmax; the confidence head can truncate the block.

Layer indexing: SpecForge/DFlash ids are layer outputs; speculators'
`aux_hidden_state_layer_ids` are layer inputs (id − 1), as in vLLM's
aux-hidden-state collection and llama.cpp's converter. Swapping the RedHat
convention for the other one moved acceptance only from 3.66 to 3.72 (noise):
adjacent residual streams are nearly interchangeable to the drafter.

Two ablations on RadixArk's DSpark confirm the less obvious parts: without
the Markov bias acceptance falls from 3.02 to 2.16; with plain RoPE instead of
its YaRN configuration, to 2.92.

Swift, 3 chat prompts × 128 tokens, greedy (plain decode 7.0 tok/s):

| Drafter | drafts | tok/s | acceptance length | draft ms | verify ms | model card |
|---|---|---|---|---|---|---|
| MTP | 3 | 11.78 | 2.70 | 32.7 | 193.7 | — |
| z-lab DFlash2 | 7 | 16.07 | 3.91 | 36.2 | 201.9 | 3.7–5.5 |
| z-lab DFlash2, Q4_0 weights | 7 | 16.87 | 3.88 | 27.8 | 197.3 | |
| RedHat DSpark | 8 | 14.84 | 3.66 | 37.7 | 202.3 | 3.8–5.7 |
| RadixArk DSpark | 7 | 12.44 | 3.02 | 37.6 | 200.1 | 2.7–4.6 |

Every greedy output with a baseline run was identical to plain decode. The
model cards measure the FP8/BF16 base model; Swift 1.5 is a fine-tune served
at Q4_K_M, and all three drafters land at 80–90% of their cards.

Tiel with jzinno's DFlash2 (plain decode 40.7 tok/s, 4 prompts × 128):

| drafts | tok/s | acceptance length | draft ms | verify ms |
|---|---|---|---|---|
| 2 | 53.90 | 2.39 | 9.9 | 32.2 |
| 3 | 50.09 | 2.86 | 12.2 | 42.5 |
| 4 | 41.92 | 3.07 | 14.4 | 56.7 |
| 8 | 39.82 | 3.57 | 18.1 | 69.3 |
| 15 | 35.89 | 3.95 | 24.9 | 82.8 |

Acceptance at 15 drafts (3.95) matches the card's 4.0, but every verify row
routes to its own experts, so the verify cost climbs steeply and 2 drafts is
best. MoE targets default to 2 drafts.

**Draft vocabulary.** Byte-level BPE ids follow merge order, so a prefix of the
LM head approximates a frequent-token vocabulary (`--draft-vocab N`).
Verification still uses the full vocabulary. On Swift/DFlash2 (Q4_0) a 32K
prefix cut the draft from 27.8 to 16.3 ms but acceptance from 3.88 to 3.43
(64K: 3.59), a net loss; for MTP (4 drafts) it was a small gain (12.15 tok/s).
Kept as an option, off by default.

## Experiment 5 — sampling, runtime and server

**Sampling.** With temperature > 0 each verify row carries its own penalty
window, and draft i is accepted with the target's probability of it (the
proposal is deterministic); on rejection the replacement is drawn from the
target distribution with the draft removed. A unit test checks the emitted
token distribution against the target's within 0.005 for a likely, an
unlikely and an out-of-top-k draft. Swift + DFlash2 at the model's defaults
(temperature 1.0, top-p 0.95, top-k 20): 12.72 vs 6.87 tok/s, acceptance 3.14.

**Memory.** `SpecOptions::memory` sizes the drafter from metadata (GGUF for
MTP; config and safetensors header otherwise) before loading; the planner
adds it to the fixed budget (MTP's K/V also per token), so the auto-fitted
context shrinks accordingly. Predicted vs measured device memory: MTP 314.3 /
312.9 MiB, z-lab DFlash2 1972.5 / 1971.0, RadixArk DSpark (8,192-slot
context cap) 1646.4 / 1644.9, RedHat DSpark 2199.0 / 2197.5, jzinno DFlash2
(2 drafts) 571.5 / 570.2 MiB.

**Binaries.** `ferrum-cli` and `ferrum-server` take `--draft mtp|DIR`
(`--draft-max`, `--draft-quant`, `--draft-context`, `--draft-p-min`,
`--draft-vocab`). The server reports `draft_n` / `draft_n_accepted` in
`timings` like llama-server, and `/metrics` counts drafted and accepted
tokens. A Tiel server run (DFlash2, 2 drafts) answered a code request at
60.8 tok/s with 68 of 78 drafts accepted.

**Small-batch GEMV at 4 rows.** llama.cpp's forward on the same Swift GGUF
after 1,024 tokens takes 146 / 151 / 196 / 353 / 555 ms at 1 / 2 / 4 / 8 / 16
rows (`llama-bench pp1..pp16 @ d1024`); Ferrum takes 144 / 155 / 203 / 214 /
216 ms after moving 4-row projections from the TensorOps tile to the batched
GEMV (the tile measured 216 ms at 4 rows).

## Experiment 6 — against llama.cpp

llama.cpp 9710a32 (`llama-server` built from source; Homebrew's 687e77892
predates DFlash2 and loads only 58 of the draft's 81 tensors). Drafts were
converted with its `convert_hf_to_gguf.py` to Q8_0, the same precision Ferrum
quantizes them to. Both servers: same GGUF, 8,192-token context, the 8 chat
prompts of `tools/spec_bench.py`, 128 tokens each, greedy, thinking on, one
server at a time. llama-server's default multi-slot setup ran out of GPU memory
with a 2 GB draft next to Swift, so its DFlash2/DSpark runs use `-np 1`.
Raw results: `docs/measurements/phase9/`.

| Swift 27B (tok/s) | drafts | Ferrum | llama.cpp | Ferrum / llama.cpp | draft acceptance Ferrum / llama.cpp |
|---|---|---|---|---|---|
| no speculation | — | 7.11 | 6.74 | 1.05× | — |
| MTP | 3 | 12.60 | 11.17 | 1.13× | 63.0% / 61.7% |
| z-lab DFlash2 | 7 | **17.78** | 9.87 | 1.80× | 47.8% / 47.5% |
| RedHat DSpark | 8 | 16.37 | 6.83 | 2.40× | 39.7% / 39.5% |
| RadixArk DSpark | 7 | 12.53 | 6.93 | 1.81× | 30.0% / 29.2% |

| Tiel 35B-A3B (tok/s) | drafts | Ferrum | llama.cpp | ratio | acceptance |
|---|---|---|---|---|---|
| no speculation | — | 41.29 | 41.14 | 1.00× | — |
| jzinno DFlash2 | 2 | **57.12** | 50.61 | 1.13× | 76.4% / 75.3% |
| jzinno DFlash2 | 8 | 44.80 | 36.59 | 1.22× | 39.0% / 39.9% |

Acceptance agrees within a point for every drafter, which cross-checks the
four drafter implementations against an independent one. The throughput gap
is the verify: llama.cpp's forward of 8–9 rows costs 2.4–3.8× a single-row
decode on Swift (Experiment 5), Ferrum's 1.5×, so llama.cpp gains little or
nothing from the 7–8-draft DFlash2/DSpark blocks. Speedup over plain Ferrum
decode: DFlash2 2.50×, RedHat DSpark 2.30×, MTP 1.77×, RadixArk DSpark 1.76×
on Swift; 1.38× on Tiel.
