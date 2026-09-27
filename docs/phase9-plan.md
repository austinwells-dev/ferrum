# Phase 9 — Speculative decoding

Phase 8 made Swift 27B and Tiel 35B-A3B fast and stable without speculation.
Phase 9 adds three drafters and a lossless verify path, on the same M5 (32 GB,
24 GiB GPU working set).

| Target | Drafter | Source | Shape |
|---|---|---|---|
| Swift 27B | MTP | `blk.64.nextn.*` in the target GGUF (Q4_0) | one gated-attention + dense-FFN block, chained |
| Swift 27B | DFlash2 | `z-lab/Qwen3.8-27B-DFlash2` | 5 layers, block 8, aux 5/19/33/47/61, window 2048 |
| Swift 27B | DSpark | `RadixArk/Qwen3.8-27B-DSpark` | 5 layers, block 7, aux 4/16/28/40/52, YaRN RoPE, no window |
| Swift 27B | DSpark | `RedHatAI/Qwen3.8-27B-speculator.dspark` | 5 layers, 8 drafts, aux 4/12/…/60 (input-of-layer), head dim 256, window 2048 |
| Tiel 35B-A3B | DFlash2 | `jzinno/Ornith-1.5-35B-A3B-DFlash2` | 6 layers, block 16, aux 1/6/11/16/22/27/32/37, window 4096 |

## Objectives

1. **Lossless.** Greedy output equals the target's greedy output; sampled output
   has the target's distribution (rejection sampling against the draft's
   deterministic proposal). Checked by re-scoring every speculative generation
   with a teacher-forced target pass: every emitted token must be the target's
   choice at its position, up to near-ties from batch-size-dependent rounding.
2. **Faster decode.** Report acceptance length and end-to-end tok/s per drafter
   against plain decode and against llama.cpp's `draft-mtp` / `draft-dflash`.
3. **Fixed memory.** Draft weights, draft K/V and verify buffers are predicted
   by the planner and allocated once; the auto-fitted context shrinks to make
   room.
4. **Same runtime features.** Sessions, prefix reuse and snapshots, streaming,
   stop strings, reasoning budgets and sampling work with speculation on.

## Design

- **Verify.** One target forward over `[anchor, d1..dk]` produces logits for
  every row. Attention layers append K/V as usual (rejected rows are
  overwritten later). Gated DeltaNet layers read the recurrent state but do not
  write it; each layer records its pre-conv qkv, g and β rows. After
  acceptance, the accepted rows are replayed through the conv and delta-rule
  kernels into the live state, which costs one extra state read and write
  instead of a per-position state copy (~150 MB per position on Swift).
- **Features.** The verify and prefill passes copy the residual stream at the
  drafter's aux layers (DFlash/DSpark) or the output-normed final hidden state
  (MTP) for the accepted rows.
- **Draft context.** DFlash/DSpark context K/V live in a ring buffer with a
  position tag per slot, so sliding-window drafts stay small and a session
  rewind never reads stale slots. Drafts without a window get a configurable
  cap.
- **Draft weights.** BF16 safetensors quantized to Q8_0 at load (Q4_0
  optional), so the existing GEMV/GEMM kernels serve them; the target's
  embedding and LM head are shared.
- **Layer indexing.** SpecForge/DFlash `target_layer_ids` name the output of
  layer *i*; speculators' `aux_hidden_state_layer_ids` name the input of layer
  *i* (output of *i*−1), as in llama.cpp's converter.

## Work order

1. Verify path: multi-row outputs, recurrent record/replay, feature capture;
   exactness test with oracle drafts; verify cost versus block width.
2. MTP for Swift.
3. Draft loader (safetensors → Q8_0), draft attention, DFlash2 (conv +
   selector) for Swift and Tiel.
4. DSpark (Markov head, confidence-based truncation) for both Swift drafters.
5. Sampling with speculation, planner, CLI/server flags, benchmarks.

Progress, measurements and decisions go in the [Phase 9 journal](phase9-journal.md).
