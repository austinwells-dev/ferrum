# Phase 11 journal — tree speculative decoding

Plan: [phase11-plan.md](phase11-plan.md). Measured on the same M5 (32 GB) as
Phases 9 and 10, with `caffeinate` and the screen unlocked; Swift 27B is
`Swift-1.5-Qwen3.8-27B-Q4_K_M.gguf` with its own MTP head, Spark 4B is
`Spark-X2.6-4B-Q4_K_M.gguf`. Tok/s are best of two interleaved rounds (the
other in parentheses); acceptance numbers are deterministic.

## Experiment 1 — tree mask in the attention kernels

Each verify row carries an ancestor bitmask and a depth. Visibility of a key
at or past the cached length is a bit test, its window is resolved once per
thread from the ancestors' depths, and a tile's window start comes from its
lowest-positioned row (rows are no longer in position order; the last row
still bounds the keys, since ancestors come first). RoPE positions are
`len + depth` through the existing position table with all three axes equal.

A runtime tree flag in the shared kernels cost the staged prompt kernel
6–13% with no tree, and 2–5% after the window precomputation. Since that
kernel also runs every prompt chunk, the tree versions became `_tree`
entries specialized at compile time; the existing kernels compile to their
previous code, and their timing returned to within run-to-run noise.

| Check | Result |
|---|---|
| Random trees vs F64 CPU reference (5,418 runs: F16/Q8, d256/d128, sinks, windows 513/40/6, rings) | worst 2.1e-4 (causal rows 1.9e-4); bound 2e-3 |
| Chain-shaped tree vs causal kernels | bit-identical, every kernel |
| Tree RoPE vs plain RoPE at each node's depth | bit-identical |
| Tree off vs the previous kernels | 480 hashed outputs bit-identical; `hybrid_spec_check` identical on Spark and Swift |

Three deliberate kernel bugs (ignoring ancestor bits, the old window start,
windowing tree keys by row) each fail the test. The third needed a tree
shape whose ancestors sit far later in the rows than their depth (a star
whose last leaf grows a chain); the random shapes never produced one.

## Experiment 2 — verify_tree and commit_path

Gated DeltaNet nodes start from their parent's state: the conv tree kernel
reads the latest three ancestors' rows (then the state above the root), and
the delta-rule tree kernel keeps each node's state slice per thread. A
node's parent is its latest ancestor in the mask, so any parents-first order
works and no reordering is needed. `commit_path` gathers the path's tape rows
for the replay, moves K/V rows (gather, then scatter, so overlapping moves
cannot race) in F16 or Q8 with scales, and moves captured rows for the
drafter. `examples/hybrid_tree_check.rs` drives oracle trees (the reference
on one random path, decoys elsewhere, every third path corrupted) beside a
chain over the same accepted tokens, 96 tokens:

| Model, cache, budget | Output vs greedy | Path rows vs chain, default kernels | Exact kernels |
|---|---|---|---|
| Swift F16, 15 | identical | max \|Δlogit\| 0.012 | 100/100 rows bit-identical |
| Swift Q8, 15 | identical | 0.046 | — |
| Swift F16, 4 | identical | 0.026 | — |
| Spark F16, 15 | identical | 0.024 | 100/100 rows bit-identical |
| Spark Q8, 15 | identical | 0.235 | — |
| Spark F16, 24 | identical | 0.034 | — |

"Exact kernels" is `FERRUM_HYBRID_EXACT=1 FERRUM_HYBRID_REFERENCE_ATTENTION=1`.
`FERRUM_HYBRID_EXACT=1` alone is not enough for bit identity: it makes the
projections row-independent, but attention still picks its kernel by row
count, so a 16-row tree and a 5-row chain round differently (Spark: 0/100
rows identical, max 0.012). Spark's Q8 numbers match its chain's own (verify
rows vs decode 0.056, replay 0.036): 8-bit K/V amplifies the last-bit
differences. A unit test checks the K/V move byte for byte (F16 and Q8,
linear and wrapping ring), and fails when the Q8 scale offset is broken.

## Experiment 3 — accept walk and prompt-lookup trees

Prompt lookup merges every earlier continuation of the matched suffix into a
trie (its first branch is the chain's proposal). Spark, 8 chat prompts × 128
tokens, greedy:

| | tok/s | accepted per pass | outputs vs plain / vs chain |
|---|---|---|---|
| plain | 33.72 | | |
| chain (3 drafts) | 37.93 | 2.03 | 8/8 identical |
| tree 4 | 37.18 | 2.07 | 8/8 / 8/8 |
| tree 8 | 37.62 | 2.08 | 8/8 / 8/8 |
| tree 15 | 37.46 | 2.09 | 8/8 / 8/8 |

The same with exact kernels: every output identical. Lookup rarely has two
different earlier continuations, so trees add little; it serves as a cheap
end-to-end test of the loop on the attention-only path.

## Experiment 4 — MTP chain plus leaf siblings

MTP runs its chain as before and also returns each step's top logits per
1,024-token block (`h_topk_blocks` on the draft head, in the same
submission). `chain_with_siblings` adds up to three alternatives per depth as
leaves, best first by the drafter's path probability (normalized over the
returned candidates), under the node budget.

**Gate**, Swift, 8 chat prompts × 128 tokens, 3 drafts: chain 11.73 tok/s at
2.87 tokens per pass, tree 15 at 12.01 and 3.27; all 16 outputs identical to
plain decoding and the tree's to the chain's. With exact kernels (3 prompts ×
64 tokens) likewise identical. Across every benchmark run below (88 runs) no
greedy output differed from plain decoding.

**Budget sweep**, Swift, one coding prompt (binary search in Rust with tests)
and one prose prompt (causes of the French Revolution), 192 tokens each.
"Depth" is MTP steps per draft; a depth-3 tree holds at most 12 nodes.

| Coding | depth 3 tok/s | per pass | depth 5 tok/s | per pass | depth 7 tok/s | per pass |
|---|---|---|---|---|---|---|
| plain | 6.21 | | 6.49 | | 6.22 | |
| chain | 10.98 (10.63) | 2.70 | 12.21 (11.50) | 3.25 | 11.11 (10.95) | 3.43 |
| tree 4 | 11.26 (10.54) | 3.00 | 11.76 (11.00) | 3.15 | 10.18 (10.17) | 3.15 |
| tree 8 | 11.24 (11.03) | 3.15 | 12.99 (12.57) | 3.77 | 11.96 (11.94) | 3.84 |
| tree 12 | 11.86 (11.84) | 3.31 | **13.10** (12.29) | 3.77 | 11.95 (11.92) | 3.92 |
| tree 15 | 11.93 (11.29) | 3.31 | 12.97 (12.22) | 3.77 | 12.07 (12.06) | 4.00 |
| tree 16 | 12.30 (11.95) | 3.31 | 8.66 (8.56) | 3.77 | 8.47 (8.47) | 3.92 |
| tree 24 | 11.96 (11.92) | 3.31 | 8.72 (8.56) | 3.84 | 8.32 (8.31) | 4.00 |

| Prose | depth 3 tok/s | per pass | depth 5 tok/s | per pass | depth 7 tok/s | per pass |
|---|---|---|---|---|---|---|
| plain | 6.60 | | 6.38 | | 6.22 | |
| chain | 12.28 (12.07) | 2.95 | 12.80 (12.28) | 3.43 | 11.71 (11.65) | 3.56 |
| tree 4 | 12.02 (11.40) | 3.00 | 11.09 (11.07) | 3.15 | 10.24 (10.21) | 3.15 |
| tree 8 | 11.90 (11.52) | 3.13 | 12.94 (12.84) | 3.82 | 12.26 (12.22) | 3.90 |
| tree 12 | 11.25 (11.16) | 3.13 | **13.30** (12.88) | 3.90 | 12.21 (12.20) | 3.98 |
| tree 15 | 11.89 (11.32) | 3.13 | 13.03 (12.76) | 3.90 | 12.13 (12.09) | 3.98 |
| tree 16 | 11.32 (11.32) | 3.13 | 9.14 (8.98) | 3.90 | 8.73 (8.73) | 3.98 |
| tree 24 | 11.57 (11.35) | 3.13 | 8.83 (8.77) | 3.90 | 8.40 (8.40) | 3.98 |

- **Tokens per pass** rise over the chain of the same depth once the budget
  reaches about 8: 6–17% at depth 3, 10–16% at depths 5 and 7 (depth 5:
  3.25 → 3.77 coding, 3.43 → 3.90 prose). The first draft is accepted 92–97%
  of the time with siblings, 75–89% without.
- **Speed.** The best tree (depth 5, 12–15 nodes) is 13.0–13.3 tok/s against
  the best chain's 12.2–12.8 (+4–7%) and the default 3-draft chain's
  11.0–12.3 (+8–19%). Depth-3 trees are within noise of the chain on prose.
- **The 17-row cliff.** Budget 16 or 24 (17 or more verify rows) at depth 5
  or 7 costs 370–390 ms per verify against 230–240 ms, and speed falls by a
  third. At depth 3 the tree never exceeds 13 rows, so budgets 16 and 24
  change nothing there.
- **Where verify time goes** (`examples/hybrid_tree_cost.rs`, 1,024 cached
  positions, median of 7): the tree kernels cost the same as the chain
  kernels at equal width, and `commit_path` the same as `commit`. Width
  itself is the cost on this build: 204 ms at 4 rows, 219 at 8, 228 at 12,
  242 at 16, rather than Phase 9's flat 1.43×.

| rows | chain verify ms | tree kernels, chain shape | MTP-shaped tree | commit(3) | commit_path via a sibling |
|---|---|---|---|---|---|
| 4 | 204.4 | 204.4 | 205.1 | 3.18 | 2.98 |
| 8 | 219.1 | 220.0 | 221.0 | 3.85 | 4.01 |
| 12 | 227.5 | 231.0 | 230.1 | 4.04 | 3.91 |
| 16 | 242.2 | 244.9 | 248.4 | 3.90 | 3.84 |

**Decision taken:** with `--draft-tree`, MTP drafts 5 deep unless
`--draft-max` says otherwise. **Open:** whether full best-first trees
(expanding siblings, which needs tree-masked MTP drafting) are worth it.
Siblings already lift first-draft acceptance to 92–97%; the remaining
misses are deeper in the chain, where an expanded sibling would need its own
continuation.
