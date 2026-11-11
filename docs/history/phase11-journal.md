# Phase 11 journal — tree speculative decoding

> **History.** Phase 11 is still in progress, so this file is still being written; earlier entries are not revised. The current docs start at [docs/README.md](../README.md).

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
`--draft-max` says otherwise (superseded by Experiment 5 for best-first
trees).

## Experiment 5 — best-first MTP trees

Siblings fix most first-guess misses (first draft accepted 92–97%), but a
sibling's own continuation is never drafted. Best-first trees expand any
node: each MTP pass runs the likeliest frontier tokens together, each
reading its parent's MTP hidden state and attending, through the tree
kernels, to the anchor's prefix and its own ancestors only, at position
anchor + depth; their top 4 tokens join the frontier. This needed the tree
mask to cover rows partway into a tree (nodes stored by earlier passes):
the tree buffer now starts with the first row's node index, a one-row pass
with a tree runs on the verify kernels rather than decode, and the key
range starts from the tree root's window. The kernel test also runs the
later half and the last node of every tree on their own (14,490 tree runs,
same worst error).

**Drafting check** (`examples/hybrid_mtp_tree_check.rs`): along 48 greedy
steps on Swift, MTP's chain and a best-first tree from the same state;
wherever the tree expanded the chain's path, its nodes' top-16 logits must
equal the chain's steps. With exact kernels 192/192 are bit-identical; with
default kernels max |Δlogit| 0.006. Drafting without the tree mask (nodes
seeing their siblings, at row positions) gives 0.325, while an argmax-only
check missed it: a few extra keys rarely move MTP's top token.

**Wave size.** Budget 15, depth 7, one coding and one prose prompt (192
tokens), interleaved:

| Nodes per pass | Coding tok/s | per pass | Prose tok/s | per pass | draft ms |
|---|---|---|---|---|---|
| 1 | 12.09 | 4.57 | 10.82 | 4.17 | 162 |
| 2 | 15.23 | 4.57 | 14.05 | 4.17 | 87 |
| **3** | 15.35 / 14.36 | 4.24 | 13.70 / 13.72 | 4.00 | 60 |
| 4 | 14.04 / 13.75 | 3.92 | 12.05 / 12.26 | 3.49 | 55 |

(Rows 1 and 2 come from one run, rows 3 and 4 from two interleaved rounds of
their own.) Waves of 4 spend the first pass on four guesses for the first
token, so trees grow wide and shallow; waves of 1 search best first exactly
but pay a pass per node. Waves of 3 are the default.

**Against chains and siblings**, same prompts, two interleaved rounds, best
(other):

| Coding | tok/s | per pass | draft ms | verify ms |
|---|---|---|---|---|
| plain | 7.06 (6.34) | | | |
| chain, 3 drafts | 12.01 (10.68) | 2.70 | 32 | 190 |
| chain, 5 drafts | 12.49 (11.42) | 3.25 | 54 | 203 |
| siblings, depth 5, 12 nodes | 13.45 (12.45) | 3.77 | 56 | 220 |
| best first, depth 5, 15 nodes | 14.81 (14.79) | 4.24 | 60 | 224 |
| **best first, depth 7, 15 nodes** | **15.12** (14.96) | 4.24 | 60 | 219 |

| Prose | tok/s | per pass | draft ms | verify ms |
|---|---|---|---|---|
| plain | 7.10 (7.02) | | | |
| chain, 3 drafts | 13.21 (13.05) | 2.95 | 32 | 188 |
| chain, 5 drafts | 13.35 (13.10) | 3.43 | 53 | 201 |
| siblings, depth 5, 12 nodes | 14.03 (13.91) | 3.90 | 56 | 220 |
| best first, depth 5, 15 nodes | 13.89 (13.87) | 4.00 | 60 | 224 |
| **best first, depth 7, 15 nodes** | **14.23** (14.09) | 4.00 | 60 | 221 |

Best-first trees beat sibling trees by 12% on code and 1% on prose, and the
default 3-draft chain by 26% and 8% (2.1× and 2.0× plain decoding). At 8 or
12 nodes best-first trees are shallower than sibling trees and lose on prose
(12.95 at 12 nodes, against siblings' 14.03), so 15 is the budget to use;
16 or more crosses the 17-row verify cliff as before. Every greedy output
checked against plain decoding was identical: the 60 runs of a first sweep
(waves of 4, depths 5 and 7, every budget, with the 8-prompt and exact-kernel
gates) and the final gate below.

**Final gate**, with the defaults below (`--draft mtp --draft-tree 15`):
8 chat prompts × 128 tokens on Swift, all identical to plain decoding, at
14.63 tok/s against plain 6.89 (2.12×) and 4.16 tokens per pass; 3 prompts
× 64 tokens with exact kernels likewise identical; the drafting check at
depth 7 bit-identical on 240/240 chain-path nodes.

**Decisions:** best-first is the default tree search (`--draft-tree-search
siblings` keeps Experiment 4's trees), with waves of 3 and, unless
`--draft-max` says otherwise, 7 MTP steps deep (5 for sibling trees).

## Experiment 6 — block-drafter trees from the candidate lattice

A block drafter already scores every depth of its block in one pass, so a
tree costs no extra drafter passes (the idea behind Kapitsa's KFlash). Each
of DFlash, DFlash2 and DSpark now implements `Drafter::draft_tree`: the
block pass takes each depth's top candidates (`h_topk_blocks`, as MTP does),
and `lattice_best_first` (`speculative.rs`) grows the tree best first by the
probability of the whole path, the product of per-depth conditionals
normalized over each depth's candidates:

- **DFlash2**: its selector's top-16 per depth, with the path selector's
  score (logit plus the bilinear predecessor/successor term the chain's
  greedy walk already uses) as the conditional on the parent's token. The
  chain is the tree's greedy path.
- **DSpark**: the top 16 of each slot's logits before the Markov bias, plus
  the chain's own pick; the conditional adds `w2[token] · w1[parent]`. The
  candidates' Markov rows are gathered on the GPU once per step (one small
  extra command buffer, +3 ms); the confidence cut-off still bounds the
  depth.
- **DFlash** (z-lab): no selector, so depths are independent and each
  node's children are the next depth's marginals.

**Gate.** `hybrid_spec --baseline --sweep 0,8,15`, 8 chat prompts × 128
tokens on Swift 27B: DFlash2 (`z-lab/Qwen3.8-27B-DFlash2`) and DSpark
(`RedHatAI/Qwen3.8-27B-speculator.dspark`) 0 greedy mismatches in all 48
speculative runs, every tree output equal to its chain's; with
`FERRUM_HYBRID_EXACT=1 FERRUM_HYBRID_REFERENCE_ATTENTION=1`, 3 prompts × 64
tokens, likewise 0 for both. On the 8-prompt gate DFlash2 runs 16.55 tok/s
as a chain (4.29 per pass), 17.52 at 8 nodes (4.65) and 18.29 at 15 (5.14,
3.00× plain); DSpark 16.64, 17.08 and 17.34 (4.10, 4.33, 4.68).

**Against chains and MTP**, Swift 27B, 192 tokens, two interleaved rounds,
best (other). Block drafters draft their trained block (7 for DFlash2, 8 for
DSpark); MTP runs 7 deep, as with `--draft-tree`.

| Coding | tok/s | per pass | draft ms | verify ms |
|---|---|---|---|---|
| MTP chain, 7 drafts | 11.96 (11.92) | 3.43 | 78 | 206 |
| MTP best first, 15 nodes | 14.61 (14.57) | 4.24 | 62 | 227 |
| DFlash2 chain | 18.69 (18.64) | 4.47 | 29 | 204 |
| DFlash2 tree, 8 nodes | **20.08** (19.84) | 4.92 | 30 | 211 |
| DFlash2 tree, 12 nodes | 18.26 (18.22) | 4.68 | 30 | 222 |
| DFlash2 tree, 15 nodes | 19.76 (19.48) | 5.19 | 30 | 230 |
| DSpark chain | **17.24** (17.12) | 4.27 | 32 | 211 |
| DSpark tree, 8 nodes | 16.86 (16.58) | 4.27 | 34 | 215 |
| DSpark tree, 12 nodes | 16.98 (16.82) | 4.47 | 34 | 224 |
| DSpark tree, 15 nodes | 16.92 (16.74) | 4.57 | 35 | 231 |

| Prose | tok/s | per pass | draft ms | verify ms |
|---|---|---|---|---|
| MTP chain, 7 drafts | 12.55 (12.53) | 3.56 | 75 | 205 |
| MTP best first, 15 nodes | 13.69 (13.67) | 4.00 | 62 | 227 |
| DFlash2 chain | 16.43 (16.37) | 3.92 | 29 | 205 |
| DFlash2 tree, 8 nodes | **18.05** (18.01) | 4.47 | 29 | 213 |
| DFlash2 tree, 12 nodes | 17.80 (17.79) | 4.57 | 29 | 222 |
| DFlash2 tree, 15 nodes | 17.88 (17.87) | 4.68 | 29 | 228 |
| DSpark chain | 14.78 (14.76) | 3.62 | 31 | 209 |
| DSpark tree, 8 nodes | 14.88 (14.82) | 3.77 | 34 | 214 |
| DSpark tree, 12 nodes | 15.54 (15.59) | 4.09 | 34 | 222 |
| DSpark tree, 15 nodes | **15.65** (15.59) | 4.17 | 34 | 227 |

DFlash2 trees beat its chain by 6% (code) and 9% (prose) at 15 nodes and by
7% and 10% at 8; KFlash reports +6% and +16% on the same 27B. Drafting
costs nothing extra, so the gain is acceptance (+16–19% per pass at 15
nodes) less the wider verify (+12%). The trees hit their limit at the
block's depth: at 15 nodes the first two depths are accepted 90–94% and
82–84% of the time, and further nodes mostly add third and fourth guesses.
Between 8 and 15 nodes the ranking flips with the prompt set (8 wins these
two by 1–2%, 15 wins the 8-prompt gate by 4%), and 12 dips on code (a
different trajectory, 4.68 per pass against 8 nodes' 4.92). DFlash2 with a
tree of 15 is the fastest configuration measured on this Mac, 1.35× MTP's
best-first trees on code and 1.31× on prose.

DSpark gains less: +6% on prose at 15 nodes, and −2% on code, where trees
add only 0.3 tokens per pass. Its candidates are taken before the Markov
bias, so the bias reorders but cannot bring in tokens outside the unbiased
top 16 (only the chain's own pick is added); building candidates from the
biased rows of every candidate parent would need a projection per parent.

**MoE targets.** On Tiel-Coder-35B-A3B (IQ4_XS) with
`z-lab/Qwen3.6-35B-A3B-DFlash` (the independent-depth case; MoE targets draft
2 deep), trees of 1–3 nodes match plain decoding, and trees of 8 and 15
diverge on 4–6 of 8 prompts, at top-2 margins 0.04–0.43. An 8-draft chain
(also 9 verify rows, no tree) diverges at the same tokens with the same
margins, and every tree output equals that chain's: this is Phase 10's
single- versus multi-row rounding on MoE, not the trees. Trees are also slow
there (verify 32 → 66 ms at 8 nodes, 0.91× plain), since every verify row
routes to its own experts: MoE targets stay on 2-draft chains.

**Decisions:** `--draft-tree N` now gives trees for DFlash, DFlash2 and
DSpark as well as MTP; defaults are unchanged (chains unless
`--draft-tree`). For a dense target, `--draft DFlash2 --draft-tree 15` is
the recommended configuration; DSpark trees help prose only.

## Experiment 7 — KFlash from Kapitsa: wide blocks, calibrated trees, a learned budget

Ported from Kapitsa (Nace AI, Apache-2.0; `src/hybrid/kflash.rs`, see
THIRD_PARTY_NOTICES): per-depth logistic calibration of the drafter's node
probabilities, fitted online from every verified node whose parent was
accepted; a cost table per verify row count; and each cycle verifying the
allowed n (2, 4, 8, 12, 16, 24, 32 rows) that maximizes E(n) − rate·cost(n).
Also from Kapitsa's DFlash2 defaults: a 16-row drafting block for a drafter
trained on 8 (mask slots carry no per-position weights), and a draft head
over the first 98,304 ids. Ferrum builds the tree on the host and measures
cycles by wall time, where Kapitsa builds on the GPU and uses GPU time.

DFlash2 on the MLX 4-bit 27B, prompts 4 and 3, 512 tokens:

| Setup | tok/s | Per pass | Draft ms | Verify ms |
|---|---|---|---|---|
| block 8, tree 15 | 23.2 | 4.52 | 24.5 | 164.8 |
| + 98,304-id head | 24.1 | 4.54 | 19.0 | 164.3 |
| block 16, head, tree 15 | 24.3 | 4.75 | 23.7 | 166.2 |
| block 16, head, auto (warm cache) | 24.5 | 4.87 | 23.7 | 169.5 |

Over all 8 prompts (256 tokens), auto reaches 25.38 tok/s, 5.02 tokens per
pass (the budget expected 5.03), against the 16-row chain's 20.97 and plain
decoding's 7.74. Every output is identical to plain decoding and to the chain,
and so is the strict-math run (`FERRUM_HYBRID_EXACT=1`, 2 prompts). The rule
settles on 16 rows on 97% of cycles: 17 or more rows leave the pair kernel
(275 ms against 189 ms). The fitted calibration shrinks the drafter's
confidence past depth 1 (beta 0.6–0.9, alpha −0.3 to −1.1).

On the Q4_K_XL GGUF (8 prompts, 256 tokens), the 16-row block with the
reduced head lifts tree 15 from 18.18 to 19.31–19.77 tok/s (5.44 per pass
against 5.08), and auto ties it (18.94–19.71). There, a 16-row check costs
1.45× a 2-row one (270 ms against 187 ms), so 20% of cycles fall back to a
single draft, which costs acceptance (5.12 per pass) about as much as it
saves in verify time.

**Decisions:** `--draft-tree auto` (KFlash, with the DFlash2 defaults above)
is the recommended setup on MLX checkpoints; `--draft-block` and
`--draft-vocab` are exposed for GGUF targets, where a fixed tree of 15 with
both is as fast.

## Experiment 8 — exact sampled trees

Kapitsa's sampled acceptance, ported as an idea (`session.rs`): every token
is drawn with a uniform keyed by the request's seed and its sequence
position, in plain decoding too, and a chain or tree row is accepted when its
token equals the row's own draw (the uniform of the position it would fill).
Rows at one depth share a position and so a uniform, and the walk emits
exactly the tokens plain decoding draws, whatever the tree. For a
deterministic draft this accepts with the draft's probability under the
target, as the rejection sampling it replaces did, and trees now apply to
sampled and penalized requests. A seed's text differs from earlier releases
(the stream changed), not its distribution.

MLX 27B, DFlash2 block 16, T 0.6 (top-k 20, top-p 0.95), 8 prompts × 256:
chain 19.10 tok/s (2.48× plain), auto trees 22.80 (2.95×; 4.61 per pass).
Five prompts equal plain sampling token for token; three part at single
tokens (chain and tree at the same token on one). In strict math
(`FERRUM_HYBRID_EXACT=1`) all three equal plain sampling, chain and tree, so
the default-kernel divergences are rounding; at the one that repeated, the
draw fell 0.001 of the distribution's mass from an edge between two tokens
(the bench now reports this draw margin for sampled runs).

## Experiment 9 — head-to-head with Kapitsa after the port

Same 4 chat prompts through each server (`h2h.py`, client-side decode
tok/s, two passes each, ranges below), the base Qwen3.8-27B, its
DFlash2 drafter, Apple M5, one slot. Kapitsa 0.2.16 runs its defaults
(16-row DFlash2 block, KFlash trees, MLX 4-bit weights).

| Setup | code-fib | code-bsearch | prose-revolution | prose-haiku |
|---|---|---|---|---|
| Kapitsa, greedy | 31.2–31.3 | 27.3–28.1 | 22.3–22.6 | 24.2–24.3 |
| Ferrum MLX, `--draft-tree auto`, greedy | 25.8–26.9 | 23.9–24.1 | 18.1–19.2 | 20.8–22.1 |
| Ferrum GGUF Q4_K_XL, auto, greedy | 17.1–17.7 | 18.6–19.0 | 14.6–14.7 | 16.4–16.5 |
| Kapitsa, T 0.6 | 22.3–22.8 | 22.9–23.1 | 23.6–23.8 | 25.4–25.8 |
| Ferrum MLX, auto, T 0.6 | 19.4–21.2 | 19.8–20.4 | 17.8–18.2 | 21.3–21.5 |
| Ferrum GGUF, auto, T 0.6 | 16.5–16.6 | 17.3–17.5 | 12.7–12.9 | 16.5–16.9 |

Before the port, Ferrum's best (GGUF, block-8 DFlash2 tree 15) reached 17.5
greedy. On MLX Ferrum now reaches 82–91% of Kapitsa greedy and 75–93% at
T 0.6. Plain decoding is level: Kapitsa 7.66–7.72 tok/s, Ferrum on the same
MLX checkpoint 7.42–7.84. Both engines' GEMVs stream at this Mac's measured
read ceiling (about 121 GB/s), so qmvf's fusion was not ported.

The rest of the gap, from both servers' budget logs: Kapitsa's 16-row cycle
costs 183–190 ms of GPU time for 5.06–5.68 tokens; Ferrum's costs 196 ms of
wall time for 5.0–5.3. Kapitsa drafts and builds the tree on the GPU in the
same command stream as the verify; Ferrum reads the candidate lattice back
and builds on the host (about 5 ms), and its verify costs 4% more at a full
context's capacity than at 8K (163 against 170 ms), which is worth a look.
On the GGUF target `auto` is 3–5% behind `--draft-block 16 --draft-vocab
98304 --draft-tree 15`, whose Q4_K verify makes 2-row cycles look cheap.

A bug found on the way: wall-clock costs include a row count's first
pipeline builds, which skewed the cached table when such a cycle set the
cost base; a count's first cycle in a process is now left out of the costs.
