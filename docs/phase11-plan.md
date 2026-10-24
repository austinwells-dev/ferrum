# Phase 11 — Tree speculative decoding

Phase 9's drafters propose a chain: the target verifies `[anchor, d1..dk]`
in one forward, and the first wrong guess discards everything after it.
Phase 11 lets a drafter also propose its next-best guesses where it is
unsure, as a tree verified in the same single target forward. The longest
root-to-node path the target agrees with is kept, plus the target's own next
token. Greedy only; the output must stay identical to plain decoding.

## Why it can pay

On the dense 27B, a verify of 4 to 16 rows costs about the same, 1.43× a
decode step, and 17 rows jumps to 2.21× (Phase 9, Experiment 1). MTP's
default chain is 3 drafts (4 rows), so up to 12 more tree nodes cost almost
nothing to verify. What they cost is drafting, and what they buy depends on
how often the drafter's second guess is the target's choice.

## Design

- **Mask.** Each verify row sees the cached sequence and only its own
  ancestors among the new rows (a per-row `u32` bitmask, so at most 32 rows),
  at rotary and window position `len + depth`. The verify and staged
  TensorOps attention kernels get `_tree` variants specialized at compile
  time, so the chain and prompt kernels are unchanged.
- **Recurrent layers.** Gated DeltaNet conv and delta-rule kernels get tree
  variants in which each node starts from its parent's state (its latest
  ancestor in the mask). As in Phase 9, verify only reads the live state and
  records a tape.
- **Commit.** `commit_path` replays the accepted path's tape rows into the
  recurrent state, moves its K/V rows to consecutive positions (F16 or Q8
  with scales, linear or ring) and moves the captured hidden state and
  features to the rows the drafter ingests.
- **Drafting.** `Drafter::draft_tree` (a chain by default). Prompt lookup
  merges every earlier continuation into a trie (a test vehicle on the
  attention-only models). MTP keeps its chain and adds each step's next-best
  tokens as leaves, best first by the drafter's path probability under a
  node budget (`chain_with_siblings`).

## Correctness gates

Per phase, on a fixed prompt set:

1. Tree output identical to plain greedy decoding. With default kernels a
   divergence is reported with the target's top-2 margin, because a 16-row
   verify runs different matrix and attention kernels than a 4-row chain or
   single-token decode.
2. With `FERRUM_HYBRID_EXACT=1 FERRUM_HYBRID_REFERENCE_ATTENTION=1` (kernels
   whose result for a row does not depend on the other rows), every accepted
   path row is bit-identical to the chain's row, and every output identical.
   `FERRUM_HYBRID_EXACT=1` alone covers only the projections.

## Work order

1. Tree mask in the attention kernels; checked against an F64 reference on
   random trees, and a chain-shaped tree bit-identical to causal rows.
2. `verify_tree` and `commit_path`, checked with oracle trees beside a chain
   (`examples/hybrid_tree_check.rs`) on Swift (DeltaNet) and Spark (rings).
3. Accept walk and session loop; prompt-lookup trees; `--draft-tree N`.
4. MTP chain plus leaf siblings; benchmark against chains, sweeping the node
   budget (4, 8, 12, 15, 16, 24) on a coding and a prose prompt.
5. (Pending a decision on Phase 4's numbers.) Full best-first MTP trees,
   expanding siblings too, with tree-masked MTP drafting.

Progress and measurements go in the [Phase 11 journal](phase11-journal.md).
