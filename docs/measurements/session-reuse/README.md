# Session reuse: snapshots keep sliding-window rings, pinned reply-header checkpoint

Branch `fix/session-cache-reuse` against `experimental` `ab8f57e`, 29 October 2026. Machine: MacBook Pro Mac17,2, Apple M5, 10 CPU cores, 32 GB, Metal 4, macOS 27.0.1. One model in memory at a time, screen unlocked (checked before every run), `caffeinate -dims`, `scripts/phase7a_cpu_preflight.py` passed first with `PHASE7A_PREFLIGHT_IGNORE=dasd`.

Models (this machine's paths are in [regression-2026-10-29.sh](regression-2026-10-29.sh)): Swift-1.5-Qwen3.8-27B Q4_K_M (dense), Tiel-Coder-35B-A3B UD-IQ4_XS and Qwen3.6-35B-A3B IQ4_XS (MoE), LFM3-VL-2B Q4_K_M and LFM3-1.7B Q4_K_M (short-conv with sliding-window attention), Spark-X2.6-4B Q4_K_M and Flint UD-Q3_K_XL (attention only, sliding-window layers).

## Regression run

[regression-2026-10-29.sh](regression-2026-10-29.sh) built both trees (`experimental` `ab8f57e` and the branch at `aa2b402`) and ran everything interleaved. Raw output: [regression-2026-10-29.txt](regression-2026-10-29.txt).

**Greedy output and memory.** `ferrum hybrid --model MODEL --raw --prompt 'The capital of France is' --max-new-tokens 48`: identical text on Swift, Tiel, LFM3-VL and Spark. Every `measured:` line is within 1 MiB of its prediction on both trees. Spark's state is 19726 MiB on the branch against 19960 MiB: snapshot slots now reserve room for the sliding-window rings they copy, and the planner fits the context in what is left ([context below](#planned-context)).

**KLD against llama.cpp** (Homebrew 0.7.0, build 11669; corpus `README.md` plus `docs/*.md` at `ab8f57e`, `-c 512`). Identical on both trees:

| Model | Chunks | Ferrum PPL | llama.cpp PPL | Mean KLD | Same top-1 |
|---|---|---|---|---|---|
| Swift 27B Q4_K_M | 8 | 7.3996 | 7.3922 | 0.000126 | 99.56% |
| Tiel 35B-A3B UD-IQ4_XS | 8 | 10.6813 | 10.7243 | 0.006947 | 96.08% |
| LFM3-1.7B Q4_K_M | 1 (the [lfm3](../lfm3/README.md) base file) | 158.0000 | 160.4486 | 0.001009 | 99.22% |

**Chat reuse.** `hybrid_chat_reuse` runs a five-turn chat through the runtime's chat path (chat template, tool call and result, 600 new tokens per turn, temperature 0) and counts, per turn, the prompt tokens that matched the previous conversation but were processed again ("lost"):

| Model | `experimental` lost (worst turn) | Branch lost (worst turn) |
|---|---|---|
| LFM3-VL-2B with a picture | 3233 (1817) | 6 (4) |
| LFM3-VL-2B text only | 3653 (1851) | 4 (4) |
| Qwen3.6-35B-A3B | 289 (289) | 3 (3) |
| Tiel-Coder-35B-A3B + DFlash2 (`--draft-max 2`) | 0 | 0 |
| Swift 27B (4 turns, 300 tokens) | 0 | 0 |

On `experimental`, LFM3 loses its cache once a turn adds more than about 570 tokens after a snapshot (the sliding-window ring has overwritten what the snapshot needed). Qwen3.6 loses it when the template drops a past reply's reasoning, one token before the end-of-prompt snapshot. The few tokens the branch still redoes are the rewritten reply header after the pinned checkpoint.

Replies differ from `experimental` on some turns (the hash column). Models with recurrent state now split the prefill at the checkpoint, and a chunk boundary rounds differently from the middle of a chunk; a near-tie later in a 600-token greedy reply then picks the other token. The session check below compares each reuse against a reference that splits at the same place, and those match exactly.

**Speed.** `hybrid_bench MODEL --pp 512,2048 --tg 128`, A-B-B-A twice (8 runs per model). Best of the four runs per tree, tokens per second:

| Model | Test | `experimental` | Branch |
|---|---|---|---|
| Swift 27B | pp512 | 194.90 | 194.09 |
| | pp2048 | 185.85 | 185.59 |
| | tg128 | 7.46 | 7.45 |
| Tiel 35B-A3B | pp512 | 969.65 | 966.42 |
| | pp2048 | 934.63 | 927.84 |
| | tg128 | 44.07 | 44.00 |
| LFM3-VL-2B | pp512 | 3048.88 | 3043.24 |
| | pp2048 | 2922.40 | 2920.27 |
| | tg128 | 101.98 | 101.96 |

Every difference is inside the run-to-run spread of the same tree: Tiel pp2048 ranges 915.67–934.63 on `experimental` and 915.15–927.84 on the branch, and drifts down over the run on both. The first Swift branch run (pp512 158.59, tg128 6.90) and the first base run (tg128 6.40) are warm-up outliers. The change does not touch the forward pass; snapshots are taken once per prompt.

## Session check

`hybrid_session_check MODEL [--draft ...]` (context 4096, two snapshot slots) runs a scripted conversation (extend, rewind, retry, replace, a pinned checkpoint, a 900-token turn, a later turn, then a rewind to the pinned reply header). It compares each greedy reply with a fresh session's and checks that the rewind reuses everything before the checkpoint. The fresh session replays the forwards that produced the cached session's rows: the same prompt chunks and the same single-row steps for its own replies. Before this change it computed the whole prefix in prompt-sized chunks, so a row the cached session had decoded one at a time was compared against a chunked one, and Flint and Qwen3.6 failed on near-ties on every tree.

Branch, working tree after `aa2b402` ([session-check/](session-check/)): **PASS** on LFM3-VL-2B, LFM3-VL-2B with lookup drafting, Spark, Flint, Qwen3.6, Tiel, Tiel with DFlash2, Tiel with MTP ([../mtp-moe](../mtp-moe/README.md)), Swift 27B and Swift 27B with MTP.

`experimental` ([session-check/experimental/](session-check/experimental/), the earlier check without `set_checkpoint`, run on `ab8f57e`): **FAIL** on all eight models run. The pinned rewind reuses nothing on LFM3, LFM3 with lookup, Spark, Tiel, Tiel with DFlash2, Swift and Qwen3.6. Flint resumes there from its rings but fails the old reference on the later turn, and Qwen3.6 also on the first extension (the reference artifacts described above).

## Planned context

`hybrid_plan MODEL ...` (GGUF metadata only), both trees: [plan-2026-10-29.txt](plan-2026-10-29.txt). Each of the two snapshot slots now also holds a copy of the sliding-window rings, which the planner takes from the K/V budget:

| Model | K/V | `experimental` auto context | Branch auto context |
|---|---|---|---|
| Spark-X2.6-4B | f16 | 564480 | 557824 (−1.2%) |
| Flint | f16 | 204288 | 198912 (−2.6%) |

Spark with Q8 K/V and Flint with Q8 K/V stay at their caps (1048576 and 262144), as do LFM3-VL-2B (128000) and Qwen3.6-35B-A3B (262144) with either format.
