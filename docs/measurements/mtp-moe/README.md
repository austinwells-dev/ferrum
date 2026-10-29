# MTP on MoE targets

Branch `fix/session-cache-reuse` (working tree after `aa2b402`), 29 October 2026. Machine: MacBook Pro Mac17,2, Apple M5, 10 CPU cores, 32 GB, Metal 4, macOS 27.0.1. Model: Tiel-Coder-35B-A3B `Tiel-Coder-35B-A3B-MTP-UD-IQ4_XS.gguf` (`peculiar-ragdoll/Tiel-Coder-35B-A3B-GGUF-MTP`, snapshot `bbe9e566`).

## What the block holds

Tiel's MTP block (`blk.40`) has a gated-attention layer like the dense 27B's, but its FFN routes to 256 experts with a shared expert, as the trunk's layers do. Its three expert tensors (`ffn_{gate,up,down}_exps`) are Q3_K, the only Q3_K tensors in the file. The `UD-Q3_K_XL` file of the same model has the same three. Ferrum has no Q3_K kernels, so the loader rewrites those blocks as Q6_K, which holds every Q3_K value exactly (same `d`, 6-bit group scales as 8-bit, 3-bit weights as 6-bit). `q3_k_rewritten_as_q6_k_keeps_every_value` checks bit-identical dequantization against llama.cpp's `dequantize_row_q3_K` and `dequantize_row_q6_K`; `q3_k_projections_match_the_cpu_dequant` runs the result through the dense and expert kernels. The three tensors grow from 330 MiB to 630 MiB (256 experts × 512 × 2048 weights each, 110 bytes per 256 as Q3_K and 210 as Q6_K).

## Speed

[tiel-2026-10-29.sh](tiel-2026-10-29.sh): `hybrid_spec MODEL --drafter D --drafts K --tokens 256 --baseline` on the 8 built-in chat prompts, greedy, two interleaved rounds, `caffeinate -dims`, screen unlocked, preflight passed. Raw output: [tiel-2026-10-29.txt](tiel-2026-10-29.txt).

| Drafter | Drafts | Round 1 tok/s | Round 2 tok/s | Plain decoding | Speedup | Accepted per step | Greedy mismatches |
|---|---|---|---|---|---|---|---|
| MTP | 1 | 50.33 | 58.74 | 36.92 / 43.22 | 1.36× | 1.85 | 0 |
| MTP | 2 | 62.69 | **63.35** | 42.93 / 43.12 | 1.46–1.47× | 2.49 | 0 |
| MTP | 3 | 59.86 | 59.92 | 42.91 / 42.96 | 1.39–1.40× | 2.83 | 0 |
| `incoai/Qwen3.6-35B-A3B-DFlash2` | 2 | 64.63 | **64.77** | 42.50 / 42.56 | 1.52× | 2.51 | 0 |

The first run of round 1 (MTP, 1 draft) was the warm-up: its plain decoding ran at 36.9 tok/s against about 43 everywhere else. Two drafts are fastest for MTP: a third accepts 0.35 more tokens per step but costs 4 ms more drafting and 3.6 ms more verifying (each draft routes to its own experts). So `--draft mtp` now defaults to 2 drafts on MoE targets. DFlash2 is about 2% faster at the same acceptance, because its single drafting pass costs 7.5 ms against MTP's two chained passes at 8.4 ms; MTP needs no download.

Memory, predicted against measured: MTP 712.5 / 709.4 MiB (1 draft) to 716.4 / 713.3 MiB (3), DFlash2 330.7 / 329.3 MiB.

## Correctness

[session-check-tiel-mtp.txt](session-check-tiel-mtp.txt): `hybrid_session_check MODEL --draft mtp` passes (every reuse identical to the replayed reference, the pinned rewind reuses everything before the checkpoint), 2.1–3.2 tokens accepted per verify step.

[spec-check-tiel-mtp.txt](spec-check-tiel-mtp.txt): `hybrid_spec_check MODEL --draft mtp` first diverges from plain decoding at token 59, where the reference's top-2 margin is 0.0012: the same near-tie plain Tiel and DFlash2 show on `experimental`. Verify rows match single-row decode to max |dlogit| 0.0101 (KL 1.55e-6), and replay after a commit matches the plain forward exactly.
