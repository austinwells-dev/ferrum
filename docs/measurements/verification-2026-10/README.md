# Claim verification, October 2026

Every claim in the README and the guides was checked against the code or re-measured on the `experimental` branch, 23 October 2026. Machine: 32 GB Apple M5 (24,576 MiB recommended GPU working set). Reference: llama.cpp build 11669 (`620d2225c`, Homebrew 0.7.0). Models: Swift-1.5-Qwen3.8-27B Q4_K_M, Tiel-Coder-35B-A3B UD-IQ4_XS, `flint-UD-Q3_K_XL`, Spark-X2.6-4B Q4_K_M. One model in memory at a time; timed runs under `caffeinate` with the CPU preflight, Ferrum and llama.cpp interleaved.

Raw output: [correctness.txt](correctness.txt) and [correctness-followup.txt](correctness-followup.txt) (tokenizers, greedy text, memory plans, KLD, speculative and session checks, long-context retrieval), [performance.txt](performance.txt) and [performance-llama-bench.txt](performance-llama-bench.txt), [server.txt](server.txt) and [vision.txt](vision.txt).

## Correctness

| Check | Result |
|---|---|
| Tokenizer ids vs `llama-tokenize` (README, architecture.md, mixed-script text; 12.5K–13.3K ids) | identical on Swift, Tiel and Spark. Flint: llama.cpp 11669 does not know the `flint` architecture |
| Greedy continuation vs `llama-simple` | identical on Swift, Tiel and Spark |
| KLD vs llama.cpp Metal, Ferrum docs and sources | Swift 1.17×10⁻⁴ (99.51% top-1); Tiel 0.0068 (95.98%); Spark 0.00028 (99.55%, PPL 20.14 vs 20.13) |
| Tiel vs a high-precision Ferrum reference (`FERRUM_HYBRID_EXACT=1`) | Ferrum 0.0074, llama.cpp 0.0068 |
| Q8 K/V vs F16 (Tiel) | 0.0095; vs llama.cpp 0.0099 |
| llama.cpp CPU vs its own Metal (Spark) | 0.0060, 97.6% top-1: Ferrum sits 20× closer to llama.cpp Metal |
| Memory plan vs Metal's accounting | within 4 MiB (Swift 3, Tiel 2, Flint 4, Spark 0 MiB on weights + scratch; ≤1 MiB on state) |
| Auto context (32 GB) | Swift 101,120 F16 / 190,208 Q8; Tiel 262,144 (K/V 5.00 → 2.66 GiB); Flint 204,288 / 262,144; Spark 564,480 / 1,048,576 |
| Flint K/V | 14 full-attention layers at 28 KiB/token plus 76 MiB of rings |
| Verify/commit with oracle drafts (`hybrid_spec_check`) | output == greedy on all four (Tiel: one near-tie at margin 0.0012) |
| Prefix reuse vs a fresh session (`hybrid_session_check`) | PASS on all four, Spark with prompt lookup and Swift with MTP |
| Greedy output with a drafter (8 prompts × 128 tokens) | 0 mismatches: Swift MTP, DFlash2, RedHat DSpark; Tiel DFlash2; Spark lookup. Flint lookup: 4/8 drift (1/8 with exact math) |
| Long-context retrieval at 10/50/90% of ~13–14K tokens | PASS on Flint, Spark and Tiel; device memory flat |
| Main vs this branch, 96 greedy tokens | identical on Swift and Tiel |
| Headless answer check (30 math questions) | Tiel DFlash2: 0 lost; Spark lookup: 0 lost |

## Performance (tok/s, best of two interleaved rounds)

| | Ferrum | llama.cpp 11669 |
|---|---|---|
| Swift pp512 / pp8192 | 175.9 / 166.7 | 159.7 / 148.3 |
| Swift tg16 at 0 / 32K | 7.53 / 6.16 | 7.09 / 6.02 |
| Tiel pp512 / pp16384 | 919.6 / 683.3 | 941.2 / 730.2 |
| Tiel tg16 at 0 / 32K | 42.78 / 31.03 | 40.55 / 29.67 |
| Spark pp512 / pp2048 / tg128 | 1197 / 1099 / 38.5 | 1207 / 1140 / 39.9 |
| Flint pp512 / pp2048 / tg128 | 965 / 925 / 50.6 | — |

Verify cost after 1,024 cached tokens (× one decode step):

| rows | Swift Ferrum | Swift llama.cpp | Tiel Ferrum | Tiel llama.cpp |
|---|---|---|---|---|
| 4 | 1.31 | 1.17 | 1.43 | 1.52 |
| 8 | 1.38 | 1.19 | 2.52 | 2.10 |
| 9 | 1.36 | 1.82 | 2.62 | 2.32 |
| 16 | 1.51 | 2.01 | 3.30 | 3.32 |
| 17 | 2.42 | 3.69 | 3.78 | 4.17 |

llama.cpp's figures are `pp N @ d1024` time over one `tg16 @ d1024` step.

Speculative decoding through each server (`tools/spec_bench.py`, 8 prompts × 128 tokens): Swift base 7.04 / 6.95, MTP 11.91 / 13.12, z-lab DFlash2 18.04 / 11.43, RedHat DSpark 16.62 / 6.07; Tiel base 41.75 / 40.25, jzinno DFlash2 (2 drafts) 60.85 / 54.67; Spark lookup 41.48 (base 36.70); Flint lookup 54.73 (base 50.10). Draft acceptance agrees within two points between the engines. llama.cpp's drafters were converted to Q8_0 GGUF with unsloth's llama.cpp `convert_hf_to_gguf.py`.

`FERRUM_EARLY_COMMIT` on the Tiel decode: 42.5 tok/s on (best of 3) against 41.5 off, about +2.5%, within the run-to-run spread.

Main against this branch (Qwen3.5 models): both issue the same dispatches per decode step (932 Swift, 864 Tiel), with best-case GPU step times within 0.2%. Decode in ABBA order (main, branch, branch, main, twice), mean of 4 runs each: Swift 7.15 main / 7.18 branch, Tiel 43.19 / 44.41 tok/s. Prefill, best of 3: Swift pp512 196.9 / 197.2, Tiel 975.3 / 973.2.

## Server, vision and sandbox

`ferrum-server` on Spark passed 45 of 45 checks: every endpoint; cache reuse (identical retry reuses all but one token, extensions reuse the prefix, and past Spark's 1,088-row K/V ring a late divergence resumes while an early one recomputes, both matching a fresh server); every thinking control including Anthropic `thinking`; `include_usage`, `return_progress` and disconnect handling; tool calls with schema-typed arguments; a clear 400 for images without a projector; `--api-key`, `--cors`, foreign-origin 403 and DNS-rebinding 403.

Vision on Tiel with its BF16 projector (0.83 GiB) passed 11 of 11: PNG, JPEG, GIF, WebP and BMP; Anthropic base64 blocks; remote URLs refused; the image cached across turns. Encoding took 0.79–0.85 s for 640×480 and 4.37 s for 1024×1024, and the description agreed with `llama-mtmd-cli`.

The Seatbelt profile, probed with `sandbox-exec` and network on against a service bound to 0.0.0.0: 127.0.0.1, `localhost`, `::1`, the LAN address and the `.local` hostname were all refused (all reachable outside the sandbox); a Unix socket outside the workspace was refused and one inside allowed; an Apple event to Finder was never delivered; `~/.zsh_history`, the Hugging Face token and browser data were unreadable; writes outside the workspace failed.

## Fixed on the way

- Spark's chat template failed to render any request that offered tools: Ferrum's `tojson` filter refused `separators=`. It now takes `separators`, `indent` and `sort_keys`, and `tests/fixtures/spark` holds Jinja2 renderings that a unit test compares byte for byte.
- A budget-cut Spark reasoning closed with Qwen's `\n</think>\n\n` instead of the bare `</think>` its template writes.
- The TUI listed GGUFs whose embedding is IQ3_XXS, IQ3_S or IQ2_S as runnable; no embedding lookup exists for those.
- `hybrid_kld MODEL BASE --write OUT` parsed `--write` as the chunk count.
- `ferrum hybrid` wrapped every prompt in Qwen's chat markup; it now uses the GGUF's own template (`--raw` for completion).

## Not re-verified

- Flint against llama.cpp (KLD, tokenizer, speed): llama.cpp 11669 and the local unsloth build (11160) do not load the `flint` architecture.
- The small-model engine (Qwen2.5, Qwen3, Granite, OLMo 2, LFM2.5 checkpoints and the 0.6–1.0× speed figure): none of the checkpoints are on this Mac.
- The agent benchmark's ten tasks failing before and passing after their fixes (`tasks_fail_before` clones them from GitHub).
