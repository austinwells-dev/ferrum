# Structured output and session-path regression checks

Branch `feat/smooth-mode-and-qol` (working tree after `6ad3d8f`), 15 November 2026, against `experimental` at `c35c0f8`. Machine: Apple M5, 32 GB, macOS 27.2.1. Models: `Spark-X2.6-4B-Q4_K_M.gguf` (dense, attention only) and `Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive-IQ4_XS.gguf` (MoE hybrid), F16 K/V.

## Correctness

[regression.sh](regression.sh) → [regression-2026-11-15.txt](regression-2026-11-15.txt). On both models: the greedy reply to one prompt (200 tokens, thinking off, through `ferrum-cli`) is byte-identical between base and branch; `hybrid_session_check` passes every case; `hybrid_spec_check` reproduces the greedy reference.

KL divergence against llama.cpp was not run: there is no `--kl-divergence-base` file for these two models on this machine, and the branch changes no kernel and no forward pass.

## Speed

[abba.sh](abba.sh) → [abba-2026-11-15.txt](abba-2026-11-15.txt): base, branch, branch, base, two rounds, under `caffeinate -dims` after the CPU preflight (`dasd` exempted, so these compare the two builds and are not absolute figures). `bench` is `hybrid_bench --pp 512,2048 --tg 128`; `chat` is `ferrum-cli` on a piped prompt of about 990 tokens with 200 greedy tokens, which runs `Session::advance`, the code the branch changes. Best of four, with the range of the four in brackets, tok/s:

| Model | Measure | Base | Branch |
|---|---|---|---|
| Spark 4B | bench pp512 | 1188.5 (1152.9–1188.5) | 1160.5 (1144.7–1160.5) |
| | bench pp2048 | 1089.0 (1079.2–1089.0) | 1082.4 (1069.6–1082.4) |
| | bench tg128 | 37.01 (36.94–37.01) | 36.98 (36.76–36.98) |
| | chat prefill | 1086 (1068–1086) | 1079 (1068–1079) |
| | chat decode | 35.1 (34.8–35.1) | 35.0 (34.8–35.0) |
| 35B MoE | bench pp512 | 928.4 (841.5–928.4) | 921.6 (899.1–921.6) |
| | bench pp2048 | 884.9 (834.9–884.9) | 877.8 (864.4–877.8) |
| | bench tg128 | 50.82 (49.98–50.82) | 50.32 (49.30–50.32) |
| | chat prefill | 884 (849–884) | 883 (800–883) |
| | chat decode | 50.5 (49.7–50.5) | 50.6 (49.3–50.6) |

`hybrid_bench` calls the forward pass directly and runs none of the changed code, so its base-to-branch differences are run-to-run spread: Spark's pp512 best differs by 2.4% between two builds of identical kernels, with the first base run the outlier (1188.5 against 1152.9–1163.2 for the other three). The chat rows, which do run the changed code, differ by at most 0.6% at best-of-four and overlap in range. One branch chat prefill on the MoE read 800 tok/s; the base range on the same measure spans 849–884.

## Structured output

[structured-output.sh](structured-output.sh) → [structured-output-2026-11-15.txt](structured-output-2026-11-15.txt), Spark 4B through `ferrum-server`. Six of seven requests ended as the request intended (five replies that parse as JSON, one schema refused with 400). Request 1, a greedy reply to a schema with `maxItems: 3`, ran on inside its third string until `max_tokens` and so did not parse: the limitation [server.md](../../server.md#structured-output) describes. Constrained replies ran at 31.9–36.1 tok/s against 37.2 unconstrained; request 7's 44.7 tok/s includes reasoning drafted by prompt lookup.
