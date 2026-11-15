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

## Idle unload

[idle-unload-2026-11-15.txt](idle-unload-2026-11-15.txt): `ferrum-server --idle-unload 4 --slot-save-path DIR` on Spark 4B, two identical requests 7 s apart. `footprint` read 23 GB loaded and 81 MB after the first unload (434 MB after the second). The request that woke the server took 5.92 s against 1.71 s, of which 4.6 s was load, warmup and restoring the slot; it reused 73 of its 74 prompt tokens from the restored cache.

## Load time

Metal hands out zero-filled buffers, and `MetalDevice::allocate` then wrote zeros over them again, which touched every page of the weights and the K/V cache once before the file was read into them. Weights and K/V no longer get that second pass (the streamed-expert arena and scratch still do).

[load-abba.sh](load-abba.sh) → [load-abba-2026-11-15.txt](load-abba-2026-11-15.txt): base `716e9b4`, A-B-B-A, two rounds, under `caffeinate -dims` after the CPU preflight (`dasd` exempted). The reply (200 greedy tokens) has the same md5 in all sixteen runs. Four runs per side, range:

| Model | Measure | Base | Branch |
|---|---|---|---|
| Spark 4B (3.3 GiB weights, 19.3 GiB K/V) | reported load, s | 0.4–0.5 | 0.4 |
| | whole process, s | 9.35–10.70 | 7.60–7.90 |
| | bench pp512 / pp2048 / tg128, best | 1212.8 / 1141.1 / 38.89 | 1208.3 / 1111.1 / 38.99 |
| | chat prefill / decode, best | 1121 / 36.4 | 1101 / 36.5 |
| 35B MoE (17.4 GiB weights, 5.3 GiB K/V) | reported load, s | 4.4–4.7 | 3.2–3.3 |
| | whole process, s | 11.88–13.74 | 11.08–11.21 |
| | bench pp512 / pp2048 / tg128, best | 961.9 / 924.1 / 53.75 | 958.0 / 920.7 / 53.74 |
| | chat prefill / decode, best | 914 / 53.2 | 914 / 53.4 |

The whole-process time includes the K/V allocation, which the reported load time does not: that is where Spark's 1.7 s goes. Speed is within the spread: the one figure that stands out, Spark's base pp2048 of 1141.1, is the first run of the session (the other three base runs read 1094.2–1106.6, the branch's four 1097.8–1111.1). Every bench run starts a new process, so each K/V page it writes on the branch is written for the first time. After the change `ferrum hybrid --raw --prompt hi` on Spark prints `measured: weights+scratch 3496 MiB (predicted 3496), state 19726 MiB (predicted 19727)`, the process footprint with the model loaded is still 23 GB, and `hybrid_session_check` passes.

## logit_bias

[logit-bias-2026-11-15.txt](logit-bias-2026-11-15.txt): banning the token `Paris` makes the greedy one-word answer come out as the same word in two tokens; an id past the vocabulary changes nothing; a value outside -100..100 is refused with 400.

## tool_choice

[tool-choice.sh](tool-choice.sh) → [tool-choice-2026-11-15.txt](tool-choice-2026-11-15.txt). The prompt asks for a French greeting and nothing else, so without a forced choice the model answers in text (case 1). On the 35B MoE, `required` and a named function produced a parsed call with thinking off, with thinking on, with a 16-token reasoning budget, streamed, and through `/v1/messages` (`any`); a name missing from `tools` got 400. On Spark 4B, whose tool-call format the runtime does not open, the same requests ran unforced with a warning each. Flint was not run.

## Final tree

[final-regression-2026-11-15.txt](final-regression-2026-11-15.txt): after the load-time, `logit_bias` and `tool_choice` changes, the greedy reply is byte-identical to `716e9b4` on both models, `hybrid_session_check` passes and `hybrid_spec_check` reproduces the plain forward.

Not adopted: thin LTO with one codegen unit. Interleaved against the default build on Spark 4B it moved neither constrained decoding (32.5–36.5 against 32.0–36.4 tok/s) nor `hybrid_bench` beyond their spread; the binary was 19% smaller (9.4 against 11.6 MB).
