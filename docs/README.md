# Ferrum documentation

## Guides

| Document | For |
|---|---|
| [server.md](server.md) | Running `ferrum-server` and `ferrum-cli`: endpoints, caching, thinking controls, speculative decoding, vision |
| [tui.md](tui.md) | The full-screen launcher: chat, the sandboxed coding agent, the speed and intelligence benchmarks |
| [development.md](development.md) | Building, testing, profiling and using the crate as a library |
| [architecture.md](architecture.md) | How the system fits together, and the runtime's safety argument |
| [experimental.md](experimental.md) | **Experimental** Flint and Spark support: status, validation numbers, known gaps |
| [vision.md](vision.md) | The image encoder and how image tokens are positioned |

## Engineering journals

Ferrum was built in phases, with each phase's experiments recorded as they happened, including the ones that failed or were reverted. These live in [`history/`](history/) and are not kept up to date: each is written as of its own phase, so early phases may describe limits or code that later phases removed. The guides above describe the current state. Raw logs and JSONL results are under [`measurements/`](measurements/).

| Phase | Focus | Write-ups |
|---|---|---|
| 1 | Metal tensor runtime, kernel ABI, safety boundary | [Results](history/phase1-results.md) |
| 2 | Transformer primitives on synthetic weights | [Results](history/phase2-results.md) |
| 3 | First real model (Qwen2.5-0.5B BF16), validated against Transformers | [Results](history/phase3-results.md) |
| 4 | Command batching, KV cache, fast GEMV/GEMM | [Results](history/phase4-results.md), [matrix optimization](history/phase4-matrix-progress.md) |
| 5 / 5.5 | Packed GGUF and MLX quantized weights | [Closeout](history/phase5-closeout.md), [experiments](history/phase5-experiments.md), [5.5 experiments](history/phase5.5-experiments.md) |
| 6 | Dense, MoE and hybrid architectures (Qwen3, Granite, OLMo 2, LFM2) | [Journal](history/phase6-architecture-journal.md) |
| 7 | Closing the gap to llama.cpp on GGUF (73 experiments) | [Journal](history/phase7a-performance-journal.md) |
| 8 | Large hybrid models (27B dense, 35B MoE), memory planner, server | [Plan](history/phase8-plan.md), [journal](history/phase8-journal.md) |
| 9 | Lossless speculative decoding: MTP, DFlash, DSpark | [Plan](history/phase9-plan.md), [journal](history/phase9-journal.md) |
| 10 | Long-context attention: Q8 K/V cache, TensorOps verify and staged prompt attention | [Journal](history/phase10-journal.md) |
| 11 | Tree speculative decoding: tree-masked verify, path commit, MTP draft trees | [Plan](history/phase11-plan.md), [journal](history/phase11-journal.md) |
| 1–4 | The original runtime design: tensor ABI, kernel dispatch, batching (formerly the second half of architecture.md) | [Runtime internals by phase](history/architecture-by-phase.md) |
| — | Image input (Qwen3-VL projector) | [vision.md](vision.md) |
| — | Re-check of every documented claim against llama.cpp build 11669 (October 2026) | [Record](measurements/verification-2026-10/README.md) |

The journals mention work branches such as `codex/phase2-transformer`. Those branches have been merged into `main` and archived as tags, for example `archive/codex/phase2-transformer`.
