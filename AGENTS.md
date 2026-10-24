# Working on Ferrum

Instructions for anyone changing this repository, human or agent. They describe how the project is run, not suggestions: CI and the branch rules on GitHub enforce the parts that can be enforced.

## What this is

A single Rust crate that runs LLMs on Apple Silicon with its own Metal kernels. The three apps (`ferrum-tui`, `ferrum-server`, `ferrum-cli`) run only the **hybrid engine** (`src/hybrid`, plus `vision`, `tokenizer`, `loader`, `tensor` and `metal`). The older general transformer engine (`model`, `ops`, `nn`, `generation`, `sampling`, `quantization`, `reference`, `metal/legacy.rs`) is behind the `legacy-engine` cargo feature, off by default, together with `tests/`, `benches/` and the non-`hybrid_` examples.

Start with [docs/README.md](docs/README.md) (guides) and [docs/architecture.md](docs/architecture.md) (current design). [docs/history/](docs/history/) is a record of how things were built and is not kept up to date: don't treat it as a description of the current code.

## Branches and pull requests

```
topic branch ──PR──► experimental ──PR──► main
```

- **Never commit or push to `main` or `experimental` directly.** Every change lands through a pull request. This includes one-line README fixes and edits made in GitHub's web editor.
- Branch from the latest `experimental`: `git switch -c <kind>/<short-name> origin/experimental`. Use `feat/`, `fix/`, `perf/`, `docs/`, `refactor/` or `chore/`.
- Open the PR **into `experimental`**. Keep it to one topic, and say in the description what changed, why, and how it was checked (see the PR template).
- `main` receives changes **only** from a PR whose head is `experimental` (CI rejects any other head). Open one when `experimental` is in a state you would hand to a user: docs true, checks below passed. Merge it with a merge commit, never squash, so `main` stays an ancestor of `experimental` and the two never diverge.
- If `main` ever gets a commit `experimental` lacks, bring it back through a PR into `experimental` before anything else goes to `main`.
- Don't force-push shared branches, and don't rewrite history that has been pushed.

Local guard: `git config core.hooksPath .githooks` enables [.githooks/pre-push](.githooks/pre-push), which refuses pushes to `main` and `experimental`.

## Before every PR

```sh
cargo fmt --check
cargo clippy --all-targets -- -D warnings
cargo clippy --all-targets --all-features -- -D warnings
cargo test --release                          # hybrid engine, apps, Metal backend (needs the GPU)
cargo test --release --features legacy-engine # only if you touched shared code or the general engine
python3 tools/check_links.py                  # every relative link and #anchor in the markdown
```

CI runs the same checks except GPU tests: hosted macOS runners have no real Apple GPU. So you must run `cargo test` locally and say so in the PR.

## Keep the docs true

The guides (`README.md`, `docs/*.md` outside `history/` and `measurements/`) describe the code **as it is on this branch**. A PR that changes behaviour updates them in the same PR:

| You changed | Update |
|---|---|
| A flag, endpoint, env var or default | [docs/server.md](docs/server.md), [docs/tui.md](docs/tui.md) or [docs/development.md](docs/development.md), whichever documents it |
| Engine design: memory planning, sessions, speculative decoding, kernels, safety boundary | [docs/architecture.md](docs/architecture.md) |
| A supported model, quant, architecture or format | README tables and [docs/experimental.md](docs/experimental.md) if it isn't production-ready |
| A performance or accuracy number | The number in the README or guide, with its new measurement record (below) |
| A module, binary or tool added, removed or moved | [docs/development.md](docs/development.md) layout table, and this file if the overview above changes |
| The number of `unsafe` blocks | The count in architecture.md's "Safety boundary" |

Rules:

- **No claim without evidence.** Every number in the README or a guide comes from a run recorded under `docs/measurements/<topic>/`, with machine, model, quant, commit and command. Don't round in Ferrum's favour, and don't keep a number the code no longer produces.
- **The README is the front door.** After any user-visible change, reread its Install, Quick start, Performance and Limitations sections and fix what is no longer true.
- **History is frozen.** Don't edit files in `docs/history/` except to fix links. Ongoing multi-PR work keeps a journal in `docs/history/` (date each entry). When it lands, move what is true into the guides.
- New experimental features go in [docs/experimental.md](docs/experimental.md) with their validation and known gaps until they meet the bar below.

## Regression checks for large changes

A change is **large** if it touches Metal kernels (`src/metal/shaders/*.metal`), the forward pass, attention or K/V cache, sampling, sessions and snapshots, speculative decoding and drafters, the memory planner, tokenizers or chat templates. So is any change of more than about 300 lines under `src/hybrid` or `src/metal`. A large change needs a measured comparison against `experimental` (the PR's base) **in the PR description** before merge.

Correctness, on at least one dense and one MoE model the change affects:

- **Greedy output is identical** to the base branch for the same prompt and token count (or explain every difference).
- `cargo run --release --example hybrid_kld -- MODEL.gguf BASE.kld`: KL divergence against llama.cpp's `llama-perplexity --kl-divergence-base` file must not get worse. Report KLD, top-1 agreement and perplexity before and after.
- `hybrid_spec_check` and `hybrid_session_check` for anything near verify, commit, sessions or drafters. The headless answer check (`keeps_its_answers` in [docs/development.md](docs/development.md)) for drafter changes.
- When the memory planner or any allocation changes, check `cargo run --release -- hybrid --model MODEL.gguf --raw --prompt hi`. Its `measured: ... (predicted ...)` line must stay within 4 MiB, and `hybrid_plan` shows the context that now fits.

Performance, with `cargo run --release --example hybrid_bench -- MODEL.gguf --pp 512,2048 --tg 128`:

- Build both binaries first and run them **interleaved A-B-B-A** (base, branch, branch, base, ...), at least two rounds. Report best-of-N and the spread. The record in `docs/measurements/verification-2026-10/regression-abba.txt` shows the format.
- Measurement hygiene, because results that break these rules don't count:
  - Run under `caffeinate -dims`, and keep the screen unlocked. Timings taken while the Mac is locked are invalid.
  - Have only one model in memory, with no other GPU work running.
  - Run `scripts/phase7a_cpu_preflight.py` first. It waits for background CPU load to settle, and can wait indefinitely on `dasd`; if so, set `PHASE7A_PREFLIGHT_IGNORE=dasd`.
- A slowdown beyond the run-to-run spread needs an explanation and an explicit decision in the PR. "Within noise" needs the numbers that show it.

Save raw output under `docs/measurements/<topic>/` and link it from the PR.

## Code

- Match the surrounding code: its naming, comment density and idioms. Comments explain why, not what.
- `unsafe` is allowed only in `src/metal`, each block with a `// SAFETY:` comment. Every GPU dispatch goes through `MetalDevice`'s checked binding path (`dispatch_hybrid`; `dispatch` in `metal/legacy.rs`), which checks bounds, ownership and aliasing before encoding.
- New kernels get a parity test against a CPU reference (see `src/hybrid/iq_tests.rs`, `mlx_tests.rs`, `tree_tests.rs`).
- Code that only the general engine uses goes behind `#[cfg(feature = "legacy-engine")]`. Don't make the apps depend on it.
- Never commit model weights, `.safetensors`, raw Instruments `.trace` bundles or anything with secrets. Keep large logs out unless a guide cites them.
- Commit messages: a short imperative subject line saying what changed (`TUI bench: count trees in the header`), then a body with why and how it was checked when that isn't obvious.
