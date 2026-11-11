## What and why

<!-- What changed, and the reason for it. Link issues or journal entries. -->

## How it was checked

<!-- Commands run and their results. CI has no GPU, so say whether `cargo test --release` passed locally. -->

- [ ] `cargo fmt --check`, both `cargo clippy` runs, `python3 tools/check_links.py`
- [ ] `cargo test --release` passed locally (and `--features legacy-engine` if shared or general-engine code changed)
- [ ] Docs updated for every behaviour, flag, default or number this changes (see AGENTS.md, "Keep the docs true"), or none needed
- [ ] README still true (Install, Quick start, Performance, Limitations)

## Large change?

<!-- Kernels, forward pass, attention/KV, sampling, sessions, speculative decoding, planner, tokenizer or templates, or >~300 lines in src/hybrid or src/metal. If so, fill in or delete this section. -->

- Models tested (dense and MoE):
- Greedy output vs base: identical / differences explained:
- KLD vs llama.cpp, before → after:
- `hybrid_bench` A-B-B-A, best-of-N with spread (base vs branch):
- Raw output: `docs/measurements/<topic>/`
