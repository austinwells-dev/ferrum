# Phase 8 journal — large hybrid models

Hardware: Apple M5 (10-core GPU), 32 GB unified memory, 24 GiB GPU working set
(`iogpu.wired_limit_mb=24576`), macOS 27. Reference: Homebrew llama.cpp 687e77892 (build 10330).
Goals and architecture notes are in the [Phase 8 plan](phase8-plan.md).

## Models

| Model | File | Size | Tensor types (trunk) |
|---|---|---|---|
| Swift 1.5 Qwen3.8-27B | `Swift-1.5-Qwen3.8-27B-Q4_K_M.gguf` (ukisai, rev a1614465) | 16.23 GiB | Q4_K, Q6_K, Q8_0, Q5_K; F32 norms, conv and gate projections |
| Tiel-Coder 35B-A3B | `Tiel-Coder-35B-A3B-MTP-UD-IQ4_XS.gguf` (peculiar-ragdoll, rev bbe9e566) | 18.12 GB | experts: IQ3_S gate/up, IQ4_XS or Q6_K down; everything else Q8_0 |

Both files carry one NextN/MTP block (`blk.64` / `blk.40`). Ferrum does not load it.

## Baseline: llama.cpp

`llama-bench -p 512,2048 -n 128 -r 2`, default settings (flash attention auto, F16 KV):

| Model | pp512 | pp2048 | tg128 |
|---|---|---|---|
| Swift 27B Q4_K_M | 149.8 ± 2.0 | 142.3 ± 2.1 | 5.98 ± 0.06 |

A 27B decode step reads about 15.3 GB of weights (everything except the token
embedding table). At the M5's ~153 GB/s this bounds decode at ~10 tok/s;
llama.cpp reaches 5.98 tok/s (~92 GB/s effective).

## Experiment 1 — hybrid engine, first light

A dedicated engine (`src/hybrid`) with its own kernel library
(`src/metal/shaders/hybrid.metal`) and an explicit-geometry dispatch path in the
Metal backend. Residual stream F32; GEMV kernels take F32 activations (ports of
llama.cpp's `mul_mv` kernels); prompt GEMMs dequantize weight tiles to F16 and
run TensorOps with F32 accumulation. Gated DeltaNet uses the sequential
recurrent kernel for both prefill and decode; attention is a reference-grade
online-softmax kernel. All state (F16 K/V for 16 attention layers, F32 conv and
delta-rule state for 48 recurrent layers) and all activation scratch are
allocated once.

Swift 27B, raw prompt "The capital of France is", greedy: the 16 generated
tokens match `llama-completion --temp 0` exactly. Decode 6.78 tok/s (148 ms
median) before any tuning, versus llama.cpp's 5.98.

### Correctness method

`examples/hybrid_kld.rs` replays the token chunks of a `llama-perplexity
--kl-divergence-base` file (512-token chunks, scoring positions 256–510 exactly
as llama-perplexity does) and reports both perplexities, KL divergence
(base ‖ Ferrum, with llama.cpp's −16 log-probability truncation) and top-1
agreement. Corpus: Ferrum's own docs and Rust sources (prose plus code).
With `--write` it emits Ferrum logits in the same format, so llama.cpp can be
scored against a Ferrum reference.

## Experiment 2 — Qwen3.5 MoE, flash attention, TensorOps precision

- **MoE** (`qwen35moe`): GPU routing (softmax, top-8, renormalized weights,
  sigmoid shared-expert gate). Decode uses per-route expert GEMVs indexed by
  expert ID; prompts use a deterministic expert-major map, a gather, one
  TensorOps GEMM per expert and a weighted combine. IQ3_S kernels added.
- **TensorOps precision**: `relaxed_precision=false` made the F16-tile GEMMs
  3.5× slower than llama.cpp's setting (`true`); switching matched both its
  speed and its numerics (Swift KLD vs llama.cpp fell from 7e-6 to 4e-6).
- **Flash attention**: simdgroup-matrix kernel with query rows packed per KV
  group (GQA shares K/V loads), 32-key blocks with online softmax, the
  sigmoid output gate applied on write, and split-K partials plus a reduce
  kernel for few-row (decode) attention. A unit test checks it against the
  reference kernel for 24/4 and 16/2 head layouts, split and unsplit.

### Accuracy

| Model | vs llama.cpp Metal (KLD / same top-1 / PPL ratio) |
|---|---|
| Swift 27B | 4e-6 / 99.85% / 1.0001 |
| Tiel 35B-A3B | 0.0085 / 93.2% / 1.004 |

Tiel's larger number is not a Ferrum error. Its expert routing matches
llama.cpp in 39 of 40 layers on a probe prompt (the remaining layer swaps one
near-tied expert); llama.cpp's own CPU backend scores KLD 0.020 / 88.9% top-1
against its Metal backend. To measure fidelity rather than agreement, a
high-precision Ferrum reference (`FERRUM_HYBRID_EXACT=1`: every projection on
the F32-activation GEMV kernels) was written with `--write` and both runtimes
were scored against it:

| Tiel, vs exact reference (2,040 positions) | Mean KLD | Same top-1 |
|---|---|---|
| llama.cpp Metal | 0.01068 | 92.40% |
| Ferrum | 0.01057 | 91.76% |

Both fast runtimes sit equally far from the exact result; this model is
sensitive to F16 GEMM rounding, and Ferrum loses nothing relative to llama.cpp.

### Throughput (`examples/hybrid_bench.rs`, llama-bench semantics)

| Model | pp512 | pp2048 | tg |
|---|---|---|---|
| Swift 27B Ferrum | 171.9 | 158.0 | 6.29 (tg64) |
| Swift 27B llama.cpp | 149.8 | 142.3 | 5.98 |
| Tiel 35B-A3B Ferrum | 634.3 | 577.0 | 34.82 (tg128) |
| Tiel 35B-A3B llama.cpp | 802.9 | 768.9 | 35.77 |

## Experiment 3 — MoE tile, TensorOps attention, vector decode attention

**Expert GEMM tile.** A layer's routes spread over 256 experts (~16 rows each
at 512 tokens), so a 128-row activation tile was mostly padding. A 32-row tile
(llama.cpp's NR1): Tiel pp512 634 → 881, pp2048 577 → 778 tok/s.

**Bandwidth ceiling.** A pure streaming-read probe (`examples/hybrid_bandwidth.rs`)
reaches 122–134 GB/s on this M5. Real-weight GEMV probes
(`examples/hybrid_kernels.rs`) run at 113–147 GB/s, so M=1 projections are at
the ceiling; Swift decode is ~85% weight streaming and ~15% small kernels.
Sweeps of rows-per-SIMD-group and SIMD groups per threadgroup for Q4_K/Q6_K
were within ±5% run-to-run noise; the llama.cpp shapes stay.

**TensorOps flash attention (prompts).** One threadgroup per 64 queries of one
head; S = Q Kᵀ and O += P V as `matmul2d` on 64×64 key blocks, with the online
softmax over S in threadgroup memory and O kept in a cooperative tensor
(rescaled per row through `get_multidimensional_index`). Tiel pp8192 528 → 747,
pp16384 398 → 623 tok/s.

**Vector decode attention.** The simdgroup-matrix kernel loaded K/V as strided
8×8 tiles and topped out near 50–60 GB/s. The single-token kernel makes QKᵀ
key-parallel (a lane reads one contiguous 512-byte K row, Q from threadgroup
memory) and PV dimension-parallel (coalesced V rows), merges four SIMD groups
pairwise in threadgroup memory, and splits keys across ~128 threadgroups.
Decode attention at 16K–64K keys: 2–3× faster (77–97 GB/s effective).
Both attention kernels are unit-tested against the reference kernel.

### Status against llama.cpp (same GGUF, same machine)

| | Ferrum | llama.cpp |
|---|---|---|
| Swift pp512 / pp2048 / pp8192 | 171.9 / 158.0 / 162.1 | 149.8 / 142.3 / 134.4 |
| Swift tg @0 / @8K / @32K | 6.65 / 6.41 / 5.85 | 5.98 / 6.00 / 5.47 |
| Tiel pp512 / pp2048 / pp8192 / pp16384 | 867 / 865 / 747 / 623 | 803 / 769 / 662 / 559 |
| Tiel tg @0 / @16K / @32K | 42.1 / 34.5 / 29.5 | 35.8 / 30.5 / 26.8 |

Decode figures are the mean of 16 steps after the given prefilled depth; run-to-run
noise on this machine is about ±5%.
