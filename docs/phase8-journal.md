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
