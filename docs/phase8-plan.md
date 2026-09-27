# Phase 8 — Large hybrid models

Phase 7 tuned GGUF kernels on sub-1B and 8B-A1B models. Phase 8 makes Ferrum usable on
the two models below, on the reference Apple M5 (32 GB, 24 GiB GPU working set).

| Target | GGUF | Architecture |
|---|---|---|
| Swift 1.5 (Qwen3.8-27B derivative) | `ukisai/Swift-1.5-Qwen3.8-27B-GGUF` Q4_K_M, 17.44 GB | `qwen35` dense hybrid, 64 layers |
| Tiel-Coder 35B-A3B (Ornith-1.5 / Qwen3.6-35B-A3B derivative) | `peculiar-ragdoll/Tiel-Coder-35B-A3B-GGUF-MTP` UD-IQ4_XS, 18.12 GB | `qwen35moe` hybrid MoE, 40 layers, 256 experts top-8 + shared expert |

## Objectives

1. **Maximum decode and prefill throughput with no loss of intelligence.** No speculative
   decoding (the MTP/nextn block is loaded by neither path). Quality is measured, not
   assumed: KL divergence and top-1 agreement of Ferrum logits against llama.cpp
   running the same GGUF, over a fixed text corpus.
2. **Stable and predictable up to high context.** Memory is fixed at load: KV storage,
   recurrent state and scratch are allocated once for the fitted context. Prefill is
   chunked with flash-style attention, so no activation scales with the square of the
   sequence length.
3. **Automatic context fitting with accurate predictions.** The runtime computes weight,
   KV, recurrent-state and scratch bytes from GGUF metadata before loading anything,
   picks the largest context that fits the working-set budget, and reports the
   prediction; measured allocations are checked against it.
4. **Agentic and chat workloads.** Multi-turn chat and tool loops re-send long, mostly
   unchanged prompts, so the runtime keeps the conversation state and prefills only
   the new suffix. Hybrid (recurrent) layers cannot be truncated, so the cache keeps
   checkpoints of recurrent state to resume from a shared prefix.

## Architecture notes (from GGUF metadata and llama.cpp 687e77892)

- Every fourth layer (`(i+1) % 4 == 0`) is gated full attention: the Q projection emits
  query and a per-head output gate interleaved per head (`[q_h | g_h]`, 2×256 each); Q/K
  RMS norm per head; partial NEOX RoPE on the first 64 of 256 dims (M-RoPE sections
  11/11/10 collapse to plain RoPE for text); θ = 1e7; attention output × sigmoid(gate).
- The other layers are Gated DeltaNet: `attn_qkv` → causal depthwise conv (kernel 4) →
  SiLU → split Q/K (16 heads × 128) and V (48 or 32 heads × 128); L2-norm Q and K;
  β = sigmoid(`ssm_beta` x); g = softplus(`ssm_alpha` x + dt_bias) · `ssm_a`; per head,
  S ← S·exp(g); d = β(v − Sᵀk); S ← S + k dᵀ; o = Sᵀ(q/√128). V head h uses K head
  h mod 16. Output: RMSNorm(o)·SiLU(z) with z = `attn_gate` x, then `ssm_out`.
- Residual: `x += op(norm(x)); x += ffn(post_norm(x))`. GGUF norm weights already
  include the +1 offset, so they are plain RMSNorm.
- Tokenizer: GPT-2 BPE with the `qwen35` pre-tokenizer regex (`\p{L}\p{M}` classes).

## Work order

1. IQ4_XS (and any other type the target files use) GEMV / GEMM / expert kernels.
2. `qwen35` dense model: loader, hybrid state cache, prefill and decode, KLD harness.
3. `qwen35moe`.
4. Throughput: fused decode kernels, chunked GDN prefill, flash attention.
5. Context planner and preallocated cache; long-context validation.
6. Session state reuse, chat templates, and sampling (temperature, top-k, top-p,
   min-p, presence/frequency/repetition penalties) for normal chatting.
7. Two binaries: `ferrum-cli` (interactive chat in the terminal) and
   `ferrum-server` (OpenAI-compatible local HTTP server for agents).

Progress, measurements and decisions are recorded in the
[Phase 8 journal](phase8-journal.md).
