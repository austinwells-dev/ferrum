# Inkling 2 Mini: port assessment

2026-11-20. An investigation only: no Ferrum code was written for it. Model facts come from Thinking Machines' announcement and model card, the Hugging Face `config.json`, `chat_template.jinja` and `tokenizer_config.json`, and llama.cpp master `29b52d3` (the merge of PR #32694: `src/models/inkling2.cpp`, `src/models/inkling2-loop.cpp`, `conversion/inkling2.py`). Ferrum facts come from `src/hybrid` on `perf/skip-cached-image-encode` at `ada39ff`. The estimates are judgement, not measurements.

## Verdict

**Large: about the size of the Qwen3.8-Flash-Next port, and harder to validate.** Ferrum already has sigmoid MoE routing, sliding-window K/V rings, sinks, output gates, short-conv state with rewind (LFM3) and a separate low-rank-gated residual path (qwen4exp `hc_layers`). The attention math, the attention-residual mixing and the loop itself are new.

| Scope | Estimate |
|---|---|
| Text only, one quant, KLD-checked | 5–7k lines, about 1.5–3 weeks |
| Plus speculation (lookup chains, DSpark drafter) | +1–1.5k lines, 3–5 days |
| Plus vision (`inkling2_vision` tower) | +1.5–2k lines, about a week |

Precedents by commit size: Flint 4.7k lines (`b23dad2`, `9842b19`, 2026-10-22); LFM3 with LFM3-VL 3.1k lines (`8b999f9`, 2026-10-28); Qwen3.8 1.1k + 3.2k lines (`a233bef`, `15a3418`, 2026-11-02 and 03).

## The model

- 26.4B total, 3.4B active per pass. 5.4B is the effective count with looping: each extra pass through the looped layers adds about 1.0B (model card).
- 24 stored layers. Layers 0–5 run once, 6–17 three times, 18–23 once: **48 visits per token**. Fixed 3 passes (`loop_config.is_varloop = false`); `TML-d24-varloop-research` takes 1–5.
- Hidden 2304, 24 Q heads and 8 KV heads of 128. Each six-layer block has five sliding-window layers (window 1024) and one full-attention layer (5, 11, 17, 23). **No RoPE.**
- MoE: 128 experts, top 8, no shared expert. Sigmoid router with a selection-only bias (`ffn_exp_probs_b`), weights scaled by 8 (`expert_weights_scale`).
- Vocabulary 200,058 in use (201,024 padded). Trained context 1M. GGUF `general.architecture = inkling2_loop`, `block_count = 24`; llama.cpp expands it to 48 slots.
- Upstream weights are NVFP4 (MXFP8 in the last 4 layers); the converter dequantizes them, so GGUFs hold ordinary types. Apache 2.0.
- A 3-layer DSpark drafter (`mtp.safetensors`, `markov_head_type = rnn_hidden`, target visits 5, 23, 41, 48). Unsloth ships `dspark-Inkling-2-Mini-Q8_0.gguf` (402 MB).

## What Ferrum already covers

| Model needs | Ferrum today |
|---|---|
| Head width 128, GQA, sliding windows with rings | Flint path: `AttnSpec.window`, `ring_rows`, snapshot rewind |
| Sigmoid routing | Flint, but it refuses a selection bias and a weight scale other than 1.0 (`src/hybrid/config.rs`, the Flint `MoeConfig` arm). Both need implementing; small. |
| Output gate | Spark's per-head sigmoid gate is close |
| Short-conv state and its verify tape | LFM3 `ShortConv` |
| Low-rank-gated residual mixing | qwen4exp `hc_layers`: precedent for a separate layer loop |
| Quants | Q4_K, Q5_K, Q6_K, Q8_0, IQ4_NL, IQ4_XS, BF16 covered. Q2_K, IQ1_S, IQ1_M, IQ2_XXS are not, so Unsloth's IQ1 and Q2 builds would be refused. |
| Byte-level BPE tokenizer | Covered; needs an `inkling2` pre-tokenizer pattern (`tokenizer()` in `src/hybrid/mod.rs`) |

## What is new, largest first

1. **Attention kernel features (riskiest; shared kernels).**
   - Position bias instead of RoPE: per query and head, `attn_rel` projects to 16 values, L2-clipped at 8.0. A learned table `attn_rel_bias` of `{16, 1024}` plus a padding row turns them into an additive bias by distance. Best computed inside flash attention as a 16-wide dot product, not stored.
   - Sinks are per token (`attn_sink_proj · x`, subtracted from the bias). Ferrum's are one constant per head.
   - Q and K are L2-clipped at 22.63, with a per-channel scale on Q (`attn_q_clip`).
   - Full-attention layers only: Q and the position vector scaled by `1 + 0.1·log(max(1, (pos+1)/8192))`; K and V pass through a static depthwise conv (kernel 4) plus a dynamic conv whose 4 taps per KV head come from `attn_{k,v}_conv_proj`, summed with the input and divided by 3 (sliding layers: static only, divided by √5); separate Q/K/V input norms.
   - V is RMS-normed with no gamma.
   - Not settled from what was read: whether the 1/√128 scale applies to the bias, and how llama.cpp's `build_inp_mem_hybrid_iswa_rel` maps distance to a table row.
2. **Loop and per-visit state.** The layer loop in `src/hybrid/engine.rs`, the memory planner (`plan.rs`), sessions and snapshots assume one state per layer. Here 48 visits share 24 sets of weights but each has its own K/V (8 full-attention slots, 40 sliding) and four short-conv states (K, V, after attention, after the FFN: 6,656 channels, 3 rows each). Wide but mostly bookkeeping.
3. **Attention residuals.** No single running residual. The stream is pushed as a block at physical layers 0, 6, 12, 18 (where the attention output replaces rather than adds to it). Each layer's attention input, its FFN input and the final output mix up to 5 RMS-normed sources with a softmax (scale 1/48) against learned queries (`attn_res_query`, three on full-attention layers; `ffn_res_query`; `output_res_query`). How block pushes interact with repeated visits needs checking against the reference.
4. **Loop messages.** On passes 2 and 3 each core layer mixes 4 messages (`loop_msg_query`, from the previous pass's block input and core output). Four rank-64 controllers apply `z = W_down·[rms(x); rms(m)]`, `x' = x − tanh(W_forget·z)·x + tanh(W_gain·z)·m` to the Q, K, V and FFN inputs. Small kernels, many dispatches. The converter folds per-pass query rows into one tensor; not confirmed to match the reference.
5. **Smaller pieces.** SITU activation `4·tanh(g/4)·σ(g) · 25·tanh(u/25)`; logit softcap 30 and truncation to 200,058 rows; static convs after attention and the FFN.
6. **Chat format.** Role tokens (`<|message_model|>`, `<|content_thinking|>`, `<|content_text|>`, `<|end_message|>`); effort as a system line `Thinking effort level: 0.9` (none 0, minimal 0.1, low 0.2, medium 0.7, high 0.9, xhigh/max 0.99); tools declared as `<|message_system|>tool_declare<|content_xml|>[…]`; calls as `<|message_model|>NAME<|content_invoke_tool_json|>{"name":…,"args":{…}}<|end_message|>`; stop on `<|content_model_end_sampling|>` (200006). Needs `chat.rs` and the server's tool parser.
7. **Optional.** DSpark: `src/hybrid/draft.rs` accepts only the `vanilla` Markov head, and target layers are visit indices, so it depends on item 2. Draft trees would be refused as for LFM3 (conv windows); lookup chains should work.

## Memory and speed on a 32 GB Mac (arithmetic only)

- UD-Q4_K_M (16.6 GB) or ggml-org IQ4_NL fits; the Q8 files (about 23 GB or more) leave no useful room.
- K/V at F16: 8 full-attention slots × 4 KiB = 32 KiB per token, about 4 GiB at 128k. The 40 sliding rings are a fixed ~170 MB.
- Each token reads about 5.4B active parameters' worth of weights and runs twice the layer dispatches, plus mixing and controller dispatches. Expect decode to be dispatch-bound until small ops are fused.

## Risks

- **The reference is hours old.** llama.cpp support merged 2026-11-20; Unsloth's draft #32695 is still open; the converter's per-pass query fold is unconfirmed. Wait a few days before trusting KLD against it.
- **Unsloth file metadata.** Their page shows `inkling2`, the merged code expects `inkling2_loop`, and a third party reported their files failing on a pre-merge build. ggml-org's IQ4_NL file is the safer first target (Ferrum has IQ4_NL kernels).
- **Homebrew llama.cpp is too old**; the KLD base needs master at or after `29b52d3`.
- **Regression rules.** The attention kernel change is a large change under [AGENTS.md](../../AGENTS.md): interleaved A-B-B-A bench and KLD on one existing dense and one MoE model, beyond Inkling's own checks.

## Suggested order

1. Config, weights, tokenizer and chat template; forward pass against a CPU reference.
2. New attention features and conv ops, each with a parity test.
3. Per-visit state, loop and controllers; greedy output and KLD against llama.cpp.
4. Memory planner and sessions.
5. Lookup drafting.
6. DSpark `rnn_hidden`.
7. Vision.

## 2026-11-21: what was built

Branch `feat/inkling2-mini`, from `experimental` at `0c20d64`. Steps 1–4 and 7 of the order above; 5 and 6 (speculation) were not started, and speculation is refused at load.

- **Target file.** Unsloth's `UD-Q4_K_XL` (architecture `inkling2`, Q8_0 attention, NVFP4 experts), not ggml-org's IQ4_NL. The reference became Unsloth's own llama.cpp build `b12568-mix-8eebad4`; master garbled these files. NVFP4 was new to Ferrum: a dequantizer shared by the generic GEMV, the expert GEMV and the GEMMs.
- **Kernels.** `inkling.metal` (depth mix, controllers, conv, prep, routing, SITU, softcap) and `_rel` variants of the 128-wide attention kernels: a relative-position bias and per-row sinks. Parity tests in `inkling_tests.rs`; NVFP4 joined the format tests in `iq_tests.rs`.
- **Surprises.** The converted query layout groups each layer's per-pass queries, so the loader regroups them per core layer. The bias extent is per layer: the sliding window on SWA layers. The reference's CPU backend disagrees with its own Metal backend (perplexity 5.76 against 5.10), so KLD is taken against Metal. The chat template has no generation prompt; forcing a `<|content_text|>` opening for thinking-off kept the model from calling tools, and llama.cpp forces nothing, so neither does Ferrum.
- **Vision.** The `inkling2v` tower took far less than the estimated week: the ViT pieces were already there. Two differences from Qwen's tower mattered: the position table resizes with half-pixel centres, and the rotary pairs are adjacent (handled by permuting Q and K rows at load). A JPEG decoder difference first looked like a tower bug, until a PNG gave cosine 0.99999.

Results and gaps: [models.md](../models.md#inkling-2-mini), raw logs in [measurements/inkling2](../measurements/inkling2/README.md).

## Sources

- https://thinkingmachines.ai/news/inkling-2-mini/
- https://thinkingmachines.ai/model-card/inkling-2-mini/
- https://huggingface.co/thinkingmachines/Inkling-2-Mini
- https://huggingface.co/blog/inkling2mini
- https://huggingface.co/unsloth/Inkling-2-Mini-GGUF
- https://github.com/ggml-org/llama.cpp/pull/32694
- https://github.com/ggml-org/llama.cpp/pull/32217
