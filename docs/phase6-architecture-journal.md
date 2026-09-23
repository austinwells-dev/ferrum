# Phase 6 architecture journal

## Starting checkpoint (2026-09-23)

- Repository root: `/Users/austinwells/Documents/ChatGPT/ferrum`.
- Branch: `codex/phase6-modern-architectures`, created from clean Phase 5.5 checkpoint `e12964cf136262e35c4237415b111f13c93cce51`.
- The retained Phase 5.5 changes are the two committed revisions `e875e68` and `e12964c`; the rejected Q5 row-sharing experiment was removed before the checkpoint. Phase 5.5 is explicitly performance-incomplete. See `docs/phase5.5-experiments.md`.
- Host: Apple M5, macOS 27, 32 GiB unified memory; Rust 1.96.0. The Hugging Face cache contains the prior official Qwen2.5 checkpoint and an official LiquidAI LFM2.5-350M snapshot, among other unrelated artifacts. No model cache path is part of Ferrum's runtime contract.
- `cargo fmt --all -- --check` and `cargo check --all-targets` passed. Library and integration tests passed under `MTL_DEBUG_LAYER=1 MTL_SHADER_VALIDATION=1`; `cargo test --all-targets -- --test-threads=1` reached Criterion's bench harness, which rejected the test flag, so subsequent validation uses separate test and bench invocations. Both ignored official Qwen2.5 real-model tests passed with Metal API and shader validation at the discovered local snapshot.

## Source and checkpoint discovery

| Role | Official source and observed revision | Relevant evidence | Status |
| --- | --- | --- | --- |
| Existing regression anchor | [Qwen/Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct), `7ae557604adf67be50417f59c2c2f167def9a775` | BF16 safetensors SHA-256 `fdf756fa7fcbe7404d5c60e26bff1a0c8b8aa1f72ced49e7dd0210fe288fb7fe`; Phase 5 records quantized revisions and hashes in `docs/measurements/phase5/model-source.json`. | Baseline validation passed. |
| First modern dense bring-up | [Qwen/Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B), `c1899de289a04d12100db370d81485cdf75e47ca` | BF16 safetensors SHA-256 `f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b`. Config declares `qwen3`, 28 layers, hidden 1024, 16 Q heads of width 128, 8 KV heads, 3072 MLP width, RMS epsilon 1e-6, RoPE theta 1e6, no attention bias, tied embeddings. Safetensors has 311 entries, including per-head `q_norm`/`k_norm` and an `lm_head.weight` byte identical to `model.embed_tokens.weight`. | BF16 and Q8_0 GGUF validated against reference. |
| Qwen shape follow-up | [Qwen/Qwen3-1.7B](https://huggingface.co/Qwen/Qwen3-1.7B), `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e` | Official config reports hidden 2048, 16 Q heads of width 128, 8 KV heads, 6144 MLP width, 28 layers. The checkpoint uses two safetensors shards; indexed payload total 4,063,479,808 B. | Validated against Transformers on eight decode steps. |
| Gemma dense candidate | [google/gemma-3-270m](https://huggingface.co/google/gemma-3-270m), `9b0cfec892e2bc2afd938c98eabe4e4a7b1e0ca1` | Small text-only checkpoint; repository access is gated by Google's license. | Access and architecture inspection pending. |
| Newer Gemma evaluation | [google/gemma-4-E2B](https://huggingface.co/google/gemma-4-E2B/blob/main/config.json) | Current official Gemma 4 text decoder adds local/global attention, different head dimensions, partial/proportional RoPE, softcapping, and other policies; the full checkpoint also contains audio/vision components. | Evaluate after smaller dense families; no support claim. |
| MoE candidate | [ibm-granite/granite-3.1-1b-a400m-instruct](https://huggingface.co/ibm-granite/granite-3.1-1b-a400m-instruct/blob/main/config.json), `0da7a48b0276d500ce5922fd2b33944091fc6c09` | Official 1B total / 400M active model; 32 experts with 8 selected per token. Config adds embedding, residual, and logit multipliers. | Research candidate, not selected conclusively. |
| Newer MoE evaluation | [ibm-granite/granite-swash-3b-a600m](https://huggingface.co/ibm-granite/granite-swash-3b-a600m/blob/main/config.json) | Current 3B/600M model adds sliding attention and a shared expert; useful later composition target. | Research candidate. |
| Hybrid candidate | [LiquidAI/LFM2.5-350M](https://huggingface.co/LiquidAI/LFM2.5-350M/blob/main/config.json), `9e6c6ccf47cd318696e137d381a7ded8fe4df09f` | Local official snapshot includes weights, config, tokenizer, generation metadata, and license. Config schedules convolutional state and full-attention layers. | Available for later state bring-up. |
| Hybrid + MoE candidate | [LiquidAI/LFM2.5-8B-A1B](https://huggingface.co/LiquidAI/LFM2.5-8B-A1B) | Current official 8B/1B family; feasibility and exact equations still need inspection. | Possible integration target. |
| Qwen3 quantized checkpoint | [Qwen/Qwen3-0.6B-GGUF](https://huggingface.co/Qwen/Qwen3-0.6B-GGUF/tree/main), `23749fefcc72300e3a2ad315e1317431b06b590a` | Official `Qwen3-0.6B-Q8_0.gguf` SHA-256 `9465e63a22add5354d9bb4b99e90117043c7124007664907259bd16d043bb031`; GGUF v3, 310 tensors, 197 Q8_0 matrices and 113 F32 tensors. | Validated against the BF16 model and reference IDs below. |

Official [Qwen model listings](https://huggingface.co/Qwen/models) currently show Qwen3.8 releases. [Transformers Qwen4-Exp documentation](https://huggingface.co/docs/transformers/en/model_doc/qwen4_exp) exists, but no practical official Qwen4 weight checkpoint has been established from the official listing. Ferrum will not claim Qwen4 support from a class definition or preview.

## Initial design evidence

Ferrum already shares embedding, linear, SwiGLU, RMSNorm, RoPE, attention, KV storage, generation, sampling, and packed quantized math. `src/model/qwen.rs`, `qwen_gguf.rs`, and `qwen_mlx.rs` are strict Qwen2-specific adapters. The current `ModelConfig` captures common dimensions, while `DecoderLayer` and `Attention` assume one pre-attention RMSNorm, one post-attention RMSNorm, full attention, and no per-head Q/K normalization. The first policy extension will cover observed Qwen2/Qwen3 differences only; the existing Qwen2 contract must remain the default for current tests and format adapters.

The official Transformers 4.57.6 `modeling_qwen3.py` installed in `.venv-reference` applies `q_norm` and `k_norm` after projection and reshape to `[batch, sequence, heads, head_dim]`, before RoPE. The current Ferrum RMSNorm kernel reduces over the last dimension and can implement that operation without new model-specific math.

## Dense milestone 1: shared Q/K policy and Qwen3

The first policy adds per-head Q/K RMS normalization and a Q/K/V bias requirement, selected at construction. The existing transformer, attention kernels, Metal device, KV cache, SwiGLU, generation, and sampler remain shared. Qwen2 retains its previous default execution path; the Qwen3 adapter validates config and maps official names into canonical weights. The official Qwen3 safetensors checkpoints serialize a duplicate tied LM head; the loader validates byte equality and retains only one copy in the model. A general indexed-safetensors directory loader checks every shard assignment, tensor name, total byte count, and shard filename.

The reference is local Transformers 4.57.6 / PyTorch 2.8.0 on CPU, using the same official checkpoint and exact prompt IDs. `tools/reference_causal.py` and the four JSON files under `docs/measurements/phase6/` retain reference IDs and top-ten logits for the 0.6B and 1.7B Hello prompt plus 2-token and 121-token 0.6B prefill cases. Ferrum matched all eight greedy IDs for each case. Maximum absolute difference over reference top-ten logits was 0.1875 (2-token prefill), 0.5 (21-token 0.6B), 0.3125 (121-token prefill), and 0.6875 (21-token 1.7B). The test bound is 1.0 for these BF16 comparisons. Snapshot branches reproduced byte-identical logits; reset and re-prefill reproduced the first greedy token. The real-model tests ran with Metal API and GPU shader validation enabled.

The Qwen3 GGUF adapter reuses the same policy and native packed Q8_0 kernels; it reads `qwen3.*` metadata and maps the added Q/K norms. Official Q8_0 and BF16 produced identical eight-token greedy IDs on the 21-token prompt. Minimum full-vocabulary Q8/BF16 logit cosine across those eight steps was 0.998014. The Q8 path retained 633,364,480 weight bytes, including 633,233,408 packed quantized bytes, versus 1,192,099,840 bytes for the 0.6B BF16 model. Its 310 GGUF tensors correspond to the unique logical model parameters.

Single cold Metal-validation diagnostic runs on the 21-token prompt recorded the following. These are useful baseline magnitudes, not warmed or paired performance claims:

| Checkpoint | Config/tokenizer load | Weight load | Construction | Retained weights | Prefill | First token | Median cached decode | Active KV after 8 tokens |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen3-0.6B BF16 | 118 ms | 323 ms | 9 ms | 1,192,099,840 B | 39.8 tok/s | 527 ms | 33.7 tok/s | 3,211,264 B |
| Qwen3-1.7B BF16 | 103 ms | 1,409 ms | 17 ms | 3,441,149,952 B | 45.2 tok/s | 465 ms | 21.1 tok/s | 3,211,264 B |
| Qwen3-0.6B Q8_0 GGUF | 584 ms total GGUF load | included | included | 633,364,480 B | 160 tok/s | 131 ms | 18.7 tok/s | 3,211,264 B |

Gemma 3 270M and Meta Llama 3.2 1B official checkpoints are gated; the configured Hugging Face credentials received access-denied responses. The phase will use available first-party checkpoints exercising different dense policies, rather than treating third-party copies as official weights. IBM Granite 4.0 dense and AllenAI OLMo 2 are under evaluation as substitutes. The current official Gemma 4 E2B architecture is substantially more complex and larger; it remains a later optional target rather than the small dense bring-up checkpoint.

The Qwen3 0.6B GGUF and safetensors checkpoints did not require any new quantized math. A dedicated MLX Qwen3 format path has not been validated yet. No MoE or persistent non-KV state support is claimed by this milestone.
