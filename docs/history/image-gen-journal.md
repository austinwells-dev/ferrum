# Image generation journal — Qwen-Image-2.1

> **History.** This work is in progress, so the file is still being written; earlier entries are not revised. The current docs start at [docs/README.md](../README.md).

Goal: text-to-image on the hybrid engine's Metal backend, for the unquantized (BF16) and quantized (GGUF, MLX 4-bit) checkpoints of `Qwen/Qwen-Image-2.1` and its 8-step distillation `Qwen/Qwen-Image-2.1-Turbo`, correctness first, on a 32 GB Mac. Reference: diffusers (`tools/imgref`). Machine for every entry: Apple M5, 32 GB.

## 2026-10-26 — what the model is

Read from diffusers' `QwenImage21Transformer2DModel`, `AutoencoderKLQwenImage21` and `QwenImage21Pipeline` (0.42.0.dev0).

- **Transformer.** One stream of text and image tokens through 32 identical blocks, width 4096 (32 heads of 128), no biases anywhere. A block is `LayerNorm → scale → attention (per-head RMSNorm on Q and K, complex-pair RoPE) → tanh-gated residual → LayerNorm → scale → SwiGLU (12,288) → tanh-gated residual`. All blocks read their scales and gates from one shared `modulation` projection of the timestep embedding, so blocks hold no modulation weights.
- **Attention is block-causal.** Text is causal; the image block is bidirectional and sees all the text. Text tokens are modulated from `t = 0` instead of the sampled timestep, so their activations, and the keys and values they produce in every layer, don't depend on the step. The pipeline caches them after step 0. Ferrum computes the text prefix once up front and then runs only the image tokens each step, which is the same arithmetic as the pipeline's step 0.
- **Latents.** The VAE compresses 16× per side into 64 channels, and the transformer takes them unpatched: one token per 16×16 pixel tile (1,024 tokens at 512², 4,096 at 1024²).
- **Text encoder.** Qwen3-VL-8B. The transformer reads the output of the last decoder layer *before* the encoder's final norm, for the tokens after the system prompt (`drop_idx` = 14). transformers 5.x returns the normalised state at that index, so the pipeline neutralises the norm with a forward hook; the harness gets this by calling the pipeline's own `encode_prompt`.
- **Sampling.** Plain Euler on the checkpoint's sigmas. Turbo ships its 8 sigmas in `model_index.json` and runs at CFG 1; the base model uses 40 steps with a dynamic exponential shift, also at CFG 1 in the model card's examples. The VAE output has four channels (RGBA).

Sizes: transformer 7.1B parameters (14.2 GB BF16; unsloth's GGUFs are 4.2 GB at Q4_K_M and 7.6 GB at Q8_0), text encoder 17.5 GB with its vision tower, VAE 0.7 GB. The encoder and transformer cannot be resident together on 32 GB, so the pipeline runs them in sequence.

## 2026-10-26 — reference harness

`tools/imgref/dump_reference.py` runs the three stages one after another under diffusers (bfloat16, MPS) and saves everything Ferrum is compared against: prompt embeddings, the seeded noise (handed to the pipeline as `latents`, so Ferrum doesn't have to reproduce torch's RNG), the rotary tables, the shared modulation, block outputs at step 0 (prefill) and step 1 (cached), each step's prediction and latents, the VAE input and the image. The full 512² Turbo run (20 prompt tokens, 1,024 image tokens, 8 steps) takes about three minutes, 2–4 s per step.

The image for `A red fox sitting in fresh snow, golden hour light.`, seed 42, is a plausible fox, which shows the harness reproduces the pipeline.

## 2026-10-26 — the three stages match diffusers

Each stage is checked against the harness dumps by `#[ignore]`d tests in `src/diffusion/reference_tests.rs`; the raw output is in [parity-2026-10-26.txt](../measurements/image-gen/parity-2026-10-26.txt).

- **A bf16 reference is a loose ruler.** The first comparison, against the bf16 dump, showed the transformer's final latents 6.5% off (relative RMS) and the prompt embeddings 5.7% off. Running the same harness in fp16 and comparing the two dumps with each other gives 6.2% on the final latents and 4.8% on the embeddings, so those errors are the reference's own rounding, not ours. Against the fp16 dump: prompt embeddings 1.0%, transformer blocks 0.07–0.7%, final latents 0.9%, decoded image 0.065% (72.8 dB PSNR).
- **Transformer.** 14.2 GB loads in about 2.5 s from the page cache; 8 steps at 512² took 14 s (about 1.75 s a step, text prefix included once). Not tuned yet.
- **Text encoder.** The hybrid engine runs Qwen3-VL's language model as an attention-only model (the Flint geometry: head width 128, GQA, no recurrent layers), and the last layer's output is read from the speculative-decoding feature capture, which sits before the final norm. 34 prompt tokens take 0.2–0.6 s. Q8_0 layers halve it to 8.6 GB at 2.7% error against fp16.
- **Decoder.** Convolutions are implicit GEMMs on TensorOps with the zero padding, 2× nearest upsampling and residuals fused. TensorOps gave wrong results for a GEMM operand with a non-unit inner stride (attention's V), so V goes through a small transpose first. 512² decodes in 0.7 s.

## 2026-10-26 — pipeline, apps, speed

- The staged pipeline, `ferrum-cli image`, the server's `/v1/images/generations` and the TUI Image tab are in. Timings: [retime-2026-10-26.txt](../measurements/image-gen/retime-2026-10-26.txt) (first taken with the display off and redone with it on).
- Every hybrid kernel's first dispatch compiles the whole concatenated source, so the image kernels would have slowed LLM startup. They now join the source only when an `i_*` or `vae_*` kernel is requested; the decoder kernels were renamed from `v_*` because the vision kernels already use that prefix.
- Two TUI tests that scan the real home directory block in `opendir` on this machine; they pass with an empty `HOME`.

## 2026-10-26 — GGUF transformers

- The unsloth GGUFs have no metadata at all (no `general.architecture`, no quantization version) and tensor names past the spec's 64 bytes, so `GgufFile::open_untagged` skips those checks; the LLM readers still require them. Names are the diffusers ones under `model.diffusion_model.`, each block's gate and up matrices are stacked in one `gate_up` tensor (gate first), and a Q8_0 file holds Q8_0 matrices with BF16 `img_in` and text projections and F32 norms.
- The GGUF is the base model, not Turbo. Its tensors differ from Turbo's by about 0.6% relative RMS (Q8_0 rounding alone is about 0.1%), so the Turbo diffusers dump cannot be the reference; the mapping is checked by pairing tensors with Turbo's (the right gate/up order shows 0.6%, the wrong one 140%), and the base runs through the 40-step dynamic-shift schedule without guidance and gives clean pictures. Record: [gguf-2026-10-26.txt](../measurements/image-gen/gguf-2026-10-26.txt).
- 512² at 40 steps: Q8_0 69 s, Q4_K_M 72 s (about 1.6 s a step, so the Q4_K kernels are not faster than Q8_0 here: denoising is compute-bound). Transformer memory 7.6 GB and 4.3 GB.

## 2026-10-26 — F16 activations and device-direct attention

- A per-kernel profile of one 1024² step ([gemm-2026-10-26.txt](../measurements/image-gen/gemm-2026-10-26.txt)) put 70% of the 8.2 s in the projections (10 TFLOPS) and 25% in attention (4.4 TFLOPS).
- The hybrid engine's GEMM reads F32 activations. The transformer now keeps what a matrix product reads (layernorm, attention and SwiGLU outputs) in F16 and uses `i_mm_*`, the same tile with half-by-half products and a 64-wide K step: BF16 10.4 → 14.2 TFLOPS, Q8_0 12.6 → 14.6, Q4_0 13.2 → 15.1. A sweep of tile shapes with both operands read directly peaked at 15.6, so the projections are near what the matrix units give here.
- Attention now reads the cached text K/V and the current rows' K/V straight from device memory, one segment at a time, instead of staging 64-key blocks through threadgroup memory: 2.0 → 1.7 s. It is still at about 5 TFLOPS; the scalar softmax and the score round trip through threadgroup memory are the likely cost, and the 32 KB threadgroup limit rules out wider key blocks with F32 scores.
- Re-timed with the display on ([retime-2026-10-26.txt](../measurements/image-gen/retime-2026-10-26.txt), A-B-B-A): 512² 25.1 → 20.0 s and 1024² 82.8 → 68.0 s end to end (about 20% faster); the profiled step is 8.0 s on a warm machine. Parity with the fp16 dump is unchanged (final latents 0.87%, image 72.8 dB).


## 2026-10-26 — text encoder from a GGUF, and a timing mishap

- `--text-file` / `--image-text-file` take the encoder's language model from a Qwen3-VL GGUF (the hybrid engine's own GGUF loader, minus the LM head). With Unsloth's UD-Q4_K_XL (4.6 GB) the encoder stage needs 4.4 GB instead of 14.5 GB and loads in under a second. Its prompt embeddings are 26.5% from the fp16 dump (Q8_0 on-load 2.7%, Q4_0 on-load 33.4%), but the pictures stay clean (34.3 dB from the Q8_0-encoder picture), so the embedding error is mostly directions the transformer ignores. Record: [gguf-2026-10-26.txt](../measurements/image-gen/gguf-2026-10-26.txt).
- With a Q4_K_M transformer and the Q4_K_XL encoder every stage peaks under 5 GB (1024²: 4.4, 5.0 and 1.9 GB), so a 32 GB Mac could keep them all resident; Ferrum still frees between stages.
- The power log showed the display off from 12:18 to 14:21, covering the first set of timings, so all timings were retaken with it on. They came out about 25% slower on the warm machine; the before/after ratios held.
