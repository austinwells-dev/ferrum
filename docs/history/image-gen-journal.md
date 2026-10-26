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
