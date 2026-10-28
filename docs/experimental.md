# Experimental: Flint, Spark, LFM3, MLX checkpoints and KFlash

> **Experimental.** This page covers Flint, Spark and LFM3 support, MLX 4-bit checkpoints, and KFlash speculative decoding (ported from Kapitsa). None of it is production-ready. Expect rough edges, and expect the behaviour, the flags and the parsing to change.

Contents: [Flint](#what-works) · [Spark-X2.6-4B](#spark-x26-4b) · [LFM3-1.7B](#lfm3-17b) · [LFM3-VL-2B](#lfm3-vl-2b) · [MLX checkpoints](#mlx-4-bit-checkpoints) · [KFlash](#kflash-speculative-decoding)

## What works

- **Loading and generation** in `ferrum-cli hybrid`, `ferrum-server` and `ferrum-tui`, from `unsloth/flint-GGUF`, file `flint-UD-Q3_K_XL.gguf` (18.5 GB).
- **Tool calls.** Flint's `<tool_call>NAME<arg_key>…</arg_key><arg_value>…</arg_value></tool_call>` blocks are parsed into OpenAI `tool_calls`. Argument values are typed by the tool's JSON schema: a `string` parameter keeps its text, anything else is read as JSON. Malformed blocks, and blocks whose name is not a declared tool, stay visible as plain text.
- **Chat template.** The GGUF's Jinja template renders identically to Jinja2 (the reference used by Hugging Face) on the cases in `tests/fixtures/flint/render_cases.json`. Regenerate them with `tests/fixtures/flint/render_cases.py`.
- **Stop tokens.** Generation ends at Flint's end-of-generation ids from the GGUF: `<|end_of_text|>`, `<|user|>`, `<|observation|>` and `<|developer|>`.

## Validation

Measured on a 32 GB M-series Mac against llama.cpp built from `e3797b1` (after Flint merged).

| Check | Result |
|---|---|
| Tokenizer ids vs `llama-tokenize` (two mixed-script samples) | identical |
| Greedy continuation, "The capital of France is" | first three tokens match llama.cpp; token 4 differs (`" \n"` vs `" is"`), likely a near tie |
| KLD vs llama.cpp Metal, 5,115 positions | mean 0.052, median 0.0049, top-1 93.5% |
| llama.cpp CPU vs its own Metal (same file) | mean 0.074, median 0.011, top-1 91.2% |
| Speed (`hybrid_bench`) | prefill 883 tok/s (pp512), 899 tok/s (pp2048); decode 52 tok/s (tg128) |
| Qwen3.5 regression (dense 27B and MoE 35B) | identical generated tokens and per-tensor sums before and after |

Ferrum sits closer to llama.cpp's Metal output than llama.cpp's own CPU backend does, so the KLD gap looks like quantization sensitivity rather than a bug. It is still one quant on one text.

## Known gaps

- One quant (`UD-Q3_K_XL`) is validated. Other Flint files may use quant types the kernels do not cover; the TUI lists the supported set when it refuses a file.
- Flint's template has no thinking switch, so the prompt itself ends in `<think>` (thinking on) or `<think></think>` (off), written the way the template writes past reasoning so later turns reuse the cached prefix. A budget-cut reasoning closes with a bare `</think>`.
- Sliding-window layers keep their K/V in a ring of 1,088 rows (window 513 plus one prompt chunk), so only the 14 full-attention layers grow with the context: 28 KiB per token plus 76 MiB of rings, against 98 KiB per token before. A session can rewind only while the rings still hold the rewind point's window; further back it recomputes from the start.
- The legacy attention kernels (`FERRUM_HYBRID_LEGACY_ATTENTION`, `FERRUM_HYBRID_SIMD_ATTENTION`) reject Flint's and Spark's geometry.
- No trained drafter (MTP, DFlash) exists for Flint. Prompt lookup (`--draft lookup`) works (1.12× on the `hybrid_spec` chat prompts) but is less exact than on the Qwen3.5 models: verifying 4 or more rows goes through F16 TensorOps tiles, and Flint's expert routing amplifies that rounding, so 4 of 8 greedy prompts drifted from plain decoding (one at a top-2 margin of 0.32). With `FERRUM_HYBRID_EXACT=1` only one near-tie flip (margin 0.035) remains, so this is rounding, not a verify error. The TUI's speed benchmark marks such rows `≈` and checks their answers before recommending them.
- Expert routing implements neither a selection bias on the scores (`exp_probs_b.bias`) nor an `expert_weights_scale` other than 1.0; this GGUF has neither, and a file that does is refused at load.
- Chat `bos_token` and `eos_token` strings come from the vocabulary, but Flint's template does not use them.

## Running it

```sh
hf download unsloth/flint-GGUF flint-UD-Q3_K_XL.gguf --local-dir ~/models
cargo run --release --bin ferrum-tui
```

Pick the model in the Models tab. It shows as runnable once the quant check passes.

# Spark-X2.6-4B

XHToken's `spark2_5` architecture: a 4.4B dense model with attention on all 36 layers (three 512-token sliding-window layers to each global one), fused QKV, a per-head sigmoid output gate, no QK norm, a GEGLU feed-forward, and separate RoPE on sliding-window layers (256 dims at a 10k base, against 64 dims at 5M on global layers). It runs on the Flint path of the hybrid engine (`Variant::Spark`), with the same K/V rings.

## Validation

`XHToken/Spark-X2.6-4B-GGUF`, file `Spark-X2.6-4B-Q4_K_M.gguf` (3.5 GB), against llama.cpp build 11669 on a 32 GB M-series Mac.

| Check | Result |
|---|---|
| Tokenizer ids vs `llama-tokenize` (mixed-script stress file, README, architecture.md; 13.4K tokens) | identical |
| Greedy continuation, "The capital of France is" | identical to `llama-simple` |
| Chat template vs Jinja2 (`tests/fixtures/spark/render_cases.json`: tools, tool history, thinking on/off, low effort) | identical |
| KLD vs llama.cpp Metal, 5,115 positions | mean 0.0012, median 0.00008, top-1 99.35%; PPL 13.303 vs 13.298 |
| llama.cpp CPU vs its own Metal (same file) | mean 0.0040, median 0.0010, top-1 97.9% |
| Speed (`hybrid_bench` vs `llama-bench`, interleaved) | pp512 1161 vs 1110-1165 tok/s; pp2048 1094 vs 1066-1121; tg128 35-38 vs 36-39 |

## Known gaps

- One quant (`Q4_K_M`) is validated. Its embedding and head gates are BF16. `Q8_0` should load but is untested; the all-BF16 file (`Spark-X2.6-4B.gguf`) is refused, since only embeddings, LM heads, routers, head gates and the fused QKV accept BF16.
- Tool calls use Flint's `<arg_key>`/`<arg_value>` format and go through the same parser. A weather-tool request through `ferrum-server` returns a call with schema-typed arguments; broader tool use is untested.
- `reasoning_effort` (`low`/`high`) is passed through to the template; the 1M trained context is capped by the memory plan.
- No trained drafter (MTP, DFlash, DSpark) exists for Spark. Prompt lookup (`--draft lookup`, "Prompt lookup" in the TUI) works and is lossless: 1.13× on the `hybrid_spec` chat prompts (35.8 → 40.4 tok/s). Plain decode already reads weights at about 81% of the M5's measured bandwidth ceiling.

# LFM3-1.7B

LiquidAI's text model (`general.architecture = lfm3`). The language model is one architecture. Still images are [LFM3-VL-2B](#lfm3-vl-2b). Audio is not implemented. Ferrum does not download weights. The file this section talks about is the hub-cache snapshot

`~/models/hub/models--LiquidAI--LFM3-1.7B-GGUF/snapshots/main/LFM3-1.7B-Q4_K_M.gguf`

(1.1 GB, Q4_K_M). The Hugging Face repo is [`LiquidAI/LFM3-1.7B-GGUF`](https://huggingface.co/LiquidAI/LFM3-1.7B-GGUF).

24 layers, hidden 2048, SwiGLU width 7680, vocab 128000, trained context 262144, tied embeddings. Thirteen gated short convolutions (kernel 7), five sliding-window attention layers (window 256) and six full-attention layers. The layer kinds come from the GGUF (KV heads are 0 on a conv layer and 4 on an attention layer); they are not hardcoded. Full-attention RoPE uses base 1,000,000 and divides the angle by `rope_freqs.weight` (in this file the first 32 factors are 1 and the last 32 are about 1e30, so those pairs do not rotate). Sliding layers rotate all 128 dimensions at base 10,000 and ignore the factors. The contract is llama.cpp's `lfm3` graph (build 11946 or newer). The llama.cpp build cited in the October verification notes (11669) cannot load this file.

The chat template is the GGUF's own Jinja. It emits `<|startoftext|>` itself (`add_bos_token` is false), opens a generation with `<think>` unless `enable_thinking` is false, and writes tool calls as `<tool_call><function=NAME><parameter=K>…</parameter></function></tool_call>`, which the existing parser accepts. `tests/fixtures/lfm3/` compares those renderings to Jinja2. Sampling metadata is temperature 0.1, top-k 50 and repetition penalty 1.1. There is no top-p key, so the usual fallback of 0.95 applies.

## Validation

`LFM3-1.7B-Q4_K_M.gguf` on a 32 GB M5, against llama.cpp build 11962. The record is [measurements/lfm3/](measurements/lfm3/). Homebrew llama.cpp build 11669 cannot load this file.

| Check | Result |
|---|---|
| Tokenizer ids vs `llama-tokenize` (mixed-script sample, 19 ids) | identical |
| Greedy continuation, "The capital of France is", 16 tokens | identical token ids |
| KLD vs llama.cpp Metal, one 512-token chunk of this repo's README (255 positions) | mean 0.0010, median 0.00082, top-1 99.22%; PPL 158.00 vs 160.45 |
| Memory plan, auto context | weights+scratch 1344 vs 1345 MiB; state 3081 vs 3082 MiB |
| `--draft lookup` vs plain greedy (12 tokens, thinking off) | same text; 3 of 3 drafts accepted |
| `--draft-tree` | refused: `draft trees are not supported for this model` |
| Qwen greedy, 16 tokens, vs `experimental` at `ec14e94` | identical ids on Swift Q4_K_M and Tiel IQ4_XS |

The perplexity is on one chunk of the README, not a standard corpus. It is only there to sit next to the KLD.

## Known gaps

- One quant (`Q4_K_M`) is the file on disk. Other quants are not loaded.
- No MTP head. `--draft lookup` can verify a chain; a draft tree is refused, because each row would need its parent's short-conv window.
- This file has no projector. Still images need [LFM3-VL-2B](#lfm3-vl-2b). Audio projectors (`lfm3a`, `lfm3a_gen`) are refused.
- Liquid's card says the model is for local agents, tool calling and structured output, and is not recommended for agentic coding or complex multi-step reasoning because of its size. Use retrieval for knowledge-heavy tasks.

# LFM3-VL-2B

The same `lfm3` language model with a still-image projector (`clip.vision.projector_type = lfm3vl`). The Hugging Face repo is [`LiquidAI/LFM3-VL-2B-GGUF`](https://huggingface.co/LiquidAI/LFM3-VL-2B-GGUF). It publishes no Q4_K_M projector (BF16, F16 and Q8_0 only), so the pair checked here is the Q4_K_M language file and `mmproj-LFM3-VL-2B-Q8_0.gguf`:

`~/models/hub/models--LiquidAI--LFM3-VL-2B-GGUF/snapshots/main/LFM3-VL-2B-Q4_K_M.gguf` (1.1 GB)

`~/models/hub/models--LiquidAI--LFM3-VL-2B-GGUF/snapshots/main/mmproj-LFM3-VL-2B-Q8_0.gguf` (194 MB)

The language file matches the 1.7B schedule: 24 layers, hidden 2048, SwiGLU 7680, vocab 128000, 13 short convolutions (kernel 7), 5 sliding-window layers (window 256) and 6 full-attention layers. Its trained context is 128000, not 262144. There is no `rope.dimension_sections`. Each image token keeps its own position. The chat template emits `<|vision_start|><|image|><|vision_end|>`, and `<|image|>` (id 125030) is the token that is repeated. A rendered prompt that contains `<|video|>` is refused. Sampling metadata is the same as the 1.7B file: temperature 0.1, top-k 50, repetition penalty 1.1, and the 0.95 top-p fallback. The card is [LFM3-VL-2B](https://docs.liquid.ai/lfm/models/lfm3-vl-2b).

The projector is a 14-block tower (hidden 1024, 16 heads, 8 KV heads, patch 16, 2×2 merge, SwiGLU, mean and std 0.5). Resize follows `clip.vision.image_resize_algo` (bicubic on this file, Catmull-Rom). The projector's layer-norm epsilon is 1e-6, not the encoder's 1e-5. Pixel limits are 65536 to 16777216. The attention-score buffer caps one image at 1773 language tokens; `--image-tokens` is clamped to that, and the default 1024 sits under it. With both files in one folder, `--mmproj auto` picks the projector whose width matches the language model.

```sh
hf download LiquidAI/LFM3-VL-2B-GGUF \
  LFM3-VL-2B-Q4_K_M.gguf mmproj-LFM3-VL-2B-Q8_0.gguf --local-dir ~/models
ferrum-cli --model ~/models/LFM3-VL-2B-Q4_K_M.gguf \
  --mmproj ~/models/mmproj-LFM3-VL-2B-Q8_0.gguf
```

`/image PATH` attaches a still image. The same flags work on `ferrum-server` and in the TUI.

## Validation

On a 32 GB M5, against llama.cpp build 11962. The record is [measurements/lfm3/](measurements/lfm3/).

| Check | Result |
|---|---|
| Language memory, auto context, no projector | weights+scratch 1344 vs 1345 MiB; state 1509 vs 1510 MiB |
| Greedy, one 256×256 image, "Describe the image in one sentence.", 16 tokens, thinking on | same sentence as `llama-mtmd-cli`; that sentence is 16 tokens |
| Qwen greedy, 16 tokens, after these kernels, vs the ids recorded at `ec14e94` | identical on Swift Q4_K_M and Tiel IQ4_XS |

The image is exactly 65536 pixels, so neither side resizes it. llama-mtmd-cli did not print the prompt length; only the continuation was compared. Ferrum's run reported 83 prompt tokens and 0.06 s to encode. Those timings are one run, not a bench. No vision-tower KLD was measured. The 27B and 35B were not re-benched: the greedy ids above are the regression check for the shared forward.

## Known gaps

- One language quant (`Q4_K_M`) and one projector quant (`Q8_0`). Other pairs are not loaded.
- Video is refused. Audio projectors are refused. The image-generation text encoder stays Qwen3-VL.
- The language memory line does not include the projector. The planner still estimates a projector the way it does for Qwen (`v.blk` and `mm` tensors at 2 bytes each, plus 640 MiB), so that reservation did not move.
- Draft trees stay refused, as on the text model.
- The score buffer, not the file's pixel cap, limits an image to 1773 language tokens.

# MLX 4-bit checkpoints

A directory holding an MLX affine 4-bit (group 64) conversion of a dense Qwen3.5-family model loads in `ferrum-cli`, `ferrum-server` and `ferrum-tui`, with no conversion step. The directory needs `config.json`, `model*.safetensors`, `tokenizer.json` and `chat_template.jinja`, and `generation_config.json` gives the default sampling. The decode GEMV and the verify-width GEMMs (integer codes on the matrix units) are adapted from [Kapitsa](https://github.com/nace-ai/kapitsa); those files are Apache-2.0 only ([THIRD_PARTY_NOTICES](../THIRD_PARTY_NOTICES)).

## Validation

The Qwen3.8-27B 4-bit conversion that Kapitsa downloads (`~/.cache/kapitsa/models/qwen3.8-27b-4bit`), on a 32 GB M5:

| Check | Result |
|---|---|
| Greedy text vs Kapitsa's plain decoding | identical, except one newline at a boundary |
| Every projection band (1–130 rows) vs a CPU dequantization (`mlx_tests`) | within 1×10⁻⁴ relative (1–2 rows), 2×10⁻³ (F16-input tiles) |
| Plain decode, same prompts through each server | Ferrum 7.42–7.84 tok/s, Kapitsa 7.66–7.72 |

## Known gaps

- One checkpoint is validated. MLX MoE checkpoints, bit widths other than 4, and group sizes other than 64 are refused at load.
- The MTP drafter reads the NextN block of a GGUF, so it is unavailable on MLX checkpoints. DFlash, DFlash2 and DSpark drafters work.
- The decode GEMV streams at about 94% of this Mac's measured read bandwidth; Kapitsa's fused norm, gate and up GEMVs (`qmvf`) were not ported, since they could save only about 3 ms per token.

# KFlash speculative decoding

`--draft-tree auto` (the TUI's *Draft tree* `auto`) turns on KFlash, ported from Kapitsa (Nace AI, Apache-2.0; `src/hybrid/kflash.rs`):

- **Calibrated trees.** The drafter's node probabilities are mapped to the probability that the model accepts them, per depth. The fit is learned online from every verified node whose parent was accepted, and trees are built best first by that calibrated path probability.
- **A learned budget.** Each step verifies the number of rows (2, 4, 8, 12, 16, 24 or 32) that maximizes the expected tokens minus the recent token rate times that row count's measured cost. Costs and calibration are cached per model, drafter and chip in `~/.cache/ferrum/kflash`, so the first run spends some cycles exploring.
- **Kapitsa's DFlash2 defaults.** A 16-row drafting block for a drafter trained on 8 (`--draft-block 16`), over the first 98,304 token ids (`--draft-vocab 98304`). Both can be set by hand for fixed trees too.
- **Exact sampled speculation**, for every speculative request, not only KFlash. Each token is drawn with a uniform keyed by the seed and its position, and a draft is kept only when it equals its own draw. Sampled output is then plain sampling's for the same seed, and trees apply to sampled requests too. This changes which text a given seed produces, not its distribution.

## Validation

MLX 4-bit 27B with its DFlash2 drafter, `hybrid_spec`, 8 chat prompts × 256 tokens:

| Check | Result |
|---|---|
| Greedy, default kernels | 25.4 tok/s (3.28× plain; the chain 2.71×); every output identical to plain decoding and to the chain |
| Greedy, `FERRUM_HYBRID_EXACT=1` | identical |
| T 0.6, default kernels | 22.8 tok/s (2.95× plain); 5 of 8 identical to plain sampling, 3 part at single tokens |
| T 0.6, `FERRUM_HYBRID_EXACT=1`, the 3 that parted | identical: rounding, not the acceptance rule (one draw fell 0.001 from a token edge) |
| Budget's expected tokens per step vs measured | 5.03 vs 5.02 |

Against Kapitsa through each server (same prompts): 82–91% of its greedy speed and 75–93% at T 0.6 on the same MLX checkpoint ([README](../README.md#performance), [Phase 11 journal](history/phase11-journal.md)).

## Known gaps

- On the Q4_K_XL GGUF, `auto` is 3–5% behind `--draft-block 16 --draft-vocab 98304 --draft-tree 15`: Q4_K verifies make 2-row steps look cheap. Use the fixed tree there.
- Ferrum builds trees on the CPU from the drafter's candidates read back each step, about 5 ms a cycle. Kapitsa builds them on the GPU, and that is most of the remaining gap.
- Cycle costs are wall time, not GPU time, so a busy GPU or a hot Mac moves the learned table; a row count's first cycle in a process (pipeline builds) is left out.
- MoE targets keep chains: every tree row routes to its own experts. Kapitsa's dynamic MoE trees were not ported.
- Prompt lookup's trees stay a test vehicle and are not offered by the TUI.

# Image generation

Qwen-Image-2.1-Turbo (8 steps, no guidance) on a 32 GB M5, drawing and editing, through `ferrum-cli image`, `ferrum-server --image-model`, the TUI Image tab and the chat and agent tool `generate_image`. Usage: [server.md](server.md#image-generation).

With a vision projector loaded, the chat and agent tools close the loop: `generate_image` adds the picture it just made to its own reply (`inject: true`, the default; `inject: false` for a call that only needs the file), and `view_image` opens any image file the sandbox allows. The model sees its output as pixels and can correct it in the next call. Without a projector both still save or read the file, and the reply says the picture was not shown.

## Validation

All against diffusers at 512², with diffusers' own noise, and repeatable with the ignored tests in `src/diffusion/reference_tests.rs` ([edit-parity-2026-10-26.txt](measurements/image-gen/edit-parity-2026-10-26.txt)). Errors are relative RMS.

- **Drawing**, against an fp16 run: prompt embeddings 1.0% (Q8_0 encoder 2.7%, Q4_0 33%, the UD-Q4_K_XL GGUF 26%), transformer blocks 0.05–0.7%, final latents 0.9%, decoded image 0.065% (the decoder alone, on diffusers' latents). A bf16 reference is itself 4.8% and 6.2% from the fp16 one on embeddings and latents, so it is the looser ruler.
- **Editing**, one 512² condition image, against a bf16 run: the VAE encoder 0.36% from diffusers' in float32; the vision tower 2.0% and its DeepStack features 0.25–1.1% from Transformers' in float32 (diffusers' own bf16 tower is 5.4% from that); the text encoder's rows after the image 2.2%; the transformer's condition-image prefix 0.3–2.2% across its blocks, and the final latents 2.2%. The prompt tokenizes identically.
- **Memory**: every stage's measured peak is within 4 MiB of the planner's prediction or below it, for drawing at 512² and 1024² and editing at 512² and 1024², and the tests fail otherwise.
- The kernels the transformer's GGUF path adds (Q4_K, Q5_K and Q6_K matrix products with F16 activations) match ggml's dequantization on the CPU; the GGUF name mapping, the gate/up split and the refusal of unsupported types are unit-tested on synthetic files.

## Known gaps

- Only the Turbo schedule is validated. The base model's 40-step dynamic-shift schedule runs, but has no guidance (no negative prompt or true CFG, which the base model is usually run with) and is not checked against diffusers.
- Seeds do not reproduce diffusers' images (ChaCha8 and Box-Muller, not torch's generator).
- Editing is checked with one condition image; two to four follow the same code but have no reference run.
- Transformer weights load from BF16 safetensors (quantized to Q8_0 or Q4_0 on load) or from an unsloth-style GGUF (`--transformer-file`) whose block matrices are Q4_K, Q5_K, Q6_K, Q8_0, Q4_0 or BF16; other types are refused at load. A text-encoder GGUF (`--text-file`) must be a Qwen3-VL-8B language-model file with the usual `blk.N` names. MLX-quantized checkpoints do not load.
- GGUF runs are validated by tensor mapping against Turbo's weights, memory estimates and pictures (Q4_K_M within 32.0 dB of Q8_0 at one seed), not against a diffusers dump of the base model ([gguf-2026-10-26.txt](measurements/image-gen/gguf-2026-10-26.txt)).
- Denoising takes about 7 s a step at 1024². The projections run at 10–12 TFLOPS, near the best the TensorOps tiles reached in a sweep, and attention at about 4.5 TFLOPS is the part left to tune ([gemm-2026-10-26.txt](measurements/image-gen/gemm-2026-10-26.txt)).
- Cancelling waits for the current denoising step to end, and decoding cannot be cancelled. With `n` above 1, a failure discards the pictures already made.
- The decoder has no tiling, so very large images are bounded by memory.
