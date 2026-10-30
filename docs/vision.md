# Vision support

Image input for the hybrid engine. Qwen3.5-family models (`qwen35`, `qwen35moe`) use the
Qwen3-VL projector (`clip.projector_type = qwen3vl_merger`). LFM3-VL uses
`clip.vision.projector_type = lfm3vl` ([below](#lfm3-vl)), and LFM3-Audio's speech input is
`clip.audio.projector_type = lfm3a` ([below](#lfm3-audio)). All ship as `mmproj*.gguf`.

## Pathways, front to back

| Stage | Where | What happens |
|---|---|---|
| Discovery | `vision::resolve_projector` | `--mmproj auto` takes a sibling `mmproj*.gguf` whose `clip.vision.projection_dim` (or `clip.audio.projection_dim`) equals the model's `embedding_length`; `none` disables; a path must be a `clip` GGUF |
| Memory plan | `PlanOptions::vision`, `vision::projector_bytes` | projector weights plus a full-size image's activations are charged before the context is fitted |
| Request | `vision::media::take_images` | finds OpenAI `image_url` / Anthropic `image` parts, decodes `data:` URLs / base64, reads each part's `detail` / `min_tokens` / `max_tokens` into an `ImageBudget`, replaces each with `{"type": "image"}` so the GGUF chat template emits its image marker (`<|image_pad|>` on Qwen, `<|image|>` on LFM3) |
| Encode | `vision::VisionModel` | decode, `smart_resize` to a multiple of 32 within `[min, max]` tokens (`VisionModel::limits`: the request's budget, never above `--image-tokens`), normalise, patchify in 2x2 block order, ViT on Metal, merger; cached by content hash and limits (`Runtime`, shared by every session, up to 8 full-size images' worth of tokens) |
| Prompt | `Runtime::expand_images` | each pad token becomes one per image position (`rows x cols` of the merged grid); the runs are found again in the token ids and become `ImageSpan`s |
| Prefix reuse | `Session::set_media`, `reusable_before_images` | pad tokens look alike, so cached tokens are reused only up to the first image whose position, grid and content hash do not all match |
| Forward | `HybridModel::forward_chunk` | rows inside a span are overwritten with the image embeddings after the token lookup; attention prep gets rotary positions from a `RopePlan` |
| Positions | `engine::rope_position`, `h_attn_prep` | Qwen text keeps counting from where an image's position grid ended (`max(rows, cols)` positions for `rows x cols` tokens); image rows use (time, row, column) positions with interleaved sections `[11, 11, 10]` read from `rope.dimension_sections`. LFM3 gives each image token its own position and does not read those sections. Text-only chunks use a scalar offset, so the hot path is unchanged |
| Surfaces | server, `ferrum-cli`, TUI | `/props` modalities, `--mmproj`, `/image`, attachment handling; OCR text remains the fallback without a projector |

## Encoder

`clip.projector_type = qwen3vl_merger`, 27 pre-norm blocks (LayerNorm, fused QKV with bias, 2-D rotary,
full non-causal attention, GELU MLP), `v.post_ln`, 2x2 merge, `mm.0` + GELU + `mm.2`. A still image repeats
its frame for the temporal conv, so the two patch-embedding taps are summed into one matrix at load time.
The learned 48x48 position table is resized bilinearly (corners aligned, as in the reference
implementation). Weights are kept F16 (BF16/F32 files are converted at load); activations are F32 and GEMMs stage
half-precision tiles with F32 accumulation. Attention runs as strided batched GEMMs through an F32 score
buffer, in head groups that bound its size. GGUF projectors with DeepStack layers are rejected at load; the image
pipeline loads the same tower from Qwen-Image's text-encoder safetensors with its DeepStack mergers
([architecture](architecture.md#image-generation)).

## LFM3-VL

`clip.vision.projector_type = lfm3vl` on an `lfm3` language model. Patches stay in raster
order. Each block is a pre-norm residual: a gated depthwise convolution or full
bidirectional attention, then a SwiGLU. The projector layer-normalises each patch
(epsilon 1e-6), interleaves every 2×2 (top-left, top-right, bottom-left, bottom-right)
and runs a GELU MLP. A still image fills both patch-embedding taps, so they add.
`<|image|>` is repeated once per language token. Any LFM3 prompt containing `<|video|>`
is refused, projector or not. The checked pair is
`LFM3-VL-2B-Q4_K_M.gguf` with `mmproj-LFM3-VL-2B-Q8_0.gguf`; that repo has no Q4_K_M
projector. Numbers and the known gaps are in [experimental.md](experimental.md#lfm3-vl-2b).

## LFM3-Audio

`clip.audio.projector_type = lfm3a` (`src/vision/lfm3a.rs`, front end in `src/vision/audio.rs`) is a
one-dimensional twin of the LFM3-VL tower that turns speech into language-model rows. A WAV clip is mixed
to mono and resampled to 16 kHz (windowed sinc, its kernel tabulated once per phase), then cut into 400-sample periodic-Hann frames every 160
samples, with no centring and the tail padded to a whole hop. A 400-point FFT (radix 2, then a direct
25-point DFT) feeds 128 Slaney mel filters, and each bin becomes `(log10(max(x, 1e-12)) + 4) / 4`.

On the GPU, `a_conv_stem` runs the first 3×3 convolution (stride 2, SiLU) and the depthwise 3×3 after it in
one pass, without storing the first stage; `a_dwconv_s2` is the second depthwise stage, and the pointwise
convolutions are GEMMs with a SiLU epilogue. Time and frequency both shrink 8×, so 128 bins become 16, and
the 256 channels × 16 bins go through one 4096→768 projection. Its columns are permuted at load, from
ggml's channel-major order to the stem's frequency-major output. Then 17 pre-norm blocks: gated short
convolutions (kernel 3, causal, the language model's `h_short_conv` with a zero state) and attention with
GQA, QK RMSNorm and NEOX RoPE, where each row sees itself and the 31 before it (an additive mask on F32
scores). Each block ends in a SwiGLU. A final RMSNorm, a LayerNorm and a GELU MLP (768→2048→2048) give one
row per 80 ms of audio.

`<|audio|>` is repeated once per row, and each row has its own position, as with LFM3 images. Clips
are cached by content (the last eight, as many as one request may carry). One clip may
last 120 s, checked from the WAV header before decoding; attention over a clip is n×n, and the memory plan
counts every activation buffer for that length (`lfm3a::activation_bytes`). The
`a.gen.*` speech generator in the same file is not loaded. Numbers and the known gaps are in
[experimental.md](experimental.md#lfm3-audio-2b).

## Validation

* `vision::tests::encoder_matches_cpu_reference` runs a small random Qwen ViT on the GPU and in plain Rust
  (several image shapes, one head group and one head per group).
  `vision::lfm3::tests::encoder_matches_cpu_reference` does the same for a tiny LFM3-VL tower, and
  `vision::lfm3a::tests::encoder_matches_cpu_reference` for a tiny speech encoder. `vision::audio` tests the
  WAV reader, the resampler, the FFT and the mel features on synthetic signals.
* `examples/hybrid_audio_check` encodes a WAV clip and compares its rows with a dump of llama.cpp's
  `mtmd` output for the same samples ([record](measurements/lfm3-audio/)).
* `hybrid::engine::tests::attention_prep_follows_the_position_table` checks the rotary kernel against a CPU
  implementation of the interleaved multi-axis scheme, and that equal axes reproduce the scalar path.
* `rope_positions_compress_around_images`, `prefix_stops_before_a_changed_or_cut_image` and the media
  parsing tests cover the position bookkeeping, reuse rules and request handling;
  `request_budgets_stay_within_the_projector_cap` and `reads_per_image_detail_and_token_limits` cover
  per-request sizes.
* End to end on Qwen3.8-27B (UD-Q4_K_XL) and Tiel-Coder 35B-A3B (BF16 projector): the description of a
  test picture agrees with `llama-mtmd-cli` at temperature 0 (same shapes, colours and layout in near-identical wording), multi-turn conversations
  with the image in history reuse the cache, and text after an image answers correctly.

## Cost

Encoding 1024 tokens (a 1024x1024 image) takes about 4.7 s on an M5; a 640x480 picture (300 tokens) about 0.8 s.
The GEMMs are plain simdgroup-matrix tiles, not the tensor-op path the language model uses, so there is room
to speed this up.

## Not covered

Video (including LFM3 `<|video|>`), speech output and audio formats other than WAV, other projector types (Gemma, LLaVA, Pixtral), DeepStack layers in chat, remote image URLs, and images on
the shared-transformer models (Qwen2.5/3, Granite, OLMo, LFM2).
