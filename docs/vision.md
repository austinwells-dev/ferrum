# Vision support

Image input for the Qwen3.5-family hybrid engine (`qwen35`, `qwen35moe`) using the Qwen3-VL
projector that ships as `mmproj*.gguf`.

## Pathways, front to back

| Stage | Where | What happens |
|---|---|---|
| Discovery | `vision::resolve_projector` | `--mmproj auto` takes a sibling `mmproj*.gguf` whose `clip.vision.projection_dim` equals the model's `embedding_length`; `none` disables; a path must be a `clip` GGUF |
| Memory plan | `PlanOptions::vision`, `vision::projector_bytes` | projector weights plus a full-size image's activations are charged before the context is fitted |
| Request | `vision::media::take_images` | finds OpenAI `image_url` / Anthropic `image` parts, decodes `data:` URLs / base64, replaces each with `{"type": "image"}` so the GGUF chat template emits `<|vision_start|><|image_pad|><|vision_end|>` |
| Encode | `vision::VisionModel` | decode, `smart_resize` to a multiple of 32 within `[min, max]` tokens, normalise, patchify in 2x2 block order, ViT on Metal, merger; cached by content hash (`Runtime`, 8 images) |
| Prompt | `Runtime::expand_images` | each pad token becomes one per image position (`rows x cols` of the merged grid); the runs are found again in the token ids and become `ImageSpan`s |
| Prefix reuse | `Session::set_media`, `reusable_before_images` | pad tokens look alike, so cached tokens are reused only up to the first image whose position, grid and content hash do not all match |
| Forward | `HybridModel::forward_chunk` | rows inside a span are overwritten with the image embeddings after the token lookup; attention prep gets rotary positions from a `RopePlan` |
| Positions | `engine::rope_position`, `h_attn_prep` | text keeps counting from where an image's position grid ended (`max(rows, cols)` positions for `rows x cols` tokens); image rows use (time, row, column) positions with interleaved sections `[11, 11, 10]` read from `rope.dimension_sections`; text-only chunks use a scalar offset, so the hot path is unchanged |
| Surfaces | server, `ferrum-cli`, TUI | `/props` modalities, `--mmproj`, `/image`, attachment handling; OCR text remains the fallback without a projector |

## Encoder

`clip.projector_type = qwen3vl_merger`, 27 pre-norm blocks (LayerNorm, fused QKV with bias, 2-D rotary,
full non-causal attention, GELU MLP), `v.post_ln`, 2x2 merge, `mm.0` + GELU + `mm.2`. A still image repeats
its frame for the temporal conv, so the two patch-embedding taps are summed into one matrix at load time.
The learned 48x48 position table is resized bilinearly (corners aligned, as in the reference
implementation). Weights are kept F16 (BF16/F32 files are converted at load); activations are F32 and GEMMs stage
half-precision tiles with F32 accumulation. Attention runs as strided batched GEMMs through an F32 score
buffer, in head groups that bound its size. DeepStack projectors are rejected at load.

## Validation

* `vision::tests::encoder_matches_cpu_reference` runs a small random ViT on the GPU and in plain Rust
  (several image shapes, one head group and one head per group).
* `hybrid::engine::tests::attention_prep_follows_the_position_table` checks the rotary kernel against a CPU
  implementation of the interleaved multi-axis scheme, and that equal axes reproduce the scalar path.
* `rope_positions_compress_around_images`, `prefix_stops_before_a_changed_or_cut_image` and the media
  parsing tests cover the position bookkeeping, reuse rules and request handling.
* End to end on Qwen3.8-27B (UD-Q4_K_XL) and Tiel-Coder 35B-A3B (BF16 projector): the description of a
  test picture agrees with `llama-mtmd-cli` at temperature 0 (same shapes, colours and layout in near-identical wording), multi-turn conversations
  with the image in history reuse the cache, and text after an image answers correctly.

## Cost

Encoding 1024 tokens (a 1024x1024 image) takes about 4.7 s on an M5; a 640x480 picture (300 tokens) about 0.8 s.
The GEMMs are plain simdgroup-matrix tiles, not the tensor-op path the language model uses, so there is room
to speed this up.

## Not covered

Video, other projector types (Gemma, LLaVA, Pixtral), DeepStack layers, remote image URLs, and images on
the shared-transformer models (Qwen2.5/3, Granite, OLMo, LFM2).
