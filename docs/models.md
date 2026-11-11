# Models and features with open gaps

What the hybrid apps run besides the two Qwen3.5-family GGUFs in the [README](../README.md#get-a-model), and what is still unchecked. These are part of this tree. They are not a separate product, and they are not made stable by which branch the commit sits on. A later merge of `experimental` into `main` would not retire the gaps below.

Checked files, commands and numbers are in [experimental branch notes](history/experimental-branch-notes.md) (the old `docs/experimental.md`, kept as a record) and under [measurements](measurements/). This page does not add a Ferrum run. The LFM3-Omni keys are from the Hugging Face file viewer for the published `mmproj`, not from a load here.

How to call the server, the TUI and the encoders is in [server.md](server.md), [tui.md](tui.md) and [vision.md](vision.md). The layer geometry is in [architecture.md](architecture.md#the-hybrid-engine).

## Flint

`unsloth/flint-GGUF`, file `flint-UD-Q3_K_XL.gguf` (18.5 GB). `general.architecture = flint`. Tool calls use `<tool_call>NAME<arg_key>…</arg_key><arg_value>…</arg_value></tool_call>`, typed by the tool schema. The GGUF template has no thinking switch: the prompt ends in `<think>` or `<think></think>`.

```sh
hf download unsloth/flint-GGUF flint-UD-Q3_K_XL.gguf --local-dir ~/models
```

One quant is validated (tokenizer, a short greedy continuation, KLD against llama.cpp after Flint merged, and `hybrid_bench`). Other Flint files may use quant types the kernels do not cover; the TUI names the set when it refuses a file. There is no MTP or DFlash drafter. Prompt lookup works and is less exact once a verify is wide enough to use F16 TensorOps tiles: 4 of 8 greedy `hybrid_spec` prompts drifted, and `FERRUM_HYBRID_EXACT=1` left one near-tie. Expert routing refuses a selection bias or an `expert_weights_scale` other than 1.0. The legacy attention env vars reject this geometry. Sliding-window layers keep a ring of 1,088 rows, so only the 14 full-attention layers grow with context. Snapshots copy those rings; a rewind is not limited to whatever the live ring still holds.

Record: [experimental branch notes](history/experimental-branch-notes.md#what-works), and the October re-check in [verification-2026-10](measurements/verification-2026-10/README.md) (llama.cpp build 11669 cannot load `flint`).

## Qwen3.8-Flash-Next

`ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF`, both shards `Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf` and `-00002-of-00002.gguf` (37.6 + 28.8 GB). Load the first; the second must sit beside it. `general.architecture = qwen4exp`. The routed experts stay in the file and stream from the SSD. On a 32 GB M5 the measured decode was 16.8–18.4 tok/s and prefill 232–315 tok/s, ahead of Strata on that machine; a DFlash2 drafter reached 23.2 tok/s on a short raw prompt and 26.5 tok/s (1.42×) in the TUI chat benchmark.

The sparse-attention indexer is not loaded. Up to its top-k of 2,048 positions it would select every position, so attention there matches dense attention; past 2,048 Ferrum attends to all positions, which the model was not trained to do. Contexts past 2,048 are not validated. Draft trees are refused. Chains can drift from plain decoding at near ties unless `FERRUM_HYBRID_EXACT=1`; drafters default to 2 drafts because each extra row reads more experts from the SSD. The separate MTP head does not load. Vision answers one test image and has no parity check against llama.cpp's `mtmd`. One quant (`Q2_0`) is validated.

Record: [experimental branch notes](history/experimental-branch-notes.md#qwen38-flash-next) and [measurements/qwen4exp](measurements/qwen4exp/).

## Spark-X2.6-4B

`XHToken/Spark-X2.6-4B-GGUF`, file `Spark-X2.6-4B-Q4_K_M.gguf` (3.5 GB). `general.architecture = spark2_5`. One quant is validated against llama.cpp build 11669 (tokenizer, greedy continuation, chat template, KLD). `Q8_0` should load and has not been run. The all-BF16 file is refused. Tool calls use Flint's `<arg_key>` / `<arg_value>` form; one weather-tool request through the server returned schema-typed arguments, and broader tool use is untested. No MTP, DFlash or DSpark drafter exists. Prompt lookup is lossless on the `hybrid_spec` prompts (1.13×).

Record: [experimental branch notes](history/experimental-branch-notes.md#spark-x26-4b).

## LFM3

LiquidAI's `lfm3` language model. The three checked language files are below. Speech output is not implemented. Draft trees are refused on every one of them (`draft trees are not supported for this model`), because a tree row would need its parent's short-conv window. `--draft lookup` can verify a chain. `general.sampling.penalty_repeat` is read only for `lfm3`.

### LFM3-1.7B

Text only. [`LiquidAI/LFM3-1.7B-GGUF`](https://huggingface.co/LiquidAI/LFM3-1.7B-GGUF), `LFM3-1.7B-Q4_K_M.gguf` (1.1 GB). Trained context 262,144. The chat template emits `<|startoftext|>` itself and opens a generation with `<think>` unless `enable_thinking` is false. Tool calls are `<tool_call><function=NAME><parameter=K>…</parameter></function></tool_call>`. Sampling metadata in the file is temperature 0.1, top-k 50 and repetition penalty 1.1; there is no top-p key, so the usual 0.95 fallback applies.

Checked against llama.cpp build 11962: tokenizer, 16 greedy tokens, KLD on one 512-token chunk of this repo's README (not a standard corpus), the memory plan, and lookup drafting. Homebrew llama.cpp build 11669 cannot load the file. Other quants were not loaded. Liquid's card says the model is for local agents, tool calling and structured output, and is not recommended for agentic coding or complex multi-step reasoning because of its size.

Record: [measurements/lfm3](measurements/lfm3/README.md).

### LFM3-VL-2B

Still images. [`LiquidAI/LFM3-VL-2B-GGUF`](https://huggingface.co/LiquidAI/LFM3-VL-2B-GGUF). The repo publishes no Q4_K_M projector, so the checked pair is `LFM3-VL-2B-Q4_K_M.gguf` (1.1 GB) and `mmproj-LFM3-VL-2B-Q8_0.gguf` (194 MB). Trained context 128,000. The marker is `<|image|>`, repeated once per language token, each with its own position. A prompt that contains `<|video|>` is refused whether or not a projector is loaded. The attention-score buffer caps one image at 1773 language tokens; `--image-tokens` is clamped to that.

```sh
hf download LiquidAI/LFM3-VL-2B-GGUF \
  LFM3-VL-2B-Q4_K_M.gguf mmproj-LFM3-VL-2B-Q8_0.gguf --local-dir ~/models
ferrum-cli --model ~/models/LFM3-VL-2B-Q4_K_M.gguf \
  --mmproj ~/models/mmproj-LFM3-VL-2B-Q8_0.gguf
```

`/image PATH` attaches a still image. The same flags work on the server and in the TUI. Greedy descriptions of two synthetic pictures matched `llama-mtmd-cli` (build 11962). No vision-tower KLD was measured. Other quant pairs were not loaded. The image-generation text encoder stays Qwen3-VL.

Record: [measurements/lfm3](measurements/lfm3/README.md). Encoder: [vision.md](vision.md#lfm3-vl).

### LFM3-Audio-2B

Speech input. [`LiquidAI/LFM3-Audio-2B-GGUF`](https://huggingface.co/LiquidAI/LFM3-Audio-2B-GGUF). Checked pair: `LFM3-Audio-2B-Q4_K_M.gguf` (1.1 GB) and `mmproj-LFM3-Audio-2B-Q8_0.gguf` (304 MB). The BF16 projector (571 MB) was checked for encoder rows too. Trained context 128,000. The template turns `input_audio`, `audio` and `audio_url` parts into `<|audio|>` (id 125031), repeated once per encoded row, with no begin or end wrapper. The projector also holds the speech generator (`a.gen.*`, `clip.gen.audio.projector_type = lfm3a_gen`); those tensors are not loaded.

WAV only (PCM 8/16/24/32-bit or float), any sample rate and channel count, mixed to mono and resampled to 16 kHz. Other formats are refused, with a hint to convert them (`afconvert -f WAVE -d LEI16@16000 in.m4a out.wav`). Rates above 384 kHz are refused. One clip may last 120 s (1500 tokens); a longer clip is refused from its header before it is decoded. At most 8 clips per request. The encoder is one pass, so attention scores grow with the square of the clip. For transcription, put Liquid's instruction in the system prompt (`Perform ASR in English.`) and only the audio in the user turn. Nothing fills that prompt in for you. The server accepts OpenAI `input_audio` parts ([server.md](server.md#audio)). The TUI chat attaches `.wav` files up to 8 MB (about 4 minutes of 16 kHz mono 16-bit, or about 43 s of 48 kHz stereo), and the Audio tab records through `ffmpeg` ([tui.md](tui.md#audio)). The recording path was checked with a loopback device, not a live microphone.

`--mmproj auto` tells a VL projector and an audio projector apart by filename (`mmproj-LFM3-VL-2B-Q8_0.gguf` belongs with `LFM3-VL-2B-*.gguf`). Renamed files in one folder need `--mmproj FILE`.

Encoder rows matched llama.cpp `mtmd` at commit `792fcf4` (cosine at least 0.999996). Greedy transcripts and one question matched `llama-mtmd-cli --jinja`. The clips were macOS `say` voices, not a word-error-rate corpus. Homebrew's llama.cpp build does not have `lfm3a`.

Record: [measurements/lfm3-audio](measurements/lfm3-audio/README.md). Encoder: [vision.md](vision.md#lfm3-audio).

### LFM3-Omni

[`LiquidAI/LFM3-Omni-2B-GGUF`](https://huggingface.co/LiquidAI/LFM3-Omni-2B-GGUF) is not a supported model here. No Ferrum run has loaded it.

The published `mmproj` metadata (Hugging Face file viewer for `mmproj-LFM3-Omni-2B`) sets `clip.has_vision_encoder`, `clip.vision.projector_type = lfm3vl`, `clip.has_audio_encoder`, `clip.audio.projector_type = lfm3a` and `clip.gen.audio.projector_type = lfm3a_gen`. It does not set `clip.projector_type`. `is_audio_projector` requires `lfm3a` and rejects a file with `clip.has_vision_encoder`, so this projector is not the speech-input path. The still-image loader reads `clip.projector_type` when that key is present and otherwise `clip.vision.projector_type`, so this file would be handed to the LFM3-VL tower, and the audio and speech-output tensors would not be loaded. A combined file whose `clip.projector_type` is `lfm3a` is refused by that loader instead (`audio is not implemented` is the text of that refusal; it is the vision loader's message, not a statement that LFM3-Audio is missing). None of those paths has been executed on an Omni file. Video is refused on every LFM3 prompt. Speech output is not implemented for any LFM3 projector.

## MLX 4-bit checkpoints

A directory of an MLX affine 4-bit (group 64) dense Qwen3.5-family model loads in the CLI, server and TUI: `config.json`, `model*.safetensors`, `tokenizer.json` and `chat_template.jinja`. The checked directory is the Qwen3.8-27B conversion Kapitsa downloads (`~/.cache/kapitsa/models/qwen3.8-27b-4bit`). Greedy text matched Kapitsa's plain decoding except one newline at a boundary. MoE checkpoints, other bit widths and other group sizes are refused. The MTP head is read from a GGUF NextN block, so it is unavailable; DFlash, DFlash2 and DSpark drafters work. The Kapitsa-derived GEMV files are Apache-2.0 only ([THIRD_PARTY_NOTICES](../THIRD_PARTY_NOTICES)).

Record: [experimental branch notes](history/experimental-branch-notes.md#mlx-4-bit-checkpoints).

## KFlash

`--draft-tree auto` (the TUI's *Draft tree* `auto`) is KFlash, ported from Kapitsa. It learns which draft-tree rows are worth checking and caches that per model, drafter and chip in `~/.cache/ferrum/kflash`. For a DFlash2 drafter it also defaults to a 16-row block and a 98,304-id draft vocabulary. Sampled speculation, for every speculative request, keeps a draft only when it equals the token drawn for that position and seed, so the text matches plain sampling for the same seed. On the Q4_K_XL GGUF, `auto` is 3–5% behind a fixed 15-wide tree with those DFlash2 defaults. MoE targets stay on chains. Prompt-lookup trees are not offered by the TUI.

On MLX 4-bit 27B with its DFlash2 drafter, greedy `hybrid_spec` output matched plain decoding (25.4 tok/s, 3.28×). Three of eight sampled prompts (T 0.6) parted by one token under default kernels and matched with `FERRUM_HYBRID_EXACT=1`. Server speed against Kapitsa on the same checkpoint was 82–91% greedy and 75–93% at T 0.6 ([README](../README.md#performance)).

Record: [experimental branch notes](history/experimental-branch-notes.md#kflash-speculative-decoding).

## Parallel requests

`ferrum-server --parallel N` (1 to 8) is the server's concurrent mode, documented in [server.md](server.md#parallel-requests). Slots take turns on the GPU, so total throughput stays about that of one request, and the drafter runs on slot 0 only. Each slot keeps its own cache, including the sliding-window rings copied into its snapshots. The October check (1, 4 and 8 slots, greedy output identical to the base, one user's third turn reusing its prefix) is [parallel-slots-2026-10](measurements/parallel-slots-2026-10/README.md). That run predates the ring snapshots; the reuse rules now are the ones in [architecture.md](architecture.md#the-hybrid-engine) and [session reuse](measurements/session-reuse/README.md).

## Image generation

Qwen-Image-2.1 drawing and editing is documented in [server.md](server.md#image-generation), including flags, the 69 s 1024² timing, and the GGUF transformer limits. It is checked against diffusers at 512² for Turbo drawing, and for one 512² edit. The base model's 40-step schedule runs and has no guidance, and it is not checked against diffusers. Seeds do not reproduce diffusers' images. Two to four condition images follow the same code and have no reference run. MLX-quantized diffusion checkpoints do not load. Cancelling waits for the current denoising step; decoding cannot be cancelled. The decoder has no tiling.

Record: [experimental branch notes](history/experimental-branch-notes.md#image-generation) and [measurements/image-gen](measurements/image-gen/).

## Sessions

A chat's cache (tokens, full-attention K/V, recurrent state, sliding-window rings and snapshots) can be saved and loaded. The TUI writes it when the chat would otherwise lose it ([tui.md](tui.md#sessions)). On the current tree, LFM3 does not drop a long turn once the live ring has moved on: the snapshot holds a copy of the ring. The sentence "on experimental, LFM3 loses its cache after about 570 tokens" in [session reuse](measurements/session-reuse/README.md) describes `experimental` at `ab8f57e`, the base of that change, not the tree after `aa2b402`. The same record's branch column is the behavior now: a few tokens redone at a rewritten reply header, and `hybrid_session_check` passing on LFM3-VL, Spark, Flint and the Qwen3.5 models that were run. Disk round-trip times are in [session cache](measurements/session-cache/README.md).
