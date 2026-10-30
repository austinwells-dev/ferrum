# LFM3-Audio-2B speech input, 2026-10-30

Machine: Apple M5, 32 GB, macOS 27.0.1. Files from [`LiquidAI/LFM3-Audio-2B-GGUF`](https://huggingface.co/LiquidAI/LFM3-Audio-2B-GGUF):
`LFM3-Audio-2B-Q4_K_M.gguf` (language), `mmproj-LFM3-Audio-2B-Q8_0.gguf` and `mmproj-LFM3-Audio-2B-BF16.gguf` (projectors).
Reference: llama.cpp built from source at `792fcf4` (Homebrew's build 11669 has no `lfm3a`).

| File | What it is |
|---|---|
| [parity-2026-10-30.sh](parity-2026-10-30.sh), [.txt](parity-2026-10-30.txt) | Encoder rows against llama.cpp's `mtmd` output for the same samples, the memory plan and CPU front end, greedy text through `ferrum-cli` and `llama-mtmd-cli --jinja`, and the LFM3-VL picture check, at `e4f76c2` (after the review fixes; first run at `bb1e87b` gave the same rows and text) |
| [embdump.cpp](embdump.cpp) | The reference dumper: feeds raw 16 kHz samples to `mtmd` and writes `mtmd_get_output_embd` |
| [server-2026-10-30.sh](server-2026-10-30.sh), [.txt](server-2026-10-30.txt) | `ferrum-server`: `/props` modalities and one `input_audio` transcription, at `e4f76c2` |
| [regression-2026-10-30.sh](regression-2026-10-30.sh), [.txt](regression-2026-10-30.txt) | `experimental` `8cb5110` against the branch at `4a2fae3`: greedy and memory on Swift 27B and Tiel 35B-A3B, KLD, `hybrid_bench` A-B-B-A ×2, an LFM3-VL picture and LFM3 text |

The clips are macOS `say` speech (the commands are in the parity script), not recorded voices, so they are not committed. The parity script writes Ferrum's decoded samples (`hybrid_audio_check --samples`) and gives exactly those to llama.cpp, so the comparison covers the mel front end and the encoder, not two WAV readers. `weather48k` is the `weather` clip as 48 kHz stereo float, which goes through Ferrum's resampler first.

## Encoder rows

Per row of 2048, Ferrum against llama.cpp:

| Clip | Projector | Rows | Cosine (min) | Relative L2 mean / max | Max abs diff |
|---|---|---|---|---|---|
| fox, 2.84 s | BF16 | 36 | 0.999999 | 0.0013 / 0.0017 | 0.0013 |
| weather, 5.99 s | BF16 | 75 | 0.999997 | 0.0013 / 0.0026 | 0.0005 |
| weather48k | BF16 | 75 | 0.999997 | 0.0013 / 0.0026 | 0.0006 |
| fox | Q8_0 | 36 | 0.999999 | 0.0012 / 0.0017 | 0.0007 |
| weather | Q8_0 | 75 | 0.999996 | 0.0013 / 0.0028 | 0.0005 |
| weather48k | Q8_0 | 75 | 0.999996 | 0.0013 / 0.0029 | 0.0009 |
| long, 112.4 s | Q8_0 | 1405 | 0.999997 | 0.0014 / 0.0026 | 0.0012 |

A 247 s clip is refused. Encode times are single runs: 21 ms for 2.84 s, 33 ms for 5.99 s and 0.57 s for 112.4 s.

## Memory plan and CPU front end

`hybrid_audio_check --memory` loads the projector, then encodes 8 silent 120 s clips (the runtime's cache holds 8) while keeping each one's rows, and compares the planner's projector budget (`vision::projector_bytes`) with the peak of Metal's allocated size: 665.8 MiB planned against 664.9 MiB measured (220.0 MiB of weights, as the loader counted them, plus 362.9 MiB to encode one clip, its 11.7 MiB of rows included, plus 7 more clips' rows), for both the BF16 and Q8_0 projectors. The first review's fix (559.3 planned, 559.4 measured) left out the projector's hidden and output buffers and the cache; the second review's fix counts them. Before either fix the planner charged a flat 400 MiB of activations.

On the 112.4 s clip as 48 kHz stereo float, the CPU front end now takes 0.18 s to decode and resample and 0.24 s for the log-mel features, against 0.56 s for the encoder. A 247 s clip of that format is refused from its WAV header in 0.54 s, which is mostly the projector load. Before the fixes, at `c092f40` (single `/usr/bin/time` runs of `hybrid_audio_check`, not in the record files), the 112 s clip took 3.16 s in all, about 2.5 s of it resampling, and the 247 s clip took 5.24 s to be refused, because it was decoded and resampled first.

## Projector auto-pick

With both LFM3 pairs linked into one folder, `ferrum-cli --model LFM3-VL-2B-Q4_K_M.gguf` (auto projector) picked `mmproj-LFM3-Audio-2B-BF16.gguf` before the fix, by alphabetical order; after it, it picks `mmproj-LFM3-VL-2B-Q8_0.gguf`, and the Audio model keeps its own projector. Checked by hand, not in the record files; `projectors_match_their_model_by_name` covers the name rule.

## Greedy text

Q4_K_M with the Q8_0 projector, temperature 0, 96 tokens at most. `llama-mtmd-cli` needs `--jinja`: without it, it builds a ChatML prompt with no `<|startoftext|>`, and the question and two-clip answers differ from the GGUF template's.

| Prompt | Both answered |
|---|---|
| System "Perform ASR in English.", fox clip | The quick brown fox jumps over the lazy dog. |
| System "Perform ASR in English.", weather clip | Tomorrow will be mostly sunny in Chicago, with a high of 72 degrees and light winds from the west. |
| weather clip + "What city is mentioned, and what is the expected high temperature? Answer in one sentence." | Chicago, with a high of 72 degrees |
| fox + weather48k clips + "Two clips are attached. Summarise each in a few words." | A quick brown fox jumps over a lazy dog. |

The two-clip answer summarises only the first clip on both sides. The server's `input_audio` request gives the same weather transcript from 95 prompt tokens.

The TUI's Audio tab was driven in tmux at `e4f76c2`: the weather48k clip dropped on the terminal gave the same transcript (95 prompt tokens), the question task gave the same answer (102 prompt tokens), the 247 s clip was refused before loading, and a recording from a silent input was refused. Its ffmpeg settings, recording `say` through the BlackHole loopback device instead of the default input, gave a clip that transcribed exactly. A live microphone was not tested.

## Regression against experimental

| Check | Base `8cb5110` | Branch `4a2fae3` |
|---|---|---|
| Greedy, 48 tokens, Swift 27B Q4_K_M and Tiel 35B-A3B IQ4_XS | | identical stdout on both |
| Memory, Swift | 16679 (16682) / 6470 (6471) MiB | same |
| Memory, Tiel | 17114 (17116) / 5183 (5184) MiB | same |
| KLD vs llama.cpp, 2040 positions, Swift | 0.000132, top-1 99.46%, PPL 7.4710 | identical |
| KLD, Tiel | 0.008268, top-1 95.20%, PPL 10.9886 | identical |
| LFM3-VL picture through `ferrum-cli`, greedy, 96 tokens | | identical (and again at `e4f76c2`, after `expand_media` was split up, in the parity record) |
| LFM3 text, raw "The capital of France is", 48 tokens | | identical |

Memory is measured weights+scratch / state, with the prediction in brackets. The raw LFM3 continuation repeats one word on both builds; that line only checks that the two builds agree.

`hybrid_bench --pp 512,2048 --tg 128`, best of four runs each, interleaved A-B-B-A twice (tok/s):

| Model | | Base | Branch | Spread across all 8 runs |
|---|---|---|---|---|
| Swift 27B | pp512 | 178.44 | 175.07 | 161.88–178.44 |
| | pp2048 | 169.90 | 170.35 | 164.52–170.35 |
| | tg128 | 6.81 | 6.82 | 6.74–6.82 |
| Tiel 35B-A3B | pp512 | 914.40 | 915.07 | 903.37–915.07 (±40 within a run) |
| | pp2048 | 866.21 | 859.82 | 854.54–866.21 |
| | tg128 | 40.34 | 40.35 | 40.05–40.35 |

Every best-of difference is inside the run-to-run spread. The largest is Swift pp512 (base 1.9% ahead), where both second-round base runs (177.59, 178.44) beat both second-round branch runs (174.33, 175.07); in the first round the bests were 174.92 (base) and 174.65 (branch). The branch only adds kernels that the language model does not call (`a_conv_stem`, `a_dwconv_s2`), so no change was expected.
