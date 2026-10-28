# LFM3

## LFM3-1.7B Q4_K_M

Working tree on `feat/lfm3` (parent `ec14e94`), 28 October 2026. Machine: MacBook Pro Mac17,2, Apple M5, 10 CPU cores, 32 GB, Metal 4, macOS 27.0.1. File:

`~/models/hub/models--LiquidAI--LFM3-1.7B-GGUF/snapshots/main/LFM3-1.7B-Q4_K_M.gguf`

(1,099,627,808 bytes). Reference: llama.cpp `v0.8.0-dev` build 11962 (`f5222f25c`), the macOS arm64 release binary. Homebrew llama.cpp 0.7.0 build 11669 cannot load `lfm3`. One model in memory at a time. The load and the greedy run used `caffeinate -dims`.

## Memory plan

`./target/release/ferrum hybrid --model <gguf> --raw --prompt hi --max-new-tokens 16`

```
memory plan: budget 23552 MiB = weights 1247 + scratch 98 + recurrent 1 + snapshots 1 + overhead 96 + KV 3081 (12.0 KiB/token x 262144); max context 262144 (trained 262144)
measured: weights+scratch 1344 MiB (predicted 1345), state 3081 MiB (predicted 3082)
```

Weights plus scratch differ by 1 MiB and state by 1 MiB, both inside the 4 MiB limit. The same line at `--context 512` prints state 12 MiB against 16 MiB after rounding to a whole MiB; the auto-context line above is the check.

## Tokenizer

`llama-tokenize -m <gguf> -f sample.txt --ids` and Ferrum's GGUF tokenizer, same bytes. 19 ids, identical:

```
[35808, 20, 229, 10869, 9, 440, 6763, 355, 229, 9792, 2136, 1679, 6903, 1400, 435, 1283, 5301, 2216, 22]
```

The sample is `Hello, 世界! The price is 12345 — isn't it?` plus a newline and `New line.`

## Greedy continuation

Prompt `The capital of France is`, 16 new tokens, temperature 0. llama.cpp: `llama-completion -c 512 -n 16 --temp 0 --top-k 0 --top-p 1 --min-p 0 --repeat-penalty 1 --no-conversation -ngl 99`. Ferrum: `ferrum hybrid --context 512 --raw --max-new-tokens 16`.

Both token sequences, prompt plus continuation:

```
[597, 5205, 302, 3980, 355, 4741, 20, 309, 278, 5205, 302, 3888, 355, 18943, 22, 440, 5205, 302, 10749, 355, 11210]
```

Text: `The capital of France is Paris, and the capital of Japan is Tokyo. The capital of Brazil is Bras`

## KL divergence

One chunk, `n_ctx=512`, of `README.md` (not a standard corpus; the perplexity is not a quality claim). llama.cpp wrote `/tmp` `base.kld` with `llama-perplexity -c 512 --chunks 1 --kl-divergence-base`. Ferrum: `hybrid_kld <gguf> base.kld 1`.

```
positions 255; Ferrum PPL 158.0000; llama.cpp PPL 160.4486; mean KLD 0.001009; median 0.000822; p99 0.004870; max 0.006760; same top-1 99.22%; 1.3 s
```

## Speculation

`ferrum-cli --context 2048 --temperature 0 --top-k 0 --top-p 1 --min-p 0 --repetition-penalty 1 --no-think --max-tokens 12`, prompt `The capital of France is`.

Plain and `--draft lookup` both answered `The capital of France is **Paris**. It is known for`. Lookup accepted 3 of 3 drafts (acceptance 4.00, counting the anchor).

`--draft lookup --draft-tree 4` exits before the first turn: `draft trees are not supported for this model`.

## Qwen regression

Same command on `origin/experimental` (`ec14e94`, worktree `/tmp/ferrum-exp`) and this branch: `ferrum hybrid --context 512 --raw --prompt "The capital of France is" --max-new-tokens 16`. One model in memory. Not a speed comparison (no interleaved `hybrid_bench`; the no-factor RoPE expression is unchanged).

| File | Generated ids, both builds |
|---|---|
| Swift-1.5-Qwen3.8-27B Q4_K_M (GGUF name "Miroslav 1.0") | `[11751, 13, 198, 760, 6511, 314, 9564, 369, 19241, 13, 198, 760, 6511, 314, 14898, 369]` |
| Tiel-Coder-35B-A3B MTP UD-IQ4_XS (GGUF name "Ornith-1.5-35B") | `[11751, 13, 11751, 369, 279, 7526, 3177, 303, 9338, 321, 369, 7172, 303, 279, 9897, 81183]` |

Swift measured weights+scratch 16679 MiB (predicted 16682), state 182 (predicted 183). Tiel 17114 vs 17116, state 73 vs 74. Paths:

`~/models/hub/models--ukisai--Swift-1.5-Qwen3.8-27B-GGUF/snapshots/a1614465cfa35d04d3e8575d713fa779662b5eab/Swift-1.5-Qwen3.8-27B-Q4_K_M.gguf`

`~/models/hub/models--peculiar-ragdoll--Tiel-Coder-35B-A3B-GGUF-MTP/snapshots/bbe9e566f39e4fc9652ac66b71968289a03c520a/Tiel-Coder-35B-A3B-MTP-UD-IQ4_XS.gguf`

The same command was run again on this branch after the LFM3-VL kernels, one model in memory. Both files produced the ids and the measured lines above. That is the regression check for the shared forward (the RoPE position table). It is not a new KLD and not an interleaved `hybrid_bench`.

## LFM3-VL-2B

Same machine and the same llama.cpp build. Language file and projector (the repo publishes no Q4_K_M projector; the pair is Q4_K_M language weights and a Q8_0 mmproj):

`~/models/hub/models--LiquidAI--LFM3-VL-2B-GGUF/snapshots/main/LFM3-VL-2B-Q4_K_M.gguf` (1,099,627,808 bytes)

`~/models/hub/models--LiquidAI--LFM3-VL-2B-GGUF/snapshots/main/mmproj-LFM3-VL-2B-Q8_0.gguf` (203,031,872 bytes)

### Language memory

`ferrum hybrid` does not load the projector. Auto context, one token, under `caffeinate -dims`:

`./target/release/ferrum hybrid --model <lang.gguf> --raw --prompt hi --max-new-tokens 1 --context auto`

```
memory plan: budget 23552 MiB = weights 1247 + scratch 98 + recurrent 1 + snapshots 1 + overhead 96 + KV 1509 (12.0 KiB/token x 128000); max context 128000 (trained 128000)
measured: weights+scratch 1344 MiB (predicted 1345), state 1509 MiB (predicted 1510)
```

Weights plus scratch differ by 1 MiB and state by 1 MiB.

### Still image, greedy

Picture: 256×256 RGB PNG (`/tmp/lfm3-vl-256.png`, 765 bytes). A red block for `x < 80`, `y < 80`, a blue block for `x > 160`, `y > 160`, off-white elsewhere. 256×256 is exactly `image_min_pixels` (65536) and a multiple of 32, so neither side resizes it. Prompt `Describe the image in one sentence.` Thinking left at the template default (`enable_thinking` true). Both runs under `caffeinate -dims`, context 512, 16 new tokens, temperature 0.

llama.cpp (`llama-mtmd-cli` has no `--no-conversation`):

```
llama-mtmd-cli -m <lang.gguf> --mmproj <mmproj.gguf> --image /tmp/lfm3-vl-256.png \
  -p "Describe the image in one sentence." -c 512 -n 16 --temp 0 --top-k 0 --top-p 1 \
  --min-p 0 --repeat-penalty 1 --no-warmup -ngl 99 --jinja \
  --image-min-tokens 64 --image-max-tokens 64
```

Ferrum:

```
printf '/image /tmp/lfm3-vl-256.png\nDescribe the image in one sentence.\n/exit\n' \
  | ./target/release/ferrum-cli --model <lang.gguf> --mmproj <mmproj.gguf> \
  --context 512 --max-tokens 16 --image-tokens 64 --image-min-tokens 64 \
  --temperature 0 --top-k 0 --top-p 1 --min-p 0 --repetition-penalty 1
```

Both printed the same 16-token sentence:

`The user wants a one-sentence description of the image. Let me carefully observe`

`llama-tokenize -m <lang.gguf> -c 64 --ids --no-bos -f` that sentence:

```
[597, 4695, 10966, 267, 734, 1522, 97365, 8818, 302, 278, 4145, 22, 6134, 622, 12518, 13895]
```

Neither CLI printed those ids itself. llama-mtmd-cli did not print the prompt length, so only the continuation was compared. Ferrum's own line: prompt 83 tokens (0 reused), prefill 1636 tok/s, 16 generated at 103.2 tok/s, stop Length, image encode 0.06 s. llama.cpp reported the image batch (`n_chunks = 1`) encoded in 193 ms. Those times are one run each, not an interleaved bench.

### Still image, resized and not square

Picture: 770×500 RGB PNG (`/tmp/lfm3-vl-770x500.png`, 2299 bytes). A red strip for `x < 48`, a blue strip for `y > 452`, off-white elsewhere. 770 and 500 are not multiples of 32. Rounding each side to the nearest multiple of 32 gives 768×512, which is 384 language tokens, and the pixel count stays inside the file's limits, so that is the whole resize. Same prompt, thinking left on, context 1024, temperature 0, image token budget 64 to 1024.

```
llama-mtmd-cli -m <lang.gguf> --mmproj <mmproj.gguf> --image /tmp/lfm3-vl-770x500.png \
  -p "Describe the image in one sentence." -c 1024 -n 48 --temp 0 --top-k 0 --top-p 1 \
  --min-p 0 --repeat-penalty 1 --no-warmup -ngl 99 --jinja \
  --image-min-tokens 64 --image-max-tokens 1024
```

```
printf '/image /tmp/lfm3-vl-770x500.png\nDescribe the image in one sentence.\n/exit\n' \
  | ./target/release/ferrum-cli --model <lang.gguf> --mmproj <mmproj.gguf> \
  --context 1024 --max-tokens 48 --image-tokens 1024 --image-min-tokens 64 \
  --temperature 0 --top-k 0 --top-p 1 --min-p 0 --repetition-penalty 1
```

Both printed the same continuation. It names the geometry: a vertical red stripe on the left.

```
The user wants a one-sentence description of the image. Let me carefully observe the image.

The image shows a flag-like design. There's a vertical red stripe on the left side, a large white/off-white central area, and
```

`llama-tokenize -m <lang.gguf> -c 64 --ids --no-bos -f` that text is 48 ids:

```
[597, 4695, 10966, 267, 734, 1522, 97365, 8818, 302, 278, 4145, 22, 6134, 622, 12518, 13895, 278, 4145, 63908, 597, 4145, 3926, 267, 11483, 8364, 1788, 22, 1978, 589, 267, 12324, 2959, 117309, 383, 278, 2690, 2891, 20, 267, 2110, 3789, 23, 2986, 32922, 4623, 2267, 20, 309]
```

Ferrum: prompt 403 tokens (0 reused), prefill 2336 tok/s, 48 generated at 95.8 tok/s, stop Length, image encode 0.39 s. llama.cpp encoded the batch in 113 ms. One run each. The first 16 ids are the square image's continuation; the rest is where this picture shows up.
