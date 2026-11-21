# Inkling 2 Mini

Working tree on `feat/inkling2-mini` (parent `0c20d64`, `experimental`), 20–21 November 2026. Machine: MacBook Pro, Apple M5, 10 CPU cores, 32 GB, macOS 27.2. Files, from [`unsloth/Inkling-2-Mini-GGUF`](https://huggingface.co/unsloth/Inkling-2-Mini-GGUF):

- `Inkling-2-Mini-UD-Q4_K_XL.gguf` (17,629,759,456 bytes; `general.architecture = inkling2`; Q8_0 attention, NVFP4 experts)
- `mmproj-F16.gguf` (639,348,160 bytes; `inkling2v`)

Reference: Unsloth's llama.cpp build `b12568-mix-8eebad4`, built from source with Metal. llama.cpp master garbles these files. One model in memory at a time.

## Greedy continuation

Prompt `The capital of France is`, raw, temperature 0. llama.cpp: [greedy-llama.txt](greedy-llama.txt). Ferrum: `ferrum hybrid --model <gguf> --raw --prompt "The capital of France is" --max-new-tokens 24`, [greedy-ferrum-capital.txt](greedy-ferrum-capital.txt). The same text: " Paris." then the multiple-choice list `A. Noun clause` … `D.`.

## Memory plan

From the same run:

```
measured: weights+scratch 17142 MiB (predicted 17144), state 5798 MiB (predicted 5799)
```

Within 2 MiB and 1 MiB. Context 177,408 without a projector; `ferrum-cli` with `mmproj-F16.gguf` fits 137,728.

## KL divergence

Base: `llama-perplexity -m <gguf> -f corpus.txt -c 512 --chunks 8 --kl-divergence-base inkling-metal.kld` ([log](llama-perplexity-metal.txt)), on 114 KB of English prose. Ferrum: `cargo run --release --example hybrid_kld -- <gguf> inkling-metal.kld`.

| Comparison | Mean KLD | Median | p99 | Same top-1 | PPL Ferrum / llama.cpp |
|---|---|---|---|---|---|
| vs llama.cpp Metal, 8 chunks ([log](kld-vs-llama-metal.txt)) | 0.002019 | 0.000468 | 0.017313 | 98.28% | 5.1209 / 5.1001 |
| vs llama.cpp CPU, 8 chunks ([log](kld-vs-llama-cpu.txt)) | 0.226889 | 0.032130 | 2.697537 | 86.67% | 5.1209 / 5.7649 |

The reference's CPU backend disagrees with its own Metal backend: its perplexity is 5.77 against 5.12 ([CPU log](llama-perplexity-cpu.txt)), so the Metal base is the one used.

Attention paths, first 2 chunks against the Metal base, `FERRUM_KLD_ROWS=N` rows per forward:

| Rows | Kernels | Mean KLD | Same top-1 | Log |
|---|---|---|---|---|
| 64 (default) | staged prompt | 0.002018 | 98.63% | [kld-rows-64.txt](kld-rows-64.txt) |
| 8 | verify | 0.002166 | 98.24% | [kld-rows-8.txt](kld-rows-8.txt) |
| 1 | decode | 0.002194 | 98.82% | [kld-rows-1.txt](kld-rows-1.txt) |

## Sessions

`cargo run --release --example hybrid_session_check -- <gguf>`: every case identical to a fresh session, PASS ([log](session-check.txt)).

## Vision

`MTMD_DEBUG_EMBEDDINGS=ref.bin llama-mtmd-cli -m <gguf> --mmproj mmproj-F16.gguf --jinja --image IMG -p hi -n 1` dumps the projector output; `cargo run --release --example hybrid_vision_parity -- mmproj-F16.gguf IMG ref.bin [MAX_TOKENS]` compares Ferrum's ([log](vision-parity.txt)). The images are llama.cpp's `tools/mtmd/test-1.jpeg` (640×488) and PNGs made from it with `sips`.

| Image | Tokens | Cosine | Relative RMS | Worst token cosine |
|---|---|---|---|---|
| `test-1.png`, 640×488 (padded to 640×520) | 208 | 0.999987 | 0.0052 | 0.99925 |
| `sq640.png`, 640×640 | 256 | 0.999971 | 0.0077 | 0.99857 |
| `pre.png`, 1680×960 (position table upsampled) | 1008 | 0.999960 | 0.0090 | 0.98327 |
| `big.png`, 2600×1500, both scale to 1680×960 (cap 1024) | 1008 | 0.999049 | 0.0437 | 0.78685 |
| `test-1.jpeg` | 208 | 0.999473 | 0.0325 | 0.97698 |

On identical pixels the tower agrees closely. The last two rows differ before the tower: llama.cpp's Pillow-style bicubic resize rounds to 8 bits between passes, and stb_image decodes JPEG slightly differently from the `image` crate. An image at or under the cap is never resized, as upstream Inkling does.

End to end, `ferrum-cli` described `test-1.jpeg` (the 1969 New York Times front page) correctly; its image took 0.38 s to encode.

## Speed

`caffeinate -dims` with the screen unlocked (checked around every run), after `scripts/phase7a_cpu_preflight.py` (`PHASE7A_PREFLIGHT_IGNORE=dasd`). Ferrum: `hybrid_bench <gguf> --pp 512,2048 --tg 128`. llama.cpp: `llama-bench -m <gguf> -p 512,2048 -n 128 -r 3`. Two interleaved rounds ([log](bench-abba.txt), end of file):

| tok/s | Ferrum round 1 / 2 | llama.cpp round 1 / 2 |
|---|---|---|
| pp512 | 535.70 / 531.92 | 259.60 / 259.60 |
| pp2048 | 501.45 / 497.75 | 221.56 / 221.87 |
| tg128 | 24.24 / 23.97 | 23.08 / 23.10 |

## Regression on other models

The change touches the shared 128- and 256-wide attention templates (the `_rel` variants are compile-time branches of them), the flash-attention routing and the forward pass. Base: a build of `0c20d64`; branch: this tree. Files: `Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive-IQ4_XS.gguf` (MoE) and `Spark-X2.6-4B-Q4_K_M.gguf` (dense). Logs in [regression/](regression/).

- **KLD**, 8 chunks against `llama-perplexity` bases made by the same reference build on the same corpus: identical to every printed digit on both. Qwen3.6: mean 0.002011, top-1 98.43%, PPL 9.0468. Spark: mean 0.000145, top-1 99.36%, PPL 9.2611.
- **Greedy**, 128 tokens from `The history of the printing press begins`: identical output and identical memory lines on both.
- **`hybrid_spec_check`** on Qwen3.6: same output as base (speculative output equals greedy; replay after commit differs by 0.0000).
- **Speed**, A-B-B-A, two rounds ([bench-abba.txt](bench-abba.txt)), best of four per side:

| tok/s | Qwen3.6 base | Qwen3.6 branch | Spark base | Spark branch |
|---|---|---|---|---|
| pp512 | 938.88 | 928.91 | 1176.10 | 1181.08 |
| pp2048 | 907.32 | 903.39 | 1104.18 | 1106.70 |
| tg128 | 53.70 | 52.89 | 39.65 | 39.63 |

Qwen3.6 drifted during that run (base pp512 ranged 788–939, tg128 51.1–53.7). Two more A-B-B-A rounds of pp512 and tg128 alone ([regression/bench-qwen36-decode-abba.txt](regression/bench-qwen36-decode-abba.txt)) gave base 940.74 / 54.96 and branch 951.51 / 55.14 (best of four), so the first run's gap is within its spread.
