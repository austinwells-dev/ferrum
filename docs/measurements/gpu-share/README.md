# GPU share (`--gpu-share`)

Branch `feat/smooth-mode-and-qol` at `f0d60bd`, 15 November 2026. Machine: Apple M5, 32 GB, macOS 27.2.1. Model: `Spark-X2.6-4B-Q4_K_M.gguf` (XHToken), F16 K/V, context 8192. One run per setting, other applications open, so these show the pacing and are not a benchmark.

```sh
P="Summarize this text in two sentences. $(printf 'The quick brown fox jumps over the lazy dog near the river bank at dawn. %.0s' $(seq 60))"
ferrum-cli -m Spark-X2.6-4B-Q4_K_M.gguf --context 8192 --no-think --temperature 0 \
  --max-tokens 100 --gpu-share off|50|25 -p "$P"
```

| `--gpu-share` | Prompt tokens | Prefill tok/s | Generated | Decode tok/s | Reply (md5) |
|---|---|---|---|---|---|
| off | 985 | 1095 | 26 | 36.0 | a5231279b8bcd187ef029209076ee5fa |
| 50 | 985 | 537 | 26 | 15.9 | a5231279b8bcd187ef029209076ee5fa |
| 25 | 985 | 271 | 26 | 7.8 | a5231279b8bcd187ef029209076ee5fa |

The reply is the same text at every setting. Speeds include the pauses. Decode at 50 is under half of plain decoding because the first step after the prompt (which also snapshots) is paced too, over only 26 tokens.

**Not measured:** whether a given share keeps a particular video smooth. The pauses give other GPU clients the device between steps; how long a step holds it depends on the model (a decode step here is about 28 ms, a 128-token prompt piece about 120 ms).

## Slot caches across a restart (`--slot-save-path`)

Same build and model, `ferrum-server --context 8192 --slot-save-path DIR`, one chat request of 35 prompt tokens and 60 generated, then SIGINT, a restart and the same request:

```
INF signal received: finishing up (again to quit now)
INF slot 0: saved 94 cached tokens (232.8 MiB) in 0.2 s
INF shutting down
...
INF slot 0: restored 94 cached tokens
```

The repeated request reported `cached_tokens: 34` of 35 (an identical retry recomputes its last token).
