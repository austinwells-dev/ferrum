# Session cache on disk

Branch `fix/session-cache-reuse` (working tree after `7fada2f`, with the fixes from the review of austinwells-dev/ferrum#12), 29 October 2026. Machine: MacBook Pro Mac17,2, Apple M5, 10 CPU cores, 32 GB, Metal 4, macOS 27.0.1, internal 1 TB SSD (APFS). One model in memory at a time, screen unlocked (checked before every model), `caffeinate -dims`, `scripts/phase7a_cpu_preflight.py` passed first with `PHASE7A_PREFLIGHT_IGNORE=dasd`.

## Round trip

[disk-2026-10-29.sh](disk-2026-10-29.sh) runs [`hybrid_session_disk`](../../../examples/hybrid_session_disk.rs) with `--lengths 2000,8000,32000` (filler sizes; the real prompt lengths are in the table). For each length: a chat turn (prefill), `save_session`; turn 2 in memory for comparison; the runtime is dropped and the model loaded again (reload), `load_session` (load), the first request again (it must reuse everything and give the same output), then `load_session` again and turn 2 from disk (it must reuse what turn 2 did in memory) and an incremental save; then turn 1 again from that save and a save of it, which is shorter than the save it appends to: it must load and give the same output. The last four models ran again with `FERRUM_KV_CACHE=q8`, at 2k and 8k. Raw output: [disk-2026-10-29.txt](disk-2026-10-29.txt). Every run passed every check.

| Model | Prompt tokens | Prefill s | Save s | Save MiB | Model reload s | Load s | Turn 2 save MiB |
|---|---|---|---|---|---|---|---|
| LFM3-VL-2B Q4_K_M | 2356 | 0.94 | 0.055 | 54.4 | 0.5 | 0.024 | 27.0 |
|  | 9356 | 3.99 | 0.095 | 136.4 | 0.4 | 0.037 | 27.0 |
|  | 39022 | 22.34 | 0.316 | 484.2 | 0.5 | 0.085 | 27.1 |
| Spark-X2.6-4B Q4_K_M | 3087 | 2.99 | 0.216 | 339.7 | 0.9 | 0.063 | 231.6 |
|  | 12492 | 13.95 | 0.422 | 670.4 | 0.9 | 0.110 | 231.7 |
|  | 51761 | 93.16 | 0.516 | 2051.1 | 0.9 | 0.294 | 231.8 |
| Qwen3.6-35B-A3B IQ4_XS | 3032 | 3.41 | 0.189 | 248.6 | 6.2 | 0.096 | 189.7 |
|  | 12437 | 16.51 | 0.313 | 432.3 | 6.2 | 0.125 | 189.7 |
|  | 51706 | 112.12 | 0.471 | 1199.4 | 6.0 | 0.278 | 189.9 |
| Tiel-Coder-35B-A3B UD-IQ4_XS | 3207 | 3.65 | 0.196 | 252.0 | 5.8 | 0.091 | 189.5 |
|  | 12612 | 16.56 | 0.304 | 435.7 | 5.9 | 0.143 | 189.8 |
|  | 51881 | 112.36 | 0.449 | 1202.9 | 6.0 | 0.305 | 189.9 |
| Tiel + DFlash2 (`--draft-max 2`) | 3207 | 3.69 | 0.177 | 252.0 | 6.2 | 0.095 | 189.5 |
|  | 12612 | 16.75 | 0.304 | 435.7 | 6.3 | 0.124 | 189.8 |
|  | 51881 | 113.28 | 0.447 | 1202.9 | 6.1 | 0.274 | 189.9 |
| Tiel + MTP | 3207 | 3.76 | 0.184 | 252.0 | 6.6 | 0.094 | 189.5 |
|  | 12612 | 17.22 | 0.316 | 435.7 | 6.7 | 0.126 | 189.8 |
|  | 51881 | 119.50 | 0.443 | 1202.9 | 6.5 | 0.269 | 189.9 |
| Swift-1.5-Qwen3.8-27B Q4_K_M | 3070 | 16.71 | 0.500 | 643.7 | 7.1 | 0.190 | 453.1 |
|  | 12475 | 75.28 | 0.697 | 1231.5 | 7.0 | 0.328 | 453.1 |
|  | 51744 | 410.19 | 1.471 | 3686.0 | 7.0 | 0.930 | 453.3 |
| LFM3-VL-2B, q8 K/V | 2356 | 0.82 | 0.037 | 29.7 | 0.4 | 0.018 | 15.2 |
|  | 9356 | 3.53 | 0.061 | 73.4 | 0.4 | 0.030 | 15.2 |
| Spark-X2.6-4B, q8 K/V | 3087 | 2.80 | 0.125 | 180.5 | 0.7 | 0.045 | 123.1 |
|  | 12492 | 13.12 | 0.223 | 356.2 | 0.6 | 0.079 | 123.1 |
| Qwen3.6-35B-A3B, q8 K/V | 3032 | 3.35 | 0.174 | 220.4 | 5.7 | 0.093 | 189.1 |
|  | 12437 | 15.92 | 0.227 | 318.0 | 5.9 | 0.137 | 189.1 |
| Swift 27B, q8 K/V | 3070 | 16.72 | 0.397 | 552.4 | 6.1 | 0.195 | 451.1 |
|  | 12475 | 73.55 | 0.669 | 864.7 | 6.3 | 0.276 | 451.1 |

A drafter starts over from the loaded tokens (`Drafter::adopt`), so its first steps after a load draft from less context; output is unaffected.

Loading is 40× (LFM3, 2k tokens) to 440× (Swift, 52k) faster than prefilling the same conversation. The model reload itself (reading the weights again, which the TUI does around a picture when the chat model doesn't fit beside the image model) takes 6–7 s on the 27B and 35B models whether or not the cache is used. Load times were taken right after the save, so the files may have come from the page cache rather than the SSD. At the SSD's several GB/s sequential read that changes them by well under a second.

A save writes the full-attention K/V rows (appending only the rows added since the last save when the conversation grew), every snapshot not already on disk, and the live state at the end: the recurrent state plus the sliding-window rings. A later save therefore writes about one snapshot-sized file and one live-state file, independent of the conversation's length: 27 MiB on LFM3, 190 MiB on the 35B MoE models, 232 MiB on Spark (its rings are large) and 453 MiB on Swift.

## SSD wear

Writes happen only when a chat's cache would otherwise be lost: Reset, `/new`, `/clear`, `/resume` switching away, a picture unloading the model, and closing the chat. Reads do not wear flash.

This machine's SSD reports, through `smartctl -a disk0`, 19.4 TB written so far and "Percentage Used: 1%", so on its own counter roughly 20 TB is about 1% of its rated life. Apple publishes no TBW figure; this ratio is the drive's own.

| Use | Assumption | Written per day | Per year | Rated life per year |
|---|---|---|---|---|
| Light, 35B MoE | 10 later saves, 2 new 12k-token chats | 2.9 GB | 1.1 TB | about 0.05% |
| Heavy, 35B MoE | 50 later saves, 10 new 52k-token chats | 22.6 GB | 8.2 TB | about 0.4% |
| Heavy, Swift 27B | 50 later saves, 10 new 52k-token chats | 62.4 GB | 22.8 TB | about 1.1% |

The cap of 16 GiB across all chats bounds space, not writes: pruning deletes files and writes nothing.
