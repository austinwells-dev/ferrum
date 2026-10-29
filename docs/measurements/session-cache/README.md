# Session cache on disk

Branch `fix/session-cache-reuse` (working tree after `a5a8ad4`), 29 October 2026. Machine: MacBook Pro Mac17,2, Apple M5, 10 CPU cores, 32 GB, Metal 4, macOS 27.0.1, internal 1 TB SSD (APFS). One model in memory at a time, screen unlocked (checked before every model), `caffeinate -dims`, `scripts/phase7a_cpu_preflight.py` passed first with `PHASE7A_PREFLIGHT_IGNORE=dasd`.

## Round trip

[disk-2026-10-29.sh](disk-2026-10-29.sh) runs [`hybrid_session_disk`](../../../examples/hybrid_session_disk.rs) with `--lengths 2000,8000,32000` (filler sizes; the real prompt lengths are in the table). For each length: a chat turn (prefill), `save_session`; turn 2 in memory for comparison; the runtime is dropped and the model loaded again (reload), `load_session` (load), the first request again (it must reuse everything and give the same output), then `load_session` again and turn 2 from disk (it must reuse what turn 2 did in memory) and an incremental save. Raw output: [disk-2026-10-29.txt](disk-2026-10-29.txt). Every model passed.

| Model | Prompt tokens | Prefill s | Save s | Save MiB | Model reload s | Load s | Turn 2 save MiB |
|---|---|---|---|---|---|---|---|
| LFM3-VL-2B Q4_K_M | 2356 | 0.94 | 0.057 | 54.4 | 0.5 | 0.022 | 27.0 |
| | 9356 | 4.10 | 0.104 | 136.4 | 0.5 | 0.037 | 27.0 |
| | 39022 | 23.82 | 0.339 | 484.2 | 0.5 | 0.096 | 27.1 |
| Spark-X2.6-4B Q4_K_M | 3087 | 3.18 | 0.229 | 339.7 | 0.8 | 0.071 | 231.6 |
| | 12492 | 15.14 | 0.428 | 670.4 | 0.8 | 0.124 | 231.7 |
| | 51761 | 101.64 | 0.505 | 2051.1 | 0.9 | 0.342 | 231.8 |
| Qwen3.6-35B-A3B IQ4_XS | 3032 | 3.86 | 0.189 | 248.6 | 6.4 | 0.100 | 189.7 |
| | 12437 | 18.33 | 0.296 | 432.3 | 6.3 | 0.138 | 189.7 |
| | 51706 | 122.37 | 0.441 | 1199.4 | 6.3 | 0.295 | 189.9 |
| Tiel-Coder-35B-A3B UD-IQ4_XS | 3207 | 4.32 | 0.184 | 252.0 | 6.0 | 0.099 | 189.5 |
| | 12612 | 18.36 | 0.290 | 435.7 | 6.0 | 0.139 | 189.8 |
| | 51881 | 121.94 | 0.437 | 1202.9 | 5.8 | 0.302 | 189.9 |
| Swift-1.5-Qwen3.8-27B Q4_K_M | 3070 | 18.67 | 0.526 | 643.7 | 7.2 | 0.238 | 453.1 |
| | 12475 | 82.86 | 0.707 | 1231.5 | 7.3 | 0.351 | 453.1 |
| | 51744 | 449.44 | 1.134 | 3686.0 | 7.1 | 0.903 | 453.3 |

Tiel with DFlash2 (`--draft-max 2`) and with MTP gave the same reuse and sizes, with save and load times within 0.4 s and 0.01 s of plain Tiel; their rows are in the raw output. A drafter starts over from the loaded tokens (`Drafter::adopt`), so its first steps after a load draft from less context; output is unaffected.

Loading is 40× (LFM3, 2k tokens) to 500× (Swift, 52k) faster than prefilling the same conversation. The model reload itself (reading the weights again, which the TUI does around a picture when the chat model doesn't fit beside the image model) takes 6–7 s on the 27B and 35B models whether or not the cache is used. Load times were taken right after the save, so the files may have come from the page cache rather than the SSD. At the SSD's several GB/s sequential read that changes them by well under a second.

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
