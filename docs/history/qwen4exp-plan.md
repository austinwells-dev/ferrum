# Qwen3.8-Flash-Next (`qwen4exp`) on Ferrum: a design from Strata (2026-11-02)

A plan, not a guide. Nothing here is implemented yet. Numbers marked **Strata** were measured by Strata's authors on their machines and are quoted from its docs. Numbers marked **est.** are arithmetic from the GGUF header and are not measurements. Everything else was read from the files on this Mac.

## Why the port is back on the table

On 2026-10-11 the port was shelved because the smallest GGUF looked like 72.5 GB and this Mac has 32 GB. [Strata](https://github.com/niko1221/strata) (MIT, C++/Metal) shows that this was the wrong conclusion. It runs the model on a **32 GB M4 Mac mini at 13.5–14.1 tok/s decode and 93.6 tok/s on a 4K prompt** (Strata, `docs/MAC.md`, Q2_0, no MTP, medians of 5 interleaved runs). llama.cpp on CPU on the same Mac manages 6.3 tok/s. Two ideas make that possible:

1. **The n-gram table (28.8 GB) never enters memory.** Every token reads 16 rows of 90 bytes straight from the SSD (`F_NOCACHE`, 4 KiB reads).
2. **The routed experts (34 GB) stream from the SSD** into a fixed arena of the most-used ones (11.5–14.7 GB on Strata's M4). The rest are read as each token routes to them, with read-ahead for the next layer.

Strata's own setup has already downloaded the files to this Mac:

| File | Size | Role |
|---|---:|---|
| `~/Documents/Strata-data/models/Q2_0/…-Q2_0-00001-of-00002.gguf` | 37.62 GB | Everything except the n-gram table. `general.architecture = qwen4exp` |
| `…-00002-of-00002.gguf` | 28.80 GB | `per_layer_token_embd.weight` only, IQ4_NL, 160 × 320,001,536 |
| `…-00001-of-00002.gguf.experts` | 33.97 GB | Strata's expert-contiguous copy ("STRX" header): each expert's gate, up and down sit together, so one read loads one expert |
| `mmproj-Qwen3.8-Flash-Next-BF16.gguf` | 0.91 GB | Vision projector |

The device reports `recommended_working_set` 25,769,803,776 B (24 GiB) and `max_buffer` 20,100,448,256 B (`ferrum info`, Apple M5, 32 GB). The disk has 295 GiB free.

## Strata in one page

Strata is three products: a Tauri app, a Python terminal app and server, and a C++ engine with CUDA, HIP, SYCL and Metal backends. Ferrum already has its own equivalents of the first two (`ferrum-tui`, `ferrum-server`). **Only the engine ideas carry over.** The parts that matter for us:

| Strata part | Where | What it teaches |
|---|---|---|
| Model geometry | `include/strata/core/geometry.hpp` | Every dimension, and which GGUF key it comes from |
| n-gram hash and table reader | `include/strata/kernels/ngram.hpp`, `include/strata/ngram/ple_reader.hpp` | The hash (ported from llama.cpp's `qwen4exp.cpp` L1092–1124) and the split issue/collect I/O API |
| PLE block | `metal/src/engine/msl/ple.metal` | The math after the two projections |
| Hyper-connections | `metal/kernels/gr.metal`, `gr2.metal` | `gr_norm`, `gr_down`, `gr_up`, and the write folded into the next read |
| QSA (sparse attention) | `metal/kernels/qsa.metal`, `qsa_prompt.metal` | Indexer, block scores, top-k selection, split-K attention |
| GDN | `docs/GDN_CONTRACT.md`, `metal/kernels/gdn*.metal` | The same Gated DeltaNet Ferrum already runs, with this model's shapes |
| Expert streaming on a Mac | `metal/src/engine/stream.cpp`, `arena_budget.h` | Arena sizing against macOS memory pressure, the experts' copy, read-ahead |
| Q2_0 | `metal/kernels/quant.metal` (`T_Q2_0`) | 64 weights per 18-byte block: an f16 scale, then 2-bit codes meaning {−1, 0, +1} |

Strata's engine is a port of llama.cpp's `qwen4exp` graph. **Ferrum should treat llama.cpp's `qwen4exp.cpp` as the reference semantics**, as we do for every other model, and Strata as a design source and a list of traps. Strata's CPU expert pool (AVX-512 kernels computing cold experts on a PC) doesn't apply: on unified memory the GPU reads every expert once it is in RAM.

## The model, as the file declares it

From the GGUF header of shard 1 (1,223 tensors) and Strata's geometry struct:

| | Value | GGUF key |
|---|---|---|
| Layers | 48: 36 GDN + 12 QSA (`layer % 4 == 3`) | `block_count`, `full_attention_interval`, `attention.compress_ratios` |
| Hidden | 2,560; vocab 248,320 (Qwen3.5's tokenizer, `pre = qwen35`) | `embedding_length` |
| GDN | 16 K heads, 48 V heads, head 128, conv 4, V head `h` reads K head `h % 16` | `ssm.*` |
| QSA | 24 Q heads, 2 KV heads, head 256, RoPE on 64 dims (MRoPE sections 11/11/10), base 1e7, q/k norm, sigmoid output gate packed in `attn_q` (12,288 = 2 × 24 × 256) | `attention.*`, `rope.*` |
| QSA indexer | 4 heads × 128, one pooled key per 4 cells, top-k 2,048 cells | `attention.indexer.*` |
| Hyper-connections | 4 residual streams (the residual is 4 × 2,560 = 10,240 wide), low rank 320, separate weights before the mixer and before the FFN, and `output_hc_*` before the head | `hyper_connection.*` |
| MoE | 512 experts, top-10, expert FFN 640, one sigmoid-gated shared expert of 640 | `expert_*` |
| PLE (n-gram embedding) | Layer 1 only. 16 heads (8 bigram, 8 trigram) × 160 dims = 2,560. Hash constants are in the header | `ple.*`, `embedding_length_per_layer_input` |
| MTP | **Not in this file.** Strata builds its draft layer from the original checkpoint (~5 GB) | |

Weight types in the Q2_0 pack (count of tensors): BF16 483, F32 292, Q2_0 202, Q3_K 91, IQ4_XS 56, Q4_K 38, Q5_K 16, Q4_0 16, Q6_K 12, Q5_0 7, IQ4_NL 7, Q8_0 2, F16 1. Q2_0 isn't only the routed experts: some shared experts, `ple_key` and some `attn_q` are Q2_0 too.

One token, one layer `L`:

```
R  : [4][2560]                                     the residual, 4 streams
if L == 1:  R += PLE(n-gram rows of the last 3 tokens, R)
x  = gr_read(R, hc_attn_*)  -> mixed [2560], inject [4]
y  = GDN(x)  or  QSA(x)                            (L % 4 == 3 is QSA)
R[c] += y * 2·sigmoid(inject[c] / 4)               gr_write
x  = gr_read(R, hc_ffn_*)
y  = Σ_top10 w_e · expert_e(x)  +  sigmoid(g·x) · shared(x)
R[c] += y * 2·sigmoid(inject[c] / 4)
...
logits = output(gr_read(R, output_hc_*))           no output_norm tensor; the read normalizes
```

## What Ferrum already has, and what is new

| Piece | Ferrum today | Work |
|---|---|---|
| GDN layers, 3:1 with full attention | `qwen35`/`qwen35moe`, the recurrent kernel, the verify tape | **Reuse.** The `grouped = 0` (GGUF `h % k_heads`) path already exists in `hybrid.metal`. New shapes: 16/48 heads |
| Gated attention, head 256, partial RoPE, q/k norm, MRoPE | `qwen35` full attention, Qwen3-VL MRoPE | **Reuse** for QSA at ≤ 2,051 cells, where QSA *is* dense attention |
| Routed MoE + sigmoid-gated shared expert, softmax top-k | `Variant::Moe` | **Extend**: 640-wide experts (`config.rs` rejects `expert_ffn % 256 != 0`), slot-indirect expert reads |
| Q3_K, Q4_K, Q5_K, Q6_K, Q4_0, Q8_0, IQ4_XS, BF16, F16 | GEMV and prefill kernels | **Reuse** |
| Q2_0, Q5_0 | none | **New** GEMV, prefill dequant, MoE variants, CPU parity |
| Q8 K/V cache | planar Q8 | **Reuse** for QSA K/V |
| Sessions, snapshots, prefix reuse, disk save | `session` | **Extend**: PLE history, indexer tail |
| Memory planner | `plan` (predicts within 4 MiB) | **Extend**: arena, I/O staging, PLE row cache, RAM headroom |
| Speculative decoding | MTP trees, DFlash, lookup | Lookup only at first; see [Speculation](#speculation-under-streaming) |
| Split GGUFs | One file per model | **New**, but only as a companion file for the n-gram table |
| Hyper-connections, PLE, QSA indexer, expert streaming | none | **New** |

## Memory on this Mac (est.)

| Part | Bytes | How |
|---|---:|---|
| Dense weights (everything in shard 1 except routed experts) | 3.65 GB | 37.62 − 33.97 |
| Routed experts | 33.97 GB | 48 × 512 × 1.3824 MB (3 × 2560 × 640 weights × 18/64 B) |
| Expert bytes touched per token | 663 MB | 10 × 48 × 1.3824 MB |
| n-gram rows per token | 64 KiB of I/O | 16 rows, one 4 KiB page each |
| QSA K/V, Q8, per token | ~13 KB | 12 layers × 2 heads × 256 × (K + V) × 34/32 B → ~430 MB at 32K |
| Indexer pooled keys | ~50 MB at 32K | 12 × 8,192 blocks × 128 × 4 B |
| GDN state | 113 MB (+4 MB conv) | 36 × 48 × 128 × 128 × 4 B |
| Snapshot | ~118 MB each | GDN state + conv + PLE history (9 × 10,240 × 4 B) + indexer tails |
| Residual scratch at chunk 1,024 | ~42 MB per buffer | 1,024 × 10,240 × 4 B |

The 24 GiB working set minus dense weights, K/V, state, two snapshots, scratch (~1.5 GB) and the PLE row cache leaves about 17 GB. RAM is the tighter limit. Strata keeps `max(7 GB, 15% of RAM)` for macOS, after a 34-minute soak swapped with less. That leaves an **arena of about 12–14 GB, or 8,700–10,000 of the 24,576 experts** (est.), in line with Strata's measured 11.5–14.7 GB on a 32 GB M4. It fits in one Metal buffer (max 20.1 GB).

## Design

### Config and weights

- `config.rs`: `Variant::Qwen4Exp` for `general.architecture = "qwen4exp"`, with three new sub-configs read from the header and cross-checked, as Strata's `read_geometry` does (`full_attention_interval` against `compress_ratios`, `n_ff_shexp == n_ff`):

  ```rust
  pub struct HyperConn { pub streams: usize, pub low_rank: usize }          // 4, 320
  pub struct Indexer { pub heads: usize, pub key_dim: usize, pub block: usize, pub top_k: usize } // 4, 128, 4, 2048
  pub struct Ple {
      pub layer: usize,                 // 1
      pub ngram: usize,                 // 3
      pub heads_per_ngram: usize,       // 8 -> 16 heads
      pub row_dim: usize,               // 160
      pub conv_kernel: usize,           // 4
      pub eos: u32,                     // 248044, which cuts the n-gram context
      pub multipliers: Vec<u64>,
      pub vocab_sizes: Vec<u64>,        // primes near 2e7, not powers of two
      pub offsets: Vec<u64>,
  }
  ```

- `loader/gguf.rs`: type 42 (Q2_0, 64 elements, 18 B) and 20 (IQ4_NL, 32 elements, 18 B) in the block table.
- `weights.rs`: `Mixer::Qsa { attn: Attention, indexer: IndexerWeights }`, per-layer `HcWeights { attn, ffn }` (BF16 down/up/inject, F32 norm), `PleWeights` on layer 1, and `Ffn::Moe` whose routed tensors are an `ExpertSource` rather than loaded buffers:

  ```rust
  pub enum ExpertSource {
      Resident(MoeTensors),            // existing qwen35moe path
      Streamed { layer: usize },       // bytes live in the file; the arena resolves (layer, expert) -> slot
  }
  ```

- The n-gram table is opened as a **companion file**: the runtime finds `-00002-of-00002.gguf` beside the model, checks that it holds exactly `per_layer_token_embd.weight` with the expected shape, and never reads it through the tensor loader.

### Hyper-connections

The residual becomes `[rows, 4, 2560]`. Every other buffer stays 2,560 wide, because every mixer and FFN reads the single `mixed` activation that `gr_read` makes. Kernels, following `gr.metal`'s contract (F32 activations, BF16 weights):

- `h_gr_norm`: per stream, apply the pending write `R'[c] = R[c] + bo · 2σ(inj_prev[c] / 4)`, then RMS-normalize with `hc_*_norm`. Folding the previous sublayer's write into the next read saves one pass over 40 KB per row.
- `h_gr_down`: `lo = silu(W_down · xn / 4)` (320 wide), `inject[c] = W_inject[c] · xn`.
- `h_gr_up`: `mixed[d] = mean_c xn[c][d] · σ(W_up[c·n + d] · lo)`.

Strata found that a Q8_0 copy of the BF16 hyper-connection weights costs a KL of 0.0007 against BF16 (Strata, M5 Max). Start exact (BF16), and measure Q8 later as an option.

Drafters that read the residual (`draft`, `mtp`) need to know it is 10,240 wide. None of the existing drafters are trained for this model, so that is only a check that the code refuses them.

### PLE: the n-gram embedding

New module `src/hybrid/ple.rs`, all safe Rust:

- `ngram_rows(token, prev: [i32; 2]) -> [u32; 16]`: 64-bit multiply and XOR over the last 1, 2 or 3 tokens, modulo each head's prime vocabulary, plus that head's offset. It runs on the host because it needs 64-bit integers and XOR. Golden vectors come from llama.cpp.
- `PleTable`: one I/O thread with a bounded queue. `issue(rows) -> Ticket` as soon as the token id is known, and `collect(ticket)` before layer 1. Pages are deduplicated and sorted. A clock-evicted row cache (Strata: 1M rows ≈ 95 MB would have served up to ~82% of reads on its corpus) is optional, and the planner budgets it. Rows decode on the CPU (IQ4_NL: 90 B → 160 f32, **split-half nibble order**) into a 10 KB upload per row.
- A keep-alive read when the table is idle, because some SSDs take 50–150 ms to wake (Strata saw it on a WD SN7100). Probably irrelevant on the internal Apple SSD: measure it before adding it.
- GPU side (`h_ple_gnorm`, `h_ple_gate_norm`, `h_ple_conv_add`): key and query grouped-norms, a scalar gate per stream `σ(sign(s)·√max(|s|, 1e-6))`, then a depthwise 4-tap conv over rows `t−9, t−6, t−3, t` of a 9-row history. It is added to `R` in place, before layer 1's `gr_read`.
- Latency: the 16 reads can only overlap the embedding and layer 0. Issue them for every row of a verify window, and for the whole prompt chunk during prefill.

### QSA: sparse full attention

For a context of **2,051 cells or fewer, QSA selects every cell and is exactly dense attention.** So bring-up can run the whole model on Ferrum's existing gated-attention kernels, provided the indexer keys are still written, so the session stays valid past 2,051 cells.

Above that, each query attends only to the 2,048 cells its indexer picks:

1. `h_qsa_index_append`: indexer K projection (BF16), RMS norm, RoPE. Raw keys go into a per-layer tail of 4 cells. When a block of 4 completes, write `pooled[b] = rope(rmsnorm(mean of the 4 raw keys) · k_norm, position of the block's first cell)`. The incomplete tail block scores with its own `dead` key and gets +1e9, so it is always selected.
2. `h_qsa_score`: per query row and block, `Σ_heads relu(q_idx · key_b)`.
3. `h_qsa_topk`: radix select of the 2,048 cells with the largest score (a cell inherits its block's score). Ties go to the lowest cell, and the result is emitted in ascending order.
4. Attention over the selected cells only: a gather variant of Ferrum's flash-decode kernel that takes an index list (512 block ids per row, since selection is block-aligned) and keeps the split-K partials and log-sum-exp merge it already has.

Decode cost is then flat in context beyond 8K. Prefill needs a per-row block mask through the chunked flash kernel (Strata's `qsa_prompt.metal` is the model). That is phase 3, because a context limit of 2,048 is enough until then.

### MoE and expert streaming

This is the core change and the only part that alters Ferrum's execution model. New module `src/hybrid/stream/`:

| File | Role |
|---|---|
| `arena.rs` | `ExpertArena`: one Metal buffer of N slots of 1.3824 MB, `slot_of[(layer, expert)]`, a decayed use count per expert, pinning of the current token's slots, and eviction of the coldest unpinned slot |
| `io.rs` | `IoPool`: a few threads doing `read_at` on the experts file with `F_NOCACHE`, filling page-aligned staging buffers they own (`Vec<u8>`, so they are `Send`) |
| `experts_file.rs` | Ferrum's own expert-contiguous file (header: model path, size, mtime, layer and expert counts, per-layer offsets), written once by `ferrum prepare`. Strata measured one read per expert at 2.97 GB/s against three at 2.30 GB/s, and decode 17% faster (Strata, M4 32 GB). Its own `.experts` file is already here, but we shouldn't depend on another project's format |
| `prefetch.rs` | Next-layer prediction: run layer `L+1`'s router on layer `L`'s FFN input (one extra 2560 × 512 BF16 GEMV per layer) and issue reads for the predicted misses while layer `L` runs |

**Decode becomes one handshake per layer.** Today Ferrum encodes a whole token and waits once. A streamed MoE layer can't be encoded until the CPU knows which experts it routes to:

```rust
for layer in 0..48 {
    encode_mixer_and_router(layer);          // gr_read, GDN/QSA, gr_write, gr_read, router top-10, shared expert
    let ids = commit_and_read_ids(layer);    // one wait; ids and weights are 10 × (u32, f32)
    let plan = arena.resolve(layer, &ids);   // hits: slot numbers; misses: claimed slots + pending reads
    encode_routed_experts(layer, &plan.hits);     // the GPU starts on the hits
    for miss in io.wait(plan.misses) {
        arena.fill(miss.slot, miss.bytes);   // scoped write into the slot
    }
    encode_routed_experts(layer, &plan.misses);   // accumulates into the same output
    prefetch.issue(layer + 1, predicted_ids(layer));
}
```

A Metal commit and wait costs on the order of 0.1 ms (est.), so 48 of them is ~5 ms per token. That is acceptable at ~70 ms per token, and Strata reports the same "fixed hand-shake per layer". The routed-expert kernel binds the whole arena as one input range and reads a small slot table (10 × slot index and weight), so `dispatch_hybrid`'s bounds and aliasing checks still cover it.

**Slot reuse must not race the GPU.** Ferrum tracks which ranges a submission *writes*. An arena slot is written by the CPU and read by the GPU, so the arena records the epoch of the last submission that read each slot, and eviction only picks slots whose epoch has completed. Slots pinned for the current token are never candidates.

**Prefill streams by layer.** A 1,024-row chunk routes to nearly all 512 experts of every layer (est.), so per chunk it reads each layer's non-resident experts once: about (1 − 0.4) × 708 MB × 48 ≈ 20 GB (est.). Bigger chunks amortize that, so prefill speed is set by chunk size and SSD bandwidth (Strata: chunk 1,024 / 2,048 / 4,096 gave 700 / 920 / 1,100 tok/s on a 128 GB Mac streaming Q5). Prefill uses a separate **ring** of staging slots (two layers' worth of misses, ~1.4 GB) so it doesn't evict decode's hot set. The planner trades chunk scratch and ring against arena size.

**Adaptivity.** The use counts follow the conversation. Swaps happen only on misses, so there is no background refill. A saved profile (`expert-profile.bin`, counts per expert) warm-starts the arena at load, filled in 1.5 GB steps that re-check free memory before each step, as Strata does after an 89 GB arena froze a 128 GB Mac.

**Safety boundary.** The first version needs no new `unsafe`: I/O threads own their staging bytes, and the model thread copies them into the slot through the existing scoped write (~2 ms per token at the expected miss volume, est.). Two optional steps each add one `unsafe` block in `src/metal`, counted in architecture.md, and each is worth taking only if measured:

- `fcntl(F_NOCACHE)` on the experts file and the n-gram shard. Without it, macOS's file cache competes with the arena for the same RAM. Use `rustix` if it covers `F_NOCACHE`, which keeps us at zero new blocks.
- Zero-copy reads straight into a slot, through a `SlotWriter` that holds the slot's pin.

### Sessions and speculative verify

- New recurrent state per session: the PLE history (9 × 10,240 f32), and per QSA layer the indexer tail (≤ 4 raw keys) and `dead` key. Pooled keys and QSA K/V are append-only, like full-attention K/V today.
- Snapshots grow to ~118 MB (est.). Two of them is the default.
- Verify of `k` rows: the PLE history gets a tape like GDN's (each row's normalized output; `commit(n)` appends the first `n`). A rejected row can complete an indexer block, so `pooled[b]` is always recomputed from the tail when a block completes, never cached across a rejection.

### Speculation under streaming

Each draft row routes to its own 10 experts, so a 3-row window touches up to 30 per layer, and on a miss-bound Mac that costs more than it saves. Strata measured MTP **slower** than one token per step on 32 GB, because its 1.2 GB do more as expert cache, and turned it off there. So:

- No MTP at first. The draft layer isn't in the GGUF anyway.
- The lookup drafter stays available, gated by measured acceptance and cost. Strata found prompt lookup paid off only on code edits (6–11% faster).
- Revisit MTP and trees only if the arena hit rate turns out high enough that compute, not I/O, is the limit.

### Planner

`PlanOptions` gains `arena: ArenaPolicy { min, max, keep_ram }` and charges the I/O staging, the prefill ring and the PLE row cache. The arena is sized last, from what is left of both the working set and the available RAM. Available means free + purgeable + file-backed inactive pages, but not other programs' inactive anonymous pages, which macOS can only free by compressing or swapping them (Strata's `arena_avail`). The `measured: ... (predicted ...)` line must still land within 4 MiB, with the arena reported separately because it is filled step by step. Querying free memory needs `host_statistics64`. If no safe crate covers it, that call goes in `src/metal` too.

## Traps to test, not trust

Every one of these was a real bug in Strata or comes with a warning in its source, and each has a plausible wrong reading that still gives finite, stable output:

1. GDN V head `h` reads K head `h % 16`, not `h / 3`.
2. Hash: XOR (not a sum) of `token · multiplier`; `%` by each head's prime vocabulary (not a mask); a missing predecessor is −1 and token 0 is real; an EOS (248044) in the context cuts it and every older token, but a token's own EOS doesn't cut its own context.
3. IQ4_NL rows: byte `j` holds elements `j` and `j + 16`, and the codebook has no −8 offset in this ggml version.
4. PLE conv taps are 3 rows apart (`t−9, t−6, t−3, t`), and the history is row-fastest (`hist[row + 9·channel]`).
5. The PLE hash and gather run once per token, outside anything replayed. Strata's CUDA graphs captured them once and repeated the first token's rows for every token.
6. QSA: `n_kv ≤ 2,051` means attend to all cells; the tail block gets +1e9; pooled keys are rotated at the position of the block's first cell.
7. Router: softmax over all 512 experts, stable descending sort (ties by index), renormalized with a 2⁻¹⁴ lower clamp.
8. `attn_q` holds Q and the gate interleaved, as in `qwen35`.
9. No `output_norm`: `output_hc_*` normalizes before the head.

Each gets a CPU-reference parity test in a new `src/hybrid/qwen4exp_tests.rs`, in the style of `iq_tests.rs`.

## Phases

Each phase is its own PR into `experimental`. All of them are "large" under AGENTS.md.

| Phase | Scope | Done when |
|---|---|---|
| **0. Formats** | Q2_0 and Q5_0 GEMV, prefill dequant and MoE kernels; type ids 42 and 20 in the loader; 640-wide experts; the `qwen4exp` config parse | CPU parity tests pass; `hybrid_plan` prints the model's geometry and memory from the header alone; existing models produce identical greedy output |
| **1. Streamed bring-up, ≤ 2,048 context** | Hyper-connections, PLE (no row cache), QSA as dense attention plus indexer writes, `ExpertArena` + `IoPool` with the per-layer handshake, staging copies, no prefetch. Context capped at 2,048 | Greedy output matches llama.cpp (CPU) on the AGENTS.md prompts; layer-by-layer activations match `llama-eval-callback` within tolerance; KLD against `llama-perplexity --kl-divergence-base` recorded under `docs/measurements/qwen4exp/` |
| **2. Speed** | `ferrum prepare` experts file, next-layer prefetch, hit/miss split dispatch, PLE row cache, saved expert profile, prefill ring and chunk planner | ABBA `hybrid_bench` against llama.cpp CPU and Strata on this Mac. Bar to merge: faster than llama.cpp. Goal: Strata's 13–14 tok/s on an M4 |
| **3. Long context** | QSA indexer selection in decode and prefill; context beyond 2,048; needle tests at 8K/16K/32K | KLD on long segments doesn't regress against phase 1's short-context numbers; needles found |
| **4. Product** | Sessions and snapshots with the new state, disk save, server/TUI integration, the `ferrum-tui` model picker, docs (README tables, `experimental.md`, `architecture.md`) | `hybrid_session_check` and `hybrid_session_disk` identical; docs true |
| **5. Optional** | Vision (`mmproj` BF16, Qwen3-VL style with MRoPE), lookup drafting gated by cost, MTP from the original checkpoint, Q8 hyper-connection read | Each with its own measured win |

## Verification notes

- **A reference build.** Strata reports llama.cpp running this Q2_0 file (56.6 tok/s on Metal on a 128 GB Mac, 6.3 tok/s on CPU on 32 GB). Q2_0 (type 42) may need a specific llama.cpp commit or fork. Find it before phase 1, and record the commit in the measurements.
- **The KLD base** has to come from llama.cpp on CPU with mmap here: about 21 tok/s on prompts (Strata, M4), so a 10K-token base takes roughly 8 minutes (est.).
- **Benchmark hygiene is stricter than usual.** Streaming numbers depend on what macOS keeps cached. Every run records the arena size, the hit rate and the bytes read, with a cold-cache run (after `purge`) kept apart from warm runs. Only one model in memory: Strata and Ferrum can't be measured side by side.
- **Memory pressure is a correctness test here.** Run a soak test of 30+ minutes and many requests, logging `vm_stat` swap-outs and `memory_pressure`, before calling any arena default safe.

## Open questions

1. Is Ferrum's per-token encode/commit model fast enough with 48 waits, or does decode need a dedicated command queue per layer, with `MTLSharedEvent` signalling the CPU instead of `waitUntilCompleted`?
2. Does the M5's SSD change the picture? Strata's 32 GB numbers are from an M4. If reads are much faster here, a smaller arena and a larger prefill chunk may be the better trade.
3. Q2_0 quality: Strata reports IQ3_XXS against Q2_0 at a KL of 0.094, but no Q2_0-against-BF16 number for this pack. The KLD base from llama.cpp will tell us what Q2_0 costs.
4. Should `ferrum prepare` write its own experts file, or read Strata's `STRX` file when it is present? Reading it saves 34 GB of disk, but ties us to a format we don't control.

## Credits

Strata (Niko Veit and contributors, MIT): the streaming design, its memory-pressure rules and the trap list. llama.cpp / ggml (MIT): the `qwen4exp` reference and the Q2_0 and IQ4_NL formats. ISTA-DASLab: the GSQ-RCO Q2_0 pack. The model is Qwen's, under its own licence.

## Journal

### 2026-11-02: phase 0, formats and config

- **Formats.** Q2_0 (type 42), Q3_K (11), Q5_0 (6) and IQ4_NL (20) now have kernels on all seven projection paths: GEMV, expert GEMV, small-batch GEMV, narrow tiles, prefill GEMM, expert GEMM and row lookup. One generic dequantizing GEMV (`MV_DEQ`) serves all four for now. A hand-tuned Q2_0 GEMV is phase 2 work, to be measured first.
- **Native Q3_K**, not the Q3_K→Q6_K rewrite the MTP block uses: the pack's 0.77 GB of dense Q3_K would grow to 1.47 GB as Q6_K, about 500 expert slots on this Mac.
- **Width limit.** Projections needed inputs that were multiples of 256; now they need whole blocks and whole 64-value steps (`PROJECTION_K_STEP`, the narrow tiles' K step). That admits the 640-wide experts and the 320-wide hyper-connection weights.
- **Parity.** `gsq_formats_*` in `iq_tests.rs` checks every path against CPU decoders written from ggml's `dequantize_row_*`, at widths that are not multiples of 256. Deliberately broken Q2_0 and Q3_K decoders fail all three tests.
- **Config.** `Variant::Qwen4Exp` parses the hyper-connection, indexer and n-gram settings, and checks `compress_ratios` against `full_attention_interval`. The loader refuses the variant with a pointer to this plan. `hybrid_plan` reads the Q2_0 file from its header: 35.03 GiB of weights, 31.64 of them routed experts, too big for the 23 GiB budget ([record](../measurements/qwen4exp/phase0-plan-2026-11-02.txt)).
- **Regression (light, as agreed: the full A-B-B-A and KLD comparison waits until the Qwen3.8 path is optimized).** All test suites pass, with and without `legacy-engine`. Greedy output over 64 tokens is identical to `experimental` on Spark-X2.6-4B Q4_K_M and Qwen3.6-35B-A3B IQ4_XS, with the same measured memory ([record](../measurements/qwen4exp/phase0-greedy-2026-11-02.txt)).

### 2026-11-02: streamed bring-up, speed, drafters and the TUI

- **Runs on the 32 GB M5.** The routed experts stream from the first shard into an arena of slots (`stream.rs`, `metal/stream.rs`): LRU kept as a sorted set, the next layer's experts read ahead from its router (top 16 per row for decode, top 10 for prompt chunks), and wrong prompt guesses demoted to first victims. The arena takes what the plan leaves (17.4 GiB at a 32K context), capped at the device's largest buffer. Prompt chunks default to 2,048 rows on streamed models.
- **Hyper-connections and the n-gram embedding** have kernels with CPU parity tests (`hc_tests.rs`). The hyper-connection projections load as Q8_0: KLD 0.0227 against 0.0222 for BF16, for less to read per token.
- **Accuracy.** KLD against llama.cpp on CPU: mean 0.0227, top-1 94.75%, PPL 7.021 vs 7.064.
- **Speed against Strata** on the same Mac, interleaved: decode 16.8–18.4 against 14.8–15.8 tok/s, prefill 232 and 315 against 190 and 282 tok/s at 1,155 and 3,815 tokens. Strata's own config passes `--spec 2` with no MTP file, so it verifies 2-token windows that never accept (67 ms each, against 62 ms without); without `--spec` it falls back to suffix lookup and reaches 15.0–16.9 tok/s, still behind.
- **Sessions and verify.** Snapshots include the n-gram conv history; verify tapes the conv inputs and leaves the history to `commit`; drafters read the stream mean of layer outputs. `hybrid_session_check`, `hybrid_session_disk` and `hybrid_spec_check` pass.
- **Drafters.** DFlash2 at 2 drafts: 1.36× on a short prompt, 1.16× at 1,155 tokens, none at 3,815. More drafts lose: each verify row routes to its own experts, often not in the arena (verify 84, 108, 150 ms at 2, 3, 4 drafts). DSpark peaks at 22.1 tok/s at 2. The MTP head is a full layer with its own 512 experts and is not loaded: it would pay the same verify cost and a MoE layer per draft. A first DFlash2 reading of 27.9 tok/s came from a run right after a plain pass on the same prompt, which left its experts in the arena; interleaved runs replaced it.
- **TUI.** The model check accepts `qwen4exp` and its BF16 and F16 tensors, MTP heads show as drafters, and the speed benchmark keeps Qwen3.8 on chains. Chat, tools, a cached follow-up, an image and DFlash2 work in `ferrum-tui`.
- **Memory.** `iogpu.wired_limit_mb` is 24,576 here; the plan stays at that less the 1 GiB reserve. No swap-outs during a drafted TUI chat.
- **Drafted output** drifts from plain decoding at near ties (margins 0.0002–0.024 at 3 drafts); `FERRUM_HYBRID_EXACT=1` makes all four test prompts identical, so it is verify rounding amplified by routing, as on Flint.
- **Context.** The full 262,144 positions fit under the cap (arena 12.1 GiB with f16 K/V, 14.9 with q8); short-prompt decode loses at most 0.4 tok/s.
- **Still open:** the QSA indexer (contexts past 2,048), draft trees, MTP, a vision parity check.
