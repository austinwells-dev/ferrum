# Phase 2 completion report

Measured on 2026-09-20 on Apple M5, arm64 macOS 27.0, Rust/Cargo 1.96.0. Phase 1 foundation: `f2e1234`. Work branch: `codex/phase2-transformer`. Phase 2 stops at synthetic model execution; no pretrained model was downloaded or executed.

## Implementation and dependencies

| Files | Change |
|---|---|
| `src/loader/mod.rs` | Safe local safetensors parser, name enumeration, dtype/shape inspection via loaded tensors, direct byte copy into shared storage, named weight errors |
| `src/tokenizer/mod.rs` | Local tokenizer.json loading, encode/decode, vocabulary size, contextual errors; independent of numerical code |
| `src/nn/mod.rs` | Embedding, linear with optional bias, RMSNorm, SwiGLU MLP |
| `src/nn/attention.rs` | Decomposed GQA attention and optional diagnostic snapshots |
| `src/nn/kv_cache.rs` | Per-layer immutable active K/V pairs, append/read/reset/capacity checks |
| `src/model/{mod,config,weights,transformer,tiny}.rs` | Validated dimensions, named construction/typed layers, transformer forward APIs, deterministic fixture serialization |
| `src/ops/transformer.rs`, `src/metal/shaders/ops.metal` | Gather, controlled layout copies, bias/scalar operations, causal masking, explicit split-half RoPE |
| `src/reference/transformer.rs`, `src/reference.rs` | Independent CPU full transformer oracle; existing primitive oracles unchanged |
| `src/transformer_smoke.rs`, `src/main.rs` | `transformer-smoke` command and instrumentation |
| `src/tensor/mod.rs` | Validated little-endian byte initialization without conversion staging |
| `src/metal/mod.rs` | Allocation/byte/dispatch counters; synchronization and four unsafe blocks retained |
| `src/error.rs`, `src/lib.rs` | New contextual errors and module exports |
| `tests/loading.rs`, `tests/transformer.rs` | Three loading/tokenizer suites and eight transformer suites |
| `Cargo.toml`, `Cargo.lock` | Two new direct dependencies and locked transitive dependencies |
| `README.md`, `docs/architecture.md`, this report | Usage, dimensions, architecture, measured results, handoff |
| `docs/measurements/phase2/*` | Raw quality, validation, numerical, smoke, and benchmark evidence |

New direct dependencies are **safetensors 0.8.0**, for mature binary/header validation and fixture serialization, and **tokenizers 0.23.2**, for standard Hugging Face tokenization. Tokenizers uses `default-features=false, features=["fancy-regex"]`; its networking/HF Hub features are disabled. No ML execution engine, Python, GGUF, or additional crate workspace was introduced. Transitive dependencies are pinned in Cargo.lock.

Safetensors validates metadata, offsets, shapes, and payload size before exposing views. Ferrum rejects unsupported dtypes and checks shape × dtype byte size again before copying directly into a fresh shared allocation. Per-tensor failures preserve the offending name; archive-level corruption produces a safetensors parse error. Safetensors represents contiguous tensors only. Whole local files are read into a byte vector; memory mapping, sharded index handling, and zero-copy file-backed Metal buffers are not implemented.

Tokenizer tests cover standard WordLevel and ByteLevel BPE JSON, loading a local file, text/IDs/decode roundtrips, empty inputs, unknown words, invalid IDs, malformed JSON, and malformed tokenizer structure. Encode does not add special tokens; decode does not skip them. Chat-template/special-token policy belongs to model integration.

## Transformer architecture and shape contract

Implemented architecture: embedding → repeated pre-norm GQA attention/residual + pre-norm SwiGLU/residual → final RMSNorm → separate or tied LM projection. Optional bias is supported on all linear projections, including Q/K/V. The intended first real-model structural target is dense full-attention Qwen2/Qwen2.5-style decoding. No broad model-family abstraction was introduced.

All dimensions come from `ModelConfig`. The primary fixture has vocabulary 32, hidden 16, intermediate 32, two layers, four query heads, two KV heads, head dimension four, context 16, epsilon 1e-5, theta 10000, and untied embeddings. Additional tests use hidden 13, intermediate 19, three query heads, head dimension six, vocabulary 17, and either one or three KV heads; they demonstrate dimensions and head layout are not hard-coded.

| Object | Logical dimensions |
|---|---|
| Embedding / LM weight | `[vocab, hidden]` |
| Hidden state | `[new_tokens, hidden]` |
| Q/K/V weights | `[q_heads*head_dim,hidden]`, `[kv_heads*head_dim,hidden]`, `[kv_heads*head_dim,hidden]` |
| Q/K/V activations | `[new_tokens,q_heads,head_dim]`, `[new_tokens,kv_heads,head_dim]`, `[new_tokens,kv_heads,head_dim]` |
| K and V cache | each `[cached+new_tokens,kv_heads,head_dim]`, per layer |
| Per-head scores/probabilities | `[new_tokens,cached+new_tokens]` |
| Per-head context / merged context | `[new_tokens,head_dim]` / `[new_tokens,q_heads*head_dim]` |
| Attention output weight | `[hidden,q_heads*head_dim]` |
| Gate/up / down weight | `[intermediate,hidden]` / `[hidden,intermediate]` |
| Norm / bias | `[hidden]` / `[projection_output]` |
| Prefill / decode logits | `[sequence,vocab]` / `[1,vocab]` |

Linear computes `X Wᵀ` with the existing tiled matmul and an optional separate bias kernel. Weight transposes are materialized at construction. Attention selects a contiguous head, transposes K, performs matmul, scalar multiplication by `1/sqrt(head_dim)`, causal masking, existing softmax, and a second matmul with V. It stacks contexts, swaps the first two axes, merges heads, and projects out. No fused kernels or asynchronous execution were introduced.

GQA maps query head `h` to KV head `h / (q_heads/kv_heads)`. The cache retains only the configured number of KV heads. Per-head working copies remain an intentional overhead. The causal condition is `key_index <= cached_length + query_row`; a dedicated kernel writes negative infinity elsewhere.

Qwen-style split-half RoPE pairs `j` and `j+D/2`, uses exponent `-2*j/D`, and applies absolute position `cached_length + token_index`. The original adjacent-pair Phase 1 API and shader are unchanged. The convention was checked against the upstream [Qwen2 implementation](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen2/modeling_qwen2.py). Phases/trigonometry are F32; outputs round to storage dtype. Tests include positions 2048 and 2049 with theta 500000. Very-long-context precision remains unestablished.

Cache append retains the first pair, then creates new contiguous concatenations. Reset drops all active pairs. Capacity is a limit, not a preallocated buffer. Model forward stages cloned Rc handles and publishes the cache only after logits succeed. Layer count/layout/length consistency, token IDs, capacity, and context position are checked. Prefill creates an empty cache and returns all position logits, including the final row; decode appends one token. Multi-token chunk appends also work. No generation loop or sampling is present.

## Numerical methodology and results

The independent scalar CPU oracle uses deterministic weights already quantized to the storage dtype. It computes reductions, dot products, and RoPE angles in F64, then rounds every operation boundary to destination dtype, including separate score scaling, SiLU/multiply, residual adds, and bias adds. It does not invoke Ferrum tensors or GPU operations. Synthetic fixture construction is shared, while transformer arithmetic/layout indexing is independent.

Each dtype compares **41 intermediate tensors**: embedding, both layer input norms, Q/K/V, rotated Q/K, every head's scores and probabilities, merged contexts, attention projections, post-attention norms, MLP results, decoder outputs, final normalized hidden state, and logits. All model weights pass through safetensors serialization and the actual production loader before GPU execution. File-based loader tests separately cover the filesystem path.

Tolerance is unchanged from Phase 1: `abs_error <= atol + rtol * abs(expected)`.

| Dtype | Absolute tolerance | Relative tolerance |
|---|---:|---:|
| F32 transformer | 3e-5 | 3e-5 |
| F32 long-offset RoPE | 5e-4 | 3e-5 |
| F16 | 2e-3 | 2e-3 |
| BF16 | 1.6e-2 | 1e-2 |

No tolerance was widened to accommodate a failing transformer. Maximum observed errors in the primary fixture:

| Comparison | F32 max absolute | F16 max absolute | BF16 max absolute |
|---|---:|---:|---:|
| All 41 intermediate tensors | 4.76837158e-7 | 0 | 0 |
| Final logits | 1.88651029e-7 | 0 | 0 |
| Final hidden state | 4.76837158e-7 | 0 | 0 |
| Cached/full logits, all five positions | 0 | 0 | 0 |
| Split-half RoPE, positions 2048/2049 | 1.90734863e-6 | 0 | 0 |

F32 per-stage maxima across the two layers:

| Stage | Maximum absolute error |
|---|---:|
| Embedding | 0 |
| Input norm | 3.57627869e-7 |
| Q / K / V | 1.19209290e-7 / 1.19209290e-7 / 1.37835741e-7 |
| RoPE Q / K | 1.19209290e-7 / 1.19209290e-7 |
| Scaled scores | 7.45058060e-8 |
| Probabilities | 5.96046448e-8 |
| Merged context | 1.49011612e-7 |
| Attention projection | 5.96046448e-8 |
| Post-attention norm | 4.76837158e-7 |
| MLP | 3.16649675e-8 |
| Decoder output | 5.96046448e-8 |

Exact low-precision agreement is an observation for these rounded deterministic fixtures, not a universal accuracy guarantee. Cached equivalence feeds `[3,8,4,11,2]` one token at a time and compares every position with full-sequence logits. It also compares active K/V and verifies a two-token prefill followed by a three-token append. Separate tied-embedding, MHA, single-KV-head, and awkward-dimension cases pass. Raw errors: [transformer log](measurements/phase2/transformer.txt).

## Timings, dispatches, and memory

Release smoke, Apple M5, F32 primary fixture; ten samples after pipeline compilation and correctness warmup. Median uses sorted sample index five. Timings include allocations, encoding, and completion waits, but exclude source loading, CPU oracle, diagnostic tracing, and result disposal. Decode samples clone the same four-token cache and append its fifth token. These short desktop measurements have scheduling variance and make no cross-engine performance claim.

| Measurement | Prefill, five tokens | Decode, cache length 4 → 5 |
|---|---:|---:|
| Median wall time | **17.521291 ms** | **17.045334 ms** |
| Metal dispatches | **119** | **123** |
| Allocations | **120** | **124** |
| Cumulative physical allocated bytes | **22,260** | **6,628** |
| Returned logits payload | 640 bytes | 128 bytes |
| Final active cache payload | 640 bytes | 640 bytes |
| Temporary allocation volume | **20,980 bytes** | **5,860 bytes** |

Model retained weight payload is **23,104 bytes**. Five-token active K/V is **640 bytes**; the configured 16-token capacity would contain **2,048 bytes**, but is not reserved. Safetensors load plus model construction/transposition took **10.040792 ms** in this process; synthetic serialization is outside that measurement. Decode dispatches exceed prefill by four because each of two layers concatenates both K and V into an existing cache.

Temporary allocation volume is total newly allocated physical bytes minus final cache and returned logits; it is **not peak live memory** and excludes pre-existing cache/source allocations. A CPU file byte vector and construction-time source/transposed matrices can coexist during loading. Keeping diagnostic traces intentionally retains additional activations. Raw output: [transformer smoke](measurements/phase2/transformer-smoke.txt).

The unchanged primitive benchmark was rerun because backend counters were added. No material regression appeared against the recorded Phase 1 estimates:

| Primitive | Phase 1 estimate µs | Phase 2 rerun estimate µs |
|---|---:|---:|
| Add F32 1,024 | 185.61 | 154.79 |
| Add F32 65,536 | 202.64 | 194.84 |
| Add F32 1,048,576 | 436.34 | 411.76 |
| RMSNorm F32 1×128 | 218.98 | 213.50 |
| RMSNorm F32 4×4096 | 854.11 | 834.76 |
| RMSNorm F32 32×4096 | 913.80 | 873.47 |
| Softmax F32 1×128 | 223.38 | 215.51 |
| Softmax F32 4×4096 | 1005.4 | 987.98 |
| Softmax F32 32×4096 | 1108.6 | 1071.6 |
| Matmul F32 17×19×23 | 180.89 | 158.15 |
| Matmul F32 128×128×128 | 239.74 | 196.84 |
| Matmul F32 256×512×256 | 399.07 | 321.84 |
| Matmul F16 256×512×256 | 328.78 | 316.83 |

These variations are not attributed to optimization: the Phase 1 primitive shaders are unchanged, runs are short, and the desktop environment is uncontrolled. Full confidence intervals and separate GPU/submission medians are in [primitive benchmark](measurements/phase2/primitive-benchmark.txt).

## Safety and quality gates

`rg -n '\bunsafe\b' src` before and after shows the same **four unsafe blocks plus one framework-linkage declaration**, all in `src/metal/mod.rs`. No raw pointer escapes, unsafe Send/Sync, Arc substitution, mutable published storage, arena, asynchronous submission, or multithreaded Metal ownership was added. New numerical/loader/tokenizer/model modules inherit unsafe denial; model/NN/loader/tokenizer explicitly forbid it. The only backend behavior addition is single-threaded counters. New MSL accesses and their host contracts remain part of the audited GPU boundary.

All required gates passed:

| Command | Result |
|---|---|
| `cargo fmt --check` | PASS |
| `cargo clippy --all-targets --all-features -- -D warnings` | PASS |
| `cargo test` | PASS: 20 integration tests (9 existing + 3 loading + 8 transformer) |
| `cargo build --release` | PASS |
| `cargo run --release -- info` | PASS: Apple M5, shared memory |
| `cargo run --release -- smoke` | PASS: all seven original operations |
| `cargo run --release -- transformer-smoke` | PASS: 41 intermediate comparisons and cached equivalence |
| `MTL_DEBUG_LAYER=1 MTL_SHADER_VALIDATION=1 cargo test --test transformer -- --test-threads=1` | PASS: all eight suites, validation enabled |
| `MTL_DEBUG_LAYER=1 MTL_SHADER_VALIDATION=1 cargo test --test correctness -- --test-threads=1` | PASS: all nine unchanged suites |
| `cargo bench --bench runtime -- --noplot` | PASS: all 13 primitive cases |

Evidence: [quality gates](measurements/phase2/quality-gates.txt), [Metal validation](measurements/phase2/metal-validation.txt), [unsafe audit](measurements/phase2/unsafe-audit.txt). Phase 1 tests and benchmark source were preserved unchanged.

## Limits and exact Phase 3 work

Execution remains synchronous, single-threaded, contiguous-only, and fresh-output-per-operation. Attention materializes per-head score/probability matrices and copies layouts; serial softmax/RMSNorm and 119–123 completion boundaries dominate the tiny workload. Cache append copies its entire active history. Weight preprocessing duplicates source/transposed storage until the caller drops its loaded source bundle. These are documented future optimization targets, not work begun here.

Only the dense full-attention Qwen2-style subset is implemented. QK-normalized Qwen3, MoE, MTP, sliding windows, YaRN/scaled or partial RoPE, multimodal models, quantization, graph execution, asynchronous ownership, and all other explicitly excluded features remain outside Phase 2. The configuration is a validated Rust structure, not a general architecture detector. Cache shape validation cannot detect reuse with a different same-shaped model; callers must keep cache/model association. Extremely large finite arithmetic and very-long-context RoPE have the same precision limits as the foundation. Validation ran on this M5 only.

Phase 3 can now:

1. Select a small real local Qwen-family checkpoint within this structural subset.
2. Parse its configuration and reject unsupported architecture flags.
3. Map its tensor names to the canonical typed construction path; assemble local shards if necessary and verify dtype/shapes.
4. Load its standard tokenizer and integrate its special tokens/chat template.
5. Invoke prefill, select the final row of logits, and perform cached single-token decode.
6. Add sampling/stopping policy and a generation loop.
7. Validate real logits/tokenization as needed and demonstrate coherent text.

No fundamental dense transformer math needs to be invented for that supported target. Phase 2 has not selected/downloaded/executed a production model, introduced chat templates or sampling, or begun real-model generation.
