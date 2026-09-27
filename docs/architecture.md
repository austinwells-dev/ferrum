# Ferrum architecture

The Phase 4 section below describes the current runtime. Earlier sections are preserved as the historical Phase 1–3 design and are superseded where noted.

## Modules and dependency direction

```text
CLI / tests / benchmarks
        │
        ├── reference (CPU oracles only)
        │
        ▼
Tensor + operation API ──► MetalDevice ──► objc2 / Apple's Metal
                                │
                                └── embedded ops.metal
```

`src/lib.rs` denies unsafe Rust except in `metal`; tensor, operations, and CPU reference modules additionally forbid it. Objective-C object types are private to the backend. There is one crate, one queue per device context, and no runtime dependency on another ML engine.

## Tensor and layout

`Tensor` contains `Rc<MetalBuffer>`, `Shape`, `Layout`, and `DType`. Fields are private. Shapes cache checked element counts; `[]` denotes a scalar with one element. Layout strides are in elements, row-major, and checked for overflow. Dimensions and strides use `SmallVec<[usize; 4]>`, supporting arbitrary rank with no dimension/stride heap allocation for the usual rank ≤4. Higher ranks spill to a vector.

All current tensors are contiguous and cover their entire logical allocation. Reshape preserves dtype, element count, and storage, rebuilding only contiguous metadata. There is no operation that can construct a slice, nonzero offset, or invalid raw memory view. Zero-sized dimensions are accepted when element and stride calculations are representable; pathological dimensions that overflow intermediate calculations are conservatively rejected.

DType's host/shader ABI is F32=0, F16=1, BF16=2. F16 uses Metal `half` loads/stores; BF16 uses two-byte integer storage and explicit conversion. Arithmetic and matmul/reduction accumulation are F32 for all three. CPU upload uses `half` crate conversions. BF16 stores implement round-to-nearest, ties-to-even and preserve NaN encoding.

## Storage

A `MetalBuffer` owns a retained `MTLBuffer`, logical length, and an `Rc` context identity. The physical Metal buffer length is `max(logical length, 4)` so empty tensors still have valid backing storage. Every allocation is zero-initialized. This simplifies initialized-memory guarantees and K=0/empty behavior, but output zeroing is included in end-to-end benchmark cost.

Storage mode is Shared. Actual CPU base-pointer alignment, physical allocation length, logical byte length, offset (currently always zero), and tensor owner count are observable. Alignment describes this allocation, not a universal allocator guarantee. Metal retains allocation ownership; `Rc` preserves it through tensor clones/reshapes. Storage can outlive the creating `MetalDevice` for CPU inspection. Operations reject tensors from a different context, even if both contexts use the same physical GPU.

Initialization converts directly into a borrowed byte slice of the shared allocation. Readback borrows initialized completed bytes and decodes directly into the returned F32 vector. Scoped closures prevent mapped references from escaping. There is no upload/readback staging vector and no CPU↔GPU blit. No storage mutation is exposed after tensor construction.

Apple's shared-memory guidance requires CPU/GPU accesses to be ordered despite shared physical memory: [MTLStorageMode.shared](https://developer.apple.com/documentation/metal/mtlstoragemode/shared). Phase 1 uses tracked hazards and command completion; managed-memory flush/synchronize calls do not apply to these shared buffers.

## MetalDevice and pipeline cache

`MetalDevice::new` discovers the default GPU, requires unified memory, creates one command queue, and retains device identity. CoreGraphics is linked for command-line discovery. Device name, recommended working set, max buffer length, unified-memory flag, and known Apple families 1–10 are exposed. The working-set recommendation is advisory, not an allocation budget.

The backend caches shader libraries by source text and pipelines by `(source, entry point)`. A separate built-in-name cache avoids copying/hashing shader source in operation hot paths. `warm_up` resolves all seven pipelines. The first compilation may benefit from Metal's persistent driver cache; measured process warm-up is not claimed to be an uncached compiler measurement.

Compilation requests safe math optimizations and precise single-precision math functions using macOS 15+ APIs. The compiler's full NSError description is returned on failure. Missing functions and pipeline creation failures have distinct contextual errors. Current bindings: [objc2-metal 0.3.2](https://docs.rs/objc2-metal/0.3.2/objc2_metal/).

`compile_kernel` returns an opaque cached pipeline; only built-in operations can dispatch. A future custom-kernel API must either be explicitly unsafe with a documented ABI/range contract or validate an expressive binding contract. Compilation alone cannot grant safe arbitrary GPU memory access.

## Kernel ABI and dispatch

Three buffers are bound: input A at index 0, optional input B at 1 (A is reused as an unused placeholder for unary kernels), and a fresh output at 2. Index 3 contains nine packed native-endian `u32` values (36 bytes):

| Index | Meaning |
|---|---|
| 0 | A element count |
| 1 | Reduction width, or matmul M |
| 2 | Matmul K |
| 3 | Matmul N |
| 4 | Dtype discriminant |
| 5 | RoPE position |
| 6 | RoPE head dimension |
| 7 | RMSNorm epsilon, F32 bits |
| 8 | RoPE theta, F32 bits |

Host validation checks context identity, matching dtypes, shapes, parameter domains, allocation sizes, and `u32` indexing before any command is committed. Matmul also guards K's final tile increment against wrap. The trusted shader guards loads/stores at tile edges. Full 16×16 groups are dispatched for matmul so every thread reaches both tile barriers. Other kernels use exact grids and up to 256 threads per group, bounded by pipeline limits. Empty grids return zero-dispatch metrics.

Add/mul/SiLU dispatch one thread per element. RMSNorm and softmax dispatch one thread per row. Softmax subtracts the row maximum before exponentiation. RoPE dispatches one thread per adjacent pair and uses `position * theta^(-2*pair/head_dim)`. Matmul loads two 16×16 threadgroup tiles, accumulates in F32, and stores one result per thread. It uses our own MSL, not MPS or MPSGraph.

## Synchronization and unsafe boundary

Command creation, encoding, commit, wait, and status inspection live in `metal::dispatch`. One public operation submits one command buffer and waits once. Completion is checked before output publication, including the GPU-error path. Inputs/outputs stay alive through the wait, and command buffers retain referenced resources. Autorelease pools bound temporary Objective-C lifetimes during compilation and dispatch.

Tensors and devices are deliberately `!Send` and `!Sync` via `Rc`. Published tensors are immutable, output allocations are fresh, and CPU initialization happens before publication. These invariants prevent safe users from racing CPU access with GPU work. An input may occupy both read-only bindings, but no output aliases an input.

Four unsafe blocks are confined to `src/metal/mod.rs`:

1. Mutable mapped byte slice during initialization: unique allocation access, initialized bounded range, no submitted work.
2. Immutable mapped byte slice during inspection: completed GPU work and a bounded initialized allocation.
3. Allocation zeroing: fresh owned shared memory of the requested physical length.
4. Encoder buffer/scalar binding: validated internal ABI and retained allocations/parameters through command encoding and completion.

The empty `unsafe extern "C"` declaration only links CoreGraphics; it declares no callable functions. No unsafe impl is used. No library/runtime `unwrap` or `expect` is used. CPU reference routines are test utilities with dimensional preconditions; they are not general runtime tensor operations.

MSL bounds, host/shader ABI consistency, Metal's driver, allocation availability, and Objective-C framework behavior remain trust boundaries. Rust's type system alone cannot establish shader memory safety. Apple validation layers provide an additional development check, not a proof.

## Observability and errors

Operations return `Output { tensor, metrics }`. Metrics include operation/output shape/dtype, logical bytes read/written, output allocation bytes, dispatch count, submission duration, synchronized wall duration, and optional GPU timestamps. No logging runs in the library hot path; callers explicitly inspect metrics. Logical bytes count each input payload once, not serial-kernel rereads or matmul tile traffic.

Submission duration starts at command creation and ends immediately after commit. Synchronized duration extends through completion. Both exclude validation, output allocation, and pipeline lookup/compilation. GPU duration is `GPUEndTime - GPUStartTime` after completion and may be unavailable. Criterion measures the wider public call including allocation and result disposal. Thus these numbers answer different questions and must not be substituted for each other.

`Error` distinguishes initialization, shader compilation, missing kernel, pipeline creation, invalid shape/reshape, dtype/context mismatch, allocation, dispatch, synchronization, parameter, and numerical-validation failures. Metal error descriptions are retained; no silent CPU fallback exists.

## Future execution constraints

No graph executor or memory planner is implemented. Phase 2 composes synchronous tensor operations without removing waits. Before asynchronous batching or allocation reuse, introduce submission/completion ownership that keeps buffers alive, prevents premature CPU mapping, and tracks when each write completes. Keep encoding distinct from completion, submit several operations together, and publish readable tensors only behind the appropriate completion token.

A future planner can consume byte size, alignment, storage mode, and liveness to allocate arenas. Nonzero offsets and views will require checked ranges and dtype alignment in both tensor metadata and encoder bindings. Thread-safe execution will require an explicit synchronization design rather than replacing `Rc` with `Arc` or adding unsafe Send/Sync implementations.

RoPE convention and scaling are model-dependent; current adjacent-pair rotation is not interchangeable with split-half variants. F32 long-position phase accuracy and native BF16/SIMD-group matmul acceleration need separate designs and tests. Serial reductions and per-operation allocations are measured optimization targets, not architectural promises.


## Phase 2 module boundaries

`loader` parses local safetensors into named, immutable tensors; `tokenizer` independently handles local Hugging Face tokenizer JSON. `model::weights` validates every required shape/dtype/device and optional projection bias before preprocessing weights. It constructs `Embedding`, `Linear`, `RmsNorm`, `Attention`, `Mlp`, and `DecoderLayer` values. Runtime execution uses those fields, never string weight lookups. `model::tiny` creates deterministic synthetic safetensors fixtures. `reference::transformer` is a scalar CPU oracle used by validation, never a numerical fallback.

`ModelConfig` requires vocabulary, hidden/intermediate sizes, layer count, query/KV head counts, head dimension, RMSNorm epsilon, RoPE theta, maximum context, tied-embedding setting, and dtype. Dimensions must be positive, the head dimension even, Q heads divisible by KV heads, and checked tensor element products within u32 indexing. Query projection width need not equal hidden size. Phase 2 configuration is explicit Rust data; Phase 3 adds the strict Qwen JSON/name adapter described below.

## Phase 2 shapes and weights

Let `S` be new tokens, `P` cached tokens, `T=P+S`, `H` hidden width, `I` intermediate width, `Q` query heads, `K` KV heads, `D` head width, and `V` vocabulary. All tensors are contiguous row-major and dtype-homogeneous.

| Tensor | Shape |
|---|---|
| Token IDs (host slice) | `[S]` u32 |
| Embedding weight | `[V,H]` |
| Hidden/residual states | `[S,H]` |
| Q / K / V projection weights | `[Q*D,H]` / `[K*D,H]` / `[K*D,H]` |
| Optional projection bias | `[output_width]` |
| Q / K / V activations | `[S,Q,D]` / `[S,K,D]` / `[S,K,D]` |
| Cached K and V, each layer | `[T,K,D]` |
| Selected Q head / K head / V head | `[S,D]` / `[T,D]` / `[T,D]` |
| Scores and probabilities per query head | `[S,T]` |
| Stacked head contexts / merged context | `[Q,S,D]` / `[S,Q*D]` |
| Attention output weight | `[H,Q*D]` |
| Gate/up / down weights | `[I,H]` / `[H,I]` |
| Norm weights | `[H]` |
| LM head weight / logits | `[V,H]` / `[S,V]` |

The synthetic canonical names are `embedding.weight`, `final_norm.weight`, `lm_head.weight`, and `layers.{index}.{input_norm,post_norm,q,k,v,o,gate,up,down}.weight`. Projection `.bias` tensors are optional. Tied mode uses the embedding matrix and does not require an LM-head weight. A transposed LM matrix still has its own allocation in this deliberately contiguous implementation. Unknown bundle tensors are ignored by model construction; source bundles can be dropped after construction.

Linear accepts rank ≥1, flattens leading dimensions, multiplies by a construction-time contiguous weight transpose, applies optional bias, and restores the leading shape. All matmuls use the unchanged Phase 1 tiled kernel. Input and output dtype must match.

Embedding gather validates every host ID before dispatch. IDs are uploaded as exact little-endian u32 bytes, never converted to F16/BF16/F32 numbers. The private carrier tensor matches the weight dtype for the existing binding contract, while its shader interpretation is u32. Byte length and four-byte alignment are checked. The public gather API accepts IDs, never a forgeable carrier tensor.

## Attention and RoPE

Each block executes RMSNorm → separate Q/K/V linears → split-half RoPE on Q/K → cache append → per-head matmul → scalar scale → causal mask → softmax → matmul → merge → output linear → residual add. The second path is RMSNorm → gate/up linears → SiLU(gate) × up → down linear → residual add.

Query head `h` reads KV head `h / (Q/K)`. Cache storage never contains expanded Q-count copies of K/V. Separate contiguous per-head working tensors are copied as needed; repeated query groups can temporarily copy the same KV head. This is intentionally unoptimized and makes reuse of existing matmul straightforward.

The Qwen split-half convention pairs component `j` with `j+D/2`, with angle `(P+token_index) * theta^(-2*j/D)` for `0 <= j < D/2`. The complete head rotates; theta is configurable. Qwen's upstream [reference implementation](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen2/modeling_qwen2.py) explicitly uses half rotation. Phase 1 `rope` remains adjacent-pair/single-position; Phase 2 `rope_split` names its different convention explicitly.

RoPE uses F32 phase/trigonometry and rounds the result to storage dtype. It does not emulate every intermediate low-precision rounding choice of another engine. The CPU oracle uses F64 angles. Existing long-context uncertainty remains: tests include offsets 2048/2049, but no very-long-context accuracy claim or scaling support is made.

Causal mask keeps key `j` exactly when `j <= P+i` for query row `i`. The kernel writes negative infinity for excluded entries without allocating a mask matrix. Shape validation requires `[S,P+S]`, so every nonempty row has a visible key. Softmax remains the existing serial F32 row reduction, with output stored in the configured dtype.

## Cache and forward semantics

`KvCache` holds one optional immutable `(K,V)` pair per layer, capacity/layout/dtype metadata, and no mutable published allocation. Initial append retains new tensors. Later appends allocate concatenated tensors, wait for completion, then replace that layer's pair. `active` borrows the current pair; reset drops all pairs. Layers can be appended independently, but a model forward requires all lengths equal and matching layer count/head layout/dtype. Payload bytes report only active storage, not a reserved maximum allocation.

`forward_prefill` creates an empty cache and returns all position logits plus the cache. `forward_decode` adds one token (including to an empty cache), returning `[1,V]`. General `forward` accepts a nonempty chunk and optional diagnostic trace. It validates token IDs, context/capacity, and cache layout, clones the cache's Rc handles, then publishes the staged cache only after final logits succeed. Errors preserve the caller's cache. A diagnostic trace may contain partial snapshots on error and retains additional tensors when enabled. Caches must be used with the model that produced them; matching shape alone does not establish matching weights.

## New copy kernel ABI and safety

The existing three-buffer/nine-u32 dispatcher is unchanged. New kernels reinterpret unused scalar slots with checked host contracts:

| Kernel | Scalar fields beyond common dtype/length |
|---|---|
| embedding_gather | p1=hidden width; B contains validated u32 IDs |
| copy_range | p1=source element offset; dispatch size=checked output count |
| concat_flat | p0=A length; dispatch size=A+B elements |
| transpose2 | p1=rows, p2=columns |
| swap01 | p1=A, p2=B, p3=C for `[A,B,C] → [B,A,C]` |
| select_head | p1=head index, p2=head count, p3=head width |
| bias_add | p1=last-axis width |
| scale | p7=F32 scalar bits |
| causal_mask | p1=key count, p5=absolute query offset |
| rope_split | p1=head count, p5=offset, p6=head width, p8=theta bits |

These are fresh-output copies, not views or arbitrary strides. Exact dispatch grids and validated shape products bound every access. Zero-size grids do not dispatch. No new unsafe Rust was added; the four backend blocks and framework linkage remain the only unsafe sites. Counter bookkeeping uses single-threaded `Cell` alongside the existing Rc ownership.

Device counters are monotonic totals of successful allocations, physical allocation bytes, and submitted dispatches. A caller snapshots before/after a forward. They include ID carrier and output allocations. Cumulative allocation volume is not peak live memory or DRAM traffic. Smoke subtracts final active cache and returned logits from allocation volume to estimate temporary allocation volume. Construction/source weight copies and diagnostic traces are excluded from steady-state samples.

## Phase 3 handoff

Fundamental dense full-attention Qwen2-style execution is ready. Phase 3 needs a locally supplied compatible small Qwen-family model, config JSON parsing/validation, real tensor-name mapping (and local shard assembly if needed), tokenizer/chat-template integration, prefill/decode calls, sampling, and coherent-text validation. It should select the supported structural subset rather than silently accept sliding-window, QK-normalized Qwen3, MoE, multimodal, or scaled-RoPE configurations. No pretrained checkpoint, network downloader, chat template, sampling loop, or generation command is included in Phase 2.


## Phase 3 production boundary

`model::qwen` parses the inspected official Qwen2.5 config, validates the dense
full-attention subset and BF16 storage, resolves mandatory official names and
Q/K/V biases, and rejects unrecognized tensors. Its construction-time remap
shares immutable tensor handles under the Phase 2 canonical names. Every shape,
dtype and device check precedes preprocessing dispatch. Runtime weight access
remains typed. `Transformer::weight_bytes` now sums the actual retained tensor
payloads, including the separate transposed tied LM allocation.

`tokenizer::qwen` validates pinned tokenizer/generation metadata, the exact
upstream chat template, 151665 defined IDs versus 151936 padded embedding rows,
and BOS/EOS/PAD mappings. It implements only the explicit system/user no-tools
branch. It uses the existing tokenizer wrapper and never runs Jinja or contacts
the network. Tokenizer maximum length does not override model context capacity.

`generation` validates prompt plus generation capacity, prefills once, copies
only the final vocabulary row through the existing checked copy kernel, performs
CPU selection, then repeatedly feeds one selected token to `forward_decode`.
The last generated token at EOS/limit is not appended to the cache. Zero requested
new tokens runs no inference. `sampling` supplies finite-logit greedy argmax,
F64 temperature/top-k/top-p probabilities and seeded ChaCha8 selection. It does
not call a CPU transformer or external engine. CPU reference code is confined
to explicit validation tools/tests.

`generate_greedy` (Phase 7B) is the greedy fast path. The model encodes a
row-argmax kernel in the same submission as the forward pass, after logits
scaling. The kernel takes the first maximal index, matching `argmax`, and
flags non-finite logits, which the host rejects exactly as `argmax` does.
Only the selected ID is read back. `ferrum run` uses it at temperature 0;
sampling still receives the full final row.

The `run` CLI composes these pieces, loads local files, drops source weights after
construction, uses tokenizers' incremental decoder for Unicode-safe text, and
reports model load, construction, tokenization, prefill, first selection, per-step
decode, counters and active KV separately. Streaming/text output is excluded
from decode kernel timings. Failed file/config/token/sampling/context checks
return contextual errors; there is no numerical fallback.

Opt-in `MetalDevice::set_profiling` aggregates operation timings in a small host
map and retains no tensors. Output allocation/zeroing and dispatch submission,
synchronized wall, and GPU timestamps are reported independently. Disabled
profiling does not change execution boundaries. The four unsafe blocks, Rc
ownership, shared storage, fresh output allocations and completion waits are
unchanged. No kernels were changed in Phase 3. See [Phase 3 results](phase3-results.md)
for exact model mapping, reference arithmetic differences, measured throughput,
and ranked Phase 4 recommendations.

## Phase 4 current execution and storage model

`MetalDevice` owns a single queue and an internal execution guard. Public tensor operations submit and wait synchronously. Transformer forward instead encodes separate tracked-hazard compute encoders into a bounded command buffer, default 1,024 dispatches. The measured model uses 603 dispatches and one completion boundary per forward. Limits 1/64/256/1024/8192 were measured; larger batches than 1024 did not improve the initial control. No broad threading or unsafe Send/Sync was introduced. Since Phase 7A Experiment 64, `MetalDevice::new()` keeps model weights in a queue residency set. It also starts one background keep-alive thread, holding only the `Send + Sync` command queue and a no-op pipeline, which dispatches an empty kernel every 500 ms for up to 3 minutes after the last submission so weight residency survives short idle gaps. `MetalDevice::without_keep_alive()` omits it.

M=1 (decode) projections use K-split GEMVs for Q5_0 (two SIMD groups per four-row tile) and factored Q4_K/Q6_K (group count chosen per projection, in `params[8]`), plus a four-row Q8_0 K-split GEMV (Phase 7A Experiments 70 and 72). A pre-norm Q5_0 MLP computes gate, up, and SiLU-multiply in one bit-identical kernel (`fuse_swiglu`, Experiment 71). Attention projections write RoPE'd K and biased V directly into their KV-cache slots (`fuse_rope_cache`, Experiment 67). Two fusions are implemented but off by default: residual add + next RMSNorm (`fuse_add_rmsnorm`, performance-neutral, Experiment 66) and split-context flash decode attention (`set_flash_decode`, Experiment 73).

Phase 7B encoder policy: a transformer forward opens its execution scope with
`execution_with_shared_encoder(!has_sparse_moe)`. For dense models, one compute
encoder spans each command buffer and uses the concurrent dispatch type. The
submission records the (buffer, byte range) reads and writes since the last
barrier. A dispatch whose inputs overlap pending writes, or whose outputs overlap
pending reads or writes, first encodes `memoryBarrierWithScope(Buffers)`.
Retained resources keep buffer identities unique for the epoch, and the arena
recycles only completed storage, so address reuse cannot hide a dependency.
Models with sparse-MoE layers, and public operations, keep one encoder per
kernel, because the measured M5 shared encoder slowed LFM2.5 short prefill.
`set_shared_encoder` and `set_concurrent_dispatch` are measurement toggles. See
Phase 7A journal Experiments 46–47.

Chunked sparse-MoE execution copies each
chunk's combined output into one preallocated device tensor. It has no host
readback or completion boundary per chunk. Chunks bound routing temporaries
to 16 MiB for dense experts and to the device's `moe_routing_temporary_limit`
(default 128 MiB) for quantized experts, whose TensorOps tiles benefit from
more routed rows per expert.

A `Submission` retains its command buffer and every bound raw buffer until completion. Completion states are Encoding/Completed/Failed. Finishing checks command-buffer status before publishing the staged cache; guard drop drains partial work on error or unwind. Each allocation tracks write ranges and their completion epochs. Mapping checks the requested logical byte range, so a failed suffix cannot invalidate a previously published KV prefix. Pending/failed output cannot be mapped. Failed storage is not recycled. Tests inject late encoding and completion failures and verify publication, retry, mapping and lifetime behavior.

Transient outputs created inside model execution use power-of-two size buckets. The arena retains at most 64 MiB of free storage; live/in-flight demand can exceed this. Reuse requires both completion and the last tensor owner dropping. Retired GPU-owned buffers count toward live/peak accounting until completion. Fresh physical storage is initialized; reused storage skips zeroing because audited kernels overwrite every logical output element before publication. Weights and persistent KV are excluded from the transient arena. Counters distinguish cumulative allocation volume, new/reused bytes, live/peak transient bytes and retained arena capacity.

Tensor views are restricted to checked contiguous subranges. Shape products, byte offsets, allocation bounds and alignment are checked; bindings and CPU reads carry offset and logical length. There are no arbitrary strides. Reshape shares storage and final-logit extraction is a view. Weight tensors remain row-major `[out,in]`; tied embedding/LM head share one allocation, with unique retained payload counted once.

KV retains logical maximum context separately from physical capacity. A first append reserves at least 256 tokens, rounded geometrically and capped at logical capacity. An append within physical capacity writes only the new suffix. Growth copies the active prefix once; reset drops storage. Immutable snapshots share a monotonic reservation watermark. A branch or retry whose offset no longer equals that watermark copies into fresh storage rather than overwriting a possibly visible/in-flight suffix. The whole model clones/stages the cache and publishes only after successful completion. The reservation may advance on a failed forward, but active published values and lengths remain unchanged. A 600-append test checks allocations only at 0/256/512, capped capacity and preserved snapshots; existing branch/transaction tests are unchanged.

Attention kernels consume sequence-major `[S,Q,D]` query and `[T,KV,D]` cache directly with checked GQA mapping. Scores are `[Q,S,T]`; context writes `[S,Q,D]` directly. Separate scaling/mask/softmax/storage round boundaries are preserved. This removes head selection, transpose, stack and merge materializations without introducing a general stride system. Ordinary decode has no layout-copy kernel; geometric KV growth and branch recovery remain explicit exceptions.

Low-precision M=1 projections use four rows per threadgroup, one SIMD group per output row. Aligned K divisible by four uses native `half4`/`bfloat4` loads and four independent F32 accumulation chains followed by SIMD reduction. Awkward widths or misaligned views use scalar GEMV. F32 projection retains ascending-order accumulation to preserve its recorded diagnostic tolerance. M>1 low-precision projection uses native SIMD-group 8×8 matrix tiles with FP32 accumulation on supported hardware. Scalar/tiled fallbacks remain, with explicit capability/compiler reporting.

RMSNorm and softmax use 256 threads per row and cross-SIMD shared reductions. Softmax retains max subtraction. RMSNorm uses the parallel sum of squares for every row. Phase 7A Experiment 59 removed the earlier BF16 midpoint safeguard, which re-summed a row serially whenever an output fell within eight F32 low-bit units of a BF16 rounding midpoint. That safeguard kept reference outputs bitwise stable but cost up to 19% of decode GPU time. Without it, two BF16 reference sequences reach exact logit ties, which their tests accept only within two BF16 steps. The Qwen2.5 BF16 real-model logit bound widened from 0.5 to 0.55, against a measured maximum of 0.516. `set_reference_math(true)` selects ordered RMSNorm/softmax for the explicit F32 reference diagnostic. Production BF16 uses the parallel paths.

Profiling distinguishes LM-head projections from transformer projections. Default batching reports GPU timestamps at command-buffer level; per-operation GPU samples are absent, not zero-cost kernels. `FERRUM_BATCH_LIMIT=1` provides isolated category attribution with additional submission/cache effects. Logical payload divided by GPU duration is an effective bandwidth proxy, not a DRAM counter. The four unsafe backend blocks plus linkage remain unchanged; no unsafe Rust was added outside this boundary.
