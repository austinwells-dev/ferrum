# Phase 1 completion report

Measured locally on 2026-09-20. Phase 1 is implemented and validated; Phase 2 has not begun.

## Device and environment

- Apple M5, arm64, macOS 27.0 build 26A428.
- Rust/Cargo 1.96.0; Rust edition 2024; deployment target macOS 15.0.
- Unified memory: true; recommended working set **25,769,803,776 bytes (24 GiB)**.
- Maximum Metal buffer length: **20,100,448,256 bytes**.
- Supports queried Apple GPU families 1–10.
- Ordinary desktop workload, no isolated power/thermal controls. Short measurements show scheduling variance and some outliers; these are baselines, not peak-throughput claims.

The initial release build failed with E0463 for proc-macro dylibs. Direct `dlopen` inspection found `mis-aligned LINKEDIT string pool` on this macOS 27 toolchain. Setting `[profile.release.build-override] strip = false` fixed the release and benchmark builds; runtime optimization is unchanged. This local finding matches an [upstream application report](https://github.com/crynta/terax-ai/issues/1050). The workaround is saved in Cargo.toml; no global toolchain/configuration changes were made.

## Files and architecture

| File | Role |
|---|---|
| `Cargo.toml`, `Cargo.lock` | Single crate, explicit binding features, locked dependencies, Criterion target, macOS release build workaround |
| `.cargo/config.toml` | macOS 15 deployment target for modern compile-option APIs |
| `.gitignore` | Excludes generated Cargo output |
| `src/lib.rs` | Public modules/types; unsafe denied outside the Metal boundary |
| `src/error.rs` | Contextual runtime error model |
| `src/metal/mod.rs` | Device, shared buffers, scoped mappings, cached libraries/pipelines, private encoding, completion, timing |
| `src/metal/shaders/ops.metal` | Seven independent runtime kernels and F16/BF16 conversion |
| `src/tensor/mod.rs` | Dtype, arbitrary-rank shape, contiguous layout, storage metadata, immutable tensors, reshape |
| `src/ops/mod.rs` | Validated tensor operations and per-operation metrics |
| `src/reference.rs` | Deterministic scalar CPU numerical oracles, separate from runtime execution |
| `src/main.rs` | `info` and numerical `smoke` CLI |
| `tests/correctness.rs` | Nine integration suites, all dtypes, shape/error/ownership/cache/numerical coverage |
| `benches/runtime.rs` | Synchronized Criterion benchmark plus separate instrumented timings |
| `README.md`, `docs/architecture.md` | Usage, safety, memory model, host/shader ABI, Phase 2 constraints |
| `docs/phase1-results.md`, `docs/measurements/*` | This report and raw measurement/validation logs |

The runtime uses one retained queue per device context, one embedded library, cached compute pipelines, immutable Rc-owned shared buffers, and a completion wait at each public operation boundary. It allocates fresh outputs. Shapes and strides stay inline through rank four; greater ranks are supported. Storage exposes allocation/logical lengths, actual alignment, ownership count, shared mode, and offset zero. Reshape does not copy data.

All four unsafe blocks are in `src/metal/mod.rs`: scoped mapped read, scoped mapped initialization, allocation zeroing, and encoder bindings. An empty unsafe extern block links CoreGraphics. No Objective-C types or raw pointers leak into tensor/operation code, and no unsafe Send/Sync implementations exist. `MTLStorageModeShared` allows CPU initialization and GPU execution against the same allocation without staging copies. Completion still must precede CPU inspection. GPU shader bounds and the host/shader ABI remain trust boundaries.

## Exact direct dependencies

| Dependency | Locked version | Role |
|---|---|---|
| `objc2` | 0.6.4 | Objective-C ownership/runtime |
| `objc2-metal` | 0.3.2 | Metal bindings; required compute/resource features only |
| `objc2-foundation` | 0.3.2 | NSString/NSError and binding support |
| `half` | 2.7.1 | F16/BF16 CPU storage conversion |
| `smallvec` | 1.16.1 | Inline small-rank tensor metadata |
| `thiserror` | 2.0.20 | Contextual errors |
| `criterion` (development only) | 0.8.2 | Statistical benchmarking |

The lockfile records transitive versions. Apple's Metal/Foundation/CoreGraphics frameworks supply the OS implementation. There are no ML engine or Python dependencies and no runtime CPU fallback.

## Numerical validation

**Nine integration suites pass**, including **130 reported CPU/GPU comparisons** plus explicit checks for BF16 rounding/NaN, zero outputs, ownership, invalid inputs, and caching. The same nine suites pass with both Metal API validation and GPU shader validation enabled. All seven release smoke operations pass.

Inputs are deterministic and quantized to the input dtype before evaluating CPU references. Reductions/matmul/RoPE use F64 CPU intermediates; expected outputs are rounded to the output dtype. Tolerance is `abs_error <= atol + rtol * abs(expected)`:

- F32: atol=3e-5, rtol=3e-5; RoPE uses atol=5e-4 for tested positions through 2048.
- F16: atol=2e-3, rtol=2e-3.
- BF16: atol=1.6e-2, rtol=1e-2.

Maximum absolute errors across the reported cases (not error bounds):

| Kernel | F32 | F16 | BF16 |
|---|---:|---:|---:|
| Add | 0 | 0 | 0 |
| Multiply | 0 | 0 | 0 |
| SiLU | 1.788139e-7 | 0 | 0 |
| RMSNorm | 3.099442e-6 | 0 | 0 |
| Softmax | 2.980232e-8 | 2.384186e-7 | 0 |
| RoPE | 1.429319e-4 | 9.765625e-4 | 3.906250e-3 |
| Matmul | 5.035400e-4 | 1.525879e-5 | 0 |

The largest F32 matmul absolute error passes the combined absolute/relative criterion; it is not below the absolute tolerance alone. Zero low-precision error means exact agreement with the rounded CPU oracle for these inputs, not exact real-number arithmetic.

Coverage includes scalars, zero length, rank six, one element, 257-element grids, odd widths 7/129, width 4096, nonmultiple matmul tiles `(17,19,23)`, `(2,4096,33)` projection-like products, zero M/K/N, softmax near ±1000, RoPE position 0/7/127/2048 and theta 10000/500000, invalid dtype/shape/reshape/epsilon/theta/context, retained storage after context destruction, immutable aliases, chains of GPU outputs, compile errors, missing entry points, and pipeline reuse.

## Benchmarks

Pipeline warm-up for the benchmark process: **38.38325 ms**. This includes all seven pipeline resolutions and can benefit from Apple's persistent compiler cache.

Each Criterion case used 10 samples, 300 ms warm-up, and a 1 s target measurement period. These are short baseline runs. Inputs and pipelines are prepared in advance; output allocation/zeroing and synchronization are included. Criterion's reported estimate and 95% confidence interval are shown below in microseconds. Matmul shapes mean M×K×N.

| Operation / dtype / shape | Estimate µs | 95% interval µs |
|---|---:|---:|
| Add F32 / 1,024 | 185.61 | 180.84–189.21 |
| Add F32 / 65,536 | 202.64 | 200.39–204.84 |
| Add F32 / 1,048,576 | 436.34 | 434.26–438.46 |
| RMSNorm F32 / 1×128 | 218.98 | 217.50–219.74 |
| RMSNorm F32 / 4×4096 | 854.11 | 851.98–856.96 |
| RMSNorm F32 / 32×4096 | 913.80 | 906.16–928.34 |
| Softmax F32 / 1×128 | 223.38 | 221.69–225.53 |
| Softmax F32 / 4×4096 | 1005.4 | 1003.5–1007.1 |
| Softmax F32 / 32×4096 | 1108.6 | 1103.3–1114.1 |
| Matmul F32 / 17×19×23 | 180.89 | 174.34–191.62 |
| Matmul F32 / 128×128×128 | 239.74 | 207.82–283.25 |
| Matmul F32 / 256×512×256 | 399.07 | 350.52–473.01 |
| Matmul F16 / 256×512×256 | 328.78 | 326.55–330.92 |

Separate instrumentation, medians of 30 invocations after five warm-ups; all values µs:

| Case | Public call | CPU submission | Synchronized dispatch | Metal GPU |
|---|---:|---:|---:|---:|
| Add F32 / 1,024 | 594.042 | 8.042 | 590.041 | 7.292 |
| Add F32 / 65,536 | 178.792 | 3.250 | 161.000 | 4.542 |
| Add F32 / 1,048,576 | 405.416 | 3.667 | 233.375 | 67.250 |
| RMSNorm F32 / 1×128 | 200.333 | 4.500 | 197.542 | 21.208 |
| RMSNorm F32 / 4×4096 | 843.583 | 11.334 | 818.542 | 628.250 |
| RMSNorm F32 / 32×4096 | 872.417 | 3.625 | 812.792 | 633.542 |
| Softmax F32 / 1×128 | 207.125 | 3.167 | 204.209 | 25.917 |
| Softmax F32 / 4×4096 | 992.708 | 10.875 | 968.417 | 781.125 |
| Softmax F32 / 32×4096 | 1084.208 | 3.500 | 1023.250 | 840.625 |
| Matmul F32 / 17×19×23 | 174.708 | 3.167 | 171.958 | 2.417 |
| Matmul F32 / 128×128×128 | 188.000 | 4.334 | 179.042 | 10.875 |
| Matmul F32 / 256×512×256 | 338.584 | 4.125 | 321.167 | 149.708 |
| Matmul F16 / 256×512×256 | 308.709 | 3.375 | 295.125 | 126.625 |

These instrumentation samples are separate from Criterion's samples. Their medians need not match Criterion estimates; the first small-add sample set showed particularly large host scheduling/wakeup cost. Public-call instrumentation stops before output disposal; Criterion includes disposal. GPU timestamps are command-buffer durations, not isolated hardware counter measurements. Logical byte metrics are not measured DRAM traffic. The serial per-row reductions clearly limit larger RMSNorm/softmax throughput. No optimization or cross-engine performance claim follows from these numbers.

## Quality gates and reproduction

Every requested command succeeded locally:

```sh
cargo fmt --check
cargo clippy --all-targets --all-features -- -D warnings
cargo test
cargo build --release
cargo run --release -- info
cargo run --release -- smoke
cargo bench --bench runtime -- --noplot
MTL_DEBUG_LAYER=1 MTL_SHADER_VALIDATION=1 cargo test --test correctness -- --test-threads=1
cargo test --test correctness -- --nocapture --test-threads=1
```

Raw logs: [quality gates](measurements/quality-gates.txt), [device](measurements/device.txt), [smoke](measurements/smoke.txt), [correctness](measurements/correctness.txt), [Metal validation](measurements/metal-validation.txt), [benchmarks](measurements/benchmark.txt). Criterion JSON remains in `target/criterion` as generated output.

## Limits and Phase 2 handoff

The Mac exposes all capabilities used here. The local macOS 27 proc-macro loading issue is handled by the checked-in build override. No remaining local blocker was found. The test run establishes behavior on this M5, not every Apple GPU or macOS release.

The runtime is synchronous, single-threaded, contiguous-only, and allocates/zeros outputs per call. Reductions are serial within each row. All arithmetic uses F32 even for F16/BF16 storage. Indexing is u32; matmul is rank two. Finite numerical behavior is tested within representative ranges, not every extreme IEEE-754 case. Long-context RoPE precision beyond tested positions is unestablished; its interleaved pairing must be matched to the eventual model.

Before Phase 2 removes waits, reuses storage, or adds batching, it must introduce explicit completion/lifetime ownership and CPU mapping exclusion. Keep offset/range/alignment checks when adding views or arenas. Do not make `Rc` resources thread-safe without a synchronization design. Preserve the independent CPU oracle tests and use the saved benchmark baseline when changing dispatch or allocation behavior. Model formats, transformer execution, attention, KV caches, and all other Phase 2 functionality remain absent.
