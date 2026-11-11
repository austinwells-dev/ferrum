//! Draft-tree verify attention. Every tree-capable kernel (the reference
//! `h_attention`, and the `_tree` variants of the four `h_attn_verify_mpp_32*`
//! kernels and of the staged TensorOps prompt kernels) runs random trees against a CPU reference over
//! F16 and Q8 caches, 256- and 128-wide heads, sinks, windows and rings. A
//! chain-shaped tree must reproduce the tree-off output bit for bit, and the
//! tree's rotary positions must equal plain RoPE at each node's depth.
use super::config::KvCache;
use super::engine::{Params, TREE_MAX_ROWS, tree_plan, tree_rows};
use crate::{DType, MetalDevice, Tensor};

fn random(n: usize, seed: u64) -> Vec<f32> {
    let mut x = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    (0..n)
        .map(|_| {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            ((x >> 40) as f32 / (1u64 << 24) as f32) * 2. - 1.
        })
        .collect()
}

/// llama.cpp q8_0 rounding of `x` in the Q8 cache layout (every value, then
/// every block scale), and the values the kernels see after dequantizing to F16.
fn quantize_q8(x: &[f32]) -> (Vec<u8>, Vec<f32>) {
    let mut values = Vec::with_capacity(x.len());
    let mut scales = Vec::new();
    let mut seen = Vec::with_capacity(x.len());
    for block in x.chunks(32) {
        let amax = block.iter().fold(0f32, |a, v| a.max(v.abs()));
        let d = half::f16::from_f32(amax / 127.);
        let id = if d.to_f32() > 0. { 1. / d.to_f32() } else { 0. };
        for v in block {
            let q = (v * id).round_ties_even().clamp(-127., 127.) as i8;
            values.push(q as u8);
            seen.push(f16(f32::from(q) * d.to_f32()));
        }
        scales.extend_from_slice(&d.to_le_bytes());
    }
    values.extend_from_slice(&scales);
    (values, seen)
}

fn f16(x: f32) -> f32 {
    half::f16::from_f32(x).to_f32()
}

/// Parents of `m` nodes in one of several shapes (row 0 is the root).
fn tree(shape: &str, m: usize, seed: u64) -> Vec<Option<usize>> {
    let r = random(m, seed);
    (0..m)
        .map(|i| match (i, shape) {
            (0, _) => None,
            (_, "chain") => Some(i - 1),
            (_, "star") => Some(0),
            (_, "binary") => Some((i - 1) / 2),
            // Uniform parents, and parents biased to the newest node (deep,
            // DFS-like trees whose depths are far from monotonic).
            (_, "random") => Some(((r[i] + 1.) / 2. * i as f32) as usize % i),
            (_, "deep") => Some(if r[i] > -0.4 { i - 1 } else { i / 3 }),
            // A star whose last leaf grows a long chain: deep nodes whose
            // ancestors sit far later in the rows than their depth, which a
            // window shorter than the chain tells apart from row order.
            (_, "late") => Some(if i < m / 2 { 0 } else { i - 1 }),
            _ => unreachable!(),
        })
        .collect()
}

#[test]
fn tree_rows_hold_ancestor_masks_and_depths() {
    let rows = tree_rows(&[None, Some(0), Some(0), Some(1), Some(3), Some(2)]).unwrap();
    assert_eq!(
        rows,
        vec![
            (0b1, 0),
            (0b11, 1),
            (0b101, 1),
            (0b1011, 2),
            (0b11011, 3),
            (0b100101, 2)
        ]
    );
    assert_eq!(tree_rows(&[None]).unwrap(), vec![(1, 0)]);
    // A chain is the causal mask.
    let chain = tree_rows(&tree("chain", 32, 0)).unwrap();
    assert!(
        chain
            .iter()
            .enumerate()
            .all(|(i, &(mask, depth))| mask == ((1u64 << (i + 1)) - 1) as u32 && depth == i as u32)
    );
    for bad in [
        vec![],
        vec![Some(0)],
        vec![None, None],
        vec![None, Some(1)],
        vec![None, Some(0), Some(3), Some(1)],
    ] {
        assert!(tree_rows(&bad).is_err(), "{bad:?}");
    }
    assert!(tree_rows(&vec![None; TREE_MAX_ROWS + 1]).is_err());
}

#[derive(Clone, Copy, Debug)]
struct Case {
    heads: usize,
    kv_heads: usize,
    dim: usize,
    m: usize,
    pos: usize,
    window: usize,
    sinks: bool,
    gated: bool,
}

fn cases() -> Vec<Case> {
    let mut out = Vec::new();
    // Qwen-shaped 256-wide gated heads (two GQA shapes), and Flint/Spark-shaped
    // 128-wide heads with sinks, sliding windows (a small one shorter than the
    // deepest trees) and no gate.
    let geometry = [
        (16, 2, 256, vec![(0, false, true)]),
        (24, 4, 256, vec![(0, false, true)]),
        (
            32,
            4,
            128,
            vec![
                (0, true, true),
                (513, true, false),
                (40, false, false),
                (6, true, false),
            ],
        ),
    ];
    for (heads, kv_heads, dim, features) in geometry {
        for m in [2usize, 5, 9, 15, 16, 25, 32] {
            for pos in [0usize, 77, 1000] {
                for &(window, sinks, gated) in &features {
                    out.push(Case {
                        heads,
                        kv_heads,
                        dim,
                        m,
                        pos,
                        window,
                        sinks,
                        gated,
                    });
                }
            }
        }
    }
    out
}

/// One cache layout of a case: linear, or a ring that wraps the keys.
struct Cache {
    kv: KvCache,
    ring: usize,
    rows: usize,
    k: Tensor,
    v: Tensor,
    /// F16 cache of the values the kernels see, in the same layout (the
    /// reference kernel reads only F16).
    k_seen: Tensor,
    v_seen: Tensor,
    /// The same values per key position, for the CPU reference.
    k_cpu: Vec<f32>,
    v_cpu: Vec<f32>,
}

fn caches(d: &MetalDevice, c: Case) -> Vec<Cache> {
    let stride = c.kv_heads * c.dim;
    let keys = c.pos + c.m;
    let ring_rows = if c.window > 0 {
        (c.window + 62 + c.m).next_multiple_of(64)
    } else {
        0
    };
    let mut rings = vec![0];
    if ring_rows > 0 && ring_rows < keys {
        rings.push(ring_rows);
    }
    let (k_raw, v_raw) = (random(keys * stride, 2), random(keys * stride, 3));
    let mut out = Vec::new();
    for kv in [KvCache::F16, KvCache::Q8] {
        for &ring in &rings {
            let rows = if ring > 0 {
                ring
            } else {
                keys.next_multiple_of(32)
            };
            // Key j lives in row j % ring, as `h_attn_prep` stores it; rows
            // no key maps to (or that a later key overwrote) hold noise.
            let lay = |raw: &[f32]| {
                let mut out = random(rows * stride, 9);
                for j in 0..keys {
                    let r = if ring > 0 { j % ring } else { j };
                    out[r * stride..][..stride].copy_from_slice(&raw[j * stride..][..stride]);
                }
                out
            };
            let (kl, vl) = (lay(&k_raw), lay(&v_raw));
            let encode = |x: &[f32]| -> (Tensor, Vec<f32>) {
                match kv {
                    KvCache::F16 => (
                        Tensor::from_f32(d, [rows, stride], DType::F16, x).unwrap(),
                        x.iter().map(|&v| f16(v)).collect(),
                    ),
                    KvCache::Q8 => {
                        let (b, seen) = quantize_q8(x);
                        let t = Tensor::from_le_bytes(
                            d,
                            [rows, kv.row_bytes(stride) / 2],
                            DType::F16,
                            &b,
                        )
                        .unwrap();
                        (t, seen)
                    }
                }
            };
            let ((k, ks), (v, vs)) = (encode(&kl), encode(&vl));
            let by_key = |seen: &[f32]| -> Vec<f32> {
                (0..keys)
                    .flat_map(|j| {
                        let r = if ring > 0 { j % ring } else { j };
                        seen[r * stride..][..stride].to_vec()
                    })
                    .collect()
            };
            out.push(Cache {
                kv,
                ring,
                rows,
                k_cpu: by_key(&ks),
                v_cpu: by_key(&vs),
                k_seen: Tensor::from_f32(d, [rows, stride], DType::F16, &ks).unwrap(),
                v_seen: Tensor::from_f32(d, [rows, stride], DType::F16, &vs).unwrap(),
                k,
                v,
            });
        }
    }
    out
}

/// Inputs shared by every kernel of a case.
struct Inputs {
    q: Tensor,
    q_cpu: Vec<f32>,
    qg: Tensor,
    qg_cpu: Vec<f32>,
    sinks: Tensor,
    sinks_cpu: Vec<f32>,
}

fn inputs(d: &MetalDevice, c: Case) -> Inputs {
    let q_raw: Vec<f32> = random(c.m * c.heads * c.dim, 1)
        .iter()
        .map(|v| v / 16.)
        .collect();
    let qg_cpu = random(c.m * 2 * c.heads * c.dim, 4);
    let sinks_cpu = random(c.heads, 5);
    Inputs {
        q: Tensor::from_f32(d, [c.m, c.heads * c.dim], DType::F16, &q_raw).unwrap(),
        q_cpu: q_raw.iter().map(|&v| f16(v)).collect(),
        qg: Tensor::from_f32(d, [c.m, 2 * c.heads * c.dim], DType::F32, &qg_cpu).unwrap(),
        qg_cpu,
        sinks: Tensor::from_f32(d, [c.heads], DType::F32, &sinks_cpu).unwrap(),
        sinks_cpu,
    }
}

/// Attention of the case's rows over its keys in F64, with the visibility
/// rules of `KeyMask` (`rows` from `tree_rows`; None = causal rows).
fn cpu_attention(c: Case, x: &Inputs, cache: &Cache, rows: Option<&[(u32, u32)]>) -> Vec<f32> {
    let (heads, dim, m, pos) = (c.heads, c.dim, c.m, c.pos);
    let stride = c.kv_heads * dim;
    let group = heads / c.kv_heads;
    let qpos = |t: usize| pos + rows.map_or(t, |r| r[t].1 as usize);
    let visible = |t: usize, key: usize| -> bool {
        let kpos = match rows {
            None if key > pos + t => return false,
            None => key,
            Some(r) if key >= pos => {
                if r[t].0 >> (key - pos) & 1 == 0 {
                    return false;
                }
                pos + r[key - pos].1 as usize
            }
            Some(_) => key,
        };
        c.window == 0 || kpos + c.window > qpos(t)
    };
    let mut out = vec![0f32; m * heads * dim];
    for t in 0..m {
        for h in 0..heads {
            let kvh = h / group;
            let q = &x.q_cpu[(t * heads + h) * dim..][..dim];
            let mut scores: Vec<(usize, f64)> = Vec::new();
            for key in 0..pos + m {
                if !visible(t, key) {
                    continue;
                }
                let k = &cache.k_cpu[key * stride + kvh * dim..][..dim];
                let s: f64 = q.iter().zip(k).map(|(a, b)| f64::from(a * b)).sum();
                scores.push((key, s));
            }
            let sink = c.sinks.then(|| f64::from(x.sinks_cpu[h]));
            let max = scores
                .iter()
                .map(|s| s.1)
                .chain(sink)
                .fold(f64::NEG_INFINITY, f64::max);
            let mut total = sink.map_or(0., |s| (s - max).exp());
            let mut acc = vec![0f64; dim];
            for &(key, s) in &scores {
                let w = (s - max).exp();
                total += w;
                let v = &cache.v_cpu[key * stride + kvh * dim..][..dim];
                for (a, &b) in acc.iter_mut().zip(v) {
                    *a += w * f64::from(b);
                }
            }
            for (i, a) in acc.iter().enumerate() {
                let mut y = a / total;
                if c.gated {
                    let gate = f64::from(x.qg_cpu[(t * heads + h) * 2 * dim + dim + i]);
                    y /= 1. + (-gate).exp();
                }
                out[(t * heads + h) * dim + i] = y as f32;
            }
        }
    }
    out
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum Kernel {
    /// `h_attention` on the F16 copy of what the kernels see.
    Reference,
    /// `h_attn_verify_mpp_32*` with `splits` key splits.
    Verify(usize),
    /// `h_flash_attn_mpp_staged*` / `h_flash_attn_mpp_q8*`.
    Staged,
}

/// Run one kernel over the case's rows `rows` with `tree` (its `_tree`
/// variant, rows `rows` of the tree) or without, and return its output.
fn run(
    d: &MetalDevice,
    c: Case,
    x: &Inputs,
    cache: &Cache,
    kernel: Kernel,
    tree: Option<&Tensor>,
    rows: std::ops::Range<usize>,
) -> Vec<f32> {
    let (heads, kv_heads, dim) = (c.heads, c.kv_heads, c.dim);
    let (m, pos) = (rows.len(), c.pos + rows.start);
    let q =
        x.q.view(rows.start * heads * dim, [m, heads * dim])
            .unwrap();
    let qg =
        x.qg.view(rows.start * 2 * heads * dim, [m, 2 * heads * dim])
            .unwrap();
    let group = heads / kv_heads;
    let stride = kv_heads * dim;
    let q8 = cache.kv == KvCache::Q8;
    let suffix = match (q8, dim) {
        (false, 256) => "",
        (true, 256) => "_q8",
        (false, _) => "_d128",
        (true, _) => "_q8_d128",
    };
    let tree_on = usize::from(tree.is_some());
    let tree_t = tree.unwrap_or(&qg);
    let sinks_t = if c.sinks { &x.sinks } else { &qg };
    let out = Tensor::zeros(d, [m, heads * dim], DType::F32).unwrap();
    let e = d.execution_with_shared_encoder(true).unwrap();
    if kernel == Kernel::Reference {
        let mut args = Params::default();
        for v in [m, heads, kv_heads, dim, pos, stride] {
            args = args.u(v).unwrap();
        }
        let mut args = args.f(1.);
        for v in [
            c.window,
            usize::from(c.gated),
            usize::from(c.sinks),
            cache.ring,
            tree_on,
        ] {
            args = args.u(v).unwrap();
        }
        d.dispatch_hybrid(
            "h_attention",
            &[
                q.binding(),
                cache.k_seen.binding(),
                cache.v_seen.binding(),
                qg.binding(),
                sinks_t.binding(),
                tree_t.binding(),
            ],
            &[out.binding()],
            &args.0,
            [heads, m, 1],
            [dim, 1, 1],
            0,
        )
        .unwrap();
        e.finish().unwrap();
        return out.to_f32();
    }
    // As `flash_attention` routes them: keys from the anchor's window start,
    // aligned to 64 in a ring.
    // From the window of the tree's root (the case's first row), the lowest.
    let first_key = if c.window > 0 {
        (c.pos + 1).saturating_sub(c.window)
    } else {
        0
    };
    let first_key = if cache.ring > 0 {
        first_key / 64 * 64
    } else {
        first_key
    };
    let span = pos + m - first_key;
    let (kps, splits) = match kernel {
        Kernel::Verify(wanted) => {
            let kps = span.div_ceil(wanted).next_multiple_of(64);
            (kps, span.div_ceil(kps))
        }
        _ => (0, 1),
    };
    let mut args = Params::default();
    for v in [
        m,
        heads,
        kv_heads,
        group,
        pos,
        stride,
        kps,
        splits,
        cache.rows,
        dim,
        first_key,
        c.window,
        usize::from(c.gated),
        usize::from(c.sinks),
        cache.ring,
    ] {
        args = args.u(v).unwrap();
    }
    // Trees run on the `_tree` variants, which take the tree after the sinks.
    let tree_suffix = if tree.is_some() { "_tree" } else { "" };
    let partial = Tensor::zeros(d, [splits * m * heads * dim], DType::F32).unwrap();
    let ml = Tensor::zeros(d, [splits * m * heads * 2], DType::F32).unwrap();
    let qk = Tensor::zeros(d, [m, heads * dim], DType::F16).unwrap();
    let bind =
        |name: String, q: &Tensor, outputs: &[&Tensor], groups: [usize; 3], shared: usize| {
            let name: &'static str = Box::leak(format!("{name}{tree_suffix}").into_boxed_str());
            let inputs = [
                q.binding(),
                cache.k.binding(),
                cache.v.binding(),
                qg.binding(),
                sinks_t.binding(),
                tree_t.binding(),
            ];
            d.dispatch_hybrid(
                name,
                &inputs[..5 + tree_on],
                &outputs.iter().map(|t| t.binding()).collect::<Vec<_>>(),
                &args.0,
                groups,
                [128, 1, 1],
                shared,
            )
            .unwrap();
        };
    match kernel {
        Kernel::Verify(_) => {
            d.dispatch_hybrid(
                "h_q_by_kv_head",
                &[q.binding()],
                &[qk.binding()],
                &args.0,
                [(m * heads * dim).div_ceil(256), 1, 1],
                [256, 1, 1],
                0,
            )
            .unwrap();
            bind(
                format!("h_attn_verify_mpp_32{suffix}"),
                &qk,
                &[&out, &partial, &ml],
                [(m * group).div_ceil(32), kv_heads, splits],
                if q8 {
                    64 * 128 * 2 + 32 * 64 * 2 + 3 * 32 * 4
                } else {
                    32 * 64 * 4 + 32 * 64 * 2 + 3 * 32 * 4
                },
            );
        }
        _ => {
            let name = match (q8, dim) {
                (false, 256) => "h_flash_attn_mpp_staged",
                (true, 256) => "h_flash_attn_mpp_q8",
                (false, _) => "h_flash_attn_mpp_staged_d128",
                (true, _) => "h_flash_attn_mpp_q8_d128",
            };
            bind(
                name.to_owned(),
                &q,
                &[&out],
                [m.div_ceil(64), heads, 1],
                64 * 64 * 4 + 64 * 64 * 2 + 3 * 64 * 4,
            );
        }
    }
    if splits > 1 {
        d.dispatch_hybrid(
            if dim == 256 {
                "h_flash_reduce"
            } else {
                "h_flash_reduce_d128"
            },
            &[
                partial.binding(),
                ml.binding(),
                qg.binding(),
                sinks_t.binding(),
            ],
            &[out.binding()],
            &args.0,
            [m * heads, 1, 1],
            [dim, 1, 1],
            0,
        )
        .unwrap();
    }
    e.finish().unwrap();
    out.to_f32()
}

fn worst(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0., f32::max)
}

/// Bound on |kernel - F64 reference| (F16 queries, keys, values and
/// probabilities). The same bound holds for causal rows, checked alongside.
const TOLERANCE: f32 = 2e-3;

#[test]
fn tree_attention_matches_the_cpu_and_a_chain_matches_causal_rows() {
    let d = MetalDevice::new().unwrap();
    let (mut checked, mut worst_causal, mut worst_tree) = (0, 0f32, 0f32);
    for c in cases() {
        let x = inputs(&d, c);
        let shapes: &[&str] = if c.m <= 2 {
            &["chain"]
        } else {
            &["chain", "star", "binary", "random", "deep", "late"]
        };
        for cache in caches(&d, c) {
            let mut kernels = vec![Kernel::Reference, Kernel::Verify(1), Kernel::Verify(3)];
            if c.m >= 16 {
                kernels.push(Kernel::Staged);
            }
            let causal = cpu_attention(c, &x, &cache, None);
            for &kernel in &kernels {
                let label = format!("{c:?} {} ring {} {kernel:?}", cache.kv, cache.ring);
                let off = run(&d, c, &x, &cache, kernel, None, 0..c.m);
                let err = worst(&off, &causal);
                assert!(err < TOLERANCE, "{label} causal: max error {err}");
                worst_causal = worst_causal.max(err);
                for (seed, shape) in shapes.iter().enumerate() {
                    let parents = tree(shape, c.m, seed as u64 + 11);
                    let rows = tree_rows(&parents).unwrap();
                    let plan = tree_plan(&d, c.pos, c.pos as i32, &parents).unwrap();
                    let got = run(&d, c, &x, &cache, kernel, plan.tree.as_ref(), 0..c.m);
                    if *shape == "chain" {
                        let same = got
                            .iter()
                            .zip(&off)
                            .all(|(a, b)| a.to_bits() == b.to_bits());
                        assert!(same, "{label}: a chain-shaped tree changed the output");
                    }
                    let want = cpu_attention(c, &x, &cache, Some(&rows));
                    let err = worst(&got, &want);
                    assert!(err < TOLERANCE, "{label} {shape} tree: max error {err}");
                    worst_tree = worst_tree.max(err);
                    checked += 1;
                    // The later half of the nodes alone, as a drafter
                    // expanding a tree in waves runs them: earlier nodes'
                    // K/V are already cached, siblings stay hidden.
                    // The later nodes alone, as a drafter expanding a tree in
                    // waves runs them: earlier nodes' K/V are already cached,
                    // and siblings stay hidden. Half the nodes, and the last.
                    let splits: &[usize] = if c.m >= 4 { &[c.m / 2, c.m - 1] } else { &[] };
                    for &split in splits.iter().filter(|_| kernel != Kernel::Staged) {
                        let plan = super::engine::tree_plan_rows(
                            &d,
                            c.pos,
                            c.pos as i32,
                            &parents,
                            split..c.m,
                        )
                        .unwrap();
                        let got = run(&d, c, &x, &cache, kernel, plan.tree.as_ref(), split..c.m);
                        let row = c.heads * c.dim;
                        let err = worst(&got, &want[split * row..]);
                        assert!(
                            err < TOLERANCE,
                            "{label} {shape} tree rows {split}..: max error {err}"
                        );
                        worst_tree = worst_tree.max(err);
                        checked += 1;
                    }
                }
            }
        }
    }
    assert!(checked > 1000, "{checked} tree runs");
    eprintln!(
        "{checked} tree runs; worst error vs F64: causal rows {worst_causal:.2e}, trees {worst_tree:.2e}"
    );
}

/// A tree's position table puts node i, stored in cache row pos + i, at
/// rotary position pos + depth(i): its query and key must be bit-identical to
/// plain RoPE of that row alone at pos + depth(i).
#[test]
fn tree_rotary_positions_are_plain_rope_at_depth() {
    let d = MetalDevice::new().unwrap();
    let (heads, kv_heads, dim, rope_dims) = (4usize, 2usize, 256usize, 64usize);
    let (theta, eps, scale) = (1.0e7f32, 1e-6f32, 0.0625f32);
    let (m, pos) = (12usize, 300usize);
    let parents = tree("deep", m, 3);
    let rows = tree_rows(&parents).unwrap();
    let plan = tree_plan(&d, pos, pos as i32, &parents).unwrap();
    let kv_width = kv_heads * dim;
    let kv_rows = pos + m;
    let qg = random(m * heads * 2 * dim, 1);
    let kk = random(m * kv_width, 2);
    let norm = Tensor::from_f32(&d, [dim], DType::F32, &vec![1.; dim]).unwrap();
    // h_attn_prep of rows `r` at cache position `at`, optionally from a table.
    let prep =
        |r: std::ops::Range<usize>, at: usize, table: Option<&Tensor>| -> (Vec<f32>, Vec<f32>) {
            let n = r.len();
            let t = |x: &[f32], w: usize| {
                Tensor::from_f32(&d, [n * w], DType::F32, &x[r.start * w..r.end * w]).unwrap()
            };
            let (qg_t, k_t) = (t(&qg, heads * 2 * dim), t(&kk, kv_width));
            let q_out = Tensor::zeros(&d, [n * heads * dim], DType::F16).unwrap();
            let cache = Tensor::zeros(&d, [kv_rows * kv_width], DType::F16).unwrap();
            let mut p = Params::default();
            for v in [n, heads, kv_heads, dim, rope_dims, at] {
                p = p.u(v).unwrap();
            }
            let mut p = p.f(theta).f(eps).u(kv_width).unwrap().f(scale).i(0);
            for v in [usize::from(table.is_some()), 0, 0, kv_rows, 1, 0, 1, 0] {
                p = p.u(v).unwrap();
            }
            let e = d.execution_with_shared_encoder(true).unwrap();
            d.dispatch_hybrid(
                "h_attn_prep",
                &[
                    qg_t.binding(),
                    k_t.binding(),
                    k_t.binding(),
                    norm.binding(),
                    norm.binding(),
                    table.unwrap_or(&qg_t).binding(),
                    qg_t.binding(),
                ],
                &[q_out.binding(), cache.binding()],
                &p.0,
                [heads + kv_heads, n, 1],
                [dim, 1, 1],
                0,
            )
            .unwrap();
            e.finish().unwrap();
            (q_out.to_f32(), cache.to_f32())
        };
    let (q_tree, k_tree) = prep(0..m, pos, plan.table.as_ref());
    let bits = |x: &[f32]| x.iter().map(|v| v.to_bits()).collect::<Vec<_>>();
    for (i, &(_, depth)) in rows.iter().enumerate() {
        let at = pos + depth as usize;
        let (q_one, k_one) = prep(i..i + 1, at, None);
        assert_eq!(
            bits(&q_tree[i * heads * dim..][..heads * dim]),
            bits(&q_one),
            "node {i} query"
        );
        assert_eq!(
            bits(&k_tree[(pos + i) * kv_width..][..kv_width]),
            bits(&k_one[at * kv_width..][..kv_width]),
            "node {i} key"
        );
    }
}

/// `commit_path`'s K/V move: rows of an accepted path (gathered, then
/// scattered back to consecutive positions) land byte for byte where a chain
/// would have stored them, with Q8 scales, across a ring's wrap, and with
/// every other row untouched.
#[test]
fn kv_rows_move_to_the_path_positions() {
    let d = MetalDevice::new().unwrap();
    let stride = 4 * 64;
    let path = [0usize, 2, 5, 6, 9];
    let first = 1; // path[0] is already in place
    for kv in [KvCache::F16, KvCache::Q8] {
        // A linear cache, and a ring whose rows wrap between the moved slots.
        for (rows, ring, position) in [(160usize, 0usize, 70usize), (128, 128, 122)] {
            let q8 = kv == KvCache::Q8;
            let bytes = rows * kv.row_bytes(stride);
            let raw: Vec<u8> = random(bytes, 21)
                .iter()
                .map(|v| ((v + 1.) * 127.5) as u8)
                .collect();
            // Raw bytes; F32 so they read back unconverted.
            let cache = Tensor::from_le_bytes(&d, [bytes / 4], DType::F32, &raw).unwrap();
            let index = |v: &[usize]| {
                Tensor::from_le_bytes(
                    &d,
                    [v.len()],
                    DType::F32,
                    &v.iter()
                        .flat_map(|&i| (i as u32).to_le_bytes())
                        .collect::<Vec<_>>(),
                )
                .unwrap()
            };
            let from = index(&path[first..]);
            let to = index(&(first..path.len()).collect::<Vec<_>>());
            let tmp = Tensor::zeros(&d, [path.len() * stride], DType::F32).unwrap();
            let moved = path.len() - first;
            let mut args = Params::default();
            for v in [moved, stride, rows, usize::from(q8), ring, position] {
                args = args.u(v).unwrap();
            }
            let e = d.execution_with_shared_encoder(true).unwrap();
            for (name, input, idx, output) in [
                ("h_kv_gather", &cache, &from, &tmp),
                ("h_kv_scatter", &tmp, &to, &cache),
            ] {
                d.dispatch_hybrid(
                    name,
                    &[input.binding(), idx.binding()],
                    &[output.binding()],
                    &args.0,
                    [(moved * stride).div_ceil(256), 1, 1],
                    [256, 1, 1],
                    0,
                )
                .unwrap();
            }
            e.finish().unwrap();
            // CPU: the same rows (values, and for Q8 their block scales) moved.
            let row = |slot: usize| {
                let r = position + slot;
                if ring > 0 { r % ring } else { r }
            };
            let (value_bytes, scale_bytes) = if q8 {
                (stride, stride / 16)
            } else {
                (2 * stride, 0)
            };
            let mut want = raw.clone();
            for (k, &slot) in path.iter().enumerate().skip(first) {
                let (src, dst) = (row(slot), row(k));
                want.copy_within(
                    src * value_bytes..(src + 1) * value_bytes,
                    dst * value_bytes,
                );
                if q8 {
                    let base = rows * value_bytes;
                    want.copy_within(
                        base + src * scale_bytes..base + (src + 1) * scale_bytes,
                        base + dst * scale_bytes,
                    );
                }
            }
            let got: Vec<u8> = cache
                .to_f32()
                .iter()
                .flat_map(|v| v.to_bits().to_le_bytes())
                .collect();
            assert!(
                got[..bytes] == want[..],
                "{kv} ring {ring}: moved K/V rows differ"
            );
        }
    }
}
