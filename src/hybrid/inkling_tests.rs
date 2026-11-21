//! Inkling 2's kernels against CPU references: the depth mix, the loop
//! controllers, the short convolutions (static and input-dependent taps,
//! carried state), sigmoid routing, the SITU activation, the attention prep,
//! the logit softcap, and the `_rel` attention kernels (staged prompt,
//! decode, verify and the split reduce) with their relative-position bias
//! and per-row sinks over F16 and Q8 caches, full and sliding, linear and ring.
use super::config::KvCache;
use super::engine::Params;
use super::tree_tests::{f16, quantize_q8, random};
use crate::{DType, MetalDevice, Tensor};

fn tensor(d: &MetalDevice, x: &[f32]) -> Tensor {
    Tensor::from_f32(d, [x.len()], DType::F32, x).unwrap()
}

fn u32s(d: &MetalDevice, v: &[u32]) -> Tensor {
    let b: Vec<u8> = v.iter().flat_map(|v| v.to_le_bytes()).collect();
    Tensor::from_le_bytes(d, [v.len()], DType::F32, &b).unwrap()
}

fn as_u32(t: &Tensor) -> Vec<u32> {
    t.to_f32().iter().map(|v| v.to_bits()).collect()
}

fn params(us: &[usize], fs: &[f32]) -> Vec<u8> {
    let mut p = Params::default();
    for &u in us {
        p = p.u(u).unwrap();
    }
    for &f in fs {
        p = p.f(f);
    }
    p.0
}

fn assert_close(got: &[f32], want: &[f32], tol: f32, what: &str) {
    assert_eq!(got.len(), want.len(), "{what}");
    let scale = want.iter().fold(0f32, |m, v| m.max(v.abs()));
    assert!(scale > 1e-3, "{what}: reference is vacuous");
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        assert!(
            (g - w).abs() <= tol * scale,
            "{what}: element {i} is {g}, expected {w} (scale {scale})"
        );
    }
}

fn rms_inv(x: &[f32], eps: f32) -> f32 {
    let ss: f64 = x.iter().map(|&v| f64::from(v) * f64::from(v)).sum();
    (1. / (ss / x.len() as f64 + f64::from(eps)).sqrt()) as f32
}

fn sigmoid(x: f32) -> f32 {
    1. / (1. + (-x).exp())
}

#[test]
fn depth_mix_matches_the_cpu() {
    let d = MetalDevice::new().unwrap();
    let (h, m, eps) = (1000, 3, 1e-6);
    for (sources, queries) in [(1, 1), (2, 3), (6, 4), (5, 2)] {
        let xs: Vec<Vec<f32>> = (0..sources)
            .map(|s| {
                random(m * h, 10 + s as u64)
                    .iter()
                    .map(|v| v * (1. + s as f32))
                    .collect()
            })
            .collect();
        let query = random(queries * h, 3);
        let ts: Vec<Tensor> = xs.iter().map(|x| tensor(&d, x)).collect();
        let qt = tensor(&d, &query);
        let out = Tensor::zeros(&d, [queries * m * h], DType::F32).unwrap();
        let inputs: Vec<_> = (0..6)
            .map(|i| ts.get(i).unwrap_or(&ts[0]).binding())
            .chain([qt.binding()])
            .collect();
        d.dispatch_hybrid(
            "h_ink_mix",
            &inputs,
            &[out.binding()],
            &params(&[h, m, sources, queries], &[eps]),
            [m, 1, 1],
            [256, 1, 1],
            0,
        )
        .unwrap();
        let mut want = vec![0f32; queries * m * h];
        for t in 0..m {
            let normed: Vec<Vec<f32>> = xs
                .iter()
                .map(|x| {
                    let row = &x[t * h..(t + 1) * h];
                    let inv = rms_inv(row, eps);
                    row.iter().map(|v| v * inv).collect()
                })
                .collect();
            for j in 0..queries {
                let q = &query[j * h..(j + 1) * h];
                let scores: Vec<f64> = normed
                    .iter()
                    .map(|n| {
                        n.iter().zip(q).map(|(a, b)| f64::from(a * b)).sum::<f64>()
                            / (h as f64).sqrt()
                    })
                    .collect();
                let max = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                let total: f64 = scores.iter().map(|s| (s - max).exp()).sum();
                for i in 0..h {
                    want[(j * m + t) * h + i] = normed
                        .iter()
                        .zip(&scores)
                        .map(|(n, s)| (s - max).exp() / total * f64::from(n[i]))
                        .sum::<f64>() as f32;
                }
            }
        }
        assert_close(
            &out.to_f32(),
            &want,
            1e-5,
            &format!("mix of {sources}x{queries}"),
        );
    }
}

#[test]
fn loop_controllers_match_the_cpu() {
    let d = MetalDevice::new().unwrap();
    let (h, m, rank, eps) = (520, 4, 32, 1e-6);
    let x: Vec<f32> = random(m * h, 1).iter().map(|v| v * 3.).collect();
    let msg = random(m * h, 2);
    let z = random(m * rank, 3);
    let gain: Vec<f32> = random(h * rank, 4).iter().map(|v| v * 0.3).collect();
    let forget: Vec<f32> = random(h * rank, 5).iter().map(|v| v * 0.3).collect();
    let (xt, mt, zt) = (tensor(&d, &x), tensor(&d, &msg), tensor(&d, &z));
    let args = params(&[h, m, rank], &[eps]);
    let z_in = Tensor::zeros(&d, [m * 2 * h], DType::F32).unwrap();
    d.dispatch_hybrid(
        "h_ink_ctrl_in",
        &[xt.binding(), mt.binding()],
        &[z_in.binding()],
        &args,
        [m, 1, 1],
        [256, 1, 1],
        0,
    )
    .unwrap();
    let y = Tensor::zeros(&d, [m * h], DType::F32).unwrap();
    let (gt, ft) = (tensor(&d, &gain), tensor(&d, &forget));
    d.dispatch_hybrid(
        "h_ink_ctrl_out",
        &[
            xt.binding(),
            mt.binding(),
            zt.binding(),
            gt.binding(),
            ft.binding(),
        ],
        &[y.binding()],
        &args,
        [(m * h).div_ceil(256), 1, 1],
        [256, 1, 1],
        0,
    )
    .unwrap();
    let mut want_in = Vec::new();
    for t in 0..m {
        for src in [&x, &msg] {
            let row = &src[t * h..(t + 1) * h];
            let inv = rms_inv(row, eps);
            want_in.extend(row.iter().map(|v| v * inv));
        }
    }
    assert_close(&z_in.to_f32(), &want_in, 1e-5, "controller input");
    let want: Vec<f32> = (0..m * h)
        .map(|i| {
            let (t, c) = (i / h, i % h);
            let dot = |w: &[f32]| {
                (0..rank)
                    .map(|k| w[c * rank + k] * z[t * rank + k])
                    .sum::<f32>()
            };
            x[i] - dot(&forget).tanh() * x[i] + dot(&gain).tanh() * msg[i]
        })
        .collect();
    assert_close(&y.to_f32(), &want, 1e-5, "controller output");
}

/// Two chunks through one state equal one chunk through all rows, with and
/// without input-dependent taps; the tape mode records inputs and leaves the
/// state alone.
#[test]
fn short_convolutions_match_the_cpu() {
    let d = MetalDevice::new().unwrap();
    let (channels, width, offset, state_width) = (300, 100, 50, 400);
    let tokens = 7;
    for dynamic in [false, true] {
        let heads = if dynamic { channels / width } else { 0 };
        let scale = 1. / ((4 * if dynamic { 2 } else { 1 } + 1) as f32).sqrt();
        let x = random(tokens * channels, 1);
        let w = random(channels * 4, 2);
        let dynv = random(tokens * heads.max(1) * 4, 3);
        let state0 = random(3 * state_width, 4);
        let st = tensor(&d, &state0);
        let wt = tensor(&d, &w);
        let tape = Tensor::zeros(&d, [tokens * state_width], DType::F32).unwrap();
        let mut got = Vec::new();
        for (from, n) in [(0, 3), (3, 4)] {
            let xt = tensor(&d, &x[from * channels..(from + n) * channels]);
            let dt = tensor(
                &d,
                &dynv[from * heads.max(1) * 4..(from + n) * heads.max(1) * 4],
            );
            let y = Tensor::zeros(&d, [n * channels], DType::F32).unwrap();
            d.dispatch_hybrid(
                "h_ink_conv",
                &[xt.binding(), wt.binding(), dt.binding()],
                &[y.binding(), st.binding(), tape.binding()],
                &params(
                    &[n, channels, channels, state_width, offset, heads, width, 0],
                    &[scale],
                ),
                [channels.div_ceil(256), 1, 1],
                [256, 1, 1],
                0,
            )
            .unwrap();
            got.extend(y.to_f32());
        }
        let mut hist: Vec<[f32; 3]> = (0..channels)
            .map(|c| [0, 1, 2].map(|r| state0[r * state_width + offset + c]))
            .collect();
        let mut want = vec![0f32; tokens * channels];
        for t in 0..tokens {
            for c in 0..channels {
                let xt = x[t * channels + c];
                let [h0, h1, h2] = hist[c];
                let mut acc =
                    xt + w[c * 4] * h0 + w[c * 4 + 1] * h1 + w[c * 4 + 2] * h2 + w[c * 4 + 3] * xt;
                if dynamic {
                    let dv = &dynv[(t * heads + c / width) * 4..][..4];
                    acc += dv[0] * xt + dv[1] * h2 + dv[2] * h1 + dv[3] * h0;
                }
                want[t * channels + c] = acc * scale;
                hist[c] = [h1, h2, xt];
            }
        }
        assert_close(&got, &want, 1e-5, &format!("conv dynamic={dynamic}"));
        let state = st.to_f32();
        for c in 0..channels {
            for r in 0..3 {
                assert_eq!(state[r * state_width + offset + c], hist[c][r], "state");
            }
        }
        // Channels outside this convolution are untouched.
        assert_eq!(state[0], state0[0]);
        assert_eq!(state[state_width - 1], state0[state_width - 1]);

        // Tape: the state stays, the inputs land at their rows.
        let xt = tensor(&d, &x);
        let dt = tensor(&d, &dynv);
        let y = Tensor::zeros(&d, [tokens * channels], DType::F32).unwrap();
        d.dispatch_hybrid(
            "h_ink_conv",
            &[xt.binding(), wt.binding(), dt.binding()],
            &[y.binding(), st.binding(), tape.binding()],
            &params(
                &[
                    tokens,
                    channels,
                    channels,
                    state_width,
                    offset,
                    heads,
                    width,
                    1,
                ],
                &[scale],
            ),
            [channels.div_ceil(256), 1, 1],
            [256, 1, 1],
            0,
        )
        .unwrap();
        assert_eq!(st.to_f32(), state, "tape leaves the state");
        let tape = tape.to_f32();
        assert_eq!(tape[2 * state_width + offset + 5], x[2 * channels + 5]);
    }
}

#[test]
fn sigmoid_routing_matches_the_cpu() {
    let d = MetalDevice::new().unwrap();
    let (experts, used, m, scale) = (64, 8, 5, 8.);
    let logits: Vec<f32> = random(m * experts, 1).iter().map(|v| v * 4.).collect();
    let mut bias: Vec<f32> = random(experts, 2).iter().map(|v| v * 0.1).collect();
    // A tie: the lower index wins.
    bias[9] = bias[3];
    let mut logits = logits;
    logits[9] = logits[3];
    let ids = Tensor::zeros(&d, [m * used], DType::F32).unwrap();
    let weights = Tensor::zeros(&d, [m * used], DType::F32).unwrap();
    d.dispatch_hybrid(
        "h_ink_route",
        &[tensor(&d, &logits).binding(), tensor(&d, &bias).binding()],
        &[ids.binding(), weights.binding()],
        &params(&[experts, used, m], &[scale]),
        [m, 1, 1],
        [experts, 1, 1],
        0,
    )
    .unwrap();
    let (got_ids, got_w) = (as_u32(&ids), weights.to_f32());
    let mut want_w = Vec::new();
    for t in 0..m {
        let prob: Vec<f32> = logits[t * experts..(t + 1) * experts]
            .iter()
            .map(|&l| sigmoid(l))
            .collect();
        let mut order: Vec<usize> = (0..experts).collect();
        order.sort_by(|&a, &b| {
            (prob[b] + bias[b])
                .total_cmp(&(prob[a] + bias[a]))
                .then(a.cmp(&b))
        });
        let picked = &order[..used];
        assert_eq!(
            got_ids[t * used..(t + 1) * used]
                .iter()
                .map(|&i| i as usize)
                .collect::<Vec<_>>(),
            picked,
            "row {t}"
        );
        let total: f32 = picked.iter().map(|&e| prob[e]).sum();
        want_w.extend(picked.iter().map(|&e| prob[e] / total * scale));
    }
    assert_close(&got_w, &want_w, 1e-5, "route weights");
}

#[test]
fn situ_activation_matches_the_cpu() {
    let d = MetalDevice::new().unwrap();
    let (width, routes, beta, lb) = (96, 5, 4., 25.);
    let gu: Vec<f32> = random(routes * 2 * width, 1)
        .iter()
        .map(|v| v * 30.)
        .collect();
    let ids: [u32; 5] = [2, 0, 1, 1, 3];
    let sorted: [u32; 5] = [1, 2, 3, 0, 4];
    let gs = [0.5f32, 1.5, 0.25, 2.];
    let ds = [3f32, 0.5, 2., 0.75];
    let w0 = random(routes, 2);
    for (sort, scaled) in [(false, false), (true, true), (false, true)] {
        let act = Tensor::zeros(&d, [routes * width], DType::F32).unwrap();
        let weights = tensor(&d, &w0);
        d.dispatch_hybrid(
            "h_ink_situ",
            &[
                tensor(&d, &gu).binding(),
                u32s(&d, &ids).binding(),
                u32s(&d, &sorted).binding(),
                tensor(&d, &gs).binding(),
                tensor(&d, &ds).binding(),
            ],
            &[act.binding(), weights.binding()],
            &params(&[width, routes, usize::from(sort)], &[beta, lb])
                .into_iter()
                .chain(params(&[usize::from(scaled)], &[]))
                .collect::<Vec<_>>(),
            [(routes * width).div_ceil(256), 1, 1],
            [256, 1, 1],
            0,
        )
        .unwrap();
        let mut want = Vec::new();
        let mut want_w = w0.clone();
        for r in 0..routes {
            let route = if sort { sorted[r] as usize } else { r };
            let s = if scaled { gs[ids[route] as usize] } else { 1. };
            for c in 0..width {
                let g = gu[r * 2 * width + c] * s;
                let u = gu[r * 2 * width + width + c] * s;
                want.push(beta * (g / beta).tanh() * sigmoid(g) * lb * (u / lb).tanh());
            }
            if scaled {
                want_w[route] *= ds[ids[route] as usize];
            }
        }
        let what = format!("situ sorted={sort} scaled={scaled}");
        assert_close(&act.to_f32(), &want, 1e-5, &what);
        assert_close(&weights.to_f32(), &want_w, 1e-6, &what);
    }
}

#[test]
fn softcap_and_accumulate_match_the_cpu() {
    let d = MetalDevice::new().unwrap();
    let x: Vec<f32> = random(1000, 1).iter().map(|v| v * 100.).collect();
    let y = tensor(&d, &x);
    d.dispatch_hybrid(
        "h_ink_softcap",
        &[],
        &[y.binding()],
        &params(&[1000], &[30.]),
        [4, 1, 1],
        [256, 1, 1],
        0,
    )
    .unwrap();
    let want: Vec<f32> = x.iter().map(|v| 30. * (v / 30.).tanh()).collect();
    assert_close(&y.to_f32(), &want, 1e-6, "softcap");
    let a = random(1000, 2);
    d.dispatch_hybrid(
        "h_ink_acc",
        &[tensor(&d, &a).binding()],
        &[y.binding()],
        &params(&[1000], &[0.]),
        [4, 1, 1],
        [256, 1, 1],
        0,
    )
    .unwrap();
    let want: Vec<f32> = want.iter().zip(&a).map(|(w, a)| w + a).collect();
    assert_close(&y.to_f32(), &want, 1e-6, "accumulate");
}

/// The F16 path of the prep (Q8 shares it up to `kv_store`, the store every
/// other prep kernel uses).
#[test]
fn attention_prep_matches_the_cpu() {
    let d = MetalDevice::new().unwrap();
    let (heads, kv, dim, r, m) = (4, 2, 128, 16, 3);
    let (qk_clip, rel_clip, eps, floor, alpha) = (22.627f32, 8f32, 1e-6f32, 8192f32, 0.1f32);
    let q_scale = 1. / (dim as f32).sqrt();
    for (pos, global) in [(5usize, false), (20000, true)] {
        let qg: Vec<f32> = random(m * heads * 2 * dim, 1)
            .iter()
            .map(|v| v * 4.)
            .collect();
        let k: Vec<f32> = random(m * kv * dim, 2).iter().map(|v| v * 4.).collect();
        let v = random(m * kv * dim, 3);
        let q_clip: Vec<f32> = random(heads * dim, 4)
            .iter()
            .map(|v| 1. + v * 0.5)
            .collect();
        let rel: Vec<f32> = random(m * heads * r, 5).iter().map(|v| v * 4.).collect();
        let rclip: Vec<f32> = random(heads * r, 6).iter().map(|v| 1. + v * 0.5).collect();
        let rows = (pos + m).next_multiple_of(32);
        let stride = kv * dim;
        let q_out = Tensor::zeros(&d, [m, heads * dim], DType::F16).unwrap();
        let kc = Tensor::zeros(&d, [rows, stride], DType::F16).unwrap();
        let vc = Tensor::zeros(&d, [rows, stride], DType::F16).unwrap();
        let rel_out = Tensor::zeros(&d, [m * heads * r], DType::F32).unwrap();
        d.dispatch_hybrid(
            "h_ink_prep",
            &[
                tensor(&d, &qg).binding(),
                tensor(&d, &k).binding(),
                tensor(&d, &v).binding(),
                tensor(&d, &q_clip).binding(),
                tensor(&d, &rel).binding(),
                tensor(&d, &rclip).binding(),
            ],
            &[
                q_out.binding(),
                kc.binding(),
                vc.binding(),
                rel_out.binding(),
            ],
            &params(
                &[m, heads, kv, pos, stride, rows, 0, usize::from(global)],
                &[qk_clip, rel_clip, eps, q_scale, floor, alpha],
            ),
            [2 * heads + 2 * kv, m, 1],
            [dim, 1, 1],
            0,
        )
        .unwrap();
        let clip = |x: &[f32], thr: f32| -> Vec<f32> {
            let norm = x.iter().map(|v| v * v).sum::<f32>().sqrt();
            let f = (thr / (norm + 1e-6)).clamp(0., 1.);
            x.iter().map(|v| v * f).collect()
        };
        let (mut wq, mut wk, mut wv, mut wr) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
        for t in 0..m {
            let p = (pos + t) as f32;
            let tau = if global {
                1. + alpha * ((p + 1.) / floor).max(1.).ln()
            } else {
                1.
            };
            for h in 0..heads {
                let x: Vec<f32> = (0..dim)
                    .map(|i| qg[(t * heads + h) * 2 * dim + i] * q_clip[h * dim + i])
                    .collect();
                wq.extend(clip(&x, qk_clip).iter().map(|v| f16(v * tau * q_scale)));
                let x: Vec<f32> = (0..r)
                    .map(|i| rel[(t * heads + h) * r + i] * rclip[h * r + i])
                    .collect();
                wr.extend(clip(&x, rel_clip).iter().map(|v| v * tau));
            }
            for h in 0..kv {
                wk.extend(
                    clip(&k[(t * kv + h) * dim..][..dim], qk_clip)
                        .iter()
                        .map(|&v| f16(v)),
                );
                let row = &v[(t * kv + h) * dim..][..dim];
                let inv = rms_inv(row, eps);
                wv.extend(row.iter().map(|x| f16(x * inv)));
            }
        }
        let what = format!("prep pos {pos}");
        assert_close(&q_out.to_f32(), &wq, 2e-3, &what);
        assert_close(&rel_out.to_f32(), &wr, 1e-5, &what);
        let at = |c: &Tensor| c.to_f32()[pos * stride..(pos + m) * stride].to_vec();
        assert_close(&at(&kc), &wk, 2e-3, &what);
        assert_close(&at(&vc), &wv, 2e-3, &what);
    }
}

// ---------------------------------------------------------------- relative-bias attention

#[derive(Clone, Copy, Debug)]
struct RelCase {
    m: usize,
    pos: usize,
    window: usize,
    extent: usize,
    kv: KvCache,
    ring: bool,
}

const HEADS: usize = 8;
const KV_HEADS: usize = 2;
const DIM: usize = 128;
const R: usize = 16;

/// Attention in F64: scores q.k plus the bias of each key `dist` back (0 from
/// `extent`), keys within the window, a per-row sink, and the output gated by
/// sigmoid(gate).
#[allow(clippy::too_many_arguments)]
fn cpu_rel_attention(
    c: RelCase,
    q: &[f32],
    k: &[f32],
    v: &[f32],
    qg: &[f32],
    sinks: &[f32],
    rel: &[f32],
    table: &[f32],
) -> Vec<f32> {
    let stride = KV_HEADS * DIM;
    let group = HEADS / KV_HEADS;
    let mut out = vec![0f32; c.m * HEADS * DIM];
    for t in 0..c.m {
        let qpos = c.pos + t;
        for h in 0..HEADS {
            let kvh = h / group;
            let qr = &q[(t * HEADS + h) * DIM..][..DIM];
            let rr = &rel[(t * HEADS + h) * R..][..R];
            let mut scores = Vec::new();
            for key in 0..=qpos {
                if c.window > 0 && qpos - key >= c.window {
                    continue;
                }
                let kr = &k[key * stride + kvh * DIM..][..DIM];
                let mut s: f64 = qr.iter().zip(kr).map(|(a, b)| f64::from(a * b)).sum();
                let dist = qpos - key;
                if dist < c.extent {
                    s += rr
                        .iter()
                        .zip(&table[dist * R..][..R])
                        .map(|(a, b)| f64::from(a * b))
                        .sum::<f64>();
                }
                scores.push((key, s));
            }
            let sink = f64::from(sinks[t * HEADS + h]);
            let max = scores.iter().map(|s| s.1).fold(sink, f64::max);
            let mut total = (sink - max).exp();
            let mut acc = vec![0f64; DIM];
            for &(key, s) in &scores {
                let w = (s - max).exp();
                total += w;
                for (a, &b) in acc.iter_mut().zip(&v[key * stride + kvh * DIM..][..DIM]) {
                    *a += w * f64::from(b);
                }
            }
            for (i, a) in acc.iter().enumerate() {
                let gate = f64::from(qg[(t * HEADS + h) * 2 * DIM + DIM + i]);
                out[(t * HEADS + h) * DIM + i] = (a / total / (1. + (-gate).exp())) as f32;
            }
        }
    }
    out
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum RelKernel {
    Staged,
    Decode(usize),
    Verify(usize),
}

/// Runs `kernel` on the case, routed as `flash_attention_in` routes it.
fn run_rel(d: &MetalDevice, c: RelCase, kernel: RelKernel, seed: u64) -> (Vec<f32>, Vec<f32>) {
    let stride = KV_HEADS * DIM;
    let group = HEADS / KV_HEADS;
    let keys = c.pos + c.m;
    let ring_rows = if c.ring {
        (c.window + 62 + c.m).next_multiple_of(64)
    } else {
        0
    };
    let rows = if c.ring {
        ring_rows
    } else {
        keys.next_multiple_of(32)
    };
    let q_raw: Vec<f32> = random(c.m * HEADS * DIM, seed)
        .iter()
        .map(|v| v / 16.)
        .collect();
    let q_cpu: Vec<f32> = q_raw.iter().map(|&v| f16(v)).collect();
    let qg = random(c.m * 2 * HEADS * DIM, seed + 1);
    let sinks = random(c.m * HEADS, seed + 2);
    let rel = random(c.m * HEADS * R, seed + 3);
    let table: Vec<f32> = random(c.extent * R, seed + 4)
        .iter()
        .map(|v| v * 0.5)
        .collect();
    let (k_raw, v_raw) = (
        random(keys * stride, seed + 5),
        random(keys * stride, seed + 6),
    );
    let lay = |raw: &[f32]| {
        let mut out = random(rows * stride, 9);
        for j in 0..keys {
            let r = if c.ring { j % ring_rows } else { j };
            out[r * stride..][..stride].copy_from_slice(&raw[j * stride..][..stride]);
        }
        out
    };
    let encode = |x: &[f32]| -> (Tensor, Vec<f32>) {
        match c.kv {
            KvCache::F16 => (
                Tensor::from_f32(d, [rows, stride], DType::F16, x).unwrap(),
                x.iter().map(|&v| f16(v)).collect(),
            ),
            KvCache::Q8 => {
                let (b, seen) = quantize_q8(x);
                let t =
                    Tensor::from_le_bytes(d, [rows, c.kv.row_bytes(stride) / 2], DType::F16, &b)
                        .unwrap();
                (t, seen)
            }
        }
    };
    let ((kc, ks), (vc, vs)) = (encode(&lay(&k_raw)), encode(&lay(&v_raw)));
    let by_key = |seen: &[f32]| -> Vec<f32> {
        (0..keys)
            .flat_map(|j| {
                let r = if c.ring { j % ring_rows } else { j };
                seen[r * stride..][..stride].to_vec()
            })
            .collect()
    };
    let want = cpu_rel_attention(
        c,
        &q_cpu,
        &by_key(&ks),
        &by_key(&vs),
        &qg,
        &sinks,
        &rel,
        &table,
    );

    let q = Tensor::from_f32(d, [c.m, HEADS * DIM], DType::F16, &q_raw).unwrap();
    let (qgt, st, relt, tt) = (
        tensor(d, &qg),
        tensor(d, &sinks),
        tensor(d, &rel),
        tensor(d, &table),
    );
    let first_key = if c.window > 0 {
        (c.pos + 1).saturating_sub(c.window)
    } else {
        0
    };
    let first_key = if c.ring {
        first_key / 64 * 64
    } else {
        first_key
    };
    let span = keys - first_key;
    let (kps, splits) = match kernel {
        RelKernel::Staged => (0, 1),
        RelKernel::Decode(wanted) => {
            let kps = span.div_ceil(wanted).next_multiple_of(32);
            (kps, span.div_ceil(kps))
        }
        RelKernel::Verify(wanted) => {
            let kps = span.div_ceil(wanted).next_multiple_of(64);
            (kps, span.div_ceil(kps))
        }
    };
    let ring = if c.ring { ring_rows } else { 0 };
    let args = params(
        &[
            c.m, HEADS, KV_HEADS, group, c.pos, stride, kps, splits, rows, DIM, first_key,
            c.window, 1, 1, ring, c.extent,
        ],
        &[],
    );
    let q8 = c.kv == KvCache::Q8;
    let out = Tensor::zeros(d, [c.m, HEADS * DIM], DType::F32).unwrap();
    let partial = Tensor::zeros(d, [splits * c.m * HEADS * DIM], DType::F32).unwrap();
    let ml = Tensor::zeros(d, [splits * c.m * HEADS * 2], DType::F32).unwrap();
    let e = d.execution_with_shared_encoder(true).unwrap();
    let tail = [qgt.binding(), st.binding(), relt.binding(), tt.binding()];
    match kernel {
        RelKernel::Staged => {
            d.dispatch_hybrid(
                if q8 {
                    "h_flash_attn_mpp_q8_d128_rel"
                } else {
                    "h_flash_attn_mpp_staged_d128_rel"
                },
                &[[q.binding(), kc.binding(), vc.binding()].as_slice(), &tail].concat(),
                &[out.binding()],
                &args,
                [c.m.div_ceil(64), HEADS, 1],
                [128, 1, 1],
                64 * 64 * 4 + 64 * 64 * 2 + 3 * 64 * 4,
            )
            .unwrap();
        }
        RelKernel::Decode(_) => {
            d.dispatch_hybrid(
                if q8 {
                    "h_attn_decode_q8_d128_rel"
                } else {
                    "h_attn_decode_d128_rel"
                },
                &[[q.binding(), kc.binding(), vc.binding()].as_slice(), &tail].concat(),
                &[out.binding(), partial.binding(), ml.binding()],
                &args,
                [KV_HEADS, splits, 1],
                [128, 1, 1],
                8 * DIM * 2 + 4 * 8 * 2 * 4 + 2 * 8 * DIM * 4,
            )
            .unwrap();
        }
        RelKernel::Verify(_) => {
            let qk = Tensor::zeros(d, [c.m, HEADS * DIM], DType::F16).unwrap();
            d.dispatch_hybrid(
                "h_q_by_kv_head",
                &[q.binding()],
                &[qk.binding()],
                &args,
                [(c.m * HEADS * DIM).div_ceil(256), 1, 1],
                [256, 1, 1],
                0,
            )
            .unwrap();
            let br = 32;
            d.dispatch_hybrid(
                if q8 {
                    "h_attn_verify_mpp_32_q8_d128_rel"
                } else {
                    "h_attn_verify_mpp_32_d128_rel"
                },
                &[[qk.binding(), kc.binding(), vc.binding()].as_slice(), &tail].concat(),
                &[out.binding(), partial.binding(), ml.binding()],
                &args,
                [(c.m * group).div_ceil(br), KV_HEADS, splits],
                [128, 1, 1],
                if q8 {
                    (64 * 128 * 2).max(br * 64 * 4) + br * 64 * 2 + 3 * br * 4
                } else {
                    br * 64 * 4 + br * 64 * 2 + 3 * br * 4
                },
            )
            .unwrap();
        }
    }
    if splits > 1 {
        d.dispatch_hybrid(
            "h_flash_reduce_rel_d128",
            &[partial.binding(), ml.binding(), qgt.binding(), st.binding()],
            &[out.binding()],
            &args,
            [c.m * HEADS, 1, 1],
            [DIM, 1, 1],
            0,
        )
        .unwrap();
    }
    e.finish().unwrap();
    (out.to_f32(), want)
}

#[test]
fn relative_bias_attention_matches_the_cpu() {
    let d = MetalDevice::new().unwrap();
    let mut seed = 100;
    // Full attention with an extent shorter than the keys (Inkling's global
    // layers), and a sliding window equal to its extent (its SWA layers),
    // linear and as a ring that wraps.
    let shapes = [(0, 50, false), (40, 40, false), (40, 40, true)];
    for kv in [KvCache::F16, KvCache::Q8] {
        for &(window, extent, ring) in &shapes {
            for pos in [0usize, 30, 300] {
                let kernels = [
                    (1, RelKernel::Decode(1)),
                    (1, RelKernel::Decode(5)),
                    (8, RelKernel::Verify(1)),
                    (8, RelKernel::Verify(3)),
                    (5, RelKernel::Verify(2)),
                    (16, RelKernel::Staged),
                    (77, RelKernel::Staged),
                ];
                for (m, kernel) in kernels {
                    let c = RelCase {
                        m,
                        pos,
                        window,
                        extent,
                        kv,
                        ring,
                    };
                    if ring && (c.window + 62 + c.m).next_multiple_of(64) >= pos + m {
                        continue; // no wrap
                    }
                    seed += 10;
                    let (got, want) = run_rel(&d, c, kernel, seed);
                    let err = got
                        .iter()
                        .zip(&want)
                        .map(|(a, b)| (a - b).abs())
                        .fold(0., f32::max);
                    // F16 queries, keys, values and probabilities.
                    assert!(err < 4e-3, "{kernel:?} {c:?}: max error {err}");
                }
            }
        }
    }
}
