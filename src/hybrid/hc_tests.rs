//! Parity of qwen4exp's hyper-connection kernels (`h_hc_expand`,
//! `h_group_rmsnorm`, `h_scaled_silu`, `h_hc_mix`, `h_hc_write`) and of its
//! n-gram embedding kernels (`h_ple_gate`, `h_ple_conv`) against CPU
//! references written from the fork's qwen4exp graph. The conv is also run
//! over a prompt split in pieces, which must equal one pass: its history
//! carries the earlier rows. Speculation adds `h_hc_mean` (what drafters
//! read), the conv's verify mode, which leaves the history alone, and
//! `h_ple_push`, which commits accepted rows to it.
use super::engine::Params;
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

fn assert_close(got: &[f32], want: &[f32], what: &str) {
    assert_eq!(got.len(), want.len(), "{what}: length");
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        assert!(
            (g - w).abs() <= 1e-4 * w.abs().max(1.),
            "{what}[{i}]: {g} vs {w}"
        );
    }
}

fn tensor(d: &MetalDevice, v: &[f32]) -> Tensor {
    Tensor::from_f32(d, [v.len()], DType::F32, v).unwrap()
}

fn sigmoid(x: f32) -> f32 {
    1. / (1. + (-x).exp())
}

fn silu(x: f32) -> f32 {
    x * sigmoid(x)
}

/// `x * w / rms(x)` with `eps`.
fn rmsnorm(x: &[f32], w: &[f32], eps: f32) -> Vec<f32> {
    let scale = 1. / (x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32 + eps).sqrt();
    x.iter().zip(w).map(|(v, g)| v * scale * g).collect()
}

const N: usize = 96;
const STREAMS: usize = 4;
const EPS: f32 = 1e-6;

#[test]
fn hyper_connection_kernels_match_the_cpu() {
    let d = MetalDevice::new().unwrap();
    for rows in [1, 3] {
        let wide = STREAMS * N;
        let params = Params::default()
            .u(N)
            .unwrap()
            .u(STREAMS)
            .unwrap()
            .u(rows)
            .unwrap()
            .0;
        let grid = [(rows * N).div_ceil(256), 1, 1];

        // The residual starts as copies of the embedding.
        let x0 = random(rows * N, 1);
        let res = Tensor::zeros(&d, [rows * wide], DType::F32).unwrap();
        d.dispatch_hybrid(
            "h_hc_expand",
            &[tensor(&d, &x0).binding()],
            &[res.binding()],
            &params,
            grid,
            [256, 1, 1],
            0,
        )
        .unwrap();
        let want: Vec<f32> = (0..rows * wide).map(|i| x0[i / wide * N + i % N]).collect();
        assert_close(&res.to_f32(), &want, "expand");

        // Each stream normalized with its own gamma row.
        let r = random(rows * wide, 2);
        let gamma = random(wide, 3);
        let xn = Tensor::zeros(&d, [rows * wide], DType::F32).unwrap();
        d.dispatch_hybrid(
            "h_group_rmsnorm",
            &[tensor(&d, &r).binding(), tensor(&d, &gamma).binding()],
            &[xn.binding()],
            &Params::default()
                .u(N)
                .unwrap()
                .f(EPS)
                .u(rows * STREAMS)
                .unwrap()
                .u(STREAMS)
                .unwrap()
                .0,
            [rows * STREAMS, 1, 1],
            [256, 1, 1],
            0,
        )
        .unwrap();
        let normed: Vec<f32> = (0..rows * STREAMS)
            .flat_map(|g| {
                let s = g % STREAMS;
                rmsnorm(&r[g * N..(g + 1) * N], &gamma[s * N..(s + 1) * N], EPS)
            })
            .collect();
        assert_close(&xn.to_f32(), &normed, "group rmsnorm");

        // silu(x / streams), in place.
        let lo = random(rows * 40, 4);
        let lo_t = tensor(&d, &lo);
        d.dispatch_hybrid(
            "h_scaled_silu",
            &[],
            &[lo_t.binding()],
            &Params::default()
                .u(lo.len())
                .unwrap()
                .f(1. / STREAMS as f32)
                .0,
            [lo.len().div_ceil(256), 1, 1],
            [256, 1, 1],
            0,
        )
        .unwrap();
        let want: Vec<f32> = lo.iter().map(|v| silu(v / STREAMS as f32)).collect();
        assert_close(&lo_t.to_f32(), &want, "scaled silu");

        // The mean of the gated streams.
        let gate = random(rows * wide, 5);
        let y = Tensor::zeros(&d, [rows * N], DType::F32).unwrap();
        d.dispatch_hybrid(
            "h_hc_mix",
            &[xn.binding(), tensor(&d, &gate).binding()],
            &[y.binding()],
            &params,
            grid,
            [256, 1, 1],
            0,
        )
        .unwrap();
        let want: Vec<f32> = (0..rows * N)
            .map(|i| {
                let (t, c) = (i / N, i % N);
                (0..STREAMS)
                    .map(|s| {
                        let at = (t * STREAMS + s) * N + c;
                        normed[at] * sigmoid(gate[at])
                    })
                    .sum::<f32>()
                    / STREAMS as f32
            })
            .collect();
        assert_close(&y.to_f32(), &want, "mix");

        // res_s += out * 2 sigmoid(inject_s / streams).
        let out = random(rows * N, 6);
        let inject = random(rows * STREAMS, 7);
        let res_t = tensor(&d, &r);
        d.dispatch_hybrid(
            "h_hc_write",
            &[tensor(&d, &out).binding(), tensor(&d, &inject).binding()],
            &[res_t.binding()],
            &params,
            grid,
            [256, 1, 1],
            0,
        )
        .unwrap();
        let want: Vec<f32> = (0..rows * wide)
            .map(|i| {
                let (t, s, c) = (i / wide, i / N % STREAMS, i % N);
                let w = 2. * sigmoid(inject[t * STREAMS + s] / STREAMS as f32);
                r[i] + out[t * N + c] * w
            })
            .collect();
        assert_close(&res_t.to_f32(), &want, "write");

        // The mean of the streams.
        let mean = Tensor::zeros(&d, [rows * N], DType::F32).unwrap();
        d.dispatch_hybrid(
            "h_hc_mean",
            &[tensor(&d, &r).binding()],
            &[mean.binding()],
            &params,
            grid,
            [256, 1, 1],
            0,
        )
        .unwrap();
        let want: Vec<f32> = (0..rows * N)
            .map(|i| {
                let (t, c) = (i / N, i % N);
                (0..STREAMS)
                    .map(|s| r[(t * STREAMS + s) * N + c])
                    .sum::<f32>()
                    / STREAMS as f32
            })
            .collect();
        assert_close(&mean.to_f32(), &want, "mean");
    }
}

/// The gate of one stream: sigmoid of the signed square root of the scaled
/// dot of the normalized key and query.
fn ple_gate(key: &[f32], res: &[f32], nk: &[f32], nq: &[f32]) -> f32 {
    let k = rmsnorm(key, nk, EPS);
    let q = rmsnorm(res, nq, EPS);
    let score = k.iter().zip(&q).map(|(a, b)| a * b).sum::<f32>() / (key.len() as f32).sqrt();
    sigmoid(score.signum() * score.abs().max(1e-6).sqrt())
}

#[test]
fn ngram_embedding_kernels_match_the_cpu() {
    let d = MetalDevice::new().unwrap();
    let (kernel, dilation) = (4, 3);
    let history = (kernel - 1) * dilation;
    let wide = STREAMS * N;
    let rows = 13;
    let key = random(rows * wide, 11);
    let res = random(rows * wide, 12);
    let value = random(rows * N, 13);
    let nk = random(wide, 14);
    let nq = random(wide, 15);
    let nc = random(wide, 16);
    let w = random(wide * kernel, 17);

    // CPU: gated value and conv input per (token, stream).
    let mut gated = vec![0f32; rows * wide];
    let mut normed = vec![0f32; rows * wide];
    for t in 0..rows {
        let v = &value[t * N..(t + 1) * N];
        for s in 0..STREAMS {
            let at = (t * STREAMS + s) * N;
            let g = ple_gate(
                &key[at..at + N],
                &res[at..at + N],
                &nk[s * N..(s + 1) * N],
                &nq[s * N..(s + 1) * N],
            );
            let gv: Vec<f32> = v.iter().map(|x| x * g).collect();
            normed[at..at + N].copy_from_slice(&rmsnorm(&gv, &nc[s * N..(s + 1) * N], EPS));
            gated[at..at + N].copy_from_slice(&gv);
        }
    }
    // Causal depthwise conv with zero history: taps at t - 9, t - 6, t - 3, t.
    let want: Vec<f32> = (0..rows * wide)
        .map(|i| {
            let (t, c) = (i / wide, i % wide);
            let conv: f32 = (0..kernel)
                .map(|k| {
                    let back = (kernel - 1 - k) * dilation;
                    match t.checked_sub(back) {
                        Some(j) => w[c * kernel + k] * normed[j * wide + c],
                        None => 0.,
                    }
                })
                .sum();
            res[i] + gated[i] + silu(conv)
        })
        .collect();

    // GPU, over the whole prompt and then over pieces sharing one history.
    for pieces in [vec![rows], vec![1, 1, 5, 6], vec![4, 9]] {
        let hist = Tensor::zeros(&d, [history * wide], DType::F32).unwrap();
        let res_t = tensor(&d, &res);
        let mut first = 0;
        for m in pieces.clone() {
            let span = first * wide..(first + m) * wide;
            let gated_t = Tensor::zeros(&d, [m * wide], DType::F32).unwrap();
            let normed_t = Tensor::zeros(&d, [m * wide], DType::F32).unwrap();
            let res_rows = res_t.view(first * wide, [m * wide]).unwrap();
            d.dispatch_hybrid(
                "h_ple_gate",
                &[
                    tensor(&d, &key[span.clone()]).binding(),
                    res_rows.binding(),
                    tensor(&d, &value[first * N..(first + m) * N]).binding(),
                    tensor(&d, &nk).binding(),
                    tensor(&d, &nq).binding(),
                    tensor(&d, &nc).binding(),
                ],
                &[gated_t.binding(), normed_t.binding()],
                &Params::default()
                    .u(N)
                    .unwrap()
                    .u(STREAMS)
                    .unwrap()
                    .u(m)
                    .unwrap()
                    .f(EPS)
                    .0,
                [m * STREAMS, 1, 1],
                [256, 1, 1],
                0,
            )
            .unwrap();
            assert_close(&gated_t.to_f32(), &gated[span.clone()], "gated");
            assert_close(&normed_t.to_f32(), &normed[span.clone()], "normed");
            d.dispatch_hybrid(
                "h_ple_conv",
                &[
                    normed_t.binding(),
                    gated_t.binding(),
                    tensor(&d, &w).binding(),
                ],
                &[hist.binding(), res_rows.binding()],
                &Params::default()
                    .u(wide)
                    .unwrap()
                    .u(m)
                    .unwrap()
                    .u(kernel)
                    .unwrap()
                    .u(dilation)
                    .unwrap()
                    // Not a verify: the history takes these rows.
                    .u(0)
                    .unwrap()
                    .0,
                [wide.div_ceil(256), 1, 1],
                [256, 1, 1],
                0,
            )
            .unwrap();
            first += m;
        }
        assert_close(&res_t.to_f32(), &want, &format!("conv over {pieces:?}"));
    }
}

/// A verify runs the conv with `keep` set: the same output, the history
/// untouched. `h_ple_push` then appends the accepted rows, keeping the newest
/// `history`, as a plain pass over those rows would have.
#[test]
fn verify_conv_keeps_its_history_until_commit() {
    let d = MetalDevice::new().unwrap();
    let (kernel, dilation) = (4, 3);
    let history = (kernel - 1) * dilation;
    let wide = STREAMS * N;
    let rows = 5;
    let hist0 = random(history * wide, 21);
    let normed = random(rows * wide, 22);
    let gated = random(rows * wide, 23);
    let res = random(rows * wide, 24);
    let w = random(wide * kernel, 25);
    let conv = |keep: usize| {
        let hist = tensor(&d, &hist0);
        let res_t = tensor(&d, &res);
        d.dispatch_hybrid(
            "h_ple_conv",
            &[
                tensor(&d, &normed).binding(),
                tensor(&d, &gated).binding(),
                tensor(&d, &w).binding(),
            ],
            &[hist.binding(), res_t.binding()],
            &Params::default()
                .u(wide)
                .unwrap()
                .u(rows)
                .unwrap()
                .u(kernel)
                .unwrap()
                .u(dilation)
                .unwrap()
                .u(keep)
                .unwrap()
                .0,
            [wide.div_ceil(256), 1, 1],
            [256, 1, 1],
            0,
        )
        .unwrap();
        (hist.to_f32(), res_t.to_f32())
    };
    let (plain_hist, plain_res) = conv(0);
    let (kept_hist, kept_res) = conv(1);
    assert_close(&kept_res, &plain_res, "verify output");
    assert_close(&kept_hist, &hist0, "verify history");

    // Commit n of the rows: the history then ends with them.
    for n in [2, rows, history + 1] {
        let span: Vec<f32> = random(n * wide, 26);
        let hist = tensor(&d, &hist0);
        d.dispatch_hybrid(
            "h_ple_push",
            &[tensor(&d, &span).binding()],
            &[hist.binding()],
            &Params::default()
                .u(wide)
                .unwrap()
                .u(n)
                .unwrap()
                .u(history)
                .unwrap()
                .0,
            [wide.div_ceil(256), 1, 1],
            [256, 1, 1],
            0,
        )
        .unwrap();
        let all: Vec<f32> = hist0.iter().chain(&span).copied().collect();
        let want = &all[all.len() - history * wide..];
        assert_close(&hist.to_f32(), want, &format!("push {n}"));
    }
    // Committing every row equals the plain pass.
    let hist = tensor(&d, &hist0);
    d.dispatch_hybrid(
        "h_ple_push",
        &[tensor(&d, &normed).binding()],
        &[hist.binding()],
        &Params::default()
            .u(wide)
            .unwrap()
            .u(rows)
            .unwrap()
            .u(history)
            .unwrap()
            .0,
        [wide.div_ceil(256), 1, 1],
        [256, 1, 1],
        0,
    )
    .unwrap();
    assert_close(&hist.to_f32(), &plain_hist, "push all");
}
