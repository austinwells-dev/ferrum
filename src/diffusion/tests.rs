//! The image transformer's kernels against straightforward CPU versions.
use super::ops::{self, HEAD_DIM, KeyValues, QkvIn};
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

fn tensor(d: &MetalDevice, dims: &[usize], dtype: DType, data: &[f32]) -> Tensor {
    Tensor::from_f32(d, dims, dtype, data).unwrap()
}

fn f16(v: f32) -> f32 {
    half::f16::from_f32(v).to_f32()
}

fn close(got: &[f32], want: &[f32], tol: f32, what: &str) {
    assert_eq!(got.len(), want.len(), "{what}: length");
    let worst = got
        .iter()
        .zip(want)
        .map(|(g, w)| (g - w).abs() / (1. + w.abs()))
        .fold(0f32, f32::max);
    assert!(worst <= tol, "{what}: worst error {worst} > {tol}");
}

#[test]
fn elementwise_kernels_match_the_cpu() {
    let d = MetalDevice::new().unwrap();
    let n = 5000;
    let x = random(n, 1);
    let v = random(n, 2);
    let xt = tensor(&d, &[n], DType::F32, &x);
    let out = Tensor::zeros(&d, [n], DType::F32).unwrap();

    ops::silu(&d, &xt, &out).unwrap();
    let want: Vec<f32> = x.iter().map(|&a| a / (1. + (-a).exp())).collect();
    close(&out.to_f32(), &want, 1e-6, "silu");

    ops::gelu_tanh(&d, &xt, &out).unwrap();
    let want: Vec<f32> = x
        .iter()
        .map(|&a| 0.5 * a * (1. + (0.797_884_6 * (a + 0.044715 * a * a * a)).tanh()))
        .collect();
    close(&out.to_f32(), &want, 1e-5, "gelu");

    let rows = 5;
    let cols = n / rows;
    let gate = random(cols, 3);
    let gt = tensor(&d, &[cols], DType::F32, &gate);
    let vt = tensor(&d, &[n], DType::F32, &v);
    let acc = tensor(&d, &[n], DType::F32, &x);
    ops::gate_add(&d, &vt, &gt, &acc).unwrap();
    let want: Vec<f32> = (0..n)
        .map(|i| x[i] + gate[i % cols].tanh() * v[i])
        .collect();
    close(&acc.to_f32(), &want, 1e-6, "gate_add");

    let acc = tensor(&d, &[n], DType::F32, &x);
    ops::euler(&d, &vt, &acc, -0.125).unwrap();
    let want: Vec<f32> = (0..n).map(|i| x[i] - 0.125 * v[i]).collect();
    close(&acc.to_f32(), &want, 1e-6, "euler");
}

#[test]
fn layernorm_scale_matches_the_cpu() {
    let d = MetalDevice::new().unwrap();
    let (rows, n) = (7, 4096);
    let x: Vec<f32> = random(rows * n, 4).iter().map(|v| v * 3. + 0.5).collect();
    let scale = random(n, 5);
    let y = Tensor::zeros(&d, [rows, n], DType::F32).unwrap();
    let xt = tensor(&d, &[rows, n], DType::F32, &x);
    let st = tensor(&d, &[n], DType::F32, &scale);
    ops::layernorm_scale(&d, &xt, &st, &y, rows, 1e-6, 1.).unwrap();
    let mut want = Vec::new();
    for row in x.chunks(n) {
        let mean = row.iter().map(|&v| f64::from(v)).sum::<f64>() / n as f64;
        let var = row
            .iter()
            .map(|&v| (f64::from(v) - mean).powi(2))
            .sum::<f64>()
            / n as f64;
        let inv = 1. / (var + 1e-6).sqrt();
        want.extend(
            row.iter()
                .zip(&scale)
                .map(|(&v, &s)| ((f64::from(v) - mean) * inv * (1. + f64::from(s))) as f32),
        );
    }
    close(&y.to_f32(), &want, 2e-5, "layernorm");
}

#[test]
fn qkv_prep_matches_the_cpu() {
    let d = MetalDevice::new().unwrap();
    let (rows, heads) = (6, 3);
    let width = heads * HEAD_DIM;
    let q = random(rows * width, 6);
    let k = random(rows * width, 7);
    let v = random(rows * width, 8);
    let qw: Vec<f32> = random(HEAD_DIM, 9).iter().map(|x| 1. + x * 0.2).collect();
    let kw: Vec<f32> = random(HEAD_DIM, 10).iter().map(|x| 1. + x * 0.2).collect();
    let angles = random(rows * 64, 11);
    let cos: Vec<f32> = angles.iter().map(|a| (a * 3.).cos()).collect();
    let sin: Vec<f32> = angles.iter().map(|a| (a * 3.).sin()).collect();

    let t = |dims: &[usize], x: &[f32]| tensor(&d, dims, DType::F32, x);
    let (qt, kt, vt) = (
        t(&[rows, width], &q),
        t(&[rows, width], &k),
        t(&[rows, width], &v),
    );
    let (qwt, kwt) = (t(&[HEAD_DIM], &qw), t(&[HEAD_DIM], &kw));
    let (ct, st) = (t(&[rows, 64], &cos), t(&[rows, 64], &sin));
    let out: Vec<Tensor> = (0..3)
        .map(|_| Tensor::zeros(&d, [rows, width], DType::F16).unwrap())
        .collect();
    let input = QkvIn {
        q: &qt,
        k: &kt,
        v: &vt,
        q_norm: &qwt,
        k_norm: &kwt,
        cos: &ct,
        sin: &st,
    };
    ops::qkv_prep(&d, &input, [&out[0], &out[1], &out[2]], rows, heads, 1e-6).unwrap();

    let rotate = |x: &[f32], w: &[f32], scale: f32| -> Vec<f32> {
        let mut y = vec![0f32; x.len()];
        for r in 0..rows {
            for h in 0..heads {
                let base = r * width + h * HEAD_DIM;
                let head = &x[base..base + HEAD_DIM];
                let ms = head.iter().map(|&v| f64::from(v).powi(2)).sum::<f64>() / HEAD_DIM as f64;
                let inv = 1. / (ms + 1e-6).sqrt();
                let n: Vec<f64> = head
                    .iter()
                    .zip(w)
                    .map(|(&v, &w)| f64::from(v) * inv * f64::from(w))
                    .collect();
                for p in 0..64 {
                    let (c, s) = (f64::from(cos[r * 64 + p]), f64::from(sin[r * 64 + p]));
                    y[base + 2 * p] = ((n[2 * p] * c - n[2 * p + 1] * s) as f32) * scale;
                    y[base + 2 * p + 1] = ((n[2 * p] * s + n[2 * p + 1] * c) as f32) * scale;
                }
            }
        }
        y
    };
    let scale = 1. / (HEAD_DIM as f32).sqrt();
    close(&out[0].to_f32(), &rotate(&q, &qw, scale), 2e-3, "q");
    close(&out[1].to_f32(), &rotate(&k, &kw, 1.), 2e-3, "k");
    let want_v: Vec<f32> = v.iter().map(|&x| f16(x)).collect();
    close(&out[2].to_f32(), &want_v, 1e-6, "v");
}

/// Softmax attention over F16-rounded inputs in f64.
fn attention_reference(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    tokens: usize,
    keys: usize,
    heads: usize,
    causal: bool,
) -> Vec<f32> {
    let width = heads * HEAD_DIM;
    let mut out = vec![0f32; tokens * width];
    for t in 0..tokens {
        for h in 0..heads {
            let visible = if causal { t + 1 } else { keys };
            let qv = &q[t * width + h * HEAD_DIM..][..HEAD_DIM];
            let scores: Vec<f64> = (0..visible)
                .map(|j| {
                    let kv = &k[j * width + h * HEAD_DIM..][..HEAD_DIM];
                    qv.iter()
                        .zip(kv)
                        .map(|(&a, &b)| f64::from(a) * f64::from(b))
                        .sum()
                })
                .collect();
            let max = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let weights: Vec<f64> = scores.iter().map(|s| (s - max).exp()).collect();
            let total: f64 = weights.iter().sum();
            for c in 0..HEAD_DIM {
                let acc: f64 = (0..visible)
                    .map(|j| weights[j] * f64::from(v[j * width + h * HEAD_DIM + c]))
                    .sum();
                out[t * width + h * HEAD_DIM + c] = (acc / total) as f32;
            }
        }
    }
    out
}

#[test]
fn attention_matches_the_cpu() {
    let d = MetalDevice::new().unwrap();
    let heads = 2;
    let width = heads * HEAD_DIM;
    // (query rows, cached prefix rows, causal): the text prefill, the image
    // steps (with a ragged tail, and a single row), and an image-only pass.
    for (tokens, prefix, causal) in [
        (150usize, 0usize, true),
        (37, 0, true),
        (100, 37, false),
        (64, 64, false),
        (1, 20, false),
        (130, 0, false),
    ] {
        let keys = prefix + tokens;
        let round = |x: Vec<f32>| -> Vec<f32> { x.iter().map(|&a| f16(a)).collect() };
        let q = round(
            random(tokens * width, 20 + tokens as u64)
                .iter()
                .map(|x| x * 2.)
                .collect(),
        );
        let k = round(
            random(keys * width, 21 + tokens as u64)
                .iter()
                .map(|x| x * 2.)
                .collect(),
        );
        let v = round(random(keys * width, 22 + tokens as u64));
        let kv = |rows: std::ops::Range<usize>, x: &[f32]| {
            tensor(
                &d,
                &[rows.len().max(1), width],
                DType::F16,
                &x[rows.start * width..rows.end * width].to_vec().pad(width),
            )
        };
        let (kp, vp) = (kv(0..prefix, &k), kv(0..prefix, &v));
        let (kc, vc) = (kv(prefix..keys, &k), kv(prefix..keys, &v));
        let qt = tensor(&d, &[tokens, width], DType::F16, &q);
        let out = Tensor::zeros(&d, [tokens, width], DType::F32).unwrap();
        let values = KeyValues {
            prefix_k: &kp,
            prefix_v: &vp,
            prefix,
            k: &kc,
            v: &vc,
        };
        ops::attention(&d, &qt, &values, &out, tokens, heads, keys, causal).unwrap();
        let want = attention_reference(&q, &k, &v, tokens, keys, heads, causal);
        close(
            &out.to_f32(),
            &want,
            3e-3,
            &format!("attention tokens {tokens} prefix {prefix} causal {causal}"),
        );
    }
}

trait Pad {
    fn pad(self, width: usize) -> Vec<f32>;
}

impl Pad for Vec<f32> {
    /// An empty slice becomes one zero row, since tensors can't be empty.
    fn pad(self, width: usize) -> Vec<f32> {
        if self.is_empty() {
            vec![0.; width]
        } else {
            self
        }
    }
}

mod decoder_kernels {
    use super::*;
    use crate::diffusion::conv::{self, Conv, Operand, Residual};

    fn rounded(x: Vec<f32>) -> Vec<f32> {
        x.into_iter().map(f16).collect()
    }

    /// One conv case: output `h x w`, input `h/up x w/up`, `k` x `k` kernel.
    struct Case {
        h: usize,
        w: usize,
        cin: usize,
        cout: usize,
        k: usize,
        up: usize,
    }

    /// 0: none, 1: same-shape residual, 2: pixel-shuffled shortcut (frames 2, repeats 4).
    fn run(c: &Case, mode: u32) {
        let d = MetalDevice::new().unwrap();
        let (ih, iw) = (c.h / c.up, c.w / c.up);
        let x = rounded(random(ih * iw * c.cin, 1 + c.cin as u64));
        let wt = rounded(
            random(c.cout * c.cin * c.k * c.k, 2 + c.cout as u64)
                .iter()
                .map(|v| v * 0.3)
                .collect(),
        );
        let bias = rounded(random(c.cout, 3));
        let (frames, repeats) = (2usize, 4usize);
        let src_channels = c.cout * frames * 4 / repeats;
        let res_same = rounded(random(c.h * c.w * c.cout, 4));
        let res_half = rounded(random((c.h / 2) * (c.w / 2) * src_channels, 5));
        let conv_w = Conv::from_values(&d, &wt, &bias, c.cin, c.cout, c.k).unwrap();
        let xt = tensor(&d, &[ih * iw, c.cin], DType::F16, &x);
        let yt = Tensor::zeros(&d, [c.h * c.w, c.cout], DType::F16).unwrap();
        let same = tensor(&d, &[c.h * c.w, c.cout], DType::F16, &res_same);
        let half = tensor(
            &d,
            &[(c.h / 2) * (c.w / 2), src_channels],
            DType::F16,
            &res_half,
        );
        let res = match mode {
            1 => Residual::Same(&same),
            2 => Residual::Shuffled {
                src: &half,
                channels: src_channels,
                frames,
                repeats,
            },
            _ => Residual::None,
        };
        conv::conv(&d, &conv_w, &xt, &yt, (c.h, c.w), c.up, res).unwrap();

        let pad = (c.k / 2) as isize;
        let mut want = vec![0f32; c.h * c.w * c.cout];
        for yy in 0..c.h {
            for xx in 0..c.w {
                for o in 0..c.cout {
                    let mut acc = f64::from(bias[o]);
                    for ky in 0..c.k {
                        for kx in 0..c.k {
                            let (iy, ix) = (
                                yy as isize + ky as isize - pad,
                                xx as isize + kx as isize - pad,
                            );
                            if iy < 0 || ix < 0 || iy >= c.h as isize || ix >= c.w as isize {
                                continue;
                            }
                            let (sy, sx) = (iy as usize / c.up, ix as usize / c.up);
                            for ch in 0..c.cin {
                                acc += f64::from(x[(sy * iw + sx) * c.cin + ch])
                                    * f64::from(wt[((o * c.cin + ch) * c.k + ky) * c.k + kx]);
                            }
                        }
                    }
                    let p = yy * c.w + xx;
                    acc += match mode {
                        1 => f64::from(res_same[p * c.cout + o]),
                        2 => {
                            let j = ((o * frames + frames - 1) * 2 + (yy & 1)) * 2 + (xx & 1);
                            let sc = j / repeats;
                            f64::from(res_half[((yy / 2) * (c.w / 2) + xx / 2) * src_channels + sc])
                        }
                        _ => 0.,
                    };
                    want[p * c.cout + o] = acc as f32;
                }
            }
        }
        close(
            &yt.to_f32(),
            &want,
            4e-3,
            &format!(
                "conv {}x{} {}->{} k{} up{} mode{mode}",
                c.h, c.w, c.cin, c.cout, c.k, c.up
            ),
        );
    }

    #[test]
    fn convolutions_match_the_cpu() {
        let case = |h, w, cin, cout, k, up| Case {
            h,
            w,
            cin,
            cout,
            k,
            up,
        };
        run(&case(9, 13, 64, 70, 3, 1), 0);
        run(&case(10, 6, 48, 20, 3, 1), 1);
        run(&case(8, 12, 32, 64, 3, 2), 0);
        run(&case(12, 10, 144, 33, 3, 2), 1);
        run(&case(7, 7, 32, 16, 1, 1), 0);
        run(&case(8, 8, 64, 32, 3, 1), 2);
        run(&case(16, 16, 288, 144, 3, 1), 0);
    }

    #[test]
    fn channel_rms_matches_the_cpu() {
        let d = MetalDevice::new().unwrap();
        for (pixels, channels, silu) in
            [(9usize, 144usize, false), (7, 1152, true), (130, 64, true)]
        {
            let x = rounded(
                random(pixels * channels, 6)
                    .iter()
                    .map(|v| v * 4.)
                    .collect(),
            );
            let gamma = random(channels, 7)
                .iter()
                .map(|v| 1. + v * 0.3)
                .collect::<Vec<_>>();
            let xt = tensor(&d, &[pixels, channels], DType::F16, &x);
            let gt = tensor(&d, &[channels], DType::F32, &gamma);
            let yt = Tensor::zeros(&d, [pixels, channels], DType::F16).unwrap();
            conv::channel_rms(&d, &xt, &gt, &yt, pixels, silu).unwrap();
            let mut want = Vec::new();
            for row in x.chunks(channels) {
                let norm = row
                    .iter()
                    .map(|&v| f64::from(v).powi(2))
                    .sum::<f64>()
                    .sqrt()
                    .max(1e-12);
                want.extend(row.iter().zip(&gamma).map(|(&v, &g)| {
                    let y = f64::from(v) / norm * (channels as f64).sqrt() * f64::from(g);
                    (if silu { y / (1. + (-y).exp()) } else { y }) as f32
                }));
            }
            close(
                &yt.to_f32(),
                &want,
                2e-3,
                &format!("rms {pixels}x{channels}"),
            );
        }
    }

    #[test]
    fn strided_gemm_and_softmax_match_the_cpu() {
        let d = MetalDevice::new().unwrap();
        // The decoder's attention: scores of Q against K read from a packed
        // [q | k | v] row, then weights times V read transposed.
        let (pixels, dim) = (150usize, 96usize);
        let row = 3 * dim;
        let qkv = rounded(random(pixels * row, 8));
        let t = tensor(&d, &[pixels, row], DType::F16, &qkv);
        let (q, k, v) = (
            t.view(0, [pixels * row]).unwrap(),
            t.view(dim, [pixels * row - dim]).unwrap(),
            t.view(2 * dim, [pixels * row - 2 * dim]).unwrap(),
        );
        let scores = Tensor::zeros(&d, [pixels, pixels], DType::F32).unwrap();
        let qa = Operand {
            t: &q,
            rows: pixels,
            stride_row: row,
            stride_k: 1,
        };
        let ka = Operand {
            t: &k,
            rows: pixels,
            stride_row: row,
            stride_k: 1,
        };
        conv::gemm(&d, &qa, &ka, &scores, dim, pixels).unwrap();
        let mut want = vec![0f32; pixels * pixels];
        for i in 0..pixels {
            for j in 0..pixels {
                want[i * pixels + j] = (0..dim)
                    .map(|c| f64::from(qkv[i * row + c]) * f64::from(qkv[j * row + dim + c]))
                    .sum::<f64>() as f32;
            }
        }
        close(&scores.to_f32(), &want, 2e-3, "scores");

        let weights = Tensor::zeros(&d, [pixels, pixels], DType::F16).unwrap();
        let scale = 1. / (dim as f32).sqrt();
        conv::softmax_rows(&d, &scores, &weights, pixels, pixels, scale).unwrap();
        let mut probs = vec![0f32; pixels * pixels];
        for i in 0..pixels {
            let s: Vec<f64> = (0..pixels)
                .map(|j| f64::from(want[i * pixels + j] * scale))
                .collect();
            let max = s.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let total: f64 = s.iter().map(|x| (x - max).exp()).sum();
            for j in 0..pixels {
                probs[i * pixels + j] = ((s[j] - max).exp() / total) as f32;
            }
        }
        close(&weights.to_f32(), &probs, 2e-3, "softmax");

        let out = Tensor::zeros(&d, [pixels, dim], DType::F16).unwrap();
        let pa = Operand {
            t: &weights,
            rows: pixels,
            stride_row: pixels,
            stride_k: 1,
        };
        let vt = Tensor::zeros(&d, [dim, pixels], DType::F16).unwrap();
        conv::transpose(&d, &v, &vt, pixels, dim, row).unwrap();
        let va = Operand {
            t: &vt,
            rows: dim,
            stride_row: pixels,
            stride_k: 1,
        };
        conv::gemm(&d, &pa, &va, &out, pixels, dim).unwrap();
        let got_probs: Vec<f32> = weights.to_f32();
        let mut want_out = vec![0f32; pixels * dim];
        for i in 0..pixels {
            for c in 0..dim {
                want_out[i * dim + c] = (0..pixels)
                    .map(|j| {
                        f64::from(got_probs[i * pixels + j]) * f64::from(qkv[j * row + 2 * dim + c])
                    })
                    .sum::<f64>() as f32;
            }
        }
        close(&out.to_f32(), &want_out, 4e-3, "weights x V");
    }
}
