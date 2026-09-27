//! Kernel and bandwidth probes on real weights (development tooling).
#![forbid(unsafe_code)]
use super::{
    engine::{HybridModel, Params},
    weights::{Ffn, Matrix, Mixer},
};
use crate::{DType, MetalDevice, Result, Tensor};

impl HybridModel {
    /// Kernel microbenchmark on real weights: every distinct (format, shape)
    /// projection of the first four layers is run `iters` times with `m`
    /// activation rows inside one command buffer. Returns
    /// (description, microseconds per call, GB/s of weight bytes).
    pub fn bench_projections(
        &self,
        d: &MetalDevice,
        m: usize,
        iters: usize,
    ) -> Result<Vec<(String, f64, f64)>> {
        self.projections(d, m, iters, false)
    }

    /// Batched-kernel correctness on real weights: every distinct projection
    /// with `m` activation rows against `m` single-row GEMVs. Returns
    /// (description, max |difference| / max |reference|, 0).
    pub fn check_projections(&self, d: &MetalDevice, m: usize) -> Result<Vec<(String, f64, f64)>> {
        self.projections(d, m, 0, true)
    }

    fn check_projection(
        &self,
        d: &MetalDevice,
        role: &str,
        w: &Matrix,
        m: usize,
    ) -> Result<(String, f64, f64)> {
        let values: Vec<f32> = (0..m * w.cols)
            .map(|i| ((i * 2654435761usize) % 2001) as f32 / 1000. - 1.)
            .collect();
        let x = Tensor::from_f32(d, [m, w.cols], DType::F32, &values)?;
        let batched = Tensor::zeros(d, [m, w.rows], DType::F32)?;
        let single = Tensor::zeros(d, [m, w.rows], DType::F32)?;
        let e = d.execution_with_shared_encoder(true)?;
        self.project(d, w, &x, m, &batched)?;
        for r in 0..m {
            let xr = x.view(r * w.cols, [1, w.cols])?;
            let yr = single.view(r * w.rows, [1, w.rows])?;
            self.project(d, w, &xr, 1, &yr)?;
        }
        e.finish()?;
        let (b, s) = (batched.to_f32(), single.to_f32());
        if std::env::var_os("FERRUM_CHECK_VERBOSE").is_some() {
            eprintln!("{role}: batched {:?} single {:?}", &b[..3], &s[..3]);
        }
        let scale = s.iter().fold(0f32, |a, v| a.max(v.abs())).max(1e-6);
        let worst = b
            .iter()
            .zip(&s)
            .fold(0f32, |a, (x, y)| a.max((x - y).abs()));
        Ok((
            format!("{role} {} [{}x{}]", w.format.name(), w.rows, w.cols),
            (worst / scale) as f64,
            0.,
        ))
    }

    fn projections(
        &self,
        d: &MetalDevice,
        m: usize,
        iters: usize,
        check: bool,
    ) -> Result<Vec<(String, f64, f64)>> {
        let mut seen = std::collections::BTreeSet::new();
        let mut mats: Vec<(&str, &Matrix)> = vec![("output", &self.weights.output)];
        for layer in self.weights.layers.iter().take(4) {
            match &layer.mixer {
                Mixer::Attention(a) => {
                    mats.extend([("attn_q", &a.q), ("attn_k", &a.k), ("attn_o", &a.o)])
                }
                Mixer::Delta(w) => mats.extend([
                    ("ssm_qkv", &w.qkv),
                    ("ssm_z", &w.z),
                    ("ssm_alpha", &w.alpha),
                    ("ssm_out", &w.out),
                ]),
            }
            match &layer.ffn {
                Ffn::Dense(f) => mats.extend([
                    ("ffn_gate", &f.gate),
                    ("ffn_up", &f.up),
                    ("ffn_down", &f.down),
                ]),
                Ffn::Moe(f) => mats.extend([
                    ("router", &f.router),
                    ("shexp_gate", &f.shared.gate),
                    ("shexp_down", &f.shared.down),
                ]),
            }
        }
        let mut results = Vec::new();
        for (role, w) in mats {
            if !seen.insert((w.format.name(), w.rows, w.cols)) {
                continue;
            }
            if check {
                results.push(self.check_projection(d, role, w, m)?);
                continue;
            }
            let x = Tensor::from_f32(d, [m, w.cols], DType::F32, &vec![0.01; m * w.cols])?;
            let y = Tensor::zeros(d, [m, w.rows], DType::F32)?;
            let run = |n: usize| -> Result<f64> {
                let e = d.execution_with_shared_encoder(true)?;
                let start = std::time::Instant::now();
                for _ in 0..n {
                    self.project(d, w, &x, m, &y)?;
                }
                e.finish()?;
                Ok(start.elapsed().as_secs_f64())
            };
            run(2)?;
            let secs = run(iters)?;
            let per = secs / iters as f64;
            results.push((
                format!("{role} {} [{}x{}]", w.format.name(), w.rows, w.cols),
                per * 1e6,
                w.byte_size() as f64 / per / 1e9,
            ));
        }
        Ok(results)
    }
}

/// Streaming-read bandwidth probe (GB/s) over `bytes` with `per_group` bytes
/// per threadgroup and `threads` threads per threadgroup.
pub fn bench_read_bandwidth(
    d: &MetalDevice,
    bytes: usize,
    per_group: usize,
    threads: usize,
) -> Result<f64> {
    let x = Tensor::zeros(d, [bytes / 4], DType::F32)?;
    let groups = bytes / per_group;
    let y = Tensor::zeros(d, [groups], DType::F32)?;
    let run = |n: usize| -> Result<f64> {
        let e = d.execution_with_shared_encoder(false)?;
        let start = std::time::Instant::now();
        for _ in 0..n {
            d.dispatch_hybrid(
                "h_bw_read",
                &[x.binding()],
                &[y.binding()],
                &Params::default().u(per_group / 16)?.0,
                [groups, 1, 1],
                [threads, 1, 1],
                0,
            )?;
        }
        e.finish()?;
        Ok(start.elapsed().as_secs_f64())
    };
    run(2)?;
    let n = 20;
    Ok(bytes as f64 * n as f64 / run(n)? / 1e9)
}

impl HybridModel {
    /// Decode attention alone: one query token against `keys` cached
    /// positions. Returns (microseconds per call, GB/s of K+V read).
    pub fn bench_decode_attention(
        &self,
        d: &MetalDevice,
        keys: usize,
        iters: usize,
    ) -> Result<(f64, f64)> {
        let c = &self.config;
        let stride = c.kv_heads * c.head_dim;
        let cap = keys.next_multiple_of(32);
        let q = Tensor::zeros(d, [1, c.heads * c.head_dim], DType::F16)?;
        let kc = Tensor::zeros(d, [cap, stride], DType::F16)?;
        let vc = Tensor::zeros(d, [cap, stride], DType::F16)?;
        let qg = Tensor::zeros(d, [1, 2 * c.heads * c.head_dim], DType::F32)?;
        let att = Tensor::zeros(d, [1, c.heads * c.head_dim], DType::F32)?;
        let s = self.scratch.borrow();
        let run = |n: usize| -> Result<f64> {
            let e = d.execution_with_shared_encoder(true)?;
            let start = std::time::Instant::now();
            for _ in 0..n {
                self.flash_attention(d, &s, &q, &kc, &vc, &qg, &att, 1, keys - 1)?;
            }
            e.finish()?;
            Ok(start.elapsed().as_secs_f64())
        };
        // Warm up until the GPU has run at least 300 ms, so clocks are up.
        let warm = std::time::Instant::now();
        while warm.elapsed().as_secs_f64() < 0.3 {
            run(iters)?;
        }
        let per = run(iters)? / iters as f64;
        Ok((per * 1e6, 2. * (keys * stride * 2) as f64 / per / 1e9))
    }
}

impl HybridModel {
    /// Prompt-chunk attention alone: `m` queries at positions `keys - m ..
    /// keys`. Returns (milliseconds per layer, effective TFLOP/s).
    pub fn bench_prefill_attention(
        &self,
        d: &MetalDevice,
        m: usize,
        keys: usize,
        iters: usize,
    ) -> Result<(f64, f64)> {
        let c = &self.config;
        let stride = c.kv_heads * c.head_dim;
        let cap = keys.next_multiple_of(32);
        let q = Tensor::zeros(d, [m, c.heads * c.head_dim], DType::F16)?;
        let kc = Tensor::zeros(d, [cap, stride], DType::F16)?;
        let vc = Tensor::zeros(d, [cap, stride], DType::F16)?;
        let qg = Tensor::zeros(d, [m, 2 * c.heads * c.head_dim], DType::F32)?;
        let att = Tensor::zeros(d, [m, c.heads * c.head_dim], DType::F32)?;
        let s = self.scratch.borrow();
        let run = |n: usize| -> Result<f64> {
            let e = d.execution_with_shared_encoder(true)?;
            let start = std::time::Instant::now();
            for _ in 0..n {
                self.flash_attention(d, &s, &q, &kc, &vc, &qg, &att, m, keys - m)?;
            }
            e.finish()?;
            Ok(start.elapsed().as_secs_f64())
        };
        let warm = std::time::Instant::now();
        while warm.elapsed().as_secs_f64() < 0.3 {
            run(1)?;
        }
        let per = run(iters)? / iters as f64;
        // QKᵀ and PV over the causal region: ~ m * (keys - m/2) pairs.
        let pairs = m as f64 * (keys as f64 - m as f64 / 2.);
        let flops = 4. * pairs * c.head_dim as f64 * c.heads as f64;
        Ok((per * 1e3, flops / per / 1e12))
    }
}

impl HybridModel {
    /// Speculative verify cost: after `context` cached positions, forward `m`
    /// tokens with a final argmax, for each width in `widths`. The recurrent
    /// state is restored between runs so every run starts from the same
    /// position. Returns (width, median milliseconds per forward).
    pub fn bench_verify_widths(
        &self,
        d: &MetalDevice,
        context: usize,
        widths: &[usize],
        reps: usize,
    ) -> Result<Vec<(usize, f64)>> {
        use super::engine::{HybridState, Output};
        let widest = widths.iter().copied().max().unwrap_or(1);
        let mut state = HybridState::new(d, &self.config, context + widest + 1)?;
        let tokens: Vec<u32> = (0..(context + widest) as u32)
            .map(|i| 1000 + (i * 7919) % 20000)
            .collect();
        for piece in tokens[..context].chunks(self.chunk()) {
            self.forward(d, &mut state, piece, Output::None)?;
        }
        let saved = state
            .recurrent()
            .map(|t| Tensor::zeros_resident(d, [t.numel()], DType::F32))
            .collect::<Result<Vec<_>>>()?;
        state.save_recurrent(d, &saved)?;
        let mut results = Vec::new();
        for &m in widths {
            let mut times = Vec::new();
            for rep in 0..reps + 2 {
                state.restore_recurrent(d, &saved, context)?;
                let start = std::time::Instant::now();
                self.forward(d, &mut state, &tokens[context..context + m], Output::Argmax)?;
                if rep >= 2 {
                    times.push(start.elapsed().as_secs_f64() * 1e3);
                }
            }
            times.sort_by(f64::total_cmp);
            results.push((m, times[times.len() / 2]));
        }
        Ok(results)
    }
}
