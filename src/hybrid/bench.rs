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
