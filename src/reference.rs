//! Scalar CPU oracles only; never used by the GPU execution path.
#![forbid(unsafe_code)]
pub fn add(a: &[f32], b: &[f32]) -> Vec<f32> {
    assert_eq!(a.len(), b.len());
    a.iter().zip(b).map(|(x, y)| x + y).collect()
}
pub fn mul(a: &[f32], b: &[f32]) -> Vec<f32> {
    assert_eq!(a.len(), b.len());
    a.iter().zip(b).map(|(x, y)| x * y).collect()
}
pub fn silu(a: &[f32]) -> Vec<f32> {
    a.iter()
        .map(|&x| {
            let s = if x >= 0. {
                1. / (1. + (-x).exp())
            } else {
                x.exp() / (1. + x.exp())
            };
            x * s
        })
        .collect()
}
pub fn rmsnorm(a: &[f32], w: &[f32], eps: f32) -> Vec<f32> {
    a.chunks(w.len())
        .flat_map(|row| {
            let inv = (row.iter().map(|&x| (x as f64).powi(2)).sum::<f64>() / w.len() as f64
                + eps as f64)
                .sqrt()
                .recip();
            row.iter()
                .zip(w)
                .map(move |(&x, &w)| (x as f64 * inv * w as f64) as f32)
        })
        .collect()
}
pub fn softmax(a: &[f32], width: usize) -> Vec<f32> {
    a.chunks(width)
        .flat_map(|row| {
            let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
            let e: Vec<_> = row.iter().map(|&x| (x as f64 - max).exp()).collect();
            let sum = e.iter().sum::<f64>();
            e.into_iter().map(move |x| (x / sum) as f32)
        })
        .collect()
}
pub fn rope(a: &[f32], head: usize, position: u32, theta: f32) -> Vec<f32> {
    let mut out = Vec::with_capacity(a.len());
    for (pair, xy) in a.chunks_exact(2).enumerate() {
        let angle =
            position as f64 * (theta as f64).powf(-((pair * 2 % head) as f64) / head as f64);
        let (co, si) = (angle.cos(), angle.sin());
        out.push((xy[0] as f64 * co - xy[1] as f64 * si) as f32);
        out.push((xy[0] as f64 * si + xy[1] as f64 * co) as f32);
    }
    out
}
pub fn matmul(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    (0..m * n)
        .map(|i| {
            let (row, col) = (i / n, i % n);
            (0..k)
                .map(|j| a[row * k + j] as f64 * b[j * n + col] as f64)
                .sum::<f64>() as f32
        })
        .collect()
}
pub fn deterministic(n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| ((i.wrapping_mul(73).wrapping_add(19) % 257) as f32 - 128.) / 73.)
        .collect()
}
pub fn check(actual: &[f32], expected: &[f32], atol: f32, rtol: f32) -> crate::Result<f32> {
    if actual.len() != expected.len() {
        return Err(crate::Error::Validation("length mismatch".into()));
    }
    let mut max = 0f32;
    for (i, (&a, &e)) in actual.iter().zip(expected).enumerate() {
        let diff = (a - e).abs();
        if !a.is_finite() || !e.is_finite() || diff > atol + rtol * e.abs() {
            return Err(crate::Error::Validation(format!(
                "index {i}: actual {a}, expected {e}, abs error {diff}"
            )));
        }
        max = max.max(diff);
    }
    Ok(max)
}

pub mod transformer;
