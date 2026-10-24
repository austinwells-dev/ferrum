//! MLX affine 4-bit projections (`Format::Mlx4`, `hybrid_mlx.metal`) against
//! a CPU dequantization: every dispatch band of `HybridModel::project` (GEMV,
//! batched GEMV, narrow and wide TensorOps tiles), and the embedding gather.
use super::engine::HybridModel;
use super::weights::{Format, Matrix};
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

/// A random MLX matrix: its three arrays as MLX stores them, and its
/// dequantized values (scale * q + bias per group of 64).
fn mlx_matrix(rows: usize, cols: usize, seed: u64) -> (Vec<u8>, Vec<u8>, Vec<u8>, Vec<f32>) {
    let codes: Vec<u8> = random(rows * cols / 2, seed)
        .iter()
        .map(|v| ((v + 1.) * 127.9) as u8)
        .collect();
    let bf16 = |v: f32| half::bf16::from_f32(v);
    let scales: Vec<half::bf16> = random(rows * cols / 64, seed + 1)
        .iter()
        .map(|v| bf16(v * 0.02))
        .collect();
    let biases: Vec<half::bf16> = random(rows * cols / 64, seed + 2)
        .iter()
        .map(|v| bf16(v * 0.1))
        .collect();
    let mut values = vec![0f32; rows * cols];
    for r in 0..rows {
        for k in 0..cols {
            let byte = codes[(r * cols + k) / 2];
            let q = if k % 2 == 0 { byte & 0xF } else { byte >> 4 };
            let g = r * cols / 64 + k / 64;
            values[r * cols + k] = scales[g].to_f32() * f32::from(q) + biases[g].to_f32();
        }
    }
    let bytes = |v: &[half::bf16]| v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<_>>();
    (codes, bytes(&scales), bytes(&biases), values)
}

#[test]
fn mlx4_projections_match_the_cpu() {
    let d = MetalDevice::new().unwrap();
    // (rows, cols): a q/k/v-shaped and an FFN-shaped matrix, odd row counts.
    for (rows, cols) in [(1030usize, 5120usize), (96, 17408)] {
        let (codes, scales, biases, w) = mlx_matrix(rows, cols, 3);
        let m = Matrix::from_mlx(&d, rows, cols, &codes, &scales, &biases).unwrap();
        assert_eq!(m.format, Format::Mlx4);
        for tokens in [1usize, 2, 3, 4, 5, 9, 16, 33, 130] {
            let x = random(tokens * cols, 7 + tokens as u64);
            let xt = Tensor::from_f32(&d, [tokens, cols], DType::F32, &x).unwrap();
            let y = Tensor::zeros(&d, [tokens, rows], DType::F32).unwrap();
            let e = d.execution_with_shared_encoder(true).unwrap();
            HybridModel::project(&d, &m, &xt, tokens, &y).unwrap();
            e.finish().unwrap();
            let got = y.to_f32();
            let (mut worst, mut scale) = (0f64, 0f64);
            for t in 0..tokens {
                for r in 0..rows {
                    let want: f64 = (0..cols)
                        .map(|k| f64::from(w[r * cols + k]) * f64::from(x[t * cols + k]))
                        .sum();
                    scale = scale.max(want.abs());
                    worst = worst.max((f64::from(got[t * rows + r]) - want).abs());
                }
            }
            // The GEMVs accumulate in F32; the tiles round weights to F16.
            let bound = if tokens <= 4 { 1e-4 } else { 2e-3 };
            assert!(
                worst / scale < bound,
                "[{rows}, {cols}] x {tokens} rows: max error {worst:.3e} of {scale:.3e}"
            );
        }
    }
}

#[test]
fn mlx4_embedding_rows_match_the_cpu() {
    let d = MetalDevice::new().unwrap();
    let (rows, cols) = (300usize, 5120usize);
    let (codes, scales, biases, w) = mlx_matrix(rows, cols, 11);
    let m = Matrix::from_mlx(&d, rows, cols, &codes, &scales, &biases).unwrap();
    let ids = [0u32, 299, 17, 17, 150];
    let ids_t = Tensor::from_le_bytes(
        &d,
        [ids.len()],
        DType::F32,
        &ids.iter().flat_map(|i| i.to_le_bytes()).collect::<Vec<_>>(),
    )
    .unwrap();
    let y = Tensor::zeros(&d, [ids.len(), cols], DType::F32).unwrap();
    let e = d.execution_with_shared_encoder(true).unwrap();
    HybridModel::get_rows(&d, &m, &ids_t, &y).unwrap();
    e.finish().unwrap();
    let got = y.to_f32();
    for (t, &id) in ids.iter().enumerate() {
        let want = &w[id as usize * cols..][..cols];
        assert!(
            got[t * cols..][..cols]
                .iter()
                .zip(want)
                .all(|(a, b)| a == b),
            "row {id}"
        );
    }
    // A prefix of the rows (a draft vocabulary) reads the same rows.
    assert_eq!(m.prefix_rows(100).row_bytes, m.row_bytes);
}
