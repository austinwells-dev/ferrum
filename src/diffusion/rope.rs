//! Rotary tables for the joint text + image sequence.
//!
//! Each head is rotated in consecutive (even, odd) pairs, `pairs()` of them,
//! split across three axes (frame, height, width). Text tokens advance one
//! shared position on all three axes. The image block sits at the frame
//! position reached after the text, and lays its tokens out on a height/width
//! grid centred on zero, so its spatial positions don't depend on the prompt
//! length.
use crate::diffusion::config::DitConfig;

const THETA: f32 = 10_000.;

pub struct RopeTable {
    pairs: usize,
    cos: Vec<f32>,
    sin: Vec<f32>,
}

impl RopeTable {
    /// Rows for `text` prompt tokens followed by one `grid.0 x grid.1` image.
    pub fn new(c: &DitConfig, text: usize, grid: (usize, usize)) -> Self {
        let (height, width) = grid;
        let pairs = c.head_dim / 2;
        let mut cos = Vec::with_capacity((text + height * width) * pairs);
        let mut sin = Vec::with_capacity(cos.capacity());
        let mut row = |frame: i64, h: i64, w: i64| {
            for (axis, position) in c.rope_axes.iter().zip([frame, h, w]) {
                for i in 0..axis / 2 {
                    let inverse = 1. / THETA.powf((2 * i) as f32 / *axis as f32);
                    let angle = position as f32 * inverse;
                    cos.push(angle.cos());
                    sin.push(angle.sin());
                }
            }
        };
        for t in 0..text as i64 {
            row(t, t, t);
        }
        // Rows count from -(n - n / 2) so an odd side puts the extra step on the negative side.
        let start = |n: usize| -((n - n / 2) as i64);
        for y in 0..height as i64 {
            for x in 0..width as i64 {
                row(text as i64, start(height) + y, start(width) + x);
            }
        }
        Self { pairs, cos, sin }
    }
    pub fn pairs(&self) -> usize {
        self.pairs
    }
    pub fn rows(&self) -> usize {
        self.cos.len() / self.pairs
    }
    /// Cosines of rows `first..`, row-major `[rows, pairs]`.
    pub fn cos(&self, first: usize) -> &[f32] {
        &self.cos[first * self.pairs..]
    }
    pub fn sin(&self, first: usize) -> &[f32] {
        &self.sin[first * self.pairs..]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config() -> DitConfig {
        DitConfig::from_json(
            r#"{"attention_head_dim":128,"axes_dims_rope":[16,56,56],"causal_condition":true,
            "context_in_dim":4096,"in_channels":64,"mlp_ratio":3,"num_attention_heads":32,
            "num_layers":32}"#,
        )
        .unwrap()
    }

    /// Values from diffusers' `QwenImage21Rope` for 3 text tokens and a 2 x 3
    /// image, at pairs 0, 1 (frame axis), 8, 9 (height) and 36, 37 (width).
    #[test]
    fn matches_the_diffusers_table() {
        let t = RopeTable::new(&config(), 3, (2, 3));
        assert_eq!((t.rows(), t.pairs()), (9, 64));
        let at = [0, 1, 8, 9, 36, 37];
        let golden: [(usize, [f32; 6], [f32; 6]); 4] = [
            (
                2,
                [
                    -0.416147, 0.806578, -0.416147, 0.131047, -0.416147, 0.131047,
                ],
                [0.909297, 0.591127, 0.909297, 0.991376, 0.909297, 0.991376],
            ),
            (
                3,
                [-0.989992, 0.582754, 0.540302, 0.752013, -0.416147, 0.131047],
                [
                    0.14112, 0.812649, -0.841471, -0.659148, -0.909297, -0.991376,
                ],
            ),
            (
                4,
                [-0.989992, 0.582754, 0.540302, 0.752013, 0.540302, 0.752013],
                [
                    0.14112, 0.812649, -0.841471, -0.659148, -0.841471, -0.659148,
                ],
            ),
            (
                8,
                [-0.989992, 0.582754, 1., 1., 1., 1.],
                [0.14112, 0.812649, 0., 0., 0., 0.],
            ),
        ];
        for (row, cos, sin) in golden {
            for (k, &pair) in at.iter().enumerate() {
                assert!(
                    (t.cos(row)[pair] - cos[k]).abs() < 1e-5,
                    "cos row {row} pair {pair}"
                );
                assert!(
                    (t.sin(row)[pair] - sin[k]).abs() < 1e-5,
                    "sin row {row} pair {pair}"
                );
            }
        }
    }
}
