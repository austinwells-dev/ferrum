//! The joint text + image sequence and its rotary tables.
//!
//! The sequence is the prompt's text with each condition image's latent
//! tokens in place of the encoder's image slots, then the target image.
//! Each head is rotated in consecutive (even, odd) pairs, `pairs()` of them,
//! split across three axes (frame, height, width). Text tokens advance one
//! shared position on all three axes. An image block sits at the frame
//! position the text before it reached, lays its tokens out on a
//! height/width grid centred on zero, and then advances the position by its
//! longer side.
use crate::diffusion::config::DitConfig;

const THETA: f32 = 10_000.;

/// A run of the sequence before the target image.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Segment {
    /// Prompt tokens.
    Text(usize),
    /// A condition image of `height x width` latent tokens, row-major.
    Image(usize, usize),
}

impl Segment {
    pub fn rows(self) -> usize {
        match self {
            Self::Text(n) => n,
            Self::Image(h, w) => h * w,
        }
    }
}

/// The joint sequence: the prefix (text and condition images, whose keys
/// and values every step reuses) then the `target` grid being denoised.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Layout {
    pub prefix: Vec<Segment>,
    pub target: (usize, usize),
}

impl Layout {
    pub fn text_to_image(text: usize, target: (usize, usize)) -> Self {
        Self {
            prefix: vec![Segment::Text(text)],
            target,
        }
    }
    pub fn prefix_rows(&self) -> usize {
        self.prefix.iter().map(|s| s.rows()).sum()
    }
    pub fn text_rows(&self) -> usize {
        self.prefix
            .iter()
            .map(|s| match s {
                Segment::Text(n) => *n,
                Segment::Image(..) => 0,
            })
            .sum()
    }
    pub fn condition_rows(&self) -> usize {
        self.prefix_rows() - self.text_rows()
    }
    pub fn target_rows(&self) -> usize {
        self.target.0 * self.target.1
    }
}

pub struct RopeTable {
    pairs: usize,
    cos: Vec<f32>,
    sin: Vec<f32>,
}

impl RopeTable {
    /// Rows for the whole sequence, prefix first.
    pub fn new(c: &DitConfig, layout: &Layout) -> Self {
        let pairs = c.head_dim / 2;
        let rows = layout.prefix_rows() + layout.target_rows();
        let mut cos = Vec::with_capacity(rows * pairs);
        let mut sin = Vec::with_capacity(rows * pairs);
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
        // Rows count from -(n - n / 2) so an odd side puts the extra step on the negative side.
        let start = |n: usize| -((n - n / 2) as i64);
        let mut position = 0i64;
        let target = Segment::Image(layout.target.0, layout.target.1);
        for segment in layout.prefix.iter().chain([&target]) {
            match *segment {
                Segment::Text(n) => {
                    for t in position..position + n as i64 {
                        row(t, t, t);
                    }
                    position += n as i64;
                }
                Segment::Image(height, width) => {
                    for y in 0..height as i64 {
                        for x in 0..width as i64 {
                            row(position, start(height) + y, start(width) + x);
                        }
                    }
                    position += height.max(width) as i64;
                }
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
        let t = RopeTable::new(&config(), &Layout::text_to_image(3, (2, 3)));
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

    /// diffusers' table for 2 text tokens, a 2 x 2 condition image, 1 text
    /// token and a 2 x 3 target: the text after the condition resumes at
    /// position 2 + max(2, 2), and the target's frame is the next one.
    #[test]
    fn places_condition_images_like_diffusers() {
        let layout = Layout {
            prefix: vec![Segment::Text(2), Segment::Image(2, 2), Segment::Text(1)],
            target: (2, 3),
        };
        let t = RopeTable::new(&config(), &layout);
        assert_eq!(t.rows(), 13);
        let at = [0, 1, 8, 9, 36, 37];
        let golden: [(usize, [f32; 6], [f32; 6]); 6] = [
            (
                1,
                [0.540302, 0.950415, 0.540302, 0.752013, 0.540302, 0.752013],
                [0.841471, 0.310984, 0.841471, 0.659148, 0.841471, 0.659148],
            ),
            (
                2,
                [-0.416147, 0.806578, 0.540302, 0.752013, 0.540302, 0.752013],
                [
                    0.909297, 0.591127, -0.841471, -0.659148, -0.841471, -0.659148,
                ],
            ),
            (
                5,
                [-0.416147, 0.806578, 1., 1., 1., 1.],
                [0.909297, 0.591127, 0., 0., 0., 0.],
            ),
            (
                6,
                [
                    -0.653644, 0.301137, -0.653644, -0.965653, -0.653644, -0.965653,
                ],
                [
                    -0.756802, 0.953581, -0.756802, 0.259834, -0.756802, 0.259834,
                ],
            ),
            (
                7,
                [0.283662, -0.010342, 0.540302, 0.752013, -0.416147, 0.131047],
                [
                    -0.958924, 0.999947, -0.841471, -0.659148, -0.909297, -0.991376,
                ],
            ),
            (
                12,
                [0.283662, -0.010342, 1., 1., 1., 1.],
                [-0.958924, 0.999947, 0., 0., 0., 0.],
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
