//! Flow-matching Euler sampling: the sigma schedule and the step it drives.
//!
//! Latents move from noise (`sigma = 1`) to the image (`sigma = 0`) along the
//! velocity the transformer predicts: `x += (sigma_next - sigma) * v`. The
//! transformer is conditioned on `sigma` directly.
use crate::{
    Error, MetalDevice, Result, Tensor, diffusion::config::SchedulerConfig, diffusion::ops,
};

/// `steps + 1` sigmas, ending at 0.
#[derive(Debug, Clone, PartialEq)]
pub struct Schedule {
    sigmas: Vec<f32>,
}

impl Schedule {
    /// A checkpoint's own sigmas (Turbo's 8-step list), used as given.
    pub fn explicit(sigmas: &[f32]) -> Self {
        let mut sigmas = sigmas.to_vec();
        sigmas.push(0.);
        Self { sigmas }
    }

    /// `steps` linearly spaced sigmas from 1 to `1 / steps`, shifted toward
    /// noise by an amount that grows with the image's token count (the base
    /// model's dynamic exponential shift), then stretched so the last sigma
    /// lands on `shift_terminal`.
    pub fn dynamic(steps: usize, image_tokens: usize, c: &SchedulerConfig) -> Result<Self> {
        // Stretching to a terminal sigma divides by how far the last step
        // moved, which is nothing for a single step (diffusers gives NaN).
        let fewest = if c.shift_terminal.is_some() { 2 } else { 1 };
        if steps < fewest {
            return Err(Error::Parameter(format!(
                "this schedule needs at least {fewest} steps"
            )));
        }
        let slope =
            (c.max_shift - c.base_shift) / (c.max_image_seq_len - c.base_image_seq_len) as f32;
        let mu = image_tokens as f32 * slope + (c.base_shift - slope * c.base_image_seq_len as f32);
        let mut sigmas: Vec<f32> = (0..steps)
            .map(|i| {
                let linear = if steps == 1 {
                    1.
                } else {
                    1. + (1. / steps as f32 - 1.) * i as f32 / (steps - 1) as f32
                };
                if c.use_dynamic_shifting {
                    mu.exp() / (mu.exp() + (1. / linear - 1.))
                } else {
                    c.shift * linear / (1. + (c.shift - 1.) * linear)
                }
            })
            .collect();
        if let Some(terminal) = c.shift_terminal {
            let last = 1. - sigmas[steps - 1];
            let scale = last / (1. - terminal);
            for s in &mut sigmas {
                *s = 1. - (1. - *s) / scale;
            }
        }
        sigmas.push(0.);
        Ok(Self { sigmas })
    }

    pub fn steps(&self) -> usize {
        self.sigmas.len() - 1
    }
    /// The sigma step `i` starts from.
    pub fn sigma(&self, i: usize) -> f32 {
        self.sigmas[i]
    }
    /// The Euler step size `sigma_next - sigma` of step `i`.
    pub fn delta(&self, i: usize) -> f32 {
        self.sigmas[i + 1] - self.sigmas[i]
    }
    /// Move `latents` along the predicted velocity for step `i`.
    pub fn advance(
        &self,
        d: &MetalDevice,
        i: usize,
        pred: &Tensor,
        latents: &Tensor,
    ) -> Result<()> {
        ops::euler(d, pred, latents, self.delta(i))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn base() -> SchedulerConfig {
        serde_json::from_str(
            r#"{"base_image_seq_len":256,"base_shift":0.5,"max_image_seq_len":8192,
            "max_shift":0.9,"shift":1.0,"shift_terminal":0.02,"use_dynamic_shifting":true}"#,
        )
        .unwrap()
    }

    fn close(a: f32, b: f32) {
        assert!((a - b).abs() < 2e-6, "{a} vs {b}");
    }

    /// Sigmas from diffusers' `FlowMatchEulerDiscreteScheduler`.
    #[test]
    fn the_base_schedule_shifts_with_image_size() {
        let s = Schedule::dynamic(40, 1024, &base()).unwrap();
        assert_eq!(s.steps(), 40);
        for (i, v) in [
            (0, 1.),
            (1, 0.984919),
            (2, 0.969517),
            (20, 0.623013),
            (38, 0.061581),
        ] {
            close(s.sigma(i), v);
        }
        close(s.sigma(39), 0.02);
        close(s.sigma(40), 0.);
        let large = Schedule::dynamic(40, 4096, &base()).unwrap();
        for (i, v) in [(1, 0.986964), (2, 0.973593), (20, 0.656666), (38, 0.067881)] {
            close(large.sigma(i), v);
        }
    }

    #[test]
    fn too_few_steps_are_an_error() {
        assert!(Schedule::dynamic(0, 1024, &base()).is_err());
        // The base model stretches to a terminal sigma, which one step can't.
        assert!(base().shift_terminal.is_some());
        assert!(Schedule::dynamic(1, 1024, &base()).is_err());
        let two = Schedule::dynamic(2, 1024, &base()).unwrap();
        assert!((0..=2).all(|i| two.sigma(i).is_finite()));
        let plain = SchedulerConfig {
            shift_terminal: None,
            ..base()
        };
        let one = Schedule::dynamic(1, 1024, &plain).unwrap();
        assert_eq!((one.steps(), one.sigma(1)), (1, 0.));
        assert!(one.sigma(0).is_finite());
    }

    #[test]
    fn an_explicit_list_is_used_as_given() {
        let list = [
            1.0, 0.978453, 0.95418, 0.926626, 0.89508, 0.845148, 0.704534, 0.414568,
        ];
        let s = Schedule::explicit(&list);
        assert_eq!(s.steps(), 8);
        close(s.delta(7), -0.414568);
        close(s.sigma(1), 0.978453);
    }
}
