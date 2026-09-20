//! Small CPU sampler: temperature, then top-k, then nucleus (top-p).
#![forbid(unsafe_code)]
use crate::{Error, Result, generation::argmax};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
#[derive(Debug, Clone, Copy)]
pub struct SamplingConfig {
    pub temperature: f64,
    pub top_k: usize,
    pub top_p: f64,
    pub seed: u64,
}
impl Default for SamplingConfig {
    fn default() -> Self {
        Self {
            temperature: 0.,
            top_k: 0,
            top_p: 1.,
            seed: 0,
        }
    }
}
impl SamplingConfig {
    pub fn validate(&self) -> Result<()> {
        if !self.temperature.is_finite()
            || self.temperature < 0.
            || !self.top_p.is_finite()
            || self.top_p <= 0.
            || self.top_p > 1.
        {
            return Err(Error::Parameter(
                "sampling: finite temperature >= 0 and 0 < top-p <= 1 required".into(),
            ));
        }
        Ok(())
    }
}
pub struct Sampler {
    config: SamplingConfig,
    rng: ChaCha8Rng,
}
impl Sampler {
    pub fn new(config: SamplingConfig) -> Result<Self> {
        config.validate()?;
        Ok(Self {
            rng: ChaCha8Rng::seed_from_u64(config.seed),
            config,
        })
    }
    pub fn sample(&mut self, logits: &[f32]) -> Result<u32> {
        let best = argmax(logits)?;
        if self.config.temperature == 0. {
            return Ok(best);
        }
        let mut candidates: Vec<_> = logits.iter().copied().enumerate().collect();
        candidates.sort_by(|(ai, a), (bi, b)| b.total_cmp(a).then(ai.cmp(bi)));
        if self.config.top_k > 0 {
            candidates.truncate(self.config.top_k);
        }
        let max = candidates[0].1 as f64;
        let mut probs: Vec<_> = candidates
            .iter()
            .map(|(_, v)| ((*v as f64 - max) / self.config.temperature).exp())
            .collect();
        let total: f64 = probs.iter().sum();
        let mut cumulative = 0.;
        let mut keep = probs.len();
        for (i, p) in probs.iter().enumerate() {
            cumulative += p;
            if cumulative >= self.config.top_p * total {
                keep = i + 1;
                break;
            }
        }
        probs.truncate(keep);
        let draw = self.rng.random::<f64>() * probs.iter().sum::<f64>();
        let mut cumulative = 0.;
        for (i, p) in probs.iter().enumerate() {
            cumulative += p;
            if draw < cumulative {
                return Ok(candidates[i].0 as u32);
            }
        }
        Ok(candidates[keep - 1].0 as u32)
    }
}
