//! A conversation session: hybrid state plus the token history it
//! represents, recurrent-state snapshots for prefix reuse across requests,
//! and the sampling/generation loop.
#![forbid(unsafe_code)]
use super::engine::{HybridModel, HybridState, Output, Penalties, Produced};
use crate::{DType, Error, MetalDevice, Result, Tensor};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use std::time::{Duration, Instant};

#[derive(Debug, Clone)]
pub struct SamplingParams {
    /// 0 selects the argmax on the GPU.
    pub temperature: f32,
    /// 0 disables. Exact up to 32 (see `Output::Candidates`).
    pub top_k: usize,
    pub top_p: f32,
    pub min_p: f32,
    pub presence_penalty: f32,
    pub frequency_penalty: f32,
    pub repetition_penalty: f32,
    /// Generated tokens the penalties look back over.
    pub penalty_last_n: usize,
    pub seed: u64,
}

impl Default for SamplingParams {
    fn default() -> Self {
        Self {
            temperature: 0.,
            top_k: 0,
            top_p: 1.,
            min_p: 0.,
            presence_penalty: 0.,
            frequency_penalty: 0.,
            repetition_penalty: 1.,
            penalty_last_n: 256,
            seed: 0,
        }
    }
}

impl SamplingParams {
    pub fn validate(&self) -> Result<()> {
        let finite = [
            self.temperature,
            self.top_p,
            self.min_p,
            self.presence_penalty,
            self.frequency_penalty,
            self.repetition_penalty,
        ]
        .iter()
        .all(|v| v.is_finite());
        if !finite
            || self.temperature < 0.
            || !(0. ..=1.).contains(&self.top_p)
            || self.top_p == 0.
            || !(0. ..=1.).contains(&self.min_p)
            || self.repetition_penalty <= 0.
        {
            return Err(Error::Parameter("invalid sampling parameters".into()));
        }
        Ok(())
    }
    fn uses_candidates(&self) -> bool {
        self.temperature > 0.
            || self.presence_penalty != 0.
            || self.frequency_penalty != 0.
            || self.repetition_penalty != 1.
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StopReason {
    /// An end-of-sequence token was sampled (it is not included in `tokens`).
    Eos,
    /// `max_tokens` reached.
    Length,
    /// The context is full.
    ContextFull,
    /// The token callback asked to stop.
    Stopped,
}

#[derive(Debug, Clone)]
pub struct Completion {
    pub tokens: Vec<u32>,
    pub stop: StopReason,
    pub prompt_tokens: usize,
    /// Prompt tokens served from the session instead of being recomputed.
    pub reused_tokens: usize,
    pub prefill: Duration,
    pub decode: Duration,
}

struct Snapshot {
    position: usize,
    tensors: Vec<Tensor>,
    stamp: u64,
}

pub struct Session {
    state: HybridState,
    /// Tokens represented by `state` (always `state.len()` of them).
    tokens: Vec<u32>,
    snapshots: Vec<Snapshot>,
    clock: u64,
}

impl Session {
    /// A session with `capacity` positions and `snapshots` recurrent-state
    /// slots for prefix reuse (see `plan::PlanOptions::snapshots`).
    pub fn new(
        d: &MetalDevice,
        model: &HybridModel,
        capacity: usize,
        snapshots: usize,
    ) -> Result<Self> {
        let state = HybridState::new(d, &model.config, capacity)?;
        let mut slots = Vec::with_capacity(snapshots);
        for _ in 0..snapshots {
            let tensors = state
                .recurrent()
                .map(|t| Tensor::zeros_resident(d, [t.numel()], DType::F32))
                .collect::<Result<Vec<_>>>()?;
            slots.push(Snapshot {
                position: 0,
                tensors,
                stamp: 0,
            });
        }
        Ok(Self {
            state,
            tokens: Vec::new(),
            snapshots: slots,
            clock: 0,
        })
    }
    pub fn len(&self) -> usize {
        self.tokens.len()
    }
    pub fn is_empty(&self) -> bool {
        self.tokens.is_empty()
    }
    pub fn capacity(&self) -> usize {
        self.state.capacity()
    }
    pub fn tokens(&self) -> &[u32] {
        &self.tokens
    }
    pub fn state_bytes(&self) -> usize {
        self.state.byte_size()
            + self
                .snapshots
                .iter()
                .flat_map(|s| &s.tensors)
                .map(Tensor::byte_size)
                .sum::<usize>()
    }
    pub fn reset(&mut self) {
        self.state.reset();
        self.tokens.clear();
        for s in &mut self.snapshots {
            s.position = 0;
            s.stamp = 0;
        }
    }

    /// Bring the state to the longest reusable prefix of `prompt`, keeping at
    /// least the final prompt token to recompute. Returns the reused length.
    fn reuse_prefix(&mut self, d: &MetalDevice, prompt: &[u32]) -> Result<usize> {
        if !self.state.is_valid() {
            self.reset();
        }
        let limit = prompt.len().saturating_sub(1);
        let common = self
            .tokens
            .iter()
            .zip(prompt)
            .take(limit)
            .take_while(|(a, b)| a == b)
            .count();
        if common == self.tokens.len() {
            return Ok(common);
        }
        // Diverged inside the cached sequence: resume from the deepest
        // snapshot at or before the divergence point.
        let best = self
            .snapshots
            .iter_mut()
            .filter(|s| s.position > 0 && s.position <= common)
            .max_by_key(|s| s.position);
        match best {
            Some(snapshot) => {
                self.clock += 1;
                snapshot.stamp = self.clock;
                let position = snapshot.position;
                self.state
                    .restore_recurrent(d, &snapshot.tensors, position)?;
                self.tokens.truncate(position);
                // Snapshots past the restore point describe a discarded branch.
                for s in &mut self.snapshots {
                    if s.position > position {
                        s.position = 0;
                        s.stamp = 0;
                    }
                }
                Ok(position)
            }
            None => {
                self.reset();
                Ok(0)
            }
        }
    }

    fn snapshot(&mut self, d: &MetalDevice) -> Result<()> {
        let position = self.tokens.len();
        if position == 0 || self.snapshots.iter().any(|s| s.position == position) {
            return Ok(());
        }
        self.clock += 1;
        let Some(slot) = self.snapshots.iter_mut().min_by_key(|s| s.stamp) else {
            return Ok(());
        };
        self.state.save_recurrent(d, &slot.tensors)?;
        slot.position = position;
        slot.stamp = self.clock;
        Ok(())
    }

    /// Generate up to `max_tokens` after `prompt`. `on_token` sees each new
    /// (non-EOS) token and may return `false` to stop. The prompt shares any
    /// prefix already in the session; divergence resumes from a snapshot.
    #[allow(clippy::too_many_arguments)]
    pub fn generate(
        &mut self,
        d: &MetalDevice,
        model: &HybridModel,
        prompt: &[u32],
        max_tokens: usize,
        params: &SamplingParams,
        eos: &[u32],
        mut on_token: impl FnMut(u32) -> Result<bool>,
    ) -> Result<Completion> {
        params.validate()?;
        if prompt.is_empty() {
            return Err(Error::Parameter("prompt must not be empty".into()));
        }
        if prompt.len() >= self.capacity() {
            return Err(Error::Parameter(format!(
                "prompt of {} tokens does not fit the {}-token context",
                prompt.len(),
                self.capacity()
            )));
        }
        let start = Instant::now();
        let reused = self.reuse_prefix(d, prompt)?;
        let mut rng = ChaCha8Rng::seed_from_u64(params.seed);
        let mut generated: Vec<u32> = Vec::new();
        let output = |generated: &[u32]| -> Output {
            if params.uses_candidates() {
                Output::Candidates(penalties(params, generated))
            } else {
                Output::Argmax
            }
        };
        let produced = model.forward(d, &mut self.state, &prompt[reused..], output(&generated))?;
        self.tokens.extend_from_slice(&prompt[reused..]);
        self.snapshot(d)?;
        let prefill = start.elapsed();
        let decode_start = Instant::now();
        let mut next = select(produced, params, &mut rng)?;
        let stop = loop {
            if eos.contains(&next) {
                break StopReason::Eos;
            }
            generated.push(next);
            if !on_token(next)? {
                break StopReason::Stopped;
            }
            if generated.len() >= max_tokens {
                break StopReason::Length;
            }
            if self.state.len() + 1 > self.capacity() {
                break StopReason::ContextFull;
            }
            let produced = model.forward(d, &mut self.state, &[next], output(&generated))?;
            self.tokens.push(next);
            next = select(produced, params, &mut rng)?;
        };
        Ok(Completion {
            tokens: generated,
            stop,
            prompt_tokens: prompt.len(),
            reused_tokens: reused,
            prefill,
            decode: decode_start.elapsed(),
        })
    }
}

fn penalties(params: &SamplingParams, generated: &[u32]) -> Penalties {
    let window = &generated[generated.len().saturating_sub(params.penalty_last_n)..];
    let mut counts = std::collections::BTreeMap::<u32, u32>::new();
    for &t in window {
        *counts.entry(t).or_default() += 1;
    }
    Penalties {
        tokens: counts.into_iter().collect(),
        presence: params.presence_penalty,
        frequency: params.frequency_penalty,
        repetition: params.repetition_penalty,
    }
}

fn select(produced: Produced, params: &SamplingParams, rng: &mut ChaCha8Rng) -> Result<u32> {
    match produced {
        Produced::Token(t) => Ok(t),
        Produced::Candidates(mut c) => sample(&mut c, params, rng),
        _ => Err(Error::Parameter("forward produced no token".into())),
    }
}

/// Temperature, top-k, min-p then top-p over GPU candidates.
pub(crate) fn sample(
    candidates: &mut [(u32, f32)],
    params: &SamplingParams,
    rng: &mut impl Rng,
) -> Result<u32> {
    if candidates.is_empty() {
        return Err(Error::Parameter("no finite logits to sample".into()));
    }
    candidates.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
    if params.temperature == 0. {
        return Ok(candidates[0].0);
    }
    let keep = if params.top_k > 0 {
        params.top_k.min(candidates.len())
    } else {
        candidates.len()
    };
    let candidates = &candidates[..keep];
    let max = candidates[0].1 as f64;
    let mut probs: Vec<f64> = candidates
        .iter()
        .map(|(_, l)| ((*l as f64 - max) / params.temperature as f64).exp())
        .collect();
    let total: f64 = probs.iter().sum();
    for p in &mut probs {
        *p /= total;
    }
    let min = params.min_p as f64 * probs[0];
    let mut keep = probs.iter().take_while(|&&p| p >= min).count().max(1);
    let mut cumulative = 0.;
    for (i, p) in probs[..keep].iter().enumerate() {
        cumulative += p;
        if cumulative >= params.top_p as f64 {
            keep = i + 1;
            break;
        }
    }
    let mass: f64 = probs[..keep].iter().sum();
    let mut draw = rng.random::<f64>() * mass;
    for (i, p) in probs[..keep].iter().enumerate() {
        if draw < *p {
            return Ok(candidates[i].0);
        }
        draw -= p;
    }
    Ok(candidates[keep - 1].0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sampling_respects_top_k_min_p_and_greedy() {
        let mut rng = ChaCha8Rng::seed_from_u64(7);
        let mut c = vec![(5, 1.0), (9, 3.0), (2, 2.0)];
        let greedy = SamplingParams::default();
        assert_eq!(sample(&mut c, &greedy, &mut rng).unwrap(), 9);
        let top1 = SamplingParams {
            temperature: 1.,
            top_k: 1,
            ..Default::default()
        };
        for _ in 0..20 {
            assert_eq!(sample(&mut c, &top1, &mut rng).unwrap(), 9);
        }
        let min_p = SamplingParams {
            temperature: 1.,
            min_p: 0.5,
            ..Default::default()
        };
        for _ in 0..50 {
            // exp(2-3)=0.37 < 0.5 of the top probability: only token 9 survives.
            assert_eq!(sample(&mut c, &min_p, &mut rng).unwrap(), 9);
        }
    }
}
