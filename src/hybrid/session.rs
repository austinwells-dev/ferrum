//! A conversation session: hybrid state plus the token history it
//! represents, recurrent-state snapshots for prefix reuse across requests,
//! and the sampling/generation loop.
#![forbid(unsafe_code)]
use super::{
    engine::{HybridModel, HybridState, Output, Penalties, Produced, RowOutput, RowsProduced},
    speculative::{Drafter, SpecStats},
};
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

/// What generation does after a token was sampled.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Flow {
    Continue,
    Stop,
    /// Feed these tokens after the sampled one (e.g. to close a reasoning
    /// block when a thinking budget runs out), then keep sampling.
    Inject(Vec<u32>),
}

/// Callbacks during `Session::generate`.
pub trait GenerationHooks {
    /// Before prefill: prompt length and how much of it the cache already holds.
    fn begin(&mut self, _prompt: usize, _reused: usize) {}
    /// After each prompt chunk: `processed` of `total` new prompt tokens are
    /// in the cache. Return false to cancel (the processed prefix stays
    /// cached, so a retry resumes where this left off).
    fn prefill(&mut self, _processed: usize, _total: usize) -> bool {
        true
    }
    /// A sampled, non-EOS token.
    fn token(&mut self, token: u32) -> Result<Flow>;
}

impl<F: FnMut(u32) -> Result<bool>> GenerationHooks for F {
    fn token(&mut self, token: u32) -> Result<Flow> {
        Ok(if self(token)? {
            Flow::Continue
        } else {
            Flow::Stop
        })
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
    /// The prefill callback cancelled before any token was generated.
    Cancelled,
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
    /// Speculative drafts proposed and accepted (0 without a drafter).
    pub drafted: usize,
    pub accepted: usize,
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
    drafter: Option<Box<dyn Drafter>>,
    /// Use the drafter for the next generations (it keeps ingesting either way).
    pub speculate: bool,
    /// The drafter must rewind to this length before its next use.
    drafter_rewind: Option<usize>,
    /// Acceptance statistics of the latest generation.
    pub spec_stats: SpecStats,
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
            drafter: None,
            speculate: false,
            drafter_rewind: None,
            spec_stats: SpecStats::default(),
        })
    }

    /// Attach a drafter (enabling the target's verify buffers) and clear the
    /// session. Generation speculates while `speculate` is true.
    pub fn set_drafter(
        &mut self,
        d: &MetalDevice,
        model: &HybridModel,
        drafter: Box<dyn Drafter>,
    ) -> Result<()> {
        model.enable_speculation(d, &drafter.geometry())?;
        self.drafter = Some(drafter);
        self.speculate = true;
        self.reset();
        Ok(())
    }

    pub fn drafter(&self) -> Option<&dyn Drafter> {
        self.drafter.as_deref()
    }

    /// Apply a pending drafter rewind.
    fn sync_drafter(&mut self, d: &MetalDevice, model: &HybridModel) -> Result<()> {
        if let (Some(len), Some(drafter)) = (self.drafter_rewind.take(), self.drafter.as_mut()) {
            drafter.rewind(d, model, len)?;
        }
        Ok(())
    }

    /// Forward `tokens` (no output) and let the drafter ingest them.
    fn extend(&mut self, d: &MetalDevice, model: &HybridModel, tokens: &[u32]) -> Result<()> {
        let pos = self.state.len();
        model.forward(d, &mut self.state, tokens, Output::None)?;
        self.tokens.extend_from_slice(tokens);
        if let Some(drafter) = self.drafter.as_mut() {
            drafter.ingest(d, model, pos, tokens)?;
        }
        Ok(())
    }

    /// Forward `tokens` with an output for the last one, and let the drafter
    /// ingest them.
    fn step(
        &mut self,
        d: &MetalDevice,
        model: &HybridModel,
        tokens: &[u32],
        output: Output,
    ) -> Result<Produced> {
        let pos = self.state.len();
        let produced = model.forward(d, &mut self.state, tokens, output)?;
        self.tokens.extend_from_slice(tokens);
        if let Some(drafter) = self.drafter.as_mut() {
            drafter.ingest(d, model, pos, tokens)?;
        }
        Ok(produced)
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
        if self.drafter.is_some() {
            self.drafter_rewind = Some(0);
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
                if self.drafter.is_some() {
                    self.drafter_rewind = Some(position);
                }
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
        if let Some(drafter) = self.drafter.as_mut() {
            drafter.snapshot(d, position)?;
        }
        Ok(())
    }

    /// Extend the session with `tokens` (sharing any cached prefix) without
    /// producing output, snapshotting at the end. Returns the reused length.
    pub fn prefill(
        &mut self,
        d: &MetalDevice,
        model: &HybridModel,
        tokens: &[u32],
    ) -> Result<usize> {
        if tokens.len() >= self.capacity() {
            return Err(Error::Parameter(format!(
                "{} tokens do not fit the {}-token context",
                tokens.len(),
                self.capacity()
            )));
        }
        let reused = self.reuse_prefix(d, tokens)?;
        self.sync_drafter(d, model)?;
        for piece in tokens[reused..].chunks(model.chunk()) {
            self.extend(d, model, piece)?;
        }
        self.snapshot(d)?;
        Ok(reused)
    }

    /// Generate up to `max_tokens` after `prompt`. `hooks` see prefill
    /// progress and each new (non-EOS) token. The prompt shares any prefix
    /// already in the session; divergence resumes from a snapshot.
    #[allow(clippy::too_many_arguments)]
    pub fn generate(
        &mut self,
        d: &MetalDevice,
        model: &HybridModel,
        prompt: &[u32],
        max_tokens: usize,
        params: &SamplingParams,
        eos: &[u32],
        hooks: &mut impl GenerationHooks,
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
        self.sync_drafter(d, model)?;
        self.spec_stats = SpecStats::default();
        hooks.begin(prompt.len(), reused);
        let mut rng = ChaCha8Rng::seed_from_u64(params.seed);
        let mut generated: Vec<u32> = Vec::new();
        let output = |generated: &[u32]| -> Output {
            if params.uses_candidates() {
                Output::Candidates(penalties(params, generated))
            } else {
                Output::Argmax
            }
        };
        // Prefill all but the final prompt token and snapshot there: the next
        // request always recomputes at least its final token, so a snapshot
        // at `len - 1` serves both a retry of this prompt and a continuation
        // of the conversation. The final token then yields the first output.
        let (body, tail) = prompt[reused..].split_at(prompt.len() - reused - 1);
        let mut processed = 0;
        for piece in body.chunks(model.chunk()) {
            self.extend(d, model, piece)?;
            processed += piece.len();
            if !hooks.prefill(processed, body.len() + 1) {
                return Ok(Completion {
                    tokens: Vec::new(),
                    stop: StopReason::Cancelled,
                    prompt_tokens: prompt.len(),
                    reused_tokens: reused,
                    prefill: start.elapsed(),
                    decode: Duration::ZERO,
                    drafted: 0,
                    accepted: 0,
                });
            }
        }
        self.snapshot(d)?;
        let produced = self.step(d, model, tail, output(&generated))?;
        hooks.prefill(body.len() + 1, body.len() + 1);
        let prefill = start.elapsed();
        let decode_start = Instant::now();
        let mut next = select(produced, params, &mut rng)?;
        let speculate = self.speculate && self.drafter.is_some();
        let stop = 'generation: loop {
            if eos.contains(&next) {
                break StopReason::Eos;
            }
            generated.push(next);
            let flow = hooks.token(next)?;
            if flow == Flow::Stop {
                break StopReason::Stopped;
            }
            if generated.len() >= max_tokens {
                break StopReason::Length;
            }
            let mut feed = vec![next];
            if let Flow::Inject(extra) = flow {
                feed.extend(extra);
            }
            if self.state.len() + feed.len() > self.capacity() {
                break StopReason::ContextFull;
            }
            // Speculative step: draft after `next`, verify, keep the prefix
            // the target agrees with (rejection sampling when sampling).
            let room = self.capacity() - self.state.len() - 1;
            let k = if speculate && feed.len() == 1 {
                let drafter = self.drafter.as_ref().expect("speculate implies a drafter");
                drafter
                    .max_drafts()
                    .min(max_tokens - generated.len())
                    .min(room)
            } else {
                0
            };
            let drafts = if k > 0 {
                let t0 = Instant::now();
                let pos = self.state.len();
                let drafter = self.drafter.as_mut().expect("speculate implies a drafter");
                let drafts = drafter.draft(d, model, pos, next, k)?;
                self.spec_stats.draft_seconds += t0.elapsed().as_secs_f64();
                drafts
            } else {
                Vec::new()
            };
            if drafts.is_empty() {
                let produced = self.step(d, model, &feed, output(&generated))?;
                generated.extend_from_slice(&feed[1..]);
                next = select(produced, params, &mut rng)?;
                continue;
            }
            let t0 = Instant::now();
            let mut rows = vec![next];
            rows.extend_from_slice(&drafts);
            let row_output = if params.uses_candidates() {
                let mut seen = generated.clone();
                let mut per_row = Vec::with_capacity(rows.len());
                for i in 0..rows.len() {
                    if i > 0 {
                        seen.push(drafts[i - 1]);
                    }
                    per_row.push(penalties(params, &seen));
                }
                RowOutput::Candidates(per_row)
            } else {
                RowOutput::Argmax
            };
            let verified = model.verify(d, &mut self.state, &rows, row_output)?;
            let (accepted, bonus) = accept(verified, &drafts, params, &mut rng)?;
            self.spec_stats.verify_seconds += t0.elapsed().as_secs_f64();
            self.spec_stats.record(drafts.len(), accepted);
            // Emit the accepted drafts one by one; the anchor row is always kept.
            let mut keep = 1;
            let mut stopped = None;
            let mut inject = None;
            for &token in &drafts[..accepted] {
                if eos.contains(&token) {
                    stopped = Some(StopReason::Eos);
                    break;
                }
                generated.push(token);
                match hooks.token(token)? {
                    Flow::Stop => {
                        stopped = Some(StopReason::Stopped);
                        break;
                    }
                    Flow::Inject(extra) => {
                        inject = Some((token, extra));
                        break;
                    }
                    Flow::Continue => {}
                }
                if generated.len() >= max_tokens {
                    stopped = Some(StopReason::Length);
                    break;
                }
                keep += 1;
            }
            let pos = self.state.len();
            model.commit(d, &mut self.state, keep)?;
            self.tokens.extend_from_slice(&rows[..keep]);
            if let Some(drafter) = self.drafter.as_mut() {
                drafter.ingest(d, model, pos, &rows[..keep])?;
            }
            if let Some(reason) = stopped {
                break 'generation reason;
            }
            if let Some((token, extra)) = inject {
                let mut feed = vec![token];
                feed.extend(extra);
                if self.state.len() + feed.len() > self.capacity() {
                    break StopReason::ContextFull;
                }
                let produced = self.step(d, model, &feed, output(&generated))?;
                generated.extend_from_slice(&feed[1..]);
                next = select(produced, params, &mut rng)?;
                continue;
            }
            next = bonus;
        };
        Ok(Completion {
            tokens: generated,
            stop,
            prompt_tokens: prompt.len(),
            reused_tokens: reused,
            prefill,
            decode: decode_start.elapsed(),
            drafted: self.spec_stats.drafted,
            accepted: self.spec_stats.accepted,
        })
    }
}

/// Decide how many drafts the target accepts and the token that follows
/// them. Greedy: drafts must equal the target's argmax. Sampling: draft i is
/// accepted with the target's probability of it (the drafter's proposal is
/// deterministic); on rejection the replacement is drawn from the target's
/// distribution with the draft removed, so the output has exactly the
/// target's distribution.
fn accept(
    verified: RowsProduced,
    drafts: &[u32],
    params: &SamplingParams,
    rng: &mut ChaCha8Rng,
) -> Result<(usize, u32)> {
    match verified {
        RowsProduced::Tokens(targets) => {
            let accepted = drafts
                .iter()
                .zip(&targets)
                .take_while(|(a, b)| a == b)
                .count();
            Ok((accepted, targets[accepted]))
        }
        RowsProduced::Logits(_) => Err(Error::Parameter("verify produced logits".into())),
        RowsProduced::Candidates(mut rows) => {
            for (i, &draft) in drafts.iter().enumerate() {
                let dist = distribution(&mut rows[i], params)?;
                let p = dist
                    .iter()
                    .find(|(t, _)| *t == draft)
                    .map_or(0., |(_, p)| *p);
                if params.temperature == 0. {
                    if p < 1. {
                        return Ok((i, dist[0].0));
                    }
                    continue;
                }
                if rng.random::<f64>() < p {
                    continue;
                }
                let rest: Vec<(u32, f64)> = dist.into_iter().filter(|(t, _)| *t != draft).collect();
                return Ok((i, draw(&rest, rng)));
            }
            let dist = distribution(&mut rows[drafts.len()], params)?;
            Ok((drafts.len(), draw(&dist, rng)))
        }
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
    let dist = distribution(candidates, params)?;
    if params.temperature == 0. {
        return Ok(dist[0].0);
    }
    Ok(draw(&dist, rng))
}

/// The sampling distribution over candidates (unnormalized probabilities,
/// most likely first). Greedy keeps only the top token with probability 1.
fn distribution(candidates: &mut [(u32, f32)], params: &SamplingParams) -> Result<Vec<(u32, f64)>> {
    if candidates.is_empty() {
        return Err(Error::Parameter("no finite logits to sample".into()));
    }
    candidates.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
    if params.temperature == 0. {
        return Ok(vec![(candidates[0].0, 1.)]);
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
    Ok(candidates[..keep]
        .iter()
        .zip(&probs)
        .map(|((t, _), p)| (*t, p / mass))
        .collect())
}

/// Draw from an (unnormalized) distribution.
fn draw(dist: &[(u32, f64)], rng: &mut impl Rng) -> u32 {
    let mass: f64 = dist.iter().map(|(_, p)| p).sum();
    let mut left = rng.random::<f64>() * mass;
    for &(t, p) in dist {
        if left < p {
            return t;
        }
        left -= p;
    }
    dist[dist.len() - 1].0
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Rejection sampling against a deterministic draft reproduces the
    /// target distribution (after temperature and top-k), whatever the draft.
    #[test]
    fn speculative_acceptance_preserves_the_distribution() {
        let logits = vec![(3, 2.0f32), (8, 1.5), (1, 1.0), (6, 0.2), (4, -1.0)];
        let params = SamplingParams {
            temperature: 0.8,
            top_k: 4,
            ..Default::default()
        };
        let mut expect = distribution(&mut logits.clone(), &params).unwrap();
        expect.sort_by_key(|&(t, _)| t);
        for draft in [3u32, 6, 4] {
            let mut rng = ChaCha8Rng::seed_from_u64(11);
            let mut counts = std::collections::BTreeMap::<u32, usize>::new();
            let trials = 200_000;
            for _ in 0..trials {
                // One draft: row 0 judges it, row 1 supplies the bonus token.
                let rows = RowsProduced::Candidates(vec![logits.clone(), logits.clone()]);
                let (accepted, bonus) = accept(rows, &[draft], &params, &mut rng).unwrap();
                let first = if accepted == 1 { draft } else { bonus };
                *counts.entry(first).or_default() += 1;
            }
            for &(t, p) in &expect {
                let got = *counts.get(&t).unwrap_or(&0) as f64 / trials as f64;
                assert!(
                    (got - p).abs() < 0.005,
                    "draft {draft}: token {t} sampled {got:.4}, expected {p:.4}"
                );
            }
            let outside: usize = counts
                .iter()
                .filter(|(t, _)| !expect.iter().any(|(e, _)| e == *t))
                .map(|(_, n)| n)
                .sum();
            assert_eq!(outside, 0, "draft {draft}: sampled outside top-k");
        }
    }

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
