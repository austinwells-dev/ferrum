//! A conversation session: hybrid state plus the token history it
//! represents, recurrent-state snapshots for prefix reuse across requests,
//! and the sampling/generation loop.
#![forbid(unsafe_code)]
use super::{
    engine::{
        HybridModel, HybridState, ImageSpan, Output, Penalties, Produced, RowOutput, RowsProduced,
    },
    kflash::{Budget, Calibration, MAX_ROWS, Record, SAVE_EVERY},
    speculative::{DraftTree, Drafter, SpecStats, accept_walk},
};
use crate::{DType, Error, MetalDevice, Result, Tensor};
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

/// Shrink a reusable prefix of `common` tokens so it never cuts through an
/// image or reuses a different one: pad tokens look alike whatever the
/// picture, so an image counts only when the cache (`cached`) holds the same
/// picture, at the same place, entirely inside the prefix.
fn reusable_before_images(common: usize, cached: &[ImageSpan], media: &[ImageSpan]) -> usize {
    for span in media {
        if span.start >= common {
            break;
        }
        let same = cached
            .iter()
            .any(|c| c.start == span.start && c.grid == span.grid && c.hash == span.hash);
        if !same || span.end() > common {
            return span.start;
        }
    }
    common
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
    /// Propose at most this many drafts per step (None: the drafter's own
    /// maximum). Lets a benchmark sweep depths without reloading the drafter.
    pub depth_cap: Option<usize>,
    /// The drafter must rewind to this length before its next use.
    drafter_rewind: Option<usize>,
    /// Draft trees of at most this many nodes instead of chains (see
    /// `set_tree_budget`).
    tree_budget: Option<usize>,
    /// KFlash's budget: each cycle verifies the tree rows it expects to pay
    /// (see `set_tree_rule`), within `tree_budget`.
    tree_rule: Option<Budget>,
    /// Acceptance statistics of the latest generation.
    pub spec_stats: SpecStats,
    /// Images of the next prompt, taken by the next `prefill`/`generate`.
    pending_media: Vec<ImageSpan>,
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
        let state = HybridState::new(d, &model.config, capacity, model.max_forward_rows())?;
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
            depth_cap: None,
            drafter_rewind: None,
            tree_budget: None,
            tree_rule: None,
            spec_stats: SpecStats::default(),
            pending_media: Vec::new(),
        })
    }

    /// Declare the images of the next prompt: each span covers pad-token
    /// positions whose embeddings the vision encoder produced. Spans must be
    /// ordered, disjoint and lie inside the prompt. Consumed by the next
    /// `prefill` or `generate`; without a call the prompt has no images.
    pub fn set_media(&mut self, media: Vec<ImageSpan>) {
        self.pending_media = media;
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

    /// Draft trees of up to `budget` nodes instead of chains (None: chains).
    /// Trees apply to every request: sampled rows are accepted when they
    /// equal their own position-keyed draw. Widens the target's verify buffers to `budget + 1` rows and
    /// clears the session.
    pub fn set_tree_budget(
        &mut self,
        d: &MetalDevice,
        model: &HybridModel,
        budget: Option<usize>,
    ) -> Result<()> {
        let drafter = self
            .drafter
            .as_ref()
            .ok_or_else(|| Error::Parameter("a draft tree needs a drafter".into()))?;
        let mut geometry = drafter.geometry();
        if let Some(budget) = budget {
            if budget == 0 {
                return Err(Error::Parameter(
                    "a draft tree needs at least one node".into(),
                ));
            }
            geometry.rows = geometry.rows.max(budget + 1);
        }
        model.enable_speculation(d, &geometry)?;
        self.tree_budget = budget;
        self.reset();
        Ok(())
    }

    pub fn tree_budget(&self) -> Option<usize> {
        self.tree_budget
    }

    /// Let KFlash's budget (`kflash::Budget`) choose each cycle's tree rows
    /// within the tree budget, and calibrate lattice trees, learning both
    /// from the cycles; `key` names its cache file (None: nothing kept).
    /// `enabled` false goes back to whole trees of the budget.
    pub fn set_tree_rule(&mut self, enabled: bool, key: Option<&str>) {
        if let Some(drafter) = self.drafter.as_mut() {
            drafter.set_calibration(&Calibration::default());
        }
        self.tree_rule =
            enabled.then(|| Budget::new(key, self.tree_budget.map_or(MAX_ROWS, |b| b + 1)));
    }

    /// Install `rule` as the tree budget's (None: whole trees of the budget)
    /// and return the previous one, so a caller can park its state.
    pub fn swap_tree_rule(&mut self, rule: Option<Budget>) -> Option<Budget> {
        if let Some(drafter) = self.drafter.as_mut() {
            drafter.set_calibration(&Calibration::default());
        }
        std::mem::replace(&mut self.tree_rule, rule)
    }

    /// The tree budget's state, for logs.
    pub fn tree_rule_summary(&self) -> Option<String> {
        self.tree_rule.as_ref().map(Budget::summary)
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
    fn reuse_prefix(
        &mut self,
        d: &MetalDevice,
        prompt: &[u32],
        media: &[ImageSpan],
    ) -> Result<usize> {
        if !self.state.is_valid() {
            self.reset();
        }
        for (i, span) in media.iter().enumerate() {
            let ordered = i == 0 || media[i - 1].end() <= span.start;
            if span.is_empty() || span.end() > prompt.len() || !ordered {
                return Err(Error::Parameter(
                    "image spans must be ordered, disjoint and inside the prompt".into(),
                ));
            }
        }
        let limit = prompt.len().saturating_sub(1);
        let mut common = self
            .tokens
            .iter()
            .zip(prompt)
            .take(limit)
            .take_while(|(a, b)| a == b)
            .count();
        common = reusable_before_images(common, self.state.media(), media);
        if common == self.tokens.len() {
            return Ok(common);
        }
        // Without recurrent state the K/V cache is valid at every prefix its
        // rings still hold, so resume exactly at the divergence. Most drafters
        // rewind only to their own snapshots, so they take the snapshot path.
        if common > 0
            && !self.state.has_recurrent()
            && self.drafter.as_ref().is_none_or(|t| t.rewinds_anywhere())
            && self.state.can_resume(common)
        {
            self.state.restore_recurrent(d, &[], common)?;
            self.tokens.truncate(common);
            if self.drafter.is_some() {
                self.drafter_rewind = Some(common);
            }
            for s in &mut self.snapshots {
                if s.position > common {
                    s.position = 0;
                    s.stamp = 0;
                }
            }
            return Ok(common);
        }
        // Diverged inside the cached sequence: resume from the deepest
        // snapshot at or before the divergence point.
        let state = &self.state;
        let best = self
            .snapshots
            .iter_mut()
            .filter(|s| s.position > 0 && s.position <= common && state.can_resume(s.position))
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
        let media = std::mem::take(&mut self.pending_media);
        let reused = self.reuse_prefix(d, tokens, &media)?;
        self.state.set_media(media);
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
        let media = std::mem::take(&mut self.pending_media);
        let reused = self.reuse_prefix(d, prompt, &media)?;
        self.state.set_media(media);
        self.sync_drafter(d, model)?;
        self.spec_stats = SpecStats::default();
        hooks.begin(prompt.len(), reused);
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
        let mut next = select(produced, params, self.state.len())?;
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
            // Speculative step: draft after `next`, verify, keep the path the
            // target agrees with: each row's own token (its argmax, or when
            // sampling its draw with the uniform of the position it fills, as
            // plain decoding draws it), so the output is plain decoding's.
            let room = self.capacity() - self.state.len() - 1;
            let k = if speculate && feed.len() == 1 {
                let drafter = self.drafter.as_ref().expect("speculate implies a drafter");
                drafter
                    .max_drafts()
                    .min(self.depth_cap.unwrap_or(usize::MAX))
                    .min(max_tokens - generated.len())
                    .min(room)
            } else {
                0
            };
            let tree_budget = self.tree_budget.filter(|_| k > 0).map(|b| b.min(room));
            let plan = self
                .tree_rule
                .as_ref()
                .filter(|_| tree_budget.is_some())
                .map(Budget::plan);
            let cycle = Instant::now();
            let mut expected = 0.;
            let drafts = if k > 0 {
                let t0 = Instant::now();
                let pos = self.state.len();
                let drafter = self.drafter.as_mut().expect("speculate implies a drafter");
                let mut drafts = match tree_budget {
                    Some(budget) => {
                        if let Some(plan) = &plan {
                            drafter.set_calibration(&plan.calibration);
                        }
                        drafter.draft_tree(d, model, pos, next, k, budget)?
                    }
                    None => DraftTree::chain(&drafter.draft(d, model, pos, next, k)?),
                };
                if let Some(plan) = plan.as_ref().filter(|_| !drafts.score.is_empty()) {
                    let (rows, e) = plan.rows(&drafts.score);
                    drafts.truncate(rows - 1);
                    expected = e;
                }
                self.spec_stats.draft_seconds += t0.elapsed().as_secs_f64();
                drafts
            } else {
                DraftTree::default()
            };
            if drafts.is_empty() {
                let produced = self.step(d, model, &feed, output(&generated))?;
                generated.extend_from_slice(&feed[1..]);
                next = select(produced, params, self.state.len())?;
                continue;
            }
            let t0 = Instant::now();
            let (rows, parents) = drafts.rows(next);
            // The accepted path (verify rows from the anchor's 0) and the
            // target's token after it.
            let anchor_pos = self.state.len();
            let depths = drafts.depths();
            let row_output = if params.uses_candidates() {
                // Each row's penalties see its own root path.
                let mut seen: Vec<Vec<u32>> = Vec::with_capacity(rows.len());
                seen.push(generated.clone());
                for (i, &token) in drafts.tokens.iter().enumerate() {
                    let mut path = seen[drafts.parents[i].map_or(0, |p| p + 1)].clone();
                    path.push(token);
                    seen.push(path);
                }
                RowOutput::Candidates(seen.iter().map(|s| penalties(params, s)).collect())
            } else {
                RowOutput::Argmax
            };
            let verified = if tree_budget.is_some() {
                model.verify_tree(d, &mut self.state, &rows, &parents, row_output)?
            } else {
                model.verify(d, &mut self.state, &rows, row_output)?
            };
            let targets = match verified {
                RowsProduced::Tokens(targets) => targets,
                RowsProduced::Candidates(mut per_row) => per_row
                    .iter_mut()
                    .enumerate()
                    .map(|(r, c)| {
                        let depth = if r == 0 { 0 } else { depths[r - 1] };
                        sample(c, params, uniform(params.seed, anchor_pos + depth + 1))
                    })
                    .collect::<Result<Vec<_>>>()?,
                RowsProduced::Logits(_) => {
                    return Err(Error::Parameter("verify produced logits".into()));
                }
            };
            // The accepted path (verify rows from the anchor's 0) and the
            // target's token after it.
            let (path, bonus) = accept_walk(&drafts, &targets);
            self.spec_stats.verify_seconds += t0.elapsed().as_secs_f64();
            self.spec_stats
                .record(drafts.len(), drafts.depth(), path.len() - 1);
            if let Some(rule) = self.tree_rule.as_mut().filter(|_| expected > 0.) {
                // Calibration records: every node whose parent is on the path.
                let mut on_path = vec![false; drafts.len() + 1];
                for &row in &path {
                    on_path[row] = true;
                }
                let records: Vec<Record> = (0..drafts.len())
                    .filter(|&i| on_path[drafts.parents[i].map_or(0, |p| p + 1)])
                    .map(|i| Record {
                        depth: depths[i],
                        lq: drafts.lq[i],
                        accepted: on_path[i + 1],
                    })
                    .collect();
                let ms = cycle.elapsed().as_secs_f64() * 1e3;
                rule.observe(rows.len(), expected, path.len() - 1, ms, &records);
            }
            // Emit the accepted drafts one by one; the anchor row is always kept.
            let mut keep = 1;
            let mut stopped = None;
            let mut inject = None;
            for &row in &path[1..] {
                let token = rows[row];
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
            let kept: Vec<u32> = path[..keep].iter().map(|&r| rows[r]).collect();
            model.commit_path(d, &mut self.state, &path[..keep])?;
            self.tokens.extend_from_slice(&kept);
            if let Some(drafter) = self.drafter.as_mut() {
                drafter.ingest(d, model, pos, &kept)?;
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
                next = select(produced, params, self.state.len())?;
                continue;
            }
            next = bonus;
        };
        if let Some(rule) = self.tree_rule.as_mut() {
            rule.save(SAVE_EVERY);
        }
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

/// The uniform that samples the token at sequence position `pos`: one per
/// position (Kapitsa's shared sampling stream), so a draft row and plain
/// decoding filling the same position draw with the same number, and
/// accepting a draft exactly when it equals its row's draw emits plain
/// decoding's tokens. For a deterministic draft that acceptance has the
/// draft's probability under the target, as rejection sampling does.
fn uniform(seed: u64, pos: usize) -> f64 {
    // SplitMix64 over the seed and position.
    let mut z = seed ^ (pos as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
    for _ in 0..2 {
        z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
    }
    (z >> 11) as f64 / (1u64 << 53) as f64
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

/// How far the draw for sequence position `pos` falls from the nearest edge
/// between two tokens of the sampling distribution over `logits`, as a
/// fraction of its mass: a draw this close to an edge flips with a logit
/// change of about that relative size (diagnostics for sampled outputs that
/// differ between kernels).
pub fn draw_margin(logits: &[f32], params: &SamplingParams, pos: usize) -> Result<f64> {
    let mut candidates: Vec<(u32, f32)> = logits
        .iter()
        .enumerate()
        .filter(|(_, l)| l.is_finite())
        .map(|(i, &l)| (i as u32, l))
        .collect();
    let dist = distribution(&mut candidates, params)?;
    let mass: f64 = dist.iter().map(|(_, p)| p).sum();
    let u = uniform(params.seed, pos);
    let mut edge = 0.;
    let mut nearest = f64::INFINITY;
    for (_, p) in &dist {
        nearest = nearest.min((u - edge / mass).abs());
        edge += p;
    }
    Ok(nearest.min((u - 1.).abs()))
}

/// The token at sequence position `pos` from a forward's output.
fn select(produced: Produced, params: &SamplingParams, pos: usize) -> Result<u32> {
    match produced {
        Produced::Token(t) => Ok(t),
        Produced::Candidates(mut c) => sample(&mut c, params, uniform(params.seed, pos)),
        _ => Err(Error::Parameter("forward produced no token".into())),
    }
}

/// Temperature, top-k, min-p then top-p over GPU candidates, drawn with the
/// uniform `u` in [0, 1).
pub(crate) fn sample(
    candidates: &mut [(u32, f32)],
    params: &SamplingParams,
    u: f64,
) -> Result<u32> {
    let dist = distribution(candidates, params)?;
    if params.temperature == 0. {
        return Ok(dist[0].0);
    }
    Ok(draw(&dist, u))
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
fn draw(dist: &[(u32, f64)], u: f64) -> u32 {
    let mass: f64 = dist.iter().map(|(_, p)| p).sum();
    let mut left = u * mass;
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

    /// A toy target: candidate logits after a history depend on its last
    /// token.
    fn toy_logits(last: u32) -> Vec<(u32, f32)> {
        (0..6u32)
            .map(|t| (t, ((t * 7 + last * 3) % 11) as f32 * 0.3))
            .collect()
    }

    /// Accepting tree rows that equal their own position-keyed draws emits
    /// plain sampling's tokens, whatever the tree.
    #[test]
    fn sampled_trees_emit_plain_sampling() {
        let params = SamplingParams {
            temperature: 0.8,
            top_k: 4,
            seed: 3,
            ..Default::default()
        };
        let plain = |start: u32, pos: usize, n: usize| -> Vec<u32> {
            let mut out = Vec::new();
            let mut last = start;
            for i in 0..n {
                last = sample(
                    &mut toy_logits(last),
                    &params,
                    uniform(params.seed, pos + i),
                )
                .unwrap();
                out.push(last);
            }
            out
        };
        let mut x = 99u64;
        let mut rand = |n: u32| {
            x = x
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (x >> 33) as u32 % n
        };
        let mut accepted_any = false;
        for trial in 0..500 {
            let (anchor, pos) = (trial % 6, 40 + trial as usize);
            // A random tree of 12 nodes over the 6 tokens.
            let mut tree = DraftTree::default();
            for i in 0..12 {
                let parent = (i > 0 && rand(3) > 0).then(|| rand(i) as usize);
                tree.push(rand(6), parent);
            }
            let (rows, _) = tree.rows(anchor);
            let depths = tree.depths();
            let targets: Vec<u32> = rows
                .iter()
                .enumerate()
                .map(|(r, &token)| {
                    let depth = if r == 0 { 0 } else { depths[r - 1] };
                    sample(
                        &mut toy_logits(token),
                        &params,
                        uniform(params.seed, pos + depth + 1),
                    )
                    .unwrap()
                })
                .collect();
            let (path, bonus) = accept_walk(&tree, &targets);
            let mut emitted: Vec<u32> = path[1..].iter().map(|&r| rows[r]).collect();
            emitted.push(bonus);
            accepted_any |= emitted.len() > 1;
            assert_eq!(
                emitted,
                plain(anchor, pos + 1, emitted.len()),
                "trial {trial}"
            );
        }
        assert!(accepted_any);
    }

    /// Position-keyed draws follow the sampling distribution.
    #[test]
    fn position_draws_follow_the_distribution() {
        let logits = vec![(3, 2.0f32), (8, 1.5), (1, 1.0), (6, 0.2), (4, -1.0)];
        let params = SamplingParams {
            temperature: 0.8,
            top_k: 4,
            seed: 11,
            ..Default::default()
        };
        let mut expect = distribution(&mut logits.clone(), &params).unwrap();
        expect.sort_by_key(|&(t, _)| t);
        let trials = 200_000;
        let mut counts = std::collections::BTreeMap::<u32, usize>::new();
        for pos in 0..trials {
            let t = sample(&mut logits.clone(), &params, uniform(params.seed, pos)).unwrap();
            *counts.entry(t).or_default() += 1;
        }
        for &(t, p) in &expect {
            let got = *counts.get(&t).unwrap_or(&0) as f64 / trials as f64;
            assert!(
                (got - p).abs() < 0.005,
                "token {t}: {got:.4}, expected {p:.4}"
            );
        }
        assert!(counts.keys().all(|t| expect.iter().any(|e| e.0 == *t)));
    }

    #[test]
    fn sampling_respects_top_k_min_p_and_greedy() {
        let mut c = vec![(5, 1.0), (9, 3.0), (2, 2.0)];
        let greedy = SamplingParams::default();
        assert_eq!(sample(&mut c, &greedy, 0.5).unwrap(), 9);
        let top1 = SamplingParams {
            temperature: 1.,
            top_k: 1,
            ..Default::default()
        };
        for i in 0..20 {
            assert_eq!(sample(&mut c, &top1, uniform(7, i)).unwrap(), 9);
        }
        let min_p = SamplingParams {
            temperature: 1.,
            min_p: 0.5,
            ..Default::default()
        };
        for i in 0..50 {
            // exp(2-3)=0.37 < 0.5 of the top probability: only token 9 survives.
            assert_eq!(sample(&mut c, &min_p, uniform(7, i)).unwrap(), 9);
        }
    }
}

#[cfg(test)]
mod media_tests {
    use super::*;
    use crate::{DType, MetalDevice};
    use std::rc::Rc;

    fn span(d: &MetalDevice, start: usize, grid: (usize, usize), hash: u64) -> ImageSpan {
        ImageSpan {
            start,
            grid,
            embedding: Rc::new(Tensor::zeros(d, [grid.0 * grid.1, 4], DType::F32).unwrap()),
            hash,
        }
    }

    #[test]
    fn prefix_stops_before_a_changed_or_cut_image() {
        let d = MetalDevice::new().unwrap();
        let cached = [span(&d, 4, (2, 2), 7)];
        // No images in the new prompt: the whole prefix is reusable.
        assert_eq!(reusable_before_images(20, &cached, &[]), 20);
        // The same image, fully inside the prefix.
        assert_eq!(
            reusable_before_images(20, &cached, &[span(&d, 4, (2, 2), 7)]),
            20
        );
        // A different picture with the same size: reuse stops where it starts.
        assert_eq!(
            reusable_before_images(20, &cached, &[span(&d, 4, (2, 2), 8)]),
            4
        );
        // The prefix ends inside the image: it cannot be partly reused.
        assert_eq!(
            reusable_before_images(6, &cached, &[span(&d, 4, (2, 2), 7)]),
            4
        );
        // An image that starts after the prefix is irrelevant.
        assert_eq!(
            reusable_before_images(4, &cached, &[span(&d, 4, (2, 2), 9)]),
            4
        );
        // Only the first mismatching image matters.
        let new = [span(&d, 4, (2, 2), 7), span(&d, 12, (1, 3), 5)];
        assert_eq!(reusable_before_images(30, &cached, &new), 12);
    }
}
