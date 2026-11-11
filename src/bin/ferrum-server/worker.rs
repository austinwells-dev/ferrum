//! The model thread: one `Runtime` with `--parallel` sessions (slots), a FIFO
//! job queue, per-request event channels, cancellation and live status for
//! the HTTP threads.
//!
//! The model's types are single-threaded, so slots share the thread: every
//! running request takes one step in turn (a prompt chunk, or one decode
//! step), and their streams interleave. Each request goes to the slot whose
//! cache holds the most of its prompt, so concurrent conversations keep their
//! own caches.
use crate::{debug, error, info, log::n, warn};
use ferrum::{
    Error,
    hybrid::{
        plan::PlanOptions,
        runtime::{ChatHooks, ChatRequest, ChatResult, Delta, Done, Runtime, SpecOptions, Turn},
        session::{Completion, SamplingParams, StopReason},
    },
};
use serde::Serialize;
use std::{
    collections::VecDeque,
    path::PathBuf,
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, AtomicUsize, Ordering},
        mpsc::{Receiver, Sender, TryRecvError, channel},
    },
    time::{Duration, Instant},
};

/// Most `--parallel` slots.
pub const MAX_PARALLEL: usize = 8;

pub enum Work {
    Chat(Box<ChatRequest>),
    Complete {
        prompt: String,
        max_tokens: usize,
        sampling: SamplingParams,
        stop: Vec<String>,
    },
}

pub enum Event {
    /// Prompt length and tokens reused from the cached conversation.
    Started {
        prompt: usize,
        reused: usize,
    },
    Progress {
        processed: usize,
        total: usize,
    },
    Delta(Delta),
    Chat(Box<ChatResult>),
    Completed {
        completion: Completion,
        text: String,
        stopped: bool,
    },
    Failed {
        status: u16,
        message: String,
    },
}

/// Vision projector settings for the model thread.
pub struct Vision {
    pub mmproj: Option<PathBuf>,
    pub max_tokens: usize,
    pub min_tokens: usize,
}

/// Which slot a request asks for: llama-server's `id_slot`, or a cache key
/// (OpenAI's `prompt_cache_key`) whose last slot it should return to.
#[derive(Debug, Clone, Default)]
pub struct Route {
    pub slot: Option<usize>,
    pub key: Option<String>,
}

pub struct Job {
    pub id: u64,
    pub work: Work,
    pub route: Route,
    pub events: Sender<Event>,
    pub cancel: Arc<AtomicBool>,
    pub enqueued: Instant,
}

#[derive(Debug, Clone, Default, Serialize)]
pub struct SlotStatus {
    pub id: usize,
    /// The request running here.
    pub request: Option<u64>,
    /// idle, prefill or generating.
    pub phase: &'static str,
    /// Tokens this slot's cache holds.
    pub tokens: usize,
    /// The cache key of the latest request here.
    pub key: Option<String>,
}

#[derive(Debug, Clone, Default, Serialize)]
pub struct Status {
    pub state: &'static str,
    pub load_error: Option<String>,
    pub model: String,
    /// Context of each slot.
    pub context: usize,
    /// Requests may carry images (a vision projector is loaded).
    pub vision: bool,
    /// Requests may carry audio (a speech projector is loaded).
    pub audio: bool,
    pub slots: Vec<SlotStatus>,
    pub requests: u64,
    pub cancelled: u64,
    pub failed: u64,
    pub prompt_tokens: u64,
    pub cached_tokens: u64,
    pub prefilled_tokens: u64,
    pub generated_tokens: u64,
    pub prefill_seconds: f64,
    pub decode_seconds: f64,
    pub last_prefill_tps: f64,
    pub last_decode_tps: f64,
    pub memory_budget_bytes: usize,
    pub memory_predicted_bytes: usize,
    /// Speculative drafter name (empty without one).
    pub drafter: String,
    pub draft_tokens: u64,
    pub draft_accepted_tokens: u64,
}

pub struct Shared {
    pub status: Mutex<Status>,
    pub queued: AtomicUsize,
}

impl Status {
    /// Slots running a request.
    pub fn busy(&self) -> usize {
        self.slots.iter().filter(|s| s.request.is_some()).count()
    }
    /// Tokens held by every slot's cache.
    pub fn cached(&self) -> usize {
        self.slots.iter().map(|s| s.tokens).sum()
    }
}

impl Shared {
    pub fn status(&self) -> Status {
        self.status.lock().map(|s| s.clone()).unwrap_or_default()
    }
    fn update(&self, f: impl FnOnce(&mut Status)) {
        if let Ok(mut s) = self.status.lock() {
            f(&mut s);
        }
    }
}

pub fn spawn(
    path: PathBuf,
    options: PlanOptions,
    spec: Option<SpecOptions>,
    alias: Option<String>,
    vision: Vision,
    shared: Arc<Shared>,
) -> Sender<Job> {
    let (tx, rx) = channel::<Job>();
    std::thread::Builder::new()
        .name("model".into())
        .spawn(move || run(path, options, spec, alias, vision, shared, rx))
        .expect("spawn model thread");
    tx
}

fn run(
    path: PathBuf,
    options: PlanOptions,
    spec: Option<SpecOptions>,
    alias: Option<String>,
    vision: Vision,
    shared: Arc<Shared>,
    jobs: Receiver<Job>,
) {
    let start = Instant::now();
    let mut rt = match Runtime::load_multimodal(
        &path,
        options,
        spec.as_ref(),
        vision.mmproj.as_deref(),
        vision.max_tokens,
    ) {
        Ok(rt) => rt,
        Err(e) => {
            error!(None, "failed to load {}: {e}", path.display());
            shared.update(|s| {
                s.state = "error";
                s.load_error = Some(e.to_string());
            });
            // Fail every request instead of hanging.
            for job in jobs {
                let _ = job.events.send(Event::Failed {
                    status: 503,
                    message: format!("model failed to load: {e}"),
                });
            }
            return;
        }
    };
    if let Some(alias) = alias {
        rt.model_name = alias;
    }
    let plan = rt.loaded.plan.clone();
    let c = &rt.loaded.model.config;
    let gib = |b: usize| b as f64 / (1u64 << 30) as f64;
    info!(
        None,
        "loaded {} ({:?}, {} layers, {} attention) in {:.1} s",
        rt.model_name,
        c.variant,
        c.layers,
        c.attention_layer_count(),
        start.elapsed().as_secs_f64()
    );
    info!(
        None,
        "context {} tokens (largest that fits: {}, trained {}); memory {:.2} GiB of {:.2} GiB budget: weights {:.2}, KV {:.2} ({}, {:.1} KiB/token), recurrent+snapshots {:.2}, scratch {:.2}",
        n(plan.context),
        n(plan.max_context),
        n(plan.trained_context),
        gib(plan.total()),
        gib(plan.budget),
        gib(plan.weights),
        gib(plan.parallel * plan.kv(plan.context)),
        c.kv_cache,
        plan.kv_per_token / 1024.,
        gib(plan.parallel * (plan.recurrent + plan.snapshots)),
        gib(plan.scratch)
    );
    let drafter = rt
        .session
        .drafter()
        .map(|d| (d.name().to_owned(), d.max_drafts()));
    if let Some((name, drafts)) = &drafter {
        info!(
            None,
            "speculative decoding: {name}, {drafts} drafts per step; drafter memory {:.2} GiB (predicted {:.2})",
            gib(rt.draft_loaded_bytes),
            gib(plan.draft_total(plan.context))
        );
    }
    if let Some(v) = rt.vision.as_mut() {
        v.model.min_tokens = vision.min_tokens;
        info!(
            None,
            "vision: {} ({:.2} GiB; images take {}..{} tokens each)",
            v.path.display(),
            gib(v.model.weight_bytes()),
            v.model.min_tokens.min(v.model.max_tokens),
            v.model.max_tokens
        );
    }
    if let Some(a) = rt.audio.as_ref() {
        info!(
            None,
            "audio: {} ({:.2} GiB; WAV clips up to {} s)",
            a.path.display(),
            gib(a.model.weight_bytes()),
            ferrum::vision::MAX_AUDIO_SECONDS
        );
    }
    let warm = Instant::now();
    match rt.warmup() {
        Ok(()) => info!(None, "warmed up in {:.1} s", warm.elapsed().as_secs_f64()),
        Err(e) => warn!(None, "warmup failed: {e}"),
    }
    let d = &rt.default_sampling;
    info!(
        None,
        "default sampling: temperature {} top-p {} top-k {} min-p {}",
        d.temperature,
        d.top_p,
        d.top_k,
        d.min_p
    );
    let parallel = rt.parallel();
    if parallel > 1 {
        info!(
            None,
            "{parallel} slots of {} tokens each{}",
            n(plan.context),
            if drafter.is_some() {
                "; speculative decoding runs on slot 0 only"
            } else {
                ""
            }
        );
    }
    shared.update(|s| {
        s.state = "ready";
        s.model = rt.model_name.clone();
        s.context = plan.context;
        s.vision = rt.supports_vision();
        s.audio = rt.supports_audio();
        s.memory_budget_bytes = plan.budget;
        s.memory_predicted_bytes = plan.total();
        s.drafter = drafter.map(|(name, _)| name).unwrap_or_default();
        s.slots = (0..parallel)
            .map(|id| SlotStatus {
                id,
                phase: "idle",
                ..SlotStatus::default()
            })
            .collect();
    });
    Scheduler {
        rt,
        shared,
        slots: (0..parallel).map(|_| Slot::default()).collect(),
        queue: VecDeque::new(),
        clock: 0,
    }
    .run(jobs);
}

/// A request running on a slot.
struct Active {
    job: Job,
    turn: Turn,
    timing: Timing,
}

#[derive(Default)]
struct Slot {
    active: Option<Active>,
    /// When it last took a request (0: never), for least-recently-used reuse.
    used: u64,
    key: Option<String>,
}

struct Scheduler {
    rt: Runtime,
    shared: Arc<Shared>,
    slots: Vec<Slot>,
    /// Jobs waiting for a slot, oldest first.
    queue: VecDeque<Job>,
    clock: u64,
}

impl Scheduler {
    fn run(mut self, jobs: Receiver<Job>) {
        loop {
            // Wait only when nothing runs; otherwise just collect arrivals.
            if self.queue.is_empty() && self.slots.iter().all(|s| s.active.is_none()) {
                match jobs.recv() {
                    Ok(job) => self.queue.push_back(job),
                    Err(_) => return,
                }
            }
            loop {
                match jobs.try_recv() {
                    Ok(job) => self.queue.push_back(job),
                    Err(TryRecvError::Empty) => break,
                    Err(TryRecvError::Disconnected) if self.queue.is_empty() => {
                        if self.slots.iter().all(|s| s.active.is_none()) {
                            return;
                        }
                        break;
                    }
                    Err(TryRecvError::Disconnected) => break,
                }
            }
            self.admit();
            for i in 0..self.slots.len() {
                if self.slots[i].active.is_some() {
                    self.step(i);
                }
            }
        }
    }

    /// Start queued jobs on free slots, oldest first; a job pinned to a busy
    /// slot waits without holding up the others.
    fn admit(&mut self) {
        let mut waiting = VecDeque::new();
        while let Some(job) = self.queue.pop_front() {
            if job.cancel.load(Ordering::SeqCst) {
                self.shared.queued.fetch_sub(1, Ordering::SeqCst);
                info!(Some(job.id), "client left while queued; skipped");
                self.shared.update(|s| s.cancelled += 1);
                continue;
            }
            if let Some(slot) = job.route.slot
                && slot >= self.slots.len()
            {
                self.shared.queued.fetch_sub(1, Ordering::SeqCst);
                let message = format!(
                    "id_slot {slot} does not exist: the server has {} slot(s) (--parallel)",
                    self.slots.len()
                );
                warn!(Some(job.id), "rejected: {message}");
                self.shared.update(|s| s.failed += 1);
                let _ = job.events.send(Event::Failed {
                    status: 400,
                    message,
                });
                continue;
            }
            let free = self.slots.iter().any(|s| s.active.is_none());
            let pinned_busy = job
                .route
                .slot
                .is_some_and(|i| self.slots[i].active.is_some());
            if !free || pinned_busy {
                waiting.push_back(job);
                continue;
            }
            self.shared.queued.fetch_sub(1, Ordering::SeqCst);
            self.start(job);
        }
        self.queue = waiting;
    }

    /// The free slot for a prompt: the one asked for; else the one whose
    /// cache holds most of it, if that is at least half of what the slot
    /// holds (so a shared system prompt does not evict another
    /// conversation); else the slot the request's cache key last used; else
    /// the least recently used (an empty one first).
    fn choose(&self, route: &Route, shared: impl Fn(usize) -> usize) -> usize {
        if let Some(slot) = route.slot {
            return slot;
        }
        let free = || (0..self.slots.len()).filter(|&i| self.slots[i].active.is_none());
        let best = free()
            .map(|i| (shared(i), i))
            .filter(|&(common, i)| common > 0 && 2 * common >= self.rt.slot(i).len())
            .max_by_key(|&(common, i)| (common, std::cmp::Reverse(i)));
        if let Some((_, i)) = best {
            return i;
        }
        if let Some(key) = &route.key
            && let Some(i) = free().find(|&i| self.slots[i].key.as_ref() == Some(key))
        {
            return i;
        }
        free()
            .min_by_key(|&i| (self.slots[i].used, i))
            .expect("a free slot")
    }

    fn start(&mut self, job: Job) {
        let waited = job.enqueued.elapsed();
        if waited > Duration::from_millis(250) {
            info!(
                Some(job.id),
                "started after {:.1} s in queue",
                waited.as_secs_f64()
            );
        }
        let mut timing = Timing::new();
        let shared = self.shared.clone();
        let started = guarded(|| match &job.work {
            Work::Chat(request) => {
                let prompt = self.rt.chat_prompt(request)?;
                let slot = self.choose_for(&job.route, &prompt.tokens, &prompt.spans);
                let mut hooks = Hooks::new(&job, &shared, &mut timing, slot);
                let turn = self.rt.start_chat(slot, request, prompt, &mut hooks)?;
                Ok((slot, turn))
            }
            Work::Complete {
                prompt,
                max_tokens,
                sampling,
                stop,
            } => {
                let tokens = self.rt.loaded.tokenizer.encode(prompt)?;
                let slot = self.choose_for(&job.route, &tokens, &[]);
                let mut hooks = Hooks::new(&job, &shared, &mut timing, slot);
                let turn = self.rt.start_complete(
                    slot,
                    prompt,
                    *max_tokens,
                    sampling,
                    stop,
                    &mut hooks,
                )?;
                Ok((slot, turn))
            }
        });
        if let Some(v) = self.rt.vision.as_ref()
            && !v.last_encode.is_zero()
        {
            info!(
                Some(job.id),
                "encoded images in {:.2} s",
                v.last_encode.as_secs_f64()
            );
        }
        if let Some(a) = self.rt.audio.as_ref()
            && !a.last_encode.is_zero()
        {
            info!(
                Some(job.id),
                "encoded audio in {:.2} s",
                a.last_encode.as_secs_f64()
            );
        }
        match started {
            Ok((slot, turn)) => {
                self.clock += 1;
                let s = &mut self.slots[slot];
                s.used = self.clock;
                s.key = job.route.key.clone();
                let key = s.key.clone();
                self.shared.update(|st| {
                    if let Some(status) = st.slots.get_mut(slot) {
                        status.request = Some(job.id);
                        status.phase = "prefill";
                        status.key = key;
                    }
                });
                self.slots[slot].active = Some(Active { job, turn, timing });
            }
            Err(failure) => {
                // A request that failed before its session moved leaves every
                // cache as it was; a panic may have, so it resets them all.
                if failure.panicked {
                    self.reset_all();
                }
                self.shared.update(|s| s.failed += 1);
                let _ = job.events.send(failure.event(job.id));
            }
        }
    }

    fn choose_for(
        &self,
        route: &Route,
        tokens: &[u32],
        spans: &[ferrum::hybrid::engine::ImageSpan],
    ) -> usize {
        let slot = self.choose(route, |i| self.rt.slot(i).shared_prefix(tokens, spans));
        if self.slots.len() > 1 {
            debug!(
                None,
                "slot {slot}: shares {} of {} prompt tokens",
                n(self.rt.slot(slot).shared_prefix(tokens, spans)),
                n(tokens.len())
            );
        }
        slot
    }

    /// One step of the request on `slot`; finishes it when it is done.
    fn step(&mut self, slot: usize) {
        let Some(mut active) = self.slots[slot].active.take() else {
            return;
        };
        let rt = &mut self.rt;
        let shared = &*self.shared;
        let outcome = guarded(|| {
            let mut hooks = Hooks::new(&active.job, shared, &mut active.timing, slot);
            rt.advance(slot, &mut active.turn, &mut hooks)
        });
        let event = match outcome {
            Ok(None) => {
                self.slots[slot].active = Some(active);
                return;
            }
            Ok(Some(done)) => {
                let event = match done {
                    Done::Chat(result) => Event::Chat(result),
                    Done::Completion(completion, text, stopped) => Event::Completed {
                        completion,
                        text,
                        stopped,
                    },
                };
                let completion = match &event {
                    Event::Chat(r) => &r.completion,
                    Event::Completed { completion, .. } => completion,
                    _ => unreachable!(),
                };
                record(
                    &self.shared,
                    &active.job,
                    slot,
                    completion,
                    &event,
                    self.rt.slot(slot).len(),
                );
                if let Some(summary) = self.rt.slot(slot).tree_rule_summary() {
                    debug!(Some(active.job.id), "{summary}");
                }
                event
            }
            Err(failure) => {
                if failure.panicked {
                    self.rt.slot_mut(slot).reset();
                }
                self.shared.update(|s| s.failed += 1);
                failure.event(active.job.id)
            }
        };
        let _ = active.job.events.send(event);
        let tokens = self.rt.slot(slot).len();
        self.shared.update(|s| {
            if let Some(status) = s.slots.get_mut(slot) {
                status.request = None;
                status.phase = "idle";
                status.tokens = tokens;
            }
        });
    }

    fn reset_all(&mut self) {
        for i in 0..self.slots.len() {
            if self.slots[i].active.is_none() {
                self.rt.slot_mut(i).reset();
            }
        }
    }
}

/// A request that failed: an error (400 for the client's, 500 otherwise) or
/// a panic.
struct Failure {
    status: u16,
    message: String,
    panicked: bool,
}

impl Failure {
    fn event(self, id: u64) -> Event {
        if self.panicked {
            error!(
                Some(id),
                "internal panic: {}; conversation cache reset", self.message
            );
        } else if self.status == 400 {
            warn!(Some(id), "rejected: {}", self.message);
        } else {
            error!(Some(id), "failed: {}", self.message);
        }
        Event::Failed {
            status: self.status,
            message: if self.panicked {
                format!("internal error: {}", self.message)
            } else {
                self.message
            },
        }
    }
}

/// Run `f`, turning errors and panics into a `Failure`.
fn guarded<T>(f: impl FnOnce() -> ferrum::Result<T>) -> Result<T, Failure> {
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(f)) {
        Ok(Ok(v)) => Ok(v),
        Ok(Err(e)) => Err(Failure {
            status: if matches!(e, Error::Parameter(_) | Error::Tokenizer(_)) {
                400
            } else {
                500
            },
            message: e.to_string(),
            panicked: false,
        }),
        Err(panic) => Err(Failure {
            status: 500,
            message: panic
                .downcast_ref::<String>()
                .cloned()
                .or_else(|| panic.downcast_ref::<&str>().map(|s| s.to_string()))
                .unwrap_or_else(|| "unknown panic".into()),
            panicked: true,
        }),
    }
}

fn record(
    shared: &Shared,
    job: &Job,
    slot: usize,
    c: &Completion,
    event: &Event,
    session_len: usize,
) {
    // A prefill cancelled part-way processed only what the session now holds.
    let prefilled = if c.stop == StopReason::Cancelled && c.tokens.is_empty() {
        session_len.saturating_sub(c.reused_tokens)
    } else {
        c.prompt_tokens - c.reused_tokens
    };
    let prefill_tps = prefilled as f64 / c.prefill.as_secs_f64().max(1e-9);
    let decode_tps = c.tokens.len() as f64 / c.decode.as_secs_f64().max(1e-9);
    let cancelled = c.stop == StopReason::Cancelled || job.cancel.load(Ordering::SeqCst);
    shared.update(|s| {
        s.requests += 1;
        s.cancelled += u64::from(cancelled);
        s.prompt_tokens += c.prompt_tokens as u64;
        s.cached_tokens += c.reused_tokens as u64;
        s.prefilled_tokens += prefilled as u64;
        s.generated_tokens += c.tokens.len() as u64;
        s.draft_tokens += c.drafted as u64;
        s.draft_accepted_tokens += c.accepted as u64;
        s.prefill_seconds += c.prefill.as_secs_f64();
        s.decode_seconds += c.decode.as_secs_f64();
        if prefilled > 0 {
            s.last_prefill_tps = prefill_tps;
        }
        if !c.tokens.is_empty() {
            s.last_decode_tps = decode_tps;
        }
    });
    let mut detail = String::new();
    if let Event::Chat(r) = event {
        if r.reasoning_tokens > 0 {
            detail.push_str(&format!(
                ", reasoning {} tok{}",
                n(r.reasoning_tokens),
                if r.budget_exhausted {
                    " (budget reached)"
                } else {
                    ""
                }
            ));
        }
        if !r.output.tool_calls.is_empty() {
            let names: Vec<&str> = r
                .output
                .tool_calls
                .iter()
                .map(|t| t.name.as_str())
                .collect();
            detail.push_str(&format!(", tool calls: {}", names.join(", ")));
        }
        if r.stopped_by_string {
            detail.push_str(", stop string");
        }
        debug!(Some(job.id), "output: {:?}", truncate(&r.text, 2000));
    }
    let stop = if cancelled && c.stop != StopReason::Cancelled {
        "client disconnected".to_owned()
    } else {
        format!("{:?}", c.stop).to_lowercase()
    };
    let slots = shared.status().slots.len();
    if slots > 1 {
        detail.push_str(&format!(", slot {slot}"));
    }
    info!(
        Some(job.id),
        "done ({stop}) in {:.2} s: prompt {} tok ({} cached), prefill {} tok in {:.2} s ({:.0} tok/s); generated {} tok in {:.2} s ({:.1} tok/s){detail}",
        job.enqueued.elapsed().as_secs_f64(),
        n(c.prompt_tokens),
        n(c.reused_tokens),
        n(prefilled),
        c.prefill.as_secs_f64(),
        prefill_tps,
        n(c.tokens.len()),
        c.decode.as_secs_f64(),
        decode_tps
    );
}

pub fn truncate(s: &str, max: usize) -> String {
    if s.len() <= max {
        return s.to_owned();
    }
    let mut end = max;
    while !s.is_char_boundary(end) {
        end -= 1;
    }
    format!("{}… (+{} bytes)", &s[..end], s.len() - end)
}

/// A request's progress-log state, kept across its steps.
struct Timing {
    prefill_total: usize,
    prefill_start: Instant,
    last_log: Instant,
    decode_start: Option<Instant>,
    disconnected: bool,
}

impl Timing {
    fn new() -> Self {
        Self {
            prefill_total: 0,
            prefill_start: Instant::now(),
            last_log: Instant::now(),
            decode_start: None,
            disconnected: false,
        }
    }
}

struct Hooks<'a> {
    job: &'a Job,
    shared: &'a Shared,
    t: &'a mut Timing,
    slot: usize,
}

impl<'a> Hooks<'a> {
    fn new(job: &'a Job, shared: &'a Shared, t: &'a mut Timing, slot: usize) -> Self {
        Self {
            job,
            shared,
            t,
            slot,
        }
    }
    fn send(&mut self, event: Event) {
        if self.job.events.send(event).is_err() {
            self.gone();
        }
    }
    fn gone(&mut self) {
        if !self.t.disconnected {
            self.t.disconnected = true;
            self.job.cancel.store(true, Ordering::SeqCst);
        }
    }
}

impl ChatHooks for Hooks<'_> {
    fn delta(&mut self, delta: Delta) -> ferrum::Result<()> {
        self.send(Event::Delta(delta));
        Ok(())
    }
    fn begin(&mut self, prompt: usize, reused: usize) {
        self.t.prefill_total = prompt - reused;
        self.t.prefill_start = Instant::now();
        self.t.last_log = Instant::now();
        let slots = self.shared.status().slots.len();
        info!(
            Some(self.job.id),
            "prompt {} tok: {} reused from cache, {} to prefill{}",
            n(prompt),
            n(reused),
            n(prompt - reused),
            if slots > 1 {
                format!(" (slot {})", self.slot)
            } else {
                String::new()
            }
        );
        self.send(Event::Started { prompt, reused });
    }
    fn prefill(&mut self, processed: usize, total: usize) -> bool {
        self.send(Event::Progress { processed, total });
        if self.t.last_log.elapsed() > Duration::from_secs(5) && processed < total {
            self.t.last_log = Instant::now();
            info!(
                Some(self.job.id),
                "prefill {}/{} tok ({:.0}%), {:.0} tok/s",
                n(processed),
                n(total),
                100. * processed as f64 / total.max(1) as f64,
                processed as f64 / self.t.prefill_start.elapsed().as_secs_f64().max(1e-9)
            );
        }
        let keep = !self.job.cancel.load(Ordering::SeqCst);
        if !keep {
            info!(
                Some(self.job.id),
                "client disconnected during prefill; stopping ({} tok cached for a retry)",
                n(processed)
            );
        }
        keep
    }
    fn cancelled(&self) -> bool {
        self.job.cancel.load(Ordering::SeqCst)
    }
    fn generated(&mut self, tokens: usize) {
        let now = Instant::now();
        let slot = self.slot;
        let t = &mut *self.t;
        let start = *t.decode_start.get_or_insert_with(|| {
            self.shared.update(|s| {
                if let Some(status) = s.slots.get_mut(slot) {
                    status.phase = "generating";
                }
            });
            debug!(
                Some(self.job.id),
                "first token {:.2} s after prefill began",
                t.prefill_start.elapsed().as_secs_f64()
            );
            t.last_log = now;
            now
        });
        if now.duration_since(t.last_log) > Duration::from_secs(10) {
            t.last_log = now;
            info!(
                Some(self.job.id),
                "generating: {} tok, {:.1} tok/s",
                n(tokens),
                tokens as f64 / now.duration_since(start).as_secs_f64().max(1e-9)
            );
        }
    }
}
