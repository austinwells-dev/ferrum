//! The model thread: one `Runtime`, a FIFO job queue, per-request event
//! channels, cancellation and live status for the HTTP threads.
use crate::{debug, error, info, log::n, warn};
use ferrum::{
    Error,
    hybrid::{
        plan::PlanOptions,
        runtime::{ChatHooks, ChatRequest, ChatResult, Delta, Runtime, SpecOptions},
        session::{Completion, SamplingParams, StopReason},
    },
};
use serde::Serialize;
use std::{
    path::PathBuf,
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, AtomicUsize, Ordering},
        mpsc::{Receiver, Sender, channel},
    },
    time::{Duration, Instant},
};

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

pub struct Job {
    pub id: u64,
    pub work: Work,
    pub events: Sender<Event>,
    pub cancel: Arc<AtomicBool>,
    pub enqueued: Instant,
}

#[derive(Debug, Clone, Default, Serialize)]
pub struct Status {
    pub state: &'static str,
    pub load_error: Option<String>,
    pub model: String,
    pub context: usize,
    pub busy_request: Option<u64>,
    pub busy_phase: &'static str,
    pub session_tokens: usize,
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
    shared: Arc<Shared>,
) -> Sender<Job> {
    let (tx, rx) = channel::<Job>();
    std::thread::Builder::new()
        .name("model".into())
        .spawn(move || run(path, options, spec, alias, shared, rx))
        .expect("spawn model thread");
    tx
}

fn run(
    path: PathBuf,
    options: PlanOptions,
    spec: Option<SpecOptions>,
    alias: Option<String>,
    shared: Arc<Shared>,
    jobs: Receiver<Job>,
) {
    let start = Instant::now();
    let mut rt = match Runtime::load_speculative(&path, options, spec.as_ref()) {
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
        "context {} tokens (largest that fits: {}, trained {}); memory {:.2} GiB of {:.2} GiB budget: weights {:.2}, KV {:.2} ({:.1} KiB/token), recurrent+snapshots {:.2}, scratch {:.2}",
        n(plan.context),
        n(plan.max_context),
        n(plan.trained_context),
        gib(plan.total()),
        gib(plan.budget),
        gib(plan.weights),
        gib(plan.kv(plan.context)),
        plan.kv_per_token / 1024.,
        gib(plan.recurrent + plan.snapshots),
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
    shared.update(|s| {
        s.state = "ready";
        s.model = rt.model_name.clone();
        s.context = plan.context;
        s.memory_budget_bytes = plan.budget;
        s.memory_predicted_bytes = plan.total();
        s.drafter = drafter.map(|(name, _)| name).unwrap_or_default();
    });
    for job in jobs {
        shared.queued.fetch_sub(1, Ordering::SeqCst);
        if job.cancel.load(Ordering::SeqCst) {
            info!(Some(job.id), "client left while queued; skipped");
            shared.update(|s| s.cancelled += 1);
            continue;
        }
        let waited = job.enqueued.elapsed();
        if waited > Duration::from_millis(250) {
            info!(
                Some(job.id),
                "started after {:.1} s in queue",
                waited.as_secs_f64()
            );
        }
        shared.update(|s| {
            s.busy_request = Some(job.id);
            s.busy_phase = "prefill";
        });
        let hooks = Hooks::new(&job, &shared);
        let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| match &job.work {
            Work::Chat(request) => rt.chat(request, hooks).map(|r| Event::Chat(Box::new(r))),
            Work::Complete {
                prompt,
                max_tokens,
                sampling,
                stop,
            } => rt.complete(prompt, *max_tokens, sampling, stop, hooks).map(
                |(completion, text, stopped)| Event::Completed {
                    completion,
                    text,
                    stopped,
                },
            ),
        }));
        let event = match outcome {
            Ok(Ok(event)) => {
                let completion = match &event {
                    Event::Chat(r) => &r.completion,
                    Event::Completed { completion, .. } => completion,
                    _ => unreachable!(),
                };
                record(&shared, &job, completion, &event, rt.session.len());
                event
            }
            Ok(Err(e)) => {
                let status = if matches!(e, Error::Parameter(_) | Error::Tokenizer(_)) {
                    400
                } else {
                    500
                };
                if status == 400 {
                    warn!(Some(job.id), "rejected: {e}");
                } else {
                    error!(Some(job.id), "failed: {e}");
                }
                shared.update(|s| s.failed += 1);
                Event::Failed {
                    status,
                    message: e.to_string(),
                }
            }
            Err(panic) => {
                let message = panic
                    .downcast_ref::<String>()
                    .cloned()
                    .or_else(|| panic.downcast_ref::<&str>().map(|s| s.to_string()))
                    .unwrap_or_else(|| "unknown panic".into());
                error!(
                    Some(job.id),
                    "internal panic: {message}; conversation cache reset"
                );
                rt.session.reset();
                shared.update(|s| s.failed += 1);
                Event::Failed {
                    status: 500,
                    message: format!("internal error: {message}"),
                }
            }
        };
        let _ = job.events.send(event);
        let tokens = rt.session.len();
        shared.update(|s| {
            s.busy_request = None;
            s.busy_phase = "idle";
            s.session_tokens = tokens;
        });
    }
}

fn record(shared: &Shared, job: &Job, c: &Completion, event: &Event, session_len: usize) {
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

struct Hooks<'a> {
    job: &'a Job,
    shared: &'a Shared,
    prefill_total: usize,
    prefill_start: Instant,
    last_log: Instant,
    decode_start: Option<Instant>,
    disconnected: bool,
}

impl<'a> Hooks<'a> {
    fn new(job: &'a Job, shared: &'a Shared) -> Self {
        Self {
            job,
            shared,
            prefill_total: 0,
            prefill_start: Instant::now(),
            last_log: Instant::now(),
            decode_start: None,
            disconnected: false,
        }
    }
    fn send(&mut self, event: Event) {
        if self.job.events.send(event).is_err() {
            self.gone();
        }
    }
    fn gone(&mut self) {
        if !self.disconnected {
            self.disconnected = true;
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
        self.prefill_total = prompt - reused;
        self.prefill_start = Instant::now();
        self.last_log = Instant::now();
        info!(
            Some(self.job.id),
            "prompt {} tok: {} reused from cache, {} to prefill",
            n(prompt),
            n(reused),
            n(prompt - reused)
        );
        self.send(Event::Started { prompt, reused });
    }
    fn prefill(&mut self, processed: usize, total: usize) -> bool {
        self.send(Event::Progress { processed, total });
        if self.last_log.elapsed() > Duration::from_secs(5) && processed < total {
            self.last_log = Instant::now();
            info!(
                Some(self.job.id),
                "prefill {}/{} tok ({:.0}%), {:.0} tok/s",
                n(processed),
                n(total),
                100. * processed as f64 / total.max(1) as f64,
                processed as f64 / self.prefill_start.elapsed().as_secs_f64().max(1e-9)
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
        let start = *self.decode_start.get_or_insert_with(|| {
            self.shared.update(|s| s.busy_phase = "generating");
            debug!(
                Some(self.job.id),
                "first token {:.2} s after prefill began",
                self.prefill_start.elapsed().as_secs_f64()
            );
            self.last_log = now;
            now
        });
        if now.duration_since(self.last_log) > Duration::from_secs(10) {
            self.last_log = now;
            info!(
                Some(self.job.id),
                "generating: {} tok, {:.1} tok/s",
                n(tokens),
                tokens as f64 / now.duration_since(start).as_secs_f64().max(1e-9)
            );
        }
    }
}
