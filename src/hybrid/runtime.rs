//! Chat runtime shared by `ferrum-cli` and `ferrum-server`: a loaded model,
//! its tokenizer and chat template, one session with prefix reuse, and a
//! streaming split of generated text into reasoning, content and tool calls.
#![forbid(unsafe_code)]
use super::{
    HybridConfig, LoadedHybrid, Variant,
    chat::{ChatTemplate, ParsedOutput, parse_output},
    config::uint,
    draft::{DraftModel, DraftOptions, DraftQuant},
    engine::{SpecGeometry, spec_bytes},
    mtp::{Mtp, mtp_state_bytes, mtp_weight_bytes},
    plan::{DraftMemory, PlanOptions},
    session::{Completion, Flow, GenerationHooks, SamplingParams, Session, StopReason},
    speculative::Drafter,
};
use crate::{
    Error, MetalDevice, Result,
    loader::gguf::{GgufFile, MetadataValue},
};
use serde_json::{Map, Value as Json};
use std::path::{Path, PathBuf};

/// Where speculative drafts come from.
#[derive(Debug, Clone, PartialEq)]
pub enum DraftSource {
    /// The target GGUF's own NextN/MTP block.
    Mtp,
    /// A DFlash / DFlash2 / DSpark checkpoint directory (or HF cache entry).
    Checkpoint(PathBuf),
}

impl DraftSource {
    /// `mtp` or a path.
    pub fn parse(value: &str) -> Self {
        if value.eq_ignore_ascii_case("mtp") {
            Self::Mtp
        } else {
            Self::Checkpoint(value.into())
        }
    }
}

/// Speculative decoding setup for `Runtime::load_speculative`.
#[derive(Debug, Clone)]
pub struct SpecOptions {
    pub source: DraftSource,
    /// Drafts per step (None: 3 for MTP; the trained block for dense
    /// targets and 2 for MoE targets, whose verify cost grows with the
    /// number of distinct experts).
    pub max_drafts: Option<usize>,
    pub draft: DraftOptions,
}

impl SpecOptions {
    pub fn new(source: DraftSource) -> Self {
        Self {
            source,
            max_drafts: None,
            draft: DraftOptions::default(),
        }
    }

    /// Apply one command-line flag shared by `ferrum-cli` and
    /// `ferrum-server`; returns false when `key` is not a draft flag.
    /// `--draft mtp|DIR` enables speculation; the others tune it:
    /// `--draft-max N`, `--draft-quant q8_0|q4_0`, `--draft-context N`
    /// (context slots for drafts without a sliding window) and
    /// `--draft-p-min P` (DSpark confidence cut-off).
    pub fn apply_flag(spec: &mut Option<Self>, key: &str, value: &str) -> Result<bool> {
        let bad = || Error::Parameter(format!("invalid {key}: {value}"));
        if key == "--draft" {
            let previous = spec.take();
            let mut next = Self::new(DraftSource::parse(value));
            if let Some(p) = previous {
                next.max_drafts = p.max_drafts;
                next.draft = p.draft;
            }
            *spec = Some(next);
            return Ok(true);
        }
        if !key.starts_with("--draft-") {
            return Ok(false);
        }
        let s = spec.get_or_insert_with(|| Self::new(DraftSource::Mtp));
        match key {
            "--draft-max" => s.max_drafts = Some(value.parse().map_err(|_| bad())?),
            "--draft-quant" => {
                s.draft.quant = match value.to_ascii_lowercase().as_str() {
                    "q8_0" => DraftQuant::Q8_0,
                    "q4_0" => DraftQuant::Q4_0,
                    _ => return Err(bad()),
                }
            }
            "--draft-context" => s.draft.context_cap = value.parse().map_err(|_| bad())?,
            "--draft-p-min" => s.draft.confidence_min = value.parse().map_err(|_| bad())?,
            _ => return Ok(false),
        }
        Ok(true)
    }

    fn drafts(&self, target: &HybridConfig, trained: usize) -> usize {
        self.max_drafts
            .unwrap_or(match (&self.source, target.variant) {
                (DraftSource::Mtp, _) => 3,
                (_, Variant::Moe) => trained.min(2),
                _ => trained,
            })
    }

    /// Memory the drafter will take next to `target`, before loading.
    pub fn memory(
        &self,
        file: &GgufFile,
        target: &HybridConfig,
        chunk: usize,
    ) -> Result<DraftMemory> {
        match &self.source {
            DraftSource::Mtp => {
                let k = self.drafts(target, 3);
                let geometry = SpecGeometry {
                    rows: k + 1,
                    aux_layers: Vec::new(),
                    final_hidden: true,
                };
                let per_token = (2 * target.kv_heads * target.head_dim * 2) as f64;
                Ok(DraftMemory {
                    weights: mtp_weight_bytes(file, target),
                    fixed: mtp_state_bytes(target, 0, chunk, k)
                        + spec_bytes(target, chunk, &geometry),
                    per_token,
                })
            }
            DraftSource::Checkpoint(dir) => {
                let (config, weights, state) = DraftModel::memory(dir, target, chunk, self.draft)?;
                let k = self.drafts(target, config.max_drafts());
                let geometry = SpecGeometry {
                    rows: k + 1,
                    aux_layers: config.target_layers.clone(),
                    final_hidden: false,
                };
                Ok(DraftMemory {
                    weights,
                    fixed: state + spec_bytes(target, chunk, &geometry),
                    per_token: 0.,
                })
            }
        }
    }

    /// Load the drafter for a session of `capacity` positions.
    pub fn build(
        &self,
        d: &MetalDevice,
        target_path: &Path,
        loaded: &LoadedHybrid,
        capacity: usize,
    ) -> Result<Box<dyn Drafter>> {
        let model = &loaded.model;
        Ok(match &self.source {
            DraftSource::Mtp => {
                let k = self.drafts(&model.config, 3);
                Box::new(Mtp::load(d, target_path, model, capacity, k)?)
            }
            DraftSource::Checkpoint(dir) => {
                let (config, _, _) =
                    DraftModel::memory(dir, &model.config, model.chunk(), self.draft)?;
                let k = self.drafts(&model.config, config.max_drafts());
                Box::new(DraftModel::load(
                    d,
                    dir,
                    model,
                    capacity,
                    Some(k),
                    self.draft,
                )?)
            }
        })
    }
}

/// Incremental text produced while generating.
#[derive(Debug, Clone, PartialEq)]
pub enum Delta {
    Reasoning(String),
    Content(String),
}

#[derive(Debug, Clone)]
pub struct ChatRequest {
    pub messages: Vec<Json>,
    pub tools: Vec<Json>,
    pub max_tokens: usize,
    pub sampling: SamplingParams,
    /// Extra chat-template variables (`enable_thinking`, `reasoning_effort`, ...).
    pub template_vars: Map<String, Json>,
    /// Stop when any of these strings appears in the visible content.
    pub stop: Vec<String>,
    /// Most reasoning tokens before `</think>` is forced (None = unlimited).
    pub reasoning_budget: Option<usize>,
    /// Leave reasoning inline (with its tags) in the content instead of
    /// splitting it out (llama-server `--reasoning-format none`).
    pub raw_reasoning: bool,
}

impl ChatRequest {
    pub fn new(messages: Vec<Json>, sampling: SamplingParams) -> Self {
        Self {
            messages,
            tools: Vec::new(),
            max_tokens: usize::MAX,
            sampling,
            template_vars: Map::new(),
            stop: Vec::new(),
            reasoning_budget: None,
            raw_reasoning: false,
        }
    }
}

#[derive(Debug, Clone)]
pub struct ChatResult {
    pub output: ParsedOutput,
    pub completion: Completion,
    /// True when a `stop` string ended generation.
    pub stopped_by_string: bool,
    /// Generated tokens that belonged to the reasoning block.
    pub reasoning_tokens: usize,
    /// True when the thinking budget closed the reasoning block.
    pub budget_exhausted: bool,
    /// Raw generated text (reasoning, content and tool-call markup).
    pub text: String,
}

/// Callbacks for one chat or completion request.
pub trait ChatHooks {
    fn delta(&mut self, delta: Delta) -> Result<()>;
    /// Prompt length and tokens reused from the session, before prefill.
    fn begin(&mut self, _prompt: usize, _reused: usize) {}
    /// Prefill progress over the new prompt tokens; false cancels.
    fn prefill(&mut self, _processed: usize, _total: usize) -> bool {
        true
    }
    /// Polled after every token; true stops generation.
    fn cancelled(&self) -> bool {
        false
    }
    /// Called after every generated token with the running count.
    fn generated(&mut self, _tokens: usize) {}
}

impl<F: FnMut(Delta) -> Result<()>> ChatHooks for F {
    fn delta(&mut self, delta: Delta) -> Result<()> {
        self(delta)
    }
}

pub struct Runtime {
    pub device: MetalDevice,
    pub loaded: LoadedHybrid,
    pub template: ChatTemplate,
    pub session: Session,
    /// The model's recommended sampling (GGUF `general.sampling.*`).
    pub default_sampling: SamplingParams,
    pub model_name: String,
    /// Device bytes the drafter took (weights, buffers, verify buffers).
    pub draft_loaded_bytes: usize,
}

impl Runtime {
    /// Compile and page in every kernel path (chunked prefill, single-token
    /// decode, sampling candidates) so the first real request runs at full
    /// speed, then clear the session.
    pub fn warmup(&mut self) -> Result<()> {
        let model = &self.loaded.model;
        let n = 66.min(model.chunk() + 2).min(self.session.capacity() - 3);
        let prompt: Vec<u32> = (0..n as u32).map(|i| 1000 + (i * 7919) % 20000).collect();
        let result = self.session.generate(
            &self.device,
            model,
            &prompt,
            2,
            &self.default_sampling,
            &[],
            &mut |_| Ok(true),
        );
        self.session.reset();
        result.map(drop)
    }

    pub fn load(path: impl AsRef<Path>, options: PlanOptions) -> Result<Self> {
        Self::load_speculative(path, options, None)
    }

    /// Load with an optional drafter: its memory is planned before any
    /// weight is read (the auto-fitted context shrinks to make room), and the
    /// session speculates by default.
    pub fn load_speculative(
        path: impl AsRef<Path>,
        mut options: PlanOptions,
        spec: Option<&SpecOptions>,
    ) -> Result<Self> {
        let path = path.as_ref();
        let device = MetalDevice::new()?;
        let file = GgufFile::open(path)?;
        let default_sampling = recommended_sampling(&file);
        if let Some(spec) = spec {
            let config = HybridConfig::from_gguf(&file)?;
            options.draft = spec.memory(&file, &config, options.chunk.max(1))?;
        }
        drop(file);
        let loaded = super::load_with(&device, path, options)?;
        let template = ChatTemplate::new(
            loaded
                .chat_template
                .as_deref()
                .ok_or_else(|| Error::Tokenizer("GGUF has no chat template".into()))?,
        )?;
        let mut session = Session::new(
            &device,
            &loaded.model,
            loaded.plan.context,
            options.snapshots,
        )?;
        let mut draft_loaded_bytes = 0;
        if let Some(spec) = spec {
            let before = device.allocated_bytes();
            let drafter = spec.build(&device, path, &loaded, loaded.plan.context)?;
            session.set_drafter(&device, &loaded.model, drafter)?;
            draft_loaded_bytes = device.allocated_bytes().saturating_sub(before);
        }
        let model_name = path.file_stem().map_or_else(
            || loaded.model.config.name.clone(),
            |s| s.to_string_lossy().into_owned(),
        );
        Ok(Self {
            device,
            loaded,
            template,
            session,
            default_sampling,
            model_name,
            draft_loaded_bytes,
        })
    }

    pub fn render(&self, request: &ChatRequest) -> Result<String> {
        let mut vars = request.template_vars.clone();
        normalize_thinking(self.template.source(), &mut vars);
        self.template.render(
            &request.messages,
            (!request.tools.is_empty()).then_some(&request.tools[..]),
            true,
            &vars,
        )
    }

    /// Run one chat turn, reusing whatever prefix the session already holds.
    pub fn chat(&mut self, request: &ChatRequest, hooks: impl ChatHooks) -> Result<ChatResult> {
        let prompt_text = self.render(request)?;
        let starts_in_reasoning = prompt_text.trim_end().ends_with("<think>");
        let (result, text) = self.run(
            &prompt_text,
            request.max_tokens,
            &request.sampling,
            &request.stop,
            starts_in_reasoning && !request.raw_reasoning,
            if starts_in_reasoning {
                request.reasoning_budget
            } else {
                None
            },
            true,
            hooks,
        )?;
        let (completion, stopped, reasoning_tokens, budget_exhausted, visible) = result;
        let mut output = if request.raw_reasoning {
            let mut parsed = parse_output(&text, false, &request.tools);
            let prefix = if starts_in_reasoning { "<think>\n" } else { "" };
            if !parsed.tool_calls.is_empty() {
                parsed.content = format!("{prefix}{}", parsed.content);
            } else {
                parsed.content = format!("{prefix}{}", text.trim_end());
            }
            parsed
        } else {
            parse_output(&text, starts_in_reasoning, &request.tools)
        };
        if stopped {
            output.content = visible;
        }
        Ok(ChatResult {
            stopped_by_string: stopped,
            output,
            completion,
            reasoning_tokens,
            budget_exhausted,
            text,
        })
    }

    /// Raw text completion (no chat template, no tool parsing).
    pub fn complete(
        &mut self,
        prompt: &str,
        max_tokens: usize,
        sampling: &SamplingParams,
        stop: &[String],
        hooks: impl ChatHooks,
    ) -> Result<(Completion, String, bool)> {
        let ((completion, stopped, _, _, visible), text) = self.run(
            prompt, max_tokens, sampling, stop, false, None, false, hooks,
        )?;
        Ok((completion, if stopped { visible } else { text }, stopped))
    }

    #[allow(clippy::too_many_arguments, clippy::type_complexity)]
    fn run(
        &mut self,
        prompt_text: &str,
        max_tokens: usize,
        sampling: &SamplingParams,
        stop: &[String],
        split_reasoning: bool,
        reasoning_budget: Option<usize>,
        tools: bool,
        hooks: impl ChatHooks,
    ) -> Result<((Completion, bool, usize, bool, String), String)> {
        let prompt = self.loaded.tokenizer.encode(prompt_text)?;
        let close_reasoning = self.loaded.tokenizer.encode("\n</think>\n\n")?;
        let tokenizer = &self.loaded.tokenizer;
        struct Driver<'a, H: ChatHooks> {
            hooks: H,
            stream: tokenizers::tokenizer::DecodeStream<
                'a,
                tokenizers::ModelWrapper,
                tokenizers::NormalizerWrapper,
                tokenizers::PreTokenizerWrapper,
                tokenizers::PostProcessorWrapper,
                tokenizers::DecoderWrapper,
            >,
            splitter: Splitter,
            failure: Option<Error>,
            reasoning_tokens: usize,
            budget: Option<usize>,
            budget_exhausted: bool,
            close: Vec<u32>,
            close_text: &'static str,
            count: usize,
        }
        impl<H: ChatHooks> Driver<'_, H> {
            fn emit(&mut self, piece: &str) -> bool {
                for delta in self.splitter.push(piece) {
                    if let Err(e) = self.hooks.delta(delta) {
                        self.failure = Some(e);
                        return false;
                    }
                }
                true
            }
        }
        impl<H: ChatHooks> GenerationHooks for Driver<'_, H> {
            fn begin(&mut self, prompt: usize, reused: usize) {
                self.hooks.begin(prompt, reused);
            }
            fn prefill(&mut self, processed: usize, total: usize) -> bool {
                self.hooks.prefill(processed, total) && !self.hooks.cancelled()
            }
            fn token(&mut self, token: u32) -> Result<Flow> {
                self.count += 1;
                self.hooks.generated(self.count);
                let in_reasoning = self.splitter.in_reasoning;
                if in_reasoning {
                    self.reasoning_tokens += 1;
                }
                let piece = self
                    .stream
                    .step(token)
                    .map_err(|e| Error::Tokenizer(e.to_string()))?;
                if let Some(piece) = piece
                    && !self.emit(&piece)
                {
                    return Ok(Flow::Stop);
                }
                if self.splitter.stopped || self.hooks.cancelled() {
                    return Ok(Flow::Stop);
                }
                if self.splitter.in_reasoning
                    && self.budget.is_some_and(|b| self.reasoning_tokens >= b)
                {
                    self.budget_exhausted = true;
                    let text = self.close_text;
                    if !self.emit(text) {
                        return Ok(Flow::Stop);
                    }
                    return Ok(Flow::Inject(self.close.clone()));
                }
                Ok(Flow::Continue)
            }
        }
        let mut driver = Driver {
            hooks,
            stream: tokenizer.decode_stream(),
            splitter: Splitter::new(split_reasoning, stop, tools),
            failure: None,
            reasoning_tokens: 0,
            budget: reasoning_budget,
            budget_exhausted: false,
            close: close_reasoning,
            close_text: "\n</think>\n\n",
            count: 0,
        };
        let completion = self.session.generate(
            &self.device,
            &self.loaded.model,
            &prompt,
            max_tokens,
            sampling,
            &self.loaded.eos_ids,
            &mut driver,
        )?;
        if let Some(e) = driver.failure.take() {
            return Err(e);
        }
        for delta in driver.splitter.finish() {
            driver.hooks.delta(delta)?;
        }
        let stopped = driver.splitter.stopped;
        let visible = driver.splitter.visible_content();
        let text = std::mem::take(&mut driver.splitter.text);
        Ok((
            (
                completion,
                stopped,
                driver.reasoning_tokens,
                driver.budget_exhausted,
                visible,
            ),
            text,
        ))
    }
}

impl StopReason {
    /// OpenAI `finish_reason` for a completion that produced no tool call.
    pub fn finish_reason(self) -> &'static str {
        match self {
            StopReason::Eos | StopReason::Stopped | StopReason::Cancelled => "stop",
            StopReason::Length | StopReason::ContextFull => "length",
        }
    }
}

/// Map thinking controls onto what this template understands:
/// `reasoning_effort` "none"/"off"/"disabled" becomes `enable_thinking=false`;
/// other efforts snap to the nearest level the template names (e.g. "high"
/// becomes "xhigh" for a template that only knows low/medium/xhigh).
pub fn normalize_thinking(template: &str, vars: &mut Map<String, Json>) {
    for key in ["enable_thinking", "thinking"] {
        if let Some(Json::String(s)) = vars.get(key) {
            let on = !matches!(
                s.to_ascii_lowercase().as_str(),
                "false" | "off" | "no" | "0" | "disabled"
            );
            vars.insert(key.into(), Json::Bool(on));
        }
    }
    let Some(effort) = vars
        .get("reasoning_effort")
        .and_then(Json::as_str)
        .map(str::to_ascii_lowercase)
    else {
        return;
    };
    if matches!(effort.as_str(), "none" | "off" | "disabled" | "false") {
        vars.remove("reasoning_effort");
        vars.insert("enable_thinking".into(), Json::Bool(false));
        return;
    }
    const LEVELS: [&str; 8] = [
        "minimal",
        "low",
        "medium",
        "high",
        "xhigh",
        "max",
        "extreme",
        "ultracode",
    ];
    let known: Vec<usize> = LEVELS
        .iter()
        .enumerate()
        .filter(|(_, l)| {
            template.contains(&format!("'{l}'")) || template.contains(&format!("\"{l}\""))
        })
        .map(|(i, _)| i)
        .collect();
    let Some(wanted) = LEVELS.iter().position(|l| *l == effort) else {
        return;
    };
    if known.is_empty() || known.contains(&wanted) {
        vars.insert("reasoning_effort".into(), Json::String(effort));
        return;
    }
    // Nearest named level, preferring the higher one on ties.
    let nearest = known
        .iter()
        .min_by_key(|&&i| (i as isize - wanted as isize).abs() * 2 - isize::from(i > wanted))
        .copied()
        .unwrap_or(wanted);
    vars.insert(
        "reasoning_effort".into(),
        Json::String(LEVELS[nearest].into()),
    );
}

/// The GGUF's recommended sampling (`general.sampling.*`), with Qwen defaults.
pub fn recommended_sampling(file: &GgufFile) -> SamplingParams {
    let md = file.metadata();
    let float = |key: &str| match md.get(key) {
        Some(MetadataValue::Float32(v)) => Some(*v),
        Some(MetadataValue::Float64(v)) => Some(*v as f32),
        _ => None,
    };
    SamplingParams {
        temperature: float("general.sampling.temp").unwrap_or(1.0),
        top_p: float("general.sampling.top_p").unwrap_or(0.95),
        top_k: uint(md, "general.sampling.top_k").unwrap_or(20),
        min_p: float("general.sampling.min_p").unwrap_or(0.),
        ..SamplingParams::default()
    }
}

/// Routes streamed text: reasoning until `</think>`, then content, holding
/// back anything that may start `<tool_call>` (tool calls are reported once
/// complete) or a stop string.
struct Splitter {
    text: String,
    in_reasoning: bool,
    /// Byte offset in `text` up to which deltas were emitted.
    emitted: usize,
    content_start: usize,
    tool_started: bool,
    stop: Vec<String>,
    stopped: bool,
    stop_at: Option<usize>,
    tools: bool,
}

impl Splitter {
    fn new(in_reasoning: bool, stop: &[String], tools: bool) -> Self {
        Self {
            text: String::new(),
            in_reasoning,
            emitted: 0,
            content_start: 0,
            tool_started: false,
            stop: stop.iter().filter(|s| !s.is_empty()).cloned().collect(),
            stopped: false,
            stop_at: None,
            tools,
        }
    }

    fn push(&mut self, piece: &str) -> Vec<Delta> {
        self.text.push_str(piece);
        self.drain(false)
    }

    fn finish(&mut self) -> Vec<Delta> {
        self.drain(true)
    }

    fn visible_content(&self) -> String {
        let end = self.stop_at.unwrap_or(self.text.len());
        self.text[self.content_start.min(end)..end]
            .trim()
            .to_owned()
    }

    fn drain(&mut self, last: bool) -> Vec<Delta> {
        let mut out = Vec::new();
        if self.in_reasoning {
            if let Some(end) = self.text[self.emitted..]
                .find("</think>")
                .map(|i| i + self.emitted)
            {
                if end > self.emitted {
                    out.push(Delta::Reasoning(self.text[self.emitted..end].to_owned()));
                }
                self.in_reasoning = false;
                self.emitted = end + "</think>".len();
                self.content_start = self.emitted;
            } else {
                let safe = if last {
                    self.text.len()
                } else {
                    hold_back(&self.text, self.emitted, &["</think>"])
                };
                if safe > self.emitted {
                    out.push(Delta::Reasoning(self.text[self.emitted..safe].to_owned()));
                    self.emitted = safe;
                }
                return out;
            }
        }
        if self.tool_started || self.stopped {
            return out;
        }
        let pending = &self.text[self.emitted..];
        let stop_hit = self
            .stop
            .iter()
            .filter_map(|s| pending.find(s.as_str()))
            .min();
        let tool_hit = if self.tools {
            pending.find("<tool_call>")
        } else {
            None
        };
        let mut end = if last {
            self.text.len()
        } else {
            let mut markers: Vec<&str> = if self.tools {
                vec!["<tool_call>"]
            } else {
                Vec::new()
            };
            markers.extend(self.stop.iter().map(String::as_str));
            hold_back(&self.text, self.emitted, &markers)
        };
        if let Some(t) = tool_hit {
            end = end.max(self.emitted).min(self.emitted + t);
            self.tool_started = true;
        }
        if let Some(s) = stop_hit
            && tool_hit.is_none_or(|t| s < t)
        {
            end = self.emitted + s;
            self.stopped = true;
            self.stop_at = Some(end);
            self.tool_started = false;
        }
        // Leading whitespace right after </think> is template formatting.
        let mut begin = self.emitted;
        if begin == self.content_start {
            let trimmed = self.text[begin..end.max(begin)].trim_start();
            if trimmed.is_empty() && !last && !self.stopped && !self.tool_started {
                return out;
            }
            begin = end.max(begin) - trimmed.len();
        }
        if end > begin {
            out.push(Delta::Content(self.text[begin..end].to_owned()));
        }
        self.emitted = end.max(self.emitted);
        out
    }
}

/// Largest offset >= `from` such that `text[..offset]` cannot be the start of
/// any marker still being generated.
fn hold_back(text: &str, from: usize, markers: &[&str]) -> usize {
    let mut safe = text.len();
    for marker in markers {
        for k in (1..marker.len()).rev() {
            if text.len() >= k
                && text.is_char_boundary(text.len() - k)
                && marker.starts_with(&text[text.len() - k..])
            {
                safe = safe.min(text.len() - k);
                break;
            }
        }
    }
    while safe > from && !text.is_char_boundary(safe) {
        safe -= 1;
    }
    safe.max(from)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run(pieces: &[&str], reasoning: bool, stop: &[&str]) -> (Vec<Delta>, Splitter) {
        let mut s = Splitter::new(
            reasoning,
            &stop.iter().map(|s| s.to_string()).collect::<Vec<_>>(),
            true,
        );
        let mut out = Vec::new();
        for p in pieces {
            out.extend(s.push(p));
        }
        out.extend(s.finish());
        (out, s)
    }

    fn join(deltas: &[Delta]) -> (String, String) {
        let mut r = String::new();
        let mut c = String::new();
        for d in deltas {
            match d {
                Delta::Reasoning(t) => r.push_str(t),
                Delta::Content(t) => c.push_str(t),
            }
        }
        (r, c)
    }

    #[test]
    fn splits_reasoning_content_and_holds_back_tool_calls() {
        let (deltas, _) = run(
            &[
                "Think",
                "ing.</th",
                "ink>\n\nHel",
                "lo <tool_",
                "call>\n<function=f>",
            ],
            true,
            &[],
        );
        let (r, c) = join(&deltas);
        assert_eq!(r, "Thinking.");
        assert_eq!(c, "Hello ");
    }

    #[test]
    fn stop_strings_truncate_content() {
        let (deltas, s) = run(&["a b ST", "OP c"], false, &["STOP"]);
        assert_eq!(join(&deltas).1, "a b ");
        assert!(s.stopped);
        assert_eq!(s.visible_content(), "a b");
    }
}

#[cfg(test)]
mod thinking_tests {
    use super::*;

    #[test]
    fn effort_snaps_to_template_levels() {
        let template = "{% if reasoning_effort not in ('xhigh', 'medium', 'low') %}";
        let mut vars = Map::new();
        vars.insert("reasoning_effort".into(), Json::String("high".into()));
        normalize_thinking(template, &mut vars);
        assert_eq!(vars["reasoning_effort"], "xhigh");
        vars.insert("reasoning_effort".into(), Json::String("minimal".into()));
        normalize_thinking(template, &mut vars);
        assert_eq!(vars["reasoning_effort"], "low");
        vars.insert("reasoning_effort".into(), Json::String("none".into()));
        normalize_thinking(template, &mut vars);
        assert_eq!(vars["enable_thinking"], false);
        assert!(!vars.contains_key("reasoning_effort"));
    }
}
