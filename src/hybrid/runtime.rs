//! Chat runtime shared by `ferrum-cli` and `ferrum-server`: a loaded model,
//! its tokenizer and chat template, one session with prefix reuse, and a
//! streaming split of generated text into reasoning, content and tool calls.
#![forbid(unsafe_code)]
use super::{
    LoadedHybrid,
    chat::{ChatTemplate, ParsedOutput, parse_output},
    config::uint,
    plan::PlanOptions,
    session::{Completion, SamplingParams, Session, StopReason},
};
use crate::{
    Error, MetalDevice, Result,
    loader::gguf::{GgufFile, MetadataValue},
};
use serde_json::{Map, Value as Json};
use std::path::Path;

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
}

#[derive(Debug, Clone)]
pub struct ChatResult {
    pub output: ParsedOutput,
    pub completion: Completion,
    /// True when a `stop` string ended generation.
    pub stopped_by_string: bool,
}

pub struct Runtime {
    pub device: MetalDevice,
    pub loaded: LoadedHybrid,
    pub template: ChatTemplate,
    pub session: Session,
    /// The model's recommended sampling (GGUF `general.sampling.*`).
    pub default_sampling: SamplingParams,
    pub model_name: String,
}

impl Runtime {
    pub fn load(path: impl AsRef<Path>, options: PlanOptions) -> Result<Self> {
        let path = path.as_ref();
        let device = MetalDevice::new()?;
        let file = GgufFile::open(path)?;
        let default_sampling = recommended_sampling(&file);
        drop(file);
        let loaded = super::load_with(&device, path, options)?;
        let template = ChatTemplate::new(
            loaded
                .chat_template
                .as_deref()
                .ok_or_else(|| Error::Tokenizer("GGUF has no chat template".into()))?,
        )?;
        let session = Session::new(
            &device,
            &loaded.model,
            loaded.plan.context,
            options.snapshots,
        )?;
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
        })
    }

    pub fn render(&self, request: &ChatRequest) -> Result<String> {
        self.template.render(
            &request.messages,
            (!request.tools.is_empty()).then_some(&request.tools[..]),
            true,
            &request.template_vars,
        )
    }

    /// Run one chat turn, reusing whatever prefix the session already holds.
    pub fn chat(
        &mut self,
        request: &ChatRequest,
        mut on_delta: impl FnMut(Delta) -> Result<()>,
    ) -> Result<ChatResult> {
        let prompt_text = self.render(request)?;
        let prompt = self.loaded.tokenizer.encode(&prompt_text)?;
        let starts_in_reasoning = prompt_text.trim_end().ends_with("<think>");
        let tokenizer = &self.loaded.tokenizer;
        let mut stream = tokenizer.decode_stream();
        let mut splitter = Splitter::new(starts_in_reasoning, &request.stop);
        let mut failure = None;
        let completion = self.session.generate(
            &self.device,
            &self.loaded.model,
            &prompt,
            request.max_tokens,
            &request.sampling,
            &self.loaded.eos_ids,
            |token| {
                let piece = match stream.step(token) {
                    Ok(Some(piece)) => piece,
                    Ok(None) => return Ok(true),
                    Err(e) => return Err(Error::Tokenizer(e.to_string())),
                };
                for delta in splitter.push(&piece) {
                    if let Err(e) = on_delta(delta) {
                        failure = Some(e);
                        return Ok(false);
                    }
                }
                Ok(!splitter.stopped)
            },
        )?;
        if let Some(e) = failure {
            return Err(e);
        }
        for delta in splitter.finish() {
            on_delta(delta)?;
        }
        let mut output = parse_output(&splitter.text, starts_in_reasoning, &request.tools);
        if splitter.stopped {
            output.content = splitter.visible_content();
        }
        Ok(ChatResult {
            stopped_by_string: splitter.stopped,
            output,
            completion,
        })
    }
}

impl StopReason {
    /// OpenAI `finish_reason` for a completion that produced no tool call.
    pub fn finish_reason(self) -> &'static str {
        match self {
            StopReason::Eos | StopReason::Stopped => "stop",
            StopReason::Length | StopReason::ContextFull => "length",
        }
    }
}

fn recommended_sampling(file: &GgufFile) -> SamplingParams {
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
}

impl Splitter {
    fn new(in_reasoning: bool, stop: &[String]) -> Self {
        Self {
            text: String::new(),
            in_reasoning,
            emitted: 0,
            content_start: 0,
            tool_started: false,
            stop: stop.iter().filter(|s| !s.is_empty()).cloned().collect(),
            stopped: false,
            stop_at: None,
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
        let tool_hit = pending.find("<tool_call>");
        let mut end = if last {
            self.text.len()
        } else {
            let mut markers: Vec<&str> = vec!["<tool_call>"];
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
