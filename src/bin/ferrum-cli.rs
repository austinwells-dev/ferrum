//! Interactive terminal chat with a Qwen3.5-family GGUF (or, experimentally,
//! Flint, Spark or LFM3).
//!
//! ferrum-cli --model FILE.gguf [--context N|auto] [--kv-cache f16|q8] [--system TEXT]
//!            [--temperature T] [--top-p P] [--top-k K] [--min-p P]
//!            [--presence-penalty X] [--repetition-penalty X] [--seed N]
//!            [--max-tokens N] [--no-think] [--reasoning-effort LEVEL]
//!            [--reasoning-budget N] [--hide-thinking]
//!            [--draft mtp|lookup|DIR|FILE.gguf] [--draft-max N] [--draft-quant q8_0|q4_0]
//!            [--draft-context N] [--draft-p-min P] [--draft-tree N|auto]
//!            [--draft-tree-search best-first|siblings] [--draft-block N] [--draft-vocab N]
//!            [--mmproj FILE|auto|none] [--image-tokens N] [--image-min-tokens N]
//!            [--gpu-share PCT|off] [--prompt TEXT] [--version]
//!
//! --prompt TEXT (-p) answers one message and exits; text piped on stdin is
//! appended to it (`cat notes.txt | ferrum-cli -m M -p "Summarize:"`).
//! --gpu-share PCT keeps the GPU busy at most PCT% of the time (5-100), so
//! video and the desktop stay smooth while the model runs; the default is
//! FERRUM_GPU_SHARE, else off.
//!
//! ferrum-cli image --model DIR --prompt TEXT [--size WxH] [--seed N] [--out FILE.png]
//!                  (text-to-image with Qwen-Image-2.1; see `ferrum-cli image --help`)
//!
//! Commands: /reset, /think on|off, /image PATH [low|high|auto|N],
//! /audio PATH.wav, /detail low|high|auto|N, /stats, /exit. End a line with `\`
//! to continue the message on the next line. Ctrl-C stops a reply; at the
//! prompt it leaves. In a terminal the prompt has line editing and history
//! (kept in ~/.cache/ferrum/cli-history).
//!
//! Image detail: `low` caps an image at 256 tokens, `high` at --image-tokens, `N`
//! at N tokens; `/detail` sets it for the images that follow, and a size after
//! `/image PATH` for that image only.
use ferrum::{
    Error, Result,
    hybrid::{
        KvCache,
        plan::PlanOptions,
        runtime::{ChatRequest, Delta, Runtime, SpecOptions},
    },
    vision::{ImageBudget, media},
};
use serde_json::{Map, Value as Json, json};
use std::{
    io::{BufRead, IsTerminal, Read, Write},
    sync::atomic::{AtomicBool, Ordering},
};

#[path = "ferrum-cli/image.rs"]
mod image;

struct Options {
    model: String,
    context: Option<usize>,
    kv_cache: KvCache,
    system: Option<String>,
    max_tokens: usize,
    think: bool,
    effort: Option<String>,
    show_thinking: bool,
    reasoning_budget: Option<usize>,
    overrides: Vec<(String, String)>,
    spec: Option<SpecOptions>,
    mmproj: Option<String>,
    image_tokens: usize,
    image_min_tokens: usize,
    gpu_share: u32,
    prompt: Option<String>,
}

/// Set by Ctrl-C while a reply is being generated (`GENERATING`).
static INTERRUPTED: AtomicBool = AtomicBool::new(false);
static GENERATING: AtomicBool = AtomicBool::new(false);

/// A reply's output, stopped by Ctrl-C.
struct Printer {
    show_thinking: bool,
    in_reasoning: bool,
}

impl ferrum::hybrid::runtime::ChatHooks for Printer {
    fn delta(&mut self, delta: Delta) -> Result<()> {
        let mut out = std::io::stdout();
        match delta {
            Delta::Reasoning(t) if self.show_thinking => {
                if !self.in_reasoning {
                    print!("\x1b[2m");
                    self.in_reasoning = true;
                }
                print!("{t}");
            }
            Delta::Reasoning(_) => {}
            Delta::Content(t) => {
                if self.in_reasoning {
                    print!("\x1b[0m\n\n");
                    self.in_reasoning = false;
                }
                print!("{t}");
            }
        }
        out.flush().ok();
        Ok(())
    }
    fn prefill(&mut self, _processed: usize, _total: usize) -> bool {
        !INTERRUPTED.load(Ordering::Relaxed)
    }
    fn cancelled(&self) -> bool {
        INTERRUPTED.load(Ordering::Relaxed)
    }
}

/// Where messages come from: one `--prompt`, a terminal with line editing
/// and history, or lines of a pipe.
enum Input {
    Once(Option<String>),
    Terminal(Box<rustyline::DefaultEditor>, Option<std::path::PathBuf>),
    Pipe(std::io::Lines<std::io::StdinLock<'static>>),
}

impl Input {
    fn new(prompt: Option<&str>) -> Result<Self> {
        let stdin = std::io::stdin();
        if let Some(prompt) = prompt {
            let mut text = prompt.to_owned();
            if !stdin.is_terminal() {
                let mut piped = String::new();
                stdin
                    .lock()
                    .read_to_string(&mut piped)
                    .map_err(|e| Error::Parameter(format!("stdin: {e}")))?;
                if !piped.trim().is_empty() {
                    text = format!("{text}\n\n{}", piped.trim_end());
                }
            }
            return Ok(Self::Once(Some(text)));
        }
        if !stdin.is_terminal() {
            return Ok(Self::Pipe(stdin.lock().lines()));
        }
        let mut editor = rustyline::DefaultEditor::new()
            .map_err(|e| Error::Parameter(format!("terminal: {e}")))?;
        let history = std::env::var_os("XDG_CACHE_HOME")
            .map(std::path::PathBuf::from)
            .or_else(|| std::env::var_os("HOME").map(|h| std::path::Path::new(&h).join(".cache")))
            .map(|dir| dir.join("ferrum").join("cli-history"));
        if let Some(path) = &history {
            let _ = editor.load_history(path);
        }
        Ok(Self::Terminal(Box::new(editor), history))
    }

    /// The next message (lines ending in `\` continue it); `None` at the end
    /// of the input or on Ctrl-C / Ctrl-D.
    fn message(&mut self) -> Result<Option<String>> {
        let mut input = String::new();
        loop {
            let line = match self {
                Self::Once(prompt) => return Ok(prompt.take()),
                Self::Terminal(editor, _) => {
                    let mark = if input.is_empty() {
                        "\n\x1b[1m>\x1b[0m "
                    } else {
                        "  "
                    };
                    match editor.readline(mark) {
                        Ok(line) => line,
                        Err(
                            rustyline::error::ReadlineError::Interrupted
                            | rustyline::error::ReadlineError::Eof,
                        ) => return Ok(None),
                        Err(e) => return Err(Error::Parameter(e.to_string())),
                    }
                }
                Self::Pipe(lines) => match lines.next() {
                    Some(line) => line.map_err(|e| Error::Parameter(e.to_string()))?,
                    None => return Ok(None),
                },
            };
            if let Some(stripped) = line.strip_suffix('\\') {
                input.push_str(stripped);
                input.push('\n');
                continue;
            }
            input.push_str(&line);
            break;
        }
        if let Self::Terminal(editor, history) = self
            && !input.trim().is_empty()
        {
            let _ = editor.add_history_entry(input.as_str());
            if let Some(path) = history {
                if let Some(dir) = path.parent() {
                    let _ = std::fs::create_dir_all(dir);
                }
                let _ = editor.save_history(path);
            }
        }
        Ok(Some(input))
    }
}

fn parse() -> Result<Options> {
    let mut o = Options {
        model: String::new(),
        context: None,
        kv_cache: KvCache::from_env(),
        system: None,
        max_tokens: 32768,
        think: true,
        effort: None,
        show_thinking: true,
        reasoning_budget: None,
        overrides: Vec::new(),
        spec: None,
        mmproj: None,
        image_tokens: ferrum::vision::DEFAULT_MAX_TOKENS,
        image_min_tokens: ferrum::vision::DEFAULT_MIN_TOKENS,
        gpu_share: ferrum::hybrid::runtime::gpu_share_from_env(),
        prompt: None,
    };
    let mut args = std::env::args().skip(1);
    while let Some(key) = args.next() {
        match key.as_str() {
            "--no-think" => o.think = false,
            "--hide-thinking" => o.show_thinking = false,
            "-V" | "--version" => {
                println!("ferrum-cli {}", env!("CARGO_PKG_VERSION"));
                std::process::exit(0);
            }
            "-h" | "--help" => {
                println!(
                    "{}",
                    include_str!("ferrum-cli.rs")
                        .lines()
                        .map_while(|l| l.strip_prefix("//!"))
                        .map(|l| l.strip_prefix(' ').unwrap_or(l))
                        .collect::<Vec<_>>()
                        .join("\n")
                );
                std::process::exit(0);
            }
            _ => {
                let value = args
                    .next()
                    .ok_or_else(|| Error::Parameter(format!("missing value for {key}")))?;
                let number = || {
                    value
                        .parse::<usize>()
                        .map_err(|_| Error::Parameter(format!("invalid {key}: {value}")))
                };
                if SpecOptions::apply_flag(&mut o.spec, &key, &value)? {
                    continue;
                }
                match key.as_str() {
                    "--model" | "-m" => o.model = value.clone(),
                    "--context" | "-c" => {
                        o.context = if value == "auto" {
                            None
                        } else {
                            Some(number()?)
                        }
                    }
                    "--kv-cache" => o.kv_cache = value.parse()?,
                    "--system" => o.system = Some(value.clone()),
                    "--mmproj" => o.mmproj = Some(value.clone()),
                    "--image-tokens" => o.image_tokens = number()?,
                    "--image-min-tokens" => o.image_min_tokens = number()?,
                    "--max-tokens" | "-n" => o.max_tokens = number()?,
                    "--reasoning-effort" => o.effort = Some(value.clone()),
                    "--reasoning-budget" => o.reasoning_budget = Some(number()?),
                    "--gpu-share" => {
                        o.gpu_share = ferrum::hybrid::runtime::parse_gpu_share(&value)?
                    }
                    "--prompt" | "-p" => o.prompt = Some(value.clone()),
                    "--temperature"
                    | "--top-p"
                    | "--top-k"
                    | "--min-p"
                    | "--presence-penalty"
                    | "--frequency-penalty"
                    | "--repetition-penalty"
                    | "--seed" => o.overrides.push((key.clone(), value.clone())),
                    _ => {
                        return Err(Error::Parameter(format!(
                            "unknown option {key} (see --help)"
                        )));
                    }
                }
            }
        }
    }
    if o.model.is_empty() {
        return Err(Error::Parameter(
            "usage: ferrum-cli --model FILE.gguf (see --help)".into(),
        ));
    }
    Ok(o)
}

fn main() {
    if let Err(e) = run() {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}

fn run() -> Result<()> {
    if std::env::args().nth(1).as_deref() == Some("image") {
        return image::run(std::env::args().skip(2));
    }
    let o = parse()?;
    eprintln!("loading {} ...", o.model);
    let mmproj = ferrum::vision::resolve_projector(o.model.as_ref(), o.mmproj.as_deref())?;
    let mut rt = Runtime::load_multimodal(
        &o.model,
        PlanOptions {
            context: o.context,
            kv_cache: o.kv_cache,
            ..Default::default()
        },
        o.spec.as_ref(),
        mmproj.as_deref(),
        o.image_tokens,
    )?;
    if let Some(vision) = rt.vision.as_mut() {
        vision.model.min_tokens = o.image_min_tokens;
        eprintln!(
            "vision: {} (up to {} tokens per image; /image PATH [low|high|N] attaches one)",
            vision.path.display(),
            vision.model.max_tokens
        );
    }
    if let Some(audio) = rt.audio.as_ref() {
        eprintln!(
            "audio: {} (WAV clips up to {} s; /audio PATH attaches one)",
            audio.path.display(),
            ferrum::vision::MAX_AUDIO_SECONDS
        );
    }
    if let Some(drafter) = rt.session.drafter() {
        eprintln!(
            "speculative decoding: {} ({} drafts per step, {:.2} GiB)",
            drafter.name(),
            drafter.max_drafts(),
            rt.draft_loaded_bytes as f64 / (1u64 << 30) as f64
        );
    }
    rt.warmup()?;
    rt.set_gpu_share(o.gpu_share);
    if o.gpu_share < 100 {
        eprintln!(
            "GPU share {}%: replies pause between steps so video and the desktop stay smooth",
            o.gpu_share
        );
    }
    // Ctrl-C stops the reply being written; with none, it leaves.
    let _ = ctrlc::set_handler(|| {
        if GENERATING.load(Ordering::Relaxed) {
            INTERRUPTED.store(true, Ordering::Relaxed);
        } else {
            println!("\x1b[0m");
            std::process::exit(130);
        }
    });
    let mut sampling = rt.default_sampling.clone();
    for (key, value) in &o.overrides {
        let f = || {
            value
                .parse::<f32>()
                .map_err(|_| Error::Parameter(format!("invalid {key}: {value}")))
        };
        match key.as_str() {
            "--temperature" => sampling.temperature = f()?,
            "--top-p" => sampling.top_p = f()?,
            "--top-k" => sampling.top_k = f()? as usize,
            "--min-p" => sampling.min_p = f()?,
            "--presence-penalty" => sampling.presence_penalty = f()?,
            "--frequency-penalty" => sampling.frequency_penalty = f()?,
            "--repetition-penalty" => sampling.repetition_penalty = f()?,
            "--seed" => sampling.seed = f()? as u64,
            _ => {}
        }
    }
    sampling.validate()?;
    let plan = &rt.loaded.plan;
    eprintln!(
        "{}: context {} tokens (max {}, {} K/V), {:.1} GiB weights, loaded in {:.1} s",
        rt.model_name,
        plan.context,
        plan.max_context,
        rt.loaded.model.config.kv_cache,
        plan.weights as f64 / (1u64 << 30) as f64,
        rt.loaded.load_time.as_secs_f64()
    );
    eprintln!(
        "sampling: temperature {} top-p {} top-k {} min-p {}; /reset /think on|off /stats /exit",
        sampling.temperature, sampling.top_p, sampling.top_k, sampling.min_p
    );
    let mut messages: Vec<Json> = Vec::new();
    if let Some(system) = &o.system {
        messages.push(json!({"role": "system", "content": system}));
    }
    let mut think = o.think;
    let mut attachments: Vec<Json> = Vec::new();
    let mut detail = ImageBudget::default();
    let mut source = Input::new(o.prompt.as_deref())?;
    let mut last_stats = String::from("no turns yet");
    loop {
        let Some(input) = source.message()? else {
            return Ok(());
        };
        let input = input.trim();
        match input {
            "" if attachments.is_empty() => continue,
            "/exit" | "/quit" => return Ok(()),
            "/reset" => {
                attachments.clear();
                messages.retain(|m| m["role"] == "system");
                rt.session.reset();
                eprintln!("(conversation cleared)");
                continue;
            }
            "/stats" => {
                eprintln!(
                    "{last_stats}; session {} / {} tokens",
                    rt.session.len(),
                    rt.session.capacity()
                );
                continue;
            }
            "/think on" | "/think off" => {
                think = input.ends_with("on");
                eprintln!("(thinking {})", if think { "on" } else { "off" });
                continue;
            }
            _ => {}
        }
        if let Some(value) = input.strip_prefix("/detail ") {
            match media::parse_detail(value) {
                Ok(budget) => {
                    detail = budget;
                    eprintln!("(images from now on: {})", value.trim());
                }
                Err(e) => eprintln!("{e}"),
            }
            continue;
        }
        if let Some(path) = input.strip_prefix("/audio ") {
            let path = path.trim();
            match std::fs::read(path) {
                Ok(bytes) if rt.supports_audio() => match ferrum::vision::clip_seconds(&bytes) {
                    Ok(seconds) => {
                        attachments.push(media::audio_part_from_bytes(&bytes));
                        eprintln!(
                            "(audio attached: {seconds:.1} s; it goes with your next message)"
                        );
                    }
                    Err(e) => eprintln!("{e}"),
                },
                Ok(_) => eprintln!("this model was loaded without an audio projector (--mmproj)"),
                Err(e) => eprintln!("cannot read {path}: {e}"),
            }
            continue;
        }
        if let Some(arg) = input.strip_prefix("/image ") {
            let (path, budget) = image_arg(arg);
            match std::fs::read(path) {
                Ok(bytes) if rt.supports_vision() => {
                    attachments.push(media::image_part_with(&bytes, budget));
                    let tokens = rt
                        .vision
                        .as_ref()
                        .map_or(0, |v| v.model.limits(budget.or(detail)).1);
                    eprintln!(
                        "(image attached: {} bytes, up to {tokens} tokens; it goes with your next message)",
                        bytes.len()
                    );
                }
                Ok(_) => eprintln!("this model was loaded without a vision projector (--mmproj)"),
                Err(e) => eprintln!("cannot read {path}: {e}"),
            }
            continue;
        }
        if attachments.is_empty() {
            messages.push(json!({"role": "user", "content": input}));
        } else {
            // An empty line sends the attachments alone (a clip to transcribe).
            let mut parts = std::mem::take(&mut attachments);
            if !input.is_empty() {
                parts.push(json!({"type": "text", "text": input}));
            }
            messages.push(json!({"role": "user", "content": parts}));
        }
        let mut vars = Map::new();
        vars.insert("enable_thinking".into(), Json::Bool(think));
        if let Some(effort) = &o.effort {
            vars.insert("reasoning_effort".into(), Json::String(effort.clone()));
        }
        let request = ChatRequest {
            messages: messages.clone(),
            tools: Vec::new(),
            max_tokens: o.max_tokens,
            sampling: sampling.clone(),
            template_vars: vars,
            stop: Vec::new(),
            reasoning_budget: o.reasoning_budget,
            raw_reasoning: false,
            image_budget: detail,
            grammar: None,
        };
        INTERRUPTED.store(false, Ordering::Relaxed);
        GENERATING.store(true, Ordering::Relaxed);
        let result = rt.chat(
            &request,
            Printer {
                show_thinking: o.show_thinking,
                in_reasoning: false,
            },
        );
        GENERATING.store(false, Ordering::Relaxed);
        print!("\x1b[0m");
        let result = match result {
            Ok(r) => r,
            Err(e) => {
                messages.pop();
                eprintln!("\nerror: {e}");
                continue;
            }
        };
        println!();
        let c = &result.completion;
        last_stats = format!(
            "prompt {} tokens ({} reused), prefill {:.0} tok/s; {} generated at {:.1} tok/s; stop {:?}",
            c.prompt_tokens,
            c.reused_tokens,
            (c.prompt_tokens - c.reused_tokens) as f64 / c.prefill.as_secs_f64().max(1e-9),
            c.tokens.len(),
            c.tokens.len() as f64 / c.decode.as_secs_f64().max(1e-9),
            c.stop
        );
        if let Some(vision) = rt.vision.as_ref()
            && !vision.last_encode.is_zero()
        {
            last_stats += &format!("; image encode {:.2} s", vision.last_encode.as_secs_f64());
        }
        if let Some(audio) = rt.audio.as_ref()
            && !audio.last_encode.is_zero()
        {
            last_stats += &format!("; audio encode {:.2} s", audio.last_encode.as_secs_f64());
        }
        let spec = &rt.session.spec_stats;
        if spec.steps > 0 {
            last_stats += &format!(
                "; speculative acceptance {:.2} ({} of {} drafts)",
                spec.acceptance_length(),
                spec.accepted,
                spec.drafted
            );
        }
        eprintln!("\x1b[2m[{last_stats}]\x1b[0m");
        let mut reply = json!({"role": "assistant", "content": result.output.content});
        if let Some(reasoning) = result.output.reasoning {
            reply["reasoning_content"] = Json::String(reasoning);
        }
        messages.push(reply);
    }
}

/// `/image`'s argument: a path, optionally followed by the image's detail
/// (`low`, `high`, `auto` or a token cap) when the whole text is not a file.
fn image_arg(arg: &str) -> (&str, ImageBudget) {
    fn unquote(p: &str) -> &str {
        p.trim().trim_matches('\'')
    }
    let arg = arg.trim();
    if !std::path::Path::new(unquote(arg)).exists()
        && let Some((path, size)) = arg.rsplit_once(' ')
        && let Ok(budget) = media::parse_detail(size)
    {
        return (unquote(path), budget);
    }
    (unquote(arg), ImageBudget::default())
}
