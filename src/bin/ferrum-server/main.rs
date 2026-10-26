//! OpenAI- and Anthropic-compatible local HTTP server for Qwen3.5-family GGUFs
//! (and, experimentally, Flint and Spark).
//!
//! The model runs on its own thread behind a FIFO queue; every connection
//! gets a thread, so health and status endpoints answer while a request is
//! generating. Up to `--parallel` requests run at once, one session (slot)
//! each, taking turns on the GPU. A slot keeps its conversation, so a request
//! that extends it only prefills the new suffix.
mod anthropic;
mod http;
mod images;
mod log;
mod openai;
mod worker;

use ferrum::diffusion::pipeline;
use ferrum::hybrid::{
    self,
    chat::ChatTemplate,
    plan::PlanOptions,
    runtime::{SpecOptions, recommended_sampling_for},
    session::SamplingParams,
};
use ferrum::tokenizer::Tokenizer;
use serde_json::{Map, Value as Json, json};
use std::{
    io::BufReader,
    net::{TcpListener, TcpStream},
    path::PathBuf,
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering},
        mpsc::{Receiver, Sender, channel},
    },
    time::Instant,
};
use worker::{Event, Job, Route, Shared, Status, Work};

const HELP: &str = "\
ferrum-server: OpenAI/Anthropic-compatible server for Qwen3.5-family GGUF models
(experimental: Flint, Spark)

usage: ferrum-server --model FILE.gguf [--image-model DIR] [options]
       ferrum-server --image-model DIR [options]       (images only)

server:
  --host ADDR               bind address (default 127.0.0.1)
  --port N                  port (default 8080)
  --api-key KEY             require `Authorization: Bearer KEY` or `x-api-key: KEY`
                            (or set FERRUM_API_KEY, which other processes can't read)
  --cors ORIGIN             let web pages from ORIGIN call the server (repeatable;
                            `*` for any). Default: only localhost pages
  --alias NAME              model name reported to clients (default: file name)
  -np, --parallel N         requests served at once, each in its own session with its
                            own context-sized cache (1-8, default 1); requests go to
                            the session whose cache holds most of their prompt
  -v, --verbose             debug logging (request bodies, outputs, first-token times)
memory:
  -c, --context N|auto      context length of each session (default auto: largest that
                            fits --parallel sessions)
  --kv-cache f16|q8         K/V cache format; q8 nearly halves its memory, so auto
                            context grows, and decodes faster at long context
                            (default f16, or FERRUM_KV_CACHE). Also -ctk/-ctv.
  --reserve-mib N           GPU memory left free for the system (default 1024)
  --snapshots N             recurrent snapshots kept for prefix reuse (default 2)
  --chunk N                 prompt tokens per forward pass (default 512)
speculative decoding (lossless; off by default):
  --draft mtp|lookup|DIR|FILE
                            drafter: the GGUF's MTP block, prompt lookup (copies
                            spans from the context; no drafter model), a DFlash /
                            DFlash2 / DSpark checkpoint directory (or its HF cache
                            entry), or a DFlash / DFlash2 GGUF
  --draft-max N             drafts per step, and draft-tree depth (default: 3 for MTP
                            and lookup, 7 for MTP trees; the trained block for dense
                            targets, 2 for MoE targets)
  --draft-tree N|auto       draft trees of up to N guesses per step
                            (1-31; 15 recommended); auto: KFlash picks the guesses
                            per cycle, learned and cached (DFlash2: block 16,
                            draft vocab 98304)
  --draft-block N           DFlash/DFlash2: draft N rows per pass
  --draft-vocab N           draft among the first N token ids
  --draft-tree-search best-first|siblings
                            how MTP grows trees (default best-first)
  --draft-quant q4_0|q8_0   draft weight precision (default q4_0)
  --draft-context N         context slots for drafts without a sliding window (8192)
  --draft-p-min P           DSpark: stop drafting below this confidence (0)
vision (images in chat requests as data: URLs / base64; Qwen3-VL style projectors):
  --mmproj FILE|auto|none   vision projector GGUF (default auto: a sibling mmproj*.gguf)
  --image-tokens N          most context tokens one image may take (default 1024)
  --image-min-tokens N      scale small images up to at least N tokens (default 64;
                            Qwen-VL reads fine print and grounds better with 1024)
                            (requests may lower both per image: `detail` low|high|N,
                            `min_tokens`/`max_tokens`, or image_max_tokens per request)
image generation (Qwen-Image-2.1 directories; each request needs about 15 GB free, 8 GB
with --image-text-quantize q8_0, and models load for the request and are freed after it):
  --image-model DIR         serve /v1/images/generations from this directory
  --image-alias NAME        model name reported for it (default: the directory name)
  --image-quantize bf16|q8_0|q4_0       quantize the transformer as it loads (default bf16)
  --image-text-quantize bf16|q8_0|q4_0  the same for the text encoder (default bf16)
  --image-text-file FILE.gguf           take the text encoder from a Qwen3-VL GGUF instead (keeps its quantization)
  --image-transformer-file FILE.gguf    take the transformer from a GGUF instead (keeps its quantization)
generation defaults (requests may override):
  --temp, --top-p, --top-k, --min-p, --presence-penalty, --frequency-penalty,
  --repeat-penalty X        sampling (default: the GGUF's recommended values)
  --max-tokens N            per-request cap when the client sends none (default: context)
thinking defaults (requests may override):
  --no-think                disable thinking (enable_thinking=false)
  --reasoning-effort LEVEL  template reasoning effort (low, medium, high, ...)
  --reasoning-budget N      force </think> after N reasoning tokens (-1 = unlimited, 0 = off)
  --reasoning-format FMT    deepseek (reasoning_content, default) | none (inline <think>)
  --chat-template-kwargs J  JSON object of extra chat-template variables

endpoints: /v1/chat/completions, /v1/completions, /v1/messages,
  /v1/messages/count_tokens, /v1/images/generations, /v1/models, /health, /props,
  /slots, /metrics, /tokenize, /detokenize, /apply-template
";

pub struct Defaults {
    pub sampling: SamplingParams,
    pub template_vars: Map<String, Json>,
    pub reasoning_budget: Option<usize>,
    pub raw_reasoning: bool,
    pub max_tokens: usize,
}

/// The chat model: its thread's queue and what requests are rendered with.
pub struct Llm {
    jobs: Mutex<Sender<Job>>,
    pub tokenizer: Tokenizer,
    pub template: ChatTemplate,
    pub defaults: Defaults,
    model_path: PathBuf,
}

pub struct Server {
    pub shared: Arc<Shared>,
    /// Absent when started with only `--image-model`.
    llm: Option<Llm>,
    pub images: Option<images::Service>,
    name: String,
    api_key: Option<String>,
    /// Extra origins web pages may call from (`--cors`).
    cors: Vec<String>,
    /// Bound to a loopback address: only local `Host` names are served.
    loopback: bool,
    next_id: AtomicU64,
    started: Instant,
}

impl Server {
    pub fn model_name(&self) -> String {
        self.name.clone()
    }

    /// The chat model. Chat routes are refused up front when there is none
    /// (`handle`), so handlers can rely on it.
    pub fn llm(&self) -> &Llm {
        self.llm
            .as_ref()
            .expect("chat routes are guarded in handle")
    }

    pub fn submit(
        &self,
        id: u64,
        work: Work,
        route: Route,
        client: &TcpStream,
    ) -> (Receiver<Event>, Arc<AtomicBool>) {
        let (tx, rx) = channel();
        let cancel = Arc::new(AtomicBool::new(false));
        http::watch_hangup(client, id, &cancel);
        let status = self.shared.status();
        let queued = self.shared.queued.fetch_add(1, Ordering::SeqCst);
        // Requests that must start or finish before a slot is free for this one.
        let ahead = (queued + status.busy() + 1).saturating_sub(status.slots.len().max(1));
        if ahead > 0 {
            info!(Some(id), "queued behind {ahead} request(s)");
        }
        let job = Job {
            id,
            work,
            route,
            events: tx,
            cancel: cancel.clone(),
            enqueued: Instant::now(),
        };
        if let Ok(jobs) = self.llm().jobs.lock()
            && jobs.send(job).is_err()
        {
            error!(Some(id), "model thread is gone");
        }
        (rx, cancel)
    }
}

pub fn error_json(message: &str, kind: &str) -> Json {
    let code = match kind {
        "invalid_request_error" => 400,
        "authentication_error" => 401,
        "permission_error" => 403,
        "not_found_error" => 404,
        "unavailable_error" => 503,
        _ => 500,
    };
    json!({"error": {"message": message, "type": kind, "code": code}})
}

struct Options {
    model: PathBuf,
    host: String,
    port: u16,
    plan: PlanOptions,
    spec: Option<SpecOptions>,
    api_key: Option<String>,
    cors: Vec<String>,
    alias: Option<String>,
    verbose: bool,
    parallel: usize,
    sampling: Vec<(String, f32)>,
    max_tokens: usize,
    vars: Map<String, Json>,
    reasoning_budget: Option<usize>,
    raw_reasoning: bool,
    mmproj: Option<String>,
    image_tokens: usize,
    image_min_tokens: usize,
    image_model: Option<String>,
    image_alias: Option<String>,
    image_settings: ferrum::diffusion::pipeline::Settings,
}

fn parse() -> Result<Options, String> {
    let mut o = Options {
        model: PathBuf::new(),
        host: "127.0.0.1".into(),
        port: 8080,
        plan: PlanOptions::default(),
        spec: None,
        api_key: None,
        cors: Vec::new(),
        alias: None,
        verbose: false,
        parallel: 1,
        sampling: Vec::new(),
        max_tokens: usize::MAX,
        vars: Map::new(),
        reasoning_budget: None,
        raw_reasoning: false,
        mmproj: None,
        image_tokens: ferrum::vision::DEFAULT_MAX_TOKENS,
        image_min_tokens: ferrum::vision::DEFAULT_MIN_TOKENS,
        image_model: None,
        image_alias: None,
        image_settings: ferrum::diffusion::pipeline::Settings::default(),
    };
    let mut args = std::env::args().skip(1);
    while let Some(key) = args.next() {
        match key.as_str() {
            "-h" | "--help" => {
                print!("{HELP}");
                std::process::exit(0);
            }
            "-v" | "--verbose" => {
                o.verbose = true;
                continue;
            }
            "--no-think" => {
                o.vars.insert("enable_thinking".into(), Json::Bool(false));
                continue;
            }
            _ => {}
        }
        let value = args.next().ok_or(format!("missing value for {key}"))?;
        let bad = || format!("invalid value for {key}: {value}");
        let number = || value.parse::<usize>().map_err(|_| bad());
        if SpecOptions::apply_flag(&mut o.spec, &key, &value).map_err(|e| e.to_string())? {
            continue;
        }
        match key.as_str() {
            "-m" | "--model" => o.model = value.clone().into(),
            "--host" => o.host = value.clone(),
            "--port" => o.port = value.parse().map_err(|_| bad())?,
            "--api-key" => o.api_key = Some(value.clone()),
            "--cors" => o.cors.push(value.trim_end_matches('/').to_owned()),
            "--alias" => o.alias = Some(value.clone()),
            "-np" | "--parallel" => {
                o.parallel = number()?;
                if !(1..=worker::MAX_PARALLEL).contains(&o.parallel) {
                    return Err(format!(
                        "--parallel takes 1..={} sessions",
                        worker::MAX_PARALLEL
                    ));
                }
            }
            "-c" | "--context" | "--ctx-size" => {
                o.plan.context = if value == "auto" || value == "0" {
                    None
                } else {
                    Some(number()?)
                }
            }
            "--kv-cache" | "-ctk" | "--cache-type-k" | "-ctv" | "--cache-type-v" => {
                o.plan.kv_cache = value.parse().map_err(|e: ferrum::Error| e.to_string())?
            }
            "--image-model" => o.image_model = Some(value.clone()),
            "--image-alias" => o.image_alias = Some(value.clone()),
            "--image-quantize" => {
                o.image_settings.transformer =
                    pipeline::parse_quantization(&value).map_err(|e| e.to_string())?
            }
            "--image-text-file" => o.image_settings.text_file = Some(value.clone().into()),
            "--image-transformer-file" => {
                o.image_settings.transformer_file = Some(value.clone().into())
            }
            "--image-text-quantize" => {
                o.image_settings.text =
                    pipeline::parse_quantization(&value).map_err(|e| e.to_string())?
            }
            "--mmproj" => o.mmproj = Some(value.clone()),
            "--image-tokens" => o.image_tokens = number()?,
            "--image-min-tokens" => o.image_min_tokens = number()?,
            "--reserve-mib" => o.plan.reserve = number()? << 20,
            "--snapshots" => o.plan.snapshots = number()?,
            "--chunk" | "--ubatch-size" => o.plan.chunk = number()?.max(1),
            "--max-tokens" | "-n" | "--n-predict" => {
                o.max_tokens = value
                    .parse::<i64>()
                    .map_err(|_| bad())
                    .map(|v| if v <= 0 { usize::MAX } else { v as usize })?
            }
            "--temp"
            | "--temperature"
            | "--top-p"
            | "--top-k"
            | "--min-p"
            | "--presence-penalty"
            | "--frequency-penalty"
            | "--repeat-penalty" => o
                .sampling
                .push((key.clone(), value.parse().map_err(|_| bad())?)),
            "--reasoning-effort" => {
                o.vars
                    .insert("reasoning_effort".into(), Json::String(value.clone()));
            }
            "--reasoning-budget" => {
                let b: i64 = value.parse().map_err(|_| bad())?;
                match b {
                    0 => {
                        o.vars.insert("enable_thinking".into(), Json::Bool(false));
                    }
                    b if b > 0 => o.reasoning_budget = Some(b as usize),
                    _ => o.reasoning_budget = None,
                }
            }
            "--reasoning-format" => {
                o.raw_reasoning = match value.as_str() {
                    "none" => true,
                    "deepseek" | "auto" | "deepseek-legacy" => false,
                    _ => return Err(bad()),
                }
            }
            "--chat-template-kwargs" => {
                let parsed: Map<String, Json> =
                    serde_json::from_str(&value).map_err(|e| format!("{key}: {e}"))?;
                o.vars.extend(parsed);
            }
            _ => return Err(format!("unknown option {key} (see --help)")),
        }
    }
    if o.model.as_os_str().is_empty() && o.image_model.is_none() {
        return Err("--model or --image-model is required (see --help)".into());
    }
    o.plan.parallel = o.parallel;
    if o.api_key.is_none() {
        o.api_key = std::env::var("FERRUM_API_KEY")
            .ok()
            .filter(|k| !k.is_empty());
    }
    Ok(o)
}

fn main() {
    let o = match parse() {
        Ok(o) => o,
        Err(e) => {
            eprintln!("error: {e}");
            std::process::exit(2);
        }
    };
    log::init(o.verbose);
    if let Err(e) = run(o) {
        error!(None, "{e}");
        std::process::exit(1);
    }
}

fn run(o: Options) -> Result<(), String> {
    let chat = !o.model.as_os_str().is_empty();
    info!(
        None,
        "ferrum-server {} starting: {}",
        env!("CARGO_PKG_VERSION"),
        if chat {
            o.model.display().to_string()
        } else {
            o.image_model.clone().unwrap_or_default()
        }
    );
    let image_name = o.image_model.as_ref().map(|dir| {
        o.image_alias.clone().unwrap_or_else(|| {
            let path = std::path::Path::new(dir);
            // A Hugging Face snapshot directory is named by its revision.
            let named = if path.parent().is_some_and(|p| p.ends_with("snapshots")) {
                path.parent().and_then(|p| p.parent())
            } else {
                Some(path)
            };
            named.and_then(|p| p.file_name()).map_or_else(
                || "image-model".into(),
                |s| {
                    s.to_string_lossy()
                        .trim_start_matches("models--")
                        .replace("--", "/")
                },
            )
        })
    });
    let name = match (&o.alias, chat) {
        (Some(alias), _) => alias.clone(),
        (None, true) => o
            .model
            .file_stem()
            .map_or_else(|| "model".into(), |s| s.to_string_lossy().into_owned()),
        (None, false) => image_name.clone().unwrap_or_default(),
    };
    let listener = TcpListener::bind((o.host.as_str(), o.port))
        .map_err(|e| format!("bind {}:{}: {e}", o.host, o.port))?;
    let shared = Arc::new(Shared {
        status: Mutex::new(Status {
            // With no chat model there is nothing to load.
            state: if chat { "loading" } else { "ready" },
            model: name.clone(),
            ..Status::default()
        }),
        queued: AtomicUsize::new(0),
    });
    let llm = if chat {
        Some(start_llm(&o, &shared)?)
    } else {
        None
    };
    let images = match (&o.image_model, image_name) {
        (Some(dir), Some(model)) => {
            info!(None, "image model {model}: {dir}");
            Some(images::Service::start(dir, o.image_settings, model)?)
        }
        _ => None,
    };
    let server = Arc::new(Server {
        shared,
        llm,
        images,
        name,
        api_key: o.api_key,
        loopback: o.host == "localhost"
            || o.host
                .trim_matches(|c| c == '[' || c == ']')
                .parse::<std::net::IpAddr>()
                .is_ok_and(|ip| ip.is_loopback()),
        cors: o.cors,
        next_id: AtomicU64::new(1),
        started: Instant::now(),
    });
    info!(
        None,
        "listening on http://{}:{} (OpenAI: /v1/chat/completions, /v1/images/generations; Anthropic: /v1/messages){}",
        o.host,
        o.port,
        if chat { "; model loading…" } else { "" }
    );
    for stream in listener.incoming() {
        let Ok(stream) = stream else { continue };
        let server = server.clone();
        let _ = std::thread::Builder::new()
            .name("conn".into())
            .spawn(move || connection(&server, stream));
    }
    Ok(())
}

/// Start the model thread for `--model` and gather what requests are rendered with.
fn start_llm(o: &Options, shared: &Arc<Shared>) -> Result<Llm, String> {
    let mut sampling = recommended_sampling_for(&o.model).map_err(|e| e.to_string())?;
    for (key, v) in &o.sampling {
        match key.as_str() {
            "--temp" | "--temperature" => sampling.temperature = *v,
            "--top-p" => sampling.top_p = *v,
            "--top-k" => sampling.top_k = *v as usize,
            "--min-p" => sampling.min_p = *v,
            "--presence-penalty" => sampling.presence_penalty = *v,
            "--frequency-penalty" => sampling.frequency_penalty = *v,
            "--repeat-penalty" => sampling.repetition_penalty = *v,
            _ => {}
        }
    }
    sampling.validate().map_err(|e| e.to_string())?;
    let tokenizer = hybrid::load_tokenizer(&o.model).map_err(|e| e.to_string())?;
    let template = hybrid::load_chat_template(&o.model)
        .map_err(|e| e.to_string())?
        .ok_or("the GGUF has no chat template")?;
    let mmproj = ferrum::vision::resolve_projector(&o.model, o.mmproj.as_deref())
        .map_err(|e| e.to_string())?;
    let jobs = worker::spawn(
        o.model.clone(),
        o.plan,
        o.spec.clone(),
        o.alias.clone(),
        worker::Vision {
            mmproj,
            max_tokens: o.image_tokens,
            min_tokens: o.image_min_tokens,
        },
        shared.clone(),
    );
    if !o.vars.is_empty() {
        info!(
            None,
            "default chat-template variables: {}",
            Json::Object(o.vars.clone())
        );
    }
    if let Some(b) = o.reasoning_budget {
        info!(None, "default reasoning budget: {b} tokens");
    }
    Ok(Llm {
        jobs: Mutex::new(jobs),
        tokenizer,
        template,
        defaults: Defaults {
            sampling,
            template_vars: o.vars.clone(),
            reasoning_budget: o.reasoning_budget,
            raw_reasoning: o.raw_reasoning,
            max_tokens: o.max_tokens,
        },
        model_path: o.model.clone(),
    })
}

/// Why a request from a web page is refused, if it is. Browsers send
/// `Origin` (command-line clients and SDKs do not): without `--cors`, only
/// pages on this machine may call the server, so a site you visit cannot
/// drive it. On a loopback address the `Host` must also be a local name,
/// which stops DNS rebinding (a site's own name pointed at 127.0.0.1).
fn cross_site(server: &Server, r: &http::Request) -> Option<String> {
    if server.loopback
        && let Some(host) = r.header("host")
        && !is_local(host)
    {
        return Some(format!("requests for host {host} are refused"));
    }
    let origin = r.header("origin")?.trim_end_matches('/');
    let allowed = server
        .cors
        .iter()
        .any(|o| o == "*" || o.eq_ignore_ascii_case(origin));
    if allowed
        || origin
            .split_once("://")
            .is_some_and(|(_, host)| is_local(host))
    {
        return None;
    }
    Some(format!(
        "web page at {origin} refused; start the server with --cors {origin} to allow it"
    ))
}

/// `localhost`, `127.0.0.1` or `[::1]`, with or without a port.
fn is_local(host: &str) -> bool {
    let name = match host.strip_prefix('[') {
        Some(v6) => v6.split(']').next().unwrap_or(""),
        None => host.split(':').next().unwrap_or(""),
    }
    .to_ascii_lowercase();
    name == "localhost" || name.ends_with(".localhost") || name == "127.0.0.1" || name == "::1"
}

fn connection(server: &Server, mut stream: TcpStream) {
    http::configure(&stream);
    let peer = stream
        .peer_addr()
        .map_or_else(|_| "?".into(), |a| a.to_string());
    let Ok(read_half) = stream.try_clone() else {
        return;
    };
    let mut reader = BufReader::new(read_half);
    loop {
        let request = match http::read_request(&mut reader) {
            Ok(Some(r)) => r,
            Ok(None) => return,
            Err(e) => {
                if e.kind() != std::io::ErrorKind::WouldBlock
                    && e.kind() != std::io::ErrorKind::TimedOut
                {
                    debug!(None, "{peer}: bad request: {e}");
                    let _ = http::json(
                        &mut stream,
                        400,
                        &error_json(&e.to_string(), "invalid_request_error"),
                        false,
                    );
                }
                return;
            }
        };
        let keep_alive = request.keep_alive;
        if handle(server, &mut stream, &request, &peer).is_err() || !keep_alive {
            return;
        }
    }
}

fn handle(
    server: &Server,
    stream: &mut TcpStream,
    r: &http::Request,
    peer: &str,
) -> std::io::Result<()> {
    let ka = r.keep_alive;
    if let Some(why) = cross_site(server, r) {
        warn!(None, "{} {} from {peer}: {why}", r.method, r.path);
        return http::json(stream, 403, &error_json(&why, "permission_error"), ka);
    }
    if r.method == "OPTIONS" {
        return http::respond(stream, 204, "text/plain", b"", ka);
    }
    let path = r.path.trim_end_matches('/');
    let quiet = matches!(
        path,
        "/health" | "/v1/health" | "/metrics" | "/slots" | "/props" | "/v1/models" | "/models"
    );
    let id = if quiet {
        0
    } else {
        server.next_id.fetch_add(1, Ordering::SeqCst)
    };
    if quiet {
        debug!(None, "{} {} from {peer}", r.method, r.path);
    } else {
        info!(
            Some(id),
            "{} {} from {peer} ({:.1} KiB){}",
            r.method,
            r.path,
            r.body.len() as f64 / 1024.,
            r.header("user-agent").map_or(String::new(), |ua| format!(
                " [{}]",
                worker::truncate(ua, 60)
            ))
        );
    }
    if let Some(key) = &server.api_key {
        let bearer = r
            .header("authorization")
            .and_then(|v| v.strip_prefix("Bearer "))
            .map(str::trim);
        let ok = bearer == Some(key.as_str()) || r.header("x-api-key") == Some(key.as_str());
        if !ok && !matches!(path, "/health" | "/v1/health") {
            warn!(Some(id), "rejected: missing or wrong API key");
            return http::json(
                stream,
                401,
                &error_json("invalid API key", "authentication_error"),
                ka,
            );
        }
    }
    let status = server.shared.status();
    let body = || -> Result<Json, String> {
        if r.body.is_empty() {
            return Ok(json!({}));
        }
        serde_json::from_slice(&r.body).map_err(|e| format!("invalid JSON body: {e}"))
    };
    if path == "/v1/images/generations" {
        return match (r.method.as_str(), body()) {
            ("POST", Ok(json_body)) => images::generations(server, stream, &json_body, id, ka),
            ("POST", Err(e)) => {
                http::json(stream, 400, &error_json(&e, "invalid_request_error"), ka)
            }
            _ => http::json(
                stream,
                405,
                &error_json("use POST", "invalid_request_error"),
                ka,
            ),
        };
    }
    let needs_model = matches!(
        path,
        "/v1/chat/completions"
            | "/chat/completions"
            | "/v1/completions"
            | "/completions"
            | "/completion"
            | "/v1/messages"
    );
    if (needs_model
        || matches!(
            path,
            "/v1/messages/count_tokens"
                | "/tokenize"
                | "/detokenize"
                | "/apply-template"
                | "/props"
        ))
        && server.llm.is_none()
    {
        return http::json(
            stream,
            404,
            &error_json(
                "this server has no chat model (start it with --model FILE.gguf)",
                "not_found_error",
            ),
            ka,
        );
    }
    if needs_model && status.state != "ready" {
        let message = match status.state {
            "error" => format!(
                "model failed to load: {}",
                status.load_error.unwrap_or_default()
            ),
            _ => "Loading model".into(),
        };
        warn!(Some(id), "rejected: {message}");
        return http::json(stream, 503, &error_json(&message, "unavailable_error"), ka);
    }
    if needs_model
        || matches!(
            path,
            "/v1/messages/count_tokens" | "/tokenize" | "/detokenize" | "/apply-template"
        )
    {
        let json_body = match body() {
            Ok(b) => b,
            Err(e) => {
                warn!(Some(id), "bad request: {e}");
                return http::json(stream, 400, &error_json(&e, "invalid_request_error"), ka);
            }
        };
        debug!(
            Some(id),
            "body: {}",
            worker::truncate(&json_body.to_string(), 4000)
        );
        return match path {
            "/v1/chat/completions" | "/chat/completions" => {
                openai::chat(server, stream, &json_body, id, ka)
            }
            "/v1/completions" | "/completions" | "/completion" => {
                openai::completion(server, stream, &json_body, id, ka)
            }
            "/v1/messages" => anthropic::messages(server, stream, &json_body, id, ka),
            "/v1/messages/count_tokens" => {
                anthropic::count_tokens(server, stream, &json_body, id, ka)
            }
            "/tokenize" => {
                let text = json_body
                    .get("content")
                    .and_then(Json::as_str)
                    .unwrap_or_default();
                match server.llm().tokenizer.encode(text) {
                    Ok(tokens) => http::json(stream, 200, &json!({"tokens": tokens}), ka),
                    Err(e) => http::json(
                        stream,
                        400,
                        &error_json(&e.to_string(), "invalid_request_error"),
                        ka,
                    ),
                }
            }
            "/detokenize" => {
                let ids: Vec<u32> = json_body
                    .get("tokens")
                    .and_then(Json::as_array)
                    .map(|a| {
                        a.iter()
                            .filter_map(|v| v.as_u64().map(|v| v as u32))
                            .collect()
                    })
                    .unwrap_or_default();
                match server.llm().tokenizer.decode(&ids) {
                    Ok(text) => http::json(stream, 200, &json!({"content": text}), ka),
                    Err(e) => http::json(
                        stream,
                        400,
                        &error_json(&e.to_string(), "invalid_request_error"),
                        ka,
                    ),
                }
            }
            _ => {
                // /apply-template
                let messages = json_body
                    .get("messages")
                    .and_then(Json::as_array)
                    .cloned()
                    .unwrap_or_default();
                let tools = json_body
                    .get("tools")
                    .and_then(Json::as_array)
                    .cloned()
                    .unwrap_or_default();
                let t = openai::thinking(&json_body, server);
                let mut vars = t.vars;
                hybrid::runtime::normalize_thinking(server.llm().template.source(), &mut vars);
                match server.llm().template.render(
                    &messages,
                    (!tools.is_empty()).then_some(&tools[..]),
                    true,
                    &vars,
                ) {
                    // The same prompt a chat request feeds the model.
                    Ok(mut prompt) => {
                        prompt.push_str(hybrid::runtime::generation_prefix(
                            &server.llm().template,
                            &vars,
                        ));
                        http::json(stream, 200, &json!({"prompt": prompt}), ka)
                    }
                    Err(e) => http::json(
                        stream,
                        400,
                        &error_json(&e.to_string(), "invalid_request_error"),
                        ka,
                    ),
                }
            }
        };
    }
    match (r.method.as_str(), path) {
        ("GET", "/health" | "/v1/health") => match status.state {
            "ready" => http::json(stream, 200, &json!({"status": "ok"}), ka),
            "error" => http::json(
                stream,
                500,
                &error_json(&status.load_error.unwrap_or_default(), "server_error"),
                ka,
            ),
            _ => http::json(
                stream,
                503,
                &error_json("Loading model", "unavailable_error"),
                ka,
            ),
        },
        ("GET", "/v1/models" | "/models") => http::json(
            stream,
            200,
            &json!({"object": "list", "data": models(server, &status)}),
            ka,
        ),
        ("GET", "/props") => {
            let d = &server.llm().defaults;
            http::json(
                stream,
                200,
                &json!({
                    "model_path": server.llm().model_path.display().to_string(),
                    "model_alias": server.name,
                    "n_ctx": status.context,
                    "total_slots": status.slots.len().max(1),
                    "chat_template": server.llm().template.source(),
                    "modalities": {"vision": status.vision, "audio": false},
                    "build_info": format!("ferrum {}", env!("CARGO_PKG_VERSION")),
                    "default_generation_settings": {
                        "n_ctx": status.context,
                        "params": {
                            "temperature": d.sampling.temperature, "top_p": d.sampling.top_p,
                            "top_k": d.sampling.top_k, "min_p": d.sampling.min_p,
                            "presence_penalty": d.sampling.presence_penalty,
                            "frequency_penalty": d.sampling.frequency_penalty,
                            "repeat_penalty": d.sampling.repetition_penalty,
                            "n_predict": if d.max_tokens == usize::MAX { -1 } else { d.max_tokens as i64 },
                            "reasoning_budget": d.reasoning_budget.map_or(-1, |b| b as i64),
                            "reasoning_format": if d.raw_reasoning { "none" } else { "deepseek" },
                            "chat_template_kwargs": d.template_vars,
                        }
                    },
                    "status": status,
                }),
                ka,
            )
        }
        ("GET", "/slots") => {
            let queued = server.shared.queued.load(Ordering::SeqCst);
            let slots: Vec<Json> = status
                .slots
                .iter()
                .map(|s| {
                    json!({
                        "id": s.id, "n_ctx": status.context, "is_processing": s.request.is_some(),
                        "id_task": s.request, "phase": s.phase, "n_cached_tokens": s.tokens,
                        "prompt_cache_key": s.key, "queued": queued
                    })
                })
                .collect();
            http::json(stream, 200, &Json::Array(slots), ka)
        }
        ("GET", "/metrics") => {
            let s = &status;
            let text = format!(
                "# TYPE ferrum_requests_total counter\nferrum_requests_total {}\n\
                 # TYPE ferrum_requests_cancelled_total counter\nferrum_requests_cancelled_total {}\n\
                 # TYPE ferrum_requests_failed_total counter\nferrum_requests_failed_total {}\n\
                 # TYPE ferrum_prompt_tokens_total counter\nferrum_prompt_tokens_total {}\n\
                 # TYPE ferrum_prompt_tokens_cached_total counter\nferrum_prompt_tokens_cached_total {}\n\
                 # TYPE ferrum_prompt_tokens_processed_total counter\nferrum_prompt_tokens_processed_total {}\n\
                 # TYPE ferrum_tokens_generated_total counter\nferrum_tokens_generated_total {}\n\
                 # TYPE ferrum_prompt_seconds_total counter\nferrum_prompt_seconds_total {:.3}\n\
                 # TYPE ferrum_generation_seconds_total counter\nferrum_generation_seconds_total {:.3}\n\
                 # TYPE ferrum_prompt_tokens_per_second gauge\nferrum_prompt_tokens_per_second {:.2}\n\
                 # TYPE ferrum_generation_tokens_per_second gauge\nferrum_generation_tokens_per_second {:.2}\n\
                 # TYPE ferrum_last_prompt_tokens_per_second gauge\nferrum_last_prompt_tokens_per_second {:.2}\n\
                 # TYPE ferrum_last_generation_tokens_per_second gauge\nferrum_last_generation_tokens_per_second {:.2}\n\
                 # TYPE ferrum_slots gauge\nferrum_slots {}\n\
                 # TYPE ferrum_requests_processing gauge\nferrum_requests_processing {}\n\
                 # TYPE ferrum_requests_queued gauge\nferrum_requests_queued {}\n\
                 # TYPE ferrum_cached_tokens gauge\nferrum_cached_tokens {}\n\
                 # TYPE ferrum_context_tokens gauge\nferrum_context_tokens {}\n\
                 # TYPE ferrum_memory_predicted_bytes gauge\nferrum_memory_predicted_bytes {}\n\
                 # TYPE ferrum_uptime_seconds gauge\nferrum_uptime_seconds {:.0}\n\
                 # TYPE ferrum_draft_tokens_total counter\nferrum_draft_tokens_total {}\n\
                 # TYPE ferrum_draft_tokens_accepted_total counter\nferrum_draft_tokens_accepted_total {}\n",
                s.requests,
                s.cancelled,
                s.failed,
                s.prompt_tokens,
                s.cached_tokens,
                s.prefilled_tokens,
                s.generated_tokens,
                s.prefill_seconds,
                s.decode_seconds,
                s.prefilled_tokens as f64 / s.prefill_seconds.max(1e-9),
                s.generated_tokens as f64 / s.decode_seconds.max(1e-9),
                s.last_prefill_tps,
                s.last_decode_tps,
                s.slots.len(),
                s.busy(),
                server.shared.queued.load(Ordering::SeqCst),
                s.cached(),
                s.context,
                s.memory_predicted_bytes,
                server.started.elapsed().as_secs_f64(),
                s.draft_tokens,
                s.draft_accepted_tokens
            );
            http::respond(
                stream,
                200,
                "text/plain; version=0.0.4",
                text.as_bytes(),
                ka,
            )
        }
        (_, "/v1/chat/completions" | "/v1/completions" | "/v1/messages") => http::json(
            stream,
            405,
            &error_json("use POST", "invalid_request_error"),
            ka,
        ),
        _ => {
            warn!(Some(id), "no route for {} {}", r.method, r.path);
            http::json(stream, 404, &error_json("not found", "not_found_error"), ka)
        }
    }
}

/// The `/v1/models` entries: the chat model, then the image model.
fn models(server: &Server, status: &Status) -> Vec<Json> {
    let mut data = Vec::new();
    if server.llm.is_some() {
        data.push(json!({
            "id": server.name, "object": "model", "created": openai::now(), "owned_by": "ferrum",
            "meta": {"n_ctx_train": status.context, "n_ctx": status.context, "status": status.state}
        }));
    }
    if let Some(images) = &server.images {
        data.push(json!({
            "id": images.model, "object": "model", "created": openai::now(), "owned_by": "ferrum",
            "meta": {"type": "image", "status": "ready"}
        }));
    }
    data
}

#[cfg(test)]
mod tests {
    use super::is_local;

    #[test]
    fn local_hosts() {
        for h in [
            "localhost",
            "localhost:8080",
            "127.0.0.1:8080",
            "[::1]:80",
            "app.localhost:3000",
        ] {
            assert!(is_local(h), "{h}");
        }
        for h in [
            "evil.com",
            "127.0.0.1.evil.com",
            "localhost.evil.com:80",
            "[::2]",
            "0.0.0.0",
            "null",
        ] {
            assert!(!is_local(h), "{h}");
        }
    }
}
