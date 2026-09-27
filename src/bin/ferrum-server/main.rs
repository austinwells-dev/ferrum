//! OpenAI- and Anthropic-compatible local HTTP server for Qwen3.5-family GGUFs.
//!
//! The model runs on its own thread behind a FIFO queue; every connection
//! gets a thread, so health and status endpoints answer while a request is
//! generating. The session keeps the previous conversation, so a request that
//! extends it only prefills the new suffix.
mod anthropic;
mod http;
mod log;
mod openai;
mod worker;

use ferrum::hybrid::{
    self, chat::ChatTemplate, plan::PlanOptions, runtime::recommended_sampling,
    session::SamplingParams,
};
use ferrum::{loader::gguf::GgufFile, tokenizer::Tokenizer};
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
use worker::{Event, Job, Shared, Status, Work};

const HELP: &str = "\
ferrum-server: OpenAI/Anthropic-compatible server for Qwen3.5-family GGUF models

usage: ferrum-server --model FILE.gguf [options]

server:
  --host ADDR               bind address (default 127.0.0.1)
  --port N                  port (default 8080)
  --api-key KEY             require `Authorization: Bearer KEY` or `x-api-key: KEY`
  --alias NAME              model name reported to clients (default: file name)
  -v, --verbose             debug logging (request bodies, outputs, first-token times)
memory:
  -c, --context N|auto      context length (default auto: largest that fits)
  --reserve-mib N           GPU memory left free for the system (default 1024)
  --snapshots N             recurrent snapshots kept for prefix reuse (default 2)
  --chunk N                 prompt tokens per forward pass (default 512)
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
  /v1/messages/count_tokens, /v1/models, /health, /props, /slots, /metrics,
  /tokenize, /detokenize, /apply-template
";

pub struct Defaults {
    pub sampling: SamplingParams,
    pub template_vars: Map<String, Json>,
    pub reasoning_budget: Option<usize>,
    pub raw_reasoning: bool,
    pub max_tokens: usize,
}

pub struct Server {
    pub shared: Arc<Shared>,
    jobs: Mutex<Sender<Job>>,
    pub tokenizer: Tokenizer,
    pub template: ChatTemplate,
    pub defaults: Defaults,
    name: String,
    model_path: PathBuf,
    api_key: Option<String>,
    next_id: AtomicU64,
    started: Instant,
}

impl Server {
    pub fn model_name(&self) -> String {
        self.name.clone()
    }

    pub fn submit(
        &self,
        id: u64,
        work: Work,
        client: &TcpStream,
    ) -> (Receiver<Event>, Arc<AtomicBool>) {
        let (tx, rx) = channel();
        let cancel = Arc::new(AtomicBool::new(false));
        http::watch_hangup(client, id, &cancel);
        let status = self.shared.status();
        let ahead = self.shared.queued.fetch_add(1, Ordering::SeqCst)
            + usize::from(status.busy_request.is_some());
        if ahead > 0 {
            info!(Some(id), "queued behind {ahead} request(s)");
        }
        let job = Job {
            id,
            work,
            events: tx,
            cancel: cancel.clone(),
            enqueued: Instant::now(),
        };
        if let Ok(jobs) = self.jobs.lock()
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
    api_key: Option<String>,
    alias: Option<String>,
    verbose: bool,
    sampling: Vec<(String, f32)>,
    max_tokens: usize,
    vars: Map<String, Json>,
    reasoning_budget: Option<usize>,
    raw_reasoning: bool,
}

fn parse() -> Result<Options, String> {
    let mut o = Options {
        model: PathBuf::new(),
        host: "127.0.0.1".into(),
        port: 8080,
        plan: PlanOptions::default(),
        api_key: None,
        alias: None,
        verbose: false,
        sampling: Vec::new(),
        max_tokens: usize::MAX,
        vars: Map::new(),
        reasoning_budget: None,
        raw_reasoning: false,
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
        match key.as_str() {
            "-m" | "--model" => o.model = value.clone().into(),
            "--host" => o.host = value.clone(),
            "--port" => o.port = value.parse().map_err(|_| bad())?,
            "--api-key" => o.api_key = Some(value.clone()),
            "--alias" => o.alias = Some(value.clone()),
            "-c" | "--context" | "--ctx-size" => {
                o.plan.context = if value == "auto" || value == "0" {
                    None
                } else {
                    Some(number()?)
                }
            }
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
    if o.model.as_os_str().is_empty() {
        return Err("--model is required (see --help)".into());
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
    info!(
        None,
        "ferrum-server {} starting: {}",
        env!("CARGO_PKG_VERSION"),
        o.model.display()
    );
    let file = GgufFile::open(&o.model).map_err(|e| e.to_string())?;
    let mut sampling = recommended_sampling(&file);
    drop(file);
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
    let name = o.alias.clone().unwrap_or_else(|| {
        o.model
            .file_stem()
            .map_or_else(|| "model".into(), |s| s.to_string_lossy().into_owned())
    });
    let listener = TcpListener::bind((o.host.as_str(), o.port))
        .map_err(|e| format!("bind {}:{}: {e}", o.host, o.port))?;
    let shared = Arc::new(Shared {
        status: Mutex::new(Status {
            state: "loading",
            model: name.clone(),
            busy_phase: "idle",
            ..Status::default()
        }),
        queued: AtomicUsize::new(0),
    });
    let jobs = worker::spawn(o.model.clone(), o.plan, o.alias.clone(), shared.clone());
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
    let server = Arc::new(Server {
        shared,
        jobs: Mutex::new(jobs),
        tokenizer,
        template,
        defaults: Defaults {
            sampling,
            template_vars: o.vars,
            reasoning_budget: o.reasoning_budget,
            raw_reasoning: o.raw_reasoning,
            max_tokens: o.max_tokens,
        },
        name,
        model_path: o.model,
        api_key: o.api_key,
        next_id: AtomicU64::new(1),
        started: Instant::now(),
    });
    info!(
        None,
        "listening on http://{}:{} (OpenAI: /v1/chat/completions, Anthropic: /v1/messages); model loading…",
        o.host,
        o.port
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
    let needs_model = matches!(
        path,
        "/v1/chat/completions"
            | "/chat/completions"
            | "/v1/completions"
            | "/completions"
            | "/completion"
            | "/v1/messages"
    );
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
                match server.tokenizer.encode(text) {
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
                match server.tokenizer.decode(&ids) {
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
                hybrid::runtime::normalize_thinking(server.template.source(), &mut vars);
                match server.template.render(
                    &messages,
                    (!tools.is_empty()).then_some(&tools[..]),
                    true,
                    &vars,
                ) {
                    Ok(prompt) => http::json(stream, 200, &json!({"prompt": prompt}), ka),
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
            &json!({"object": "list", "data": [{
                "id": server.name, "object": "model", "created": openai::now(), "owned_by": "ferrum",
                "meta": {"n_ctx_train": status.context, "n_ctx": status.context, "status": status.state}
            }]}),
            ka,
        ),
        ("GET", "/props") => {
            let d = &server.defaults;
            http::json(
                stream,
                200,
                &json!({
                    "model_path": server.model_path.display().to_string(),
                    "model_alias": server.name,
                    "n_ctx": status.context,
                    "total_slots": 1,
                    "chat_template": server.template.source(),
                    "modalities": {"vision": false, "audio": false},
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
        ("GET", "/slots") => http::json(
            stream,
            200,
            &json!([{
                "id": 0, "n_ctx": status.context, "is_processing": status.busy_request.is_some(),
                "id_task": status.busy_request, "phase": status.busy_phase,
                "n_cached_tokens": status.session_tokens, "queued": server.shared.queued.load(Ordering::SeqCst)
            }]),
            ka,
        ),
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
                 # TYPE ferrum_requests_processing gauge\nferrum_requests_processing {}\n\
                 # TYPE ferrum_requests_queued gauge\nferrum_requests_queued {}\n\
                 # TYPE ferrum_cached_tokens gauge\nferrum_cached_tokens {}\n\
                 # TYPE ferrum_context_tokens gauge\nferrum_context_tokens {}\n\
                 # TYPE ferrum_memory_predicted_bytes gauge\nferrum_memory_predicted_bytes {}\n\
                 # TYPE ferrum_uptime_seconds gauge\nferrum_uptime_seconds {:.0}\n",
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
                u8::from(s.busy_request.is_some()),
                server.shared.queued.load(Ordering::SeqCst),
                s.session_tokens,
                s.context,
                s.memory_predicted_bytes,
                server.started.elapsed().as_secs_f64()
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
