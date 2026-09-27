//! OpenAI-compatible local HTTP server for a Qwen3.5-family GGUF.
//!
//! ferrum-server --model FILE.gguf [--host 127.0.0.1] [--port 8080]
//!               [--context N|auto] [--snapshots N] [--api-key KEY]
//!
//! Endpoints: POST /v1/chat/completions (streaming and non-streaming, tools,
//! reasoning_content), GET /v1/models, GET /health. Requests are served one
//! at a time; the session keeps the previous conversation so a follow-up
//! request that extends it only prefills the new suffix.
use ferrum::{
    Error, Result,
    hybrid::{
        plan::PlanOptions,
        runtime::{ChatRequest, ChatResult, Delta, Runtime},
        session::SamplingParams,
    },
};
use serde_json::{Map, Value as Json, json};
use std::{
    io::{BufRead, BufReader, Read, Write},
    net::{TcpListener, TcpStream},
    time::{SystemTime, UNIX_EPOCH},
};

struct Options {
    model: String,
    host: String,
    port: u16,
    context: Option<usize>,
    snapshots: usize,
    api_key: Option<String>,
}

fn parse() -> Result<Options> {
    let mut o = Options {
        model: String::new(),
        host: "127.0.0.1".into(),
        port: 8080,
        context: None,
        snapshots: 2,
        api_key: None,
    };
    let mut args = std::env::args().skip(1);
    while let Some(key) = args.next() {
        if key == "-h" || key == "--help" {
            println!(
                "usage: ferrum-server --model FILE.gguf [--host 127.0.0.1] [--port 8080] [--context N|auto] [--snapshots 2] [--api-key KEY]"
            );
            std::process::exit(0);
        }
        let value = args
            .next()
            .ok_or_else(|| Error::Parameter(format!("missing value for {key}")))?;
        let bad = || Error::Parameter(format!("invalid {key}: {value}"));
        match key.as_str() {
            "--model" | "-m" => o.model = value.clone(),
            "--host" => o.host = value.clone(),
            "--port" => o.port = value.parse().map_err(|_| bad())?,
            "--context" | "-c" => {
                o.context = if value == "auto" {
                    None
                } else {
                    Some(value.parse().map_err(|_| bad())?)
                }
            }
            "--snapshots" => o.snapshots = value.parse().map_err(|_| bad())?,
            "--api-key" => o.api_key = Some(value.clone()),
            _ => return Err(Error::Parameter(format!("unknown option {key}"))),
        }
    }
    if o.model.is_empty() {
        return Err(Error::Parameter(
            "usage: ferrum-server --model FILE.gguf (see --help)".into(),
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
    let o = parse()?;
    eprintln!("loading {} ...", o.model);
    let mut rt = Runtime::load(
        &o.model,
        PlanOptions {
            context: o.context,
            snapshots: o.snapshots,
            ..Default::default()
        },
    )?;
    let plan = &rt.loaded.plan;
    eprintln!(
        "{}: context {} tokens (max {}); predicted {:.1} GiB of a {:.1} GiB budget",
        rt.model_name,
        plan.context,
        plan.max_context,
        plan.total() as f64 / (1u64 << 30) as f64,
        plan.budget as f64 / (1u64 << 30) as f64
    );
    let listener = TcpListener::bind((o.host.as_str(), o.port))
        .map_err(|e| Error::Parameter(format!("bind {}:{}: {e}", o.host, o.port)))?;
    eprintln!("listening on http://{}:{}/v1", o.host, o.port);
    let mut next_id = 0u64;
    for stream in listener.incoming() {
        let Ok(stream) = stream else { continue };
        next_id += 1;
        if let Err(e) = handle(&mut rt, stream, &o, next_id) {
            eprintln!("request {next_id}: {e}");
        }
    }
    Ok(())
}

struct Request {
    method: String,
    path: String,
    headers: Vec<(String, String)>,
    body: Vec<u8>,
}

fn read_request(stream: &TcpStream) -> std::io::Result<Request> {
    let mut reader = BufReader::new(stream);
    let mut line = String::new();
    reader.read_line(&mut line)?;
    let mut parts = line.split_whitespace();
    let method = parts.next().unwrap_or_default().to_owned();
    let path = parts
        .next()
        .unwrap_or_default()
        .split('?')
        .next()
        .unwrap_or_default()
        .to_owned();
    let mut headers = Vec::new();
    loop {
        let mut h = String::new();
        if reader.read_line(&mut h)? == 0 || h.trim().is_empty() {
            break;
        }
        if let Some((k, v)) = h.split_once(':') {
            headers.push((k.trim().to_ascii_lowercase(), v.trim().to_owned()));
        }
    }
    let length: usize = headers
        .iter()
        .find(|(k, _)| k == "content-length")
        .and_then(|(_, v)| v.parse().ok())
        .unwrap_or(0);
    let mut body = vec![0; length.min(256 << 20)];
    reader.read_exact(&mut body)?;
    Ok(Request {
        method,
        path,
        headers,
        body,
    })
}

fn respond(stream: &mut TcpStream, status: u16, body: &Json) -> std::io::Result<()> {
    let text = body.to_string();
    let reason = match status {
        200 => "OK",
        400 => "Bad Request",
        401 => "Unauthorized",
        404 => "Not Found",
        _ => "Internal Server Error",
    };
    write!(
        stream,
        "HTTP/1.1 {status} {reason}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nAccess-Control-Allow-Origin: *\r\nConnection: close\r\n\r\n{text}",
        text.len()
    )
}

fn error_body(message: &str, kind: &str) -> Json {
    json!({"error": {"message": message, "type": kind}})
}

fn handle(rt: &mut Runtime, mut stream: TcpStream, o: &Options, id: u64) -> Result<()> {
    let io = |e: std::io::Error| Error::Parameter(format!("connection: {e}"));
    let request = read_request(&stream).map_err(io)?;
    if request.method == "OPTIONS" {
        write!(
            stream,
            "HTTP/1.1 204 No Content\r\nAccess-Control-Allow-Origin: *\r\nAccess-Control-Allow-Headers: *\r\nAccess-Control-Allow-Methods: GET, POST, OPTIONS\r\nConnection: close\r\n\r\n"
        )
        .map_err(io)?;
        return Ok(());
    }
    if let Some(key) = &o.api_key {
        let ok = request.headers.iter().any(|(k, v)| {
            k == "authorization" && v.strip_prefix("Bearer ").is_some_and(|t| t == key)
        });
        if !ok {
            return respond(
                &mut stream,
                401,
                &error_body("invalid API key", "authentication_error"),
            )
            .map_err(io);
        }
    }
    match (request.method.as_str(), request.path.as_str()) {
        ("GET", "/health") => respond(&mut stream, 200, &json!({"status": "ok"})).map_err(io),
        ("GET", "/v1/models") => respond(
            &mut stream,
            200,
            &json!({"object": "list", "data": [{
                "id": rt.model_name, "object": "model", "owned_by": "ferrum",
                "context_length": rt.session.capacity()
            }]}),
        )
        .map_err(io),
        ("POST", "/v1/chat/completions") => {
            let body: Json = match serde_json::from_slice(&request.body) {
                Ok(b) => b,
                Err(e) => {
                    return respond(
                        &mut stream,
                        400,
                        &error_body(&format!("invalid JSON: {e}"), "invalid_request_error"),
                    )
                    .map_err(io);
                }
            };
            match chat_request(&body, rt) {
                Ok((chat, stream_mode)) => {
                    if stream_mode {
                        stream_chat(rt, stream, &chat, id)
                    } else {
                        match rt.chat(&chat, |_| Ok(())) {
                            Ok(result) => {
                                respond(&mut stream, 200, &completion_body(rt, &result, id))
                                    .map_err(io)
                            }
                            Err(e) => respond(
                                &mut stream,
                                500,
                                &error_body(&e.to_string(), "server_error"),
                            )
                            .map_err(io),
                        }
                    }
                }
                Err(e) => respond(
                    &mut stream,
                    400,
                    &error_body(&e.to_string(), "invalid_request_error"),
                )
                .map_err(io),
            }
        }
        _ => respond(
            &mut stream,
            404,
            &error_body("not found", "invalid_request_error"),
        )
        .map_err(io),
    }
}

fn chat_request(body: &Json, rt: &Runtime) -> Result<(ChatRequest, bool)> {
    let bad = |m: &str| Error::Parameter(m.to_owned());
    let messages = body["messages"]
        .as_array()
        .ok_or_else(|| bad("messages must be an array"))?
        .clone();
    if messages.is_empty() {
        return Err(bad("messages must not be empty"));
    }
    let tools = body
        .get("tools")
        .and_then(Json::as_array)
        .cloned()
        .unwrap_or_default();
    let d = &rt.default_sampling;
    let f = |key: &str, default: f32| {
        body.get(key)
            .and_then(Json::as_f64)
            .map_or(default, |v| v as f32)
    };
    let sampling = SamplingParams {
        temperature: f("temperature", d.temperature),
        top_p: f("top_p", d.top_p),
        top_k: body
            .get("top_k")
            .and_then(Json::as_u64)
            .map_or(d.top_k, |v| v as usize),
        min_p: f("min_p", d.min_p),
        presence_penalty: f("presence_penalty", d.presence_penalty),
        frequency_penalty: f("frequency_penalty", d.frequency_penalty),
        repetition_penalty: f("repetition_penalty", d.repetition_penalty),
        seed: body.get("seed").and_then(Json::as_u64).unwrap_or_else(|| {
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map_or(0, |t| t.as_nanos() as u64)
        }),
        ..d.clone()
    };
    sampling.validate()?;
    let room = rt.session.capacity();
    let max_tokens = body
        .get("max_completion_tokens")
        .or_else(|| body.get("max_tokens"))
        .and_then(Json::as_u64)
        .map_or(room, |v| v as usize);
    let stop = match body.get("stop") {
        Some(Json::String(s)) => vec![s.clone()],
        Some(Json::Array(a)) => a
            .iter()
            .filter_map(|v| v.as_str().map(str::to_owned))
            .collect(),
        _ => Vec::new(),
    };
    let mut template_vars: Map<String, Json> = body
        .get("chat_template_kwargs")
        .and_then(Json::as_object)
        .cloned()
        .unwrap_or_default();
    if let Some(effort) = body.get("reasoning_effort") {
        template_vars.insert("reasoning_effort".into(), effort.clone());
    }
    let stream = body.get("stream").and_then(Json::as_bool).unwrap_or(false);
    Ok((
        ChatRequest {
            messages,
            tools,
            max_tokens,
            sampling,
            template_vars,
            stop,
        },
        stream,
    ))
}

fn created() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |t| t.as_secs())
}

fn tool_calls_json(result: &ChatResult, id: u64) -> Vec<Json> {
    result
        .output
        .tool_calls
        .iter()
        .enumerate()
        .map(|(i, call)| {
            json!({
                "index": i,
                "id": format!("call_{id}_{i}"),
                "type": "function",
                "function": {"name": call.name, "arguments": Json::Object(call.arguments.clone()).to_string()}
            })
        })
        .collect()
}

fn finish_reason(result: &ChatResult) -> &'static str {
    if !result.output.tool_calls.is_empty() {
        "tool_calls"
    } else if result.stopped_by_string {
        "stop"
    } else {
        result.completion.stop.finish_reason()
    }
}

fn usage(result: &ChatResult) -> Json {
    let c = &result.completion;
    json!({
        "prompt_tokens": c.prompt_tokens,
        "completion_tokens": c.tokens.len(),
        "total_tokens": c.prompt_tokens + c.tokens.len(),
        "prompt_tokens_details": {"cached_tokens": c.reused_tokens},
        "timings": {
            "prompt_ms": c.prefill.as_secs_f64() * 1e3,
            "predicted_ms": c.decode.as_secs_f64() * 1e3,
            "predicted_per_second": c.tokens.len() as f64 / c.decode.as_secs_f64().max(1e-9)
        }
    })
}

fn completion_body(rt: &Runtime, result: &ChatResult, id: u64) -> Json {
    let mut message = json!({"role": "assistant", "content": result.output.content});
    if let Some(r) = &result.output.reasoning {
        message["reasoning_content"] = Json::String(r.clone());
    }
    let calls = tool_calls_json(result, id);
    if !calls.is_empty() {
        message["tool_calls"] = Json::Array(calls);
    }
    json!({
        "id": format!("chatcmpl-{id}"),
        "object": "chat.completion",
        "created": created(),
        "model": rt.model_name,
        "choices": [{"index": 0, "message": message, "finish_reason": finish_reason(result)}],
        "usage": usage(result)
    })
}

fn stream_chat(rt: &mut Runtime, mut stream: TcpStream, chat: &ChatRequest, id: u64) -> Result<()> {
    let io = |e: std::io::Error| Error::Parameter(format!("connection: {e}"));
    write!(
        stream,
        "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\nAccess-Control-Allow-Origin: *\r\nConnection: close\r\n\r\n"
    )
    .map_err(io)?;
    let model = rt.model_name.clone();
    let chunk = |delta: Json, finish: Option<&str>| {
        json!({
            "id": format!("chatcmpl-{id}"),
            "object": "chat.completion.chunk",
            "created": created(),
            "model": model,
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]
        })
    };
    let send = |stream: &mut TcpStream, value: &Json| -> Result<()> {
        write!(stream, "data: {value}\n\n").map_err(io)?;
        stream.flush().map_err(io)
    };
    send(
        &mut stream,
        &chunk(json!({"role": "assistant", "content": ""}), None),
    )?;
    let result = rt.chat(chat, |delta| {
        let d = match delta {
            Delta::Reasoning(t) => json!({"reasoning_content": t}),
            Delta::Content(t) => json!({"content": t}),
        };
        send(&mut stream, &chunk(d, None))
    });
    match result {
        Ok(result) => {
            let calls = tool_calls_json(&result, id);
            if !calls.is_empty() {
                send(&mut stream, &chunk(json!({"tool_calls": calls}), None))?;
            }
            let mut last = chunk(json!({}), Some(finish_reason(&result)));
            last["usage"] = usage(&result);
            send(&mut stream, &last)?;
        }
        Err(e) => {
            send(&mut stream, &error_body(&e.to_string(), "server_error"))?;
        }
    }
    write!(stream, "data: [DONE]\n\n").map_err(io)?;
    Ok(())
}
