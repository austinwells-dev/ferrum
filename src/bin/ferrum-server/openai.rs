//! OpenAI-compatible endpoints: /v1/chat/completions, /v1/completions.
use crate::{
    Server, error_json,
    http::{self, Chunked},
    info, warn,
    worker::{Event, Work},
};
use ferrum::hybrid::{
    runtime::{ChatRequest, ChatResult, Delta},
    session::{Completion, SamplingParams},
};
use serde_json::{Map, Value as Json, json};
use std::{
    net::TcpStream,
    sync::{Arc, atomic::AtomicBool, mpsc::Receiver},
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

pub fn now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |t| t.as_secs())
}

/// Thinking controls a request asked for, from every common spelling:
/// `chat_template_kwargs`, `enable_thinking`, `think` (Ollama),
/// `reasoning_effort` (OpenAI), `reasoning: {effort, enabled, max_tokens,
/// exclude}` (OpenRouter), `thinking: {type, budget_tokens}` (Anthropic),
/// `reasoning_budget` / `thinking_budget_tokens`, `reasoning_format`.
pub struct Thinking {
    pub vars: Map<String, Json>,
    pub budget: Option<usize>,
    pub raw: bool,
    /// Omit reasoning from the response (it is still generated).
    pub exclude: bool,
}

pub fn thinking(body: &Json, server: &Server) -> Thinking {
    let d = &server.defaults;
    let mut vars = d.template_vars.clone();
    if let Some(kwargs) = body.get("chat_template_kwargs").and_then(Json::as_object) {
        for (k, v) in kwargs {
            vars.insert(k.clone(), v.clone());
        }
    }
    let mut budget = d.reasoning_budget;
    let enable = |vars: &mut Map<String, Json>, on: bool| {
        vars.insert("enable_thinking".into(), Json::Bool(on));
    };
    for key in ["enable_thinking", "think"] {
        match body.get(key) {
            Some(Json::Bool(on)) => enable(&mut vars, *on),
            Some(Json::String(level)) => {
                // Ollama accepts "low" | "medium" | "high" for think.
                enable(&mut vars, true);
                vars.insert("reasoning_effort".into(), Json::String(level.clone()));
            }
            _ => {}
        }
    }
    if let Some(effort) = body.get("reasoning_effort").and_then(Json::as_str) {
        vars.insert("reasoning_effort".into(), Json::String(effort.into()));
    }
    let mut exclude = false;
    if let Some(r) = body.get("reasoning").and_then(Json::as_object) {
        if let Some(effort) = r.get("effort").and_then(Json::as_str) {
            vars.insert("reasoning_effort".into(), Json::String(effort.into()));
        }
        if let Some(on) = r.get("enabled").and_then(Json::as_bool) {
            enable(&mut vars, on);
        }
        if let Some(max) = r.get("max_tokens").and_then(Json::as_u64) {
            budget = Some(max as usize);
        }
        exclude = r.get("exclude").and_then(Json::as_bool).unwrap_or(false);
    }
    if let Some(t) = body.get("thinking").and_then(Json::as_object) {
        match t.get("type").and_then(Json::as_str) {
            Some("disabled") => enable(&mut vars, false),
            Some("enabled" | "adaptive") => enable(&mut vars, true),
            _ => {}
        }
        if let Some(b) = t.get("budget_tokens").and_then(Json::as_u64) {
            budget = Some(b as usize);
        }
    }
    for key in [
        "reasoning_budget",
        "thinking_budget_tokens",
        "thinking_budget",
    ] {
        if let Some(b) = body.get(key).and_then(Json::as_i64) {
            budget = (b >= 0).then_some(b as usize);
        }
    }
    if budget == Some(0) {
        enable(&mut vars, false);
        budget = None;
    }
    let raw = match body.get("reasoning_format").and_then(Json::as_str) {
        Some("none") => true,
        Some(_) => false,
        None => d.raw_reasoning,
    };
    Thinking {
        vars,
        budget,
        raw,
        exclude,
    }
}

/// Sampling from the request over the server defaults; llama-server field
/// names (`n_predict`, `repeat_penalty`, ...) are accepted too.
pub fn sampling(body: &Json, server: &Server) -> Result<SamplingParams, String> {
    let d = &server.defaults.sampling;
    let f = |keys: &[&str], default: f32| {
        keys.iter()
            .find_map(|k| body.get(*k).and_then(Json::as_f64))
            .map_or(default, |v| v as f32)
    };
    let seed = body.get("seed").and_then(Json::as_i64).filter(|&s| s >= 0);
    let s = SamplingParams {
        temperature: f(&["temperature"], d.temperature),
        top_p: f(&["top_p"], d.top_p),
        top_k: body
            .get("top_k")
            .and_then(Json::as_i64)
            .map_or(d.top_k, |v| v.max(0) as usize),
        min_p: f(&["min_p"], d.min_p),
        presence_penalty: f(&["presence_penalty"], d.presence_penalty),
        frequency_penalty: f(&["frequency_penalty"], d.frequency_penalty),
        repetition_penalty: f(
            &["repetition_penalty", "repeat_penalty"],
            d.repetition_penalty,
        ),
        penalty_last_n: body
            .get("repeat_last_n")
            .and_then(Json::as_u64)
            .map_or(d.penalty_last_n, |v| v as usize),
        seed: seed.map_or_else(
            || {
                SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .map_or(0, |t| t.as_nanos() as u64)
            },
            |s| s as u64,
        ),
    };
    s.validate().map_err(|e| e.to_string())?;
    Ok(s)
}

pub fn stop_strings(body: &Json, key: &str) -> Vec<String> {
    match body.get(key) {
        Some(Json::String(s)) => vec![s.clone()],
        Some(Json::Array(a)) => a
            .iter()
            .filter_map(|v| v.as_str().map(str::to_owned))
            .collect(),
        _ => Vec::new(),
    }
}

pub fn max_tokens(body: &Json, keys: &[&str], server: &Server) -> usize {
    keys.iter()
        .find_map(|k| body.get(*k).and_then(Json::as_i64))
        .filter(|&v| v > 0)
        .map_or(server.defaults.max_tokens, |v| v as usize)
}

fn chat_request(body: &Json, server: &Server, id: u64) -> Result<(ChatRequest, bool), String> {
    let messages = body
        .get("messages")
        .and_then(Json::as_array)
        .filter(|m| !m.is_empty())
        .ok_or("messages must be a non-empty array")?
        .clone();
    if body.get("n").and_then(Json::as_u64).is_some_and(|n| n > 1) {
        return Err("n > 1 is not supported".into());
    }
    let mut tools = body
        .get("tools")
        .and_then(Json::as_array)
        .cloned()
        .unwrap_or_default();
    match body.get("tool_choice") {
        Some(Json::String(c)) if c == "none" => tools.clear(),
        Some(Json::String(c)) if c == "required" => {
            warn!(
                Some(id),
                "tool_choice \"required\" is not enforced (no grammar); the model decides"
            )
        }
        Some(Json::Object(_)) => warn!(
            Some(id),
            "a forced tool_choice is not enforced; the model decides"
        ),
        _ => {}
    }
    for unsupported in ["response_format", "logprobs", "grammar", "json_schema"] {
        if body
            .get(unsupported)
            .is_some_and(|v| !v.is_null() && v != &json!(false) && v != &json!({"type": "text"}))
        {
            warn!(Some(id), "ignoring unsupported field {unsupported}");
        }
    }
    let t = thinking(body, server);
    let request = ChatRequest {
        messages,
        tools,
        max_tokens: max_tokens(
            body,
            &["max_completion_tokens", "max_tokens", "n_predict"],
            server,
        ),
        sampling: sampling(body, server)?,
        template_vars: t.vars,
        stop: stop_strings(body, "stop"),
        reasoning_budget: t.budget,
        raw_reasoning: t.raw,
    };
    Ok((request, t.exclude))
}

fn describe(request: &ChatRequest, exclude: bool, stream: bool) -> String {
    let s = &request.sampling;
    let mut thinking = match request.template_vars.get("enable_thinking") {
        Some(Json::Bool(false)) => "off".to_owned(),
        _ => "on".to_owned(),
    };
    if let Some(effort) = request
        .template_vars
        .get("reasoning_effort")
        .and_then(Json::as_str)
    {
        thinking.push_str(&format!(", effort {effort}"));
    }
    if let Some(b) = request.reasoning_budget {
        thinking.push_str(&format!(", budget {b}"));
    }
    if request.raw_reasoning {
        thinking.push_str(", inline");
    }
    if exclude {
        thinking.push_str(", excluded");
    }
    format!(
        "{} messages, {} tools, {}, max {} tok; temperature {} top-p {} top-k {} min-p {}; thinking {thinking}",
        request.messages.len(),
        request.tools.len(),
        if stream { "stream" } else { "no stream" },
        if request.max_tokens == usize::MAX {
            "∞".into()
        } else {
            crate::log::n(request.max_tokens)
        },
        s.temperature,
        s.top_p,
        s.top_k,
        s.min_p
    )
}

pub fn timings(c: &Completion) -> Json {
    let prefilled = c.prompt_tokens - c.reused_tokens;
    let pm = c.prefill.as_secs_f64() * 1e3;
    let dm = c.decode.as_secs_f64() * 1e3;
    json!({
        "cache_n": c.reused_tokens,
        "prompt_n": prefilled,
        "prompt_ms": pm,
        "prompt_per_token_ms": pm / prefilled.max(1) as f64,
        "prompt_per_second": prefilled as f64 / (pm / 1e3).max(1e-9),
        "predicted_n": c.tokens.len(),
        "predicted_ms": dm,
        "predicted_per_token_ms": dm / c.tokens.len().max(1) as f64,
        "predicted_per_second": c.tokens.len() as f64 / (dm / 1e3).max(1e-9),
    })
}

fn usage(c: &Completion) -> Json {
    json!({
        "prompt_tokens": c.prompt_tokens,
        "completion_tokens": c.tokens.len(),
        "total_tokens": c.prompt_tokens + c.tokens.len(),
        "prompt_tokens_details": {"cached_tokens": c.reused_tokens},
    })
}

fn tool_calls(result: &ChatResult, id: u64) -> Vec<Json> {
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

fn message(result: &ChatResult, id: u64, exclude: bool) -> Json {
    let mut m = json!({"role": "assistant", "content": result.output.content});
    if let Some(r) = result.output.reasoning.as_ref().filter(|_| !exclude) {
        m["reasoning_content"] = Json::String(r.clone());
    }
    let calls = tool_calls(result, id);
    if !calls.is_empty() {
        m["tool_calls"] = Json::Array(calls);
    }
    m
}

/// Wait for events, calling `idle(elapsed_since_last_event)` roughly every
/// second while nothing arrives. `idle` returning false aborts the wait.
pub fn next_event(rx: &Receiver<Event>, mut idle: impl FnMut() -> bool) -> Option<Event> {
    loop {
        match rx.recv_timeout(Duration::from_secs(1)) {
            Ok(e) => return Some(e),
            Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {
                if !idle() {
                    return None;
                }
            }
            Err(_) => return None,
        }
    }
}

pub fn chat(
    server: &Server,
    stream: &mut TcpStream,
    body: &Json,
    id: u64,
    keep_alive: bool,
) -> std::io::Result<()> {
    let (request, exclude) = match chat_request(body, server, id) {
        Ok(r) => r,
        Err(e) => {
            warn!(Some(id), "bad request: {e}");
            return http::json(
                stream,
                400,
                &error_json(&e, "invalid_request_error"),
                keep_alive,
            );
        }
    };
    let streaming = body.get("stream").and_then(Json::as_bool).unwrap_or(false);
    info!(Some(id), "chat: {}", describe(&request, exclude, streaming));
    let (rx, cancel) = server.submit(id, Work::Chat(Box::new(request)), stream);
    let include_usage = body
        .pointer("/stream_options/include_usage")
        .and_then(Json::as_bool)
        .unwrap_or(false);
    let progress = body
        .get("return_progress")
        .and_then(Json::as_bool)
        .unwrap_or(false);
    if streaming {
        chat_stream(
            server,
            stream,
            &rx,
            &cancel,
            id,
            keep_alive,
            exclude,
            include_usage,
            progress,
        )
    } else {
        wait_json(stream, &rx, &cancel, id, keep_alive, |event| match event {
            Event::Chat(result) => Some(json!({
                "id": format!("chatcmpl-{id}"),
                "object": "chat.completion",
                "created": now(),
                "model": server.model_name(),
                "choices": [{"index": 0, "message": message(&result, id, exclude), "finish_reason": finish_reason(&result)}],
                "usage": usage(&result.completion),
                "timings": timings(&result.completion),
            })),
            _ => None,
        })
    }
}

/// Non-streaming reply. Headers wait for the result so errors keep their
/// status code; after 5 s a chunked body starts and a space (valid leading
/// JSON whitespace) is sent every 5 s so client idle timeouts do not fire.
pub fn wait_json(
    stream: &mut TcpStream,
    rx: &Receiver<Event>,
    cancel: &Arc<AtomicBool>,
    id: u64,
    keep_alive: bool,
    mut finish: impl FnMut(Event) -> Option<Json>,
) -> std::io::Result<()> {
    let start = Instant::now();
    let mut body: Option<Chunked> = None;
    let mut last_ping = Instant::now();
    let mut tick = |body: &mut Option<Chunked>| -> std::io::Result<()> {
        if start.elapsed() < Duration::from_secs(5) || last_ping.elapsed() < Duration::from_secs(5)
        {
            return Ok(());
        }
        last_ping = Instant::now();
        if body.is_none() {
            *body = Some(Chunked::start(
                stream,
                200,
                "application/json; charset=utf-8",
                keep_alive,
            )?);
        }
        body.as_mut().map_or(Ok(()), |b| b.send(b" "))
    };
    loop {
        let mut io_error = None;
        let event = next_event(rx, || match tick(&mut body) {
            Ok(()) => true,
            Err(e) => {
                io_error = Some(e);
                false
            }
        });
        if io_error.is_none()
            && let Err(e) = tick(&mut body)
        {
            io_error = Some(e);
        }
        if let Some(e) = io_error {
            cancel.store(true, std::sync::atomic::Ordering::SeqCst);
            info!(Some(id), "client disconnected while waiting; stopping");
            return Err(e);
        }
        let Some(event) = event else {
            return Ok(());
        };
        let value = match event {
            Event::Failed { status, message } => {
                let err = error_json(
                    &message,
                    if status == 400 {
                        "invalid_request_error"
                    } else {
                        "server_error"
                    },
                );
                Some((status, err))
            }
            other => finish(other).map(|v| (200, v)),
        };
        if let Some((status, value)) = value {
            return match body.take() {
                Some(mut b) => {
                    b.send(value.to_string().as_bytes())?;
                    b.finish()
                }
                None => http::json(stream, status, &value, keep_alive),
            };
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn chat_stream(
    server: &Server,
    stream: &mut TcpStream,
    rx: &Receiver<Event>,
    cancel: &Arc<AtomicBool>,
    id: u64,
    keep_alive: bool,
    exclude: bool,
    include_usage: bool,
    progress: bool,
) -> std::io::Result<()> {
    let model = server.model_name();
    let chunk = |delta: Json, finish: Option<&str>| {
        json!({
            "id": format!("chatcmpl-{id}"),
            "object": "chat.completion.chunk",
            "created": now(),
            "model": model,
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]
        })
    };
    // Hold headers until the request starts (or 5 s pass) so early errors
    // still get a proper status code.
    let start = Instant::now();
    let mut pending = None;
    while start.elapsed() < Duration::from_secs(5) {
        match rx.recv_timeout(Duration::from_millis(100)) {
            Ok(e) => {
                pending = Some(e);
                break;
            }
            Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {}
            Err(_) => break,
        }
    }
    if let Some(Event::Failed { status, message }) = &pending {
        let kind = if *status == 400 {
            "invalid_request_error"
        } else {
            "server_error"
        };
        return http::json(stream, *status, &error_json(message, kind), keep_alive);
    }
    let mut out = Chunked::start(stream, 200, "text/event-stream", keep_alive)?;
    let fail = |e: std::io::Error, cancel: &Arc<AtomicBool>| {
        cancel.store(true, std::sync::atomic::Ordering::SeqCst);
        info!(Some(id), "client disconnected; stopping");
        e
    };
    let data = |value: &Json| format!("data: {value}\n\n");
    out.send(data(&chunk(json!({"role": "assistant", "content": null}), None)).as_bytes())
        .map_err(|e| fail(e, cancel))?;
    let mut last_ping = Instant::now();
    let mut queued_note = false;
    let mut next = pending;
    loop {
        let event = match next.take() {
            Some(e) => e,
            None => {
                let mut ping_error = None;
                let e = next_event(rx, || {
                    if last_ping.elapsed() >= Duration::from_secs(3) {
                        last_ping = Instant::now();
                        let note = if !queued_note {
                            queued_note = true;
                            ": waiting\n\n"
                        } else {
                            ": keep-alive\n\n"
                        };
                        if let Err(e) = out.send(note.as_bytes()) {
                            ping_error = Some(e);
                            return false;
                        }
                    }
                    true
                });
                if let Some(e) = ping_error {
                    return Err(fail(e, cancel));
                }
                match e {
                    Some(e) => e,
                    None => return out.finish(),
                }
            }
        };
        let payload = match event {
            Event::Started { prompt, reused } => {
                progress.then(|| data(&json!({"id": format!("chatcmpl-{id}"), "object": "chat.completion.chunk", "created": now(), "model": model,
                    "choices": [], "prompt_progress": {"total": prompt, "cache": reused, "processed": reused, "time_ms": 0}})))
            }
            Event::Progress { processed, total } => {
                if progress {
                    Some(data(&json!({"id": format!("chatcmpl-{id}"), "object": "chat.completion.chunk", "created": now(), "model": model,
                        "choices": [], "prompt_progress": {"total": total, "processed": processed, "time_ms": start.elapsed().as_millis()}})))
                } else if last_ping.elapsed() >= Duration::from_secs(3) {
                    Some(format!(": prefill {processed}/{total}\n\n"))
                } else {
                    None
                }
            }
            Event::Delta(Delta::Reasoning(t)) => (!exclude).then(|| data(&chunk(json!({"reasoning_content": t}), None))),
            Event::Delta(Delta::Content(t)) => Some(data(&chunk(json!({"content": t}), None))),
            Event::Chat(result) => {
                let mut tail = String::new();
                let calls = tool_calls(&result, id);
                if !calls.is_empty() {
                    tail.push_str(&data(&chunk(json!({"tool_calls": calls}), None)));
                }
                let mut last = chunk(json!({}), Some(finish_reason(&result)));
                last["usage"] = usage(&result.completion);
                last["timings"] = timings(&result.completion);
                tail.push_str(&data(&last));
                if include_usage {
                    tail.push_str(&data(&json!({"id": format!("chatcmpl-{id}"), "object": "chat.completion.chunk",
                        "created": now(), "model": model, "choices": [], "usage": usage(&result.completion)})));
                }
                tail.push_str("data: [DONE]\n\n");
                out.send(tail.as_bytes()).map_err(|e| fail(e, cancel))?;
                return out.finish();
            }
            Event::Failed { status, message } => {
                let kind = if status == 400 { "invalid_request_error" } else { "server_error" };
                let text = format!("data: {}\n\ndata: [DONE]\n\n", error_json(&message, kind));
                out.send(text.as_bytes()).map_err(|e| fail(e, cancel))?;
                return out.finish();
            }
            Event::Completed { .. } => None,
        };
        if let Some(p) = payload {
            last_ping = Instant::now();
            out.send(p.as_bytes()).map_err(|e| fail(e, cancel))?;
        }
    }
}

/// POST /v1/completions (and /completion, llama-server's native route).
pub fn completion(
    server: &Server,
    stream: &mut TcpStream,
    body: &Json,
    id: u64,
    keep_alive: bool,
) -> std::io::Result<()> {
    let prompt = match body.get("prompt") {
        Some(Json::String(p)) => p.clone(),
        Some(Json::Array(a)) if a.len() == 1 && a[0].is_string() => {
            a[0].as_str().unwrap_or_default().to_owned()
        }
        _ => {
            return http::json(
                stream,
                400,
                &error_json("prompt must be a string", "invalid_request_error"),
                keep_alive,
            );
        }
    };
    let sampling = match sampling(body, server) {
        Ok(s) => s,
        Err(e) => {
            return http::json(
                stream,
                400,
                &error_json(&e, "invalid_request_error"),
                keep_alive,
            );
        }
    };
    let max = max_tokens(body, &["max_tokens", "n_predict"], server);
    let stop = stop_strings(body, "stop");
    let streaming = body.get("stream").and_then(Json::as_bool).unwrap_or(false);
    info!(
        Some(id),
        "completion: prompt {} chars, max {max} tok, {}",
        prompt.len(),
        if streaming { "stream" } else { "no stream" }
    );
    let (rx, cancel) = server.submit(
        id,
        Work::Complete {
            prompt,
            max_tokens: max,
            sampling,
            stop,
        },
        stream,
    );
    let model = server.model_name();
    let body_for = |text: &str, c: &Completion, stopped: bool, obj: &str| {
        let finish = if stopped {
            "stop"
        } else {
            c.stop.finish_reason()
        };
        json!({
            "id": format!("cmpl-{id}"),
            "object": obj,
            "created": now(),
            "model": model,
            "choices": [{"index": 0, "text": text, "finish_reason": finish, "logprobs": null}],
            "usage": usage(c),
            "timings": timings(c),
        })
    };
    if !streaming {
        return wait_json(stream, &rx, &cancel, id, keep_alive, |event| match event {
            Event::Completed {
                completion,
                text,
                stopped,
            } => Some(body_for(&text, &completion, stopped, "text_completion")),
            _ => None,
        });
    }
    let mut out = Chunked::start(stream, 200, "text/event-stream", keep_alive)?;
    let mut last_ping = Instant::now();
    loop {
        let mut ping_error = None;
        let event = next_event(&rx, || {
            if last_ping.elapsed() >= Duration::from_secs(3) {
                last_ping = Instant::now();
                if let Err(e) = out.send(b": keep-alive\n\n") {
                    ping_error = Some(e);
                    return false;
                }
            }
            true
        });
        if let Some(e) = ping_error {
            cancel.store(true, std::sync::atomic::Ordering::SeqCst);
            return Err(e);
        }
        let Some(event) = event else {
            return out.finish();
        };
        let text = match event {
            Event::Delta(Delta::Content(t) | Delta::Reasoning(t)) => Some(format!(
                "data: {}\n\n",
                json!({"id": format!("cmpl-{id}"), "object": "text_completion", "created": now(), "model": model,
                    "choices": [{"index": 0, "text": t, "finish_reason": null}]})
            )),
            Event::Completed {
                completion,
                stopped,
                ..
            } => {
                let last = body_for("", &completion, stopped, "text_completion");
                out.send(format!("data: {last}\n\ndata: [DONE]\n\n").as_bytes())?;
                return out.finish();
            }
            Event::Failed { message, .. } => {
                out.send(
                    format!(
                        "data: {}\n\ndata: [DONE]\n\n",
                        error_json(&message, "server_error")
                    )
                    .as_bytes(),
                )?;
                return out.finish();
            }
            _ => None,
        };
        if let Some(t) = text {
            last_ping = Instant::now();
            if let Err(e) = out.send(t.as_bytes()) {
                cancel.store(true, std::sync::atomic::Ordering::SeqCst);
                info!(Some(id), "client disconnected; stopping");
                return Err(e);
            }
        }
    }
}
