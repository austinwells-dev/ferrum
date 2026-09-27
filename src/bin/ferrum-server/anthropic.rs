//! Anthropic Messages API: POST /v1/messages (streaming and not) and
//! /v1/messages/count_tokens, mapped onto the same chat runtime.
use crate::{
    Server, http,
    http::Chunked,
    info,
    openai::{self, next_event},
    warn,
    worker::{Event, Work},
};
use ferrum::hybrid::runtime::{ChatRequest, ChatResult, Delta};
use serde_json::{Map, Value as Json, json};
use std::{
    net::TcpStream,
    sync::atomic::Ordering,
    time::{Duration, Instant},
};

fn error(message: &str, kind: &str) -> Json {
    json!({"type": "error", "error": {"type": kind, "message": message}})
}

fn text_of(content: &Json) -> String {
    match content {
        Json::String(s) => s.clone(),
        Json::Array(blocks) => blocks
            .iter()
            .filter(|b| b.get("type").and_then(Json::as_str) == Some("text"))
            .filter_map(|b| b.get("text")?.as_str())
            .collect::<Vec<_>>()
            .join("\n"),
        _ => String::new(),
    }
}

/// Anthropic messages and tools → OpenAI-style messages and tools.
fn convert(body: &Json, id: u64) -> Result<(Vec<Json>, Vec<Json>), String> {
    let mut messages = Vec::new();
    if let Some(system) = body.get("system") {
        let text = text_of(system);
        if !text.is_empty() {
            messages.push(json!({"role": "system", "content": text}));
        }
    }
    for m in body
        .get("messages")
        .and_then(Json::as_array)
        .ok_or("messages must be an array")?
    {
        let role = m
            .get("role")
            .and_then(Json::as_str)
            .ok_or("message role missing")?;
        let content = m.get("content").cloned().unwrap_or(Json::Null);
        let blocks = match &content {
            Json::String(s) => vec![json!({"type": "text", "text": s})],
            Json::Array(b) => b.clone(),
            _ => Vec::new(),
        };
        match role {
            "user" => {
                let mut text = Vec::new();
                for b in &blocks {
                    match b.get("type").and_then(Json::as_str) {
                        Some("tool_result") => {
                            let mut result = text_of(b.get("content").unwrap_or(&Json::Null));
                            if b.get("is_error").and_then(Json::as_bool) == Some(true) {
                                result = format!("Error: {result}");
                            }
                            messages.push(json!({
                                "role": "tool",
                                "tool_call_id": b.get("tool_use_id").cloned().unwrap_or(Json::Null),
                                "content": result
                            }));
                        }
                        Some("text") => text.push(
                            b.get("text")
                                .and_then(Json::as_str)
                                .unwrap_or_default()
                                .to_owned(),
                        ),
                        Some(other) => {
                            warn!(Some(id), "ignoring unsupported user content block {other}")
                        }
                        None => {}
                    }
                }
                if !text.is_empty() {
                    messages.push(json!({"role": "user", "content": text.join("\n")}));
                }
            }
            "assistant" => {
                let mut text = Vec::new();
                let mut reasoning = Vec::new();
                let mut calls = Vec::new();
                for b in &blocks {
                    match b.get("type").and_then(Json::as_str) {
                        Some("text") => text.push(b.get("text").and_then(Json::as_str).unwrap_or_default().to_owned()),
                        Some("thinking") => {
                            reasoning.push(b.get("thinking").and_then(Json::as_str).unwrap_or_default().to_owned())
                        }
                        Some("tool_use") => calls.push(json!({
                            "id": b.get("id").cloned().unwrap_or(Json::Null),
                            "type": "function",
                            "function": {
                                "name": b.get("name").cloned().unwrap_or(Json::Null),
                                "arguments": b.get("input").cloned().unwrap_or(json!({})).to_string()
                            }
                        })),
                        _ => {}
                    }
                }
                let mut msg = json!({"role": "assistant", "content": text.join("\n")});
                if !reasoning.is_empty() {
                    msg["reasoning_content"] = Json::String(reasoning.join("\n"));
                }
                if !calls.is_empty() {
                    msg["tool_calls"] = Json::Array(calls);
                }
                messages.push(msg);
            }
            other => return Err(format!("unsupported role {other}")),
        }
    }
    let mut tools = Vec::new();
    for t in body
        .get("tools")
        .and_then(Json::as_array)
        .into_iter()
        .flatten()
    {
        match (t.get("name"), t.get("input_schema")) {
            (Some(name), Some(schema)) => tools.push(json!({"type": "function", "function": {
                "name": name, "description": t.get("description").cloned().unwrap_or(Json::String(String::new())),
                "parameters": schema}})),
            _ => warn!(Some(id), "ignoring server tool {}", t.get("type").and_then(Json::as_str).unwrap_or("?")),
        }
    }
    if body.pointer("/tool_choice/type").and_then(Json::as_str) == Some("none") {
        tools.clear();
    }
    Ok((messages, tools))
}

fn request(body: &Json, server: &Server, id: u64) -> Result<ChatRequest, String> {
    let (messages, tools) = convert(body, id)?;
    if messages.is_empty() {
        return Err("messages must not be empty".into());
    }
    let t = openai::thinking(body, server);
    Ok(ChatRequest {
        messages,
        tools,
        max_tokens: openai::max_tokens(body, &["max_tokens"], server),
        sampling: openai::sampling(body, server)?,
        template_vars: t.vars,
        stop: openai::stop_strings(body, "stop_sequences"),
        reasoning_budget: t.budget,
        raw_reasoning: false,
    })
}

fn stop_reason(r: &ChatResult) -> &'static str {
    if !r.output.tool_calls.is_empty() {
        "tool_use"
    } else if r.stopped_by_string {
        "stop_sequence"
    } else if r.completion.stop.finish_reason() == "length" {
        "max_tokens"
    } else {
        "end_turn"
    }
}

fn usage(r: &ChatResult) -> Json {
    let c = &r.completion;
    json!({
        "input_tokens": c.prompt_tokens - c.reused_tokens,
        "cache_read_input_tokens": c.reused_tokens,
        "cache_creation_input_tokens": 0,
        "output_tokens": c.tokens.len(),
    })
}

fn tool_blocks(r: &ChatResult, id: u64) -> Vec<Json> {
    r.output
        .tool_calls
        .iter()
        .enumerate()
        .map(|(i, c)| json!({"type": "tool_use", "id": format!("toolu_{id}_{i}"), "name": c.name, "input": Json::Object(c.arguments.clone())}))
        .collect()
}

pub fn count_tokens(
    server: &Server,
    stream: &mut TcpStream,
    body: &Json,
    id: u64,
    keep_alive: bool,
) -> std::io::Result<()> {
    let result = convert(body, id).and_then(|(messages, tools)| {
        server
            .template
            .render(
                &messages,
                (!tools.is_empty()).then_some(&tools[..]),
                true,
                &Map::new(),
            )
            .and_then(|text| server.tokenizer.encode(&text))
            .map_err(|e| e.to_string())
    });
    match result {
        Ok(ids) => http::json(stream, 200, &json!({"input_tokens": ids.len()}), keep_alive),
        Err(e) => http::json(stream, 400, &error(&e, "invalid_request_error"), keep_alive),
    }
}

pub fn messages(
    server: &Server,
    stream: &mut TcpStream,
    body: &Json,
    id: u64,
    keep_alive: bool,
) -> std::io::Result<()> {
    let req = match request(body, server, id) {
        Ok(r) => r,
        Err(e) => {
            warn!(Some(id), "bad request: {e}");
            return http::json(stream, 400, &error(&e, "invalid_request_error"), keep_alive);
        }
    };
    let streaming = body.get("stream").and_then(Json::as_bool).unwrap_or(false);
    info!(
        Some(id),
        "messages: {} messages, {} tools, {}, max {} tok, thinking {}{}",
        req.messages.len(),
        req.tools.len(),
        if streaming { "stream" } else { "no stream" },
        req.max_tokens,
        if req.template_vars.get("enable_thinking") == Some(&Json::Bool(false)) {
            "off"
        } else {
            "on"
        },
        req.reasoning_budget
            .map_or(String::new(), |b| format!(", budget {b}"))
    );
    let (rx, cancel) = server.submit(id, Work::Chat(Box::new(req)));
    let model = server.model_name();
    if !streaming {
        return openai::wait_json(stream, &rx, &cancel, id, keep_alive, |event| match event {
            Event::Chat(r) => {
                let mut content = Vec::new();
                if let Some(t) = &r.output.reasoning {
                    content.push(json!({"type": "thinking", "thinking": t, "signature": ""}));
                }
                if !r.output.content.is_empty() {
                    content.push(json!({"type": "text", "text": r.output.content}));
                }
                content.extend(tool_blocks(&r, id));
                Some(json!({
                    "id": format!("msg_{id}"), "type": "message", "role": "assistant", "model": model,
                    "content": content, "stop_reason": stop_reason(&r), "stop_sequence": null, "usage": usage(&r)
                }))
            }
            _ => None,
        });
    }
    // Streaming: message_start, one content block per kind, message_delta, message_stop.
    let mut out = Chunked::start(stream, 200, "text/event-stream", keep_alive)?;
    let event = |name: &str, data: Json| format!("event: {name}\ndata: {data}\n\n");
    let gone = |e: std::io::Error| {
        cancel.store(true, Ordering::SeqCst);
        info!(Some(id), "client disconnected; stopping");
        e
    };
    out.send(
        event(
            "message_start",
            json!({"type": "message_start", "message": {"id": format!("msg_{id}"), "type": "message", "role": "assistant",
                "model": model, "content": [], "stop_reason": null, "stop_sequence": null,
                "usage": {"input_tokens": 0, "output_tokens": 0}}}),
        )
        .as_bytes(),
    )
    .map_err(gone)?;
    // (kind, index) of the open block.
    let mut open: Option<(&'static str, usize)> = None;
    let mut next_index = 0usize;
    let mut last_ping = Instant::now();
    loop {
        let mut ping_error = None;
        let ev = next_event(&rx, || {
            if last_ping.elapsed() >= Duration::from_secs(3) {
                last_ping = Instant::now();
                if let Err(e) = out.send(event("ping", json!({"type": "ping"})).as_bytes()) {
                    ping_error = Some(e);
                    return false;
                }
            }
            true
        });
        if let Some(e) = ping_error {
            return Err(gone(e));
        }
        let Some(ev) = ev else { return out.finish() };
        let mut text = String::new();
        let mut switch_to = |kind: &'static str, text: &mut String| -> usize {
            match open {
                Some((k, i)) if k == kind => i,
                _ => {
                    if let Some((_, i)) = open {
                        text.push_str(&event(
                            "content_block_stop",
                            json!({"type": "content_block_stop", "index": i}),
                        ));
                    }
                    let i = next_index;
                    next_index += 1;
                    let block = if kind == "thinking" {
                        json!({"type": "thinking", "thinking": "", "signature": ""})
                    } else {
                        json!({"type": "text", "text": ""})
                    };
                    text.push_str(&event(
                        "content_block_start",
                        json!({"type": "content_block_start", "index": i, "content_block": block}),
                    ));
                    open = Some((kind, i));
                    i
                }
            }
        };
        match ev {
            Event::Delta(Delta::Reasoning(t)) => {
                let i = switch_to("thinking", &mut text);
                text.push_str(&event(
                    "content_block_delta",
                    json!({"type": "content_block_delta", "index": i,
                    "delta": {"type": "thinking_delta", "thinking": t}}),
                ));
            }
            Event::Delta(Delta::Content(t)) => {
                let i = switch_to("text", &mut text);
                text.push_str(&event(
                    "content_block_delta",
                    json!({"type": "content_block_delta", "index": i,
                    "delta": {"type": "text_delta", "text": t}}),
                ));
            }
            Event::Chat(r) => {
                if let Some((kind, i)) = open.take() {
                    if kind == "thinking" {
                        text.push_str(&event(
                            "content_block_delta",
                            json!({"type": "content_block_delta", "index": i,
                            "delta": {"type": "signature_delta", "signature": ""}}),
                        ));
                    }
                    text.push_str(&event(
                        "content_block_stop",
                        json!({"type": "content_block_stop", "index": i}),
                    ));
                }
                for block in tool_blocks(&r, id) {
                    let i = next_index;
                    next_index += 1;
                    let input = block["input"].to_string();
                    let mut start = block.clone();
                    start["input"] = json!({});
                    text.push_str(&event(
                        "content_block_start",
                        json!({"type": "content_block_start", "index": i, "content_block": start}),
                    ));
                    text.push_str(&event(
                        "content_block_delta",
                        json!({"type": "content_block_delta", "index": i,
                        "delta": {"type": "input_json_delta", "partial_json": input}}),
                    ));
                    text.push_str(&event(
                        "content_block_stop",
                        json!({"type": "content_block_stop", "index": i}),
                    ));
                }
                text.push_str(&event("message_delta", json!({"type": "message_delta",
                    "delta": {"stop_reason": stop_reason(&r), "stop_sequence": null}, "usage": usage(&r)})));
                text.push_str(&event("message_stop", json!({"type": "message_stop"})));
                out.send(text.as_bytes()).map_err(gone)?;
                return out.finish();
            }
            Event::Failed { status, message } => {
                let kind = if status == 400 {
                    "invalid_request_error"
                } else {
                    "api_error"
                };
                out.send(event("error", error(&message, kind)).as_bytes())
                    .map_err(gone)?;
                return out.finish();
            }
            _ => {}
        }
        if !text.is_empty() {
            last_ping = Instant::now();
            out.send(text.as_bytes()).map_err(gone)?;
        }
    }
}
