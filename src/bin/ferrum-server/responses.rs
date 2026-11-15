//! OpenAI's Responses API: POST /v1/responses, stateless. The request is
//! rewritten as a chat completion (`chat_body`) and the reply is reported as
//! output items: a reasoning item, a message, and function calls.
use crate::{
    Server, error_json,
    http::{self, Chunked},
    info, openai, warn,
    worker::{Event, Work},
};
use ferrum::hybrid::runtime::{ChatResult, Delta};
use serde_json::{Map, Value as Json, json};
use std::{
    net::TcpStream,
    sync::{Arc, atomic::AtomicBool, mpsc::Receiver},
    time::{Duration, Instant},
};

/// A message's `content` in chat-completion form: `input_text` and
/// `output_text` parts become text, `input_image` an image part.
fn content(value: &Json) -> Result<Json, String> {
    let parts = match value {
        Json::Array(parts) => parts,
        other => return Ok(other.clone()),
    };
    let mut out = Vec::new();
    for part in parts {
        match part.get("type").and_then(Json::as_str) {
            Some("input_text" | "output_text" | "text") => out.push(
                json!({"type": "text", "text": part.get("text").cloned().unwrap_or_default()}),
            ),
            Some("input_image") => {
                let url = part
                    .get("image_url")
                    .and_then(Json::as_str)
                    .ok_or("an input_image needs an image_url (a data: or http(s) URL)")?;
                let mut image = json!({"url": url});
                if let Some(detail) = part.get("detail") {
                    image["detail"] = detail.clone();
                }
                out.push(json!({"type": "image_url", "image_url": image}));
            }
            Some("refusal") => {}
            Some(other) => return Err(format!("unsupported content part {other:?}")),
            None => return Err("a content part needs a type".into()),
        }
    }
    // Text alone stays a string, as chat templates expect of assistant turns.
    if out.iter().all(|p| p["type"] == "text") {
        let text: Vec<&str> = out.iter().filter_map(|p| p["text"].as_str()).collect();
        return Ok(Json::String(text.concat()));
    }
    Ok(Json::Array(out))
}

/// The request as a `/v1/chat/completions` body.
fn chat_body(body: &Json, id: u64) -> Result<Json, String> {
    if body
        .get("previous_response_id")
        .is_some_and(|v| !v.is_null())
    {
        return Err(
            "previous_response_id is not supported: responses are not stored, send the conversation in input".into(),
        );
    }
    let mut messages = Vec::new();
    if let Some(text) = body.get("instructions").and_then(Json::as_str) {
        messages.push(json!({"role": "system", "content": text}));
    }
    match body.get("input") {
        Some(Json::String(text)) => messages.push(json!({"role": "user", "content": text})),
        Some(Json::Array(items)) => {
            for item in items {
                let kind = item.get("type").and_then(Json::as_str);
                match (kind, item.get("role").and_then(Json::as_str)) {
                    (None | Some("message"), Some(role)) => {
                        let role = if role == "developer" { "system" } else { role };
                        let text = content(item.get("content").unwrap_or(&Json::Null))?;
                        messages.push(json!({"role": role, "content": text}));
                    }
                    (Some("function_call"), _) => {
                        let call = json!({
                            "id": item.get("call_id").cloned().unwrap_or_default(),
                            "type": "function",
                            "function": {
                                "name": item.get("name").cloned().unwrap_or_default(),
                                "arguments": item.get("arguments").cloned().unwrap_or(json!("{}")),
                            }
                        });
                        // Calls of one turn share its assistant message.
                        match messages.last_mut().and_then(|m| m.get_mut("tool_calls")) {
                            Some(Json::Array(calls)) => calls.push(call),
                            _ => messages.push(
                                json!({"role": "assistant", "content": "", "tool_calls": [call]}),
                            ),
                        }
                    }
                    (Some("function_call_output"), _) => {
                        let output = match item.get("output") {
                            Some(Json::String(text)) => Json::String(text.clone()),
                            Some(other) => content(other)?,
                            None => Json::String(String::new()),
                        };
                        messages.push(json!({
                            "role": "tool",
                            "tool_call_id": item.get("call_id").cloned().unwrap_or_default(),
                            "content": output,
                        }));
                    }
                    // Earlier reasoning is not fed back.
                    (Some("reasoning"), _) => {}
                    (other, _) => {
                        return Err(format!(
                            "unsupported input item {:?}",
                            other.unwrap_or("without a role or type")
                        ));
                    }
                }
            }
        }
        _ => return Err("input must be a string or an array of items".into()),
    }
    if !messages.iter().any(|m| m["role"] != "system") {
        return Err("input must hold at least one message".into());
    }
    let mut chat = Map::new();
    chat.insert("messages".into(), Json::Array(messages));
    let mut tools = Vec::new();
    for tool in body
        .get("tools")
        .and_then(Json::as_array)
        .into_iter()
        .flatten()
    {
        match tool.get("type").and_then(Json::as_str) {
            Some("function") => tools.push(json!({"type": "function", "function": {
                "name": tool.get("name").cloned().unwrap_or_default(),
                "description": tool.get("description").cloned().unwrap_or(json!("")),
                "parameters": tool.get("parameters").cloned().unwrap_or(json!({"type": "object"})),
            }})),
            other => warn!(
                Some(id),
                "ignoring tool of type {} (only function tools run here)",
                other.unwrap_or("?")
            ),
        }
    }
    if !tools.is_empty() {
        chat.insert("tools".into(), Json::Array(tools));
        match body.get("tool_choice") {
            Some(Json::Object(choice)) if choice.get("type").is_some_and(|t| t == "function") => {
                chat.insert("tool_choice".into(), Json::Object(choice.clone()));
            }
            Some(Json::Object(_)) => {
                return Err("tool_choice may be auto, none, required or a function".into());
            }
            Some(choice) => {
                chat.insert("tool_choice".into(), choice.clone());
            }
            None => {}
        }
    }
    // text.format is response_format with the schema one level up.
    match body.pointer("/text/format") {
        Some(format) if format.get("type").is_some_and(|t| t == "json_schema") => {
            chat.insert(
                "response_format".into(),
                json!({"type": "json_schema", "json_schema": {"schema": format.get("schema").cloned().unwrap_or(Json::Null)}}),
            );
        }
        Some(format) if !format.is_null() => {
            chat.insert("response_format".into(), format.clone());
        }
        _ => {}
    }
    if let Some(effort) = body.pointer("/reasoning/effort").filter(|v| !v.is_null()) {
        chat.insert("reasoning_effort".into(), effort.clone());
    }
    if let Some(max) = body.get("max_output_tokens").filter(|v| !v.is_null()) {
        chat.insert("max_tokens".into(), max.clone());
    }
    // Sampling and this server's own extensions pass through unchanged.
    for key in [
        "temperature",
        "top_p",
        "top_k",
        "min_p",
        "seed",
        "presence_penalty",
        "frequency_penalty",
        "repetition_penalty",
        "repeat_penalty",
        "model",
        "logit_bias",
        "grammar",
        "chat_template_kwargs",
        "reasoning_budget",
        "thinking_budget_tokens",
        "id_slot",
        "prompt_cache_key",
        "session_id",
    ] {
        if let Some(value) = body.get(key).filter(|v| !v.is_null()) {
            chat.insert(key.into(), value.clone());
        }
    }
    Ok(Json::Object(chat))
}

fn usage(result: &ChatResult) -> Json {
    let c = &result.completion;
    json!({
        "input_tokens": c.prompt_tokens,
        "input_tokens_details": {"cached_tokens": c.reused_tokens},
        "output_tokens": c.tokens.len(),
        "output_tokens_details": {"reasoning_tokens": result.reasoning_tokens},
        "total_tokens": c.prompt_tokens + c.tokens.len(),
    })
}

/// The response object. `output` and `usage` are filled when it is done.
fn envelope(server: &Server, body: &Json, id: u64, created: u64) -> Json {
    let echo = |key: &str, default: Json| body.get(key).cloned().unwrap_or(default);
    json!({
        "id": format!("resp_{id}"),
        "object": "response",
        "created_at": created,
        "status": "in_progress",
        "error": null,
        "incomplete_details": null,
        "instructions": echo("instructions", Json::Null),
        "max_output_tokens": echo("max_output_tokens", Json::Null),
        "model": server.model_name(body),
        "output": [],
        "parallel_tool_calls": true,
        "previous_response_id": null,
        "reasoning": echo("reasoning", json!({"effort": null, "summary": null})),
        "store": false,
        "temperature": echo("temperature", Json::Null),
        "text": echo("text", json!({"format": {"type": "text"}})),
        "tool_choice": echo("tool_choice", json!("auto")),
        "tools": echo("tools", json!([])),
        "top_p": echo("top_p", Json::Null),
        "truncation": "disabled",
        "usage": null,
        "metadata": echo("metadata", json!({})),
    })
}

fn reasoning_item(id: u64, text: &str) -> Json {
    json!({
        "id": format!("rs_{id}"),
        "type": "reasoning",
        "summary": [],
        "content": [{"type": "reasoning_text", "text": text}],
    })
}

fn message_item(id: u64, text: &str, status: &str) -> Json {
    json!({
        "id": format!("msg_{id}"),
        "type": "message",
        "status": status,
        "role": "assistant",
        "content": [{"type": "output_text", "text": text, "annotations": []}],
    })
}

fn call_items(result: &ChatResult, id: u64) -> Vec<Json> {
    result
        .output
        .tool_calls
        .iter()
        .enumerate()
        .map(|(i, call)| {
            json!({
                "id": format!("fc_{id}_{i}"),
                "type": "function_call",
                "status": "completed",
                "call_id": format!("call_{id}_{i}"),
                "name": call.name,
                "arguments": Json::Object(call.arguments.clone()).to_string(),
            })
        })
        .collect()
}

/// `envelope` completed with the reply's items.
fn finished(mut response: Json, result: &ChatResult, id: u64, exclude: bool) -> Json {
    let mut output = Vec::new();
    if let Some(reasoning) = result
        .output
        .reasoning
        .as_ref()
        .filter(|r| !exclude && !r.is_empty())
    {
        output.push(reasoning_item(id, reasoning));
    }
    let calls = call_items(result, id);
    if !result.output.content.is_empty() || calls.is_empty() {
        output.push(message_item(id, &result.output.content, "completed"));
    }
    output.extend(calls);
    // Out of tokens (or context) is an incomplete response, not an error.
    let cut = result.output.tool_calls.is_empty()
        && !result.stopped_by_string
        && result.completion.stop.finish_reason() == "length";
    response["status"] = json!(if cut { "incomplete" } else { "completed" });
    if cut {
        response["incomplete_details"] = json!({"reason": "max_output_tokens"});
    }
    response["output"] = Json::Array(output);
    response["usage"] = usage(result);
    response
}

pub fn responses(
    server: &Server,
    stream: &mut TcpStream,
    body: &Json,
    id: u64,
    keep_alive: bool,
) -> std::io::Result<()> {
    let parsed = chat_body(body, id).and_then(|chat| {
        let (request, exclude) = openai::chat_request(&chat, server, id)?;
        Ok((request, exclude, openai::route(&chat, server)?))
    });
    let (request, exclude, route) = match parsed {
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
    info!(
        Some(id),
        "responses: {}",
        openai::describe(&request, exclude, streaming)
    );
    let created = openai::now();
    let (rx, cancel) = server.submit(id, Work::Chat(Box::new(request)), route, stream);
    if streaming {
        let response = envelope(server, body, id, created);
        return events(stream, &rx, &cancel, id, keep_alive, exclude, response);
    }
    openai::wait_json(stream, &rx, &cancel, id, keep_alive, |event| match event {
        Event::Chat(result) => Some(finished(
            envelope(server, body, id, created),
            &result,
            id,
            exclude,
        )),
        _ => None,
    })
}

/// The item a streamed reply is writing.
#[derive(PartialEq, Clone, Copy)]
enum Open {
    None,
    Reasoning,
    Message,
}

/// The streamed form: `response.created`, then each output item opened, its
/// text in deltas, and closed, then `response.completed`. Function calls
/// arrive whole once the reply has ended (the runtime parses them then).
fn events(
    stream: &mut TcpStream,
    rx: &Receiver<Event>,
    cancel: &Arc<AtomicBool>,
    id: u64,
    keep_alive: bool,
    exclude: bool,
    response: Json,
) -> std::io::Result<()> {
    // Hold headers until the request starts (or 5 s pass) so early errors
    // still get a proper status code.
    let start = Instant::now();
    let mut next = None;
    while start.elapsed() < Duration::from_secs(5) {
        match rx.recv_timeout(Duration::from_millis(100)) {
            Ok(e) => {
                next = Some(e);
                break;
            }
            Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {}
            Err(_) => break,
        }
    }
    if let Some(Event::Failed { status, message }) = &next {
        let kind = if *status == 400 {
            "invalid_request_error"
        } else {
            "server_error"
        };
        return http::json(stream, *status, &error_json(message, kind), keep_alive);
    }
    let mut out = Chunked::start(stream, 200, "text/event-stream", keep_alive)?;
    let mut sequence = 0u64;
    let mut frame = |kind: &str, mut fields: Json| {
        fields["type"] = json!(kind);
        fields["sequence_number"] = json!(sequence);
        sequence += 1;
        format!("event: {kind}\ndata: {fields}\n\n")
    };
    let gone = |e: std::io::Error| {
        cancel.store(true, std::sync::atomic::Ordering::SeqCst);
        info!(Some(id), "client disconnected; stopping");
        e
    };
    let mut text = frame("response.created", json!({"response": response}));
    text.push_str(&frame(
        "response.in_progress",
        json!({"response": response}),
    ));
    out.send(text.as_bytes()).map_err(gone)?;

    let mut open = Open::None;
    let mut index = 0usize;
    let mut reasoning = String::new();
    let mut content = String::new();
    let mut last_ping = Instant::now();
    loop {
        let event = match next.take() {
            Some(e) => e,
            None => {
                let mut ping_error = None;
                let e = openai::next_event(rx, || {
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
                    return Err(gone(e));
                }
                match e {
                    Some(e) => e,
                    None => return out.finish(),
                }
            }
        };
        let mut text = String::new();
        // Close the item being written; `index` moves to the next one.
        let close = |open: &mut Open,
                     index: &mut usize,
                     text: &mut String,
                     frame: &mut dyn FnMut(&str, Json) -> String| {
            match *open {
                Open::None => return,
                Open::Reasoning => {
                    let item = format!("rs_{id}");
                    text.push_str(&frame("response.reasoning_text.done", json!({"item_id": item, "output_index": *index, "content_index": 0, "text": reasoning})));
                    text.push_str(&frame(
                        "response.output_item.done",
                        json!({"output_index": *index, "item": reasoning_item(id, &reasoning)}),
                    ));
                }
                Open::Message => {
                    let item = format!("msg_{id}");
                    text.push_str(&frame("response.output_text.done", json!({"item_id": item, "output_index": *index, "content_index": 0, "text": content})));
                    text.push_str(&frame("response.content_part.done", json!({"item_id": item, "output_index": *index, "content_index": 0, "part": {"type": "output_text", "text": content, "annotations": []}})));
                    text.push_str(&frame("response.output_item.done", json!({"output_index": *index, "item": message_item(id, &content, "completed")})));
                }
            }
            *open = Open::None;
            *index += 1;
        };
        match event {
            Event::Delta(Delta::Reasoning(delta)) => {
                if exclude {
                    continue;
                }
                if open != Open::Reasoning {
                    close(&mut open, &mut index, &mut text, &mut frame);
                    open = Open::Reasoning;
                    text.push_str(&frame("response.output_item.added", json!({"output_index": index, "item": {"id": format!("rs_{id}"), "type": "reasoning", "summary": [], "content": []}})));
                }
                text.push_str(&frame("response.reasoning_text.delta", json!({"item_id": format!("rs_{id}"), "output_index": index, "content_index": 0, "delta": delta})));
                reasoning.push_str(&delta);
            }
            Event::Delta(Delta::Content(delta)) => {
                if open != Open::Message {
                    close(&mut open, &mut index, &mut text, &mut frame);
                    open = Open::Message;
                    let item = format!("msg_{id}");
                    text.push_str(&frame("response.output_item.added", json!({"output_index": index, "item": {"id": item, "type": "message", "status": "in_progress", "role": "assistant", "content": []}})));
                    text.push_str(&frame("response.content_part.added", json!({"item_id": item, "output_index": index, "content_index": 0, "part": {"type": "output_text", "text": "", "annotations": []}})));
                }
                text.push_str(&frame("response.output_text.delta", json!({"item_id": format!("msg_{id}"), "output_index": index, "content_index": 0, "delta": delta})));
                content.push_str(&delta);
            }
            Event::Chat(result) => {
                close(&mut open, &mut index, &mut text, &mut frame);
                for call in call_items(&result, id) {
                    let mut pending = call.clone();
                    pending["status"] = json!("in_progress");
                    pending["arguments"] = json!("");
                    text.push_str(&frame(
                        "response.output_item.added",
                        json!({"output_index": index, "item": pending}),
                    ));
                    text.push_str(&frame("response.function_call_arguments.delta", json!({"item_id": call["id"], "output_index": index, "delta": call["arguments"]})));
                    text.push_str(&frame("response.function_call_arguments.done", json!({"item_id": call["id"], "output_index": index, "arguments": call["arguments"]})));
                    text.push_str(&frame(
                        "response.output_item.done",
                        json!({"output_index": index, "item": call}),
                    ));
                    index += 1;
                }
                let done = finished(response, &result, id, exclude);
                let kind = if done["status"] == "incomplete" {
                    "response.incomplete"
                } else {
                    "response.completed"
                };
                text.push_str(&frame(kind, json!({"response": done})));
                out.send(text.as_bytes()).map_err(gone)?;
                return out.finish();
            }
            Event::Failed { message, .. } => {
                text.push_str(&frame(
                    "error",
                    json!({"code": "server_error", "message": message, "param": null}),
                ));
                out.send(text.as_bytes()).map_err(gone)?;
                return out.finish();
            }
            Event::Started { .. } | Event::Progress { .. } | Event::Completed { .. } => {
                if last_ping.elapsed() < Duration::from_secs(3) {
                    continue;
                }
                text.push_str(": working\n\n");
            }
        }
        last_ping = Instant::now();
        out.send(text.as_bytes()).map_err(gone)?;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn input_items_become_chat_messages() {
        let body = json!({
            "instructions": "Be brief.",
            "input": [
                {"role": "developer", "content": "Use tools."},
                {"type": "message", "role": "user", "content": [
                    {"type": "input_text", "text": "What time "},
                    {"type": "input_text", "text": "is it in Oslo?"}]},
                {"type": "reasoning", "summary": []},
                {"type": "function_call", "call_id": "c1", "name": "get_time", "arguments": "{\"city\":\"Oslo\"}"},
                {"type": "function_call_output", "call_id": "c1", "output": "12:00"},
                {"role": "user", "content": [
                    {"type": "input_text", "text": "And this?"},
                    {"type": "input_image", "image_url": "data:image/png;base64,AAAA"}]}
            ],
            "tools": [
                {"type": "function", "name": "get_time", "parameters": {"type": "object"}},
                {"type": "web_search"}
            ],
            "tool_choice": {"type": "function", "name": "get_time"},
            "max_output_tokens": 50,
            "reasoning": {"effort": "low"},
            "text": {"format": {"type": "json_schema", "name": "x", "schema": {"type": "object"}}},
            "temperature": 0.5
        });
        let chat = chat_body(&body, 1).unwrap();
        let m = chat["messages"].as_array().unwrap();
        assert_eq!(m.len(), 6);
        assert_eq!(m[0], json!({"role": "system", "content": "Be brief."}));
        assert_eq!(m[1]["role"], "system");
        assert_eq!(m[2]["content"], "What time is it in Oslo?");
        assert_eq!(m[3]["tool_calls"][0]["function"]["name"], "get_time");
        assert_eq!(
            m[4],
            json!({"role": "tool", "tool_call_id": "c1", "content": "12:00"})
        );
        assert_eq!(
            m[5]["content"][1]["image_url"]["url"],
            "data:image/png;base64,AAAA"
        );
        assert_eq!(chat["tools"].as_array().unwrap().len(), 1);
        assert_eq!(chat["tools"][0]["function"]["name"], "get_time");
        assert_eq!(chat["tool_choice"]["name"], "get_time");
        assert_eq!(chat["max_tokens"], 50);
        assert_eq!(chat["reasoning_effort"], "low");
        assert_eq!(
            chat["response_format"]["json_schema"]["schema"]["type"],
            "object"
        );
        assert_eq!(chat["temperature"], 0.5);
    }

    #[test]
    fn refuses_what_it_cannot_honour() {
        let refused = |body: Json| chat_body(&body, 1).unwrap_err();
        assert!(
            refused(json!({"input": "hi", "previous_response_id": "resp_1"}))
                .contains("not stored")
        );
        assert!(refused(json!({})).contains("input must be"));
        assert!(
            refused(json!({"input": [], "instructions": "x"})).contains("at least one message")
        );
        assert!(
            refused(json!({"input": [{"type": "computer_call"}]}))
                .contains("unsupported input item")
        );
        assert!(
            refused(json!({"input": [{"role": "user", "content": [{"type": "input_file"}]}]}))
                .contains("unsupported content part")
        );
    }
}
