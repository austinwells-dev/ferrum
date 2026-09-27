//! Chat templates embedded in GGUF files, rendered the way Hugging Face
//! `apply_chat_template` does, and parsing of generated text into
//! reasoning, content and XML-style tool calls.
#![forbid(unsafe_code)]
use crate::{Error, Result};
use minijinja::{Environment, ErrorKind, Value};
use serde_json::{Map, Value as Json};

pub struct ChatTemplate {
    env: Environment<'static>,
}

/// Python `json.dumps(value, ensure_ascii=False)`: insertion order, `", "`
/// and `": "` separators, no HTML escaping (Hugging Face's `tojson`).
fn python_json(value: &Json, indent: Option<usize>, depth: usize, out: &mut String) {
    let newline = |out: &mut String, depth: usize| {
        if let Some(width) = indent {
            out.push('\n');
            out.push_str(&" ".repeat(width * depth));
        }
    };
    match value {
        Json::Object(map) => {
            if map.is_empty() {
                out.push_str("{}");
                return;
            }
            out.push('{');
            for (i, (k, v)) in map.iter().enumerate() {
                if i > 0 {
                    out.push(if indent.is_some() { ',' } else { ',' });
                    if indent.is_none() {
                        out.push(' ');
                    }
                }
                newline(out, depth + 1);
                out.push_str(&serde_json::to_string(k).unwrap_or_default());
                out.push_str(": ");
                python_json(v, indent, depth + 1, out);
            }
            newline(out, depth);
            out.push('}');
        }
        Json::Array(items) => {
            if items.is_empty() {
                out.push_str("[]");
                return;
            }
            out.push('[');
            for (i, v) in items.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                    if indent.is_none() {
                        out.push(' ');
                    }
                }
                newline(out, depth + 1);
                python_json(v, indent, depth + 1, out);
            }
            newline(out, depth);
            out.push(']');
        }
        Json::Number(n) => {
            // Python prints floats with a fractional part (1.0, not 1).
            if n.is_f64() {
                let f = n.as_f64().unwrap_or(0.);
                if f.fract() == 0. && f.abs() < 1e16 {
                    out.push_str(&format!("{f:.1}"));
                } else {
                    out.push_str(&n.to_string());
                }
            } else {
                out.push_str(&n.to_string());
            }
        }
        Json::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
        Json::Null => out.push_str("null"),
        Json::String(s) => out.push_str(&serde_json::to_string(s).unwrap_or_default()),
    }
}

impl ChatTemplate {
    pub fn new(source: &str) -> Result<Self> {
        let mut env = Environment::new();
        env.set_trim_blocks(true);
        env.set_lstrip_blocks(true);
        env.set_unknown_method_callback(minijinja_contrib::pycompat::unknown_method_callback);
        env.add_function(
            "raise_exception",
            |message: String| -> std::result::Result<Value, minijinja::Error> {
                Err(minijinja::Error::new(ErrorKind::InvalidOperation, message))
            },
        );
        env.add_filter(
            "tojson",
            |value: Value,
             kwargs: minijinja::value::Kwargs|
             -> std::result::Result<Value, minijinja::Error> {
                let indent: Option<usize> = kwargs.get("indent")?;
                kwargs.assert_all_used()?;
                let json: Json = serde_json::to_value(&value).map_err(|e| {
                    minijinja::Error::new(ErrorKind::InvalidOperation, e.to_string())
                })?;
                let mut out = String::new();
                python_json(&json, indent, 0, &mut out);
                Ok(Value::from_safe_string(out))
            },
        );
        env.add_template_owned("chat", source.to_owned())
            .map_err(|e| Error::Tokenizer(format!("chat template: {e:#}")))?;
        Ok(Self { env })
    }

    /// Render `messages` (OpenAI-style JSON objects). `extra` carries template
    /// variables such as `enable_thinking` or `reasoning_effort`.
    pub fn render(
        &self,
        messages: &[Json],
        tools: Option<&[Json]>,
        add_generation_prompt: bool,
        extra: &Map<String, Json>,
    ) -> Result<String> {
        let mut context = Map::new();
        context.insert(
            "messages".into(),
            Json::Array(messages.iter().map(normalize_message).collect()),
        );
        if let Some(tools) = tools.filter(|t| !t.is_empty()) {
            context.insert("tools".into(), Json::Array(tools.to_vec()));
        }
        context.insert(
            "add_generation_prompt".into(),
            Json::Bool(add_generation_prompt),
        );
        context.insert("bos_token".into(), Json::String(String::new()));
        context.insert("eos_token".into(), Json::String("<|im_end|>".into()));
        for (k, v) in extra {
            context.insert(k.clone(), v.clone());
        }
        let template = self
            .env
            .get_template("chat")
            .map_err(|e| Error::Tokenizer(e.to_string()))?;
        template
            .render(Value::from_serialize(&context))
            .map_err(|e| Error::Tokenizer(format!("chat template: {e:#}")))
    }
}

/// OpenAI tool-call arguments arrive as a JSON string; templates iterate them
/// as a mapping (`arguments|items`), as Hugging Face clients pass them.
fn normalize_message(message: &Json) -> Json {
    let mut message = message.clone();
    if let Some(calls) = message.get_mut("tool_calls").and_then(Json::as_array_mut) {
        for call in calls {
            if let Some(args) = call.pointer_mut("/function/arguments")
                && let Some(text) = args.as_str()
                && let Ok(parsed) = serde_json::from_str::<Json>(text)
                && parsed.is_object()
            {
                *args = parsed;
            }
        }
    }
    message
}

#[derive(Debug, Clone, PartialEq)]
pub struct ToolCall {
    pub name: String,
    /// Arguments as a JSON object.
    pub arguments: Map<String, Json>,
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct ParsedOutput {
    pub reasoning: Option<String>,
    pub content: String,
    pub tool_calls: Vec<ToolCall>,
}

/// Split generated text into reasoning (up to `</think>`), visible content and
/// `<tool_call><function=…><parameter=…>…</parameter></function></tool_call>`
/// blocks. `starts_in_reasoning` is true when the prompt ended inside `<think>`.
/// `tools` (OpenAI tool definitions) type the parameter values by schema.
pub fn parse_output(text: &str, starts_in_reasoning: bool, tools: &[Json]) -> ParsedOutput {
    let mut rest = text;
    let mut reasoning = None;
    if starts_in_reasoning {
        match rest.find("</think>") {
            Some(end) => {
                reasoning = Some(rest[..end].trim().to_owned());
                rest = &rest[end + "</think>".len()..];
            }
            None => {
                return ParsedOutput {
                    reasoning: Some(rest.trim().to_owned()),
                    ..Default::default()
                };
            }
        }
    } else if let Some(stripped) = rest.trim_start().strip_prefix("<think>")
        && let Some(end) = stripped.find("</think>")
    {
        reasoning = Some(stripped[..end].trim().to_owned());
        rest = &stripped[end + "</think>".len()..];
    }
    let mut content = String::new();
    let mut tool_calls = Vec::new();
    let mut cursor = rest;
    while let Some(start) = cursor.find("<tool_call>") {
        content.push_str(&cursor[..start]);
        let after = &cursor[start + "<tool_call>".len()..];
        let (body, next) = match after.find("</tool_call>") {
            Some(end) => (&after[..end], &after[end + "</tool_call>".len()..]),
            None => (after, ""),
        };
        if let Some(call) = parse_tool_call(body, tools) {
            tool_calls.push(call);
        } else {
            // Unparseable: keep it visible rather than dropping model output.
            content.push_str(&cursor[start..cursor.len() - next.len()]);
        }
        cursor = next;
    }
    content.push_str(cursor);
    ParsedOutput {
        reasoning,
        content: content.trim().to_owned(),
        tool_calls,
    }
}

fn parse_tool_call(body: &str, tools: &[Json]) -> Option<ToolCall> {
    let start = body.find("<function=")? + "<function=".len();
    let name_end = start + body[start..].find('>')?;
    let name = body[start..name_end].trim().to_owned();
    let schema = tools.iter().find_map(|t| {
        let f = t.get("function").unwrap_or(t);
        (f.get("name")?.as_str()? == name).then(|| f.pointer("/parameters/properties").cloned())?
    });
    let mut arguments = Map::new();
    let mut cursor = &body[name_end + 1..];
    while let Some(p) = cursor.find("<parameter=") {
        let after = &cursor[p + "<parameter=".len()..];
        let key_end = after.find('>')?;
        let key = after[..key_end].trim().to_owned();
        let value_start = &after[key_end + 1..];
        let value_end = value_start
            .find("</parameter>")
            .or_else(|| value_start.find("<parameter="))
            .or_else(|| value_start.find("</function>"))
            .unwrap_or(value_start.len());
        let raw = value_start[..value_end]
            .strip_prefix('\n')
            .unwrap_or(&value_start[..value_end]);
        let raw = raw.strip_suffix('\n').unwrap_or(raw);
        let declared = schema
            .as_ref()
            .and_then(|s| s.get(&key))
            .and_then(|p| p.get("type"))
            .and_then(Json::as_str);
        let value = match declared {
            Some("string") => Json::String(raw.to_owned()),
            _ => serde_json::from_str(raw.trim()).unwrap_or_else(|_| Json::String(raw.to_owned())),
        };
        arguments.insert(key, value);
        cursor = &value_start[value_end..];
    }
    Some(ToolCall { name, arguments })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tojson_matches_python_json_dumps() {
        let value: Json = serde_json::from_str(r#"{"b": 1, "a": [1.0, "x<y"], "c": {}}"#).unwrap();
        let mut out = String::new();
        python_json(&value, None, 0, &mut out);
        assert_eq!(out, r#"{"b": 1, "a": [1.0, "x<y"], "c": {}}"#);
    }

    #[test]
    fn parses_reasoning_content_and_typed_tool_calls() {
        let tools: Vec<Json> = vec![
            serde_json::json!({"type": "function", "function": {"name": "read",
            "parameters": {"properties": {"path": {"type": "string"}, "lines": {"type": "integer"}}}}}),
        ];
        let text = "Let me look.\n</think>\n\nReading it.\n\n<tool_call>\n<function=read>\n<parameter=path>\n42\n</parameter>\n<parameter=lines>\n10\n</parameter>\n</function>\n</tool_call>";
        let parsed = parse_output(text, true, &tools);
        assert_eq!(parsed.reasoning.as_deref(), Some("Let me look."));
        assert_eq!(parsed.content, "Reading it.");
        assert_eq!(parsed.tool_calls.len(), 1);
        assert_eq!(parsed.tool_calls[0].name, "read");
        assert_eq!(
            parsed.tool_calls[0].arguments["path"],
            Json::String("42".into())
        );
        assert_eq!(parsed.tool_calls[0].arguments["lines"], Json::from(10));
    }

    #[test]
    fn renders_qwen_style_template_with_python_methods() {
        let t = ChatTemplate::new(
            "{%- for m in messages %}{%- if m.content.startswith('x') %}X{% endif %}<|im_start|>{{ m.role }}\n{{ m.content|trim }}<|im_end|>\n{%- endfor %}{%- if tools %}{{ tools[0]|tojson }}{% endif %}{%- if add_generation_prompt %}<|im_start|>assistant\n{% endif %}",
        )
        .unwrap();
        let out = t
            .render(
                &[serde_json::json!({"role": "user", "content": " xhi "})],
                Some(&[serde_json::json!({"name": "f", "b": 2})]),
                true,
                &Map::new(),
            )
            .unwrap();
        assert_eq!(
            out,
            "<|im_start|>user\nxhi<|im_end|>{\"name\": \"f\", \"b\": 2}<|im_start|>assistant\n"
        );
    }
}
