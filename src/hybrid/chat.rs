//! Chat templates embedded in GGUF files, rendered the way Hugging Face
//! `apply_chat_template` does, and parsing of generated text into
//! reasoning, content and XML-style tool calls.
#![forbid(unsafe_code)]
use crate::{Error, Result};
use minijinja::{Environment, ErrorKind, Value};
use serde_json::{Map, Value as Json};

pub struct ChatTemplate {
    env: Environment<'static>,
    source: String,
    bos_token: String,
    eos_token: String,
    /// See `opens_thinking`.
    opens_thinking: bool,
}

/// Keyword arguments of Python's `json.dumps` that `tojson` accepts.
struct JsonStyle<'a> {
    indent: Option<usize>,
    /// Between items, and between a key and its value (`separators`).
    item: &'a str,
    key: &'a str,
    sort_keys: bool,
}

impl JsonStyle<'_> {
    /// `json.dumps` defaults: `", "` between items (`","` when indenting).
    fn new(indent: Option<usize>) -> Self {
        Self {
            indent,
            item: if indent.is_some() { "," } else { ", " },
            key: ": ",
            sort_keys: false,
        }
    }
}

/// Python `json.dumps(value, ensure_ascii=False, ...)`: insertion order unless
/// `sort_keys`, no HTML escaping (Hugging Face's `tojson`).
fn python_json(value: &Json, style: &JsonStyle, depth: usize, out: &mut String) {
    let newline = |out: &mut String, depth: usize| {
        if let Some(width) = style.indent {
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
            let mut entries: Vec<_> = map.iter().collect();
            if style.sort_keys {
                entries.sort_by(|a, b| a.0.cmp(b.0));
            }
            out.push('{');
            for (i, (k, v)) in entries.into_iter().enumerate() {
                if i > 0 {
                    out.push_str(style.item);
                }
                newline(out, depth + 1);
                out.push_str(&serde_json::to_string(k).unwrap_or_default());
                out.push_str(style.key);
                python_json(v, style, depth + 1, out);
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
                    out.push_str(style.item);
                }
                newline(out, depth + 1);
                python_json(v, style, depth + 1, out);
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
    /// A template for a model whose chat turns end with `<|im_end|>` (Qwen3.5).
    pub fn new(source: &str) -> Result<Self> {
        Self::with_special_tokens(source, "", "<|im_end|>")
    }

    /// A template with the model's `bos_token` and `eos_token` strings.
    pub fn with_special_tokens(source: &str, bos_token: &str, eos_token: &str) -> Result<Self> {
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
                let mut style = JsonStyle::new(kwargs.get("indent")?);
                // Hugging Face's default is ensure_ascii=False.
                let ensure_ascii: bool =
                    kwargs.get::<Option<bool>>("ensure_ascii")?.unwrap_or(false);
                style.sort_keys = kwargs.get::<Option<bool>>("sort_keys")?.unwrap_or(false);
                let separators: Option<Vec<String>> = kwargs.get("separators")?;
                kwargs.assert_all_used()?;
                if let Some(s) = &separators {
                    let [item, key] = s.as_slice() else {
                        return Err(minijinja::Error::new(
                            ErrorKind::InvalidOperation,
                            "tojson separators must be (item_separator, key_separator)",
                        ));
                    };
                    (style.item, style.key) = (item, key);
                }
                let json: Json = serde_json::to_value(&value).map_err(|e| {
                    minijinja::Error::new(ErrorKind::InvalidOperation, e.to_string())
                })?;
                let mut out = String::new();
                python_json(&json, &style, 0, &mut out);
                if ensure_ascii {
                    // Non-ASCII only occurs inside JSON strings, so escaping the
                    // output is the same as escaping the strings.
                    out = ascii_escape(&out);
                }
                Ok(Value::from_safe_string(out))
            },
        );
        env.add_template_owned("chat", source.to_owned())
            .map_err(|e| Error::Tokenizer(format!("chat template: {e:#}")))?;
        Ok(Self {
            env,
            source: source.to_owned(),
            bos_token: bos_token.to_owned(),
            eos_token: eos_token.to_owned(),
            opens_thinking: false,
        })
    }

    /// Mark the template as one whose prompt must open or close the think
    /// block itself (Flint's template has no thinking switch).
    pub fn with_opens_thinking(mut self, on: bool) -> Self {
        self.opens_thinking = on;
        self
    }

    pub fn source(&self) -> &str {
        &self.source
    }

    /// Whether the generation prompt needs `<think>` (or an empty think block)
    /// appended, because the template does not write one (Flint).
    pub fn opens_thinking(&self) -> bool {
        self.opens_thinking
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
        context.insert("bos_token".into(), Json::String(self.bos_token.clone()));
        context.insert("eos_token".into(), Json::String(self.eos_token.clone()));
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

/// Python `json.dumps(ensure_ascii=True)` escaping of every non-ASCII character.
fn ascii_escape(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    for c in text.chars() {
        if c.is_ascii() {
            out.push(c);
        } else {
            let mut units = [0u16; 2];
            for unit in c.encode_utf16(&mut units) {
                out.push_str(&format!("\\u{unit:04x}"));
            }
        }
    }
    out
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

/// One `<tool_call>` body in either format: Qwen3.5's `<function=…>` with
/// `<parameter=…>` blocks, or Flint's `name<arg_key>…</arg_key><arg_value>…`
/// pairs. `None` leaves the text in the content.
fn parse_tool_call(body: &str, tools: &[Json]) -> Option<ToolCall> {
    if body.contains("<function=") {
        parse_qwen_tool_call(body, tools)
    } else {
        parse_flint_tool_call(body, tools)
    }
}

/// The function object of tool `name` in the OpenAI tool list.
fn tool_function<'a>(tools: &'a [Json], name: &str) -> Option<&'a Json> {
    tools
        .iter()
        .map(|t| t.get("function").unwrap_or(t))
        .find(|f| f.get("name").and_then(Json::as_str) == Some(name))
}

/// The declared parameter schema of tool `name` in the OpenAI tool list.
fn tool_schema(tools: &[Json], name: &str) -> Option<Json> {
    tool_function(tools, name)?
        .pointer("/parameters/properties")
        .cloned()
}

/// A parameter value typed by its declared schema: strings stay as written,
/// anything else is read as JSON and kept as text when that fails.
fn typed_value(schema: Option<&Json>, key: &str, raw: &str) -> Json {
    let declared = schema
        .and_then(|s| s.get(key))
        .and_then(|p| p.get("type"))
        .and_then(Json::as_str);
    match declared {
        Some("string") => Json::String(raw.to_owned()),
        _ => serde_json::from_str(raw.trim()).unwrap_or_else(|_| Json::String(raw.to_owned())),
    }
}

fn parse_qwen_tool_call(body: &str, tools: &[Json]) -> Option<ToolCall> {
    let start = body.find("<function=")? + "<function=".len();
    let name_end = start + body[start..].find('>')?;
    let name = body[start..name_end].trim().to_owned();
    let schema = tool_schema(tools, &name);
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
        arguments.insert(key.clone(), typed_value(schema.as_ref(), &key, raw));
        cursor = &value_start[value_end..];
    }
    Some(ToolCall { name, arguments })
}

/// Flint: `<tool_call>NAME<arg_key>K</arg_key><arg_value>V</arg_value>…</tool_call>`,
/// with no separators between the pairs (the template writes them back to back).
/// The bare name carries no marker of its own, so it must name a declared tool.
fn parse_flint_tool_call(body: &str, tools: &[Json]) -> Option<ToolCall> {
    let (name, mut rest) = match body.find("<arg_key>") {
        Some(p) => (&body[..p], &body[p..]),
        None => (body, ""),
    };
    let name = name.trim();
    tool_function(tools, name)?;
    let schema = tool_schema(tools, name);
    let mut arguments = Map::new();
    while !rest.trim().is_empty() {
        let after = rest.trim_start().strip_prefix("<arg_key>")?;
        let key_end = after.find("</arg_key>")?;
        let key = after[..key_end].trim();
        if key.is_empty() {
            return None;
        }
        let after = after[key_end + "</arg_key>".len()..]
            .trim_start()
            .strip_prefix("<arg_value>")?;
        let value_end = after.find("</arg_value>")?;
        let raw = &after[..value_end];
        arguments.insert(key.to_owned(), typed_value(schema.as_ref(), key, raw));
        rest = &after[value_end + "</arg_value>".len()..];
    }
    Some(ToolCall {
        name: name.to_owned(),
        arguments,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tojson_matches_python_json_dumps() {
        let value: Json = serde_json::from_str(r#"{"b": 1, "a": [1.0, "x<y"], "c": {}}"#).unwrap();
        let mut out = String::new();
        python_json(&value, &JsonStyle::new(None), 0, &mut out);
        assert_eq!(out, r#"{"b": 1, "a": [1.0, "x<y"], "c": {}}"#);
    }

    /// `indent`, `separators` and `sort_keys` as `json.dumps` applies them.
    #[test]
    fn tojson_takes_python_keyword_arguments() {
        let t = ChatTemplate::new(
            "{{ v | tojson(separators=(',', ':')) }}|{{ v | tojson(indent=2) }}|{{ v | tojson(sort_keys=true) }}",
        )
        .unwrap();
        let out = t
            .render(
                &[],
                None,
                false,
                &serde_json::from_str(r#"{"v": {"b": 1, "a": [1.0, "x"], "c": {}}}"#).unwrap(),
            )
            .unwrap();
        assert_eq!(
            out,
            "{\"b\":1,\"a\":[1.0,\"x\"],\"c\":{}}|\
             {\n  \"b\": 1,\n  \"a\": [\n    1.0,\n    \"x\"\n  ],\n  \"c\": {}\n}|\
             {\"a\": [1.0, \"x\"], \"b\": 1, \"c\": {}}"
        );
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
    fn parses_flint_tool_calls_typed_by_schema() {
        let tools: Vec<Json> = vec![serde_json::json!({"type": "function", "function": {
        "name": "get_weather",
        "parameters": {"properties": {
            "city": {"type": "string"}, "days": {"type": "integer"}, "metric": {"type": "boolean"}
        }}}})];
        let text = "Let me check.\n<tool_call>get_weather<arg_key>city</arg_key><arg_value>10</arg_value><arg_key>days</arg_key><arg_value>3</arg_value><arg_key>metric</arg_key><arg_value>true</arg_value></tool_call>";
        let parsed = parse_output(text, false, &tools);
        assert_eq!(parsed.content, "Let me check.");
        assert_eq!(parsed.tool_calls.len(), 1);
        let call = &parsed.tool_calls[0];
        assert_eq!(call.name, "get_weather");
        // "city" is a string in the schema, so "10" stays text.
        assert_eq!(call.arguments["city"], Json::String("10".into()));
        assert_eq!(call.arguments["days"], Json::from(3));
        assert_eq!(call.arguments["metric"], Json::Bool(true));
    }

    #[test]
    fn types_the_image_tools_arguments() {
        let tools: Vec<Json> = vec![
            serde_json::json!({"type": "function", "function": {
                "name": "view_image",
                "parameters": {"properties": {"path": {"type": "string"}}}}}),
            serde_json::json!({"type": "function", "function": {
                "name": "generate_image",
                "parameters": {"properties": {
                    "prompt": {"type": "string"},
                    "inject": {"type": "boolean"},
                    "seed": {"type": "integer"}}}}}),
        ];
        let text = "<tool_call>\n<function=view_image>\n<parameter=path>\nimages/fox.png\n\
                    </parameter>\n</function>\n</tool_call>\n\
                    <tool_call>\n<function=generate_image>\n<parameter=prompt>\na fox\n\
                    </parameter>\n<parameter=inject>\nfalse\n</parameter>\n\
                    <parameter=seed>\n7\n</parameter>\n</function>\n</tool_call>";
        let parsed = parse_output(text, false, &tools);
        assert_eq!(parsed.tool_calls.len(), 2);
        assert_eq!(parsed.tool_calls[0].name, "view_image");
        assert_eq!(
            parsed.tool_calls[0].arguments["path"],
            Json::String("images/fox.png".into())
        );
        assert_eq!(parsed.tool_calls[1].name, "generate_image");
        // A boolean from the schema is a JSON boolean, not the text "false".
        assert_eq!(parsed.tool_calls[1].arguments["inject"], Json::Bool(false));
        assert_eq!(parsed.tool_calls[1].arguments["seed"], Json::from(7));
    }

    #[test]
    fn flint_calls_without_arguments_and_malformed_blocks() {
        let tools = vec![serde_json::json!({"name": "now"})];
        let parsed = parse_output("<tool_call>now</tool_call>", false, &tools);
        assert_eq!(parsed.tool_calls.len(), 1);
        assert_eq!(parsed.tool_calls[0].name, "now");
        assert!(parsed.tool_calls[0].arguments.is_empty());
        // A block that is not a call stays visible; a truncated pair is not
        // half-parsed.
        let text = "See <tool_call>not a call</tool_call> then <tool_call>now<arg_key>x</arg_key><arg_value>1</tool_call>";
        let parsed = parse_output(text, false, &tools);
        assert!(parsed.tool_calls.is_empty());
        assert!(parsed.content.contains("not a call"));
        assert!(parsed.content.contains("<arg_value>1"));
        // A bare word that names no declared tool is text, with or without tools.
        for tools in [&tools[..], &[]] {
            let parsed = parse_output("<tool_call>later</tool_call>", false, tools);
            assert!(parsed.tool_calls.is_empty());
            assert_eq!(parsed.content, "<tool_call>later</tool_call>");
        }
    }

    /// Every case in `cases` (a render_cases.json) renders byte for byte as
    /// Jinja2 renders `template` (see the fixture's render_cases.py).
    fn renders_like_jinja2(template: &ChatTemplate, cases: &str) {
        let cases: Vec<Json> = serde_json::from_str(cases).unwrap();
        for case in cases {
            let name = case["name"].as_str().unwrap();
            let messages = case["messages"].as_array().unwrap();
            let tools = case["tools"].as_array().map(Vec::as_slice);
            let extra = case
                .get("extra")
                .and_then(Json::as_object)
                .cloned()
                .unwrap_or_default();
            let out = template
                .render(
                    messages,
                    tools,
                    case["add_generation_prompt"].as_bool().unwrap(),
                    &extra,
                )
                .unwrap();
            assert_eq!(out, case["expect"].as_str().unwrap(), "case {name}");
        }
    }

    #[test]
    fn renders_flint_template_like_jinja2() {
        let template = ChatTemplate::with_special_tokens(
            include_str!("../../tests/fixtures/flint/chat_template.jinja"),
            "<|begin_of_text|>",
            "<|end_of_text|>",
        )
        .unwrap();
        renders_like_jinja2(
            &template,
            include_str!("../../tests/fixtures/flint/render_cases.json"),
        );
    }

    /// Spark's template calls `tojson(separators=...)`, reads `loop.previtem`
    /// and `loop.nextitem`, and switches on `enable_thinking` and
    /// `reasoning_effort`.
    #[test]
    fn renders_spark_template_like_jinja2() {
        let template = ChatTemplate::with_special_tokens(
            include_str!("../../tests/fixtures/spark/chat_template.jinja"),
            "",
            "",
        )
        .unwrap();
        renders_like_jinja2(
            &template,
            include_str!("../../tests/fixtures/spark/render_cases.json"),
        );
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
