//! The chat runtime end to end on a real model:
//! `FERRUM_TEST_MODEL=MODEL.gguf cargo test --release e2e -- --ignored`.
//! It covers what the kernel parity tests cannot: template, session reuse,
//! and the paths that change how a token is chosen.

use super::{
    grammar::Grammar,
    plan::PlanOptions,
    runtime::{ChatRequest, ChatResult, Delta, ForcedTool, Runtime},
    session::SamplingParams,
};
use serde_json::{Map, Value as Json, json};
use std::sync::Arc;

fn request(text: &str, max_tokens: usize) -> ChatRequest {
    let mut request = ChatRequest::new(
        vec![json!({"role": "user", "content": text})],
        SamplingParams::default(),
    );
    request.max_tokens = max_tokens;
    // Short replies: no reasoning before them.
    request.template_vars = Map::from_iter([("enable_thinking".to_owned(), Json::Bool(false))]);
    request
}

fn run(rt: &mut Runtime, request: &ChatRequest) -> ChatResult {
    rt.chat(request, |_: Delta| Ok(())).expect("chat")
}

#[test]
#[ignore = "needs FERRUM_TEST_MODEL (a chat GGUF) and the GPU"]
fn e2e_chat_reuse_constraints_and_forced_tools() {
    let path = std::env::var("FERRUM_TEST_MODEL").expect("FERRUM_TEST_MODEL");
    let options = PlanOptions {
        context: Some(4096),
        ..Default::default()
    };
    let mut rt = Runtime::load(path, options).expect("load");

    // Greedy decoding repeats itself, the second time from the cache.
    let plain = request("Name the capital of France in one word.", 16);
    let first = run(&mut rt, &plain);
    assert!(!first.output.content.trim().is_empty());
    assert_eq!(first.completion.reused_tokens, 0);
    let again = run(&mut rt, &plain);
    assert_eq!(again.completion.tokens, first.completion.tokens);
    assert!(again.completion.reused_tokens > 0);

    // Banning the first token of that reply changes it.
    let mut biased = plain.clone();
    biased.logit_bias = vec![(first.completion.tokens[0], f32::NEG_INFINITY)];
    let other = run(&mut rt, &biased);
    assert_ne!(other.completion.tokens[0], first.completion.tokens[0]);

    // A schema-constrained reply is JSON of that shape.
    let mut shaped = request("Give the capital of France.", 64);
    let schema = json!({
        "type": "object",
        "properties": {"capital": {"type": "string"}},
        "required": ["capital"],
        "additionalProperties": false
    });
    shaped.grammar = Some(Arc::new(Grammar::from_schema(&schema).expect("schema")));
    let reply = run(&mut rt, &shaped);
    let value: Json = serde_json::from_str(reply.output.content.trim()).expect("JSON reply");
    assert!(value["capital"].is_string(), "{value}");

    // A forced tool call is made even when the prompt does not ask for one.
    if rt.forces_tools() {
        let mut forced = request("Say hello in French, nothing else.", 96);
        forced.tools = vec![json!({"type": "function", "function": {
            "name": "get_time",
            "description": "Current time in a city",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"]
            }
        }})];
        assert!(run(&mut rt, &forced).output.tool_calls.is_empty());
        forced.force_tool = Some(ForcedTool::Named("get_time".into()));
        let called = run(&mut rt, &forced);
        assert_eq!(called.output.tool_calls.len(), 1, "{}", called.text);
        assert_eq!(called.output.tool_calls[0].name, "get_time");
    }
}
