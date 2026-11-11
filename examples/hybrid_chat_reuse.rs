//! How much of each chat turn's prompt comes from the session's cache, over
//! a scripted conversation shaped like a TUI chat: a tool call, a long tool
//! result (a picture with `--image`, else text), then follow-up questions with
//! thinking on. Chat templates drop past reasoning once a newer user message
//! exists, so each follow-up diverges from the cache at the previous answer;
//! a hybrid model resumes there only from a snapshot.
//!
//! Per turn it prints the prompt length, the prefix the cache shares with it
//! (`possible`), what the session reused, the prefill time and a hash of the
//! reply (to compare builds turn by turn). `--strict N`
//! exits 1 when a turn reuses more than N tokens less than it could.
//!
//! usage: hybrid_chat_reuse MODEL.gguf [--mmproj FILE --image PNG]
//!        [--context N] [--max-tokens N] [--turns N] [--strict N] [--draft ...]
use ferrum::{
    Result,
    hybrid::{
        plan::PlanOptions,
        runtime::{ChatHooks, ChatRequest, Delta, Runtime, SpecOptions},
        session::SamplingParams,
    },
    vision::media,
};
use serde_json::{Value as Json, json};
use std::time::Instant;

struct Hooks {
    reused: usize,
}

impl ChatHooks for Hooks {
    fn delta(&mut self, _: Delta) -> Result<()> {
        Ok(())
    }
    fn begin(&mut self, _prompt: usize, reused: usize) {
        self.reused = reused;
    }
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let flag = |name: &str| {
        args.iter()
            .position(|a| a == name)
            .and_then(|i| args.get(i + 1))
    };
    let count = |name: &str, default: usize| flag(name).map_or(default, |v| v.parse().unwrap());
    let mut spec = None;
    for pair in args[2..].windows(2) {
        SpecOptions::apply_flag(&mut spec, &pair[0], &pair[1])?;
    }
    let image = flag("--image").map(|p| std::fs::read(p).expect("--image"));
    let mut rt = Runtime::load_multimodal(
        &args[1],
        PlanOptions {
            context: Some(count("--context", 16384)),
            ..Default::default()
        },
        spec.as_ref(),
        flag("--mmproj").map(std::path::Path::new),
        ferrum::vision::DEFAULT_MAX_TOKENS,
    )?;
    rt.warmup()?;
    let max_tokens = count("--max-tokens", 600);
    let strict = flag("--strict").map(|v| v.parse::<usize>().unwrap());

    let (tool, ask) = match image {
        Some(_) => (
            json!({"type": "function", "function": {
                "name": "generate_image",
                "description": "Draw a picture from a prompt and show it to you.",
                "parameters": {"type": "object", "properties": {"prompt": {"type": "string"}}, "required": ["prompt"]}}}),
            "Draw a cat on a red sofa with generate_image, then tell me what you see.",
        ),
        None => (
            json!({"type": "function", "function": {
                "name": "read_file",
                "description": "Read a file from the project.",
                "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}}),
            "Read src/hybrid/session.rs with read_file and explain what reuse_prefix does.",
        ),
    };
    let tool_text = include_str!("../src/hybrid/session.rs")
        .chars()
        .take(6000)
        .collect::<String>();
    let follow_ups = [
        "Explain that again, step by step, in more detail.",
        "What could go wrong with it? Think it through carefully.",
        "Summarise everything so far in a short list.",
        "Which part matters most, and why?",
    ];
    let mut messages = vec![
        json!({"role": "system", "content": "You are a helpful assistant."}),
        json!({"role": "user", "content": ask}),
    ];
    let sampling = SamplingParams {
        temperature: 0.0,
        ..Default::default()
    };
    let turns = count("--turns", 5);
    let mut lost_total = 0;
    let mut worst = 0;
    println!("| turn | prompt | possible | reused | lost | prefill s | reply |");
    println!("|---|---|---|---|---|---|---|");
    let mut asked = 0;
    let mut served = false;
    for turn in 0..turns {
        let mut request = ChatRequest::new(messages.clone(), sampling.clone());
        request.tools = vec![tool.clone()];
        request.max_tokens = max_tokens;
        let prompt = rt.chat_prompt(&request)?;
        let possible = rt.session.shared_prefix(&prompt.tokens, &prompt.spans);
        let len = prompt.tokens.len();
        if args.iter().any(|a| a == "--why") && possible < rt.session.len() {
            let tok = &rt.loaded.tokenizer;
            let show = |ids: &[u32]| {
                ids.iter()
                    .map(|&t| tok.decode_lenient(&[t]).unwrap_or_default())
                    .collect::<Vec<_>>()
            };
            let from = possible.saturating_sub(4);
            let cached = rt.session.tokens();
            eprintln!(
                "turn {turn} diverges at {possible} (cached {}):\n  cached {:?}\n  prompt {:?}",
                cached.len(),
                show(&cached[from..(possible + 6).min(cached.len())]),
                show(&prompt.tokens[from..(possible + 6).min(len)])
            );
        }
        let mut hooks = Hooks { reused: 0 };
        let started = Instant::now();
        let mut pending = rt.start_chat(0, &request, prompt, &mut hooks)?;
        let mut first = None;
        let result = loop {
            if first.is_none() && pending.decoding() {
                first = Some(started.elapsed().as_secs_f64());
            }
            if let Some(done) = rt.advance(0, &mut pending, &mut hooks)? {
                let ferrum::hybrid::runtime::Done::Chat(r) = done else {
                    unreachable!()
                };
                break *r;
            }
        };
        let lost = possible - hooks.reused.min(possible);
        lost_total += lost;
        worst = worst.max(lost);
        // FNV-1a of the raw reply.
        let hash = result.text.bytes().fold(0xcbf29ce484222325u64, |h, b| {
            (h ^ u64::from(b)).wrapping_mul(0x100000001b3)
        });
        println!(
            "| {turn} | {len} | {possible} | {} | {lost} | {:.2} | {:08x} |",
            hooks.reused,
            first.unwrap_or_default(),
            hash >> 32
        );
        let mut reply = json!({"role": "assistant", "content": result.output.content});
        if let Some(r) = result.output.reasoning {
            reply["reasoning_content"] = Json::String(r);
        }
        // Every call gets a reply, as in the TUI: the first the real result,
        // later ones a short note, so the conversation moves on.
        let mut calls = Vec::new();
        for call in &result.output.tool_calls {
            calls.push(json!({"id": format!("call_{turn}_{}", calls.len()), "type": "function", "function": {
                "name": call.name,
                "arguments": Json::Object(call.arguments.clone()).to_string()}}));
        }
        if !calls.is_empty() {
            reply["tool_calls"] = Json::Array(calls.clone());
            messages.push(reply);
            for call in &calls {
                let content = match (&image, served) {
                    (Some(bytes), false) => json!([
                        media::image_part_with(bytes, Default::default()),
                        {"type": "text", "text": "saved cat.png (1024x1024) in 60 s"},
                    ]),
                    (None, false) => Json::String(tool_text.clone()),
                    (_, true) => Json::String("done; nothing more to read".into()),
                };
                served = true;
                messages.push(json!({"role": "tool", "tool_call_id": call["id"], "name": call["function"]["name"], "content": content}));
            }
        } else {
            messages.push(reply);
            messages.push(json!({"role": "user", "content": follow_ups[asked % follow_ups.len()]}));
            asked += 1;
        }
    }
    println!("lost {lost_total} tokens in all, {worst} at worst");
    if strict.is_some_and(|n| worst > n) {
        std::process::exit(1);
    }
    Ok(())
}
