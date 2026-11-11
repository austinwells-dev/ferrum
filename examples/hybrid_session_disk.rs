//! A session saved to disk must resume as if it had stayed in memory. For
//! each conversation length: one chat turn, then a save; the runtime is
//! dropped and the model loaded again (as the TUI does around a picture),
//! the session loaded from disk, and the same request run again. Its output
//! must be identical and it must reuse the cache; a second turn must reuse
//! as much as it did in memory. Then turn 1 again from that save, which makes
//! the conversation shorter than the save it appends to: its save must load
//! and resume with the same output. Prints prefill time against save and
//! load time, and the bytes each save wrote.
//!
//! usage: hybrid_session_disk MODEL.gguf [--lengths 2000,8000,32000]
//!        [--dir DIR] [--draft ...]
use ferrum::{
    Result,
    hybrid::{
        plan::PlanOptions,
        runtime::{ChatHooks, ChatRequest, ChatResult, Delta, Done, Runtime, SpecOptions},
        session::SamplingParams,
    },
};
use serde_json::{Value as Json, json};
use std::{path::PathBuf, time::Instant};

#[derive(Default)]
struct Hooks {
    reused: usize,
    prompt: usize,
    prefill: Option<Instant>,
    prefill_s: f64,
}

impl ChatHooks for Hooks {
    fn delta(&mut self, _: Delta) -> Result<()> {
        Ok(())
    }
    fn begin(&mut self, prompt: usize, reused: usize) {
        self.prompt = prompt;
        self.reused = reused;
        self.prefill = Some(Instant::now());
    }
    fn prefill(&mut self, processed: usize, total: usize) -> bool {
        if processed == total
            && let Some(start) = self.prefill
        {
            self.prefill_s = start.elapsed().as_secs_f64();
        }
        true
    }
}

/// One chat turn on slot 0, with what its hooks saw.
fn turn(rt: &mut Runtime, request: &ChatRequest) -> Result<(ChatResult, Hooks)> {
    let mut hooks = Hooks::default();
    let prompt = rt.chat_prompt(request)?;
    let mut pending = rt.start_chat(0, request, prompt, &mut hooks)?;
    loop {
        if let Some(done) = rt.advance(0, &mut pending, &mut hooks)? {
            let Done::Chat(result) = done else {
                unreachable!("a chat turn ends in a chat result")
            };
            return Ok((*result, hooks));
        }
    }
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let flag = |name: &str| {
        args.iter()
            .position(|a| a == name)
            .and_then(|i| args.get(i + 1))
    };
    let lengths: Vec<usize> = flag("--lengths")
        .map_or("2000,8000,32000", String::as_str)
        .split(',')
        .map(|n| n.parse().unwrap())
        .collect();
    let dir = flag("--dir").map_or_else(
        || std::env::temp_dir().join(format!("ferrum-session-disk-{}", std::process::id())),
        PathBuf::from,
    );
    let mut spec = None;
    for pair in args[2..].windows(2) {
        SpecOptions::apply_flag(&mut spec, &pair[0], &pair[1])?;
    }
    // The filler takes 1.2 to 1.7 tokens per requested one, depending on
    // the tokenizer; the table reports the real count.
    let context = lengths.iter().max().unwrap() * 2 + 2048;
    let load = || {
        let mut rt = Runtime::load_speculative(
            &args[1],
            PlanOptions {
                context: Some(context),
                ..Default::default()
            },
            spec.as_ref(),
        )?;
        rt.warmup()?;
        Ok::<_, ferrum::Error>(rt)
    };
    let sampling = SamplingParams {
        temperature: 0.0,
        ..Default::default()
    };
    let mut ok = true;
    println!(
        "| tokens | prefill s | save s | save MiB | reload s | load s | reused after load | same output | turn 2 reused (memory / disk) | turn 2 save MiB | shorter re-save |"
    );
    println!("|---|---|---|---|---|---|---|---|---|---|---|");
    for &length in &lengths {
        let _ = std::fs::remove_dir_all(&dir);
        let mut rt = load()?;
        // Varied text sized by `length`.
        let filler: String = (0..length / 12)
            .map(|i| {
                format!(
                    "Note {i}: the value of item {} is {}. ",
                    i * 7 % 101,
                    i * 13 % 997
                )
            })
            .collect();
        let messages = vec![
            json!({"role": "system", "content": "You are a helpful assistant."}),
            json!({"role": "user", "content": format!("{filler}\nWhich item has the largest value? Answer briefly.")}),
        ];
        let request = |messages: &Vec<Json>| {
            let mut r = ChatRequest::new(messages.clone(), sampling.clone());
            r.max_tokens = 48;
            r
        };
        let (a, first) = turn(&mut rt, &request(&messages))?;
        let started = Instant::now();
        let saved = rt.save_session(&dir)?;
        let save_s = started.elapsed().as_secs_f64();

        // Turn 2 in memory, for comparison, then back to the saved state.
        let mut next = messages.clone();
        let mut reply = json!({"role": "assistant", "content": a.output.content});
        if let Some(r) = &a.output.reasoning {
            reply["reasoning_content"] = Json::String(r.clone());
        }
        next.push(reply);
        next.push(json!({"role": "user", "content": "And the smallest?"}));
        let (_, memory) = turn(&mut rt, &request(&next))?;

        drop(rt);
        let started = Instant::now();
        let mut rt = load()?;
        let reload_s = started.elapsed().as_secs_f64();
        let started = Instant::now();
        let loaded = rt.load_session(&dir)?;
        let load_s = started.elapsed().as_secs_f64();
        let (b, again) = turn(&mut rt, &request(&messages))?;
        let same = a.text == b.text;
        // The disk cache again (the retry above extended it), then turn 2.
        rt.session.reset();
        rt.load_session(&dir)?;
        let (_, disk) = turn(&mut rt, &request(&next))?;
        let appended = rt.save_session(&dir)?;
        // Shorter than the save it appends to.
        rt.session.reset();
        rt.load_session(&dir)?;
        let (c, _) = turn(&mut rt, &request(&messages))?;
        rt.save_session(&dir)?;
        rt.session.reset();
        let shorter = rt.load_session(&dir)?;
        let (e, after) = turn(&mut rt, &request(&messages))?;
        let shorter_ok = shorter.is_some()
            && after.reused + 1 >= after.prompt
            && c.text == a.text
            && e.text == a.text;
        let reused_all = loaded.is_some() && again.reused + 1 >= again.prompt;
        ok &= same && reused_all && disk.reused >= memory.reused && shorter_ok;
        println!(
            "| {} | {:.2} | {save_s:.3} | {:.1} | {reload_s:.1} | {load_s:.3} | {} of {} | {} | {} / {} | {:.1} | {} |",
            first.prompt,
            first.prefill_s,
            saved.bytes as f64 / (1 << 20) as f64,
            again.reused,
            again.prompt,
            if same { "yes" } else { "NO" },
            memory.reused,
            disk.reused,
            appended.bytes as f64 / (1 << 20) as f64,
            if shorter_ok { "ok" } else { "FAILED" },
        );
        if !same {
            println!("  memory: {:?}\n  disk:   {:?}", a.text, b.text);
        }
    }
    let _ = std::fs::remove_dir_all(&dir);
    println!("{}", if ok { "PASS" } else { "FAIL" });
    std::process::exit(if ok { 0 } else { 1 });
}
