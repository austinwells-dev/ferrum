//! How long each turn of an image chat takes to start: one new picture per
//! user turn, so turn N's prompt carries N pictures. A chat re-sends its whole
//! history every turn; only pictures past the session's cached prefix should
//! be encoded.
//!
//! Per turn it prints the image count, prompt length, reused tokens, the
//! pictures it encoded (as `ChatHooks::encoding` reports them), time
//! encoding them, time until the first output token, and a hash of the
//! greedy reply (to compare builds turn by turn).
//!
//! usage: hybrid_image_turns MODEL.gguf --mmproj FILE [--turns N]
//!        [--size PIXELS] [--context N] [--max-tokens N]
use ferrum::{
    Result,
    hybrid::{
        plan::PlanOptions,
        runtime::{ChatHooks, ChatRequest, Delta, Done, Runtime},
        session::SamplingParams,
    },
};
use serde_json::json;
use std::time::Instant;

struct Hooks {
    reused: usize,
    /// Images the turn reported encoding (`ChatHooks::encoding`).
    encoded: usize,
}

impl ChatHooks for Hooks {
    fn delta(&mut self, _: Delta) -> Result<()> {
        Ok(())
    }
    fn begin(&mut self, _prompt: usize, reused: usize) {
        self.reused = reused;
    }
    fn encoding(&mut self, _done: usize, total: usize) -> bool {
        self.encoded = total;
        true
    }
}

/// A distinct, deterministic picture per turn: coloured bands and a gradient.
fn picture(turn: usize, size: u32) -> String {
    let img = image::RgbImage::from_fn(size, size, |x, y| {
        let band = ((x + y * 3 + turn as u32 * 41) / 37) % 6;
        let base = [
            [200, 40, 40],
            [40, 160, 60],
            [50, 70, 210],
            [220, 200, 40],
            [30, 30, 30],
            [240, 240, 240],
        ][band as usize];
        image::Rgb(base.map(|c: u32| ((c + x * 255 / size) / 2) as u8))
    });
    let mut png = Vec::new();
    img.write_to(&mut std::io::Cursor::new(&mut png), image::ImageFormat::Png)
        .expect("png");
    format!("data:image/png;base64,{}", base64(&png))
}

fn base64(bytes: &[u8]) -> String {
    const T: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::with_capacity(bytes.len().div_ceil(3) * 4);
    for chunk in bytes.chunks(3) {
        let n = chunk
            .iter()
            .enumerate()
            .fold(0u32, |n, (i, &b)| n | u32::from(b) << (16 - 8 * i));
        for i in 0..4 {
            out.push(if i <= chunk.len() {
                T[(n >> (18 - 6 * i) & 63) as usize] as char
            } else {
                '='
            });
        }
    }
    out
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let flag = |name: &str| {
        args.iter()
            .position(|a| a == name)
            .and_then(|i| args.get(i + 1))
    };
    let count = |name: &str, default: usize| flag(name).map_or(default, |v| v.parse().unwrap());
    let mut rt = Runtime::load_multimodal(
        &args[1],
        PlanOptions {
            context: Some(count("--context", 32768)),
            ..Default::default()
        },
        None,
        Some(std::path::Path::new(flag("--mmproj").expect("--mmproj"))),
        ferrum::vision::DEFAULT_MAX_TOKENS,
    )?;
    rt.warmup()?;
    let sampling = SamplingParams {
        temperature: 0.0,
        ..Default::default()
    };
    let size = count("--size", 1024) as u32;
    let mut messages = vec![json!({"role": "system", "content": "You are a helpful assistant."})];
    println!("| turn | images | prompt | reused | encoded | encode s | first token s | reply |");
    println!("|---|---|---|---|---|---|---|---|");
    for turn in 0..count("--turns", 12) {
        messages.push(json!({"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": picture(turn, size)}},
            {"type": "text", "text": "Name the main colours in this picture, in one short sentence."},
        ]}));
        let mut request = ChatRequest::new(messages.clone(), sampling.clone());
        request.max_tokens = count("--max-tokens", 24);
        request
            .template_vars
            .insert("enable_thinking".into(), json!(false));
        let mut hooks = Hooks {
            reused: 0,
            encoded: 0,
        };
        let started = Instant::now();
        let prompt = rt.chat_prompt(&request)?;
        let len = prompt.tokens.len();
        let mut pending = rt.start_chat(0, &request, prompt, &mut hooks)?;
        let encode = rt
            .vision
            .as_ref()
            .map_or(0., |v| v.last_encode.as_secs_f64());
        let mut first = None;
        let result = loop {
            if first.is_none() && pending.decoding() {
                first = Some(started.elapsed().as_secs_f64());
            }
            if let Some(Done::Chat(r)) = rt.advance(0, &mut pending, &mut hooks)? {
                break *r;
            }
        };
        let hash = result.text.bytes().fold(0xcbf29ce484222325u64, |h, b| {
            (h ^ u64::from(b)).wrapping_mul(0x100000001b3)
        });
        println!(
            "| {turn} | {} | {len} | {} | {} | {encode:.2} | {:.2} | {:08x} |",
            turn + 1,
            hooks.reused,
            hooks.encoded,
            first.unwrap_or_default(),
            hash >> 32
        );
        messages.push(json!({"role": "assistant", "content": result.output.content}));
    }
    Ok(())
}
