//! Long-context check: hide a passphrase at several depths of a long filler
//! document, ask for it through the chat template, and report prefill
//! throughput, the answer and device memory before and after.
//!
//! usage: hybrid_needle MODEL.gguf FILLER.txt TOKENS [depths=0.1,0.5,0.9]
use ferrum::{
    Result,
    hybrid::{
        plan::PlanOptions,
        runtime::{ChatRequest, Runtime},
        session::SamplingParams,
    },
};
use serde_json::{Map, Value as Json, json};

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let target: usize = args[3].parse().expect("token count");
    let depths: Vec<f64> = args
        .get(4)
        .map_or("0.1,0.5,0.9".into(), |v| v.clone())
        .split(',')
        .map(|v| v.parse().expect("depth"))
        .collect();
    let filler = std::fs::read_to_string(&args[2]).expect("filler text");
    let mut rt = Runtime::load(
        &args[1],
        PlanOptions {
            context: Some(target + 1024),
            ..Default::default()
        },
    )?;
    let mem_start = rt.device.allocated_bytes();
    // Filler paragraphs, repeated to the target length.
    let paragraphs: Vec<&str> = filler
        .split("\n\n")
        .filter(|p| p.trim().len() > 40)
        .collect();
    let per_token = {
        let all = paragraphs.join("\n\n");
        all.len() as f64 / rt.loaded.tokenizer.encode(&all)?.len() as f64
    };
    let target_chars = (target as f64 * per_token * 0.97) as usize;
    for (round, depth) in depths.iter().enumerate() {
        let code = format!(
            "{:04}-{}",
            1000 + round * 7919 % 9000,
            ["amber", "violet", "cobalt", "saffron"][round % 4]
        );
        let needle = format!("\n\nIMPORTANT: the secret passphrase is {code}. Remember it.\n\n");
        let build = |chars: usize| {
            let mut doc = String::new();
            let insert_at = (chars as f64 * depth) as usize;
            let mut inserted = false;
            let mut i = 0;
            while doc.len() < chars {
                if !inserted && doc.len() >= insert_at {
                    doc.push_str(&needle);
                    inserted = true;
                }
                doc.push_str(paragraphs[i % paragraphs.len()]);
                doc.push_str("\n\n");
                i += 1;
            }
            if !inserted {
                doc.push_str(&needle);
            }
            doc
        };
        // Shrink until the rendered document is within the target token count.
        let mut chars = target_chars;
        let mut doc = build(chars);
        for _ in 0..4 {
            let tokens = rt.loaded.tokenizer.encode(&doc)?.len();
            if tokens <= target {
                break;
            }
            chars = (chars as f64 * target as f64 / tokens as f64 * 0.99) as usize;
            doc = build(chars);
        }
        let mut vars = Map::new();
        vars.insert("enable_thinking".into(), Json::Bool(false));
        let request = ChatRequest {
            messages: vec![json!({"role": "user", "content": format!(
                "{doc}\n\nWhat is the secret passphrase mentioned in the document above? Reply with the passphrase only.")})],
            tools: Vec::new(),
            max_tokens: 24,
            sampling: SamplingParams::default(),
            template_vars: vars,
            stop: Vec::new(),
        };
        rt.session.reset();
        let result = rt.chat(&request, |_| Ok(()))?;
        let c = &result.completion;
        println!(
            "depth {depth:.2}: {} prompt tokens, prefill {:.1} s ({:.0} tok/s), decode {:.1} tok/s, answer {:?} -> {}",
            c.prompt_tokens,
            c.prefill.as_secs_f64(),
            c.prompt_tokens as f64 / c.prefill.as_secs_f64(),
            c.tokens.len() as f64 / c.decode.as_secs_f64().max(1e-9),
            result.output.content,
            if result.output.content.contains(&code) {
                "PASS"
            } else {
                "FAIL"
            }
        );
    }
    println!(
        "device memory: {:.1} MiB at start, {:.1} MiB at end (plan predicted {:.1} MiB)",
        mem_start as f64 / 1048576.,
        rt.device.allocated_bytes() as f64 / 1048576.,
        rt.loaded.plan.total() as f64 / 1048576.
    );
    Ok(())
}
