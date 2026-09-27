//! Speculative decoding benchmark and losslessness check on chat prompts.
//! usage: hybrid_spec MODEL.gguf --drafter mtp|DRAFT_DIR [--drafts K]
//!        [--tokens 256] [--context 8192] [--temp 0] [--baseline] [--prompts N]
//!        [--pmin P]  (DSpark confidence cut-off)
//!
//! Each prompt is rendered with the model's chat template (thinking on) and
//! generated with speculation. With --baseline the same prompt is first
//! generated without speculation; greedy outputs are compared token by token
//! and any divergence is reported with the target's top-2 logit margin at
//! that point (a near-tie means single-row and multi-row kernels rounded
//! differently, not a verification error).
use ferrum::{
    Result,
    hybrid::{
        HybridState, Output, Produced,
        draft::DraftOptions,
        plan::PlanOptions,
        runtime::{ChatRequest, DraftSource, Runtime, SpecOptions},
        session::SamplingParams,
        speculative::SpecStats,
    },
};
use serde_json::json;
use std::time::Instant;

const PROMPTS: &[&str] = &[
    "Write a Python function that returns the n-th Fibonacci number using memoization, then explain its complexity.",
    "A train leaves at 3:40 pm and travels 210 km at 84 km/h. At what time does it arrive? Show your steps.",
    "Explain the difference between a mutex and a semaphore, with a short example of each in C.",
    "Summarize the causes of the French Revolution in five bullet points.",
    "Implement binary search in Rust on a sorted slice of i32 and add unit tests.",
    "What is the derivative of x^3 * ln(x)? Explain each step.",
    "Write a haiku about autumn leaves, then explain the imagery you chose.",
    "Convert this JSON to YAML: {\"name\": \"ferrum\", \"version\": 9, \"features\": [\"metal\", \"speculation\"]}",
];

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let get = |key: &str, default: &str| -> String {
        args.iter()
            .position(|a| a == key)
            .map_or(default.to_owned(), |i| args[i + 1].clone())
    };
    let has = |key: &str| args.iter().any(|a| a == key);
    let max_tokens: usize = get("--tokens", "256").parse().expect("tokens");
    let context: usize = get("--context", "8192").parse().expect("context");
    let temperature: f32 = get("--temp", "0").parse().expect("temp");
    let prompts: usize = get("--prompts", &PROMPTS.len().to_string())
        .parse()
        .expect("prompts");
    let drafter_arg = get("--drafter", "mtp");
    let drafts: usize = get("--drafts", "0").parse().expect("drafts");
    let spec = SpecOptions {
        source: DraftSource::parse(&drafter_arg),
        max_drafts: (drafts > 0).then_some(drafts),
        draft: DraftOptions {
            confidence_min: get("--pmin", "0").parse().expect("pmin"),
            vocab: get("--vocab", "0").parse().ok().filter(|&v: &usize| v > 0),
            quant: if get("--quant", "q4_0") == "q4_0" {
                ferrum::hybrid::draft::DraftQuant::Q4_0
            } else {
                ferrum::hybrid::draft::DraftQuant::Q8_0
            },
            ..Default::default()
        },
    };
    let mut rt = Runtime::load_speculative(
        &args[1],
        PlanOptions {
            context: Some(context),
            ..Default::default()
        },
        Some(&spec),
    )?;
    let plan = &rt.loaded.plan;
    let drafter = rt.session.drafter().expect("drafter attached");
    println!(
        "drafter {} (max {} drafts); memory predicted {:.1} MiB, measured {:.1} MiB",
        drafter.name(),
        drafter.max_drafts(),
        plan.draft_total(plan.context) as f64 / 1048576.,
        rt.draft_loaded_bytes as f64 / 1048576.
    );
    rt.warmup()?;
    let sampling = SamplingParams {
        temperature,
        top_k: if temperature > 0. { 20 } else { 0 },
        top_p: if temperature > 0. { 0.95 } else { 1. },
        seed: 7,
        ..Default::default()
    };
    let (mut base_tokens, mut base_secs) = (0usize, 0f64);
    let (mut spec_tokens, mut spec_secs) = (0usize, 0f64);
    let mut total = SpecStats::default();
    let mut mismatches = 0;
    for (index, prompt) in PROMPTS.iter().take(prompts).enumerate() {
        let request = ChatRequest::new(
            vec![json!({"role": "user", "content": prompt})],
            sampling.clone(),
        );
        let text = rt.render(&request)?;
        let tokens = rt.loaded.tokenizer.encode(&text)?;
        let eos = rt.loaded.eos_ids.clone();
        let run = |rt: &mut Runtime, speculate: bool| -> Result<(Vec<u32>, f64)> {
            rt.session.reset();
            rt.session.speculate = speculate;
            let start = Instant::now();
            let completion = rt.session.generate(
                &rt.device,
                &rt.loaded.model,
                &tokens,
                max_tokens,
                &sampling,
                &eos,
                &mut |_| Ok(true),
            )?;
            let _ = start;
            Ok((completion.tokens, completion.decode.as_secs_f64()))
        };
        let base = if has("--baseline") {
            let (t, secs) = run(&mut rt, false)?;
            base_tokens += t.len();
            base_secs += secs;
            Some((t, secs))
        } else {
            None
        };
        let (spec, secs) = run(&mut rt, true)?;
        let stats = rt.session.spec_stats.clone();
        spec_tokens += spec.len();
        spec_secs += secs;
        let mut line = format!(
            "prompt {index}: {} tokens, {:.1} tok/s, acceptance {:.2} ({} steps)",
            spec.len(),
            spec.len() as f64 / secs,
            stats.acceptance_length(),
            stats.steps
        );
        if let Some((base, bsecs)) = base {
            line += &format!(", baseline {:.1} tok/s", base.len() as f64 / bsecs);
            if temperature == 0. {
                match base.iter().zip(&spec).position(|(a, b)| a != b) {
                    None if base.len() == spec.len() => line += ", identical",
                    None => {
                        line += &format!(", prefix-identical ({} vs {})", base.len(), spec.len())
                    }
                    Some(i) => {
                        mismatches += 1;
                        let margin = margin(&rt, &tokens, &base[..i])?;
                        line += &format!(", diverges at {i} (target top-2 margin {margin:.4})");
                    }
                }
            }
        }
        println!("{line}");
        total.steps += stats.steps;
        total.drafted += stats.drafted;
        total.accepted += stats.accepted;
        if total.accepted_at.len() < stats.accepted_at.len() {
            total.accepted_at.resize(stats.accepted_at.len(), 0);
        }
        for (a, b) in total.accepted_at.iter_mut().zip(&stats.accepted_at) {
            *a += b;
        }
        total.draft_seconds += stats.draft_seconds;
        total.verify_seconds += stats.verify_seconds;
    }
    println!(
        "\nspeculative: {:.2} tok/s, acceptance length {:.3}, draft {:.1} ms/step, verify {:.1} ms/step",
        spec_tokens as f64 / spec_secs,
        total.acceptance_length(),
        total.draft_seconds * 1e3 / total.steps.max(1) as f64,
        total.verify_seconds * 1e3 / total.steps.max(1) as f64,
    );
    let rates: Vec<String> = total
        .accepted_at
        .iter()
        .map(|&a| format!("{:.0}%", 100. * a as f64 / total.steps.max(1) as f64))
        .collect();
    println!("acceptance by position: {}", rates.join(" "));
    if base_tokens > 0 {
        println!(
            "baseline: {:.2} tok/s; speedup {:.2}x; greedy mismatches {mismatches}",
            base_tokens as f64 / base_secs,
            (spec_tokens as f64 / spec_secs) / (base_tokens as f64 / base_secs)
        );
    }
    Ok(())
}

/// Target top-2 logit margin after `prompt` + `prefix`.
fn margin(rt: &Runtime, prompt: &[u32], prefix: &[u32]) -> Result<f32> {
    let model = &rt.loaded.model;
    let mut state = HybridState::new(&rt.device, &model.config, prompt.len() + prefix.len() + 1)?;
    let mut all = prompt.to_vec();
    all.extend_from_slice(prefix);
    let Produced::Logits(logits) = model.forward(&rt.device, &mut state, &all, Output::Logits)?
    else {
        unreachable!()
    };
    let mut sorted = logits;
    sorted.sort_by(|a, b| b.total_cmp(a));
    Ok(sorted[0] - sorted[1])
}
