//! Speculative decoding benchmark and losslessness check on chat prompts.
//! usage: hybrid_spec MODEL.gguf --drafter mtp|DRAFT_DIR [--drafts K]
//!        [--tokens 256] [--context 8192] [--temp 0] [--baseline] [--prompts N]
//!        [--only I,J]  (these prompt indices instead of the first N)
//!        [--tree N]  (draft trees of N nodes; greedy only)
//!        [--sweep 0,4,8,15]  (each tree budget in turn, 0 = chains, on one load)
//!        [--tree-search best-first|siblings]  (how MTP grows trees)
//!        [--pmin P]  (DSpark confidence cut-off)
//!        [--depth N --filler FILE]  (generate after N tokens of FILE in the prompt)
//!
//! Each prompt is rendered with the model's chat template (thinking on) and
//! generated with speculation. With --baseline the same prompt is first
//! generated without speculation; greedy outputs are compared token by token
//! and any divergence is reported with the target's top-2 logit margin at
//! that point (a near-tie means single-row and multi-row kernels rounded
//! differently, not a verification error). In a sweep, tree outputs are also
//! compared with the chain's.
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
    let depth: usize = get("--depth", "0").parse().expect("depth");
    let drafter_arg = get("--drafter", "mtp");
    let drafts: usize = get("--drafts", "0").parse().expect("drafts");
    let list = |key: &str| -> Option<Vec<usize>> {
        has(key).then(|| {
            get(key, "")
                .split(',')
                .map(|v| v.parse().expect("list of numbers"))
                .collect()
        })
    };
    // Each configuration is a tree budget, or None for chains.
    let configs: Vec<Option<usize>> = match list("--sweep") {
        Some(budgets) => budgets.iter().map(|&b| (b > 0).then_some(b)).collect(),
        None => vec![list("--tree").map(|b| b[0])],
    };
    let widest = configs.iter().flatten().max().copied();
    let spec = SpecOptions {
        source: DraftSource::parse(&drafter_arg),
        max_drafts: (drafts > 0).then_some(drafts),
        tree: widest,
        draft: DraftOptions {
            confidence_min: get("--pmin", "0").parse().expect("pmin"),
            vocab: get("--vocab", "0").parse().ok().filter(|&v: &usize| v > 0),
            quant: if get("--quant", "q4_0") == "q4_0" {
                ferrum::hybrid::draft::DraftQuant::Q4_0
            } else {
                ferrum::hybrid::draft::DraftQuant::Q8_0
            },
            tree_search: if get("--tree-search", "best-first") == "siblings" {
                ferrum::hybrid::draft::TreeSearch::Siblings
            } else {
                ferrum::hybrid::draft::TreeSearch::BestFirst
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
    // Long-context runs: the first `depth` tokens of the filler file precede
    // every prompt, so drafting and verification run against a deep cache.
    let filler = if depth > 0 {
        let text = std::fs::read_to_string(get("--filler", "")).expect("--filler FILE");
        let ids = rt.loaded.tokenizer.encode(&text)?;
        assert!(ids.len() >= depth, "filler has only {} tokens", ids.len());
        rt.loaded.tokenizer.decode(&ids[..depth])?
    } else {
        String::new()
    };
    let label = |config: Option<usize>| config.map_or("chain".to_owned(), |b| format!("tree {b}"));
    let selected: Vec<usize> =
        list("--only").unwrap_or_else(|| (0..prompts.min(PROMPTS.len())).collect());
    let (mut base_tokens, mut base_secs) = (0usize, 0f64);
    // Per configuration: tokens, seconds, statistics, greedy mismatches.
    let mut totals = vec![(0usize, 0f64, SpecStats::default(), 0usize); configs.len()];
    for &index in &selected {
        let prompt = PROMPTS[index];
        let content = if depth > 0 {
            format!("Reference material:\n\n{filler}\n\n---\n\n{prompt}")
        } else {
            prompt.to_string()
        };
        let request = ChatRequest::new(
            vec![json!({"role": "user", "content": content})],
            sampling.clone(),
        );
        let text = rt.render(&request)?;
        let tokens = rt.loaded.tokenizer.encode(&text)?;
        let eos = rt.loaded.eos_ids.clone();
        let run = |rt: &mut Runtime, speculate: bool| -> Result<(Vec<u32>, f64)> {
            rt.session.reset();
            rt.session.speculate = speculate;
            let completion = rt.session.generate(
                &rt.device,
                &rt.loaded.model,
                &tokens,
                max_tokens,
                &sampling,
                &eos,
                &mut |_| Ok(true),
            )?;
            Ok((completion.tokens, completion.decode.as_secs_f64()))
        };
        let base = if has("--baseline") {
            let (t, secs) = run(&mut rt, false)?;
            base_tokens += t.len();
            base_secs += secs;
            println!(
                "prompt {index}: baseline {} tokens, {:.1} tok/s",
                t.len(),
                t.len() as f64 / secs
            );
            Some(t)
        } else {
            None
        };
        let mut chain_output: Option<Vec<u32>> = None;
        for (c, &config) in configs.iter().enumerate() {
            if configs.len() > 1 || config.is_some() {
                rt.session
                    .set_tree_budget(&rt.device, &rt.loaded.model, config)?;
            }
            let (spec, secs) = run(&mut rt, true)?;
            let stats = rt.session.spec_stats.clone();
            let mut line = format!(
                "prompt {index} {}: {} tokens, {:.1} tok/s, acceptance {:.2} ({} steps)",
                label(config),
                spec.len(),
                spec.len() as f64 / secs,
                stats.acceptance_length(),
                stats.steps
            );
            if temperature == 0. {
                if let Some(base) = &base {
                    match base.iter().zip(&spec).position(|(a, b)| a != b) {
                        None if base.len() == spec.len() => line += ", identical",
                        None => {
                            line +=
                                &format!(", prefix-identical ({} vs {})", base.len(), spec.len())
                        }
                        Some(i) => {
                            totals[c].3 += 1;
                            let margin = margin(&rt, &tokens, &base[..i])?;
                            line += &format!(", diverges at {i} (target top-2 margin {margin:.4})");
                        }
                    }
                }
                match (&chain_output, config) {
                    (None, None) => chain_output = Some(spec.clone()),
                    (Some(chain), Some(_)) => {
                        match chain.iter().zip(&spec).position(|(a, b)| a != b) {
                            None if chain.len() == spec.len() => line += ", = chain",
                            None => line += ", chain prefix",
                            Some(i) => {
                                let margin = margin(&rt, &tokens, &chain[..i])?;
                                line += &format!(
                                    ", leaves the chain at {i} (target top-2 margin {margin:.4})"
                                );
                            }
                        }
                    }
                    _ => {}
                }
            }
            println!("{line}");
            let t = &mut totals[c];
            t.0 += spec.len();
            t.1 += secs;
            t.2.steps += stats.steps;
            t.2.drafted += stats.drafted;
            t.2.accepted += stats.accepted;
            if t.2.accepted_at.len() < stats.accepted_at.len() {
                t.2.accepted_at.resize(stats.accepted_at.len(), 0);
            }
            for (a, b) in t.2.accepted_at.iter_mut().zip(&stats.accepted_at) {
                *a += b;
            }
            t.2.draft_seconds += stats.draft_seconds;
            t.2.verify_seconds += stats.verify_seconds;
        }
    }
    println!();
    if base_tokens > 0 {
        println!("baseline: {:.2} tok/s", base_tokens as f64 / base_secs);
    }
    for (c, &config) in configs.iter().enumerate() {
        let (tokens, secs, total, mismatches) = &totals[c];
        let rate = *tokens as f64 / secs;
        let rates: Vec<String> = total
            .accepted_at
            .iter()
            .map(|&a| format!("{:.0}%", 100. * a as f64 / total.steps.max(1) as f64))
            .collect();
        let mut line = format!(
            "{}: {rate:.2} tok/s, acceptance length {:.3}, {:.1} drafts/step, draft {:.1} ms/step, verify {:.1} ms/step, accepted by depth {}",
            label(config),
            total.acceptance_length(),
            total.drafted as f64 / total.steps.max(1) as f64,
            total.draft_seconds * 1e3 / total.steps.max(1) as f64,
            total.verify_seconds * 1e3 / total.steps.max(1) as f64,
            rates.join(" ")
        );
        if base_tokens > 0 {
            line += &format!(
                ", speedup {:.2}x, greedy mismatches {mismatches}",
                rate / (base_tokens as f64 / base_secs)
            );
        }
        println!("{line}");
    }
    Ok(())
}

/// Target top-2 logit margin after `prompt` + `prefix`.
fn margin(rt: &Runtime, prompt: &[u32], prefix: &[u32]) -> Result<f32> {
    let model = &rt.loaded.model;
    let mut state = HybridState::new(
        &rt.device,
        &model.config,
        prompt.len() + prefix.len() + 1,
        model.max_forward_rows(),
    )?;
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
