//! Prefix reuse must not change results: greedy generations from a session
//! that extends or rewinds (restoring a recurrent snapshot) a cached
//! conversation are compared with a fresh session that computes the same
//! prefix/suffix split from scratch.
//!
//! usage: hybrid_session_check MODEL.gguf [--draft mtp|DIR]
//!
//! With --draft both sessions speculate with their own drafter, which also
//! checks that drafters follow extensions, snapshot rewinds and resets.
use ferrum::{
    Result,
    hybrid::{
        self,
        plan::PlanOptions,
        runtime::{DraftSource, SpecOptions},
        session::{SamplingParams, Session},
    },
};

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let d = ferrum::MetalDevice::new()?;
    let loaded = hybrid::load_with(
        &d,
        &args[1],
        PlanOptions {
            context: Some(4096),
            snapshots: 2,
            ..Default::default()
        },
    )?;
    let model = &loaded.model;
    let tok = &loaded.tokenizer;
    let greedy = SamplingParams::default();
    let eos = &loaded.eos_ids;
    let mut cached = Session::new(&d, model, 4096, 2)?;
    let mut fresh = Session::new(&d, model, 4096, 0)?;
    if let Some(i) = args.iter().position(|a| a == "--draft") {
        let spec = SpecOptions::new(DraftSource::parse(&args[i + 1]));
        let path = std::path::Path::new(&args[1]);
        for session in [&mut cached, &mut fresh] {
            let drafter = spec.build(&d, path, &loaded, 4096)?;
            println!("drafter {}", drafter.name());
            session.set_drafter(&d, model, drafter)?;
        }
    }
    let base = "<|im_start|>user\nList three prime numbers and explain why each is prime.<|im_end|>\n<|im_start|>assistant\n";
    let p1 = tok.encode(base)?;
    let first = cached.generate(&d, model, &p1, 40, &greedy, eos, &mut |_| Ok(true))?;
    // 1. Extension: previous prompt + generation + a new turn.
    let mut p2 = p1.clone();
    p2.extend(&first.tokens);
    p2.extend(tok.encode(
        "<|im_end|>\n<|im_start|>user\nNow two more.<|im_end|>\n<|im_start|>assistant\n",
    )?);
    // 2. Retry: the identical prompt again (resumes from the `len - 1` snapshot).
    let retry = p1.clone();
    // 3. Divergence inside the cache: same first turn, different question.
    let p3 = tok.encode("<|im_start|>user\nList three prime numbers and explain why each is prime.<|im_end|>\n<|im_start|>assistant\nSure")?;
    // 4. Divergence before any snapshot: entirely different prompt.
    let p4 = tok.encode(
        "<|im_start|>user\nWhat is the boiling point of water?<|im_end|>\n<|im_start|>assistant\n",
    )?;
    let mut ok = true;
    for (name, prompt) in [
        ("extend", &p2),
        ("rewind", &p3),
        ("retry", &retry),
        ("replace", &p4),
        ("extend again", &p2),
    ] {
        let a = cached.generate(&d, model, prompt, 32, &greedy, eos, &mut |_| Ok(true))?;
        // Reference: a fresh session split at the same point, so both sides run
        // identical kernels (single-row and chunk kernels round differently).
        fresh.reset();
        if a.reused_tokens > 0 {
            fresh.prefill(&d, model, &prompt[..a.reused_tokens])?;
        }
        let b = fresh.generate(&d, model, prompt, 32, &greedy, eos, &mut |_| Ok(true))?;
        assert_eq!(b.reused_tokens, a.reused_tokens);
        let same = a.tokens == b.tokens;
        ok &= same;
        println!(
            "{name:<13} prompt {:>4} tokens, reused {:>4}: {} ({} tokens, {} verify steps, acceptance {:.2})",
            prompt.len(),
            a.reused_tokens,
            if same { "identical" } else { "DIFFERENT" },
            a.tokens.len(),
            cached.spec_stats.steps,
            cached.spec_stats.acceptance_length()
        );
        if !same {
            println!("  cached: {:?}\n  fresh:  {:?}", a.tokens, b.tokens);
        }
    }
    println!("{}", if ok { "PASS" } else { "FAIL" });
    std::process::exit(if ok { 0 } else { 1 });
}
