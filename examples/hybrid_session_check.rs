//! Prefix reuse must not change results: greedy generations from a session
//! that extends or rewinds (restoring a recurrent snapshot, or for
//! attention-only models resuming at the divergence point) a cached
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
        // The cached session computed p1's body as a chunk and its final
        // token alone; attention-only models (no snapshots needed) resume
        // past that boundary, so the reference splits there too.
        let splits: &[&[u32]] = if name == "rewind" {
            &[&p1[..p1.len() - 1]]
        } else {
            &[]
        };
        ok &= compare(
            &d,
            model,
            eos,
            &mut cached,
            &mut fresh,
            name,
            prompt,
            splits,
            None,
        )?
        .0;
    }
    // 5. A pinned checkpoint at the header of the reply to `p2`'s question,
    // kept through two later turns (which take the other slot), then a
    // rewind into that reply, as a chat template does when it drops the
    // reply's reasoning. A long turn in between moves sliding-window rings
    // (LFM3, Flint) well past the checkpoint, so it resumes only from the
    // snapshot's own copy of them.
    let header = tok.encode("<|im_start|>assistant\n")?.len();
    let checkpoint = p2.len() - header;
    // From an empty cache, so the checkpoint lies past the reused prefix.
    cached.reset();
    cached.set_checkpoint(checkpoint, true);
    let pin = Some(checkpoint);
    ok &= compare(
        &d,
        model,
        eos,
        &mut cached,
        &mut fresh,
        "pin",
        &p2,
        &[],
        pin,
    )?
    .0;
    // A model with recurrent state prefilled `p2` in two chunks, split at the
    // checkpoint; the reference splits there too (chunks round differently).
    let recurrent = hybrid::engine::recurrent_bytes(&model.config) > 0;
    let split = &p2[..checkpoint];
    let splits: &[&[u32]] = if recurrent { &[split] } else { &[] };
    let filler = "Write out the multiplication table from 1 to 12 in full. ".repeat(60);
    let mut next = p2.clone();
    for (name, question) in [("long turn", filler.as_str()), ("later turn", "Thanks.")] {
        // A scripted reply rather than the decoded one: the reference then
        // prefills the same chunks (single-row decode rounds differently).
        next.extend(tok.encode("Here you are.")?);
        next.extend(tok.encode(&format!(
            "<|im_end|>\n<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n"
        ))?);
        ok &= compare(
            &d,
            model,
            eos,
            &mut cached,
            &mut fresh,
            name,
            &next,
            splits,
            None,
        )?
        .0;
    }
    let mut rewound = p2[..checkpoint].to_vec();
    rewound.extend(tok.encode("<|im_start|>assistant\nTwo more primes are")?);
    let (same, reused) = compare(
        &d,
        model,
        eos,
        &mut cached,
        &mut fresh,
        "pinned rewind",
        &rewound,
        splits,
        None,
    )?;
    // Everything before the checkpoint is reused; only the header is redone.
    if reused < checkpoint {
        println!("  expected to resume at the pinned checkpoint, position {checkpoint}");
    }
    ok &= same && reused >= checkpoint;
    println!("{}", if ok { "PASS" } else { "FAIL" });
    std::process::exit(if ok { 0 } else { 1 });
}

/// Generate from `prompt` on the cached session and on a fresh one split at
/// the same point (so both run identical kernels: single-row and chunk
/// kernels round differently): whether the greedy tokens match, and how many
/// prompt tokens the cached session reused.
#[allow(clippy::too_many_arguments)]
fn compare(
    d: &ferrum::MetalDevice,
    model: &hybrid::HybridModel,
    eos: &[u32],
    cached: &mut Session,
    fresh: &mut Session,
    name: &str,
    prompt: &[u32],
    splits: &[&[u32]],
    checkpoint: Option<usize>,
) -> Result<(bool, usize)> {
    let greedy = SamplingParams::default();
    let a = cached.generate(d, model, prompt, 32, &greedy, eos, &mut |_| Ok(true))?;
    fresh.reset();
    // Prefill where the cached session's earlier chunks ended (`splits`,
    // when they lie inside what it reused), then the reused prefix.
    for split in splits {
        if split.len() < a.reused_tokens {
            fresh.prefill(d, model, split)?;
        }
    }
    if a.reused_tokens > 0 {
        fresh.prefill(d, model, &prompt[..a.reused_tokens])?;
    }
    if let Some(at) = checkpoint {
        fresh.set_checkpoint(at, false);
    }
    let b = fresh.generate(d, model, prompt, 32, &greedy, eos, &mut |_| Ok(true))?;
    assert_eq!(b.reused_tokens, a.reused_tokens);
    let same = a.tokens == b.tokens;
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
    Ok((same, a.reused_tokens))
}
