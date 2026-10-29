//! Prefix reuse must not change results: greedy generations from a session
//! that extends or rewinds (restoring a recurrent snapshot, or for
//! attention-only models resuming at the divergence point) a cached
//! conversation are compared with a fresh session that computes the same
//! prefix from scratch, replaying the cached session's forwards: the same
//! prompt chunks and the same single-row steps for its own replies (chunk
//! and single-row kernels round differently, which a near-tie turns into a
//! different token).
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
        session::{Completion, SamplingParams, Session},
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
    let recurrent = hybrid::engine::recurrent_bytes(&model.config) > 0;
    let mut forwards = Forwards::default();
    let first = cached.generate(&d, model, &p1, 40, &greedy, eos, &mut |_| Ok(true))?;
    forwards.record(model, &first, p1.len(), None, cached.len());
    let mut c = Compare {
        d: &d,
        model,
        eos,
        cached: &mut cached,
        fresh: &mut fresh,
        forwards,
    };
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
        ok &= c.run(name, prompt, None)?.0;
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
    c.cached.reset();
    c.forwards.0.clear();
    c.cached.set_checkpoint(checkpoint, true);
    // Only a model with recurrent state splits its prefill there.
    ok &= c.run("pin", &p2, recurrent.then_some(checkpoint))?.0;
    let filler = "Write out the multiplication table from 1 to 12 in full. ".repeat(60);
    let mut next = p2.clone();
    for (name, question) in [("long turn", filler.as_str()), ("later turn", "Thanks.")] {
        next.extend(tok.encode("Here you are.")?);
        next.extend(tok.encode(&format!(
            "<|im_end|>\n<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n"
        ))?);
        ok &= c.run(name, &next, None)?.0;
    }
    let mut rewound = p2[..checkpoint].to_vec();
    rewound.extend(tok.encode("<|im_start|>assistant\nTwo more primes are")?);
    let (same, reused) = c.run("pinned rewind", &rewound, None)?;
    // Everything before the checkpoint is reused; only the header is redone.
    if reused < checkpoint {
        println!("  expected to resume at the pinned checkpoint, position {checkpoint}");
    }
    ok &= same && reused >= checkpoint;
    println!("{}", if ok { "PASS" } else { "FAIL" });
    std::process::exit(if ok { 0 } else { 1 });
}

/// The forwards that computed the cached session's rows, in order: each a
/// range of positions run as one chunk (a single row for a decode step).
/// Without a drafter this is exact; a drafter's verify runs several rows
/// at once, which the replay approximates with single rows.
#[derive(Default)]
struct Forwards(Vec<(usize, usize)>);

impl Forwards {
    /// A generation that reused `done.reused_tokens` of a `prompt`-token
    /// prompt and left the session `len` tokens long: the rows past the
    /// reused prefix went in as prompt chunks (split at `checkpoint`), then
    /// the final prompt token and each fed-back reply token alone.
    fn record(
        &mut self,
        model: &hybrid::HybridModel,
        done: &Completion,
        prompt: usize,
        checkpoint: Option<usize>,
        len: usize,
    ) {
        let reused = done.reused_tokens;
        self.0.retain(|r| r.0 < reused);
        if let Some(last) = self.0.last_mut() {
            last.1 = last.1.min(reused);
        }
        let body = prompt - 1;
        let mut bounds = vec![reused];
        bounds.extend(checkpoint.filter(|&at| at > reused && at < body));
        bounds.push(body);
        for pair in bounds.windows(2) {
            let mut at = pair[0];
            while at < pair[1] {
                let end = (at + model.chunk()).min(pair[1]);
                self.0.push((at, end));
                at = end;
            }
        }
        self.0.extend((body..len).map(|i| (i, i + 1)));
    }
}

struct Compare<'a> {
    d: &'a ferrum::MetalDevice,
    model: &'a hybrid::HybridModel,
    eos: &'a [u32],
    cached: &'a mut Session,
    fresh: &'a mut Session,
    forwards: Forwards,
}

impl Compare<'_> {
    /// Generate from `prompt` on the cached session, then on a fresh one that
    /// first replays the forwards behind what the cached one reused, and
    /// splits its prefill at `checkpoint` as the cached one did: whether the
    /// greedy tokens match, and how many prompt tokens were reused.
    fn run(
        &mut self,
        name: &str,
        prompt: &[u32],
        checkpoint: Option<usize>,
    ) -> Result<(bool, usize)> {
        let (d, model, eos) = (self.d, self.model, self.eos);
        let greedy = SamplingParams::default();
        let history = self.cached.tokens().to_vec();
        let a = self
            .cached
            .generate(d, model, prompt, 32, &greedy, eos, &mut |_| Ok(true))?;
        self.fresh.reset();
        for &(_, end) in self.forwards.0.iter().filter(|r| r.0 < a.reused_tokens) {
            let end = end.min(a.reused_tokens);
            self.fresh.prefill(d, model, &history[..end])?;
        }
        if let Some(at) = checkpoint {
            self.fresh.set_checkpoint(at, false);
        }
        let b = self
            .fresh
            .generate(d, model, prompt, 32, &greedy, eos, &mut |_| Ok(true))?;
        assert_eq!(b.reused_tokens, a.reused_tokens);
        self.forwards
            .record(model, &a, prompt.len(), checkpoint, self.cached.len());
        let same = a.tokens == b.tokens;
        println!(
            "{name:<13} prompt {:>4} tokens, reused {:>4}: {} ({} tokens, {} verify steps, acceptance {:.2})",
            prompt.len(),
            a.reused_tokens,
            if same { "identical" } else { "DIFFERENT" },
            a.tokens.len(),
            self.cached.spec_stats.steps,
            self.cached.spec_stats.acceptance_length()
        );
        if !same {
            println!("  cached: {:?}\n  fresh:  {:?}", a.tokens, b.tokens);
        }
        Ok((same, a.reused_tokens))
    }
}
