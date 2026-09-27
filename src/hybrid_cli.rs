//! `ferrum hybrid`: load a Qwen3.5-family GGUF and generate.
use ferrum::{
    Error, MetalDevice, Result,
    hybrid::{self, HybridState, Output, Produced},
};
use std::{
    io::Write,
    time::{Duration, Instant},
};

struct Options {
    model: String,
    prompt: String,
    raw: bool,
    max_new: usize,
    context: Option<usize>,
    chunk: usize,
    repeat: usize,
}

fn parse() -> Result<Options> {
    let mut o = Options {
        model: String::new(),
        prompt: String::new(),
        raw: false,
        max_new: 32,
        context: None,
        chunk: 512,
        repeat: 1,
    };
    let mut args = std::env::args().skip(2);
    while let Some(key) = args.next() {
        if key == "--raw" {
            o.raw = true;
            continue;
        }
        let value = args
            .next()
            .ok_or_else(|| Error::Parameter(format!("missing value for {key}")))?;
        let number = || {
            value
                .parse::<usize>()
                .map_err(|_| Error::Parameter(format!("invalid value for {key}: {value}")))
        };
        match key.as_str() {
            "--model" => o.model = value.clone(),
            "--prompt" => o.prompt = value.clone(),
            "--prompt-file" => {
                o.prompt = std::fs::read_to_string(&value)
                    .map_err(|e| Error::Parameter(format!("{value}: {e}")))?
            }
            "--max-new-tokens" => o.max_new = number()?,
            "--context" => {
                o.context = if value == "auto" {
                    None
                } else {
                    Some(number()?)
                }
            }
            "--chunk" => o.chunk = number()?,
            "--repeat" => o.repeat = number()?,
            _ => return Err(Error::Parameter(format!("unknown option {key}"))),
        }
    }
    if o.model.is_empty() || o.prompt.is_empty() {
        return Err(Error::Parameter(
            "usage: hybrid --model FILE.gguf --prompt TEXT [--raw] [--max-new-tokens N] [--context N] [--chunk N] [--repeat N]".into(),
        ));
    }
    Ok(o)
}

fn ms(d: Duration) -> f64 {
    d.as_secs_f64() * 1e3
}

fn print_profile(label: &str, profile: ferrum::metal::Profile) {
    let mut entries: Vec<_> = profile.into_iter().collect();
    entries.sort_by_key(|e| std::cmp::Reverse(e.1.gpu));
    let total: Duration = entries.iter().map(|(_, e)| e.gpu).sum();
    eprintln!("{label}: GPU {:.2} ms total", ms(total));
    for (name, e) in entries.iter().take(24) {
        eprintln!(
            "  {name:<22} {:>6} calls {:>9.3} ms  {:>5.1}%",
            e.calls,
            ms(e.gpu),
            100. * e.gpu.as_secs_f64() / total.as_secs_f64().max(1e-12)
        );
    }
}

pub fn run(d: &MetalDevice) -> Result<()> {
    let o = parse()?;
    let profile = std::env::var("FERRUM_HYBRID_PROFILE").is_ok_and(|v| v != "0");
    if profile {
        // One command buffer per kernel so each dispatch has its own GPU time.
        d.set_batch_limit(1)?;
    }
    let loaded = hybrid::load_with(
        d,
        &o.model,
        hybrid::plan::PlanOptions {
            context: o.context,
            chunk: o.chunk,
            ..Default::default()
        },
    )?;
    let plan = &loaded.plan;
    let mib = |b: usize| b as f64 / (1u64 << 20) as f64;
    eprintln!(
        "memory plan: budget {:.0} MiB = weights {:.0} + scratch {:.0} + recurrent {:.0} + snapshots {:.0} + overhead {:.0} + KV {:.0} ({:.1} KiB/token x {}); max context {} (trained {})",
        mib(plan.budget),
        mib(plan.weights),
        mib(plan.scratch),
        mib(plan.recurrent),
        mib(plan.snapshots),
        mib(plan.overhead),
        mib(plan.kv(plan.context)),
        plan.kv_per_token / 1024.,
        plan.context,
        plan.max_context,
        plan.trained_context,
    );
    let model = &loaded.model;
    let c = &model.config;
    eprintln!(
        "{} ({:?}): {} layers ({} attention), hidden {}, vocab {}; weights {:.2} GiB; scratch {:.1} MiB; load {:.2} s",
        c.name,
        c.variant,
        c.layers,
        c.attention_layer_count(),
        c.hidden,
        c.vocab,
        model.weight_bytes() as f64 / (1u64 << 30) as f64,
        model.scratch_bytes() as f64 / (1u64 << 20) as f64,
        loaded.load_time.as_secs_f64()
    );
    let text = if o.raw {
        o.prompt.clone()
    } else {
        format!(
            "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n",
            o.prompt
        )
    };
    let prompt = loaded.tokenizer.encode(&text)?;
    let before_state = d.allocated_bytes();
    let mut state = HybridState::new(d, c, plan.context)?;
    let state_measured = d.allocated_bytes() - before_state;
    eprintln!(
        "measured: weights+scratch {:.0} MiB (predicted {:.0}), state {:.0} MiB (predicted {:.0})",
        mib(loaded.loaded_bytes),
        mib(plan.weights + plan.scratch),
        mib(state_measured),
        mib(plan.recurrent + plan.kv(plan.context)),
    );
    eprintln!(
        "prompt {} tokens; state {:.1} MiB for {} positions",
        prompt.len(),
        state.byte_size() as f64 / (1u64 << 20) as f64,
        state.capacity()
    );
    for round in 0..o.repeat {
        state.reset();
        d.set_profiling(profile && round + 1 == o.repeat);
        d.take_profile();
        let start = Instant::now();
        let Produced::Token(mut token) = model.forward(d, &mut state, &prompt, Output::Argmax)?
        else {
            unreachable!("argmax output yields a token")
        };
        let prefill = start.elapsed();
        if profile && round + 1 == o.repeat {
            print_profile("prefill", d.take_profile());
        }
        let mut generated = vec![token];
        let mut steps = Vec::new();
        let mut stream = loaded.tokenizer.decode_stream();
        let mut out = std::io::stdout();
        if round == 0
            && let Some(piece) = stream
                .step(token)
                .map_err(|e| Error::Tokenizer(e.to_string()))?
        {
            print!("{piece}");
        }
        while generated.len() < o.max_new && !loaded.eos_ids.contains(&token) {
            let t = Instant::now();
            let Produced::Token(next) = model.forward(d, &mut state, &[token], Output::Argmax)?
            else {
                unreachable!()
            };
            steps.push(t.elapsed());
            token = next;
            generated.push(token);
            if round == 0
                && !loaded.eos_ids.contains(&token)
                && let Some(piece) = stream
                    .step(token)
                    .map_err(|e| Error::Tokenizer(e.to_string()))?
            {
                print!("{piece}");
                out.flush().ok();
            }
        }
        println!();
        if profile && round + 1 == o.repeat && !steps.is_empty() {
            print_profile(&format!("decode ({} steps)", steps.len()), d.take_profile());
        }
        let decode: Duration = steps.iter().sum();
        let mut sorted = steps.clone();
        sorted.sort();
        eprintln!(
            "round {round}: prefill {} tok in {:.1} ms ({:.1} tok/s); decode {} tok in {:.1} ms ({:.2} tok/s, median {:.1} ms)",
            prompt.len(),
            ms(prefill),
            prompt.len() as f64 / prefill.as_secs_f64(),
            steps.len(),
            ms(decode),
            steps.len() as f64 / decode.as_secs_f64().max(1e-9),
            sorted.get(sorted.len() / 2).map_or(0., |&d| ms(d)),
        );
        if round == 0 {
            eprintln!("generated IDs: {generated:?}");
        }
    }
    Ok(())
}
