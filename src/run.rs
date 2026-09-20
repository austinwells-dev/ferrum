use ferrum::{
    Error, MetalDevice, Result,
    generation::{self, Generation},
    loader::Weights,
    model::qwen::{self, QwenConfig},
    sampling::{Sampler, SamplingConfig},
    tokenizer::qwen::{DEFAULT_SYSTEM, QwenTokenizer},
};
use std::{
    io::{self, Write},
    path::PathBuf,
    time::Instant,
};
struct Options {
    model: PathBuf,
    prompt: String,
    system: String,
    raw: bool,
    diagnostic: bool,
    max_new: usize,
    sampling: SamplingConfig,
    warmup: bool,
    profile: bool,
}
impl Options {
    fn parse() -> Result<Self> {
        let mut o = Self {
            model: PathBuf::new(),
            prompt: String::new(),
            system: DEFAULT_SYSTEM.into(),
            raw: false,
            diagnostic: false,
            max_new: 32,
            sampling: Default::default(),
            warmup: false,
            profile: false,
        };
        let mut supplied_prompt = false;
        let mut args = std::env::args().skip(2);
        while let Some(key) = args.next() {
            match key.as_str() {
                "--raw" => o.raw = true,
                "--tokenizer-diagnostic" => o.diagnostic = true,
                "--warmup" => o.warmup = true,
                "--profile" => o.profile = true,
                "--model" | "--prompt" | "--system" | "--max-new-tokens" | "--temperature"
                | "--top-k" | "--top-p" | "--seed" => {
                    let v = args
                        .next()
                        .ok_or_else(|| Error::Parameter(format!("missing value for {key}")))?;
                    let bad = || Error::Parameter(format!("invalid value for {key}: {v}"));
                    match key.as_str() {
                        "--model" => o.model = v.into(),
                        "--prompt" => {
                            o.prompt = v;
                            supplied_prompt = true;
                        }
                        "--system" => o.system = v,
                        "--max-new-tokens" => o.max_new = v.parse().map_err(|_| bad())?,
                        "--temperature" => o.sampling.temperature = v.parse().map_err(|_| bad())?,
                        "--top-k" => o.sampling.top_k = v.parse().map_err(|_| bad())?,
                        "--top-p" => o.sampling.top_p = v.parse().map_err(|_| bad())?,
                        "--seed" => o.sampling.seed = v.parse().map_err(|_| bad())?,
                        _ => {}
                    }
                }
                _ => return Err(Error::Parameter(format!("unknown option {key}"))),
            }
        }
        if o.model.as_os_str().is_empty() || !supplied_prompt {
            return Err(Error::Parameter("usage: run --model DIR --prompt TEXT [--raw] [--max-new-tokens 32] [--temperature 0] [--top-k 0] [--top-p 1] [--seed 0] [--warmup] [--profile] [--tokenizer-diagnostic]".into()));
        }
        o.sampling.validate()?;
        if !o.model.is_dir() {
            return Err(Error::Config(format!(
                "missing model directory {}",
                o.model.display()
            )));
        }
        Ok(o)
    }
}
fn ms(d: std::time::Duration) -> f64 {
    d.as_secs_f64() * 1000.
}
fn report(r: &Generation, prompt: usize) {
    eprintln!(
        "\n---\nprompt: {prompt} tokens; generated: {}; stop: {:?}",
        r.tokens.len(),
        r.stop
    );
    eprintln!(
        "prefill: {:.3} ms ({:.3} tok/s); first token: {:.3} ms",
        ms(r.prefill),
        if r.prefill.is_zero() {
            0.
        } else {
            prompt as f64 / r.prefill.as_secs_f64()
        },
        ms(r.first_token)
    );
    eprintln!(
        "prefill + final row: {} dispatches; {} allocations; {} allocation bytes",
        r.prefill_counters.dispatches,
        r.prefill_counters.allocations,
        r.prefill_counters.allocated_bytes
    );
    eprintln!("prefill runtime counters: {:?}", r.prefill_counters);
    if !r.decode.is_empty() {
        let mut sorted = r.decode.clone();
        sorted.sort();
        eprintln!(
            "decode: median {:.3} ms ({:.3} tok/s); per-token ms: {:?}",
            ms(sorted[sorted.len() / 2]),
            1. / sorted[sorted.len() / 2].as_secs_f64(),
            r.decode.iter().map(|&d| ms(d)).collect::<Vec<_>>()
        );
        eprintln!("decode counters per step: {:?}", r.decode_counters);
    }
    eprintln!(
        "sampling: {:.3} ms total; active KV: {} bytes; generated IDs: {:?}",
        ms(r.sampling),
        r.kv_bytes,
        r.tokens
    );
}
pub fn run(d: &MetalDevice) -> Result<()> {
    let o = Options::parse()?;
    if let Ok(limit) = std::env::var("FERRUM_BATCH_LIMIT") {
        d.set_batch_limit(
            limit
                .parse()
                .map_err(|_| Error::Parameter("invalid FERRUM_BATCH_LIMIT".into()))?,
        )?;
    }
    let start = Instant::now();
    let qc = QwenConfig::from_file(o.model.join("config.json"))?;
    let c = qc.convert()?;
    let tok = QwenTokenizer::load(&o.model, &qc)?;
    let tokenizer_load = start.elapsed();
    let start = Instant::now();
    let (text, ids) = tok.encode_prompt(&o.prompt, &o.system, o.raw)?;
    let tokenization = start.elapsed();
    generation::validate_context(ids.len(), o.max_new, c.max_context_length)?;
    if o.diagnostic {
        eprintln!(
            "input: {:?}\nformatted: {:?}\ntoken count: {}\nIDs: {:?}\nroundtrip: {:?}",
            o.prompt,
            text,
            ids.len(),
            ids,
            tok.tokenizer.decode(&ids)?
        );
    }
    let start = Instant::now();
    let source = Weights::from_file(d, o.model.join("model.safetensors"))?;
    let load = start.elapsed();
    let source_bytes = source.bytes();
    let count = source.names().count();
    let before = d.counters();
    let start = Instant::now();
    let model = qwen::construct(d, c.clone(), &source)?;
    let construction = start.elapsed();
    let construction_counters = generation::counter_delta(before, d.counters());
    drop(source);
    eprintln!(
        "Ferrum\nModel: Qwen2.5-0.5B-Instruct (local checkpoint)\nDevice: {}\nBackend: Metal\nDtype: {:?}\nLayers: {}; hidden: {}; intermediate: {}; Q heads: {}; KV heads: {}; head dim: {}\nParameters: {}; tensors: {}\nSource weight bytes: {}\nRetained weight bytes: {}\nConstruction/transposition bytes: {}\nRecommended working set: {} bytes",
        d.name(),
        c.dtype,
        c.num_layers,
        c.hidden_size,
        c.intermediate_size,
        c.num_attention_heads,
        c.num_key_value_heads,
        c.head_dim,
        source_bytes / 2,
        count,
        source_bytes,
        model.weight_bytes(),
        construction_counters.allocated_bytes,
        d.recommended_max_working_set()
    );
    eprintln!(
        "config/tokenizer load: {:.3} ms; weight load: {:.3} ms; construction: {:.3} ms; tokenization: {:.3} ms\nPrompt: {} tokens; sampler: {:?}",
        ms(tokenizer_load),
        ms(load),
        ms(construction),
        ms(tokenization),
        ids.len(),
        o.sampling
    );
    if o.warmup && o.max_new > 0 {
        let _ = generation::generate(
            d,
            &model,
            &ids,
            o.max_new.min(2),
            &tok.eos_ids,
            generation::argmax,
            |_| Ok(()),
        )?;
    }
    d.set_profiling(o.profile);
    let mut sampler = Sampler::new(o.sampling)?;
    let mut stream = tok.tokenizer.decode_stream();
    let mut emitted = String::new();
    let mut stdout = io::stdout().lock();
    let ioerr = |e: io::Error| Error::Parameter(format!("stdout: {e}"));
    let r = generation::generate(
        d,
        &model,
        &ids,
        o.max_new,
        &tok.eos_ids,
        |logits| sampler.sample(logits),
        |id| {
            if !tok.tokenizer.is_defined(id) {
                return Err(Error::Tokenizer(format!(
                    "generated undefined padded vocabulary ID {id}"
                )));
            }
            if let Some(fragment) = stream
                .step(id)
                .map_err(|e| Error::Tokenizer(e.to_string()))?
            {
                emitted.push_str(&fragment);
                stdout.write_all(fragment.as_bytes()).map_err(ioerr)?;
                stdout.flush().map_err(ioerr)?;
            }
            Ok(())
        },
    )?;
    // Flush any final incomplete byte sequence using the tokenizer's documented
    // replacement policy; completed Unicode was already streamed intact.
    let visible: Vec<_> = r
        .tokens
        .iter()
        .copied()
        .filter(|id| !tok.eos_ids.contains(id))
        .collect();
    let complete = tok.tokenizer.decode(&visible)?;
    let tail = complete
        .strip_prefix(&emitted)
        .ok_or_else(|| Error::Tokenizer("stream/full decode prefix mismatch".into()))?;
    stdout.write_all(tail.as_bytes()).map_err(ioerr)?;
    writeln!(stdout).map_err(ioerr)?;
    report(&r, ids.len());
    if o.profile {
        eprintln!(
            "PROFILE prefill: {}",
            serde_json::to_string(&r.prefill_profile)
                .map_err(|e| Error::Validation(e.to_string()))?
        );
        for (i, p) in r.decode_profiles.iter().enumerate() {
            eprintln!(
                "PROFILE decode {i}: {}",
                serde_json::to_string(p).map_err(|e| Error::Validation(e.to_string()))?
            );
        }
        d.set_profiling(false);
    }
    Ok(())
}
