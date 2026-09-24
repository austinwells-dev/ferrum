use ferrum::{
    Error, MetalDevice, Result,
    generation::{self, Generation},
    loader::Weights,
    model::{
        granite, granite_moe, lfm2, lfm2_moe, olmo2,
        qwen::{self, QwenConfig},
        qwen_gguf, qwen_mlx, qwen3,
    },
    sampling::{Sampler, SamplingConfig},
    tokenizer::{
        Tokenizer,
        qwen::{DEFAULT_SYSTEM, QwenTokenizer, chat_prompt},
    },
};
enum PromptFormat {
    Qwen,
    Granite,
    Lfm2,
    Plain,
}
struct RuntimeTokenizer {
    tokenizer: Tokenizer,
    eos_ids: Vec<u32>,
    format: PromptFormat,
}
impl From<QwenTokenizer> for RuntimeTokenizer {
    fn from(value: QwenTokenizer) -> Self {
        Self {
            tokenizer: value.tokenizer,
            eos_ids: value.eos_ids,
            format: PromptFormat::Qwen,
        }
    }
}
impl RuntimeTokenizer {
    fn encode_prompt(&self, prompt: &str, system: &str, raw: bool) -> Result<(String, Vec<u32>)> {
        let text = if raw {
            prompt.to_owned()
        } else {
            match self.format {
                PromptFormat::Qwen => chat_prompt(system, prompt),
                PromptFormat::Granite => format!(
                    "<|start_of_role|>system<|end_of_role|>{system}<|end_of_text|>\n<|start_of_role|>user<|end_of_role|>{prompt}<|end_of_text|>\n<|start_of_role|>assistant<|end_of_role|>"
                ),
                PromptFormat::Lfm2 => {
                    let system_message = if system == DEFAULT_SYSTEM {
                        String::new()
                    } else {
                        format!("<|im_start|>system\n{system}<|im_end|>\n")
                    };
                    format!(
                        "<|startoftext|>{system_message}<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
                    )
                }
                PromptFormat::Plain => {
                    if system != DEFAULT_SYSTEM {
                        return Err(Error::Tokenizer(
                            "this model has no chat template; use --raw for custom prompts".into(),
                        ));
                    }
                    prompt.to_owned()
                }
            }
        };
        let ids = self.tokenizer.encode(&text)?;
        Ok((text, ids))
    }
}
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
            profile: std::env::args().nth(1).as_deref() == Some("profile"),
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
            return Err(Error::Parameter("usage: run --model DIR_OR_GGUF --prompt TEXT [--raw] [--max-new-tokens 32] [--temperature 0] [--top-k 0] [--top-p 1] [--seed 0] [--warmup] [--profile] [--tokenizer-diagnostic]".into()));
        }
        o.sampling.validate()?;
        if !o.model.is_dir() && !o.model.is_file() {
            return Err(Error::Config(format!(
                "missing model path {}",
                o.model.display()
            )));
        }
        Ok(o)
    }
}
fn ms(d: std::time::Duration) -> f64 {
    d.as_secs_f64() * 1000.
}
fn report(r: &Generation, prompt: usize, profile: bool) {
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
    if profile {
        eprintln!("prefill runtime counters: {:?}", r.prefill_counters);
    }
    if !r.decode.is_empty() {
        let mut sorted = r.decode.clone();
        sorted.sort();
        eprintln!(
            "decode: median {:.3} ms ({:.3} tok/s); per-token ms: {:?}",
            ms(sorted[sorted.len() / 2]),
            1. / sorted[sorted.len() / 2].as_secs_f64(),
            r.decode.iter().map(|&d| ms(d)).collect::<Vec<_>>()
        );
        if profile {
            eprintln!("decode counters per step: {:?}", r.decode_counters);
        }
    }
    eprintln!(
        "sampling: {:.3} ms total; active KV: {} bytes; convolution/state: {} bytes; generated IDs: {:?}",
        ms(r.sampling),
        r.kv_bytes,
        r.state_bytes,
        r.tokens
    );
}
pub fn run(d: &MetalDevice) -> Result<()> {
    let o = Options::parse()?;
    if let Ok(value) = std::env::var("FERRUM_NATIVE_MATMUL") {
        d.set_native_matmul(value != "0")?;
    }
    if let Ok(limit) = std::env::var("FERRUM_BATCH_LIMIT") {
        d.set_batch_limit(
            limit
                .parse()
                .map_err(|_| Error::Parameter("invalid FERRUM_BATCH_LIMIT".into()))?,
        )?;
    }
    let (
        c,
        tok,
        model,
        source_bytes,
        tensor_count,
        parameter_count,
        quantized_bytes,
        config_tokenizer_load,
        weight_load,
        construction,
        construction_bytes,
        source_label,
    ) = if o.model.is_file() {
        let before = d.counters();
        let start = Instant::now();
        let loaded = qwen_gguf::load(d, &o.model)?;
        let elapsed = start.elapsed();
        let allocated = generation::counter_delta(before, d.counters()).allocated_bytes;
        (
            loaded.config,
            loaded.tokenizer.into(),
            loaded.model,
            loaded.source_tensor_bytes,
            loaded.tensor_count,
            loaded.parameter_count,
            loaded.quantized_tensor_bytes,
            elapsed,
            std::time::Duration::ZERO,
            std::time::Duration::ZERO,
            allocated,
            if loaded.architecture == "qwen3" {
                "Qwen3 GGUF"
            } else {
                "Qwen2 GGUF"
            },
        )
    } else {
        let config_path = o.model.join("config.json");
        let metadata: serde_json::Value = serde_json::from_slice(
            &std::fs::read(&config_path)
                .map_err(|e| Error::Config(format!("{}: {e}", config_path.display())))?,
        )
        .map_err(|e| Error::Config(format!("{}: {e}", config_path.display())))?;
        if metadata["model_type"] == "lfm2_moe" {
            let before = d.counters();
            let loaded = lfm2_moe::load(d, &o.model)?;
            let allocated = generation::counter_delta(before, d.counters()).allocated_bytes;
            (
                loaded.config,
                RuntimeTokenizer {
                    tokenizer: loaded.tokenizer,
                    eos_ids: loaded.eos_ids,
                    format: PromptFormat::Lfm2,
                },
                loaded.model,
                loaded.source_tensor_bytes,
                loaded.tensor_count,
                loaded.parameter_count,
                0,
                loaded.config_tokenizer_load,
                loaded.weight_load,
                loaded.construction,
                allocated,
                "LFM2.5-8B-A1B hybrid MoE safetensors",
            )
        } else if metadata["model_type"] == "lfm2" {
            let before = d.counters();
            let loaded = lfm2::load(d, &o.model)?;
            let allocated = generation::counter_delta(before, d.counters()).allocated_bytes;
            (
                loaded.config,
                RuntimeTokenizer {
                    tokenizer: loaded.tokenizer,
                    eos_ids: loaded.eos_ids,
                    format: PromptFormat::Lfm2,
                },
                loaded.model,
                loaded.source_tensor_bytes,
                loaded.tensor_count,
                loaded.parameter_count,
                0,
                loaded.config_tokenizer_load,
                loaded.weight_load,
                loaded.construction,
                allocated,
                "LFM2.5-230M hybrid safetensors",
            )
        } else if metadata["model_type"] == "qwen3" {
            let before = d.counters();
            let loaded = qwen3::load(d, &o.model)?;
            let allocated = generation::counter_delta(before, d.counters()).allocated_bytes;
            (
                loaded.config,
                loaded.tokenizer.into(),
                loaded.model,
                loaded.source_tensor_bytes,
                loaded.tensor_count,
                loaded.parameter_count,
                0,
                loaded.config_tokenizer_load,
                loaded.weight_load,
                loaded.construction,
                allocated,
                "Qwen3 safetensors",
            )
        } else if metadata["model_type"] == "granitemoehybrid" {
            let before = d.counters();
            let loaded = granite::load(d, &o.model)?;
            let allocated = generation::counter_delta(before, d.counters()).allocated_bytes;
            (
                loaded.config,
                RuntimeTokenizer {
                    tokenizer: loaded.tokenizer,
                    eos_ids: loaded.eos_ids,
                    format: PromptFormat::Granite,
                },
                loaded.model,
                loaded.source_tensor_bytes,
                loaded.tensor_count,
                loaded.parameter_count,
                0,
                loaded.config_tokenizer_load,
                loaded.weight_load,
                loaded.construction,
                allocated,
                "Granite 4 dense safetensors",
            )
        } else if metadata["model_type"] == "granitemoe" {
            let before = d.counters();
            let loaded = granite_moe::load(d, &o.model)?;
            let allocated = generation::counter_delta(before, d.counters()).allocated_bytes;
            (
                loaded.config,
                RuntimeTokenizer {
                    tokenizer: loaded.tokenizer,
                    eos_ids: loaded.eos_ids,
                    format: PromptFormat::Granite,
                },
                loaded.model,
                loaded.source_tensor_bytes,
                loaded.tensor_count,
                loaded.parameter_count,
                0,
                loaded.config_tokenizer_load,
                loaded.weight_load,
                loaded.construction,
                allocated,
                "Granite 3.1 1B-A400M MoE safetensors",
            )
        } else if metadata["model_type"] == "olmo2" {
            let before = d.counters();
            let loaded = olmo2::load(d, &o.model)?;
            let allocated = generation::counter_delta(before, d.counters()).allocated_bytes;
            (
                loaded.config,
                RuntimeTokenizer {
                    tokenizer: loaded.tokenizer,
                    eos_ids: loaded.eos_ids,
                    format: PromptFormat::Plain,
                },
                loaded.model,
                loaded.source_tensor_bytes,
                loaded.tensor_count,
                loaded.parameter_count,
                0,
                loaded.config_tokenizer_load,
                loaded.weight_load,
                loaded.construction,
                allocated,
                "OLMo 2 safetensors",
            )
        } else if metadata["model_type"] == "qwen2" {
            let qc = QwenConfig::from_file(&config_path)?;
            if qc.extra.contains_key("quantization") {
                let before = d.counters();
                let loaded = qwen_mlx::load(d, &o.model)?;
                let allocated = generation::counter_delta(before, d.counters()).allocated_bytes;
                (
                    loaded.config,
                    loaded.tokenizer.into(),
                    loaded.model,
                    loaded.source_tensor_bytes,
                    loaded.tensor_count,
                    loaded.parameter_count,
                    loaded.quantized_tensor_bytes,
                    loaded.config_tokenizer_load,
                    loaded.weight_load,
                    loaded.construction,
                    allocated,
                    "Qwen2 MLX affine Q4",
                )
            } else {
                let start = Instant::now();
                let c = qc.convert()?;
                let tok = QwenTokenizer::load(&o.model, &qc)?;
                let tokenizer_load = start.elapsed();
                let start = Instant::now();
                let source = Weights::from_file(d, o.model.join("model.safetensors"))?;
                let weight_load = start.elapsed();
                let source_bytes = source.bytes();
                let tensor_count = source.names().count();
                let parameter_count = source.names().try_fold(0usize, |total, name| {
                    total
                        .checked_add(source.get(name)?.numel())
                        .ok_or_else(|| Error::Shape("source parameter count overflow".into()))
                })?;
                let before = d.counters();
                let start = Instant::now();
                let model = qwen::construct(d, c.clone(), &source)?;
                let construction = start.elapsed();
                let construction_bytes =
                    generation::counter_delta(before, d.counters()).allocated_bytes;
                drop(source);
                (
                    c,
                    tok.into(),
                    model,
                    source_bytes,
                    tensor_count,
                    parameter_count,
                    0,
                    tokenizer_load,
                    weight_load,
                    construction,
                    construction_bytes,
                    "Qwen2.5-0.5B-Instruct safetensors",
                )
            }
        } else {
            return Err(Error::Config(format!(
                "unsupported model_type {:?}",
                metadata["model_type"]
            )));
        }
    };
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
    eprintln!(
        "Ferrum\nModel: {source_label}\nDevice: {}\nBackend: Metal\nDtype: {:?}\nLayers: {}; hidden: {}; intermediate: {}; Q heads: {}; KV heads: {}; head dim: {}\nParameters: {}; tensors: {}\nSource tensor bytes: {}\nRetained weight bytes: {}\nQuantized tensor bytes: {}\nLoader/construction allocations: {}\nRecommended working set: {} bytes",
        d.name(),
        c.dtype,
        c.num_layers,
        c.hidden_size,
        c.intermediate_size,
        c.num_attention_heads,
        c.num_key_value_heads,
        c.head_dim,
        parameter_count,
        tensor_count,
        source_bytes,
        model.weight_bytes(),
        quantized_bytes,
        construction_bytes,
        d.recommended_max_working_set()
    );
    eprintln!(
        "config/tokenizer/model load: {:.3} ms; weight read: {:.3} ms; construction: {:.3} ms; tokenization: {:.3} ms\nPrompt: {} tokens; sampler: {:?}",
        ms(config_tokenizer_load),
        ms(weight_load),
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
        let _ = model.take_moe_stats();
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
    let moe_stats = model.take_moe_stats();
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
    report(&r, ids.len(), o.profile);
    if o.profile {
        let mut sorted = r.decode.clone();
        sorted.sort();
        let median = sorted.get(sorted.len() / 2).copied().unwrap_or_default();
        eprintln!(
            "SUMMARY {}",
            serde_json::json!({"prompt_tokens":ids.len(),"generated_ids":r.tokens,"prefill_ms":ms(r.prefill),"prefill_tps":ids.len() as f64/r.prefill.as_secs_f64(),"first_token_ms":ms(r.first_token),"decode_median_ms":ms(median),"decode_tps":if median.is_zero(){0.}else{1./median.as_secs_f64()},"decode_ms":r.decode.iter().map(|x|ms(*x)).collect::<Vec<_>>(),"prefill_counters":r.prefill_counters,"decode_counters":r.decode_counters,"sampling_ms":ms(r.sampling),"retained_weight_bytes":model.weight_bytes(),"kv_active_bytes":r.kv_bytes,"kv_reserved_bytes":r.kv_reserved_bytes,"state_bytes":r.state_bytes,"moe":if moe_stats.assignments==0 {serde_json::Value::Null} else {serde_json::json!({"router_projection_enqueue_ms":ms(moe_stats.router_projection_enqueue),"route_selection_wall_ms":ms(moe_stats.routing),"routing_boundary_wait_ms":ms(moe_stats.routing_boundary_wait),"expert_dispatch_enqueue_ms":ms(moe_stats.expert_dispatch),"combine_enqueue_ms":ms(moe_stats.combine_dispatch),"active_expert_visits":moe_stats.active_experts,"assignments":moe_stats.assignments,"peak_temporary_bytes":moe_stats.peak_temporary_bytes})}})
        );
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
