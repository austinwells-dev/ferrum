//! Paired production-generation matrix for the Phase 4 BF16 baseline and GGUF.
#[path = "support/phase5_common.rs"]
mod phase5_common;

use ferrum::{
    Error, MetalDevice, Result, generation,
    loader::Weights,
    model::{Transformer, qwen},
    tokenizer::qwen::{DEFAULT_SYSTEM, QwenTokenizer},
};

struct GenerationCase<'a> {
    model: &'a Transformer,
    label: &'static str,
    model_kind: &'a str,
    kernel_path: &'static str,
    pair: usize,
    order: usize,
    prompt: &'a [u32],
    generated: usize,
}

fn run_case(device: &MetalDevice, case: GenerationCase<'_>) -> Result<()> {
    let started = std::time::Instant::now();
    let result = generation::generate(
        device,
        case.model,
        case.prompt,
        case.generated,
        &[],
        generation::argmax,
        |_| Ok(()),
    )?;
    let elapsed = started.elapsed();
    let mut decode_ms: Vec<_> = result
        .decode
        .iter()
        .map(|duration| duration.as_secs_f64() * 1000.0)
        .collect();
    decode_ms.sort_by(f64::total_cmp);
    let median_decode_ms = decode_ms.get(decode_ms.len() / 2).copied();
    println!(
        "{}",
        serde_json::json!({
            "case": case.label,
            "model": case.model_kind,
            "kernel_path": case.kernel_path,
            "pair": case.pair,
            "pair_order": case.order,
            "generation_ms": elapsed.as_secs_f64() * 1000.0,
            "generation_tps": result.tokens.len() as f64 / elapsed.as_secs_f64(),
            "post_first_token_tps": if elapsed > result.first_token { (result.tokens.len().saturating_sub(1)) as f64 / (elapsed - result.first_token).as_secs_f64() } else { 0.0 },
            "prefill_ms": result.prefill.as_secs_f64() * 1000.0,
            "prefill_tps": case.prompt.len() as f64 / result.prefill.as_secs_f64(),
            "first_token_ms": result.first_token.as_secs_f64() * 1000.0,
            "decode_median_ms": median_decode_ms,
            "decode_tps": median_decode_ms.map(|ms| 1000.0 / ms),
            "decode_aggregate_tps": result.decode.len() as f64 / result.decode.iter().sum::<std::time::Duration>().as_secs_f64(),
            "decode_ms": result.decode.iter().map(|duration| duration.as_secs_f64() * 1000.0).collect::<Vec<_>>(),
            "sampling_ms": result.sampling.as_secs_f64() * 1000.0,
            "emit_callback": "noop",
            "prompt_ids": case.prompt,
            "generated_ids": result.tokens,
            "prefill_counters": result.prefill_counters,
            "decode_counters": result.decode_counters,
            "kv_active_bytes": result.kv_bytes,
            "kv_reserved_bytes": result.kv_reserved_bytes,
            "prefill_profile": result.prefill_profile,
            "decode_profiles": result.decode_profiles,
        })
    );
    Ok(())
}

fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().collect();
    if !(3..=4).contains(&args.len()) {
        return Err(Error::Parameter(
            "usage: phase5_matrix BF16_CHECKPOINT_DIR QUANTIZED_GGUF_OR_MLX_DIR [PAIRED_REPEATS]"
                .into(),
        ));
    }
    let bf16_dir = std::path::Path::new(&args[1]);
    let quantized_path = std::path::Path::new(&args[2]);
    let repeats = args
        .get(3)
        .map(|value| value.parse::<usize>())
        .transpose()
        .map_err(|_| Error::Parameter("paired repeats must be an integer".into()))?
        .unwrap_or(3);
    if repeats == 0 {
        return Err(Error::Parameter("paired repeats must be positive".into()));
    }

    let device = MetalDevice::new()?;
    if let Ok(limit) = std::env::var("FERRUM_BATCH_LIMIT") {
        device.set_batch_limit(
            limit
                .parse()
                .map_err(|_| Error::Parameter("invalid FERRUM_BATCH_LIMIT".into()))?,
        )?;
    }
    device.set_profiling(std::env::var_os("FERRUM_MATRIX_PROFILE").is_some());
    let kernel_ab = std::env::var("FERRUM_PHASE5_KERNEL_AB").ok();
    let compare_mpp_direct = kernel_ab.as_deref() == Some("mpp-direct");
    let compare_mpp_k64 = kernel_ab.as_deref() == Some("mpp-k64");
    if kernel_ab.is_some() && !compare_mpp_direct && !compare_mpp_k64 {
        return Err(Error::Parameter(
            "FERRUM_PHASE5_KERNEL_AB must be mpp-direct or mpp-k64".into(),
        ));
    }
    if (compare_mpp_direct || compare_mpp_k64)
        && device.capabilities()?["mpp_available"].as_bool() != Some(true)
    {
        return Err(Error::Parameter(
            "FERRUM_PHASE5_KERNEL_AB requires Metal 4 MPP support".into(),
        ));
    }

    let bf16_config = qwen::QwenConfig::from_file(bf16_dir.join("config.json"))?;
    let bf16_tokenizer = QwenTokenizer::load(bf16_dir, &bf16_config)?;
    let bf16_source = Weights::from_file(&device, bf16_dir.join("model.safetensors"))?;
    let bf16_model = qwen::construct(&device, bf16_config.convert()?, &bf16_source)?;
    drop(bf16_source);

    let quantized = phase5_common::load(&device, quantized_path)?;
    let reference_config = bf16_model.config();
    let quantized_config = &quantized.config;
    if quantized_config.vocab_size != reference_config.vocab_size
        || quantized_config.hidden_size != reference_config.hidden_size
        || quantized_config.intermediate_size != reference_config.intermediate_size
        || quantized_config.num_layers != reference_config.num_layers
        || quantized_config.num_attention_heads != reference_config.num_attention_heads
        || quantized_config.num_key_value_heads != reference_config.num_key_value_heads
        || quantized_config.head_dim != reference_config.head_dim
        || quantized_config.max_context_length != reference_config.max_context_length
    {
        return Err(Error::Config(format!(
            "quantized dimensions {:?} differ from BF16 reference {:?}",
            quantized_config, reference_config
        )));
    }
    let short = bf16_tokenizer
        .encode_prompt("Hello!", DEFAULT_SYSTEM, false)?
        .1;
    let prose_text = "A runtime executes a sequence of tensor operations. Memory ownership determines when storage can be reused. Numerical tests compare computed results with reference values. Measurements distinguish time spent computing from time spent waiting. ".repeat(100);
    let prose = bf16_tokenizer.encode_prompt(&prose_text, "", true)?.1;
    let quantized_short = quantized
        .tokenizer
        .encode_prompt("Hello!", DEFAULT_SYSTEM, false)?
        .1;
    let quantized_prose = quantized.tokenizer.encode_prompt(&prose_text, "", true)?.1;
    if quantized_short != short || quantized_prose != prose {
        return Err(Error::Tokenizer("BF16/GGUF prompt token IDs differ".into()));
    }
    let quantized_label = quantized_path
        .file_stem()
        .and_then(|value| value.to_str())
        .unwrap_or("quantized-model");
    let cases = [
        ("short", short.clone(), 17),
        ("128", prose[..128].to_vec(), 17),
        ("512", prose[..512].to_vec(), 17),
        ("1024", prose[..1024].to_vec(), 17),
        ("sustained-128-decode", short, 129),
        ("long-horizon-1601", prose[..368].to_vec(), 1601),
    ];
    let selected = std::env::var("FERRUM_PHASE5_CASE").ok();
    eprintln!(
        "Phase 5 paired matrix: device={}; BF16 retained={} bytes; {} {}@{} source={} bytes; quantized retained={} bytes; packed={} bytes; tensors={}; parameters={}; repeats={}",
        device.name(),
        bf16_model.weight_bytes(),
        quantized.format,
        quantized.repository,
        quantized.revision,
        quantized.source_tensor_bytes,
        quantized.model.weight_bytes(),
        quantized.quantized_tensor_bytes,
        quantized.tensor_count,
        quantized.parameter_count,
        repeats,
    );

    for (label, prompt, generated) in cases {
        if selected.as_ref().is_some_and(|case| case != label) {
            continue;
        }
        if prompt.len() + generated > bf16_model.config().max_context_length {
            return Err(Error::Config(format!("case {label} exceeds model context")));
        }
        if compare_mpp_direct || compare_mpp_k64 {
            // Compile both projection pipelines and warm both paths before timing.
            let kernel_modes = if compare_mpp_direct {
                [("mpp-k128", true, false), ("direct", false, false)]
            } else {
                [("mpp-k128", true, false), ("mpp-k64", true, true)]
            };
            for (_, native_matmul, tile_k64) in kernel_modes {
                device.set_native_matmul(native_matmul)?;
                device.set_mlx_affine4_mpp_tile_k64(Some(tile_k64))?;
                generation::generate(
                    &device,
                    &quantized.model,
                    &prompt,
                    2,
                    &[],
                    generation::argmax,
                    |_| Ok(()),
                )?;
            }
            device.set_native_matmul(true)?;
            device.set_mlx_affine4_mpp_tile_k64(None)?;
            for pair in 0..repeats {
                let order = if pair % 2 == 0 {
                    kernel_modes
                } else {
                    [kernel_modes[1], kernel_modes[0]]
                };
                for (position, (kernel_path, native_matmul, tile_k64)) in
                    order.into_iter().enumerate()
                {
                    device.set_native_matmul(native_matmul)?;
                    device.set_mlx_affine4_mpp_tile_k64(Some(tile_k64))?;
                    run_case(
                        &device,
                        GenerationCase {
                            model: &quantized.model,
                            label,
                            model_kind: quantized_label,
                            kernel_path,
                            pair,
                            order: position,
                            prompt: &prompt,
                            generated,
                        },
                    )?;
                }
            }
            device.set_native_matmul(true)?;
            device.set_mlx_affine4_mpp_tile_k64(None)?;
            continue;
        }
        for model in [&bf16_model, &quantized.model] {
            generation::generate(
                &device,
                model,
                &prompt,
                generated,
                &[],
                generation::argmax,
                |_| Ok(()),
            )?;
        }
        for pair in 0..repeats {
            let order = if pair % 2 == 0 {
                [("bf16", &bf16_model), (quantized_label, &quantized.model)]
            } else {
                [(quantized_label, &quantized.model), ("bf16", &bf16_model)]
            };
            for (position, (kind, model)) in order.into_iter().enumerate() {
                run_case(
                    &device,
                    GenerationCase {
                        model,
                        label,
                        model_kind: kind,
                        kernel_path: "production-default",
                        pair,
                        order: position,
                        prompt: &prompt,
                        generated,
                    },
                )?;
            }
        }
    }
    Ok(())
}
