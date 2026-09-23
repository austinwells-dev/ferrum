//! Persistent JSON-lines runner for paired Phase 5.5 external comparisons.
//! Model loading is outside per-generation timings; each request starts with a fresh KV state.
#[path = "support/phase5_common.rs"]
mod phase5_common;

use ferrum::{Error, MetalDevice, Result, generation};
use serde_json::{Value, json};
use std::{
    io::{self, BufRead, Write},
    path::Path,
};

fn request_usize(request: &Value, key: &str) -> Result<usize> {
    request[key]
        .as_u64()
        .and_then(|value| usize::try_from(value).ok())
        .ok_or_else(|| {
            Error::Parameter(format!("request field {key} must be a nonnegative integer"))
        })
}

fn request_string<'a>(request: &'a Value, key: &str) -> Result<&'a str> {
    request[key]
        .as_str()
        .ok_or_else(|| Error::Parameter(format!("request field {key} must be a string")))
}

fn prompt_ids(request: &Value) -> Result<Vec<u32>> {
    let values = request["prompt_ids"]
        .as_array()
        .ok_or_else(|| Error::Parameter("request field prompt_ids must be an array".into()))?;
    let ids = values
        .iter()
        .map(|value| {
            value
                .as_u64()
                .and_then(|id| u32::try_from(id).ok())
                .ok_or_else(|| Error::Parameter("prompt_ids must contain uint32 values".into()))
        })
        .collect::<Result<Vec<_>>>()?;
    if ids.is_empty() {
        return Err(Error::Parameter("prompt_ids cannot be empty".into()));
    }
    Ok(ids)
}

fn handle(
    device: &MetalDevice,
    model: &phase5_common::QuantizedModel,
    request: &Value,
) -> Result<Value> {
    let prompt = prompt_ids(request)?;
    let max_new_tokens = request_usize(request, "max_new_tokens")?;
    let case = request_string(request, "case")?;
    let pair = request_usize(request, "pair")?;
    let pair_order = request_string(request, "pair_order")?;
    let warmup = request["warmup"].as_bool().unwrap_or(false);
    if max_new_tokens == 0 {
        return Err(Error::Parameter("max_new_tokens must be positive".into()));
    }

    let start = std::time::Instant::now();
    let result = generation::generate(
        device,
        &model.model,
        &prompt,
        max_new_tokens,
        &[],
        generation::argmax,
        |_| Ok(()),
    )?;
    let generation_ms = start.elapsed().as_secs_f64() * 1000.0;
    let mut decode_ms = result
        .decode
        .iter()
        .map(|duration| duration.as_secs_f64() * 1000.0)
        .collect::<Vec<_>>();
    decode_ms.sort_by(f64::total_cmp);
    let decode_median_ms = decode_ms.get(decode_ms.len() / 2).copied();

    Ok(json!({
        "runtime": "ferrum",
        "case": case,
        "pair": pair,
        "pair_order": pair_order,
        "warmup": warmup,
        "model_format": model.format,
        "model_repository": model.repository,
        "model_revision": model.revision,
        "prompt_tokens": prompt.len(),
        "generated_tokens": result.tokens.len(),
        "prompt_ids": prompt,
        "generated_ids": result.tokens,
        "generation_ms": generation_ms,
        "generation_tps": result.tokens.len() as f64 / (generation_ms / 1000.0).max(f64::MIN_POSITIVE),
        "prefill_ms": result.prefill.as_secs_f64() * 1000.0,
        "prefill_tps": prompt.len() as f64 / result.prefill.as_secs_f64(),
        "first_token_ms": result.first_token.as_secs_f64() * 1000.0,
        "decode_ms": result.decode.iter().map(|duration| duration.as_secs_f64() * 1000.0).collect::<Vec<_>>(),
        "decode_median_ms": decode_median_ms,
        "decode_tps": decode_median_ms.map(|ms| 1000.0 / ms),
        "decode_aggregate_tps": result.decode.len() as f64
            / result.decode.iter().sum::<std::time::Duration>().as_secs_f64().max(f64::MIN_POSITIVE),
        "sampling_ms": result.sampling.as_secs_f64() * 1000.0,
        "retained_weight_bytes": model.model.weight_bytes(),
        "source_tensor_bytes": model.source_tensor_bytes,
        "packed_quantized_bytes": model.quantized_tensor_bytes,
        "prefill_counters": result.prefill_counters,
        "decode_counters": result.decode_counters,
        "kv_active_bytes": result.kv_bytes,
        "kv_reserved_bytes": result.kv_reserved_bytes,
        "prefill_profile": result.prefill_profile,
        "decode_profiles": result.decode_profiles,
    }))
}

fn main() -> Result<()> {
    let args = std::env::args().collect::<Vec<_>>();
    if args.len() != 2 {
        return Err(Error::Parameter(
            "usage: phase55_bench GGUF_FILE_OR_MLX_MODEL_DIR".into(),
        ));
    }
    let device = MetalDevice::new()?;
    device.set_profiling(std::env::var_os("FERRUM_MATRIX_PROFILE").is_some());
    if let Ok(value) = std::env::var("FERRUM_Q4_0_GEMV_8ROWS") {
        let enabled = match value.as_str() {
            "1" | "true" => true,
            "0" | "false" => false,
            _ => return Err(Error::Parameter("invalid FERRUM_Q4_0_GEMV_8ROWS".into())),
        };
        device.set_q4_0_gemv_8rows(enabled)?;
    }
    if let Ok(value) = std::env::var("FERRUM_MLX_AFFINE4_GEMV_QUAD") {
        let enabled = match value.as_str() {
            "1" | "true" => true,
            "0" | "false" => false,
            _ => {
                return Err(Error::Parameter(
                    "invalid FERRUM_MLX_AFFINE4_GEMV_QUAD".into(),
                ));
            }
        };
        device.set_mlx_affine4_gemv_quad(enabled)?;
    }
    if let Ok(value) = std::env::var("FERRUM_BATCH_LIMIT") {
        let limit = value
            .parse::<usize>()
            .map_err(|_| Error::Parameter("invalid FERRUM_BATCH_LIMIT".into()))?;
        device.set_batch_limit(limit)?;
    }
    let model = phase5_common::load(&device, Path::new(&args[1]))?;
    eprintln!(
        "phase55_bench ready: format={} weights={} packed={} tensors={}",
        model.format,
        model.model.weight_bytes(),
        model.quantized_tensor_bytes,
        model.tensor_count
    );

    let stdin = io::stdin();
    let mut stdout = io::BufWriter::new(io::stdout().lock());
    for line in stdin.lock().lines() {
        let line = line.map_err(|error| Error::Parameter(format!("stdin read failed: {error}")))?;
        if line.trim().is_empty() {
            continue;
        }
        let response = match serde_json::from_str::<Value>(&line) {
            Ok(request) => handle(&device, &model, &request),
            Err(error) => Err(Error::Parameter(format!("invalid JSON request: {error}"))),
        };
        match response {
            Ok(value) => writeln!(stdout, "{value}")
                .map_err(|error| Error::Parameter(format!("stdout write failed: {error}")))?,
            Err(error) => writeln!(stdout, "{}", json!({"error": error.to_string()}))
                .map_err(|error| Error::Parameter(format!("stdout write failed: {error}")))?,
        }
        stdout
            .flush()
            .map_err(|error| Error::Parameter(format!("stdout flush failed: {error}")))?;
    }
    Ok(())
}
