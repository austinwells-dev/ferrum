#![recursion_limit = "256"]

//! Persistent JSON-lines runner for paired Phase 5.5 external comparisons.
//! Model loading is outside per-generation timings; each request starts with a fresh KV state.
#[path = "support/phase5_common.rs"]
mod phase5_common;

use ferrum::{
    Error, MetalDevice, Result, generation,
    loader::gguf::{GgufFile, MetadataValue},
    model::{Transformer, lfm2_moe, qwen_gguf},
};
use serde_json::{Value, json};
use std::{
    io::{self, BufRead, Write},
    path::Path,
};

struct BenchModel {
    model: Transformer,
    format: &'static str,
    repository: &'static str,
    revision: &'static str,
    source_tensor_bytes: usize,
    quantized_tensor_bytes: usize,
    tensor_count: usize,
    parameter_count: usize,
}

fn load_model(device: &MetalDevice, path: &Path) -> Result<BenchModel> {
    if path.is_dir() {
        let loaded = phase5_common::load(device, path)?;
        return Ok(BenchModel {
            model: loaded.model,
            format: loaded.format,
            repository: loaded.repository,
            revision: loaded.revision,
            source_tensor_bytes: loaded.source_tensor_bytes,
            quantized_tensor_bytes: loaded.quantized_tensor_bytes,
            tensor_count: loaded.tensor_count,
            parameter_count: loaded.parameter_count,
        });
    }

    let gguf = GgufFile::open(path)?;
    let architecture = match gguf.metadata_value("general.architecture") {
        Some(MetadataValue::String(value)) => value.as_str(),
        _ => {
            return Err(Error::Config(
                "GGUF is missing string general.architecture".into(),
            ));
        }
    };
    match architecture {
        "qwen2" | "qwen3" => {
            let loaded = qwen_gguf::load(device, path)?;
            let (repository, revision) = if loaded.architecture == "qwen3" {
                (
                    "Qwen/Qwen3-0.6B-GGUF",
                    "23749fefcc72300e3a2ad315e1317431b06b590a",
                )
            } else {
                (
                    "Qwen/Qwen2.5-0.5B-Instruct-GGUF",
                    "9217f5db79a29953eb74d5343926648285ec7e67",
                )
            };
            Ok(BenchModel {
                model: loaded.model,
                format: "GGUF",
                repository,
                revision,
                source_tensor_bytes: loaded.source_tensor_bytes,
                quantized_tensor_bytes: loaded.quantized_tensor_bytes,
                tensor_count: loaded.tensor_count,
                parameter_count: loaded.parameter_count,
            })
        }
        "lfm2moe" => {
            let metadata_dir = std::env::var_os("FERRUM_PHASE7A_LFM2_METADATA")
                .map(std::path::PathBuf::from)
                .ok_or_else(|| {
                    Error::Parameter(
                        "LFM2-MoE GGUF requires FERRUM_PHASE7A_LFM2_METADATA to point to its official config/tokenizer directory".into(),
                    )
                })?;
            let loaded = lfm2_moe::load_gguf(device, path, metadata_dir)?;
            Ok(BenchModel {
                model: loaded.model,
                format: "GGUF",
                repository: "LiquidAI/LFM2.5-8B-A1B-GGUF",
                revision: "49c14831707011e64d70b2ebd8462ba08d608434",
                source_tensor_bytes: loaded.source_tensor_bytes,
                quantized_tensor_bytes: loaded.quantized_tensor_bytes,
                tensor_count: loaded.tensor_count,
                parameter_count: loaded.parameter_count,
            })
        }
        other => Err(Error::Config(format!(
            "benchmark GGUF architecture {other:?} is unsupported"
        ))),
    }
}

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

fn handle(device: &MetalDevice, model: &BenchModel, request: &Value) -> Result<Value> {
    let prompt = prompt_ids(request)?;
    let max_new_tokens = request_usize(request, "max_new_tokens")?;
    let case = request_string(request, "case")?;
    let pair = request_usize(request, "pair")?;
    let pair_order = request_string(request, "pair_order")?;
    let warmup = request["warmup"].as_bool().unwrap_or(false);
    let attention_softmax_prefix = match request.get("attention_softmax_prefix") {
        None | Some(Value::Null) => {
            device.set_attention_softmax_prefix(true)?;
            true
        }
        Some(Value::Bool(enabled)) => {
            device.set_attention_softmax_prefix(*enabled)?;
            *enabled
        }
        Some(_) => {
            return Err(Error::Parameter(
                "request field attention_softmax_prefix must be a boolean".into(),
            ));
        }
    };
    let gguf_mpp_min_rows = match request.get("gguf_mpp_min_rows") {
        None | Some(Value::Null) => device.gguf_mpp_min_rows(),
        Some(value) => {
            let rows = value
                .as_u64()
                .and_then(|rows| usize::try_from(rows).ok())
                .ok_or_else(|| {
                    Error::Parameter("request field gguf_mpp_min_rows must be an integer".into())
                })?;
            device.set_gguf_mpp_min_rows(rows)?;
            Some(rows)
        }
    };
    let q8_0_mpp_tile_k64 = match request.get("q8_0_mpp_tile_k64") {
        None | Some(Value::Null) => {
            device.set_q8_0_mpp_tile_k64(true)?;
            true
        }
        Some(Value::Bool(enabled)) => {
            device.set_q8_0_mpp_tile_k64(*enabled)?;
            *enabled
        }
        Some(_) => {
            return Err(Error::Parameter(
                "request field q8_0_mpp_tile_k64 must be a boolean".into(),
            ));
        }
    };
    let q8_0_gemv_k_split = match request.get("q8_0_gemv_k_split") {
        None | Some(Value::Null) => {
            device.set_q8_0_gemv_k_split(true)?;
            true
        }
        Some(Value::Bool(enabled)) => {
            device.set_q8_0_gemv_k_split(*enabled)?;
            *enabled
        }
        Some(_) => {
            return Err(Error::Parameter(
                "request field q8_0_gemv_k_split must be a boolean".into(),
            ));
        }
    };
    let q5_1_mpp_tile_k64 = match request.get("q5_1_mpp_tile_k64") {
        None | Some(Value::Null) => {
            device.set_q5_1_mpp_tile_k64(true)?;
            true
        }
        Some(Value::Bool(enabled)) => {
            device.set_q5_1_mpp_tile_k64(*enabled)?;
            *enabled
        }
        Some(_) => {
            return Err(Error::Parameter(
                "request field q5_1_mpp_tile_k64 must be a boolean".into(),
            ));
        }
    };
    let q4_k_mpp_tile_k64 = match request.get("q4_k_mpp_tile_k64") {
        None | Some(Value::Null) => {
            device.set_q4_k_mpp_tile_k64(true)?;
            true
        }
        Some(Value::Bool(enabled)) => {
            device.set_q4_k_mpp_tile_k64(*enabled)?;
            *enabled
        }
        Some(_) => {
            return Err(Error::Parameter(
                "request field q4_k_mpp_tile_k64 must be a boolean".into(),
            ));
        }
    };
    let q4_k_mpp_tile_m128 = match request.get("q4_k_mpp_tile_m128") {
        None | Some(Value::Null) => true,
        Some(Value::Bool(enabled)) => {
            device.set_q4_k_mpp_tile_m128(*enabled)?;
            *enabled
        }
        Some(_) => {
            return Err(Error::Parameter(
                "request field q4_k_mpp_tile_m128 must be a boolean".into(),
            ));
        }
    };
    let q4_k_expert_project_8rows = match request.get("q4_k_expert_project_8rows") {
        None | Some(Value::Null) => {
            device.set_q4_k_expert_project_8rows(true)?;
            true
        }
        Some(Value::Bool(enabled)) => {
            device.set_q4_k_expert_project_8rows(*enabled)?;
            *enabled
        }
        Some(_) => {
            return Err(Error::Parameter(
                "request field q4_k_expert_project_8rows must be a boolean".into(),
            ));
        }
    };
    let q6_k_expert_project_8rows = match request.get("q6_k_expert_project_8rows") {
        None | Some(Value::Null) => {
            device.set_q6_k_expert_project_8rows(true)?;
            true
        }
        Some(Value::Bool(enabled)) => {
            device.set_q6_k_expert_project_8rows(*enabled)?;
            *enabled
        }
        Some(_) => {
            return Err(Error::Parameter(
                "request field q6_k_expert_project_8rows must be a boolean".into(),
            ));
        }
    };
    let q5_k_mpp_tile_k64 = match request.get("q5_k_mpp_tile_k64") {
        None | Some(Value::Null) => {
            device.set_q5_k_mpp_tile_k64(true)?;
            true
        }
        Some(Value::Bool(enabled)) => {
            device.set_q5_k_mpp_tile_k64(*enabled)?;
            *enabled
        }
        Some(_) => {
            return Err(Error::Parameter(
                "request field q5_k_mpp_tile_k64 must be a boolean".into(),
            ));
        }
    };
    let q5_0_gemv_n4 = match request.get("q5_0_gemv_n4") {
        None | Some(Value::Null) => {
            device.set_q5_0_gemv_n4(true)?;
            true
        }
        Some(Value::Bool(enabled)) => {
            device.set_q5_0_gemv_n4(*enabled)?;
            *enabled
        }
        Some(_) => {
            return Err(Error::Parameter(
                "request field q5_0_gemv_n4 must be a boolean".into(),
            ));
        }
    };
    let q5_1_gemv_n4 = match request.get("q5_1_gemv_n4") {
        None | Some(Value::Null) => {
            device.set_q5_1_gemv_n4(true)?;
            true
        }
        Some(Value::Bool(enabled)) => {
            device.set_q5_1_gemv_n4(*enabled)?;
            *enabled
        }
        Some(_) => {
            return Err(Error::Parameter(
                "request field q5_1_gemv_n4 must be a boolean".into(),
            ));
        }
    };
    let q5_k_gemv_8rows = match request.get("q5_k_gemv_8rows") {
        None | Some(Value::Null) => {
            device.set_q5_k_gemv_8rows(false)?;
            false
        }
        Some(Value::Bool(enabled)) => {
            device.set_q5_k_gemv_8rows(*enabled)?;
            *enabled
        }
        Some(_) => {
            return Err(Error::Parameter(
                "request field q5_k_gemv_8rows must be a boolean".into(),
            ));
        }
    };
    let expert_tensorops = match request.get("moe_expert_tensorops") {
        None | Some(Value::Null) => device.moe_expert_tensorops_enabled(),
        Some(Value::Bool(enabled)) => {
            device.set_moe_expert_tensorops(*enabled)?;
            *enabled
        }
        Some(_) => {
            return Err(Error::Parameter(
                "request field moe_expert_tensorops must be a boolean".into(),
            ));
        }
    };
    let expert_tensorops_tile_k64 = match request.get("moe_expert_tensorops_tile_k64") {
        None | Some(Value::Null) => device.moe_expert_tensorops_tile_k64(),
        Some(Value::Bool(enabled)) => {
            device.set_moe_expert_tensorops_tile_k64(*enabled)?;
            *enabled
        }
        Some(_) => {
            return Err(Error::Parameter(
                "request field moe_expert_tensorops_tile_k64 must be a boolean".into(),
            ));
        }
    };
    let expert_tensorops_min_routes = match request
        .get("moe_expert_tensorops_min_routes_per_expert")
    {
        None | Some(Value::Null) => device.moe_expert_tensorops_min_routes_per_expert(),
        Some(value) => {
            let routes = value
                .as_u64()
                .and_then(|routes| usize::try_from(routes).ok())
                .filter(|routes| *routes > 0)
                .ok_or_else(|| Error::Parameter(
                    "request field moe_expert_tensorops_min_routes_per_expert must be a positive integer".into(),
                ))?;
            device.set_moe_expert_tensorops_min_routes_per_expert(routes)?;
            routes
        }
    };
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
        "attention_softmax_prefix": attention_softmax_prefix,
        "gguf_mpp_min_rows": gguf_mpp_min_rows,
        "moe_expert_tensorops": expert_tensorops,
        "moe_expert_tensorops_tile_k64": expert_tensorops_tile_k64,
        "moe_expert_tensorops_min_routes_per_expert": expert_tensorops_min_routes,
        "q8_0_mpp_tile_k64": q8_0_mpp_tile_k64,
        "q8_0_gemv_k_split": q8_0_gemv_k_split,
        "q5_1_mpp_tile_k64": q5_1_mpp_tile_k64,
        "q4_k_mpp_tile_k64": q4_k_mpp_tile_k64,
        "q4_k_mpp_tile_m128": q4_k_mpp_tile_m128,
        "q4_k_expert_project_8rows": q4_k_expert_project_8rows,
        "q6_k_expert_project_8rows": q6_k_expert_project_8rows,
        "q5_k_mpp_tile_k64": q5_k_mpp_tile_k64,
        "q5_0_gemv_n4": q5_0_gemv_n4,
        "q5_1_gemv_n4": q5_1_gemv_n4,
        "q5_k_gemv_8rows": q5_k_gemv_8rows,
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
        "tensor_count": model.tensor_count,
        "parameter_count": model.parameter_count,
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
    if let Ok(value) = std::env::var("FERRUM_Q4_K_GEMV_8ROWS") {
        let enabled = match value.as_str() {
            "1" | "true" => true,
            "0" | "false" => false,
            _ => return Err(Error::Parameter("invalid FERRUM_Q4_K_GEMV_8ROWS".into())),
        };
        device.set_q4_k_gemv_8rows(enabled)?;
    }
    if let Ok(value) = std::env::var("FERRUM_Q6_K_GEMV_8ROWS") {
        let enabled = match value.as_str() {
            "1" | "true" => true,
            "0" | "false" => false,
            _ => return Err(Error::Parameter("invalid FERRUM_Q6_K_GEMV_8ROWS".into())),
        };
        device.set_q6_k_gemv_8rows(enabled)?;
    }
    if let Ok(value) = std::env::var("FERRUM_MOE_GPU_ROUTING") {
        let enabled = match value.as_str() {
            "1" | "true" => true,
            "0" | "false" => false,
            _ => return Err(Error::Parameter("invalid FERRUM_MOE_GPU_ROUTING".into())),
        };
        device.set_moe_gpu_routing(enabled)?;
    }
    if let Ok(value) = std::env::var("FERRUM_MOE_EXPERT_TENSOROPS") {
        let enabled = match value.as_str() {
            "1" | "true" => true,
            "0" | "false" => false,
            _ => {
                return Err(Error::Parameter(
                    "invalid FERRUM_MOE_EXPERT_TENSOROPS".into(),
                ));
            }
        };
        device.set_moe_expert_tensorops(enabled)?;
    }
    if let Ok(value) = std::env::var("FERRUM_MOE_EXPERT_TENSOROPS_MIN_ROUTES_PER_EXPERT") {
        let routes = value
            .parse::<usize>()
            .ok()
            .filter(|routes| *routes > 0)
            .ok_or_else(|| {
                Error::Parameter("invalid FERRUM_MOE_EXPERT_TENSOROPS_MIN_ROUTES_PER_EXPERT".into())
            })?;
        device.set_moe_expert_tensorops_min_routes_per_expert(routes)?;
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
    let model = load_model(&device, Path::new(&args[1]))?;
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
