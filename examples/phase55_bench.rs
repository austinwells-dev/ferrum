#![recursion_limit = "256"]

//! Persistent JSON-lines runner for paired Phase 5.5 external comparisons.
//! Model loading is outside per-generation timings; each request starts with a fresh KV state.
#[path = "support/phase5_common.rs"]
mod phase5_common;

use ferrum::{
    Error, MetalDevice, Result, generation,
    loader::gguf::{GgufFile, MetadataValue},
    model::{Transformer, granite_moe, lfm2_moe, qwen_gguf},
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
        let config: Value = serde_json::from_slice(
            &std::fs::read(path.join("config.json"))
                .map_err(|e| Error::Config(format!("{}: {e}", path.display())))?,
        )
        .map_err(|e| Error::Config(format!("{} config.json: {e}", path.display())))?;
        if config["model_type"] == "granitemoe" {
            let loaded = granite_moe::load(device, path)?;
            return Ok(BenchModel {
                model: loaded.model,
                format: "BF16 safetensors",
                repository: "ibm-granite/granite-3.1-1b-a400m-instruct",
                revision: "0da7a48b0276d500ce5922fd2b33944091fc6c09",
                source_tensor_bytes: loaded.source_tensor_bytes,
                quantized_tensor_bytes: 0,
                tensor_count: loaded.tensor_count,
                parameter_count: loaded.parameter_count,
            });
        }
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
        "granitemoe" => Err(Error::Config(
            "Granite MoE GGUF is not supported by the Phase 5.5 runner; use its official BF16 safetensors directory for internal candidate/control measurements".into(),
        )),
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

fn phase7a_cpu_preflight() -> Result<()> {
    let guard = Path::new(env!("CARGO_MANIFEST_DIR")).join("scripts/phase7a_cpu_preflight.py");
    let output = std::process::Command::new("python3")
        .arg(guard)
        .arg("--wait")
        // stdout carries the JSON-lines benchmark protocol to the parent.
        .output()
        .map_err(|error| Error::Config(format!("failed to run Phase 7A CPU preflight: {error}")))?;
    eprint!("{}", String::from_utf8_lossy(&output.stdout));
    eprint!("{}", String::from_utf8_lossy(&output.stderr));
    if !output.status.success() {
        return Err(Error::Config(format!(
            "Phase 7A CPU preflight failed with status {}",
            output.status
        )));
    }
    Ok(())
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
    let attention_softmax_prefix_reuse = match request.get("attention_softmax_prefix_reuse") {
        None | Some(Value::Null) => {
            device.set_attention_softmax_prefix_reuse(true)?;
            true
        }
        Some(Value::Bool(enabled)) => {
            device.set_attention_softmax_prefix_reuse(*enabled)?;
            *enabled
        }
        Some(_) => {
            return Err(Error::Parameter(
                "request field attention_softmax_prefix_reuse must be a boolean".into(),
            ));
        }
    };
    let gguf_mpp_min_rows = match request.get("gguf_mpp_min_rows") {
        None | Some(Value::Null) => {
            device.clear_gguf_mpp_min_rows()?;
            device.gguf_mpp_min_rows()
        }
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
    let gguf_mpp_q5_1_min_rows = match request.get("gguf_mpp_q5_1_min_rows") {
        None | Some(Value::Null) => {
            device.clear_gguf_mpp_q5_1_min_rows()?;
            device.gguf_mpp_q5_1_min_rows()
        }
        Some(value) => {
            let rows = value
                .as_u64()
                .and_then(|rows| usize::try_from(rows).ok())
                .ok_or_else(|| {
                    Error::Parameter(
                        "request field gguf_mpp_q5_1_min_rows must be an integer".into(),
                    )
                })?;
            device.set_gguf_mpp_q5_1_min_rows(rows)?;
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
        None | Some(Value::Null) => {
            device.set_q4_k_mpp_tile_m128(true)?;
            true
        }
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
    let q4_k_expert_project_16rows = match request.get("q4_k_expert_project_16rows") {
        None | Some(Value::Null) => {
            device.set_q4_k_expert_project_16rows(true)?;
            true
        }
        Some(Value::Bool(enabled)) => {
            device.set_q4_k_expert_project_16rows(*enabled)?;
            *enabled
        }
        Some(_) => {
            return Err(Error::Parameter(
                "request field q4_k_expert_project_16rows must be a boolean".into(),
            ));
        }
    };
    let q4_k_expert_project_16rows_pairs = match request.get("q4_k_expert_project_16rows_pairs") {
        None | Some(Value::Null) => {
            device.set_q4_k_expert_project_16rows_pairs(true)?;
            true
        }
        Some(Value::Bool(enabled)) => {
            device.set_q4_k_expert_project_16rows_pairs(*enabled)?;
            *enabled
        }
        Some(_) => {
            return Err(Error::Parameter(
                "request field q4_k_expert_project_16rows_pairs must be a boolean".into(),
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
            device.set_q5_k_gemv_8rows(true)?;
            true
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
    let moe_gpu_routing_prefill = match request.get("moe_gpu_routing_prefill") {
        None | Some(Value::Null) => {
            device.set_moe_gpu_routing_prefill(true)?;
            true
        }
        Some(Value::Bool(enabled)) => {
            device.set_moe_gpu_routing_prefill(*enabled)?;
            *enabled
        }
        Some(_) => {
            return Err(Error::Parameter(
                "request field moe_gpu_routing_prefill must be a boolean".into(),
            ));
        }
    };
    let shared_encoder = match request.get("shared_encoder") {
        None | Some(Value::Null) => {
            device.set_shared_encoder(true)?;
            true
        }
        Some(Value::Bool(enabled)) => {
            device.set_shared_encoder(*enabled)?;
            *enabled
        }
        Some(_) => {
            return Err(Error::Parameter(
                "request field shared_encoder must be a boolean".into(),
            ));
        }
    };
    let concurrent_dispatch = match request.get("concurrent_dispatch") {
        None | Some(Value::Null) => {
            device.set_concurrent_dispatch(true)?;
            true
        }
        Some(Value::Bool(enabled)) => {
            device.set_concurrent_dispatch(*enabled)?;
            *enabled
        }
        Some(_) => {
            return Err(Error::Parameter(
                "request field concurrent_dispatch must be a boolean".into(),
            ));
        }
    };
    let attention_context_decode_wide = match request.get("attention_context_decode_wide") {
        None | Some(Value::Null) => {
            device.set_attention_context_decode_wide(true)?;
            true
        }
        Some(Value::Bool(enabled)) => {
            device.set_attention_context_decode_wide(*enabled)?;
            *enabled
        }
        Some(_) => {
            return Err(Error::Parameter(
                "request field attention_context_decode_wide must be a boolean".into(),
            ));
        }
    };
    let moe_chunk_device_copy = match request.get("moe_chunk_device_copy") {
        None | Some(Value::Null) => {
            device.set_moe_chunk_device_copy(true)?;
            true
        }
        Some(Value::Bool(enabled)) => {
            device.set_moe_chunk_device_copy(*enabled)?;
            *enabled
        }
        Some(_) => {
            return Err(Error::Parameter(
                "request field moe_chunk_device_copy must be a boolean".into(),
            ));
        }
    };
    let q4_k_factored = match request.get("q4_k_factored") {
        None | Some(Value::Null) => {
            device.set_q4_k_factored(true)?;
            true
        }
        Some(Value::Bool(enabled)) => {
            device.set_q4_k_factored(*enabled)?;
            *enabled
        }
        Some(_) => {
            return Err(Error::Parameter(
                "request field q4_k_factored must be a boolean".into(),
            ));
        }
    };
    let q6_k_factored = match request.get("q6_k_factored") {
        None | Some(Value::Null) => {
            device.set_q6_k_factored(true)?;
            true
        }
        Some(Value::Bool(enabled)) => {
            device.set_q6_k_factored(*enabled)?;
            *enabled
        }
        Some(_) => {
            return Err(Error::Parameter(
                "request field q6_k_factored must be a boolean".into(),
            ));
        }
    };
    let mpp_fast_dequant = match request.get("mpp_fast_dequant") {
        None | Some(Value::Null) => {
            device.set_mpp_fast_dequant(true)?;
            true
        }
        Some(Value::Bool(enabled)) => {
            device.set_mpp_fast_dequant(*enabled)?;
            *enabled
        }
        Some(_) => {
            return Err(Error::Parameter(
                "request field mpp_fast_dequant must be a boolean".into(),
            ));
        }
    };
    let moe_expert_tile_pairs = match request.get("moe_expert_tile_pairs") {
        None | Some(Value::Null) => {
            device.set_moe_expert_tile_pairs(true)?;
            true
        }
        Some(Value::Bool(enabled)) => {
            device.set_moe_expert_tile_pairs(*enabled)?;
            *enabled
        }
        Some(_) => {
            return Err(Error::Parameter(
                "request field moe_expert_tile_pairs must be a boolean".into(),
            ));
        }
    };
    let moe_routing_temporary_mib = match request.get("moe_routing_temporary_mib") {
        None | Some(Value::Null) => 128,
        Some(value) => value
            .as_u64()
            .map(|v| v as usize)
            .filter(|v| *v > 0)
            .ok_or_else(|| {
                Error::Parameter("request field moe_routing_temporary_mib must be positive".into())
            })?,
    };
    device.set_moe_routing_temporary_limit(moe_routing_temporary_mib * 1024 * 1024)?;
    let attention_scores_vector = match request.get("attention_scores_vector") {
        None | Some(Value::Null) => {
            device.set_attention_scores_vector(true)?;
            true
        }
        Some(Value::Bool(enabled)) => {
            device.set_attention_scores_vector(*enabled)?;
            *enabled
        }
        Some(_) => {
            return Err(Error::Parameter(
                "request field attention_scores_vector must be a boolean".into(),
            ));
        }
    };
    let dense_mpp_tile_pairs = match request.get("dense_mpp_tile_pairs") {
        None | Some(Value::Null) => {
            device.set_dense_mpp_tile_pairs(true)?;
            true
        }
        Some(Value::Bool(enabled)) => {
            device.set_dense_mpp_tile_pairs(*enabled)?;
            *enabled
        }
        Some(_) => {
            return Err(Error::Parameter(
                "request field dense_mpp_tile_pairs must be a boolean".into(),
            ));
        }
    };
    let rope_table = match request.get("rope_table") {
        None | Some(Value::Null) => {
            device.set_rope_table(true)?;
            true
        }
        Some(Value::Bool(enabled)) => {
            device.set_rope_table(*enabled)?;
            *enabled
        }
        Some(_) => {
            return Err(Error::Parameter(
                "request field rope_table must be a boolean".into(),
            ));
        }
    };
    let arena_epoch_reuse = match request.get("arena_epoch_reuse") {
        None | Some(Value::Null) => {
            device.set_arena_epoch_reuse(true)?;
            true
        }
        Some(Value::Bool(enabled)) => {
            device.set_arena_epoch_reuse(*enabled)?;
            *enabled
        }
        Some(_) => {
            return Err(Error::Parameter(
                "request field arena_epoch_reuse must be a boolean".into(),
            ));
        }
    };
    let moe_expert_grouped = match request.get("moe_expert_grouped") {
        None | Some(Value::Null) => {
            device.set_moe_expert_grouped(true)?;
            true
        }
        Some(Value::Bool(enabled)) => {
            device.set_moe_expert_grouped(*enabled)?;
            *enabled
        }
        Some(_) => {
            return Err(Error::Parameter(
                "request field moe_expert_grouped must be a boolean".into(),
            ));
        }
    };
    let fuse_add_rmsnorm = match request.get("fuse_add_rmsnorm") {
        None | Some(Value::Null) => {
            device.set_fuse_add_rmsnorm(false)?;
            false
        }
        Some(Value::Bool(enabled)) => {
            device.set_fuse_add_rmsnorm(*enabled)?;
            *enabled
        }
        Some(_) => {
            return Err(Error::Parameter(
                "request field fuse_add_rmsnorm must be a boolean".into(),
            ));
        }
    };
    let fuse_rope_cache = match request.get("fuse_rope_cache") {
        None | Some(Value::Null) => {
            device.set_fuse_rope_cache(true)?;
            true
        }
        Some(Value::Bool(enabled)) => {
            device.set_fuse_rope_cache(*enabled)?;
            *enabled
        }
        Some(_) => {
            return Err(Error::Parameter(
                "request field fuse_rope_cache must be a boolean".into(),
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

    let device_argmax = match request.get("device_argmax") {
        None | Some(Value::Null) => true,
        Some(Value::Bool(enabled)) => *enabled,
        Some(_) => {
            return Err(Error::Parameter(
                "request field device_argmax must be a boolean".into(),
            ));
        }
    };

    let start = std::time::Instant::now();
    let result = if device_argmax {
        generation::generate_greedy(device, &model.model, &prompt, max_new_tokens, &[], |_| {
            Ok(())
        })?
    } else {
        generation::generate(
            device,
            &model.model,
            &prompt,
            max_new_tokens,
            &[],
            generation::argmax,
            |_| Ok(()),
        )?
    };
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
        "attention_softmax_prefix_reuse": attention_softmax_prefix_reuse,
        "gguf_mpp_min_rows": gguf_mpp_min_rows,
        "gguf_mpp_q5_1_min_rows": gguf_mpp_q5_1_min_rows,
        "moe_expert_tensorops": expert_tensorops,
        "moe_expert_tensorops_tile_k64": expert_tensorops_tile_k64,
        "moe_expert_tensorops_min_routes_per_expert": expert_tensorops_min_routes,
        "q8_0_mpp_tile_k64": q8_0_mpp_tile_k64,
        "q8_0_gemv_k_split": q8_0_gemv_k_split,
        "q5_1_mpp_tile_k64": q5_1_mpp_tile_k64,
        "q4_k_mpp_tile_k64": q4_k_mpp_tile_k64,
        "q4_k_mpp_tile_m128": q4_k_mpp_tile_m128,
        "q4_k_expert_project_8rows": q4_k_expert_project_8rows,
        "q4_k_expert_project_16rows": q4_k_expert_project_16rows,
        "q4_k_expert_project_16rows_pairs": q4_k_expert_project_16rows_pairs,
        "q6_k_expert_project_8rows": q6_k_expert_project_8rows,
        "q5_k_mpp_tile_k64": q5_k_mpp_tile_k64,
        "q5_0_gemv_n4": q5_0_gemv_n4,
        "q5_1_gemv_n4": q5_1_gemv_n4,
        "q5_k_gemv_8rows": q5_k_gemv_8rows,
        "moe_gpu_routing_prefill": moe_gpu_routing_prefill,
        "shared_encoder": shared_encoder,
        "q4_k_factored": q4_k_factored,
        "attention_scores_vector": attention_scores_vector,
        "q6_k_factored": q6_k_factored,
        "mpp_fast_dequant": mpp_fast_dequant,
        "dense_mpp_tile_pairs": dense_mpp_tile_pairs,
        "rope_table": rope_table,
        "fuse_add_rmsnorm": fuse_add_rmsnorm,
        "fuse_rope_cache": fuse_rope_cache,
        "moe_expert_grouped": moe_expert_grouped,
        "resident_weights": device.resident_weights(),
        "keep_alive": std::env::var("FERRUM_KEEP_ALIVE").as_deref() != Ok("0"),
        "arena_epoch_reuse": arena_epoch_reuse,
        "moe_expert_tile_pairs": moe_expert_tile_pairs,
        "moe_routing_temporary_mib": moe_routing_temporary_mib,
        "device_argmax": device_argmax,
        "moe_chunk_device_copy": moe_chunk_device_copy,
        "attention_context_decode_wide": attention_context_decode_wide,
        "concurrent_dispatch": concurrent_dispatch,
        "model_format": model.format,
        "model_repository": std::env::var("FERRUM_PHASE7A_MODEL_REPOSITORY")
            .unwrap_or_else(|_| model.repository.to_owned()),
        "model_revision": std::env::var("FERRUM_PHASE7A_MODEL_REVISION")
            .unwrap_or_else(|_| model.revision.to_owned()),
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
    phase7a_cpu_preflight()?;
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
    if let Ok(value) = std::env::var("FERRUM_RESIDENT_WEIGHTS") {
        let enabled = match value.as_str() {
            "1" | "true" => true,
            "0" | "false" => false,
            _ => return Err(Error::Parameter("invalid FERRUM_RESIDENT_WEIGHTS".into())),
        };
        device.set_resident_weights(enabled);
    }
    // MetalDevice::new starts the residency keep-alive; FERRUM_KEEP_ALIVE=0 stops it.
    if std::env::var("FERRUM_KEEP_ALIVE").as_deref() == Ok("0") {
        device.set_residency_keep_alive(None)?;
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
