use ferrum::{
    MetalDevice,
    generation::{argmax, final_logits},
    model::lfm2_moe,
};

#[test]
#[ignore = "requires the pinned official Q4_K_M GGUF and matching Transformers metadata directory"]
fn official_lfm2_moe_q4_gguf_matches_llama_cpp_and_replays_hybrid_state() {
    let gguf = std::path::PathBuf::from(
        std::env::var_os("FERRUM_LFM2_MOE_GGUF").expect("official GGUF file"),
    );
    let metadata_dir = std::path::PathBuf::from(
        std::env::var_os("FERRUM_LFM2_MOE_MODEL").expect("official model metadata directory"),
    );
    let reference: serde_json::Value = serde_json::from_str(include_str!(
        "../docs/measurements/phase6/lfm2-8b-a1b-q4-k-m-reference-hello.json"
    ))
    .unwrap();
    let prompt_ids: Vec<u32> = serde_json::from_value(reference["prompt_ids"].clone()).unwrap();
    let expected_ids: Vec<u32> =
        serde_json::from_value(reference["generated_ids"].clone()).unwrap();

    let device = MetalDevice::new().unwrap();
    let loaded = lfm2_moe::load_gguf(&device, gguf, metadata_dir).unwrap();
    println!(
        "LFM2-MoE Q4_K_M tensors={} parameters={} source_tensor_bytes={} quantized_tensor_bytes={} config_tokenizer_load={:?} weight_load={:?} construction={:?}",
        loaded.tensor_count,
        loaded.parameter_count,
        loaded.source_tensor_bytes,
        loaded.quantized_tensor_bytes,
        loaded.config_tokenizer_load,
        loaded.weight_load,
        loaded.construction
    );
    assert_eq!(loaded.tensor_count, 256);
    assert_eq!(loaded.source_tensor_bytes, 5_147_326_208);
    assert_eq!(loaded.parameter_count, 8_467_856_128);
    assert!(loaded.quantized_tensor_bytes > 4_500_000_000);
    assert!(loaded.quantized_tensor_bytes <= loaded.source_tensor_bytes);
    assert_eq!(loaded.tokenizer.vocab_size(), 125_017);
    assert_eq!(
        loaded
            .tokenizer
            .encode("<|startoftext|><|im_start|>user\nHello!<|im_end|>\n<|im_start|>assistant\n")
            .unwrap(),
        prompt_ids
    );

    let (mut logits, mut cache) = loaded
        .model
        .forward_prefill_last(&device, &prompt_ids)
        .unwrap();
    let original = cache.clone();
    assert_eq!(cache.len().unwrap(), prompt_ids.len());
    assert_eq!(cache.state_bytes(), 221_184);
    let rows = reference["rows"].as_array().unwrap();
    let mut generated = Vec::with_capacity(rows.len());
    let mut maximum_delta_difference = 0.0f32;
    for row in rows {
        let values = final_logits(&device, &logits).unwrap();
        let token = argmax(&values).unwrap();
        assert_eq!(token as u64, row["token"].as_u64().unwrap());
        for (rank, item) in row["top10"].as_array().unwrap().iter().enumerate() {
            let id = item[0].as_u64().unwrap() as usize;
            let ferrum_delta = values[id] - values[token as usize];
            let reference_delta =
                item[1].as_f64().unwrap() as f32 - row["top10"][0][1].as_f64().unwrap() as f32;
            let difference = (ferrum_delta - reference_delta).abs();
            maximum_delta_difference = maximum_delta_difference.max(difference);
            assert!(
                difference <= 4.0,
                "step {}, rank {rank}, ID {id}: Ferrum delta={ferrum_delta}, llama.cpp delta={reference_delta}",
                row["step"]
            );
        }
        generated.push(token);
        if generated.len() < rows.len() {
            logits = loaded
                .model
                .forward_decode(&device, token, &mut cache)
                .unwrap();
        }
    }
    assert_eq!(generated, expected_ids);
    println!("LFM2-MoE Q4_K_M max top-10 logit-delta difference: {maximum_delta_difference}");
    assert_eq!(cache.len().unwrap(), prompt_ids.len() + generated.len() - 1);
    let stats = loaded.model.take_moe_stats();
    assert_eq!(
        stats.assignments,
        (prompt_ids.len() + generated.len() - 1) * 22 * 4
    );
    // GPU routing keeps selections on Metal and does not count distinct experts.
    assert!(!stats.active_experts_known || stats.active_experts > 0);
    assert!(stats.peak_temporary_bytes < 16 * 1024 * 1024);
    println!(
        "LFM2-MoE Q4_K_M assignments={} active_expert_visits={} peak_routing_temporary_bytes={}",
        stats.assignments, stats.active_experts, stats.peak_temporary_bytes
    );

    let mut branch_a = original.clone();
    let mut branch_b = original;
    let branch_logits_a = loaded
        .model
        .forward_decode(&device, generated[0], &mut branch_a)
        .unwrap();
    let branch_logits_b = loaded
        .model
        .forward_decode(&device, generated[0], &mut branch_b)
        .unwrap();
    assert_eq!(branch_logits_a.to_f32(), branch_logits_b.to_f32());
    assert_eq!(branch_a.state_bytes(), branch_b.state_bytes());

    let mut reset_cache = loaded.model.new_cache().unwrap();
    let reset_logits = loaded
        .model
        .forward(&device, &prompt_ids, &mut reset_cache, None)
        .unwrap();
    assert_eq!(
        argmax(&final_logits(&device, &reset_logits).unwrap()).unwrap(),
        generated[0]
    );
}

#[test]
#[ignore = "requires the pinned official Q4_K_M GGUF and matching Transformers metadata directory"]
fn official_lfm2_moe_q4_gguf_profiles_generation() {
    let gguf = std::path::PathBuf::from(
        std::env::var_os("FERRUM_LFM2_MOE_GGUF").expect("official GGUF file"),
    );
    let metadata_dir = std::path::PathBuf::from(
        std::env::var_os("FERRUM_LFM2_MOE_MODEL").expect("official model metadata directory"),
    );
    let reference: serde_json::Value = serde_json::from_str(include_str!(
        "../docs/measurements/phase6/lfm2-8b-a1b-q4-k-m-reference-hello.json"
    ))
    .unwrap();
    let prompt_ids: Vec<u32> = serde_json::from_value(reference["prompt_ids"].clone()).unwrap();
    let expected_ids: Vec<u32> =
        serde_json::from_value(reference["generated_ids"].clone()).unwrap();

    let device = MetalDevice::new().unwrap();
    let loaded = lfm2_moe::load_gguf(&device, gguf, metadata_dir).unwrap();
    let retained_model_bytes = loaded.model.weight_bytes();
    device.set_profiling(true);
    let generation_started = std::time::Instant::now();
    let generation = ferrum::generation::generate(
        &device,
        &loaded.model,
        &prompt_ids,
        expected_ids.len(),
        &loaded.eos_ids,
        argmax,
        |_| Ok(()),
    )
    .unwrap();
    let complete_generation_wall = generation_started.elapsed();
    device.set_profiling(false);
    assert_eq!(generation.tokens, expected_ids);

    let mut decode_times = generation.decode.clone();
    decode_times.sort();
    let median_decode = decode_times[decode_times.len() / 2];
    let transient_peak_bytes = generation
        .decode_counters
        .iter()
        .map(|counters| counters.transient_peak_bytes)
        .chain([generation.prefill_counters.transient_peak_bytes])
        .max()
        .unwrap_or_default();
    let stats = loaded.model.take_moe_stats();
    let profile = serde_json::json!({
        "model": "LiquidAI/LFM2.5-8B-A1B-GGUF",
        "revision": "49c14831707011e64d70b2ebd8462ba08d608434",
        "file": "LFM2.5-8B-A1B-Q4_K_M.gguf",
        "runtime": "Ferrum release profile; no warmup; Apple M5",
        "prompt_tokens": prompt_ids.len(),
        "generated_ids": generation.tokens,
        "config_tokenizer_load_ms": loaded.config_tokenizer_load.as_secs_f64() * 1e3,
        "weight_load_ms": loaded.weight_load.as_secs_f64() * 1e3,
        "construction_ms": loaded.construction.as_secs_f64() * 1e3,
        "source_tensor_bytes": loaded.source_tensor_bytes,
        "retained_quantized_tensor_bytes": loaded.quantized_tensor_bytes,
        "retained_model_bytes": retained_model_bytes,
        "prefill_ms": generation.prefill.as_secs_f64() * 1e3,
        "prefill_tokens_per_second": prompt_ids.len() as f64 / generation.prefill.as_secs_f64(),
        "first_token_ms": generation.first_token.as_secs_f64() * 1e3,
        "cached_decode_ms": generation.decode.iter().map(|time| time.as_secs_f64() * 1e3).collect::<Vec<_>>(),
        "cached_decode_median_ms": median_decode.as_secs_f64() * 1e3,
        "cached_decode_tokens_per_second": 1.0 / median_decode.as_secs_f64(),
        "complete_generation_wall_ms": complete_generation_wall.as_secs_f64() * 1e3,
        "complete_generation_tokens_per_second": generation.tokens.len() as f64 / complete_generation_wall.as_secs_f64(),
        "prompt_plus_generation_tokens_per_second": (prompt_ids.len() + generation.tokens.len()) as f64 / complete_generation_wall.as_secs_f64(),
        "active_kv_bytes": generation.kv_bytes,
        "reserved_kv_bytes": generation.kv_reserved_bytes,
        "hybrid_state_bytes": generation.state_bytes,
        "transient_peak_bytes": transient_peak_bytes,
        "moe": {
            "assignments": stats.assignments,
            "active_expert_visits": stats.active_experts,
            "peak_temporary_bytes": stats.peak_temporary_bytes,
            "router_projection_enqueue_ms": stats.router_projection_enqueue.as_secs_f64() * 1e3,
            "route_selection_wall_ms": stats.routing.as_secs_f64() * 1e3,
            "routing_boundary_wait_ms": stats.routing_boundary_wait.as_secs_f64() * 1e3,
            "expert_dispatch_enqueue_ms": stats.expert_dispatch.as_secs_f64() * 1e3,
            "combine_enqueue_ms": stats.combine_dispatch.as_secs_f64() * 1e3
        }
    });
    println!("Q4PROFILE {profile}");
    println!(
        "PROFILE prefill: {}",
        serde_json::to_string(&generation.prefill_profile).unwrap()
    );
    for (step, profile) in generation.decode_profiles.iter().enumerate() {
        println!(
            "PROFILE decode {step}: {}",
            serde_json::to_string(profile).unwrap()
        );
    }
}
