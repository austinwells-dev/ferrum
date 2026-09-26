use ferrum::{
    MetalDevice,
    generation::{argmax, final_logits},
    model::{
        architecture::MoeRoutingPolicy,
        granite_moe::{self, GraniteMoeConfig},
    },
};

/// One BF16 step at `x`'s magnitude: the resolution of reference logits.
fn bf16_step(x: f32) -> f32 {
    2f32.powf(x.abs().log2().floor() - 7.)
}

/// Greedy parity that tolerates only exact BF16 ties. Parallel RMSNorm sums may
/// round a row differently from the reference; a different token is accepted
/// only when the reference's own top two are within two BF16 steps (one step
/// of rounding on each reference logit) and Ferrum
/// scores the reference token within one step of its choice. The reference
/// token is then followed so later steps stay checked.
fn reference_token(values: &[f32], row: &serde_json::Value) -> u32 {
    let expected = row["token"].as_u64().unwrap() as usize;
    let token = argmax(values).unwrap() as usize;
    if token != expected {
        let top = row["top10"].as_array().unwrap();
        let (first, second) = (
            top[0][1].as_f64().unwrap() as f32,
            top[1][1].as_f64().unwrap() as f32,
        );
        assert!(
            first - second <= 2. * bf16_step(first)
                && values[token] - values[expected] <= bf16_step(values[token]),
            "step {}: token {token} ({}) instead of {expected} ({}); reference margin {}",
            row["step"],
            values[token],
            values[expected],
            first - second
        );
    }
    expected as u32
}

#[test]
fn official_granite_moe_config_selects_shared_sparse_policy() {
    let source: GraniteMoeConfig =
        serde_json::from_str(include_str!("fixtures/granite_moe/config.json")).unwrap();
    let (config, policy) = source.convert().unwrap();
    assert_eq!(config.hidden_size, 1024);
    assert_eq!(config.intermediate_size, 512);
    assert_eq!(config.num_attention_heads, 16);
    assert_eq!(config.num_key_value_heads, 8);
    assert_eq!(policy.attention_scale, Some(0.015625));
    assert_eq!(policy.embedding_multiplier, 12.);
    assert_eq!(policy.residual_multiplier, 0.22);
    assert_eq!(policy.logits_divisor, 6.);
    assert_eq!(policy.moe, Some(MoeRoutingPolicy::softmax(32, 8)));

    for (key, value) in [
        ("model_type", serde_json::json!("granite")),
        ("num_local_experts", serde_json::json!(0)),
        ("num_experts_per_tok", serde_json::json!(33)),
        ("hidden_act", serde_json::json!("gelu")),
        ("attention_bias", serde_json::json!(true)),
        ("tie_word_embeddings", serde_json::json!(false)),
        ("new_semantics", serde_json::json!(true)),
    ] {
        let mut value_json: serde_json::Value =
            serde_json::from_str(include_str!("fixtures/granite_moe/config.json")).unwrap();
        value_json[key] = value;
        let invalid: GraniteMoeConfig = serde_json::from_value(value_json).unwrap();
        assert!(invalid.convert().is_err(), "{key}");
    }
}

#[test]
#[ignore = "requires official Granite 3.1 1B-A400M checkpoint; set FERRUM_GRANITE_MOE_MODEL"]
fn official_granite_moe_matches_reference_and_replays_cache_state() {
    let dir = std::path::PathBuf::from(
        std::env::var_os("FERRUM_GRANITE_MOE_MODEL").expect("official model directory"),
    );
    let device = MetalDevice::new().unwrap();
    let loaded = granite_moe::load(&device, &dir).unwrap();
    assert_eq!(loaded.tensor_count, 218);
    assert_eq!(loaded.parameter_count, 1_334_628_352);

    let reference: serde_json::Value = serde_json::from_str(include_str!(
        "../docs/measurements/phase6/granite-3.1-1b-a400m-reference-hello.json"
    ))
    .unwrap();
    let ids: Vec<u32> = serde_json::from_value(reference["prompt_ids"].clone()).unwrap();
    assert_eq!(loaded.tokenizer.encode("Hello!").unwrap(), ids);
    let (mut logits, mut cache) = loaded.model.forward_prefill_last(&device, &ids).unwrap();
    let original = cache.clone();
    let mut generated = Vec::new();
    let mut maximum = 0.0f32;
    for row in reference["rows"].as_array().unwrap() {
        let values = final_logits(&device, &logits).unwrap();
        let token = reference_token(&values, row);
        for item in row["top10"].as_array().unwrap() {
            let id = item[0].as_u64().unwrap() as usize;
            let difference = (values[id] - item[1].as_f64().unwrap() as f32).abs();
            maximum = maximum.max(difference);
            assert!(
                difference <= 1.0,
                "step {}, ID {id}, difference {difference}",
                row["step"]
            );
        }
        generated.push(token);
        if generated.len() < reference["rows"].as_array().unwrap().len() {
            logits = loaded
                .model
                .forward_decode(&device, token, &mut cache)
                .unwrap();
        }
    }
    assert_eq!(serde_json::json!(generated), reference["generated_ids"]);
    println!("Granite MoE max top-10 logit difference: {maximum}");
    let stats = loaded.model.take_moe_stats();
    assert_eq!(
        stats.assignments,
        (ids.len() + generated.len() - 1) * 24 * 8
    );
    // GPU routing keeps selections on Metal and does not count distinct experts.
    assert!(!stats.active_experts_known || stats.active_experts > 0);
    assert!(stats.peak_temporary_bytes < 16 * 1024 * 1024);

    let mut branch_a = original.clone();
    let mut branch_b = original;
    assert_eq!(
        loaded
            .model
            .forward_decode(&device, generated[0], &mut branch_a)
            .unwrap()
            .to_f32(),
        loaded
            .model
            .forward_decode(&device, generated[0], &mut branch_b)
            .unwrap()
            .to_f32(),
    );
    cache.reset();
    let again = loaded
        .model
        .forward(&device, &ids, &mut cache, None)
        .unwrap();
    assert_eq!(
        argmax(&final_logits(&device, &again).unwrap()).unwrap(),
        generated[0]
    );
    let _ = loaded.model.take_moe_stats();

    let long: serde_json::Value = serde_json::from_str(include_str!(
        "../docs/measurements/phase6/granite-3.1-1b-a400m-reference-long.json"
    ))
    .unwrap();
    let text = long["prompt_text"].as_str().unwrap();
    let repeat = long["repeat"].as_u64().unwrap() as usize;
    let ids: Vec<u32> = serde_json::from_value(long["prompt_ids"].clone()).unwrap();
    assert!(
        ids.len() > 256,
        "the metadata test prompt must exceed 256 tokens"
    );
    assert_eq!(loaded.tokenizer.encode(&text.repeat(repeat)).unwrap(), ids);
    let (logits, _) = loaded.model.forward_prefill_last(&device, &ids).unwrap();
    let values = final_logits(&device, &logits).unwrap();
    assert_eq!(
        argmax(&values).unwrap() as u64,
        long["rows"][0]["token"].as_u64().unwrap()
    );
    for item in long["rows"][0]["top10"].as_array().unwrap() {
        let id = item[0].as_u64().unwrap() as usize;
        assert!((values[id] - item[1].as_f64().unwrap() as f32).abs() <= 1.0);
    }
    let stats = loaded.model.take_moe_stats();
    assert_eq!(stats.assignments, ids.len() * 24 * 8);
    assert!(stats.peak_temporary_bytes < 16 * 1024 * 1024);
}
