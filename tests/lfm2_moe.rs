use ferrum::model::{
    architecture::{LayerFeedForwardPolicy, LayerOperatorPolicy, MoeScoringFunction},
    lfm2_moe::{self, Lfm2MoeConfig},
};
use ferrum::{
    MetalDevice,
    generation::{argmax, final_logits},
};

fn official_config() -> Lfm2MoeConfig {
    serde_json::from_str(include_str!("fixtures/lfm2_moe/config.json")).unwrap()
}

#[test]
fn official_lfm2_moe_config_composes_hybrid_layers_and_routed_experts() {
    let (config, policy) = official_config().convert().unwrap();
    assert_eq!(config.vocab_size, 128_000);
    assert_eq!(config.hidden_size, 2_048);
    assert_eq!(config.intermediate_size, 7_168);
    assert_eq!(config.num_layers, 24);
    assert_eq!(config.num_attention_heads, 32);
    assert_eq!(config.num_key_value_heads, 8);
    assert_eq!(config.max_context_length, 32_768);
    assert_eq!(policy.layer_policies.as_ref().unwrap().len(), 24);
    assert_eq!(
        policy
            .layer_policies
            .as_ref()
            .unwrap()
            .iter()
            .filter(|layer| matches!(layer.operator, LayerOperatorPolicy::ShortConv { .. }))
            .count(),
        18
    );
    assert_eq!(
        policy
            .layer_policies
            .as_ref()
            .unwrap()
            .iter()
            .filter(|layer| matches!(layer.operator, LayerOperatorPolicy::Attention))
            .count(),
        6
    );
    for (index, layer) in policy.layer_policies.as_ref().unwrap().iter().enumerate() {
        if index < 2 {
            assert!(matches!(
                layer.feed_forward,
                LayerFeedForwardPolicy::Dense {
                    intermediate_size: 7_168
                }
            ));
        } else {
            match layer.feed_forward {
                LayerFeedForwardPolicy::Sparse {
                    routing,
                    intermediate_size,
                } => {
                    assert_eq!(routing.experts, 32);
                    assert_eq!(routing.top_k, 4);
                    assert_eq!(routing.scoring_function, MoeScoringFunction::Sigmoid);
                    assert!(routing.normalize_top_k_prob);
                    assert_eq!(routing.normalization_epsilon, 1e-6);
                    assert_eq!(routing.routed_scaling_factor, 1.);
                    assert_eq!(intermediate_size, 1_792);
                }
                LayerFeedForwardPolicy::Dense { .. } => panic!("expected sparse layer {index}"),
            }
        }
    }
}

#[test]
fn unsupported_lfm2_moe_config_semantics_are_rejected() {
    for (path, replacement) in [
        ("model_type", serde_json::json!("lfm2")),
        ("conv_bias", serde_json::json!(true)),
        ("tie_word_embeddings", serde_json::json!(false)),
        ("num_experts_per_tok", serde_json::json!(33)),
    ] {
        let mut value: serde_json::Value =
            serde_json::from_str(include_str!("fixtures/lfm2_moe/config.json")).unwrap();
        value[path] = replacement;
        let config: Lfm2MoeConfig = serde_json::from_value(value).unwrap();
        assert!(config.convert().is_err(), "{path}");
    }

    let mut value: serde_json::Value =
        serde_json::from_str(include_str!("fixtures/lfm2_moe/config.json")).unwrap();
    value["layer_types"][2] = "linear_attention".into();
    let config: Lfm2MoeConfig = serde_json::from_value(value).unwrap();
    assert!(config.convert().is_err());
}

#[test]
#[ignore = "requires official LFM2.5-8B-A1B checkpoint; set FERRUM_LFM2_MOE_MODEL"]
fn official_lfm2_moe_matches_reference_and_replays_hybrid_state() {
    let dir = std::path::PathBuf::from(
        std::env::var_os("FERRUM_LFM2_MOE_MODEL").expect("official model directory"),
    );
    let reference: serde_json::Value = serde_json::from_str(include_str!(
        "../docs/measurements/phase6/lfm2-8b-a1b-reference-hello.json"
    ))
    .unwrap();
    let ids: Vec<u32> = serde_json::from_value(reference["prompt_ids"].clone()).unwrap();
    let device = MetalDevice::new().unwrap();
    let loaded = lfm2_moe::load(&device, &dir).unwrap();
    assert_eq!(loaded.tensor_count, 2_302);
    assert_eq!(loaded.source_tensor_bytes, 16_935_715_072);
    assert_eq!(loaded.parameter_count, 8_467_856_128);
    assert_eq!(loaded.tokenizer.vocab_size(), 125_017);
    assert_eq!(
        loaded
            .tokenizer
            .encode("<|startoftext|><|im_start|>user\nHello!<|im_end|>\n<|im_start|>assistant\n")
            .unwrap(),
        ids
    );

    let (mut logits, mut cache) = loaded.model.forward_prefill_last(&device, &ids).unwrap();
    let original = cache.clone();
    assert_eq!(cache.len().unwrap(), ids.len());
    assert_eq!(cache.state_bytes(), 221_184);
    let rows = reference["rows"].as_array().unwrap();
    let mut generated = Vec::with_capacity(rows.len());
    let mut maximum = 0.0f32;
    for row in rows {
        let values = final_logits(&device, &logits).unwrap();
        let token = argmax(&values).unwrap();
        assert_eq!(token as u64, row["token"].as_u64().unwrap());
        for item in row["top10"].as_array().unwrap() {
            let id = item[0].as_u64().unwrap() as usize;
            let difference = (values[id] - item[1].as_f64().unwrap() as f32).abs();
            maximum = maximum.max(difference);
            assert!(
                difference <= 3.25,
                "step {}, ID {id}, difference {difference}",
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
    assert_eq!(serde_json::json!(generated), reference["generated_ids"]);
    println!("LFM2-MoE max top-10 logit difference: {maximum}");
    assert_eq!(cache.len().unwrap(), ids.len() + generated.len() - 1);
    let stats = loaded.model.take_moe_stats();
    assert_eq!(
        stats.assignments,
        (ids.len() + generated.len() - 1) * 22 * 4
    );
    // GPU routing keeps selections on Metal and does not count distinct experts.
    assert!(!stats.active_experts_known || stats.active_experts > 0);
    assert!(stats.peak_temporary_bytes < 16 * 1024 * 1024);

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
        .forward(&device, &ids, &mut reset_cache, None)
        .unwrap();
    assert_eq!(
        argmax(&final_logits(&device, &reset_logits).unwrap()).unwrap(),
        generated[0]
    );
}
