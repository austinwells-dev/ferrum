use ferrum::model::{
    architecture::{LayerFeedForwardPolicy, LayerOperatorPolicy},
    lfm2::Lfm2Config,
};
use ferrum::{
    MetalDevice,
    generation::{argmax, final_logits},
};

fn official_config() -> Lfm2Config {
    serde_json::from_str(include_str!("fixtures/lfm2/config.json")).unwrap()
}

#[test]
fn official_lfm2_config_selects_hybrid_layer_schedule() {
    let (config, policy) = official_config().convert().unwrap();
    assert_eq!(config.vocab_size, 65_536);
    assert_eq!(config.hidden_size, 1_024);
    assert_eq!(config.intermediate_size, 2_560);
    assert_eq!(config.num_layers, 14);
    assert_eq!(config.num_attention_heads, 16);
    assert_eq!(config.num_key_value_heads, 8);
    assert_eq!(config.max_context_length, 32_768);
    assert_eq!(policy.qk_norm_epsilon, Some(1e-5));
    let operators: Vec<_> = (0..config.num_layers)
        .map(|index| policy.layer(index, config.intermediate_size).unwrap())
        .collect();
    assert_eq!(
        operators
            .iter()
            .filter(|layer| matches!(layer.operator, LayerOperatorPolicy::ShortConv { .. }))
            .count(),
        8
    );
    assert_eq!(
        operators
            .iter()
            .filter(|layer| matches!(layer.operator, LayerOperatorPolicy::Attention))
            .count(),
        6
    );
    assert!(operators.iter().all(|layer| matches!(
        layer.feed_forward,
        LayerFeedForwardPolicy::Dense {
            intermediate_size: 2_560
        }
    )));
}

#[test]
fn unsupported_lfm2_semantics_are_rejected() {
    for (path, replacement) in [
        ("model_type", serde_json::json!("lfm2_moe")),
        ("conv_bias", serde_json::json!(true)),
        ("use_pos_enc", serde_json::json!(false)),
        ("tie_word_embeddings", serde_json::json!(false)),
    ] {
        let mut value: serde_json::Value =
            serde_json::from_str(include_str!("fixtures/lfm2/config.json")).unwrap();
        value[path] = replacement;
        let config: Lfm2Config = serde_json::from_value(value).unwrap();
        assert!(config.convert().is_err(), "{path}");
    }

    let mut value: serde_json::Value =
        serde_json::from_str(include_str!("fixtures/lfm2/config.json")).unwrap();
    value["layer_types"][2] = "linear_attention".into();
    let config: Lfm2Config = serde_json::from_value(value).unwrap();
    assert!(config.convert().is_err());
}

#[test]
#[ignore = "requires official LFM2.5-230M checkpoint; set FERRUM_LFM2_MODEL"]
fn official_lfm2_matches_reference_and_replays_hybrid_state() {
    let dir = std::path::PathBuf::from(
        std::env::var_os("FERRUM_LFM2_MODEL").expect("official model directory"),
    );
    let reference: serde_json::Value = serde_json::from_str(include_str!(
        "../docs/measurements/phase6/lfm2-230m-reference-hello.json"
    ))
    .unwrap();
    let ids: Vec<u32> = serde_json::from_value(reference["prompt_ids"].clone()).unwrap();
    let device = MetalDevice::new().unwrap();
    let loaded = ferrum::model::lfm2::load(&device, &dir).unwrap();
    assert_eq!(loaded.tensor_count, 132);
    assert_eq!(loaded.parameter_count, 229_693_184);
    assert_eq!(loaded.tokenizer.vocab_size(), 64_402);
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
    assert_eq!(cache.state_bytes(), 49_152);
    let mut generated = Vec::new();
    let mut maximum = 0.0f32;
    for row in reference["rows"].as_array().unwrap() {
        let values = final_logits(&device, &logits).unwrap();
        let token = argmax(&values).unwrap();
        if token as u64 != row["token"].as_u64().unwrap() {
            let mut top = values.iter().copied().enumerate().collect::<Vec<_>>();
            top.sort_by(|a, b| b.1.total_cmp(&a.1));
            panic!(
                "step {} greedy mismatch: Ferrum {token}, reference {}; Ferrum top five {:?}",
                row["step"],
                row["token"],
                &top[..5]
            );
        }
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
    println!("LFM2 max top-10 logit difference: {maximum}");

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

    let long: serde_json::Value = serde_json::from_str(include_str!(
        "../docs/measurements/phase6/lfm2-230m-reference-long.json"
    ))
    .unwrap();
    let long_ids: Vec<u32> = serde_json::from_value(long["prompt_ids"].clone()).unwrap();
    assert_eq!(long_ids.len(), 138);
    assert_eq!(
        loaded
            .tokenizer
            .encode(&format!(
                "<|startoftext|><|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n",
                "Hello! ".repeat(64)
            ))
            .unwrap(),
        long_ids
    );
    let (mut logits, mut cache) = loaded
        .model
        .forward_prefill_last(&device, &long_ids)
        .unwrap();
    assert_eq!(cache.len().unwrap(), long_ids.len());
    assert_eq!(cache.state_bytes(), 49_152);
    for row in long["rows"].as_array().unwrap() {
        let values = final_logits(&device, &logits).unwrap();
        let token = argmax(&values).unwrap();
        assert_eq!(token as u64, row["token"].as_u64().unwrap());
        for item in row["top10"].as_array().unwrap() {
            let id = item[0].as_u64().unwrap() as usize;
            assert!(
                (values[id] - item[1].as_f64().unwrap() as f32).abs() <= 1.0,
                "long prompt step {}, ID {id}",
                row["step"]
            );
        }
        if row["step"].as_u64().unwrap() + 1 < long["rows"].as_array().unwrap().len() as u64 {
            logits = loaded
                .model
                .forward_decode(&device, token, &mut cache)
                .unwrap();
        }
    }
}
