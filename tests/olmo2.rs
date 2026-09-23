use ferrum::{
    MetalDevice,
    generation::{argmax, final_logits},
    model::{
        architecture::{ProjectionBias, QkNormLayout, ResidualTopology},
        olmo2::{self, Olmo2Config},
    },
};

#[test]
fn official_olmo2_config_selects_postnorm_and_projection_norm() {
    let source: Olmo2Config =
        serde_json::from_str(include_str!("fixtures/olmo2/config.json")).unwrap();
    let (config, policy) = source.convert().unwrap();
    assert_eq!(config.hidden_size, 2048);
    assert_eq!(config.num_attention_heads, 16);
    assert_eq!(config.num_key_value_heads, 16);
    assert_eq!(policy.qk_norm_layout, QkNormLayout::Projection);
    assert_eq!(policy.residual_topology, ResidualTopology::PostNorm);
    assert_eq!(policy.qkv_bias, ProjectionBias::Forbidden);

    let mut invalid: serde_json::Value =
        serde_json::from_str(include_str!("fixtures/olmo2/config.json")).unwrap();
    invalid["rope_scaling"] = serde_json::json!({"rope_type":"dynamic"});
    let invalid: Olmo2Config = serde_json::from_value(invalid).unwrap();
    assert!(invalid.convert().is_err());
}

#[test]
#[ignore = "requires official OLMo 2 1B checkpoint; set FERRUM_OLMO2_MODEL"]
fn official_olmo2_matches_reference_and_replays_state() {
    let dir = std::path::PathBuf::from(
        std::env::var_os("FERRUM_OLMO2_MODEL").expect("official model directory"),
    );
    let reference: serde_json::Value = serde_json::from_str(include_str!(
        "../docs/measurements/phase6/olmo2-1b-reference-hello.json"
    ))
    .unwrap();
    let ids: Vec<u32> = serde_json::from_value(reference["prompt_ids"].clone()).unwrap();
    let device = MetalDevice::new().unwrap();
    let loaded = olmo2::load(&device, &dir).unwrap();
    assert_eq!(loaded.tensor_count, 179);
    assert_eq!(loaded.tokenizer.encode("Hello!").unwrap(), ids);
    let (mut logits, mut cache) = loaded.model.forward_prefill_last(&device, &ids).unwrap();
    let original = cache.clone();
    let mut generated = Vec::new();
    let mut maximum = 0.0f32;
    for row in reference["rows"].as_array().unwrap() {
        let values = final_logits(&device, &logits).unwrap();
        let token = argmax(&values).unwrap();
        assert_eq!(token as u64, row["token"].as_u64().unwrap());
        for item in row["top10"].as_array().unwrap() {
            let id = item[0].as_u64().unwrap() as usize;
            let difference = (values[id] - item[1].as_f64().unwrap() as f32).abs();
            maximum = maximum.max(difference);
            assert!(
                difference <= 0.1,
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
    println!("OLMo 2 max top-10 logit difference: {maximum}");
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

    let long: serde_json::Value = serde_json::from_str(include_str!(
        "../docs/measurements/phase6/olmo2-1b-reference-long.json"
    ))
    .unwrap();
    let ids: Vec<u32> = serde_json::from_value(long["prompt_ids"].clone()).unwrap();
    assert_eq!(loaded.tokenizer.encode(&"Hello! ".repeat(12)).unwrap(), ids);
    let (mut logits, mut cache) = loaded.model.forward_prefill_last(&device, &ids).unwrap();
    for row in long["rows"].as_array().unwrap() {
        let values = final_logits(&device, &logits).unwrap();
        let token = argmax(&values).unwrap();
        assert_eq!(token as u64, row["token"].as_u64().unwrap());
        for item in row["top10"].as_array().unwrap() {
            let id = item[0].as_u64().unwrap() as usize;
            assert!((values[id] - item[1].as_f64().unwrap() as f32).abs() <= 0.1);
        }
        if row["step"].as_u64().unwrap() + 1 < long["rows"].as_array().unwrap().len() as u64 {
            logits = loaded
                .model
                .forward_decode(&device, token, &mut cache)
                .unwrap();
        }
    }
}
