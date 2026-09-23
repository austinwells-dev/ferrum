use ferrum::{
    MetalDevice,
    generation::{argmax, final_logits},
    model::{
        architecture::ProjectionBias,
        qwen3::{self, Qwen3Config},
    },
};

#[test]
fn official_qwen3_config_selects_head_norm_and_config_driven_shapes() {
    let config: Qwen3Config =
        serde_json::from_str(include_str!("fixtures/qwen3/config.json")).unwrap();
    let (c, policy) = config.convert().unwrap();
    assert_eq!(c.hidden_size, 1024);
    assert_eq!(c.num_attention_heads * c.head_dim, 2048);
    assert_eq!(policy.qk_norm_epsilon, Some(1e-6));
    assert_eq!(policy.qkv_bias, ProjectionBias::Forbidden);
    let specs = qwen3::specifications(&c, &policy);
    assert_eq!(specs.len(), 310);
    assert!(specs.iter().any(|(source, _, shape)| {
        source == "model.layers.0.self_attn.q_proj.weight" && shape == &[2048, 1024]
    }));
    assert!(specs.iter().any(|(source, _, shape)| {
        source == "model.layers.0.self_attn.q_norm.weight" && shape == &[128]
    }));
    assert!(
        !specs
            .iter()
            .any(|(source, _, _)| source == "lm_head.weight")
    );

    let mut invalid: serde_json::Value =
        serde_json::from_str(include_str!("fixtures/qwen3/config.json")).unwrap();
    invalid["rope_scaling"] = serde_json::json!({"rope_type": "dynamic"});
    let invalid: Qwen3Config = serde_json::from_value(invalid).unwrap();
    assert!(invalid.convert().is_err());
}

#[test]
#[ignore = "requires official local Qwen3-0.6B checkpoint; set FERRUM_QWEN3_MODEL"]
fn official_qwen3_logits_generation_and_state_replay() {
    let dir =
        std::path::PathBuf::from(std::env::var_os("FERRUM_QWEN3_MODEL").expect("model directory"));
    let device = MetalDevice::new().unwrap();
    let loaded = qwen3::load(&device, &dir).unwrap();
    assert_eq!(loaded.tensor_count, 311);
    assert_eq!(loaded.config.vocab_size, 151936);
    assert!(loaded.model.weight_bytes() < loaded.source_tensor_bytes);
    let reference: serde_json::Value = serde_json::from_str(include_str!(
        "../docs/measurements/phase6/qwen3-0.6b-reference-hello.json"
    ))
    .unwrap();
    let ids: Vec<u32> = serde_json::from_value(reference["prompt_ids"].clone()).unwrap();
    let (formatted, actual_ids) = loaded
        .tokenizer
        .encode_prompt("Hello!", "You are a helpful assistant.", false)
        .unwrap();
    assert!(formatted.ends_with("<|im_start|>assistant\n"));
    assert_eq!(actual_ids, ids);

    let (mut logits, mut state) = loaded.model.forward_prefill_last(&device, &ids).unwrap();
    let prefill_state = state.clone();
    let mut generated = Vec::new();
    let mut max_selected_difference = 0.0f32;
    for row in reference["rows"].as_array().unwrap() {
        let values = final_logits(&device, &logits).unwrap();
        assert_eq!(values.len(), loaded.config.vocab_size);
        let token = argmax(&values).unwrap();
        assert_eq!(token as u64, row["token"].as_u64().unwrap());
        for item in row["top10"].as_array().unwrap() {
            let id = item[0].as_u64().unwrap() as usize;
            let difference = (values[id] - item[1].as_f64().unwrap() as f32).abs();
            max_selected_difference = max_selected_difference.max(difference);
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
                .forward_decode(&device, token, &mut state)
                .unwrap();
        }
    }
    assert_eq!(serde_json::json!(generated), reference["generated_ids"]);
    println!("Qwen3 max selected top-10 logit difference: {max_selected_difference}");

    // Replaying the published prefill state must reproduce the first decode step.
    let mut branch_a = prefill_state.clone();
    let mut branch_b = prefill_state;
    let a = loaded
        .model
        .forward_decode(&device, generated[0], &mut branch_a)
        .unwrap();
    let b = loaded
        .model
        .forward_decode(&device, generated[0], &mut branch_b)
        .unwrap();
    assert_eq!(a.to_f32(), b.to_f32());
    state.reset();
    let reset = loaded
        .model
        .forward(&device, &ids, &mut state, None)
        .unwrap();
    assert_eq!(
        argmax(&final_logits(&device, &reset).unwrap()).unwrap(),
        generated[0]
    );
}

#[test]
#[ignore = "requires official local Qwen3-1.7B checkpoint; set FERRUM_QWEN3_1_7B_MODEL"]
fn official_qwen3_second_size_sharded_reference() {
    let dir = std::path::PathBuf::from(
        std::env::var_os("FERRUM_QWEN3_1_7B_MODEL").expect("model directory"),
    );
    let device = MetalDevice::new().unwrap();
    let loaded = qwen3::load(&device, &dir).unwrap();
    assert_eq!(loaded.config.hidden_size, 2048);
    assert_eq!(loaded.config.intermediate_size, 6144);
    assert_eq!(loaded.tensor_count, 311);
    let reference: serde_json::Value = serde_json::from_str(include_str!(
        "../docs/measurements/phase6/qwen3-1.7b-reference-hello.json"
    ))
    .unwrap();
    let ids: Vec<u32> = serde_json::from_value(reference["prompt_ids"].clone()).unwrap();
    let (mut logits, mut cache) = loaded.model.forward_prefill_last(&device, &ids).unwrap();
    let mut generated = Vec::new();
    let mut max_selected_difference = 0.0f32;
    for row in reference["rows"].as_array().unwrap() {
        let values = final_logits(&device, &logits).unwrap();
        let token = argmax(&values).unwrap();
        assert_eq!(token as u64, row["token"].as_u64().unwrap());
        for item in row["top10"].as_array().unwrap() {
            let id = item[0].as_u64().unwrap() as usize;
            let difference = (values[id] - item[1].as_f64().unwrap() as f32).abs();
            max_selected_difference = max_selected_difference.max(difference);
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
    println!("Qwen3 1.7B max selected top-10 logit difference: {max_selected_difference}");
}

#[test]
#[ignore = "requires official local Qwen3-0.6B checkpoint; set FERRUM_QWEN3_MODEL"]
fn official_qwen3_short_and_long_prefill_reference() {
    let dir =
        std::path::PathBuf::from(std::env::var_os("FERRUM_QWEN3_MODEL").expect("model directory"));
    let device = MetalDevice::new().unwrap();
    let loaded = qwen3::load(&device, &dir).unwrap();
    for data in [
        include_str!("../docs/measurements/phase6/qwen3-0.6b-reference-short.json"),
        include_str!("../docs/measurements/phase6/qwen3-0.6b-reference-121.json"),
    ] {
        let reference: serde_json::Value = serde_json::from_str(data).unwrap();
        let ids: Vec<u32> = serde_json::from_value(reference["prompt_ids"].clone()).unwrap();
        let (mut logits, mut cache) = loaded.model.forward_prefill_last(&device, &ids).unwrap();
        let mut generated = Vec::new();
        let mut max_selected_difference = 0.0f32;
        for row in reference["rows"].as_array().unwrap() {
            let values = final_logits(&device, &logits).unwrap();
            let token = argmax(&values).unwrap();
            assert_eq!(token as u64, row["token"].as_u64().unwrap());
            for item in row["top10"].as_array().unwrap() {
                let id = item[0].as_u64().unwrap() as usize;
                let difference = (values[id] - item[1].as_f64().unwrap() as f32).abs();
                max_selected_difference = max_selected_difference.max(difference);
                assert!(
                    difference <= 1.0,
                    "{} tokens, step {}, ID {id}, difference {difference}",
                    ids.len(),
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
        println!(
            "Qwen3 {}-token prefill: max selected top-10 logit difference {max_selected_difference}",
            ids.len()
        );
    }
}
