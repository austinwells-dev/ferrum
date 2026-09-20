use ferrum::{
    DType, MetalDevice, Tensor,
    generation::{argmax, final_logits, validate_context},
    loader::Weights,
    model::{
        qwen::{self, QwenConfig},
        tiny,
    },
    tokenizer::qwen::chat_prompt,
};
fn config() -> QwenConfig {
    QwenConfig::from_bytes(include_bytes!("fixtures/qwen/config.json")).unwrap()
}
#[test]
fn official_config_and_rejections() {
    let c = config().convert().unwrap();
    assert_eq!(
        (c.hidden_size, c.head_dim, c.num_layers, c.vocab_size),
        (896, 64, 24, 151936)
    );
    for (key, value) in [
        ("model_type", serde_json::json!("qwen3")),
        ("architectures", serde_json::json!(["Other"])),
        ("torch_dtype", serde_json::json!("float32")),
        ("hidden_size", serde_json::json!(895)),
        ("intermediate_size", serde_json::json!(0)),
        ("num_hidden_layers", serde_json::json!(0)),
        ("num_attention_heads", serde_json::json!(0)),
        ("num_key_value_heads", serde_json::json!(3)),
        ("rms_norm_eps", serde_json::json!(0)),
        ("rope_theta", serde_json::json!(-1)),
        ("max_position_embeddings", serde_json::json!(0)),
        ("use_sliding_window", serde_json::json!(true)),
        ("hidden_act", serde_json::json!("gelu")),
        ("rope_scaling", serde_json::json!({"type":"yarn"})),
        ("attention_bias", serde_json::json!(false)),
        ("mlp_bias", serde_json::json!(true)),
        ("partial_rotary_factor", serde_json::json!(0.5)),
        ("bos_token_id", serde_json::json!(151936)),
        ("new_semantics", serde_json::json!(true)),
    ] {
        let mut v: serde_json::Value =
            serde_json::from_slice(include_bytes!("fixtures/qwen/config.json")).unwrap();
        v[key] = value;
        assert!(
            QwenConfig::from_bytes(&serde_json::to_vec(&v).unwrap())
                .unwrap()
                .convert()
                .is_err(),
            "{key}"
        );
    }
    assert!(QwenConfig::from_bytes(b"{}").is_err());
    assert!(QwenConfig::from_file("/nonexistent/ferrum/config.json").is_err());
}
#[test]
fn official_names_small_payload_and_failures() {
    let d = MetalDevice::new().unwrap();
    let mut c = ferrum::model::ModelConfig::tiny(DType::BF16);
    c.tie_word_embeddings = true;
    let original = tiny::weights(&c).unwrap();
    let specs = qwen::specifications(&c);
    assert!(
        specs
            .iter()
            .any(|(n, _, s)| n == "model.layers.0.self_attn.k_proj.bias" && s == &[8])
    );
    let official: tiny::CpuWeights = specs
        .iter()
        .map(|(n, canonical, _)| (n.clone(), original[canonical].clone()))
        .collect();
    let w = Weights::from_bytes(&d, &tiny::serialize(&c, &official).unwrap()).unwrap();
    let mapped = qwen::map_weights(&d, &c, &w).unwrap();
    for (n, _, _) in &specs {
        assert!(w.optional(n).is_some());
    }
    let model = qwen::construct(&d, c.clone(), &w).unwrap();
    let canonical = ferrum::model::Transformer::from_weights(&d, c.clone(), &mapped).unwrap();
    assert_eq!(
        model.forward_prefill(&d, &[1, 2]).unwrap().0.to_f32(),
        canonical.forward_prefill(&d, &[1, 2]).unwrap().0.to_f32()
    );
    for kind in 0..4 {
        let mut bad = official.clone();
        let name = "model.layers.0.self_attn.k_proj.bias";
        match kind {
            0 => {
                bad.remove(name);
            }
            1 => bad.get_mut(name).unwrap().0 = vec![2, 4],
            2 => {
                bad.insert("unexpected".into(), (vec![1], vec![0.]));
            }
            _ => {}
        }
        let mut dtype = c.clone();
        if kind == 3 {
            dtype.dtype = DType::F16;
        }
        let w = Weights::from_bytes(&d, &tiny::serialize(&dtype, &bad).unwrap()).unwrap();
        let before = d.counters().dispatches;
        let error = match qwen::construct(&d, c.clone(), &w) {
            Ok(_) => panic!("accepted malformed weights"),
            Err(e) => e.to_string(),
        };
        assert!(error.contains(if kind == 2 {
            "unexpected"
        } else if kind == 3 {
            "dtype"
        } else {
            name
        }));
        assert_eq!(before, d.counters().dispatches);
    }
}
#[test]
fn final_row_argmax_context_and_template() {
    let d = MetalDevice::new().unwrap();
    let t = Tensor::from_f32(
        &d,
        [3, 4],
        DType::BF16,
        &[99., 0., 0., 0., 0., 98., 0., 0., 1., 2., 3., 4.],
    )
    .unwrap();
    assert_eq!(final_logits(&d, &t).unwrap(), [1., 2., 3., 4.]);
    assert_eq!(argmax(&final_logits(&d, &t).unwrap()).unwrap(), 3);
    assert_eq!(argmax(&[2., 2., 1.]).unwrap(), 0);
    for x in [
        vec![],
        vec![f32::NAN],
        vec![f32::INFINITY],
        vec![f32::NEG_INFINITY],
    ] {
        assert!(argmax(&x).is_err());
    }
    assert!(final_logits(&d, &t.reshape([12]).unwrap()).is_err());
    assert!(validate_context(0, 1, 10).is_err());
    assert!(validate_context(11, 0, 10).is_err());
    assert!(validate_context(8, 3, 10).is_err());
    assert!(validate_context(1, usize::MAX, 10).is_err());
    assert!(validate_context(8, 2, 10).is_ok());
    assert_eq!(
        chat_prompt("S", "Hello!"),
        "<|im_start|>system\nS<|im_end|>\n<|im_start|>user\nHello!<|im_end|>\n<|im_start|>assistant\n"
    );
}
#[test]
fn sampler_distribution_filters_and_determinism() {
    use ferrum::sampling::{Sampler, SamplingConfig};
    for (temperature, top_p) in [
        (-1., 1.),
        (f64::NAN, 1.),
        (f64::INFINITY, 1.),
        (1., 0.),
        (1., 1.1),
        (1., f64::NAN),
    ] {
        assert!(
            Sampler::new(SamplingConfig {
                temperature,
                top_p,
                ..Default::default()
            })
            .is_err()
        );
    }
    let c = SamplingConfig {
        temperature: 0.7,
        top_k: 3,
        top_p: 0.95,
        seed: 42,
    };
    let mut a = Sampler::new(c).unwrap();
    let mut b = Sampler::new(c).unwrap();
    let logits = [1., 2., 3., 4., -100.];
    let x: Vec<_> = (0..100).map(|_| a.sample(&logits).unwrap()).collect();
    assert_eq!(
        x,
        (0..100)
            .map(|_| b.sample(&logits).unwrap())
            .collect::<Vec<_>>()
    );
    assert!(x.iter().all(|&id| (1..=3).contains(&id)));
    assert!(x.contains(&2));
    assert!(x.contains(&3));
    for c in [
        SamplingConfig {
            temperature: 1.,
            top_k: 1,
            ..Default::default()
        },
        SamplingConfig {
            temperature: 1.,
            top_p: 0.01,
            ..Default::default()
        },
        SamplingConfig::default(),
    ] {
        let mut s = Sampler::new(c).unwrap();
        for _ in 0..10 {
            assert_eq!(s.sample(&logits).unwrap(), 3);
        }
    }
    let mut s = Sampler::new(SamplingConfig {
        temperature: f64::MIN_POSITIVE,
        ..Default::default()
    })
    .unwrap();
    assert_eq!(s.sample(&[-f32::MAX, f32::MAX]).unwrap(), 1);
    let mut s = Sampler::new(SamplingConfig {
        temperature: 1.,
        ..Default::default()
    })
    .unwrap();
    let mut counts = [0; 3];
    for _ in 0..6000 {
        counts[s.sample(&[0., 0., 0.]).unwrap() as usize] += 1;
    }
    assert!(
        counts.iter().all(|&n| (1800..2200).contains(&n)),
        "{counts:?}"
    );
}
#[test]
fn generation_stops_and_does_not_append_terminal_token() {
    use ferrum::generation::{StopReason, generate, stop_reason};
    let d = MetalDevice::new().unwrap();
    let c = ferrum::model::ModelConfig::tiny(DType::BF16);
    let w = Weights::from_bytes(
        &d,
        &tiny::serialize(&c, &tiny::weights(&c).unwrap()).unwrap(),
    )
    .unwrap();
    let model = ferrum::model::Transformer::from_weights(&d, c.clone(), &w).unwrap();
    for (maximum, eos, expected, reason) in [
        (4, vec![3], 1, StopReason::Eos),
        (3, vec![4], 3, StopReason::MaxNewTokens),
        (0, vec![4], 0, StopReason::MaxNewTokens),
    ] {
        let mut emitted = Vec::new();
        let r = generate(
            &d,
            &model,
            &[1, 2],
            maximum,
            &eos,
            |_| Ok(3),
            |id| {
                emitted.push(id);
                Ok(())
            },
        )
        .unwrap();
        assert_eq!(r.tokens.len(), expected);
        assert_eq!(r.stop, reason);
        assert_eq!(r.decode.len(), expected.saturating_sub(1));
        if expected > 0 {
            assert_eq!(
                r.kv_bytes,
                2 * c.num_layers
                    * (2 + expected - 1)
                    * c.num_key_value_heads
                    * c.head_dim
                    * c.dtype.size_bytes()
            );
        }
        if reason == StopReason::Eos {
            assert!(emitted.is_empty());
        }
    }
    let before = d.counters().dispatches;
    assert!(generate(&d, &model, &[1; 15], 2, &[3], argmax, |_| Ok(())).is_err());
    assert_eq!(before, d.counters().dispatches);
    assert_eq!(stop_reason(3, 4, 4, &[3]), Some(StopReason::Eos));
}
#[test]
fn profile_is_opt_in_and_counts_operations() {
    let d = MetalDevice::new().unwrap();
    let x = Tensor::from_f32(&d, [2], DType::F32, &[1., 2.]).unwrap();
    d.add(&x, &x).unwrap();
    assert!(d.take_profile().is_empty());
    d.set_profiling(true);
    d.add(&x, &x).unwrap();
    d.add(&x, &x).unwrap();
    let p = d.take_profile();
    assert_eq!(p["add"].calls, 2);
    assert_eq!(p["add"].allocation_bytes, 16);
    assert!(d.take_profile().is_empty());
    d.set_profiling(false);
}
#[test]
fn special_token_and_template_metadata_rejections() {
    use ferrum::tokenizer::qwen::validate_metadata;
    let tc: serde_json::Value =
        serde_json::from_str(include_str!("fixtures/qwen/tokenizer_config.json")).unwrap();
    let gc: serde_json::Value =
        serde_json::from_str(include_str!("fixtures/qwen/generation_config.json")).unwrap();
    validate_metadata(&config(), &tc, &gc).unwrap();
    for (key, value) in [
        ("add_bos_token", serde_json::json!(true)),
        ("bos_token", serde_json::json!("BOS")),
        ("eos_token", serde_json::json!("EOS")),
        ("pad_token", serde_json::json!("PAD")),
        ("chat_template", serde_json::json!("different")),
        ("add_prefix_space", serde_json::json!(true)),
    ] {
        let mut bad = tc.clone();
        bad[key] = value;
        assert!(validate_metadata(&config(), &bad, &gc).is_err(), "{key}");
    }
    for (key, value) in [
        ("bos_token_id", serde_json::json!(0)),
        ("eos_token_id", serde_json::json!([151645])),
        ("pad_token_id", serde_json::json!(999999)),
    ] {
        let mut bad = gc.clone();
        bad[key] = value;
        assert!(validate_metadata(&config(), &tc, &bad).is_err(), "{key}");
    }
}
