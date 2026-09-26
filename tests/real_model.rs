//! Explicit local-only ~1 GB checkpoint test. No downloads.
use ferrum::{
    MetalDevice,
    generation::{self, StopReason},
    loader::Weights,
    model::qwen::{self, QwenConfig},
    tokenizer::qwen::{DEFAULT_SYSTEM, QwenTokenizer},
};
#[test]
#[ignore = "requires official local Qwen checkpoint; set FERRUM_QWEN_MODEL"]
fn official_qwen_bf16_cached_generation() {
    let dir = std::env::var_os("FERRUM_QWEN_MODEL")
        .map(std::path::PathBuf::from)
        .expect("set FERRUM_QWEN_MODEL to the official checkpoint directory");
    let d = MetalDevice::new().unwrap();
    if let Ok(value) = std::env::var("FERRUM_NATIVE_MATMUL") {
        d.set_native_matmul(value != "0").unwrap();
    }
    let qc = QwenConfig::from_file(dir.join("config.json")).unwrap();
    let c = qc.convert().unwrap();
    let tok = QwenTokenizer::load(&dir, &qc).unwrap();
    let (raw, ids) = tok.encode_prompt("Hello!", DEFAULT_SYSTEM, false).unwrap();
    let reference: serde_json::Value = serde_json::from_str(include_str!(
        "../docs/measurements/phase3/reference-bf16.json"
    ))
    .unwrap();
    assert_eq!(raw, reference["raw"].as_str().unwrap());
    assert_eq!(serde_json::json!(ids), reference["prompt_ids"]);
    let weights = Weights::from_file(&d, dir.join("model.safetensors")).unwrap();
    assert_eq!(weights.names().count(), 290);
    assert_eq!(weights.bytes(), 988065536);
    let model = qwen::construct(&d, c.clone(), &weights).unwrap();
    drop(weights);
    assert_eq!(model.weight_bytes(), 988065536);
    let mut step = 0;
    let mut emitted = Vec::new();
    let r = generation::generate(
        &d,
        &model,
        &ids,
        16,
        &tok.eos_ids,
        |values| {
            assert_eq!(values.len(), 151936);
            assert!(values.iter().all(|v| v.is_finite()));
            let token = generation::argmax(values)?;
            if step < 5 {
                let expected = &reference["steps"][step];
                assert_eq!(token as u64, expected["token"].as_u64().unwrap());
                // Empirical BF16 cross-engine check, not a universal floating-point error bound.
                // F32 diagnosis independently agrees to < 8e-5 (see phase3-results).
                // Parallel RMSNorm (no ordered midpoint fallback, Phase 7A Experiment 59)
                // measured a 0.516 maximum over these five steps; 0.5 held only with it.
                for (id, v) in expected["selected"].as_object().unwrap() {
                    let id: usize = id.parse().unwrap();
                    assert!(
                        (values[id] as f64 - v.as_f64().unwrap()).abs() <= 0.55,
                        "step {step}, ID {id}"
                    );
                }
                for pair in expected["top10"].as_array().unwrap() {
                    let id = pair[0].as_u64().unwrap() as usize;
                    assert!(
                        (values[id] as f64 - pair[1].as_f64().unwrap()).abs() <= 0.55,
                        "step {step}, top ID {id}"
                    );
                }
            }
            step += 1;
            Ok(token)
        },
        |id| {
            assert!((id as usize) < tok.tokenizer.vocab_size());
            emitted.push(id);
            Ok(())
        },
    )
    .unwrap();
    assert_eq!(r.stop, StopReason::Eos);
    assert!(r.tokens.len() > 5);
    assert!(tok.eos_ids.contains(r.tokens.last().unwrap()));
    assert_eq!(r.decode.len(), r.tokens.len() - 1);
    assert_eq!(
        r.kv_bytes,
        2 * c.num_layers
            * (ids.len() + r.tokens.len() - 1)
            * c.num_key_value_heads
            * c.head_dim
            * 2
    );
    // On-device greedy selection yields the same IDs and stop behavior.
    let greedy =
        generation::generate_greedy(&d, &model, &ids, 16, &tok.eos_ids, |_| Ok(())).unwrap();
    assert_eq!(greedy.tokens, r.tokens);
    assert_eq!(greedy.stop, r.stop);
    assert_eq!(greedy.kv_bytes, r.kv_bytes);
    let text = tok.tokenizer.decode(&emitted).unwrap();
    assert!(text.starts_with("Hello! How can I "));
    assert!(text.ends_with("you today?"));
    println!("{text}\n{r:?}");
    // Tokenizer stream matches full decode even across split multi-byte Unicode.
    let unicode = tok.tokenizer.encode("Hello 世界 🌙 café!").unwrap();
    let mut stream = tok.tokenizer.decode_stream();
    let mut output = String::new();
    for id in unicode {
        if let Some(s) = stream.step(id).unwrap() {
            output.push_str(&s);
        }
    }
    assert_eq!(output, "Hello 世界 🌙 café!");
}

#[test]
#[ignore = "128-token lifetime stress requires official local Qwen checkpoint"]
fn official_qwen_long_lifetime_stress() {
    let dir =
        std::path::PathBuf::from(std::env::var_os("FERRUM_QWEN_MODEL").expect("model directory"));
    let d = MetalDevice::new().unwrap();
    let qc = QwenConfig::from_file(dir.join("config.json")).unwrap();
    let config = qc.convert().unwrap();
    let tok = QwenTokenizer::load(&dir, &qc).unwrap();
    let (_, ids) = tok.encode_prompt("Hello!", DEFAULT_SYSTEM, false).unwrap();
    let w = Weights::from_file(&d, dir.join("model.safetensors")).unwrap();
    let model = qwen::construct(&d, config, &w).unwrap();
    drop(w);
    // Suppress EOS solely to stress 128 successive actual cached invocations.
    let result =
        generation::generate(&d, &model, &ids, 128, &[], generation::argmax, |_| Ok(())).unwrap();
    assert_eq!(result.tokens.len(), 128);
    assert_eq!(result.decode.len(), 127);
    assert!(
        result
            .decode_counters
            .iter()
            .all(|c| c.completion_waits <= 2)
    );
    println!("stress_final_counters={:?}", result.decode_counters.last());
    let (first, mut cache) = model.forward_prefill(&d, &ids).unwrap();
    let expected = first.to_f32();
    cache.reset();
    let again = model.forward(&d, &ids, &mut cache, None).unwrap();
    assert_eq!(again.to_f32(), expected);
}
