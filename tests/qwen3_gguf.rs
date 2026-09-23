use ferrum::{
    MetalDevice,
    generation::{argmax, final_logits},
    model::{qwen_gguf, qwen3},
};

#[test]
#[ignore = "requires official Qwen3 BF16 and Q8_0 GGUF; set FERRUM_QWEN3_MODEL and FERRUM_QWEN3_GGUF"]
fn official_qwen3_q8_uses_packed_path_and_matches_greedy_reference() {
    let dense_dir = std::path::PathBuf::from(
        std::env::var_os("FERRUM_QWEN3_MODEL").expect("BF16 model directory"),
    );
    let gguf_path = std::path::PathBuf::from(
        std::env::var_os("FERRUM_QWEN3_GGUF").expect("official Q8_0 GGUF path"),
    );
    let device = MetalDevice::new().unwrap();
    let dense = qwen3::load(&device, dense_dir).unwrap();
    let q8 = qwen_gguf::load(&device, gguf_path).unwrap();
    assert_eq!(q8.architecture, "qwen3");
    assert_eq!(q8.config.hidden_size, dense.config.hidden_size);
    assert_eq!(q8.config.head_dim, dense.config.head_dim);
    assert_eq!(q8.tensor_count, 310);
    assert!(q8.quantized_tensor_bytes > 600_000_000);
    assert!(q8.model.weight_bytes() < dense.model.weight_bytes());
    let reference: serde_json::Value = serde_json::from_str(include_str!(
        "../docs/measurements/phase6/qwen3-0.6b-reference-hello.json"
    ))
    .unwrap();
    let ids: Vec<u32> = serde_json::from_value(reference["prompt_ids"].clone()).unwrap();
    let (mut dense_logits, mut dense_cache) =
        dense.model.forward_prefill_last(&device, &ids).unwrap();
    let (mut q8_logits, mut q8_cache) = q8.model.forward_prefill_last(&device, &ids).unwrap();
    let mut cosine_min = 1.0f64;
    let mut generated = Vec::new();
    for row in reference["rows"].as_array().unwrap() {
        let a = final_logits(&device, &dense_logits).unwrap();
        let b = final_logits(&device, &q8_logits).unwrap();
        let (dot, aa, bb) =
            a.iter()
                .zip(&b)
                .fold((0.0f64, 0.0f64, 0.0f64), |(dot, aa, bb), (&x, &y)| {
                    let (x, y) = (x as f64, y as f64);
                    (dot + x * y, aa + x * x, bb + y * y)
                });
        let cosine = dot / (aa * bb).sqrt();
        cosine_min = cosine_min.min(cosine);
        assert!(
            cosine > 0.99,
            "Q8/BF16 logit cosine {cosine} at step {}",
            row["step"]
        );
        let dense_token = argmax(&a).unwrap();
        let q8_token = argmax(&b).unwrap();
        assert_eq!(dense_token, q8_token);
        assert_eq!(q8_token as u64, row["token"].as_u64().unwrap());
        generated.push(q8_token);
        if generated.len() < reference["rows"].as_array().unwrap().len() {
            dense_logits = dense
                .model
                .forward_decode(&device, dense_token, &mut dense_cache)
                .unwrap();
            q8_logits = q8
                .model
                .forward_decode(&device, q8_token, &mut q8_cache)
                .unwrap();
        }
    }
    assert_eq!(serde_json::json!(generated), reference["generated_ids"]);
    println!("Qwen3 Q8/BF16 minimum full-vocabulary logit cosine: {cosine_min:.6}");
}
