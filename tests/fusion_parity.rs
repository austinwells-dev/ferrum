//! Graph fusions must not change a single output bit on real models.
use ferrum::{MetalDevice, generation::final_logits, model::qwen_gguf};

fn bits(values: &[f32]) -> Vec<u32> {
    values.iter().map(|v| v.to_bits()).collect()
}

/// Prefill plus eight greedy decode steps; returns every step's logits bits.
fn run(device: &MetalDevice, model: &ferrum::model::Transformer, prompt: &[u32]) -> Vec<Vec<u32>> {
    let (logits, mut cache) = model.forward_prefill_last(device, prompt).unwrap();
    let mut values = final_logits(device, &logits).unwrap();
    let mut steps = vec![bits(&values)];
    for _ in 0..8 {
        let token = ferrum::generation::argmax(&values).unwrap();
        let logits = model.forward_decode(device, token, &mut cache).unwrap();
        values = final_logits(device, &logits).unwrap();
        steps.push(bits(&values));
    }
    steps
}

fn check(path: std::ffi::OsString) {
    let device = MetalDevice::new().unwrap();
    let loaded = qwen_gguf::load(&device, std::path::PathBuf::from(path)).unwrap();
    // A prompt long enough for multi-row prefill and several KV-cache growths.
    let prompt: Vec<u32> = (0..300).map(|i| 100 + (i * 37) % 5000).collect();
    let mut outputs = Vec::new();
    for fused in [false, true] {
        device.set_fuse_add_rmsnorm(fused).unwrap();
        device.set_fuse_rope_cache(fused).unwrap();
        outputs.push(run(&device, &loaded.model, &prompt));
    }
    assert_eq!(outputs[0].len(), outputs[1].len());
    for (step, (unfused, fused)) in outputs[0].iter().zip(&outputs[1]).enumerate() {
        assert!(unfused == fused, "logits differ at step {step}");
    }
}

#[test]
#[ignore = "requires the Qwen2.5-0.5B Q4_K_M GGUF (projection biases); set FERRUM_QWEN25_GGUF"]
fn qwen25_fusions_are_bit_identical() {
    check(std::env::var_os("FERRUM_QWEN25_GGUF").expect("Qwen2.5 GGUF path"));
}

#[test]
#[ignore = "requires the Qwen3-0.6B Q8_0 GGUF (per-head q/k norms); set FERRUM_QWEN3_GGUF"]
fn qwen3_fusions_are_bit_identical() {
    check(std::env::var_os("FERRUM_QWEN3_GGUF").expect("Qwen3 GGUF path"));
}
