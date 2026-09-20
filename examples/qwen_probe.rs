use ferrum::{
    MetalDevice, Result,
    generation::{argmax, final_logits},
    loader::Weights,
    model::qwen,
    tokenizer::qwen::{DEFAULT_SYSTEM, QwenTokenizer},
};
fn main() -> Result<()> {
    let dir = std::env::args()
        .nth(1)
        .ok_or_else(|| ferrum::Error::Parameter("model directory required".into()))?;
    let dir = std::path::Path::new(&dir);
    let d = MetalDevice::new()?;
    let cfg = qwen::QwenConfig::from_file(dir.join("config.json"))?;
    let mut c = cfg.convert()?;
    let diagnostic_f32 = std::env::args().any(|a| a == "--diagnostic-f32");
    let tok = QwenTokenizer::load(dir, &cfg)?;
    let (text, ids) = tok.encode_prompt("Hello!", DEFAULT_SYSTEM, false)?;
    println!("prompt={text:?}\nids={ids:?}");
    let start = std::time::Instant::now();
    let w = if diagnostic_f32 {
        c.dtype = ferrum::DType::F32;
        let bytes = std::fs::read(dir.join("model.safetensors"))
            .map_err(|e| ferrum::Error::Safetensors(e.to_string()))?;
        let source = safetensors::SafeTensors::deserialize(&bytes)
            .map_err(|e| ferrum::Error::Safetensors(e.to_string()))?;
        let data: Vec<_> = source
            .tensors()
            .into_iter()
            .map(|(name, t)| {
                let values: Vec<u8> = t
                    .data()
                    .chunks_exact(2)
                    .flat_map(|b| {
                        half::bf16::from_bits(u16::from_le_bytes([b[0], b[1]]))
                            .to_f32()
                            .to_le_bytes()
                    })
                    .collect();
                (name, t.shape().to_vec(), values)
            })
            .collect();
        let views = data
            .iter()
            .map(|(n, s, b)| {
                safetensors::tensor::TensorView::new(safetensors::Dtype::F32, s.clone(), b)
                    .map(|v| (n.as_str(), v))
            })
            .collect::<std::result::Result<Vec<_>, _>>()
            .map_err(|e| ferrum::Error::Safetensors(e.to_string()))?;
        let converted = safetensors::serialize(views, None)
            .map_err(|e| ferrum::Error::Safetensors(e.to_string()))?;
        Weights::from_bytes(&d, &converted)?
    } else {
        Weights::from_file(&d, dir.join("model.safetensors"))?
    };
    println!("source={} load={:?}", w.bytes(), start.elapsed());
    let start = std::time::Instant::now();
    let before = d.counters();
    let model = qwen::construct(&d, c, &w)?;
    println!(
        "retained={} construction={:?} transpose_bytes={} recommended={}",
        model.weight_bytes(),
        start.elapsed(),
        d.counters().allocated_bytes - before.allocated_bytes,
        d.recommended_max_working_set()
    );
    drop(w);
    let start = std::time::Instant::now();
    let (mut logits, mut cache) = model.forward_prefill(&d, &ids)?;
    let mut steps = Vec::new();
    let mut generated = Vec::new();
    for step in 0..8 {
        let values = final_logits(&d, &logits)?;
        let id = argmax(&values)?;
        let mut order: Vec<_> = (0..values.len()).collect();
        order.sort_by(|&a, &b| values[b].total_cmp(&values[a]).then(a.cmp(&b)));
        let selected: std::collections::BTreeMap<_, _> = [0, 1, 13, 198, 9707, 151643, 151645]
            .map(|i| (i.to_string(), values[i]))
            .into();
        let rec = serde_json::json!({"token":id,"top10":order[..10].iter().map(|&i|(i,values[i])).collect::<Vec<_>>(),"selected":selected});
        println!("step {step}: {rec}");
        steps.push(rec);
        generated.push(id);
        if tok.eos_ids.contains(&id) || step == 7 {
            break;
        }
        logits = model.forward_decode(&d, id, &mut cache)?;
    }
    println!(
        "elapsed={:?} text={:?}",
        start.elapsed(),
        tok.tokenizer.decode(&generated)?
    );
    let report = serde_json::json!({"prompt_ids":ids,"steps":steps,"generated_ids":generated});
    std::fs::write(
        if diagnostic_f32 {
            "docs/measurements/phase3/ferrum-probe-f32.json"
        } else {
            "docs/measurements/phase3/ferrum-probe.json"
        },
        serde_json::to_vec_pretty(&report).map_err(|e| ferrum::Error::Validation(e.to_string()))?,
    )
    .map_err(|e| ferrum::Error::Validation(e.to_string()))?;
    Ok(())
}
