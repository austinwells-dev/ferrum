//! Teacher-forced diagnostic only: compare logits at the same token history.
use ferrum::{
    MetalDevice, Result,
    generation::{argmax, final_logits},
    loader::Weights,
    model::qwen,
    tokenizer::qwen::{DEFAULT_SYSTEM, QwenTokenizer},
};
fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().collect();
    let dir = std::path::Path::new(&args[1]);
    let d = MetalDevice::new()?;
    let qc = qwen::QwenConfig::from_file(dir.join("config.json"))?;
    let tok = QwenTokenizer::load(dir, &qc)?;
    let (_, ids) = tok.encode_prompt("What is Rust?", DEFAULT_SYSTEM, false)?;
    let w = Weights::from_file(&d, dir.join("model.safetensors"))?;
    let model = qwen::construct(&d, qc.convert()?, &w)?;
    drop(w);
    let teacher: Option<Vec<u32>> = args.get(2).map(|path| {
        let text = std::fs::read_to_string(path).unwrap();
        serde_json::from_str::<serde_json::Value>(&text).unwrap()["tokens"]
            .as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_u64().unwrap() as u32)
            .collect()
    });
    let (mut logits, mut cache) = model.forward_prefill(&d, &ids)?;
    let mut tokens = Vec::new();
    let mut steps = Vec::new();
    for step in 0..32 {
        let values = final_logits(&d, &logits)?;
        let chosen = argmax(&values)?;
        let mut order: Vec<_> = (0..values.len()).collect();
        order.sort_by(|&a, &b| values[b].total_cmp(&values[a]).then(a.cmp(&b)));
        steps.push(serde_json::json!({"chosen":chosen,"top10":order[..10].iter().map(|&i|(i,values[i],tok.tokenizer.decode(&[i as u32]).unwrap_or_default())).collect::<Vec<_>>()}));
        let next = teacher.as_ref().map_or(chosen, |t| t[step]);
        tokens.push(next);
        if step < 31 {
            logits = model.forward_decode(&d, next, &mut cache)?;
        }
    }
    println!(
        "{}",
        serde_json::json!({"tokens":tokens,"steps":steps,"text":tok.tokenizer.decode(&tokens)?})
    );
    Ok(())
}
