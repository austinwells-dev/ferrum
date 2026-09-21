//! Teacher-forced numerical diagnostic using a recorded runtime-matrix history.
use ferrum::{
    MetalDevice, Result,
    generation::{argmax, final_logits},
    loader::Weights,
    model::qwen,
};
fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().collect();
    let dir = std::path::Path::new(&args[1]);
    let rows = std::fs::read_to_string(&args[2]).unwrap();
    let row: serde_json::Value = rows
        .lines()
        .map(|x| serde_json::from_str::<serde_json::Value>(x).unwrap())
        .find(|x| x["case"].as_str() == Some(&args[3]) && x["run"] == 0)
        .unwrap();
    let ids: Vec<u32> = serde_json::from_value(row["prompt_ids"].clone()).unwrap();
    let teacher: Vec<u32> = serde_json::from_value(row["generated_ids"].clone()).unwrap();
    let d = MetalDevice::new()?;
    let cfg = qwen::QwenConfig::from_file(dir.join("config.json"))?;
    let weights = Weights::from_file(&d, dir.join("model.safetensors"))?;
    let model = qwen::construct(&d, cfg.convert()?, &weights)?;
    drop(weights);
    let (mut logits, mut cache) = model.forward_prefill_last(&d, &ids)?;
    for (step, &token) in teacher.iter().enumerate() {
        let values = final_logits(&d, &logits)?;
        let mut top: Vec<_> = values.iter().copied().enumerate().collect();
        top.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
        top.truncate(10);
        println!(
            "{}",
            serde_json::json!({"step":step,"chosen":argmax(&values)?,"teacher":token,"top10":top})
        );
        if step + 1 < teacher.len() {
            logits = model.forward_decode(&d, token, &mut cache)?;
        }
    }
    Ok(())
}
