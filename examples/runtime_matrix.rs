//! End-to-end production generation matrix; no alternate inference path.
use ferrum::{
    MetalDevice, Result, generation,
    loader::Weights,
    model::qwen,
    tokenizer::qwen::{DEFAULT_SYSTEM, QwenTokenizer},
};
fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().collect();
    let dir = std::path::Path::new(&args[1]);
    let repeats = args
        .get(2)
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(3);
    let d = MetalDevice::new()?;
    if let Ok(limit) = std::env::var("FERRUM_BATCH_LIMIT") {
        d.set_batch_limit(limit.parse().unwrap())?;
    }
    d.set_profiling(std::env::var_os("FERRUM_MATRIX_PROFILE").is_some());
    let qc = qwen::QwenConfig::from_file(dir.join("config.json"))?;
    let tok = QwenTokenizer::load(dir, &qc)?;
    let w = Weights::from_file(&d, dir.join("model.safetensors"))?;
    let model = qwen::construct(&d, qc.convert()?, &w)?;
    drop(w);
    let (_, short) = tok.encode_prompt("Hello!", DEFAULT_SYSTEM, false)?;
    let prose = "A runtime executes a sequence of tensor operations. Memory ownership determines when storage can be reused. Numerical tests compare computed results with reference values. Measurements distinguish time spent computing from time spent waiting. ".repeat(100);
    let corpus = tok.tokenizer.encode(&prose)?;
    let mut cases = vec![
        ("short", short.clone(), 17),
        ("medium", corpus[..128].to_vec(), 17),
        ("512", corpus[..512].to_vec(), 17),
        ("1024", corpus[..1024].to_vec(), 17),
        ("sustained", short, 129),
    ];
    if std::env::var_os("FERRUM_MATRIX_LONG").is_some() {
        cases.push(("long-horizon", corpus[..368].to_vec(), 1601));
    }
    let filter = std::env::var("FERRUM_MATRIX_CASE").ok();
    for (label, prompt, count) in cases {
        if filter.as_ref().is_some_and(|x| x != label) {
            continue;
        }
        // Warm pipelines and allocator through an actual generation, then discard its cache.
        generation::generate(&d, &model, &prompt, count, &[], generation::argmax, |_| {
            Ok(())
        })?;
        for run in 0..repeats {
            let generation_start = std::time::Instant::now();
            let r =
                generation::generate(&d, &model, &prompt, count, &[], generation::argmax, |_| {
                    Ok(())
                })?;
            let generation_seconds = generation_start.elapsed().as_secs_f64();
            let post_first_seconds = generation_seconds - r.first_token.as_secs_f64();
            let mut dec: Vec<_> = r.decode.iter().map(|x| x.as_secs_f64() * 1000.).collect();
            dec.sort_by(f64::total_cmp);
            println!(
                "{}",
                serde_json::json!({"case":label,"run":run,"generation_ms":generation_seconds*1000.,"generation_tps":r.tokens.len() as f64/generation_seconds,"post_first_token_tps":(r.tokens.len()-1) as f64/post_first_seconds,"sampling_ms":r.sampling.as_secs_f64()*1000.,"decode_aggregate_tps":r.decode.len() as f64/r.decode.iter().sum::<std::time::Duration>().as_secs_f64(),"emit_callback":"noop","prompt_ids":prompt,"prefill_ms":r.prefill.as_secs_f64()*1000.,"prefill_tps":prompt.len() as f64/r.prefill.as_secs_f64(),"first_token_ms":r.first_token.as_secs_f64()*1000.,"decode_median_ms":dec[dec.len()/2],"decode_tps":1000./dec[dec.len()/2],"decode_ms":r.decode.iter().map(|x|x.as_secs_f64()*1000.).collect::<Vec<_>>(),"prefill_counters":r.prefill_counters,"decode_counters":r.decode_counters,"kv_active_bytes":r.kv_bytes,"kv_reserved_bytes":r.kv_reserved_bytes,"generated_ids":r.tokens,"prefill_profile":r.prefill_profile,"decode_profiles":r.decode_profiles})
            );
        }
    }
    Ok(())
}
