//! Prompt-chunk attention timing at several depths.
//! usage: hybrid_prefill_attention MODEL.gguf [keys,...] [chunk=512]
fn main() -> ferrum::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let depths: Vec<usize> = args
        .get(2)
        .map_or("4096,32768,131072".into(), |v| v.clone())
        .split(',')
        .map(|v| v.parse().expect("keys"))
        .collect();
    let m: usize = args.get(3).map_or(512, |v| v.parse().expect("chunk"));
    let d = ferrum::MetalDevice::new()?;
    d.set_batch_limit(100_000)?;
    let loaded = ferrum::hybrid::load(&d, &args[1], m, 1)?;
    for keys in depths {
        let (ms, tflops) = loaded.model.bench_prefill_attention(&d, m, keys, 5)?;
        println!("chunk {m} at {keys:>7} keys: {ms:>8.2} ms/layer  {tflops:>6.2} TFLOP/s");
    }
    Ok(())
}
