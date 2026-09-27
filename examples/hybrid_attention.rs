//! Decode-attention kernel timing at several context depths.
//! usage: hybrid_attention MODEL.gguf [keys,...]
fn main() -> ferrum::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let depths: Vec<usize> = args
        .get(2)
        .map_or("1024,8192,32768,131072".into(), |v| v.clone())
        .split(',')
        .map(|v| v.parse().expect("keys"))
        .collect();
    let d = ferrum::MetalDevice::new()?;
    d.set_batch_limit(100_000)?;
    let loaded = ferrum::hybrid::load(&d, &args[1], 1, 1)?;
    for keys in depths {
        let (us, gbs) = loaded.model.bench_decode_attention(&d, keys, 50)?;
        println!("keys {keys:>7}: {us:>8.1} us/layer  {gbs:>6.1} GB/s");
    }
    Ok(())
}
