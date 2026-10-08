//! Few-row attention kernel timing at several context depths: `m` query
//! tokens (1 = decode, 2.. = speculative verify) against `keys` cached
//! positions, for each K/V cache format.
//! usage: hybrid_attention MODEL.gguf [keys,...] [--m 1,3,5,9] [--kv f16,q8]
use ferrum::hybrid::{self, KvCache};

fn list(args: &[String], key: &str, default: &str) -> Vec<String> {
    args.iter()
        .position(|a| a == key)
        .map_or(default.to_owned(), |i| args[i + 1].clone())
        .split(',')
        .map(str::to_owned)
        .collect()
}

fn main() -> ferrum::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let depths: Vec<usize> = args
        .get(2)
        .filter(|a| !a.starts_with("--"))
        .map_or("1024,8192,32768,131072".into(), |v| v.clone())
        .split(',')
        .map(|v| v.parse().expect("keys"))
        .collect();
    let widths: Vec<usize> = list(&args, "--m", "1")
        .iter()
        .map(|v| v.parse().expect("m"))
        .collect();
    let formats: Vec<KvCache> = list(&args, "--kv", "f16")
        .iter()
        .map(|v| v.parse().expect("kv format"))
        .collect();
    let d = ferrum::MetalDevice::new()?;
    d.set_batch_limit(100_000)?;
    let loaded = hybrid::load(&d, &args[1], 16, 1)?;
    for &kv in &formats {
        for &m in &widths {
            for &keys in &depths {
                let (us, gbs) = loaded.model.bench_attention(&d, kv, m, keys, 50)?;
                println!(
                    "kv {kv:>3}  m {m:>2}  keys {keys:>7}: {us:>8.1} us/layer  {gbs:>6.1} GB/s"
                );
            }
        }
    }
    Ok(())
}
