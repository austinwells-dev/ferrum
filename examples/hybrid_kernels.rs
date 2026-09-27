//! Projection kernel throughput on real hybrid-model weights.
//! usage: hybrid_kernels MODEL.gguf [m=1] [iters=200]
fn main() -> ferrum::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let m: usize = args.get(2).map_or(1, |v| v.parse().expect("m"));
    let iters: usize = args.get(3).map_or(200, |v| v.parse().expect("iters"));
    let d = ferrum::MetalDevice::new()?;
    d.set_batch_limit(100_000)?;
    let loaded = ferrum::hybrid::load(&d, &args[1], m.max(1), 1)?;
    for (name, us, gbs) in loaded.model.bench_projections(&d, m, iters)? {
        println!("{name:<40} {us:>9.1} us  {gbs:>7.1} GB/s");
    }
    Ok(())
}
