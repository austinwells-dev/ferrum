//! Streaming-read bandwidth ceiling of the GPU.
fn main() -> ferrum::Result<()> {
    let d = ferrum::MetalDevice::new()?;
    d.set_batch_limit(100_000)?;
    for &(span, threads) in &[
        (4096usize, 128usize),
        (16384, 256),
        (65536, 256),
        (262144, 1024),
        (8192, 256),
    ] {
        let gbs = ferrum::hybrid::bench::bench_read_bandwidth(&d, 1 << 30, span, threads)?;
        println!("1 GiB, {span:>7} B/threadgroup, {threads:>4} threads: {gbs:.1} GB/s");
    }
    Ok(())
}
