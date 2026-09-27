//! Speculative verify cost: target forward time versus rows per forward.
//! usage: hybrid_verify_cost MODEL.gguf [--context 1024] [--widths 1,2,4,8,16] [--reps 5]
use ferrum::{MetalDevice, Result, hybrid};

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let get = |key: &str, default: &str| -> String {
        args.iter()
            .position(|a| a == key)
            .map_or(default.to_owned(), |i| args[i + 1].clone())
    };
    let context: usize = get("--context", "1024").parse().expect("context");
    let widths: Vec<usize> = get("--widths", "1,2,3,4,5,6,8,9,12,16,17")
        .split(',')
        .map(|s| s.parse().expect("width"))
        .collect();
    let reps: usize = get("--reps", "5").parse().expect("reps");
    let d = MetalDevice::new()?;
    let loaded = hybrid::load(&d, &args[1], 512, 1)?;
    let results = loaded
        .model
        .bench_verify_widths(&d, context, &widths, reps)?;
    let base = results[0].1;
    println!("| rows | ms | x one row | ms per row |\n|---|---|---|---|");
    for (m, ms) in results {
        println!(
            "| {m} | {ms:.1} | {:.2} | {:.1} |",
            ms / base,
            ms / m as f64
        );
    }
    Ok(())
}
