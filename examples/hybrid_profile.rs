//! GPU time per kernel of one decode step.
//! usage: hybrid_profile MODEL [--context 512] [--steps 4]
//!
//! Each dispatch runs as its own command buffer (batch limit 1), so its GPU
//! timestamps time it alone; that serializes the step, so the total is a
//! little above a batched step's GPU time. Kernels are summed over `steps`
//! decode steps after `context` cached positions and reported per step.
use ferrum::{
    MetalDevice, Result,
    hybrid::{self, HybridState, Output},
};

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let get = |key: &str, default: &str| -> String {
        args.iter()
            .position(|a| a == key)
            .map_or(default.to_owned(), |i| args[i + 1].clone())
    };
    let context: usize = get("--context", "512").parse().expect("context");
    let steps: usize = get("--steps", "4").parse().expect("steps");
    let d = MetalDevice::new()?;
    let loaded = hybrid::load(&d, &args[1], 512, 1)?;
    let model = &loaded.model;
    let mut state = HybridState::new(
        &d,
        &model.config,
        context + steps + 8,
        model.max_forward_rows(),
    )?;
    let prompt: Vec<u32> = (0..context as u32)
        .map(|i| 1000 + (i * 7919) % 20000)
        .collect();
    model.forward(&d, &mut state, &prompt, Output::None)?;
    model.forward(&d, &mut state, &[1000], Output::Argmax)?;
    d.set_batch_limit(1)?;
    d.set_profiling(true);
    for _ in 0..steps {
        model.forward(&d, &mut state, &[1000], Output::Argmax)?;
    }
    let profile = d.take_profile();
    d.set_profiling(false);
    let mut rows: Vec<(String, f64, usize)> = profile
        .into_iter()
        .map(|(name, e)| {
            (
                name,
                e.gpu.as_secs_f64() * 1e3 / steps as f64,
                e.calls / steps,
            )
        })
        .collect();
    rows.sort_by(|a, b| b.1.total_cmp(&a.1));
    let total: f64 = rows.iter().map(|r| r.1).sum();
    println!("| kernel | GPU ms per step | share | calls per step |\n|---|---|---|---|");
    for (name, ms, calls) in &rows {
        println!("| {name} | {ms:.2} | {:.1}% | {calls} |", 100. * ms / total);
    }
    println!(
        "| total | {total:.2} | | {} |",
        rows.iter().map(|r| r.2).sum::<usize>()
    );
    Ok(())
}
