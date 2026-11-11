mod hybrid_cli;
#[cfg(feature = "legacy-engine")]
mod run;
#[cfg(feature = "legacy-engine")]
mod transformer_smoke;
#[cfg(feature = "legacy-engine")]
use ferrum::{DType, Tensor, reference as cpu};
use ferrum::{MetalDevice, Result};
fn main() -> Result<()> {
    let device = MetalDevice::new()?;
    match std::env::args().nth(1).as_deref().unwrap_or("info") {
        "info" => {
            println!("Capabilities: {}", device.capabilities()?);
            println!(
                "Runtime device: {}\nBackend: Metal\nUnified memory: {}\nRecommended max working set: {} bytes\nMax buffer: {} bytes\nSupported Apple GPU families (queried 1–10): {:?}",
                device.name(),
                device.has_unified_memory(),
                device.recommended_max_working_set(),
                device.max_buffer_length(),
                device.apple_families()
            );
        }
        #[cfg(feature = "legacy-engine")]
        "run" | "profile" => run::run(&device)?,
        "hybrid" => hybrid_cli::run(&device)?,
        #[cfg(feature = "legacy-engine")]
        "transformer-smoke" => transformer_smoke::run(&device)?,
        #[cfg(feature = "legacy-engine")]
        "smoke" => {
            println!("Metal device: {}", device.name());
            let start = std::time::Instant::now();
            device.warm_up()?;
            println!("Pipeline warm-up: {:?}", start.elapsed());
            let x = cpu::deterministic(3 * 128);
            let w = vec![1.; 128];
            let a = Tensor::from_f32(&device, vec![3, 128], DType::F32, &x)?;
            let weight = Tensor::from_f32(&device, vec![128], DType::F32, &w)?;
            let bdata = cpu::deterministic(128 * 65);
            let b = Tensor::from_f32(&device, vec![128, 65], DType::F32, &bdata)?;
            for (output, reference) in [
                (device.add(&a, &a)?, cpu::add(&x, &x)),
                (device.mul(&a, &a)?, cpu::mul(&x, &x)),
                (device.silu(&a)?, cpu::silu(&x)),
                (
                    device.rmsnorm(&a, &weight, 1e-5)?,
                    cpu::rmsnorm(&x, &w, 1e-5),
                ),
                (device.softmax(&a)?, cpu::softmax(&x, 128)),
                (
                    device.rope(&a, 128, 7, 10000.)?,
                    cpu::rope(&x, 128, 7, 10000.),
                ),
                (device.matmul(&a, &b)?, cpu::matmul(&x, &bdata, 3, 128, 65)),
            ] {
                let error = cpu::check(&output.tensor.to_f32(), &reference, 2e-4, 2e-4)?;
                println!(
                    "{:<10} PASS max_abs={:.3e} wall={:?} gpu={:?}",
                    output.metrics.operation,
                    error,
                    output.metrics.timing.synchronized,
                    output.metrics.timing.gpu
                );
            }
            println!("Phase 1 Metal smoke test: PASS");
        }
        other => {
            let commands = if cfg!(feature = "legacy-engine") {
                "info, smoke, transformer-smoke, run, profile or hybrid"
            } else {
                "info or hybrid (smoke, transformer-smoke, run and profile need --features legacy-engine)"
            };
            return Err(ferrum::Error::Parameter(format!(
                "unknown command {other}; use {commands}"
            )));
        }
    }
    Ok(())
}
