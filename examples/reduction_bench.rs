//! Standalone reduction latency across short/medium/long contexts and awkward widths.
use ferrum::{DType, MetalDevice, Tensor, reference};
fn main() -> ferrum::Result<()> {
    let d = MetalDevice::new()?;
    for ty in [DType::F32, DType::F16, DType::BF16] {
        for width in [21, 128, 896, 2049, 4096] {
            let x = Tensor::from_f32(&d, [14, width], ty, &reference::deterministic(14 * width))?;
            let w = Tensor::from_f32(&d, [width], ty, &vec![1.; width])?;
            for norm in [false, true] {
                let mut wall = Vec::new();
                let mut gpu = Vec::new();
                for sample in 0..12 {
                    let start = std::time::Instant::now();
                    let y = if norm {
                        d.rmsnorm(&x, &w, 1e-5)?
                    } else {
                        d.softmax(&x)?
                    };
                    if sample >= 3 {
                        wall.push(start.elapsed().as_secs_f64() * 1e6);
                        gpu.push(y.metrics.timing.gpu.unwrap_or_default().as_secs_f64() * 1e6);
                    }
                }
                wall.sort_by(f64::total_cmp);
                gpu.sort_by(f64::total_cmp);
                println!(
                    "{}",
                    serde_json::json!({"dtype":format!("{ty:?}"),"operation":if norm{"rmsnorm"}else{"softmax"},"rows":14,"width":width,"wall_us":wall[4],"gpu_us":gpu[4]})
                );
            }
        }
    }
    Ok(())
}
