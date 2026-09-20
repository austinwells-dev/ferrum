use ferrum::{DType, MetalDevice, Tensor, nn::Linear};
fn main() -> ferrum::Result<()> {
    let d = MetalDevice::new()?;
    if let Ok(value) = std::env::var("FERRUM_NATIVE_MATMUL") {
        d.set_native_matmul(value != "0")?;
    }
    for (label, k, n) in [
        ("q", 896, 896),
        ("kv", 896, 128),
        ("gate", 896, 4864),
        ("down", 4864, 896),
        ("lm_head", 896, 151936),
        ("awkward", 129, 257),
    ] {
        for dtype in [DType::BF16, DType::F16, DType::F32] {
            let w = Tensor::from_f32(&d, [n, k], dtype, &vec![0.01; n * k])?;
            let linear = Linear::new(&d, w, None)?;
            for m in [1, 21] {
                let x = Tensor::from_f32(&d, [m, k], dtype, &vec![0.5; m * k])?;
                let _ = linear.forward(&d, &x)?;
                let mut wall = Vec::new();
                let mut gpu = Vec::new();
                for _ in 0..9 {
                    let before = d.counters();
                    let start = std::time::Instant::now();
                    let y = linear.forward(&d, &x)?;
                    wall.push(start.elapsed().as_secs_f64() * 1000.);
                    gpu.push((d.counters().gpu - before.gpu).as_secs_f64() * 1000.);
                    std::hint::black_box(y);
                }
                wall.sort_by(f64::total_cmp);
                gpu.sort_by(f64::total_cmp);
                println!(
                    "{}",
                    serde_json::json!({"label":label,"dtype":format!("{dtype:?}"),"m":m,"k":k,"n":n,"wall_ms":wall[4],"gpu_ms":gpu[4],"min_ms":wall[0],"max_ms":wall[8]})
                );
            }
        }
    }
    Ok(())
}
