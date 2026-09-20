//! Every sample waits for GPU completion. Inputs and pipelines are prepared outside measurement.
use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use ferrum::{DType, MetalDevice, Tensor, ops::Output, reference};
use std::{
    hint::black_box,
    time::{Duration, Instant},
};

fn input(d: &MetalDevice, shape: Vec<usize>, dtype: DType) -> Tensor {
    let n = shape.iter().product();
    Tensor::from_f32(d, shape, dtype, &reference::deterministic(n)).unwrap()
}
fn instrument(label: &str, mut f: impl FnMut() -> Output) {
    for _ in 0..5 {
        black_box(f());
    }
    let mut wall = Vec::new();
    let mut submit = Vec::new();
    let mut gpu = Vec::new();
    let mut end_to_end = Vec::new();
    for _ in 0..30 {
        let start = Instant::now();
        let out = f();
        end_to_end.push(start.elapsed().as_secs_f64() * 1e6);
        wall.push(out.metrics.timing.synchronized.as_secs_f64() * 1e6);
        submit.push(out.metrics.timing.submission.as_secs_f64() * 1e6);
        if let Some(t) = out.metrics.timing.gpu {
            gpu.push(t.as_secs_f64() * 1e6);
        }
        black_box(out);
    }
    fn median(x: &mut [f64]) -> f64 {
        x.sort_by(f64::total_cmp);
        x[x.len() / 2]
    }
    let gpu = if gpu.is_empty() {
        "unavailable".into()
    } else {
        format!("{:.3}", median(&mut gpu))
    };
    println!(
        "TIMING {label}: median_us e2e={:.3} submit={:.3} synchronized={:.3} gpu={gpu}",
        median(&mut end_to_end),
        median(&mut submit),
        median(&mut wall)
    );
}
fn benchmarks(c: &mut Criterion) {
    let d = MetalDevice::new().unwrap();
    let start = Instant::now();
    d.warm_up().unwrap();
    println!(
        "DEVICE {}; pipeline warm-up {:?}; samples include fresh output allocation and synchronization",
        d.name(),
        start.elapsed()
    );
    let mut group = c.benchmark_group("metal");
    group
        .sample_size(10)
        .warm_up_time(Duration::from_millis(300))
        .measurement_time(Duration::from_secs(1));
    for n in [1024, 65536, 1048576] {
        let a = input(&d, vec![n], DType::F32);
        let b = input(&d, vec![n], DType::F32);
        instrument(&format!("add/f32/{n}"), || d.add(&a, &b).unwrap());
        group.throughput(Throughput::Bytes((n * 12) as u64));
        group.bench_with_input(BenchmarkId::new("add_f32", n), &n, |bencher, _| {
            bencher.iter(|| black_box(d.add(black_box(&a), black_box(&b)).unwrap()))
        });
    }
    for (rows, width) in [(1, 128), (4, 4096), (32, 4096)] {
        let a = input(&d, vec![rows, width], DType::F32);
        let w = input(&d, vec![width], DType::F32);
        instrument(&format!("rmsnorm/f32/{rows}x{width}"), || {
            d.rmsnorm(&a, &w, 1e-5).unwrap()
        });
        group.throughput(Throughput::Elements((rows * width) as u64));
        group.bench_function(format!("rmsnorm_f32/{rows}x{width}"), |b| {
            b.iter(|| black_box(d.rmsnorm(black_box(&a), black_box(&w), 1e-5).unwrap()))
        });
        instrument(&format!("softmax/f32/{rows}x{width}"), || {
            d.softmax(&a).unwrap()
        });
        group.bench_function(format!("softmax_f32/{rows}x{width}"), |b| {
            b.iter(|| black_box(d.softmax(black_box(&a)).unwrap()))
        });
    }
    for (m, k, n, dtype) in [
        (17, 19, 23, DType::F32),
        (128, 128, 128, DType::F32),
        (256, 512, 256, DType::F32),
        (256, 512, 256, DType::F16),
    ] {
        let a = input(&d, vec![m, k], dtype);
        let b = input(&d, vec![k, n], dtype);
        instrument(&format!("matmul/{dtype:?}/{m}x{k}x{n}"), || {
            d.matmul(&a, &b).unwrap()
        });
        group.throughput(Throughput::Elements((2 * m * k * n) as u64));
        group.bench_function(format!("matmul_{dtype:?}/{m}x{k}x{n}"), |bench| {
            bench.iter(|| black_box(d.matmul(black_box(&a), black_box(&b)).unwrap()))
        });
    }
    group.finish();
}
criterion_group!(benches, benchmarks);
criterion_main!(benches);
