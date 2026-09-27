//! llama-bench-style throughput for hybrid GGUFs.
//! usage: hybrid_bench MODEL.gguf [--pp 512,2048] [--tg 128] [--reps 3] [--chunk 512]
//!
//! ppN: prefill N tokens into an empty state (one forward, final-token argmax).
//! tgN: N greedy single-token decode steps from an empty state.
//! Each measurement is preceded by one untimed warmup run.
use ferrum::{
    MetalDevice, Result,
    hybrid::{self, HybridState, Output},
};
use std::time::Instant;

fn stats(v: &[f64]) -> (f64, f64) {
    let mean = v.iter().sum::<f64>() / v.len() as f64;
    let var = v.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (v.len().max(2) - 1) as f64;
    (mean, var.sqrt())
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let get = |key: &str, default: &str| -> String {
        args.iter()
            .position(|a| a == key)
            .map_or(default.to_owned(), |i| args[i + 1].clone())
    };
    let pp: Vec<usize> = get("--pp", "512,2048")
        .split(',')
        .filter(|s| !s.is_empty())
        .map(|s| s.parse().expect("pp size"))
        .collect();
    let tg: usize = get("--tg", "128").parse().expect("tg");
    let depths: Vec<usize> = get("--depth", "")
        .split(',')
        .filter(|s| !s.is_empty())
        .map(|s| s.parse().expect("depth"))
        .collect();
    let reps: usize = get("--reps", "3").parse().expect("reps");
    let chunk: usize = get("--chunk", "512").parse().expect("chunk");
    let d = MetalDevice::new()?;
    let loaded = hybrid::load(&d, &args[1], chunk, 1)?;
    let model = &loaded.model;
    let max_pp = pp.iter().copied().max().unwrap_or(0);
    let pp_depth_max = get("--pp-depth", "")
        .split(',')
        .filter(|s| !s.is_empty())
        .map(|s| s.parse::<usize>().expect("pp depth") + 512)
        .max()
        .unwrap_or(0);
    let max_depth = depths.iter().copied().max().unwrap_or(0).max(pp_depth_max);
    let mut state = HybridState::new(&d, &model.config, max_pp.max(tg).max(max_depth + 32) + 1)?;
    // Deterministic pseudo-text tokens in the ordinary-word ID range.
    let tokens: Vec<u32> = (0..max_pp.max(max_depth) as u32)
        .map(|i| 1000 + (i * 7919) % 20000)
        .collect();
    println!("| model | test | t/s |\n|---|---|---|");
    // Prefill of one 512-token chunk after `depth` cached positions.
    for depth in get("--pp-depth", "")
        .split(',')
        .filter(|s| !s.is_empty())
        .map(|s| s.parse::<usize>().expect("pp depth"))
    {
        state.reset();
        model.forward(&d, &mut state, &tokens[..depth], Output::None)?;
        let start = Instant::now();
        model.forward(&d, &mut state, &tokens[depth..depth + 512], Output::Argmax)?;
        println!(
            "| {} | pp512 @ d{depth} | {:.2} |",
            model.config.name,
            512. / start.elapsed().as_secs_f64()
        );
    }
    // Decode speed after a prefilled context of each depth (16 timed steps).
    for &depth in &depths {
        state.reset();
        if depth > 0 {
            model.forward(&d, &mut state, &tokens[..depth], Output::None)?;
        }
        let mut token = 1000u32;
        let mut times = Vec::new();
        for _ in 0..17 {
            let start = Instant::now();
            if let hybrid::Produced::Token(t) =
                model.forward(&d, &mut state, &[token], Output::Argmax)?
            {
                token = t;
            }
            times.push(start.elapsed().as_secs_f64());
        }
        times.remove(0);
        let (mean, _) = stats(&times);
        println!(
            "| {} | tg@{depth} | {:.2} (step {:.2} ms) |",
            model.config.name,
            1. / mean,
            mean * 1e3
        );
    }
    for &n in &pp {
        let mut rates = Vec::new();
        for rep in 0..=reps {
            state.reset();
            let start = Instant::now();
            model.forward(&d, &mut state, &tokens[..n], Output::Argmax)?;
            if rep > 0 {
                rates.push(n as f64 / start.elapsed().as_secs_f64());
            }
        }
        let (mean, sd) = stats(&rates);
        println!("| {} | pp{n} | {mean:.2} ± {sd:.2} |", model.config.name);
    }
    if tg > 0 {
        let mut rates = Vec::new();
        for rep in 0..=reps {
            state.reset();
            let mut token = 1000u32;
            let start = Instant::now();
            for _ in 0..tg {
                if let hybrid::Produced::Token(t) =
                    model.forward(&d, &mut state, &[token], Output::Argmax)?
                {
                    token = t;
                }
            }
            if rep > 0 {
                rates.push(tg as f64 / start.elapsed().as_secs_f64());
            }
        }
        let (mean, sd) = stats(&rates);
        println!("| {} | tg{tg} | {mean:.2} ± {sd:.2} |", model.config.name);
        // One decode step's runtime counters.
        let before = d.counters();
        let start = Instant::now();
        model.forward(&d, &mut state, &[1000], Output::Argmax)?;
        let wall = start.elapsed();
        let after = d.counters();
        eprintln!(
            "decode step: wall {:.2} ms; {} dispatches in {} command buffers; encode {:.2} ms; wait {:.2} ms; GPU {:.2} ms",
            wall.as_secs_f64() * 1e3,
            after.dispatches - before.dispatches,
            after.command_buffers - before.command_buffers,
            (after.encode - before.encode).as_secs_f64() * 1e3,
            (after.wait - before.wait).as_secs_f64() * 1e3,
            (after.gpu - before.gpu).as_secs_f64() * 1e3,
        );
    }
    Ok(())
}
