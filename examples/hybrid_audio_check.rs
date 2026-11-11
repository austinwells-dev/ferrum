//! Encode a WAV clip with an LFM3-Audio projector and compare the rows with a
//! reference dump (llama.cpp's `mtmd_get_output_embd` for the same samples).
//!
//! usage: hybrid_audio_check MMPROJ.gguf CLIP.wav [REF.f32] [--samples OUT.f32] [--memory]
//!
//! `--samples` writes the decoded 16 kHz mono samples as raw little-endian
//! f32, so the reference encoder can read exactly what Ferrum encoded (no
//! second WAV decoder or resampler in the comparison). REF is raw f32
//! `[rows, width]`. Reports the shapes, the encode time, and per-row cosine
//! similarity and relative error, and times the CPU front end. `--memory`
//! compares the memory planner's projector budget with the device bytes the
//! weights take plus the peak of encoding `AUDIO_CACHE_CLIPS` full-length
//! clips while keeping their rows, as the runtime's cache does.
use ferrum::{MetalDevice, Result, vision, vision::AudioModel, vision::audio};

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!(
            "usage: hybrid_audio_check MMPROJ.gguf CLIP.wav [REF.f32] [--samples OUT.f32] [--memory]"
        );
        std::process::exit(2);
    }
    let samples_out = args
        .iter()
        .position(|a| a == "--samples")
        .map(|i| args[i + 1].clone());
    let reference = args.get(3).filter(|a| !a.starts_with("--"));
    let memory = args.iter().any(|a| a == "--memory");
    let d = MetalDevice::new()?;
    let before = d.allocated_bytes();
    let model = AudioModel::load(&d, &args[1])?;
    let loaded = d.allocated_bytes() - before;
    let bytes = std::fs::read(&args[2]).expect("read clip");
    let start = std::time::Instant::now();
    let samples = audio::decode_wav(&bytes)?;
    let decode = start.elapsed();
    if let Some(path) = samples_out {
        let raw: Vec<u8> = samples.iter().flat_map(|v| v.to_le_bytes()).collect();
        std::fs::write(&path, raw).expect("write samples");
    }
    model.warm(&d)?;
    let start = std::time::Instant::now();
    let embedding = model.encode_samples(&d, &samples, 0)?;
    let elapsed = start.elapsed();
    let rows = embedding.tokens();
    let width = model.out_dim();
    let got = embedding.rows.to_f32();
    println!(
        "{:.2} s of audio -> {rows} rows of {width} in {:.1} ms",
        samples.len() as f64 / audio::SAMPLE_RATE as f64,
        elapsed.as_secs_f64() * 1e3
    );
    let mib = |b: usize| b as f64 / (1 << 20) as f64;
    println!(
        "CPU front end: WAV decode and resample {:.1} ms, log-mel {:.1} ms",
        decode.as_secs_f64() * 1e3,
        {
            let start = std::time::Instant::now();
            audio::MelBank::new(128).features(&samples);
            start.elapsed().as_secs_f64() * 1e3
        }
    );
    if memory {
        // The worst case the plan covers: a full-length clip encoded while
        // the runtime's cache holds the rows of as many others as it keeps.
        // Counted from here: the clip checked above still holds its rows,
        // which the runtime would count as one of the cached clips.
        let base = d.allocated_bytes();
        let longest = vec![0.; vision::MAX_AUDIO_SECONDS * audio::SAMPLE_RATE];
        let mut kept = Vec::new();
        let mut peak = 0;
        for i in 0..vision::AUDIO_CACHE_CLIPS {
            let held = d.allocated_bytes() - base;
            kept.push(model.encode_samples(&d, &longest, i as u64)?);
            peak = peak.max(held + model.activation_bytes());
        }
        let planned = vision::projector_bytes(std::path::Path::new(&args[1]));
        let rows = (d.allocated_bytes() - base) / kept.len();
        println!(
            "memory: planned {:.1} MiB; measured weights {:.1} MiB (loader counted {:.1}) + one {} s clip's encode {:.1} MiB (its rows {:.1} MiB) + {} more cached clips' rows = peak {:.1} MiB",
            mib(planned),
            mib(loaded),
            mib(model.weight_bytes()),
            vision::MAX_AUDIO_SECONDS,
            mib(model.activation_bytes()),
            mib(rows),
            vision::AUDIO_CACHE_CLIPS - 1,
            mib(loaded + peak),
        );
    }
    let Some(reference) = reference else {
        return Ok(());
    };
    let raw = std::fs::read(reference).expect("read reference");
    let want: Vec<f32> = raw
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect();
    if want.len() != got.len() {
        eprintln!(
            "reference has {} values ({} rows), Ferrum {} ({rows} rows)",
            want.len(),
            want.len() / width,
            got.len()
        );
        std::process::exit(1);
    }
    let mut cosines = Vec::with_capacity(rows);
    let mut rel = Vec::with_capacity(rows);
    for (g, w) in got.chunks_exact(width).zip(want.chunks_exact(width)) {
        let dot: f64 = g.iter().zip(w).map(|(a, b)| f64::from(a * b)).sum();
        let ng: f64 = g.iter().map(|a| f64::from(a * a)).sum::<f64>().sqrt();
        let nw: f64 = w.iter().map(|a| f64::from(a * a)).sum::<f64>().sqrt();
        let diff: f64 = g
            .iter()
            .zip(w)
            .map(|(a, b)| f64::from((a - b) * (a - b)))
            .sum::<f64>()
            .sqrt();
        cosines.push(dot / (ng * nw).max(1e-30));
        rel.push(diff / nw.max(1e-30));
    }
    let mean = |v: &[f64]| v.iter().sum::<f64>() / v.len() as f64;
    let max_abs = got
        .iter()
        .zip(&want)
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    println!(
        "cosine mean {:.6} min {:.6}; relative L2 error mean {:.4} max {:.4}; max abs diff {max_abs:.4}",
        mean(&cosines),
        cosines.iter().copied().fold(f64::INFINITY, f64::min),
        mean(&rel),
        rel.iter().copied().fold(0., f64::max),
    );
    Ok(())
}
