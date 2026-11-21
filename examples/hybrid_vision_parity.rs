//! Compare Ferrum's embedding of one image with llama.cpp's.
//!
//! llama.cpp writes the projector output it feeds the language model when
//! `MTMD_DEBUG_EMBEDDINGS` holds a path (`[i32 tokens][i32 width][f32 ...]`):
//!
//!     MTMD_DEBUG_EMBEDDINGS=ref.bin llama-mtmd-cli -m MODEL --mmproj MMPROJ \
//!         --image IMG -p hi -n 1 --jinja
//!
//! This encodes the same image and prints, over all rows, the cosine
//! similarity, the largest absolute difference and the RMS of the difference
//! relative to the reference's RMS, plus the worst row.
//!
//! usage: hybrid_vision_parity MMPROJ.gguf IMAGE REF.bin [MAX_TOKENS]
//! (MAX_TOKENS defaults to 4096; match llama.cpp's `--image-max-tokens`)
use ferrum::{MetalDevice, Result, vision::VisionModel};

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let (mmproj, image, reference, max_tokens) = match &args[1..] {
        [m, i, r] => (m, i, r, 4096),
        [m, i, r, n] => (m, i, r, n.parse().unwrap_or(4096)),
        _ => {
            eprintln!("usage: hybrid_vision_parity MMPROJ.gguf IMAGE REF.bin [MAX_TOKENS]");
            std::process::exit(2);
        }
    };
    let d = MetalDevice::new()?;
    let model = VisionModel::load(&d, mmproj, max_tokens)?;
    let embedding = model.encode(&d, &std::fs::read(image)?)?;
    let ours = embedding.rows.to_f32();
    let raw = std::fs::read(reference)?;
    let word = |i: usize| i32::from_le_bytes(raw[4 * i..4 * i + 4].try_into().unwrap()) as usize;
    let (tokens, width) = (word(0), word(1));
    let theirs: Vec<f32> = raw[8..]
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect();
    println!(
        "ferrum: {} tokens (grid {:?}); llama.cpp: {tokens} x {width}",
        embedding.tokens(),
        embedding.grid
    );
    if ours.len() != theirs.len() {
        eprintln!("shapes differ");
        std::process::exit(1);
    }
    let stats = |a: &[f32], b: &[f32]| {
        let (mut dot, mut aa, mut bb, mut diff, mut max) = (0f64, 0f64, 0f64, 0f64, 0f32);
        for (x, y) in a.iter().zip(b) {
            dot += f64::from(x * y);
            aa += f64::from(x * x);
            bb += f64::from(y * y);
            diff += f64::from((x - y) * (x - y));
            max = max.max((x - y).abs());
        }
        (dot / (aa.sqrt() * bb.sqrt()), max, (diff / bb).sqrt())
    };
    let (cos, max, rel) = stats(&ours, &theirs);
    println!("all rows: cosine {cos:.6}, max |diff| {max:.4}, relative RMS {rel:.5}");
    let worst = (0..tokens)
        .map(|r| {
            (
                r,
                stats(
                    &ours[r * width..(r + 1) * width],
                    &theirs[r * width..(r + 1) * width],
                ),
            )
        })
        .min_by(|a, b| a.1.0.total_cmp(&b.1.0))
        .unwrap();
    println!(
        "worst row {}: cosine {:.6}, max |diff| {:.4}, relative RMS {:.5}",
        worst.0, worst.1.0, worst.1.1, worst.1.2
    );
    Ok(())
}
