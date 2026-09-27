//! Compare Ferrum's hybrid-engine logits with a llama.cpp
//! `llama-perplexity --kl-divergence-base` file for the same GGUF.
//!
//! usage: hybrid_kld MODEL.gguf BASE.kld [max_chunks] [--write OUT.kld]
//!
//! With `--write`, Ferrum's own logits for the base file's tokens are also
//! written in the same format, so llama-perplexity (or this tool) can be
//! scored against a Ferrum reference, e.g. one run with FERRUM_HYBRID_EXACT=1.
//!
//! For every chunk the model starts from an empty state and scores positions
//! n_ctx/2 .. n_ctx-2 exactly as llama-perplexity does. Reports Ferrum and
//! llama.cpp perplexity, mean/percentile KL divergence (base ‖ Ferrum), and
//! top-1 agreement.
#![allow(clippy::needless_range_loop)]
use ferrum::{
    MetalDevice, Result,
    hybrid::{self, HybridState},
};
use std::io::Read;

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("usage: hybrid_kld MODEL.gguf BASE.kld [max_chunks]");
        std::process::exit(2);
    }
    let max_chunks: usize = args
        .get(3)
        .map_or(usize::MAX, |v| v.parse().expect("chunk count"));
    let write_path = args
        .iter()
        .position(|a| a == "--write")
        .map(|i| args[i + 1].clone());
    let mut f = std::fs::File::open(&args[2]).expect("open base file");
    let mut bytes = Vec::new();
    f.read_to_end(&mut bytes).expect("read base file");
    assert_eq!(
        &bytes[..8],
        b"_logits_",
        "not a llama.cpp KL-divergence base file"
    );
    let u32_at = |o: usize| u32::from_le_bytes(bytes[o..o + 4].try_into().unwrap());
    let n_ctx = u32_at(8) as usize;
    let n_vocab = u32_at(12) as usize;
    let n_chunk = u32_at(16) as usize;
    let tokens: Vec<u32> = (0..n_ctx * n_chunk).map(|i| u32_at(20 + 4 * i)).collect();
    let mut offset = 20 + 4 * n_ctx * n_chunk;
    let nv = 2 * n_vocab.div_ceil(2) + 4;
    let first = n_ctx / 2;
    let scored = n_ctx - 1 - first;

    let device = MetalDevice::new()?;
    let rows = 64;
    let loaded = hybrid::load(&device, &args[1], rows, rows)?;
    let model = &loaded.model;
    assert_eq!(
        model.config.vocab, n_vocab,
        "vocabulary differs from base file"
    );
    let mut state = HybridState::new(&device, &model.config, n_ctx)?;

    let (mut nll, mut nll_base, mut kld_sum, mut same_top, mut count) =
        (0f64, 0f64, 0f64, 0usize, 0usize);
    let mut klds = Vec::new();
    let chunks_used = n_chunk.min(max_chunks);
    let mut written = write_path.as_ref().map(|_| {
        let mut out = Vec::new();
        out.extend_from_slice(b"_logits_");
        out.extend_from_slice(&(n_ctx as u32).to_le_bytes());
        out.extend_from_slice(&(n_vocab as u32).to_le_bytes());
        out.extend_from_slice(&(chunks_used as u32).to_le_bytes());
        for t in &tokens[..chunks_used * n_ctx] {
            out.extend_from_slice(&t.to_le_bytes());
        }
        out
    });
    let start = std::time::Instant::now();
    for chunk in 0..chunks_used {
        state.reset();
        let seq = &tokens[chunk * n_ctx..(chunk + 1) * n_ctx];
        let mut logits = Vec::with_capacity(n_ctx * n_vocab);
        for piece in seq.chunks(rows) {
            logits.extend(model.forward_all_logits(&device, &mut state, piece)?);
        }
        for i in 0..scored {
            let pos = first + i;
            let row = &logits[pos * n_vocab..(pos + 1) * n_vocab];
            let base = &bytes[offset + i * nv * 2..offset + (i + 1) * nv * 2];
            let base_u16 = |j: usize| u16::from_le_bytes([base[8 + 2 * j], base[9 + 2 * j]]);
            let scale = f32::from_le_bytes(base[0..4].try_into().unwrap());
            let min_log_prob = f32::from_le_bytes(base[4..8].try_into().unwrap());
            let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let log_sum: f64 = row
                .iter()
                .map(|&x| ((x - max) as f64).exp())
                .sum::<f64>()
                .ln();
            let target = seq[pos + 1] as usize;
            if let Some(out) = written.as_mut() {
                // llama-perplexity's encoding: scale, min log-prob, then u16 per token.
                let min_logit = row
                    .iter()
                    .copied()
                    .fold(f32::INFINITY, f32::min)
                    .max(max - 16.);
                let min_log_prob = min_logit - max - log_sum as f32;
                let scale = (max - min_logit) / 65535.;
                out.extend_from_slice(&scale.to_le_bytes());
                out.extend_from_slice(&min_log_prob.to_le_bytes());
                for &x in row {
                    let q = if scale > 0. && x > min_logit {
                        ((x - min_logit) / scale).round() as u16
                    } else {
                        0
                    };
                    out.extend_from_slice(&q.to_le_bytes());
                }
                if n_vocab % 2 == 1 {
                    out.extend_from_slice(&0u16.to_le_bytes());
                }
            }
            nll += log_sum - (row[target] - max) as f64;
            nll_base -= (scale * base_u16(target) as f32 + min_log_prob) as f64;
            let mut kl = 0f64;
            let (mut top, mut top_base) = (0usize, 0usize);
            let (mut top_v, mut top_base_v) = (f32::NEG_INFINITY, u16::MIN);
            for j in 0..n_vocab {
                // Same truncation as llama-perplexity: the base file clamps
                // log-probabilities below max-16, so those terms are skipped.
                let lp_base = scale * base_u16(j) as f32 + min_log_prob;
                if lp_base > -16. {
                    let lp = (row[j] - max) as f64 - log_sum;
                    kl += (lp_base as f64).exp() * (lp_base as f64 - lp);
                }
                if row[j] > top_v {
                    top_v = row[j];
                    top = j;
                }
                if base_u16(j) > top_base_v {
                    top_base_v = base_u16(j);
                    top_base = j;
                }
            }
            kld_sum += kl;
            klds.push(kl);
            same_top += usize::from(top == top_base);
            count += 1;
        }
        offset += scored * nv * 2;
        eprintln!(
            "chunk {chunk}: ppl {:.4} (llama.cpp {:.4}) mean KLD {:.6} same top {:.2}%",
            (nll / count as f64).exp(),
            (nll_base / count as f64).exp(),
            kld_sum / count as f64,
            100. * same_top as f64 / count as f64
        );
    }
    if let (Some(path), Some(out)) = (&write_path, written) {
        std::fs::write(path, out).expect("write base file");
        eprintln!("wrote {path}");
    }
    klds.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let pct = |p: f64| klds[((klds.len() - 1) as f64 * p) as usize];
    println!(
        "positions {count}; Ferrum PPL {:.4}; llama.cpp PPL {:.4}; mean KLD {:.6}; median {:.6}; p99 {:.6}; max {:.6}; same top-1 {:.2}%; {:.1} s",
        (nll / count as f64).exp(),
        (nll_base / count as f64).exp(),
        kld_sum / count as f64,
        pct(0.5),
        pct(0.99),
        klds[klds.len() - 1],
        100. * same_top as f64 / count as f64,
        start.elapsed().as_secs_f64()
    );
    Ok(())
}
