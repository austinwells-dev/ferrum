//! Speculative verify/commit check with oracle drafts.
//! usage: hybrid_spec_check MODEL.gguf [--tokens 96] [--block 8]
//!
//! 1. Plain greedy decode after a fixed prompt gives the reference sequence.
//! 2. From the same prompt, a speculative loop drafts the reference
//!    continuation with every third draft block corrupted at a varying
//!    position (so full, partial and zero acceptance all occur), verifies,
//!    commits the accepted prefix and continues from the target's token.
//! 3. The replayed recurrent state is compared with one built by plain
//!    forwards: next-token logits after `commit(n)` versus after forwarding
//!    the same `n` tokens.
//!
//! Greedy speculative output must match the reference except at near-ties,
//! where single-row and multi-row kernels may round differently; any
//! divergence is reported with the reference logit margin at that position.
use ferrum::{
    MetalDevice, Result,
    hybrid::{self, HybridState, Output, Produced, RowOutput, RowsProduced, SpecGeometry},
};

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let get = |key: &str, default: &str| -> String {
        args.iter()
            .position(|a| a == key)
            .map_or(default.to_owned(), |i| args[i + 1].clone())
    };
    let n_new: usize = get("--tokens", "96").parse().expect("tokens");
    let block: usize = get("--block", "8").parse().expect("block");
    let d = MetalDevice::new()?;
    let loaded = hybrid::load(&d, &args[1], 512, 1)?;
    let model = &loaded.model;
    let prompt = loaded.tokenizer.encode(
        "The following is a detailed explanation of how a hash map works in \
         computer science, including hashing, buckets, collisions and resizing.\n\n",
    )?;
    let capacity = prompt.len() + n_new + block + 64;

    // Reference: plain greedy decode, keeping each step's top-2 margin.
    let mut state = HybridState::new(&d, &model.config, capacity)?;
    let Produced::Logits(mut logits) = model.forward(&d, &mut state, &prompt, Output::Logits)?
    else {
        unreachable!()
    };
    let mut reference = Vec::new();
    let mut margins = Vec::new();
    for _ in 0..n_new {
        let (best, margin) = top2(&logits);
        reference.push(best);
        margins.push(margin);
        let Produced::Logits(next) = model.forward(&d, &mut state, &[best], Output::Logits)? else {
            unreachable!()
        };
        logits = next;
    }

    // Speculative run with oracle drafts.
    model.enable_speculation(
        &d,
        &SpecGeometry {
            rows: block + 1,
            aux_layers: vec![1, model.config.layers / 2],
            final_hidden: true,
        },
    )?;
    let mut state = HybridState::new(&d, &model.config, capacity)?;
    let Produced::Token(mut anchor) = model.forward(&d, &mut state, &prompt, Output::Argmax)?
    else {
        unreachable!()
    };
    let mut out = vec![anchor];
    let (mut steps, mut accepted_total) = (0usize, 0usize);
    let mut corruptions = 0;
    while out.len() < n_new {
        let at = out.len();
        let k = block.min(n_new - at);
        let mut drafts: Vec<u32> = reference[at..at + k].to_vec();
        if steps % 3 == 1 {
            let bad = steps % (k + 1);
            if bad < k {
                drafts[bad] = drafts[bad].wrapping_add(17) % model.config.vocab as u32;
                corruptions += 1;
            }
        }
        let mut rows = vec![anchor];
        rows.extend(&drafts);
        let RowsProduced::Tokens(targets) =
            model.verify(&d, &mut state, &rows, RowOutput::Argmax)?
        else {
            unreachable!()
        };
        let accepted = drafts
            .iter()
            .zip(&targets)
            .take_while(|(a, b)| a == b)
            .count();
        model.commit(&d, &mut state, accepted + 1)?;
        out.extend(&drafts[..accepted]);
        anchor = targets[accepted];
        out.push(anchor);
        steps += 1;
        accepted_total += accepted;
    }
    out.truncate(n_new);
    let first_diff = out.iter().zip(&reference).position(|(a, b)| a != b);
    println!(
        "{steps} verify steps, {:.2} tokens/step, {corruptions} corrupted blocks",
        (accepted_total + steps) as f64 / steps as f64
    );
    match first_diff {
        None => println!("speculative output == greedy reference ({n_new} tokens)"),
        Some(i) => println!(
            "first divergence at token {i}: reference margin {:.4} (near-tie if small)",
            margins[i]
        ),
    }

    // Numerics: verify-row logits versus single-row decode logits at the
    // same positions (reference prefix, one full-width verify).
    {
        let rows = block + 1;
        let mut a = HybridState::new(&d, &model.config, capacity)?;
        model.forward(&d, &mut a, &prompt, Output::None)?;
        let RowsProduced::Logits(wide) =
            model.verify(&d, &mut a, &reference[..rows], RowOutput::Logits)?
        else {
            unreachable!()
        };
        let mut b = HybridState::new(&d, &model.config, capacity)?;
        model.forward(&d, &mut b, &prompt, Output::None)?;
        let (mut worst, mut kl_max) = (0f32, 0f64);
        for (i, row) in wide.iter().enumerate() {
            let Produced::Logits(single) =
                model.forward(&d, &mut b, &[reference[i]], Output::Logits)?
            else {
                unreachable!()
            };
            worst = worst.max(
                row.iter()
                    .zip(&single)
                    .fold(0f32, |m, (x, y)| m.max((x - y).abs())),
            );
            kl_max = kl_max.max(kl(&single, row));
        }
        println!(
            "verify rows vs single-row decode over {rows} positions: max |dlogit| {worst:.4}, max KL {kl_max:.2e}"
        );
    }

    // Replay check: commit(n) versus plain forwards of the same n tokens.
    let n = block / 2 + 1;
    let tail = &reference[..n];
    let mut a = HybridState::new(&d, &model.config, capacity)?;
    model.forward(&d, &mut a, &prompt, Output::None)?;
    model.verify(&d, &mut a, &reference[..block + 1], RowOutput::Argmax)?;
    model.commit(&d, &mut a, n)?;
    let Produced::Logits(la) = model.forward(&d, &mut a, &[reference[n]], Output::Logits)? else {
        unreachable!()
    };
    let mut b = HybridState::new(&d, &model.config, capacity)?;
    model.forward(&d, &mut b, &prompt, Output::None)?;
    model.forward(&d, &mut b, tail, Output::None)?;
    let Produced::Logits(lb) = model.forward(&d, &mut b, &[reference[n]], Output::Logits)? else {
        unreachable!()
    };
    let scale = lb.iter().fold(0f32, |m, v| m.max(v.abs()));
    let worst = la
        .iter()
        .zip(&lb)
        .fold(0f32, |m, (x, y)| m.max((x - y).abs()));
    println!(
        "replay after commit({n}) vs plain forward: max |dlogit| {worst:.4} (scale {scale:.1}), argmax {} vs {}",
        top2(&la).0,
        top2(&lb).0
    );
    Ok(())
}

fn top2(logits: &[f32]) -> (u32, f32) {
    let (mut best, mut second) = ((0u32, f32::NEG_INFINITY), f32::NEG_INFINITY);
    for (i, &v) in logits.iter().enumerate() {
        if v > best.1 {
            second = best.1;
            best = (i as u32, v);
        } else if v > second {
            second = v;
        }
    }
    (best.0, best.1 - second)
}

/// KL(p || q) of the softmax distributions of two logit rows.
fn kl(p: &[f32], q: &[f32]) -> f64 {
    let lse = |x: &[f32]| {
        let m = x.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b)) as f64;
        m + x.iter().map(|&v| (v as f64 - m).exp()).sum::<f64>().ln()
    };
    let (lp, lq) = (lse(p), lse(q));
    p.iter()
        .zip(q)
        .map(|(&a, &b)| {
            let pa = (a as f64 - lp).exp();
            pa * ((a as f64 - lp) - (b as f64 - lq))
        })
        .sum()
}
