//! Cost of a draft-tree verify against a chain verify of the same width.
//! usage: hybrid_tree_cost MODEL.gguf [--context 1024] [--widths 4,8,12,16] [--reps 7]
//!
//! After `context` cached positions, for each verify width m (rows,
//! anchor included) this times, median of `reps`:
//! - `verify` of an m-row chain (the Phase 9 kernels);
//! - `verify_tree` of a chain-shaped tree (the tree kernels, same work);
//! - `verify_tree` of an MTP-shaped tree (a 3-deep chain, the rest siblings);
//! - `commit` of 3 rows, and `commit_path` of a path through a sibling.
use ferrum::{
    MetalDevice, Result,
    hybrid::{self, HybridState, Output, RowOutput, SpecGeometry},
};
use std::time::Instant;

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let get = |key: &str, default: &str| -> String {
        args.iter()
            .position(|a| a == key)
            .map_or(default.to_owned(), |i| args[i + 1].clone())
    };
    let context: usize = get("--context", "1024").parse().expect("context");
    let widths: Vec<usize> = get("--widths", "4,8,12,16")
        .split(',')
        .map(|s| s.parse().expect("width"))
        .collect();
    let reps: usize = get("--reps", "7").parse().expect("reps");
    let d = MetalDevice::new()?;
    let loaded = hybrid::load(&d, &args[1], 512, 1)?;
    let model = &loaded.model;
    let widest = widths.iter().copied().max().unwrap_or(4);
    model.enable_speculation(
        &d,
        &SpecGeometry {
            rows: widest,
            aux_layers: Vec::new(),
            final_hidden: true,
        },
    )?;
    let prompt: Vec<u32> = (0..context as u32)
        .map(|i| 1000 + (i * 7919) % 20000)
        .collect();
    let mut state = HybridState::new(
        &d,
        &model.config,
        context + (reps + 2) * 4 * widths.len() * 2 + widest + 8,
        model.max_forward_rows(),
    )?;
    model.forward(&d, &mut state, &prompt, Output::None)?;
    let median = |mut v: Vec<f64>| {
        v.sort_by(f64::total_cmp);
        v[v.len() / 2]
    };
    let time = |f: &mut dyn FnMut() -> Result<()>| -> Result<f64> {
        let t = Instant::now();
        f()?;
        Ok(t.elapsed().as_secs_f64() * 1e3)
    };
    println!(
        "| rows | chain verify ms | tree kernels, chain shape | MTP-shaped tree | commit(3) | commit_path via a sibling |"
    );
    println!("|---|---|---|---|---|---|");
    for &m in &widths {
        let tokens: Vec<u32> = (0..m as u32).map(|i| 3000 + i * 13).collect();
        let chain: Vec<Option<usize>> = (0..m).map(|i| i.checked_sub(1)).collect();
        // Rows 1..=3 a chain; every later row a sibling under rows 0..=2.
        let mtp: Vec<Option<usize>> = (0..m)
            .map(|i| match i {
                0 => None,
                1..=3 => Some(i - 1),
                _ => Some((i - 4) % 3),
            })
            .collect();
        let (mut a, mut b, mut c, mut e, mut f) = (vec![], vec![], vec![], vec![], vec![]);
        for _ in 0..reps {
            a.push(time(&mut || {
                model.verify(&d, &mut state, &tokens, RowOutput::Argmax)?;
                Ok(())
            })?);
            b.push(time(&mut || {
                model.verify_tree(&d, &mut state, &tokens, &chain, RowOutput::Argmax)?;
                Ok(())
            })?);
            c.push(time(&mut || {
                model.verify_tree(&d, &mut state, &tokens, &mtp, RowOutput::Argmax)?;
                Ok(())
            })?);
            e.push(time(&mut || model.commit(&d, &mut state, 3))?);
            // A path through the first sibling under the anchor, when there is one.
            let path: Vec<usize> = if m > 4 { vec![0, 4] } else { vec![0, 1] };
            model.verify_tree(&d, &mut state, &tokens, &mtp, RowOutput::Argmax)?;
            f.push(time(&mut || model.commit_path(&d, &mut state, &path))?);
        }
        println!(
            "| {m} | {:.1} | {:.1} | {:.1} | {:.2} | {:.2} |",
            median(a),
            median(b),
            median(c),
            median(e),
            median(f)
        );
    }
    Ok(())
}
