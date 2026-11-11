//! Draft-tree verify / commit_path check with oracle trees.
//! usage: hybrid_tree_check MODEL.gguf [--tokens 96] [--budget 15]
//!
//! 1. Plain greedy decode after a fixed prompt gives the reference sequence.
//! 2. From the same prompt, a tree loop drafts `budget` nodes per step: the
//!    reference continuation lies on one random root-to-node path (every
//!    third step corrupted at a moving depth, so full, partial and zero
//!    acceptance occur), every other node is a decoy, and the shapes vary
//!    (deep, bushy, chain-like). The accept walk follows the child whose
//!    token is its parent row's argmax; `commit_path` keeps the walk.
//! 3. Alongside, a second state verifies each accepted path as a chain and
//!    commits all of it. Each path row's logits are compared with the
//!    chain's row: bit for bit with FERRUM_HYBRID_EXACT=1 and
//!    FERRUM_HYBRID_REFERENCE_ATTENTION=1 (row-independent kernels), and as
//!    max |dlogit| otherwise.
//! 4. Replay: next-token logits after `commit_path` versus plain forwards of
//!    the same tokens.
//!
//! Tree output must match the reference except at near-ties, where
//! single-row and multi-row kernels may round differently; a divergence is
//! reported with the reference's top-2 margin there.
use ferrum::{
    MetalDevice, Result,
    hybrid::{self, HybridState, Output, Produced, RowOutput, RowsProduced, SpecGeometry},
};

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
}

/// A draft tree of `nodes` rows (row 0 is the anchor): parents, and the path
/// from the root to one of the deepest nodes.
fn shape(rng: &mut Rng, nodes: usize) -> (Vec<Option<usize>>, Vec<usize>) {
    let style = rng.below(3);
    let mut parents = vec![None];
    let mut depth = vec![0usize];
    for i in 1..nodes {
        let p = match style {
            // Mostly a chain with side branches.
            0 if rng.below(4) > 0 => i - 1,
            // Bushy: anywhere.
            1 => rng.below(i),
            // Two-level fan-out, then chains.
            2 if i < nodes / 3 => 0,
            2 => i - 1 - rng.below(i.min(3)),
            _ => rng.below(i),
        };
        parents.push(Some(p));
        depth.push(depth[p] + 1);
    }
    let deepest = depth.iter().max().copied().unwrap_or(0);
    let candidates: Vec<usize> = (0..nodes).filter(|&i| depth[i] == deepest).collect();
    let mut node = candidates[rng.below(candidates.len())];
    let mut path = vec![node];
    while let Some(p) = parents[node] {
        path.push(p);
        node = p;
    }
    path.reverse();
    (parents, path)
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let get = |key: &str, default: &str| -> String {
        args.iter()
            .position(|a| a == key)
            .map_or(default.to_owned(), |i| args[i + 1].clone())
    };
    let n_new: usize = get("--tokens", "96").parse().expect("tokens");
    let budget: usize = get("--budget", "15").parse().expect("budget");
    let d = MetalDevice::new()?;
    let loaded = hybrid::load(&d, &args[1], 512, 1)?;
    let model = &loaded.model;
    let vocab = model.config.vocab as u32;
    let prompt = loaded.tokenizer.encode(
        "The following is a detailed explanation of how a hash map works in \
         computer science, including hashing, buckets, collisions and resizing.\n\n",
    )?;
    let capacity = prompt.len() + n_new + budget + 64;
    let new_state = || HybridState::new(&d, &model.config, capacity, model.max_forward_rows());

    // Reference: plain greedy decode, keeping each step's top-2 margin.
    let mut state = new_state()?;
    let Produced::Logits(mut logits) = model.forward(&d, &mut state, &prompt, Output::Logits)?
    else {
        unreachable!()
    };
    let mut reference = Vec::new();
    let mut margins = Vec::new();
    for _ in 0..n_new + budget + 1 {
        let (best, margin) = top2(&logits);
        reference.push(best);
        margins.push(margin);
        let Produced::Logits(next) = model.forward(&d, &mut state, &[best], Output::Logits)? else {
            unreachable!()
        };
        logits = next;
    }
    drop(state);

    model.enable_speculation(
        &d,
        &SpecGeometry {
            rows: budget + 1,
            aux_layers: vec![1, model.config.layers / 2],
            final_hidden: true,
        },
    )?;
    let mut tree_state = new_state()?;
    let mut chain_state = new_state()?;
    let Produced::Token(mut anchor) =
        model.forward(&d, &mut tree_state, &prompt, Output::Argmax)?
    else {
        unreachable!()
    };
    model.forward(&d, &mut chain_state, &prompt, Output::None)?;
    let mut rng = Rng(0x5eed_1234_abcd_0001);
    let mut out = vec![anchor];
    let (mut steps, mut accepted_total, mut corruptions) = (0usize, 0usize, 0usize);
    let (mut rows_compared, mut rows_identical, mut worst_chain) = (0usize, 0usize, 0f32);
    while out.len() < n_new {
        let at = out.len();
        let (parents, path) = shape(&mut rng, budget + 1);
        let mut tokens = vec![0u32; budget + 1];
        tokens[0] = anchor;
        let on_path = |i: usize| path.iter().position(|&p| p == i);
        let mut depth = vec![0usize; budget + 1];
        for i in 1..=budget {
            depth[i] = depth[parents[i].unwrap()] + 1;
            tokens[i] = match on_path(i) {
                Some(k) => reference[at + k - 1],
                None => {
                    // A decoy never repeats the reference token its depth would accept.
                    let mut t = (rng.next() % vocab as u64) as u32;
                    if t == reference[at + depth[i] - 1] {
                        t = (t + 1) % vocab;
                    }
                    t
                }
            };
        }
        if steps % 3 == 1 && path.len() > 1 {
            let k = 1 + steps % (path.len() - 1);
            tokens[path[k]] = (tokens[path[k]] + 17) % vocab;
            corruptions += 1;
        }
        let RowsProduced::Logits(rows) =
            model.verify_tree(&d, &mut tree_state, &tokens, &parents, RowOutput::Logits)?
        else {
            unreachable!()
        };
        // Accept walk: follow the child whose token is its parent row's argmax.
        let mut walk = vec![0usize];
        loop {
            let cur = *walk.last().unwrap();
            let want = argmax(&rows[cur]);
            match (1..=budget).find(|&i| parents[i] == Some(cur) && tokens[i] == want) {
                Some(child) => walk.push(child),
                None => break,
            }
        }
        let accepted: Vec<u32> = walk[1..].iter().map(|&i| tokens[i]).collect();
        let bonus = argmax(&rows[*walk.last().unwrap()]);
        model.commit_path(&d, &mut tree_state, &walk)?;

        // The same tokens as a chain verify, row for row.
        let mut chain_rows = vec![anchor];
        chain_rows.extend(&accepted);
        let RowsProduced::Logits(chain) =
            model.verify(&d, &mut chain_state, &chain_rows, RowOutput::Logits)?
        else {
            unreachable!()
        };
        model.commit(&d, &mut chain_state, chain_rows.len())?;
        for (k, &row) in walk.iter().enumerate() {
            let (a, b) = (&rows[row], &chain[k]);
            rows_compared += 1;
            if a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits()) {
                rows_identical += 1;
            }
            worst_chain =
                worst_chain.max(a.iter().zip(b).fold(0f32, |m, (x, y)| m.max((x - y).abs())));
        }

        out.extend(&accepted);
        anchor = bonus;
        out.push(anchor);
        steps += 1;
        accepted_total += accepted.len();
    }
    out.truncate(n_new);
    println!(
        "{steps} tree verifies of {budget} drafts, {:.2} tokens/step, {corruptions} corrupted paths",
        (accepted_total + steps) as f64 / steps as f64
    );
    match out.iter().zip(&reference).position(|(a, b)| a != b) {
        None => println!("tree output == greedy reference ({n_new} tokens)"),
        Some(i) => println!(
            "first divergence at token {i}: reference margin {:.4} (near-tie if small)",
            margins[i]
        ),
    }
    println!(
        "tree path rows vs chain rows: {rows_identical}/{rows_compared} bit-identical, max |dlogit| {worst_chain:.4}"
    );

    // Replay: a path that leaves the first rows (the reference under a decoy
    // first child), committed, then one more token, versus plain forwards.
    let mut a = new_state()?;
    model.forward(&d, &mut a, &prompt, Output::None)?;
    let n = 4;
    // Rows: 0 anchor, 1 decoy, 2.. the reference chain under the anchor.
    let mut tokens = vec![reference[0], (reference[1] + 1) % vocab];
    let mut parents = vec![None, Some(0)];
    for (k, &token) in reference.iter().enumerate().take(n).skip(1) {
        tokens.push(token);
        parents.push(Some(if k == 1 { 0 } else { k }));
    }
    model.verify_tree(&d, &mut a, &tokens, &parents, RowOutput::Argmax)?;
    let walk: Vec<usize> = std::iter::once(0).chain(2..n + 1).collect();
    model.commit_path(&d, &mut a, &walk)?;
    let Produced::Logits(la) = model.forward(&d, &mut a, &[reference[n]], Output::Logits)? else {
        unreachable!()
    };
    let mut b = new_state()?;
    model.forward(&d, &mut b, &prompt, Output::None)?;
    model.forward(&d, &mut b, &reference[..n], Output::None)?;
    let Produced::Logits(lb) = model.forward(&d, &mut b, &[reference[n]], Output::Logits)? else {
        unreachable!()
    };
    let scale = lb.iter().fold(0f32, |m, v| m.max(v.abs()));
    let worst = la
        .iter()
        .zip(&lb)
        .fold(0f32, |m, (x, y)| m.max((x - y).abs()));
    println!(
        "replay after commit_path({walk:?}) vs plain forward: max |dlogit| {worst:.4} (scale {scale:.1}), argmax {} vs {}",
        argmax(&la),
        argmax(&lb)
    );
    Ok(())
}

/// First maximal index, as `h_argmax` picks it.
fn argmax(logits: &[f32]) -> u32 {
    let mut best = 0;
    for (i, &v) in logits.iter().enumerate() {
        if v > logits[best] {
            best = i;
        }
    }
    best as u32
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
