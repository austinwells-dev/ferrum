//! Best-first MTP draft trees against MTP's own chain.
//! usage: hybrid_mtp_tree_check MODEL.gguf [--steps 64] [--depth 5] [--budget 15]
//!
//! Along a plain greedy generation, every step drafts MTP's chain and a
//! best-first tree from the same state. A tree pass computes each node from
//! its parent's hidden state with attention to its own ancestors only, at
//! position anchor + depth, so wherever the tree expanded the chain's path
//! its nodes must reproduce the chain's steps: the same top tokens with the
//! same logits. Bit for bit with FERRUM_HYBRID_EXACT=1
//! FERRUM_HYBRID_REFERENCE_ATTENTION=1 (row-independent kernels); otherwise
//! reported as the largest logit difference.
use ferrum::{
    MetalDevice, Result,
    hybrid::{
        self, HybridState, Output, Produced, SpecGeometry, draft::TreeSearch, mtp::Mtp,
        speculative::Drafter,
    },
};

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let get = |key: &str, default: &str| -> String {
        args.iter()
            .position(|a| a == key)
            .map_or(default.to_owned(), |i| args[i + 1].clone())
    };
    let steps: usize = get("--steps", "64").parse().expect("steps");
    let depth: usize = get("--depth", "5").parse().expect("depth");
    let budget: usize = get("--budget", "15").parse().expect("budget");
    let d = MetalDevice::new()?;
    let loaded = hybrid::load(&d, &args[1], 512, 1)?;
    let model = &loaded.model;
    let prompt = loaded.tokenizer.encode(
        "Write a Rust function that parses a comma-separated list of integers, \
         returning an error for malformed input, with unit tests.\n\n",
    )?;
    let capacity = prompt.len() + steps + 64;
    let mut mtp = Mtp::load(
        &d,
        &args[1],
        model,
        capacity,
        depth,
        None,
        budget,
        TreeSearch::BestFirst,
    )?;
    model.enable_speculation(
        &d,
        &SpecGeometry {
            rows: budget + 1,
            aux_layers: Vec::new(),
            final_hidden: true,
        },
    )?;
    let mut state = HybridState::new(&d, &model.config, capacity, model.max_forward_rows())?;
    let Produced::Token(mut anchor) = model.forward(&d, &mut state, &prompt, Output::Argmax)?
    else {
        unreachable!()
    };
    mtp.ingest(&d, model, 0, &prompt)?;
    let (mut misses, mut checked, mut followed, mut nodes, mut deepest) = (0, 0, 0, 0, 0);
    let (mut identical, mut compared, mut worst) = (0, 0, 0f32);
    for step in 0..steps {
        let pos = state.len();
        let (chain, chain_top) = mtp.draft_traced(&d, model, pos, anchor, depth)?;
        let (tree, tree_top) = mtp.draft_tree_traced(&d, model, pos, anchor, depth, budget)?;
        // The top 16 (token, logit) pairs, best first.
        let top16 = |c: &[(u32, f32)]| {
            let mut c = c.to_vec();
            c.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
            c.truncate(16);
            c
        };
        let mut compare = |k: usize, slot: usize| {
            compared += 1;
            let (a, b) = (top16(&chain_top[k]), top16(&tree_top[slot]));
            if a.iter()
                .zip(&b)
                .all(|(x, y)| x.0 == y.0 && x.1.to_bits() == y.1.to_bits())
            {
                identical += 1;
            }
            for (x, y) in a.iter().zip(&b) {
                worst = worst.max((x.1 - y.1).abs());
            }
        };
        compare(0, 0);
        assert!(tree.len() <= budget, "tree of {} nodes", tree.len());
        assert!(
            tree.parents
                .iter()
                .enumerate()
                .all(|(i, p)| p.is_none_or(|p| p < i)),
            "parents must come first"
        );
        nodes += tree.len();
        deepest = deepest.max(tree.depth());
        // Follow the chain through the tree while the tree expanded its nodes.
        let mut node = None;
        for (k, &token) in chain.iter().enumerate() {
            let children: Vec<usize> = (0..tree.len())
                .filter(|&i| tree.parents[i] == node)
                .collect();
            if children.is_empty() {
                break;
            }
            checked += 1;
            match children.iter().find(|&&i| tree.tokens[i] == token) {
                Some(&child) => {
                    followed += 1;
                    node = Some(child);
                    // The chain's step k + 1 ran on chain[k], as the child's pass did.
                    if !tree_top[child + 1].is_empty() && k + 1 < chain_top.len() {
                        compare(k + 1, child + 1);
                    }
                }
                None => {
                    misses += 1;
                    println!(
                        "step {step}: chain token {token} missing among {:?}",
                        children.iter().map(|&i| tree.tokens[i]).collect::<Vec<_>>()
                    );
                    break;
                }
            }
        }
        // Plain greedy step; the drafter ingests the committed token.
        let Produced::Token(next) = model.forward(&d, &mut state, &[anchor], Output::Argmax)?
        else {
            unreachable!()
        };
        mtp.ingest(&d, model, pos, &[anchor])?;
        anchor = next;
    }
    println!(
        "{steps} steps: trees of {:.1} nodes on average (deepest {deepest}); chain followed {followed}/{checked} expanded nodes, {misses} misses",
        nodes as f64 / steps as f64
    );
    println!(
        "anchor and expanded chain-path nodes vs chain steps: {identical}/{compared} with bit-identical top-16 logits, max |dlogit| {worst:.5}"
    );
    Ok(())
}
