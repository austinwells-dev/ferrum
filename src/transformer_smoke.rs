use ferrum::{
    DType, MetalDevice, Result,
    loader::Weights,
    model::{ModelConfig, Transformer, tiny},
    nn::attention::Trace,
    reference,
};
use std::time::{Duration, Instant};
fn delta(a: ferrum::metal::Counters, b: ferrum::metal::Counters) -> ferrum::metal::Counters {
    ferrum::metal::Counters {
        allocations: b.allocations - a.allocations,
        allocated_bytes: b.allocated_bytes - a.allocated_bytes,
        dispatches: b.dispatches - a.dispatches,
    }
}
pub fn run(d: &MetalDevice) -> Result<()> {
    let c = ModelConfig::tiny(DType::F32);
    let tokens = [3, 8, 4, 11, 2];
    println!(
        "Ferrum transformer smoke\ndevice: {}\ndtype: {:?}\nlayers: {}\nhidden: {}\nq heads: {}\nkv heads: {}\nsequence: {}",
        d.name(),
        c.dtype,
        c.num_layers,
        c.hidden_size,
        c.num_attention_heads,
        c.num_key_value_heads,
        tokens.len()
    );
    let w = tiny::weights(&c)?;
    let bytes = tiny::serialize(&c, &w)?;
    let load_start = Instant::now();
    let loaded = Weights::from_bytes(d, &bytes)?;
    let model = Transformer::from_weights(d, c.clone(), &loaded)?;
    println!(
        "safetensors load PASS ({:?}, includes projection transpose)",
        load_start.elapsed()
    );
    drop(loaded);
    let oracle = reference::transformer::forward(&c, &w, &tokens)?;
    let mut trace = Trace::new();
    let mut cache = model.new_cache()?;
    let full = model.forward(d, &tokens, &mut cache, Some(&mut trace))?;
    let mut max = 0f32;
    for (name, t) in &trace {
        let e = reference::check(&t.to_f32(), &oracle[name], 3e-5, 3e-5)?;
        max = max.max(e);
    }
    println!(
        "embedding / attention / MLP / decoder blocks / final logits PASS\n{} intermediate comparisons: max_abs={max:.8e}",
        trace.len()
    );
    drop(trace);
    let mut incremental = model.new_cache()?;
    let values = full.to_f32();
    let mut cached_max = 0f32;
    for (i, &token) in tokens.iter().enumerate() {
        let logits = model.forward_decode(d, token, &mut incremental)?;
        cached_max = cached_max.max(reference::check(
            &logits.to_f32(),
            &values[i * c.vocab_size..(i + 1) * c.vocab_size],
            3e-5,
            3e-5,
        )?);
    }
    println!("KV cache / cached decode equivalence PASS max_abs={cached_max:.8e}");
    // Compilation has completed. Ten independent samples, with identical cache lengths.
    let (_, prefix) = model.forward_prefill(d, &tokens[..4])?;
    let mut prefill = Vec::new();
    let mut decode = Vec::new();
    let mut pm = Default::default();
    let mut dm = Default::default();
    for _ in 0..10 {
        let before = d.counters();
        let start = Instant::now();
        let result = model.forward_prefill(d, &tokens)?;
        prefill.push(start.elapsed());
        pm = delta(before, d.counters());
        drop(result);
        let mut step = prefix.clone();
        let before = d.counters();
        let start = Instant::now();
        let result = model.forward_decode(d, tokens[4], &mut step)?;
        decode.push(start.elapsed());
        dm = delta(before, d.counters());
        drop(result);
    }
    fn median(v: &mut [Duration]) -> Duration {
        v.sort();
        v[v.len() / 2]
    }
    println!(
        "prefill median (10 samples): {:?}\ndecode token median (10 samples, 4 -> 5): {:?}",
        median(&mut prefill),
        median(&mut decode)
    );
    println!(
        "prefill: {} dispatches, {} allocations, {} allocated bytes",
        pm.dispatches, pm.allocations, pm.allocated_bytes
    );
    println!(
        "decode: {} dispatches, {} allocations, {} allocated bytes",
        dm.dispatches, dm.allocations, dm.allocated_bytes
    );
    println!(
        "weight memory: {} bytes\nKV cache memory (5 tokens): {} bytes\nKV cache capacity payload: {} bytes",
        model.weight_bytes(),
        cache.bytes(),
        2 * c.num_layers
            * c.max_context_length
            * c.num_key_value_heads
            * c.head_dim
            * c.dtype.size_bytes()
    );
    println!(
        "prefill temporary allocation volume: {} bytes\ndecode temporary allocation volume: {} bytes (cumulative allocated bytes minus returned logits and final active KV; not peak live memory)",
        pm.allocated_bytes - cache.bytes() - full.byte_size(),
        dm.allocated_bytes - cache.bytes() - c.vocab_size * c.dtype.size_bytes()
    );
    println!("Phase 2 transformer smoke: PASS");
    Ok(())
}
