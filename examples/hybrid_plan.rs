//! Memory plan for each K/V cache format, from GGUF metadata only (no
//! weights are read): the weights (and the routed experts among them), the
//! largest context that fits and the K/V cost. A model that doesn't fit
//! prints why instead of a context.
//! usage: hybrid_plan MODEL.gguf [MODEL.gguf ...]
use ferrum::{
    MetalDevice, Result,
    hybrid::{HybridConfig, KvCache, plan},
    loader::gguf::GgufFile,
};

fn main() -> Result<()> {
    let budget = MetalDevice::new()?.recommended_max_working_set() as usize;
    let gib = |b: usize| b as f64 / (1u64 << 30) as f64;
    println!(
        "| model | K/V | weights (experts) | KiB/token | auto context | K/V at auto context |\n|---|---|---|---|---|---|"
    );
    for path in std::env::args().skip(1) {
        let file = GgufFile::open(&path)?;
        for kv in [KvCache::F16, KvCache::Q8] {
            let mut config = HybridConfig::from_gguf(&file)?;
            config.kv_cache = kv;
            let weights = format!(
                "{:.2} GiB ({:.2})",
                gib(plan::weight_bytes(&file, &config)),
                gib(plan::routed_expert_bytes(&file, &config))
            );
            let options = plan::PlanOptions {
                kv_cache: kv,
                ..Default::default()
            };
            match plan::plan(&file, &config, budget, options) {
                Ok(p) => println!(
                    "| {} | {kv} | {weights} | {:.1} | {} | {:.2} GiB |",
                    config.name,
                    p.kv_per_token / 1024.,
                    p.context,
                    gib(p.kv(p.context))
                ),
                Err(e) => println!(
                    "| {} | {kv} | {weights} | | does not fit: {e} | |",
                    config.name
                ),
            }
        }
    }
    Ok(())
}
