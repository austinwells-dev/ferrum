//! Memory plan for each K/V cache format, from GGUF metadata only (no
//! weights are read): the largest context that fits and the K/V cost.
//! usage: hybrid_plan MODEL.gguf [MODEL.gguf ...]
use ferrum::{
    MetalDevice, Result,
    hybrid::{HybridConfig, KvCache, plan},
    loader::gguf::GgufFile,
};

fn main() -> Result<()> {
    let budget = MetalDevice::new()?.recommended_max_working_set() as usize;
    println!(
        "| model | K/V | KiB/token | auto context | K/V at auto context |\n|---|---|---|---|---|"
    );
    for path in std::env::args().skip(1) {
        let file = GgufFile::open(&path)?;
        for kv in [KvCache::F16, KvCache::Q8] {
            let mut config = HybridConfig::from_gguf(&file)?;
            config.kv_cache = kv;
            let p = plan::plan(
                &file,
                &config,
                budget,
                plan::PlanOptions {
                    kv_cache: kv,
                    ..Default::default()
                },
            )?;
            println!(
                "| {} | {kv} | {:.1} | {} | {:.2} GiB |",
                config.name,
                p.kv_per_token / 1024.,
                p.context,
                p.kv(p.context) as f64 / (1u64 << 30) as f64
            );
        }
    }
    Ok(())
}
