//! Teacher-forced Phase 5 quality check: BF16 reference versus a GGUF model.
use ferrum::{
    Error, MetalDevice, Result,
    loader::Weights,
    model::{qwen, qwen_gguf},
    tokenizer::qwen::QwenTokenizer,
};

const CORPUS: &str = "The capital of France is Paris. Rust is a systems programming language with ownership and borrowing. Quantized weights store scale metadata with compact integer blocks.";

fn top_k(values: &[f32], k: usize) -> Vec<usize> {
    let mut indices: Vec<_> = (0..values.len()).collect();
    let order = |&left: &usize, &right: &usize| {
        values[right]
            .total_cmp(&values[left])
            .then_with(|| left.cmp(&right))
    };
    indices.select_nth_unstable_by(k - 1, order);
    indices[..k].sort_unstable_by(order);
    indices.truncate(k);
    indices
}

fn nll(values: &[f32], target: u32) -> f64 {
    let max = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let normalizer: f64 = values
        .iter()
        .map(|&value| f64::from((value - max).exp()))
        .sum();
    f64::from(max) + normalizer.ln() - f64::from(values[target as usize])
}

fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().collect();
    if args.len() != 3 {
        return Err(Error::Parameter(
            "usage: qwen_quant_quality BF16_CHECKPOINT_DIR Q8_0_GGUF".into(),
        ));
    }
    let bf16_dir = std::path::Path::new(&args[1]);
    let gguf_path = std::path::Path::new(&args[2]);
    let device = MetalDevice::new()?;

    let bf16_config = qwen::QwenConfig::from_file(bf16_dir.join("config.json"))?;
    let bf16_tokenizer = QwenTokenizer::load(bf16_dir, &bf16_config)?;
    let bf16_weights = Weights::from_file(&device, bf16_dir.join("model.safetensors"))?;
    let bf16_model = qwen::construct(&device, bf16_config.convert()?, &bf16_weights)?;
    drop(bf16_weights);

    let quantized = qwen_gguf::load(&device, gguf_path)?;
    let corpus = CORPUS.repeat(16);
    let (text, bf16_ids) = bf16_tokenizer.encode_prompt(&corpus, "", true)?;
    let (quantized_text, quantized_ids) = quantized.tokenizer.encode_prompt(&corpus, "", true)?;
    if text != quantized_text || bf16_ids != quantized_ids {
        return Err(Error::Tokenizer(
            "BF16 and GGUF tokenizers produced different corpus IDs".into(),
        ));
    }

    let bf16_logits = bf16_model.forward_prefill(&device, &bf16_ids)?.0.to_f32();
    let quantized_logits = quantized
        .model
        .forward_prefill(&device, &quantized_ids)?
        .0
        .to_f32();
    let vocabulary = bf16_model.config().vocab_size;
    let positions = bf16_ids.len();
    if quantized.model.config().vocab_size != vocabulary
        || bf16_logits.len() != positions * vocabulary
        || quantized_logits.len() != bf16_logits.len()
    {
        return Err(Error::Shape("quality probe logits shape mismatch".into()));
    }

    let mut absolute_sum = 0.0f64;
    let mut squared_sum = 0.0f64;
    let mut max_absolute = 0.0f32;
    let mut dot = 0.0f64;
    let mut bf16_norm = 0.0f64;
    let mut quantized_norm = 0.0f64;
    let mut top1_matches = 0usize;
    let mut top5_overlap = 0usize;
    let mut bf16_nll_sum = 0.0f64;
    let mut quantized_nll_sum = 0.0f64;
    for position in 0..positions {
        let start = position * vocabulary;
        let reference = &bf16_logits[start..start + vocabulary];
        let actual = &quantized_logits[start..start + vocabulary];
        for (&left, &right) in reference.iter().zip(actual) {
            let delta = (left - right).abs();
            max_absolute = max_absolute.max(delta);
            absolute_sum += f64::from(delta);
            squared_sum += f64::from(delta) * f64::from(delta);
            dot += f64::from(left) * f64::from(right);
            bf16_norm += f64::from(left) * f64::from(left);
            quantized_norm += f64::from(right) * f64::from(right);
        }
        let reference_top = top_k(reference, 5);
        let quantized_top = top_k(actual, 5);
        top1_matches += usize::from(reference_top[0] == quantized_top[0]);
        top5_overlap += reference_top
            .iter()
            .filter(|id| quantized_top.contains(id))
            .count();
        if position + 1 < positions {
            let target = bf16_ids[position + 1];
            bf16_nll_sum += nll(reference, target);
            quantized_nll_sum += nll(actual, target);
        }
    }
    let logit_count = (positions * vocabulary) as f64;
    let scored_tokens = positions.saturating_sub(1) as f64;
    let bf16_perplexity = (bf16_nll_sum / scored_tokens).exp();
    let quantized_perplexity = (quantized_nll_sum / scored_tokens).exp();
    println!(
        "{}",
        serde_json::json!({
            "format": "GGUF Q8_0",
            "quantized_repository": "Qwen/Qwen2.5-0.5B-Instruct-GGUF",
            "quantized_revision": "9217f5db79a29953eb74d5343926648285ec7e67",
            "quantized_file": "qwen2.5-0.5b-instruct-q8_0.gguf",
            "reference": format!("Qwen2.5-0.5B-Instruct BF16 safetensors revision {}", qwen::REVISION),
            "device": device.name(),
            "parameter_count": quantized.parameter_count,
            "bf16_retained_weight_bytes": bf16_model.weight_bytes(),
            "gguf_source_tensor_bytes": quantized.source_tensor_bytes,
            "q8_0_retained_weight_bytes": quantized.model.weight_bytes(),
            "q8_0_retained_packed_bytes": quantized.quantized_tensor_bytes,
            "corpus": corpus,
            "corpus_repetitions": 16,
            "token_ids": bf16_ids,
            "positions": positions,
            "vocabulary": vocabulary,
            "logit_max_abs_error": max_absolute,
            "logit_mean_abs_error": absolute_sum / logit_count,
            "logit_rmse": (squared_sum / logit_count).sqrt(),
            "logit_cosine_similarity": dot / (bf16_norm.sqrt() * quantized_norm.sqrt()),
            "top1_position_agreement": top1_matches as f64 / positions as f64,
            "top5_mean_overlap": top5_overlap as f64 / (positions * 5) as f64,
            "teacher_forced_tokens": positions.saturating_sub(1),
            "bf16_perplexity": bf16_perplexity,
            "q8_0_perplexity": quantized_perplexity,
            "perplexity_ratio": quantized_perplexity / bf16_perplexity,
        })
    );
    Ok(())
}
