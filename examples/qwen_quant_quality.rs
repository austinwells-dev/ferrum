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

fn teacher_forced_decode_logits(
    device: &MetalDevice,
    model: &ferrum::model::Transformer,
    ids: &[u32],
) -> Result<Vec<f32>> {
    if ids.len() < 2 {
        return Ok(Vec::new());
    }
    let (first_logits, mut cache) = model.forward_prefill_last(device, &ids[..1])?;
    let mut logits = first_logits.to_f32();
    for &token in &ids[1..ids.len() - 1] {
        logits.extend(model.forward_decode(device, token, &mut cache)?.to_f32());
    }
    Ok(logits)
}

fn quality_metrics(
    reference_logits: &[f32],
    actual_logits: &[f32],
    targets: &[u32],
    vocabulary: usize,
) -> Result<serde_json::Value> {
    if vocabulary == 0
        || !reference_logits.len().is_multiple_of(vocabulary)
        || actual_logits.len() != reference_logits.len()
        || targets.len() > reference_logits.len() / vocabulary
    {
        return Err(Error::Shape(
            "quality logits or targets have invalid dimensions".into(),
        ));
    }
    let positions = reference_logits.len() / vocabulary;
    let mut absolute_sum = 0.0f64;
    let mut squared_sum = 0.0f64;
    let mut max_absolute = 0.0f32;
    let mut dot = 0.0f64;
    let mut reference_norm = 0.0f64;
    let mut actual_norm = 0.0f64;
    let mut top1_matches = 0usize;
    let mut top5_overlap = 0usize;
    let mut reference_nll = 0.0f64;
    let mut actual_nll = 0.0f64;
    for position in 0..positions {
        let start = position * vocabulary;
        let reference = &reference_logits[start..start + vocabulary];
        let actual = &actual_logits[start..start + vocabulary];
        for (&left, &right) in reference.iter().zip(actual) {
            let delta = (left - right).abs();
            max_absolute = max_absolute.max(delta);
            absolute_sum += f64::from(delta);
            squared_sum += f64::from(delta) * f64::from(delta);
            dot += f64::from(left) * f64::from(right);
            reference_norm += f64::from(left) * f64::from(left);
            actual_norm += f64::from(right) * f64::from(right);
        }
        let reference_top = top_k(reference, 5);
        let actual_top = top_k(actual, 5);
        top1_matches += usize::from(reference_top[0] == actual_top[0]);
        top5_overlap += reference_top
            .iter()
            .filter(|id| actual_top.contains(id))
            .count();
        if let Some(&target) = targets.get(position) {
            reference_nll += nll(reference, target);
            actual_nll += nll(actual, target);
        }
    }
    let logit_count = (positions * vocabulary) as f64;
    let reference_perplexity = (reference_nll / targets.len() as f64).exp();
    let actual_perplexity = (actual_nll / targets.len() as f64).exp();
    Ok(serde_json::json!({
        "positions": positions,
        "teacher_forced_tokens": targets.len(),
        "vocabulary": vocabulary,
        "logit_max_abs_error": max_absolute,
        "logit_mean_abs_error": absolute_sum / logit_count,
        "logit_rmse": (squared_sum / logit_count).sqrt(),
        "logit_cosine_similarity": dot / (reference_norm.sqrt() * actual_norm.sqrt()),
        "top1_position_agreement": top1_matches as f64 / positions as f64,
        "top5_mean_overlap": top5_overlap as f64 / (positions * 5) as f64,
        "bf16_perplexity": reference_perplexity,
        "quantized_perplexity": actual_perplexity,
        "perplexity_ratio": actual_perplexity / reference_perplexity,
    }))
}

fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().collect();
    if args.len() != 3 {
        return Err(Error::Parameter(
            "usage: qwen_quant_quality BF16_CHECKPOINT_DIR QUANTIZED_GGUF".into(),
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

    let prefill_quality =
        quality_metrics(&bf16_logits, &quantized_logits, &bf16_ids[1..], vocabulary)?;
    // Feed the same ground-truth stream through one-token decode calls so this
    // report also measures the M=1 kernels used after the initial prompt.
    let bf16_decode_logits = teacher_forced_decode_logits(&device, &bf16_model, &bf16_ids)?;
    let quantized_decode_logits =
        teacher_forced_decode_logits(&device, &quantized.model, &quantized_ids)?;
    let decode_quality = quality_metrics(
        &bf16_decode_logits,
        &quantized_decode_logits,
        &bf16_ids[1..],
        vocabulary,
    )?;
    println!(
        "{}",
        serde_json::json!({
            "format": "GGUF",
            "quantized_repository": "Qwen/Qwen2.5-0.5B-Instruct-GGUF",
            "quantized_revision": "9217f5db79a29953eb74d5343926648285ec7e67",
            "quantized_file": gguf_path.file_name().and_then(|name| name.to_str()).unwrap_or("unknown"),
            "reference": format!("Qwen2.5-0.5B-Instruct BF16 safetensors revision {}", qwen::REVISION),
            "device": device.name(),
            "parameter_count": quantized.parameter_count,
            "bf16_retained_weight_bytes": bf16_model.weight_bytes(),
            "gguf_source_tensor_bytes": quantized.source_tensor_bytes,
            "quantized_retained_weight_bytes": quantized.model.weight_bytes(),
            "quantized_retained_packed_bytes": quantized.quantized_tensor_bytes,
            "corpus": corpus,
            "corpus_repetitions": 16,
            "token_ids": bf16_ids,
            "positions": positions,
            "vocabulary": vocabulary,
            "logit_max_abs_error": prefill_quality["logit_max_abs_error"],
            "logit_mean_abs_error": prefill_quality["logit_mean_abs_error"],
            "logit_rmse": prefill_quality["logit_rmse"],
            "logit_cosine_similarity": prefill_quality["logit_cosine_similarity"],
            "top1_position_agreement": prefill_quality["top1_position_agreement"],
            "top5_mean_overlap": prefill_quality["top5_mean_overlap"],
            "teacher_forced_tokens": prefill_quality["teacher_forced_tokens"],
            "bf16_perplexity": prefill_quality["bf16_perplexity"],
            "quantized_perplexity": prefill_quality["quantized_perplexity"],
            "perplexity_ratio": prefill_quality["perplexity_ratio"],
            "prefill_quality": prefill_quality,
            "cached_decode_quality": decode_quality,
        })
    );
    Ok(())
}
