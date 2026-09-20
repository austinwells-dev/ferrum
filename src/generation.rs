//! CPU token selection and orchestration; all model arithmetic remains on Metal.
#![forbid(unsafe_code)]
use crate::{Error, MetalDevice, Result, Tensor};
/// Copy only the final [vocabulary] row using the existing checked copy operation.
pub fn final_logits(d: &MetalDevice, logits: &Tensor) -> Result<Vec<f32>> {
    let shape = logits.shape().dimensions();
    if shape.len() != 2 || shape.contains(&0) {
        return Err(Error::Shape(
            "logits require nonempty [sequence,vocabulary]".into(),
        ));
    }
    Ok(
        d.copy_range(logits, (shape[0] - 1) * shape[1], &[shape[1]])?
            .tensor
            .to_f32(),
    )
}
pub fn argmax(logits: &[f32]) -> Result<u32> {
    if logits.is_empty()
        || logits.len() > u32::MAX as usize
        || logits.iter().any(|x| !x.is_finite())
    {
        return Err(Error::Parameter(
            "argmax requires nonempty finite vocabulary logits".into(),
        ));
    }
    let mut best = 0;
    for i in 1..logits.len() {
        if logits[i] > logits[best] {
            best = i;
        }
    }
    Ok(best as u32)
}
pub fn validate_context(prompt: usize, max_new: usize, capacity: usize) -> Result<()> {
    if prompt == 0 {
        return Err(Error::Parameter(
            "prompt must contain at least one token".into(),
        ));
    }
    if prompt > capacity {
        return Err(Error::Parameter(format!(
            "prompt too long: {prompt} tokens, context capacity {capacity}"
        )));
    }
    if prompt.checked_add(max_new).is_none_or(|n| n > capacity) {
        return Err(Error::Parameter(format!(
            "context overflow: prompt {prompt} + max-new-tokens {max_new} exceeds {capacity}"
        )));
    }
    Ok(())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StopReason {
    Eos,
    MaxNewTokens,
}
pub fn stop_reason(token: u32, produced: usize, max_new: usize, eos: &[u32]) -> Option<StopReason> {
    if eos.contains(&token) {
        Some(StopReason::Eos)
    } else if produced >= max_new {
        Some(StopReason::MaxNewTokens)
    } else {
        None
    }
}
#[derive(Debug)]
pub struct Generation {
    pub tokens: Vec<u32>,
    pub stop: StopReason,
    pub prefill: std::time::Duration,
    pub first_token: std::time::Duration,
    pub decode: Vec<std::time::Duration>,
    pub sampling: std::time::Duration,
    pub prefill_counters: crate::metal::Counters,
    pub decode_counters: Vec<crate::metal::Counters>,
    pub kv_bytes: usize,
    pub prefill_profile: crate::metal::Profile,
    pub decode_profiles: Vec<crate::metal::Profile>,
}
pub fn counter_delta(
    a: crate::metal::Counters,
    b: crate::metal::Counters,
) -> crate::metal::Counters {
    crate::metal::Counters {
        allocations: b.allocations - a.allocations,
        allocated_bytes: b.allocated_bytes - a.allocated_bytes,
        dispatches: b.dispatches - a.dispatches,
        command_buffers: b.command_buffers - a.command_buffers,
        completion_waits: b.completion_waits - a.completion_waits,
        encode: b.encode - a.encode,
        wait: b.wait - a.wait,
        gpu: b.gpu - a.gpu,
        reused_bytes: b.reused_bytes - a.reused_bytes,
        transient_live_bytes: b.transient_live_bytes,
        transient_peak_bytes: b.transient_peak_bytes,
        arena_capacity: b.arena_capacity,
        arena_high_water: b.arena_high_water,
    }
}
/// The callback sees non-EOS IDs only. It may buffer incomplete UTF-8 sequences.
pub fn generate(
    d: &MetalDevice,
    model: &crate::model::Transformer,
    prompt: &[u32],
    max_new: usize,
    eos: &[u32],
    mut select: impl FnMut(&[f32]) -> Result<u32>,
    mut emit: impl FnMut(u32) -> Result<()>,
) -> Result<Generation> {
    use std::time::{Duration, Instant};
    validate_context(prompt.len(), max_new, model.config().max_context_length)?;
    if eos
        .iter()
        .any(|&id| id as usize >= model.config().vocab_size)
    {
        return Err(Error::Parameter("EOS ID outside model vocabulary".into()));
    }
    let mut result = Generation {
        tokens: Vec::new(),
        stop: StopReason::MaxNewTokens,
        prefill: Duration::ZERO,
        first_token: Duration::ZERO,
        decode: Vec::new(),
        sampling: Duration::ZERO,
        prefill_counters: Default::default(),
        decode_counters: Vec::new(),
        kv_bytes: 0,
        prefill_profile: Default::default(),
        decode_profiles: Vec::new(),
    };
    if max_new == 0 {
        return Ok(result);
    }
    let start = Instant::now();
    d.reset_transient_peak();
    let before = d.counters();
    let (logits, mut cache) = model.forward_prefill(d, prompt)?;
    result.prefill = start.elapsed();
    let mut values = final_logits(d, &logits)?;
    drop(logits);
    result.prefill_counters = counter_delta(before, d.counters());
    result.prefill_profile = d.take_profile();
    for step in 0..max_new {
        let sampling_start = Instant::now();
        let token = select(&values)?;
        result.sampling += sampling_start.elapsed();
        if token as usize >= model.config().vocab_size {
            return Err(Error::Token {
                id: token,
                vocab: model.config().vocab_size,
            });
        }
        if step == 0 {
            result.first_token = start.elapsed();
        }
        result.tokens.push(token);
        let stop = stop_reason(token, result.tokens.len(), max_new, eos);
        if !eos.contains(&token) {
            emit(token)?;
        }
        if let Some(reason) = stop {
            result.stop = reason;
            break;
        }
        d.reset_transient_peak();
        let before = d.counters();
        let start = Instant::now();
        let logits = model.forward_decode(d, token, &mut cache)?;
        values = final_logits(d, &logits)?;
        result.decode.push(start.elapsed());
        result.decode_profiles.push(d.take_profile());
        result
            .decode_counters
            .push(counter_delta(before, d.counters()));
    }
    result.kv_bytes = cache.bytes();
    Ok(result)
}
