//! Hybrid-engine geometry read from GGUF metadata (Qwen3.5 family, Flint, Spark).
#![forbid(unsafe_code)]
use crate::{
    Error, Result,
    loader::gguf::{GgufReader, MetadataValue},
};
use std::collections::BTreeMap;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Variant {
    /// `qwen35`: every layer has a dense SwiGLU feed-forward block.
    Dense,
    /// `qwen35moe`: routed experts plus a sigmoid-gated shared expert.
    Moe,
    /// `flint`: full attention on every layer (some sliding-window), learned
    /// sinks, dense feed-forward for the leading layers, then sigmoid-routed
    /// experts with an ungated shared expert.
    Flint,
    /// `spark2_5`: full attention on every layer (sliding-window except every
    /// fourth), fused QKV, a per-head sigmoid output gate, no QK norm, its own
    /// RoPE width and base on sliding-window layers, and a GEGLU feed-forward.
    Spark,
}

/// How one attention layer masks, rotates and gates its keys and queries.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AttnSpec {
    /// Sliding window in tokens: a key at `pos - key >= window` is masked.
    /// `None` attends to the whole context.
    pub window: Option<usize>,
    /// Rotary embedding on Q and K.
    pub rope: bool,
    /// Learned per-head sink logits (`attn_sinks`) join the softmax denominator.
    pub sinks: bool,
    /// Sigmoid output gate fused into `attn_q` (Qwen3.5 gated attention).
    pub gated: bool,
}

/// Storage format of the attention K/V cache.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum KvCache {
    /// Half precision: 2 bytes per element.
    #[default]
    F16,
    /// 8-bit values with one F16 scale per 32 elements (llama.cpp's q8_0
    /// scheme): 1.0625 bytes per element. A cache tensor holds every row's
    /// values first, then every row's scales.
    Q8,
}

impl KvCache {
    /// Bytes of one cached row of `width` elements.
    pub fn row_bytes(self, width: usize) -> usize {
        match self {
            Self::F16 => width * 2,
            Self::Q8 => width + width / 32 * 2,
        }
    }
    /// A zeroed, resident cache of `rows` rows. Q8 storage is an F16-typed
    /// byte blob of `[rows, row_bytes / 2]`, so `shape()[0]` is still the row count.
    pub fn alloc(self, d: &crate::MetalDevice, rows: usize, width: usize) -> Result<crate::Tensor> {
        crate::Tensor::zeros_resident(d, [rows, self.row_bytes(width) / 2], crate::DType::F16)
    }
    /// `FERRUM_KV_CACHE` (f16 or q8), else F16. An unparsable value is
    /// ignored with a warning.
    pub fn from_env() -> Self {
        match std::env::var("FERRUM_KV_CACHE") {
            Ok(v) => v.parse().unwrap_or_else(|e| {
                eprintln!("ignoring FERRUM_KV_CACHE: {e}");
                Self::F16
            }),
            Err(_) => Self::F16,
        }
    }
    pub fn name(self) -> &'static str {
        match self {
            Self::F16 => "f16",
            Self::Q8 => "q8",
        }
    }
}

impl std::fmt::Display for KvCache {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.name())
    }
}

impl std::str::FromStr for KvCache {
    type Err = Error;
    fn from_str(s: &str) -> Result<Self> {
        match s.to_ascii_lowercase().as_str() {
            "f16" => Ok(Self::F16),
            "q8" | "q8_0" => Ok(Self::Q8),
            other => Err(Error::Config(format!(
                "unknown KV cache format {other:?} (expected f16 or q8)"
            ))),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MoeConfig {
    pub experts: usize,
    pub experts_used: usize,
    pub expert_ffn: usize,
    pub shared_ffn: usize,
    /// Scores are sigmoid(logit) rather than softmax over all experts.
    pub sigmoid: bool,
    /// Selected weights are divided by their sum.
    pub norm_weights: bool,
    /// The shared expert has a sigmoid gate (`ffn_gate_inp_shexp`).
    pub shared_gate: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub struct HybridConfig {
    pub variant: Variant,
    pub name: String,
    pub vocab: usize,
    pub hidden: usize,
    /// Trunk layers executed per token. The trailing NextN/MTP block is excluded.
    pub layers: usize,
    /// Layers with full attention (the rest are Gated DeltaNet).
    pub attention_layers: Vec<bool>,
    /// Per-layer attention options, one entry per trunk layer.
    pub attn: Vec<AttnSpec>,
    /// Leading layers with a dense feed-forward block (all of them for
    /// `qwen35`, none for `qwen35moe`).
    pub leading_dense: usize,
    pub heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub rope_dims: usize,
    pub rope_theta: f32,
    /// RoPE width and base of sliding-window layers, when the GGUF gives them
    /// their own (`spark2_5`: 256 dims at a 10k base against 64 at 5M).
    pub swa_rope: Option<(usize, f32)>,
    /// Interleaved multi-axis RoPE sections (time, height, width, extra) used
    /// to position image tokens; zeros when the GGUF declares none.
    pub rope_sections: [usize; 4],
    pub eps: f32,
    /// Dense FFN width (unused by MoE layers).
    pub ffn: usize,
    /// Dense FFN activation is GELU (tanh) rather than SiLU: `gelu(gate) * up`.
    pub gelu_ffn: bool,
    pub conv_kernel: usize,
    /// Per-head state width of the delta rule (both K and V head dims).
    pub ssm_head_dim: usize,
    pub ssm_k_heads: usize,
    pub ssm_v_heads: usize,
    pub moe: Option<MoeConfig>,
    pub trained_context: usize,
    /// K/V cache format (a load option, not GGUF metadata).
    pub kv_cache: KvCache,
}

impl HybridConfig {
    pub fn from_gguf<R: std::io::Read + std::io::Seek>(file: &GgufReader<R>) -> Result<Self> {
        let md = file.metadata();
        let arch = string(md, "general.architecture")?;
        let variant = match arch {
            "qwen35" => Variant::Dense,
            "qwen35moe" => Variant::Moe,
            "flint" => Variant::Flint,
            "spark2_5" => Variant::Spark,
            other => {
                return Err(Error::Config(format!(
                    "GGUF architecture {other:?} is not a supported hybrid"
                )));
            }
        };
        let key = |suffix: &str| format!("{arch}.{suffix}");
        let blocks = uint(md, &key("block_count"))?;
        let nextn = match md.get(&key("nextn_predict_layers")) {
            Some(_) => uint(md, &key("nextn_predict_layers"))?,
            None => 0,
        };
        let layers = blocks
            .checked_sub(nextn)
            .filter(|&l| l > 0)
            .ok_or_else(|| Error::Config("no trunk layers".into()))?;
        let attention_layers = match variant {
            Variant::Flint | Variant::Spark => vec![true; layers],
            _ => match md.get(&key("attention.recurrent_layers")) {
                Some(MetadataValue::Array { values, .. }) => values
                    .iter()
                    .take(layers)
                    .map(|v| match v {
                        MetadataValue::Bool(recurrent) => Ok(!recurrent),
                        _ => Err(Error::Config("recurrent_layers must be booleans".into())),
                    })
                    .collect::<Result<Vec<_>>>()?,
                _ => {
                    let interval = match md.get(&key("full_attention_interval")) {
                        Some(_) => uint(md, &key("full_attention_interval"))?,
                        None => 4,
                    };
                    (0..layers).map(|i| (i + 1) % interval == 0).collect()
                }
            },
        };
        if attention_layers.len() != layers {
            return Err(Error::Config("recurrent layer mask is too short".into()));
        }
        let head_dim = uint(md, &key("attention.key_length"))?;
        if uint(md, &key("attention.value_length"))? != head_dim {
            return Err(Error::Config("K/V head dimensions differ".into()));
        }
        // Flint and Spark have no recurrent layers, so their SSM keys are
        // absent and unused.
        let (ssm_head_dim, ssm_v_heads, ssm_k_heads, conv_kernel) = match variant {
            Variant::Flint | Variant::Spark => (0, 0, 0, 4),
            _ => {
                let ssm_head_dim = uint(md, &key("ssm.state_size"))?;
                let ssm_v_heads = uint(md, &key("ssm.time_step_rank"))?;
                let ssm_k_heads = uint(md, &key("ssm.group_count"))?;
                if uint(md, &key("ssm.inner_size"))? != ssm_v_heads * ssm_head_dim {
                    return Err(Error::Config(
                        "ssm.inner_size must equal time_step_rank * state_size".into(),
                    ));
                }
                (
                    ssm_head_dim,
                    ssm_v_heads,
                    ssm_k_heads,
                    uint(md, &key("ssm.conv_kernel"))?,
                )
            }
        };
        let moe = match variant {
            Variant::Dense | Variant::Spark => None,
            Variant::Moe => Some(MoeConfig {
                experts: uint(md, &key("expert_count"))?,
                experts_used: uint(md, &key("expert_used_count"))?,
                expert_ffn: uint(md, &key("expert_feed_forward_length"))?,
                shared_ffn: uint(md, &key("expert_shared_feed_forward_length"))?,
                sigmoid: false,
                norm_weights: true,
                shared_gate: true,
            }),
            Variant::Flint => {
                let gating = uint(md, &key("expert_gating_func"))?;
                if gating != 2 {
                    return Err(Error::Config(format!(
                        "expert gating function {gating} is not sigmoid"
                    )));
                }
                // Neither a weight scale nor a selection bias on the scores is
                // implemented; refuse checkpoints that carry one rather than
                // route their experts wrongly.
                let scale_key = key("expert_weights_scale");
                if md.contains_key(&scale_key) && float(md, &scale_key)? != 1.0 {
                    return Err(Error::Config(format!(
                        "{scale_key} {} is unsupported (only 1.0)",
                        float(md, &scale_key)?
                    )));
                }
                if let Some(bias) = file
                    .tensors()
                    .keys()
                    .find(|n| n.ends_with(".exp_probs_b.bias"))
                {
                    return Err(Error::Config(format!(
                        "expert selection bias {bias} is unsupported"
                    )));
                }
                Some(MoeConfig {
                    experts: uint(md, &key("expert_count"))?,
                    experts_used: uint(md, &key("expert_used_count"))?,
                    expert_ffn: uint(md, &key("expert_feed_forward_length"))?,
                    shared_ffn: uint(md, &key("expert_shared_feed_forward_length"))?,
                    sigmoid: true,
                    norm_weights: flag(md, &key("expert_weights_norm"), false),
                    shared_gate: false,
                })
            }
        };
        let leading_dense = match variant {
            Variant::Dense | Variant::Spark => layers,
            Variant::Moe => 0,
            Variant::Flint => uint_or(md, &key("leading_dense_block_count"), 0)?,
        };
        let attn = attention_specs(md, &key, variant, layers)?;
        let embedding = file.tensor("token_embd.weight")?.logical_matrix_shape()?;
        let rope_dims = match variant {
            Variant::Flint | Variant::Spark => uint_or(md, &key("rope.dimension_count"), head_dim)?,
            Variant::Dense | Variant::Moe => uint(md, &key("rope.dimension_count"))?,
        };
        let rope_theta = float(md, &key("rope.freq_base"))?;
        // llama.cpp falls back to the global width and base for either key.
        let swa_rope = match variant {
            Variant::Spark => Some((
                uint_or(md, &key("rope.dimension_count_swa"), rope_dims)?,
                match md.get(&key("rope.freq_base_swa")) {
                    Some(_) => float(md, &key("rope.freq_base_swa"))?,
                    None => rope_theta,
                },
            )),
            _ => None,
        };
        let config = Self {
            variant,
            name: md
                .get("general.name")
                .and_then(|v| match v {
                    MetadataValue::String(s) => Some(s.clone()),
                    _ => None,
                })
                .unwrap_or_default(),
            vocab: embedding[0],
            hidden: uint(md, &key("embedding_length"))?,
            layers,
            attention_layers,
            attn,
            leading_dense,
            heads: uint(md, &key("attention.head_count"))?,
            kv_heads: uint(md, &key("attention.head_count_kv"))?,
            head_dim,
            rope_dims,
            rope_theta,
            swa_rope,
            rope_sections: rope_sections(md, &key("rope.dimension_sections")),
            eps: float(md, &key("attention.layer_norm_rms_epsilon"))?,
            ffn: match variant {
                Variant::Moe => 0,
                Variant::Dense | Variant::Flint | Variant::Spark => {
                    uint(md, &key("feed_forward_length"))?
                }
            },
            gelu_ffn: variant == Variant::Spark,
            conv_kernel,
            ssm_head_dim,
            ssm_k_heads,
            ssm_v_heads,
            moe,
            trained_context: uint(md, &key("context_length"))?,
            kv_cache: KvCache::F16,
        };
        config.validate()?;
        Ok(config)
    }

    fn validate(&self) -> Result<()> {
        let bad = |m: &str| Err(Error::Config(format!("unsupported hybrid geometry: {m}")));
        if embedding_mismatch(self) {
            return bad("embedding width");
        }
        let head_dim_ok = match self.variant {
            Variant::Flint | Variant::Spark => matches!(self.head_dim, 128 | 256),
            Variant::Dense | Variant::Moe => self.head_dim == 256,
        };
        let rope_ok = |dims: usize| dims <= self.head_dim && dims.is_multiple_of(2);
        if !head_dim_ok
            || !rope_ok(self.rope_dims)
            || self.swa_rope.is_some_and(|(dims, _)| !rope_ok(dims))
        {
            return bad("attention head dimension with even RoPE width");
        }
        if self.kv_heads == 0 || !self.heads.is_multiple_of(self.kv_heads) {
            return bad("grouped-query head ratio");
        }
        if self.recurrent_layer_count() > 0 {
            if self.conv_kernel != 4 {
                return bad("convolution kernel must be 4");
            }
            if self.ssm_head_dim != 128
                || self.ssm_k_heads == 0
                || !self.ssm_v_heads.is_multiple_of(self.ssm_k_heads)
            {
                return bad("delta-rule heads must be 128 wide with V heads a multiple of K heads");
            }
        }
        if self.attn.len() != self.layers || self.leading_dense > self.layers {
            return bad("per-layer attention options and dense layer count");
        }
        if !self.hidden.is_multiple_of(256)
            || (self.variant != Variant::Moe && !self.ffn.is_multiple_of(256))
        {
            return bad("projection widths must be multiples of 256");
        }
        if let Some(moe) = self.moe
            && (moe.experts == 0
                || moe.experts_used == 0
                || moe.experts_used > moe.experts
                || !moe.expert_ffn.is_multiple_of(256)
                || !moe.shared_ffn.is_multiple_of(256))
        {
            return bad("expert geometry");
        }
        Ok(())
    }

    pub fn is_attention(&self, layer: usize) -> bool {
        self.attention_layers[layer]
    }
    /// Whether `layer` runs the dense feed-forward (rather than experts).
    pub fn dense_ffn(&self, layer: usize) -> bool {
        layer < self.leading_dense
    }
    /// Query rows of `attn_q` per token: heads * head_dim, doubled when the
    /// output gate is fused into the projection.
    pub fn query_width(&self) -> usize {
        let gated = self.attn.iter().any(|a| a.gated);
        self.heads * self.head_dim * if gated { 2 } else { 1 }
    }
    pub fn attention_layer_count(&self) -> usize {
        self.attention_layers.iter().filter(|&&a| a).count()
    }
    pub fn recurrent_layer_count(&self) -> usize {
        self.layers - self.attention_layer_count()
    }
    /// Q and K channels of the delta-rule convolution input.
    pub fn ssm_key_dim(&self) -> usize {
        self.ssm_k_heads * self.ssm_head_dim
    }
    pub fn ssm_value_dim(&self) -> usize {
        self.ssm_v_heads * self.ssm_head_dim
    }
    pub fn conv_channels(&self) -> usize {
        2 * self.ssm_key_dim() + self.ssm_value_dim()
    }
    /// Bytes of K plus V for one cached position across all attention layers.
    pub fn kv_bytes_per_token(&self) -> usize {
        self.attention_layer_count() * 2 * self.kv_cache.row_bytes(self.kv_heads * self.head_dim)
    }
    /// Fixed recurrent state bytes (conv window + delta-rule matrices, F32).
    pub fn recurrent_state_bytes(&self) -> usize {
        let conv = (self.conv_kernel - 1) * self.conv_channels() * 4;
        let ssm = self.ssm_v_heads * self.ssm_head_dim * self.ssm_head_dim * 4;
        self.recurrent_layer_count() * (conv + ssm)
    }
}

/// Per-layer attention options. Flint and Spark mask layers flagged in
/// `sliding_window_pattern` to `sliding_window`, and Flint skips RoPE where
/// `rope_pattern` is false; Qwen3.5 has neither pattern and always gates Q.
/// (Spark's per-head gate is a separate weight, not the fused Q gate.)
fn attention_specs(
    md: &BTreeMap<String, MetadataValue>,
    key: &dyn Fn(&str) -> String,
    variant: Variant,
    layers: usize,
) -> Result<Vec<AttnSpec>> {
    let sliding = bool_array(md, &key("attention.sliding_window_pattern"), layers)?;
    let window = match sliding {
        Some(_) => Some(uint(md, &key("attention.sliding_window"))?),
        None => None,
    };
    let rope = bool_array(md, &key("attention.rope_pattern"), layers)?;
    Ok((0..layers)
        .map(|i| AttnSpec {
            window: sliding.as_ref().filter(|p| p[i]).and(window),
            rope: rope.as_ref().is_none_or(|p| p[i]),
            sinks: variant == Variant::Flint,
            gated: matches!(variant, Variant::Dense | Variant::Moe),
        })
        .collect())
}

fn bool_array(
    md: &BTreeMap<String, MetadataValue>,
    key: &str,
    layers: usize,
) -> Result<Option<Vec<bool>>> {
    match md.get(key) {
        None => Ok(None),
        Some(MetadataValue::Array { values, .. }) => {
            let flags = values
                .iter()
                .map(|v| match v {
                    MetadataValue::Bool(b) => Ok(*b),
                    _ => Err(Error::Config(format!("{key} must hold booleans"))),
                })
                .collect::<Result<Vec<_>>>()?;
            if flags.len() != layers {
                return Err(Error::Config(format!(
                    "{key} has {} entries for {layers} layers",
                    flags.len()
                )));
            }
            Ok(Some(flags))
        }
        Some(_) => Err(Error::Config(format!("{key} must be an array"))),
    }
}

/// A boolean metadata value, or `default` when the key is absent.
fn flag(md: &BTreeMap<String, MetadataValue>, key: &str, default: bool) -> bool {
    match md.get(key) {
        Some(MetadataValue::Bool(b)) => *b,
        _ => default,
    }
}

/// An unsigned metadata value, or `default` when the key is absent.
fn uint_or(md: &BTreeMap<String, MetadataValue>, key: &str, default: usize) -> Result<usize> {
    if md.contains_key(key) {
        uint(md, key)
    } else {
        Ok(default)
    }
}

fn embedding_mismatch(c: &HybridConfig) -> bool {
    c.hidden == 0 || c.vocab == 0
}

fn string<'a>(md: &'a BTreeMap<String, MetadataValue>, key: &str) -> Result<&'a str> {
    match md.get(key) {
        Some(MetadataValue::String(s)) => Ok(s),
        _ => Err(Error::Config(format!("missing GGUF metadata {key}"))),
    }
}

/// `rope.dimension_sections` as four section sizes (zeros when absent).
fn rope_sections(md: &BTreeMap<String, MetadataValue>, key: &str) -> [usize; 4] {
    let mut out = [0; 4];
    if let Some(MetadataValue::Array { values, .. }) = md.get(key) {
        for (o, v) in out.iter_mut().zip(values) {
            *o = match v {
                MetadataValue::Uint32(v) => *v as usize,
                MetadataValue::Int32(v) if *v >= 0 => *v as usize,
                MetadataValue::Uint64(v) => *v as usize,
                _ => 0,
            };
        }
    }
    out
}

pub(crate) fn uint(md: &BTreeMap<String, MetadataValue>, key: &str) -> Result<usize> {
    let value = match md.get(key) {
        Some(MetadataValue::Uint32(v)) => u64::from(*v),
        Some(MetadataValue::Uint64(v)) => *v,
        Some(MetadataValue::Int32(v)) if *v >= 0 => *v as u64,
        _ => return Err(Error::Config(format!("missing GGUF metadata {key}"))),
    };
    usize::try_from(value).map_err(|_| Error::Config(format!("{key} exceeds usize")))
}

fn float(md: &BTreeMap<String, MetadataValue>, key: &str) -> Result<f32> {
    match md.get(key) {
        Some(MetadataValue::Float32(v)) if v.is_finite() && *v > 0. => Ok(*v),
        Some(MetadataValue::Float64(v)) if v.is_finite() && *v > 0. => Ok(*v as f32),
        _ => Err(Error::Config(format!(
            "missing or invalid GGUF metadata {key}"
        ))),
    }
}
