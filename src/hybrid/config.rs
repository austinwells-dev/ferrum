//! Hybrid-engine geometry read from GGUF metadata (Qwen3.5 family, Flint, Spark,
//! LFM3, and the not yet runnable Qwen3.8-Flash-Next).
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
    /// `lfm3`: short convolution on some layers, sliding-window and full
    /// attention on the rest, sandwich RMSNorms, and partial RoPE on the
    /// full-attention layers via a frequency-factor vector.
    Lfm3,
    /// `qwen4exp` (Qwen3.8-Flash-Next): the `qwen35moe` layout plus four-stream
    /// hyper-connections, indexer-selected sparse attention and an n-gram
    /// embedding on layer 1 (`Qwen4ExpConfig`). Parsed and planned only: the
    /// loader refuses it (docs/history/qwen4exp-plan.md).
    Qwen4Exp,
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

/// What `qwen4exp` adds to the `qwen35moe` geometry, from its GGUF metadata.
#[derive(Debug, Clone, PartialEq)]
pub struct Qwen4ExpConfig {
    /// Residual streams (`hyper_connection.count`): the residual is
    /// `hc_streams * hidden` wide, and every sublayer reads one mix of them.
    pub hc_streams: usize,
    /// Width of each hyper-connection's low-rank gate (`hyper_connection.low_rank`).
    pub hc_low_rank: usize,
    pub indexer: IndexerConfig,
    pub ple: PleConfig,
}

/// The sparse-attention indexer on the full-attention layers: each query
/// attends to the `top_k` cells whose `block`-cell group scores highest.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct IndexerConfig {
    pub heads: usize,
    pub key_dim: usize,
    /// Cells pooled into one indexer key (`attention.compress_ratios`).
    pub block: usize,
    pub top_k: usize,
}

/// The per-layer n-gram embedding ("PLE"): for each token, `ngram - 1`
/// n-gram orders times `heads_per_ngram` hashed rows of `row_dim` values,
/// read from `per_layer_token_embd.weight` in the second GGUF shard.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PleConfig {
    /// Layers that add the embedding to their residual (`[1]`).
    pub layers: Vec<usize>,
    /// Longest n-gram (3: bigrams and trigrams).
    pub ngram: usize,
    pub heads_per_ngram: usize,
    /// Values per table row (`embedding_length_per_layer_input`).
    pub row_dim: usize,
    pub conv_kernel: usize,
    /// A token that cuts the n-gram context (with everything before it).
    pub eos: u32,
    /// Per-position hash multipliers, one per n-gram position.
    pub multipliers: Vec<u64>,
    /// Each head's table size (primes, not powers of two) and first row.
    pub vocab_sizes: Vec<u64>,
    pub offsets: Vec<u64>,
}

impl PleConfig {
    pub fn heads(&self) -> usize {
        (self.ngram - 1) * self.heads_per_ngram
    }
    /// Rows the table must hold: the last head's offset plus its size.
    pub fn table_rows(&self) -> u64 {
        self.offsets.last().copied().unwrap_or(0) + self.vocab_sizes.last().copied().unwrap_or(0)
    }
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
    /// Delta-rule V head h reads K head h / (v_heads / k_heads), as in Hugging
    /// Face checkpoints; GGUF conversions reorder the V heads so it reads
    /// h % k_heads.
    pub ssm_grouped_heads: bool,
    pub moe: Option<MoeConfig>,
    pub qwen4exp: Option<Qwen4ExpConfig>,
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
            "lfm3" => Variant::Lfm3,
            "qwen4exp" => Variant::Qwen4Exp,
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
        if variant == Variant::Lfm3 {
            if nextn != 0 {
                return Err(Error::Config("lfm3 NextN blocks are not supported".into()));
            }
            if let Some(name) = file.tensors().keys().find(|n| {
                n.contains("ffn_gate_exps") || n.contains("ffn_up_exps") || n.ends_with(".bias")
            }) {
                return Err(Error::Config(format!(
                    "lfm3 expert or bias tensors are not supported ({name})"
                )));
            }
        }
        // LFM3 stores KV heads per layer (0 on short-conv layers). One shared
        // count is enough when every attention layer agrees.
        let kv_heads_array = if variant == Variant::Lfm3 {
            Some(i32_array(md, &key("attention.head_count_kv"), layers)?)
        } else {
            None
        };
        let attention_layers = match variant {
            Variant::Flint | Variant::Spark => vec![true; layers],
            Variant::Lfm3 => kv_heads_array
                .as_ref()
                .expect("lfm3 kv-head array")
                .iter()
                .enumerate()
                .map(|(i, &n)| {
                    if n < 0 {
                        Err(Error::Config(format!(
                            "lfm3.attention.head_count_kv[{i}] is negative"
                        )))
                    } else {
                        Ok(n > 0)
                    }
                })
                .collect::<Result<Vec<_>>>()?,
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
            Variant::Lfm3 => {
                let kernel = uint(md, &key("shortconv.l_cache"))?;
                if !(2..=9).contains(&kernel) {
                    return Err(Error::Config(format!(
                        "shortconv.l_cache {kernel} is outside 2..=9"
                    )));
                }
                (0, 0, 0, kernel)
            }
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
            Variant::Dense | Variant::Spark | Variant::Lfm3 => None,
            Variant::Moe | Variant::Qwen4Exp => Some(MoeConfig {
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
            Variant::Dense | Variant::Spark | Variant::Lfm3 => layers,
            Variant::Moe | Variant::Qwen4Exp => 0,
            Variant::Flint => uint_or(md, &key("leading_dense_block_count"), 0)?,
        };
        let attn = attention_specs(md, &key, variant, layers)?;
        if variant == Variant::Lfm3 {
            for (i, spec) in attn.iter().enumerate() {
                if spec.window.is_some() && !attention_layers[i] {
                    return Err(Error::Config("sliding window on a short-conv layer".into()));
                }
            }
        }
        let kv_heads = match &kv_heads_array {
            Some(counts) => {
                let mut heads = None;
                for &n in counts {
                    if n > 0 {
                        let n = n as usize;
                        if let Some(prev) = heads
                            && prev != n
                        {
                            return Err(Error::Config(
                                "lfm3 attention layers disagree on KV heads".into(),
                            ));
                        }
                        heads = Some(n);
                    }
                }
                heads.ok_or_else(|| Error::Config("lfm3 has no attention layer".into()))?
            }
            None => uint(md, &key("attention.head_count_kv"))?,
        };
        let embedding = file.tensor("token_embd.weight")?.logical_matrix_shape()?;
        let rope_dims = match variant {
            Variant::Flint | Variant::Spark => uint_or(md, &key("rope.dimension_count"), head_dim)?,
            Variant::Dense | Variant::Moe | Variant::Lfm3 | Variant::Qwen4Exp => {
                uint(md, &key("rope.dimension_count"))?
            }
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
            // Sliding layers rotate the full head; full-attention layers keep
            // `rope_theta` and divide by `rope_freqs`.
            Variant::Lfm3 => Some((rope_dims, float(md, &key("rope.freq_base_swa"))?)),
            _ => None,
        };
        let qwen4exp = match variant {
            Variant::Qwen4Exp => Some(qwen4exp(md, &key, &attention_layers)?),
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
            kv_heads,
            head_dim,
            rope_dims,
            rope_theta,
            swa_rope,
            rope_sections: rope_sections(md, &key("rope.dimension_sections")),
            eps: float(md, &key("attention.layer_norm_rms_epsilon"))?,
            ffn: match variant {
                Variant::Moe | Variant::Qwen4Exp => 0,
                Variant::Dense | Variant::Flint | Variant::Spark | Variant::Lfm3 => {
                    uint(md, &key("feed_forward_length"))?
                }
            },
            gelu_ffn: variant == Variant::Spark,
            conv_kernel,
            ssm_head_dim,
            ssm_k_heads,
            ssm_v_heads,
            ssm_grouped_heads: false,
            moe,
            qwen4exp,
            trained_context: uint(md, &key("context_length"))?,
            kv_cache: KvCache::F16,
        };
        config.validate()?;
        Ok(config)
    }

    pub(crate) fn validate(&self) -> Result<()> {
        let bad = |m: &str| Err(Error::Config(format!("unsupported hybrid geometry: {m}")));
        if embedding_mismatch(self) {
            return bad("embedding width");
        }
        let head_dim_ok = match self.variant {
            Variant::Flint | Variant::Spark => matches!(self.head_dim, 128 | 256),
            Variant::Lfm3 => self.head_dim == 128 && self.rope_dims == self.head_dim,
            Variant::Dense | Variant::Moe | Variant::Qwen4Exp => self.head_dim == 256,
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
        if self.variant == Variant::Lfm3 {
            if !(2..=9).contains(&self.conv_kernel) {
                return bad("short convolution kernel must be 2..=9");
            }
        } else if self.recurrent_layer_count() > 0 {
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
        if !self.hidden.is_multiple_of(256) || !self.ffn.is_multiple_of(256) {
            return bad("projection widths must be multiples of 256");
        }
        // Expert widths only feed projections (Qwen3.8's are 640).
        let step = super::weights::PROJECTION_K_STEP;
        if let Some(moe) = self.moe
            && (moe.experts == 0
                || moe.experts_used == 0
                || moe.experts_used > moe.experts
                || !moe.expert_ffn.is_multiple_of(step)
                || !moe.shared_ffn.is_multiple_of(step))
        {
            return bad("expert geometry");
        }
        if let Some(q) = &self.qwen4exp {
            let ple = &q.ple;
            if q.hc_streams == 0
                || !q.hc_low_rank.is_multiple_of(step)
                || q.indexer.heads == 0
                || q.indexer.block == 0
                || q.indexer.top_k == 0
                || ple.ngram < 2
                || ple.heads() * ple.row_dim != self.hidden
                || ple.multipliers.len() != ple.ngram
                || ple.vocab_sizes.len() != ple.heads()
                || ple.offsets.len() != ple.heads()
                || ple.layers.iter().any(|&l| l >= self.layers)
            {
                return bad("qwen4exp hyper-connection, indexer or n-gram geometry");
            }
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
    /// LFM3 keeps only the short-conv window, `(l_cache - 1) * hidden` per
    /// conv layer, and no delta-rule matrix.
    pub fn recurrent_state_bytes(&self) -> usize {
        if self.variant == Variant::Lfm3 {
            let conv = (self.conv_kernel - 1) * self.hidden * 4;
            return self.recurrent_layer_count() * conv;
        }
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
            gated: matches!(variant, Variant::Dense | Variant::Moe | Variant::Qwen4Exp),
        })
        .collect())
}

/// The `qwen4exp` additions. The full-attention layers are given twice, by
/// `full_attention_interval` and by the non-zero `attention.compress_ratios`
/// entries (each the indexer's block), and the two must agree.
fn qwen4exp(
    md: &BTreeMap<String, MetadataValue>,
    key: &dyn Fn(&str) -> String,
    attention_layers: &[bool],
) -> Result<Qwen4ExpConfig> {
    let ratios = u64_array(md, &key("attention.compress_ratios"))?;
    if ratios.len() != attention_layers.len()
        || ratios
            .iter()
            .zip(attention_layers)
            .any(|(&r, &a)| (r != 0) != a)
    {
        return Err(Error::Config(
            "attention.compress_ratios disagrees with full_attention_interval".into(),
        ));
    }
    let mut blocks = ratios.iter().filter(|&&r| r != 0);
    let block = blocks.next().copied().unwrap_or(0);
    if blocks.any(|&r| r != block) {
        return Err(Error::Config(
            "attention.compress_ratios differ between layers".into(),
        ));
    }
    let to_usize = |v: Vec<u64>| v.into_iter().map(|v| v as usize).collect();
    Ok(Qwen4ExpConfig {
        hc_streams: uint(md, &key("hyper_connection.count"))?,
        hc_low_rank: uint(md, &key("hyper_connection.low_rank"))?,
        indexer: IndexerConfig {
            heads: uint(md, &key("attention.indexer.head_count"))?,
            key_dim: uint(md, &key("attention.indexer.key_length"))?,
            block: block as usize,
            top_k: uint(md, &key("attention.indexer.top_k"))?,
        },
        ple: PleConfig {
            layers: to_usize(u64_array(md, &key("ple.layers"))?),
            ngram: uint(md, &key("ple.ngram_size"))?,
            heads_per_ngram: uint(md, &key("ple.heads_per_ngram"))?,
            row_dim: uint(md, &key("embedding_length_per_layer_input"))?,
            conv_kernel: uint(md, &key("ple.conv_kernel"))?,
            eos: u32::try_from(uint(md, &key("ple.eos_token_id"))?)
                .map_err(|_| Error::Config("ple.eos_token_id exceeds u32".into()))?,
            multipliers: u64_array(md, &key("ple.layer_multipliers"))?,
            vocab_sizes: u64_array(md, &key("ple.head_vocab_sizes"))?,
            offsets: u64_array(md, &key("ple.head_offsets"))?,
        },
    })
}

/// A metadata array of non-negative integers of any width.
fn u64_array(md: &BTreeMap<String, MetadataValue>, key: &str) -> Result<Vec<u64>> {
    let Some(MetadataValue::Array { values, .. }) = md.get(key) else {
        return Err(Error::Config(format!("missing GGUF metadata array {key}")));
    };
    values
        .iter()
        .map(|v| match *v {
            MetadataValue::Uint8(n) => Ok(u64::from(n)),
            MetadataValue::Uint16(n) => Ok(u64::from(n)),
            MetadataValue::Uint32(n) => Ok(u64::from(n)),
            MetadataValue::Uint64(n) => Ok(n),
            MetadataValue::Int32(n) if n >= 0 => Ok(n as u64),
            MetadataValue::Int64(n) if n >= 0 => Ok(n as u64),
            _ => Err(Error::Config(format!(
                "{key} must hold non-negative integers"
            ))),
        })
        .collect()
}

fn i32_array(md: &BTreeMap<String, MetadataValue>, key: &str, layers: usize) -> Result<Vec<i32>> {
    let MetadataValue::Array { values, .. } = md
        .get(key)
        .ok_or_else(|| Error::Config(format!("missing GGUF metadata {key}")))?
    else {
        return Err(Error::Config(format!("{key} must be an array")));
    };
    let counts = values
        .iter()
        .map(|v| match v {
            MetadataValue::Int32(n) => Ok(*n),
            MetadataValue::Uint32(n) => i32::try_from(*n)
                .map_err(|_| Error::Config(format!("{key} value does not fit i32"))),
            _ => Err(Error::Config(format!("{key} must hold integers"))),
        })
        .collect::<Result<Vec<_>>>()?;
    if counts.len() != layers {
        return Err(Error::Config(format!(
            "{key} has {} entries for {layers} layers",
            counts.len()
        )));
    }
    Ok(counts)
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    enum Meta {
        U32(u32),
        F32(f32),
        Str(&'static str),
        U64s(Vec<u64>),
    }

    /// A GGUF holding `md` and one F32 `token_embd.weight` of `[vocab, hidden]`.
    fn gguf(md: &[(String, Meta)], vocab: usize, hidden: usize) -> Vec<u8> {
        let s = |b: &mut Vec<u8>, v: &str| {
            b.extend_from_slice(&(v.len() as u64).to_le_bytes());
            b.extend_from_slice(v.as_bytes());
        };
        let mut b = b"GGUF".to_vec();
        b.extend_from_slice(&3u32.to_le_bytes());
        b.extend_from_slice(&1u64.to_le_bytes());
        b.extend_from_slice(&(md.len() as u64).to_le_bytes());
        for (k, v) in md {
            s(&mut b, k);
            match v {
                Meta::U32(x) => {
                    b.extend_from_slice(&4u32.to_le_bytes());
                    b.extend_from_slice(&x.to_le_bytes());
                }
                Meta::F32(x) => {
                    b.extend_from_slice(&6u32.to_le_bytes());
                    b.extend_from_slice(&x.to_le_bytes());
                }
                Meta::Str(x) => {
                    b.extend_from_slice(&8u32.to_le_bytes());
                    s(&mut b, x);
                }
                Meta::U64s(xs) => {
                    b.extend_from_slice(&9u32.to_le_bytes());
                    b.extend_from_slice(&10u32.to_le_bytes());
                    b.extend_from_slice(&(xs.len() as u64).to_le_bytes());
                    for x in xs {
                        b.extend_from_slice(&x.to_le_bytes());
                    }
                }
            }
        }
        s(&mut b, "token_embd.weight");
        b.extend_from_slice(&2u32.to_le_bytes());
        b.extend_from_slice(&(hidden as u64).to_le_bytes());
        b.extend_from_slice(&(vocab as u64).to_le_bytes());
        b.extend_from_slice(&0u32.to_le_bytes());
        b.extend_from_slice(&0u64.to_le_bytes());
        b.resize(b.len().div_ceil(32) * 32, 0);
        b.resize(b.len() + vocab * hidden * 4, 0);
        b
    }

    /// Qwen3.8-Flash-Next's metadata keys at a small scale: 8 layers (full
    /// attention on 3 and 7), 640-wide experts, 16 n-gram heads of 16 values.
    fn qwen4exp_metadata(ratios: Vec<u64>) -> Vec<(String, Meta)> {
        use Meta::*;
        let k = |s: &str| format!("qwen4exp.{s}");
        let heads = 16u64;
        let sizes: Vec<u64> = (0..heads).map(|i| 1009 + 10 * i).collect();
        let offsets = sizes
            .iter()
            .scan(0, |at, n| {
                let o = *at;
                *at += n;
                Some(o)
            })
            .collect();
        vec![
            ("general.architecture".into(), Str("qwen4exp")),
            ("general.name".into(), Str("tiny qwen4exp")),
            (k("block_count"), U32(8)),
            (k("context_length"), U32(4096)),
            (k("embedding_length"), U32(256)),
            (k("attention.head_count"), U32(2)),
            (k("attention.head_count_kv"), U32(1)),
            (k("attention.key_length"), U32(256)),
            (k("attention.value_length"), U32(256)),
            (k("attention.layer_norm_rms_epsilon"), F32(1e-6)),
            (k("rope.dimension_count"), U32(64)),
            (k("rope.freq_base"), F32(1e7)),
            (k("full_attention_interval"), U32(4)),
            (k("ssm.conv_kernel"), U32(4)),
            (k("ssm.state_size"), U32(128)),
            (k("ssm.group_count"), U32(2)),
            (k("ssm.time_step_rank"), U32(4)),
            (k("ssm.inner_size"), U32(512)),
            (k("expert_count"), U32(8)),
            (k("expert_used_count"), U32(2)),
            (k("expert_feed_forward_length"), U32(640)),
            (k("expert_shared_feed_forward_length"), U32(640)),
            (k("hyper_connection.count"), U32(4)),
            (k("hyper_connection.low_rank"), U32(320)),
            (k("attention.indexer.head_count"), U32(4)),
            (k("attention.indexer.key_length"), U32(128)),
            (k("attention.indexer.top_k"), U32(2048)),
            (k("attention.compress_ratios"), U64s(ratios)),
            (k("ple.layers"), U64s(vec![1])),
            (k("ple.ngram_size"), U32(3)),
            (k("ple.heads_per_ngram"), U32(8)),
            (k("ple.conv_kernel"), U32(4)),
            (k("ple.eos_token_id"), U32(3)),
            (k("embedding_length_per_layer_input"), U32(16)),
            (
                k("ple.layer_multipliers"),
                U64s(vec![23703573157769, 20109073645365, 8052911324071]),
            ),
            (k("ple.head_vocab_sizes"), U64s(sizes)),
            (k("ple.head_offsets"), U64s(offsets)),
        ]
    }

    fn parse(md: &[(String, Meta)]) -> Result<HybridConfig> {
        let file = GgufReader::from_reader(Cursor::new(gguf(md, 4, 256)))?;
        HybridConfig::from_gguf(&file)
    }

    #[test]
    fn qwen4exp_parses_its_additions() {
        let c = parse(&qwen4exp_metadata(vec![0, 0, 0, 4, 0, 0, 0, 4])).unwrap();
        assert_eq!(c.variant, Variant::Qwen4Exp);
        assert_eq!(
            c.attention_layers,
            [false, false, false, true, false, false, false, true]
        );
        assert!(
            c.attn
                .iter()
                .all(|a| a.gated && a.rope && a.window.is_none())
        );
        let moe = c.moe.unwrap();
        assert_eq!((moe.experts, moe.experts_used, moe.expert_ffn), (8, 2, 640));
        assert!(!moe.sigmoid && moe.norm_weights && moe.shared_gate);
        let q = c.qwen4exp.unwrap();
        assert_eq!((q.hc_streams, q.hc_low_rank), (4, 320));
        assert_eq!(
            q.indexer,
            IndexerConfig {
                heads: 4,
                key_dim: 128,
                block: 4,
                top_k: 2048
            }
        );
        assert_eq!(q.ple.layers, [1]);
        assert_eq!(q.ple.heads(), 16);
        assert_eq!(q.ple.multipliers[0], 23703573157769);
        assert_eq!(
            q.ple.table_rows(),
            (0..16).map(|i| 1009 + 10 * i).sum::<u64>()
        );
    }

    #[test]
    fn qwen4exp_layer_lists_must_agree() {
        let err = parse(&qwen4exp_metadata(vec![0, 0, 4, 0, 0, 0, 0, 4])).unwrap_err();
        assert!(err.to_string().contains("compress_ratios"), "{err}");
        let err = parse(&qwen4exp_metadata(vec![0, 0, 0, 4, 0, 0, 0, 8])).unwrap_err();
        assert!(err.to_string().contains("differ"), "{err}");
    }
}
