//! Qwen3.5-family hybrids from MLX checkpoints (a directory of
//! `config.json`, `model*.safetensors`, `tokenizer.json` and a chat
//! template), with affine 4-bit, group-64 linears (`Format::Mlx4`).
//!
//! The conversions follow mlx-lm's `qwen3_5` sanitize and Kapitsa's loader
//! (github.com/nace-ai/kapitsa, `src/model/qwen35.cpp`), which runs these
//! checkpoints as published: MLX conversions store RMSNorm weights with the
//! `1 + w` already applied (the Gated DeltaNet output norm stays plain `w`),
//! `conv1d` as `[channels, 4, 1]`, `A_log` from which the delta rule's decay
//! is `-exp(A_log)`, and value heads in Hugging Face order, where V head h
//! reads K head `h / (v_heads / k_heads)` (`HybridConfig::ssm_grouped_heads`).
#![forbid(unsafe_code)]
use super::{
    config::{AttnSpec, HybridConfig, KvCache, Variant},
    weights::{AttentionWeights, DeltaWeights, DenseFfn, Ffn, Layer, Matrix, Mixer, Weights},
};
use crate::{DType, Error, MetalDevice, Result, Tensor, tokenizer::Tokenizer};
use serde_json::Value as Json;
use std::{
    collections::BTreeMap,
    io::{Read, Seek, SeekFrom},
    path::{Path, PathBuf},
};

/// Whether `path` is an MLX checkpoint directory.
pub fn is_mlx_dir(path: &Path) -> bool {
    path.is_dir()
        && path.join("config.json").is_file()
        && (path.join("model.safetensors").is_file()
            || path.join("model.safetensors.index.json").is_file())
}

/// One tensor of the checkpoint: its shard, dtype, shape and byte range.
#[derive(Clone)]
struct Entry {
    file: PathBuf,
    dtype: String,
    shape: Vec<usize>,
    start: u64,
    len: usize,
}

/// The safetensors headers of every shard; tensor bytes are read on demand.
pub(crate) struct Checkpoint {
    tensors: BTreeMap<String, Entry>,
    /// Name prefix of the text model (`language_model.` in VLM conversions).
    prefix: String,
}

fn io(path: &Path, e: std::io::Error) -> Error {
    Error::Safetensors(format!("{}: {e}", path.display()))
}

impl Checkpoint {
    pub(crate) fn open(dir: &Path) -> Result<Self> {
        let index = dir.join("model.safetensors.index.json");
        let files: Vec<PathBuf> = if index.is_file() {
            let bytes = std::fs::read(&index).map_err(|e| io(&index, e))?;
            let index: Json = serde_json::from_slice(&bytes)
                .map_err(|e| Error::Safetensors(format!("{}: {e}", index.display())))?;
            let mut names: Vec<String> = index["weight_map"]
                .as_object()
                .ok_or_else(|| Error::Safetensors("index has no weight_map".into()))?
                .values()
                .filter_map(|v| v.as_str().map(str::to_owned))
                .collect();
            names.sort();
            names.dedup();
            names.into_iter().map(|n| dir.join(n)).collect()
        } else {
            vec![dir.join("model.safetensors")]
        };
        let mut tensors = BTreeMap::new();
        for file in files {
            let mut f = std::fs::File::open(&file).map_err(|e| io(&file, e))?;
            let mut len = [0u8; 8];
            f.read_exact(&mut len).map_err(|e| io(&file, e))?;
            let n = u64::from_le_bytes(len) as usize;
            let mut header = vec![0u8; n];
            f.read_exact(&mut header).map_err(|e| io(&file, e))?;
            let header: BTreeMap<String, Json> = serde_json::from_slice(&header)
                .map_err(|e| Error::Safetensors(format!("{}: {e}", file.display())))?;
            for (name, info) in header {
                if name == "__metadata__" {
                    continue;
                }
                let offsets = info["data_offsets"]
                    .as_array()
                    .ok_or_else(|| Error::Safetensors(format!("{name}: no data_offsets")))?;
                let (a, b) = (
                    offsets[0].as_u64().unwrap_or(0),
                    offsets[1].as_u64().unwrap_or(0),
                );
                tensors.insert(
                    name,
                    Entry {
                        file: file.clone(),
                        dtype: info["dtype"].as_str().unwrap_or("").to_owned(),
                        shape: info["shape"]
                            .as_array()
                            .map(|s| {
                                s.iter()
                                    .filter_map(Json::as_u64)
                                    .map(|v| v as usize)
                                    .collect()
                            })
                            .unwrap_or_default(),
                        start: 8 + n as u64 + a,
                        len: (b - a) as usize,
                    },
                );
            }
        }
        let prefix = if tensors.contains_key("language_model.model.embed_tokens.weight") {
            "language_model."
        } else if tensors.contains_key("model.embed_tokens.weight") {
            ""
        } else {
            return Err(Error::Config(format!(
                "{}: no text-model embedding (model.embed_tokens.weight)",
                dir.display()
            )));
        };
        Ok(Self {
            tensors,
            prefix: prefix.to_owned(),
        })
    }

    fn entry(&self, name: &str) -> Result<&Entry> {
        self.tensors
            .get(&format!("{}{name}", self.prefix))
            .ok_or_else(|| Error::Weight {
                name: name.into(),
                message: "missing from the checkpoint".into(),
            })
    }

    fn has(&self, name: &str) -> bool {
        self.tensors.contains_key(&format!("{}{name}", self.prefix))
    }

    fn bytes(&self, name: &str) -> Result<Vec<u8>> {
        let e = self.entry(name)?;
        let mut f = std::fs::File::open(&e.file).map_err(|err| io(&e.file, err))?;
        f.seek(SeekFrom::Start(e.start))
            .map_err(|err| io(&e.file, err))?;
        let mut out = vec![0u8; e.len];
        f.read_exact(&mut out).map_err(|err| io(&e.file, err))?;
        Ok(out)
    }

    /// A BF16 (or F32) tensor of `len` values as F32.
    fn floats(&self, name: &str, len: usize) -> Result<Vec<f32>> {
        let e = self.entry(name)?.clone();
        let bytes = self.bytes(name)?;
        let values: Vec<f32> = match e.dtype.as_str() {
            "BF16" => bytes
                .chunks_exact(2)
                .map(|b| f32::from_bits(u32::from(u16::from_le_bytes([b[0], b[1]])) << 16))
                .collect(),
            "F32" => bytes
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                .collect(),
            other => {
                return Err(Error::Weight {
                    name: name.into(),
                    message: format!("{other} values (expected BF16 or F32)"),
                });
            }
        };
        if values.len() != len || values.iter().any(|v| !v.is_finite()) {
            return Err(Error::Weight {
                name: name.into(),
                message: format!("expected {len} finite values, found {}", values.len()),
            });
        }
        Ok(values)
    }

    /// The bytes the loader will allocate for `name`'s quantized linear (codes
    /// plus scales plus biases), or a vector's F32 copy.
    fn device_bytes(&self, name: &str) -> usize {
        let len = |n: &str| self.tensors.get(n).map_or(0, |e| e.len);
        if let Some(base) = name.strip_suffix(".weight")
            && self.tensors.contains_key(&format!("{base}.scales"))
        {
            return len(name) + len(&format!("{base}.scales")) + len(&format!("{base}.biases"));
        }
        match self.tensors.get(name) {
            Some(e) if e.dtype == "BF16" => e.len * 2,
            Some(e) => e.len,
            None => 0,
        }
    }
}

/// `config.json`'s text model (VLM conversions nest it under `text_config`).
fn text_config(dir: &Path) -> Result<Json> {
    let path = dir.join("config.json");
    let bytes = std::fs::read(&path).map_err(|e| io(&path, e))?;
    let root: Json = serde_json::from_slice(&bytes)
        .map_err(|e| Error::Config(format!("{}: {e}", path.display())))?;
    Ok(root.get("text_config").cloned().unwrap_or(root))
}

/// The hybrid geometry of an MLX checkpoint's `config.json`.
pub fn config(dir: &Path) -> Result<HybridConfig> {
    let t = text_config(dir)?;
    let model_type = t["model_type"].as_str().unwrap_or("");
    if !model_type.starts_with("qwen3_5") {
        return Err(Error::Config(format!(
            "MLX model type {model_type:?} is not a Qwen3.5-family hybrid"
        )));
    }
    if t.get("num_experts").and_then(Json::as_u64).unwrap_or(0) > 0 {
        return Err(Error::Config(
            "MLX mixture-of-experts checkpoints are not supported yet; use the GGUF".into(),
        ));
    }
    let root_path = dir.join("config.json");
    let root: Json =
        serde_json::from_slice(&std::fs::read(&root_path).map_err(|e| io(&root_path, e))?)
            .map_err(|e| Error::Config(format!("{}: {e}", root_path.display())))?;
    let quant = root
        .get("quantization")
        .or_else(|| root.get("quantization_config"))
        .cloned()
        .unwrap_or(Json::Null);
    let (bits, group) = (quant["bits"].as_u64(), quant["group_size"].as_u64());
    if bits != Some(4) || group != Some(64) || quant["mode"].as_str().is_some_and(|m| m != "affine")
    {
        return Err(Error::Config(format!(
            "MLX quantization {quant} is unsupported (affine 4-bit, group 64)"
        )));
    }
    let u = |key: &str| -> Result<usize> {
        t[key]
            .as_u64()
            .map(|v| v as usize)
            .ok_or_else(|| Error::Config(format!("config.json: missing {key}")))
    };
    let layers = u("num_hidden_layers")?;
    let attention_layers: Vec<bool> = match t["layer_types"].as_array() {
        Some(types) => types
            .iter()
            .take(layers)
            .map(|v| v.as_str() == Some("full_attention"))
            .collect(),
        None => {
            let interval = t["full_attention_interval"].as_u64().unwrap_or(4) as usize;
            (0..layers).map(|i| (i + 1) % interval == 0).collect()
        }
    };
    let rope = t.get("rope_parameters").cloned().unwrap_or(t.clone());
    let head_dim = u("head_dim")?;
    let partial = rope["partial_rotary_factor"]
        .as_f64()
        .or_else(|| t["partial_rotary_factor"].as_f64())
        .unwrap_or(0.25);
    let theta = rope["rope_theta"]
        .as_f64()
        .or_else(|| t["rope_theta"].as_f64())
        .unwrap_or(1e7) as f32;
    let mut rope_sections = [0usize; 4];
    if let Some(s) = rope["mrope_section"].as_array() {
        for (slot, v) in rope_sections.iter_mut().zip(s) {
            *slot = v.as_u64().unwrap_or(0) as usize;
        }
    }
    let ssm_head_dim = u("linear_key_head_dim")?;
    if u("linear_value_head_dim")? != ssm_head_dim {
        return Err(Error::Config("linear K and V head dims differ".into()));
    }
    let gated = t["attn_output_gate"].as_bool().unwrap_or(true);
    let config = HybridConfig {
        variant: Variant::Dense,
        name: dir
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_default(),
        vocab: u("vocab_size")?,
        hidden: u("hidden_size")?,
        layers,
        attention_layers,
        attn: vec![
            AttnSpec {
                window: None,
                rope: true,
                sinks: false,
                gated,
            };
            layers
        ],
        leading_dense: layers,
        heads: u("num_attention_heads")?,
        kv_heads: u("num_key_value_heads")?,
        head_dim,
        rope_dims: (head_dim as f64 * partial).round() as usize,
        rope_theta: theta,
        swa_rope: None,
        rope_sections,
        eps: t["rms_norm_eps"].as_f64().unwrap_or(1e-6) as f32,
        ffn: u("intermediate_size")?,
        gelu_ffn: false,
        conv_kernel: u("linear_conv_kernel_dim")?,
        ssm_head_dim,
        ssm_k_heads: u("linear_num_key_heads")?,
        ssm_v_heads: u("linear_num_value_heads")?,
        ssm_grouped_heads: true,
        moe: None,
        trained_context: u("max_position_embeddings")?,
        kv_cache: KvCache::F16,
    };
    config.validate()?;
    Ok(config)
}

/// Weight bytes the loader will allocate (page-rounded per tensor).
pub fn weight_bytes(dir: &Path, c: &HybridConfig) -> Result<usize> {
    let ck = Checkpoint::open(dir)?;
    let trunk = |name: &str| -> bool {
        let Some(rest) = name.strip_prefix(&format!("{}model.layers.", ck.prefix)) else {
            return name.starts_with(&format!("{}model.", ck.prefix))
                || name.starts_with(&format!("{}lm_head.", ck.prefix));
        };
        rest.split('.')
            .next()
            .and_then(|l| l.parse::<usize>().ok())
            .is_some_and(|l| l < c.layers)
    };
    let mut total = 0;
    for name in ck.tensors.keys() {
        if !trunk(name) || name.ends_with(".scales") || name.ends_with(".biases") {
            continue;
        }
        total += super::engine::page(ck.device_bytes(name));
    }
    if !ck.has("lm_head.weight") {
        total += super::engine::page(
            ck.device_bytes(&format!("{}model.embed_tokens.weight", ck.prefix)),
        );
    }
    Ok(total)
}

/// Load the trunk's weights.
pub(crate) fn load_weights(d: &MetalDevice, dir: &Path, c: &HybridConfig) -> Result<Weights> {
    let ck = Checkpoint::open(dir)?;
    let linear = |name: &str, rows: usize, cols: usize| -> Result<Matrix> {
        let shape = &ck.entry(&format!("{name}.weight"))?.shape;
        if shape != &[rows, cols / 8] {
            return Err(Error::Weight {
                name: name.into(),
                message: format!("expected 4-bit [{rows}, {cols}], found words {shape:?}"),
            });
        }
        Matrix::from_mlx(
            d,
            rows,
            cols,
            &ck.bytes(&format!("{name}.weight"))?,
            &ck.bytes(&format!("{name}.scales"))?,
            &ck.bytes(&format!("{name}.biases"))?,
        )
    };
    let vector = |name: &str, len: usize, map: &dyn Fn(f32) -> f32| -> Result<Tensor> {
        let v: Vec<f32> = ck.floats(name, len)?.into_iter().map(map).collect();
        Tensor::from_f32(d, [len], DType::F32, &v)
    };
    let same = |v: f32| v;
    let embedding = linear("model.embed_tokens", c.vocab, c.hidden)?;
    let output = if ck.has("lm_head.weight") {
        linear("lm_head", c.vocab, c.hidden)?
    } else {
        linear("model.embed_tokens", c.vocab, c.hidden)?
    };
    let output_norm = vector("model.norm.weight", c.hidden, &same)?;
    let mut layers = Vec::with_capacity(c.layers);
    for i in 0..c.layers {
        let p = format!("model.layers.{i}.");
        let attn_norm = vector(&format!("{p}input_layernorm.weight"), c.hidden, &same)?;
        let post_norm = vector(
            &format!("{p}post_attention_layernorm.weight"),
            c.hidden,
            &same,
        )?;
        let mixer = if c.is_attention(i) {
            let a = format!("{p}self_attn.");
            let q = c.heads * c.head_dim;
            let kv = c.kv_heads * c.head_dim;
            Mixer::Attention(AttentionWeights {
                q: linear(
                    &format!("{a}q_proj"),
                    if c.attn[i].gated { 2 * q } else { q },
                    c.hidden,
                )?,
                k: linear(&format!("{a}k_proj"), kv, c.hidden)?,
                v: linear(&format!("{a}v_proj"), kv, c.hidden)?,
                o: linear(&format!("{a}o_proj"), c.hidden, q)?,
                q_norm: Some(vector(&format!("{a}q_norm.weight"), c.head_dim, &same)?),
                k_norm: Some(vector(&format!("{a}k_norm.weight"), c.head_dim, &same)?),
                sinks: None,
                head_gate: None,
            })
        } else {
            let a = format!("{p}linear_attn.");
            let v_heads = c.ssm_v_heads;
            Mixer::Delta(DeltaWeights {
                qkv: linear(&format!("{a}in_proj_qkv"), c.conv_channels(), c.hidden)?,
                z: linear(&format!("{a}in_proj_z"), c.ssm_value_dim(), c.hidden)?,
                alpha: linear(&format!("{a}in_proj_a"), v_heads, c.hidden)?,
                beta: linear(&format!("{a}in_proj_b"), v_heads, c.hidden)?,
                out: linear(&format!("{a}out_proj"), c.hidden, c.ssm_value_dim())?,
                // [channels, 4, 1]: each channel's taps are contiguous, as the conv kernel reads them.
                conv: vector(
                    &format!("{a}conv1d.weight"),
                    c.conv_channels() * c.conv_kernel,
                    &same,
                )?,
                a: vector(&format!("{a}A_log"), v_heads, &|v| -v.exp())?,
                dt_bias: vector(&format!("{a}dt_bias"), v_heads, &same)?,
                norm: vector(&format!("{a}norm.weight"), c.ssm_head_dim, &same)?,
            })
        };
        let m = format!("{p}mlp.");
        let ffn = Ffn::Dense(DenseFfn {
            gate: linear(&format!("{m}gate_proj"), c.ffn, c.hidden)?,
            up: linear(&format!("{m}up_proj"), c.ffn, c.hidden)?,
            down: linear(&format!("{m}down_proj"), c.hidden, c.ffn)?,
        });
        layers.push(Layer {
            attn_norm,
            post_norm,
            post_attn: None,
            post_ffw: None,
            mixer,
            ffn,
        });
    }
    Ok(Weights {
        embedding,
        output_norm,
        output,
        rope_freqs: None,
        layers,
    })
}

/// Tokenizer, end-of-generation ids, chat template and its (bos, eos)
/// strings of an MLX checkpoint.
pub(crate) struct ChatFiles {
    pub tokenizer: Tokenizer,
    pub eos_ids: Vec<u32>,
    pub template: Option<String>,
    pub special: (String, String),
}

pub(crate) fn chat_files(dir: &Path) -> Result<ChatFiles> {
    let tokenizer = Tokenizer::from_file(dir.join("tokenizer.json"))?;
    let read_json = |name: &str| -> Json {
        std::fs::read(dir.join(name))
            .ok()
            .and_then(|b| serde_json::from_slice(&b).ok())
            .unwrap_or(Json::Null)
    };
    let tok_config = read_json("tokenizer_config.json");
    let generation = read_json("generation_config.json");
    let mut eos_ids: Vec<u32> = match &generation["eos_token_id"] {
        Json::Array(ids) => ids
            .iter()
            .filter_map(Json::as_u64)
            .map(|v| v as u32)
            .collect(),
        Json::Number(n) => n.as_u64().map(|v| vec![v as u32]).unwrap_or_default(),
        _ => Vec::new(),
    };
    let token_string = |v: &Json| -> String {
        match v {
            Json::String(s) => s.clone(),
            Json::Object(o) => o
                .get("content")
                .and_then(Json::as_str)
                .unwrap_or("")
                .to_owned(),
            _ => String::new(),
        }
    };
    let bos = token_string(&tok_config["bos_token"]);
    let eos = token_string(&tok_config["eos_token"]);
    // Chat turns end with <|im_end|>; raw completions may emit <|endoftext|>.
    for special in [eos.as_str(), "<|im_end|>", "<|endoftext|>"] {
        if let Ok(ids) = tokenizer.encode(special)
            && let [id] = ids[..]
            && !special.is_empty()
            && !eos_ids.contains(&id)
        {
            eos_ids.push(id);
        }
    }
    if eos_ids.is_empty() {
        return Err(Error::Tokenizer(format!(
            "{}: no end-of-generation token",
            dir.display()
        )));
    }
    let template = std::fs::read_to_string(dir.join("chat_template.jinja"))
        .ok()
        .or_else(|| tok_config["chat_template"].as_str().map(str::to_owned));
    Ok(ChatFiles {
        tokenizer,
        eos_ids,
        template,
        special: (bos, eos),
    })
}
