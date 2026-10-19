//! A tiny random DFlash2 drafter written both as an HF checkpoint and as a
//! llama.cpp-style GGUF must load to the same config and the same weights.
use super::*;
use crate::hybrid::{KvCache, Variant};

const VOCAB: usize = 512;
const TARGET_HIDDEN: usize = 256;

fn random_bf16(n: usize, seed: u64) -> Vec<u16> {
    let mut x = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    (0..n)
        .map(|_| {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            let v = ((x >> 40) as f32 / (1u64 << 24) as f32) * 2. - 1.;
            (v.to_bits() >> 16) as u16
        })
        .collect()
}

fn config_json() -> Json {
    serde_json::json!({
        "hidden_size": 256,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "head_dim": 128,
        "num_hidden_layers": 2,
        "intermediate_size": 512,
        "rms_norm_eps": 1e-6,
        "use_sliding_window": true,
        "sliding_window": 64,
        "rope_theta": 1e6,
        "dflash_config": {
            "block_size": 4,
            "mask_token_id": 7,
            "target_layer_ids": [0, 2],
            "selector_top_k": 4,
            "selector_rank": 32,
            "conv_kernel_size": 2,
            "conv_group_size": 16
        }
    })
}

/// HF names, llama.cpp GGUF names (as its converter writes them, spelled out
/// here rather than derived from `gguf::tensor_name`) and HF shapes.
fn tensors() -> Vec<(String, String, Vec<usize>)> {
    let (h, q, kv, ffn, rank) = (256, 256, 128, 512, 32);
    let proj = 2 * 2 * (h / 16);
    let mut out: Vec<(String, String, Vec<usize>)> = [
        ("fc.weight", "fc.weight", vec![h, 2 * TARGET_HIDDEN]),
        ("hidden_norm.weight", "enc.output_norm.weight", vec![h]),
        ("norm.weight", "output_norm.weight", vec![h]),
        (
            "candidate_selector.hidden_projection.weight",
            "selector_hidden.weight",
            vec![rank, h],
        ),
        (
            "candidate_selector.predecessor_codebook",
            "selector_predecessor.weight",
            vec![VOCAB, rank],
        ),
        (
            "candidate_selector.successor_codebook",
            "selector_successor.weight",
            vec![VOCAB, rank],
        ),
    ]
    .into_iter()
    .map(|(a, b, s)| (a.to_string(), b.to_string(), s))
    .collect();
    for i in 0..2 {
        for (name, gguf, shape) in [
            ("input_layernorm.weight", "attn_norm.weight", vec![h]),
            (
                "post_attention_layernorm.weight",
                "ffn_norm.weight",
                vec![h],
            ),
            ("self_attn.q_proj.weight", "attn_q.weight", vec![q, h]),
            ("self_attn.k_proj.weight", "attn_k.weight", vec![kv, h]),
            ("self_attn.v_proj.weight", "attn_v.weight", vec![kv, h]),
            ("self_attn.o_proj.weight", "attn_output.weight", vec![h, q]),
            ("self_attn.q_norm.weight", "attn_q_norm.weight", vec![128]),
            ("self_attn.k_norm.weight", "attn_k_norm.weight", vec![128]),
            ("mlp.gate_proj.weight", "ffn_gate.weight", vec![ffn, h]),
            ("mlp.up_proj.weight", "ffn_up.weight", vec![ffn, h]),
            ("mlp.down_proj.weight", "ffn_down.weight", vec![h, ffn]),
            (
                "attention_conv.base_kernel",
                "attn_conv_base",
                vec![2, 2, h],
            ),
            (
                "attention_conv.kernel_projection.weight",
                "attn_conv_proj.weight",
                vec![proj, h],
            ),
            ("mlp_conv.base_kernel", "ffn_conv_base", vec![2, 2, h]),
            (
                "mlp_conv.kernel_projection.weight",
                "ffn_conv_proj.weight",
                vec![proj, h],
            ),
        ] {
            out.push((
                format!("layers.{i}.{name}"),
                format!("blk.{i}.{gguf}"),
                shape,
            ));
        }
    }
    out
}

fn weights() -> Vec<(String, String, Vec<usize>, Vec<u16>)> {
    tensors()
        .into_iter()
        .enumerate()
        .map(|(i, (name, gguf, shape))| {
            let n = shape.iter().product();
            (name, gguf, shape, random_bf16(n, i as u64 + 1))
        })
        .collect()
}

fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("ferrum-draft-{}-{name}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn write_checkpoint(dir: &Path) {
    let mut header = serde_json::Map::new();
    let mut data = Vec::new();
    for (name, _, shape, values) in weights() {
        let start = data.len();
        for v in &values {
            data.extend_from_slice(&v.to_le_bytes());
        }
        header.insert(
            name,
            serde_json::json!({
                "dtype": "BF16",
                "shape": shape,
                "data_offsets": [start, data.len()],
            }),
        );
    }
    let header = serde_json::to_vec(&Json::Object(header)).unwrap();
    let mut file = (header.len() as u64).to_le_bytes().to_vec();
    file.extend_from_slice(&header);
    file.extend_from_slice(&data);
    std::fs::write(dir.join("model.safetensors"), file).unwrap();
    std::fs::write(dir.join("config.json"), config_json().to_string()).unwrap();
}

/// How the GGUF stores matrices (vectors and conv kernels are always F32,
/// as llama.cpp's converter keeps them).
#[derive(Clone, Copy)]
enum Matrices {
    Bf16,
    Q8_0,
}

enum Meta {
    U32(u32),
    F32(f32),
    Bool(bool),
    Str(&'static str),
    I32s(Vec<i32>),
    Bools(Vec<bool>),
}

/// ggml's reference Q8_0 (the converter's `quantize_q8_0`).
fn q8_0(values: &[f32]) -> Vec<u8> {
    let mut out = Vec::new();
    for block in values.chunks_exact(32) {
        let amax = block.iter().fold(0f32, |m, v| m.max(v.abs()));
        let d = amax / 127.;
        let id = if d != 0. { 1. / d } else { 0. };
        out.extend_from_slice(&half::f16::from_f32(d).to_le_bytes());
        out.extend(block.iter().map(|v| ((v * id).round() as i8) as u8));
    }
    out
}

fn gguf_metadata(extra: Vec<(&'static str, Meta)>) -> Vec<(&'static str, Meta)> {
    use Meta::*;
    let mut md = vec![
        ("general.architecture", Str("dflash")),
        ("general.quantization_version", U32(2)),
        ("dflash.embedding_length", U32(256)),
        ("dflash.block_count", U32(2)),
        ("dflash.feed_forward_length", U32(512)),
        ("dflash.attention.head_count", U32(2)),
        ("dflash.attention.head_count_kv", U32(1)),
        ("dflash.attention.key_length", U32(128)),
        ("dflash.attention.value_length", U32(128)),
        ("dflash.attention.layer_norm_rms_epsilon", F32(1e-6)),
        ("dflash.attention.causal", Bool(false)),
        ("dflash.attention.sliding_window", U32(64)),
        (
            "dflash.attention.sliding_window_pattern",
            Bools(vec![true, true]),
        ),
        ("dflash.rope.freq_base", F32(1e6)),
        ("dflash.rope.dimension_sections", I32s(vec![64, 0, 0, 0])),
        ("dflash.block_size", U32(4)),
        ("dflash.conv_kernel_size", U32(2)),
        ("dflash.conv_group_size", U32(16)),
        ("dflash.selector_rank", U32(32)),
        ("dflash.selector_top_k", U32(4)),
        ("dflash.target_layers", I32s(vec![1, 3])),
        ("tokenizer.ggml.model", Str("gpt2")),
        ("tokenizer.ggml.pre", Str("qwen35")),
        ("tokenizer.ggml.mask_token_id", U32(7)),
    ];
    md.extend(extra);
    md
}

fn write_gguf(path: &Path, matrices: Matrices, md: Vec<(&'static str, Meta)>, extra: &[&str]) {
    let s = |b: &mut Vec<u8>, v: &str| {
        b.extend_from_slice(&(v.len() as u64).to_le_bytes());
        b.extend_from_slice(v.as_bytes());
    };
    let mut tensors: Vec<(String, Vec<usize>, u32, Vec<u8>)> = Vec::new();
    for (_, gguf, shape, values) in weights() {
        let dims: Vec<usize> = shape.iter().rev().copied().collect();
        let (type_id, bytes) = match (shape.len(), matrices) {
            (2, Matrices::Bf16) => (30, values.iter().flat_map(|v| v.to_le_bytes()).collect()),
            (2, Matrices::Q8_0) => {
                let f: Vec<f32> = values
                    .iter()
                    .map(|&v| f32::from_bits(u32::from(v) << 16))
                    .collect();
                (8, q8_0(&f))
            }
            _ => (
                0,
                values
                    .iter()
                    .flat_map(|&v| f32::from_bits(u32::from(v) << 16).to_le_bytes())
                    .collect(),
            ),
        };
        tensors.push((gguf, dims, type_id, bytes));
    }
    for name in extra {
        tensors.push((name.to_string(), vec![2], 0, vec![0; 8]));
    }
    let mut b = b"GGUF".to_vec();
    b.extend_from_slice(&3u32.to_le_bytes());
    b.extend_from_slice(&(tensors.len() as u64).to_le_bytes());
    b.extend_from_slice(&(md.len() as u64).to_le_bytes());
    for (k, v) in md {
        s(&mut b, k);
        let u32s = |b: &mut Vec<u8>, x: u32| b.extend_from_slice(&x.to_le_bytes());
        match v {
            Meta::U32(x) => {
                u32s(&mut b, 4);
                u32s(&mut b, x);
            }
            Meta::F32(x) => {
                u32s(&mut b, 6);
                b.extend_from_slice(&x.to_le_bytes());
            }
            Meta::Bool(x) => {
                u32s(&mut b, 7);
                b.push(u8::from(x));
            }
            Meta::Str(x) => {
                u32s(&mut b, 8);
                s(&mut b, x);
            }
            Meta::I32s(xs) => {
                u32s(&mut b, 9);
                u32s(&mut b, 5);
                b.extend_from_slice(&(xs.len() as u64).to_le_bytes());
                for x in xs {
                    b.extend_from_slice(&x.to_le_bytes());
                }
            }
            Meta::Bools(xs) => {
                u32s(&mut b, 9);
                u32s(&mut b, 7);
                b.extend_from_slice(&(xs.len() as u64).to_le_bytes());
                b.extend(xs.into_iter().map(u8::from));
            }
        }
    }
    let mut offset = 0u64;
    for (name, dims, type_id, bytes) in &tensors {
        s(&mut b, name);
        b.extend_from_slice(&(dims.len() as u32).to_le_bytes());
        for &d in dims {
            b.extend_from_slice(&(d as u64).to_le_bytes());
        }
        b.extend_from_slice(&type_id.to_le_bytes());
        b.extend_from_slice(&offset.to_le_bytes());
        offset += (bytes.len() as u64).div_ceil(32) * 32;
    }
    b.resize(b.len().div_ceil(32) * 32, 0);
    for (_, _, _, bytes) in &tensors {
        b.extend_from_slice(bytes);
        b.resize(b.len().div_ceil(32) * 32, 0);
    }
    std::fs::write(path, b).unwrap();
}

/// Every value a load produced, in a fixed order.
fn snapshot(w: &DraftWeights) -> Vec<(String, Vec<u8>)> {
    let mut out = Vec::new();
    let mut m = |name: String, m: &Matrix| out.push((name, m.packed_bytes()));
    m("fc".into(), &w.fc);
    for (i, l) in w.layers.iter().enumerate() {
        for (n, x) in [
            ("q", &l.q),
            ("k", &l.k),
            ("v", &l.v),
            ("o", &l.o),
            ("gate", &l.gate),
            ("up", &l.up),
            ("down", &l.down),
        ] {
            m(format!("{i}.{n}"), x);
        }
        for (n, c) in [("attn_conv", &l.attn_conv), ("mlp_conv", &l.mlp_conv)] {
            m(format!("{i}.{n}.proj"), &c.as_ref().unwrap().proj);
        }
    }
    let selector = w.selector.as_ref().unwrap();
    m("selector.hidden".into(), &selector.hidden);
    let mut t = |name: String, t: &Tensor| {
        out.push((
            name,
            t.to_f32().iter().flat_map(|v| v.to_le_bytes()).collect(),
        ));
    };
    t("hidden_norm".into(), &w.hidden_norm);
    t("norm".into(), &w.norm);
    for (i, l) in w.layers.iter().enumerate() {
        t(format!("{i}.in_norm"), &l.in_norm);
        t(format!("{i}.post_norm"), &l.post_norm);
        t(format!("{i}.q_norm"), &l.q_norm);
        t(format!("{i}.k_norm"), &l.k_norm);
        t(
            format!("{i}.attn_conv.base"),
            &l.attn_conv.as_ref().unwrap().base,
        );
        t(
            format!("{i}.mlp_conv.base"),
            &l.mlp_conv.as_ref().unwrap().base,
        );
    }
    out.push(("bytes".into(), w.bytes.to_le_bytes().to_vec()));
    out
}

fn codebooks(w: &DraftWeights) -> (Vec<u16>, Vec<u16>) {
    let s = w.selector.as_ref().unwrap();
    (s.prev.clone(), s.next.clone())
}

fn load(d: &MetalDevice, path: &Path, precision: DraftQuant) -> (DraftConfig, DraftWeights) {
    let (config, mut src, _) = open(path).unwrap();
    let w = load_weights(d, &config, &mut src, TARGET_HIDDEN, VOCAB, VOCAB, precision).unwrap();
    (config, w)
}

fn assert_same(a: &[(String, Vec<u8>)], b: &[(String, Vec<u8>)]) {
    assert_eq!(a.len(), b.len());
    for ((name, x), (_, y)) in a.iter().zip(b) {
        assert!(x == y, "{name} differs");
    }
}

fn target() -> HybridConfig {
    HybridConfig {
        variant: Variant::Dense,
        name: "tiny".into(),
        vocab: VOCAB,
        hidden: TARGET_HIDDEN,
        layers: 4,
        attention_layers: vec![false, false, false, true],
        heads: 2,
        kv_heads: 1,
        head_dim: 256,
        rope_dims: 64,
        rope_theta: 1e7,
        rope_sections: [0; 4],
        eps: 1e-6,
        ffn: 512,
        conv_kernel: 4,
        ssm_head_dim: 128,
        ssm_k_heads: 1,
        ssm_v_heads: 1,
        moe: None,
        trained_context: 4096,
        kv_cache: KvCache::F16,
    }
}

#[test]
fn gguf_drafter_matches_its_checkpoint() {
    let d = MetalDevice::new().unwrap();
    let dir = scratch("parity");
    let ckpt = dir.join("checkpoint");
    std::fs::create_dir_all(&ckpt).unwrap();
    write_checkpoint(&ckpt);
    let bf16 = dir.join("draft-bf16.gguf");
    let q8 = dir.join("draft-q8_0.gguf");
    write_gguf(&bf16, Matrices::Bf16, gguf_metadata(vec![]), &[]);
    write_gguf(&q8, Matrices::Q8_0, gguf_metadata(vec![]), &[]);

    for precision in [DraftQuant::Q4_0, DraftQuant::Q8_0] {
        let (want_config, want) = load(&d, &ckpt, precision);
        assert_eq!(want_config.kind, Kind::DFlash2 { top_k: 4, rank: 32 });
        assert_eq!(want_config.target_layers, [0, 2]);
        assert_eq!(want_config.window, Some(64));

        // BF16 matrices decode exactly, so every encoding matches.
        let (config, got) = load(&d, &bf16, precision);
        assert_eq!(config, want_config);
        assert_same(&snapshot(&got), &snapshot(&want));
        assert_eq!(codebooks(&got), codebooks(&want));

        // Q8_0 matrices: kept as stored, they equal the checkpoint quantized
        // to Q8_0; re-quantized to Q4_0 they take the same space.
        let (config, got) = load(&d, &q8, precision);
        assert_eq!(config, want_config);
        let (got, want) = (snapshot(&got), snapshot(&want));
        match precision {
            DraftQuant::Q8_0 => assert_same(&got, &want),
            DraftQuant::Q4_0 => {
                for ((name, x), (_, y)) in got.iter().zip(&want) {
                    assert_eq!(x.len(), y.len(), "{name} size");
                }
            }
        }

        // The memory estimate agrees between formats and with the load.
        let t = target();
        let options = DraftOptions {
            quant: precision,
            ..DraftOptions::default()
        };
        let from_ckpt = DraftModel::memory(&ckpt, &t, 64, options).unwrap();
        let from_gguf = DraftModel::memory(&bf16, &t, 64, options).unwrap();
        assert_eq!(from_gguf, from_ckpt);
        let (_, loaded) = load(&d, &bf16, precision);
        assert_eq!(from_gguf.1, loaded.bytes);
    }

    // Q8_0 codebooks round to BF16 within one Q8_0 step.
    let (_, want) = load(&d, &ckpt, DraftQuant::Q8_0);
    let (_, got) = load(&d, &q8, DraftQuant::Q8_0);
    let bf = |v: u16| f32::from_bits(u32::from(v) << 16);
    for (a, b) in codebooks(&got).0.iter().zip(&codebooks(&want).0) {
        assert!(
            (bf(*a) - bf(*b)).abs() <= 1. / 127. + 1e-2,
            "{} vs {}",
            bf(*a),
            bf(*b)
        );
    }
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn gguf_drafter_rejects_what_it_cannot_run() {
    let dir = scratch("reject");
    let path = dir.join("draft.gguf");
    let error = |md, extra: &[&str]| {
        write_gguf(&path, Matrices::Bf16, md, extra);
        match open(&path) {
            Ok(_) => String::new(),
            Err(e) => e.to_string(),
        }
    };
    assert!(error(gguf_metadata(vec![]), &[]).is_empty());
    assert!(error(gguf_metadata(vec![]), &["blk.0.attn_sinks.weight"]).contains("attn_sinks"));
    assert!(error(gguf_metadata(vec![]), &["markov_w1.weight"]).contains("markov_w1"));
    assert!(
        error(
            gguf_metadata(vec![("dflash.logit_scale", Meta::F32(2.))]),
            &[]
        )
        .contains("logit_scale")
    );
    let mut causal = gguf_metadata(vec![]);
    causal.retain(|(k, _)| *k != "dflash.attention.causal");
    causal.push(("dflash.attention.causal", Meta::Bool(true)));
    assert!(error(causal, &[]).contains("causal"));
    let mut mixed = gguf_metadata(vec![]);
    mixed.retain(|(k, _)| *k != "dflash.attention.sliding_window_pattern");
    mixed.push((
        "dflash.attention.sliding_window_pattern",
        Meta::Bools(vec![true, false]),
    ));
    assert!(error(mixed, &[]).contains("mixed"));
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn gguf_tokenizer_must_match_the_target() {
    let mut target = std::collections::BTreeMap::new();
    target.insert(
        "tokenizer.ggml.pre".to_string(),
        MetadataValue::String("qwen35".into()),
    );
    let mut draft = target.clone();
    assert!(check_tokenizer(&draft, &target).is_ok());
    draft.insert(
        "tokenizer.ggml.pre".to_string(),
        MetadataValue::String("llama-bpe".into()),
    );
    assert!(check_tokenizer(&draft, &target).is_err());
}

/// The OrcaCyber DFlash2 drafter (or any DFlash GGUF) loads end to end:
/// `FERRUM_DFLASH_GGUF=path cargo test --release -- --ignored real_dflash_gguf`.
#[test]
#[ignore]
fn real_dflash_gguf_loads() {
    let path = PathBuf::from(std::env::var("FERRUM_DFLASH_GGUF").expect("FERRUM_DFLASH_GGUF"));
    let d = MetalDevice::new().unwrap();
    let (config, mut src, _) = open(&path).unwrap();
    eprintln!("{config:?}");
    let target_hidden = config.hidden;
    let vocab = src
        .shape("candidate_selector.predecessor_codebook")
        .map_or(248_320, |s| s[0]);
    let w = load_weights(
        &d,
        &config,
        &mut src,
        target_hidden,
        vocab,
        vocab,
        DraftQuant::Q8_0,
    )
    .unwrap();
    eprintln!("{} layers, {} MiB on device", w.layers.len(), w.bytes >> 20);
}
