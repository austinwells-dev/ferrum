use super::*;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

fn rand_vec(rng: &mut ChaCha8Rng, n: usize, scale: f32) -> Vec<f32> {
    (0..n).map(|_| rng.random_range(-scale..scale)).collect()
}

fn erf(x: f32) -> f32 {
    // Abramowitz & Stegun 7.1.26 (|error| < 1.5e-7).
    let t = 1. / (1. + 0.327_591_1 * x.abs());
    let poly =
        ((((1.061_405_4 * t - 1.453_152) * t) + 1.421_413_7) * t - 0.284_496_74) * t + 0.254_829_6;
    let y = 1. - poly * t * (-x * x).exp();
    y.copysign(x)
}

fn gelu_tanh(x: f32) -> f32 {
    0.5 * x * (1. + (0.797_884_6 * (x + 0.044715 * x * x * x)).tanh())
}

fn gelu_erf(x: f32) -> f32 {
    0.5 * x * (1. + erf(x * std::f32::consts::FRAC_1_SQRT_2))
}

/// `out[m, n] = a[m, k] . w[n, k] + bias[n]`
fn linear(a: &[f32], w: &[f32], b: &[f32], m: usize, n: usize, k: usize) -> Vec<f32> {
    let mut out = vec![0.; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut s = b[j];
            for t in 0..k {
                s += a[i * k + t] * w[j * k + t];
            }
            out[i * n + j] = s;
        }
    }
    out
}

fn layernorm(x: &[f32], w: &[f32], b: &[f32], width: usize, eps: f32) -> Vec<f32> {
    let mut out = Vec::with_capacity(x.len());
    for row in x.chunks(width) {
        let mean = row.iter().sum::<f32>() / width as f32;
        let var = row.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / width as f32;
        let inv = 1. / (var + eps).sqrt();
        out.extend(
            row.iter()
                .enumerate()
                .map(|(i, v)| (v - mean) * inv * w[i] + b[i]),
        );
    }
    out
}

fn tiny_config() -> VisionConfig {
    VisionConfig {
        patch: 4,
        hidden: 64,
        ffn: 96,
        layers: 2,
        heads: 2,
        eps: 1e-6,
        merge: 2,
        out_dim: 40,
        table: 6,
        mean: [0.5; 3],
        std: [0.5; 3],
    }
}

fn synthetic(d: &MetalDevice, config: VisionConfig, seed: u64) -> VisionModel {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let h = config.hidden;
    let mat = |rng: &mut ChaCha8Rng, rows: usize, cols: usize| {
        let scale = 1. / (cols as f32).sqrt();
        Tensor::from_f32(
            d,
            [rows, cols],
            DType::F16,
            &rand_vec(rng, rows * cols, scale),
        )
        .unwrap()
    };
    let vec = |rng: &mut ChaCha8Rng, n: usize, base: f32| {
        let v: Vec<f32> = rand_vec(rng, n, 0.1).iter().map(|x| x + base).collect();
        Tensor::from_f32(d, [n], DType::F32, &v).unwrap()
    };
    let layers = (0..config.layers)
        .map(|_| Layer {
            ln1: (vec(&mut rng, h, 1.), vec(&mut rng, h, 0.)),
            ln2: (vec(&mut rng, h, 1.), vec(&mut rng, h, 0.)),
            qkv: (mat(&mut rng, 3 * h, h), vec(&mut rng, 3 * h, 0.)),
            out: (mat(&mut rng, h, h), vec(&mut rng, h, 0.)),
            up: (mat(&mut rng, config.ffn, h), vec(&mut rng, config.ffn, 0.)),
            down: (mat(&mut rng, h, config.ffn), vec(&mut rng, h, 0.)),
        })
        .collect();
    let merged = h * 4;
    let merger = |rng: &mut ChaCha8Rng, post_shuffle: bool| {
        let norm = if post_shuffle { merged } else { h };
        Merger {
            norm: (vec(rng, norm, 1.), vec(rng, norm, 0.)),
            fc1: (mat(rng, merged, merged), vec(rng, merged, 0.)),
            fc2: (
                mat(rng, config.out_dim, merged),
                vec(rng, config.out_dim, 0.),
            ),
            post_shuffle,
        }
    };
    let output = merger(&mut rng, false);
    let deepstack = (0..config.layers)
        .map(|l| (l, merger(&mut rng, true)))
        .collect();
    VisionModel {
        patch: (mat(&mut rng, h, config.patch_len()), vec(&mut rng, h, 0.)),
        position: rand_vec(&mut rng, config.table * config.table * h, 0.5),
        merger: output,
        deepstack,
        zero: Tensor::zeros(d, [1], DType::F32).unwrap(),
        layers,
        config,
        max_tokens: 64,
        min_tokens: DEFAULT_MIN_TOKENS,
        weight_bytes: 0,
        score_bytes: SCORE_BYTES,
        lfm3: None,
        marker: "<|image_pad|>",
        filter: image::imageops::FilterType::CatmullRom,
    }
}

/// A merger on CPU over `x` (`n` patch rows of `h`, block order).
fn merge_reference(
    m: &Merger,
    x: &[f32],
    n: usize,
    h: usize,
    eps: f32,
    out_dim: usize,
) -> Vec<f32> {
    let w = |t: &(Tensor, Tensor)| (t.0.to_f32(), t.1.to_f32());
    let (g, b) = w(&m.norm);
    // Per-patch and post-shuffle norms differ only in the width normalised.
    let xn = layernorm(x, &g, &b, if m.post_shuffle { 4 * h } else { h }, eps);
    let rows = n / 4;
    let (w1, b1) = w(&m.fc1);
    let mid: Vec<f32> = linear(&xn, &w1, &b1, rows, 4 * h, 4 * h)
        .into_iter()
        .map(gelu_erf)
        .collect();
    let (w2, b2) = w(&m.fc2);
    linear(&mid, &w2, &b2, rows, out_dim, 4 * h)
}

/// The output rows and the DeepStack features.
fn reference(m: &VisionModel, image: &Prepared) -> (Vec<f32>, Vec<Vec<f32>>) {
    let c = &m.config;
    let (gh, gw) = (image.height / c.patch, image.width / c.patch);
    let (n, h, hd) = (gh * gw, c.hidden, c.head_dim());
    let w = |t: &(Tensor, Tensor)| (t.0.to_f32(), t.1.to_f32());
    let (pw, pb) = w(&m.patch);
    let mut x = linear(&m.patchify(image), &pw, &pb, n, h, c.patch_len());
    for (a, b) in x.iter_mut().zip(m.position_rows(gh, gw)) {
        *a += b;
    }
    let (cos, sin) = rope_tables(gh, gw, c.merge, hd);
    let mut deepstack = Vec::new();
    for (index, layer) in m.layers.iter().enumerate() {
        let (g1, b1) = w(&layer.ln1);
        let xn = layernorm(&x, &g1, &b1, h, c.eps);
        let (qw, qb) = w(&layer.qkv);
        let mut qkv = linear(&xn, &qw, &qb, n, 3 * h, h);
        for t in 0..n {
            for head in 0..2 * c.heads {
                let base = t * 3 * h
                    + if head < c.heads {
                        head * hd
                    } else {
                        h + (head - c.heads) * hd
                    };
                for i in 0..hd / 2 {
                    let (co, si) = (cos[t * hd / 2 + i], sin[t * hd / 2 + i]);
                    let (x0, x1) = (qkv[base + i], qkv[base + i + hd / 2]);
                    qkv[base + i] = x0 * co - x1 * si;
                    qkv[base + i + hd / 2] = x1 * co + x0 * si;
                }
            }
        }
        let mut att = vec![0.; n * h];
        for head in 0..c.heads {
            for q in 0..n {
                let mut scores: Vec<f32> = (0..n)
                    .map(|k| {
                        (0..hd)
                            .map(|i| {
                                qkv[q * 3 * h + head * hd + i] * qkv[k * 3 * h + h + head * hd + i]
                            })
                            .sum::<f32>()
                            / (hd as f32).sqrt()
                    })
                    .collect();
                let max = scores.iter().cloned().fold(f32::MIN, f32::max);
                let mut sum = 0.;
                for s in &mut scores {
                    *s = (*s - max).exp();
                    sum += *s;
                }
                for i in 0..hd {
                    att[q * h + head * hd + i] = (0..n)
                        .map(|k| scores[k] / sum * qkv[k * 3 * h + 2 * h + head * hd + i])
                        .sum();
                }
            }
        }
        let (ow, ob) = w(&layer.out);
        for (a, b) in x.iter_mut().zip(linear(&att, &ow, &ob, n, h, h)) {
            *a += b;
        }
        let (g2, b2) = w(&layer.ln2);
        let xn = layernorm(&x, &g2, &b2, h, c.eps);
        let (uw, ub) = w(&layer.up);
        let hid: Vec<f32> = linear(&xn, &uw, &ub, n, c.ffn, h)
            .into_iter()
            .map(gelu_tanh)
            .collect();
        let (dw, db) = w(&layer.down);
        for (a, b) in x.iter_mut().zip(linear(&hid, &dw, &db, n, h, c.ffn)) {
            *a += b;
        }
        for (_, merger) in m.deepstack.iter().filter(|(l, _)| *l == index) {
            deepstack.push(merge_reference(merger, &x, n, h, c.eps, c.out_dim));
        }
    }
    (
        merge_reference(&m.merger, &x, n, h, c.eps, c.out_dim),
        deepstack,
    )
}

fn test_image(width: usize, height: usize, seed: u64) -> Prepared {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    Prepared {
        width,
        height,
        pixels: rand_vec(&mut rng, 3 * width * height, 1.),
        hash: seed,
    }
}

fn compare(label: &str, got: &[f32], want: &[f32]) {
    assert_eq!(got.len(), want.len(), "{label}: length");
    let scale = want.iter().map(|v| v.abs()).fold(0., f32::max).max(1e-3);
    let worst = got
        .iter()
        .zip(want)
        .map(|(a, b)| (a - b).abs())
        .fold(0., f32::max);
    assert!(
        worst <= 0.02 * scale,
        "{label}: worst error {worst} against output scale {scale}"
    );
}

#[test]
fn encoder_matches_cpu_reference() {
    let d = MetalDevice::new().unwrap();
    let mut model = synthetic(&d, tiny_config(), 7);
    for (width, height) in [(16, 16), (32, 24), (24, 40)] {
        let image = test_image(width, height, (width * height) as u64);
        let (want, want_deepstack) = reference(&model, &image);
        for score_bytes in [SCORE_BYTES, 1] {
            model.score_bytes = score_bytes;
            let got = model.encode_prepared(&d, &image).unwrap();
            assert_eq!(got.grid, (height / 8, width / 8));
            compare(
                &format!("{width}x{height} scores {score_bytes}"),
                &got.rows.to_f32(),
                &want,
            );
            assert_eq!(got.deepstack.len(), want_deepstack.len());
            for (i, (g, w)) in got.deepstack.iter().zip(&want_deepstack).enumerate() {
                compare(&format!("{width}x{height} deepstack {i}"), &g.to_f32(), w);
            }
        }
    }
}

#[test]
fn position_table_is_identity_at_native_size() {
    let d = MetalDevice::new().unwrap();
    let model = synthetic(&d, tiny_config(), 3);
    let t = model.config.table;
    let rows = model.position_rows(t, t);
    let h = model.config.hidden;
    for (token, (row, col)) in block_order(t, t, 2).enumerate() {
        let want = &model.position[(row * t + col) * h..(row * t + col + 1) * h];
        assert_eq!(&rows[token * h..(token + 1) * h], want);
    }
}

#[test]
fn smart_resize_follows_the_reference_rounding() {
    // Already a multiple: unchanged.
    assert_eq!(smart_resize(512, 768, 32, 65536, 1 << 20), (768, 512));
    // Rounded to the nearest multiple.
    assert_eq!(smart_resize(500, 770, 32, 65536, 1 << 20), (768, 512));
    // Tiny images grow to the minimum pixel count.
    let (w, h) = smart_resize(32, 32, 32, 65536, 1 << 20);
    assert!(w * h >= 65536 && w % 32 == 0 && h % 32 == 0);
    // Huge images shrink under the cap, keeping the aspect ratio roughly.
    let (w, h) = smart_resize(4000, 6000, 32, 65536, 1 << 20);
    assert!(w * h <= 1 << 20 && w % 32 == 0 && h % 32 == 0);
    assert!(w > h);
    // Degenerate strips still produce at least one merged token.
    let (w, h) = smart_resize(1, 4000, 32, 65536, 1 << 20);
    assert!(h >= 32 && w >= 32);
}

#[test]
fn block_order_keeps_merge_blocks_adjacent() {
    let order: Vec<_> = block_order(4, 4, 2).collect();
    assert_eq!(&order[..4], &[(0, 0), (0, 1), (1, 0), (1, 1)]);
    assert_eq!(&order[4..8], &[(0, 2), (0, 3), (1, 2), (1, 3)]);
    assert_eq!(order.len(), 16);
}

#[test]
fn prepare_decodes_resizes_and_normalises() {
    let d = MetalDevice::new().unwrap();
    let model = synthetic(&d, tiny_config(), 1);
    // A 2x2 PNG: opaque red, transparent (shown white), blue, black.
    let mut img = image::RgbaImage::new(2, 2);
    img.put_pixel(0, 0, image::Rgba([255, 0, 0, 255]));
    img.put_pixel(1, 0, image::Rgba([0, 0, 0, 0]));
    img.put_pixel(0, 1, image::Rgba([0, 0, 255, 255]));
    img.put_pixel(1, 1, image::Rgba([0, 0, 0, 255]));
    let mut png = Vec::new();
    img.write_to(&mut std::io::Cursor::new(&mut png), image::ImageFormat::Png)
        .unwrap();
    let p = model.prepare(&png).unwrap();
    assert_eq!((p.width % 8, p.height % 8), (0, 0));
    assert_eq!(p.pixels.len(), 3 * p.width * p.height);
    let mut other = png.clone();
    other.push(0);
    assert_ne!(model.prepare(&other).map(|p| p.hash).unwrap_or(0), p.hash);
    assert!(p.pixels.iter().all(|v| (-1.0001..=1.0001).contains(v)));
    assert!(model.prepare(b"not an image").is_err());
}

#[test]
fn request_budgets_stay_within_the_projector_cap() {
    let d = MetalDevice::new().unwrap();
    let mut model = synthetic(&d, tiny_config(), 1);
    model.max_tokens = 64;
    model.min_tokens = 16;
    let budget = |min, max| ImageBudget {
        min_tokens: min,
        max_tokens: max,
    };
    assert_eq!(model.limits(ImageBudget::default()), (16, 64));
    // A request may lower the cap, and the floor follows it down.
    assert_eq!(model.limits(budget(None, Some(8))), (8, 8));
    assert_eq!(model.limits(budget(Some(4), Some(32))), (4, 32));
    // It may not raise the cap the memory plan was made for.
    assert_eq!(model.limits(budget(Some(100), Some(1 << 20))), (64, 64));
    // A large image shrinks to the request's cap (factor 8: one token per 8x8).
    let (w, h) = model.target_size_with(640, 640, budget(None, Some(9)));
    assert!((w / 8) * (h / 8) <= 9, "{w}x{h}");
    let (w, h) = model.target_size(640, 640);
    assert!((w / 8) * (h / 8) <= 64 && (w / 8) * (h / 8) > 9, "{w}x{h}");
    let fill = ImageBudget {
        min_tokens: Some(1),
        ..Default::default()
    };
    assert_eq!(fill.or(budget(Some(2), Some(3))), budget(Some(1), Some(3)));
}

#[test]
fn safetensors_names_follow_transformers() {
    for (gguf, hf) in [
        ("v.patch_embd.bias", "patch_embed.proj.bias"),
        ("v.position_embd.weight", "pos_embed.weight"),
        ("v.blk.3.ln1.weight", "blocks.3.norm1.weight"),
        ("v.blk.3.ln2.bias", "blocks.3.norm2.bias"),
        ("v.blk.26.attn_qkv.weight", "blocks.26.attn.qkv.weight"),
        ("v.blk.0.attn_out.bias", "blocks.0.attn.proj.bias"),
        ("v.blk.1.ffn_up.weight", "blocks.1.mlp.linear_fc1.weight"),
        ("v.blk.1.ffn_down.weight", "blocks.1.mlp.linear_fc2.weight"),
        ("v.post_ln.weight", "merger.norm.weight"),
        ("mm.0.weight", "merger.linear_fc1.weight"),
        ("mm.2.bias", "merger.linear_fc2.bias"),
        (
            "v.deepstack.2.norm.weight",
            "deepstack_merger_list.2.norm.weight",
        ),
        (
            "v.deepstack.0.fc1.bias",
            "deepstack_merger_list.0.linear_fc1.bias",
        ),
        (
            "v.deepstack.1.fc2.weight",
            "deepstack_merger_list.1.linear_fc2.weight",
        ),
    ] {
        assert_eq!(
            transformers_name("model.visual.", gguf),
            format!("model.visual.{hf}"),
            "{gguf}"
        );
    }
}

#[test]
fn projectors_match_their_model_by_name() {
    let vl = "lfm3-vl-2b-q4_k_m";
    let audio = "lfm3-audio-2b-q4_k_m";
    assert!(named_after("mmproj-lfm3-vl-2b-q8_0.gguf", vl));
    assert!(!named_after("mmproj-lfm3-audio-2b-q8_0.gguf", vl));
    assert!(named_after("mmproj-lfm3-audio-2b-bf16.gguf", audio));
    assert!(!named_after("mmproj-lfm3-vl-2b-f16.gguf", audio));
    assert!(named_after(
        "qwen3.8-27b-mmproj.gguf",
        "qwen3.8-27b-ud-q4_k_xl"
    ));
    assert!(named_after(
        "mmproj-swift-1.5-qwen3.8-27b-q4_k_m.gguf",
        "swift-1.5-qwen3.8-27b-q4_k_m"
    ));
    assert!(!named_after(
        "mmproj-bf16.gguf",
        "swift-1.5-qwen3.8-27b-q4_k_m"
    ));
    assert!(!named_after("mmproj.gguf", "anything"));
}

/// A header-and-zeros GGUF: string, u32 and f32 metadata, F32 tensors.
fn write_gguf(path: &Path, md: &[(&str, Option<&str>, f64)], tensors: &[(&str, &[usize])]) {
    let s = |b: &mut Vec<u8>, v: &str| {
        b.extend_from_slice(&(v.len() as u64).to_le_bytes());
        b.extend_from_slice(v.as_bytes());
    };
    let mut b = b"GGUF".to_vec();
    b.extend_from_slice(&3u32.to_le_bytes());
    b.extend_from_slice(&(tensors.len() as u64).to_le_bytes());
    b.extend_from_slice(&(md.len() as u64).to_le_bytes());
    for &(key, text, number) in md {
        s(&mut b, key);
        match text {
            Some(t) => {
                b.extend_from_slice(&8u32.to_le_bytes());
                s(&mut b, t);
            }
            None if number.fract() != 0. => {
                b.extend_from_slice(&6u32.to_le_bytes());
                b.extend_from_slice(&(number as f32).to_le_bytes());
            }
            None => {
                b.extend_from_slice(&4u32.to_le_bytes());
                b.extend_from_slice(&(number as u32).to_le_bytes());
            }
        }
    }
    let mut offset = 0u64;
    for (name, dims) in tensors {
        s(&mut b, name);
        b.extend_from_slice(&(dims.len() as u32).to_le_bytes());
        for &d in *dims {
            b.extend_from_slice(&(d as u64).to_le_bytes());
        }
        b.extend_from_slice(&0u32.to_le_bytes());
        b.extend_from_slice(&offset.to_le_bytes());
        offset += (dims.iter().product::<usize>() * 4).next_multiple_of(32) as u64;
    }
    b.resize(b.len().next_multiple_of(32), 0);
    b.resize(b.len() + offset as usize, 0);
    std::fs::write(path, b).unwrap();
}

#[test]
fn audio_projector_budget_counts_weights_encode_and_cache() {
    let path = std::env::temp_dir().join(format!("ferrum-lfm3a-{}.gguf", std::process::id()));
    let md: &[(&str, Option<&str>, f64)] = &[
        ("general.architecture", Some("clip"), 0.),
        ("clip.audio.projector_type", Some("lfm3a"), 0.),
        ("clip.audio.num_mel_bins", None, 128.),
        ("clip.audio.embedding_length", None, 64.),
        ("clip.audio.attention.head_count", None, 4.),
        ("clip.audio.attention.head_count_kv", None, 2.),
        ("clip.audio.feed_forward_length", None, 96.),
        ("clip.audio.block_count", None, 3.),
        ("clip.audio.window_size", None, 32.),
        ("clip.audio.attention.layer_norm_epsilon", None, 1e-5),
        ("clip.audio.projection_dim", None, 48.),
    ];
    let tensors: &[(&str, &[usize])] = &[
        ("a.conv1d.0.bias", &[16]),
        ("a.blk.0.ffn_up.weight", &[64, 96]),
        ("mm.a.mlp.1.bias", &[40]),
        ("mm.a.mlp.3.weight", &[40, 48]),
        // The speech generator is not loaded, so not planned.
        ("a.gen.code.proj_in.weight", &[4096, 4096]),
    ];
    write_gguf(&path, md, tensors);
    assert!(is_audio_projector(&path));
    let config = audio_config(GgufFile::open(&path).unwrap().metadata()).unwrap();
    let frames = audio::MelBank::frames(MAX_AUDIO_SECONDS * audio::SAMPLE_RATE);
    let page = crate::hybrid::engine::page;
    let weights = page(16 * 2) + page(64 * 96 * 2) + page(40 * 2) + page(40 * 48 * 2);
    let want = weights
        + lfm3a::activation_bytes(&config, 16, 40, frames)
        + (AUDIO_CACHE_CLIPS - 1) * lfm3a::output_bytes(&config, frames);
    assert_eq!(projector_bytes(&path), want);
    let _ = std::fs::remove_file(&path);
}
