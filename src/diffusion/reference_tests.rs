//! The transformer against diffusers' dumps (`tools/imgref`). These need the
//! model and a dump on disk, so they are ignored by default:
//!
//! ```sh
//! FERRUM_QWEN_IMAGE=~/.cache/huggingface/hub/models--Qwen--Qwen-Image-2.1-Turbo/snapshots/<rev> \
//! FERRUM_IMGREF=~/models/imgref/fox-512 \
//!   cargo test --release -- --ignored --nocapture dit_
//! ```
use super::{
    checkpoint::Checkpoint,
    config::ModelDir,
    dit::{Dit, Rope, Scratch},
    pipeline::{Pipeline, Request, Settings, Stage, parse_quantization},
    schedule::Schedule,
    text::TextEncoder,
    vae::Vae,
};
use crate::{DType, MetalDevice, Tensor};
use std::path::PathBuf;

fn dir(var: &str) -> PathBuf {
    std::env::var_os(var)
        .unwrap_or_else(|| panic!("set {var} (see the module docs)"))
        .into()
}

/// Relative RMS error, and the largest absolute error over the reference's RMS.
fn compare(what: &str, got: &[f32], want: &[f32]) -> f32 {
    assert_eq!(got.len(), want.len(), "{what}: length");
    let rms = |x: &mut dyn Iterator<Item = f32>| {
        let (n, s) = x.fold((0usize, 0f64), |(n, s), v| {
            (n + 1, s + f64::from(v).powi(2))
        });
        (s / n as f64).sqrt() as f32
    };
    let scale = rms(&mut want.iter().copied());
    let err = rms(&mut got.iter().zip(want).map(|(g, w)| g - w));
    let worst = got
        .iter()
        .zip(want)
        .map(|(g, w)| (g - w).abs())
        .fold(0f32, f32::max);
    eprintln!(
        "{what:>22}: rms {scale:8.3}  rel rms error {:.5}  max error / rms {:.4}",
        err / scale,
        worst / scale
    );
    err / scale
}

struct Setup {
    d: MetalDevice,
    dump: Checkpoint,
    dit: Dit,
    rope: Rope,
    scratch: Scratch,
    text: usize,
    tokens: usize,
    sigmas: Vec<f32>,
}

fn setup() -> Setup {
    let d = MetalDevice::new().unwrap();
    let model = ModelDir::open(dir("FERRUM_QWEN_IMAGE")).unwrap();
    let dump = Checkpoint::open(dir("FERRUM_IMGREF")).unwrap();
    let started = std::time::Instant::now();
    let dit = Dit::load(
        &d,
        &Checkpoint::open(model.path("transformer")).unwrap(),
        &model.dit,
        std::env::var("FERRUM_DIT_QUANT").map_or(None, |q| parse_quantization(&q).unwrap()),
    )
    .unwrap();
    eprintln!(
        "loaded {:.1} GB in {:.1}s",
        dit.weight_bytes() as f64 / 1e9,
        started.elapsed().as_secs_f32()
    );
    let text = dump.entry("prompt_embeds").unwrap().shape[0];
    let tokens = dump.entry("noise").unwrap().shape[0];
    let side = (tokens as f32).sqrt() as usize;
    assert_eq!(side * side, tokens, "the dump is square");
    let rope = Rope::new(&d, &model.dit, text, (side, side)).unwrap();
    let scratch = Scratch::new(&d, &model.dit, text.max(tokens)).unwrap();
    Setup {
        d,
        dump,
        dit,
        rope,
        scratch,
        text,
        tokens,
        sigmas: model.sample_sigmas.expect("a Turbo checkpoint"),
    }
}

impl Setup {
    fn f32(&self, name: &str) -> Vec<f32> {
        self.dump.read_f32(name).unwrap()
    }
    fn tensor(&self, name: &str) -> Tensor {
        let e = self.dump.entry(name).unwrap();
        Tensor::from_f32(&self.d, e.shape.clone(), DType::F32, &self.f32(name)).unwrap()
    }
}

#[test]
#[ignore = "needs the Turbo checkpoint and a tools/imgref dump"]
fn dit_blocks_match_diffusers() {
    let s = setup();
    let width = 4096;
    let layers = [0usize, 1, 15, 31];
    let mut text_out = Vec::new();
    let embeds = s.tensor("prompt_embeds");
    let mut probe = |l: usize, x: &Tensor| {
        if layers.contains(&l) {
            text_out.push((l, x.to_f32()[..s.text * width].to_vec()));
        }
    };
    let prefix = s
        .dit
        .prefill(&s.d, &s.scratch, &embeds, &s.rope, Some(&mut probe))
        .unwrap();
    for (l, got) in &text_out {
        let want = s.f32(&format!("block_out.{l}"));
        compare(&format!("text block {l}"), got, &want[..s.text * width]);
    }

    // Step 0 runs the image rows over the cached text: the same arithmetic
    // as the pipeline's first call, whose image rows follow its text rows.
    let noise = s.tensor("noise");
    let pred = Tensor::zeros(&s.d, [s.tokens, 64], DType::F32).unwrap();
    let mut image_out = Vec::new();
    let mut probe = |l: usize, x: &Tensor| {
        if layers.contains(&l) {
            image_out.push((l, x.to_f32()));
        }
    };
    s.dit
        .step(
            &s.d,
            &s.scratch,
            &prefix,
            &s.rope,
            &noise,
            s.sigmas[0],
            &pred,
            Some(&mut probe),
        )
        .unwrap();
    for (l, got) in &image_out {
        let want = s.f32(&format!("block_out.{l}"));
        compare(
            &format!("image block {l}, step 0"),
            got,
            &want[s.text * width..],
        );
    }
    let rel = compare("prediction, step 0", &pred.to_f32(), &s.f32("pred.0"));
    assert!(rel < 0.05, "step 0 prediction error {rel}");

    // Step 1 starts from diffusers' own latents, so errors don't compound.
    let latents = s.tensor("latents.0");
    image_out.clear();
    let mut probe = |l: usize, x: &Tensor| {
        if layers.contains(&l) {
            image_out.push((l, x.to_f32()));
        }
    };
    s.dit
        .step(
            &s.d,
            &s.scratch,
            &prefix,
            &s.rope,
            &latents,
            s.sigmas[1],
            &pred,
            Some(&mut probe),
        )
        .unwrap();
    for (l, got) in &image_out {
        compare(
            &format!("image block {l}, step 1"),
            got,
            &s.f32(&format!("cached_out.{l}")),
        );
    }
    let rel = compare("prediction, step 1", &pred.to_f32(), &s.f32("pred.1"));
    assert!(rel < 0.05, "step 1 prediction error {rel}");
}

#[test]
#[ignore = "needs the Turbo checkpoint and a tools/imgref dump"]
fn dit_sampling_matches_diffusers_latents() {
    let s = setup();
    let embeds = s.tensor("prompt_embeds");
    let prefix = s
        .dit
        .prefill(&s.d, &s.scratch, &embeds, &s.rope, None)
        .unwrap();
    let latents = s.tensor("noise");
    let pred = Tensor::zeros(&s.d, [s.tokens, 64], DType::F32).unwrap();
    let schedule = Schedule::explicit(&s.sigmas);
    let started = std::time::Instant::now();
    for i in 0..schedule.steps() {
        s.dit
            .step(
                &s.d,
                &s.scratch,
                &prefix,
                &s.rope,
                &latents,
                schedule.sigma(i),
                &pred,
                None,
            )
            .unwrap();
        schedule.advance(&s.d, i, &pred, &latents).unwrap();
        compare(
            &format!("latents after step {i}"),
            &latents.to_f32(),
            &s.f32(&format!("latents.{i}")),
        );
    }
    eprintln!("8 steps in {:.1}s", started.elapsed().as_secs_f32());
    let rel = compare("final latents", &latents.to_f32(), &s.f32("latents.7"));
    assert!(rel < 0.1, "final latent error {rel}");
}

#[test]
#[ignore = "needs the Turbo checkpoint and a tools/imgref dump"]
fn vae_decodes_like_diffusers() {
    let d = MetalDevice::new().unwrap();
    let model = ModelDir::open(dir("FERRUM_QWEN_IMAGE")).unwrap();
    let dump = Checkpoint::open(dir("FERRUM_IMGREF")).unwrap();
    let started = std::time::Instant::now();
    let vae = Vae::load(
        &d,
        &Checkpoint::open(model.path("vae")).unwrap(),
        &model.vae,
    )
    .unwrap();
    eprintln!(
        "loaded {:.2} GB in {:.1}s",
        vae.weight_bytes() as f64 / 1e9,
        started.elapsed().as_secs_f32()
    );

    // The dump is channel-first; the decoder wants one row of channels per pixel.
    let z = dump.entry("vae_in").unwrap().shape.clone();
    let (c, h, w) = (z[0], z[1], z[2]);
    let chw = dump.read_f32("vae_in").unwrap();
    let nhwc: Vec<f32> = (0..h * w)
        .flat_map(|p| (0..c).map(move |ch| (ch, p)))
        .map(|(ch, p)| chw[ch * h * w + p])
        .collect();
    let input = Tensor::from_f32(&d, [h * w, c], DType::F16, &nhwc).unwrap();

    let started = std::time::Instant::now();
    let image = vae.decode(&d, &input, (h, w)).unwrap();
    eprintln!("decoded in {:.2}s", started.elapsed().as_secs_f32());

    let want = dump.read_f32("image").unwrap();
    let want_shape = dump.entry("image").unwrap().shape.clone();
    assert_eq!(want_shape[1..], [image.height, image.width]);
    let channels = want_shape[0].min(image.channels);
    let (n, got_c) = (image.height * image.width, image.channels);
    let (mut got_cmp, mut want_cmp) = (Vec::new(), Vec::new());
    for ch in 0..channels {
        for p in 0..n {
            got_cmp.push(image.pixels[p * got_c + ch]);
            want_cmp.push(want[ch * n + p].clamp(-1., 1.));
        }
    }
    let rel = compare("decoded image", &got_cmp, &want_cmp);
    let mse = got_cmp
        .iter()
        .zip(&want_cmp)
        .map(|(g, w)| f64::from((g - w) / 2.).powi(2))
        .sum::<f64>()
        / got_cmp.len() as f64;
    eprintln!("PSNR {:.1} dB (on [0, 1])", -10. * mse.log10());
    assert!(rel < 0.05, "image error {rel}");
}

fn meta() -> serde_json::Value {
    let path = dir("FERRUM_IMGREF").join("meta.json");
    serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap()
}

fn encoder_error(quantize: Option<crate::hybrid::weights::Format>) -> f32 {
    let d = MetalDevice::new().unwrap();
    let model = ModelDir::open(dir("FERRUM_QWEN_IMAGE")).unwrap();
    let dump = Checkpoint::open(dir("FERRUM_IMGREF")).unwrap();
    let meta = meta();
    let ids: Vec<u32> = meta["input_ids"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap() as u32)
        .collect();
    let skip = meta["drop_idx"].as_u64().unwrap() as usize;

    let started = std::time::Instant::now();
    let ckpt = Checkpoint::open(model.path("text_encoder")).unwrap();
    let encoder = TextEncoder::load(&d, &ckpt, &model.text, 256, quantize).unwrap();
    eprintln!(
        "{quantize:?}: loaded {:.1} GB in {:.1}s",
        encoder.weight_bytes() as f64 / 1e9,
        started.elapsed().as_secs_f32()
    );
    let started = std::time::Instant::now();
    let got = encoder.encode(&d, &ids, skip).unwrap();
    eprintln!(
        "encoded {} tokens in {:.2}s",
        ids.len(),
        started.elapsed().as_secs_f32()
    );
    let want = dump.read_f32("prompt_embeds").unwrap();
    compare(&format!("prompt embeds {quantize:?}"), &got.to_f32(), &want)
}

/// Error bound against a dump: a bf16 reference is itself 4.8% from the
/// same model run in fp16 on these embeddings (docs/measurements), so it
/// can only bound ours more loosely.
fn bound(bf16: f32, f16: f32) -> f32 {
    if meta()["dtype"] == "bfloat16" {
        bf16
    } else {
        f16
    }
}

#[test]
#[ignore = "needs the Turbo checkpoint and a tools/imgref dump"]
fn text_encoder_matches_diffusers() {
    let rel = encoder_error(None);
    assert!(rel < bound(0.08, 0.03), "prompt embedding error {rel}");
}

#[test]
#[ignore = "needs the Turbo checkpoint and a tools/imgref dump"]
fn text_encoder_q8_stays_close_to_diffusers() {
    let rel = encoder_error(Some(crate::hybrid::weights::Format::Q8_0));
    assert!(rel < bound(0.08, 0.05), "Q8_0 prompt embedding error {rel}");
}

#[test]
#[ignore = "needs the Turbo checkpoint and a tools/imgref dump"]
fn template_tokenizes_like_diffusers() {
    let pipeline = Pipeline::open(dir("FERRUM_QWEN_IMAGE"), Settings::default()).unwrap();
    let meta = meta();
    let (ids, skip) = pipeline.tokenize(meta["prompt"].as_str().unwrap()).unwrap();
    let want: Vec<u64> = meta["input_ids"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap())
        .collect();
    assert_eq!(ids.iter().map(|&i| u64::from(i)).collect::<Vec<_>>(), want);
    assert_eq!(skip as u64, meta["drop_idx"].as_u64().unwrap());
}

/// The whole pipeline at the dump's size. `FERRUM_DIT_GGUF` takes the
/// transformer from a GGUF (the published ones hold the base model, not Turbo,
/// so only the memory estimates and the picture can be checked). Writes the PNG to `FERRUM_IMAGE_OUT`
/// when set, and prints each stage's predicted and measured device bytes.
#[test]
#[ignore = "needs the Turbo checkpoint"]
fn pipeline_generates_an_image() {
    let d = MetalDevice::new().unwrap();
    let settings = Settings {
        text: std::env::var("FERRUM_TEXT_QUANT").map_or(None, |q| parse_quantization(&q).unwrap()),
        transformer: std::env::var("FERRUM_DIT_QUANT")
            .map_or(None, |q| parse_quantization(&q).unwrap()),
        transformer_file: std::env::var_os("FERRUM_DIT_GGUF").map(PathBuf::from),
    };
    let pipeline = Pipeline::open(dir("FERRUM_QWEN_IMAGE"), settings).unwrap();
    let side = std::env::var("FERRUM_SIDE").map_or(512, |s| s.parse().unwrap());
    let mut request = Request::new("A red fox sitting in fresh snow, golden hour light.");
    (request.width, request.height, request.seed) = (side, side, 42);
    let predicted = pipeline.stage_bytes(&request).unwrap();

    let mut stage_peaks = Vec::new();
    d.reset_allocated_peak();
    let started = std::time::Instant::now();
    let generated = pipeline
        .generate(&d, &request, &mut |stage| {
            if matches!(stage, Stage::LoadTransformer | Stage::LoadDecoder) {
                stage_peaks.push(d.counters().allocated_peak);
                d.reset_allocated_peak();
            }
            if let Stage::Denoise { step, steps } = stage {
                eprint!("\rstep {step}/{steps}");
            }
        })
        .unwrap();
    stage_peaks.push(d.counters().allocated_peak);
    eprintln!(
        "\ntotal {:.1}s, {:?}",
        started.elapsed().as_secs_f32(),
        generated.timings
    );
    for ((name, want), got) in predicted.iter().zip(&stage_peaks) {
        eprintln!(
            "{name:>12}: predicted {:8.1} MiB, measured peak {:8.1} MiB",
            *want as f64 / 1048576.,
            *got as f64 / 1048576.
        );
    }
    assert_eq!((generated.width, generated.height), (side, side));
    if let Some(path) = std::env::var_os("FERRUM_IMAGE_OUT") {
        std::fs::write(path, generated.png().unwrap()).unwrap();
    }
}

/// Per-kernel GPU time of one denoising step at `FERRUM_SIDE` (default 1024)
/// pixels, on random inputs: the first thing to read before tuning the
/// transformer. `FERRUM_DIT_QUANT` as for the pipeline test.
#[test]
#[ignore = "needs the Turbo checkpoint; prints a profile"]
fn dit_step_profile() {
    let d = MetalDevice::new().unwrap();
    let model = ModelDir::open(dir("FERRUM_QWEN_IMAGE")).unwrap();
    let quantize =
        std::env::var("FERRUM_DIT_QUANT").map_or(None, |q| parse_quantization(&q).unwrap());
    let ckpt = Checkpoint::open(model.path("transformer")).unwrap();
    let dit = Dit::load(&d, &ckpt, &model.dit, quantize).unwrap();
    let side = std::env::var("FERRUM_SIDE").map_or(1024, |s| s.parse::<usize>().unwrap()) / 16;
    let (text, tokens) = (32, side * side);
    let rope = Rope::new(&d, &model.dit, text, (side, side)).unwrap();
    let scratch = Scratch::new(&d, &model.dit, text.max(tokens)).unwrap();
    let noise = |rows: usize, cols: usize| {
        let v: Vec<f32> = (0..rows * cols)
            .map(|i| ((i * 2654435761) % 1000) as f32 / 500. - 1.)
            .collect();
        Tensor::from_f32(&d, [rows, cols], DType::F32, &v).unwrap()
    };
    let prefix = dit
        .prefill(&d, &scratch, &noise(text, model.dit.context), &rope, None)
        .unwrap();
    let latents = noise(tokens, 64);
    let pred = Tensor::zeros(&d, [tokens, 64], DType::F32).unwrap();
    let step = || {
        dit.step(&d, &scratch, &prefix, &rope, &latents, 0.9, &pred, None)
            .unwrap()
    };
    step(); // compile the kernels
    d.set_profiling(true);
    let started = std::time::Instant::now();
    step();
    let wall = started.elapsed();
    let profile = d.take_profile();
    d.set_profiling(false);
    let mut rows: Vec<_> = profile.iter().collect();
    rows.sort_by_key(|(_, e)| std::cmp::Reverse(e.gpu));
    let total: f64 = rows.iter().map(|(_, e)| e.gpu.as_secs_f64()).sum();
    eprintln!(
        "step wall {:.2}s, kernel GPU time {total:.2}s",
        wall.as_secs_f64()
    );
    for (name, e) in rows.iter().take(14) {
        eprintln!(
            "{name:>24}: {:>6.0} ms over {:>4} calls ({:.0}%)",
            e.gpu.as_secs_f64() * 1e3,
            e.calls,
            e.gpu.as_secs_f64() / total * 100.
        );
    }
}
