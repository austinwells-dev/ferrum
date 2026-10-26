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
    rope::Layout,
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
    layout: Layout,
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
    let layout = Layout::text_to_image(text, (side, side));
    let rope = Rope::new(&d, &model.dit, &layout).unwrap();
    let scratch = Scratch::new(&d, &model.dit, text.max(tokens)).unwrap();
    Setup {
        d,
        dump,
        dit,
        rope,
        layout,
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
        .prefill(
            &s.d,
            &s.scratch,
            &s.layout,
            &embeds,
            None,
            &s.rope,
            Some(&mut probe),
        )
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
    assert!(
        rel < bound(&meta(), 0.015, 0.004),
        "step 0 prediction error {rel}"
    );

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
    assert!(
        rel < bound(&meta(), 0.04, 0.008),
        "step 1 prediction error {rel}"
    );
}

#[test]
#[ignore = "needs the Turbo checkpoint and a tools/imgref dump"]
fn dit_sampling_matches_diffusers_latents() {
    let s = setup();
    let embeds = s.tensor("prompt_embeds");
    let prefix = s
        .dit
        .prefill(&s.d, &s.scratch, &s.layout, &embeds, None, &s.rope, None)
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
    assert!(rel < bound(&meta(), 0.1, 0.02), "final latent error {rel}");
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
    // The decoder reads the dump's own latents, so this is the VAE alone.
    assert!(rel < 0.002, "image error {rel}");
}

fn meta() -> serde_json::Value {
    let path = dir("FERRUM_IMGREF").join("meta.json");
    serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap()
}

fn encoder_error(quantize: Option<crate::hybrid::weights::Format>) -> f32 {
    encoder_error_from(quantize, None)
}

/// The encoder's embeddings against the dump, from the checkpoint (quantized
/// to `quantize` as it loads) or from a GGUF.
fn encoder_error_from(
    quantize: Option<crate::hybrid::weights::Format>,
    gguf: Option<&std::path::Path>,
) -> f32 {
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
    let encoder = match gguf {
        Some(path) => TextEncoder::load_gguf(&d, path, &model.text, 256).unwrap(),
        None => {
            let ckpt = Checkpoint::open(model.path("text_encoder")).unwrap();
            TextEncoder::load(&d, &ckpt, &model.text, 256, quantize).unwrap()
        }
    };
    eprintln!(
        "{quantize:?}: loaded {:.1} GB in {:.1}s",
        encoder.weight_bytes() as f64 / 1e9,
        started.elapsed().as_secs_f32()
    );
    let started = std::time::Instant::now();
    let got = encoder.encode(&d, &ids, skip, Vec::new()).unwrap();
    eprintln!(
        "encoded {} tokens in {:.2}s",
        ids.len(),
        started.elapsed().as_secs_f32()
    );
    let want = dump.read_f32("prompt_embeds").unwrap();
    compare(&format!("prompt embeds {quantize:?}"), &got.to_f32(), &want)
}

/// Error bound against a dump made in `meta`'s dtype. A bf16 reference is
/// itself 4.8% from the same model run in fp16 on the prompt embeddings
/// (docs/measurements), so it can only bound ours more loosely; the bounds
/// are 1.5 to 3 times what each was measured at.
fn bound(meta: &serde_json::Value, bf16: f32, f16: f32) -> f32 {
    if meta["dtype"] == "bfloat16" {
        bf16
    } else {
        f16
    }
}

#[test]
#[ignore = "needs the Turbo checkpoint and a tools/imgref dump"]
fn text_encoder_matches_diffusers() {
    let rel = encoder_error(None);
    assert!(
        rel < bound(&meta(), 0.08, 0.02),
        "prompt embedding error {rel}"
    );
}

#[test]
#[ignore = "needs the Turbo checkpoint and a tools/imgref dump"]
fn text_encoder_q8_stays_close_to_diffusers() {
    let rel = encoder_error(Some(crate::hybrid::weights::Format::Q8_0));
    assert!(
        rel < bound(&meta(), 0.08, 0.04),
        "Q8_0 prompt embedding error {rel}"
    );
}

/// Four-bit layers are the lowest the encoder offers; this reports the error
/// the README's numbers rest on and only guards against a broken kernel.
#[test]
#[ignore = "needs the Turbo checkpoint and a tools/imgref dump"]
fn text_encoder_q4_0_error_is_bounded() {
    let rel = encoder_error(Some(crate::hybrid::weights::Format::Q4_0));
    eprintln!("Q4_0 prompt embedding error {rel}");
    assert!(rel < 0.4, "Q4_0 prompt embedding error {rel}");
}

/// A Qwen3-VL GGUF at `FERRUM_TEXT_GGUF` (the Unsloth UD-Q4_K_XL one is the
/// reference case). Its 4-bit layers put it about 27% from either dump
/// (docs/measurements/image-gen); the bound is Q4_0 quantized on load, so a
/// misnamed or misread tensor (near 100%) fails and so does any file that
/// does worse than plain four-bit.
#[test]
#[ignore = "needs the Turbo checkpoint, a tools/imgref dump and FERRUM_TEXT_GGUF"]
fn text_encoder_gguf_error_is_bounded() {
    let path = dir("FERRUM_TEXT_GGUF");
    let rel = encoder_error_from(None, Some(&path));
    eprintln!("GGUF prompt embedding error {rel}");
    assert!(rel < 0.34, "GGUF prompt embedding error {rel}");
}

#[test]
#[ignore = "needs the Turbo checkpoint and a tools/imgref dump"]
fn template_tokenizes_like_diffusers() {
    let pipeline = Pipeline::open(dir("FERRUM_QWEN_IMAGE"), Settings::default()).unwrap();
    let meta = meta();
    let (ids, skip) = pipeline
        .tokenize(meta["prompt"].as_str().unwrap(), &[])
        .unwrap();
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
fn pipeline_settings() -> Settings {
    Settings {
        text: std::env::var("FERRUM_TEXT_QUANT").map_or(None, |q| parse_quantization(&q).unwrap()),
        transformer: std::env::var("FERRUM_DIT_QUANT")
            .map_or(None, |q| parse_quantization(&q).unwrap()),
        transformer_file: std::env::var_os("FERRUM_DIT_GGUF").map(PathBuf::from),
        text_file: std::env::var_os("FERRUM_TEXT_GGUF").map(PathBuf::from),
    }
}

/// Run `request`, check each stage's peak allocation against the planner's
/// prediction, and save the picture to `FERRUM_IMAGE_OUT` if it is set.
fn generate_within_plan(pipeline: &Pipeline, request: &Request) -> super::pipeline::Generated {
    let d = MetalDevice::new().unwrap();
    let predicted = pipeline.stage_bytes(request).unwrap();
    // The stages each window holds: the vision tower and image encoder run
    // one after the other between the first two boundaries.
    let windows: [&[&str]; 4] = [
        &["vision", "image encoder"],
        &["encoder"],
        &["transformer"],
        &["decoder"],
    ];
    let mut peaks = Vec::new();
    let started = std::time::Instant::now();
    let generated = pipeline
        .generate(&d, request, &mut |stage| {
            if matches!(
                stage,
                Stage::EncodeImages
                    | Stage::LoadEncoder
                    | Stage::LoadTransformer
                    | Stage::LoadDecoder
            ) {
                peaks.push(d.counters().allocated_peak);
                d.reset_allocated_peak();
            }
            if let Stage::Denoise { step, steps } = stage {
                eprint!("\rstep {step}/{steps}");
            }
        })
        .unwrap();
    peaks.push(d.counters().allocated_peak);
    eprintln!(
        "\ntotal {:.1}s, {:?}",
        started.elapsed().as_secs_f32(),
        generated.timings
    );
    // Each boundary reports the peak of the window before it.
    let measured = &peaks[1..];
    let windows = &windows[windows.len() - measured.len()..];
    for (names, &got) in windows.iter().zip(measured) {
        let want = predicted
            .iter()
            .filter(|(n, _)| names.contains(n))
            .map(|p| p.1)
            .max()
            .unwrap_or(0);
        eprintln!(
            "{:>26}: predicted {:8.1} MiB, measured peak {:8.1} MiB",
            names.join(" + "),
            want as f64 / 1048576.,
            got as f64 / 1048576.
        );
        // The tolerance the language model's planner is held to.
        assert!(
            got <= want + (4 << 20),
            "{names:?} allocated more than predicted"
        );
    }
    if let Some(path) = std::env::var_os("FERRUM_IMAGE_OUT") {
        std::fs::write(path, generated.png().unwrap()).unwrap();
    }
    generated
}

#[test]
#[ignore = "needs the Turbo checkpoint"]
fn pipeline_generates_an_image() {
    let pipeline = Pipeline::open(dir("FERRUM_QWEN_IMAGE"), pipeline_settings()).unwrap();
    let side = std::env::var("FERRUM_SIDE").map_or(512, |s| s.parse().unwrap());
    let mut request = Request::new("A red fox sitting in fresh snow, golden hour light.");
    (request.size, request.seed) = (Some((side, side)), 42);
    let generated = generate_within_plan(&pipeline, &request);
    assert_eq!((generated.width, generated.height), (side, side));
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
    let rope = Rope::new(&d, &model.dit, &Layout::text_to_image(text, (side, side))).unwrap();
    let scratch = Scratch::new(&d, &model.dit, text.max(tokens)).unwrap();
    let noise = |rows: usize, cols: usize| {
        let v: Vec<f32> = (0..rows * cols)
            .map(|i| ((i * 2654435761) % 1000) as f32 / 500. - 1.)
            .collect();
        Tensor::from_f32(&d, [rows, cols], DType::F32, &v).unwrap()
    };
    let prefix = dit
        .prefill(
            &d,
            &scratch,
            &Layout::text_to_image(text, (side, side)),
            &noise(text, model.dit.context),
            None,
            &rope,
            None,
        )
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

/// Editing, against a dump made with `--image` (`FERRUM_IMGREF_EDIT`).
mod edit {
    use super::*;
    use crate::diffusion::vae_encoder::VaeEncoder;

    fn dump() -> Checkpoint {
        Checkpoint::open(dir("FERRUM_IMGREF_EDIT")).unwrap()
    }

    /// The dump's resized RGBA condition image `i`.
    fn condition(dump: &Checkpoint, i: usize) -> (Vec<u8>, usize, usize) {
        let name = format!("condition.{i}");
        let shape = dump.entry(&name).unwrap().shape.clone();
        let rgba = dump
            .read_f32(&name)
            .unwrap()
            .iter()
            .map(|&v| v as u8)
            .collect();
        (rgba, shape[1], shape[0])
    }

    #[test]
    #[ignore = "needs the Turbo checkpoint and an edit dump"]
    fn vae_encoder_matches_diffusers() {
        let d = MetalDevice::new().unwrap();
        let model = ModelDir::open(dir("FERRUM_QWEN_IMAGE")).unwrap();
        let dump = dump();
        let encoder = VaeEncoder::load(
            &d,
            &Checkpoint::open(model.path("vae")).unwrap(),
            &model.vae,
        )
        .unwrap();
        let (rgba, width, height) = condition(&dump, 0);
        let latents = encoder.encode(&d, &rgba, width, height).unwrap().to_f32();
        let fp32 = compare(
            "vs fp32 VAE",
            &latents,
            &dump.read_f32("vae_encoded.0").unwrap(),
        );
        let pipeline = compare(
            "vs pipeline's VAE",
            &latents,
            &dump.read_f32("condition_latents").unwrap(),
        );
        assert!(
            fp32 < 0.01,
            "condition latents error {fp32} against the fp32 VAE"
        );
        assert!(
            pipeline < 0.05,
            "condition latents error {pipeline} against the pipeline"
        );
    }

    #[test]
    #[ignore = "needs the Turbo checkpoint and an edit dump"]
    fn vision_tower_matches_transformers() {
        let d = MetalDevice::new().unwrap();
        let model = ModelDir::open(dir("FERRUM_QWEN_IMAGE")).unwrap();
        let vision = model.vision.as_ref().expect("a vision tower");
        let dump = dump();
        let ckpt = Checkpoint::open(model.path("text_encoder")).unwrap();
        let tower = crate::diffusion::text::load_vision(&d, &ckpt, vision).unwrap();
        let (rgba, width, height) = condition(&dump, 0);
        let image = image::RgbaImage::from_raw(width as u32, height as u32, rgba).unwrap();
        let got = tower
            .encode_prepared(&d, &tower.prepare_rgba(&image).unwrap())
            .unwrap();
        assert_eq!(got.grid, (height / 32, width / 32));
        // Against the tower in float32; the pipeline's own BF16 run is
        // printed for scale (it is about 5% from float32 itself).
        let rows = got.rows.to_f32();
        compare(
            "vs BF16 pipeline",
            &rows,
            &dump.read_f32("vision_embeds").unwrap(),
        );
        let rel = compare(
            "vision vs fp32",
            &rows,
            &dump.read_f32("vision_fp32").unwrap(),
        );
        // F16 activations through 27 blocks.
        assert!(rel < 0.03, "vision embedding error {rel}");
        assert_eq!(got.deepstack.len(), vision.deepstack.len());
        for (k, features) in got.deepstack.iter().enumerate() {
            let rel = compare(
                &format!("deepstack {k} vs fp32"),
                &features.to_f32(),
                &dump.read_f32(&format!("deepstack_fp32.{k}")).unwrap(),
            );
            assert!(rel < 0.03, "DeepStack {k} error {rel}");
        }
    }

    fn edit_meta() -> serde_json::Value {
        let path = dir("FERRUM_IMGREF_EDIT").join("meta.json");
        serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap()
    }

    fn sizes(meta: &serde_json::Value) -> Vec<(usize, usize)> {
        meta["images"]
            .as_array()
            .unwrap()
            .iter()
            .map(|i| {
                let side = |k: &str| i[k].as_u64().unwrap() as usize;
                (side("width"), side("height"))
            })
            .collect()
    }

    #[test]
    #[ignore = "needs the Turbo checkpoint and an edit dump"]
    fn edit_prompt_tokenizes_like_transformers() {
        let pipeline = Pipeline::open(dir("FERRUM_QWEN_IMAGE"), Settings::default()).unwrap();
        let meta = edit_meta();
        let (ids, skip) = pipeline
            .tokenize(meta["prompt"].as_str().unwrap(), &sizes(&meta))
            .unwrap();
        let want: Vec<u32> = meta["input_ids"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as u32)
            .collect();
        assert_eq!(ids, want);
        assert_eq!(skip as u64, meta["drop_idx"].as_u64().unwrap());
    }

    /// The encoder's text rows after the image, the ones it changes. Fed the
    /// dump's own vision features, this checks where the image and its
    /// DeepStack features land given the same inputs as diffusers. Through
    /// Ferrum's tower (2% from float32, where the BF16 dump is 5%) it is
    /// printed for scale.
    #[test]
    #[ignore = "needs the Turbo checkpoint and an edit dump"]
    fn text_encoder_with_an_image_matches_diffusers() {
        let d = MetalDevice::new().unwrap();
        let model = ModelDir::open(dir("FERRUM_QWEN_IMAGE")).unwrap();
        let pipeline = Pipeline::open(dir("FERRUM_QWEN_IMAGE"), Settings::default()).unwrap();
        let (dump, meta) = (dump(), edit_meta());
        let prompt = pipeline
            .prompt(meta["prompt"].as_str().unwrap(), &sizes(&meta))
            .unwrap();
        let ckpt = Checkpoint::open(model.path("text_encoder")).unwrap();
        let vision = model.vision.as_ref().unwrap();
        let tower = crate::diffusion::text::load_vision(&d, &ckpt, vision).unwrap();
        let (rgba, width, height) = condition(&dump, 0);
        let image = image::RgbaImage::from_raw(width as u32, height as u32, rgba).unwrap();
        let ours = tower
            .encode_prepared(&d, &tower.prepare_rgba(&image).unwrap())
            .unwrap();
        drop(tower);
        let (start, grid) = prompt.spans[0];
        let feature = |name: &str| {
            let e = dump.entry(name).unwrap();
            let v = dump.read_f32(name).unwrap();
            std::rc::Rc::new(Tensor::from_f32(&d, e.shape.clone(), DType::F32, &v).unwrap())
        };
        let theirs = crate::hybrid::engine::ImageSpan {
            start,
            grid,
            embedding: feature("vision_embeds"),
            deepstack: (0..vision.deepstack.len())
                .map(|k| feature(&format!("deepstack.{k}")))
                .collect(),
            hash: ours.hash,
        };
        let ours = crate::hybrid::engine::ImageSpan {
            start,
            grid,
            embedding: ours.rows,
            deepstack: ours.deepstack,
            hash: ours.hash,
        };

        let pad = dump.read_f32("image_pad_mask").unwrap();
        let want: Vec<f32> = dump
            .read_f32("prompt_embeds")
            .unwrap()
            .chunks_exact(model.text.hidden)
            .zip(&pad)
            .filter(|&(_, &p)| p == 0.)
            .flat_map(|(r, _)| r.iter().copied())
            .collect();
        // The text rows before the image are the text-only case, tested
        // above. They include `<|im_start|>`, whose final row is a large
        // cancellation that BF16 runs disagree on by 25-40%, and with few
        // text rows that one row would set the whole-prompt error.
        let after = (start - prompt.skip) * model.text.hidden;
        let encoder = TextEncoder::load(&d, &ckpt, &model.text, prompt.ids.len(), None).unwrap();
        let error = |what: &str, span| {
            let rows = encoder.encode(&d, &prompt.ids, prompt.skip, vec![span]);
            let got = prompt.text_rows(&d, &rows.unwrap()).unwrap().to_f32();
            compare(what, &got, &want);
            compare(&format!("{what}, after it"), &got[after..], &want[after..])
        };
        error("with Ferrum's tower", ours);
        let rel = error("with the dump's tower", theirs);
        assert!(
            rel < bound(&meta, 0.05, 0.03),
            "embedding error {rel} after the image"
        );
    }

    /// The whole edit through the pipeline, from the dump's condition image:
    /// it runs, keeps to the planner's memory and makes a picture the size
    /// diffusers chose (scaled with `FERRUM_RESOLUTION`). The numerics are
    /// the tests above.
    #[test]
    #[ignore = "needs the Turbo checkpoint and an edit dump"]
    fn pipeline_edits_an_image() {
        let pipeline = Pipeline::open(dir("FERRUM_QWEN_IMAGE"), pipeline_settings()).unwrap();
        let (dump, meta) = (dump(), edit_meta());
        let (rgba, width, height) = condition(&dump, 0);
        let mut request = Request::new(meta["prompt"].as_str().unwrap());
        request.images = vec![crate::diffusion::pipeline::InputImage {
            width,
            height,
            rgba,
        }];
        // `FERRUM_RESOLUTION` edits at another size than the dump's.
        let dumped = meta["resolution"].as_u64().unwrap() as usize;
        request.resolution =
            std::env::var("FERRUM_RESOLUTION").map_or(dumped, |r| r.parse().unwrap());
        request.seed = 42;
        let generated = generate_within_plan(&pipeline, &request);
        let side = |k: &str| meta[k].as_u64().unwrap() as usize * request.resolution / dumped;
        assert_eq!(
            (generated.width, generated.height),
            (side("width"), side("height"))
        );
        // Not one flat colour.
        let first = &generated.rgba[..4];
        assert!(generated.rgba.chunks_exact(4).any(|px| px != first));
    }

    /// The transformer over the dump's own embeddings and condition latents:
    /// the block-causal prefill, then two steps and the whole schedule.
    #[test]
    #[ignore = "needs the Turbo checkpoint and an edit dump"]
    fn dit_with_a_condition_image_matches_diffusers() {
        let d = MetalDevice::new().unwrap();
        let model = ModelDir::open(dir("FERRUM_QWEN_IMAGE")).unwrap();
        let pipeline = Pipeline::open(dir("FERRUM_QWEN_IMAGE"), Settings::default()).unwrap();
        let (dump, meta) = (dump(), edit_meta());
        let prompt = pipeline
            .prompt(meta["prompt"].as_str().unwrap(), &sizes(&meta))
            .unwrap();
        let target = (
            meta["height"].as_u64().unwrap() as usize / 16,
            meta["width"].as_u64().unwrap() as usize / 16,
        );
        let layout = Layout {
            target,
            ..prompt.layout.clone()
        };
        let dit = Dit::load(
            &d,
            &Checkpoint::open(model.path("transformer")).unwrap(),
            &model.dit,
            None,
        )
        .unwrap();
        let (prefix_rows, tokens) = (layout.prefix_rows(), layout.target_rows());
        let rope = Rope::new(&d, &model.dit, &layout).unwrap();
        let scratch = Scratch::new(&d, &model.dit, prefix_rows.max(tokens)).unwrap();
        let tensor = |name: &str| {
            let e = dump.entry(name).unwrap();
            Tensor::from_f32(
                &d,
                e.shape.clone(),
                DType::F32,
                &dump.read_f32(name).unwrap(),
            )
            .unwrap()
        };
        let width = model.dit.hidden();
        let pad = dump.read_f32("image_pad_mask").unwrap();
        let text: Vec<f32> = dump
            .read_f32("prompt_embeds")
            .unwrap()
            .chunks_exact(model.dit.context)
            .zip(&pad)
            .filter(|&(_, &p)| p == 0.)
            .flat_map(|(r, _)| r.iter().copied())
            .collect();
        let embeds = Tensor::from_f32(
            &d,
            [text.len() / model.dit.context, model.dit.context],
            DType::F32,
            &text,
        )
        .unwrap();
        let conditions = tensor("condition_latents");
        let layers = [0usize, 15, 31];
        let mut prefix_out = Vec::new();
        let mut probe = |l: usize, x: &Tensor| {
            if layers.contains(&l) {
                prefix_out.push((l, x.to_f32()[..prefix_rows * width].to_vec()));
            }
        };
        let prefix = dit
            .prefill(
                &d,
                &scratch,
                &layout,
                &embeds,
                Some(&conditions),
                &rope,
                Some(&mut probe),
            )
            .unwrap();
        for (l, got) in &prefix_out {
            let want = dump.read_f32(&format!("block_out.{l}")).unwrap();
            compare(
                &format!("prefix block {l}"),
                got,
                &want[..prefix_rows * width],
            );
        }
        let sigmas = model.sample_sigmas.clone().expect("a Turbo checkpoint");
        let latents = tensor("noise");
        let pred = Tensor::zeros(&d, [tokens, 64], DType::F32).unwrap();
        let schedule = Schedule::explicit(&sigmas);
        for i in 0..schedule.steps() {
            dit.step(
                &d,
                &scratch,
                &prefix,
                &rope,
                &latents,
                schedule.sigma(i),
                &pred,
                None,
            )
            .unwrap();
            if i < 2 {
                let rel = compare(
                    &format!("prediction, step {i}"),
                    &pred.to_f32(),
                    &dump.read_f32(&format!("pred.{i}")).unwrap(),
                );
                // Step 1 starts from our own step 0 here, not diffusers'.
                let most = [bound(&meta, 0.015, 0.004), bound(&meta, 0.04, 0.008)][i];
                assert!(rel < most, "step {i} prediction error {rel}");
            }
            schedule.advance(&d, i, &pred, &latents).unwrap();
        }
        let rel = compare(
            "final latents",
            &latents.to_f32(),
            &dump
                .read_f32(&format!("latents.{}", schedule.steps() - 1))
                .unwrap(),
        );
        assert!(rel < bound(&meta, 0.1, 0.02), "final latent error {rel}");
    }
}
