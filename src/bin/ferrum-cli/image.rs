//! `ferrum-cli image`: one text-to-image generation, written as a PNG.
use ferrum::{
    Error, MetalDevice, Result,
    diffusion::pipeline::{Pipeline, Request, Settings, Stage, parse_quantization},
};
use std::{
    io::IsTerminal,
    time::{SystemTime, UNIX_EPOCH},
};

pub const USAGE: &str = "\
usage: ferrum-cli image --model DIR --prompt TEXT [options]

  --model, -m DIR        a Qwen-Image-2.1 directory (config and safetensors, as on Hugging Face)
  --prompt, -p TEXT      what to draw
  --size WxH             image size, multiples of 32 from 256 to 2048 (default 1024x1024)
  --seed N               noise seed (default: random, printed)
  --steps N              denoising steps (default: the checkpoint's own schedule)
  --out, -o FILE         PNG to write (default ferrum-SEED.png)
  --quantize bf16|q8_0|q4_0       transformer matrices, quantized as they load (default bf16)
  --text-quantize bf16|q8_0|q4_0  text encoder matrices, likewise (default bf16)
  --text-file FILE.gguf           take the text encoder from a Qwen3-VL GGUF (e.g. UD-Q4_K_XL) instead
  --transformer-file FILE.gguf    take the transformer from a GGUF (any quantization it holds)
        instead of the directory's safetensors; --model still supplies the encoder, decoder and schedule
";

struct Options {
    model: String,
    request: Request,
    out: Option<String>,
    settings: Settings,
}

fn parse(args: impl Iterator<Item = String>) -> Result<Options> {
    let mut o = Options {
        model: String::new(),
        request: Request::new(""),
        out: None,
        settings: Settings::default(),
    };
    let mut seed = None;
    let mut args = args;
    while let Some(key) = args.next() {
        if matches!(key.as_str(), "-h" | "--help") {
            print!("{USAGE}");
            std::process::exit(0);
        }
        let value = args
            .next()
            .ok_or_else(|| Error::Parameter(format!("missing value for {key}")))?;
        let number = || {
            value
                .parse::<usize>()
                .map_err(|_| Error::Parameter(format!("invalid {key}: {value}")))
        };
        match key.as_str() {
            "--model" | "-m" => o.model = value.clone(),
            "--prompt" | "-p" => o.request.prompt = value.clone(),
            "--size" => {
                let (w, h) = value
                    .split_once(['x', 'X'])
                    .and_then(|(w, h)| Some((w.parse().ok()?, h.parse().ok()?)))
                    .ok_or_else(|| Error::Parameter(format!("invalid --size {value} (WxH)")))?;
                (o.request.width, o.request.height) = (w, h);
            }
            "--seed" => seed = Some(number()? as u64),
            "--steps" => o.request.steps = Some(number()?),
            "--out" | "-o" => o.out = Some(value.clone()),
            "--quantize" => o.settings.transformer = parse_quantization(&value)?,
            "--text-quantize" => o.settings.text = parse_quantization(&value)?,
            "--text-file" => o.settings.text_file = Some(value.clone().into()),
            "--transformer-file" => o.settings.transformer_file = Some(value.clone().into()),
            _ => {
                return Err(Error::Parameter(format!(
                    "unknown option {key} (see `ferrum-cli image --help`)"
                )));
            }
        }
    }
    if o.model.is_empty() || o.request.prompt.is_empty() {
        return Err(Error::Parameter(format!(
            "--model and --prompt are required\n{USAGE}"
        )));
    }
    o.request.seed = seed.unwrap_or_else(|| {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_or(0, |t| t.as_nanos() as u64 & 0xffff_ffff)
    });
    Ok(o)
}

pub fn run(args: impl Iterator<Item = String>) -> Result<()> {
    let o = parse(args)?;
    let device = MetalDevice::new()?;
    let pipeline = Pipeline::open(&o.model, o.settings)?;
    let tty = std::io::stderr().is_terminal();
    let started = std::time::Instant::now();
    let generated = pipeline.generate(&device, &o.request, &mut |stage| match stage {
        Stage::LoadEncoder => eprintln!("loading the text encoder ..."),
        Stage::Encode => eprintln!("encoding the prompt ..."),
        Stage::LoadTransformer => eprintln!("loading the transformer ..."),
        Stage::Denoise { step, steps } => {
            // A log has no cursor to return to: only a terminal gets each step.
            if tty {
                eprint!("\rdenoising: step {step}/{steps}");
            }
            if step == steps {
                eprintln!("{}denoised {steps} steps", if tty { "\r" } else { "" });
            }
        }
        Stage::LoadDecoder => eprintln!("loading the decoder ..."),
        Stage::Decode => eprintln!("decoding ..."),
    })?;
    let out = o
        .out
        .unwrap_or_else(|| format!("ferrum-{}.png", generated.seed));
    std::fs::write(&out, generated.png()?).map_err(|e| Error::Parameter(format!("{out}: {e}")))?;
    let t = generated.timings;
    eprintln!(
        "{out}: {}x{}, seed {}, {:.1} s (load {:.1}, encode {:.1}, denoise {:.1}, decode {:.1})",
        generated.width,
        generated.height,
        generated.seed,
        started.elapsed().as_secs_f32(),
        t.load.as_secs_f32(),
        t.encode.as_secs_f32(),
        t.denoise.as_secs_f32(),
        t.decode.as_secs_f32()
    );
    Ok(())
}
