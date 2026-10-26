//! `generate_image`: chat and the agent draw or edit pictures with the Image
//! tab's Qwen-Image model. The chat model's thread runs it, because the
//! picture needs the GPU and often the chat model's memory: when the
//! pipeline's stages do not fit beside the chat model, the chat model is
//! unloaded for the picture and loaded again after it (the next turn then
//! reads the conversation afresh, since its cache went with the model).
use crate::agent::{Sandbox, arg_int, arg_str, def};
use crate::imagetab::stage_label;
use crate::*;
use ferrum::{
    Error, MetalDevice,
    diffusion::pipeline::{InputImage, MAX_IMAGES, Observer, Pipeline, Request, Settings, Stage},
};
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
    mpsc::Sender,
};

pub const NAME: &str = "generate_image";

/// The Image tab's model and weight settings, which the tool uses.
#[derive(Clone)]
pub struct ImageModel {
    pub dir: PathBuf,
    pub settings: Settings,
}

pub fn def_json() -> Json {
    def(
        NAME,
        "Draw a picture from a text prompt with the local Qwen-Image model, or edit pictures in the workspace by \
         describing the change. Saves a PNG in the workspace and returns its path. Takes 20-30 s at 512x512 and \
         1-2 minutes at 1024x1024, and may unload you from memory meanwhile.",
        json!({
            "prompt": {"type": "string", "description": "What to draw, or for an edit the change to make."},
            "images": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Edit: up to 4 workspace images to change or combine. Leave out to draw a new picture."
            },
            "size": {"type": "string", "description": "WIDTHxHEIGHT, multiples of 32 from 256 to 2048 (default 1024x1024). For an edit, the area to keep the last image's shape at."},
            "path": {"type": "string", "description": "Where to save the PNG, relative to the workspace (default images/ferrum-<time>-<seed>.png)."},
            "seed": {"type": "integer", "description": "Noise seed, to repeat a picture."}
        }),
        &["prompt"],
    )
}

/// A checked call: what to generate and where it goes.
pub struct Job {
    pub request: Request,
    pub out: PathBuf,
    pub sandbox: Sandbox,
}

/// The call's arguments as a pipeline request, reading edit images and
/// placing the output inside the workspace.
pub fn job(sb: &Sandbox, args: &Map<String, Json>) -> Result<Job, String> {
    let prompt = arg_str(args, "prompt")?.trim();
    if prompt.is_empty() {
        return Err("the prompt is empty".into());
    }
    let mut request = Request::new(prompt);
    let paths: Vec<&str> = match args.get("images") {
        None | Some(Json::Null) => Vec::new(),
        Some(Json::String(p)) => vec![p.as_str()],
        Some(Json::Array(list)) => list
            .iter()
            .map(|v| v.as_str().ok_or("images must be workspace paths"))
            .collect::<Result<_, _>>()?,
        Some(_) => return Err("images must be a list of workspace paths".into()),
    };
    if paths.len() > MAX_IMAGES {
        return Err(format!("an edit takes at most {MAX_IMAGES} images"));
    }
    for p in paths {
        let bytes = sb.read_bytes(&sb.resolve(p)?)?;
        request
            .images
            .push(InputImage::decode(&bytes).map_err(|e| format!("{p}: {e}"))?);
    }
    let size = match args.get("size").and_then(Json::as_str) {
        Some(s) => s
            .split_once(['x', 'X'])
            .and_then(|(w, h)| Some((w.trim().parse().ok()?, h.trim().parse().ok()?)))
            .ok_or_else(|| format!("size must look like 1024x1024, not {s:?}"))?,
        None => (1024, 1024),
    };
    // An edit keeps the last image's shape at the size's area.
    if request.images.is_empty() {
        request.size = Some(size);
    } else {
        request.resolution = ((size.0 * size.1) as f64).sqrt().round() as usize;
    }
    request.seed = arg_int(args, "seed").map_or_else(
        || {
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map_or(0, |t| t.as_nanos() as u64 & 0xffff_ffff)
        },
        |s| s as u64,
    );
    let out = match args.get("path").and_then(Json::as_str) {
        Some(p) if !p.to_ascii_lowercase().ends_with(".png") => {
            return Err(format!(
                "{p}: the picture is saved as PNG, so the path must end in .png"
            ));
        }
        Some(p) => p.to_string(),
        None => {
            let stamp = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map_or(0, |t| t.as_secs());
            format!("images/ferrum-{stamp}-{}.png", request.seed)
        }
    };
    Ok(Job {
        request,
        out: sb.resolve(&out)?,
        sandbox: sb.clone(),
    })
}

/// A job for the chat model's thread, with where to send its stages and its
/// result (`(turn, ok, text, seconds)`, as for the other tools).
pub struct Work {
    pub idx: usize,
    pub job: Job,
    pub cancel: Arc<AtomicBool>,
    pub reply: Sender<(usize, bool, String, f32)>,
}

struct Watch<'a> {
    progress: &'a dyn Fn(String),
    cancel: &'a AtomicBool,
}

impl Observer for Watch<'_> {
    fn stage(&mut self, stage: Stage) {
        (self.progress)(stage_label(stage));
    }
    fn cancelled(&self) -> bool {
        self.cancel.load(Ordering::SeqCst)
    }
}

/// Make the picture, calling `release` to free the chat model first when
/// the stages would not fit beside it. Returns the text for the model.
pub fn run(
    model: &ImageModel,
    work: &Work,
    release: &mut dyn FnMut(),
    progress: &dyn Fn(String),
) -> Result<String, String> {
    let device = MetalDevice::new().map_err(|e| e.to_string())?;
    let pipeline = Pipeline::open(&model.dir, model.settings.clone()).map_err(|e| e.to_string())?;
    let request = &work.job.request;
    if !pipeline
        .fits_now(&device, request)
        .map_err(|e| e.to_string())?
    {
        progress("unloading the chat model to make room".into());
        release();
    }
    let started = Instant::now();
    let mut watch = Watch {
        progress,
        cancel: &work.cancel,
    };
    let image = pipeline
        .generate(&device, request, &mut watch)
        .map_err(|e| match e {
            Error::Cancelled => "cancelled by the user".to_string(),
            Error::Allocation(_) => {
                "not enough free memory for this picture: try a smaller size, or q8_0 in the Image tab".into()
            }
            e => e.to_string(),
        })?;
    let png = image.png().map_err(|e| e.to_string())?;
    let sb = &work.job.sandbox;
    if let Some(dir) = work.job.out.parent() {
        fs::create_dir_all(dir).map_err(|e| e.to_string())?;
    }
    sb.write_bytes(&work.job.out, &png)?;
    Ok(format!(
        "saved {} ({}x{}, seed {}) in {:.0} s",
        sb.rel(&work.job.out),
        image.width,
        image.height,
        image.seed,
        started.elapsed().as_secs_f32()
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sandbox(tag: &str) -> (Sandbox, PathBuf) {
        let dir =
            std::env::temp_dir().join(format!("ferrum-imagetool-{}-{tag}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        let sb = Sandbox::new(&dir.display().to_string(), false).expect("sandbox");
        (sb, dir)
    }

    fn args(v: Json) -> Map<String, Json> {
        v.as_object().unwrap().clone()
    }

    fn png(sb: &Sandbox, name: &str, w: u32, h: u32) {
        let mut bytes = Vec::new();
        image::RgbaImage::new(w, h)
            .write_to(
                &mut std::io::Cursor::new(&mut bytes),
                image::ImageFormat::Png,
            )
            .unwrap();
        sb.write_bytes(&sb.workspace.join(name), &bytes).unwrap();
    }

    #[test]
    fn a_new_picture_takes_its_size_and_lands_in_the_workspace() {
        let (sb, dir) = sandbox("new");
        let job = job(
            &sb,
            &args(json!({"prompt": "a fox", "size": "768x512", "seed": 3})),
        )
        .unwrap();
        assert_eq!((job.request.size, job.request.seed), (Some((768, 512)), 3));
        assert!(job.request.images.is_empty());
        let out = sb.rel(&job.out);
        assert!(
            out.starts_with("images/ferrum-") && out.ends_with("-3.png"),
            "{out}"
        );
        let named = job_at(&sb, "art/fox.png");
        assert_eq!(sb.rel(&named), "art/fox.png");
        let _ = fs::remove_dir_all(dir);
    }

    fn job_at(sb: &Sandbox, path: &str) -> PathBuf {
        job(sb, &args(json!({"prompt": "a fox", "path": path})))
            .unwrap()
            .out
    }

    #[test]
    fn an_edit_reads_workspace_images_and_keeps_their_shape() {
        let (sb, dir) = sandbox("edit");
        png(&sb, "in.png", 64, 32);
        let job = job(
            &sb,
            &args(json!({"prompt": "make it blue", "images": ["in.png"], "size": "512x512"})),
        )
        .unwrap();
        assert_eq!(job.request.images.len(), 1);
        assert_eq!(job.request.size, None);
        assert_eq!(job.request.resolution, 512);
        assert_eq!(job.request.output_size(), (736, 352));
        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn bad_calls_are_refused() {
        let (sb, dir) = sandbox("bad");
        png(&sb, "in.png", 32, 32);
        for bad in [
            json!({}),
            json!({"prompt": "  "}),
            json!({"prompt": "a", "size": "big"}),
            json!({"prompt": "a", "path": "out.jpg"}),
            json!({"prompt": "a", "path": "/tmp/escape.png"}),
            json!({"prompt": "a", "path": "../escape.png"}),
            json!({"prompt": "a", "images": ["missing.png"]}),
            json!({"prompt": "a", "images": ["/etc/hosts"]}),
            json!({"prompt": "a", "images": [1]}),
            json!({"prompt": "a", "images": ["in.png", "in.png", "in.png", "in.png", "in.png"]}),
        ] {
            assert!(job(&sb, &args(bad.clone())).is_err(), "{bad}");
        }
        let _ = fs::remove_dir_all(dir);
    }

    /// `FERRUM_IMAGE_MODEL` is a Qwen-Image-2.1-Turbo directory. The draw
    /// runs beside a stand-in chat model that leaves too little room, so it
    /// has to be released; the edit then has the memory to itself.
    #[test]
    #[ignore = "needs the Qwen-Image model on disk"]
    fn draws_then_edits_its_own_picture() {
        let model = ImageModel {
            dir: std::env::var("FERRUM_IMAGE_MODEL")
                .expect("FERRUM_IMAGE_MODEL")
                .into(),
            settings: Settings::default(),
        };
        let (sb, dir) = sandbox("run");
        let call = |a: Json, release: &mut dyn FnMut()| {
            let (reply, _) = std::sync::mpsc::channel();
            let work = Work {
                idx: 0,
                job: job(&sb, &args(a)).unwrap(),
                cancel: Arc::new(AtomicBool::new(false)),
                reply,
            };
            let text = run(&model, &work, release, &|_| {}).unwrap();
            (sb.rel(&work.job.out), text)
        };
        let device = MetalDevice::new().unwrap();
        let free = (device.recommended_max_working_set() as usize) - device.allocated_bytes();
        // In pieces: one buffer cannot be that large.
        let piece = 1 << 30;
        let ballast: Vec<_> = (0..free / piece - 1)
            .map(|_| device.allocate(piece).unwrap())
            .collect();
        let mut chat = Some(ballast);
        let (out, text) = call(
            json!({"prompt": "a red fox", "size": "256x256", "seed": 1}),
            &mut || chat = None,
        );
        assert!(chat.is_none(), "the stand-in chat model was not released");
        assert!(
            text.starts_with(&format!("saved {out} (256x256, seed 1)")),
            "{text}"
        );
        let (edited, text) = call(
            json!({
                "prompt": "Give the fox a blue hat.", "images": [out], "size": "256x256", "path": "hat.png"
            }),
            &mut || panic!("nothing to release"),
        );
        assert_eq!(edited, "hat.png");
        assert!(text.contains("(256x256"), "{text}");
        let _ = fs::remove_dir_all(dir);
    }
}
