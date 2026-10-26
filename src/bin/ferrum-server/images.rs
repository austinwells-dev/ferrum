//! `POST /v1/images/generations` and `/v1/images/edits` (OpenAI's images
//! API) on their own thread.
//!
//! The pipeline loads and frees its models for every request, so it needs
//! the memory of one stage at a time next to whatever else is resident; a request that would not fit is refused with
//! 503 instead of loading. Requests queue FIFO. Images come back as base64
//! PNG (`response_format: "b64_json"`); there are no URLs to serve. Edits
//! take their images as multipart file uploads (`image` or `image[]`, as
//! OpenAI's clients send them) or, in a JSON body, as base64 or data: URLs.
use crate::{debug, error, http, info, multipart, warn, worker::truncate};
use ferrum::{
    Error, MetalDevice,
    diffusion::pipeline::{
        self, Generated, InputImage, Observer, Pipeline, Request, Settings, Stage,
    },
    vision::media::{base64_decode, base64_encode, decode_data_url},
};
use serde_json::{Value as Json, json};
use std::{
    net::TcpStream,
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, Ordering},
        mpsc::{Receiver, Sender, channel},
    },
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

/// Most images one request may ask for (they generate one after another).
const MAX_N: usize = 4;

enum Event {
    Image { png: Vec<u8>, seed: u64 },
    Failed { status: u16, message: String },
    Done,
}

struct Job {
    id: u64,
    request: Request,
    n: usize,
    events: Sender<Event>,
    cancel: Arc<AtomicBool>,
}

pub struct Service {
    jobs: Mutex<Sender<Job>>,
    pub model: String,
}

impl Service {
    /// Open the model directory and start the thread that serves it.
    pub fn start(dir: &str, settings: Settings, model: String) -> Result<Self, String> {
        let pipeline = Pipeline::open(dir, settings).map_err(|e| format!("{dir}: {e}"))?;
        let (tx, rx) = channel();
        std::thread::Builder::new()
            .name("images".into())
            .spawn(move || worker(&pipeline, &rx))
            .map_err(|e| e.to_string())?;
        Ok(Self {
            jobs: Mutex::new(tx),
            model,
        })
    }
}

struct Watch<'a> {
    job: &'a Job,
}

impl Observer for Watch<'_> {
    fn stage(&mut self, stage: Stage) {
        debug!(Some(self.job.id), "{stage:?}");
    }
    fn cancelled(&self) -> bool {
        self.job.cancel.load(Ordering::SeqCst)
    }
}

fn worker(pipeline: &Pipeline, jobs: &Receiver<Job>) {
    let device = match MetalDevice::new() {
        Ok(d) => d,
        Err(e) => {
            error!(None, "image thread: {e}");
            return;
        }
    };
    for job in jobs {
        run(&job, &mut |request, observer| {
            pipeline.generate(&device, request, observer)
        });
    }
}

/// The pipeline's `generate`, or a stand-in for it in tests.
type Generate<'a> = dyn FnMut(&Request, &mut dyn Observer) -> ferrum::Result<Generated> + 'a;

/// Generate `job`'s images one after another, sending each, or the failure
/// that ends the job, and then `Done`.
fn run(job: &Job, generate: &mut Generate) {
    let id = Some(job.id);
    for i in 0..job.n {
        let mut request = job.request.clone();
        request.seed = request.seed.wrapping_add(i as u64);
        let started = Instant::now();
        // A panic fails this request, not the thread every later one needs.
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            generate(&request, &mut Watch { job })
        }));
        let event = match result {
            Err(panic) => {
                let message = panic
                    .downcast_ref::<String>()
                    .map(String::as_str)
                    .or_else(|| panic.downcast_ref::<&str>().copied())
                    .unwrap_or("unknown panic");
                error!(id, "image generation panicked: {message}");
                Event::Failed {
                    status: 500,
                    message: format!("image generation panicked: {message}"),
                }
            }
            Ok(Ok(image)) => match image.png() {
                Ok(png) => {
                    let t = image.timings;
                    info!(
                        id,
                        "image {}/{}: {}x{} seed {} in {:.1} s (load {:.1}, encode {:.1}, denoise {:.1}, decode {:.1})",
                        i + 1,
                        job.n,
                        image.width,
                        image.height,
                        image.seed,
                        started.elapsed().as_secs_f32(),
                        t.load.as_secs_f32(),
                        t.encode.as_secs_f32(),
                        t.denoise.as_secs_f32(),
                        t.decode.as_secs_f32()
                    );
                    Event::Image {
                        png,
                        seed: image.seed,
                    }
                }
                Err(e) => Event::Failed {
                    status: 500,
                    message: e.to_string(),
                },
            },
            Ok(Err(Error::Cancelled)) => {
                info!(id, "cancelled");
                break;
            }
            Ok(Err(e @ Error::Allocation(_))) => {
                warn!(id, "refused: {e}");
                Event::Failed {
                    status: 503,
                    message: format!(
                        "not enough free memory for this image ({e}); use a smaller size, quantize with --image-quantize q8_0, or stop other models"
                    ),
                }
            }
            Ok(Err(e @ Error::Parameter(_))) => Event::Failed {
                status: 400,
                message: e.to_string(),
            },
            Ok(Err(e)) => {
                error!(id, "generation failed: {e}");
                Event::Failed {
                    status: 500,
                    message: e.to_string(),
                }
            }
        };
        let failed = matches!(event, Event::Failed { .. });
        if job.events.send(event).is_err() || failed {
            break;
        }
    }
    let _ = job.events.send(Event::Done);
}

/// `"1024x1024"` as a width and height.
fn parse_size(size: &str) -> Option<(usize, usize)> {
    let (w, h) = size.split_once(['x', 'X'])?;
    Some((w.trim().parse().ok()?, h.trim().parse().ok()?))
}

fn bad(stream: &mut TcpStream, message: &str, ka: bool) -> std::io::Result<()> {
    http::json(
        stream,
        400,
        &crate::error_json(message, "invalid_request_error"),
        ka,
    )
}

/// `POST /v1/images/generations`: a JSON body.
pub fn generations(
    server: &crate::Server,
    stream: &mut TcpStream,
    body: &Json,
    id: u64,
    ka: bool,
) -> std::io::Result<()> {
    serve(server, stream, body, Vec::new(), id, ka)
}

/// `POST /v1/images/edits`: multipart with image files, or JSON with
/// `image` as one base64 string / data: URL or a list of them.
pub fn edits(
    server: &crate::Server,
    stream: &mut TcpStream,
    request: &http::Request,
    id: u64,
    ka: bool,
) -> std::io::Result<()> {
    let parsed = match request.header("content-type").and_then(multipart::boundary) {
        Some(boundary) => form(&request.body, boundary),
        None => serde_json::from_slice::<Json>(&request.body)
            .map_err(|e| format!("invalid JSON body: {e}"))
            .and_then(|body| {
                let images = json_images(&body)?;
                Ok((body, images))
            }),
    };
    match parsed {
        Ok((_, images)) if images.is_empty() => bad(stream, "an edit needs at least one image", ka),
        Ok((body, images)) => serve(server, stream, &body, images, id, ka),
        Err(e) => bad(stream, &e, ka),
    }
}

/// A multipart form as a JSON object of its text fields (numbers where they
/// parse), and its image files in order.
fn form(body: &[u8], boundary: &str) -> Result<(Json, Vec<Vec<u8>>), String> {
    let mut fields = serde_json::Map::new();
    let mut images = Vec::new();
    for part in multipart::parse(body, boundary)? {
        if matches!(part.name.as_str(), "image" | "image[]") {
            images.push(part.body.to_vec());
            continue;
        }
        let text = String::from_utf8(part.body.to_vec())
            .map_err(|_| format!("form field {} is not UTF-8", part.name))?;
        let value = text.parse::<u64>().map_or(Json::String(text), Json::from);
        fields.insert(part.name, value);
    }
    Ok((Json::Object(fields), images))
}

/// `image` in a JSON body: base64 or data: URLs, one or a list.
fn json_images(body: &Json) -> Result<Vec<Vec<u8>>, String> {
    let one = |v: &Json| -> Result<Vec<u8>, String> {
        let s = v
            .as_str()
            .ok_or("each image must be a base64 string or data: URL")?;
        let bytes = if s.starts_with("data:") {
            decode_data_url(s)
        } else {
            base64_decode(s)
        };
        bytes.map_err(|e| e.to_string())
    };
    match body.get("image") {
        None | Some(Json::Null) => Ok(Vec::new()),
        Some(Json::Array(list)) => list.iter().map(one).collect(),
        Some(v) => one(v).map(|b| vec![b]),
    }
}

/// The pipeline request a body asks for, or why it is invalid.
fn request(body: &Json, images: &[Vec<u8>]) -> Result<(Request, usize), String> {
    let prompt = body.get("prompt").and_then(Json::as_str).unwrap_or("");
    if prompt.trim().is_empty() {
        return Err("prompt is required".into());
    }
    let uint = |key: &str| -> Result<Option<u64>, String> {
        match body.get(key) {
            None | Some(Json::Null) => Ok(None),
            Some(v) => v
                .as_u64()
                .map(Some)
                .ok_or_else(|| format!("{key} must be a non-negative integer")),
        }
    };
    let n = uint("n")?.unwrap_or(1);
    if !(1..=MAX_N as u64).contains(&n) {
        return Err(format!("n must be 1 to {MAX_N}"));
    }
    if body
        .get("response_format")
        .and_then(Json::as_str)
        .is_some_and(|f| f != "b64_json")
    {
        return Err(
            "only response_format \"b64_json\" is supported (the server keeps no files to link to)"
                .into(),
        );
    }
    let mut request = Request::new(prompt);
    request.size = match body.get("size").and_then(Json::as_str) {
        None | Some("auto") => None,
        Some(size) => {
            let (w, h) = parse_size(size)
                .ok_or_else(|| format!("size must look like 1024x1024 or auto, not {size:?}"))?;
            pipeline::check_size(w, h).map_err(|e| e.to_string())?;
            Some((w, h))
        }
    };
    if images.len() > pipeline::MAX_IMAGES {
        return Err(format!(
            "{} images; an edit takes at most {}",
            images.len(),
            pipeline::MAX_IMAGES
        ));
    }
    request.images = images
        .iter()
        .map(|b| InputImage::decode(b))
        .collect::<Result<_, _>>()
        .map_err(|e| e.to_string())?;
    if let Some(r) = uint("resolution")? {
        request.resolution = r as usize;
    }
    request.steps = uint("steps")?.map(|s| s as usize);
    request.seed = uint("seed")?.unwrap_or_else(|| {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_or(0, |t| t.as_nanos() as u64 & 0xffff_ffff)
    });
    Ok((request, n as usize))
}

fn serve(
    server: &crate::Server,
    stream: &mut TcpStream,
    body: &Json,
    images: Vec<Vec<u8>>,
    id: u64,
    ka: bool,
) -> std::io::Result<()> {
    let Some(service) = &server.images else {
        return http::json(
            stream,
            404,
            &crate::error_json(
                "this server has no image model (start it with --image-model DIR)",
                "not_found_error",
            ),
            ka,
        );
    };
    let (request, n) = match request(body, &images) {
        Ok(r) => r,
        Err(e) => return bad(stream, &e, ka),
    };
    drop(images);
    let (width, height) = request.output_size();
    info!(
        Some(id),
        "image request: {width}x{height}, {} condition images, n {n}, prompt {:?}",
        request.images.len(),
        truncate(&request.prompt, 80)
    );

    let (tx, rx) = channel();
    let cancel = Arc::new(AtomicBool::new(false));
    http::watch_hangup(stream, id, &cancel);
    let job = Job {
        id,
        request,
        n,
        events: tx,
        cancel: cancel.clone(),
    };
    if !service.jobs.lock().is_ok_and(|jobs| jobs.send(job).is_ok()) {
        error!(Some(id), "image thread is gone");
        return http::json(
            stream,
            500,
            &crate::error_json("image thread is gone", "server_error"),
            ka,
        );
    }
    let mut data = Vec::new();
    loop {
        match rx.recv_timeout(Duration::from_millis(500)) {
            Ok(Event::Image { png, seed }) => {
                data.push(json!({"b64_json": base64_encode(&png), "seed": seed}));
            }
            Ok(Event::Failed { status, message }) => {
                let kind = if status == 503 {
                    "unavailable_error"
                } else if status == 400 {
                    "invalid_request_error"
                } else {
                    "server_error"
                };
                return http::json(stream, status, &crate::error_json(&message, kind), ka);
            }
            Ok(Event::Done) => break,
            Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => {
                // The worker dropped the job without finishing it.
                if !cancel.load(Ordering::SeqCst) {
                    error!(Some(id), "the image thread ended without an answer");
                    return http::json(
                        stream,
                        500,
                        &crate::error_json(
                            "the image thread ended without an answer",
                            "server_error",
                        ),
                        ka,
                    );
                }
                break;
            }
            Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {}
        }
    }
    if cancel.load(Ordering::SeqCst) {
        return Ok(());
    }
    let created = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |t| t.as_secs());
    http::json(stream, 200, &json!({"created": created, "data": data}), ka)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Run a job of `n` images through `generate`; the events it sent.
    fn events(n: usize, cancel: bool, generate: &mut Generate) -> Vec<Event> {
        let (tx, rx) = channel();
        let job = Job {
            id: 1,
            request: Request::new("a fox"),
            n,
            events: tx,
            cancel: Arc::new(AtomicBool::new(cancel)),
        };
        run(&job, generate);
        drop(job);
        rx.iter().collect()
    }

    fn image(seed: u64) -> Generated {
        Generated {
            width: 1,
            height: 1,
            rgba: vec![255; 4],
            timings: Default::default(),
            seed,
        }
    }

    /// `(status, or 200 for an image)` per event, `Done` as 0.
    fn statuses(events: &[Event]) -> Vec<u16> {
        events
            .iter()
            .map(|e| match e {
                Event::Image { .. } => 200,
                Event::Failed { status, .. } => *status,
                Event::Done => 0,
            })
            .collect()
    }

    #[test]
    fn each_image_takes_the_next_seed() {
        let got = events(3, false, &mut |r, _| Ok(image(r.seed)));
        let seeds: Vec<u64> = got
            .iter()
            .filter_map(|e| match e {
                Event::Image { seed, .. } => Some(*seed),
                _ => None,
            })
            .collect();
        assert_eq!(seeds, [0, 1, 2]);
        assert!(matches!(got.last(), Some(Event::Done)));
    }

    #[test]
    fn failures_map_to_statuses_and_end_the_job() {
        let fail = |e: fn() -> Error| {
            let mut first = true;
            statuses(&events(3, false, &mut move |r, _| {
                if std::mem::take(&mut first) {
                    Ok(image(r.seed))
                } else {
                    Err(e())
                }
            }))
        };
        assert_eq!(fail(|| Error::Allocation(16 << 30)), [200, 503, 0]);
        assert_eq!(fail(|| Error::Parameter("0 steps".into())), [200, 400, 0]);
        assert_eq!(fail(|| Error::Shape("odd".into())), [200, 500, 0]);
    }

    #[test]
    fn a_panic_fails_its_request_and_the_next_one_runs() {
        let got = events(1, false, &mut |_, _| panic!("sigmas[steps - 1]"));
        assert_eq!(statuses(&got), [500, 0]);
        let Event::Failed { message, .. } = &got[0] else {
            unreachable!()
        };
        assert!(message.contains("sigmas[steps - 1]"), "{message}");
        assert_eq!(
            statuses(&events(1, false, &mut |r, _| Ok(image(r.seed)))),
            [200, 0]
        );
    }

    #[test]
    fn a_cancelled_job_ends_without_an_answer() {
        let got = events(4, true, &mut |r, observer| {
            if observer.cancelled() {
                Err(Error::Cancelled)
            } else {
                Ok(image(r.seed))
            }
        });
        assert_eq!(statuses(&got), [0]);
    }

    #[test]
    fn sizes_parse_as_width_by_height() {
        assert_eq!(parse_size("1024x768"), Some((1024, 768)));
        assert_eq!(parse_size("512X512"), Some((512, 512)));
        assert_eq!(parse_size("big"), None);
        assert_eq!(parse_size("1024x"), None);
    }

    #[test]
    fn requests_are_validated() {
        let ok = request(&json!({"prompt": "a fox"}), &[]).unwrap();
        assert_eq!((ok.0.size, ok.1, ok.0.images.len()), (None, 1, 0));
        let sized = request(&json!({"prompt": "a", "size": "512x768", "n": 2}), &[]).unwrap();
        assert_eq!((sized.0.size, sized.1), (Some((512, 768)), 2));
        for bad in [
            json!({}),
            json!({"prompt": " "}),
            json!({"prompt": "a", "n": 0}),
            json!({"prompt": "a", "n": 5}),
            json!({"prompt": "a", "steps": -1}),
            json!({"prompt": "a", "steps": "eight"}),
            json!({"prompt": "a", "size": "100x100"}),
            json!({"prompt": "a", "response_format": "url"}),
        ] {
            assert!(request(&bad, &[]).is_err(), "{bad}");
        }
        assert!(request(&json!({"prompt": "a"}), &[b"not an image".to_vec()]).is_err());
        let png = {
            let mut out = Vec::new();
            image::RgbaImage::new(64, 32)
                .write_to(&mut std::io::Cursor::new(&mut out), image::ImageFormat::Png)
                .unwrap();
            out
        };
        let edit = request(&json!({"prompt": "a"}), std::slice::from_ref(&png)).unwrap();
        assert_eq!(edit.0.images.len(), 1);
        assert_eq!(edit.0.output_size(), (1440, 736));
        assert!(request(&json!({"prompt": "a"}), &vec![png; 5]).is_err());
    }
}
