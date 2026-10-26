//! `POST /v1/images/generations` (OpenAI's images API) on its own thread.
//!
//! The pipeline loads and frees its three models for every request, so it
//! needs the memory of one stage at a time (about 15 GB for BF16) next to
//! whatever else is resident; a request that would not fit is refused with
//! 503 instead of loading. Requests queue FIFO. Images come back as base64
//! PNG (`response_format: "b64_json"`); there are no URLs to serve.
use crate::{debug, error, http, info, warn, worker::truncate};
use ferrum::{
    Error, MetalDevice,
    diffusion::pipeline::{self, Observer, Pipeline, Request, Settings, Stage},
    vision::media::base64_encode,
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
        let id = Some(job.id);
        for i in 0..job.n {
            let mut request = job.request.clone();
            request.seed = request.seed.wrapping_add(i as u64);
            let started = Instant::now();
            let result = pipeline.generate(&device, &request, &mut Watch { job: &job });
            let event = match result {
                Ok(image) => match image.png() {
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
                Err(Error::Cancelled) => {
                    info!(id, "cancelled");
                    break;
                }
                Err(e @ Error::Allocation(_)) => {
                    warn!(id, "refused: {e}");
                    Event::Failed {
                        status: 503,
                        message: format!(
                            "not enough free memory for this image ({e}); use a smaller --size, quantize with --image-quantize q8_0, or stop other models"
                        ),
                    }
                }
                Err(e @ Error::Parameter(_)) => Event::Failed {
                    status: 400,
                    message: e.to_string(),
                },
                Err(e) => {
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

pub fn generations(
    server: &crate::Server,
    stream: &mut TcpStream,
    body: &Json,
    id: u64,
    ka: bool,
) -> std::io::Result<()> {
    let Some(images) = &server.images else {
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
    let prompt = body.get("prompt").and_then(Json::as_str).unwrap_or("");
    if prompt.trim().is_empty() {
        return bad(stream, "prompt is required", ka);
    }
    let n = body.get("n").map_or(Some(1), Json::as_u64);
    let Some(n) = n.filter(|n| (1..=MAX_N as u64).contains(n)) else {
        return bad(stream, &format!("n must be 1 to {MAX_N}"), ka);
    };
    if body
        .get("response_format")
        .and_then(Json::as_str)
        .is_some_and(|f| f != "b64_json")
    {
        return bad(
            stream,
            "only response_format \"b64_json\" is supported (the server keeps no files to link to)",
            ka,
        );
    }
    let size = body
        .get("size")
        .and_then(Json::as_str)
        .unwrap_or("1024x1024");
    let Some((width, height)) = parse_size(size) else {
        return bad(
            stream,
            &format!("size must look like 1024x1024, not {size:?}"),
            ka,
        );
    };
    if let Err(e) = pipeline::check_size(width, height) {
        return bad(stream, &e.to_string(), ka);
    }
    let mut request = Request::new(prompt);
    (request.width, request.height) = (width, height);
    request.steps = body.get("steps").and_then(Json::as_u64).map(|s| s as usize);
    request.seed = body.get("seed").and_then(Json::as_u64).unwrap_or_else(|| {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_or(0, |t| t.as_nanos() as u64 & 0xffff_ffff)
    });

    let (tx, rx) = channel();
    let cancel = Arc::new(AtomicBool::new(false));
    http::watch_hangup(stream, id, &cancel);
    let job = Job {
        id,
        request,
        n: n as usize,
        events: tx,
        cancel: cancel.clone(),
    };
    info!(
        Some(id),
        "image request: {width}x{height}, n {n}, prompt {:?}",
        truncate(prompt, 80)
    );
    if !images.jobs.lock().is_ok_and(|jobs| jobs.send(job).is_ok()) {
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
            Ok(Event::Done) | Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => break,
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
    use super::parse_size;

    #[test]
    fn sizes_parse_as_width_by_height() {
        assert_eq!(parse_size("1024x768"), Some((1024, 768)));
        assert_eq!(parse_size("512X512"), Some((512, 512)));
        assert_eq!(parse_size("big"), None);
        assert_eq!(parse_size("1024x"), None);
    }
}
