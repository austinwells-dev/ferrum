//! Minimal HTTP/1.1: keep-alive, Content-Length and chunked request bodies,
//! `Expect: 100-continue`, JSON responses and chunked streaming responses.
use std::{
    io::{self, BufRead, BufReader, Read, Write},
    net::TcpStream,
    os::fd::AsRawFd,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    thread,
    time::Duration,
};

pub struct Request {
    pub method: String,
    pub path: String,
    pub headers: Vec<(String, String)>,
    pub body: Vec<u8>,
    pub keep_alive: bool,
}

impl Request {
    pub fn header(&self, name: &str) -> Option<&str> {
        self.headers
            .iter()
            .find(|(k, _)| k == name)
            .map(|(_, v)| v.as_str())
    }
}

pub const MAX_BODY: usize = 512 << 20;

/// Read one request. `Ok(None)` means the client closed the connection.
pub fn read_request(reader: &mut BufReader<TcpStream>) -> std::io::Result<Option<Request>> {
    let mut line = String::new();
    if reader.read_line(&mut line)? == 0 {
        return Ok(None);
    }
    let line = line.trim_end();
    if line.is_empty() {
        return read_request(reader);
    }
    let mut parts = line.split_whitespace();
    let method = parts.next().unwrap_or_default().to_owned();
    let target = parts.next().unwrap_or_default();
    let version = parts.next().unwrap_or("HTTP/1.1");
    let (path, _query) = target.split_once('?').unwrap_or((target, ""));
    let mut headers = Vec::new();
    loop {
        let mut h = String::new();
        if reader.read_line(&mut h)? == 0 || h.trim().is_empty() {
            break;
        }
        if let Some((k, v)) = h.split_once(':') {
            headers.push((k.trim().to_ascii_lowercase(), v.trim().to_owned()));
        }
    }
    let get = |name: &str| {
        headers
            .iter()
            .find(|(k, _)| k == name)
            .map(|(_, v)| v.clone())
    };
    if get("expect").is_some_and(|v| v.eq_ignore_ascii_case("100-continue")) {
        reader
            .get_mut()
            .write_all(b"HTTP/1.1 100 Continue\r\n\r\n")?;
    }
    let chunked =
        get("transfer-encoding").is_some_and(|v| v.to_ascii_lowercase().contains("chunked"));
    let body = if chunked {
        let mut body = Vec::new();
        loop {
            let mut size = String::new();
            reader.read_line(&mut size)?;
            let size = usize::from_str_radix(size.trim().split(';').next().unwrap_or("0"), 16)
                .map_err(|_| std::io::Error::other("bad chunk size"))?;
            if size == 0 {
                let mut trailer = String::new();
                while reader.read_line(&mut trailer)? > 0 && !trailer.trim().is_empty() {
                    trailer.clear();
                }
                break;
            }
            if body.len() + size > MAX_BODY {
                return Err(std::io::Error::other("request body too large"));
            }
            let start = body.len();
            body.resize(start + size, 0);
            reader.read_exact(&mut body[start..])?;
            let mut crlf = [0; 2];
            reader.read_exact(&mut crlf)?;
        }
        body
    } else {
        let length: usize = get("content-length")
            .and_then(|v| v.parse().ok())
            .unwrap_or(0);
        if length > MAX_BODY {
            return Err(std::io::Error::other("request body too large"));
        }
        let mut body = vec![0; length];
        reader.read_exact(&mut body)?;
        body
    };
    let connection = get("connection").unwrap_or_default().to_ascii_lowercase();
    let keep_alive = if version == "HTTP/1.0" {
        connection.contains("keep-alive")
    } else {
        !connection.contains("close")
    };
    Ok(Some(Request {
        method,
        path: path.to_owned(),
        headers,
        body,
        keep_alive,
    }))
}

pub fn reason(status: u16) -> &'static str {
    match status {
        200 => "OK",
        204 => "No Content",
        400 => "Bad Request",
        401 => "Unauthorized",
        404 => "Not Found",
        405 => "Method Not Allowed",
        413 => "Payload Too Large",
        501 => "Not Implemented",
        503 => "Service Unavailable",
        _ => "Internal Server Error",
    }
}

const COMMON: &str = "Access-Control-Allow-Origin: *\r\nAccess-Control-Allow-Headers: *\r\nAccess-Control-Allow-Methods: GET, POST, OPTIONS\r\nServer: ferrum\r\n";

pub fn respond(
    stream: &mut TcpStream,
    status: u16,
    content_type: &str,
    body: &[u8],
    keep_alive: bool,
) -> std::io::Result<()> {
    let head = format!(
        "HTTP/1.1 {status} {}\r\nContent-Type: {content_type}\r\nContent-Length: {}\r\n{COMMON}Connection: {}\r\n\r\n",
        reason(status),
        body.len(),
        if keep_alive { "keep-alive" } else { "close" }
    );
    stream.write_all(head.as_bytes())?;
    stream.write_all(body)?;
    stream.flush()
}

pub fn json(
    stream: &mut TcpStream,
    status: u16,
    body: &serde_json::Value,
    keep_alive: bool,
) -> std::io::Result<()> {
    respond(
        stream,
        status,
        "application/json; charset=utf-8",
        body.to_string().as_bytes(),
        keep_alive,
    )
}

/// A chunked-encoding response body (streams and slow JSON replies). It
/// writes through its own handle to the connection's socket.
pub struct Chunked {
    stream: TcpStream,
    finished: bool,
}

impl Chunked {
    pub fn start(
        stream: &TcpStream,
        status: u16,
        content_type: &str,
        keep_alive: bool,
    ) -> std::io::Result<Self> {
        let mut stream = stream.try_clone()?;
        let extra = if content_type.starts_with("text/event-stream") {
            "Cache-Control: no-cache\r\nX-Accel-Buffering: no\r\n"
        } else {
            ""
        };
        let head = format!(
            "HTTP/1.1 {status} {}\r\nContent-Type: {content_type}\r\nTransfer-Encoding: chunked\r\n{extra}{COMMON}Connection: {}\r\n\r\n",
            reason(status),
            if keep_alive { "keep-alive" } else { "close" }
        );
        stream.write_all(head.as_bytes())?;
        stream.flush()?;
        Ok(Self {
            stream,
            finished: false,
        })
    }

    pub fn send(&mut self, data: &[u8]) -> std::io::Result<()> {
        if data.is_empty() {
            return Ok(());
        }
        write!(self.stream, "{:x}\r\n", data.len())?;
        self.stream.write_all(data)?;
        self.stream.write_all(b"\r\n")?;
        self.stream.flush()
    }

    pub fn finish(mut self) -> std::io::Result<()> {
        self.finished = true;
        self.stream.write_all(b"0\r\n\r\n")?;
        self.stream.flush()
    }
}

impl Drop for Chunked {
    fn drop(&mut self) {
        if !self.finished {
            let _ = self.stream.write_all(b"0\r\n\r\n");
            let _ = self.stream.flush();
        }
    }
}

pub fn configure(stream: &TcpStream) {
    let _ = stream.set_nodelay(true);
    let _ = stream.set_read_timeout(Some(Duration::from_secs(300)));
    let _ = stream.set_write_timeout(Some(Duration::from_secs(60)));
}

/// Set `cancel` as soon as the client closes the connection, without waiting
/// for a write to fail: the first write after a hang-up usually still lands
/// in the kernel's send buffer, and a slow prefill chunk may not write for
/// seconds. Polls every 200 ms until the handler and the job both drop
/// `cancel`.
pub fn watch_hangup(stream: &TcpStream, id: u64, cancel: &Arc<AtomicBool>) {
    let Ok(stream) = stream.try_clone() else {
        return;
    };
    let cancel = Arc::downgrade(cancel);
    thread::spawn(move || {
        loop {
            thread::sleep(Duration::from_millis(200));
            let Some(cancel) = cancel.upgrade() else {
                return;
            };
            if cancel.load(Ordering::SeqCst) {
                return;
            }
            if peer_closed(&stream) {
                crate::info!(Some(id), "client closed the connection; cancelling");
                cancel.store(true, Ordering::SeqCst);
                return;
            }
        }
    });
}

/// A non-blocking peek: 0 bytes means the peer sent FIN; an error other than
/// "would block" means the connection was reset. Pipelined request bytes
/// are left in place for the next request.
fn peer_closed(stream: &TcpStream) -> bool {
    unsafe extern "C" {
        fn recv(socket: i32, buffer: *mut u8, length: usize, flags: i32) -> isize;
    }
    const MSG_PEEK: i32 = 0x2;
    const MSG_DONTWAIT: i32 = 0x80;
    let mut byte = 0u8;
    // SAFETY: a one-byte peek into a live local buffer on an open socket.
    let n = unsafe { recv(stream.as_raw_fd(), &mut byte, 1, MSG_PEEK | MSG_DONTWAIT) };
    match n {
        0 => true,
        n if n > 0 => false,
        _ => !matches!(
            io::Error::last_os_error().kind(),
            io::ErrorKind::WouldBlock | io::ErrorKind::Interrupted
        ),
    }
}
