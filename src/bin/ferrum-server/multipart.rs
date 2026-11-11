//! Just enough `multipart/form-data` for OpenAI's image edit uploads: named
//! text fields and files, no nesting, no transfer encodings.

/// One form part: its field name and content.
pub struct Part<'a> {
    pub name: String,
    pub body: &'a [u8],
}

/// The boundary of a `multipart/form-data; boundary=...` content type.
pub fn boundary(content_type: &str) -> Option<&str> {
    let (kind, params) = content_type.split_once(';')?;
    if !kind.trim().eq_ignore_ascii_case("multipart/form-data") {
        return None;
    }
    params.split(';').find_map(|p| {
        let (k, v) = p.trim().split_once('=')?;
        k.eq_ignore_ascii_case("boundary")
            .then(|| v.trim_matches('"'))
            .filter(|b| !b.is_empty())
    })
}

/// The parts of `body`, or a message saying why it is malformed.
pub fn parse<'a>(body: &'a [u8], boundary: &str) -> Result<Vec<Part<'a>>, String> {
    let delimiter = format!("--{boundary}");
    let delimiter = delimiter.as_bytes();
    let mut rest = after(body, delimiter).ok_or("no multipart boundary in the body")?;
    let mut parts = Vec::new();
    loop {
        if rest.starts_with(b"--") {
            return Ok(parts);
        }
        rest = rest
            .strip_prefix(b"\r\n")
            .ok_or("malformed multipart boundary line")?;
        let head_end = find(rest, b"\r\n\r\n").ok_or("multipart part without headers")?;
        let head = std::str::from_utf8(&rest[..head_end])
            .map_err(|_| "multipart headers are not UTF-8")?;
        let content = &rest[head_end + 4..];
        let mut next = Vec::with_capacity(delimiter.len() + 2);
        next.extend_from_slice(b"\r\n");
        next.extend_from_slice(delimiter);
        let end = find(content, &next).ok_or("multipart part is not terminated")?;
        let disposition = head
            .lines()
            .find_map(|l| {
                let (k, v) = l.split_once(':')?;
                k.trim()
                    .eq_ignore_ascii_case("content-disposition")
                    .then_some(v)
            })
            .ok_or("multipart part without Content-Disposition")?;
        parts.push(Part {
            name: param(disposition, "name").ok_or("multipart part without a name")?,
            body: &content[..end],
        });
        rest = &content[end + next.len()..];
    }
}

/// A `key="value"` parameter of a header value.
fn param(header: &str, key: &str) -> Option<String> {
    header.split(';').find_map(|p| {
        let (k, v) = p.trim().split_once('=')?;
        (k.trim() == key).then(|| v.trim().trim_matches('"').to_string())
    })
}

fn find(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    haystack.windows(needle.len()).position(|w| w == needle)
}

fn after<'a>(haystack: &'a [u8], needle: &[u8]) -> Option<&'a [u8]> {
    find(haystack, needle).map(|i| &haystack[i + needle.len()..])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reads_fields_and_files() {
        let body = b"preamble\r\n--XyZ\r\nContent-Disposition: form-data; name=\"prompt\"\r\n\r\nadd a hat\r\n\
--XyZ\r\nContent-Disposition: form-data; name=\"image[]\"; filename=\"a.png\"\r\nContent-Type: image/png\r\n\r\n\x89PNG\r\n--X\r\n\
--XyZ--\r\n";
        let parts = parse(body, "XyZ").unwrap();
        assert_eq!(parts.len(), 2);
        assert_eq!(
            (parts[0].name.as_str(), parts[0].body),
            ("prompt", &b"add a hat"[..])
        );
        assert_eq!(parts[1].name, "image[]");
        // A file may hold CRLF and a near-miss of the boundary.
        assert_eq!(parts[1].body, b"\x89PNG\r\n--X");
    }

    #[test]
    fn rejects_truncated_bodies() {
        let body = b"--b\r\nContent-Disposition: form-data; name=\"x\"\r\n\r\nno end";
        assert!(parse(body, "b").is_err());
        assert!(parse(b"nothing", "b").is_err());
    }

    #[test]
    fn finds_the_boundary() {
        assert_eq!(
            boundary("multipart/form-data; boundary=----abc"),
            Some("----abc")
        );
        assert_eq!(boundary("Multipart/Form-Data; boundary=\"q\""), Some("q"));
        assert_eq!(boundary("application/json"), None);
    }
}
