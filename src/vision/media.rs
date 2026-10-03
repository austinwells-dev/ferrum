//! Images inside chat messages: finding them, decoding their payloads and
//! replacing them with the placeholder the chat template understands.
#![forbid(unsafe_code)]
use crate::{Error, Result};
use serde_json::{Value as Json, json};

/// Most images one request may carry.
pub const MAX_IMAGES: usize = 16;

/// Decode standard or URL-safe base64, ignoring whitespace and padding.
pub fn base64_decode(text: &str) -> Result<Vec<u8>> {
    let mut out = Vec::with_capacity(text.len() / 4 * 3);
    let (mut acc, mut bits) = (0u32, 0u32);
    for c in text.bytes() {
        let v = match c {
            b'A'..=b'Z' => c - b'A',
            b'a'..=b'z' => c - b'a' + 26,
            b'0'..=b'9' => c - b'0' + 52,
            b'+' | b'-' => 62,
            b'/' | b'_' => 63,
            b'=' | b' ' | b'\n' | b'\r' | b'\t' => continue,
            _ => return Err(Error::Parameter("invalid base64 in image data".into())),
        };
        acc = (acc << 6) | u32::from(v);
        bits += 6;
        if bits >= 8 {
            bits -= 8;
            out.push((acc >> bits) as u8);
            acc &= (1 << bits) - 1;
        }
    }
    Ok(out)
}

/// Standard base64 with padding.
pub fn base64_encode(data: &[u8]) -> String {
    const ALPHABET: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::with_capacity(data.len().div_ceil(3) * 4);
    for chunk in data.chunks(3) {
        let n = chunk
            .iter()
            .enumerate()
            .fold(0u32, |n, (i, &b)| n | u32::from(b) << (16 - 8 * i));
        for i in 0..4 {
            if i <= chunk.len() {
                out.push(ALPHABET[(n >> (18 - 6 * i) & 63) as usize] as char);
            } else {
                out.push('=');
            }
        }
    }
    out
}

/// A chat content part carrying `bytes` as an inline image.
pub fn image_part_from_bytes(bytes: &[u8]) -> Json {
    let mime = match bytes {
        [0x89, b'P', b'N', b'G', ..] => "image/png",
        [0xff, 0xd8, ..] => "image/jpeg",
        [b'G', b'I', b'F', ..] => "image/gif",
        [b'R', b'I', b'F', b'F', ..] => "image/webp",
        [b'B', b'M', ..] => "image/bmp",
        _ => "application/octet-stream",
    };
    json!({"type": "image_url", "image_url": {"url": format!("data:{mime};base64,{}", base64_encode(bytes))}})
}

/// The bytes of a `data:<type>;base64,<payload>` URL.
pub fn decode_data_url(url: &str) -> Result<Vec<u8>> {
    let Some(rest) = url.strip_prefix("data:") else {
        return Err(Error::Parameter(
            "images must be sent inline as data: URLs or base64; remote and file URLs are not fetched"
                .into(),
        ));
    };
    match rest.split_once(',') {
        Some((head, payload)) if head.ends_with(";base64") => base64_decode(payload),
        _ => Err(Error::Parameter(
            "image data: URLs must be base64 encoded".into(),
        )),
    }
}

fn url_of(part: &Json, key: &str) -> Option<String> {
    match part.get(key)? {
        Json::String(s) => Some(s.clone()),
        Json::Object(o) => o.get("url")?.as_str().map(str::to_owned),
        _ => None,
    }
}

/// The encoded bytes of an image content part, if `part` is one.
fn image_part(part: &Json) -> Option<Result<Vec<u8>>> {
    let kind = part.get("type").and_then(Json::as_str);
    if matches!(kind, Some("image_url" | "input_image")) || part.get("image_url").is_some() {
        return Some(match url_of(part, "image_url") {
            Some(url) => decode_data_url(&url),
            None => Err(Error::Parameter("image_url part has no url".into())),
        });
    }
    if kind == Some("image") || part.get("image").is_some() {
        if let Some(source) = part.get("source") {
            return Some(match source.get("type").and_then(Json::as_str) {
                Some("base64") => match source.get("data").and_then(Json::as_str) {
                    Some(data) => base64_decode(data),
                    None => Err(Error::Parameter("image source has no data".into())),
                },
                Some("url") => match source.get("url").and_then(Json::as_str) {
                    Some(url) => decode_data_url(url),
                    None => Err(Error::Parameter("image source has no url".into())),
                },
                _ => Err(Error::Parameter("unsupported image source".into())),
            });
        }
        let url = url_of(part, "image").or_else(|| url_of(part, "url"));
        return Some(match url {
            Some(url) => decode_data_url(&url),
            None => Err(Error::Parameter("image part has no data".into())),
        });
    }
    None
}

/// Does any message carry an image part?
pub fn has_images(messages: &[Json]) -> bool {
    messages.iter().any(|m| {
        m.get("content")
            .and_then(Json::as_array)
            .is_some_and(|parts| parts.iter().any(|p| image_part(p).is_some()))
    })
}

/// Replace every image part with `{"type": "image"}` (what chat templates
/// look for) and return the decoded images in prompt order.
pub fn take_images(messages: &[Json]) -> Result<(Vec<Json>, Vec<Vec<u8>>)> {
    let mut images = Vec::new();
    let mut out = Vec::with_capacity(messages.len());
    for message in messages {
        let mut message = message.clone();
        if let Some(parts) = message.get_mut("content").and_then(Json::as_array_mut) {
            for part in parts {
                if let Some(bytes) = image_part(part) {
                    if images.len() >= MAX_IMAGES {
                        return Err(Error::Parameter(format!(
                            "at most {MAX_IMAGES} images per request"
                        )));
                    }
                    images.push(bytes?);
                    *part = json!({"type": "image"});
                }
            }
        }
        out.push(message);
    }
    Ok((out, images))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn base64_round_trips_known_vectors() {
        assert_eq!(base64_decode("").unwrap(), b"");
        assert_eq!(base64_decode("Zg==").unwrap(), b"f");
        assert_eq!(base64_decode("Zm8=").unwrap(), b"fo");
        assert_eq!(base64_decode("Zm9v").unwrap(), b"foo");
        assert_eq!(base64_decode("Zm9vYg").unwrap(), b"foob");
        assert_eq!(base64_decode("Zm9v\nYmFy").unwrap(), b"foobar");
        for data in [
            &b""[..],
            b"f",
            b"fo",
            b"foo",
            b"foob",
            &[0xfb, 0xff, 0x00, 0x10],
        ] {
            assert_eq!(base64_decode(&base64_encode(data)).unwrap(), data);
        }
        assert_eq!(base64_encode(b"fo"), "Zm8=");
        assert_eq!(base64_decode("-_8").unwrap(), [0xfb, 0xff]);
        assert!(base64_decode("a*b").is_err());
    }

    #[test]
    fn finds_openai_and_anthropic_images() {
        let messages = vec![
            json!({"role": "system", "content": "be brief"}),
            json!({"role": "user", "content": [
                {"type": "text", "text": "what is this?"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,Zm9v"}},
                {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "YmFy"}},
                {"type": "image_url", "image_url": "data:image/jpeg;base64,YmF6"},
            ]}),
        ];
        assert!(has_images(&messages));
        let (clean, images) = take_images(&messages).unwrap();
        assert_eq!(images, [b"foo".to_vec(), b"bar".to_vec(), b"baz".to_vec()]);
        assert_eq!(clean[1]["content"][0]["text"], "what is this?");
        assert_eq!(clean[1]["content"][1], json!({"type": "image"}));
        assert!(!has_images(&clean[..1]));
        // The input is untouched.
        assert!(messages[1]["content"][1].get("image_url").is_some());
    }

    #[test]
    fn rejects_remote_and_malformed_images() {
        let remote = [json!({"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": "https://example.com/a.png"}}]})];
        assert!(take_images(&remote).is_err());
        let plain = [json!({"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": "data:text/plain,hello"}}]})];
        assert!(take_images(&plain).is_err());
        let text_only = [json!({"role": "user", "content": "hi"})];
        assert!(!has_images(&text_only));
        let many: Vec<Json> = (0..=MAX_IMAGES)
            .map(|_| {
                json!({"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,Zm9v"}}]})
            })
            .collect();
        assert!(take_images(&many).is_err());
    }
}
