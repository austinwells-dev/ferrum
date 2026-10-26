//! Images inside chat messages: finding them, decoding their payloads and
//! replacing them with the placeholder the chat template understands.
#![forbid(unsafe_code)]
use super::{ImageBudget, LOW_DETAIL_TOKENS};
use crate::{Error, Result};
use serde_json::{Value as Json, json};

/// Most images one request may carry.
pub const MAX_IMAGES: usize = 16;

/// One image of a request: its encoded bytes and the token limits its
/// content part asked for.
#[derive(Debug, Clone, PartialEq)]
pub struct ImageInput {
    pub bytes: Vec<u8>,
    pub budget: ImageBudget,
}

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
    image_part_with(bytes, ImageBudget::default())
}

/// `image_part_from_bytes` asking for `budget` (as `min_tokens` and
/// `max_tokens` next to the URL).
pub fn image_part_with(bytes: &[u8], budget: ImageBudget) -> Json {
    let mime = match bytes {
        [0x89, b'P', b'N', b'G', ..] => "image/png",
        [0xff, 0xd8, ..] => "image/jpeg",
        [b'G', b'I', b'F', ..] => "image/gif",
        [b'R', b'I', b'F', b'F', ..] => "image/webp",
        [b'B', b'M', ..] => "image/bmp",
        _ => "application/octet-stream",
    };
    let mut url = json!({"url": format!("data:{mime};base64,{}", base64_encode(bytes))});
    if let Some(n) = budget.min_tokens {
        url["min_tokens"] = json!(n);
    }
    if let Some(n) = budget.max_tokens {
        url["max_tokens"] = json!(n);
    }
    json!({"type": "image_url", "image_url": url})
}

/// Parse a per-image size setting: `low`, `high`, `auto` (OpenAI's
/// `detail`), or a token cap.
pub fn parse_detail(value: &str) -> Result<ImageBudget> {
    match value.trim().to_ascii_lowercase().as_str() {
        "auto" | "default" | "" => Ok(ImageBudget::default()),
        "low" => Ok(ImageBudget {
            min_tokens: None,
            max_tokens: Some(LOW_DETAIL_TOKENS),
        }),
        // The projector's cap; `VisionModel::limits` clamps to it.
        "high" => Ok(ImageBudget {
            min_tokens: None,
            max_tokens: Some(usize::MAX),
        }),
        n => match n.parse::<usize>() {
            Ok(n) if n > 0 => Ok(ImageBudget {
                min_tokens: None,
                max_tokens: Some(n),
            }),
            _ => Err(Error::Parameter(format!(
                "image detail must be low, high, auto or a token count, not {value:?}"
            ))),
        },
    }
}

/// The token limits an image part asks for: OpenAI's `detail`, and Ferrum's
/// `min_tokens` / `max_tokens`, on the part or inside its `image_url` /
/// `source` object (the inner one wins).
fn part_budget(part: &Json) -> Result<ImageBudget> {
    let mut budget = ImageBudget::default();
    let inner = ["image_url", "source", "image"]
        .iter()
        .filter_map(|k| part.get(*k).filter(|v| v.is_object()))
        .collect::<Vec<_>>();
    for obj in std::iter::once(part).chain(inner) {
        if let Some(detail) = obj.get("detail") {
            let detail = detail
                .as_str()
                .ok_or_else(|| Error::Parameter("image detail must be a string".into()))?;
            budget = parse_detail(detail)?.or(budget);
        }
        for (key, slot) in [
            ("min_tokens", &mut budget.min_tokens),
            ("max_tokens", &mut budget.max_tokens),
        ] {
            if let Some(v) = obj.get(key) {
                match v.as_u64() {
                    Some(n) if n > 0 => *slot = Some(n as usize),
                    _ => {
                        return Err(Error::Parameter(format!(
                            "image {key} must be a positive integer"
                        )));
                    }
                }
            }
        }
    }
    Ok(budget)
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
pub fn take_images(messages: &[Json]) -> Result<(Vec<Json>, Vec<ImageInput>)> {
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
                    images.push(ImageInput {
                        bytes: bytes?,
                        budget: part_budget(part)?,
                    });
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
        let bytes: Vec<&[u8]> = images.iter().map(|i| &i.bytes[..]).collect();
        assert_eq!(bytes, [b"foo", b"bar", b"baz"]);
        assert!(images.iter().all(|i| i.budget == ImageBudget::default()));
        assert_eq!(clean[1]["content"][0]["text"], "what is this?");
        assert_eq!(clean[1]["content"][1], json!({"type": "image"}));
        assert!(!has_images(&clean[..1]));
        // The input is untouched.
        assert!(messages[1]["content"][1].get("image_url").is_some());
    }

    #[test]
    fn reads_per_image_detail_and_token_limits() {
        let messages = [json!({"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,Zm9v", "detail": "low"}},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,Zm9v", "detail": "high"}},
            {"type": "image_url", "detail": "low",
             "image_url": {"url": "data:image/png;base64,Zm9v", "max_tokens": 100, "min_tokens": 50}},
            {"type": "image", "max_tokens": 300,
             "source": {"type": "base64", "media_type": "image/png", "data": "YmFy"}},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,Zm9v", "detail": "512"}},
        ]})];
        let (_, images) = take_images(&messages).unwrap();
        let caps: Vec<_> = images
            .iter()
            .map(|i| (i.budget.min_tokens, i.budget.max_tokens))
            .collect();
        assert_eq!(
            caps,
            [
                (None, Some(LOW_DETAIL_TOKENS)),
                (None, Some(usize::MAX)),
                (Some(50), Some(100)),
                (None, Some(300)),
                (None, Some(512)),
            ]
        );
        for bad in [
            json!({"detail": "huge"}),
            json!({"max_tokens": 0}),
            json!({"min_tokens": "x"}),
        ] {
            let mut url = json!({"url": "data:image/png;base64,Zm9v"});
            url.as_object_mut()
                .unwrap()
                .extend(bad.as_object().unwrap().clone());
            let m = [json!({"role": "user", "content": [{"type": "image_url", "image_url": url}]})];
            assert!(take_images(&m).is_err(), "{m:?}");
        }
        // What the CLI and TUI send round-trips.
        let budget = ImageBudget {
            min_tokens: Some(8),
            max_tokens: Some(64),
        };
        let m = [json!({"role": "user", "content": [image_part_with(b"foo", budget)]})];
        assert_eq!(take_images(&m).unwrap().1[0].budget, budget);
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
