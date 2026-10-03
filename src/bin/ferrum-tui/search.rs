//! The `web_search` tool: one query, several back ends. The endpoint comes from
//! the user's settings, never from the model, so the request runs outside the
//! sandbox. That lets a SearXNG server on this Mac or LAN work.
use serde_json::Value as Json;
use std::process::Command;

pub const PROVIDERS: &[&str] = &["duckduckgo", "instant", "searxng", "brave", "custom"];

#[derive(Clone, Default)]
pub struct Config {
    pub provider: String,
    /// SearXNG base address, or the custom URL template (with `{query}`).
    pub url: String,
    /// API key, or `$ENV_VAR` naming where to read it.
    pub key: String,
}

impl Config {
    pub fn new(provider: &str, url: &str, key: &str) -> Self {
        Self {
            provider: provider.into(),
            url: url.trim().into(),
            key: key.trim().into(),
        }
    }

    fn key(&self) -> Result<String, String> {
        match self.key.strip_prefix('$') {
            Some(var) => std::env::var(var).map_err(|_| format!("${var} is not set")),
            None if self.key.is_empty() => Err("set a Search API key (or $ENV_VAR)".into()),
            None => Ok(self.key.clone()),
        }
    }
}

/// Run a search; the text is what the model sees.
pub fn search(cfg: &Config, query: &str) -> Result<String, String> {
    let q = crate::coding::url_encode(query);
    let (url, headers): (String, Vec<String>) = match cfg.provider.as_str() {
        "" | "duckduckgo" => (
            "https://html.duckduckgo.com/html/?q=".to_string() + &q,
            vec![],
        ),
        "instant" => (
            format!("https://api.duckduckgo.com/?q={q}&format=json&no_html=1&skip_disambig=1"),
            vec![],
        ),
        "searxng" => {
            if cfg.url.is_empty() {
                return Err("set Search URL to your SearXNG address".into());
            }
            let base = cfg.url.trim_end_matches('/');
            let base = if base.ends_with("/search") {
                base.to_string()
            } else {
                format!("{base}/search")
            };
            (format!("{base}?q={q}&format=json"), vec![])
        }
        "brave" => (
            format!("https://api.search.brave.com/res/v1/web/search?q={q}&count=8"),
            vec![format!("X-Subscription-Token: {}", cfg.key()?)],
        ),
        "custom" => {
            if !cfg.url.contains("{query}") {
                return Err("Search URL needs a {query} placeholder".into());
            }
            let mut url = cfg.url.replace("{query}", &q);
            let mut headers = vec![];
            if !cfg.key.is_empty() {
                let key = cfg.key()?;
                if url.contains("{key}") {
                    url = url.replace("{key}", &key);
                } else {
                    headers.push(format!("Authorization: Bearer {key}"));
                }
            }
            (url, headers)
        }
        other => return Err(format!("unknown search provider {other}")),
    };
    if !(url.starts_with("http://") || url.starts_with("https://")) {
        return Err("the search address must start with http:// or https://".into());
    }
    let mut cmd = Command::new("/usr/bin/curl");
    cmd.args([
        "-sSL",
        "--max-time",
        "20",
        "-A",
        "Mozilla/5.0 (Macintosh) ferrum",
    ]);
    cmd.args(["-H", "Accept: application/json, text/html;q=0.8"]);
    for h in &headers {
        cmd.args(["-H", h]);
    }
    cmd.args(["-w", "\n%{http_code}", "--", &url]);
    let out = cmd
        .output()
        .map_err(|e| format!("could not run curl: {e}"))?;
    if !out.status.success() {
        return Err(format!(
            "search failed: {}",
            String::from_utf8_lossy(&out.stderr).trim()
        ));
    }
    let body = String::from_utf8_lossy(&out.stdout).to_string();
    let (body, status) = body.rsplit_once('\n').unwrap_or((&body, ""));
    if status != "200" {
        let hint = if cfg.provider == "searxng" && status == "403" {
            " (enable the json format under search.formats in the SearXNG settings.yml)"
        } else {
            ""
        };
        return Err(format!("the search server answered {status}{hint}"));
    }
    let results = match cfg.provider.as_str() {
        "" | "duckduckgo" => crate::coding::parse_search(body),
        _ => parse_json(&cfg.provider, body)?,
    };
    if results.is_empty() {
        return Ok("no results".into());
    }
    Ok(results
        .iter()
        .take(8)
        .enumerate()
        .map(|(i, (t, u, s))| format!("{}. {t}\n   {u}\n   {s}", i + 1))
        .collect::<Vec<_>>()
        .join("\n"))
}

fn text(v: &Json, keys: &[&str]) -> String {
    keys.iter()
        .find_map(|k| v[*k].as_str().filter(|s| !s.is_empty()))
        .map(|s| crate::agent::html_to_text(s).trim().to_string())
        .unwrap_or_default()
}

/// (title, url, snippet) from a JSON answer.
pub fn parse_json(provider: &str, body: &str) -> Result<Vec<(String, String, String)>, String> {
    let json: Json = serde_json::from_str(body).map_err(|_| {
        format!(
            "the search server did not answer with JSON: {}",
            crate::agent::clip_output(body.trim(), 120)
        )
    })?;
    let mut out = Vec::new();
    if provider == "instant" {
        let (head, link, abs) = (
            text(&json, &["Heading"]),
            text(&json, &["AbstractURL"]),
            text(&json, &["AbstractText"]),
        );
        if !abs.is_empty() {
            out.push((head, link, abs));
        }
        let mut topics: Vec<&Json> = json["RelatedTopics"]
            .as_array()
            .into_iter()
            .flatten()
            .collect();
        while let Some(t) = topics.pop() {
            if let Some(sub) = t["Topics"].as_array() {
                topics.extend(sub);
            } else if let (s, u) = (text(t, &["Text"]), text(t, &["FirstURL"]))
                && !s.is_empty()
            {
                let title = s.split(" - ").next().unwrap_or(&s).to_string();
                out.push((title, u, s));
            }
        }
        return Ok(out);
    }
    // searxng: results[]; brave: web.results[]; others: first array of objects
    let list = [
        &json["results"],
        &json["web"]["results"],
        &json["items"],
        &json["data"],
    ]
    .into_iter()
    .find_map(Json::as_array)
    .ok_or("no results list in the search answer")?;
    for r in list {
        let (t, u) = (
            text(r, &["title", "name"]),
            text(r, &["url", "link", "href"]),
        );
        if !t.is_empty() && !u.is_empty() {
            out.push((
                t,
                u,
                text(r, &["content", "snippet", "description", "body"]),
            ));
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reads_searxng_and_brave_shapes() {
        let s = r#"{"results":[{"title":"A","url":"https://a.test","content":"about <b>a</b>"}]}"#;
        let r = parse_json("searxng", s).unwrap();
        assert_eq!(
            r[0],
            ("A".into(), "https://a.test".into(), "about a".into())
        );
        let b = r#"{"web":{"results":[{"title":"B","url":"https://b.test","description":"d"}]}}"#;
        assert_eq!(parse_json("brave", b).unwrap()[0].1, "https://b.test");
        assert!(parse_json("custom", "<html>").is_err());
    }

    #[test]
    fn reads_instant_answers_with_nested_topics() {
        let s = r#"{"Heading":"Rust","AbstractText":"A language","AbstractURL":"https://r.test",
          "RelatedTopics":[{"Text":"Cargo - build tool","FirstURL":"https://c.test"},
          {"Name":"g","Topics":[{"Text":"Crate - package","FirstURL":"https://k.test"}]}]}"#;
        let r = parse_json("instant", s).unwrap();
        assert_eq!(r.len(), 3);
        assert!(r.iter().any(|x| x.1 == "https://k.test"));
    }

    #[test]
    fn needs_settings() {
        assert!(search(&Config::new("searxng", "", ""), "x").is_err());
        assert!(search(&Config::new("custom", "https://x.test/?q=", ""), "x").is_err());
        assert!(search(&Config::new("brave", "", ""), "x").is_err());
    }

    #[test]
    fn talks_to_a_local_searxng() {
        use std::io::{Read, Write};
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        let server = std::thread::spawn(move || {
            let (mut c, _) = listener.accept().unwrap();
            let mut buf = [0u8; 2048];
            let n = c.read(&mut buf).unwrap();
            let req = String::from_utf8_lossy(&buf[..n]).to_string();
            let body = r#"{"results":[{"title":"Hit","url":"https://h.test","content":"found"}]}"#;
            let _ = write!(
                c,
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                body.len()
            );
            req
        });
        let cfg = Config::new("searxng", &format!("http://127.0.0.1:{port}/"), "");
        let out = search(&cfg, "rust lang").unwrap();
        assert!(
            out.contains("Hit") && out.contains("https://h.test"),
            "{out}"
        );
        let req = server.join().unwrap();
        assert!(
            req.starts_with("GET /search?q=rust+lang&format=json"),
            "{req}"
        );
    }
}
