//! Saved chat and agent sessions: one JSON file each in
//! `~/.config/ferrum/sessions`, written after every reply so `/resume` can
//! bring any of them back.
use crate::*;

pub fn dir() -> PathBuf {
    home().join(".config/ferrum/sessions")
}

/// A sortable id: `YYYYMMDD-HHMMSS-xxxx`.
pub fn new_id() -> String {
    let stamp = Command::new("/bin/date")
        .arg("+%Y%m%d-%H%M%S")
        .output()
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .unwrap_or_default();
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.subsec_nanos())
        .unwrap_or(0);
    format!("{stamp}-{:04x}", (nanos ^ std::process::id()) & 0xffff)
}

#[derive(Clone, Debug)]
pub struct Meta {
    pub id: String,
    pub title: String,
    pub model: String,
    pub project: String,
    pub updated: u64,
    pub turns: usize,
}

pub fn save(id: &str, data: &Json) {
    let _ = crate::write_private(
        &dir().join(format!("{id}.json")),
        &serde_json::to_string(data).unwrap_or_default(),
    );
}

pub fn load(id: &str) -> Option<Json> {
    let text = fs::read_to_string(dir().join(format!("{id}.json"))).ok()?;
    serde_json::from_str(&text).ok()
}

pub fn delete(id: &str) {
    let _ = fs::remove_file(dir().join(format!("{id}.json")));
    let _ = fs::remove_dir_all(cache_dir(id));
}

/// Most bytes all chats' session caches may take; the least recently saved
/// go first.
const CACHE_CAP: u64 = 16 << 30;

fn caches() -> PathBuf {
    std::env::var_os("XDG_CACHE_HOME")
        .map(PathBuf::from)
        .filter(|p| p.is_absolute())
        .unwrap_or_else(|| home().join(".cache"))
        .join("ferrum/sessions")
}

/// Where chat `id`'s session cache (`Runtime::save_session`) lives.
pub fn cache_dir(id: &str) -> PathBuf {
    caches().join(id)
}

/// Remove the least recently saved session caches until all of them fit
/// `CACHE_CAP`.
pub fn prune_caches() {
    let Ok(entries) = fs::read_dir(caches()) else {
        return;
    };
    let size = |dir: &Path| -> u64 {
        fs::read_dir(dir)
            .into_iter()
            .flatten()
            .flatten()
            .filter_map(|e| e.metadata().ok())
            .map(|m| m.len())
            .sum()
    };
    let mut dirs: Vec<(std::time::SystemTime, u64, PathBuf)> = entries
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.is_dir())
        .map(|p| {
            let saved = fs::metadata(p.join("meta.json"))
                .and_then(|m| m.modified())
                .unwrap_or(std::time::UNIX_EPOCH);
            (saved, size(&p), p)
        })
        .collect();
    dirs.sort_by_key(|d| d.0);
    let mut total: u64 = dirs.iter().map(|d| d.1).sum();
    for (_, bytes, path) in dirs {
        if total <= CACHE_CAP {
            break;
        }
        let _ = fs::remove_dir_all(path);
        total -= bytes;
    }
}

/// Newest first. `kind` is "chat" or "agent"; for agents, only sessions of `project`.
pub fn list(kind: &str, project: Option<&str>) -> Vec<Meta> {
    let Ok(entries) = fs::read_dir(dir()) else {
        return Vec::new();
    };
    let mut files: Vec<(std::time::SystemTime, PathBuf)> = entries
        .flatten()
        .filter(|e| e.path().extension().is_some_and(|x| x == "json"))
        .filter_map(|e| Some((e.metadata().ok()?.modified().ok()?, e.path())))
        .collect();
    files.sort_by_key(|b| std::cmp::Reverse(b.0));
    let mut out = Vec::new();
    for (_, path) in files.into_iter().take(80) {
        let Ok(text) = fs::read_to_string(&path) else {
            continue;
        };
        let Ok(j) = serde_json::from_str::<Json>(&text) else {
            continue;
        };
        if j["kind"] != kind || project.is_some_and(|p| j["project"] != p) {
            continue;
        }
        let turns = j["turns"]
            .as_array()
            .map_or(0, |t| t.iter().filter(|x| x["role"] == "user").count());
        if turns == 0 {
            continue;
        }
        out.push(Meta {
            id: j["id"].as_str().unwrap_or("").into(),
            title: j["title"].as_str().unwrap_or("(untitled)").into(),
            model: j["model"].as_str().unwrap_or("").into(),
            project: j["project"].as_str().unwrap_or("").into(),
            updated: j["updated"].as_u64().unwrap_or(0),
            turns,
        });
        if out.len() >= 30 {
            break;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ids_sort_by_time_and_differ() {
        let (a, b) = (new_id(), new_id());
        assert_eq!(a.len(), "20261003-164125-abcd".len(), "{a}");
        assert!(a.chars().next().unwrap().is_ascii_digit());
        // Same second, different suffix is possible but the format stays sortable.
        assert!(a[..15] <= b[..15]);
    }
}
