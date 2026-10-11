//! SKILL.md files the agent can load on demand. Only each skill's name and
//! description go in the prompt; `skill(name)` returns the full text.
use std::fs;
use std::path::{Path, PathBuf};

const MAX_SKILLS: usize = 40;
const BODY_LIMIT: usize = 12_000;

#[derive(Clone, Debug, PartialEq)]
pub struct Skill {
    pub name: String,
    pub description: String,
    pub file: PathBuf,
    /// The skill's folder as the agent should write it: relative to the project,
    /// or absolute for a global skill.
    pub dir: String,
    /// From `~/.ferrum/skills` rather than the project.
    pub global: bool,
}

/// Name and description from the `---` frontmatter, falling back to the folder
/// name and the first line of text.
pub fn parse(text: &str, fallback: &str) -> (String, String) {
    let (mut name, mut desc) = (String::new(), String::new());
    let body = match text.strip_prefix("---") {
        Some(rest) => {
            let end = rest.find("\n---").unwrap_or(rest.len());
            let mut lines = rest[..end].lines().peekable();
            while let Some(line) = lines.next() {
                let Some((key, value)) = line.split_once(':') else {
                    continue;
                };
                let value = value.trim().trim_matches(|c| c == '"' || c == '\'');
                let folded = matches!(value, ">" | "|" | ">-" | "|-");
                let mut got = value.to_string();
                if folded {
                    got.clear();
                    while let Some(next) = lines.peek() {
                        if !next.starts_with(' ') && !next.starts_with('\t') {
                            break;
                        }
                        got += &format!("{} ", next.trim());
                        lines.next();
                    }
                }
                match key.trim() {
                    "name" => name = got.trim().to_string(),
                    "description" => desc = got.trim().to_string(),
                    _ => {}
                }
            }
            rest.get(end + 4..).unwrap_or("")
        }
        None => text,
    };
    if desc.is_empty() {
        desc = body
            .lines()
            .map(|l| l.trim().trim_start_matches('#').trim())
            .find(|l| !l.is_empty())
            .unwrap_or("")
            .to_string();
    }
    if name.is_empty() {
        name = fallback.to_string();
    }
    (name, desc)
}

fn read_skill(file: &Path, fallback: &str, project: &Path) -> Option<Skill> {
    let text = crate::agent::read_no_follow(file).ok()?;
    let (name, description) = parse(&text, fallback);
    let folder = file.parent()?;
    let (dir, global) = match folder.strip_prefix(project) {
        Ok(r) if r.as_os_str().is_empty() => (".".to_string(), false),
        Ok(r) => (r.display().to_string(), false),
        Err(_) => (folder.display().to_string(), true),
    };
    Some(Skill {
        name,
        description,
        file: file.to_path_buf(),
        dir,
        global,
    })
}

fn is_skill_file(name: &str) -> bool {
    name.eq_ignore_ascii_case("SKILL.md")
}

/// Skills in `<dir>/*/SKILL.md`: one folder level, no recursion. Symlinked
/// folders are skipped so a link cannot lead outside the project.
fn scan(dir: &Path, project: &Path, out: &mut Vec<Skill>) {
    let Ok(entries) = fs::read_dir(dir) else {
        return;
    };
    let mut entries: Vec<_> = entries.flatten().collect();
    entries.sort_by_key(|e| e.file_name());
    for e in entries {
        let folder = e.file_name().to_string_lossy().to_string();
        if !e.file_type().map(|t| t.is_dir()).unwrap_or(false) || folder.starts_with('.') {
            continue;
        }
        let Ok(inner) = fs::read_dir(e.path()) else {
            continue;
        };
        // file_type does not follow links: a SKILL.md symlinked to a secret
        // would otherwise be read here, outside the sandbox.
        let file = inner.flatten().find(|f| {
            is_skill_file(&f.file_name().to_string_lossy())
                && f.file_type().is_ok_and(|t| t.is_file())
        });
        if let Some(f) = file {
            out.extend(read_skill(&f.path(), &folder, project));
        }
    }
}

/// Skills in `<project>/*/`, `<project>/skills/*/` and `<project>/.ferrum/skills/*/`,
/// then `~/.ferrum/skills/*/`. A project skill wins over a global one of the same name.
pub fn discover(project: &Path) -> Vec<Skill> {
    let mut out = Vec::new();
    for dir in [
        project.join("skills"),
        project.join(".ferrum").join("skills"),
        project.to_path_buf(),
    ] {
        scan(&dir, project, &mut out);
    }
    let home = expand_home();
    let mut global = Vec::new();
    scan(&home.join(".ferrum").join("skills"), &home, &mut global);
    for mut s in global {
        s.global = true;
        s.dir = s
            .file
            .parent()
            .map(|d| d.display().to_string())
            .unwrap_or_default();
        out.push(s);
    }
    let mut seen = std::collections::HashSet::new();
    out.retain(|s| seen.insert(s.name.to_lowercase()));
    out.truncate(MAX_SKILLS);
    out
}

fn expand_home() -> PathBuf {
    std::env::var_os("HOME")
        .map(PathBuf::from)
        .unwrap_or_default()
}

/// `text` on one line, cut with an ellipsis.
pub fn one_line(text: &str, max: usize) -> String {
    let flat = text.split_whitespace().collect::<Vec<_>>().join(" ");
    if flat.chars().count() <= max {
        flat
    } else {
        format!(
            "{}…",
            flat.chars().take(max.saturating_sub(1)).collect::<String>()
        )
    }
}

/// The prompt section listing what is available.
pub fn prompt(skills: &[Skill]) -> String {
    if skills.is_empty() {
        return String::new();
    }
    let mut s = String::from(
        "SKILLS\nInstructions for specific kinds of work are available. When a task matches one, call `skill` with its name and follow it. Do not load a skill that does not match.\n",
    );
    for k in skills {
        let desc = one_line(&k.description, 160);
        let loc = if k.global { "~/.ferrum/skills" } else { &k.dir };
        s += &format!("- {} ({loc}): {desc}\n", k.name);
    }
    s
}

/// The full skill for the model: its text, and the files beside it.
pub fn load(skills: &[Skill], name: &str) -> Result<String, String> {
    let wanted = name.trim().trim_start_matches('/').to_lowercase();
    let skill = skills
        .iter()
        .find(|s| s.name.to_lowercase() == wanted)
        .ok_or_else(|| {
            format!(
                "no skill named {name:?}; available: {}",
                skills
                    .iter()
                    .map(|s| s.name.as_str())
                    .collect::<Vec<_>>()
                    .join(", ")
            )
        })?;
    let text = crate::agent::read_no_follow(&skill.file).map_err(|e| e.to_string())?;
    let dir = skill.file.parent().unwrap_or(Path::new("."));
    let mut files = Vec::new();
    collect(dir, dir, 0, &mut files);
    let mut out = format!("Skill {} ({}/SKILL.md):\n\n", skill.name, skill.dir);
    out += &crate::agent::clip_output(text.trim(), BODY_LIMIT);
    if !files.is_empty() {
        out += &format!(
            "\n\nFiles next to it; paths here are from the project folder{}:\n",
            if skill.global {
                " (this skill lives outside it: use `cat` in bash to read these)"
            } else {
                ""
            }
        );
        for f in files.iter().take(30) {
            out += &format!("- {}/{f}\n", skill.dir);
        }
    }
    Ok(out)
}

fn collect(root: &Path, dir: &Path, depth: usize, out: &mut Vec<String>) {
    let Ok(entries) = fs::read_dir(dir) else {
        return;
    };
    let mut entries: Vec<_> = entries.flatten().collect();
    entries.sort_by_key(|e| e.file_name());
    for e in entries {
        let path = e.path();
        let name = e.file_name().to_string_lossy().to_string();
        if name.starts_with('.') || is_skill_file(&name) {
            continue;
        }
        let Ok(kind) = e.file_type() else { continue };
        if kind.is_dir() && depth < 2 {
            collect(root, &path, depth + 1, out);
        } else if kind.is_file() {
            out.push(
                path.strip_prefix(root)
                    .unwrap_or(&path)
                    .display()
                    .to_string(),
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reads_frontmatter_and_falls_back() {
        let t = "---\nname: deploy\ndescription: Ship the app to staging\n---\n# Deploy\nsteps";
        assert_eq!(
            parse(t, "x"),
            ("deploy".into(), "Ship the app to staging".into())
        );
        let folded = "---\nname: a\ndescription: >\n  first part\n  second part\n---\nbody";
        assert_eq!(parse(folded, "x").1, "first part second part");
        assert_eq!(
            parse("# Release notes\n\nWrite them.", "rel"),
            ("rel".into(), "Release notes".into())
        );
    }

    #[test]
    fn finds_project_skills_one_level_down_and_loads_them() {
        let dir = std::env::temp_dir().join(format!("ferrum-skills-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(dir.join("skills/lint/scripts")).unwrap();
        fs::create_dir_all(dir.join("tools/other")).unwrap();
        fs::create_dir_all(dir.join("deep/er/lint")).unwrap();
        fs::write(dir.join("SKILL.md"), "not scanned").unwrap();
        fs::write(
            dir.join("skills/lint/SKILL.md"),
            "---\nname: lint\ndescription: Run the linter\n---\nRun scripts/run.sh",
        )
        .unwrap();
        fs::write(dir.join("skills/lint/scripts/run.sh"), "echo ok").unwrap();
        fs::write(dir.join("tools/other/SKILL.md"), "x").unwrap();
        fs::write(dir.join("deep/er/lint/SKILL.md"), "too deep").unwrap();
        fs::create_dir_all(dir.join("mine")).unwrap();
        fs::write(dir.join("mine/SKILL.md"), "My own skill").unwrap();
        let mut found = Vec::new();
        for d in [dir.join("skills"), dir.clone()] {
            scan(&d, &dir, &mut found);
        }
        let names: Vec<_> = found.iter().map(|s| s.name.as_str()).collect();
        assert_eq!(names, ["lint", "mine"], "{names:?}");
        let text = load(&found, "LINT").unwrap();
        assert!(
            text.contains("Run scripts/run.sh") && text.contains("- skills/lint/scripts/run.sh\n")
        );
        assert!(load(&found, "nope").unwrap_err().contains("lint"));
        assert!(prompt(&found).contains("lint (skills/lint): Run the linter"));
        // A SKILL.md that is a symlink (say, to a key) is never read.
        let secret = dir.with_extension("secret");
        fs::write(&secret, "TOP SECRET").unwrap();
        fs::create_dir_all(dir.join("skills/leak")).unwrap();
        std::os::unix::fs::symlink(&secret, dir.join("skills/leak/SKILL.md")).unwrap();
        let mut again = Vec::new();
        scan(&dir.join("skills"), &dir, &mut again);
        assert!(again.iter().all(|s| s.name != "leak"), "{again:?}");
        fs::remove_file(dir.join("skills/lint/SKILL.md")).unwrap();
        std::os::unix::fs::symlink(&secret, dir.join("skills/lint/SKILL.md")).unwrap();
        assert!(load(&found, "lint").is_err());
        let _ = fs::remove_file(secret);
        let _ = fs::remove_dir_all(&dir);
    }
}
