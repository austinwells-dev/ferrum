//! SKILL.md files the agent can load on demand. Only each skill's name and
//! description go in the prompt; `skill(name)` returns the full text.
use std::fs;
use std::path::{Path, PathBuf};

const MAX_SKILLS: usize = 40;
const MAX_DEPTH: usize = 4;
const BODY_LIMIT: usize = 12_000;
const SKIP: &[&str] = &["node_modules", "target", "vendor", "dist", "build", "venv"];

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
    let text = fs::read_to_string(file).ok()?;
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

fn walk(dir: &Path, depth: usize, project: &Path, out: &mut Vec<Skill>) {
    let Ok(entries) = fs::read_dir(dir) else {
        return;
    };
    let mut entries: Vec<_> = entries.flatten().collect();
    entries.sort_by_key(|e| e.file_name());
    for e in &entries {
        let name = e.file_name().to_string_lossy().to_string();
        if e.path().is_file() && is_skill_file(&name) && out.len() < MAX_SKILLS {
            let folder = dir
                .file_name()
                .map(|n| n.to_string_lossy().to_string())
                .unwrap_or_default();
            let folder = if dir == project {
                project
                    .file_name()
                    .map(|n| n.to_string_lossy().to_string())
                    .unwrap_or(folder)
            } else {
                folder
            };
            out.extend(read_skill(&e.path(), &folder, project));
        }
    }
    if depth >= MAX_DEPTH {
        return;
    }
    for e in entries {
        let name = e.file_name().to_string_lossy().to_string();
        let hidden = name.starts_with('.') && name != ".ferrum";
        // symlinks are not followed: a link must not lead outside the project
        let real = e.file_type().map(|t| t.is_dir()).unwrap_or(false);
        if real && !hidden && !SKIP.contains(&name.as_str()) {
            walk(&e.path(), depth + 1, project, out);
        }
    }
}

/// Skills found anywhere in the project (a few folders deep), then the ones in
/// `~/.ferrum/skills`. A project skill wins over a global one of the same name.
pub fn discover(project: &Path) -> Vec<Skill> {
    let mut out = Vec::new();
    walk(project, 0, project, &mut out);
    let home = expand_home();
    let mut global = Vec::new();
    walk(&home.join(".ferrum").join("skills"), 1, &home, &mut global);
    for mut s in global {
        s.global = true;
        s.dir = s
            .file
            .parent()
            .map(|d| d.display().to_string())
            .unwrap_or_default();
        if !out.iter().any(|p| p.name == s.name) && out.len() < MAX_SKILLS {
            out.push(s);
        }
    }
    out
}

fn expand_home() -> PathBuf {
    std::env::var_os("HOME")
        .map(PathBuf::from)
        .unwrap_or_default()
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
        let desc = crate::agent::clip_output(&k.description, 160);
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
    let text = fs::read_to_string(&skill.file).map_err(|e| e.to_string())?;
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
    fn finds_project_skills_and_loads_them() {
        let dir = std::env::temp_dir().join(format!("ferrum-skills-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(dir.join("tools/lint/scripts")).unwrap();
        fs::create_dir_all(dir.join("node_modules/pkg")).unwrap();
        fs::write(
            dir.join("SKILL.md"),
            "---\ndescription: Root skill\n---\nhi",
        )
        .unwrap();
        fs::write(
            dir.join("tools/lint/SKILL.md"),
            "---\nname: lint\ndescription: Run the linter\n---\nRun scripts/run.sh",
        )
        .unwrap();
        fs::write(dir.join("tools/lint/scripts/run.sh"), "echo ok").unwrap();
        fs::write(dir.join("node_modules/pkg/SKILL.md"), "ignored").unwrap();
        let mut found = Vec::new();
        walk(&dir, 0, &dir, &mut found);
        let names: Vec<_> = found.iter().map(|s| s.name.as_str()).collect();
        assert_eq!(found.len(), 2, "{names:?}");
        assert!(names.contains(&"lint"));
        let text = load(&found, "LINT").unwrap();
        assert!(
            text.contains("Run scripts/run.sh") && text.contains("- tools/lint/scripts/run.sh\n")
        );
        assert!(load(&found, "nope").unwrap_err().contains("lint"));
        assert!(prompt(&found).contains("lint (tools/lint): Run the linter"));
        let _ = fs::remove_dir_all(&dir);
    }
}
