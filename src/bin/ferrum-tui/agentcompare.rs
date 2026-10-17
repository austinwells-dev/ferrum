//! The agent benchmark's Compare view: one column per saved run (tasks
//! solved), best first, and under the columns a grid of which tasks each run
//! solved, so it shows where models differ.
use crate::agentbench::{self, AgentResult};
use crate::agentui::{duration, kilo};
use crate::benchui::ago;
use crate::evalcompare::{centered_text, column_cell, model_name};
use crate::evalui::score_color;
use crate::*;

#[derive(Clone, Default)]
pub struct AgentCompare {
    pub sel: usize,
    /// Only runs with thinking on (true) or off (false); None: both.
    pub thinking: Option<bool>,
    /// `x` was pressed once: the next `x` deletes every saved agent run.
    pub confirm_clear: bool,
}

impl AgentCompare {
    fn next_thinking(&mut self) {
        self.thinking = match self.thinking {
            None => Some(false),
            Some(false) => Some(true),
            Some(true) => None,
        };
    }

    fn thinking_label(&self) -> &'static str {
        match self.thinking {
            None => "thinking on and off",
            Some(false) => "thinking off",
            Some(true) => "thinking on",
        }
    }
}

/// One line of a column's labels.
type Label<'a> = Box<dyn Fn(&AgentResult) -> String + 'a>;

/// The runs Compare shows: most solved first, then fewest generated tokens.
pub fn compare_runs<'a>(results: &'a [AgentResult], c: &AgentCompare) -> Vec<&'a AgentResult> {
    agentbench::ranked(results)
        .into_iter()
        .filter(|r| c.thinking.is_none_or(|on| r.thinking.is_some() == on))
        .collect()
}

/// The grid cell for a task in a run: solved, not solved, or not run.
fn task_mark(r: &AgentResult, task: &str) -> (&'static str, Color) {
    match r.results.iter().find(|t| t.id == task) {
        Some(t) if t.solved => ("✓", GOOD),
        Some(_) => ("✗", BAD),
        None => ("·", FAINT),
    }
}

impl App {
    pub fn agent_compare_key(&mut self, key: KeyEvent) {
        let Some(b) = self.bench.as_mut() else { return };
        let n =
            b.ag.compare
                .as_ref()
                .map_or(0, |c| compare_runs(&b.ag.results, c).len());
        let Some(c) = b.ag.compare.as_mut() else {
            return;
        };
        let clearing = c.confirm_clear;
        c.confirm_clear = false;
        match key.code {
            KeyCode::Esc | KeyCode::Char('q') | KeyCode::Char('c') => b.ag.compare = None,
            KeyCode::Left | KeyCode::Up | KeyCode::Char('h') | KeyCode::Char('k') => {
                c.sel = c.sel.saturating_sub(1)
            }
            KeyCode::Right
            | KeyCode::Down
            | KeyCode::Char('l')
            | KeyCode::Char('j')
            | KeyCode::Tab => c.sel = (c.sel + 1).min(n.saturating_sub(1)),
            KeyCode::Char('t') => {
                c.next_thinking();
                c.sel = 0;
            }
            KeyCode::Char('x') if clearing => {
                b.ag.results.clear();
                agentbench::write_results(&agentbench::results_path(), &b.ag.results);
                c.sel = 0;
                self.status = Some(("cleared every agent result".into(), true));
            }
            KeyCode::Char('x') => c.confirm_clear = true,
            _ => {}
        }
    }

    pub fn draw_agent_compare(&self, f: &mut Frame, area: Rect, b: &BenchState) {
        let block = panel("Compare · agent", true);
        let inner = block.inner(area);
        f.render_widget(block, area);
        let width = inner.width as usize;
        let Some(c) = &b.ag.compare else { return };
        let runs = compare_runs(&b.ag.results, c);
        let mut lines = vec![
            Line::styled(
                format!(
                    " every model · {} · {} run{} · most solved first, then fewest generated tokens",
                    c.thinking_label(),
                    runs.len(),
                    if runs.len() == 1 { "" } else { "s" }
                ),
                Style::new().fg(DIM),
            ),
            Line::styled(
                clip(
                    " columns: tasks solved · tok/solve: generated tokens per solved task · ✓/✗ per task below",
                    width,
                ),
                Style::new().fg(FAINT),
            ),
        ];
        if c.confirm_clear {
            lines.push(Line::styled(
                format!(
                    " press x again to delete all {} saved agent runs · any other key keeps them",
                    b.ag.results.len()
                ),
                Style::new().fg(BAD).add_modifier(Modifier::BOLD),
            ));
        }
        lines.push(Line::default());
        if runs.is_empty() {
            lines.push(Line::styled(
                if b.ag.results.is_empty() {
                    "   no agent runs yet: run the agent tasks first"
                } else {
                    "   nothing matches this filter (t: thinking)"
                },
                Style::new().fg(GOLD),
            ));
            f.render_widget(Paragraph::new(lines), inner);
            return;
        }
        let tasks = agentbench::tasks();
        // Task names on the left; one column per run to their right.
        let gutter = (width / 3).clamp(20, 38);
        let room = width.saturating_sub(gutter + 1);
        let col_w = (room / runs.len()).clamp(10, 18);
        let visible = (room / col_w).max(1);
        let n = runs.len();
        let sel = c.sel.min(n - 1);
        let first = sel
            .saturating_sub(visible - 1)
            .min(n.saturating_sub(visible));
        let shown = &runs[first..(first + visible).min(n)];
        let fixed = lines.len() + 1 + 4 + 2 + tasks.len() + 3;
        let chart_h = (inner.height as usize).saturating_sub(fixed).clamp(4, 14);
        let full = (chart_h - 1) * 8;
        let pad = || Span::raw(" ".repeat(gutter));
        for row in 0..chart_h {
            let y = chart_h - 1 - row;
            let mut spans = vec![pad()];
            for (k, r) in shown.iter().enumerate() {
                let t = r.totals();
                let share = t.solved as f64 / t.done.max(1) as f64;
                let filled = (share * full as f64).round() as usize;
                let on = first + k == sel;
                if y == filled.div_ceil(8) {
                    spans.push(Span::styled(
                        centered_text(&format!("{}/{}", t.solved, t.done), col_w),
                        Style::new()
                            .fg(if on { Color::White } else { score_color(share) })
                            .add_modifier(Modifier::BOLD),
                    ));
                } else {
                    spans.push(Span::raw(" "));
                    spans.push(Span::styled(
                        column_cell(y, filled, filled).repeat(col_w - 3),
                        Style::new().fg(score_color(share)),
                    ));
                    spans.push(Span::raw("  "));
                }
            }
            lines.push(Line::from(spans));
        }
        lines.push(Line::styled(
            format!("{}{}", " ".repeat(gutter), "─".repeat(shown.len() * col_w)),
            Style::new().fg(FAINT),
        ));
        let speed = |r: &AgentResult| {
            b.saved
                .iter()
                .find(|x| x.machine == b.machine.label() && x.model == r.model)
                .map(|x| format!(" {:.2}×", x.speedup))
                .unwrap_or_default()
        };
        let labels: [(&str, Label); 4] = [
            ("model", Box::new(|r| model_name(&r.model))),
            (
                "setup",
                Box::new(|r| {
                    format!(
                        "{}{}",
                        r.thinking
                            .map_or("no think".to_string(), |t| format!("think {t}")),
                        if r.draft == "plain decoding" {
                            ""
                        } else {
                            " +d"
                        }
                    )
                }),
            ),
            (
                "tok/solve",
                Box::new(|r| r.totals().per_solve().map_or("–".to_string(), kilo)),
            ),
            (
                "time",
                Box::new(|r| format!("{}{}", duration(r.totals().secs), speed(r))),
            ),
        ];
        for (name, label) in &labels {
            let mut spans = vec![Span::styled(
                format!("{:>w$} ", name, w = gutter - 1),
                Style::new().fg(FAINT),
            )];
            for (k, r) in shown.iter().enumerate() {
                let on = first + k == sel;
                spans.push(Span::styled(
                    format!(" {:<w$}", clip(&label(r), col_w - 2), w = col_w - 1),
                    Style::new()
                        .fg(if on { Color::White } else { DIM })
                        .bg(if on { SELECTED } else { Color::Reset }),
                ));
            }
            lines.push(Line::from(spans));
        }
        lines.push(Line::default());
        lines.push(Line::styled(
            format!("{:<w$}", "  TASKS", w = gutter),
            Style::new().fg(DIM).add_modifier(Modifier::BOLD),
        ));
        for task in &tasks {
            let mut spans = vec![Span::styled(
                format!("  {:<w$}", clip(&task.id, gutter - 3), w = gutter - 2),
                Style::new().fg(TEXT),
            )];
            for (k, r) in shown.iter().enumerate() {
                let (mark, color) = task_mark(r, &task.id);
                spans.push(Span::styled(
                    centered_text(mark, col_w),
                    Style::new().fg(color).bg(if first + k == sel {
                        SELECTED
                    } else {
                        Color::Reset
                    }),
                ));
            }
            lines.push(Line::from(spans));
        }
        lines.push(Line::default());
        let r = runs[sel];
        let t = r.totals();
        lines.push(Line::from(vec![
            Span::styled(
                format!(" ▲ {}", model_name(&r.model)),
                Style::new().fg(Color::White).add_modifier(Modifier::BOLD),
            ),
            Span::styled(
                clip(
                    &format!(
                        "  {}/{} solved · {} generated · {} read · {} replies · {} · thinking {} · {} · {}{}",
                        t.solved,
                        t.done,
                        kilo(t.generated),
                        kilo(t.prompt),
                        t.turns,
                        duration(t.secs),
                        r.thinking.map_or("off".to_string(), |b| format!("≤{b}")),
                        r.draft,
                        ago(r.when),
                        if r.tasks != agentbench::tasks_fingerprint() {
                            " · older tasks"
                        } else {
                            ""
                        }
                    ),
                    width.saturating_sub(model_name(&r.model).chars().count() + 4),
                ),
                Style::new().fg(DIM),
            ),
        ]));
        if n > visible {
            lines.push(Line::styled(
                format!(
                    " runs {}-{} of {n} · ←→ scrolls",
                    first + 1,
                    first + shown.len()
                ),
                Style::new().fg(FAINT),
            ));
        }
        f.render_widget(Paragraph::new(lines), inner);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agentbench::TaskResult;

    fn run(model: &str, thinking: Option<usize>, solved: &[&str]) -> AgentResult {
        AgentResult {
            model: model.into(),
            when: 1,
            thinking,
            draft: "plain decoding".into(),
            kv_cache: "F16".into(),
            tasks: "t".into(),
            results: ["a", "b", "c"]
                .iter()
                .map(|id| TaskResult {
                    id: id.to_string(),
                    solved: solved.contains(id),
                    turns: 3,
                    generated: 1000,
                    prompt: 500,
                    secs: 5.0,
                    end: "finished".into(),
                    test: "passed".into(),
                    log: None,
                })
                .collect(),
        }
    }

    #[test]
    fn filters_runs_and_marks_tasks() {
        let results = vec![
            run("/m/one.gguf", None, &["a"]),
            run("/m/two.gguf", Some(2048), &["a", "b"]),
        ];
        let mut c = AgentCompare::default();
        let models: Vec<&str> = compare_runs(&results, &c)
            .iter()
            .map(|r| r.model.as_str())
            .collect();
        assert_eq!(
            models,
            vec!["/m/two.gguf", "/m/one.gguf"],
            "most solved first"
        );
        c.next_thinking();
        assert_eq!(compare_runs(&results, &c)[0].model, "/m/one.gguf");
        c.next_thinking();
        assert_eq!(compare_runs(&results, &c)[0].model, "/m/two.gguf");
        c.next_thinking();
        assert_eq!(compare_runs(&results, &c).len(), 2);
        assert_eq!(task_mark(&results[0], "a").0, "✓");
        assert_eq!(task_mark(&results[0], "b").0, "✗");
        assert_eq!(task_mark(&results[0], "z").0, "·");
    }
}
