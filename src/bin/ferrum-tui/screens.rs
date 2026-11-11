//! Every screen drawn at a narrow, a middling and a wide size with made-up
//! results: nothing may panic, and screens built of panes must reach the
//! bottom of the window (no blank rows under them). Set
//! FERRUM_SHOW_SCREENS=1 to print them.
use crate::agentbench::TaskResult;
use crate::agentui::Feed;
use crate::bench::{ChunkRow, Match, Row};
use crate::benchui::BenchPhase;
use crate::eval::Tally;
use crate::evalui::Measure;
use crate::*;
use ratatui::{Terminal, backend::TestBackend};

const SIZES: [(u16, u16); 4] = [(80, 30), (130, 40), (180, 50), (240, 60)];

/// Draw `app` at `w`×`h`; with `fills`, the row above the key bar must hold
/// the bottom edge of a pane.
fn render(app: &App, w: u16, h: u16, name: &str, fills: bool) {
    let mut term = Terminal::new(TestBackend::new(w, h)).unwrap();
    term.draw(|f| app.draw(f)).unwrap();
    let buf = term.backend().buffer();
    let row = |y: u16| (0..w).map(|x| buf[(x, y)].symbol()).collect::<String>();
    if std::env::var_os("FERRUM_SHOW_SCREENS").is_some() {
        println!("==== {name} {w}x{h}");
        for y in 0..h {
            println!("{}", row(y).trim_end());
        }
    }
    if fills {
        let last = row(h - 2);
        assert!(
            last.contains('╰'),
            "{name} at {w}x{h} leaves the bottom empty: {last:?}"
        );
    }
}

fn sweep(app: &App, name: &str, fills: bool) {
    for (w, h) in SIZES {
        render(app, w, h, name, fills);
    }
}

fn row(cand: usize, label: &str, depth: Option<usize>, tps: f64, same: Option<Match>) -> Row {
    Row {
        cand,
        label: label.into(),
        draft_value: String::new(),
        quant: if depth.is_some() {
            "q4_0".into()
        } else {
            String::new()
        },
        depth,
        tree: None,
        tps,
        accept: depth.map(|_| 2.4),
        per_prompt: vec![tps; 3],
        same,
        check: None,
        tokens: 576,
    }
}

#[test]
fn every_screen_draws_at_every_size() {
    let mut app = App::new(None);
    app.screen = Screen::Home;
    sweep(&app, "home", false);
    app.screen = Screen::Settings;
    for sel in [0, 4, 9] {
        app.set_sel = sel;
        sweep(&app, "settings", true);
    }
    app.go_tab(Tab::Models);
    sweep(&app, "models", true);
    app.go_tab(Tab::Image);
    for state in 0..4 {
        app.image.demo(state);
        sweep(&app, "image", true);
    }
    app.image.demo(0);
    app.buffer = app.image.prompt.repeat(6);
    app.editing = Editing::ImagePrompt;
    sweep(&app, "image prompt", false);
    app.editing = Editing::No;
    app.go_tab(Tab::Audio);
    for state in 0..4 {
        app.audio.demo(state);
        sweep(&app, "audio", true);
    }
    app.buffer = "~/Music/Ferrum/ferrum-1761850000.wav".into();
    app.editing = Editing::AudioClip;
    sweep(&app, "audio clip", false);
    app.editing = Editing::No;
    for mode in [Mode::Chat, Mode::Agent, Mode::Serve] {
        app.open_pick(mode);
        sweep(&app, "setups", true);
        app.open_editor(mode, Origin::OneTime);
        sweep(&app, "editor", true);
    }
    app.serve = Some(Serve::fake(
        (0..60)
            .map(|i| format!("12:00:{:02} INF [{i}] done (stop) in 9.10 s: prompt 412 tok (380 cached), prefill 32 tok in 0.05 s (640 tok/s); generated {} tok in 9.02 s ({:.1} tok/s)", i % 60, 300 + i * 7, 38.0 + (i % 9) as f64))
            .collect(),
    ));
    app.screen = Screen::Serve;
    sweep(&app, "serve", true);
    app.serve = None;

    app.open_bench();
    for (m, measure) in [Measure::Speed, Measure::Intelligence, Measure::Agent]
        .into_iter()
        .enumerate()
    {
        for (p, phase) in [BenchPhase::Setup, BenchPhase::Running, BenchPhase::Results]
            .into_iter()
            .enumerate()
        {
            let b = app.bench.as_mut().unwrap();
            b.measure = measure;
            b.phase = phase;
            b.rows = vec![
                row(0, "no drafter", None, 21.4, None),
                row(1, "MTP head", Some(3), 30.2, Some(Match::Exact)),
                row(1, "MTP head", Some(5), 33.9, Some(Match::Close)),
                row(
                    2,
                    "z-lab/Qwen3.8-27B-DFlash2 (dflash)",
                    Some(7),
                    49.8,
                    Some(Match::Differs),
                ),
            ];
            b.chunk_rows = vec![
                ChunkRow {
                    chunk: 256,
                    prefill_tps: 410.0,
                },
                ChunkRow {
                    chunk: 512,
                    prefill_tps: 520.0,
                },
            ];
            b.skipped = vec![(
                "RedHatAI/speculator (dspark)".into(),
                "hidden size 5120 ≠ model's 2048".into(),
            )];
            b.expected = 8;
            b.iq.tallies = (0..eval::CATEGORIES.len())
                .map(|i| Tally {
                    right: i * 3 % 10,
                    n: 10,
                    total: 100,
                })
                .collect();
            b.iq.answers = (0..40)
                .map(|i| {
                    (
                        i % eval::CATEGORIES.len(),
                        format!("q-{i}"),
                        i % 3 != 0,
                        Some(format!("{i}")),
                    )
                })
                .collect();
            b.iq.expected = 105;
            b.ag.done =
                b.ag.tasks
                    .iter()
                    .take(4)
                    .enumerate()
                    .map(|(i, t)| TaskResult {
                        id: t.id.clone(),
                        solved: i % 2 == 0,
                        turns: 12,
                        generated: 8000,
                        prompt: 40_000,
                        secs: 300.0,
                        end: "finished".into(),
                        test: "passed".into(),
                        log: None,
                    })
                    .collect();
            b.ag.feed = vec![
                Feed::Thinking("The test says setext headings break when a list follows the underline. I should look at how the block processor matches the dashes.".into()),
                Feed::Tool {
                    summary: "grep setext markdown/blockprocessors.py".into(),
                    ok: true,
                    output: "455: class SetextHeaderProcessor(BlockProcessor):\n463:     RE = re.compile(r'^.*?\\n[=-]+[ ]*(\\n|$)', re.M)".into(),
                },
                Feed::Reply(1, 812),
                Feed::Text("The underline pattern also matches a list item. I will require the line to be only dashes.".into()),
                Feed::Tool {
                    summary: "bash python -m pytest tests/test_syntax/blocks -q".into(),
                    ok: false,
                    output: "F.............\n1 failed, 14 passed".into(),
                },
            ];
            b.iq.feed = vec![
                Feed::Question {
                    title: "Knowledge · mmlu-pro-4411 · question 41 of 105".into(),
                    text: "Which of the following best describes the role of the Golgi apparatus?\n(A) protein synthesis\n(B) modifying, sorting and packaging proteins\n(C) ATP production\n(D) lipid breakdown".into(),
                },
                Feed::Thinking("Golgi modifies and packages proteins from the ER; synthesis is ribosomes, ATP is mitochondria.".into()),
                Feed::Text("The answer is (B).".into()),
                Feed::Verdict { correct: true, answer: "B".into() },
            ];
            b.ag.current = Some((
                4,
                6,
                563,
                5500,
                "read_file markdown/inlinepatterns.py".into(),
            ));
            b.stage = "z-lab/Qwen3.8-27B-DFlash2 (dflash) · depth 5 · prompt 2 of 3".into();
            b.iq.stage = "question 41 of 105 · knowledge".into();
            let name = format!(
                "bench {} {}",
                ["speed", "iq", "agent"][m],
                ["setup", "running", "results"][p]
            );
            for pane in 0..4 {
                app.bench_pane = pane;
                sweep(&app, &name, true);
            }
        }
    }
}

/// The whole screen as text, rows joined by newlines.
fn screen_text(app: &App, w: u16, h: u16) -> String {
    let mut term = Terminal::new(TestBackend::new(w, h)).unwrap();
    term.draw(|f| app.draw(f)).unwrap();
    let buf = term.backend().buffer();
    (0..h)
        .map(|y| (0..w).map(|x| buf[(x, y)].symbol()).collect::<String>())
        .collect::<Vec<_>>()
        .join("\n")
}

#[test]
fn a_running_speed_bench_lists_only_ticked_candidates() {
    use crate::bench::{Candidate, Source};
    let mut app = App::new(None);
    app.open_bench();
    let drafter = |label: &str| Source::Drafter {
        path: format!("/m/{label}").into(),
        label: label.into(),
        kind: "dflash2",
        max_depth: 7,
    };
    // A dense model: drafters try trees on it.
    app.models.push(crate::catalog::Model {
        path: "/m/dense.gguf".into(),
        name: "dense".into(),
        origin: String::new(),
        size: 1,
        info: Some(crate::catalog::Info {
            arch: "qwen35".into(),
            name: None,
            layers: None,
            experts: None,
            train_ctx: None,
            sampling: String::new(),
        }),
        hybrid: None,
        mtp: false,
        mtp_why: String::new(),
        vision: None,
        audio: false,
        problem: None,
        tokenizer: Default::default(),
    });
    let dense = app.models.len() - 1;
    let b = app.bench.as_mut().unwrap();
    b.model = dense;
    b.measure = Measure::Speed;
    b.cands = vec![
        Candidate {
            source: Source::Baseline,
            quant: "q4_0",
            on: true,
        },
        Candidate {
            source: drafter("Skipped-Drafter"),
            quant: "q4_0",
            on: false,
        },
        Candidate {
            source: drafter("Ticked-Drafter"),
            quant: "q4_0",
            on: true,
        },
    ];
    // The run measured the ticked drafter; its rows name its queue index.
    b.rows = vec![
        row(0, "no drafter", None, 8.0, None),
        row(2, "Ticked-Drafter", Some(2), 19.0, Some(Match::Exact)),
    ];
    for phase in [BenchPhase::Setup, BenchPhase::Running, BenchPhase::Results] {
        app.bench.as_mut().unwrap().phase = phase;
        let text = screen_text(&app, 240, 60);
        assert!(text.contains("Ticked-Drafter"), "{phase:?}");
        assert_eq!(
            text.contains("Skipped-Drafter"),
            phase == BenchPhase::Setup,
            "{phase:?}: an unticked drafter shows only in the setup list"
        );
    }
    // The setup's plan lists the trees each drafter tries, and counts them:
    // the baseline, 3 depths and 2 trees of the one ticked drafter.
    app.bench.as_mut().unwrap().phase = BenchPhase::Setup;
    let text = screen_text(&app, 240, 60);
    assert!(text.contains("T15, auto"), "the plan lists no trees");
    assert!(text.contains("6 runs"), "the plan does not count the trees");
    // Progress lands on the ticked drafter: 1 of its 3 depths and 2 trees
    // measured.
    app.bench.as_mut().unwrap().phase = BenchPhase::Running;
    let text = screen_text(&app, 240, 60);
    let queue: Vec<&str> = text
        .lines()
        .filter(|l| l.contains("Ticked-Drafter"))
        .collect();
    assert!(
        queue.iter().any(|l| l.contains(" 1/5")),
        "no 1/5 progress on the ticked drafter: {queue:?}"
    );
    // On an MoE model the plan has no trees: 3 depths, no tree runs.
    if let Some(info) = app.models[dense].info.as_mut() {
        info.arch = "qwen35moe".into();
    }
    app.bench.as_mut().unwrap().phase = BenchPhase::Setup;
    let text = screen_text(&app, 240, 60);
    assert!(!text.contains("T15, auto"), "trees planned on an MoE model");
    assert!(text.contains("4 runs"), "an MoE plan still counts trees");
}
