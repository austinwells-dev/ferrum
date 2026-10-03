use crate::*;

#[derive(Clone, Copy, PartialEq)]
pub enum Mode {
    Chat,
    Serve,
}

#[derive(Clone, Copy, PartialEq)]
pub enum Scope {
    Both,
    Chat,
    Serve,
}

pub enum Kind {
    /// First option is the default and is never passed on the command line.
    Cycle(&'static [&'static str]),
    Num {
        step: f64,
        min: f64,
        max: f64,
        base: f64,
        int: bool,
    },
    Text {
        presets: &'static [&'static str],
        secret: bool,
    },
}

pub struct Field {
    pub key: &'static str,
    pub label: &'static str,
    pub group: &'static str,
    pub scope: Scope,
    pub kind: Kind,
    /// Empty means "not set": the binary uses its own default.
    pub value: String,
    pub default: &'static str,
    pub help: &'static str,
}

pub fn field(
    key: &'static str,
    label: &'static str,
    group: &'static str,
    scope: Scope,
    kind: Kind,
    default: &'static str,
    help: &'static str,
) -> Field {
    Field {
        key,
        label,
        group,
        scope,
        kind,
        value: String::new(),
        default,
        help,
    }
}

pub fn num(step: f64, min: f64, max: f64, base: f64, int: bool) -> Kind {
    Kind::Num {
        step,
        min,
        max,
        base,
        int,
    }
}

pub fn text(presets: &'static [&'static str]) -> Kind {
    Kind::Text {
        presets,
        secret: false,
    }
}

pub fn fields() -> Vec<Field> {
    use Scope::{Both, Chat, Serve};
    let m = "Model";
    let r = "Reasoning";
    let s = "Sampling";
    let d = "Speculative decoding";
    let v = "Server";
    vec![
        field(
            "context",
            "Context",
            m,
            Both,
            Kind::Cycle(&[
                "auto", "4096", "8192", "16384", "32768", "65536", "131072", "262144",
            ]),
            "auto",
            "Tokens of context. auto picks the largest that fits in memory.",
        ),
        field(
            "max_tokens",
            "Max tokens",
            m,
            Both,
            num(1024., 1., 1_000_000., 8192., true),
            "default",
            "Cap on tokens generated per reply.",
        ),
        field(
            "think",
            "Thinking",
            r,
            Both,
            Kind::Cycle(&["on", "off"]),
            "on",
            "Let the model reason before answering.",
        ),
        field(
            "effort",
            "Effort",
            r,
            Both,
            Kind::Cycle(&["default", "low", "medium", "high"]),
            "default",
            "Reasoning effort passed to the chat template.",
        ),
        field(
            "budget",
            "Reasoning budget",
            r,
            Both,
            num(256., 0., 100_000., 2048., true),
            "unlimited",
            "Force the end of thinking after this many tokens.",
        ),
        field(
            "show",
            "Show thinking",
            r,
            Chat,
            Kind::Cycle(&["show", "hide"]),
            "show",
            "Print the reasoning stream in the chat.",
        ),
        field(
            "temperature",
            "Temperature",
            s,
            Both,
            num(0.05, 0., 2., 0.7, false),
            "model default",
            "Higher is more random. Backspace resets to the model's default.",
        ),
        field(
            "top_p",
            "Top-p",
            s,
            Both,
            num(0.05, 0., 1., 0.95, false),
            "model default",
            "Nucleus sampling cutoff.",
        ),
        field(
            "top_k",
            "Top-k",
            s,
            Both,
            num(5., 0., 500., 20., true),
            "model default",
            "Keep only the k most likely tokens (0 disables).",
        ),
        field(
            "min_p",
            "Min-p",
            s,
            Both,
            num(0.01, 0., 1., 0.05, false),
            "model default",
            "Drop tokens below this fraction of the top token's probability.",
        ),
        field(
            "presence",
            "Presence penalty",
            s,
            Both,
            num(0.1, -2., 2., 0.5, false),
            "0",
            "Penalize tokens that already appeared.",
        ),
        field(
            "frequency",
            "Frequency penalty",
            s,
            Both,
            num(0.1, -2., 2., 0.5, false),
            "0",
            "Penalize tokens by how often they appeared.",
        ),
        field(
            "repeat",
            "Repeat penalty",
            s,
            Both,
            num(0.05, 0.5, 2., 1.1, false),
            "1",
            "Multiplicative repetition penalty.",
        ),
        field(
            "system",
            "System prompt",
            s,
            Chat,
            text(&[]),
            "none",
            "Enter to type a system prompt for the chat.",
        ),
        field(
            "draft",
            "Drafter",
            d,
            Both,
            text(&["off", "mtp"]),
            "off",
            "Lossless speedup. ←→ cycles, Enter opens the picker of detected DSpark/DFlash drafters.",
        ),
        field(
            "draft_max",
            "Drafts per step",
            d,
            Both,
            num(1., 1., 16., 3., true),
            "auto",
            "How many tokens the drafter proposes at a time.",
        ),
        field(
            "draft_quant",
            "Draft precision",
            d,
            Both,
            Kind::Cycle(&["q4_0", "q8_0"]),
            "q4_0",
            "Weight precision of the drafter.",
        ),
        field(
            "tools",
            "Tools",
            "Agent tools",
            Chat,
            Kind::Cycle(&["off", "ask", "auto"]),
            "off",
            "Let the model run commands and edit files in a sandbox. ask: approve writes, commands and downloads; auto: no prompts.",
        ),
        field(
            "workspace",
            "Workspace folder",
            "Agent tools",
            Chat,
            text(&[]),
            "~/ferrum-workspace",
            "The only folder the tools can write to (Enter to change). Attachments are copied into it.",
        ),
        field(
            "network",
            "Network",
            "Agent tools",
            Chat,
            Kind::Cycle(&["on", "off"]),
            "on",
            "on lets the model fetch web pages and commands use the internet (never this Mac's own services).",
        ),
        field(
            "host",
            "Host",
            v,
            Serve,
            text(&["127.0.0.1", "0.0.0.0"]),
            "127.0.0.1",
            "127.0.0.1 is this Mac only; 0.0.0.0 exposes it to the network.",
        ),
        field(
            "port",
            "Port",
            v,
            Serve,
            num(1., 1., 65535., 8080., true),
            "8080",
            "Server listens on this port.",
        ),
        field(
            "api_key",
            "API key",
            v,
            Serve,
            Kind::Text {
                presets: &[],
                secret: true,
            },
            "none",
            "Require this bearer token from clients (Enter to type).",
        ),
        field(
            "alias",
            "Model alias",
            v,
            Serve,
            text(&[]),
            "file name",
            "Model name reported to clients.",
        ),
        field(
            "reserve",
            "Reserve (MiB)",
            v,
            Serve,
            num(256., 0., 65536., 1024., true),
            "1024",
            "GPU memory left free for the system.",
        ),
        field(
            "snapshots",
            "Snapshots",
            v,
            Serve,
            num(1., 0., 32., 2., true),
            "2",
            "Recurrent snapshots kept for prefix reuse.",
        ),
        field(
            "chunk",
            "Prefill chunk",
            v,
            Serve,
            num(128., 1., 8192., 512., true),
            "512",
            "Prompt tokens per forward pass.",
        ),
        field(
            "verbose",
            "Verbose log",
            v,
            Serve,
            Kind::Cycle(&["off", "on"]),
            "off",
            "Log request bodies and timings.",
        ),
    ]
}
