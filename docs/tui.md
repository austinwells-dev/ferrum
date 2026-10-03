# TUI guide

```sh
ferrum-tui                 # or: ferrum-tui MODEL.gguf
alias ferrum=ferrum-tui    # optional
```

After a short startup animation, the home screen offers **Chat**, **Agents**, **Serve**, **Benchmark** and **Settings**. Choosing Chat, Agents or Serve lists your saved **favorites** (a model, its settings and an optional drafter), along with **New setup** (saved as a favorite when you launch it) and **One-time run**.

Ferrum looks for models in the current folder, `./models`, `~/models`, `~/Downloads`, the Hugging Face cache and LM Studio's folders. You can add more folders in Settings. Models, quants and drafters that Ferrum can't run are greyed out with the reason, for example an unsupported architecture or a drafter whose hidden size doesn't match the model.

Type `/help` in any chat for the full command list.

## Chat

Chat runs the model inside the TUI with streaming markdown and a separate reasoning view (`/thoughts` shows or hides it, `/think on|off` turns reasoning on or off).

**Attachments.** You can attach files in three ways: with `/attach PATH`, by dropping them onto the terminal, or from the clipboard with Ctrl-V or `/paste` (files, screenshots or text). `/screenshot` lets you drag out a screen region. If the model has a vision projector next to it, images go to the model as pixels. Otherwise, macOS Vision reads the text in the image and that text is sent instead.

**Tools.** Set *Tools* to `ask` or `auto` in a setup (or type `/tools ask`) and the model can read, write and edit files, run shell commands and fetch web pages. In `ask` mode you approve every write, command and download.

## Agents

The Agents tab is a local coding agent that works on a real project folder. Pick a setup, set the *Project folder* (by default, the folder you started Ferrum from) and give it a task.

**Tools:** `read_file`, `grep` (ripgrep), `glob`, `list_dir`, `edit_file`, `write_file`, `bash`, `check` (cargo, go, python or js errors), background processes (`process_start`, `process_output`, `process_stop`), `web_search`, `fetch_url`, a `todo` plan shown in a sidebar, and `ask_user`.

**Approval modes** (`/tools ...`):

| Mode | Behaviour |
|---|---|
| `edits` | File edits run freely; commands and downloads ask first |
| `ask` | Every change asks first |
| `auto` | Never asks |

`/plan on` makes the agent read-only until you run `/plan off`.

### Web search

`web_search` is available in Chat and Agents whenever *Network* is on. Choose the back end in the setup (*Web search*, *Search URL*, *Search API key*):

| Provider | Needs | Notes |
|---|---|---|
| `duckduckgo` | nothing | Default. Scrapes DuckDuckGo's HTML results |
| `instant` | nothing | DuckDuckGo's answer API. Summaries and related topics only, often empty for ordinary queries |
| `searxng` | *Search URL*, e.g. `http://localhost:8080` | Needs `json` under `search.formats` in the server's `settings.yml` |
| `brave` | *Search API key* | Brave Search API |
| `custom` | *Search URL* with `{query}` (and optionally `{key}`) | Any GET endpoint that returns JSON with a `results` list of `title`, `url` and `content` |

Write the key as `$NAME` to read it from an environment variable instead of saving it in `tui.json`. The search request is made from your setting, never from model output, and runs outside the sandbox so that a SearXNG server on this Mac or LAN can be reached. The model only supplies the query. `ask` mode asks before each search.

### Sandbox

Every command and download runs inside a macOS Seatbelt sandbox (`sandbox-exec`, profile in [`sandbox.sb`](../src/bin/ferrum-tui/sandbox.sb)):

- the whole disk is read-only except the workspace: the project folder in Agents, and in Chat a folder of its own for every session (a new `<date>-<time>-<model>` subfolder of `~/ferrum-workspace`, set by *Sandboxes folder*),
- keys and credentials such as `~/.ssh` are hidden,
- the network can be switched off entirely,
- localhost services on your Mac can never be reached.

### Context optimisers

Local models have small contexts, so the agent works to keep its context useful:

- Long tool output is stored and searchable with BM25. The model gets a short excerpt and a handle it can follow up with (`ctx_search`, `ctx_read`). `bash` takes an `intent` and returns only the lines relevant to it.
- Re-reading a file that hasn't changed returns a short notice instead of the contents.
- At 55% context, old tool output is elided in one batch, so the prompt cache is rebuilt once instead of every turn. At 80% the session is snapshotted.
- Repeated identical tool calls are stopped.
- The system prompt includes a repo map, the commands available on the machine, and your `AGENTS.md` / `CLAUDE.md`.

`/ctx` shows context use and what was saved; `/compact` snapshots on demand.

### Ponytail (minimal-code mode)

`/ponytail lite|full|ultra|off` adds minimal-code guidance to every turn. The agent reads first, then climbs a ladder: is this needed at all, is it already in the codebase, does the standard library, the platform or an installed dependency cover it, can it be one line, and only then the smallest new code. Deferred shortcuts are marked `ponytail:`.

| Command | What it does |
|---|---|
| `/review` | Look for over-engineering in your uncommitted diff |
| `/audit` | Look for code in the repo that doesn't need to exist |
| `/debt` | List the shortcuts marked `ponytail:` |

## Sessions

Every chat and agent session is saved after each reply to `~/.config/ferrum/sessions` (nothing is written before your first message). `/new` starts a fresh session in the model that is already loaded and keeps the old one; `/resume` lists earlier sessions (agents show only the ones for the current project), and Enter brings one back with its transcript, tool results and, for agents, its plan, touched files and stored output. The model re-reads the conversation with your next message. `x` twice deletes a session from the list.

## Serve

Serve starts `ferrum-server` with the chosen setup and shows its endpoint, health and live log. See the [server guide](server.md).

## Benchmark

The Benchmark tab finds the fastest speculative-decoding setup for a model on *this* Mac.

1. Pick a model. The tab lists plain decoding, the model's MTP head (if the GGUF has a usable one) and every compatible drafter it can find. Compatibility is checked by hidden size, layer count and vocabulary.
2. Each drafter is loaded once, and its draft depth is swept (`quick` tries a few depths, `full` tries every depth). The tab measures greedy decode speed on three prompts (code, prose and structured data) and the mean number of tokens accepted per verify step.
3. Every run's output is compared with plain decoding. Rows whose text differs are marked `≠` and are never recommended.
4. An optional stage also sweeps the prefill chunk size.

Results are ranked by speedup over plain decoding and remembered per machine and model in `~/.config/ferrum/bench.json`. Press `s` to save the winner as a Chat/Agents favorite, or `v` to save it as a Serve favorite (including the chunk size).
