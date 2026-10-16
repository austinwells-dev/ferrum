# TUI guide

```sh
ferrum-tui                 # or: ferrum-tui MODEL.gguf
alias ferrum=ferrum-tui    # optional
```

After a short startup animation, the home screen offers **Chat**, **Agents**, **Serve**, **Benchmark** and **Settings**. Choosing Chat, Agents or Serve lists your saved **favorites** (a model, its settings and an optional drafter), along with **New setup** (saved as a favorite when you launch it) and **One-time run**.

Ferrum looks for models in the current folder, `./models`, `~/models`, `~/Downloads`, the Hugging Face cache and LM Studio's folders. You can add more folders in Settings. Models, quants and drafters that Ferrum can't run are greyed out with the reason, for example an unsupported architecture or a drafter whose hidden size doesn't match the model.

Type `/help` in any chat for the full command list.

Each setup's **Model** group has *Context* and *KV cache*. *KV cache* `q8` stores the attention cache in 8 bits, which nearly halves its memory (so `auto` context grows) and speeds up long conversations; `f16` (the default) is exact. It applies to Chat, Agents, Serve and the Benchmark tab, and is saved with favorites. See [K/V cache format](server.md#kv-cache-format).

## Chat

Chat runs the model inside the TUI with streaming markdown (syntax-highlighted code blocks for common languages, tables that wrap to the window, links and horizontal rules) and a separate reasoning view (`/thoughts` shows or hides it, `/think on|off` turns reasoning on or off).

**Attachments.** You can attach files in three ways: with `/attach PATH`, by dropping them onto the terminal, or from the clipboard with Ctrl-V or `/paste` (files, screenshots or text). `/screenshot` lets you drag out a screen region. If the model has a vision projector next to it, images go to the model as pixels. Otherwise, macOS Vision reads the text in the image and that text is sent instead.

**Previews.** Images and videos are shown inline: attachments waiting to be sent appear in a strip above the message box, and sent ones appear under your message (videos show their first frame). `/previews off` hides them. Ctrl-O or `/preview` opens the newest one full screen. `/preview N` opens the Nth one and `/preview PATH` opens any file. In the preview, ←/→ moves between them, space pauses, `o` opens the file in its default app and esc closes it. Terminals that support the kitty graphics protocol (kitty, Ghostty, WezTerm) show the real pixels; other terminals get a 24-bit colour half-block rendering. To choose the mode yourself, set `FERRUM_GRAPHICS=kitty` or `FERRUM_GRAPHICS=blocks`. Inside tmux the half-block mode is used. GIFs animate. Videos (`mp4`, `mov`, `m4v`, `webm`, `mkv`, `avi`, up to 512 MB) play through `ffmpeg` if it is installed. Without ffmpeg you get the Quick Look poster frame. Models can't see videos, so the model is only told that the file is in the workspace.

**Tools.** Set *Tools* to `ask` or `auto` in a setup (or type `/tools ask`) and the model can read, write and edit files, run shell commands and fetch web pages. In `ask` mode you approve every write, command and download.

## Agents

The Agents tab is a local coding agent that works on a real project folder. Pick a setup, set the *Project folder* (by default, the folder you started Ferrum from) and give it a task.

**Tools:** `read_file`, `grep` (ripgrep), `glob`, `list_dir`, `edit_file`, `write_file`, `bash`, `check` (cargo, go, python or js errors), background processes (`process_start`, `process_output`, `process_stop`), `web_search`, `fetch_url`, a `todo` plan shown in a sidebar, and `ask_user`.

**Approval modes** (`/tools ...`):

| Mode | Behaviour |
|---|---|
| `edits` | File edits run freely; commands, `check` (it builds the project) and downloads ask first |
| `ask` | Every change asks first |
| `auto` | Never asks |

`/plan on` makes the agent read-only until you run `/plan off`.

### Web search

`web_search` is available in Chat and Agents whenever *Network* is on. Choose the back end under **Settings → Web search** (*Provider*, *Search URL*, *API key*). It applies to every setup:

| Provider | Needs | Notes |
|---|---|---|
| `duckduckgo` | nothing | Default. Scrapes DuckDuckGo's HTML results |
| `instant` | nothing | DuckDuckGo's answer API. Summaries and related topics only, often empty for ordinary queries |
| `searxng` | *Search URL*, e.g. `http://localhost:8080` | Needs `json` under `search.formats` in the server's `settings.yml` |
| `brave` | *Search API key* | Brave Search API |
| `custom` | *Search URL* with `{query}` (and optionally `{key}`) | Any GET endpoint that returns JSON with a `results` list of `title`, `url` and `content` |

Write the key as `$NAME` to read it from an environment variable instead of saving it in `tui.json`. The search request is made from your setting, never from model output, and runs outside the sandbox so that a SearXNG server on this Mac or LAN can be reached. The model only supplies the query. `ask` mode asks before each search.

### Skills

The agent loads `SKILL.md` files on demand. It checks `<project>/skills/*/SKILL.md`, `<project>/.ferrum/skills/*/SKILL.md`, `<project>/*/SKILL.md` and `~/.ferrum/skills/*/SKILL.md`: one folder level below each, never deeper. The first skill with a given name wins, in that order.

Each file may start with frontmatter:

```
---
name: deploy
description: Ship the app to staging
---
```

Without frontmatter, the folder name and the first line of text are used. Only the name and description go into the prompt; the agent calls `skill(name)` to read the whole file and see the files beside it, and uses its normal tools to read or run them (so scripts run in the sandbox). Skills appear in the `/` menu. `/skills` rescans and lists what was found, and `/name [task]` loads a skill and starts on the task. Up to 40 skills are listed.

### Sandbox

Every command and download runs inside a macOS Seatbelt sandbox (`sandbox-exec`, profile in [`sandbox.sb`](../src/bin/ferrum-tui/sandbox.sb)):

- the whole disk is read-only except the workspace: the project folder in Agents, and in Chat a folder of its own for every session (a new `<date>-<time>-<model>` subfolder of `~/ferrum-workspace`, set by *Sandboxes folder*),
- keys and credentials such as `~/.ssh` are hidden,
- the network can be switched off entirely,
- localhost services on your Mac can never be reached, and neither can Unix sockets outside the workspace (tmux, Docker, ssh-agent),
- it cannot start programs outside the sandbox through `open` or Apple events, or use the clipboard or keychain,
- commands left running in the background end with the command (use `process_start` for servers).

The project's `.git/config`, `.git/hooks` and `.git` itself are read-only to the agent (it can still commit and branch), because git settings and hooks run programs later, outside the sandbox, for example when a git-aware shell prompt runs `git status`. The TUI's own `git` calls (branch, status, `/review`) run sandboxed as well, and the file tools, which run in the TUI itself, refuse symlinks out of the workspace rather than follow them. Shell history, Cargo and Hugging Face tokens, and the 1Password, Brave, Arc and Edge data folders are hidden too.

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
3. Every run's output is compared with plain decoding. Rows that match are marked `✓`. Rows whose text differs early are marked `≠` and are never recommended. Rows marked `≈` matched for at least the first 25% of every answer before a near-tie flipped (batched verification rounds slightly differently).
4. If a `≈` row would beat every exact row, the tab first asks it 30 bundled math questions, and plain decoding the same questions (its answers are cached per model in `~/.config/ferrum/answer-check.json`). The row is marked `≈✓` and may be recommended if it gets at most one more question wrong than plain decoding, after subtracting questions only it got right. Otherwise it is marked `≈✗`. Rows marked `≈?` were not checked and are not recommended. Checks run fastest row first and stop at the first pass.
5. An optional stage also sweeps the prefill chunk size.

Results are ranked by speedup over plain decoding and remembered per machine and model in `~/.config/ferrum/bench.json`. Press `s` to save the winner as a Chat/Agents favorite, or `v` to save it as a Serve favorite (including the chunk size).

### Measuring intelligence

Set *Measure* to `intelligence` to score a model on four question sets instead of timing it: math (GSM8K), knowledge (MMLU), following instructions (IFEval) and reading code (CRUXEval, predicting what a function returns). Every answer is checked by a program, not by another model. Generation is greedy, so a result can be reproduced exactly.

The question sets are not part of the binary. Install them once (this downloads samples from Hugging Face into `~/.config/ferrum/eval`, or `$FERRUM_EVAL_DIR`):

```bash
python3 tools/eval_import.py
```

- *Questions*: `quick` asks the first 15 questions of each set and **predicts** the full score with a 95% range (shown as `~62% ±9`). `full` asks all of them and gives the **actual** score.
- *Thinking*: `off` (the default), or on with a reasoning budget per question. Thinking usually raises the score a lot and makes the run several times longer, so results with and without it are kept apart.
- *Drafter*: `off` uses plain decoding. `tuned` uses the drafter this Mac's speed benchmark recommended for the model, which is faster and gives the same answers up to near-tie flips.

The score is the mean of the four category scores. Complete runs are saved in `~/.config/ferrum/eval.json`, the newest per model, question files and setup. Previous results for the model are listed under the options. A stopped run is shown but not saved.

These are public benchmarks that models may have seen in training. Use the scores to compare models and settings on this Mac, not as leaderboard numbers.
