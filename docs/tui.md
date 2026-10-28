# TUI guide

```sh
ferrum-tui                 # or: ferrum-tui MODEL.gguf
alias ferrum=ferrum-tui    # optional
```

After a short startup animation, Ferrum opens on **Home**: one button per tab. The tabs run along the top of every screen: **1 Home**, **2 Chat**, **3 Agents**, **4 Serve**, **5 Bench**, **6 Models** and **7 Settings**. Press a number to switch tabs, `?` to list the keys of the current screen, and ctrl-p for a command palette. Changing tabs never stops anything: a loaded chat model, a running server and a benchmark run all keep going, and the right end of the tab bar names the model that is loaded (by a chat, an agent, the server or a benchmark run), what it is doing right now (thinking, writing, running a tool, the benchmark task or question it is on, with the newest words it wrote), and how much memory the models hold.

Chat, Agents and Serve show your saved **favorites** (a model, its settings and an optional drafter), along with **New setup** (saved as a favorite when you launch it) and **One-time run**. The preview beside the list shows the setup's settings, how its model measured on this Mac and the exact command. Once a session or the server is running, its tab shows that instead. Only one model is loaded at a time: launching something else (a chat, an agent, the server or a benchmark) while a model is loaded asks you to press the key again to unload the first.

Ferrum looks for models in the current folder, `./models`, `~/models`, `~/Downloads`, the Hugging Face cache and LM Studio's folders. You can add more folders in Settings. **Models** lists every file it found in one table with its quant, size, architecture, context, vision support, the drafters that fit it, and its best speed, intelligence and agent results on this Mac (`o` changes the sort). Files Ferrum can't run stay listed with the reason, such as an unsupported architecture. From the table, `enter` starts a chat with the model, `a` an agent, `s` a server and `b` a benchmark.

Type `/help` in any chat for the full command list.

## Layout

Every screen fills the window with panes, and one pane in each split takes whatever room the others leave, so no rows go unused. On a narrow window (under 100 columns by default) the panes stack, and Bench shows one group at a time (`tab` moves between them).

`[` and `]` move the split on the current screen, and the setting is kept. **Settings › Layout** holds the same sizes and a few more:

| Setting | What it changes |
|---|---|
| Setup list width | the list's share of the width in Chat, Agents and Serve |
| Bench setup column | the setup and queue column's share on Bench |
| Serve endpoint column | the endpoint column's share beside the server log |
| Models table height | the most of the height the model table takes |
| Chat side panel | `auto` (from 110 columns), `always` or `off`; ctrl-b toggles it in a session |
| Side panel width | columns of the chat and agent side panel |
| Chat text width | where chat prose wraps; the conversation pane itself always fills |
| One pane group below | the width under which screens stack their panes |

They are saved in `~/.config/ferrum/tui.json` under `layout`. The key bar lists the keys for whatever is selected, and messages appear at its right end. Settings, the setup editor and Bench pin a short explanation of the selected row at the bottom of their pane.

In a chat, typing goes to the message box. Esc on an empty box hands the keys to the conversation: ↑↓ scroll, the number keys (1–8) switch tabs, and typing goes back to the box. ctrl-d there unloads the model.

Each setup's **Model** group has *Context* and *KV cache*. *KV cache* `q8` stores the attention cache in 8 bits, which nearly halves its memory (so `auto` context grows) and speeds up long conversations; `f16` (the default) is exact. It applies to Chat, Agents, Serve and the Benchmark tab, and is saved with favorites. See [K/V cache format](server.md#kv-cache-format).

## Chat

Chat runs the model inside the TUI with streaming markdown (syntax-highlighted code blocks for common languages, tables that wrap to the window, links and horizontal rules) and a separate reasoning view (`/thoughts` shows or hides it, `/think on|off` turns reasoning on or off).

**Attachments.** You can attach files in three ways: with `/attach PATH`, by dropping them onto the terminal, or from the clipboard with Ctrl-V or `/paste` (files, screenshots or text). `/screenshot` lets you drag out a screen region. If the model has a vision projector next to it, images go to the model as pixels. Otherwise, macOS Vision reads the text in the image and that text is sent instead. `/detail low|high|auto|N` sets how many tokens the images in the next messages may take (low is 256, high is the *Image tokens* setting), and the Model group's *Image tokens* and *Image min tokens* set the cap and floor (`--image-tokens`, `--image-min-tokens`) for Chat, Agents and Serve.

**Previews.** Images and videos are shown inline: attachments waiting to be sent appear in a strip above the message box, and sent ones appear under your message (videos show their first frame). `/previews off` hides them. Ctrl-O or `/preview` opens the newest one full screen. `/preview N` opens the Nth one and `/preview PATH` opens any file. In the preview, ←/→ moves between them, space pauses, `o` opens the file in its default app and esc closes it. Terminals that support the kitty graphics protocol (kitty, Ghostty, WezTerm) or iTerm2's inline images show the real pixels; other terminals get a 24-bit colour half-block rendering. To choose the mode yourself, set `FERRUM_GRAPHICS=kitty`, `FERRUM_GRAPHICS=iterm` or `FERRUM_GRAPHICS=blocks`. Inside tmux the half-block mode is used. GIFs animate. Videos (`mp4`, `mov`, `m4v`, `webm`, `mkv`, `avi`, up to 512 MB) play through `ffmpeg` if it is installed. Without ffmpeg you get the Quick Look poster frame. Models can't see videos, so the model is only told that the file is in the workspace.

**Tools.** Set *Tools* to `ask` or `auto` in a setup (or type `/tools ask`) and the model can read, write and edit files, run shell commands, fetch web pages, and look at image files with `view_image` when a vision projector is loaded. In `ask` mode you approve every write, command and download.

**Pictures.** When the Image tab has a Qwen-Image model, chats and agents with tools on also get `generate_image`: the model can draw a picture from a prompt, or edit pictures in its workspace, and the PNG is saved there (`images/` unless it names a path). The finished picture is added to the tool's reply as an image, so the model can see what it drew — and a mistake it can see is one it can fix; pass `inject: false` for a call when you only need the file. It uses the Image tab's model and quantization choices. If the picture's stages don't fit beside the chat model, the chat model is unloaded while the picture is made and loaded again afterwards, so the next reply reads the conversation afresh (the tool's line shows each stage). Esc stops it after the current step. In `ask` mode it asks first, and plan mode refuses it.

**Looking at images.** With a vision projector loaded, the model itself can open an image file with `view_image` (PNG, JPEG, GIF, WebP or BMP): it names a path, and the pixels appear in that tool's reply, just as if you had attached the file. No attachment, paste or screenshot is needed — pointing at a path is enough. With the sandbox on, the path must be in the workspace (or its attachments); with `/sandbox off` it may be any file you can read. A picture shown this way counts against the 16 images one request can carry, as do `generate_image` results.

## Agents

The Agents tab is a local coding agent that works on a real project folder. Pick a setup, set the *Project folder* (by default, the folder you started Ferrum from) and give it a task.

**Tools:** `read_file`, `view_image` (show the model an image file), `grep` (ripgrep), `glob`, `list_dir`, `edit_file`, `write_file`, `bash`, `check` (cargo, go, python or js errors), background processes (`process_start`, `process_output`, `process_stop`), `web_search`, `fetch_url`, a `todo` plan shown in a sidebar, `ask_user`, and `generate_image` when an image model is set up (see Pictures above).

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
- At 85% context the session is snapshotted once (as `/compact` does), so the prompt cache is rebuilt once instead of every turn. Tool output stays in full until then.
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

## Image

The Image tab (key `7`, or `i` on Home) draws with a Qwen-Image checkpoint found in the usual model folders. `p` edits the prompt, ↑↓ pick a row (new picture or edit, model, size, seed, transformer and encoder quantization), ←→ change it, `r` draws a new seed and `g` or enter generates. To edit, set the first row to the picture on screen and describe the change in the prompt: the result keeps that picture's shape at the chosen size's area. The picture appears in the pane (on terminals that show images) and is saved as a PNG in `$FERRUM_IMAGES`, default `~/Pictures/Ferrum`. `o` opens it, and esc stops a running generation. Chat models are unloaded first, since the stages need most of the memory. The settings are kept in `~/.config/ferrum/image.json`. See [Image generation](server.md#image-generation).

## Sessions

Every chat and agent session is saved after each reply to `~/.config/ferrum/sessions` (nothing is written before your first message). `/new` starts a fresh session in the model that is already loaded and keeps the old one; `/resume` lists earlier sessions (agents show only the ones for the current project), and Enter brings one back with its transcript, tool results and, for agents, its plan, touched files and stored output. The model re-reads the conversation with your next message. `x` twice deletes a session from the list.

## Serve

Serve starts `ferrum-server` with the chosen setup and shows its endpoint, routes and live log, with a traffic chart of each request's decode speed. The server keeps running on other tabs; `x` twice stops it. *Parallel requests* (1–8) sets `--parallel`: each request gets a session with its own cache, and `auto` context shrinks so all of them fit. See the [server guide](server.md).

## Benchmark

The Bench tab measures a model on *this* Mac. Every measure uses the same frame: the setup and its queue on the left, a chart and a table of rows filling the rest, and notes on the marks in a third column on windows 170 columns or wider. `m` (or the *Measure* row) cycles through **speed** (below), which finds the fastest speculative-decoding setup; **intelligence** ([Measuring intelligence](#measuring-intelligence)), which scores answers to question sets; and **agent** ([Measuring agentic coding](#measuring-agentic-coding)), which has the coding agent fix real bugs.

1. Pick a model. The tab lists plain decoding, the model's MTP head (if the GGUF has a usable one) and every compatible drafter it can find. Compatibility is checked by hidden size, layer count and vocabulary.
2. Each drafter is loaded once, and its draft depth is swept (`quick` tries a few depths, `full` tries every depth). The tab measures greedy decode speed on three prompts (code, prose and structured data) and the mean number of tokens accepted per verify step. On dense models, the MTP head and every drafter model then try two **draft trees**, each on its own load: a tree of up to 15 guesses (`T15` in the table), and KFlash's learned budget (`auto`, which for DFlash2 also drafts a 16-row block over a reduced draft vocabulary). KFlash learns this Mac's costs and the drafter's calibration as it runs and keeps them in `~/.cache/ferrum/kflash`, so its first benchmark spends some cycles exploring. A tree wins only when it is faster: on a near tie the chain is recommended. MoE models and Flint skip the trees: each tree row routes to its own experts, so trees check slower than plain decoding there, and the wide check's rounding drifts from it. LFM3 skips them too, because a tree row would need its parent's short-conv window.
3. Every run's output is compared with plain decoding. Rows that match are marked `✓`. Rows whose text differs early are marked `≠` and are never recommended. Rows marked `≈` matched for at least the first 25% of every answer before a near-tie flipped (batched verification rounds slightly differently).
4. If a `≈` row would beat every exact row, the tab first asks it 30 bundled math questions, and plain decoding the same questions (its answers are cached per model in `~/.config/ferrum/answer-check.json`). The row is marked `≈✓` and may be recommended if it gets at most one more question wrong than plain decoding, after subtracting questions only it got right. Otherwise it is marked `≈✗`. Rows marked `≈?` were not checked and are not recommended. Checks run fastest row first and stop at the first pass.
5. An optional stage also sweeps the prefill chunk size.

In Chat, Agents and Serve settings, *Draft tree* (`off`, `auto`, `15`, `8`), *Draft block* and *Draft vocabulary* set the same options by hand (`--draft-tree`, `--draft-block`, `--draft-vocab`).

Results are ranked by speedup over plain decoding and remembered per machine and model in `~/.config/ferrum/bench.json`. Press `s` to save the winner as a Chat/Agents favorite, or `v` to save it as a Serve favorite (including the chunk size).

### Measuring intelligence

Set *Measure* to `intelligence` to score a model on seven question sets instead of timing it. Four are about answering questions: math, knowledge, following instructions and reading code (predicting what a Python function returns). Three are about real work: calling tools, finding a fact in a long document, and writing code that has to pass tests. Every answer is checked by a program, not by another model. Generation is greedy, so a result can be reproduced exactly.

*Difficulty* picks the sets:

| | `hard` (the default) | `standard` |
|---|---|---|
| Math | MATH-500, levels 3-5, problems with a plain-number answer | GSM8K |
| Knowledge | MMLU-Pro (10 options) | MMLU (4 options) |
| Instructions | IFEval, the prompts with the most rules | IFEval |
| Code reading | CRUXEval, the longest functions | CRUXEval |
| Tool calling | BFCL parallel and parallel-multiple: several calls in one turn | BFCL simple and multiple: one call, sometimes choosing among tools |
| Long context | a vault code hidden at 10-90% depth in 16K and 32K-token logs, with codes for other vaults as decoys | the same in 4K and 8K-token logs, no decoys |
| Writing code | HumanEval: complete a function, then its tests run | MBPP: write a function from a description and one example test |

The standard question sets are easy for current mid-size models, which mostly score in the 80s and 90s. Use them for small models. Each difficulty is installed and scored separately, and its results are never mixed with the other's.

Tool calls are graded like BFCL: the right functions, each called once in any order, with every argument one of its allowed values (numbers by value, strings ignoring case, spaces and punctuation) and nothing extra. Code is graded by running the answer's last ```` ```python ```` block and the hidden tests with `python3`, inside the same Seatbelt sandbox the agent uses but with no writes and no network, for at most 10 seconds. Writing code needs `python3` on your `PATH`. Long-context questions need a bigger context, so a run sizes its context from its longest question (about 38K tokens for the hard sets, plus the thinking budget). On a model that cannot fit that next to its weights, the run fails at load with the reason.

The question sets are not part of the binary. Select *Question sets* and press Enter to install them: the tab downloads a fixed sample of 100 questions per set from Hugging Face (with the system `curl`, 50 for long context, which is generated locally) into `~/.config/ferrum/eval` (the hard sets into its `hard` folder), or `$FERRUM_EVAL_DIR`. The same questions are picked every time. Requests are spaced out, and if Hugging Face rate-limits the download (HTTP 429), the tab waits and retries. Each finished set is kept, so if the install fails or you leave the tab (which cancels it), installing again picks up where it stopped. The question files are only replaced once all seven sets are in. Setting `HF_TOKEN` raises the rate limit. To remove the sets, press Enter on the line twice. Saved results stay. Results from different question files are never mixed, and older ones are marked as such.

Each answer gets a token budget after any thinking. Instructions that ask for a length get room for it (about 1.6 tokens per required word, 80 per required paragraph), and tool calls room for the calls expected. Results saved before these budgets were added (October 2026) cut off the longest answers, so their instruction-following and tool-calling scores are slightly low. They count as older results and are marked as such.

- *Questions*: `quick` asks the first 15 questions of each set (105 in all) and **predicts** the full score with a 95% range (shown as `~62% ±9`). `full` asks all of them and gives the **actual** score.
- *Thinking*: `off` (the default), or on with a reasoning budget per question. Thinking usually raises the score a lot and makes the run several times longer, so results with and without it are kept apart.
- *Drafter*: `off` uses plain decoding. `tuned` uses the drafter this Mac's speed benchmark recommended for the model, which is faster and gives the same answers up to near-tie flips.

The score is the mean of the seven category scores. Complete runs are saved in `~/.config/ferrum/eval.json`, the newest per model, question files and setup. Previous results for the model are listed under the options. A stopped run is shown but not saved.

Press `c` in the intelligence setup or results to open **Compare**: every saved result from every model, best first, one bar each. Each row is labeled with the model, the difficulty, the run size, thinking, and whether a drafter was used, plus this Mac's best speedup for the model from the speed benchmark. A predicted score's 95% range is shaded (`▒`). On a screen at least 140 columns wide, Compare uses the full width and draws each result as a column, best on the left, with its labels under the axis and the selected result's details below; `←`/`→` move between columns. It opens on the setup's difficulty. `d` cycles difficulty (hard, standard, both), `t` cycles thinking (all, off, on), `f` shows only full runs, and Enter opens a result's category bars. Pressing `x` twice deletes every saved result.

These are public benchmarks that models may have seen in training. Use the scores to compare models and settings on this Mac, not as leaderboard numbers.

While the test runs, the category chart shrinks to one line per category and a live feed fills the rest of the tab: each question as it is asked (long-context questions show their first and last lines), the model's thinking and answer as they stream, and whether the answer was graded right, with the answer the grader read. The feed keeps roughly the last 200,000 characters.

### Measuring agentic coding

Set *Measure* to `agent` to see how well a model works as a coding agent. This is a separate measurement from intelligence. There are ten tasks, each a real merged bug-fix pull request from a small open-source project with no dependencies: seven in Python (Python-Markdown, more-itertools, tomli, simplejson, idna) and three in Rust (semver, humantime, bstr). For each task:

1. The repository is checked out at the commit before the fix, and only that commit, so the fix is not in its history.
2. The model gets a description of the bug, written like an issue, and the command that runs the tests. It works with the same tools and Seatbelt sandbox as the Agents tab, with the network off. It has up to 50 replies and 30 minutes. It is told that only that command decides the task, since other tests in a project's suite may fail for unrelated reasons (a missing optional dependency). Python checkouts get their package's entry points (from `pyproject.toml`, in a `.dist-info` folder git ignores), as if installed, so plugin lookups work: without them Python-Markdown's full suite fails about 390 tests. With thinking on, only the newest reply's thinking is sent back to the model. Older thinking is dropped from the prompt, because chat templates such as Qwen's would otherwise keep all of it and fill the context within a dozen replies. The saved transcript keeps all of it.
3. Afterwards the tests from the actual pull request are added (for the Rust tasks whose tests sit inside the source, a test written from them) and run in the sandbox. The task is solved if they pass. They include the project's existing tests in the same file, so a fix that breaks something else fails.

Each task was checked to fail before its fix and pass after it.

Select *Tasks* and press Enter to install them: this clones each repository at its commit with `git`, and for Rust runs `cargo fetch` so the tests can run offline. They go into `~/.config/ferrum/agent` (or `$FERRUM_AGENT_DIR`). Pressing Enter twice removes them. The tasks need `git`, `python3` and `cargo`. *Thinking* and *Drafter* work as for intelligence. The agent runs with a 64K context, or 32K when that does not fit.

The result is the number of tasks solved, plus what they cost: tokens generated (thinking included), prompt tokens read (not counting ones reused from the cache), replies and time, per task and in total. **Generated tokens per solved task** is the efficiency figure. Two models that solve the same number of tasks are told apart by it. Each task's transcript is saved in `~/.config/ferrum/agent-runs/<run>/<task>.json`: the whole conversation (replies, thinking, tool calls and their output), the agent's changes as a diff, and the test output. While the run goes on, the task tiles shrink to one line each and the rest of the tab shows a live feed of the current task: the model's thinking (dimmed behind a `┊`), its replies as they stream, and each tool call with whether it worked and the first lines of its output. On the results screen, `↑`/`↓` pick a finished task and Enter opens its transcript. That is where to look for why a task failed: a loop, a wrong guess, or a fix that was nearly right. Complete runs are saved in `~/.config/ferrum/agent.json`, the newest per model and setup. The setup and results screens list every saved run, most tasks solved first, then fewest generated tokens. Press `c` on either screen for **Compare**. It shows one column per run with the tasks it solved, and under it the model, thinking and drafter, tokens per solved task and time. Below the columns is a grid of ✓/✗ per task, lined up with them, which shows exactly which tasks separate two models. It uses the full width on wide screens. `←`/`→` pick a run, `t` filters by thinking, and pressing `x` twice deletes every saved agent run. A full run takes from minutes to several hours, depending on the model.
