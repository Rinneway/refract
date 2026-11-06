# Command line

`refract` works the same on Windows, macOS and Linux. Install it in one of three ways:

- **uv:** `uv sync --all-extras`, then `uv run refract …`.
- **pip:** `pip install "refract-agent[api,mcp,pdf] @ git+https://github.com/Rinneway/refract"`, then `refract …`.
- **Docker:** `docker run --rm -it refract refract …`; see [deployment.md](deployment.md).

Every command prints help with `--help`.

- [Without a model](#without-a-model)
- [The interactive console: `refract shell`](#the-interactive-console-refract-shell)
- [Commands](#commands)
- [Environment variables](#environment-variables)
- [Exit codes and output](#exit-codes-and-output)
- [Windows, macOS, Linux](#windows-macos-linux)

## Without a model

There are two ways to run the agent with no key and no model:

| | `refract demo [rate\|injection\|dollar]` | `--demo` on `analyze`, `feed`, `brief`, `shell`, `serve`, `mcp` |
| --- | --- | --- |
| Model answers | written by hand to show one check at work | recorded from real runs: qwen3.5:9b drafts and revises, gemma4:12b verifies (local Ollama) |
| Pairs | three scenarios | the eight recorded pairs, plus the scenarios (`refract demo --list`) |
| Code | tools, retrieval, rule checks, the impact ceiling, the action policy, pruning and the trace run as the current code in both modes; a code change shows up in the replayed insight | |
| Trace | the models are named `scripted` | the models are named `recorded:<model>` and the replayed recording is in `trace.config.replay`; tokens and cost are 0 |

In demo mode, three things need a live model and give a clear message instead:

- a pair without a recording (the message lists the pairs that have one);
- a pasted news text;
- follow-up questions (`chat`, `ask`).

A recording is made for one reader role and language. A run asked for another role or language is shown as recorded, with a note saying so. `--demo` always uses the packaged demo data, so `--data` is ignored.

To make your own recordings, see [evaluation.md](evaluation.md#rerunning-with-local-models): `refract demo --record FILE` makes them, and `REFRACT_RECORDINGS=FILE` replays them.

## The interactive console: `refract shell`

```bash
refract shell                          # with the model from refract.toml / .env
refract shell --demo                   # no model
refract shell --profile demo_coffee_ip --lang en
refract shell --demo --script session.txt     # commands from a file, then exit
```

A session:

```text
refract[—]> profiles
   1  demo_building_materials       company          ООО «Демо-Стройснаб» (вымышленная компания)
   2  demo_coffee_ip                sole_proprietor  ИП — кофейня «Демо-кофе» (вымышленный)
   …
refract[—]> use 2
refract[demo_coffee_ip]> feed
   1  direct     2026-10-20 Правительство предложило снизить порог доходов для НДС на упрощёнке
        сильная связь — Налоги и сборы: …
refract[demo_coffee_ip]> analyze 1
  … Профиль
  … Планирую анализ
  …
## Порог НДС на упрощёнке снижается: …
refract[demo_coffee_ip]> А что будет с ценами в кофейне?
refract[demo_coffee_ip]> trace
refract[demo_coffee_ip]> save runs
```

| Command | What it does |
| --- | --- |
| `profiles` | the subjects in the data folder, numbered; `*` marks the chosen one |
| `use <n\|id>` | choose the subject (a number from `profiles` or an id) |
| `profile` | the chosen subject, its reader role and language |
| `news` (`events`) | every stored news item, newest first, numbered |
| `feed [n]` | news ranked for the subject by code, strongest links first, with the reason; default 10 |
| `analyze <n\|id>` | explain a news item for the subject; the agent's steps appear as they run |
| `paste` | explain a text you paste: the title on one line, then the text, then an empty line |
| `brief [n]` | the n strongest direct news explained in one short brief; default 3 |
| `ask <question>` | a follow-up question about the last insight; the conversation is kept until the next analysis |
| any text | after an insight, a line of three or more words, or one ending in `?`, is a question about it |
| `sources` | every statement of the last insight: id, origin (fact, AI inference, house view, hypothesis), sources |
| `trace` | the trace table: steps, models and tools, time, cost, verdicts, notes |
| `save [folder]` | save the last run (default `runs/`); `refract chat --run` and `load` read it |
| `load <file>` | load a saved run: its subject becomes the chosen one, and you can ask about it |
| `role <role>` | reader role: `owner`, `ceo`, `cfo`, `chief_accountant`, `treasurer`, `entrepreneur`, `individual`, `other`; default by the subject's kind |
| `lang <ru\|en>` | language of the insights and of the console |
| `status` | data folder, model, chosen subject, last insight |
| `clear` | clear the screen |
| `help` | the commands |
| `exit`, `quit` | leave; also Ctrl+D (macOS, Linux) or Ctrl+Z then Enter (Windows) |

- Numbers refer to the last list shown. Subjects always keep their `profiles` numbers.
- In demo mode, the news that has a recording for the chosen subject is marked `▶ demo`.
- **Ctrl+C** stops the running analysis and returns to the prompt; at the prompt it starts a fresh line.
- **Editing and history.** On macOS and Linux, Python's `readline` gives line editing, Tab completion of commands, and history kept in `~/.refract_history`. The Windows console has its own editing: arrows and F7 for the history of the session.
- **Colours.** Colours are used only on a terminal that shows them. On Windows 10 and later, the console's VT mode is switched on. `NO_COLOR=1` or `--no-color` turns them off. Scripts and pipes get plain text.
- **Errors.** A mistake (an unknown command, a wrong number, a missing file, a model error) is a message, and the session goes on.
- **Scripts.** `--script FILE` reads one command per line; `#` starts a comment, and each command is echoed as `> command`. Use `--script -` to read from stdin. The exit code is 1 if any command failed, so a script works as a test: CI runs [`tests/fixtures/shell_script.txt`](../tests/fixtures/shell_script.txt) on all three systems.

## Commands

Global options come before the command: `refract --config FILE <command>` reads that TOML file instead of `./refract.toml`, and `refract --version` prints the version.

### Analysis

**`refract analyze`** runs the agent for one subject and one news item. It prints the insight as Markdown, then the trace.

```text
refract analyze --profile ID (--event ID | --title T --text X | --title T --text-file F)
                [--publisher P] [--url U] [--role R] [--lang ru|en]
                [--ablation NAME] [--iterations N] [--out DIR] [--json] [--quiet] [--data DIR] [--demo]
```

| Option | Meaning |
| --- | --- |
| `--profile` | subject id (`refract profiles`) |
| `--event` | a stored event id (`refract events`) |
| `--title`, `--text` / `--text-file` | a news text instead of a stored event; the file may be UTF-8 or Windows-1251 |
| `--publisher`, `--url` | where the pasted text came from (shown as its source) |
| `--role` | reader role; default `cfo` |
| `--lang` | language of the insight; default from the config (`ru`) |
| `--ablation` | switch a component off: `no_plan`, `no_verifier`, `no_rules`, `no_revise`, `no_self_check`, `no_cap`, `no_policy`, `no_attribution`, `no_reconcile` |
| `--iterations` | revise rounds (default 1) |
| `--out DIR` | save the insight, trace and state as `DIR/<insight_id>.json` |
| `--json` | print the insight as JSON instead of Markdown |
| `--quiet` | no step labels on stderr |

**`refract chat --run FILE --q "…" [--q "…"]`** asks follow-up questions about a saved run. Repeat `--q` for a conversation: each question sees the earlier answers.

**`refract feed --profile ID`** ranks the stored news for one subject by code.

```text
refract feed --profile ID [--top N] [--since YYYY-MM-DD] [--all] [--no-tag]
             [--analyze K [--role R] [--out DIR]] [--lang ru|en] [--json] [--data DIR] [--demo]
```

- Each item is `direct` (at least one moderate link), `background` (only weak links) or, with `--all`, unrelated.
- News without factors is tagged once by the model, unless you pass `--no-tag` or run in demo mode.
- `--analyze K` explains the K strongest direct items.

**`refract brief --profile ID [--top 3] [--since …] [--role R] [--out DIR] [--lang …] [--json]`** explains the strongest direct news of the feed, each in one paragraph: the headline, impact, the first "why it matters" statement and the first action. Each item is one agent run.

### Data

| Command | What it does |
| --- | --- |
| `refract profiles [--data DIR]` | the subjects: id, kind, name |
| `refract events [--data DIR]` | the stored news: id, date, language, title |
| `refract news fetch --feeds FILE --data DIR [--limit N]` | collect RSS / Atom feeds into deduplicated events; a repeated fetch adds only what is new |
| `refract theses --docs DIR --out FILE [--workers N]` | extract the institution's theses from Markdown, text or PDF research; only theses whose quote is found verbatim are kept (uses the model) |

Data folders are described in [integrations.md](integrations.md#your-own-data) and [data-model.md](data-model.md). A file that does not load is skipped with a message on stderr naming it.

### Interfaces for other programs

| Command | What it does |
| --- | --- |
| `refract serve [--host 127.0.0.1] [--port 8000] [--data DIR] [--demo]` | REST API with OpenAPI docs at `/docs` (extra `api`) |
| `refract mcp [--data DIR] [--demo]` | MCP server over stdio for Claude Desktop, Cursor and other clients (extra `mcp`) |
| `refract schemas --out DIR` | JSON Schemas of the contracts (insight, trace, feed, brief, …) |

Endpoints and tools: [integrations.md](integrations.md).

### Models

| Command | What it does |
| --- | --- |
| `refract models` | the configured aliases, their ids, and what each endpoint lists |
| `refract check [--models main,verifier]` | for each model: does it answer, structured output, context window, hidden reasoning, embeddings; with hints for the usual mistakes |

### Demo and recordings

| Command | What it does |
| --- | --- |
| `refract demo [rate\|injection\|dollar] [--json]` | a scripted run (see above) |
| `refract demo --list` | the pairs `--demo` can replay: subject, event, role, language, recorded or scripted |
| `refract demo --record FILE [--pairs FILE] [--data DIR]` | run the configured models on the recorded pairs (or on a pairs file) and write a new recordings file |

### Evaluation

```text
refract eval pairs     --out FILE [--languages ru,en] [--sample N --seed S] [--data DIR]
refract eval run       --out DIR [--pairs FILE] [--systems …] [--workers N] [--force]
refract eval judge     --out DIR [--judges a,b] [--systems …] [--workers N] [--force]
refract eval label     export|import --out DIR [--name human] [--sample N --seed S] [--lang ru|en] [--file F]
refract eval relevance export|import --out DIR [--lang ru|en] [--file F] [--force]
refract eval report    --out DIR [--reference full] [--judges …]
```

What each step does, the systems, the metrics and the statistics: [evaluation.md](evaluation.md).

## Environment variables

Environment variables override `refract.toml`. A `.env` file in the current folder is read first. Every setting: [configuration.md](configuration.md).

| Variable | Meaning |
| --- | --- |
| `LLM_BASE_URL` (`LLM_BASE_URLS`, comma-separated) | OpenAI-compatible endpoint(s), e.g. `http://localhost:11434/v1` |
| `LLM_API_KEY` (or `OPENAI_API_KEY`) | the key; any non-empty value for local Ollama or vLLM |
| `REFRACT_MODEL`, `REFRACT_VERIFIER_MODEL` | model ids of the `main` and `verifier` aliases |
| `REFRACT_EMBEDDING_MODEL` | an embedding model for search by meaning (e.g. `bge-m3`) |
| `REFRACT_LANGUAGE` | default language, `ru` or `en` |
| `REFRACT_CONFIG` | path of the TOML config (default `./refract.toml`) |
| `REFRACT_DATA_DIR` | data folder (default `examples/demo`, or the packaged demo data after `pip install`) |
| `REFRACT_TIMEOUT_SCALE`, `LLM_TIMEOUT_S` | multiplier of step timeouts (3–5 for local models); request timeout |
| `LLM_CACHE_DIR`, `LLM_LOG_PATH` | answer cache (default `.cache/llm`); call log (default `logs/llm_calls.jsonl`) |
| `REFRACT_API_TOKEN` | for `serve`: require `Authorization: Bearer <token>` |
| `REFRACT_RECORDINGS` | for `--demo`: replay this recordings file instead of the packaged one |
| `NO_COLOR` | no colours in `refract shell` |

## Exit codes and output

| Code | Meaning |
| --- | --- |
| 0 | done |
| 1 | the agent gave no insight (`analyze`); a pair failed (`demo --record`); a check failed (`check`); a command of a `shell --script` failed; a labels sheet had bad rows |
| 2 | a usage error: an unknown subject, event or file, a broken config or data file, something demo mode cannot do; the message on stderr says which |

- The result goes to stdout: Markdown, or JSON with `--json`.
- Progress, notes and messages go to stderr. So `refract analyze … --json > insight.json` gives a clean file, and `refract … | head` works.

## Windows, macOS, Linux

- **Paths.** On Windows, write paths with `\` or `/`; both work: `--out runs\exp1`, `--data C:\data\refract`.
- **Text encoding.**
  - **Output.** Russian text is written as UTF-8 to pipes and files on every system, including `refract analyze … > out.md` in PowerShell or cmd (Windows code pages cannot hold Cyrillic, so output is not left to them). The Windows console shows Cyrillic directly.
  - **Input.** Data files may be UTF-8 with or without a BOM; research texts and `--text-file` may also be Windows-1251.
- **Shells.**
  - **Environment variables.** In PowerShell: `$env:REFRACT_LANGUAGE = "en"`. In cmd: `set REFRACT_LANGUAGE=en`. In bash or zsh: `export REFRACT_LANGUAGE=en`, or put the variable before the command.
  - **Quoting.** Quote a `--text` that has spaces in every shell. In PowerShell, also quote anything with braces, like `"HEAD^{tree}"` in git.
- **Console.**
  - **Line editing.** macOS and Linux have readline (line editing, Tab, a history file). On Windows, the console's own editing works, and no extra package is needed.
  - **Exit.** Ctrl+D on macOS and Linux; Ctrl+Z then Enter on Windows. `exit` works everywhere.
- **Python.** 3.10 to 3.13. CI tests Linux on all four versions, and Windows and macOS on 3.10 and 3.13.
