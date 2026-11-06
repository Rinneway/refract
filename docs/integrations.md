# Integrations

| Interface | Install | Start |
| --- | --- | --- |
| Python API | `pip install refract-agent` | `from refract import Refract` |
| CLI | same | `refract --help` |
| REST API + SSE | `pip install "refract-agent[api]"` | `refract serve --port 8000` |
| MCP server | `pip install "refract-agent[mcp]"` | `refract mcp` |
| PDF research | `pip install "refract-agent[pdf]"` | put PDFs into `knowledge/docs/` |
| Docker | — | `docker build -t refract . && docker run -p 8000:8000 --env-file .env refract` |

Until the package is published, install from the repository: `pip install "refract-agent[api,mcp,pdf] @ git+https://github.com/Rinneway/refract"`, or `uv sync --all-extras` in a clone.

## Python

```python
from datetime import datetime

from refract import Refract, AgentConfig, ablation
from refract.render import insight_markdown, trace_markdown

rf = Refract.from_dir("examples/demo")  # settings from .env / refract.toml

# an event from the store, or any text
run = rf.analyze("demo_grain_farm", "ev_wheat_en", role="cfo", language="en")
event = rf.news_from_text("Порты Азовского моря закрыты из-за шторма", "…", publisher="Пресс-служба порта")
run2 = rf.analyze("demo_grain_farm", event, language="ru")

print(insight_markdown(run.insight, rf.settings.institution))
print(trace_markdown(run.trace))

# follow-up
reply = rf.chat(run.insight.insight_id, "How exposed are we to freight costs?").response

# which stored news concern the subject, strongest links first (code; untagged news is tagged once by the model)
feed = rf.feed("demo_grain_farm", top=5, since=datetime(2026, 10, 1))
for item in feed.items:
    print(item.relevance, item.impact_ceiling.value, item.title, item.links[0])

# the strongest news, each explained by the agent and cut to a paragraph by code (one agent run per item)
brief = rf.brief("demo_grain_farm", top=3)

# components off, e.g. for a quick draft without the review
quick = rf.analyze("demo_grain_farm", "ev_wheat_en", config=ablation("no_verifier"))

# persistence
path = rf.save_run(run, "runs/")
again = rf.load_run(path)
```

Without a model: `Refract.demo()` replays recorded answers of real models (`refract.demo.Replay`); a pair without a recording, a pasted text or `chat` raises `DemoError`, which the REST API returns as 409 and `/health` reports `"demo": true`.

Lower level: `Agent(llm, store, config=AgentConfig(...), settings=..., policy=..., impact_policy=..., on_progress=fn)`, `ChatAgent`, `DataStore`, `LLMClient`, `ScriptedLLM`.

## CLI

The full reference — every command and option, the interactive console, environment variables, exit codes and notes for Windows, macOS and Linux — is in [cli.md](cli.md). In short:

```text
refract shell [--demo] [--profile ID] [--lang ru|en] [--script FILE]   interactive console
refract demo [rate|injection|dollar] | --list | --record FILE         without a model / new recordings
refract profiles | events                                             what is in the data folder
refract analyze --profile ID (--event ID | --title T --text X) …       one insight
refract chat --run FILE --q "…"                                       follow-up questions about a saved run
refract feed --profile ID [--analyze K]                               news ranked for the subject
refract brief --profile ID [--top 3]                                  the strongest news, a paragraph each
refract news fetch --feeds FILE --data DIR                            RSS / Atom → events
refract theses --docs DIR --out FILE                                  verified theses from research
refract eval pairs|run|judge|label|relevance|report …                 experiments, see evaluation.md
refract serve | mcp                                                   REST API · MCP server
refract check | models | schemas                                      endpoint check · models · JSON Schemas
```

`--demo` (on `analyze`, `feed`, `brief`, `shell`, `serve`, `mcp`) runs without a model; `--data DIR` points to your own data folder.

## REST API

Open `/docs` for the OpenAPI UI.

| Method | Path | Body / result |
| --- | --- | --- |
| GET | `/health` | `{status, version, language, llm_configured, demo, data_problems}` |
| GET | `/profiles` · `/profiles/{id}` | list · `Profile` |
| GET | `/profiles/{id}/feed?top=10&since=…&tag=true&all=false&language=…` | `Feed`: stored news ranked for the subject |
| POST | `/profiles/{id}/brief?top=3&since=…&role=…&language=…` | `Brief`: the strongest news explained (synchronous, one agent run per item) |
| GET | `/events?limit=50` · `/events/{id}` | list · `NewsEvent` |
| POST | `/events` | `{title, text, publisher?, url?, published_at?, language?}` → `NewsEvent` |
| POST | `/insights` | `{profile_id, event_id | news, role?, language?, ablation?}` → `{job_id}`; with `?wait=true` → `Insight` |
| GET | `/jobs/{job_id}` | `InsightProgress`: status and steps |
| GET | `/jobs/{job_id}/stream` | Server-Sent Events: `started`, `done`, `finished`, then `closed` with the `insight_id` |
| GET | `/insights/{id}` · `/trace` · `/markdown` | `Insight` · `Trace` · Markdown |
| POST | `/insights/{id}/chat` | `{question, history?}` → `ChatResponse` |

```bash
JOB=$(curl -s localhost:8000/insights -H 'content-type: application/json' \
  -d '{"profile_id":"demo_coffee_ip","event_id":"ev_usn_vat_ru","role":"entrepreneur"}' | jq -r .job_id)
curl -N localhost:8000/jobs/$JOB/stream
```

```javascript
const es = new EventSource(`/jobs/${jobId}/stream`);
es.addEventListener("started", (e) => show(JSON.parse(e.data).label));
es.addEventListener("closed", (e) => { es.close(); load(JSON.parse(e.data).insight_id); });
```

With `REFRACT_API_TOKEN` set, every endpoint except `/health` needs `Authorization: Bearer <token>` (EventSource cannot send headers — put the API behind a proxy that adds it, or keep the stream endpoint on a private network). Insights live in memory — the newest 500 runs and 1000 jobs; older ones and everything after a restart answer 404 (save runs with the Python API if you need them later).

## MCP

```json
{
  "mcpServers": {
    "refract": {
      "command": "refract",
      "args": ["mcp", "--data", "/absolute/path/to/data"],
      "env": {
        "LLM_BASE_URL": "https://openrouter.ai/api/v1",
        "LLM_API_KEY": "…",
        "REFRACT_MODEL": "…",
        "REFRACT_VERIFIER_MODEL": "…"
      }
    }
  }
}
```

Without a model: `"args": ["mcp", "--demo"]` — `analyze_news` replays recorded answers for the demo pairs (`refract demo --list`); other pairs, pasted texts and `ask_followup` return a message saying what needs a model.

With uv instead of a global install: `"command": "uv", "args": ["--directory", "/path/to/refract", "run", "refract", "mcp"]`.

| Tool | Arguments | Returns |
| --- | --- | --- |
| `list_profiles` | — | ids, kinds and summaries |
| `get_profile` | `profile_id` | the profile as JSON |
| `news_feed` | `profile_id`, `top`, `since` | the news that concern the subject, strongest links first, with the reasons |
| `news_brief` | `profile_id`, `top` | the strongest news explained by the agent, a paragraph each |
| `list_events` | `limit` | latest events |
| `analyze_news` | `profile_id`, `event_id` or `title` + `text`, `role`, `language`, `publisher`, `url` | the insight in Markdown with its `insight_id` |
| `ask_followup` | `insight_id`, `question` | the answer with origins and sources |
| `get_trace` | `insight_id` | the trace table |
| `search_research` | `query`, `limit` | research excerpts as JSON |

Works with the `mcp` SDK 1.x (`FastMCP`) and 2.x (`MCPServer`).

## News feeds

```json
[{"name": "Bank of England — news", "url": "https://www.bankofengland.co.uk/rss/news", "language": "en"}]
```

`refract news fetch --feeds feeds.json --data my_data` writes `my_data/events/rss_<timestamp>.json` with the news that is not stored yet (articles whose URL is already in `my_data/events/` are left out, so run it as often as you like). Articles with similar titles within 48 hours become one event with several articles. Feeds in any encoding the document declares (many Russian feeds are windows-1251) or only the HTTP header names are read correctly; for Atom, the article link is taken, not the comments. Only titles, links, dates and the feeds' own summaries are stored; if you have the right to use full texts, put them into `NewsArticle.text` yourself. The URLs in `examples/feeds.example.json` are examples — check them before relying on them.

## Your own data

1. Write profiles as `profiles/<id>.json` (`refract schemas` gives the JSON Schema; `examples/demo/profiles/` has six examples of all kinds).
2. Put research documents into `knowledge/docs/` and run `refract theses --docs knowledge/docs --out knowledge/theses.json`; review the result — only theses whose quote is verbatim on the page are kept.
3. Add events from feeds, from your own pipeline (`NewsEvent` JSON) or as text through any interface.
4. Set `[institution]` and, if needed, your own `[policies]`.

Files may be saved as UTF-8 with or without a byte-order mark (Windows Notepad adds one); research texts may also be in Windows-1251. A file or an object that does not load — broken JSON, a profile that fails its schema, a damaged PDF — is skipped with a note that names the file and the reason: the CLI prints the notes, `GET /health` lists them in `data_problems`, Python has them in `rf.store.problems`. A repeated id is reported too; the last object wins.
