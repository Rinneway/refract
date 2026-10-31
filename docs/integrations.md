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
from refract import Refract, AgentConfig, ablation
from refract.render import insight_markdown, trace_markdown

rf = Refract.from_dir("examples/demo")              # settings from .env / refract.toml

# an event from the store, or any text
run = rf.analyze("demo_grain_farm", "ev_wheat_en", role="cfo", language="en")
event = rf.news_from_text("Порты Азовского моря закрыты из-за шторма", "…", publisher="Пресс-служба порта")
run2 = rf.analyze("demo_grain_farm", event, language="ru")

print(insight_markdown(run.insight, rf.settings.institution))
print(trace_markdown(run.trace))

# follow-up
reply = rf.chat(run.insight.insight_id, "How exposed are we to freight costs?").response

# components off, e.g. for a quick draft without the review
quick = rf.analyze("demo_grain_farm", "ev_wheat_en", config=ablation("no_verifier"))

# persistence
path = rf.save_run(run, "runs/")
again = rf.load_run(path)
```

Lower level: `Agent(llm, store, config=AgentConfig(...), settings=..., policy=..., impact_policy=..., on_progress=fn)`, `ChatAgent`, `DataStore`, `LLMClient`, `ScriptedLLM`.

## CLI

```text
refract demo [rate|injection|dollar]           scripted run without a key
refract profiles | events [--data DIR]
refract analyze --profile ID (--event ID | --title T --text X | --text-file F)
                [--role R] [--lang ru|en] [--ablation NAME] [--iterations N] [--out DIR] [--json]
refract chat --run FILE --q "…" [--q "…"]
refract theses --docs DIR --out FILE           extract verified theses from research
refract news fetch --feeds FILE --data DIR     RSS / Atom → events
refract eval pairs|run|judge|report …          experiments, see evaluation.md
refract serve [--host H] [--port P]            REST API
refract mcp                                    MCP server over stdio
refract check [--models main,verifier]         endpoint ready? structured output, context window, reasoning
refract models                                 configured models and what the endpoints offer
refract schemas --out DIR                      JSON Schemas of the contracts
```

## REST API

Open `/docs` for the OpenAPI UI.

| Method | Path | Body / result |
| --- | --- | --- |
| GET | `/health` | `{status, version, language, llm_configured}` |
| GET | `/profiles` · `/profiles/{id}` | list · `Profile` |
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

With `REFRACT_API_TOKEN` set, every endpoint except `/health` needs `Authorization: Bearer <token>` (EventSource cannot send headers — put the API behind a proxy that adds it, or keep the stream endpoint on a private network). Insights live in memory; restarting the API forgets them (save runs with the Python API if you need them later).

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

With uv instead of a global install: `"command": "uv", "args": ["--directory", "/path/to/refract", "run", "refract", "mcp"]`.

| Tool | Arguments | Returns |
| --- | --- | --- |
| `list_profiles` | — | ids, kinds and summaries |
| `get_profile` | `profile_id` | the profile as JSON |
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

`refract news fetch --feeds feeds.json --data my_data` writes `my_data/events/rss_<timestamp>.json`. Articles with similar titles within 48 hours become one event with several articles. Only titles, links, dates and the feeds' own summaries are stored; if you have the right to use full texts, put them into `NewsArticle.text` yourself. The URLs in `examples/feeds.example.json` are examples — check them before relying on them.

## Your own data

1. Write profiles as `profiles/<id>.json` (`refract schemas` gives the JSON Schema; `examples/demo/profiles/` has six examples of all kinds).
2. Put research documents into `knowledge/docs/` and run `refract theses --docs knowledge/docs --out knowledge/theses.json`; review the result — only theses whose quote is verbatim on the page are kept.
3. Add events from feeds, from your own pipeline (`NewsEvent` JSON) or as text through any interface.
4. Set `[institution]` and, if needed, your own `[policies]`.
