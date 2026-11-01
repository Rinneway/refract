# Configuration

Settings are read in this order, later sources win: built-in defaults → `refract.toml` (or the file in `REFRACT_CONFIG`) → environment variables (`.env` is loaded first). Start from [`refract.example.toml`](../refract.example.toml) and [`.env.example`](../.env.example).

## Providers

Any endpoint that speaks the OpenAI Chat Completions API works.

| Provider | `LLM_BASE_URL` | `LLM_API_KEY` | Notes |
| --- | --- | --- | --- |
| OpenAI | `https://api.openai.com/v1` | your key | native structured output |
| OpenRouter | `https://openrouter.ai/api/v1` | your key | model ids like `deepseek/deepseek-chat`; cost is read from `usage.cost` when present |
| LiteLLM proxy | your proxy URL | proxy key | cost is read from the `x-litellm-response-cost` header |
| vLLM | `http://localhost:8000/v1` | any string | |
| Ollama (local) | `http://localhost:11434/v1` | any string, e.g. `ollama` | set `OLLAMA_CONTEXT_LENGTH=16384`; ready config: [`examples/config/ollama-local.toml`](../examples/config/ollama-local.toml) |
| Ollama Cloud | `https://ollama.com/v1` | a key from ollama.com → Settings → Keys, in `OLLAMA_API_KEY` | billed in usage credits since Aug 2026: the Free plan covers a few starter models, others answer HTTP 402 until credits are added; one request at a time on Free; config: [`examples/config/ollama-cloud.toml`](../examples/config/ollama-cloud.toml) reads the key from `OLLAMA_API_KEY` (`[llm].api_key_env`), so local and cloud configs share one `.env` |
| LM Studio | `http://localhost:1234/v1` | any string | |

Several URLs separated by commas form a failover chain: on a network error or HTTP 5xx the next URL is tried. HTTP 400/422 on a structured-output request makes the client retry in JSON mode and then with the JSON Schema written into the system prompt; the working mode is remembered per model.

`refract models` prints the configured aliases and the models each endpoint offers. `refract check` sends three short requests per model and reports whether structured output works (and in which mode), whether the context window holds an agent-sized prompt (~9 000 tokens), and whether the model burns hidden reasoning tokens.

### Local models

- **Context window.** Ollama truncates prompts longer than its `num_ctx` without an error; the agent then works on the end of the prompt only. Set `OLLAMA_CONTEXT_LENGTH` (16384 is enough) and restart Ollama. The client also warns in the trace when the provider reports far fewer prompt tokens than the prompt can hold.
- **Reasoning.** Thinking models spend minutes per step on hidden reasoning; `extra_body = { reasoning_effort = "none" }` switches it off on Ollama. An answer cut at `max_tokens` is retried once with twice the budget.
- **Timeouts.** `[llm].timeout_scale` (or `REFRACT_TIMEOUT_SCALE`) multiplies every step timeout; 3–5 suits a consumer GPU.
- **Concurrency.** One local server serves one request well; run experiments with `--workers 1`.

## Models and routing

Models are referred to by alias. `main` and `verifier` always exist: from `[models.*]` in the TOML file, from `REFRACT_MODEL` / `REFRACT_VERIFIER_MODEL`, or the default `gpt-4o-mini`; without a verifier, `main` checks its own drafts (it works, but a model of another family catches more).

```toml
[models.main]
id = "z-ai/glm-5.3-flash"
extra_body = { reasoning = { enabled = false } }   # passed to the provider as is
price_in = 0.10    # example price per 1M input tokens; used only if the provider reports no cost
price_out = 0.40

[routes.verify]
model = "verifier"
fallback = "main"   # used if the first model gives no schema-valid answer
max_tokens = 2500
timeout_s = 60
temperature = 0.0
```

Steps and their defaults:

| Route | Model | Fallback | max_tokens | timeout, s |
| --- | --- | --- | --- | --- |
| `classify_news` | main | — | 900 | 30 |
| `plan` | main | — | 1200 | 30 |
| `draft_insight` | main | verifier | 4000 | 90 |
| `verify` | verifier | main | 2500 | 60 |
| `revise` | main | verifier | 4000 | 90 |
| `headline` | main | verifier | 300 | 20 |
| `chat_plan` | main | — | 800 | 25 |
| `chat` | main | verifier | 2000 | 60 |
| `extract_theses` | main | — | 2500 | 90 |
| `judge` | verifier | — | 2500 | 120 |

A failed plan falls back to the default plan, so the plan route has no fallback model by design: it keeps latency predictable.

## Cache, log and cost

- `LLM_CACHE_DIR` (default `.cache/llm`) — answers by SHA-256 of the request; a repeated run is free and identical.
- `LLM_LOG_PATH` (default `logs/llm_calls.jsonl`) — one line per call: step, model, endpoint, tokens, reasoning tokens, cost, currency, latency, cache flag, error.
- `[llm].currency` — the currency shown in traces and reports. Costs come from the provider or the price table.

## Language

`REFRACT_LANGUAGE` or `language` in the TOML file sets the default; every interface accepts a per-run language (`--lang`, `"language"` in the API, the `language` tool argument). Prompts, labels, rendered context, disclaimers and trace notes follow it.

## Institution

The institution is whoever publishes the house view and may offer products — a bank, a consultancy, an accounting firm, or simply “our research desk”.

```toml
[institution]
name = "Demo Bank"
name_ru = "Демо-банк"
name_gen = "Демо-банка"   # Russian genitive, used in «позиция Демо-банка»
```

## Policies

```toml
[actions]
offer_products = false        # keep the advice, never suggest a product

[policies]
roles = "my_policies/roles.json"
actions = "my_policies/actions.json"
```

The files have the format of `src/refract/policies/*.json`. Role keys must be values of `ReaderRole`; rule conditions understand `factors` (with `prefix:*` wildcards), `event_types`, `subject_kinds`, `has` / `has_not` (`products.<name>`, `foreign_trade.active`, `foreign_trade.countries`, `tax_regime`), `via: counterparty` with `min_share`, and conditions for the model: `direction_for_subject`, `event_move`, `note`.

## Retrieval

`REFRACT_EMBEDDING_MODEL` (or `[retrieval].embedding_model`) turns research search into BM25 + dense retrieval fused by reciprocal rank fusion. Embeddings come from the same endpoint and are cached in `<cache_dir>/embeddings/`.

## Impact thresholds

`ImpactPolicy` (`agent/impact.py`) holds the thresholds of the impact ceiling; pass your own to `Refract(..., impact_policy=ImpactPolicy(...))` or `Agent(...)` to run sensitivity experiments.
