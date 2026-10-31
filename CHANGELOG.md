# Changelog

## Unreleased

### Local models

- `refract check`: structured output, context window and hidden reasoning of every configured model, with hints for the usual mistakes (a placeholder key on a remote endpoint, a missing `ollama pull`, an environment variable overriding the config).
- Ready configs for Ollama, local and cloud (`examples/config/`).
- LLM client: a warning when the provider truncates the prompt; a retry with a larger budget when an answer is cut at `max_tokens`; `timeout_scale` for slow local models.

## 0.1.0 — 2026-10-31

First public release: the agent as a standalone project, re-implemented on synthetic data.

- Agent: tagging, planning, read-only tools, drafting, rule checks, second-model review, revision, finalisation; follow-up chat; progress events.
- Subjects: companies, sole proprietors, self-employed people and private individuals; reader roles including entrepreneur and individual.
- Russian and English: prompts, labels, context and disclaimers by run language.
- International vocabulary: NACE Rev. 2 / OKVED 2 divisions, ISO country codes with Russian and English aliases, ISO currencies.
- Impact ceiling with configurable thresholds; action policy in JSON with products that can be switched off.
- Number check ignores digits inside identifiers and checks small percentages.
- Language check: a warning when the insight is not in the run language; the house view is restored only from theses in that language.
- Provider-agnostic LLM client: failover, structured-output fallback, cost from provider or price table, disk cache, JSONL log.
- Knowledge: Markdown / text / PDF ingestion, BM25 and hybrid retrieval, thesis extraction with quote verification.
- Interfaces: Python API, CLI, REST API with SSE, MCP server, RSS/Atom collector, scripted demo.
- Evaluation: ablations, two baselines, LLM judges, bootstrap CIs, sign tests, inter-judge agreement, human-label metrics, results by output language and a ru/en parity check (`examples/demo/pairs_parity.jsonl`).
- Synthetic bilingual demo data.
