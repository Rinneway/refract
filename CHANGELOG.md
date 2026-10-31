# Changelog

## Unreleased

### Local models

- `refract check`: structured output, context window and hidden reasoning of every configured model, with hints for the usual mistakes (a placeholder key on a remote endpoint, a missing `ollama pull`, an environment variable overriding the config).
- Ready configs for Ollama, local and cloud (`examples/config/`).
- LLM client: a warning when the provider truncates the prompt; a retry with a larger budget when an answer is cut at `max_tokens`; `timeout_scale` for slow local models.

### Robustness to small models

Found by running qwen3.5:9b with gemma4:e4b as the reviewer through local Ollama and replaying the traces.

- Attribution step: missing `features` and `source_ids` are filled by code from shared word stems, numbers and a small bilingual concept lexicon (“НДС”, “VAT” → taxes); “Risk:” prefixes removed; each repair is noted in the trace. New ablation `no_attribution`.
- `why_it_matters` statements use `SubjectClaim`: `features` is required in the JSON schema, while a missing field is still parsed.
- A required block that review emptied comes back with the actions that relied on it; if nothing can come back, code words it from the first key fact or the strongest event–subject link.
- Structural findings of the reviewer (unknown or missing source, duplicate id, wrong language) stand only if the rule check confirms them.
- Reconciling the reviewer: an error whose own explanation calls it wording (“does not contradict, but too categorical”) or an assumption becomes a warning; the statement keeps a caveat or becomes a hypothesis instead of being cut, and no revision round is spent on it. New ablation `no_reconcile`.
- Headlines: an over-long headline (in the draft or a rewrite) is cut at a clause boundary instead of being rejected; the fallback headline is the first conclusion about the subject; a softened headline is rewritten.
- A house view repeated in “why it matters” is flagged by the rule check and removed at finalisation; an action stays while one of the statements it relies on stays.
- Prompts: `verify-v7` — overstated certainty and self-described wording remarks are warnings; `insight-v9` — “why it matters” is the model's own conclusion, a bill or a proposal is written conditionally.
- Duplicate plan steps run once.
- Regression tests replaying two real runs answer by answer (`tests/fixtures/qwen_usn_run.json`).

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
