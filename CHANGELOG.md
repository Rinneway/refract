# Changelog

## Unreleased

### After the first local run

- The demo recordings are now the answers of the current prompts (qwen3.5:9b and gemma4:12b, November 2026); the October recordings stay in `tests/fixtures` for the regression cases found in them. Results and the faults found: [docs/results/smoke-2026-11.md](docs/results/smoke-2026-11.md).
- Impact links: tax news that names a tax regime (the simplified system, professional income, the patent…) is a weak link for a subject on another regime.
- Actions: fewer than two after the checks are topped up from fired policy rules that code can decide alone, skipping what a kept action already says. In the second local run the checks often left one generic step, and both judges scored actions lowest for the agent.
- `eval run` stops when the folder's answers were made with other models or prompts (a forgotten `REFRACT_VERIFIER_MODEL` once mixed a gemma4:e4b reviewer into a gemma4:12b study, and the old answers were silently kept); `--force` redoes every answer. The report's header names the experiment's models.
- Judges are saved under the model, not the alias; a judge found under both names is counted once, so it no longer shows perfect agreement with itself.
- Rule check `arithmetic`: a rule of thumb from the sources (“every 10% rise in freight cuts the margin by about 1.5 pp”) applied to the news is recomputed; a wrong result (“a 15% jump cuts it by 1.5 pp”) is an error with the right value in the message.
- Rule check `off_topic`: a house view that cites only theses on other topics than the event is an error and is cut; the fallback that restores a removed house view skips such theses too.
- Internal counterparty codes (`cp_import_309`) never reach the reader: they are replaced by what is known — “the supplier with 14% of outgoing payments”.
- `.env` is read from the current folder, not from the folder the package was installed from.
- Fixed: Windows paths in `shell load/save`; pyright without `tomli`.

### Console, demo mode, packaging

- `refract shell`: an interactive console on Windows, macOS and Linux — subjects, feed, analysis with live steps, pasted texts, follow-up questions, sources, trace, brief, save and load. It uses only the standard library: readline where Python has it, ANSI colours only on a terminal that shows them (VT mode on the Windows console, `NO_COLOR`), and Ctrl+C stops the analysis, not the session. `--script` runs a command file.
- Demo mode without a model: `--demo` on `analyze`, `feed`, `brief`, `shell`, `serve`, `mcp`, and `Refract.demo()`, replays recorded answers of qwen3.5:9b and gemma4:12b on eight demo pairs, packaged in the wheel. Checks, the impact ceiling and the policy run as current code; the trace names the model as recorded. A pair without a recording, a pasted text or a follow-up question gives a message listing what can be replayed (409 in the API, `/health` → `demo`). `refract demo --list`.
- `refract demo --record FILE` writes fresh recordings from the configured models; `REFRACT_RECORDINGS` replays them.
- Docker: an image built from the wheel, running as a non-root user with a health check, which starts the API in demo mode; `docker-compose.yml` with Ollama, the models pulled on first start.
- CI: lint with pyright; tests on Linux (3.10–3.13), Windows and macOS; the CLI and a console script without a model; the wheel installed alone on all three systems; the image answering in demo mode.
- Russian output on a Windows pipe or file (`refract … > out.md`) no longer fails: stdout and stderr switch to UTF-8.
- Docs: [docs/cli.md](docs/cli.md), a full reference of every command, and [docs/deployment.md](docs/deployment.md).
- Removed: the web page from the previous unreleased change. The agent is used from the console, from Python, over REST and over MCP.

### Feed, brief, search

- `Refract.brief`, `refract brief`, `POST /profiles/{id}/brief`, MCP `news_brief`: the strongest direct news of the feed, each explained by the agent and cut down by code to a paragraph (headline, impact, the first reason, the first action), highest impact first.
- Search by meaning: with `REFRACT_EMBEDDING_MODEL` (e.g. `bge-m3`), research, news and theses are ranked by BM25 and embeddings fused by reciprocal rank fusion, so a Russian query finds an English item. Vectors are cached per text and model. A failing embedding model leaves search lexical and is reported (`data_problems`, `refract check`).
- Rule check: advice must fit the kind of subject — no business products (VAT, suppliers, payroll, a business account…) for a household, no staff, payroll or VAT for a self-employed person, no household steps (tax deductions, a personal budget, pension savings) for a company; such an action is an error and is cut.
- `refract eval relevance export|import`: a blind sheet of every subject × news item, scored against the links computed by code — precision, recall, κ, the share judged relevant at each strength, and every disagreement with its links. No model is called.

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
- Numbers: a percentage in a statement must be a percentage in the context (“20%” is not confirmed by the 20 of a date); a statement's number is cited from the source that contains it.
- Feature keys small models invent (`sector`, `revenue`, `loans`) are mapped to the listed ones.
- A risk or opportunity that says there is no effect (“…this is not a direct risk”) is a rule error.
- When a revision round runs anyway, the reviser also sees the reconciled wording remarks and softens the text itself.
- A reviewer's “number not in sources” stands only if the number check finds it too; “not a settled fact” is read as a remark on wording.
- Research has a country scope: `countries` on theses and excerpts (document front matter `countries: RU`); subjects from other countries do not see them. A UK company no longer gets the Russian policy-rate forecast.
- A house view that lost its source cites the thesis sharing its numbers, across languages.
- Actions: a product named only in the text follows the policy like the `product` field (`product_patterns`); `contact_manager` without a manager in the text becomes `analysis` or `operational`.
- A revision is a patch: statements and actions the review did not flag and the model dropped are kept (`checks.merge_revision`).
- Evaluation: only systems that were run are judged; failed runs and judge calls are retried instead of being cached as results; a rejected key stops `judge` with a hint; `eval` prints environment overrides of the config.
- Experiments keep a manifest (models behind the aliases, prompt versions) and warn when a later run uses other models; the report has a Models table that flags steps made with different models.
- HTTP 402 (no credits) also stops `judge`, with a hint; `ollama-local.toml` gets two local judges from other families (ministral-3:14b, phi4:14b); the Ollama Cloud config and docs describe its credit billing.
- `[llm].api_key_env`: a config can read its key from its own variable — `ollama-cloud.toml` uses `OLLAMA_API_KEY`, so one `.env` serves local and cloud runs.
- `Trace.totals` counts cached calls, the reviewer's errors and how many code lifted, statements restored after revision and pruned; the evaluation report has a Process table with them.
- Duplicate plan steps run once.
- Regression tests replaying fourteen real runs answer by answer, in Russian and English, with gemma4:e4b and gemma4:12b as reviewers (`tests/fixtures/`), including an 8-pair smoke evaluation.

### Checks, evaluation and interfaces

- New rule check: a counterparty called the largest or main (“крупнейший поставщик”, “main customer”) with the share of another counterparty is an error — gemma4:12b wrote “the largest supplier (14%)” about the importer while cement has 22%, and the reviewer caught it only sometimes.
- Impact ceiling: a product held with the institution and priced off the event's main factor (loans, mortgage, leasing, factoring, deposits and the policy rate) is a moderate link when no sensitivity covers that factor. A coffee shop with a loan used to get “no link” to a rate decision.
- A run without an insight says why: `Trace.failure_reason()`, the `error` of the progress event and of an API job, a line on stderr in the CLI; an emptied required block gets a plain note instead of a validation message.
- CLI: an unknown profile, event or ablation, a missing file and a saved run without an insight end with a message and exit code 2 instead of a traceback.
- `insight-v11`: compare a new tax with what the subject pays now; “largest” only for the counterparty with the largest share; every risk names its mechanism; every action says what, with whom and by when; no business advice to a private person. The examples in the prompts no longer use numbers from the demo data (14%, 31%), so the prompt carries no answers for evaluation pairs.
- Human labels: `refract eval label export` writes a blinded, shuffled sheet with one page per answer (context, answer, the judge's rubric); `refract eval label import` saves it as a judge named `human`. Agreement is reported for every pair of judges, annotators first.
- `refract eval pairs --sample N --seed S` keeps a reproducible random sample.
- `agent/checks.py` is split into a package by responsibility — `rules`, `review`, `attribution`, `repair`, `messages`; `checks.<name>` keeps working.
- End-to-end tests drive every interface — CLI, Python API, REST API, MCP — through the real HTTP client against a fake OpenAI-compatible server; `docs/schemas` is checked against the code.

### Bugs found by fuzzing and review

- Numbers: thousands separated by a no-break, thin or narrow no-break space (“12 600”) were split into two numbers, and the second was reported as invented. The separators in the number pattern had become ordinary spaces; they are escapes now.
- A schema-invalid answer is no longer lost when the model's repair fails too: code keeps its valid part (enum case and wording, a single item where a list is expected, lists over their maximum, invalid optional fields and items) and notes every change in the trace. On corrupted real answers of small models, failed runs fell from 87 to 3 in 600.
- Repeated statement and action ids are renumbered instead of failing the insight.
- LLM cache: atomic writes; a damaged or foreign cache file is a miss, not a crash on every later run; the call log is written under a lock.
- Chat never returns an empty answer.
- Only the newest runs (500, `keep_runs`) and API jobs (1000) stay in memory.
- Impact links: national factors (the policy rate, taxes, demand…) link only to subjects of the countries the news is about; trade matches by the three-digit group, what is traded; a counterparty matched by its section letter is a weak link whatever its share, as documented. The tagger (`classify-v5`) names the country of domestic news and, in trade, the group.
- The retelling baseline now sees the same tagged event as every other system.
- `tests/test_robustness.py` replays 150 corrupted real answers across all ablations on every test run; pyright is clean apart from optional imports.

### Personal news feed

- `Refract.feed`, `refract feed`, `GET /profiles/{id}/feed`, MCP `news_feed`: the stored news ranked for one subject by code — direct, background or unrelated — with the links as reasons; untagged news is tagged once; `--analyze K` explains the strongest items.
- Evaluation: the feed's verdict against `gold_relevant` (precision, recall, F1, κ, and the share judged relevant at each link strength).

### Data and search

- Data folders saved on Windows: files with a byte-order mark load, research texts may be in Windows-1251; a file or object that does not load is skipped with a note naming it (CLI, `GET /health` → `data_problems`), repeated ids are reported, a broken `refract.toml` is a message.
- Research: a long paragraph is cut at sentence ends instead of becoming one chunk of any size.
- Search: light stemming, so «ставка» finds «ставку» and «rate» finds «rates».
- RSS: windows-1251 feeds, Atom article links (not the comments), `content:encoded`; a repeated `news fetch` adds only what is new.
- Links read as sentences (“Construction and real estate: the largest customer is a developer…”), for readers and for the model; API jobs show step labels in the request language from the start.

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
