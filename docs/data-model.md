# Data model

All contracts are Pydantic v2 models in `src/refract/schemas/` with `extra="forbid"`. JSON Schemas are in [`docs/schemas/`](schemas/) (`refract schemas --out docs/schemas` regenerates them).

## A data folder

```text
my_data/
  profiles/*.json          Profile — one object or a list per file
  events/*.json            NewsEvent
  knowledge/theses*.json   Thesis — the institution's verified positions
  knowledge/docs/          research documents: .md, .txt, .pdf
```

`examples/demo/` is a complete example. Convert your own sources (a CRM, a client base, a form) into `Profile` objects once; the agent does not care where they came from.

## Profile

| Field | Notes |
| --- | --- |
| `profile_id`, `kind` | `company`, `sole_proprietor`, `self_employed`, `individual` |
| `name`, `country`, `region`, `city` | country as ISO 3166-1 alpha-2 |
| `industry` | `code`, `name`, `section`, `scheme` (`nace` / `okved`); may be empty for individuals |
| `tax_regime` | free text, e.g. “simplified 6%”, “НПД” |
| `scale` | `revenue`, `group_revenue`, `currency`, `segment` (`large` … `micro`, `household`), `employees`, `age_years` |
| `industry_position` | revenue percentile among peers in your sample, with the method |
| `products` | flags of products held with the institution: `current_account`, `loans`, `mortgage`, `leasing`, `deposits`, `payroll`, `factoring`, `guarantees`, `acquiring`, `fx_account`, `investments` |
| `foreign_trade` | `active`, `countries`, `direction` |
| `group_id`, `group_members`, `affiliations` | confirmed group members and hypothetical links through shared persons |
| `counterparties` | payment links: `role` (supplier / customer), `amount`, `share_of_flow`, the counterparty's `industry` and `region` |
| `exposures` | sensitivities to factors: `factor`, `direction`, `strength` 0–1, `via` (own / counterparty / group), `reason` |
| `user_provided` | what the reader said about themselves — rendered as untrusted text |
| `data_quality`, `summary` | limitations shown to the model; a 2–6 sentence description |

The agent turns a profile into keyed features — `kind`, `industry`, `scale.segment`, `region`, `tax_regime`, `products.<name>`, `products.absent`, `foreign_trade`, `group`, `counterparties.<role>.<id>`, `affiliations.<id>`, every exposure factor, `user.<key>` — and statements must cite these keys.

## News event

`event_id`, `title`, `summary`, `published_at`, `articles` (one or more: `article_id`, `url`, `publisher`, `title`, `text`, `language`), `key_facts`, and the tags: `event_type`, `macro_factors` (≤ 3, the main one first), `industry_codes` (two-digit divisions the news names directly), `industry_sections`, `countries` (the countries the news is about, the domestic one included: national factors link only to subjects of these countries; empty — unknown), `regions`, `commodities`, `importance`, `language`. Events without factors are tagged by the agent on first use.

## Factor vocabulary

`schemas/vocab.py`: `policy_rate`, `inflation`, `fx:USD`, `fx:EUR`, `fx:CNY`, `oil`, `gas`, `metals`, `coal`, `agri`, `gold`, `energy_prices`, `consumer_demand`, `construction`, `credit_market`, `bonds`, `equities`, `labor_market`, `taxes`, `tariffs`, `sanctions`, `logistics`, `budget`, `regulation`, `sme_support`, `gdp`, plus prefixed factors `sector:<NACE division>`, `region:<name>`, `country:<ISO code>`, `commodity:<name>`, `fx:<ISO 4217>`. Events, exposures and theses share it — matching factors is what links them.

NACE Rev. 2 and OKVED 2 share sections (A–U) and two-digit divisions, so profiles may use either scheme.

## Research

- `KnowledgeChunk` — a fragment of a document (≈ 800 characters, never across a page or a Markdown `##` section) with `doc_title`, `doc_date`, `page`, `language`, `countries` (from the document's front matter).
- `Thesis` — a position of the institution: `topic`, `statement`, `figures`, `horizon`, `factors`, the exact `quote`, `doc_title`, `doc_date`, `page`, `countries`, `is_stale`. `countries` (ISO codes) ties a thesis to economies — a central bank's rate, a national tax rule; empty means it applies everywhere. Subjects from other countries never see it. Theses older than a year are ranked last and are never used to restore a removed house view.

## Insight

`InsightContent` is what the model writes; `Insight` adds identifiers, `language`, `role`, `sources`, `models`, `trace_id`, `disclaimer` and `validated`.

| Block | Type | Ids |
| --- | --- | --- |
| `headline` | ≤ 160 characters | `headline` in review findings |
| `what_happened` | `Claim[]`, ≥ 1, facts | `c1…` |
| `why_it_matters` | `Claim[]`, ≥ 1, with `features` | `c…` |
| `risks`, `opportunities` | `Effect[]`: a claim with `impact`, `horizon`, optional `estimate` | `r…`, `o…` |
| `scenarios` | name, description, plausibility, triggers, sources | — |
| `house_view` | `Claim[]` with origin `house_view` | `b…` |
| `actions` | `Action[]`: `text`, `kind`, `rationale`, `product`, `claim_ids` | `a…` |
| `overall_impact`, `overall_direction` | low / medium / high; negative / positive / mixed / unclear | — |

`violations(insight)` lists the hard grounding rules; `Insight` refuses to be created while any is violated (except in the `no_rules` ablation).

## Trace

`Trace` holds the run's configuration (agent switches, language, prompt versions), the currency, and a list of `TraceStep`: kind (`plan`, `tool_call`, `llm_call`, `draft`, `verify`, `revise`, `finalize`), name, model or tool, input and output summaries, tokens, cost, latency, cache flag, source ids, verdict and findings. `totals` sums LLM calls, tool calls, tokens, cost, latency and revision rounds.
