# The agent

Input: a subject profile, a news event, a reader role and a language. Output: an `Insight` and a `Trace`; the `AgentState` lets the chat continue on the same context.

## Steps

| Step | What happens | Model | Code |
| --- | --- | --- | --- |
| Profile | the subject's profile is loaded without the model | — | `agent/tools.py` |
| Tagging | only if the event has no factors: event type, up to three factors (the main one first), NACE sections and two-digit divisions the news names directly, countries (ISO), regions, commodities, importance 0–1, language | `classify_news` route | `prompts/classify.py` |
| Plan | 3–6 tool calls with arguments and a purpose; the house view is always added; if the model fails, a default plan | `plan` | `prompts/plan.py` |
| Tools | house view theses, research search, counterparties, group, related news | — | `agent/tools.py`, `knowledge/store.py` |
| Draft | the insight by the JSON schema: what happened, why it matters, risks, opportunities, scenarios, house view, actions, impact and direction | `draft_insight` | `prompts/insight.py` |
| Attribution | small models often leave `features` and `source_ids` empty: code fills them from what the statement visibly refers to — shared word stems, numbers and concept words (“НДС”, “VAT” → `taxes`, `tax_regime`) with the feature descriptions; unknown keys are mapped to listed ones (`sector` → `industry`, `loans` → `products.loans`); a source is the one that contains the statement's numbers (a number found in two sources cites both), else the one the statement nearly retells (≥ 50% of stems), else the subject data together with the source it partly retells (≥ 25%); a cited source that lacks the statement's number gets the source that has it. “Risk:” prefixes are removed. Every repair is a note in the trace | — | `checks.attribute` |
| Rule checks | grounding rules, feature keys, numbers absent from the context (including the headline; a percentage must be a percentage there, not the 20 of a date), a risk or opportunity that says there is no effect, impact above the ceiling, an empty house view while theses exist, text not in the run language (a warning) | — | `agent/checks.py` |
| Review | a second model sees the author's full context and flags contradictions, invented numbers, misattributed house views, retelling, the headline; it may ask for more searches | `verify` | `prompts/verify.py` |
| Lifting false alarms | a reviewer's error on a fact or house-view statement becomes a warning if the statement is nearly verbatim from the cited source, all its numbers are there and it has no inference connective; a structural error (unknown or missing source, a number absent from the context, duplicate id, wrong language) stands only if the rule check found it too | — | `checks.confirm_by_sources`, `checks.overrule_structural` |
| Reconciling the reviewer | small reviewers often write “does not contradict the context, but too categorical” and mark it an error. Code makes the severity agree with the explanation: a contradiction or house-view error becomes a warning when its own message (quotes excluded) calls it wording (“too categorical”, “soften”, “not a settled fact”, “generalisation”) or an assumption, when it is a house-view finding on a statement that does not credit the institution, or a “no source” finding while a research source is cited. Numbers are never reconciled, and neither is a statement with a number absent from the context. If a revision round runs anyway, the reviser sees the reconciled remarks with their fix hints and softens the wording itself. At finalisation the statement gets a caveat and confidence ≤ 0.6; an assumption becomes a `hypothesis` (≤ 0.5); a reconciled headline is rewritten | — | `checks.reconcile`, `checks.soften` |
| Revise | with errors: first extra research searches if the reviewer asked, then a revision of the draft by the findings; `max_iterations` rounds (default 1) | `revise` | `prompts/revise.py` |
| Finalize | cut statements with unresolved errors and actions that rely on them; restore a required block with a caveat if it became empty, with the actions that relied on it; if nothing can be restored, word the block by code — the first key fact of the news, or the strongest event–subject link with its feature (confidence ≤ 0.5, with a caveat); restore the house view from cited theses in the run language if the review removed all of it; remove a house view repeated in “why it matters”; rewrite a rejected headline (a long rewrite is cut at a clause boundary such as “, but”, never mid-thought; the fallback is the first conclusion about the subject, not a repeated house view); keep an action while at least one statement it relies on stays; clamp the impact to the ceiling; enforce the action policy; keep only cited sources | `headline` (if needed) | `agent/orchestrator.py` |

## Provenance

| Origin | Meaning | Hard rule |
| --- | --- | --- |
| `fact` | stated in the news or in the subject data | at least one source |
| `house_view` | the institution's published position | cites a research source (thesis or document excerpt); only in the `house_view` block |
| `ai` | the model's inference | sources it rests on |
| `hypothesis` | an assumption the data cannot confirm | sources or features, and a caveat |

Every `why_it_matters` statement must list feature keys of the subject in `features` (the field is required in the JSON schema; if a model omits it, the attribution step fills it). A feature key is valid if it equals a key from the rendered feature list or refines one (`products.loans.rate`).

An empty `house_view` means the institution has no research on the topic; interfaces show this explicitly.

## Impact ceiling

The model tends to give one news item the same level for every reader. The ceiling is computed by code from links between the event and the subject:

| Link | Strength | Ceiling |
| --- | --- | --- |
| the event names the subject's main activity (two-digit NACE/OKVED division); a country the subject trades with; the division of a counterparty with > 30% of the flow; sensitivity ≥ 0.8 to the event's main factor | strong | high |
| a counterparty with 10–30% of the flow; sensitivity 0.5–0.8; the subject's region; strong sensitivity to a secondary factor | moderate | medium |
| only the section letter matches; weak sensitivity; a counterparty matched by section only | weak | low |
| no links | — | low |
| no sensitivities computed for the subject and no strong link | — | medium |

`high` also needs a strong event (`importance` ≥ 0.6); a moderate link to a very weak event (< 0.3) gives `low`. A sensitivity via the group, or to a secondary factor of the event, is one step weaker. Countries are compared by ISO code with Russian and English aliases (“ОАЭ”, “UAE”, “Объединённые Арабские Эмираты” → `AE`).

The model sees the links and the ceiling in the context; an impact above it is a warning in the review and is clamped at finalisation, with a note in the trace. All thresholds are fields of `ImpactPolicy`.

## Action policy

Rules live in `policies/actions.json`. Code decides which rules fire from the subject's features and the event's tags (factors, event types, subject kinds, held and missing products, trade countries, large customers in the industry the event names). The model sees the fired rules with their conditions that code cannot check (“the event is negative for the subject”, “the rate stays high”) and adapts the wording. Code then enforces:

- a product only from a fired rule, matched by name in either language and replaced by its canonical name in the run language; a rule requiring a direction is honoured only if the insight's direction matches;
- at most one `apply_product`;
- with high negative impact the first action is `contact_manager`;
- no products for tax events, and none at all if `offer_products = false`;
- at most three actions; if none remain, the fallback action.

## Reader roles

`policies/roles.json` gives each role a main question, depth, tone, the blocks to expand and the maximum number of statements per block, in both languages: owner, CEO, CFO, chief accountant, treasurer, entrepreneur (sole proprietor or self-employed), private individual, other.

## Follow-up chat

A chat turn is a small agent on the insight's context: the model classifies the question (`clarify`, `slice`, `source`, `out_of_scope`) and may call up to three tools; it answers by a schema with an origin for every statement; code drops statements with unknown sources, facts without sources, house views without research and numbers absent from the context, and rebuilds the answer from verified statements if its text contains an invented number. Off-topic questions are declined without tools.

## Security

- Tools are read-only and bound to one subject; arguments are coerced and capped (≤ 8 theses, ≤ 3 excerpts, ≤ 10 counterparties).
- External text is wrapped in `<news>` tags with inner tags removed; every system prompt says it is data; instruction-like phrases are flagged and the reviewer treats obeying them as an error.
- The impact level and products are bounded by code, so a news text cannot raise them (see `refract demo injection`).
- Statements with rule errors (unknown source, invented number) never reach the reader.
- Keys live only in the environment; the REST API can require a bearer token.

## Ablations

`AgentConfig` switches components off; `refract.agent.ABLATIONS` names the combinations used in experiments:

| Name | Change |
| --- | --- |
| `full` | nothing |
| `no_plan` | a fixed default plan instead of the model's plan |
| `no_verifier` | no second-model review (rule checks stay) |
| `no_rules` | no rule checks; the result is not validated (`Insight.validated = false`) |
| `no_revise` | no revision rounds; unresolved errors are cut |
| `no_self_check` | no review, no rule checks, no revision, no pruning, no attribution repair, no reconciling |
| `no_cap` | no impact ceiling in the context and no clamping |
| `no_policy` | no action rules in the context and no enforcement |
| `no_attribution` | no code repair of missing features and sources |
| `no_reconcile` | the reviewer's severity is taken as is |

## Progress events

`Agent(on_progress=fn)` receives `{"status": "started" | "done" | "finished", "name", "label", "step", …}` for every step; `label` is localised. `agent/progress.py` turns them into an `InsightProgress` snapshot for a loading screen; the REST API streams them over SSE. A failing callback never breaks the run; an unknown subject or event still sends `finished` with `ok = false`.
