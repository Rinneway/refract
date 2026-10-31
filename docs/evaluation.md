# Evaluation

The harness answers two kinds of questions: does the agent beat simpler systems, and what does each component contribute. It is built so that results can go into a paper: resumable runs, fixed prompts with versions, cached model answers, code-computed metrics next to judge scores, and statistics with uncertainty.

## Workflow

```bash
refract eval pairs  --data DATA --out runs/exp1/pairs.jsonl --languages ru,en
refract eval run    --data DATA --out runs/exp1 --systems full,no_plan,no_verifier,no_rules,no_revise,no_self_check,no_cap,no_policy,single_call,retelling
refract eval judge  --data DATA --out runs/exp1 --judges verifier,judge2
refract eval report --out runs/exp1 --reference full
```

- `pairs` — every profile × every event × every language, with the default reader role for each subject kind. Edit the file to choose pairs, set roles or add human labels (`gold_impact`, `gold_relevant`).
- `run` — writes `runs/<system>/<pair>.json`: the answer, the exact context the system saw, cost, latency, number of LLM calls, the trace (for agent variants) and code-computed metrics. Existing files are skipped (`--force` recomputes).
- `judge` — writes `judge/<judge>/<system>/<pair>.json`. The judge sees the system's own context and its answer in one neutral form, without the system's name. The first judge is primary; the second is used for agreement.
- `report` — `report.md` and `summary.json`.

An event without factors is tagged once and the tags are shared by all systems (`_events/`), so systems differ only in what is being compared.

## Systems

| System | What it is | Question it answers |
| --- | --- | --- |
| `full` | the whole agent | — |
| `no_plan` | default plan instead of the model's | does planning help? |
| `no_verifier` | rule checks only | what does the second model add? |
| `no_rules` | second model only, no code checks, unvalidated output | what do deterministic checks add? |
| `no_revise` | no revision round | does revising beat cutting? |
| `no_self_check` | no checks, no revision, no pruning | the value of self-verification as a whole |
| `no_cap` | no impact ceiling | is impact calibrated without it? |
| `no_policy` | no action rules | do actions stay relevant and safe? |
| `single_call` | the same insight prompt with the house view matched by factors, one call | is the agentic loop worth its cost? |
| `retelling` | one call, plain prompt, no house view | the naive baseline |

## Metrics

**By the judge** (`eval/judge.py`, version in every file):

| Metric | Scale |
| --- | --- |
| what happened, why it matters, consequences, actions, house view, grounding | 0–2 each; the total is 0–12 |
| overall usefulness | 1–5 |
| unsupported statements | count per answer |
| retelling instead of interpretation | yes / no |
| impact level calibrated | yes / no |

**By code** (no judge needed, `harness._objective`): rule errors that reach the reader (unknown sources, facts without sources, house views without research, unknown feature keys, invented numbers), invented numbers per answer, impact above the computed ceiling, cost, latency p50/p90, LLM calls.

**Against human labels** (if `gold_impact` is filled): accuracy and quadratic-weighted κ of the stated impact level.

## Statistics

- 95% percentile-bootstrap confidence intervals of means (2 000 resamples, fixed seed).
- Pairwise comparison with the reference system on the same pairs: wins / ties / losses by judge total, exact two-sided sign test, mean paired difference with a bootstrap CI.
- Agreement between the first two judges: Spearman correlation of totals, quadratic-weighted κ per criterion and for the overall score, exact agreement.

With 30 pairs a difference of about one point of the total is detectable; plan at least 50–100 pairs per comparison for a paper, and correct for multiple comparisons if you test many ablations against `full`.

## Human labels

Fill `gold_impact` (and optionally `gold_relevant`) in `pairs.jsonl` for a sample of pairs, ideally by two annotators with an adjudication pass; report their agreement (κ) before comparing systems to them. A simple guideline: *high* — a decision is needed within weeks; *medium* — specific decisions may need adjusting within a quarter; *low* — background information.

## Checking the English prompts

The English prompts are translations of prompts tuned on Russian news. Before reporting English results, check that they behave like the Russian ones. `examples/demo/pairs_parity.jsonl` has seven subject × event × role triples — each in Russian and in English — covering a company, a sole proprietor, a self-employed person, a household, Russian and English news, and a prompt injection.

```bash
cp refract.example.toml refract.toml      # OpenRouter ids of the original models; edit if needed
# .env: LLM_API_KEY=<your OpenRouter key>
refract models                            # the endpoint answers and knows the ids
refract eval run    --pairs examples/demo/pairs_parity.jsonl --out runs/parity --systems full,single_call
refract eval judge  --out runs/parity --judges judge2,verifier   # a judge that neither wrote nor reviewed goes first
refract eval report --out runs/parity --reference full
```

What to look at in `runs/parity/report.md`:

| Check | Where | Good sign |
| --- | --- | --- |
| The answer is in the requested language | “Language ok”, by language | 100% for `en` and `ru` |
| Grounding does not degrade | “Rule errors”, by language | `en` not worse than `ru` |
| Quality parity | “Language parity”, en − ru | the 95% CI includes 0 |
| Calibration and safety | “Above ceiling”; the injection pair in `runs/parity/runs/full/` | 0%; impact not `high`, no product |
| Structured output | `logs/llm_calls.jsonl` | no `:repair` steps piling up on English runs |

Then read three or four English insights next to their Russian twins (`runs/parity/runs/full/*__en.json` and `*__ru.json`): the wording of the house view, the `features` cited, the actions. With seven triples the parity check catches gross problems only; for a paper, repeat it on 30 or more triples of fresh news. A difference that persists points to a prompt (see `trace.config.prompts` for versions), not to the model, if `single_call` shows the same gap.

Two cross-language effects are by design: a reviewer's finding against a translated statement cannot be lifted by the lexical source check (it compares words), and a house view removed by the review is restored only from theses in the run language. Both make English runs on Russian research slightly more conservative.

## Threats to validity

- **Judge bias.** LLM judges prefer longer and more structured answers and their own family's style. Use a judge from another family than the author, report two judges and their agreement, and validate on human labels.
- **Shared context.** All systems see the same subject data and research; the comparison is about how the context is used, not about retrieval coverage.
- **Synthetic data.** The demo data is invented and small; it shows the mechanics and is not a benchmark. Use your own pairs for claims.
- **Leakage.** Choose news published after the prompts were last changed; the original experiments used a held-out set of fresh news not seen during development.
- **Costs and latency** depend on the provider and caching; report them with the provider and model ids, and with a cold cache.

## Results from the original setting

In the original setting (Russian news, three real client profiles, October 2026; the data is not public), the full agent with GLM-5.3-Flash as the author and DeepSeek-V4.1-Flash as the reviewer scored 8.6 / 12 by the primary judge (9.5 by a second judge) against 6.8 for one-call retelling, answered 30 / 30 pairs, had no rule errors reaching the reader, and kept the impact level calibrated in all pairs. One call with the same prompt and the house view scored 9.1 on the pairs it answered but failed on 3 of 30 and had rule errors in 19% of answers. The agent took a median of about 70 s per fresh news item. These numbers cannot be reproduced from this repository and are given only as context for designing new experiments.
