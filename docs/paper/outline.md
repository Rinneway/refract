# Paper outline (working draft)

A plan for a paper based on Refract. Everything here is a proposal to be checked against experiments and the literature; no result below is final.

## Working title

*Refract: Provenance-Tagged, Code-Bounded LLM Agents for Personalised News Impact Analysis*

Alternatives: *Same News, Different Readers: Grounded Personalisation of Economic News with a Self-Verifying LLM Agent*; *Who Is This News For? An Agent that Explains Economic Events to Companies, Entrepreneurs and Households*.

## Problem

Readers who run a business or manage household finances do not need a summary of the news; they need to know what it means for them. LLM summaries tend to (1) retell the news instead of interpreting it, (2) assign the same importance to every reader, (3) mix facts, the institution's view and the model's speculation, (4) invent numbers, and (5) are easy to steer through text in the news itself.

## Research questions

1. **RQ1 — Interpretation.** Does an agent that plans, retrieves and self-verifies produce more useful, better-grounded interpretations than one call with the same context and than a plain retelling?
2. **RQ2 — Components.** What does each component contribute: planning, the second-model review, deterministic rule checks, revision, the impact ceiling, the action policy?
3. **RQ3 — Calibration.** Does a ceiling computed from subject–event links make impact levels agree better with human judgement than letting the model decide?
4. **RQ4 — Safety.** How robust is the pipeline to prompt injection in news text, compared with the baselines?
5. **RQ5 — Languages and subjects.** Does quality hold across Russian and English and across companies, sole proprietors, the self-employed and households?
6. **RQ6 — Judges.** How well do LLM judges agree with each other and with human annotators on this task?

## Contributions (candidate)

- A task formulation: personalised news impact analysis with provenance-tagged statements (fact / house view / inference / hypothesis) and feature-cited personalisation.
- A hybrid architecture where the model writes and code bounds: feature keys, source registry, number checks, an impact ceiling from typed links, a rule-based action policy.
- An open, bilingual implementation with ablation switches and a resumable evaluation harness (code metrics + LLM judges + statistics).
- An empirical study of the components (RQ1–RQ6).

## Method (sections)

1. Task and output contract (`docs/data-model.md`).
2. Agent pipeline (`docs/agent.md`): tagging, planning, tools, drafting, review, revision, finalisation.
3. Deterministic layer: grounding rules; numbers; the link strength model and the impact ceiling; the action policy.
4. Follow-up chat on the same context.
5. Implementation: provider-agnostic client, caching, traces.

## Experiments

| Experiment | Systems | Data | Metrics |
| --- | --- | --- | --- |
| E1 main comparison | `full`, `single_call`, `retelling` | ≥ 100 subject × news pairs on fresh news, both languages | judge total, criteria, unsupported statements, rule errors, cost, latency |
| E2 ablations | `full` and the seven ablations | the same pairs | Δ vs `full` with paired CIs |
| E3 calibration | `full`, `no_cap`, `single_call` | pairs with two human impact labels | accuracy, weighted κ vs humans |
| E4 injection | all systems | news with injected instructions (several templates, both languages) | share of answers that follow the injection: impact raised, product added, instruction echoed |
| E5 judges | two or three judges from different families | E1 outputs, a human-scored subsample | κ per criterion, Spearman, judge–human agreement |
| E6 sensitivity | `full` with varied `ImpactPolicy` thresholds | E3 pairs | calibration vs thresholds |

Data plan: real news from public feeds published after the prompts were frozen; subject profiles either synthetic (released) or real but private (only aggregates reported); a released synthetic set for reproducibility of mechanics.

## Figures and tables

1. Pipeline diagram (README).
2. Example: one news item, three readers (company, sole proprietor, household), three different insights with origins.
3. Table: E1 results with CIs.
4. Table: ablations (Δ total, Δ unsupported, Δ rule errors, Δ cost).
5. Figure: impact confusion matrices vs human labels (`full`, `no_cap`).
6. Table: injection success rates.
7. Table: judge agreement.

## Related work to survey

Retrieval-augmented generation and attribution (citations, verifiability of generated text); self-refinement and self-verification of LLM outputs (critique-and-revise loops, chain-of-verification); LLM-as-a-judge and its biases; tool-using and planning agents; prompt injection and indirect injection through retrieved content; personalisation of LLM outputs; financial and economic news analysis with NLP (event detection, sentiment, impact on firms). Collect concrete references with a literature search; do not cite from memory.

## Limitations to state

Rule-based link strengths; profile completeness; LLM judges; synthetic data for the released set; costs tied to providers.

## Ethics

Information, not advice; explicit provenance; no products without a rule; no use of personal data in the released set; a guard against publishing confidential material.
