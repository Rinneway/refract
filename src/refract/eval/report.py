"""Aggregate an experiment into `report.md` and `summary.json`.

Per system: answer rate; judge total over six questions (0–12) with a 95% bootstrap CI; per-criterion means;
unsupported statements per answer; retelling and impact-calibration rates by the judge; and code-computed metrics —
rule errors reaching the reader, invented numbers, impact above the computed ceiling; cost and latency. Then
pairwise comparison with a reference system (wins / ties / losses, sign test, paired bootstrap difference),
agreement between judges (quadratic-weighted κ per criterion, Spearman on totals) and, if pairs carry human labels,
accuracy and κ of the impact level against them.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .harness import Pair, _read, _safe
from .judge import CRITERIA
from .stats import bootstrap_ci, mean, paired_bootstrap_diff, percentile, sign_test, spearman, weighted_kappa

LEVELS = ["low", "medium", "high"]


def _judge_scores(out: Path, judge: str, system: str, pairs: list[Pair]) -> dict[str, dict]:
    res = {}
    for p in pairs:
        d = _read(out / "judge" / _safe(judge) / system / f"{p.id}.json")
        if d and d.get("score"):
            res[p.id] = d["score"]
    return res


def _total(score: dict) -> int:
    return sum(score[c]["score"] for c in CRITERIA)


def summarize(
    out: str | Path, pairs: list[Pair], judges: list[str], reference: str = "full"
) -> dict[str, Any]:
    out = Path(out)
    systems = (
        sorted(p.name for p in (out / "runs").iterdir() if p.is_dir()) if (out / "runs").exists() else []
    )
    primary = judges[0] if judges else None
    S: dict[str, Any] = {"pairs": len(pairs), "judges": judges, "reference": reference, "systems": {}}
    for system in systems:
        runs = {p.id: _read(out / "runs" / system / f"{p.id}.json") for p in pairs}
        ok = {k: v for k, v in runs.items() if v and v.get("content")}
        objective = [v.get("objective") or {} for v in ok.values()]
        structured = [o for o in objective if o.get("rule_errors") is not None]
        m: dict[str, Any] = {
            "n": len(pairs),
            "answered": len(ok),
            "cost_mean": mean([v.get("cost", 0.0) for v in ok.values()]),
            "latency_p50_s": percentile([v.get("latency_ms", 0) / 1000 for v in ok.values()], 0.5),
            "latency_p90_s": percentile([v.get("latency_ms", 0) / 1000 for v in ok.values()], 0.9),
            "llm_calls_mean": mean([v.get("llm_calls", 0) for v in ok.values()]),
            "over_cap_rate": mean([float(o.get("over_cap", False)) for o in objective])
            if objective
            else None,
            "rule_error_rate": mean([float(bool(o["rule_errors"])) for o in structured])
            if structured
            else None,
            "invented_numbers_mean": mean([o.get("invented_numbers", 0) for o in structured])
            if structured
            else None,
            "language_ok_rate": mean([float(o.get("language_ok", True)) for o in objective])
            if objective
            else None,
            "judges": {},
            "by_language": {},
        }
        for lang in sorted({p.language for p in pairs}):
            subset = [p for p in pairs if p.language == lang]
            got = [ok[p.id] for p in subset if p.id in ok]
            obj = [g.get("objective") or {} for g in got]
            st = [o for o in obj if o.get("rule_errors") is not None]
            row: dict[str, Any] = {
                "n": len(subset),
                "answered": len(got),
                "language_ok_rate": mean([float(o.get("language_ok", True)) for o in obj]) if obj else None,
                "rule_error_rate": mean([float(bool(o["rule_errors"])) for o in st]) if st else None,
                "over_cap_rate": mean([float(o.get("over_cap", False)) for o in obj]) if obj else None,
            }
            if primary:
                sc = _judge_scores(out, primary, system, subset)
                row["judge_total_mean"] = mean([_total(v) for v in sc.values()]) if sc else None
            m["by_language"][lang] = row
        if primary and {"ru", "en"} <= {p.language for p in pairs}:
            # the same subject, event and role in both languages — the paired difference of judge totals
            sc = _judge_scores(out, primary, system, pairs)
            totals = {
                (p.profile_id, p.event_id, p.role, p.language): _total(sc[p.id]) for p in pairs if p.id in sc
            }
            matched = sorted(
                {k[:3] for k in totals if k[3] == "en"} & {k[:3] for k in totals if k[3] == "ru"}
            )
            if matched:
                d, lo, hi = paired_bootstrap_diff(
                    [totals[(*k, "en")] for k in matched], [totals[(*k, "ru")] for k in matched]
                )
                m["language_parity"] = {"n": len(matched), "en_minus_ru": d, "ci95": [lo, hi]}
        gold = [
            (p.gold_impact, (runs[p.id] or {}).get("objective", {}).get("impact"))
            for p in pairs
            if p.gold_impact and runs.get(p.id) and runs[p.id].get("content")
        ]
        if gold:
            m["gold_impact_accuracy"] = mean([float(g == y) for g, y in gold])
            pairs_ok = [(LEVELS.index(g), LEVELS.index(y)) for g, y in gold if y in LEVELS]
            m["gold_impact_kappa"] = (
                weighted_kappa([a for a, _ in pairs_ok], [b for _, b in pairs_ok], [0, 1, 2])
                if pairs_ok
                else None
            )
        for judge in judges:
            scores = _judge_scores(out, judge, system, pairs)
            if not scores:
                continue
            totals = [_total(s) for s in scores.values()]
            lo, hi = bootstrap_ci(totals)
            m["judges"][judge] = {
                "n": len(scores),
                "total_mean": mean(totals),
                "total_ci95": [lo, hi],
                "overall_mean": mean([s["overall"] for s in scores.values()]),
                "criteria": {c: mean([s[c]["score"] for s in scores.values()]) for c in CRITERIA},
                "unsupported_mean": mean([len(s.get("unsupported", [])) for s in scores.values()]),
                "retelling_rate": mean([float(s["retelling"]) for s in scores.values()]),
                "impact_calibrated_rate": mean([float(s["impact_calibrated"]) for s in scores.values()]),
            }
        S["systems"][system] = m

    if primary and reference in systems:
        ref = _judge_scores(out, primary, reference, pairs)
        S["pairwise"] = {}
        for system in systems:
            if system == reference:
                continue
            other = _judge_scores(out, primary, system, pairs)
            common = sorted(set(ref) & set(other))
            a, b = [_total(ref[k]) for k in common], [_total(other[k]) for k in common]
            wins = sum(x > y for x, y in zip(a, b, strict=True))
            losses = sum(x < y for x, y in zip(a, b, strict=True))
            diff, lo, hi = paired_bootstrap_diff(a, b) if common else (float("nan"),) * 3
            S["pairwise"][system] = {
                "n": len(common),
                "reference_wins": wins,
                "ties": len(common) - wins - losses,
                "reference_losses": losses,
                "sign_test_p": sign_test(wins, losses),
                "mean_diff": diff,
                "diff_ci95": [lo, hi],
            }

    if len(judges) >= 2:
        j1, j2 = judges[:2]
        agree: dict[str, Any] = {"judges": [j1, j2], "criteria": {}}
        x1, x2 = [], []
        crit = {c: ([], []) for c in CRITERIA}
        overall = ([], [])
        for system in systems:
            s1, s2 = _judge_scores(out, j1, system, pairs), _judge_scores(out, j2, system, pairs)
            for k in set(s1) & set(s2):
                x1.append(_total(s1[k]))
                x2.append(_total(s2[k]))
                overall[0].append(s1[k]["overall"])
                overall[1].append(s2[k]["overall"])
                for c in CRITERIA:
                    crit[c][0].append(s1[k][c]["score"])
                    crit[c][1].append(s2[k][c]["score"])
        agree["n"] = len(x1)
        agree["total_spearman"] = spearman(x1, x2)
        agree["overall_kappa"] = weighted_kappa(overall[0], overall[1], [1, 2, 3, 4, 5])
        for c, (a, b) in crit.items():
            agree["criteria"][c] = {
                "kappa": weighted_kappa(a, b, [0, 1, 2]),
                "exact": mean([float(p == q) for p, q in zip(a, b, strict=True)]),
            }
        S["agreement"] = agree
    return S


def _f(x: Any, digits: int = 2) -> str:
    if x is None or (isinstance(x, float) and x != x):
        return "—"
    return f"{x:.{digits}f}" if isinstance(x, float) else str(x)


def _pct(x: Any) -> str:
    return "—" if x is None or (isinstance(x, float) and x != x) else f"{100 * x:.0f}%"


def render(S: dict[str, Any]) -> str:
    primary = S["judges"][0] if S["judges"] else None
    lines = [
        "# Evaluation report",
        "",
        f"Pairs: {S['pairs']}. Judges: {', '.join(S['judges']) or '—'}. Reference: `{S['reference']}`.",
        "",
    ]
    lines += [
        "| System | Answered | Judge total /12 [95% CI] | Overall /5 | Unsupported / answer | Retelling | Impact calibrated (judge) | Above ceiling (code) | Rule errors (code) | Invented numbers | Language ok | Cost | p50 s |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for name, m in S["systems"].items():
        j = m["judges"].get(primary, {}) if primary else {}
        ci = j.get("total_ci95", [None, None])
        total = f"{_f(j.get('total_mean'))} [{_f(ci[0])}–{_f(ci[1])}]" if j else "—"
        lines.append(
            f"| `{name}` | {m['answered']}/{m['n']} | {total} | {_f(j.get('overall_mean'))} | {_f(j.get('unsupported_mean'))} | "
            f"{_pct(j.get('retelling_rate'))} | {_pct(j.get('impact_calibrated_rate'))} | {_pct(m.get('over_cap_rate'))} | "
            f"{_pct(m.get('rule_error_rate'))} | {_f(m.get('invented_numbers_mean'))} | {_pct(m.get('language_ok_rate'))} | "
            f"{_f(m.get('cost_mean'), 4)} | {_f(m.get('latency_p50_s'), 0)} |"
        )
    langs = sorted({lang for m in S["systems"].values() for lang in m.get("by_language", {})})
    if len(langs) > 1:
        lines += [
            "",
            "## By output language",
            "",
            "| System | Language | Answered | Judge total /12 | Language ok | Rule errors | Above ceiling |",
            "|---|---|---|---|---|---|---|",
        ]
        for name, m in S["systems"].items():
            for lang in langs:
                if r := m["by_language"].get(lang):
                    lines.append(
                        f"| `{name}` | {lang} | {r['answered']}/{r['n']} | {_f(r.get('judge_total_mean'))} | "
                        f"{_pct(r.get('language_ok_rate'))} | {_pct(r.get('rule_error_rate'))} | {_pct(r.get('over_cap_rate'))} |"
                    )
        if parity := {k: m["language_parity"] for k, m in S["systems"].items() if m.get("language_parity")}:
            lines += [
                "",
                "Language parity — judge total of the English insight minus the Russian one for the same subject, event and role:",
                "",
                "| System | n | en − ru [95% CI] |",
                "|---|---|---|",
            ]
            for name, p in parity.items():
                lines.append(
                    f"| `{name}` | {p['n']} | {_f(p['en_minus_ru'])} [{_f(p['ci95'][0])}–{_f(p['ci95'][1])}] |"
                )
    if primary:
        lines += [
            "",
            f"## Criteria (judge `{primary}`, 0–2)",
            "",
            "| System | " + " | ".join(CRITERIA) + " |",
            "|---|" + "---|" * len(CRITERIA),
        ]
        for name, m in S["systems"].items():
            j = m["judges"].get(primary)
            if j:
                lines.append(f"| `{name}` | " + " | ".join(_f(j["criteria"][c]) for c in CRITERIA) + " |")
    if S.get("pairwise"):
        lines += [
            "",
            f"## `{S['reference']}` against each system (judge `{primary}`, total score)",
            "",
            "| System | n | wins | ties | losses | sign test p | mean diff [95% CI] |",
            "|---|---|---|---|---|---|---|",
        ]
        for name, p in S["pairwise"].items():
            lines.append(
                f"| `{name}` | {p['n']} | {p['reference_wins']} | {p['ties']} | {p['reference_losses']} | {_f(p['sign_test_p'], 3)} | "
                f"{_f(p['mean_diff'])} [{_f(p['diff_ci95'][0])}–{_f(p['diff_ci95'][1])}] |"
            )
    if S.get("agreement"):
        a = S["agreement"]
        lines += [
            "",
            f"## Agreement between `{a['judges'][0]}` and `{a['judges'][1]}` (n = {a['n']})",
            "",
            f"Spearman on totals: {_f(a['total_spearman'])}; weighted κ on overall: {_f(a['overall_kappa'])}.",
            "",
            "| Criterion | weighted κ | exact agreement |",
            "|---|---|---|",
        ]
        for c, v in a["criteria"].items():
            lines.append(f"| {c} | {_f(v['kappa'])} | {_pct(v['exact'])} |")
    gold = {k: m for k, m in S["systems"].items() if "gold_impact_accuracy" in m}
    if gold:
        lines += [
            "",
            "## Impact level against human labels",
            "",
            "| System | accuracy | weighted κ |",
            "|---|---|---|",
        ]
        for name, m in gold.items():
            lines.append(
                f"| `{name}` | {_pct(m['gold_impact_accuracy'])} | {_f(m.get('gold_impact_kappa'))} |"
            )
    lines += [
        "",
        "Rule errors and invented numbers are computed by code on what the reader sees; “above ceiling” compares the stated impact with the ceiling computed from subject–event links.",
    ]
    return "\n".join(lines) + "\n"


def write_report(
    out: str | Path, pairs: list[Pair], judges: list[str], reference: str = "full"
) -> dict[str, Any]:
    out = Path(out)
    S = summarize(out, pairs, judges, reference)
    (out / "summary.json").write_text(
        json.dumps(S, ensure_ascii=False, indent=1, default=str), encoding="utf-8"
    )
    (out / "report.md").write_text(render(S), encoding="utf-8")
    return S
