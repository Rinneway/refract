"""Human-readable rendering of insights, chat answers and traces (Markdown)."""

from __future__ import annotations

from .config import Institution
from .i18n import norm_lang, t
from .schemas import ChatResponse, Feed, Insight, Trace

_TITLES = {
    "ru": {
        "what_happened": "Что произошло", "why_it_matters": "Почему это важно для вас", "risks": "Риски",
        "opportunities": "Возможности", "scenarios": "Сценарии", "house_view": "Позиция {institution_gen}",
        "actions": "Что можно сделать", "sources": "Источники", "impact": "Влияние", "trace": "Как получен вывод",
        "no_house_view": "По этой теме у {institution_gen} пока нет позиции.", "product": "продукт",
    },
    "en": {
        "what_happened": "What happened", "why_it_matters": "Why it matters to you", "risks": "Risks",
        "opportunities": "Opportunities", "scenarios": "Scenarios", "house_view": "{institution} view",
        "actions": "What you can do", "sources": "Sources", "impact": "Impact", "trace": "How this was produced",
        "no_house_view": "{institution} has no published view on this topic yet.", "product": "product",
    },
}  # fmt: skip


def _inst(institution: Institution | None, lang: str) -> dict[str, str]:
    inst = institution or Institution()
    return {"institution": inst.display(lang), "institution_gen": inst.genitive(lang)}


def insight_markdown(ins: Insight, institution: Institution | None = None) -> str:
    lang = norm_lang(ins.language)
    T, names = _TITLES[lang], _inst(institution, lang)
    out = [
        f"## {ins.headline}",
        "",
        f"**{T['impact']}:** {ins.overall_impact.value} · {ins.overall_direction.value}",
    ]

    def claims(key: str, items) -> None:
        if not items:
            return
        out.extend(["", f"### {T[key].format(**names)}"])
        for c in items:
            label = t(f"origin.{c.origin.value}", lang, **names)
            extra = f" ({c.impact.value}, {c.horizon.value})" if hasattr(c, "impact") else ""
            cites = f" [{', '.join(c.source_ids)}]" if c.source_ids else ""
            out.append(f"- *{label}*{extra}: {c.text}{cites}")
            if c.caveat:
                out.append(f"  > {c.caveat}")

    claims("what_happened", ins.what_happened)
    claims("why_it_matters", ins.why_it_matters)
    claims("risks", ins.risks)
    claims("opportunities", ins.opportunities)
    if ins.scenarios:
        out.extend(["", f"### {T['scenarios']}"])
        out += [f"- **{s.name}** ({s.plausibility}): {s.description}" for s in ins.scenarios]
    if ins.house_view:
        claims("house_view", ins.house_view)
    else:
        out.extend(["", f"### {T['house_view'].format(**names)}", T["no_house_view"].format(**names)])
    if ins.actions:
        out.extend(["", f"### {T['actions']}"])
        for a in ins.actions:
            product = f" — {T['product']}: {a.product}" if a.product else ""
            out.append(f"- **{a.text}** ({a.kind.value}{product}). {a.rationale}")
    out.extend(["", f"### {T['sources']}"])
    for s in ins.sources:
        where = f", {t('ctx.page', lang)} {s.page}" if s.page else ""
        link = f" <{s.url}>" if s.url and s.url.startswith("http") else ""
        out.append(f"- `{s.source_id}` {s.title}{' — ' + s.publisher if s.publisher else ''}{where}{link}")
    out.extend(["", f"_{ins.disclaimer}_"])
    return "\n".join(out)


def trace_markdown(trace: Trace) -> str:
    lang = norm_lang(trace.language)
    lines = [
        f"### {_TITLES[lang]['trace']}",
        "",
        "| # | step | model / tool | ms | cost | verdict | notes |",
        "|---|---|---|---|---|---|---|",
    ]
    for s in trace.steps:
        who = s.model or s.tool or ""
        notes = "; ".join(s.issues)[:160].replace("|", "/")
        lines.append(
            f"| {s.index} | {s.name} | {who} | {s.latency_ms} | {s.cost:.4f} | {s.verdict or ''} | {notes} |"
        )
    tt = trace.totals
    lines.append("")
    lines.append(
        f"LLM calls: {tt.llm_calls}, tools: {tt.tool_calls}, tokens: {tt.tokens_in}/{tt.tokens_out}, "
        f"cost: {tt.cost:.4f} {trace.currency}, model time: {tt.latency_ms / 1000:.1f} s, revisions: {tt.iterations}"
    )
    return "\n".join(lines)


def chat_markdown(res: ChatResponse, institution: Institution | None = None) -> str:
    lang = norm_lang(res.language)
    names = _inst(institution, lang)
    out = [res.answer]
    for c in res.claims:
        out.append(
            f"- *{t(f'origin.{c.origin.value}', lang, **names)}*: {c.text} [{', '.join(c.source_ids)}]"
        )
    if res.follow_ups:
        out.append("")
        out += [f"→ {q}" for q in res.follow_ups]
    return "\n".join(out)


_FEED = {
    "ru": {
        "title": "Новости для {profile}", "direct": "касается напрямую", "background": "фон", "none": "не касается",
        "ceiling": "предел влияния", "considered": "Подходящих событий: {n}, показано: {k}.",
        "untagged": "Не размечены (учтены только связи по отрасли и стране): {ids}.", "empty": "Нет событий.",
    },
    "en": {
        "title": "News for {profile}", "direct": "concerns you directly", "background": "background",
        "none": "unrelated", "ceiling": "impact ceiling", "considered": "Matching events: {n}, shown: {k}.",
        "untagged": "Not tagged (only activity and country links count): {ids}.", "empty": "No events.",
    },
}  # fmt: skip


def feed_markdown(feed: Feed, links: int = 2) -> str:
    """The feed as a list: title, date, relevance and impact ceiling, then the strongest links."""
    lang = norm_lang(feed.language)
    T = _FEED[lang]
    out = [f"## {T['title'].format(profile=feed.profile_id)}", ""]
    if not feed.items:
        out.append(T["empty"])
    for i, it in enumerate(feed.items, 1):
        date = it.published_at.strftime("%Y-%m-%d")
        out.append(
            f"{i}. **{it.title}** — {date}, `{it.event_id}` · {T[it.relevance]} · {T['ceiling']}: "
            f"{it.impact_ceiling.value}"
        )
        out += [f"   - {x}" for x in it.links[:links]]
    out += ["", T["considered"].format(n=feed.considered, k=len(feed.items))]
    if feed.untagged:
        out.append(T["untagged"].format(ids=", ".join(feed.untagged)))
    return "\n".join(out)
