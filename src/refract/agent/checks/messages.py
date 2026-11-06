"""Texts of check findings and repair notes, in the languages of a run."""

from __future__ import annotations

from ...i18n import norm_lang

FINDINGS = {
    "kind_business": {
        "ru": "{cid}: совет для бизнеса («{phrase}») частному лицу — субъект не ведёт бизнеса",
        "en": "{cid}: business advice (“{phrase}”) for a private person — the subject runs no business",
    },
    "kind_staff": {
        "ru": "{cid}: «{phrase}» не относится к самозанятому — у него нет сотрудников и НДС",
        "en": "{cid}: “{phrase}” does not apply to the self-employed — no staff and no VAT",
    },
    "kind_household": {
        "ru": "{cid}: совет для домохозяйства («{phrase}») компании",
        "en": "{cid}: household advice (“{phrase}”) for a company",
    },
    "kind_hint": {
        "ru": "предложите шаг, который этот субъект может сделать сам, исходя из его профиля",
        "en": "suggest a step this subject can take itself, given its profile",
    },
    "superlative": {
        "ru": "{cid}: «{phrase} … {pct}%» не сходится с профилем: {pct}% — доля {other}, а крупнейший {role} — "
        "{top} ({top_pct}%)",
        "en": "{cid}: “{phrase} … {pct}%” does not match the profile: {pct}% is the share of {other}, the largest "
        "{role} is {top} ({top_pct}%)",
    },
    "superlative_none": {
        "ru": "{cid}: «{phrase} … {pct}%» не сходится с профилем: {pct}% — доля {other}",
        "en": "{cid}: “{phrase} … {pct}%” does not match the profile: {pct}% is the share of {other}",
    },
    "superlative_hint": {
        "ru": "назовите этого контрагента по отрасли или стране и его доле ({pct}%), без «крупнейший» и «основной»",
        "en": "name this counterparty by its industry or country and its share ({pct}%), without “largest” or “main”",
    },
    "supplier": {"ru": "поставщик", "en": "supplier"},
    "customer": {"ru": "покупатель", "en": "customer"},
    "of_supplier": {"ru": "поставщика", "en": "supplier"},
    "of_customer": {"ru": "покупателя", "en": "customer"},
    "features": {"ru": "неизвестные признаки субъекта: {bad}", "en": "unknown subject features: {bad}"},
    "features_hint": {
        "ru": "используй ключи из списка «Признаки»",
        "en": "use keys from the “Features” list",
    },
    "headline_numbers": {
        "ru": "в заголовке числа, которых нет в контексте: {nums}",
        "en": "numbers in the headline that are not in the context: {nums}",
    },
    "headline_hint": {
        "ru": "возьми числа из утверждений инсайта или убери их из заголовка",
        "en": "take numbers from the insight's statements or drop them from the headline",
    },
    "numbers": {
        "ru": "числа, которых нет в контексте: {nums}",
        "en": "numbers that are not in the context: {nums}",
    },
    "numbers_hint": {
        "ru": "убери число или возьми его из источника",
        "en": "drop the number or take it from a source",
    },
    "cap": {
        "ru": "уровень влияния выше предела {level} ({reason})",
        "en": "impact above the ceiling {level} ({reason})",
    },
    "cap_claims": {"ru": "; утверждения {ids}", "en": "; statements {ids}"},
    "cap_hint": {
        "ru": "понизь overall_impact и impact до {level} или ниже",
        "en": "lower overall_impact and impact to {level} or below",
    },
    "house_empty": {
        "ru": "есть тезисы института, но блок его позиции пуст",
        "en": "theses are available but the house_view block is empty",
    },
    "house_hint": {
        "ru": "добавь позицию института, если тезисы относятся к событию",
        "en": "add the house view if the theses concern the event",
    },
    "language": {
        "ru": "инсайт написан не по-русски: доля кириллицы {share:.0%}",
        "en": "the insight is not in English: {share:.0%} of letters are Latin",
    },
    "language_hint": {"ru": "перепиши все утверждения по-русски", "en": "rewrite every statement in English"},
    "confirmed": {
        "ru": "{cid}: замечание «{code}» снято — утверждение совпадает с источником",
        "en": "{cid}: finding “{code}” lifted — the statement matches its source",
    },
    "lifted": {"ru": " [снято кодом: совпадает с источником]", "en": " [lifted by code: matches the source]"},
    "structural": {
        "ru": "{cid}: замечание «{code}» снято — код проверяет это сам и ошибки не нашёл",
        "en": "{cid}: finding “{code}” lifted — code checks this itself and found no problem",
    },
    "overruled": {
        "ru": " [снято кодом: структура в порядке]",
        "en": " [lifted by code: the structure is fine]",
    },
    "why_house": {
        "ru": "{cid}: в «почему важно» — позиция института; она уже в блоке house_view, здесь нужен вывод для субъекта",
        "en": "{cid}: “why it matters” repeats the house view, which belongs in house_view; say what it means here",
    },
    "why_house_hint": {
        "ru": "origin=ai: что эта позиция значит для субъекта, через его признаки",
        "en": "origin=ai: what the position means for the subject, through its features",
    },
    "softened": {
        "ru": " [понижено кодом: сама проверка называет это формулировкой, а не ошибкой]",
        "en": " [downgraded by code: the review itself calls this wording, not an error]",
    },
    "softened_note": {
        "ru": "{cid}: «{code}» понижено до предупреждения — {why}",
        "en": "{cid}: “{code}” downgraded to a warning — {why}",
    },
    "why_wording": {"ru": "замечание о формулировке", "en": "a remark on wording"},
    "why_assumption": {
        "ru": "это допущение — станет гипотезой",
        "en": "an assumption — becomes a hypothesis",
    },
    "why_unattributed": {
        "ru": "утверждение не приписано институту",
        "en": "the statement is not attributed to the institution",
    },
    "why_cited": {"ru": "ссылка на аналитику есть", "en": "a research source is cited"},
    "no_effect": {
        "ru": "{cid}: это не риск и не возможность — утверждение говорит, что эффекта нет",
        "en": "{cid}: neither a risk nor an opportunity — the statement says there is no effect",
    },
    "no_effect_hint": {
        "ru": "удали утверждение или перенеси мысль в why_it_matters",
        "en": "delete it or move the thought to why_it_matters",
    },
}


def finding(key: str, lang: str, **kw: object) -> str:
    return FINDINGS[key][norm_lang(lang)].format(**kw)


NOTES = {
    "house_source": {
        "ru": "{cid}: позиции института добавлен источник из аналитики — {ids}",
        "en": "{cid}: the house view now cites its research source — {ids}",
    },
    "number_source": {
        "ru": "{cid}: добавлен источник числа — {ids}",
        "en": "{cid}: added the source of its number — {ids}",
    },
    "mapped": {
        "ru": "{cid}: ключи признаков приведены к списку — {old} → {new}",
        "en": "{cid}: feature keys mapped to the list — {old} → {new}",
    },
    "features": {
        "ru": "{cid}: признаки определены кодом по тексту — {keys}",
        "en": "{cid}: features inferred by code from the text — {keys}",
    },
    "sources": {
        "ru": "{cid}: источник определён кодом по тексту — {ids}",
        "en": "{cid}: source inferred by code from the text — {ids}",
    },
    "fallback_why": {
        "ru": "блок why_it_matters пуст — добавлена связь события с субъектом, рассчитанная кодом",
        "en": "why_it_matters was empty — added the event–subject link computed by code",
    },
    "fallback_what": {
        "ru": "блок what_happened пуст — добавлен факт из новости",
        "en": "what_happened was empty — added a fact from the news",
    },
    "fallback_caveat": {
        "ru": "Сформулировано кодом по признакам субъекта: модель не дала подтверждённого вывода.",
        "en": "Worded by code from the subject's features: the model gave no verified conclusion.",
    },
}
