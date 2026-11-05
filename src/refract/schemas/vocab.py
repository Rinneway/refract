"""Shared vocabulary of events, profiles and research.

An event is tagged with factors (`NewsEvent.macro_factors`), a profile with sensitivities to them
(`Exposure.factor`), a thesis with the factors it covers (`Thesis.factors`). Matching factors is what links a
piece of news to a subject and to the institution's position.

Besides the keys below, prefixed factors are allowed: `sector:<two-digit NACE division>`, `region:<name>`,
`country:<ISO 3166-1 alpha-2>`, `commodity:<name>`, `fx:<ISO 4217>`.
"""

from __future__ import annotations

import re

FACTORS: dict[str, dict[str, str]] = {
    "policy_rate": {
        "ru": "Ключевая ставка центробанка и стоимость кредитов",
        "en": "Central bank policy rate and cost of credit",
    },
    "inflation": {"ru": "Инфляция и цены", "en": "Inflation and prices"},
    "fx:USD": {"ru": "Курс доллара", "en": "US dollar exchange rate"},
    "fx:EUR": {"ru": "Курс евро", "en": "Euro exchange rate"},
    "fx:CNY": {"ru": "Курс юаня", "en": "Chinese yuan exchange rate"},
    "oil": {"ru": "Нефть и нефтепродукты", "en": "Oil and oil products"},
    "gas": {"ru": "Газ", "en": "Natural gas"},
    "metals": {"ru": "Металлы и сталь", "en": "Metals and steel"},
    "coal": {"ru": "Уголь", "en": "Coal"},
    "agri": {"ru": "Сельхозсырьё и продовольствие", "en": "Agricultural commodities and food"},
    "gold": {"ru": "Золото", "en": "Gold"},
    "energy_prices": {"ru": "Электроэнергия и коммунальные тарифы", "en": "Electricity and utility tariffs"},
    "consumer_demand": {"ru": "Потребительский спрос", "en": "Consumer demand"},
    "construction": {"ru": "Строительство и недвижимость", "en": "Construction and real estate"},
    "credit_market": {"ru": "Кредитование и ипотека", "en": "Lending and mortgages"},
    "bonds": {"ru": "Облигации и госдолг", "en": "Bonds and government debt"},
    "equities": {"ru": "Фондовый рынок", "en": "Equity market"},
    "labor_market": {"ru": "Рынок труда и зарплаты", "en": "Labour market and wages"},
    "taxes": {"ru": "Налоги и сборы", "en": "Taxes and levies"},
    "tariffs": {"ru": "Пошлины и торговые барьеры", "en": "Customs duties and trade barriers"},
    "sanctions": {"ru": "Санкции и ограничения", "en": "Sanctions and restrictions"},
    "logistics": {"ru": "Логистика и транспорт", "en": "Logistics and transport"},
    "budget": {"ru": "Госбюджет и госзаказ", "en": "Public budget and procurement"},
    "regulation": {"ru": "Отраслевое регулирование", "en": "Industry regulation"},
    "sme_support": {
        "ru": "Поддержка малого бизнеса и самозанятых",
        "en": "Support for small business and the self-employed",
    },
    "gdp": {"ru": "Экономический рост", "en": "Economic growth"},
}

FACTOR_PREFIXES = ("sector:", "region:", "country:", "commodity:", "fx:")

THESIS_TOPICS: dict[str, dict[str, str]] = {
    "policy_rate": {
        "ru": "Ключевая ставка и денежно-кредитная политика",
        "en": "Policy rate and monetary policy",
    },
    "inflation": {"ru": "Инфляция", "en": "Inflation"},
    "gdp": {"ru": "Рост экономики", "en": "Economic growth"},
    "consumption": {"ru": "Потребление", "en": "Consumption"},
    "labor_market": {"ru": "Рынок труда", "en": "Labour market"},
    "fx": {"ru": "Валютный курс", "en": "Exchange rates"},
    "bonds": {"ru": "Облигации и долговой рынок", "en": "Bonds and debt market"},
    "commodities": {"ru": "Сырьевые рынки", "en": "Commodity markets"},
    "budget": {"ru": "Бюджет", "en": "Public budget"},
    "credit": {"ru": "Кредитование", "en": "Credit"},
    "real_estate": {"ru": "Недвижимость", "en": "Real estate"},
    "taxes": {"ru": "Налоги", "en": "Taxes"},
    "sme": {"ru": "Малый бизнес", "en": "Small business"},
    "industry": {"ru": "Отрасли", "en": "Industries"},
}


def factor_label(factor: str, lang: str = "ru") -> str:
    """A factor as a reader names it: 'policy_rate' → «Ключевая ставка центробанка и стоимость кредитов»;
    prefixed factors keep their value ('sector:41' → «отрасль 41», 'country:CN' → «страна CN»)."""
    lang = "en" if lang == "en" else "ru"
    if factor in FACTORS:
        return FACTORS[factor][lang]
    prefix, _, value = factor.partition(":")
    names = {
        "sector": ("отрасль", "industry"),
        "region": ("регион", "region"),
        "country": ("страна", "country"),
        "commodity": ("сырьё", "commodity"),
        "fx": ("курс", "exchange rate of"),
    }
    if value and prefix in names:
        return f"{names[prefix][0 if lang == 'ru' else 1]} {value}"
    return factor


def is_known_factor(factor: str) -> bool:
    return factor in FACTORS or factor.startswith(FACTOR_PREFIXES)


def factors_prompt(lang: str = "ru") -> str:
    """The factor vocabulary as text for prompts."""
    lang = "en" if lang == "en" else "ru"
    lines = [f"- {key} — {desc[lang]}" for key, desc in FACTORS.items()]
    lines.append(
        "- а также sector:<две цифры NACE/ОКВЭД>, region:<регион>, country:<код ISO>, commodity:<сырьё>, fx:<код валюты>"
        if lang == "ru"
        else "- and also sector:<two-digit NACE division>, region:<region>, country:<ISO code>, commodity:<name>, fx:<currency code>"
    )
    return "\n".join(lines)


def topics_prompt(lang: str = "ru") -> str:
    lang = "en" if lang == "en" else "ru"
    return ", ".join(f"{k} ({v[lang]})" for k, v in THESIS_TOPICS.items())


# ------------------------------------------------------------------------------------------ NACE / OKVED

# Sections by the two-digit division. Identical in NACE Rev. 2 and OKVED 2.
SECTIONS: list[tuple[int, int, str]] = [
    (1, 3, "A"), (5, 9, "B"), (10, 33, "C"), (35, 35, "D"), (36, 39, "E"), (41, 43, "F"), (45, 47, "G"),
    (49, 53, "H"), (55, 56, "I"), (58, 63, "J"), (64, 66, "K"), (68, 68, "L"), (69, 75, "M"), (77, 82, "N"),
    (84, 84, "O"), (85, 85, "P"), (86, 88, "Q"), (90, 93, "R"), (94, 96, "S"), (97, 98, "T"), (99, 99, "U"),
]  # fmt: skip


def division(code: str | None) -> str | None:
    """Two-digit division of an activity code: '46.73' → '46'."""
    head = (code or "").strip().split(".")[0]
    return head.zfill(2) if head.isdigit() and len(head) <= 2 else None


# In trade the division says only “wholesale” or “retail”; what is traded is in the three-digit group
# (46.2 agricultural raw materials, 46.7 fuel, metals and building materials).
TRADE_DIVISIONS = frozenset({"45", "46", "47"})


def group(code: str | None) -> str | None:
    """Three-digit group of an activity code: '46.73' → '46.7', '46.2' → '46.2', '46' → None."""
    two = division(code)
    parts = (code or "").strip().split(".")
    if two is None or len(parts) < 2 or not parts[1][:1].isdigit():
        return None
    return f"{two}.{parts[1][0]}"


def section_of(code: str | None) -> str | None:
    two = division(code)
    if two is None:
        return None
    return next((sec for lo, hi, sec in SECTIONS if lo <= int(two) <= hi), None)


# ------------------------------------------------------------------------------------------ countries

# ISO 3166-1 alpha-2 → names and common aliases in Russian and English (lower case, «ё» → «е»).
COUNTRIES: dict[str, tuple[str, ...]] = {
    "RU": ("россия", "российская федерация", "рф", "russia", "russian federation"),
    "CN": ("китай", "кнр", "китайская народная республика", "china", "prc", "people's republic of china"),
    "HK": ("гонконг", "сянган", "hong kong"),
    "AE": (
        "оаэ",
        "объединенные арабские эмираты",
        "арабские эмираты",
        "эмираты",
        "uae",
        "united arab emirates",
        "emirates",
    ),
    "TR": ("турция", "турецкая республика", "туркие", "turkey", "turkiye", "türkiye"),
    "IN": ("индия", "india"),
    "KZ": ("казахстан", "kazakhstan"),
    "BY": ("беларусь", "белоруссия", "belarus"),
    "AM": ("армения", "armenia"),
    "UZ": ("узбекистан", "uzbekistan"),
    "KG": ("киргизия", "кыргызстан", "киргизстан", "kyrgyzstan"),
    "TJ": ("таджикистан", "tajikistan"),
    "AZ": ("азербайджан", "azerbaijan"),
    "GE": ("грузия", "georgia"),
    "MD": ("молдова", "молдавия", "moldova"),
    "TM": ("туркменистан", "туркмения", "turkmenistan"),
    "KR": ("южная корея", "республика корея", "корея", "south korea", "republic of korea", "korea"),
    "KP": ("кндр", "северная корея", "north korea"),
    "JP": ("япония", "japan"),
    "VN": ("вьетнам", "социалистическая республика вьетнам", "vietnam", "viet nam"),
    "TH": ("таиланд", "тайланд", "thailand"),
    "MY": ("малайзия", "malaysia"),
    "ID": ("индонезия", "indonesia"),
    "MN": ("монголия", "mongolia"),
    "IR": ("иран", "исламская республика иран", "iran"),
    "SA": ("саудовская аравия", "королевство саудовская аравия", "saudi arabia"),
    "EG": ("египет", "арабская республика египет", "egypt"),
    "TN": ("тунис", "tunisia"),
    "MA": ("марокко", "morocco"),
    "ZA": ("юар", "южная африка", "south africa"),
    "BR": ("бразилия", "brazil"),
    "AR": ("аргентина", "argentina"),
    "MX": ("мексика", "mexico"),
    "US": (
        "сша",
        "соединенные штаты",
        "соединенные штаты америки",
        "америка",
        "usa",
        "us",
        "united states",
        "united states of america",
    ),
    "CA": ("канада", "canada"),
    "GB": (
        "великобритания",
        "британия",
        "соединенное королевство",
        "англия",
        "uk",
        "united kingdom",
        "britain",
        "great britain",
    ),
    "DE": ("германия", "фрг", "germany"),
    "FR": ("франция", "france"),
    "IT": ("италия", "italy"),
    "ES": ("испания", "spain"),
    "NL": ("нидерланды", "голландия", "netherlands", "holland"),
    "PL": ("польша", "poland"),
    "CZ": ("чехия", "чешская республика", "czechia", "czech republic"),
    "FI": ("финляндия", "finland"),
    "SE": ("швеция", "sweden"),
    "NO": ("норвегия", "norway"),
    "CH": ("швейцария", "switzerland"),
    "AT": ("австрия", "austria"),
    "RS": ("сербия", "serbia"),
    "IL": ("израиль", "israel"),
    "SG": ("сингапур", "singapore"),
    "MM": ("мьянма", "бирма", "myanmar", "burma"),
    "EU": ("евросоюз", "европейский союз", "ес", "eu", "european union"),
}

# Main currency by country — for linking trade countries to fx:<code> factors.
COUNTRY_CURRENCY: dict[str, str] = {
    "CN": "CNY", "HK": "HKD", "AE": "AED", "TR": "TRY", "IN": "INR", "KZ": "KZT", "BY": "BYN", "AM": "AMD",
    "UZ": "UZS", "KG": "KGS", "TJ": "TJS", "AZ": "AZN", "GE": "GEL", "KR": "KRW", "JP": "JPY", "VN": "VND",
    "US": "USD", "GB": "GBP", "DE": "EUR", "FR": "EUR", "IT": "EUR", "ES": "EUR", "NL": "EUR", "FI": "EUR",
    "AT": "EUR", "EU": "EUR", "CH": "CHF", "RU": "RUB", "BR": "BRL", "SA": "SAR", "EG": "EGP", "TH": "THB",
}  # fmt: skip

_ALIAS = {alias: code for code, names in COUNTRIES.items() for alias in names}
_PREFIXES = ("республика ", "королевство ", "государство ", "republic of ", "kingdom of ", "state of ")


def _clean(name: str) -> str:
    return re.sub(r"[\s\-–—]+", " ", name.lower().replace("ё", "е")).strip(" .,()")


def country_code(name: str | None) -> str:
    """ISO 3166-1 alpha-2 code for a country name in Russian or English; unknown names are returned cleaned.

    «Объединённые Арабские Эмираты», «ОАЭ» and “UAE” all give “AE”; “AE” stays “AE”.
    """
    raw = (name or "").strip()
    if len(raw) == 2 and raw.isalpha() and raw.upper() in COUNTRIES:
        return raw.upper()
    key = _clean(raw)
    if key in _ALIAS:
        return _ALIAS[key]
    for prefix in _PREFIXES:
        if key.startswith(prefix) and key[len(prefix) :] in _ALIAS:
            return _ALIAS[key[len(prefix) :]]
    return key
