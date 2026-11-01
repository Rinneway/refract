"""Text checks without an LLM: quote grounding, where numbers come from, delimiting external text."""

from __future__ import annotations

import difflib
import re

# A number token: digits with optional thousand groups (“1 200”, “1,200”) and an optional decimal part. Digits glued
# to letters or underscores (source ids like src_doc_2026q4_t5, “Q3”) are not numbers of the text.
NUM = re.compile(r"(?<!\w)(?:\d{1,3}(?:[  ,]\d{3})+(?:[.,]\d+)?|\d+(?:[.,]\d+)?)")

# External text (news, what a user typed) goes to the model inside <news>…</news>, so the model can tell data from
# instructions. Tags inside the text itself are removed, so the text cannot close the block early.
_NEWS_TAG = re.compile(r"<\s*/?\s*news\b[^>]*>", re.IGNORECASE)
_INJECTION = re.compile(
    r"(игнорируй|проигнорируй|забудь|не\s+учитывай|отмени)\s+(?:\w+\s+){0,3}?(инструкц|указани|правил|промпт)"
    r"|ignore\s+(?:all\s+)?(?:the\s+)?(?:previous|prior|above|earlier)\s+(?:instructions|rules|prompts?)"
    r"|disregard\s+(?:all\s+)?(?:the\s+)?(?:previous|prior|above)"
    r"|system\s*prompt|системн\w*\s+(промпт|инструкц)|ты\s+теперь\s+\w+|you\s+are\s+now\b"
    r"|новые\s+инструкции|new\s+instructions",
    re.IGNORECASE,
)


def untrusted(text: str, source_id: str | None = None) -> str:
    """External text wrapped in delimiters for a prompt."""
    attr = f' id="{source_id}"' if source_id else ""
    return f"<news{attr}>\n{_NEWS_TAG.sub(' ', text or '').strip()}\n</news>"


def looks_like_injection(text: str) -> bool:
    """Whether external text seems to contain an instruction to the model. Used to warn, never to refuse."""
    return bool(_INJECTION.search(text or ""))


def normalize(text: str) -> str:
    text = (text or "").replace("-\n", "").replace("ё", "е").replace("Ё", "Е")
    for a, b in (("«", '"'), ("»", '"'), ("“", '"'), ("”", '"'), ("’", "'"), ("—", "-"), ("–", "-")):
        text = text.replace(a, b)
    return re.sub(r"\s+", " ", text).strip().lower()


def quote_grounding(quote: str, source: str) -> float:
    """Share of the quote found in the source: 1.0 — verbatim."""
    q, s = normalize(quote), normalize(source)
    if not q:
        return 0.0
    if q in s:
        return 1.0
    sm = difflib.SequenceMatcher(None, q, s, autojunk=False)
    return sum(b.size for b in sm.get_matching_blocks()) / len(q)


def _variants(token: str) -> set[float]:
    """All readings of a number token. “1,200” is 1200 in English and 1.2 in Russian — both are kept, so a
    claim and its source match whichever convention each of them uses."""
    out: set[float] = set()
    compact = token.replace(" ", "").replace(" ", "")
    candidates = set()
    if re.fullmatch(
        r"\d{1,3}(?:,\d{3})+(?:\.\d+)?", compact
    ):  # “1,200” or “12,600.5”: thousands or a decimal comma
        candidates.add(compact.replace(",", ""))
    if compact.count(",") <= 1:  # “3,5” or “3.5”; also “12 600 000” after removing the spaces
        candidates.add(compact.replace(",", "."))
    for c in candidates:
        try:
            out.add(round(float(c), 3))
        except ValueError:
            continue
    return out


def numbers(text: str) -> set[float]:
    out: set[float] = set()
    for m in NUM.findall(text or ""):
        out |= _variants(m)
    return out


_PERCENT = re.compile(r"\s?(%|п\.\s?п\.|pp\b|p\.p\.|percent|процент)", re.IGNORECASE)


def percents(text: str) -> set[float]:
    """Numbers written as percentages or percentage points (“20%”, “1,5 п.п.”, “5 percent”)."""
    out: set[float] = set()
    for m in NUM.finditer(text or ""):
        if _PERCENT.match(text[m.end() : m.end() + 12]):
            out |= _variants(m.group(0))
    return out


def first_percent(text: str) -> tuple[str, set[float]] | None:
    """The first percentage in `text`: the token as written and its readings (“14%” → ("14", {14.0}))."""
    for m in NUM.finditer(text or ""):
        if _PERCENT.match(text[m.end() : m.end() + 12]):
            return m.group(0), _variants(m.group(0))
    return None


def invented_numbers(texts: list[str], context: str) -> set[float]:
    """Numbers from `texts` that do not occur in `context`.

    A percentage must occur in the context as a percentage: “20%” is not confirmed by the 20 in a date. Small
    integers (quarters, months, “two steps”) are ignored unless they are percentages: “2%” must come from a source,
    “2 quarters” need not.
    """
    known, known_pct = numbers(context), percents(context)
    out: set[float] = set()
    for t in texts:
        for m in NUM.finditer(t or ""):
            readings = _variants(m.group(0))
            percent = bool(_PERCENT.match(t[m.end() : m.end() + 12]))
            if readings & (known_pct if percent else known):
                continue
            if not percent and all(r.is_integer() and r <= 12 for r in readings):
                continue
            out.add(min(readings))
    return out


_STOP_RU = {
    "этот", "этого", "этом", "года", "году", "также", "может", "могут", "будет", "более", "менее", "после",
    "только", "через", "который", "которая", "которые", "компании", "компания", "компанию", "банка", "банк",
    "ставки", "ставка", "уровне", "вашей", "ваша", "ваше", "ваши", "вашего", "вашим", "около", "чтобы", "нужно",
    "если", "когда",
}  # fmt: skip
_STOP_EN = {
    "this", "that", "these", "those", "with", "from", "into", "will", "would", "could", "should", "also", "more",
    "less", "after", "only", "through", "which", "there", "their", "company", "bank", "rate", "level", "year",
    "have", "been", "being", "than", "they", "were", "your", "about", "when", "what",
}  # fmt: skip
_STOP = {w[:5] for w in _STOP_RU | _STOP_EN}


def stems(text: str) -> set[str]:
    """Crude stems: the first five letters of words of four or more letters, without frequent words."""
    return {w[:5] for w in re.findall(r"[а-яa-z]+", normalize(text)) if len(w) >= 4} - _STOP


def lexical_support(text: str, source: str) -> float:
    """Share of the statement's stems found in the source: 1.0 — every word comes from the source."""
    own = stems(text)
    return len(own & stems(source)) / len(own) if own else 0.0


def script_share(text: str, lang: str) -> float:
    """Share of letters written in the script of `lang`: Cyrillic for ru, Latin for en. 1.0 for text without letters.

    Acronyms such as USD or NACE keep a Russian text a little below 1.0; a text in the wrong language is near 0.
    """
    letters = [ch for ch in (text or "").lower() if ch.isalpha()]
    if not letters:
        return 1.0
    cyrillic = sum("а" <= ch <= "я" or ch == "ё" for ch in letters)
    latin = sum("a" <= ch <= "z" for ch in letters)
    own = cyrillic if (lang or "").lower().startswith("ru") else latin
    return own / len(letters)


# Where a sentence can end without losing its meaning: before “, но” / “, but”, a semicolon, a dash or a colon.
_CLAUSE = re.compile(
    r",\s+(?:но|а|однако|хотя|при этом|причём|but|while|although|whereas|yet)\b|\s(?:while|whereas|but)\s"
    r"|;|\s[—–-]\s|:\s",
    re.IGNORECASE,
)


def trim_sentence(text: str, limit: int = 160, min_len: int = 40) -> str:
    """Shorten a one-sentence text to `limit` characters.

    The cut is made at the last clause boundary that keeps at least `min_len` characters (“X may do Y, but Z…” →
    “X may do Y”), so the result is still a whole statement; only without a boundary is it cut at a word with “…”.
    """
    text = " ".join((text or "").split())
    if len(text) <= limit:
        return text
    cuts = [m.start() for m in _CLAUSE.finditer(text) if min_len <= m.start() <= limit]
    if cuts:
        return text[: max(cuts)].rstrip(" ,;:—–-")
    return text[: limit - 1].rsplit(" ", 1)[0].rstrip(",;:—–- ") + "…"
