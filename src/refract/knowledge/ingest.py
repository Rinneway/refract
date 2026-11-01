"""Research documents → searchable chunks. Markdown and plain text out of the box, PDF with `pip install refract-agent[pdf]`.

A document may start with a small front matter block:

    ---
    title: Policy rate outlook
    date: 2026-10-01
    language: en
    countries: GB
    ---

`countries` (ISO codes, comma-separated) ties the document to economies: its excerpts and theses are not used for
subjects from other countries. Without it the document applies everywhere.

Without it the title is the first heading or the file name, and the date is unknown. PDFs keep page numbers, so
the house view can cite “p. 3”; Markdown files are split by `## ` headings, and the section number plays the role
of the page.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

from ..schemas import KnowledgeChunk

CHUNK_CHARS = 800
_FRONT = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)


@dataclass
class Document:
    doc_id: str
    title: str
    doc_date: date | None
    language: str
    pages: list[tuple[int, str]] = field(default_factory=list)  # (page or section number, text)
    path: Path | None = None
    countries: list[str] = field(default_factory=list)


def _countries(value: str | None) -> list[str]:
    return [c.strip().upper() for c in re.split(r"[,;\s]+", value or "") if c.strip()]


def _front_matter(text: str) -> tuple[dict[str, str], str]:
    m = _FRONT.match(text)
    if not m:
        return {}, text
    meta = {}
    for line in m.group(1).splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            meta[k.strip().lower()] = v.strip().strip("\"'")
    return meta, text[m.end() :]


def _date(value: str | None) -> date | None:
    try:
        return date.fromisoformat(value) if value else None
    except ValueError:
        return None


def _guess_lang(text: str) -> str:
    cyr = sum("а" <= ch.lower() <= "я" for ch in text[:4000])
    lat = sum("a" <= ch.lower() <= "z" for ch in text[:4000])
    return "ru" if cyr >= lat else "en"


def read_text_document(path: Path) -> Document:
    meta, body = _front_matter(path.read_text(encoding="utf-8"))
    heading = re.search(r"^#\s+(.+)$", body, re.MULTILINE)
    title = meta.get("title") or (heading.group(1).strip() if heading else path.stem)
    sections = [s for s in re.split(r"\n(?=##\s)", body) if s.strip()]
    return Document(
        doc_id=meta.get("id") or path.stem,
        title=title,
        doc_date=_date(meta.get("date")),
        language=meta.get("language") or _guess_lang(body),
        pages=[(i, s.strip()) for i, s in enumerate(sections, 1)],
        path=path,
        countries=_countries(meta.get("countries")),
    )


def read_pdf_document(path: Path) -> Document:
    try:
        from pypdf import PdfReader
    except ImportError as e:  # pragma: no cover
        raise RuntimeError("PDF support needs: pip install 'refract-agent[pdf]'") from e
    reader = PdfReader(str(path))
    pages = [(i, (page.extract_text() or "").strip()) for i, page in enumerate(reader.pages, 1)]
    meta = reader.metadata or {}
    text = " ".join(t for _, t in pages)
    return Document(
        doc_id=path.stem,
        title=str(getattr(meta, "title", None) or path.stem),
        doc_date=None,
        language=_guess_lang(text),
        pages=[p for p in pages if p[1]],
        path=path,
    )


def read_documents(folder: Path) -> list[Document]:
    docs = []
    for path in sorted(folder.glob("**/*")):
        if path.suffix.lower() in (".md", ".markdown", ".txt"):
            docs.append(read_text_document(path))
        elif path.suffix.lower() == ".pdf":
            docs.append(read_pdf_document(path))
    return docs


def split_chunks(doc: Document, max_chars: int = CHUNK_CHARS) -> list[KnowledgeChunk]:
    """Paragraphs merged into chunks of about `max_chars`; a chunk never crosses a page or section."""
    out: list[KnowledgeChunk] = []
    for page, text in doc.pages:
        parts = [p.strip() for p in re.split(r"\n\s*\n", text) if len(p.strip()) > 40]
        buf, n = "", 0
        for part in [*parts, None]:
            if part is None or (buf and len(buf) + len(part) > max_chars):
                if buf:
                    n += 1
                    out.append(
                        KnowledgeChunk(
                            chunk_id=f"{doc.doc_id}_p{page}_c{n}",
                            doc_id=doc.doc_id,
                            doc_title=doc.title,
                            doc_date=doc.doc_date,
                            page=page,
                            text=buf,
                            language=doc.language,
                            countries=doc.countries,
                        )
                    )
                buf = ""
            if part is not None:
                buf = f"{buf}\n{part}".strip()
    return out
