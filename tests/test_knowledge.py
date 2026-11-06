from refract.knowledge import BM25, HybridIndex
from refract.knowledge.ingest import read_documents
from refract.knowledge.store import DataStore
from refract.textutil import quote_grounding
from tests.conftest import DEMO


def test_documents_and_front_matter() -> None:
    docs = {d.doc_id: d for d in read_documents(DEMO / "knowledge" / "docs")}
    macro = docs["demo_macro_2026q4"]
    assert macro.language == "ru" and str(macro.doc_date) == "2026-10-01" and len(macro.pages) == 5
    assert docs["demo_fxcom_2026m10"].language == "en"


def test_demo_theses_are_grounded(store) -> None:
    docs = {d.doc_id: dict(d.pages) for d in read_documents(DEMO / "knowledge" / "docs")}
    assert len(store.theses) >= 10
    for th in store.theses:
        assert quote_grounding(th.quote, docs[th.doc_id][th.page]) == 1.0, th.thesis_id


def test_search(store) -> None:
    hits = store.search_chunks("прогноз ключевой ставки на конец года")
    assert hits and hits[0].doc_id == "demo_macro_2026q4"
    en = store.search_chunks("dollar forecast GBP/USD hedging")
    assert en and en[0].doc_id == "demo_fxcom_2026m10"
    theses = store.find_theses(["policy_rate"], ["policy_rate"], "ставка")
    assert theses[0].topic == "policy_rate"


VECTORS = {
    "ставка центробанка": [1, 0, 0],
    "курс доллара": [0, 1, 0],
    "урожай пшеницы": [0, 0, 1],
    "зерно": [0, 0, 1],
}


def fake_embed(calls: list[int] | None = None):
    def embed(texts):
        if calls is not None:
            calls.append(len(texts))
        return [VECTORS.get(x, [0.3, 0.3, 0.3]) for x in texts]

    return embed


def test_hybrid_index_fuses_dense_scores(tmp_path) -> None:
    from refract.knowledge.search import Embeddings

    docs = ["ставка центробанка", "курс доллара", "урожай пшеницы"]
    calls: list[int] = []
    index = HybridIndex(docs, Embeddings(fake_embed(calls), "m1", tmp_path))
    assert index.top("зерно", 1)[0][0] == 2  # no lexical overlap — found by embeddings
    assert BM25(docs).top("зерно") == []
    again = HybridIndex([*docs, "новая новость"], Embeddings(fake_embed(calls), "m1", tmp_path))
    assert again.top("зерно", 1)[0][0] == 2
    assert calls == [3, 1, 1]  # only the new text; the repeated query is cached too
    assert len(list(tmp_path.glob("emb_*.jsonl"))) == 1
    HybridIndex(docs, Embeddings(fake_embed(calls), "m2", tmp_path))  # another model: its own vectors
    assert len(list(tmp_path.glob("emb_*.jsonl"))) == 2 and calls[-1] == 3


def test_a_failing_embedding_model_leaves_search_lexical(tmp_path) -> None:
    from refract.knowledge.search import Embeddings

    def broken(texts):
        raise RuntimeError("model 'bge-m3' not found, try pulling it first")

    store = DataStore.from_dir(DEMO, embed=broken, cache_dir=tmp_path, embed_model="bge-m3")
    assert store.search_chunks("ставка", 2) and store.search_events("ставка", 1)  # BM25 still answers
    assert any("embeddings unavailable (bge-m3)" in p and "BM25" in p for p in store.problems)
    assert Embeddings(broken).vectors(["x"]) is None


def test_news_are_found_across_languages_with_embeddings(store, tmp_path) -> None:
    def embed(texts):  # a toy multilingual model: wheat and grain in either language are one direction
        return [
            [1.0, 0.0] if any(w in t.lower() for w in ("wheat", "зерн", "пшениц")) else [0.0, 1.0]
            for t in texts
        ]

    store_e = DataStore(list(store.profiles.values()), list(store.events.values()), store.theses, embed=embed)
    assert store.search_events("цены на зерно", 1) == []  # BM25 alone: no common words with the English news
    assert store_e.search_events("цены на зерно", 1)[0].event_id == "ev_wheat_en"


def test_research_is_scoped_to_the_subjects_country(store, tmp_path) -> None:
    from refract.knowledge.ingest import read_text_document

    uk = {t.thesis_id for t in store.find_theses(["policy_rate", "fx:USD"], [], k=8, country="GB")}
    ru = {t.thesis_id for t in store.find_theses(["policy_rate", "fx:USD"], [], k=8, country="RU")}
    assert not any(t.startswith("demo_macro_") for t in uk)  # a Russian rate forecast is not for a UK company
    assert "demo_fxcom_2026m10_t2" in uk and "demo_fxcom_2026m10_t2" not in ru  # UK exporters only
    assert any(t.startswith("demo_macro_") for t in ru)
    assert all(not c.doc_id.startswith("demo_macro") for c in store.search_chunks("ставка", 3, country="GB"))
    doc = tmp_path / "d.md"
    doc.write_text(
        "---\nid: d\ncountries: ru, gb\n---\n# T\n\n## S\n\n" + "Текст раздела. " * 10, encoding="utf-8"
    )
    assert read_text_document(doc).countries == ["RU", "GB"]


def test_a_long_paragraph_is_cut_at_sentence_ends() -> None:
    from pathlib import Path

    from refract.knowledge.ingest import Document, split_chunks

    text = " ".join(
        f"Предложение {i}: банк ожидает снижения ставки до {10 + i % 5}% к концу года." for i in range(300)
    )
    doc = Document(doc_id="d", title="t", doc_date=None, language="ru", pages=[(1, text)], path=Path("d.txt"))
    chunks = split_chunks(doc)
    assert len(chunks) > 20 and max(len(c.text) for c in chunks) <= 800
    assert all(c.text.endswith("года.") for c in chunks)  # whole sentences
    assert " ".join(c.text for c in chunks) == text  # nothing lost
    unbroken = Document(
        doc_id="u", title="t", doc_date=None, language="ru", pages=[(1, "x" * 2500)], path=Path("u")
    )
    assert [len(c.text) for c in split_chunks(unbroken)] == [800, 800, 800, 100]


def test_search_matches_word_forms(store) -> None:
    from refract.knowledge.search import stem

    assert {stem(w) for w in ("ставка", "ставку", "ставки", "ставке")} == {"ставк"}
    assert stem("налога") == stem("налог") and stem("рубля") == stem("рубль") and stem("кредитов") == "кредит"
    assert stem("rates") == stem("rate") and stem("policies") == "policy" and stem("2026") == "2026"
    assert store.search_events("ставка", k=1)[0].event_id == "ev_rate_hold_ru"  # the title says «ставку»
