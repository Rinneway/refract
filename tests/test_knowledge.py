from refract.knowledge import BM25, HybridIndex
from refract.knowledge.ingest import read_documents
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


def test_hybrid_index_fuses_dense_scores(tmp_path) -> None:
    docs = ["ставка центробанка", "курс доллара", "урожай пшеницы"]
    vectors = {
        "ставка центробанка": [1, 0, 0],
        "курс доллара": [0, 1, 0],
        "урожай пшеницы": [0, 0, 1],
        "зерно": [0, 0, 1],
    }
    index = HybridIndex(
        docs, embed=lambda xs: [vectors.get(x, [0.3, 0.3, 0.3]) for x in xs], cache_dir=tmp_path
    )
    assert index.top("зерно", 1)[0][0] == 2  # no lexical overlap — found by embeddings
    assert list(tmp_path.glob("emb_*.json"))
    assert BM25(docs).top("зерно") == []


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
