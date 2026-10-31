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
