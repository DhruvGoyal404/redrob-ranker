"""The live demo's JD-adaptive ranker (app/demo_rank.py) - prove it actually responds
to the JD and still demotes traps. Embeddings are mocked, so this runs without torch."""
from src import traps
import demo_rank
from tests.factory import make_raw, normalize, skill


def _assess(recs):
    return [traps.assess(r) for r in recs]


def _three_similar():
    return [normalize(make_raw(
        candidate_id=f"CAND_000000{i}", title="ML Engineer", yoe=6.0,
        summary="Built retrieval and ranking systems with embeddings and vector search.",
        skills=[skill("Embeddings"), skill("Vector Search")])) for i in (1, 2, 3)]


def test_jd_relevance_drives_order_and_is_responsive():
    recs = _three_similar(); tr = _assess(recs)
    # JD-A: candidate 3 most relevant
    a = demo_rank.rank(recs, tr, semantic=[0.1, 0.2, 0.9], bm25=[0, 0, 0])
    # JD-B (a different JD): relevance flips -> candidate 1 most relevant
    b = demo_rank.rank(recs, tr, semantic=[0.9, 0.2, 0.1], bm25=[0, 0, 0])
    assert a[0][0]["candidate_id"].endswith("3")
    assert b[0][0]["candidate_id"].endswith("1")
    assert a[0][0]["candidate_id"] != b[0][0]["candidate_id"]   # the JD genuinely re-ranks


def test_bm25_also_contributes():
    recs = _three_similar(); tr = _assess(recs)
    rows = demo_rank.rank(recs, tr, semantic=[0, 0, 0], bm25=[0.1, 0.9, 0.2])
    assert rows[0][0]["candidate_id"].endswith("2")


def test_traps_demoted_even_at_max_relevance():
    good = normalize(make_raw(candidate_id="CAND_0000001", title="ML Engineer", yoe=6.0,
                              summary="Built retrieval ranking embeddings vector search",
                              skills=[skill("Embeddings")]))
    stuffer = normalize(make_raw(candidate_id="CAND_0000002", title="HR Manager", yoe=8.0,
                                 skills=[skill("RAG"), skill("Pinecone"), skill("LLM"),
                                         skill("Embeddings")]))
    recs = [good, stuffer]; tr = _assess(recs)
    assert tr[1]["is_stuffer"]
    rows = demo_rank.rank(recs, tr, semantic=[0.1, 0.99], bm25=[0, 0])  # stuffer max relevance
    assert rows[0][0]["candidate_id"] == "CAND_0000001"
    assert rows[-1][1]["is_stuffer"]


def test_confidence_and_reasoning_are_jd_adaptive():
    rec = normalize(make_raw(title="ML Engineer", yoe=6.0,
                             summary="Built retrieval ranking embeddings vector search",
                             skills=[skill("Embeddings")]))
    tr = traps.assess(rec)
    rows = demo_rank.rank([rec], [tr], semantic=[0.9], bm25=[0.9])
    _, _, info = rows[0]
    conf = demo_rank.confidence(info, tr)
    text = demo_rank.reasoning(rec, info, tr, conf)
    assert conf in ("High", "Moderate", "Low")
    assert "match to this JD" in text
    assert text.startswith(conf)
