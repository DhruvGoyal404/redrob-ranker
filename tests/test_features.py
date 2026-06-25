"""Feature extraction: word-boundary matching and title classification."""
from src import features
from tests.factory import make_raw, normalize, skill


def test_short_terms_do_not_match_inside_words():
    # "rag" inside "average/leverage/storage", "ner" inside "owner" must NOT match.
    raw = make_raw(summary="On average we leverage storage; the owner is a partner.",
                   career=[{"company": "Acme", "title": "Manager",
                            "start_date": "2020-01-01", "end_date": None,
                            "duration_months": 36, "is_current": True,
                            "industry": "X", "company_size": "201-500",
                            "description": "On average we leverage storage."}])
    ev = features.evidence_groups(normalize(raw))
    assert ev["nlp"] == 0


def test_real_evidence_terms_match():
    raw = make_raw(summary="Built retrieval and ranking with embeddings and vector search.")
    ev = features.evidence_groups(normalize(raw))
    assert ev["retrieval_ranking"] > 0
    assert ev["embeddings"] > 0
    assert ev["vector_db"] > 0


def test_title_classification():
    assert features.title_class(normalize(make_raw(title="ML Engineer"))) == "relevant"
    assert features.title_class(normalize(make_raw(title="HR Manager"))) == "nontech"
    assert features.title_class(normalize(make_raw(title="Software Engineer"))) == "adjacent"


def test_ai_skill_count_uses_boundaries():
    rec = normalize(make_raw(skills=[skill("RAG"), skill("Cooking"), skill("NLP")]))
    assert features.ai_skill_count(rec) == 2
