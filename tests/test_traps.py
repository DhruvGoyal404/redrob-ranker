"""Trap gate: honeypots and keyword-stuffers must be caught; real engineers must not."""
from src import traps
from tests.factory import make_raw, normalize, skill, with_assessments


def test_clean_ml_candidate_is_not_a_trap():
    raw = make_raw(
        title="ML Engineer", yoe=6.0,
        summary="Built retrieval and ranking systems with embeddings and a vector "
                "database at a product company; designed NDCG evaluation.",
        skills=[skill("Embeddings"), skill("Vector Search"), skill("NLP")])
    with_assessments(raw, {"Embeddings": 80, "Vector Search": 75})
    rec = normalize(raw)
    t = traps.assess(rec)
    assert not t["is_honeypot"]
    assert not t["is_stuffer"]


def test_honeypot_expert_skill_zero_months():
    raw = make_raw(skills=[skill("RAG", proficiency="expert", months=0)])
    t = traps.assess(normalize(raw))
    assert t["is_honeypot"]
    assert "expert_skill_with_zero_months" in t["honeypot_reasons"]


def test_honeypot_role_longer_than_career():
    raw = make_raw(yoe=4.0, career=[{
        "company": "Acme", "title": "ML Engineer", "start_date": "2010-01-01",
        "end_date": "2024-01-01", "duration_months": 168, "is_current": False,
        "industry": "Software", "company_size": "201-500",
        "description": "Long role."}])
    t = traps.assess(normalize(raw))
    assert t["is_honeypot"]  # 168 months >> 4y career


def test_keyword_stuffer_nontech_title_with_unbacked_ai_skills():
    raw = make_raw(
        title="HR Manager", yoe=8.0,
        summary="HR Manager across recruitment and payroll.",
        skills=[skill("RAG"), skill("Pinecone"), skill("LLM"), skill("Embeddings")])
    t = traps.assess(normalize(raw))
    assert t["is_stuffer"]
    assert not t["is_honeypot"]
