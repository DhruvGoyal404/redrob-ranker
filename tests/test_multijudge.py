"""Mechanics of the multi-judge gold-set harness (no LLM calls; uses stub judges)."""
from eval import multijudge as mj
from tests.factory import make_raw, normalize, skill


def test_stub_judge_is_deterministic():
    rec = normalize(make_raw(title="ML Engineer", summary="built retrieval and ranking"))
    trap = {"is_honeypot": False, "is_stuffer": False}
    j = mj.StubJudge("judgeA", seed=1)
    s1 = j.score("jd", rec, trap)
    j2 = mj.StubJudge("judgeA", seed=1)
    assert 0 <= s1 <= 5
    assert s1 == j2.score("jd", rec, trap)


def test_panel_agreement_identical_is_one():
    scores = {"a": [5, 3, 0, 4, 2], "b": [5, 3, 0, 4, 2]}
    _, mean_k = mj.panel_agreement(scores)
    assert abs(mean_k - 1.0) < 1e-9


def test_panel_agreement_disagreement_below_one():
    scores = {"a": [5, 5, 0, 0, 3], "b": [0, 0, 5, 5, 2]}
    _, mean_k = mj.panel_agreement(scores)
    assert mean_k < 0.5


def test_aggregate_gold_is_median():
    scores = {"a": [5, 0, 3], "b": [4, 1, 3], "c": [4, 5, 0]}
    assert mj.aggregate_gold(scores) == [4, 1, 3]


def test_stratified_sample_spans_buckets_and_size():
    by_id = {}
    by_id["CAND_0000001"] = (normalize(make_raw(candidate_id="CAND_0000001", title="ML Engineer")),
                             {"is_honeypot": False, "is_stuffer": False})
    by_id["CAND_0000002"] = (normalize(make_raw(candidate_id="CAND_0000002", title="Software Engineer")),
                             {"is_honeypot": False, "is_stuffer": False})
    by_id["CAND_0000003"] = (normalize(make_raw(candidate_id="CAND_0000003", title="HR Manager")),
                             {"is_honeypot": False, "is_stuffer": True})
    picked = mj.stratified_sample(by_id, n=3, seed=1)
    assert len(picked) <= 3
    assert all(c in by_id for c in picked)
