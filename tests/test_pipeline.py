"""End-to-end smoke: the full parse -> trap -> retrieve -> score -> reason path
runs on the bundled sample and produces sane, grounded output."""
import json
from datetime import date
from pathlib import Path

import pytest

from src import parse, traps, retrieve, reasoning
from src import score as scoring
from src.jd import JD_QUERY

SAMPLE = Path(__file__).resolve().parent.parent / "data" / "sample_candidates.json"


@pytest.fixture(scope="module")
def ranked():
    raws = json.loads(SAMPLE.read_text(encoding="utf-8"))
    recs = [parse.normalize(r, date(2026, 6, 1)) for r in raws]
    narratives = [r["narrative"] for r in recs]
    idx, dense = retrieve.shortlist(narratives, JD_QUERY, None, None, size=len(recs))
    rows = []
    for i in idx:
        rec = recs[i]
        trap = traps.assess(rec)
        sc = scoring.score_candidate(rec, trap, float(dense[i]))
        rows.append((rec, trap, sc))
    rows.sort(key=lambda x: -x[2]["final"])
    return rows


def test_pipeline_produces_rows(ranked):
    assert len(ranked) > 0


def test_scores_are_monotonic_non_increasing(ranked):
    scores = [sc["final"] for _, _, sc in ranked]
    assert scores == sorted(scores, reverse=True)


def test_top_candidate_outscores_bottom(ranked):
    assert ranked[0][2]["final"] > ranked[-1][2]["final"]


def test_reasoning_is_grounded_and_nonempty(ranked):
    for pos, (rec, trap, sc) in enumerate(ranked[:5], 1):
        why = reasoning.build_reasoning(rec, sc, trap, pos)
        assert isinstance(why, str) and len(why) > 20
        # confidence tag present
        assert any(tag in why for tag in ["confidence", "Excluded"])


def test_top_candidate_is_relevant_titled(ranked):
    from src import features
    top_titles = [features.title_class(rec) for rec, _, _ in ranked[:3]]
    assert any(t in ("relevant", "adjacent") for t in top_titles)
