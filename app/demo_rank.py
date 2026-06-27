"""
JD-adaptive ranking for the LIVE DEMO.

This is a separate, general-purpose "rank candidates for ANY job description" engine.
It deliberately does NOT touch the submission pipeline (src/score.py, rank.py), which
stays tuned to the challenge's one target role and produced the validated submission.csv.

How it ranks (so editing the JD genuinely re-ranks):
  * RELEVANCE to the typed JD - dense semantic similarity (bge-small) fused with BM25
    keyword relevance - is the PRIMARY signal. Change the JD and this changes.
  * UNIVERSAL QUALITY - skill trust (proficiency x assessment x endorsements) and a mild
    experience-reasonableness term - modulates the relevance (these don't depend on the JD).
  * BEHAVIORAL AVAILABILITY - recruiter responsiveness / recency - a light multiplier.
  * TRAP GATE - honeypots and keyword-stuffers are driven to the bottom, JD-agnostic.

We reuse src.score.score_candidate purely to read its *universal* signals (skill_trust,
experience_band, availability) and the trap modifiers - never its challenge-JD-specific
title/domain/must-have components - so this ranker is genuinely role-agnostic.
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np

from src import score as scoring

_EV_LABEL = {
    "retrieval_ranking": "retrieval/ranking",
    "embeddings": "embeddings",
    "vector_db": "vector search",
    "evaluation": "ranking evaluation (NDCG/MRR)",
    "nlp": "NLP/LLM",
    "ml_core": "applied ML",
}
_EV_ORDER = ["retrieval_ranking", "embeddings", "vector_db", "evaluation", "nlp", "ml_core"]


_PROF_W = {"beginner": 0.25, "intermediate": 0.5, "advanced": 0.8, "expert": 1.0}


def _norm(x) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    lo, hi = float(x.min()), float(x.max())
    return (x - lo) / (hi - lo) if hi > lo else np.zeros_like(x)


def _skill_quality(rec: dict) -> float:
    """ROLE-AGNOSTIC skill quality over ALL of a candidate's skills (proficiency x
    duration x platform-assessment x endorsements). Unlike src.score._skill_trust this
    is not AI-specific, so the demo doesn't favour AI candidates for non-AI roles.
    Top-5 average so a few genuinely strong skills count, not diluted by a long tail."""
    vals = []
    for s in rec.get("skills", []):
        prof = _PROF_W.get(s.get("proficiency"), 0.4)
        dur = min(1.0, (s.get("months") or 0) / 24.0)
        a = s.get("assessment")
        assess = (a / 100.0) if a is not None else 0.5
        endo = min(1.0, (s.get("endorsements") or 0) / 50.0)
        vals.append(prof * (0.4 * dur + 0.4 * assess + 0.2 * endo))
    vals.sort(reverse=True)
    top = vals[:5] or [0.0]
    return sum(top) / len(top)


def rank(records: List[dict], traps_list: List[dict],
         semantic, bm25, w_sem: float = 0.6, w_rel: float = 0.55) -> List:
    """Rank candidates for the typed JD.

    `semantic`, `bm25`: per-candidate relevance arrays to the JD (semantic = cosine of
    the dense JD/candidate vectors; bm25 = lexical). Returns a list of (rec, trap, info)
    sorted best-first, where info carries the final score + a transparent breakdown.

    Score = (w_rel * JD-relevance + (1-w_rel) * universal-quality) * availability * trap.
    The blend is ADDITIVE so a genuine candidate is never zeroed by low JD-relevance, and a
    keyword-stuffer with high JD-relevance is still driven down by the hard trap multiplier
    (the whole point: stuffers have high *keyword* relevance but no real evidence).
    """
    sem_n, bm_n = _norm(semantic), _norm(bm25)
    rel = w_sem * sem_n + (1.0 - w_sem) * bm_n            # JD relevance, [0, 1]
    rows = []
    for i, (rec, trap) in enumerate(zip(records, traps_list)):
        sc = scoring.score_candidate(rec, trap, float(sem_n[i]))   # for experience + availability
        skill = _skill_quality(rec)                                 # role-agnostic skill quality
        expb = sc["components"]["experience_band"]
        avail = float(sc["modifiers"].get("availability", 1.0))
        quality = 0.7 * skill + 0.3 * expb                          # universal, [0, 1]
        base = w_rel * float(rel[i]) + (1.0 - w_rel) * quality      # JD-relevance dominant
        trap_mult = 0.001 if trap["is_honeypot"] else 0.05 if trap["is_stuffer"] else 1.0
        final = base * avail * trap_mult
        rows.append((rec, trap, {
            "final": final, "jd_relevance": float(rel[i]),
            "semantic": float(sem_n[i]), "bm25": float(bm_n[i]),
            "skill_trust": skill, "experience_band": expb, "quality": quality,
            "availability": avail, "evidence": sc["evidence"],
        }))
    rows.sort(key=lambda r: -r[2]["final"])
    return rows


def confidence(info: Dict, trap: Dict) -> str:
    if trap["is_honeypot"]:
        return "Excluded"
    if trap["is_stuffer"]:
        return "Low"
    rel, q = info["jd_relevance"], info["quality"]
    if rel >= 0.60 and q >= 0.45:
        return "High"
    if rel >= 0.32:
        return "Moderate"
    return "Low"


def reasoning(rec: dict, info: Dict, trap: Dict, conf: str) -> str:
    """A grounded, JD-adaptive one-liner: relevance to THIS JD + demonstrated evidence."""
    if trap["is_honeypot"]:
        return (f"Excluded - {rec['title']}: internally inconsistent profile "
                f"(impossible tenure/skill timeline).")
    if trap["is_stuffer"]:
        return (f"Low confidence - {rec['title']}: non-technical profile listing AI skills "
                f"with no demonstrated backing (keyword-stuffer, demoted).")
    rel_pct = round(100 * info["jd_relevance"])
    parts = [f"{conf} confidence - {rec['title']}, {rec['yoe']:.1f} yrs",
             f"{rel_pct}% match to this JD (semantic + keyword)"]
    ev = info["evidence"]
    demo_ev = [_EV_LABEL[g] for g in _EV_ORDER if ev.get(g)][:3]
    if demo_ev:
        parts.append("demonstrates " + ", ".join(demo_ev))
    rr = rec["signals"].get("recruiter_response_rate")
    if rr and rr >= 0.6:
        parts.append(f"responsive to recruiters ({rr:.0%})")
    loc = rec.get("location")
    if loc:
        parts.append(f"{loc}-based")
    return "; ".join(parts) + "."
