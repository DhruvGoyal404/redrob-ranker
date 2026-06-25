"""
Signal-based weighted scorer — the core ranker.

Design principle (straight from the JD's note to participants): score on
*demonstrated evidence of retrieval/ranking/applied-ML work*, not on the presence
of AI keywords. Every candidate gets an additive base score across seven
transparent components, then multiplicative modifiers for availability, location,
and the JD's explicit disqualifiers. Honeypots and stuffers are driven to the
bottom. The full per-component breakdown is returned for grounded reasoning and
the live demo, and so the weights are defensible in the Stage-5 interview.
"""
from __future__ import annotations

from typing import Dict, List

from . import config, features

_PROF_W = {"beginner": 0.25, "intermediate": 0.5, "advanced": 0.8, "expert": 1.0}
_IR_GROUPS = ["retrieval_ranking", "embeddings", "vector_db", "nlp", "evaluation", "ml_core"]


# --------------------------------------------------------------------------- #
# Additive base components — each returns a value in [0, 1].
# --------------------------------------------------------------------------- #
def _title_role_fit(rec: dict) -> float:
    base = {"relevant": 1.0, "adjacent": 0.6, "other": 0.35,
            "offdomain": 0.15, "nontech": 0.05}[features.title_class(rec)]
    # A past ML/AI role lifts an otherwise-adjacent current title.
    for c in rec["career"]:
        if features._any_term(c["title_lower"], config.RELEVANT_TITLE_TERMS):
            base = max(base, 0.8)
            break
    return base


def _domain_evidence(rec: dict, ev: Dict[str, int], backed) -> float:
    distinct = sum(1 for g in _IR_GROUPS if ev.get(g, 0) > 0)
    total = sum(ev.get(g, 0) for g in _IR_GROUPS)
    return (
        0.50 * (distinct / len(_IR_GROUPS))
        + 0.30 * min(1.0, total / 8.0)
        + 0.20 * min(1.0, len(backed) / 3.0)
    )


def _must_have_coverage(rec: dict, ev: Dict[str, int], backed) -> float:
    satisfied = 0
    for groups in config.MUST_HAVES.values():
        if any(ev.get(g, 0) > 0 for g in groups) or any(g in backed for g in groups):
            satisfied += 1
    return satisfied / len(config.MUST_HAVES)


def _experience_band(yoe: float) -> float:
    if config.EXP_PEAK_LO <= yoe <= config.EXP_PEAK_HI:
        return 1.0
    if yoe < config.EXP_PEAK_LO:
        lo = config.EXP_OK_LO
        return max(0.0, (yoe - lo) / (config.EXP_PEAK_LO - lo)) * 0.7 + 0.3 if yoe >= lo \
            else max(0.0, yoe / lo) * 0.3
    hi = config.EXP_OK_HI
    return max(0.0, 1.0 - (yoe - config.EXP_PEAK_HI) / (hi - config.EXP_PEAK_HI)) * 0.6 + 0.4 \
        if yoe <= hi else 0.35


def _skill_trust(rec: dict) -> float:
    vals: List[float] = []
    for s in rec["skills"]:
        if not features._any_skill_is_ai(s):
            continue
        prof = _PROF_W.get(s["proficiency"], 0.4)
        dur = min(1.0, s["months"] / 24.0)
        assess = (s["assessment"] / 100.0) if s["assessment"] is not None else 0.5
        endo = min(1.0, s["endorsements"] / 50.0)
        vals.append(prof * (0.4 * dur + 0.4 * assess + 0.2 * endo))
    return (sum(vals) / len(vals)) if vals else 0.0


def _nice_to_have(rec: dict) -> float:
    hits = len(features._NICE_PATTERN.findall(rec["narrative_lower"]))
    return min(1.0, hits / 3.0)


# --------------------------------------------------------------------------- #
# Multiplicative modifiers.
# --------------------------------------------------------------------------- #
def _availability(rec: dict) -> float:
    sig = rec["signals"]
    d = rec["days_since_active"]
    recency = 1.0 if d is None else max(0.0, min(1.0, 1.0 - (d - 60) / 180.0)) if d > 60 else 1.0
    resp = sig.get("recruiter_response_rate", 0.0) or 0.0
    otw = 1.0 if sig.get("open_to_work_flag") else 0.6
    icr = sig.get("interview_completion_rate", 0.0) or 0.0
    raw = 0.40 * recency + 0.30 * resp + 0.15 * otw + 0.15 * icr
    m = config.MODIFIERS
    return m["availability_floor"] + (m["availability_ceil"] - m["availability_floor"]) * raw


def _location(rec: dict) -> float:
    cls = features.location_class(rec)
    m = config.MODIFIERS
    if cls == "preferred":
        return m["location_pref"]
    if cls == "welcome":
        return m["location_welcome"]
    return 1.0 if rec["signals"].get("willing_to_relocate") else m["location_far_norelocate"]


def _disqualifier_mult(rec: dict, ev: Dict[str, int]) -> Dict[str, float]:
    m = config.MODIFIERS
    out = {}
    if features.consulting_only(rec):
        out["consulting_only"] = m["consulting_only"]
    if features.title_class(rec) == "offdomain" and ev.get("nlp", 0) == 0:
        out["offdomain_only"] = m["offdomain_only"]
    if features._any_term(rec["narrative_lower"], config.RESEARCH_ONLY_TERMS) \
            and ev.get("ml_core", 0) == 0:
        out["research_only"] = m["research_only"]
    if features.is_title_hopper(rec):
        out["title_hopper"] = m["title_hopper"]
    return out


# --------------------------------------------------------------------------- #
# Top-level scoring.
# --------------------------------------------------------------------------- #
def score_candidate(rec: dict, trap: dict, sem_sim: float) -> Dict:
    """Return {'final': float, 'base': float, 'components': {...},
    'modifiers': {...}} for one candidate. sem_sim in [0,1]."""
    ev = features.evidence_groups(rec)
    backed = features.skill_evidence_groups(rec)

    comp = {
        "title_role_fit": _title_role_fit(rec),
        "domain_evidence": _domain_evidence(rec, ev, backed),
        "must_have_coverage": _must_have_coverage(rec, ev, backed),
        "semantic_similarity": max(0.0, min(1.0, sem_sim)),
        "experience_band": _experience_band(rec["yoe"]),
        "skill_trust": _skill_trust(rec),
        "nice_to_have": _nice_to_have(rec),
    }
    base = sum(config.WEIGHTS[k] * comp[k] for k in config.WEIGHTS)

    mods = {"availability": _availability(rec), "location": _location(rec)}
    mods.update(_disqualifier_mult(rec, ev))
    if trap["is_stuffer"]:
        mods["stuffer"] = config.MODIFIERS["stuffer"]
    if trap["is_honeypot"]:
        mods["honeypot"] = config.MODIFIERS["honeypot"]

    final = base
    for v in mods.values():
        final *= v

    return {
        "final": final,
        "base": base,
        "components": comp,
        "modifiers": mods,
        "evidence": ev,
        "backed": sorted(backed),
    }
