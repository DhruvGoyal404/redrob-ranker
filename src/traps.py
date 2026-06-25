"""
Validity / trap gate — runs FIRST, highest ROI.

Two jobs:
  1. honeypot detection: subtly *impossible* profiles (the spec's tier-0 set).
     Only clear internal contradictions are flagged, so we never demote a real
     candidate by accident (precision over recall — a missed honeypot just needs
     to lose on merit; a false positive ejects a genuine top-100 pick).
  2. keyword-stuffer detection: non-technical title + AI skills with no
     corroborating work evidence (the JD's "Marketing Manager with every AI
     keyword" trap).

Returns structured reasons so reasoning.py can explain and eval/ can score
trap-catch rate.
"""
from __future__ import annotations

from typing import Dict, List

from . import config, features


def honeypot_reasons(rec: dict) -> List[str]:
    """Return a list of impossibility reasons; empty list = not a honeypot."""
    reasons: List[str] = []
    yoe = rec["yoe"]
    yoe_months = yoe * 12

    # 1. "expert" proficiency but zero months of use.
    for s in rec["skills"]:
        if s["proficiency"] == "expert" and s["months"] == 0:
            reasons.append("expert_skill_with_zero_months")
            break

    # NOTE: we deliberately do NOT flag "skill months > career months" — in this
    # synthetic dataset skill durations are assigned independently of career length
    # for normal profiles (~9% of the pool), so it is noise, not an impossibility.
    # The exception (expert proficiency + huge duration on zero-experience) is
    # already covered by rule 1 and the experience/timeline checks below.

    # 3. a single role longer than the entire career.
    if yoe > 0 and rec["career"]:
        if max(c["months"] for c in rec["career"]) > yoe_months + 12:
            reasons.append("role_duration_exceeds_career")

    # 4. summed role months wildly exceed total experience (impossible overlap).
    if yoe > 0 and rec["total_career_months"] > yoe_months + 36:
        reasons.append("career_months_exceed_experience")

    # 5/6. per-role date sanity: end<start, or duration longer than the span.
    for c in rec["career"]:
        if c["start"] and c["end"]:
            if c["end"] < c["start"]:
                reasons.append("end_date_before_start_date")
            span = (c["end"] - c["start"]).days / 30.4
            if c["months"] > span + 3:
                reasons.append("role_duration_exceeds_date_span")

    # 7. claims more experience than the earliest start date allows.
    starts = [c["start"] for c in rec["career"] if c["start"]]
    if starts and yoe > 0:
        from datetime import date
        ref = date.fromisoformat(config.REFERENCE_DATE)
        implied_years = (ref - min(starts)).days / 365.25
        if yoe > implied_years + 2.0:
            reasons.append("experience_exceeds_timeline")

    # de-dup while preserving order
    seen, out = set(), []
    for r in reasons:
        if r not in seen:
            seen.add(r)
            out.append(r)
    return out


def stuffer_info(rec: dict) -> Dict:
    """Detect keyword-stuffer. Returns {is_stuffer, ai_skills, evidence}.

    A stuffer = non-technical current title + several AI skills + no corroborating
    evidence in the work narrative and no genuinely backed AI skill.
    """
    tclass = features.title_class(rec)
    n_ai = features.ai_skill_count(rec)
    work_ev = features.evidence_groups(rec)
    backed = features.skill_evidence_groups(rec)
    evidence_total = sum(work_ev.values())
    held_tech_role = features.has_relevant_or_adjacent_role(rec)

    # Stuffer = non-technical/off-domain person who has never held an ML/SWE role,
    # lists several AI skills, shows only incidental (<3 hits) AI language in their
    # actual work narrative, and has no platform-assessed AI skill to back it up.
    is_stuffer = (
        tclass in ("nontech", "offdomain")
        and n_ai >= 3
        and not held_tech_role
        and evidence_total < 3
        and len(backed) == 0
    )
    return {
        "is_stuffer": is_stuffer,
        "title_class": tclass,
        "ai_skill_count": n_ai,
        "evidence_total": evidence_total,
        "held_tech_role": held_tech_role,
        "backed_ai_skills": sorted(backed),
    }


def assess(rec: dict) -> Dict:
    """Full trap assessment for one record."""
    hp = honeypot_reasons(rec)
    st = stuffer_info(rec)
    return {
        "is_honeypot": len(hp) > 0,
        "honeypot_reasons": hp,
        **st,
    }
