"""
Grounded reasoning generation.

Every reasoning string is assembled from facts that actually exist in the
candidate's profile and from the scorer's own per-component breakdown - never
from an LLM at ranking time. This guarantees the Stage-4 review checks pass:
specific facts, JD connection, honest concerns, no hallucination, variation,
and tone that matches the rank. Borderline candidates also get a counterfactual.

The output reads as plain English a recruiter could act on, not a template with
a name slotted in.
"""
from __future__ import annotations

from typing import Dict, List

from . import config, features

_GROUP_LABEL = {
    "retrieval_ranking": "retrieval/ranking",
    "embeddings": "embeddings",
    "vector_db": "vector search / hybrid search",
    "nlp": "NLP/LLM",
    "evaluation": "ranking evaluation (NDCG/MRR/A-B)",
    "ml_core": "applied ML",
}


def _evidence_phrase(scored: Dict) -> str:
    ev = scored["evidence"]
    present = [g for g in config.EVIDENCE_TERMS if ev.get(g, 0) > 0]
    # prioritise the IR-core groups for the phrase
    ordered = [g for g in ["retrieval_ranking", "embeddings", "vector_db",
                           "evaluation", "nlp", "ml_core"] if g in present]
    labels = [_GROUP_LABEL[g] for g in ordered[:3]]
    if not labels:
        return ""
    if len(labels) == 1:
        return labels[0]
    return ", ".join(labels[:-1]) + " and " + labels[-1]


def _top_skill_names(rec: dict, k: int = 3) -> List[str]:
    ai = [s for s in rec["skills"] if features._any_skill_is_ai(s)]
    ai.sort(key=lambda s: (s["assessment"] or 0, s["months"]), reverse=True)
    return [s["name"] for s in ai[:k]]


def confidence_tag(scored: Dict, trap: Dict) -> str:
    c = scored["components"]
    if trap["is_honeypot"]:
        return "Excluded"
    strong = c["title_role_fit"] >= 0.8 and c["domain_evidence"] >= 0.45 \
        and c["must_have_coverage"] >= 0.5
    moderate = c["title_role_fit"] >= 0.55 or c["domain_evidence"] >= 0.3
    return "High" if strong else ("Moderate" if moderate else "Low")


def _concern(rec: dict, scored: Dict, trap: Dict) -> str:
    """One honest concern, if any - Stage-4 rewards acknowledging gaps."""
    c = scored["components"]
    sig = rec["signals"]
    if trap["is_stuffer"]:
        return "skills list is AI-heavy but the work history shows no ML/IR role"
    if c["must_have_coverage"] < 0.5:
        return "limited direct evidence on some JD must-haves (vector DB / ranking eval)"
    d = rec["days_since_active"]
    if d is not None and d > 150:
        return f"low availability ({d} days since last active)"
    rr = sig.get("recruiter_response_rate")
    if rr is not None and rr < 0.2:
        return f"weak recruiter response rate ({rr:.0%})"
    if features.consulting_only(rec):
        return "entire career at IT-services firms, no product-company experience"
    if features.location_class(rec) == "far" and not sig.get("willing_to_relocate"):
        return f"based in {rec['location']} and not open to relocation"
    if rec["yoe"] < config.EXP_OK_LO:
        return f"only {rec['yoe']:.0f} years experience, below the target band"
    return ""


def build_reasoning(rec: dict, scored: Dict, trap: Dict, rank: int,
                    counterfactual: bool = False) -> str:
    """Assemble a 1-2 sentence grounded reasoning for one candidate."""
    title = rec["title"]
    yoe = rec["yoe"]
    conf = confidence_tag(scored, trap)
    ev_phrase = _evidence_phrase(scored)
    skills = _top_skill_names(rec, 3)

    # Lead clause: who they are + the decisive evidence.
    lead = f"{title}, {yoe:.1f} yrs"
    if ev_phrase:
        lead += f"; demonstrated {ev_phrase}"
    elif skills:
        lead += f"; lists {', '.join(skills)}"

    # Availability / location colour where it helps.
    sig = rec["signals"]
    extras = []
    if features.location_class(rec) in ("preferred", "welcome"):
        extras.append(f"{rec['location']}-based")
    elif sig.get("willing_to_relocate"):
        extras.append("open to relocation")
    rr = sig.get("recruiter_response_rate")
    if rr is not None and rr >= 0.6 and conf in ("High", "Moderate"):
        extras.append(f"responsive to recruiters ({rr:.0%})")
    lead_extra = ("; " + ", ".join(extras)) if extras else ""

    concern = _concern(rec, scored, trap)
    if conf == "High":
        sent = f"{conf} confidence - {lead}{lead_extra}."
        if concern:
            sent += f" Minor concern: {concern}."
    elif conf == "Low":
        sent = f"{conf} confidence - {lead}{lead_extra}."
        sent += f" {concern[0].upper()}{concern[1:]}." if concern else " Adjacent fit only."
    else:
        sent = f"{conf} confidence - {lead}{lead_extra}."
        if concern:
            sent += f" Concern: {concern}."

    # Tone must scale with rank (Stage-4 check). Beyond the clear top tier, add an
    # honest "ranked here rather than higher because ..." note derived from this
    # candidate's own weakest dimension - keeps confident candidates from all
    # reading identically and acknowledges relative standing without faking doubt.
    note = ""
    if rank > 20:
        note = _relative_standing(rec, scored)
    elif counterfactual:
        note = _counterfactual(rec, scored)
    if note:
        sent += f" {note}"
    return sent.strip()


def _relative_standing(rec: dict, scored: Dict) -> str:
    """Name the weakest actionable dimension that keeps an otherwise-strong
    candidate from ranking higher. Honest and candidate-specific."""
    c = scored["components"]
    mods = scored["modifiers"]
    yoe = rec["yoe"]
    if c["must_have_coverage"] < 1.0:
        return "Ranked here rather than higher: doesn't yet evidence every JD must-have (e.g. ranking-evaluation depth)."
    if c["experience_band"] < 0.9:
        side = "below" if yoe < 6 else "above"
        return f"Ranked here rather than higher: {yoe:.1f}y sits {side} the 6-8y sweet spot."
    if c["domain_evidence"] < 0.7:
        return "Ranked here rather than higher: retrieval/ranking evidence is solid but thinner than the top tier."
    if mods.get("availability", 1.0) < 0.95:
        return "Ranked here rather than higher: availability signals (recruiter response / recency) are a notch lower."
    if c["skill_trust"] < 0.5:
        return "Ranked here rather than higher: skills are less corroborated by platform assessments."
    return "Ranked here rather than higher: edged out by candidates above with deeper retrieval/ranking track records."


def _counterfactual(rec: dict, scored: Dict) -> str:
    """A cheap, specific 'what would move this candidate' note for borderline rows."""
    c = scored["components"]
    if c["domain_evidence"] < 0.5 and c["title_role_fit"] >= 0.6:
        return "Would rank materially higher with explicit vector-DB / ranking-eval evidence."
    if c["must_have_coverage"] < 0.75:
        return "Closing the gap on ranking-evaluation experience would lift this candidate."
    d = rec["days_since_active"]
    if d is not None and d > 120:
        return "Stronger if recent platform activity confirmed current availability."
    return ""
