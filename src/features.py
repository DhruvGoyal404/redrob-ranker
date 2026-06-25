"""
Shared feature extraction — term matching used by the trap gate, the scorer and
the reasoning generator so all three read the profile the same way.
"""
from __future__ import annotations

import re
from typing import Dict, List, Set

from . import config


def _term_pattern(term: str) -> str:
    """Word-boundary regex for one term. Short/ambiguous tokens (<=3 chars, e.g.
    'rag', 'ner', 'nlp', 'llm', 'e5') require both-side boundaries so they match
    real tokens, not substrings inside words like 'average' or 'owner'. Longer
    terms use a left boundary + prefix so morphological variants match
    (rank/ranking/ranked, embedding/embeddings, retrieval)."""
    esc = re.escape(term.strip())
    if len(term.strip()) <= 3:
        return r"\b" + esc + r"\b"
    left = r"\b" if term[0].isalnum() else ""
    return left + esc


def _compile_group(terms: List[str]) -> "re.Pattern":
    return re.compile("|".join(_term_pattern(t) for t in terms), re.IGNORECASE)


# Precompile once at import.
_EVIDENCE_PATTERNS = {g: _compile_group(t) for g, t in config.EVIDENCE_TERMS.items()}
_NICE_PATTERN = _compile_group(config.NICE_TO_HAVE_TERMS)


def _any_term(text: str, terms: List[str]) -> bool:
    return any(t in text for t in terms)


def title_class(rec: dict) -> str:
    """Classify the current title: 'relevant' | 'adjacent' | 'nontech' |
    'offdomain' | 'other'. Order matters: relevant wins over adjacent."""
    t = rec["title_lower"]
    if _any_term(t, config.RELEVANT_TITLE_TERMS):
        return "relevant"
    if _any_term(t, config.OFFDOMAIN_TITLE_TERMS):
        return "offdomain"
    if _any_term(t, config.NONTECH_TITLE_TERMS):
        return "nontech"
    if _any_term(t, config.ADJACENT_TITLE_TERMS):
        return "adjacent"
    return "other"


def evidence_groups(rec: dict) -> Dict[str, int]:
    """Count evidence hits per group across career descriptions + headline +
    summary (the *work narrative*), NOT the bare skills list. This is the
    'outcome evidence, not vocabulary' read that catches plain-language Tier-5s
    and ignores keyword stuffing in the skills array."""
    # Narrative text = headline + summary + every role description.
    work_text = " ".join(
        [rec["headline"].lower(), rec["summary"].lower()]
        + [c["description_lower"] for c in rec["career"]]
        + [c["title_lower"] for c in rec["career"]]
    )
    out: Dict[str, int] = {}
    for group, pat in _EVIDENCE_PATTERNS.items():
        out[group] = len(pat.findall(work_text))
    return out


def skill_evidence_groups(rec: dict) -> Set[str]:
    """Evidence groups that appear as *genuinely assessed* skills — an AI/IR skill
    with a real Redrob assessment score >= 60. Self-reported duration alone does
    NOT count as backing (stuffers fake duration); the platform assessment is the
    trustworthy signal per redrob_signals_doc."""
    backed: Set[str] = set()
    for s in rec["skills"]:
        if s["assessment"] is None or s["assessment"] < 60:
            continue
        nm = s["name_lower"]
        for group, pat in _EVIDENCE_PATTERNS.items():
            if pat.search(nm):
                backed.add(group)
    return backed


def _any_skill_is_ai(skill: dict) -> bool:
    """True if a single skill entry names an AI/ML/IR skill."""
    return any(pat.search(skill["name_lower"]) for pat in _EVIDENCE_PATTERNS.values())


def ai_skill_count(rec: dict) -> int:
    """Number of distinct AI/ML/IR-flavoured skills listed (any group)."""
    n = 0
    for s in rec["skills"]:
        nm = s["name_lower"]
        if any(pat.search(nm) for pat in _EVIDENCE_PATTERNS.values()):
            n += 1
    return n


def consulting_only(rec: dict) -> bool:
    """True if every company in the career history is an IT-services/consulting
    firm (JD explicit disqualifier) — and there is at least one job."""
    if not rec["companies"]:
        return False
    def is_consult(c):
        return any(f in c for f in config.CONSULTING_FIRMS)
    return all(is_consult(c) for c in rec["companies"])


def location_class(rec: dict) -> str:
    loc = rec["location_lower"]
    if _any_term(loc, config.PREFERRED_LOCATIONS):
        return "preferred"
    if _any_term(loc, config.WELCOME_LOCATIONS):
        return "welcome"
    return "far"


def has_relevant_or_adjacent_role(rec: dict) -> bool:
    """True if any role in the career history (or the current title) is a
    relevant or adjacent technical title — i.e. the person has actually held an
    ML/SWE-type role at some point, not just listed skills."""
    titles = [rec["title_lower"]] + [c["title_lower"] for c in rec["career"]]
    for t in titles:
        if _any_term(t, config.RELEVANT_TITLE_TERMS) or _any_term(t, config.ADJACENT_TITLE_TERMS):
            return True
    return False


def is_title_hopper(rec: dict) -> bool:
    """JD: switching companies every ~1.5y chasing titles. Flag short average
    tenure across several completed jobs."""
    completed = [c for c in rec["career"] if not c["is_current"] and c["months"] > 0]
    return len(completed) >= 3 and rec["avg_tenure_months"] < 18
