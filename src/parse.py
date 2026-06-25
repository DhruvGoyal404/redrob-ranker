"""
Stream-parse candidates.jsonl into compact, normalized records.

Handles both plain .jsonl and gzipped .jsonl.gz. Designed to be memory-safe
(line-by-line) and fast — no third-party deps. The normalized record exposes
everything the trap gate, scorer and reasoning need, with the raw profile kept
for grounded reasoning / fact-checking.
"""
from __future__ import annotations

import gzip
import io
import json
from datetime import date
from typing import Dict, Iterator, List


def _open(path: str):
    if str(path).endswith(".gz"):
        return io.TextIOWrapper(gzip.open(path, "rb"), encoding="utf-8")
    return open(path, "r", encoding="utf-8")


def iter_raw(path: str) -> Iterator[dict]:
    """Yield raw candidate dicts one per line."""
    with _open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def _pdate(s):
    try:
        return date.fromisoformat(s)
    except (TypeError, ValueError):
        return None


def build_narrative(raw: dict) -> str:
    """Career-narrative document for retrieval: headline + summary + role
    descriptions + skill names. This is what BM25 and the embedder see."""
    p = raw.get("profile", {})
    parts: List[str] = [p.get("headline", ""), p.get("summary", "")]
    for h in raw.get("career_history", []):
        parts.append(h.get("title", ""))
        parts.append(h.get("description", ""))
    parts.append(" ".join(s.get("name", "") for s in raw.get("skills", [])))
    return " ".join(x for x in parts if x).strip()


def normalize(raw: dict, reference_date: date) -> dict:
    """Return a normalized record with derived fields used downstream."""
    p = raw.get("profile", {})
    sig = raw.get("redrob_signals", {})
    assess = sig.get("skill_assessment_scores", {}) or {}
    assess_lower = {k.lower(): v for k, v in assess.items()}

    skills = []
    for s in raw.get("skills", []):
        name = s.get("name", "")
        skills.append({
            "name": name,
            "name_lower": name.lower(),
            "proficiency": s.get("proficiency", ""),
            "months": s.get("duration_months", 0) or 0,
            "endorsements": s.get("endorsements", 0) or 0,
            "assessment": assess_lower.get(name.lower()),
        })

    career = []
    companies = []
    for h in raw.get("career_history", []):
        sd, ed = _pdate(h.get("start_date")), _pdate(h.get("end_date"))
        career.append({
            "company": h.get("company", ""),
            "company_lower": h.get("company", "").lower(),
            "title": h.get("title", ""),
            "title_lower": h.get("title", "").lower(),
            "start": sd,
            "end": ed,
            "months": h.get("duration_months", 0) or 0,
            "is_current": bool(h.get("is_current")),
            "industry": h.get("industry", ""),
            "company_size": h.get("company_size", ""),
            "description": h.get("description", ""),
            "description_lower": h.get("description", "").lower(),
        })
        companies.append(h.get("company", "").lower())

    narrative = build_narrative(raw)

    rec = {
        "candidate_id": raw.get("candidate_id", ""),
        "name": p.get("anonymized_name", ""),
        "headline": p.get("headline", ""),
        "summary": p.get("summary", ""),
        "title": p.get("current_title", ""),
        "title_lower": p.get("current_title", "").lower(),
        "company": p.get("current_company", ""),
        "company_size": p.get("current_company_size", ""),
        "industry": p.get("current_industry", ""),
        "location": p.get("location", ""),
        "location_lower": p.get("location", "").lower(),
        "country": p.get("country", ""),
        "yoe": float(p.get("years_of_experience", 0) or 0),
        "skills": skills,
        "career": career,
        "companies": companies,
        "education": raw.get("education", []),
        "signals": sig,
        "narrative": narrative,
        "narrative_lower": narrative.lower(),
        # raw kept for grounded reasoning / hallucination checks
        "_raw": raw,
    }

    # --- derived career aggregates ---
    rec["total_career_months"] = sum(c["months"] for c in career)
    rec["num_jobs"] = len(career)
    completed = [c for c in career if not c["is_current"] and c["months"] > 0]
    rec["avg_tenure_months"] = (
        sum(c["months"] for c in completed) / len(completed) if completed else 0.0
    )
    # recency of activity in days
    la = _pdate(sig.get("last_active_date"))
    rec["days_since_active"] = (reference_date - la).days if la else None
    return rec


def load_all(path: str, reference_date: date) -> List[dict]:
    return [normalize(r, reference_date) for r in iter_raw(path)]
