"""Tiny helpers to build schema-valid candidate records for tests."""
from datetime import date

from src import parse

REF = date(2026, 6, 1)


def _signals(**over):
    base = {
        "profile_completeness_score": 90, "signup_date": "2022-01-01",
        "last_active_date": "2026-05-15", "open_to_work_flag": True,
        "profile_views_received_30d": 20, "applications_submitted_30d": 3,
        "recruiter_response_rate": 0.7, "avg_response_time_hours": 6.0,
        "skill_assessment_scores": {}, "connection_count": 300,
        "endorsements_received": 100, "notice_period_days": 30,
        "expected_salary_range_inr_lpa": {"min": 30, "max": 50},
        "preferred_work_mode": "hybrid", "willing_to_relocate": True,
        "github_activity_score": 60, "search_appearance_30d": 15,
        "saved_by_recruiters_30d": 3, "interview_completion_rate": 0.9,
        "offer_acceptance_rate": 0.5, "verified_email": True,
        "verified_phone": True, "linkedin_connected": True,
    }
    base.update(over)
    return base


def make_raw(candidate_id="CAND_0000001", title="ML Engineer", yoe=6.0,
             location="Pune, Maharashtra", skills=None, career=None,
             summary="", signals=None):
    return {
        "candidate_id": candidate_id,
        "profile": {
            "anonymized_name": "Test", "headline": title, "summary": summary,
            "location": location, "country": "India", "years_of_experience": yoe,
            "current_title": title, "current_company": "Acme",
            "current_company_size": "201-500", "current_industry": "Software",
        },
        "career_history": career or [{
            "company": "Acme", "title": title, "start_date": "2019-01-01",
            "end_date": None, "duration_months": int(yoe * 12), "is_current": True,
            "industry": "Software", "company_size": "201-500",
            "description": summary or "Worked on software.",
        }],
        "education": [], "skills": skills or [],
        "redrob_signals": _signals(**(signals or {})),
    }


def normalize(raw):
    return parse.normalize(raw, REF)


def skill(name, proficiency="advanced", months=24, endorsements=20):
    return {"name": name, "proficiency": proficiency, "endorsements": endorsements,
            "duration_months": months}


def with_assessments(raw, scores):
    """Set Redrob platform assessment scores (skill_name -> 0..100)."""
    raw["redrob_signals"]["skill_assessment_scores"] = scores
    return raw
