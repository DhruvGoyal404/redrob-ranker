"""
Redrob Ranker - live sandbox demo.

The magic moment first: pick or paste a job description, drop in (or use the
bundled) candidate sample, and watch a ranked, explained shortlist appear - each
row with a confidence tag, the specific evidence behind it, honest concerns, and
trap flags. A side panel shows how many honeypots and keyword-stuffers were caught
and demoted, and you can inject a stuffer to watch it sink.

This runs the SAME pipeline as rank.py (src/*), just on a small sample. Embeddings
are optional here (this is the sandbox, not the constrained ranking path).
Deploy target: Streamlit Community Cloud.
"""
import os
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_TORCH", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import json
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "app"))

from src import config, parse, traps, retrieve, reasoning
from src import score as scoring

# JD presets live in app/presets.py. Fall back to the single challenge JD if that
# module is unavailable for any reason, so the demo always boots.
try:
    from presets import PRESETS, DEFAULT_PRESET
except Exception:
    from src.jd import JD_QUERY
    PRESETS = {"Senior AI Engineer (challenge JD)": JD_QUERY}
    DEFAULT_PRESET = "Senior AI Engineer (challenge JD)"

st.set_page_config(page_title="Redrob Ranker", page_icon="🧭", layout="wide")
REF = date.fromisoformat(config.REFERENCE_DATE)
SAMPLE = _ROOT / "data" / "sample_candidates.json"
_CONF_COLOR = {"High": "#1b5e20", "Moderate": "#7a5c00", "Low": "#5a1e1e", "Excluded": "#4a148c"}


@st.cache_resource(show_spinner=False)
def get_model():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(config.EMBED_MODEL, device="cpu")


@st.cache_data(show_spinner=False)
def _embed(texts):
    return get_model().encode(list(texts), normalize_embeddings=True, convert_to_numpy=True)


def synthetic_stuffer():
    """A textbook keyword-stuffer: non-technical title, every AI buzzword as a
    skill, but a work history with no ML evidence. The ranker should demote it."""
    return {
        "candidate_id": "CAND_9999999",
        "profile": {
            "anonymized_name": "Demo Stuffer", "headline": "HR Manager | AI enthusiast",
            "summary": "HR Manager with 8 years across recruitment, payroll and "
                       "employee engagement. Passionate about AI.",
            "location": "Noida, Uttar Pradesh", "country": "India",
            "years_of_experience": 8.0, "current_title": "HR Manager",
            "current_company": "SomeCorp", "current_company_size": "201-500",
            "current_industry": "Human Resources",
        },
        "career_history": [{
            "company": "SomeCorp", "title": "HR Manager",
            "start_date": "2018-01-01", "end_date": None, "duration_months": 96,
            "is_current": True, "industry": "Human Resources",
            "company_size": "201-500",
            "description": "Led recruitment, onboarding, payroll and employee "
                           "engagement programs. Managed a team of HR generalists.",
        }],
        "education": [{"institution": "Some University", "degree": "MBA",
                       "field_of_study": "Human Resources", "start_year": 2014,
                       "end_year": 2016, "grade": None, "tier": "tier_3"}],
        "skills": [{"name": n, "proficiency": "expert", "endorsements": 40,
                    "duration_months": 36}
                   for n in ["RAG", "Pinecone", "Vector Search", "LLM", "Embeddings",
                             "Transformers", "Hugging Face Transformers", "PyTorch"]],
        "redrob_signals": {
            "profile_completeness_score": 95, "signup_date": "2023-01-01",
            "last_active_date": "2026-05-20", "open_to_work_flag": True,
            "profile_views_received_30d": 50, "applications_submitted_30d": 10,
            "recruiter_response_rate": 0.9, "avg_response_time_hours": 2.0,
            "skill_assessment_scores": {}, "connection_count": 500,
            "endorsements_received": 300, "notice_period_days": 30,
            "expected_salary_range_inr_lpa": {"min": 20, "max": 30},
            "preferred_work_mode": "hybrid", "willing_to_relocate": True,
            "github_activity_score": -1, "search_appearance_30d": 40,
            "saved_by_recruiters_30d": 5, "interview_completion_rate": 0.9,
            "offer_acceptance_rate": -1, "verified_email": True,
            "verified_phone": True, "linkedin_connected": True,
        },
    }


def load_records(uploaded, inject_stuffer):
    if uploaded is not None:
        text = uploaded.getvalue().decode("utf-8")
        raws = ([json.loads(l) for l in text.splitlines() if l.strip()]
                if uploaded.name.endswith(".jsonl") else json.loads(text))
    elif SAMPLE.exists():
        raws = json.loads(SAMPLE.read_text(encoding="utf-8"))
    else:
        return []
    if isinstance(raws, dict):
        raws = [raws]
    if inject_stuffer:
        raws = raws + [synthetic_stuffer()]
    return [parse.normalize(r, REF) for r in raws]


def run_ranking(records, jd_text):
    narratives = [r["narrative"] for r in records]
    try:                                    # full hybrid: BM25 + dense bge-small (PyTorch)
        doc_emb, jd_emb = _embed(tuple(narratives)), _embed((jd_text,))
    except Exception:                       # graceful fallback if the model can't load
        doc_emb = jd_emb = None
        st.warning("Embedding model unavailable - ranking with BM25 + the signal scorer.")
    idx, dense_norm = retrieve.shortlist(narratives, jd_text, doc_emb, jd_emb, size=len(records))
    rows = []
    for i in idx:
        rec = records[i]
        trap = traps.assess(rec)
        sc = scoring.score_candidate(rec, trap, float(dense_norm[i]))
        rows.append((rec, trap, sc))
    rows.sort(key=lambda x: -x[2]["final"])
    return rows


# ---------------------------------------------------------------- UI
st.title("🧭 Redrob Intelligent Candidate Ranker")
st.caption("Senior AI Engineer - Founding Team | evidence-based ranking, not keyword matching")

st.info(
    "**This demo runs the full hybrid pipeline live** - BM25 + dense `bge-small` embeddings "
    "fused with RRF, then the 7-component signal scorer + trap gate - the same pipeline that "
    "produced the submission. Edit the JD below and re-rank to see the semantic match respond. "
    "(First run loads the embedding model, ~30 s.)", icon="🧭")

with st.sidebar:
    st.header("How it works")
    st.markdown(
        "1. **Trap gate** - honeypots & keyword-stuffers detected and demoted.\n"
        "2. **Hybrid retrieval** - BM25 + dense embeddings fused with RRF.\n"
        "3. **Signal scorer** - 7 transparent components x availability/location/"
        "disqualifier modifiers.\n"
        "4. **Grounded reasoning** - facts from the profile + confidence + concerns."
    )
    st.divider()
    inject_stuffer = st.checkbox(
        "Inject a keyword-stuffer", value=False,
        help="Adds one HR-Manager-with-every-AI-skill profile so you can watch the "
             "trap gate demote it.")
    uploaded = st.file_uploader("Candidate sample (.json or .jsonl)", type=["json", "jsonl"])
    top_n = st.slider("Show top N", 5, 50, 15)

# This demo ranks for the challenge's single target role (the Senior AI Engineer JD).
# Our signal scorer is TUNED to this role's must-haves (read from src/config.py), so we show
# that JD honestly rather than imply the demo adapts to arbitrary roles - it does not, and
# pretending so would be a different, weaker keyword ranker. Editing the text only shifts the
# optional semantic-embeddings component (14% of the score), not the tuned signal scorer.
jd_text = st.text_area(
    'Target role we rank for - the challenge\'s "Senior AI Engineer, Founding Team" JD',
    PRESETS[DEFAULT_PRESET], height=160)
st.caption("Edit the JD and re-rank - the dense semantic-match component responds. Because we "
           "score *demonstrated evidence*, not raw keyword overlap, the order shifts thoughtfully "
           "rather than swinging on cosmetic wording (a pure keyword-matcher would swing).")

if st.button("⚡ Rank candidates", type="primary"):
    records = load_records(uploaded, inject_stuffer)
    if not records:
        st.error("No candidates loaded. Upload a sample or add data/sample_candidates.json.")
        st.stop()
    with st.spinner(f"Ranking {len(records)} candidates ..."):
        st.session_state.ranked = run_ranking(records, jd_text)

# ---- results render OUTSIDE the button block so widgets below stay interactive ----
if "ranked" in st.session_state:
    ranked = st.session_state.ranked
    n_hp = sum(1 for _, t, _ in ranked if t["is_honeypot"])
    n_st = sum(1 for _, t, _ in ranked if t["is_stuffer"])

    m1, m2, m3 = st.columns(3)
    m1.metric("Candidates scored", len(ranked))
    m2.metric("Keyword-stuffers demoted", n_st)
    m3.metric("Honeypots demoted", n_hp)

    table = []
    for pos, (rec, trap, sc) in enumerate(ranked[:top_n], 1):
        why = reasoning.build_reasoning(rec, sc, trap, pos)
        conf = reasoning.confidence_tag(sc, trap)
        flags = ("🚫 honeypot" if trap["is_honeypot"] else "") + \
                ("  ⚠️ stuffer" if trap["is_stuffer"] else "")
        table.append({"rank": pos, "candidate_id": rec["candidate_id"],
                      "title": rec["title"], "yoe": rec["yoe"],
                      "confidence": conf, "score": round(sc["final"], 3),
                      "flags": flags.strip(), "reasoning": why})
    df = pd.DataFrame(table)

    def _row_style(row):
        bg = "background-color: rgba(120,30,30,0.35)" if row["flags"] else ""
        return [bg] * len(row)

    def _conf_style(val):
        return f"color: white; background-color: {_CONF_COLOR.get(val, '')}"

    styled = (df.style.apply(_row_style, axis=1)
              .map(_conf_style, subset=["confidence"]))
    st.dataframe(styled, use_container_width=True, hide_index=True)

    st.download_button(
        "⬇ Download ranked CSV",
        pd.DataFrame([{"candidate_id": r["candidate_id"], "rank": r["rank"],
                       "score": r["score"], "reasoning": r["reasoning"]} for r in table]
                     ).to_csv(index=False).encode("utf-8"),
        "ranked_sample.csv", "text/csv")

    st.subheader("Why this candidate? (score breakdown)")
    pick = st.selectbox("Inspect a candidate", [r["candidate_id"] for r in table])
    rec, trap, sc = next(x for x in ranked if x[0]["candidate_id"] == pick)
    st.markdown(f"**{rec['title']}** | {rec['yoe']:.1f} yrs | {rec['location']}  \n"
                f"_{reasoning.build_reasoning(rec, sc, trap, 1)}_")
    c1, c2 = st.columns([3, 2])
    with c1:
        st.markdown("**Weighted contribution per component**")
        contrib = {k: round(config.WEIGHTS[k] * v, 4) for k, v in sc["components"].items()}
        st.bar_chart(pd.Series(contrib, name="contribution"))
    with c2:
        st.markdown("**Multiplicative modifiers**")
        st.dataframe(pd.DataFrame([{"modifier": k, "x": round(v, 3)}
                                   for k, v in sc["modifiers"].items()]), hide_index=True)
        st.metric("Final score", round(sc["final"], 4))
else:
    st.info("Pick a job description above (or paste your own) and hit **Rank candidates**.")
