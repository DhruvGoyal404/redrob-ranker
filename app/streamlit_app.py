"""
Redrob Intelligent Candidate Ranker - live product demo.

A general "rank candidates for any role" engine: pick or paste a job description and a
ranked, explained shortlist appears - relevance to the JD (dense bge-small embeddings +
BM25) drives the order, modulated by demonstrated-skill quality and behavioral
availability, and gated by a trap detector that demotes honeypots and keyword-stuffers.
Every pick carries a confidence tag and a grounded one-line justification.

The ranking here (app/demo_rank.py) is JD-ADAPTIVE so switching roles re-ranks visibly.
The competition submission (rank.py) uses our scorer tuned to the one challenge role and
is unchanged by this demo.
"""
import os
os.environ.setdefault("USE_TF", "0")
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

from src import config, parse, traps, retrieve

try:
    from presets import PRESETS, DEFAULT_PRESET
    import demo_rank
except Exception:                                   # pragma: no cover - defensive boot
    from src.jd import JD_QUERY
    PRESETS = {"Senior AI Engineer (challenge role)": JD_QUERY}
    DEFAULT_PRESET = "Senior AI Engineer (challenge role)"
    import demo_rank

st.set_page_config(page_title="Redrob Candidate Ranker", page_icon="🧭", layout="wide")
REF = date.fromisoformat(config.REFERENCE_DATE)
SAMPLE = _ROOT / "data" / "demo_candidates.json"   # 50-row challenge sample + role-diverse demo profiles
if not SAMPLE.exists():
    SAMPLE = _ROOT / "data" / "sample_candidates.json"
_CONF_COLOR = {"High": "#155e3b", "Moderate": "#7a5c00", "Low": "#6e2222", "Excluded": "#43306e"}

st.markdown("""
<style>
  #MainMenu, footer {visibility: hidden;}
  div[data-testid="stToolbar"] {visibility: hidden; height: 0;}
  .block-container {padding-top: 2.2rem; max-width: 1180px;}
  h1 {font-weight: 800; letter-spacing: -0.5px;}
  div[data-testid="stMetricValue"] {font-size: 2rem;}
  section[data-testid="stSidebar"] {border-right: 1px solid rgba(255,255,255,.06);}
</style>
""", unsafe_allow_html=True)


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
    """JD-adaptive ranking: dense semantic + BM25 relevance to the typed JD, modulated by
    universal quality and gated by the trap detector (see app/demo_rank.py)."""
    narratives = [r["narrative"] for r in records]
    try:                                              # dense semantic relevance (PyTorch)
        doc_emb, jd_emb = _embed(tuple(narratives)), _embed((jd_text,))
        semantic = retrieve.dense_scores(doc_emb, jd_emb)
    except Exception:                                 # graceful: BM25-only relevance
        semantic = np.zeros(len(records), dtype=float)
        st.warning("Embedding model unavailable - ranking on BM25 keyword relevance + quality.")
    bm25 = retrieve.bm25_scores(narratives, jd_text)
    traps_list = [traps.assess(r) for r in records]
    return demo_rank.rank(records, traps_list, semantic, bm25)


# ============================================================================== UI
st.title("🧭 Redrob Intelligent Candidate Ranker")
st.markdown("Rank candidates for **any role** by *evidence of the right work* and relevance to "
            "the job description - with keyword-trap detection and a grounded reason for every pick.")

with st.expander("How the ranking works"):
    st.markdown(
        "- **Relevance to the JD** (dense `bge-small` embeddings + BM25 keyword match) is the "
        "primary signal - editing the role re-ranks the list.\n"
        "- **Demonstrated-skill quality** (proficiency x platform assessment x endorsements) and "
        "**behavioral availability** (recruiter responsiveness, recency) modulate it.\n"
        "- A **trap gate** drives honeypots (internally impossible profiles) and keyword-stuffers "
        "(non-technical profiles padded with AI skills) to the bottom, regardless of keyword match.\n"
        "- The competition `submission.csv` uses our scorer tuned to the one challenge role; this "
        "demo is the general, role-agnostic product."
    )

with st.sidebar:
    st.subheader("Demo controls")
    inject_stuffer = st.toggle(
        "Inject a keyword-stuffer", value=False,
        help="Adds one HR-Manager-with-every-AI-skill profile to show the trap gate demote it.")
    top_n = st.slider("Results to show", 5, 50, 15)
    uploaded = st.file_uploader("Use your own candidates (.json / .jsonl)",
                                type=["json", "jsonl"])
    st.caption("Empty = the bundled 50-profile sample.")

# ---- Role / JD selector: pick a preset OR "Custom" to paste your own; box is editable ----
CUSTOM = "✏️  Custom - paste your own JD"
names = [CUSTOM] + list(PRESETS)
st.session_state.setdefault("jd_text", PRESETS[DEFAULT_PRESET])
left, right = st.columns([1, 2])
with left:
    sel = st.selectbox("Role  ·  or pick Custom →", names,
                       index=names.index(DEFAULT_PRESET), key="preset_select")
    if st.session_state.get("_last_preset") != sel:
        st.session_state._last_preset = sel
        st.session_state.jd_text = "" if sel == CUSTOM else PRESETS[sel]
    go = st.button("⚡ Rank candidates", type="primary", use_container_width=True)
with right:
    st.text_area("Job description  ·  edit this, or pick **Custom** to paste any role",
                 key="jd_text", height=190,
                 placeholder="Paste any job description here, then hit Rank candidates...")
jd_text = st.session_state.jd_text

if go:
    records = load_records(uploaded, inject_stuffer)
    if not records:
        st.error("No candidates loaded. Upload a sample or add data/sample_candidates.json.")
        st.stop()
    with st.spinner(f"Ranking {len(records)} candidates for this role ..."):
        st.session_state.ranked = run_ranking(records, jd_text)

# ---- results render OUTSIDE the button block so the inspector below stays interactive ----
if "ranked" in st.session_state:
    ranked = st.session_state.ranked
    n_hp = sum(1 for _, t, _ in ranked if t["is_honeypot"])
    n_st = sum(1 for _, t, _ in ranked if t["is_stuffer"])

    rels = [info["jd_relevance"] for _, t, info in ranked
            if not (t["is_honeypot"] or t["is_stuffer"])]
    max_rel = max(rels) if rels else 1.0
    n_strong = sum(1 for r in rels if r >= 0.55 * max_rel)   # within 55% of the best fit

    st.divider()
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Candidates scored", len(ranked))
    m2.metric("Strong fits for this role", n_strong,
              help="Non-trap candidates whose relevance to THIS job description is within 55% "
                   "of the best match - it changes as you switch roles (a deeper talent pool "
                   "for some roles than others).")
    m3.metric("Keyword-stuffers demoted", n_st)
    m4.metric("Honeypots demoted", n_hp)

    table = []
    for pos, (rec, trap, info) in enumerate(ranked[:top_n], 1):
        conf = demo_rank.confidence(info, trap)
        why = demo_rank.reasoning(rec, info, trap, conf)
        flags = ("🚫 honeypot" if trap["is_honeypot"] else "") + \
                ("  ⚠️ stuffer" if trap["is_stuffer"] else "")
        table.append({"rank": pos, "candidate_id": rec["candidate_id"], "title": rec["title"],
                      "yoe": rec["yoe"], "confidence": conf,
                      "JD match": round(info["jd_relevance"], 3),
                      "score": round(info["final"], 3), "flags": flags.strip(), "reasoning": why})
    df = pd.DataFrame(table)

    def _row_style(row):
        return ["background-color: rgba(120,30,30,0.35)" if row["flags"] else ""] * len(row)

    def _conf_style(val):
        return f"color: white; background-color: {_CONF_COLOR.get(val, '')}"

    styled = df.style.apply(_row_style, axis=1).map(_conf_style, subset=["confidence"])
    st.dataframe(styled, use_container_width=True, hide_index=True)

    st.download_button(
        "⬇ Download ranked CSV",
        pd.DataFrame([{"candidate_id": r["candidate_id"], "rank": r["rank"],
                       "score": r["score"], "reasoning": r["reasoning"]} for r in table]
                     ).to_csv(index=False).encode("utf-8"),
        "ranked_candidates.csv", "text/csv")

    st.subheader("Why this candidate?")
    pick = st.selectbox("Inspect a candidate", [r["candidate_id"] for r in table])
    rec, trap, info = next(x for x in ranked if x[0]["candidate_id"] == pick)
    conf = demo_rank.confidence(info, trap)
    st.markdown(f"**{rec['title']}** &nbsp;·&nbsp; {rec['yoe']:.1f} yrs &nbsp;·&nbsp; "
                f"{rec.get('location', 'n/a')}  \n_{demo_rank.reasoning(rec, info, trap, conf)}_")
    c1, c2 = st.columns([3, 2])
    with c1:
        st.markdown("**What drove the score** (each in 0-1)")
        drivers = {"JD relevance": info["jd_relevance"], "  · semantic": info["semantic"],
                   "  · BM25 keyword": info["bm25"], "skill trust": info["skill_trust"],
                   "experience fit": info["experience_band"]}
        st.bar_chart(pd.Series(drivers, name="signal"))
    with c2:
        st.markdown("**Multipliers**")
        st.dataframe(pd.DataFrame([
            {"factor": "availability", "x": round(info["availability"], 3)},
            {"factor": "honeypot" if trap["is_honeypot"] else
                       "stuffer" if trap["is_stuffer"] else "trap gate",
             "x": 0.001 if trap["is_honeypot"] else 0.05 if trap["is_stuffer"] else 1.0},
        ]), hide_index=True)
        st.metric("Final score", round(info["final"], 4))
else:
    st.info("Pick a role (or paste a JD) and hit **Rank candidates**.")
