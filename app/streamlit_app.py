"""
Redrob Ranker — live sandbox demo.

The magic moment first: paste a job description, drop in (or use the bundled)
candidate sample, and watch a ranked, *explained* shortlist appear — with the
per-component score breakdown, confidence tag, honest concerns, and trap flags
for every candidate. Then the sidebar explains the mechanics.

This runs the SAME pipeline as rank.py (src/*), just on a small sample, and
embeds the sample at runtime (network allowed here — this is the sandbox, not the
constrained ranking path). Deploy target: Streamlit Community Cloud.
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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import config, parse, traps, retrieve, reasoning
from src import score as scoring
from src.jd import JD_QUERY

st.set_page_config(page_title="Redrob Ranker", layout="wide")
REF = date.fromisoformat(config.REFERENCE_DATE)
SAMPLE = Path(__file__).resolve().parent.parent / "data" / "sample_candidates.json"


@st.cache_resource(show_spinner=False)
def get_model():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(config.EMBED_MODEL, device="cpu")


def load_sample_records(uploaded, use_semantic):
    if uploaded is not None:
        text = uploaded.getvalue().decode("utf-8")
        raws = ([json.loads(l) for l in text.splitlines() if l.strip()]
                if uploaded.name.endswith(".jsonl")
                else json.loads(text))
    elif SAMPLE.exists():
        raws = json.loads(SAMPLE.read_text(encoding="utf-8"))
    else:
        return []
    if isinstance(raws, dict):
        raws = [raws]
    return [parse.normalize(r, REF) for r in raws]


def run_ranking(records, jd_text, use_semantic):
    narratives = [r["narrative"] for r in records]
    doc_emb = jd_emb = None
    if use_semantic:
        model = get_model()
        doc_emb = model.encode(narratives, normalize_embeddings=True, convert_to_numpy=True)
        jd_emb = model.encode([jd_text], normalize_embeddings=True, convert_to_numpy=True)
    idx, dense_norm = retrieve.shortlist(
        narratives, jd_text, doc_emb, jd_emb, size=len(records))
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
st.caption("Senior AI Engineer — Founding Team · evidence-based ranking, not keyword matching")

with st.sidebar:
    st.header("How it works")
    st.markdown(
        "1. **Trap gate** — honeypots & keyword-stuffers detected and demoted.\n"
        "2. **Hybrid retrieval** — BM25 + dense embeddings fused with RRF.\n"
        "3. **Signal scorer** — 7 transparent components × availability/location/"
        "disqualifier modifiers.\n"
        "4. **Grounded reasoning** — facts from the profile + confidence + concerns."
    )
    use_semantic = st.checkbox(
        "Use semantic embeddings (bge-small)", value=False,
        help="Off by default for reliability on free 1 GB hosting. Turn on to add "
             "the dense layer (loads PyTorch — heavier). The full ranker still runs "
             "via BM25 + the signal scorer when off.")
    uploaded = st.file_uploader("Candidate sample (.json or .jsonl)", type=["json", "jsonl"])
    top_n = st.slider("Show top N", 5, 50, 15)

jd_text = st.text_area("Job description / query", JD_QUERY, height=140)

if st.button("⚡ Rank candidates", type="primary"):
    records = load_sample_records(uploaded, use_semantic)
    if not records:
        st.error("No candidates loaded. Upload a sample or add data/sample_candidates.json.")
        st.stop()
    with st.spinner(f"Ranking {len(records)} candidates ..."):
        ranked = run_ranking(records, jd_text, use_semantic)

    st.success(f"Ranked {len(records)} candidates. Top {top_n}:")
    table, dl = [], []
    for pos, (rec, trap, sc) in enumerate(ranked[:top_n], 1):
        why = reasoning.build_reasoning(rec, sc, trap, pos, counterfactual=(pos > top_n // 2))
        flags = []
        if trap["is_honeypot"]:
            flags.append("🚫 honeypot")
        if trap["is_stuffer"]:
            flags.append("⚠️ stuffer")
        table.append({
            "rank": pos, "candidate_id": rec["candidate_id"], "title": rec["title"],
            "yoe": rec["yoe"], "score": round(sc["final"], 3),
            "flags": " ".join(flags), "reasoning": why,
        })
        dl.append({"candidate_id": rec["candidate_id"], "rank": pos,
                   "score": round(sc["final"], 6), "reasoning": why})
    df = pd.DataFrame(table)
    st.dataframe(df, use_container_width=True, hide_index=True)

    st.download_button("⬇ Download ranked CSV",
                       pd.DataFrame(dl).to_csv(index=False).encode("utf-8"),
                       "submission_sample.csv", "text/csv")

    st.subheader("Score breakdown")
    pick = st.selectbox("Inspect a candidate",
                        [r["candidate_id"] for r in table], index=0)
    rec, trap, sc = next(x for x in ranked if x[0]["candidate_id"] == pick)
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Base components** (weighted)")
        st.dataframe(pd.DataFrame(
            [{"component": k, "value": round(v, 3), "weight": config.WEIGHTS[k]}
             for k, v in sc["components"].items()]), hide_index=True)
    with c2:
        st.markdown("**Multiplicative modifiers**")
        st.dataframe(pd.DataFrame(
            [{"modifier": k, "x": round(v, 3)} for k, v in sc["modifiers"].items()]),
            hide_index=True)
        st.metric("Final score", round(sc["final"], 4))
