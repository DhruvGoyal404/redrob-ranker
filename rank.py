"""
rank.py — produce submission.csv from candidates.jsonl.

This is the single Stage-3 reproduce command. It is CPU-only, makes no network
calls, imports no torch / sentence-transformers, and runs well within the
5-minute / 16 GB budget. Dense embeddings are read from ./artifacts (built
offline by precompute.py); if they are absent it degrades gracefully to a
BM25-only ranking so the pipeline still runs.

    python rank.py --candidates ./candidates.jsonl --out ./submission.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from datetime import date
from pathlib import Path

import numpy as np

from src import config, parse, traps, score as scoring, reasoning
from src.jd import JD_QUERY
from src import retrieve


def _load_embeddings(artifacts: Path, ids_in_order):
    """Load precomputed candidate embeddings + JD embedding, aligned to the
    current candidate order. Returns (doc_emb, jd_emb) or (None, None)."""
    emb_p, ids_p, jd_p = (artifacts / "embeddings.npy",
                          artifacts / "ids.json", artifacts / "jd_embedding.npy")
    if not (emb_p.exists() and ids_p.exists() and jd_p.exists()):
        print("[rank] no embedding cache found -> BM25-only fallback")
        return None, None
    emb = np.load(emb_p)
    jd = np.load(jd_p)
    cached_ids = json.loads(ids_p.read_text(encoding="utf-8"))
    if cached_ids == ids_in_order:
        return emb, jd
    # Re-align by id (covers a reordered/subset candidates file).
    pos = {cid: i for i, cid in enumerate(cached_ids)}
    if not all(cid in pos for cid in ids_in_order):
        print("[rank] embedding cache doesn't cover all candidates -> BM25-only")
        return None, None
    idx = np.array([pos[cid] for cid in ids_in_order])
    return emb[idx], jd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", required=True)
    ap.add_argument("--out", default="./submission.csv")
    ap.add_argument("--artifacts", default="./artifacts")
    ap.add_argument("--top", type=int, default=100)
    ap.add_argument("--validate", action="store_true",
                    help="run the official validator after writing")
    args = ap.parse_args()

    t0 = time.time()
    ref = date.fromisoformat(config.REFERENCE_DATE)

    print(f"[rank] loading candidates from {args.candidates} ...")
    records = parse.load_all(args.candidates, ref)
    ids_in_order = [r["candidate_id"] for r in records]
    narratives = [r["narrative"] for r in records]
    print(f"[rank] {len(records)} candidates loaded in {time.time()-t0:.1f}s")

    doc_emb, jd_emb = _load_embeddings(Path(args.artifacts), ids_in_order)

    t1 = time.time()
    shortlist_idx, dense_norm = retrieve.shortlist(
        narratives, JD_QUERY, doc_emb, jd_emb, config.SHORTLIST_SIZE)
    print(f"[rank] hybrid retrieval -> {len(shortlist_idx)} shortlisted "
          f"in {time.time()-t1:.1f}s")

    # Detailed scoring over the shortlist.
    scored_rows = []
    for i in shortlist_idx:
        rec = records[i]
        trap = traps.assess(rec)
        if trap["is_honeypot"]:
            continue  # tier-0; never in top-100 (and the >10% DQ rule)
        sc = scoring.score_candidate(rec, trap, float(dense_norm[i]))
        scored_rows.append((sc["final"], rec, sc, trap))

    # Sort by final score; take a margin above top for borderline reasoning.
    scored_rows.sort(key=lambda r: -r[0])
    top = scored_rows[: args.top]

    # Normalize scores to [0,1], round, then re-sort by (-score, candidate_id)
    # so the validator's monotonic + tie-break rules both hold exactly.
    max_final = max((r[0] for r in top), default=1.0) or 1.0
    prepared = []
    for final, rec, sc, trap in top:
        prepared.append((round(final / max_final, 6), rec, sc, trap))
    prepared.sort(key=lambda r: (-r[0], r[1]["candidate_id"]))

    # Reasoning: counterfactual on the borderline band (ranks ~40-70).
    rows = []
    prev = None
    for rank_pos, (score_val, rec, sc, trap) in enumerate(prepared, start=1):
        if prev is not None and score_val > prev:
            score_val = prev  # guard against any float drift
        prev = score_val
        cf = 40 <= rank_pos <= 70
        why = reasoning.build_reasoning(rec, sc, trap, rank_pos, counterfactual=cf)
        rows.append({
            "candidate_id": rec["candidate_id"],
            "rank": rank_pos,
            "score": f"{score_val:.6f}",
            "reasoning": why,
        })

    out_p = Path(args.out)
    with open(out_p, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["candidate_id", "rank", "score", "reasoning"])
        w.writeheader()
        w.writerows(rows)
    print(f"[rank] wrote {len(rows)} rows to {out_p} in total {time.time()-t0:.1f}s")

    if args.validate:
        import subprocess, sys
        val = Path(__file__).parent / "validate_submission.py"
        if val.exists():
            subprocess.run([sys.executable, str(val), str(out_p)])


if __name__ == "__main__":
    main()
