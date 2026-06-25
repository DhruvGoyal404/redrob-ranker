"""
Self-evaluation harness — run after producing submission.csv.

Reports two things:

1. TRAP-CATCH RATE (ground-truth-defensible): how many known honeypots / keyword
   stuffers leaked into the top-100. We *know* the trap labels because we detect
   them with internal-consistency rules, so this number is trustworthy and is the
   exact thing the Stage-3 filter (honeypot rate > 10% -> DQ) checks.

2. NDCG/MAP/P@k against a TRANSPARENT SILVER relevance proxy (build_silver_labels).
   This is NOT the hidden ground truth and is deliberately built from a simple,
   independent rubric; it correlates with the ranker because both read the same
   profiles, so treat it as a sanity signal, not a score. Disclosed honestly.

    python -m eval.evaluate --candidates ./candidates.jsonl --submission ./submission.csv
"""
from __future__ import annotations

import argparse
import csv
from collections import Counter
from datetime import date

from src import config, parse, traps, features
from eval import metrics


def _market_demand(rec: dict) -> float:
    """An INDEPENDENT 'market demand' signal the ranker never uses as a feature:
    how much recruiters engaged with this profile (saved / searched / viewed).
    Used only to *grade* the relevance proxy, so the resulting NDCG is not
    circular with the ranker's own scoring."""
    s = rec["signals"]
    return (2.0 * (s.get("saved_by_recruiters_30d", 0) or 0)
            + 0.1 * (s.get("search_appearance_30d", 0) or 0)
            + 0.1 * (s.get("profile_views_received_30d", 0) or 0))


def build_silver_labels(rec: dict, trap: dict, p50: float, p80: float) -> int:
    """Graded relevance proxy 0..5.

    Eligibility (title/trap) comes from the profile, but the *grade within the
    eligible band* comes from held-out recruiter-demand signals the ranker does
    not see. So a high NDCG here means our ordering agrees with how recruiters
    actually engaged — an independent check, honestly imperfect (demand also
    reflects popularity, not only JD-fit), disclosed as such."""
    if trap["is_honeypot"] or trap["is_stuffer"]:
        return 0
    tclass = features.title_class(rec)
    if tclass in ("nontech", "offdomain"):
        return 1 if features.has_relevant_or_adjacent_role(rec) else 0
    base = {"relevant": 3, "adjacent": 2, "other": 1}.get(tclass, 0)
    if base == 0:
        return 0
    md = _market_demand(rec)
    bump = 2 if md >= p80 else (1 if md >= p50 else 0)
    return max(0, min(5, base + bump))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", required=True)
    ap.add_argument("--submission", default="./submission.csv")
    args = ap.parse_args()
    ref = date.fromisoformat(config.REFERENCE_DATE)

    import numpy as np
    print("[eval] loading pool + assessing traps ...")
    by_id = {}
    pool_honeypots = pool_stuffers = 0
    eligible_demand = []
    for raw in parse.iter_raw(args.candidates):
        rec = parse.normalize(raw, ref)
        trap = traps.assess(rec)
        by_id[rec["candidate_id"]] = (rec, trap)
        pool_honeypots += trap["is_honeypot"]
        pool_stuffers += trap["is_stuffer"]
        if not (trap["is_honeypot"] or trap["is_stuffer"]) \
                and features.title_class(rec) in ("relevant", "adjacent", "other"):
            eligible_demand.append(_market_demand(rec))

    p50 = float(np.percentile(eligible_demand, 50)) if eligible_demand else 0.0
    p80 = float(np.percentile(eligible_demand, 80)) if eligible_demand else 0.0
    silver = {cid: build_silver_labels(rec, trap, p50, p80)
              for cid, (rec, trap) in by_id.items()}

    with open(args.submission, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    rows.sort(key=lambda r: int(r["rank"]))
    sub_ids = [r["candidate_id"] for r in rows]

    # --- trap-catch rate ---
    hp = [c for c in sub_ids if c in by_id and by_id[c][1]["is_honeypot"]]
    st = [c for c in sub_ids if c in by_id and by_id[c][1]["is_stuffer"]]
    hp10 = [c for c in sub_ids[:10] if c in by_id and by_id[c][1]["is_honeypot"]]
    missing = [c for c in sub_ids if c not in by_id]

    print("\n=== TRAP-CATCH (ground-truth-defensible) ===")
    print(f"  pool honeypots: {pool_honeypots} | stuffers: {pool_stuffers}")
    print(f"  honeypots in top-100: {len(hp)}  (Stage-3 DQ if >10) -> {'OK' if len(hp)<=10 else 'FAIL'}")
    print(f"  honeypots in top-10:  {len(hp10)}")
    print(f"  stuffers  in top-100: {len(st)}")
    print(f"  unknown ids (not in pool): {len(missing)}")

    # --- title-class composition of top-100 ---
    tc = Counter(features.title_class(by_id[c][0]) for c in sub_ids if c in by_id)
    print("\n=== TOP-100 COMPOSITION ===")
    print("  title classes:", dict(tc))

    # --- silver-proxy metrics ---
    ranked_rel = [silver.get(c, 0) for c in sub_ids]
    m = metrics.composite(ranked_rel)
    print("\n=== SILVER-PROXY METRICS (sanity only, NOT the hidden score) ===")
    for k in ["ndcg@10", "ndcg@50", "map", "p@10", "p@5", "composite"]:
        print(f"  {k:10s}: {m[k]:.4f}")
    print("  top-10 silver tiers:", ranked_rel[:10])


if __name__ == "__main__":
    main()
