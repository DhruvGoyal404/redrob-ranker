"""
Ranking metrics matching the challenge's scoring formula:
    composite = 0.50*NDCG@10 + 0.30*NDCG@50 + 0.15*MAP + 0.05*P@10
where relevance is a graded tier (0..5) and "relevant" = tier >= 3.

These run against a *self-built* relevance proxy (see build_silver_labels), NOT
the hidden ground truth - so they are a sanity signal, honestly disclosed, not a
claim of true score.
"""
from __future__ import annotations

import math
from typing import Dict, List


def dcg(gains: List[float]) -> float:
    return sum(g / math.log2(i + 2) for i, g in enumerate(gains))


def ndcg_at_k(ranked_rel: List[float], k: int) -> float:
    top = ranked_rel[:k]
    ideal = sorted(ranked_rel, reverse=True)[:k]
    idcg = dcg(ideal)
    return dcg(top) / idcg if idcg > 0 else 0.0


def precision_at_k(ranked_rel: List[float], k: int, thresh: float = 3.0) -> float:
    top = ranked_rel[:k]
    return sum(1 for r in top if r >= thresh) / k if k else 0.0


def average_precision(ranked_rel: List[float], thresh: float = 3.0) -> float:
    hits, ap = 0, 0.0
    for i, r in enumerate(ranked_rel, start=1):
        if r >= thresh:
            hits += 1
            ap += hits / i
    total_rel = sum(1 for r in ranked_rel if r >= thresh)
    return ap / total_rel if total_rel else 0.0


def composite(ranked_rel: List[float]) -> Dict[str, float]:
    m = {
        "ndcg@10": ndcg_at_k(ranked_rel, 10),
        "ndcg@50": ndcg_at_k(ranked_rel, 50),
        "map": average_precision(ranked_rel),
        "p@10": precision_at_k(ranked_rel, 10),
        "p@5": precision_at_k(ranked_rel, 5),
    }
    m["composite"] = (0.50 * m["ndcg@10"] + 0.30 * m["ndcg@50"]
                      + 0.15 * m["map"] + 0.05 * m["p@10"])
    return m
