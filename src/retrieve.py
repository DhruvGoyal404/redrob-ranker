"""
Hybrid retrieval: BM25 (sparse) + dense cosine, fused with Reciprocal Rank
Fusion (RRF, k=60). Produces a high-recall shortlist that the scorer then ranks
in detail.

At ranking time this module imports NO torch / sentence-transformers and makes
NO network calls - dense vectors are loaded from the precomputed cache and the
JD vector is precomputed too. BM25 is rebuilt from the narratives in-process
(pure-python rank_bm25), which is fast and keeps the cache small.
"""
from __future__ import annotations

import re
from typing import List, Optional

import numpy as np
from rank_bm25 import BM25Okapi

_TOKEN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> List[str]:
    return _TOKEN.findall(text.lower())


def bm25_scores(narratives: List[str], query: str) -> np.ndarray:
    corpus = [tokenize(n) for n in narratives]
    bm25 = BM25Okapi(corpus)
    return np.asarray(bm25.get_scores(tokenize(query)), dtype=np.float32)


def dense_scores(doc_emb: np.ndarray, jd_emb: np.ndarray) -> np.ndarray:
    """Cosine similarity of each row of doc_emb against jd_emb. Assumes vectors
    are L2-normalized at precompute time; falls back to normalizing here."""
    q = jd_emb.astype(np.float32).ravel()
    qn = np.linalg.norm(q)
    if qn > 0:
        q = q / qn
    d = doc_emb.astype(np.float32)
    norms = np.linalg.norm(d, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (d / norms) @ q


def _ranks_from_scores(scores: np.ndarray) -> np.ndarray:
    """Return rank position (0 = best) for each index, by descending score."""
    order = np.argsort(-scores, kind="stable")
    ranks = np.empty_like(order)
    ranks[order] = np.arange(len(order))
    return ranks


def rrf_fuse(bm25: np.ndarray, dense: np.ndarray, k: int = 60) -> np.ndarray:
    """Reciprocal Rank Fusion of two score arrays -> fused score per index."""
    rb = _ranks_from_scores(bm25)
    rd = _ranks_from_scores(dense)
    return 1.0 / (k + rb + 1) + 1.0 / (k + rd + 1)


def shortlist(
    narratives: List[str],
    query: str,
    doc_emb: Optional[np.ndarray],
    jd_emb: Optional[np.ndarray],
    size: int,
):
    """Return (shortlist_indices, dense_sim_normalized_full).

    dense_sim_normalized is min-max scaled to [0,1] over the pool so the scorer
    can use it directly as the semantic_similarity component. If no embeddings
    are supplied (pure-BM25 fallback), dense contributes zeros.
    """
    bm25 = bm25_scores(narratives, query)
    if doc_emb is not None and jd_emb is not None:
        dense = dense_scores(doc_emb, jd_emb)
    else:
        dense = np.zeros(len(narratives), dtype=np.float32)

    fused = rrf_fuse(bm25, dense)
    idx = np.argsort(-fused, kind="stable")[:size]

    lo, hi = float(dense.min()), float(dense.max())
    dense_norm = (dense - lo) / (hi - lo) if hi > lo else np.zeros_like(dense)
    return idx, dense_norm
