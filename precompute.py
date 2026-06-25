"""
Offline pre-computation (allowed to exceed the 5-minute ranking budget).

Embeds every candidate's career-narrative with a small CPU-friendly model
(BAAI/bge-small-en-v1.5, 384-d) and the fixed JD query, then caches both to
disk. This is the ONLY script that imports sentence-transformers or needs
network (to fetch the model once). rank.py reads the cache and never does.

Usage:
    python precompute.py --candidates ./candidates.jsonl --artifacts ./artifacts
"""
from __future__ import annotations

import os

# Force the HuggingFace stack onto PyTorch only - a broken TensorFlow install is
# present in this environment and transformers will otherwise try to import it.
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_TORCH", "1")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import argparse
import json
import time
from datetime import date
from pathlib import Path

import numpy as np

from src import config, parse
from src.jd import JD_QUERY


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", required=True)
    ap.add_argument("--artifacts", default="./artifacts")
    ap.add_argument("--model", default=config.EMBED_MODEL)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--device", default="auto", help="auto|cpu|cuda")
    ap.add_argument("--max-seq-length", type=int, default=192)
    args = ap.parse_args()

    art = Path(args.artifacts)
    art.mkdir(parents=True, exist_ok=True)
    ref = date.fromisoformat(config.REFERENCE_DATE)

    t0 = time.time()
    print(f"[precompute] loading + normalizing {args.candidates} ...")
    ids, narratives = [], []
    for raw in parse.iter_raw(args.candidates):
        rec = parse.normalize(raw, ref)
        ids.append(rec["candidate_id"])
        narratives.append(rec["narrative"])
    print(f"[precompute] {len(ids)} candidates in {time.time()-t0:.1f}s")

    import torch
    from sentence_transformers import SentenceTransformer  # heavy; offline-only
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        torch.set_num_threads(os.cpu_count() or 4)
    print(f"[precompute] loading model {args.model} on {device} "
          f"(cuda_available={torch.cuda.is_available()}) ...")
    model = SentenceTransformer(args.model, device=device)
    model.max_seq_length = args.max_seq_length

    t1 = time.time()
    emb = model.encode(
        narratives,
        batch_size=args.batch_size,
        normalize_embeddings=True,
        show_progress_bar=True,
        convert_to_numpy=True,
    ).astype(np.float32)
    print(f"[precompute] embedded candidates in {time.time()-t1:.1f}s -> {emb.shape}")

    jd_emb = model.encode([JD_QUERY], normalize_embeddings=True,
                          convert_to_numpy=True).astype(np.float32)

    np.save(art / "embeddings.npy", emb)
    np.save(art / "jd_embedding.npy", jd_emb)
    (art / "ids.json").write_text(json.dumps(ids), encoding="utf-8")
    meta = {"model": args.model, "dim": int(emb.shape[1]), "n": len(ids),
            "reference_date": config.REFERENCE_DATE}
    (art / "embed_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[precompute] wrote artifacts to {art} in total {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
