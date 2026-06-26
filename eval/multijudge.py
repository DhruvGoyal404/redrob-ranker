"""
Multi-judge LLM gold-set evaluation  (OFFLINE - network/LLM calls allowed here;
this is NOT the ranking step and never runs inside rank.py).

Why this exists: there is no public gold ranking. The cheap always-on sanity check
is the silver-proxy in evaluate.py. This module is the rigorous version from the
strategy playbook: have 2-3 *independent* LLM judges each score a stratified sample
of candidates for JD relevance, measure inter-rater agreement (quadratic-weighted
Cohen's kappa) so reviewers can calibrate trust, aggregate the judges into a gold
relevance, and score our ranker (NDCG@k / Precision@k) against it - with limitations
disclosed honestly.

Judges are pluggable:
  --backend anthropic   (uses ANTHROPIC_API_KEY; --models claude-sonnet-4-6,...)
  --backend openai      (uses OPENAI_API_KEY;     --models gpt-4o,gpt-4o-mini,...)
  --backend ollama      (local models on a GPU box; --models llama3.1,qwen2.5,...)
  --backend stub        (deterministic synthetic judges - tests the harness, no API)

Example (real run):
  python -m eval.multijudge --candidates ./candidates.jsonl \
      --backend anthropic --models claude-sonnet-4-6,claude-opus-4-1 --n 40 --out gold_report.json
"""
from __future__ import annotations

import argparse
import json
import random
import re
from datetime import date

from src import config, parse, traps, features
from src import score as scoring
from eval import metrics

# ----------------------------------------------------------------------------- prompts
JUDGE_SYSTEM = (
    "You are an expert technical recruiter evaluating how well a candidate fits a "
    "specific role. Judge strictly on DEMONSTRATED EVIDENCE in the profile (what they "
    "actually built and shipped), not on keyword presence. A non-technical person who "
    "lists AI skills with no real project evidence is NOT a fit and should score low."
)
JUDGE_USER = """JOB DESCRIPTION:
{jd}

CANDIDATE PROFILE
Title: {title}
Experience: {yoe} years
Location: {loc}
Skills: {skills}
Career narrative:
{narrative}

Rate this candidate's fit for the role on a 0-5 integer scale:
5 = ideal hire   4 = strong fit   3 = relevant, worth a call
2 = weak / adjacent only   1 = poor fit   0 = irrelevant, or a keyword-stuffer with no real evidence

Respond with ONLY a JSON object, no prose:
{{"score": <integer 0-5>, "reason": "<one short sentence>"}}"""


def profile_fields(rec: dict) -> dict:
    skills = rec.get("skills") or []
    names = [s.get("name", "") for s in skills if isinstance(s, dict)][:18]
    narrative = (rec.get("narrative") or "")[:1600]
    return {"jd": config.JD_QUERY if hasattr(config, "JD_QUERY") else "",
            "title": rec.get("title", "?"), "yoe": f"{rec.get('yoe', 0):.1f}",
            "loc": rec.get("location", "?"), "skills": ", ".join(n for n in names if n),
            "narrative": narrative}


def _parse_score(text: str) -> int:
    """Pull the integer score out of a judge's reply, robustly."""
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        try:
            obj = json.loads(m.group(0))
            return max(0, min(5, int(round(float(obj["score"])))))
        except Exception:
            pass
    m = re.search(r"([0-5])", text)
    return int(m.group(1)) if m else 0


# ----------------------------------------------------------------------------- judges
class AnthropicJudge:
    def __init__(self, model):
        import anthropic
        self.client = anthropic.Anthropic()
        self.model = model

    def score(self, jd, rec, trap):
        f = profile_fields(rec); f["jd"] = jd
        msg = self.client.messages.create(
            model=self.model, max_tokens=200, temperature=0, system=JUDGE_SYSTEM,
            messages=[{"role": "user", "content": JUDGE_USER.format(**f)}])
        return _parse_score(msg.content[0].text)


class OpenAIJudge:
    def __init__(self, model):
        import openai
        self.client = openai.OpenAI()
        self.model = model

    def score(self, jd, rec, trap):
        f = profile_fields(rec); f["jd"] = jd
        r = self.client.chat.completions.create(
            model=self.model, temperature=0, max_tokens=200,
            messages=[{"role": "system", "content": JUDGE_SYSTEM},
                      {"role": "user", "content": JUDGE_USER.format(**f)}])
        return _parse_score(r.choices[0].message.content)


class OllamaJudge:
    def __init__(self, model, host="http://localhost:11434"):
        self.model, self.host = model, host

    def score(self, jd, rec, trap):
        import requests
        f = profile_fields(rec); f["jd"] = jd
        r = requests.post(f"{self.host}/api/chat", timeout=120, json={
            "model": self.model, "stream": False, "options": {"temperature": 0},
            "messages": [{"role": "system", "content": JUDGE_SYSTEM},
                         {"role": "user", "content": JUDGE_USER.format(**f)}]})
        return _parse_score(r.json()["message"]["content"])


class StubJudge:
    """Deterministic synthetic judge for testing the harness without any API. It
    derives a latent relevance from the profile (using our own assessment as a
    pseudo-ground-truth) and adds per-judge Gaussian noise, so inter-judge
    agreement is realistic and the reported kappa is a meaningful smoke test."""
    def __init__(self, name, noise=0.7, seed=0):
        self.rng = random.Random((hash(name) ^ seed) & 0xFFFFFFFF)
        self.noise = noise

    def score(self, jd, rec, trap):
        if trap["is_honeypot"] or trap["is_stuffer"]:
            latent = 0.3
        else:
            tc = features.title_class(rec)
            latent = {"relevant": 4.3, "adjacent": 3.0, "other": 2.0,
                      "nontech": 0.6, "offdomain": 1.0}.get(tc, 2.0)
            ev = features.evidence_groups(rec)
            latent += 0.5 * min(2, sum(1 for g in ("retrieval_ranking", "embeddings",
                                                   "vector_db", "evaluation") if ev.get(g)))
            if config.EXP_OK_LO <= rec.get("yoe", 0) <= config.EXP_OK_HI:
                latent += 0.3
        return max(0, min(5, int(round(latent + self.rng.gauss(0, self.noise)))))


def make_panel(specs):
    """Build a (possibly mixed-family) judge panel from 'backend:model' specs,
    e.g. ['anthropic:claude-sonnet-4-6', 'openai:gpt-4o', 'ollama:qwen2.5']."""
    cls = {"anthropic": AnthropicJudge, "openai": OpenAIJudge, "ollama": OllamaJudge}
    judges = {}
    for i, spec in enumerate(specs):
        backend, _, model = spec.partition(":")
        model = model or backend
        name = f"{backend}:{model}"
        if backend == "stub":
            judges[name] = StubJudge(model, seed=i)
        elif backend in cls:
            judges[name] = cls[backend](model)
        else:
            raise SystemExit(f"unknown backend in judge spec '{spec}'")
    return judges


# ----------------------------------------------------------------------------- sampling
def stratified_sample(by_id, n, seed):
    """Pick a sample that spans the difficulty range: clearly-relevant, adjacent,
    traps, and irrelevant - so the gold set actually exercises the ranker."""
    rng = random.Random(seed)
    buckets = {"relevant": [], "adjacent": [], "trap": [], "other": []}
    for cid, (rec, trap) in by_id.items():
        if trap["is_honeypot"] or trap["is_stuffer"]:
            buckets["trap"].append(cid)
        else:
            tc = features.title_class(rec)
            buckets["relevant" if tc == "relevant" else
                    "adjacent" if tc == "adjacent" else "other"].append(cid)
    quota = {"relevant": round(0.40 * n), "adjacent": round(0.25 * n),
             "trap": round(0.20 * n), "other": round(0.15 * n)}
    picked = []
    for k, q in quota.items():
        rng.shuffle(buckets[k])
        picked += buckets[k][:q]
    return picked[:n]


# ----------------------------------------------------------------------------- agreement
def quadratic_kappa(a, b, k=5):
    """Quadratic-weighted Cohen's kappa for two raters on a 0..k ordinal scale.
    (Implemented directly so the harness needs no scikit-learn.)"""
    import numpy as np
    n = k + 1
    O = np.zeros((n, n))
    for x, y in zip(a, b):
        O[x, y] += 1
    W = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            W[i, j] = (i - j) ** 2 / (k ** 2)
    act_a = O.sum(axis=1)
    act_b = O.sum(axis=0)
    E = np.outer(act_a, act_b) / O.sum()
    denom = (W * E).sum()
    return 1.0 if denom == 0 else float(1.0 - (W * O).sum() / denom)


def panel_agreement(scores: dict):
    """Pairwise quadratic-weighted Cohen's kappa across judges + the mean."""
    names = list(scores)
    pairs, vals = {}, []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            k = quadratic_kappa(scores[names[i]], scores[names[j]])
            pairs[f"{names[i]} <-> {names[j]}"] = round(k, 3)
            vals.append(k)
    mean_k = float(sum(vals) / len(vals)) if vals else float("nan")
    return pairs, mean_k


def aggregate_gold(scores: dict):
    """Per-candidate gold relevance = median judge score (rounded)."""
    import statistics
    names = list(scores)
    n = len(scores[names[0]])
    return [int(round(statistics.median(scores[j][i] for j in names))) for i in range(n)]


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", required=True)
    ap.add_argument("--judges", default="",
                    help="comma-separated backend:model judge specs, e.g. "
                         "anthropic:claude-sonnet-4-6,openai:gpt-4o,ollama:qwen2.5")
    ap.add_argument("--backend", choices=["anthropic", "openai", "ollama", "stub"], default="stub",
                    help="single-family fallback when --judges is not given")
    ap.add_argument("--models", default="a,b,c", help="comma-separated model names for --backend")
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--limit", type=int, default=0, help="only read first N pool lines (speed)")
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    ref = date.fromisoformat(config.REFERENCE_DATE)
    jd = config.JD_QUERY if hasattr(config, "JD_QUERY") else __import__(
        "src.jd", fromlist=["JD_QUERY"]).JD_QUERY

    print(f"[gold] loading pool from {args.candidates} ...")
    by_id = {}
    for i, raw in enumerate(parse.iter_raw(args.candidates)):
        if args.limit and i >= args.limit:
            break
        rec = parse.normalize(raw, ref)
        by_id[rec["candidate_id"]] = (rec, traps.assess(rec))
    print(f"[gold] {len(by_id)} candidates in pool")

    sample = stratified_sample(by_id, args.n, args.seed)
    print(f"[gold] sampled {len(sample)} candidates (stratified across relevant/adjacent/trap/other)")

    if args.judges:
        specs = [s.strip() for s in args.judges.split(",") if s.strip()]
    else:
        specs = [f"{args.backend}:{m.strip()}" for m in args.models.split(",") if m.strip()]
    judges = make_panel(specs)
    print(f"[gold] judges: {list(judges)}")

    # Each judge scores every sampled candidate (resilient to transient API errors).
    import time
    def safe_score(judge, rec, trap):
        for attempt in range(3):
            try:
                return judge.score(jd, rec, trap)
            except Exception as e:
                if attempt == 2:
                    print(f"[gold]   judge error after retries ({e}); recording neutral 2")
                    return 2
                time.sleep(1.5 * (attempt + 1))

    scores = {name: [] for name in judges}
    for n_done, cid in enumerate(sample, 1):
        rec, trap = by_id[cid]
        for name, judge in judges.items():
            scores[name].append(safe_score(judge, rec, trap))
        if n_done % 10 == 0:
            print(f"[gold]   judged {n_done}/{len(sample)}")

    pairs, mean_k = panel_agreement(scores)
    gold = aggregate_gold(scores)

    # Our ranker's ordering of the SAME sample (signal-based core score).
    ours = []
    for idx, cid in enumerate(sample):
        rec, trap = by_id[cid]
        sc = scoring.score_candidate(rec, trap, 0.0)
        ours.append((sc["final"], idx))
    ours.sort(key=lambda x: -x[0])
    gold_in_our_order = [gold[idx] for _, idx in ours]
    m = metrics.composite(gold_in_our_order)

    from collections import Counter
    print("\n=== INTER-RATER AGREEMENT (quadratic Cohen's kappa) ===")
    for pair, k in pairs.items():
        print(f"  {pair}: {k}")
    print(f"  mean pairwise kappa: {mean_k:.3f}  "
          f"({'substantial' if mean_k >= 0.6 else 'moderate' if mean_k >= 0.4 else 'fair/low'})")
    print("\n=== GOLD RELEVANCE DISTRIBUTION (median of judges) ===")
    print(" ", dict(sorted(Counter(gold).items())))
    print("\n=== OUR RANKER vs LLM GOLD (relevant = gold >= 3) ===")
    for k in ["ndcg@10", "ndcg@50", "map", "p@10", "p@5", "composite"]:
        print(f"  {k:10s}: {m[k]:.4f}")
    print("\nNote: judges are LLMs, not ground truth (known leniency/position bias); we "
          "report agreement so trust can be calibrated, and gold is the median of "
          "independent judges. Honest, triangulated proxy - not a hidden score.")

    if args.out:
        json.dump({"judges": specs, "n": len(sample),
                   "pairwise_kappa": pairs, "mean_kappa": mean_k,
                   "gold_distribution": dict(Counter(gold)), "our_metrics": m,
                   "sample_ids": sample, "judge_scores": scores, "gold": gold},
                  open(args.out, "w"), indent=2)
        print(f"\n[gold] wrote {args.out}")


if __name__ == "__main__":
    main()
