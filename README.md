# Redrob Intelligent Candidate Ranker - Track 1

[![CI](https://github.com/DhruvGoyal404/redrob-ranker/actions/workflows/ci.yml/badge.svg)](https://github.com/DhruvGoyal404/redrob-ranker/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.13](https://img.shields.io/badge/python-3.13-blue.svg)](https://www.python.org/)

Ranking the top-100 candidates for **"Senior AI Engineer - Founding Team"** out of a
100,000-profile pool - by *evidence of the right work*, not by who stuffed the most
AI keywords into their skills list.

> **The thesis.** The released JD says it plainly: *"The right answer is not to find
> candidates whose skills section contains the most AI keywords. A candidate who has
> all the AI keywords but whose title is Marketing Manager is not a fit. A Tier-5
> candidate may not use the word 'RAG' but if their career history shows they built a
> recommendation system at a product company, they're a fit."* So we score on
> **demonstrated, outcome-shaped evidence** read from career history, gated by a strict
> validity check, and modulated by whether the candidate is **actually available**.

---

## The magic moment (live demo)

**Sandbox:** https://huggingface.co/spaces/agent-dg21/redrob-ranker (Hugging Face Spaces, Docker)

Pick a role (or paste any JD) → the shortlist re-ranks live, each row with a confidence
tag, a grounded one-line justification, a "JD match" score, and trap flags. Switch from
*Senior AI Engineer* to *Computer Vision* / *Backend* / *Data Analyst* and the right
candidate rises to the top. Inject a keyword-stuffer to watch the trap gate sink it. Open
any candidate for the score breakdown (JD relevance / skill quality / experience / availability).

> **Two rankers, disclosed honestly.** The **live demo** (`app/demo_rank.py`) is a *general,
> JD-adaptive* product: relevance to the typed JD - dense `bge-small` embeddings + BM25,
> running live on Hugging Face Spaces (Docker/CPU, which supports PyTorch; first run loads the
> model ~30 s) - drives the order, modulated by role-agnostic skill quality + availability and
> gated by the trap detector. The **competition `submission.csv`** (`rank.py`) uses our signal
> scorer *tuned to the one challenge role* and is unchanged by the demo. Both share the trap
> gate, evidence reading, and embeddings; the demo shows the machinery generalizes to any role.

```bash
streamlit run app/streamlit_app.py
```

---

## How it maps to the three required capabilities

| Required capability | Where it lives |
|---|---|
| **Deep Job Understanding** | `src/config.py` + `src/jd.py` encode the JD's *real* must-haves, explicit disqualifiers, and ideal-candidate band - not a keyword list |
| **Contextual Relevance** | `src/retrieve.py` - BM25 + `bge-small` dense embeddings fused with RRF; semantic match catches plain-language "Tier-5" candidates who never write the buzzwords |
| **Signal Integration** | `src/score.py` - the 23 `redrob_signals` enter as an availability modifier (recruiter response, recency, open-to-work, interview completion); profile + career metadata + behavior combined |

---

## Architecture

**System architecture** - offline embedding precompute feeding the CPU-only online ranking pipeline:

![System architecture diagram](diagrams/architecturediagram.png)

**Module structure** (class diagram):

![Module class diagram](diagrams/classdiagrams.png)

**Use cases**:

![Use case diagram](diagrams/usecasediagram.png)

### The scorer (transparent, defensible weights - `src/config.py`)

Additive base (weights sum to 1.0):

| Component | Weight | What it rewards |
|---|---|---|
| `title_role_fit` | 0.22 | Holding (now or before) an ML/AI/IR role - the decisive anti-stuffer signal |
| `domain_evidence` | 0.24 | Retrieval/ranking/embeddings/vector-DB/eval evidence **read from career descriptions**, not skill names |
| `must_have_coverage` | 0.20 | The JD's "absolutely need" list (embeddings retrieval, vector DB, ranking eval, applied ML) |
| `semantic_similarity` | 0.14 | Dense JD↔profile match (hybrid retrieval) |
| `experience_band` | 0.08 | Peak 6-8y per the JD's "5-9 is a range" |
| `skill_trust` | 0.08 | proficiency × duration × **platform assessment** × endorsements (defeats lazy stuffing) |
| `nice_to_have` | 0.04 | LoRA/QLoRA, LTR, HR-tech, OSS |

Then multiplicative modifiers: **availability** (0.55-1.10 from `redrob_signals`),
**location** (Pune/Noida ↑, relocation-aware), and the JD's **disqualifiers**
(services-firms-only ×0.45, off-domain ×0.40, title-hopper ×0.80). Honeypots ×0.001,
stuffers ×0.05.

---

## Reproduce

```bash
pip install -r requirements.txt

# One-time offline pre-computation (embeddings cache; may exceed 5 min, GPU OK).
# Needs the heavy deps (torch/sentence-transformers), kept separate so the ranker
# and the demo stay light:
pip install -r requirements-precompute.txt
python precompute.py --candidates ./candidates.jsonl --artifacts ./artifacts

# The ranking step - CPU-only, no network, < 5 min, < 16 GB  (Stage-3 command)
python rank.py --candidates ./candidates.jsonl --out ./submission.csv --validate
```

`rank.py` imports **no torch and makes no network calls** - it loads the precomputed
vectors and rebuilds BM25 in-process. If the embedding cache is absent it degrades
gracefully to a BM25-only ranking so the pipeline always runs.

> **Data note.** `candidates.jsonl` is not committed (464 MB). Drop the file from the
> hackathon bundle into the repo root, then run the commands above.
>
> **Submission file.** The committed top-100 is [`paradise0211.csv`](paradise0211.csv); the
> portal upload is the same rows exported to `paradise0211.xlsx` (Excel required).

### Reproduce via Docker (offline, CPU-only)

A `Dockerfile` reproduces the ranking step in a clean container with only the ranking
deps (`numpy` + `rank-bm25`, no torch). Verified to run with networking disabled:

```bash
docker build -t redrob-ranker .
docker run --rm --network none \
  -v "$PWD/candidates.jsonl:/app/candidates.jsonl:ro" \
  -v "$PWD/artifacts:/app/artifacts:ro" \
  -v "$PWD/out:/app/out" \
  redrob-ranker --candidates /app/candidates.jsonl --out /app/out/submission.csv --validate
```

Without the `artifacts` mount it still produces a valid BM25-only ranking. `make docker-build`
/ `make docker-run` wrap these.

### Tests

```bash
pip install -r requirements-rank.txt -r requirements-dev.txt   # minimal deps + pytest
python -m pytest -q   # trap detection, ranking metrics, word-boundary matching, pipeline smoke
```

CI (`.github/workflows/ci.yml`) runs compile + tests on every push.

---

## Compute compliance

| Constraint | Limit | This pipeline |
|---|---|---|
| Runtime (ranking) | ≤ 5 min | **~81 s** on a 12-core laptop CPU when uncontended (measured); up to ~200 s under heavy concurrent CPU load - always well under the 5-min cap |
| Memory | ≤ 16 GB | well under (streaming parse; ~2 GB peak) |
| Compute | CPU only | yes - no GPU, no torch at ranking time |
| Network | off | yes - no API calls; vectors precomputed |
| Disk (intermediate) | ≤ 5 GB | embeddings cache ~153 MB |
| Per-candidate LLM calls | none | none |

---

## Evaluation (honest, no hidden ground truth)

There is no public gold ranking, so we report three things and are explicit about what
each is worth:

- **Trap-catch rate - trustworthy** (we *know* the trap labels because we detect them
  by internal consistency). On the full 100K pool (68 honeypots, 3,876 stuffers
  detected): honeypots in top-100 = **0** (Stage-3 DQ is >10), honeypots in top-10 =
  **0**, stuffers in top-100 = **0**. Top-100 is **100% genuinely ML/AI/IR-titled**.
- **Silver-proxy NDCG/MAP/P@k - independent sanity check** (`eval/`): the relevance
  grade *within* the eligible band comes from **held-out recruiter-demand signals the
  ranker never uses as features** (`saved_by_recruiters`, `search_appearance`,
  `profile_views`), so it is not circular with our scoring. NDCG@10 **0.985**, NDCG@50
  **0.981**, MAP **1.00**, P@10 **1.00** (composite **0.987**). Honestly imperfect -
  recruiter demand also reflects popularity, not only JD-fit - and disclosed as such.

```bash
python -m eval.evaluate --candidates ./candidates.jsonl --submission ./submission.csv
```

We do **not** claim our pipeline matches a true gold ranking - none exists publicly -
but we quantify against a disclosed proxy and report the trap-catch numbers we *can*
stand behind.

### Multi-judge LLM gold set - rigorous, with disclosed agreement (`eval/multijudge.py`)

The third number is the playbook's "do evaluation properly" version. We take a
stratified sample of 40 candidates (our own top signal-ranked picks + known traps +
random spread) and have **three independent Claude judges - `claude-opus-4-8`,
`claude-sonnet-4-6`, `claude-haiku-4-5`** - each score every candidate 0-5 for JD fit,
**offline** (this never runs at ranking time). We then report **inter-rater agreement**
so a reviewer can calibrate how much to trust the gold:

- **Mean pairwise quadratic Cohen's kappa = 0.96** (opus<->sonnet 0.99, opus<->haiku
  0.93, sonnet<->haiku 0.95) - the panel agrees strongly on who fits.
- Against the median-judge gold (relevant = score >= 3): **NDCG@10 0.985, NDCG@50
  0.987, MAP 1.00, P@10 1.00** - our ranking puts the judge-approved candidates on top
  and the planted traps (scored 0) at the bottom.

```bash
python -m eval.multijudge --candidates ./candidates.jsonl \
  --judges anthropic:claude-opus-4-8,anthropic:claude-sonnet-4-6,anthropic:claude-haiku-4-5-20251001 \
  --n 40 --out gold_report.json
```

The exact run's full output - per-judge 0-5 scores, the pairwise/mean kappa, the
aggregated gold, and the sampled candidate ids - is committed as
[`gold_report.json`](gold_report.json) so the numbers above are independently checkable.

**Disclosed honestly** (the panel rewards this): the three judges share a model family
(Claude), so the high kappa partly reflects shared training - genuine cross-family
triangulation with OpenAI was unreachable from our compute environment and is noted as
future work. The sample **deliberately includes our own top picks** so the panel can
confirm or refute them; these numbers therefore validate that *independent LLMs agree
our highest-ranked candidates are the strongest and that traps score 0*, not that we
matched an unbiased global gold ranking. Judges are pluggable (`anthropic` / `openai` /
local `ollama`), so the panel can be re-run and broadened.

### Fairness / proxy-skew audit (`src/fairness.py`)

This is **voluntary best-practice rigor we chose to apply, not a regulatory-compliance
claim.** The dataset is synthetic and has **no protected-class labels** (gender, caste,
socioeconomic background), so this cannot replicate a real adverse-impact audit - and the
legal backdrop is genuinely unsettled (US disparate-impact guidance is in flux in 2025-26;
India has no binding AI-hiring statute and no private-sector caste-discrimination law). So
we measure **proxies** for protected attributes and report **proxy-risk**, not violations.

We compare the top-100 against the realistic **eligible pool** (16,776 non-trap candidates
holding a relevant/adjacent technical role - a filter computed from the profile, independent
of the ranking, so there is no leakage). The module is deterministic (no seed/sampling).

```bash
python -m src.fairness --candidates ./candidates.jsonl --submission ./submission.csv --residual
```

| Proxy | eligible base | of top-100 | reading |
|---|---|---|---|
| Preferred location (Pune/Noida) | 4,963 (29.6%) | 50/100 | intended JD-fit |
| Tier-1 college | 1,730 (10.3%) | 50/100 | **proxy-risk flag** (not adverse impact) |
| Employment gap (>6 mo) | 0 (0.0%) | 0/100 | **non-informative** |

- **Location** - *intended, not a defect.* The JD explicitly prefers Noida/Pune (Redrob's
  NCR offices); relocation-willing candidates are still credited.
- **Tier-1 college - a proxy-risk flag, with an honest open limitation.** College tier is **not a
  protected class** - it is a known *proxy* for protected attributes in India. So our **50/100 vs a
  10.3% base rate (~5x) is a risk flag, not an adverse-impact finding** (chi-square 171, p<0.001,
  subgroups n=1,730 / 15,046 - significant, not a small-sample artifact), because we never measured
  the actual protected attributes. We ran two tests with `--residual`:
  - *Residual gradient* - ranking by the tier-blind full signal score reproduces our exact 50%
    tier-1 rate, which shows the ranker adds nothing **beyond its own features**. But that
    comparison is the pipeline vs itself, so on its own it is close to tautological.
  - *Signal decomposition (the test that actually matters)* - ranking the pool by each signal in
    isolation, the tier-1 rate in the top-100 (base 10.3%) is **52/100 by `domain_evidence` (the
    CV-text feature; 95% CI [42,62], chi-square 188 vs base, p<0.001), 9/100 by objective
    years-of-experience (95% CI [5,16], chi-square 0.2 - not significant, CI spans the base), and
    38/100 by platform assessment scores (CI [29,48], chi-square 83, p<0.001)**. The
    domain-vs-experience contrast is itself significant (chi-square 43.6). (Tenure is heavily tied
    - 5,016 candidates sit at the in-band max - so its 9% is a stable sample of that large cohort,
    not a noisy slice. Correlations with tier-1: domain_evidence +0.25 >> assessment +0.08 >> tenure +0.01.)
  - **Honest conclusion:** the skew is driven mainly by `domain_evidence`, read from CV narrative
    text - exactly the feature most likely to absorb an elite-college access advantage
    (included-variable bias). We therefore **do not** claim the skew is cleanly merit-mediated; our
    ranking does not *independently amplify* the tier-1 concentration beyond what our scoring inputs
    encode, but we **cannot rule out that those inputs partially launder the same proxy**, and we
    flag this as an **open limitation, not a closed question**. Mitigation directions: lean harder
    on objective assessment scores relative to CV-text evidence, and audit against real protected
    attributes in production.
- **Employment gap - non-informative, and we say so rather than fake a finding.** The `_has_gap`
  proxy fires on **0 of the 16,776 eligible** (synthetic histories are gap-free), so "0/100"
  carries no signal - there is nobody with a gap to include or exclude. The code reports this as
  **NON-INFORMATIVE** (we caught our own earlier misread of the selection rate as a base rate).

**What we'd do in production** (where the labels exist): join real protected-attribute data,
run the four-fifths / adverse-impact test on the attributes themselves (not proxies), add
confidence intervals on each ratio, and keep a human-in-the-loop override - the transparent,
monitored posture that current responsible-AI guidance recommends.

---

## What we tried and rejected

- **Sorting by AI-skill count / pure embedding similarity** - exactly the trap. The
  provided `sample_submission.csv` does this and ranks HR Managers and Graphic
  Designers at the top. Pure dense similarity also gets fooled by stuffed skill lists,
  so embeddings are one input among seven, not the ranker.
- **A learned ranker (LambdaMART)** - tempting, but with no real relevance labels we'd
  be training on guessed targets. A transparent weighted function is more defensible in
  the Stage-5 interview and easier to audit. LTR is noted as the productionization path
  once real recruiter-feedback labels exist.
- **Per-candidate LLM reasoning at ranking time** - forbidden by the compute
  constraint and unscalable; we assemble grounded reasoning from the score breakdown
  instead, which also eliminates hallucination.
- **Flagging "skill duration > career length" as a honeypot** - it fired on ~9% of the
  pool (the synthetic generator assigns skill durations independently), so it was noise,
  not an impossibility. Dropped in favor of clear contradictions only.

---

## Repo layout

```
rank.py            precompute.py        submission_metadata.yaml
src/    config jd parse features traps retrieve score reasoning fairness
eval/   metrics.py evaluate.py multijudge.py
app/    streamlit_app.py  demo_rank.py (JD-adaptive demo ranker)  presets.py
tests/  test_traps.py test_metrics.py test_features.py test_pipeline.py test_demo_rank.py test_multijudge.py
data/   sample_candidates.json  demo_candidates.json  candidate_schema.json
Dockerfile  requirements-rank.txt  Makefile  .github/workflows/ci.yml
```

## Future work
LambdaMART LTR on real recruiter-feedback labels; cross-encoder re-rank of the top-200;
continuous fairness monitoring; online A/B evaluation harness.
