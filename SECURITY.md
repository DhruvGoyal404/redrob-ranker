# Security Policy

## Reporting a vulnerability

If you discover a security issue, please **do not open a public issue**. Instead, email
**dhruv621999goyal@gmail.com** with details and steps to reproduce. We'll acknowledge
within a few days and work on a fix.

## Data handling

This project ranks candidate profiles, so we treat data carefully:

- **No candidate data is committed.** The full `candidates.jsonl` (100K profiles) is
  git-ignored; only the organizer-provided 50-row synthetic `data/sample_candidates.json`
  (already public in the hackathon bundle) is included, for the demo.
- **No secrets, API keys, or credentials** are stored in this repository.
- **No candidate data is sent to any hosted LLM.** The ranking step makes no network
  calls and no per-candidate model API calls; embeddings are computed locally/offline.

## Responsible-use note

This is a candidate-ranking system. We deliberately:

- Score on demonstrated work evidence, not keyword presence, to reduce a known
  proxy-discrimination channel.
- Do **not** use `education.tier` (college prestige) as a positive ranking feature.
- Ship a four-fifths proxy-skew audit (`src/fairness.py`) and report it honestly,
  including residual skew, rather than claiming the system is "unbiased."

Any production deployment of a hiring model should add human-in-the-loop review and
ongoing bias monitoring (see the fairness section in the README).

## Supported versions

This is a hackathon submission; the `main` branch is the supported version.
