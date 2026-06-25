# Contributing

This is a hackathon submission for the **India Runs by Redrob AI - Track 1**
Intelligent Candidate Discovery & Ranking Challenge. It is primarily a competition
entry, but contributions, issues, and suggestions are welcome.

## Getting started

```bash
pip install -r requirements.txt
# Drop the hackathon candidates.jsonl into the repo root, then:
python precompute.py --candidates ./candidates.jsonl --artifacts ./artifacts   # offline, GPU-friendly
python rank.py --candidates ./candidates.jsonl --out ./submission.csv --validate
python -m eval.evaluate --candidates ./candidates.jsonl --submission ./submission.csv
streamlit run app/streamlit_app.py
```

## Project conventions

- **The ranking step (`rank.py`) must stay CPU-only, offline, and < 5 minutes.** Do not
  add torch / network / hosted-LLM calls to the ranking path. Heavy work (embeddings)
  belongs in `precompute.py`.
- **Score on evidence, not keywords.** New signals should read demonstrated work from
  `career_history`, not reward bare skill-name presence.
- Keep scoring weights and trap heuristics in `src/config.py` so they stay transparent
  and tunable in one place.
- Match the surrounding code style; concentrate comments where the logic is non-obvious
  (trap heuristics, fusion, weights).

## Pull requests

1. Branch from `main`.
2. Keep changes focused; explain the *why* in the PR description.
3. Run `python -m py_compile` on changed files and re-run the validator + eval before
   opening the PR.

## Reporting issues

Open a GitHub issue with steps to reproduce. For anything security-related, see
[SECURITY.md](SECURITY.md).
