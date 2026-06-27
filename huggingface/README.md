---
title: Redrob Intelligent Candidate Ranker
emoji: 🧭
colorFrom: blue
colorTo: indigo
sdk: streamlit
sdk_version: 1.40.0
app_file: app/streamlit_app.py
pinned: false
license: mit
---

# Redrob Intelligent Candidate Ranker — live demo

Evidence-based AI candidate ranker for the India Runs (Redrob) Track-1 challenge.
This Space runs the **full hybrid pipeline live** — BM25 + dense `bge-small` embeddings
fused with RRF, then a 7-component signal scorer + trap gate, with grounded reasoning.

Paste/edit the target JD, hit **Rank candidates**, and inspect the per-candidate score
breakdown. Toggle **Inject a keyword-stuffer** to watch the trap gate demote it.

Code + methodology: https://github.com/DhruvGoyal404/redrob-ranker

> This file is the Hugging Face Space config (the frontmatter above). It is intentionally
> separate from the repo's main README.md so the GitHub README stays clean.
