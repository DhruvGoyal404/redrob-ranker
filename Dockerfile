# Reproduces the RANKING step (rank.py) in a clean, CPU-only container that matches
# the challenge constraints: no GPU, no network needed at run time, < 5 min, < 16 GB.
# Only the ranking deps are installed (numpy + rank-bm25); no torch, no pandas.
#
# Build:
#   docker build -t redrob-ranker .
#
# Run (offline, mounting the candidate file and the precomputed embeddings):
#   docker run --rm --network none \
#     -v "$PWD/candidates.jsonl:/app/candidates.jsonl:ro" \
#     -v "$PWD/artifacts:/app/artifacts:ro" \
#     -v "$PWD/out:/app/out" \
#     redrob-ranker --candidates /app/candidates.jsonl --out /app/out/submission.csv --validate
#
# Without the artifacts mount it still runs, degrading to a valid BM25-only ranking.
FROM python:3.13-slim

WORKDIR /app

COPY requirements-rank.txt .
RUN pip install --no-cache-dir -r requirements-rank.txt

# Only what the ranking step needs.
COPY src/ ./src/
COPY rank.py validate_submission.py ./

ENTRYPOINT ["python", "rank.py"]
CMD ["--candidates", "/app/candidates.jsonl", "--out", "/app/out/submission.csv", "--validate"]
