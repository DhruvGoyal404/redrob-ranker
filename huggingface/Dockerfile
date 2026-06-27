# Hugging Face Spaces (Docker SDK) - runs the Streamlit demo on port 7860.
# Copy this file to the ROOT of your HF Space (as "Dockerfile"), along with
# requirements.txt, README.md (the frontmatter one), and the app/ src/ data/ folders.
FROM python:3.11-slim

WORKDIR /app

# Writable caches for the sentence-transformers model download (HF runs non-root).
ENV HF_HOME=/app/.cache \
    TRANSFORMERS_CACHE=/app/.cache \
    USE_TF=0 \
    TOKENIZERS_PARALLELISM=false

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/
COPY src/ ./src/
COPY data/ ./data/

RUN mkdir -p /app/.cache && chmod -R 777 /app/.cache

EXPOSE 7860
CMD ["streamlit", "run", "app/streamlit_app.py", \
     "--server.port=7860", "--server.address=0.0.0.0", \
     "--server.enableCORS=false", "--server.enableXsrfProtection=false"]
