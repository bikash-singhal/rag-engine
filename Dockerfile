FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt ./

RUN pip install --no-cache-dir -r requirements.txt

# Pre-download HF models
RUN python - <<'PY'
from sentence_transformers import SentenceTransformer, CrossEncoder

SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
CrossEncoder("cross-encoder/ms-marco-MiniLM-L6-v2")

print("Models downloaded successfully.")
PY

COPY src ./src
COPY data/documents ./data/documents
COPY data/indexes/default ./data/indexes/default

RUN useradd -m appuser && \
    mkdir -p data/indexes logs && \
    chown -R appuser:appuser /app

USER appuser

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]