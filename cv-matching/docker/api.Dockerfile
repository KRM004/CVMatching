# API FastAPI  —  docker compose up --build  (contexte = racine du projet)
FROM python:3.11-slim
WORKDIR /app
ENV PYTHONUNBUFFERED=1 CVMATCH_HOME=/app HF_HOME=/app/.hf_cache
# PyTorch CPU (évite ~2 Go de dépendances CUDA)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir ".[ml,llm]"
# Modèle Sentence-BERT téléchargé à la construction (démarrage rapide ensuite)
ARG SBERT_MODEL=paraphrase-multilingual-MiniLM-L12-v2
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('${SBERT_MODEL}')"
EXPOSE 8000
CMD ["uvicorn", "cvmatch.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
