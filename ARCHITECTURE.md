# InfernoGraph Architecture

InfernoGraph compares three pipelines for the same question:

1. LLM Only
2. Basic RAG
3. InfernoGraph (graph-aware)

Core backend components:

- `backend/main.py` exposes `/compare`, `/benchmark`, `/health`, `/compliance`.
- `backend/engine.py` runs retrieval, graph routing, compression, and scoring.
- `backend/router.py` selects retrieval mode.
- `backend/compression_engine.py` compresses evidence to triples.

Frontend:

- `frontend/dashboard.html` presents side-by-side pipeline outputs.

Dataset:

- `datasets/manifest.json` tracks Round 1 dataset stats and licensing evidence.
