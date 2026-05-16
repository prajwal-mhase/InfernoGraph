# InfernoGraph

Adaptive Multi-Hop GraphRAG Engine for Ultra-Low-Token Reasoning.

InfernoGraph is a competition-ready GraphRAG inference optimization platform for the TigerGraph GraphRAG Inference Hackathon. It compares three pipelines side by side:

1. LLM Only
2. Basic RAG
3. InfernoGraph GraphRAG

The core idea is simple: do not send raw chunks when the graph can send typed evidence paths.

## What Makes It Different

- Adaptive router chooses one of four retrieval modes per query.
- Multi-hop graph traversal finds disease -> pathway -> protein -> drug -> company chains.
- Graph-aware compression converts chunks into compact evidence triples.
- Confidence-aware fallback adds raw chunks only when graph evidence is weak.
- Benchmark API reports tokens, cost, latency, local judge score, pass/fail, and BERTScore F1 proxy.
- Dashboard visualizes the reasoning graph, token heatmap, and monthly cost projection.

## Run Locally

```bash
pip install -r requirements.txt
uvicorn backend.main:app --reload
```

Open the dashboard:

```bash
cd frontend
python -m http.server 5500
```

Then visit:

```text
http://127.0.0.1:5500/dashboard.html
```

Backend API:

```text
http://127.0.0.1:8000
```

## API

- `POST /compare` runs one question through all three pipelines.
- `GET /benchmark` runs the bundled evaluation set.
- `GET /health` returns backend health and asset status.
- `GET /compliance` returns live hackathon rule checks.

## Reproduce Benchmark

```bash
python benchmarking/run_benchmark.py
```

Outputs:

- `benchmarking/output/benchmark_report.json`
- `benchmarking/output/benchmark_cases.csv`
- updated `BENCHMARK_REPORT.md`

## Rule Compliance

Run the live audit:

```bash
uvicorn backend.main:app --reload
```

Then open:

```text
http://127.0.0.1:8000/compliance
```

## Gemini Configuration

Set Gemini environment variables before running if you want live model calls:

```bash
set GEMINI_API_KEY=your_key
set GEMINI_API_URL=https://generativelanguage.googleapis.com
set GEMINI_MODEL=gemini-2.0-flash
```

If Gemini is unavailable (quota/model issues), InfernoGraph falls back to deterministic local synthesis so the demo stays runnable.

## Dataset

The Round 1 dataset is built from PubMed Central Open Access:

```bash
python datasets/build_open_access_pmc_dataset.py --target-tokens 2000000
python datasets/validate_dataset.py
```

The generated manifest records:

- `2,012,461` tokens
- `166` open-access PMC documents

## Official Accuracy Evaluation

```bash
pip install -r requirements-official.txt
set HF_API_TOKEN=your_token
python evaluation/official_hf_eval.py
```

## Docker

```bash
docker compose up --build
```

## Quick Start (Windows PowerShell)

From the repository root you can use the included helpers:

```powershell
.\start_backend.ps1    # starts the FastAPI backend on port 8000
.\start_frontend.ps1   # serves frontend on port 5500
```
