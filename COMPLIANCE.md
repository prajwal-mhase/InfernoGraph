# Hackathon Rule Compliance

Generated from `GET /compliance`.

| Rule | Status | Evidence |
|---|---|---|
| Three required pipelines | Pass | `POST /compare` returns `LLM Only`, `Basic RAG`, `InfernoGraph`. |
| Interactive comparison dashboard | Pass | `frontend/dashboard.html` compares the three pipelines. |
| Benchmark metrics | Pass | `GET /benchmark` returns token/cost/latency and quality fields per pipeline. |
| Round 1 dataset | Pass | `datasets/manifest.json` includes token and document counts with target status. |
| Public/properly licensed text | Pass | Dataset manifest records PMC Open Access source and license preservation note. |
| TigerGraph GraphRAG path | Pass | `graph_pipeline/tigergraph_schema.gsql` and `docs/TIGERGRAPH_GRAPHRAG.md` are present. |
| Deliverables | Pass | Architecture, benchmark report, demo script, blog post, and social post are present. |
| Round 2 scale plan | Pass | `docs/ROUND2_SCALE_PLAN.md` is present. |
