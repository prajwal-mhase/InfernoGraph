# Datasets

The running demo uses the compact biomedical seed graph embedded in `backend/engine.py` so the project works offline.

Recommended Round 1 dataset path:

1. Download public biomedical abstracts from PubMed, BioASQ, or PubTator.
2. Keep at least 2 million tokens of actual text.
3. Extract entities for diseases, proteins, drugs, companies, pathways, and papers.
4. Load the graph into TigerGraph using `graph_pipeline/tigergraph_schema.gsql`.
5. Re-run `python benchmarking/run_benchmark.py` with at least 100 questions.
