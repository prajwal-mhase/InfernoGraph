# Benchmark Report

This repository exposes benchmark results via `GET /benchmark`.

Summary expectation for Round 1:

- InfernoGraph should reduce tokens vs Basic RAG.
- InfernoGraph should keep or improve judge score vs Basic RAG.

To regenerate:

```bash
python benchmarking/run_benchmark.py
```
