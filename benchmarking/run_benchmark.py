from __future__ import annotations

import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.engine import InfernoGraphEngine


OUTPUT_DIR = ROOT / "benchmarking" / "output"


def main() -> None:
    engine = InfernoGraphEngine()
    report = engine.benchmark()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    json_path = OUTPUT_DIR / "benchmark_report.json"
    csv_path = OUTPUT_DIR / "benchmark_cases.csv"
    md_path = ROOT / "BENCHMARK_REPORT.md"

    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "case_id",
                "question",
                "pipeline",
                "tokens",
                "prompt_tokens",
                "completion_tokens",
                "cost_usd",
                "latency_ms",
                "judge_score",
                "judge_pass",
                "bertscore_f1_proxy",
            ]
        )
        for index, case in enumerate(report["cases"], start=1):
            for pipeline in case["pipelines"]:
                metrics = pipeline["metrics"]
                writer.writerow(
                    [
                        f"Q{index}",
                        case["question"],
                        pipeline["name"],
                        metrics["tokens"],
                        metrics["prompt_tokens"],
                        metrics["completion_tokens"],
                        metrics["cost_usd"],
                        metrics["latency_ms"],
                        metrics["judge_score"],
                        metrics["judge_pass"],
                        metrics["bertscore_f1_proxy"],
                    ]
                )

    md_path.write_text(render_markdown(report), encoding="utf-8")

    print(f"Wrote {json_path}")
    print(f"Wrote {csv_path}")
    print(f"Updated {md_path}")


def render_markdown(report: dict) -> str:
    aggregate_rows = []
    for name, row in report["aggregates"].items():
        aggregate_rows.append(
            "| {name} | {tokens} | ${cost:.5f} | {latency} ms | {judge:.3f} | {pass_rate:.0%} |".format(
                name=name,
                tokens=row["avg_tokens"],
                cost=row["avg_cost_usd"],
                latency=row["avg_latency_ms"],
                judge=row["avg_judge_score"],
                pass_rate=row["pass_rate"],
            )
        )

    headline = report["headline"]
    manifest_path = ROOT / "datasets" / "manifest.json"
    dataset_line = "Dataset manifest not found. Run `python datasets/build_open_access_pmc_dataset.py --target-tokens 2000000`."
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        dataset_line = (
            f"{manifest.get('token_count', 0):,} tokens across "
            f"{manifest.get('document_count', 0):,} PubMed Central Open Access documents."
        )
    return """# Benchmark Report

Local deterministic benchmark over the bundled biomedical evaluation set.

## Headline

- Token reduction vs Basic RAG: {token_reduction:.0%}
- InfernoGraph judge pass rate: {pass_rate:.0%}
- InfernoGraph BERTScore F1 proxy: {bert_proxy:.3f}

## Dataset Compliance

{dataset_line}

## Aggregate Metrics

| Pipeline | Avg Tokens | Avg Cost | Avg Latency | Avg Judge Score | Pass Rate |
|---|---:|---:|---:|---:|---:|
{rows}

## Method

Each evaluation question is run through the same three pipelines: LLM Only, Basic RAG, and InfernoGraph. The local judge measures required entity coverage plus lexical semantic F1 against bundled gold answers. For official submission scoring, replace the local proxy with the Hugging Face LLM-as-a-Judge and BERTScore scripts from `evaluation/accuracy_eval.py`.

Official-style adapters are included in `evaluation/official_hf_eval.py`. When `HF_API_TOKEN` and `bert-score` are available, it writes Hugging Face judge and BERTScore outputs to `evaluation/output/`.

## Reproduce

```bash
python benchmarking/run_benchmark.py
python evaluation/official_hf_eval.py
python datasets/validate_dataset.py
```
""".format(
        token_reduction=headline["token_reduction_vs_basic_rag"],
        pass_rate=headline["judge_pass_rate"],
        bert_proxy=headline["bertscore_f1_proxy"],
        dataset_line=dataset_line,
        rows="\n".join(aggregate_rows),
    )


if __name__ == "__main__":
    main()
