from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.engine import InfernoGraphEngine


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate InfernoGraph answer quality.")
    parser.add_argument(
        "--question",
        default=None,
        help="Optional single question. Defaults to the bundled benchmark set.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print full JSON output.",
    )
    args = parser.parse_args()

    engine = InfernoGraphEngine()
    if args.question:
        report = engine.compare(args.question)
    else:
        report = engine.benchmark()

    if args.json:
        print(json.dumps(report, indent=2))
        return

    if args.question:
        print_single(report)
    else:
        print_benchmark(report)


def print_single(report: dict) -> None:
    print(report["question"])
    for pipeline in report["pipelines"]:
        metrics = pipeline["metrics"]
        print(
            "{name}: judge={judge:.3f}, pass={passed}, bert_proxy={bert:.3f}, tokens={tokens}".format(
                name=pipeline["name"],
                judge=metrics["judge_score"],
                passed=metrics["judge_pass"],
                bert=metrics["bertscore_f1_proxy"],
                tokens=metrics["tokens"],
            )
        )


def print_benchmark(report: dict) -> None:
    print("InfernoGraph evaluation aggregate")
    print("Token reduction vs Basic RAG: {:.0%}".format(report["headline"]["token_reduction_vs_basic_rag"]))
    print("Judge pass rate: {:.0%}".format(report["headline"]["judge_pass_rate"]))
    print("BERTScore F1 proxy: {:.3f}".format(report["headline"]["bertscore_f1_proxy"]))
    for name, row in report["aggregates"].items():
        print(
            "{name}: tokens={tokens}, judge={judge:.3f}, pass_rate={pass_rate:.0%}".format(
                name=name,
                tokens=row["avg_tokens"],
                judge=row["avg_judge_score"],
                pass_rate=row["pass_rate"],
            )
        )


if __name__ == "__main__":
    main()
