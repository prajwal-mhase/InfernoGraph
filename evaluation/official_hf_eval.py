from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import sys
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.engine import InfernoGraphEngine


OUTPUT_DIR = ROOT / "evaluation" / "output"


def hf_judge(question: str, reference: str, answer: str, model: str) -> dict:
    token = os.getenv("HF_API_TOKEN")
    if not token:
        return {
            "judge_pass": None,
            "judge_score": None,
            "judge_model": model,
            "reason": "HF_API_TOKEN is not set; local judge remains active.",
        }

    prompt = (
        "You are grading a GraphRAG benchmark answer. Return only PASS or FAIL.\n"
        "PASS if the answer is factually aligned with the reference and answers the question.\n\n"
        f"Question: {question}\n"
        f"Reference: {reference}\n"
        f"Candidate answer: {answer}\n"
        "Grade:"
    )
    body = json.dumps({"inputs": prompt, "parameters": {"max_new_tokens": 8}}).encode("utf-8")
    request = urllib.request.Request(
        f"https://api-inference.huggingface.co/models/{model}",
        data=body,
        method="POST",
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            data = json.loads(response.read().decode("utf-8"))
    except urllib.error.URLError as error:
        return {"judge_pass": None, "judge_score": None, "judge_model": model, "reason": str(error)}

    generated = json.dumps(data).upper()
    passed = "PASS" in generated and "FAIL" not in generated
    return {
        "judge_pass": passed,
        "judge_score": 1.0 if passed else 0.0,
        "judge_model": model,
        "raw": data,
    }


def bertscore(candidate: str, reference: str) -> dict:
    try:
        from bert_score import score
    except ImportError:
        return {
            "bertscore_precision": None,
            "bertscore_recall": None,
            "bertscore_f1": None,
            "reason": "Install `bert-score` to run official BERTScore.",
        }

    precision, recall, f1 = score([candidate], [reference], lang="en", rescale_with_baseline=True)
    return {
        "bertscore_precision": round(float(precision[0]), 4),
        "bertscore_recall": round(float(recall[0]), 4),
        "bertscore_f1": round(float(f1[0]), 4),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run official-style Hugging Face judge + BERTScore evaluation.")
    parser.add_argument("--judge-model", default="google/flan-t5-large")
    parser.add_argument("--pipeline", default="InfernoGraph", choices=["LLM Only", "Basic RAG", "InfernoGraph"])
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    args = parser.parse_args()

    engine = InfernoGraphEngine()
    rows = []
    for case in engine.kb.evaluation_cases:
        comparison = engine.compare(case.question)
        pipeline = next(item for item in comparison["pipelines"] if item["name"] == args.pipeline)
        answer = pipeline["answer"]
        judge = hf_judge(case.question, case.gold_answer, answer, args.judge_model)
        bert = bertscore(answer, case.gold_answer)
        local = pipeline["metrics"]
        rows.append(
            {
                "case_id": case.id,
                "pipeline": args.pipeline,
                "question": case.question,
                "local_judge_pass": local["judge_pass"],
                "local_judge_score": local["judge_score"],
                "local_bertscore_proxy": local["bertscore_f1_proxy"],
                **judge,
                **bert,
            }
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "official_hf_accuracy.json"
    csv_path = output_dir / "official_hf_accuracy.csv"
    json_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted({key for row in rows for key in row}))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {json_path}")
    print(f"Wrote {csv_path}")


if __name__ == "__main__":
    main()
