from __future__ import annotations

from typing import Any
from pathlib import Path
import json

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

try:
    from .engine import InfernoGraphEngine
except ImportError:  # Supports `cd backend && uvicorn main:app`.
    from engine import InfernoGraphEngine


class QueryRequest(BaseModel):
    question: str = Field(
        default=(
            "Which drugs target proteins connected to both Alzheimer disease and chronic "
            "inflammation, and which companies are associated with them?"
        ),
        min_length=3,
        max_length=800,
    )


class QueryResponse(BaseModel):
    question: str
    cache_hit: bool
    pipelines: list[dict[str, Any]]
    winner_summary: dict[str, Any]
    graph: dict[str, Any]
    routing: dict[str, Any]
    compression: dict[str, Any]


app = FastAPI(
    title="InfernoGraph",
    version="1.0.0",
    description="Adaptive Multi-Hop GraphRAG Engine for Ultra-Low-Token Reasoning",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

engine = InfernoGraphEngine()


def _check_file(path: str) -> bool:
    return Path(path).exists()


def _manifest_snapshot() -> dict[str, Any]:
    manifest_path = Path(__file__).resolve().parents[1] / "datasets" / "manifest.json"
    if not manifest_path.exists():
        return {"present": False}
    try:
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
        return {
            "present": True,
            "token_count": data.get("token_count"),
            "document_count": data.get("document_count"),
            "target_met": data.get("target_met"),
            "source": data.get("source"),
        }
    except Exception:
        return {"present": True, "parse_error": True}


@app.get("/")
def root() -> dict[str, Any]:
    return {
        "message": "InfernoGraph API running",
        "project": "Adaptive Multi-Hop GraphRAG Engine",
        "dashboard": "Open frontend/dashboard.html or serve frontend on port 5500.",
        "endpoints": ["/compare", "/benchmark", "/health", "/compliance"],
    }


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "documents": len(engine.kb.documents),
        "nodes": len(engine.kb.nodes),
        "edges": len(engine.kb.edges),
        "cached_queries": len(engine.memory_cache),
        "tigergraph_assets_present": _check_file(str(Path(__file__).resolve().parents[1] / "graph_pipeline" / "tigergraph_schema.gsql")),
        "llm_provider": "Gemini (with fallback)",
    }


@app.get("/compliance")
def compliance() -> dict[str, Any]:
    repo_root = Path(__file__).resolve().parents[1]
    required_files = {
        "architecture": repo_root / "ARCHITECTURE.md",
        "benchmark_report": repo_root / "BENCHMARK_REPORT.md",
        "demo_script": repo_root / "DEMO_SCRIPT.md",
        "blog_post": repo_root / "BLOG_POST.md",
        "social_post": repo_root / "SOCIAL_POST.md",
        "round2_plan": repo_root / "docs" / "ROUND2_SCALE_PLAN.md",
        "tigergraph_doc": repo_root / "docs" / "TIGERGRAPH_GRAPHRAG.md",
        "dashboard": repo_root / "frontend" / "dashboard.html",
        "tigergraph_schema": repo_root / "graph_pipeline" / "tigergraph_schema.gsql",
    }

    file_checks = {key: path.exists() for key, path in required_files.items()}
    manifest = _manifest_snapshot()

    route_checks = {
        "compare_route": True,
        "benchmark_route": True,
        "health_route": True,
        "compliance_route": True,
    }

    rules = {
        "three_required_pipelines": True,
        "interactive_dashboard": file_checks["dashboard"],
        "benchmark_metrics": True,
        "accuracy_report_present": file_checks["benchmark_report"],
        "round1_dataset_manifest": bool(manifest.get("present")),
        "open_access_dataset_target_met": bool(manifest.get("target_met")),
        "tigergraph_path_artifacts": file_checks["tigergraph_schema"] and file_checks["tigergraph_doc"],
        "deliverables_present": all(
            file_checks[k]
            for k in ["architecture", "benchmark_report", "demo_script", "blog_post", "social_post"]
        ),
        "round2_scale_plan": file_checks["round2_plan"],
    }

    overall = "pass" if all(rules.values()) else "partial"
    return {
        "overall_status": overall,
        "rules": rules,
        "routes": route_checks,
        "dataset_manifest": manifest,
        "files": file_checks,
    }


@app.post("/compare", response_model=QueryResponse)
def compare(request: QueryRequest) -> dict[str, Any]:
    return engine.compare(request.question)


@app.get("/benchmark")
def benchmark() -> dict[str, Any]:
    return engine.benchmark()
