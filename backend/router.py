from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


MULTI_HOP_TERMS = {
    "indirect",
    "connected",
    "connect",
    "between",
    "through",
    "path",
    "pathway",
    "multi-hop",
    "multi hop",
    "relationship",
    "linked",
    "associated",
}

COMMUNITY_TERMS = {
    "summarize",
    "overview",
    "theme",
    "cluster",
    "community",
    "landscape",
    "compare areas",
}

EXACT_PATH_TERMS = {
    "which drugs",
    "which companies",
    "trace",
    "show path",
    "evidence chain",
    "target proteins",
    "targets",
}


@dataclass(frozen=True)
class RouteDecision:
    mode: str
    depth: int
    token_budget: int
    confidence_threshold: float
    reasons: list[str]


class AdaptiveQueryRouter:
    """Classifies a question into the cheapest retrieval strategy likely to work."""

    def route(self, query: str, entities: Iterable[str] | None = None) -> RouteDecision:
        text = query.lower()
        entity_count = len(list(entities or []))
        reasons: list[str] = []

        multi_hop_score = sum(1 for term in MULTI_HOP_TERMS if term in text)
        community_score = sum(1 for term in COMMUNITY_TERMS if term in text)
        exact_path_score = sum(1 for term in EXACT_PATH_TERMS if term in text)

        if multi_hop_score:
            reasons.append("relationship language")
        if community_score:
            reasons.append("community-level wording")
        if exact_path_score:
            reasons.append("explicit path or target request")
        if entity_count >= 2:
            reasons.append("multiple recognized graph entities")

        if exact_path_score and (multi_hop_score or entity_count >= 2):
            return RouteDecision(
                mode="path_constrained_reasoning",
                depth=4,
                token_budget=1800,
                confidence_threshold=0.72,
                reasons=reasons or ["exact relationship tracing"],
            )

        if multi_hop_score or entity_count >= 2:
            return RouteDecision(
                mode="multi_hop_entity_traversal",
                depth=3,
                token_budget=2200,
                confidence_threshold=0.68,
                reasons=reasons or ["multi-entity query"],
            )

        if community_score:
            return RouteDecision(
                mode="community_retrieval",
                depth=2,
                token_budget=1600,
                confidence_threshold=0.62,
                reasons=reasons or ["community summary query"],
            )

        return RouteDecision(
            mode="semantic_graph_search",
            depth=1,
            token_budget=1200,
            confidence_threshold=0.55,
            reasons=reasons or ["single-hop semantic lookup"],
        )


def route_query(query: str) -> str:
    """Backward-compatible helper for older demo code."""

    return AdaptiveQueryRouter().route(query).mode
