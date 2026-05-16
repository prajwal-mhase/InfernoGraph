from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Iterable, Sequence


@dataclass(frozen=True)
class EvidenceTriple:
    source: str
    relation: str
    target: str
    confidence: float = 0.8
    source_doc: str | None = None

    def compact(self) -> str:
        suffix = f" [{self.source_doc}]" if self.source_doc else ""
        return f"{self.source} -> {self.relation} -> {self.target}{suffix}"


@dataclass(frozen=True)
class CompressedContext:
    text: str
    triples: list[EvidenceTriple]
    original_tokens: int
    compressed_tokens: int
    compression_ratio: float
    evidence_density: float


def estimate_tokens(text: str) -> int:
    """Stable token estimate without provider-specific SDKs."""

    if not text:
        return 0
    return max(1, round(len(text.split()) * 1.32))


def _dedupe_triples(triples: Iterable[EvidenceTriple]) -> list[EvidenceTriple]:
    seen: OrderedDict[tuple[str, str, str], EvidenceTriple] = OrderedDict()
    for triple in triples:
        key = (triple.source.lower(), triple.relation.lower(), triple.target.lower())
        if key not in seen or triple.confidence > seen[key].confidence:
            seen[key] = triple
    return list(seen.values())


def compress_graph_evidence(
    triples: Iterable[EvidenceTriple],
    raw_context: Sequence[str] | None = None,
    max_triples: int = 16,
) -> CompressedContext:
    ranked = sorted(
        _dedupe_triples(triples),
        key=lambda item: (item.confidence, item.source.lower(), item.target.lower()),
        reverse=True,
    )[:max_triples]

    lines = [triple.compact() for triple in ranked]
    compressed = "\n".join(lines)
    original_text = "\n\n".join(raw_context or lines)
    original_tokens = estimate_tokens(original_text)
    compressed_tokens = estimate_tokens(compressed)
    ratio = 0.0
    if original_tokens:
        ratio = max(0.0, 1.0 - (compressed_tokens / original_tokens))
    density = round(len(ranked) / max(1, compressed_tokens), 4)

    return CompressedContext(
        text=compressed,
        triples=ranked,
        original_tokens=original_tokens,
        compressed_tokens=compressed_tokens,
        compression_ratio=round(ratio, 4),
        evidence_density=density,
    )


def compress_paths(paths: Sequence[Sequence[EvidenceTriple]], raw_context: Sequence[str] | None = None) -> CompressedContext:
    flattened: list[EvidenceTriple] = []
    for path in paths:
        flattened.extend(path)
    return compress_graph_evidence(flattened, raw_context=raw_context)


def compress_context(triples):
    """Compatibility wrapper used by the initial scaffold."""

    evidence = [EvidenceTriple(source=s, relation=r, target=o) for s, r, o in triples]
    return compress_graph_evidence(evidence).text
