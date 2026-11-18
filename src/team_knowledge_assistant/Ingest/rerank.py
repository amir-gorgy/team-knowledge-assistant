"""Re-order search hits with a cross-encoder (a second, slower, more accurate pass).

Vector search embeds the question and each chunk separately and compares the two vectors, so
the model never sees them side by side. A cross-encoder reads the question and one chunk
together and scores how well that chunk answers it. That is far more accurate, but it costs
one model run per (question, chunk) pair, so it only re-orders a short candidate list that a
fast first stage (vector or hybrid search) has already picked.

The model runs in-process on the CPU through ONNX (fastembed); it is downloaded on first use.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from functools import cache

from fastembed.rerank.cross_encoder import TextCrossEncoder

from team_knowledge_assistant.config import settings


@dataclass
class RerankedHit:
    id: int
    source: str
    heading_path: list[str]
    content: str
    score: float  # cross-encoder relevance: higher = better. A raw model score, not 0-1, no fixed range
    first_stage_rank: int  # where the first stage (vector/hybrid) had it, 1-based


@cache
def _model(name: str) -> TextCrossEncoder:
    # Loading takes a few seconds (and a download the first time), so load each model once.
    return TextCrossEncoder(model_name=name)


def as_document(heading_path: Sequence[str], content: str) -> str:
    """The text the cross-encoder reads: the heading path first, so a chunk that is mostly code
    still says which page and section it belongs to."""
    return " > ".join(heading_path) + "\n\n" + content


def rerank(query: str, hits: Sequence, k: int = 5, model: str | None = None) -> list[RerankedHit]:
    """Score every hit against `query` and return the best `k`, best first.

    `hits` is any first-stage result (Hit, KeywordHit, HybridHit): anything with id, source,
    heading_path and content.
    """
    if not hits:
        return []
    scores = list(_model(model or settings.reranker_model).rerank(
        query, [as_document(h.heading_path, h.content) for h in hits]
    ))
    ranked = sorted(zip(hits, scores, range(1, len(hits) + 1)), key=lambda t: t[1], reverse=True)
    return [
        RerankedHit(h.id, h.source, h.heading_path, h.content, float(score), first_rank)
        for h, score, first_rank in ranked[:k]
    ]
