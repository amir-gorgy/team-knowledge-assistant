from dataclasses import dataclass

import pytest

from team_knowledge_assistant.Ingest import rerank as rerank_module
from team_knowledge_assistant.Ingest.rerank import as_document, rerank


@dataclass
class FakeHit:
    id: int
    source: str
    heading_path: list[str]
    content: str


class FakeModel:
    """Scores a document by how many times it contains the word "yes"."""

    def rerank(self, query, documents):
        return [float(d.count("yes")) for d in documents]


@pytest.fixture(autouse=True)
def fake_model(monkeypatch):
    monkeypatch.setattr(rerank_module, "_model", lambda name: FakeModel())


def hit(id_, content):
    return FakeHit(id_, "doc.md", ["Doc", f"Section {id_}"], content)


def test_orders_by_score_and_keeps_first_stage_rank():
    hits = [hit(1, "no"), hit(2, "yes yes"), hit(3, "yes")]
    out = rerank("q", hits, k=3)
    assert [h.id for h in out] == [2, 3, 1]
    assert [h.first_stage_rank for h in out] == [2, 3, 1]
    assert [h.score for h in out] == [2.0, 1.0, 0.0]


def test_cuts_to_k():
    hits = [hit(i, "yes" * i) for i in range(1, 6)]
    assert [h.id for h in rerank("q", hits, k=2)] == [5, 4]


def test_empty_input_gives_nothing():
    assert rerank("q", [], k=5) == []


def test_document_starts_with_heading_path():
    assert as_document(["Page", "Section"], "body") == "Page > Section\n\nbody"


def test_heading_path_is_scored_too():
    # The heading text is part of what the model reads, so a heading can lift a chunk.
    hits = [hit(1, "body"), FakeHit(2, "doc.md", ["yes"], "body")]
    assert rerank("q", hits, k=1)[0].id == 2
