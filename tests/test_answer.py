import httpx2
import pytest
from fastapi.testclient import TestClient

from team_knowledge_assistant import answer as answer_module
from team_knowledge_assistant import main
from team_knowledge_assistant.answer import (
    NOT_FOUND_TEXT,
    Answer,
    ChatTruncatedError,
    Source,
    answer_from_hits,
    build_messages,
    parse_answer,
)
from team_knowledge_assistant.config import settings
from team_knowledge_assistant.Ingest.rerank import RerankedHit


def hit(id_, score=1.0, content="text"):
    return RerankedHit(id_, f"data/docs/Doc {id_}.md", ["Doc", f"Section {id_}"], content, score, id_)


@pytest.fixture
def fake_chat(monkeypatch):
    """Replace the model: returns `reply`, and records the messages it was sent."""
    calls = []

    def install(reply):
        def chat(messages):
            calls.append(messages)
            return reply

        monkeypatch.setattr(answer_module, "chat", chat)
        return calls

    return install


# build_messages


def test_sources_are_numbered_from_one_with_file_and_heading_path():
    system, user = build_messages("q?", [hit(7), hit(3)])
    assert system["role"] == "system" and "NOT_FOUND" in system["content"]
    assert "[1] Doc 7.md > Doc > Section 7\ntext" in user["content"]
    assert "[2] Doc 3.md > Doc > Section 3\ntext" in user["content"]
    assert user["content"].endswith("Question: q?")


# parse_answer


def test_citations_map_to_hits_in_order_of_first_use():
    hits = [hit(1), hit(2), hit(3)]
    a = parse_answer("Do this [3]. Then that [1]. Again [3].", hits)
    assert a.found
    assert [(c.n, c.source) for c in a.citations] == [(3, "data/docs/Doc 3.md"), (1, "data/docs/Doc 1.md")]


def test_several_numbers_in_one_bracket_are_split():
    a = parse_answer("Both say so [1, 2].", [hit(1), hit(2)])
    assert a.text == "Both say so [1][2]."
    assert [c.n for c in a.citations] == [1, 2]


def test_invented_citation_is_removed_from_text_and_sources():
    a = parse_answer("Real [1]. Made up [7].", [hit(1), hit(2)])
    assert a.text == "Real [1]. Made up ."
    assert [c.n for c in a.citations] == [1]


def test_answer_without_valid_citations_is_not_found():
    for reply in ["An answer citing nothing.", "Only invented ones [9]."]:
        a = parse_answer(reply, [hit(1)])
        assert (a.found, a.text, a.not_found_reason) == (False, NOT_FOUND_TEXT, "no_valid_citations")


def test_not_found_reply():
    a = parse_answer("NOT_FOUND", [hit(1)])
    assert (a.found, a.citations, a.not_found_reason) == (False, [], "model_said_not_found")


def test_not_found_after_an_explanation_is_still_not_found():
    # Qwen did this with the gate off: explained with citations, then ended with NOT_FOUND.
    reply = "The sources cover body size limits [1] but not per-user rate limiting.\n\nNOT_FOUND"
    assert parse_answer(reply, [hit(1)]).not_found_reason == "model_said_not_found"


def test_status_constant_containing_not_found_is_a_real_answer():
    a = parse_answer("Use status_code=status.HTTP_404_NOT_FOUND [1].", [hit(1)])
    assert a.found


# answer_from_hits: the gate


def test_nothing_passes_the_gate_so_the_model_is_never_called(fake_chat):
    calls = fake_chat("should not be used [1]")
    below = settings.reranker_not_found_below - 0.1
    a = answer_from_hits("q?", [hit(1, score=below), hit(2, score=below)])
    assert (a.found, a.not_found_reason) == (False, "no_relevant_chunks")
    assert calls == []


def test_only_hits_that_pass_the_gate_are_sent_and_numbered(fake_chat):
    calls = fake_chat("From the second chunk [2].")
    low = settings.reranker_not_found_below - 0.1
    hits = [hit(1, score=5.0), hit(2, score=low), hit(3, score=2.0)]
    a = answer_from_hits("q?", hits)
    user = calls[0][1]["content"]
    assert "Section 2" not in user  # dropped by the gate
    assert "[2] Doc 3.md" in user  # so retrieved hit 3 is source [2]
    assert a.citations[0].source == "data/docs/Doc 3.md"


# answer_backend "copilot": no model, the caller writes the answer


@pytest.fixture
def copilot_backend(monkeypatch):
    monkeypatch.setattr(settings, "answer_backend", "copilot")


def test_copilot_backend_returns_gated_numbered_sources_without_calling_the_model(fake_chat, copilot_backend):
    calls = fake_chat("should not be used [1]")
    low = settings.reranker_not_found_below - 0.1
    a = answer_from_hits("q?", [hit(1, score=5.0), hit(2, score=low), hit(3, score=2.0)])
    assert calls == []
    assert (a.text, a.found, a.citations) == (None, True, [])
    assert [(s.n, s.source) for s in a.sources] == [(1, "data/docs/Doc 1.md"), (2, "data/docs/Doc 3.md")]


def test_copilot_backend_not_found_is_the_same_as_local(fake_chat, copilot_backend):
    calls = fake_chat("should not be used [1]")
    a = answer_from_hits("q?", [hit(1, score=settings.reranker_not_found_below - 0.1)])
    assert calls == []
    assert a == Answer(NOT_FOUND_TEXT, False, [], "no_relevant_chunks")


def test_mcp_ask_adds_instructions_only_when_no_answer_was_written(monkeypatch):
    from team_knowledge_assistant import mcp_server

    sources = [Source(1, "a.md", ["A"], "text")]
    monkeypatch.setattr(mcp_server, "answer", lambda q, k: Answer(None, True, [], sources=sources))
    assert mcp_server.ask("q?")["instructions"] == mcp_server.COPILOT_INSTRUCTIONS

    monkeypatch.setattr(mcp_server, "answer", lambda q, k: Answer("Done [1].", True, []))
    assert "instructions" not in mcp_server.ask("q?")


# POST /ask


def test_ask_endpoint_returns_the_answer(monkeypatch):
    monkeypatch.setattr(main, "answer", lambda q, k: Answer(NOT_FOUND_TEXT, False, [], "no_relevant_chunks"))
    r = TestClient(main.app).post("/ask", json={"question": "q?"})
    assert r.status_code == 200
    assert r.json() == {
        "text": NOT_FOUND_TEXT, "found": False, "citations": [], "not_found_reason": "no_relevant_chunks", "sources": []
    }


@pytest.mark.parametrize(
    ("error", "status"),
    [(ChatTruncatedError("cut off"), 502), (httpx2.ConnectError("refused"), 503)],
)
def test_ask_endpoint_maps_failures(monkeypatch, error, status):
    def fail(q, k):
        raise error

    monkeypatch.setattr(main, "answer", fail)
    assert TestClient(main.app).post("/ask", json={"question": "q?"}).status_code == status


def test_ask_endpoint_rejects_empty_question():
    assert TestClient(main.app).post("/ask", json={"question": ""}).status_code == 422
