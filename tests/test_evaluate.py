from dataclasses import dataclass

import pytest

from team_knowledge_assistant.scripts.evaluate import gold_rank, reciprocal_rank


@dataclass
class FakeHit:
    source: str
    heading_path: list[str]


A = {"file": "a.md", "heading_path": ["A", "Intro"]}
B = {"file": "b.md", "heading_path": ["B", "Usage"]}


def hits(*pairs):
    return [FakeHit(f"data/sample-docs-md/{f}", list(p)) for f, p in pairs]


def test_any_takes_the_best_gold_rank():
    h = hits(("x.md", ["X"]), ("b.md", ["B", "Usage"]), ("a.md", ["A", "Intro"]))
    assert gold_rank(h, [A, B], "any") == 2


def test_all_takes_the_worst_gold_rank():
    h = hits(("a.md", ["A", "Intro"]), ("x.md", ["X"]), ("b.md", ["B", "Usage"]))
    assert gold_rank(h, [A, B], "all") == 3


def test_all_is_none_when_one_gold_is_missing():
    h = hits(("a.md", ["A", "Intro"]), ("x.md", ["X"]))
    assert gold_rank(h, [A, B], "all") is None


def test_any_is_none_when_no_gold_is_found():
    assert gold_rank(hits(("x.md", ["X"])), [A], "any") is None


def test_matches_on_file_name_not_full_path():
    h = [FakeHit("/elsewhere/a.md", ["A", "Intro"])]
    assert gold_rank(h, [A], "any") == 1


def test_parent_heading_is_not_a_match():
    # Same file, but the hit is the section above the gold one.
    assert gold_rank(hits(("a.md", ["A"])), [A], "any") is None


def test_split_section_counts_at_its_first_piece():
    # A long section becomes several chunks with the same heading path.
    h = hits(("x.md", ["X"]), ("a.md", ["A", "Intro"]), ("a.md", ["A", "Intro"]))
    assert gold_rank(h, [A], "any") == 2


def test_none_match_cannot_be_ranked():
    with pytest.raises(ValueError):
        gold_rank([], [], "none")


def test_reciprocal_rank():
    assert [reciprocal_rank(r) for r in (1, 2, 4, None)] == [1.0, 0.5, 0.25, 0.0]
