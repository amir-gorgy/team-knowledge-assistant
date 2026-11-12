import pytest

from team_knowledge_assistant.Ingest.chunking import MarkdownChunker, clean_heading


def paths(chunks):
    """The heading path of each chunk, joined into one string for easy comparison."""
    return [" > ".join(c.heading_path) for c in chunks]


def squash(text):
    """Text with all whitespace removed, to check that splitting keeps every character."""
    return "".join(text.split())


# --- split_by_headings ---


def test_heading_path_follows_hierarchy():
    text = "# A\na\n## B\nb\n### C\nc\n## D\nd"
    chunks = MarkdownChunker().chunk(text, "doc.md")
    assert paths(chunks) == ["A", "A > B", "A > B > C", "A > D"]


def test_hash_inside_code_block_is_not_a_heading():
    text = "# A\n```bash\n# just a comment\n```\nafter"
    chunks = MarkdownChunker().chunk(text, "doc.md")
    assert len(chunks) == 1
    assert "# just a comment" in chunks[0].text
    assert "after" in chunks[0].text


def test_heading_only_section_is_dropped_but_stays_in_path():
    text = "# Title\n## Section\nbody"
    chunks = MarkdownChunker().chunk(text, "doc.md")
    assert paths(chunks) == ["Title > Section"]


def test_preamble_is_kept_by_default():
    chunks = MarkdownChunker().chunk("intro\n# A\na", "doc.md")
    assert chunks[0].text == "intro"
    assert chunks[0].heading_path == []


def test_preamble_is_dropped_with_skip_preamble():
    chunks = MarkdownChunker(skip_preamble=True).chunk("site nav\n# A\na", "doc.md")
    assert paths(chunks) == ["A"]


def test_source_is_set_on_every_chunk():
    chunks = MarkdownChunker().chunk("# A\na\n# B\nb", "docs/x.md")
    assert {c.source for c in chunks} == {"docs/x.md"}


def test_each_chunk_has_its_own_heading_path_list():
    chunks = MarkdownChunker().chunk("# A\na\n## B\nb", "doc.md")
    chunks[0].heading_path.append("changed")
    assert chunks[1].heading_path == ["A", "B"]


def test_empty_text_gives_no_chunks():
    assert MarkdownChunker().chunk("", "doc.md") == []


# --- clean_heading ---


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Use the **`scopes`**", "Use the scopes"),
        ("Data <u>conversion</u> ", "Data conversion"),
        ("See [the docs](https://example.com)", "See the docs"),
        ("Title ##", "Title"),
        ("response_model", "response_model"),
    ],
)
def test_clean_heading(raw, expected):
    assert clean_heading(raw) == expected


# --- split_large ---


def test_large_section_is_split_without_losing_text():
    paragraphs = [f"Paragraph {i} " + "word " * 15 for i in range(10)]
    text = "# Big\n\n" + "\n\n".join(paragraphs)
    # No overlap, so the pieces joined together must equal the original exactly.
    chunker = MarkdownChunker(max_chars=200, overlap=0)

    chunks = chunker.chunk(text, "doc.md")

    assert len(chunks) > 1
    assert all(len(c.text) <= 200 for c in chunks)
    assert all(c.heading_path == ["Big"] for c in chunks)
    assert squash("".join(c.text for c in chunks)) == squash(text)


def test_small_section_is_not_split():
    chunks = MarkdownChunker(max_chars=200).chunk("# A\nshort", "doc.md")
    assert len(chunks) == 1


def test_code_block_with_blank_lines_stays_whole():
    code = "```python\nx = 1\n\n\ny = 2\n```"
    text = "# A\n\n" + "intro " * 20 + "\n\n" + code
    chunks = MarkdownChunker(max_chars=120).chunk(text, "doc.md")
    assert any(code in c.text for c in chunks)


def test_over_long_line_is_cut():
    chunks = MarkdownChunker(max_chars=50, overlap=0).chunk("# H\n" + "a" * 130, "doc.md")
    assert all(len(c.text) <= 50 for c in chunks)
    assert squash("".join(c.text for c in chunks)) == squash("# H" + "a" * 130)


def test_oversized_code_block_is_kept_whole():
    code = "```python\n" + "\n".join(f"value_{i} = {i}" for i in range(30)) + "\n```"
    text = "# A\n\nSome intro text.\n\n" + code
    chunks = MarkdownChunker(max_chars=100).chunk(text, "doc.md")
    # The code block is far over 100 chars, but appears whole in exactly one chunk.
    assert sum(code in c.text for c in chunks) == 1


# --- overlap ---


def sentences_text(n):
    return " ".join(f"Sentence number {i} is here." for i in range(n))


def test_each_piece_starts_with_the_end_of_the_previous_one():
    text = "# Big\n\n" + "\n\n".join(sentences_text(6) for _ in range(6))
    chunker = MarkdownChunker(max_chars=400, overlap=0.1)  # overlap up to 40 chars

    chunks = chunker.chunk(text, "doc.md")

    assert len(chunks) > 1
    for previous, current in zip(chunks, chunks[1:]):
        repeated = current.text.split("\n\n")[0]
        assert previous.text.endswith(repeated)
        assert 0 < len(repeated) <= 40
        assert len(current.text) <= 400


def test_overlap_uses_whole_sentences():
    text = "# Big\n\n" + "\n\n".join(sentences_text(6) for _ in range(6))
    chunks = MarkdownChunker(max_chars=400, overlap=0.1).chunk(text, "doc.md")
    for chunk in chunks[1:]:
        assert chunk.text.startswith("Sentence number")


def test_overlap_never_repeats_code():
    code = "```python\nx = 1\ny = 2\n```"
    text = "# A\n\n" + "\n\n".join([sentences_text(4), code] * 6)
    chunks = MarkdownChunker(max_chars=200, overlap=0.2).chunk(text, "doc.md")
    for chunk in chunks:
        fences = sum(line.startswith("```") for line in chunk.text.splitlines())
        assert fences % 2 == 0  # no chunk holds half a code block
    assert squash("".join(c.text for c in chunks)).count(squash(code)) == 6


def test_zero_overlap_repeats_nothing():
    text = "# Big\n\n" + "\n\n".join(sentences_text(6) for _ in range(6))
    chunks = MarkdownChunker(max_chars=400, overlap=0).chunk(text, "doc.md")
    assert squash("".join(c.text for c in chunks)) == squash(text)
