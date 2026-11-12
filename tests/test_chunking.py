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
    chunker = MarkdownChunker(max_chars=200)

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
    chunks = MarkdownChunker(max_chars=50).chunk("# H\n" + "a" * 130, "doc.md")
    assert all(len(c.text) <= 50 for c in chunks)
    assert squash("".join(c.text for c in chunks)) == squash("# H" + "a" * 130)
