"""Convert the PDFs in sample-docs-pdf/ to Markdown files in sample-docs-md/.

pymupdf4llm turns a PDF into Markdown with headings and ``` code fences, but inside
code blocks it drops every space and all indentation:

    defcreate_access_token(data:dict,expires_delta:timedelta|None=None):
    to_encode=data.copy()

The PDF itself still has the spaces. So after converting, each code block is
rebuilt from the PDF's raw characters:

  1. Collect every line set in a monospace font (code) with its position on the page.
  2. Find which of those lines a code block came from, by comparing them with all
     whitespace removed (the one thing the broken and the raw versions agree on).
  3. Replace the block with the raw lines, indented by how far right each line
     starts, with blank lines where the vertical gap between lines is larger.

Line wraps can't be undone: a code line too long for the page continues on the
next line at indentation 0. Inside brackets that is still valid Python.

Run from anywhere: uv run python data/pdf-to-md.py
"""

import bisect
import re
import statistics
from dataclasses import dataclass
from pathlib import Path

import pymupdf

DATA_DIR = Path(__file__).parent
PDF_DIR = DATA_DIR / "sample-docs-pdf"
MD_DIR = DATA_DIR / "sample-docs-md"

# A fenced code block in Markdown: the opening ``` line, the code (group 1), the closing ```.
# re.M lets ^ and $ match at each line; re.S lets .*? run across lines.
FENCED_BLOCK_PATTERN = re.compile(r"^```[^\n]*\n(.*?)^```[ \t]*$", re.M | re.S)

# Bit in a PDF text span's "flags" that is set when its font is monospaced.
MONOSPACE_FLAG = 8

# Cap on blank lines inserted between two code lines, so a large gap
# doesn't produce a wall of empty lines.
MAX_BLANK_LINES = 2


@dataclass
class CodeLine:
    """One line of code as it sits on the PDF page."""

    text: str  # the characters, without leading or trailing whitespace
    x: float  # left edge, in points; deeper indentation = larger x
    y: float  # top edge, in points; used to spot blank lines
    page: int
    char_width: float  # average width of one character, in points


@dataclass
class RepairStats:
    """How many code blocks were found, and how many could be rebuilt."""

    blocks: int = 0
    repaired: int = 0
    not_found: int = 0  # no matching lines in the PDF; left as pymupdf4llm wrote them


def pdf_to_markdown(pdf_path: Path) -> tuple[str, RepairStats]:
    """Convert a PDF to Markdown with readable code blocks."""
    doc = pymupdf.open(pdf_path)
    markdown = doc.to_markdown(use_ocr=False, header=False, footer=False)
    return repair_code_blocks(markdown, doc)


def squash(text: str) -> str:
    """Text with all whitespace removed: the form the broken code blocks are in."""
    return "".join(text.split())


def extract_code_lines(doc: pymupdf.Document) -> list[CodeLine]:
    """Every line of the PDF set entirely in a monospace font, in reading order.

    Lines that mix fonts are prose with inline `code` in them, so they are skipped.
    """
    code_lines = []
    for page in doc:
        # "rawdict" gives every character with its own bounding box.
        for block in page.get_text("rawdict")["blocks"]:
            for line in block.get("lines", []):  # image blocks have no "lines"
                spans = [span for span in line["spans"] if span["chars"]]
                if not spans or not all(span["flags"] & MONOSPACE_FLAG for span in spans):
                    continue
                chars = [char for span in spans for char in span["chars"]]
                text = "".join(char["c"] for char in chars).strip()
                if not text:
                    continue
                width = (chars[-1]["bbox"][2] - chars[0]["bbox"][0]) / len(chars)
                code_lines.append(
                    CodeLine(
                        text=text,
                        x=line["bbox"][0],
                        y=line["bbox"][1],
                        page=page.number,
                        char_width=width,
                    )
                )
    return code_lines


def repair_code_blocks(markdown: str, doc: pymupdf.Document) -> tuple[str, RepairStats]:
    """Replace each fenced code block in markdown with its lines rebuilt from the PDF."""
    code_lines = extract_code_lines(doc)
    stats = RepairStats()
    if not code_lines:
        return markdown, stats

    # Put all code lines, squashed, into one long string, and remember where each
    # line starts and ends in it. A code block from the Markdown, squashed the same
    # way, can then be found in it with a plain str.find().
    squashed_lines = [squash(line.text) for line in code_lines]
    all_code = "".join(squashed_lines)
    line_starts = []
    position = 0
    for squashed in squashed_lines:
        line_starts.append(position)
        position += len(squashed)
    line_ends = [start + len(s) for start, s in zip(line_starts, squashed_lines)]

    # Typical distance from one code line to the next; a gap twice this size
    # means one blank line.
    gaps = [
        b.y - a.y
        for a, b in zip(code_lines, code_lines[1:])
        if a.page == b.page and b.y > a.y
    ]
    line_height = statistics.median(gaps) if gaps else None

    # Code blocks appear in the same order in the Markdown and the PDF, so each
    # search starts where the previous match ended. That also matches a code example
    # repeated several times in one document to the right copy.
    search_from = 0

    def rebuild(match: re.Match) -> str:
        nonlocal search_from
        stats.blocks += 1
        target = squash(match.group(1))
        if not target:
            return match.group(0)

        found_at = all_code.find(target, search_from)
        if found_at == -1:  # out of order: try once more from the start
            found_at = all_code.find(target)
        if found_at == -1:
            stats.not_found += 1
            return match.group(0)
        found_end = found_at + len(target)

        # Which code lines does the match cover? It must start at the start of a
        # line and end at the end of one; anything else is a coincidental match.
        first = bisect.bisect_right(line_starts, found_at) - 1
        last = bisect.bisect_left(line_ends, found_end)
        if line_starts[first] != found_at or line_ends[last] != found_end:
            stats.not_found += 1
            return match.group(0)

        search_from = found_end
        stats.repaired += 1
        opening_fence = match.group(0).split("\n", 1)[0]  # keeps a language tag, if any
        code = format_code(code_lines[first : last + 1], line_height)
        return f"{opening_fence}\n{code}\n```"

    return FENCED_BLOCK_PATTERN.sub(rebuild, markdown), stats


def format_code(lines: list[CodeLine], line_height: float | None) -> str:
    """Join code lines into text, restoring indentation and blank lines."""
    # The leftmost line of the block is indentation 0. Every char_width further
    # right is one more space (4 spaces per Python indentation level).
    left = min(line.x for line in lines)
    char_width = statistics.median(line.char_width for line in lines)

    out = []
    for previous, line in zip([None, *lines], lines):
        # A bigger-than-usual gap above this line means blank lines were there.
        # Gaps across a page break say nothing, so they are skipped.
        if previous and line_height and previous.page == line.page:
            blank_lines = round((line.y - previous.y) / line_height) - 1
            out.extend([""] * max(0, min(blank_lines, MAX_BLANK_LINES)))
        indent = round((line.x - left) / char_width)
        out.append(" " * indent + line.text)
    return "\n".join(out)


if __name__ == "__main__":
    total = RepairStats()
    for pdf_path in sorted(PDF_DIR.glob("*.pdf")):
        markdown, stats = pdf_to_markdown(pdf_path)
        out_path = MD_DIR / f"{pdf_path.stem}.md"
        out_path.write_text(markdown, encoding="utf-8")

        total.blocks += stats.blocks
        total.repaired += stats.repaired
        total.not_found += stats.not_found
        if stats.not_found:
            print(f"{pdf_path.name}: {stats.not_found} of {stats.blocks} code blocks not repaired")

    print(f"Code blocks: {total.blocks}, repaired: {total.repaired}, not repaired: {total.not_found}")
