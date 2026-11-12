"""Split Markdown documents into chunks that are small enough to embed and cite.

Chunking happens in two passes:
  1. split_by_headings: one chunk per section (a heading plus the text under it),
     skipping sections that have no text of their own.
  2. split_large: sections longer than max_chars are cut at paragraph breaks,
     and every piece keeps the section's heading_path. Consecutive pieces overlap
     by about 10%, and code blocks are never split.

Use MarkdownChunker.chunk() to run both.
"""

from dataclasses import dataclass, field
import re

# A Markdown heading line: 1-6 '#' characters, whitespace, then the heading text.
# Group 1 captures the '#'s (its length is the heading level),
# group 2 captures the heading text.
HEADINGS_PATTERN = re.compile(r"^(#{1,6})\s+(.*)")

# Patterns used by clean_heading() to strip Markdown/HTML formatting from heading text.
LINK_PATTERN = re.compile(r"\[([^\]]*)\]\([^)]*\)")  # [text](url) -> keeps "text" as group 1
HTML_TAG_PATTERN = re.compile(r"<[^>]+>")  # e.g. <u>, </u>, <mark>
CLOSING_HASHES_PATTERN = re.compile(r"\s+#+\s*$")  # optional closing hashes: "## Title ##"

# Where one sentence ends and the next begins: whitespace after '.', '!' or '?'.
# Used to build the overlap from whole sentences.
SENTENCE_BREAK_PATTERN = re.compile(r"(?<=[.!?])\s+")


@dataclass
class Chunk:
    """One piece of a document, plus what's needed to cite it later."""

    text: str
    source: str  # where the text came from, e.g. the file path
    # Headings above this chunk, outermost first, e.g. ["SQL Databases", "Create the Tables"].
    # default_factory gives each Chunk its own list; a plain `= []` would be shared by all of them.
    heading_path: list[str] = field(default_factory=list)


def clean_heading(text: str) -> str:
    """Turn raw heading text into plain text for heading_path and citations.

    "Use the **`scopes`**"     -> "Use the scopes"
    "Data <u>conversion</u>"   -> "Data conversion"
    "See [the docs](http://x)" -> "See the docs"

    Underscores are kept on purpose: in these docs they are part of names like
    `response_model`, not emphasis.
    """
    text = LINK_PATTERN.sub(r"\1", text)
    text = HTML_TAG_PATTERN.sub("", text)
    text = CLOSING_HASHES_PATTERN.sub("", text)
    text = text.replace("*", "").replace("`", "")
    # Collapse runs of whitespace left behind by the removals into single spaces.
    return " ".join(text.split())


def is_fence(line: str) -> bool:
    """True for a line that opens or closes a ``` code block.

    Only fences at the start of the line are detected; indented fences and
    ~~~ fences are not. All fences in the current sample docs are of this kind.
    """
    return line.startswith("```")


def contains_code(block: str) -> bool:
    """True if the block has a ``` fence line anywhere in it."""
    return any(is_fence(line) for line in block.splitlines())


class MarkdownChunker:
    def __init__(
        self,
        max_chars: int = 2500,
        overlap: float = 0.1,
        skip_preamble: bool = False,
    ):
        """
        Args:
            max_chars: Longest chunk to produce. Longer sections are split by split_large().
                The one exception is a single code block longer than this: it is kept
                whole, because half a function is a poor search result.
            overlap: Fraction of max_chars that the start of each piece repeats from
                the end of the previous piece (0.1 = 10%, 0 = no overlap). Only used
                when a section is split; whole sentences are repeated where possible.
            skip_preamble: Drop the text before the first heading. Turn this on for
                documents converted from web pages, where that text is the site's
                navigation (e.g. "FastAPI Learn Deployment") rather than content.
                Leave it off for normal Markdown, where it can be a real introduction.
        """
        self.max_chars = max_chars
        self.overlap_chars = int(max_chars * overlap)
        self.skip_preamble = skip_preamble

    def chunk(self, text: str, source: str) -> list[Chunk]:
        """Split a Markdown document into chunks of at most max_chars characters."""
        chunks = []
        for section in self.split_by_headings(text, source):
            chunks.extend(self.split_large(section))
        return chunks

    def split_by_headings(self, text: str, source: str) -> list[Chunk]:
        """
        Splits the given markdown text into chunks based on headings.

        Each chunk is one section: a heading line plus everything up to the next
        heading. Sections with only a heading and no text are dropped; their
        heading still appears in the heading_path of the sections below them.
        Lines starting with '#' inside ``` code blocks are not treated as headings.

        Args:
            text (str): The markdown text to be split.
            source (str): The source of the markdown text (e.g., filename).

        Returns:
            list[Chunk]: A list of Chunk objects containing the split text and heading paths.
        """
        chunks = []
        path = []  # headings we are currently under; path[i] is the level-(i+1) heading
        buffer = []  # lines of the section being collected
        in_code = False  # True while between an opening and closing ``` fence
        # False until the first heading: the lines collected before it are the preamble.
        # After that, buffer[0] is always the section's heading line.
        seen_heading = False

        def flush():
            # Turn the collected lines into a Chunk, then start a new, empty buffer.
            body_lines = buffer[1:] if seen_heading else buffer
            has_body = "\n".join(body_lines).strip() != ""
            is_skipped_preamble = not seen_heading and self.skip_preamble
            # Skip empty or heading-only sections (fix: these were 72 junk chunks
            # like "# About HTTPS"), and the preamble when asked to.
            if has_body and not is_skipped_preamble:
                # Copy the path: it keeps changing as we read on, and each chunk
                # must keep the path it had when it was created.
                chunk_text = "\n".join(buffer).strip()
                chunks.append(Chunk(text=chunk_text, source=source, heading_path=path.copy()))
            buffer.clear()

        for line in text.splitlines():
            if is_fence(line):
                in_code = not in_code
            # Inside a code block, '# ...' is a comment, not a heading.
            m = None if in_code else HEADINGS_PATTERN.match(line)
            if m:
                # A new heading ends the previous section.
                flush()
                seen_heading = True
                level = len(m.group(1))
                # Keep the headings above this level and replace the rest with this one.
                # E.g. path ["A", "B", "C"] then a level-2 heading "D" -> ["A", "D"].
                path[:] = path[: level - 1] + [clean_heading(m.group(2))]
            # The heading line is kept in the chunk text, so the chunk still names its topic.
            buffer.append(line)
        # Save the last section, which has no heading after it to trigger a flush.
        flush()
        return chunks

    def split_large(self, chunk: Chunk) -> list[Chunk]:
        """Split a chunk longer than max_chars into pieces at paragraph breaks.

        Paragraphs (and whole code blocks) are packed into a piece until the next
        one would push it over max_chars, then a new piece starts. Each new piece
        begins with the last sentences of the previous one (the overlap), so text
        near a cut still appears with its context in at least one piece. Every
        piece keeps the original source and heading_path, so it can still be cited.
        """
        if len(chunk.text) <= self.max_chars:
            return [chunk]

        def joined_length(blocks: list[str]) -> int:
            # Length of the blocks once joined with blank lines ("\n\n").
            return len("\n\n".join(blocks))

        pieces = []  # finished pieces, each a list of blocks
        current = []  # blocks of the piece being filled
        for block in self._blocks(chunk.text):
            if contains_code(block):
                # Never cut code: an oversized code block becomes one piece of its own,
                # longer than max_chars.
                parts = [block]
            else:
                # Plain text with no blank lines to split at is cut at line breaks.
                parts = self._split_by_lines(block)

            for part in parts:
                if current and joined_length(current + [part]) > self.max_chars:
                    pieces.append(current)
                    # Start the next piece with the end of the one just finished,
                    # unless that would push the new piece over the limit.
                    tail = self._overlap_tail(current[-1])
                    fits = tail and joined_length([tail, part]) <= self.max_chars
                    current = [tail] if fits else []
                current.append(part)
        if current:
            pieces.append(current)

        # list(...) gives each piece its own copy of the heading path.
        return [
            Chunk(
                text="\n\n".join(piece),
                source=chunk.source,
                heading_path=list(chunk.heading_path),
            )
            for piece in pieces
        ]

    def _overlap_tail(self, block: str) -> str:
        """The end of a block to repeat at the start of the next piece.

        Takes as many whole sentences from the end of the block as fit in
        overlap_chars. If even the last sentence is too long, takes the last
        overlap_chars characters, starting at a word boundary. Code is never
        repeated, so a piece never starts in the middle of a code block.
        """
        if self.overlap_chars <= 0 or contains_code(block):
            return ""

        # Positions where a sentence starts: the beginning of the block, plus right
        # after every sentence break. Walk them from the end, keeping the earliest
        # start whose tail still fits. Slicing (rather than re-joining sentences)
        # keeps the tail an exact copy of the block's ending.
        sentence_starts = [0] + [m.end() for m in SENTENCE_BREAK_PATTERN.finditer(block)]
        tail = ""
        for start in reversed(sentence_starts):
            if len(block) - start > self.overlap_chars:
                break
            tail = block[start:]
        if tail:
            return tail

        # No whole sentence fits: take the last characters, dropping the first
        # (probably cut-off) word. Text with no spaces at all gets no overlap.
        end = block[-self.overlap_chars :]
        _, space, rest = end.partition(" ")
        return rest.strip() if space else ""

    def _blocks(self, text: str) -> list[str]:
        """Split text into paragraphs at blank lines, keeping each code block in one piece.

        Blank lines inside a ``` block are part of the code, so they don't end the block.
        """
        blocks = []
        current = []
        in_code = False
        for line in text.splitlines():
            if is_fence(line):
                in_code = not in_code
            if not line.strip() and not in_code:
                # A blank line outside code ends the current paragraph.
                if current:
                    blocks.append("\n".join(current))
                    current = []
            else:
                current.append(line)
        if current:
            blocks.append("\n".join(current))
        return blocks

    def _split_by_lines(self, block: str) -> list[str]:
        """Split one text block longer than max_chars at line breaks.

        This is the fallback for text with no blank lines to split at. It is never
        used for code: split_large keeps code blocks whole.
        A single line longer than max_chars is cut at max_chars characters.
        """
        if len(block) <= self.max_chars:
            return [block]

        parts = []
        current = ""
        for line in block.splitlines():
            # Cut over-long lines into max_chars-sized slices first.
            for start in range(0, max(len(line), 1), self.max_chars):
                segment = line[start : start + self.max_chars]
                # +1 for the "\n" that joins two lines.
                if current and len(current) + 1 + len(segment) > self.max_chars:
                    parts.append(current)
                    current = segment
                else:
                    current = f"{current}\n{segment}" if current else segment
        if current:
            parts.append(current)
        return parts
