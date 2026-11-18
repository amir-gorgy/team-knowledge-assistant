"""Build sample-docs-md/ from FastAPI's own Markdown docs on GitHub.

FastAPI writes its docs in Markdown (docs/en/docs/**/*.md in github.com/fastapi/fastapi),
so no PDF conversion is needed. Three things still have to change for the ingest pipeline:

  1. Code examples live in separate files and are pulled in by include tags:
         {* ../../docs_src/query_params/tutorial001.py hl[9] *}     a fenced code block
         {* ../../docs_src/sql_databases/tutorial001.py ln[1:11] *} only lines 1-11
         {!../../docs_src/templates/templates/item.html!}           raw text, already in a fence
     Each tag is replaced by the file's text, so chunks contain the actual code.
     hl[...] (highlighted lines) and title[...] only change how the site shows the code.
  2. Headings end with an anchor, "## Defaults { #defaults }", which would end up in
     every heading path. It is removed.
  3. Pages sit in nested folders (tutorial/security/...), but ingest only reads the top
     level of a folder. Each page is written flat as "<page title> - FastAPI.md".

Left out: reference/ (one-line stubs that only point at generated API docs), release-notes.md,
and pages with no "# " title (fragments such as translation-banner.md) or a leading "_".

Run from the repo root (needs git; downloads ~10 MB):
    uv run python data/fastapi-docs-to-md.py              # FastAPI 0.165.0 into data/sample-docs-md
    uv run python data/fastapi-docs-to-md.py --ref 0.166.0
"""

import argparse
import re
import subprocess
import sys
import tempfile
from pathlib import Path

DATA_DIR = Path(__file__).parent
MD_DIR = DATA_DIR / "sample-docs-md"
REPO_URL = "https://github.com/fastapi/fastapi.git"
DEFAULT_REF = "0.165.0"  # a release tag, so the same command gives the same docs on every machine

SKIP_DIRS = {"reference"}
SKIP_FILES = {"release-notes.md"}

# {* path options *} -> a fenced code block. Options: hl[..], ln[..], title[..].
FENCE_INCLUDE = re.compile(r"\{\*\s*(\.\./\S+)(.*?)\*\}")  # path must start ../, so Python's {**d} is left alone
# {! path !} or {!> path !} -> the file's raw text (the page already wraps it in a fence).
RAW_INCLUDE = re.compile(r"\{!>?\s*(\.\./\S+?)\s*!\}")
LINES_OPTION = re.compile(r"ln\[([^\]]*)\]")
# [ \t], not \s: \s also matches the newline and would join the heading to the next line.
HEADING_ANCHOR = re.compile(r"^(#{1,6} .*?)[ \t]*\{[ \t]*#[^}]*\}[ \t]*$", re.M)
TITLE = re.compile(r"^# (.+)$", re.M)
FENCE_LANG = {".py": "python", ".html": "html", ".js": "javascript", ".json": "json", ".toml": "toml"}


def pick_lines(text: str, spec: str) -> str:
    """Keep only the lines ln[...] names: "1:11", "29" or "1:2,12:16,29" (1-based, inclusive).

    A gap between ranges becomes a "# ..." line, like the docs site shows.
    """
    lines = text.splitlines()
    out: list[str] = []
    last = 0
    for part in spec.split(","):
        start, _, end = part.strip().partition(":")
        first, final = int(start), int(end or start)
        if out and first > last + 1:
            out.append("# ...")
        out.extend(lines[first - 1 : final])
        last = final
    return "\n".join(out)


def convert(page: str, base: Path) -> str:
    """Resolve include tags against `base` (docs/en, where the ../../ paths start) and strip anchors."""
    missing: list[str] = []

    def read(path: str) -> str:
        file = (base / path).resolve()
        if not file.is_file():
            missing.append(path)
            return f"(missing: {path})"
        return file.read_text(encoding="utf-8").rstrip("\n")

    def fence(match: re.Match) -> str:
        path, options = match.group(1), match.group(2)
        code = read(path)
        if ln := LINES_OPTION.search(options):
            code = pick_lines(code, ln.group(1))
        lang = FENCE_LANG.get(Path(path).suffix, "")
        return f"```{lang}\n{code}\n```"

    page = FENCE_INCLUDE.sub(fence, page)
    page = RAW_INCLUDE.sub(lambda m: read(m.group(1)), page)
    page = HEADING_ANCHOR.sub(r"\1", page)
    for path in missing:
        print(f"  warning: include not found: {path}", file=sys.stderr)
    return page


def fetch(ref: str, into: Path) -> Path:
    """Sparse, shallow clone at one tag or branch: the docs, their examples, and the one
    library folder a page quotes (how-to/configure-swagger-ui.md shows fastapi/openapi/docs.py)."""
    repo = into / "fastapi"
    git = ["git", "-c", "advice.detachedHead=false"]
    subprocess.run([*git, "clone", "-q", "--depth", "1", "--branch", ref, "--filter=blob:none",
                    "--sparse", REPO_URL, str(repo)], check=True)
    subprocess.run([*git, "-C", str(repo), "sparse-checkout", "set", "docs/en/docs", "docs_src", "fastapi/openapi"], check=True)
    return repo


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ref", default=DEFAULT_REF, help="FastAPI git tag or branch")
    parser.add_argument("--src", type=Path, help="an existing fastapi checkout (skips the download)")
    parser.add_argument("--out", type=Path, default=MD_DIR)
    args = parser.parse_args()

    if args.out.exists() and any(args.out.glob("*.md")):
        sys.exit(f"{args.out} already has .md files; move or delete them first")
    args.out.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        repo = args.src or fetch(args.ref, Path(tmp))
        base = repo / "docs" / "en"
        pages = sorted((base / "docs").rglob("*.md"))
        written: dict[str, Path] = {}
        for page in pages:
            rel = page.relative_to(base / "docs")
            if rel.parts[0] in SKIP_DIRS or rel.name in SKIP_FILES or rel.name.startswith("_"):
                continue
            text = convert(page.read_text(encoding="utf-8"), base)
            title = TITLE.search(text)
            if not title:
                continue
            # "/" can't go in a file name; FastAPI titles don't use it today, but be safe.
            name = f"{title.group(1).strip().replace('/', '-')} - FastAPI.md"
            if name in written:
                sys.exit(f"two pages share the title in {name}: {written[name]} and {rel}")
            written[name] = rel
            (args.out / name).write_text(text.rstrip("\n") + "\n", encoding="utf-8")

    print(f"{len(written)} pages from FastAPI {args.ref if not args.src else args.src} -> {args.out}")


if __name__ == "__main__":
    main()
