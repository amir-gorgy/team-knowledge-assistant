import pymupdf
from pathlib import Path

DATA_DIR = Path(__file__).parent
PDF_DIR = DATA_DIR / "sample-docs-pdf"
MD_DIR = DATA_DIR / "sample-docs-md"

for file in PDF_DIR.glob("*.pdf"):
    pdf_path = file
    doc = pymupdf.open(pdf_path)
    md = doc.to_markdown(use_ocr=False, header=False, footer=False)
    out_path = MD_DIR / f"{pdf_path.stem}.md"
    out_path.write_text(md, encoding="utf-8")
    

# doc = pymupdf.open(PDF_DIR / "About FastAPI versions - FastAPI.pdf")
# md = doc.to_markdown(use_ocr=False, header=False, footer=False)

# out_path = MD_DIR / "About FastAPI versions - FastAPI.md"
# out_path.write_text(md, encoding="utf-8")