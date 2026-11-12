from pathlib import Path
import re

total_flags = 0
zero_headings = 0

for file in Path("data/sample-docs-md").glob("*.md"):
    text = file.read_text(encoding="utf-8")
    # counts number of headings
    headings = len(re.findall(r"^#{1,6} ", text, re.MULTILINE))
    if headings == 0:
        zero_headings += 1
        print(f"{file.name} has no headings")
    # counts number of fences
    fences = len(re.findall(r"^```", text, re.MULTILINE))
    
    flag = "" if fences % 2 == 0 else " <-- Unbalanced fences"
    if flag:
        total_flags += 1
    print(f"{file.name} headings={headings} fences={fences//2} {flag}")

print(f"Total files with issues: {total_flags}. Total files with zero headings: {zero_headings}.")