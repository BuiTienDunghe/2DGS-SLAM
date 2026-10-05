"""plan v6 quick: put the tables of v6q_tables.md into the report template (placeholders @@E5@@, @@E1@@, @@E4@@).

usage: python tools/v6q_assemble.py REPORT_DIR
"""
import os
import sys

O = sys.argv[1]
tables = open(os.path.join(O, "v6q_tables.md"), encoding="utf-8").read()
parts = {}
for block in tables.split("\n### ")[1:]:
    title, body = block.split("\n", 1)
    parts[title.strip()] = body.strip()
txt = open(os.path.join(O, "v6q_report.tpl.md"), encoding="utf-8").read()
for k in ("E5", "E1", "E4"):
    assert f"@@{k}@@" in txt and parts.get(k), k
    txt = txt.replace(f"@@{k}@@", parts[k])
open(os.path.join(O, "v6q_report.md"), "w", encoding="utf-8").write(txt)
print("written", os.path.join(O, "v6q_report.md"), len(txt.splitlines()), "lines")
