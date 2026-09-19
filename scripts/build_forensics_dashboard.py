"""Inline the forensics JSON into the dashboard template. Read-only on both inputs.

Usage: .venv/bin/python scripts/build_forensics_dashboard.py <out.html>
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
tpl = (ROOT / "docs" / "forensics_dashboard.template.html").read_text()
data = (ROOT / "data" / "processed" / "forensics_dashboard.json").read_text()
# guard: the payload sits in a <script type=application/json>, so it must not
# contain a closing script tag or a comment opener that would end the element
assert "</script" not in data and "<!--" not in data, "payload would break out of its script tag"
body = tpl.replace("__DATA__", data)

# A complete, self-contained document: no CDN, no font fetch, no analytics, no
# network of any kind — open it with file:// and it works offline forever.
# The template's <title> is hoisted into <head>; everything else is body content.
title = "Wallet forensics — the shape of the edge"
if body.lstrip().startswith("<title>"):
    end = body.index("</title>") + len("</title>")
    title = body[body.index("<title>") + 7:body.index("</title>")]
    body = body[end:]

DOC = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="referrer" content="no-referrer">
<title>{title}</title>
<style>
  html {{ background: #f9f9f7; }}
  @media (prefers-color-scheme: dark) {{ html {{ background: #0d0d0d; }} }}
  body {{ margin: 0; }}
  *, *::before, *::after {{ box-sizing: border-box; }}
</style>
</head>
<body>
{body}
</body>
</html>
"""

out = Path(sys.argv[1])
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(DOC.format(title=title, body=body))
print(f"wrote {out} — {out.stat().st_size/1e6:.2f} MB")
