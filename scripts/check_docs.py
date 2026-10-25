#!/usr/bin/env python3
"""Check local Markdown links, JSON syntax and overlay payload fingerprints."""

import ast
import json
from pathlib import Path
import re
from urllib.parse import unquote
from sparkfleet.overlays import inspect_overlay

root = Path(__file__).resolve().parents[1]
for p in root.rglob("*.md"):
    if any(x in p.parts for x in (".git", ".venv", "work", "build", "dist")):
        continue
    for target in re.findall(r"\]\(([^\s)]+)(?:\s+[^)]*)?\)", p.read_text()):
        if target.startswith(("http:", "https:", "#", "mailto:")):
            continue
        path = unquote(target.split("#", 1)[0])
        if path and not (p.parent / path).exists():
            raise ValueError(f"Broken local link: {p.relative_to(root)} -> {path}")
for folder in ["examples", "overlays", "benchmarks"]:
    for p in (root / folder).rglob("*.json"):
        json.loads(p.read_text())
for p in (root / "overlays").rglob("*.py"):
    ast.parse(p.read_text(), filename=str(p))
for p in (root / "overlays").iterdir():
    if p.is_dir():
        inspect_overlay(p)
print("Local docs links, JSON, overlay syntax and payload fingerprints: PASS")
