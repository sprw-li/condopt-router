"""Check that relative markdown links in the docs actually resolve.

Scans README, ``docs/``, ``research/``, ``skill/`` and ``examples/`` for
``[label](relative/path.md)`` links and reports targets that do not exist.
External URLs, anchors and mail links are ignored.

Run:
    py -3 scripts/check_docs_links.py
    py -3 scripts/check_docs_links.py --json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LINK = re.compile(r"\[[^\]]*\]\(([^)]+)\)")
SCAN_DIRS = ("docs", "research", "skill", "examples")
SKIP_PREFIXES = ("http://", "https://", "mailto:", "#", "data:")


def markdown_files() -> list[Path]:
    files = [ROOT / "README.md"]
    for folder in SCAN_DIRS:
        files.extend(sorted((ROOT / folder).rglob("*.md")))
    return [path for path in files if path.exists()]


FENCE = re.compile(r"```.*?```", re.DOTALL)
INLINE_CODE = re.compile(r"`[^`\n]*`")


def _strip_code(text: str) -> str:
    """Code fences and inline spans may contain ``[...](...)`` lookalikes."""
    return INLINE_CODE.sub("", FENCE.sub("", text))


def check() -> list[dict]:
    broken = []
    for path in markdown_files():
        text = _strip_code(path.read_text(encoding="utf-8"))
        for match in LINK.finditer(text):
            target = match.group(1).strip()
            if not target or target.startswith(SKIP_PREFIXES):
                continue
            clean = target.split("#", 1)[0].strip()
            if not clean:
                continue
            if "://" in clean:
                continue
            if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", clean):
                # Bare identifiers such as ``[best_name](ctx)`` in prose about
                # code are not file links.
                continue
            resolved = (path.parent / clean).resolve()
            if not resolved.exists():
                broken.append({
                    "file": str(path.relative_to(ROOT)),
                    "link": target,
                })
    return broken


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    broken = check()
    files = markdown_files()
    if args.json:
        print(json.dumps({"files": len(files), "broken": broken}, ensure_ascii=False, indent=2))
    else:
        for item in broken:
            print(f"BROKEN {item['file']} -> {item['link']}")
        print(f"docs_links: {len(files)} files, {len(broken)} broken")
    return 1 if broken else 0


if __name__ == "__main__":
    raise SystemExit(main())
