"""Backlog metric for the optimize loop: number of *open* backlog items.

Lower is better.  The gate for the loop is ``scripts/release_check.py`` (it must stay
green), so this number may only drop while correctness holds.

An item counts as open when its row in ``docs/IMPROVEMENT_BACKLOG.md`` is neither
struck through (``~~X~~``) nor marked ``已修``.  Rows are the tables under the
``P0``/``P1`` headings.

Run:
    py -3 scripts/backlog_metric.py      # prints a single integer
    py -3 scripts/backlog_metric.py -v   # prints the open items too
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BACKLOG = ROOT / "docs" / "IMPROVEMENT_BACKLOG.md"

#: rows that look like a backlog entry: ``| A7 | … |`` or ``| # | … |``
ITEM_ROW = re.compile(r"^\|\s*(A\d+|\d+)\s*\|")


def open_items() -> list[str]:
    if not BACKLOG.exists():
        return []
    items: list[str] = []
    for line in BACKLOG.read_text(encoding="utf-8").splitlines():
        match = ITEM_ROW.match(line)
        if not match:
            continue
        identifier = match.group(1)
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        title = cells[1] if len(cells) > 1 else ""
        # a struck-through title or an explicit "已修" mark means it is closed
        closed = "~~" in title or "已修" in title
        # the "已修（有回归测试）" table also uses rows; those are closed by definition
        if not closed:
            items.append(f"{identifier}: {title}")
    return items


def metric() -> int:
    return len(open_items())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()
    items = open_items()
    if args.verbose:
        for item in items:
            print(item, file=sys.stderr)
    print(metric())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
