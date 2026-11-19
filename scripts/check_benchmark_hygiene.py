"""Benchmark-hygiene check: loop caps must scale with the campaign size.

This project lost two rounds to a fixed ``for _ in range(60)`` loop that silently
truncated budget-120 campaigns (see ``research/EVIDENCE_POLICY.md`` §7).  A batch is
two actions (ask + tell), so a benchmark loop needs at least ``2 * budget / batch``
steps -- safest is an explicit budget-scaled bound.

Rules enforced here:

1. every *hard-coded* step cap inside a benchmark loop must be large enough for the
   largest budget the script offers by default (or be expressed via the budget);
2. benchmark scripts that check delivery evidence must record the progress
   trajectory themselves (``study.record_progress()``) or go through the adapter.

Run:
    py -3 scripts/check_benchmark_hygiene.py
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"

#: benchmark scripts that drive their own ask/tell loop (not through run_campaign)
#: script -> the largest budget its evidence uses (loop caps must cover 2+ actions
#: per evaluation, so the practical requirement is >= 4 * budget)
MANUAL_LOOPS = {
    "benchmark_confirm_budget.py": 20,
    "benchmark_hard_feasibility.py": 60,
    "benchmark_tiny_budget.py": 8,
    "benchmark_transfer_moo.py": 60,
    "benchmark_transfer_reliability_band.py": 30,
    "benchmark_mid_budget.py": 120,
    "benchmark_delivery_evidence.py": 24,
    "benchmark_stagnation.py": 80,
}


def check() -> dict:
    problems: list[str] = []
    checked = 0
    for name, max_budget in MANUAL_LOOPS.items():
        path = SCRIPTS / name
        if not path.exists():
            problems.append(f"{name}: listed as a manual loop but missing")
            continue
        checked += 1
        text = path.read_text(encoding="utf-8")
        caps = [int(value) for value in re.findall(r"for _ in range\(\s*(\d+)\s*\)", text)]
        needs = max(60, 4 * max_budget)
        for cap in caps:
            if cap < needs:
                problems.append(
                    f"{name}: hard-coded loop cap {cap} < {needs} "
                    f"(two actions per batch; largest budget {max_budget})"
                )
        # a budget-scaled cap is the recommended form
        if caps and not re.search(r"for _ in range\([^)]*budget", text):
            problems.append(
                f"{name}: loop cap does not reference `budget` (must scale with size)"
            )
        if "record_progress" not in text and "run_campaign" not in text:
            # only a problem when the script reports delivery evidence
            if "delivery_checklist" in text or "delivery_ok" in text:
                problems.append(
                    f"{name}: reports delivery evidence without recording progress"
                )
    return {"checked": checked, "problems": problems, "ok": not problems}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    report = check()
    if args.json:
        import json as _json

        print(_json.dumps(report, ensure_ascii=False, indent=2))
    else:
        for problem in report["problems"]:
            print("PROBLEM", problem)
        print(
            f"benchmark_hygiene: scripts={report['checked']} "
            f"problems={len(report['problems'])}"
        )
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())