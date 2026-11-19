"""Lead-side acceptance runner for the parallel backlog fan-out.

Runs each item's own acceptance command (the executable criterion recorded in
``docs/IMPROVEMENT_BACKLOG.md``) and prints a table with exit codes, so "done" is a
result rather than a claim.  Missing test files are reported as ``MISSING`` instead
of being silently skipped.

Run:
    py -3 scripts/verify_backlog_items.py            # item checks only
    py -3 scripts/verify_backlog_items.py --full     # + full pytest
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

ITEMS: list[tuple[str, list[str], list[str]]] = [
    ("A4 等效性检验", ["-m", "pytest", "-q", "tests/test_noise_floor_equivalence.py"],
     ["tests/test_noise_floor_equivalence.py"]),
    ("A11 基准协议", ["-m", "pytest", "-q", "tests/test_benchmark_protocol.py"],
     ["tests/test_benchmark_protocol.py"]),
    ("A7+A8 CLI/Schema", ["-m", "pytest", "-q", "tests/test_cli_errors.py",
                          "tests/test_answers_schema.py"],
     ["tests/test_cli_errors.py", "tests/test_answers_schema.py"]),
    ("A9 审计字段", ["-m", "pytest", "-q", "tests/test_audit_fields.py"],
     ["tests/test_audit_fields.py"]),
    ("A12 可行性语义", ["-m", "pytest", "-q", "tests/test_feasibility_semantics.py"],
     ["tests/test_feasibility_semantics.py"]),
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", action="store_true", help="also run the whole suite")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    rows: list[dict] = []
    for name, command, required_files in ITEMS:
        missing = [path for path in required_files if not (ROOT / path).exists()]
        if missing:
            rows.append({"item": name, "status": "MISSING", "exit": None,
                         "detail": ", ".join(missing)})
            continue
        result = subprocess.run(
            [sys.executable, *command], cwd=ROOT, capture_output=True,
            text=True, encoding="utf-8",
        )
        tail = (result.stdout + result.stderr).strip().splitlines()[-1:] or [""]
        rows.append({
            "item": name,
            "status": "PASS" if result.returncode == 0 else "FAIL",
            "exit": result.returncode,
            "detail": tail[0][:110],
        })

    full = None
    if args.full:
        result = subprocess.run(
            [sys.executable, "-m", "pytest", "-q"], cwd=ROOT,
            capture_output=True, text=True, encoding="utf-8",
        )
        full = {
            "exit": result.returncode,
            "detail": ((result.stdout + result.stderr).strip().splitlines() or [""])[-1][:120],
        }

    if args.json:
        import json

        print(json.dumps({"items": rows, "full_suite": full}, ensure_ascii=False, indent=2))
    else:
        for row in rows:
            print(f"[{row['status']:7}] {row['item']}: {row['detail']}")
        if full is not None:
            print(f"[{'PASS' if full['exit'] == 0 else 'FAIL':7}] full suite: {full['detail']}")

    failed = [row for row in rows if row["status"] != "PASS"]
    if full is not None and full["exit"] != 0:
        failed.append({"item": "full suite"})
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
