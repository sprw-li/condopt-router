"""End-to-end negative control for the golden-number gate (backlog A3).

Requirement: "故意改坏一个数字，检查必须变红".  This script backs up a document,
corrupts one registered number, runs the gate as a subprocess, restores the file and
reports both exit codes -- so the proof is reproducible and leaves no damage.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

def _utf8_child_env() -> dict:
    """Environment for child processes so their stdio is UTF-8, not the codepage."""
    import os

    return {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"}


ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "docs" / "GOLDEN_NUMBERS.json"


def main() -> int:
    entries = json.loads(REGISTRY.read_text(encoding="utf-8", errors="replace"))["entries"]
    # pick an entry whose document we can safely edit and restore
    entry = next(
        (item for item in entries if item["doc"].startswith("research/")),
        entries[0],
    )
    target = ROOT / entry["doc"]
    original = target.read_text(encoding="utf-8", errors="replace")
    lines = original.splitlines(keepends=True)
    corrupted_lines = []
    hit = False
    for line in lines:
        if not hit and entry["line_hint"] in line and entry["number_text"] in line:
            corrupted_lines.append(line.replace(entry["number_text"], "999999", 1))
            hit = True
        else:
            corrupted_lines.append(line)
    if not hit:
        print("could not corrupt:", entry["id"])
        return 2

    gate = [sys.executable, "scripts/check_golden_numbers.py"]
    clean = subprocess.run(gate, cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace")
    try:
        target.write_text("".join(corrupted_lines), encoding="utf-8", errors="replace")
        dirty = subprocess.run(gate, cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace")
    finally:
        target.write_text(original, encoding="utf-8", errors="replace")
        restored = subprocess.run(
            gate, cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace"
        )

    print(json.dumps({
        "entry": entry["id"],
        "doc": entry["doc"],
        "clean_exit": clean.returncode,
        "corrupted_exit": dirty.returncode,
        "restored_exit": restored.returncode,
        "corrupted_output": (dirty.stdout + dirty.stderr).strip().splitlines()[:3],
        "ok": clean.returncode == 0 and dirty.returncode != 0 and restored.returncode == 0,
    }, ensure_ascii=False, indent=2))
    return 0 if (clean.returncode == 0 and dirty.returncode != 0 and restored.returncode == 0) else 1


if __name__ == "__main__":
    raise SystemExit(main())
