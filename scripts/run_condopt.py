"""Run the condopt CLI from a source checkout without installing the package.

Usage:
    py -3 scripts/run_condopt.py route --space configs/space.example.json --budget 30
"""

import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from condopt.cli import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
