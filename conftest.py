import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent  # this file sits at the repository root
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
