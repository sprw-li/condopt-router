"""Encoding hygiene checks.

A PowerShell UTF-8 round-trip previously corrupted non-ASCII source files
and left a BOM in one module.  These checks keep that from silently
recurring: every Python file must parse, and no source file may start with
a UTF-8 BOM.
"""

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIRS = ["src", "tests", "scripts", "skill"]


def iter_files():
    for directory in SOURCE_DIRS:
        base = ROOT / directory
        if not base.exists():
            continue
        for path in base.rglob("*"):
            if path.suffix in (".py", ".md") and "__pycache__" not in path.parts:
                yield path


def test_no_utf8_bom():
    offenders = [
        str(path.relative_to(ROOT))
        for path in iter_files()
        if path.read_bytes().startswith(b"\xef\xbb\xbf")
    ]
    assert not offenders, f"files with UTF-8 BOM: {offenders}"


def test_python_files_parse_as_utf8():
    failures = []
    for path in iter_files():
        if path.suffix != ".py":
            continue
        try:
            text = path.read_text(encoding="utf-8")
            ast.parse(text)
        except (SyntaxError, UnicodeDecodeError) as exc:
            failures.append(f"{path.relative_to(ROOT)}: {exc}")
    assert not failures, failures
