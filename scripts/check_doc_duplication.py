"""Detect duplicated rule text between the agent skill and the docs.

The skill should point at ``docs/`` rather than restating long benchmark
numbers, otherwise the two copies drift apart.  This checker extracts
rule-like sentences (thresholds, measured numbers, defaults) from ``skill/``
and ``docs/`` and reports near-duplicates by token Jaccard similarity.

Run:
    py -3 scripts/check_doc_duplication.py
    py -3 scripts/check_doc_duplication.py --json --threshold 0.6
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RULE_HINT = re.compile(r"(默认|实测|建议|阈值|>|<|≥|≤|\d+%|\d+\.\d+)")
SENTENCE = re.compile(r"[^。；\n]{20,200}[。；]?")


def _sentences(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8")
    text = re.sub(r"```.*?```", "", text, flags=re.DOTALL)
    out = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith(("#", "|", "-", "*", ">", "```")):
            continue
        for match in SENTENCE.finditer(stripped):
            sentence = match.group(0).strip()
            if RULE_HINT.search(sentence):
                out.append(sentence)
    return out


def _tokens(sentence: str) -> set[str]:
    cleaned = re.sub(r"[^\w\u4e00-\u9fff]+", " ", sentence)
    return {token for token in cleaned.split() if len(token) > 1}


def _similarity(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def find_duplicates(threshold: float = 0.6) -> list[dict]:
    skill_files = sorted((ROOT / "skill").rglob("*.md"))
    doc_files = sorted((ROOT / "docs").rglob("*.md"))
    matches: list[dict] = []
    for skill_file in skill_files:
        skill_sentences = _sentences(skill_file)
        skill_tokens = [_tokens(sentence) for sentence in skill_sentences]
        for doc_file in doc_files:
            for doc_sentence in _sentences(doc_file):
                doc_tokens = _tokens(doc_sentence)
                for index, sentence in enumerate(skill_sentences):
                    score = _similarity(skill_tokens[index], doc_tokens)
                    if score >= threshold:
                        matches.append({
                            "similarity": round(score, 3),
                            "skill": str(skill_file.relative_to(ROOT)),
                            "doc": str(doc_file.relative_to(ROOT)),
                            "skill_text": sentence[:120],
                            "doc_text": doc_sentence[:120],
                        })
    matches.sort(key=lambda item: -item["similarity"])
    return matches


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threshold", type=float, default=0.6)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--max-duplicates", type=int, default=0, help="fail above this count")
    args = parser.parse_args()
    matches = find_duplicates(args.threshold)
    if args.json:
        print(json.dumps({"threshold": args.threshold, "count": len(matches), "matches": matches},
                         ensure_ascii=False, indent=2))
    else:
        for item in matches:
            print(f"[{item['similarity']}] {item['skill']} <-> {item['doc']}")
            print(f"    skill: {item['skill_text']}")
            print(f"    doc:   {item['doc_text']}")
        print(f"doc_duplication: {len(matches)} near-duplicate rule sentences (threshold {args.threshold})")
    return 1 if len(matches) > args.max_duplicates else 0


if __name__ == "__main__":
    raise SystemExit(main())