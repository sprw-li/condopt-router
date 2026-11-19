"""Regenerate the index of ``research/NEGATIVE_RESULTS.md`` (idempotent).

Earlier hand-patched runs prepended a fresh index each time, leaving three index
blocks behind.  This script parses the numbered sections, ignores everything
before the first heading, and rewrites preamble + one index + sections.

Run:
    py -3 scripts/build_negative_index.py
    py -3 scripts/build_negative_index.py --check
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOC = ROOT / "research" / "NEGATIVE_RESULTS.md"

SUMMARIES = {
    1: "带噪约束 2 目标上 cqnehvi 不如 cqehvi",
    2: "Sobol 播种的 CMA-ES 不如默认初始化",
    3: "mqehvi_nd 输给标量化（保留但标为实验性）",
    4: "TPE + 物性描述符反而更差（84.1 vs 90.2）",
    5: "垃圾描述符与 one-hot 相当（不必苛求描述符精度）",
    6: "真实 ART 六点上代理预测失败（样本量太小）",
    7: "4 维 MC-EHVI 性能坑（已修复）",
    8: "描述符在多保真多目标上降级为中性",
    9: "停滞后探索重启不如继续；调参优于继续",
    10: "多保真 + 约束：全保真槽位可能全部不可行（已修复交付证据）",
    11: "小样本下 transfer-audit 会误判（真实数据；已加低置信度保护）",
    12: "transfer-audit 在真实小样本上没有排序能力（保护机制 20/20 生效）",
}


def build() -> str:
    text = DOC.read_text(encoding="utf-8")
    first_heading = re.search(r"^## ", text, re.MULTILINE)
    assert first_heading is not None
    preamble = text[: first_heading.start()].rstrip()

    # sections are the ## N. blocks; anything else in between is dropped
    sections = list(re.finditer(r"^## (\d+)\. (.*)$", text, re.MULTILINE))
    assert sections, "no numbered entries found"
    pieces: dict[int, str] = {}
    for index, match in enumerate(sections):
        start = match.start()
        end = sections[index + 1].start() if index + 1 < len(sections) else len(text)
        pieces[int(match.group(1))] = text[start:end].strip()

    # the footer belongs to the file, not to the last section
    last = max(pieces)
    pieces[last] = re.sub(
        r"\n> 索引由 `scripts/build_negative_index\.py`.*$", "", pieces[last]
    ).strip()

    lines = [preamble, "", "## 索引", "", "| # | 主题 | 一句话结论 |", "|---|---|---|"]
    for number in sorted(pieces):
        title = re.match(r"^## \d+\. (.*)$", pieces[number], re.MULTILINE).group(1)
        lines.append(f"| {number} | {title} | {SUMMARIES.get(number, '')} |")
    lines.append("")
    for number in sorted(pieces):
        lines.append(pieces[number])
        lines.append("")
    lines.append("> 索引由 `scripts/build_negative_index.py` 生成（幂等）；`tests/test_negative_results.py` 校验。")
    return "\n".join(lines).rstrip() + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    content = build()
    current = DOC.read_text(encoding="utf-8")
    if args.check:
        if current.strip() != content.strip():
            print("negative-results index is stale; run scripts/build_negative_index.py")
            return 1
        print("negative-results index is up to date")
        return 0
    DOC.write_text(content, encoding="utf-8")
    print(f"wrote {DOC} ({len(content.splitlines())} lines)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())