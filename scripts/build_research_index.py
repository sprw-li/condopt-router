"""Regenerate ``research/README.md``: an index of every research note.

Groups notes into curated headline results, specs/designs, review archives and
working notes, so the evidence base stays navigable as the folder grows.

Run:
    py -3 scripts/build_research_index.py        # writes research/README.md
    py -3 scripts/build_research_index.py --check  # fail if the index is stale
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

def _force_utf8_stdio() -> None:
    """Emit UTF-8 regardless of the Windows console codepage (cp936 here)."""
    import sys

    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
        except (AttributeError, ValueError):  # pragma: no cover
            pass


ROOT = Path(__file__).resolve().parents[1]
RESEARCH = ROOT / "research"

HEADLINES = [
    ("BENCHMARK_SPEC.md", "基准分层与指标定义（Tier A–G）"),
    ("BENCHMARK_TIER_CD.md", "Tier C/D 高基数与混合空间基准"),
    ("MF_BENCHMARK.md", "多保真基准：mfbo 调参 93.2 → 95.3"),
    ("TRANSFER_BENCHMARK.md", "暖启动收益 +37%（good 源）、坏源随 top_k 恶化、prior_reliability"),
    ("REAL_DATA_LOO.md", "真实 ART 六点 LOO：Spearman −1.00 → −0.83，仍无预测力（样本量）"),
    ("REAL_PRIOR_TRANSFER.md", "真实先验 → 代理战役：12 次配对运行 warm 12/12 命中，cold 0/12"),
    ("DESCRIPTOR_FEATURES.md", "描述符特征：3/3 命中、噪声鲁棒、Tier C 上只配 GP 类"),
    ("NEGATIVE_RESULTS.md", "负结果清单：cqnehvi / Sobol 播种 / mqehvi_nd / TPE+描述符 / …"),
    ("ROUTER_CALIBRATION.md", "路由权重的留一校准（LOPO）"),
    ("AUTONOMOUS_LAB_POSITION.md", "面向 AI 全自动实验室的定位说明"),
    ("TRANSFER_ADAPTIVE.md", "分阶段迁移（已 API 化）：有偏源 +0.71 vs 固定 k3/k5 −1.54/−0.33；否决稳定"),
    ("TRANSFER_ADAPTIVE.md", "分阶段迁移：有偏源 +0.71 vs 固定 k3/k5 的 −1.54/−0.33"),
    ("EXTERNAL_BENCHMARK_SCAN.md", "外部同类项目扫描（78 个项目）：状态机/基准协议/审计护栏四块差距"),
    ("EXTERNAL_REVIEW_RESPONSE.md", "外部审查（2026-11-11）逐条回应：门禁环境依赖、qNEHVI/TPE/CMA-ES 命名边界、benchmark 协议待办"),
    ("NOISE_FLOOR.md", "先估噪声底再声称差异：描述符 HV 差异 t=−0.704 不显著"),
    ("AUDIT_CALIBRATION.md", "审计阈值校准：中等偏差源分不出，低置信度建议封顶 0.6"),
    ("TRANSFER_TOP_K.md", "修掉先验导致的平台期误判后，暖启动由 −31 变 +6~+9（top_k 扫描）"),
    ("TRANSFER_COST.md", "成本预算下的暖启动：修掉 3 个真实缺陷；budget 24 下收益仍不明显"),
    ("MID_BUDGET.md", "中段预算停止策略：40–60 预算优先更优但交付证据 2/5 → 1/5"),
    ("TINY_BUDGET.md", "极小预算：空间填充 vs 代理实测（约束场景 sobol 仅 1/4 有前沿）"),
    ("TRANSFER_MOO.md", "多目标暖启动：全保真 +0.5% HV/+20% 前沿；多保真中性或略负"),
    ("TRANSFER_AUDIT.md", "transfer-audit 分级：一致源 1.0、坏源 0.3–0.6、小样本降级"),
    ("ADAPTER_LONG_RUN.md", "执行器适配器长跑：120 次预算，0 重复 id，失败全部补测"),
    ("HARD_FEASIBILITY.md", "极低可行率：写进变量范围 5/5 vs 约束搜索 1/5"),
    ("EVIDENCE_AUDIT.md", "证据声明复核：25 条声明重跑（约 390 s）"),
    ("STAGNATION_TUNING.md", "停滞后调参优于继续（−205 vs −320），patience=3 最优"),
]

GROUPS = [
    ("规格与设计", re.compile(r"_SPEC|_PLAN$|SPEC_V\d")),
    ("实现与验收", re.compile(r"IMPL|ACCEPTANCE|APPLY|PATCH")),
    ("历史评审归档", re.compile(r"^REVIEW_v\d+")),
    ("调研与参考", re.compile(r"SURVEY|SCAN|REFERENCES|ARCHITECTURE_PATTERNS|CAPABILITIES")),
    ("优先级与交接", re.compile(r"NEXT_|OUTBOX|NEXT_ROUND")),
    (
        "历史归档（不再维护）",
        re.compile(r"^REVIEW_v\d+|APPLY_PLAN|FIRST_STEP|FAST_CI|SPRINT\d|_PLAN$"),
    ),
]


def _counts() -> dict:
    """Live counts so the index states how much evidence exists, not just where."""
    import importlib.util

    negative = RESEARCH / "NEGATIVE_RESULTS.md"
    entries = 0
    if negative.exists():
        entries = len(
            re.findall(
                r"^## \d+\.",
                negative.read_text(encoding="utf-8"),
                re.MULTILINE,
            )
        )

    def _load(name: str, path: Path):
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module

    claims = len(_load("verify_evidence", ROOT / "scripts" / "verify_evidence.py").CLAIMS)
    scenarios = len(
        _load("scenario_matrix", ROOT / "scripts" / "scenario_matrix.py").SCENARIOS
    )
    files = [p for p in RESEARCH.glob("*.md") if p.name != "README.md"]
    archive_pattern = re.compile(r"^REVIEW_v\d+|APPLY_PLAN|FIRST_STEP|FAST_CI|SPRINT\d|_PLAN$")
    return {
        "notes": len(files),
        "active": len([p for p in files if not archive_pattern.search(p.name)]),
        "archived": len([p for p in files if archive_pattern.search(p.name)]),
        "negative": entries,
        "claims": claims,
        "scenarios": scenarios,
    }


def _first_line(path: Path) -> str:
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            return stripped.lstrip("#").strip()
        if stripped:
            return stripped[:120]
    return "(empty)"


def build() -> str:
    counts = _counts()
    files = sorted(path for path in RESEARCH.glob("*.md") if path.name != "README.md")
    by_name = {path.name: path for path in files}
    lines = [
        "# 研究笔记索引",
        "",
        "本目录是优化器/router 的**证据库**：每条默认值都能追溯到这里的数字。",
        "索引由 `scripts/build_research_index.py` 生成，`tests/test_research_index.py` 防止漏登记。",
        "",
        "## 概览",
        "",
        "| 指标 | 数量 | 入口 |",
        "|---|---|---|",
        f"| 研究笔记 | {counts['notes']}（活跃 {counts['active']} / 历史归档 {counts['archived']}） | 本目录 |",
        f"| 负结果条目 | {counts['negative']} | [`NEGATIVE_RESULTS.md`](NEGATIVE_RESULTS.md) |",
        f"| 证据声明（可重跑） | {counts['claims']} | [`EVIDENCE_AUDIT.md`](EVIDENCE_AUDIT.md) |",
        f"| 场景断言 | {counts['scenarios']} | [`../docs/SCENARIOS.md`](../docs/SCENARIOS.md) |",
        "",
        "## 核心结论",
        "",
        "| 文档 | 结论 |",
        "|---|---|",
    ]
    for name, summary in HEADLINES:
        if name in by_name:
            lines.append(f"| [`{name}`]({name}) | {summary} |")
    listed = {name for name, _ in HEADLINES}
    lines += ["", "## 全部文档", ""]
    for title, pattern in GROUPS:
        group = [path for path in files if path.name not in listed and pattern.search(path.name)]
        if not group:
            continue
        listed.update(path.name for path in group)
        lines += [f"### {title}", "", "| 文档 | 标题 |", "|---|---|"]
        for path in group:
            lines.append(f"| [`{path.name}`]({path.name}) | {_first_line(path)} |")
        lines.append("")
    rest = [path for path in files if path.name not in listed]
    if rest:
        lines += ["### 其他工作笔记", "", "| 文档 | 标题 |", "|---|---|"]
        for path in rest:
            lines.append(f"| [`{path.name}`]({path.name}) | {_first_line(path)} |")
        lines.append("")
    lines += ["## Ralph 循环日志", "", "见 [`ralph/LOG.md`](ralph/LOG.md)。", ""]
    return "\n".join(lines)


def main() -> int:
    _force_utf8_stdio()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="fail if the index is stale")
    args = parser.parse_args()
    content = build()
    target = RESEARCH / "README.md"
    if args.check:
        current = target.read_text(encoding="utf-8") if target.exists() else ""
        if current.strip() != content.strip():
            print("research index is stale; run scripts/build_research_index.py")
            return 1
        print("research index is up to date")
        return 0
    target.write_text(content, encoding="utf-8")
    print(f"wrote {target} ({len(content.splitlines())} lines)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
