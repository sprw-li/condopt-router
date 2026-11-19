"""Generate ``RELEASE_NOTES.md``: the one-page summary of what this project delivers.

Counts come from the code and evidence base (registry, gates, scenarios, decision
matrix, claims, negative results), so the note cannot silently go stale.

Run:
    py -3 scripts/build_release_notes.py
    py -3 scripts/build_release_notes.py --check
"""

from __future__ import annotations

import argparse
import importlib.util
import re
import sys
from pathlib import Path

def _utf8_child_env() -> dict:
    """Environment for child processes so their stdio is UTF-8, not the codepage."""
    import os

    return {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"}


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
DOC = ROOT / "RELEASE_NOTES.md"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def counts() -> dict:
    from condopt import registry

    gates = _load("release_check", ROOT / "scripts" / "release_check.py")
    scenarios = _load("scenario_matrix", ROOT / "scripts" / "scenario_matrix.py")
    decisions = _load("decision_matrix", ROOT / "scripts" / "decision_matrix.py")
    evidence = _load("verify_evidence", ROOT / "scripts" / "verify_evidence.py")
    negative = (ROOT / "research" / "NEGATIVE_RESULTS.md").read_text(encoding="utf-8", errors="replace")
    # collected cases (parametrised tests expand), not just ``def test_`` functions
    collected = 0
    try:
        import subprocess

        result = subprocess.run(
            [sys.executable, "-m", "pytest", "--collect-only", "-q"],
            cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace", env=_utf8_child_env(),
        )
        collected = sum(
            int(match.group(1))
            for line in result.stdout.splitlines()
            if (match := re.match(r".+\.py:\s*(\d+)$", line.strip()))
        )
    except Exception:  # pragma: no cover - fall back to a static count
        collected = 0
    if not collected:
        test_files = list((ROOT / "tests").glob("test_*.py"))
        for path in test_files:
            source = path.read_text(encoding="utf-8", errors="replace")
            collected += len(re.findall(r"^def test_", source, re.MULTILINE))
    return {
        "algorithms": len(registry.names()),
        "gates": len(
            set(re.findall(r'record\(\s*\n?\s*"([a-z0-9_]+)"', (ROOT / "scripts" / "release_check.py").read_text(encoding="utf-8", errors="replace")))
        ),
        "scenarios": len(scenarios.SCENARIOS),
        "decision_rules": len(decisions.RULES),
        "suppressions": len(decisions.evaluate_suppressions()),
        "claims": len(evidence.CLAIMS),
        "negatives": len(re.findall(r"^## \d+\.", negative, re.MULTILINE)),
        "tests": collected,
    }


CAPABILITIES = [
    ("算法库", "18 个 ask/tell 优化器（GP-BO/TPE/ParEGO/qEHVI/qNEHVI/cqEHVI/多保真家族/CMA-ES/空间填充），统一 registry 与参数 schema"),
    ("路由器", "按问题特征（维度/目标数/约束/多保真/噪声/预算/地貌）选算法与参数，给出理由、备选与告警"),
    ("决策层", "开工五段建议（算法/停止/可行率/先验/交付口径）、交付槽位、确认复孔、停滞调参"),
    ("执行器适配器", "一个 measure() 方法接入真实仪器：失败补测、重试上限、并行批、超时、失败归因、进度曲线"),
    ("多目标/多保真/约束", "Pareto 前沿、可交付（可行）HV、保真度通道、成本硬上限"),
    ("迁移", "transfer_from（前沿选择 + 预算记账 + 元数据清洗）、transfer-audit（含置信度分级）"),
    ("交付自查", "report/digest 里逐条打勾：可行的最终保真观测、前沿+进展、推荐条件已复孔确认"),
    ("skill", "condopt-router：问卷 → 生成配置 → 选算法 → 迭代 → 收尾，文档流程有可执行断言"),
    (
        "证据纪律",
        lambda live: (
            "DEFAULTS 每条带出处、登记表（数据集/方向/tradeoff）、"
            f"{live['claims']} 条可重跑声明、{live['negatives']} 条负结果、测量卫生检查"
        ),
    ),
]


STEPS = [
    ("1. 问卷 → 配置", "py -3 scripts/run_condopt.py prepare --answers configs/answers.example.json --out work"),
    ("2. 建战役", "py -3 scripts/run_condopt.py init --run work/run --space work/space.json --objective work/objective.json --context work/context.json --budget 24 --batch 4"),
    ("3. 取条件", "py -3 scripts/run_condopt.py ask --run work/run --n 4"),
    ("4. 回填实测", "py -3 scripts/run_condopt.py tell --run work/run --results results.json"),
    ("5. 看决策/收尾", "py -3 scripts/run_condopt.py step --run work/run --json  # 或 digest / report"),
    ("发布自检", "py -3 scripts/release_check.py  # 全部门禁"),
]


KEY_FINDINGS = [
    ("真实数据暖启动", "ART Table 2 六点做源、代理 oracle 做目标：warm 12/12 命中真实最优，cold 0/12"),
    ("多保真 MOO 默认", "mfparego 在 3 个数据集上不落后 momfbo（674.5/641.0/27318 vs 669.5/636.3/26998）"),
    ("高维规则", "cmaes 在 8–14 维是“高维 GP 不可靠”的兜底；窄谷型 6 维起就该切，多峰型未必"),
    ("极小预算", "低于建模下限时屏蔽模型类；约束/MOO 保留代理 + 告警（纯 sobol 在约束下仅 1/4 有可行前沿）"),
    ("预算优先", "40–120 预算下更优（92.9/98.8/100.6/102.2 vs 71.8/65.6/76.9/65.6），交付自查可通过"),
    ("交付证据", "四种问题形态 12/12 通过自查；为交付最多保留 1 次评估"),
    ("迁移可信度", "机制已修（mfbo 曾忽略 objective_std）；但结果对 reliability 不敏感，分级只作策略性约定"),
]


LIMITATIONS = [
    "真实测量只有 ART Table 2 六点，其余为合成基准或代理 oracle，结论外推需谨慎",
    "多保真暖启动收益弱（+0.35%~+1%），不保证正收益",
    "约束场景保证“有可行的最终保真证据”，但可行区内的最优性仍受预算/代理精度限制",
    "skill 的安装依赖共享目录 <agent-library>，可能被其他会话并发改动（安装器已自带逻辑）",
]


def build() -> str:
    live = counts()
    lines = [
        "# 条件优化 router（condopt）交付说明",
        "",
        "**一句话**：把实验条件优化从“随手挑一个算法”变成**有理由的算法选择 + 可审计的闭环执行 + 可交付的证据**。",
        "面向 AI 全自动实验室的前置技术：agent 用 `condopt-router` skill 决策，`condopt` 负责算法、状态与证据。",
        "",
        "## 规模（自动统计）",
        "",
        "| 项 | 数量 |",
        "|---|---|",
        f"| 优化算法 | **{live['algorithms']}** |",
        f"| 发布门禁 | **{live['gates']}** |",
        f"| 单测 | **{live['tests']}** |",
        f"| 路由场景断言 | **{live['scenarios']}** |",
        f"| 决策检查（规则 + 抑制断言） | **{live['decision_rules'] + live['suppressions']}**"
        f"（{live['decision_rules']} 规则 + {live['suppressions']} 抑制） |",
        f"| 可重跑证据声明 | **{live['claims']}** |",
        f"| 负结果条目 | **{live['negatives']}** |",
        "",
        "## 能力",
        "",
        "| 层 | 提供什么 |",
        "|---|---|",
    ]
    for layer, description in CAPABILITIES:
        if callable(description):
            description = description(live)
        lines.append(f"| **{layer}** | {description} |")
    lines += ["", "## 怎么用（六条命令）", "", "```powershell"]
    for title, command in STEPS:
        lines += [f"# {title}", command]
    lines += ["```", "", "## 关键实测结论（都有归档与门禁）", ""]
    for topic, finding in KEY_FINDINGS:
        lines.append(f"- **{topic}**：{finding}")
    lines += [
        "",
        "这些数字的出处：`docs/DEFAULTS.md`（逐条证据）、`docs/DECISIONS.md`（决策矩阵）、"
        "`research/README.md`（证据库索引）、`research/EVIDENCE_POLICY.md`（证据规则）。",
        "",
        "## 已知限制（诚实清单）",
        "",
    ]
    for item in LIMITATIONS:
        lines.append(f"- {item}")
    lines += [
        "",
        "## 发布自检",
        "",
        "```powershell",
        "py -3 scripts/release_check.py    # 全部门禁（含证据重跑、场景/决策断言、端到端演练、skill 流程）",
        "```",
        "",
        "本文件由 `scripts/build_release_notes.py` 生成（`--check` 可校验），计数直接取自代码与证据库。",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    content = build()
    if args.check:
        current = DOC.read_text(encoding="utf-8", errors="replace") if DOC.exists() else ""
        if current.strip() != content.strip():
            print("RELEASE_NOTES.md is stale; run scripts/build_release_notes.py")
            return 1
        print("RELEASE_NOTES.md is up to date")
        return 0
    DOC.write_text(content, encoding="utf-8", errors="replace")
    print(f"wrote {DOC} ({len(content.splitlines())} lines)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())