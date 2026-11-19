"""Regenerate the long-run comparison table inside research/ADAPTER_LONG_RUN.md.

The table is generated from real drill runs so the documented numbers cannot drift
from the code.  Markers delimit the generated block.

Run:
    py -3 scripts/build_adapter_table.py            # 4 configs, full budgets
    py -3 scripts/build_adapter_table.py --quick    # small budgets (format check)
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

START = "<!-- adapter-table:start -->"
END = "<!-- adapter-table:end -->"
DOC = ROOT / "research" / "ADAPTER_LONG_RUN.md"

_spec = importlib.util.spec_from_file_location(
    "adapter_long_run", ROOT / "scripts" / "adapter_long_run.py"
)
_module = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
sys.modules[_spec.name] = _module
_spec.loader.exec_module(_module)

CONFIGS = (
    # (label, problem, budget, batch)
    ("单目标 + MF + 约束", "mf_constrained", 120, 6),
    ("2 目标 + 约束", "moo_constrained", 40, 4),
    ("MF + 2 目标 + 约束", "mf_moo_constrained", 40, 4),
    ("MF + 4 目标", "mf_moo4", 24, 4),
)

COLUMNS = (
    "配置",
    "问题",
    "算法",
    "观测",
    "尝试/失败",
    "成本",
    "墙钟(s)",
    "可行全保真",
    "可交付 HV",
    "raw HV",
    "重复id",
)


def _fmt_curve(curve) -> str:
    if not curve:
        return "—"
    return f"{curve[0]:.2f} → {curve[1]:.2f}"


def render(rows: list[dict]) -> str:
    lines = [
        START,
        "| " + " | ".join(COLUMNS) + " |",
        "|" + "---|" * len(COLUMNS),
    ]
    for row in rows:
        lines.append(
            "| {label} | `{problem}` | `{algorithm}` | {n_observations} | "
            "{attempts}/{failures} | {cost} | {wall} | {full} | {feasible_hv} | {raw_hv} | {dups} |".format(
                label=row["label"],
                problem=row["problem"],
                algorithm=row["algorithm"],
                n_observations=row["n_observations"],
                attempts=row["attempts"],
                failures=row["failures"],
                cost=row["cost_spent"],
                wall=row["wall_seconds"],
                full=row["n_feasible_full_fidelity"],
                feasible_hv=_fmt_curve(row.get("feasible_hypervolume_curve")),
                raw_hv=(
                    "—" if row.get("final_hypervolume") is None
                    else f"{row['final_hypervolume']:.2f}"
                ),
                dups=row["duplicate_suggestion_ids"],
            )
        )
    lines += [
        END,
        "",
        "> 由 `scripts/build_adapter_table.py` 依据真实运行生成（含失败补测、指标口径检查）。",
    ]
    return "\n".join(lines)


def build(quick: bool = False) -> list[dict]:
    rows = []
    for label, problem, budget, batch in CONFIGS:
        effective_budget = 16 if quick else budget
        report = _module.run(
            budget=effective_budget,
            batch_size=batch,
            workers=4,
            seed=0,
            patience=100,
            problem_name=problem,
        )
        report["label"] = label
        report["problem"] = problem
        rows.append(report)
    return rows


def write_doc(table: str) -> None:
    text = DOC.read_text(encoding="utf-8")
    if START in text and END in text:
        head = text.split(START)[0].rstrip()
        tail = text.split(END, 1)[1].lstrip()
        DOC.write_text(head + "\n\n" + table + "\n" + tail, encoding="utf-8")
    else:
        DOC.write_text(text.rstrip() + "\n\n## 长跑对照表\n\n" + table + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--no-write", action="store_true")
    args = parser.parse_args()
    rows = build(quick=args.quick)
    table = render(rows)
    if not args.no_write:
        write_doc(table)
        print(f"table written -> {DOC}")
    for row in rows:
        print(
            f"  {row['label']}: {row['algorithm']} obs={row['n_observations']} "
            f"fail={row['failures']} cost={row['cost_spent']} "
            f"feasible_hv={_fmt_curve(row.get('feasible_hypervolume_curve'))}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())