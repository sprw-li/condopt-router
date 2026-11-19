"""Re-run selected evidence scripts and check the documented claims still hold.

The scenario matrix proves *which* default is chosen; this script proves the
*numbers behind it* are still reproducible.  Each claim runs a benchmark script
at a reduced budget and compares the parsed summary.

Run:
    py -3 scripts/verify_evidence.py            # all claims (~10 min)
    py -3 scripts/verify_evidence.py --quick    # two cheapest claims
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

def _force_utf8_stdio() -> None:
    """Emit UTF-8 regardless of the Windows console codepage (cp936 here)."""
    import sys

    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
        except (AttributeError, ValueError):  # pragma: no cover
            pass


def _utf8_child_env() -> dict:
    """Environment for child processes so their stdio is UTF-8, not the codepage."""
    import os

    return {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"}


ROOT = Path(__file__).resolve().parents[1]


@dataclass
class Claim:
    name: str
    command: list[str]
    check: str
    description: str
    quick: bool = False
    #: registry key in docs/DEFAULT_EVIDENCE.json that this claim backs
    decision: str = ""


CLAIMS: tuple[Claim, ...] = (
    Claim(
        "descriptor_hv_difference_is_not_detected",
        [
            sys.executable, "scripts/noise_floor.py", "--mode", "paired",
            "--problem", "moo_solvent_constrained",
            "--problem-b", "moo_solvent_constrained_desc",
            "--algorithm", "cqehvi", "--budget", "40",
            "--seeds", "0", "1", "2", "3", "4",
        ],
        "noise-floor: paired t-test not significant (significant_95 == false)",
        "描述符没有改变可行 HV：配对检验未检出差异（A4 的 TOST 口径下 CI 跨越边界 ⇒ 只能写“未检出”，不能写“等价”）",
        decision="descriptor_features",
    ),
    Claim(
        "descriptors_raise_coverage_not_quality",
        [
            sys.executable, "scripts/benchmark_momfbo.py",
            "--problem", "moo_solvent_constrained_desc", "--budget", "40",
            "--seeds", "0", "1", "2", "--algorithms", "cqehvi",
            "--out", "runs/_claim_desc/hv_desc.csv",
        ],
        "descriptors-moo: desc runs produce feasible points (n_feasible_full_mean >= 20)",
        "MOO+约束下描述符版能产出大量可行点（覆盖提升），需要与 one-hot 版并排解读",
        decision="descriptor_features",
    ),
    Claim(
        "descriptors_only_help_a_constraint_aware_optimizer",
        [
            sys.executable, "scripts/benchmark_solvent_constrained.py",
            "--budget", "40", "--seeds", "0", "1", "2", "3", "4",
        ],
        "descriptors: constrained feasible_rate(desc) > constrained feasible_rate(one-hot) "
        "and bayes best_feasible(desc) < bayes best_feasible(one-hot)",
        "描述符只对约束感知算法有益；对约束盲算法更差",
        decision="descriptor_features",
    ),
    Claim(
        "staged_transfer_can_lose_when_the_veto_misses",
        [
            sys.executable, "scripts/benchmark_transfer_adaptive.py",
            "--problem", "discrete_catalyst", "--budget", "24", "--seeds", "0", "1", "2", "3",
        ],
        "veto-miss: biased staged delta < biased fixed_k1 delta and veto count == 0",
        "分阶段迁移的保护取决于审计能否否决：该问题上 0/4 被否，分阶段差于 fixed_k1",
        decision="transfer_top_k",
    ),
    Claim(
        "staged_api_matches_the_hand_rolled_policy",
        [
            sys.executable, "scripts/benchmark_transfer_adaptive.py",
            "--seeds", "0", "1",
        ],
        "staged-api: staged_audit delta == staged_recheck delta for every source quality",
        "Study.transfer_staged 与手工分阶段配方结果逐位一致（API 化未改变行为）",
        decision="transfer_top_k",
    ),
    Claim(
        "transfer_veto_band_is_safe_from_0_4",
        [
            sys.executable, "scripts/benchmark_transfer_adaptive.py",
            "--sweep-veto", "0.2", "0.4", "0.6", "--seeds", "0", "1",
        ],
        "veto-band: threshold 0.2 worst_delta < 0 and threshold 0.4/0.6 worst_delta > 0",
        "否决线 0.2 起不到保护作用（有偏源导致负收益），0.4–0.6 安全",
        decision="transfer_top_k",
    ),
    Claim(
        "staged_transfer_avoids_the_biased_loss",
        [
            sys.executable, "scripts/benchmark_transfer_adaptive.py",
            "--seeds", "0", "1",
        ],
        "staged: biased staged delta > biased fixed_k3 delta and > biased fixed_k5 delta",
        "分阶段迁移（小 top_k + 两批后审计）在有偏源上不亏，而固定 k=3/5 会亏",
        decision="transfer_top_k",
    ),
    Claim(
        "audit_is_blind_to_the_size_of_the_bias",
        [
            sys.executable, "scripts/benchmark_audit_calibration.py",
            "--problem", "ackley10", "--budget", "20", "--source-points", "30",
            "--fine", "--seeds", "0", "1",
        ],
        "audit-blindness: one distinct audit_mean across all bias levels, deltas of both signs",
        "审计分在偏差 0~−18 上完全不变，而收益有正有负 → 只能当一票否决器",
        decision="transfer_reliability_grading",
    ),
    Claim(
        "audit_cannot_separate_a_biased_source",
        [
            sys.executable, "scripts/benchmark_audit_calibration.py",
            "--problem", "ackley10", "--budget", "20", "--source-points", "30",
            "--seeds", "0", "1",
        ],
        "audit-calibration: biased audit >= 0.6 while delta_vs_cold < 0",
        "审计对“中等偏差”无分辨力：有偏源仍得高分却带来负收益（故低置信度封顶 0.6）",
        decision="transfer_reliability_grading",
    ),
    Claim(
        "prior_reliability_is_observable_when_powered",
        [
            sys.executable, "scripts/benchmark_reliability_power.py",
            "--seeds", "0", "1",
        ],
        "reliability: r1.0 delta > r0.4 delta",
        "有功效实验中，可靠源打折会损失收益（reliability 机制可观测）",
        decision="transfer_reliability_grading",
    ),
    Claim(
        "warm_start_beats_cold_after_plateau_fix",
        [
            sys.executable, "scripts/sweep_transfer_top_k.py",
            "--top-ks", "0", "3", "--seeds", "0", "1", "2",
        ],
        "top-k: k=3 delta_vs_cold > 0",
        "修掉“先验导致平台期误判”后，好源暖启动（k=3）应优于冷启动",
        decision="transfer_top_k",
    ),
    Claim(
        "constrained_beats_bayes_on_hard_feasibility",
        [
            sys.executable, "scripts/benchmark_constrained_second.py",
            "--problems", "constraint_hard", "--budget", "30", "--seeds", "0", "1", "2",
        ],
        "constrained: feasible_mean > bayes and >= 2/3 runs feasible",
        "硬可行域（6%）上 constrained 有可行解且更优；bayes 多数运行找不到可行点",
        decision="constrained_single_objective",
    ),
    Claim(
        "noise_advice_matches_measurement",
        [
            sys.executable, "scripts/benchmark_noise_policy.py",
            "--budget", "24", "--seeds", "0", "1",
        ],
        "noise-policy: ei_default >= ucb on every noise level",
        "高噪声下 UCB+大 kappa 不比默认 EI 好（所以建议里不默认切 UCB）",
        decision="noise_policy",
    ),
    Claim(
        "tiny_budget_keeps_the_surrogate",
        [
            sys.executable, "scripts/benchmark_tiny_budget.py",
            "--budgets", "4", "--seeds", "0", "1",
        ],
        "tiny-budget: parego.feasible_hv >= sobol.feasible_hv",
        "极小预算下代理类不逊于纯空间填充（MOO+约束：parego 3-4/4 有前沿 vs sobol 1/4）",
        decision="tiny_budget_policy",
    ),
    Claim(
        "confirm_batch_three_is_not_worse",
        [
            sys.executable, "scripts/benchmark_confirm_budget.py",
            "--problem", "branin", "--budget", "12", "--seeds", "0", "1", "2",
        ],
        "confirm-budget: confirm_n=3.best_mean >= confirm_n=2.best_mean",
        "默认 3 次复孔在 branin 上不劣于 2 次（保持默认的另一半依据）",
        decision="confirm_n",
    ),
    Claim(
        "mid_budget_policies_agree_below_20",
        [
            sys.executable, "scripts/benchmark_mid_budget.py",
            "--budgets", "12", "--seeds", "0", "1",
        ],
        "mid-budget: policies equal at budget 12",
        "预算 12 时预算优先与停滞优先结果一致（阈值放在 60 无副作用）",
        decision="budget_first_threshold",
    ),
    Claim(
        "transfer_audit_grades_bad_sources_low",
        [
            sys.executable, "scripts/benchmark_transfer_audit.py",
            "--seeds", "0", "1",
        ],
        "transfer-grades: consistent.reliability_mean >= 0.8 > bad <= 0.6",
        "一致源建议值 >=0.8，而 shifted/shuffled/flipped <=0.6（分级阈值的依据）",
        decision="transfer_reliability_grading",
    ),
    Claim(
        "cmaes_beats_bayes_10d",
        [
            sys.executable, "scripts/benchmark_highdim.py",
            "--budget", "50", "--batch", "6", "--seeds", "0", "1",
            "--algorithms", "bayes", "cmaes",
        ],
        "cmaes < bayes on rosenbrock10 and ackley10",
        "10 维光滑连续上 cmaes 优于 bayes（−18.1 vs −155.4）",
        quick=True,
        decision="cmaes_high_dim",
    ),
    Claim(
        "constrained_beats_bayes_on_constraints",
        [
            sys.executable, "scripts/benchmark_solvent_constrained.py",
            "--budget", "40", "--seeds", "0", "1",
            "--algorithms", "constrained", "bayes",
        ],
        "constrained > bayes best_feasible",
        "带硬约束时约束感知算法更好（90.39 vs 84.7）",
        quick=True,
        decision="constrained_single_objective,descriptor_features",
    ),
    Claim(
        "mfparego_matches_momfbo_two_objectives",
        [
            sys.executable, "scripts/benchmark_momfbo.py",
            "--problem", "mf_moo", "--budget", "30", "--batch", "4",
            "--seeds", "0", "1", "2", "--algorithms", "momfbo", "mfparego",
        ],
        "mfparego >= 0.97 * momfbo feasible HV",
        "2 目标 + 多保真（无约束，第 2 个独立数据集）：mfparego 674.5 vs momfbo 669.5",
        decision="mfparego_multi_fidelity_moo",
    ),
    Claim(
        "mfparego_matches_or_beats_momfbo",
        [
            sys.executable, "scripts/benchmark_momfbo.py",
            "--problem", "mf_moo3", "--budget", "30", "--batch", "4",
            "--seeds", "0", "1", "--algorithms", "momfbo", "mfparego",
        ],
        "mfparego >= 0.97 * momfbo feasible HV",
        "3 目标 + 多保真上 mfparego 不落后（27399 vs 26869）",
        quick=True,
        decision="mfparego_multi_fidelity_moo",
    ),
    Claim(
        "mfparego_matches_momfbo_under_constraints",
        [
            sys.executable, "scripts/benchmark_momfbo.py",
            "--problem", "mf_moo_constrained", "--budget", "30", "--batch", "4",
            "--seeds", "0", "1", "--algorithms", "momfbo", "mfparego",
        ],
        "mfparego >= 0.97 * momfbo feasible HV (constrained)",
        "2 目标 + 约束 + 多保真上 mfparego 不落后（642.5 vs 635.6）",
        decision="mfparego_multi_fidelity_moo",
    ),
    Claim(
        "retune_beats_stopping_on_stagnation",
        [
            sys.executable, "scripts/benchmark_stagnation.py",
            "--problems", "rosenbrock10", "--budget", "40", "--batch", "4",
            "--seeds", "0", "1", "--arms", "stop", "tune",
        ],
        "tune best_mean > stop best_mean",
        "停滞时调参继续优于直接停止（−205.4 vs −373.7）",
        decision="stagnation_retune",
    ),
    Claim(
        "deliverable_hv_is_tracked_and_bounded",
        [
            sys.executable, "examples/multiobjective_executor.py",
        ],
        "0 < feasible_hypervolume <= hypervolume",
        "约束多目标的交付口径：可行 HV 被单独记录且不超过 raw HV（实测 80.94 vs 143.03）",
        quick=True,
        decision="delivery_evidence_policy",
    ),
    Claim(
        "feasible_full_evidence_always_delivered",
        [
            sys.executable, "scripts/benchmark_feasible_full.py",
            "--budget", "20", "--batch", "3", "--seeds", "0", "1",
        ],
        "has_feasible_full_rate == 1.0 for the aware arm",
        "约束 + 多保真下总能拿到可行的全保真证据（0.40 → 1.00）",
        decision="delivery_evidence_policy",
    ),
)


def _run(command: list[str]) -> tuple[int, str]:
    result = subprocess.run(
        command, cwd=str(ROOT), capture_output=True, text=True, encoding="utf-8", errors="replace"
    )
    return result.returncode, (result.stdout or "") + (result.stderr or "")


def _json_lines(output: str) -> list[dict]:
    rows = []
    for line in output.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def _summaries(output: str) -> dict[str, dict]:
    return {
        row["algorithm"]: row
        for row in _json_lines(output)
        if "algorithm" in row and "n" in row
    }


def _by_key(rows: list[dict], key: str, metric: str) -> dict[str, float]:
    buckets: dict[str, list[float]] = {}
    for row in rows:
        value = row.get(metric)
        if value is None:
            continue
        buckets.setdefault(str(row.get(key)), []).append(float(value))
    return {name: sum(items) / len(items) for name, items in buckets.items()}


def evaluate_claim(claim: Claim) -> dict:
    import time

    started = time.perf_counter()
    code, output = _run(claim.command)
    elapsed = round(time.perf_counter() - started, 1)
    ok = code == 0
    detail = ""
    if ok and claim.check.startswith("cmaes"):
        summaries = _summaries(output)
        cmaes = summaries.get("cmaes")
        bayes = summaries.get("bayes")
        if not cmaes or not bayes:
            ok, detail = False, "missing summary rows"
        else:
            cmaes_mean = cmaes.get("best_score_mean", cmaes.get("hv_feasible_mean"))
            bayes_mean = bayes.get("best_score_mean", bayes.get("hv_feasible_mean"))
            per_problem = {}
            for row in _json_lines(output):
                if "problem" not in row:
                    continue
                per_problem.setdefault(row["problem"], {})[row["algorithm"]] = row
            failures = [
                name
                for name, pair in per_problem.items()
                if "cmaes" in pair and "bayes" in pair
                and not (
                    pair["cmaes"].get("best_score_mean", 0)
                    > pair["bayes"].get("best_score_mean", 0)
                )
            ]
            ok = not failures
            detail = f"cmaes={cmaes_mean} bayes={bayes_mean} failing={failures}"
    elif ok and claim.check.startswith("noise-floor:"):
        rows = _json_lines(output)
        row = next((item for item in rows if "significant_95" in item), None)
        if row is None:
            ok, detail = False, "missing paired report"
        else:
            ok = row["significant_95"] is False
            detail = (
                f"mean_delta={row.get('mean_delta')} t={row.get('t_statistic')} "
                f"critical={row.get('t_critical_95')} significant={row.get('significant_95')}"
            )
    elif ok and claim.check.startswith("descriptors-moo:"):
        rows = _json_lines(output)
        desc = next(
            (row for row in rows if "n_feasible_full_mean" in row), None
        )
        if desc is None:
            ok, detail = False, "missing summary row"
        else:
            n_feasible = float(desc["n_feasible_full_mean"])
            hv = float(desc["hv_feasible_full_mean"])
            ok = n_feasible >= 20 and hv > 0
            detail = f"desc feasible_points={n_feasible} feasible_hv={hv}"
    elif ok and claim.check.startswith("descriptors:"):
        rows = [
            row for row in _json_lines(output)
            if "problem" in row and "algorithm" in row and "feasible_rate_mean" in row
        ]
        def arm(problem: str, algorithm: str):
            for row in rows:
                if row["problem"] == problem and row["algorithm"] == algorithm:
                    return row
            return None

        one_hot = arm("solvent_constrained", "constrained")
        desc = arm("solvent_constrained_desc", "constrained")
        bayes_one_hot = arm("solvent_constrained", "bayes")
        bayes_desc = arm("solvent_constrained_desc", "bayes")
        if None in (one_hot, desc, bayes_one_hot, bayes_desc):
            ok, detail = False, "missing encoding arms"
        else:
            ok = (
                float(desc["feasible_rate_mean"]) > float(one_hot["feasible_rate_mean"])
                and float(bayes_desc["best_feasible_mean"])
                < float(bayes_one_hot["best_feasible_mean"])
            )
            detail = (
                f"constrained feasible {one_hot['feasible_rate_mean']} -> {desc['feasible_rate_mean']}; "
                f"bayes best {bayes_one_hot['best_feasible_mean']} -> {bayes_desc['best_feasible_mean']}"
            )
    elif ok and claim.check.startswith("veto-miss:"):
        rows = [
            row for row in _json_lines(output)
            if "quality" in row and "policy" in row and "delta_vs_cold_mean" in row
        ]
        def arm(quality: str, policy: str):
            for row in rows:
                if row["quality"] == quality and row["policy"] == policy:
                    return row
            return None

        staged = arm("biased", "staged_audit")
        fixed = arm("biased", "fixed_k1")
        if not staged or not fixed:
            ok, detail = False, "missing biased arms"
        else:
            staged_delta = float(staged["delta_vs_cold_mean"])
            fixed_delta = float(fixed["delta_vs_cold_mean"])
            ok = int(staged.get("vetoed") or 0) == 0 and staged_delta < fixed_delta
            detail = (
                f"biased staged={staged_delta} (vetoed={staged.get('vetoed')}) "
                f"fixed_k1={fixed_delta}"
            )
    elif ok and claim.check.startswith("staged-api:"):
        rows = [
            row for row in _json_lines(output)
            if "quality" in row and "policy" in row and "delta_vs_cold_mean" in row
        ]
        def delta(quality: str, policy: str):
            for row in rows:
                if row["quality"] == quality and row["policy"] == policy:
                    return float(row["delta_vs_cold_mean"])
            return None

        qualities = ("good", "biased", "shuffled")
        pairs = {
            quality: (delta(quality, "staged_audit"), delta(quality, "staged_recheck"))
            for quality in qualities
        }
        missing = [quality for quality, pair in pairs.items() if None in pair]
        if missing:
            ok, detail = False, f"missing arms for {missing}"
        else:
            mismatches = {
                quality: pair for quality, pair in pairs.items()
                if abs(pair[0] - pair[1]) > 1e-9
            }
            ok = not mismatches
            detail = f"deltas={ {q: pairs[q][0] for q in qualities} } mismatches={mismatches}"
    elif ok and claim.check.startswith("veto-band:"):
        summary = None
        for row in _json_lines(output):
            if "veto_sweep" in row:
                summary = row["veto_sweep"]
        if not summary:
            ok, detail = False, "missing veto_sweep report"
        else:
            by_threshold = {round(float(item["threshold"]), 2): item for item in summary}
            loose = by_threshold.get(0.2, {}).get("worst_delta")
            safe = [
                by_threshold.get(value, {}).get("worst_delta")
                for value in (0.4, 0.6)
            ]
            ok = (
                loose is not None
                and float(loose) < 0
                and all(value is not None and float(value) > 0 for value in safe)
            )
            detail = f"worst_delta t0.2={loose} t0.4/t0.6={safe}"
    elif ok and claim.check.startswith("staged:"):
        rows = [
            row for row in _json_lines(output)
            if "quality" in row and "policy" in row and "delta_vs_cold_mean" in row
        ]
        def arm(quality: str, policy: str):
            for row in rows:
                if row["quality"] == quality and row["policy"] == policy:
                    return float(row["delta_vs_cold_mean"])
            return None

        staged = arm("biased", "staged_audit")
        k3 = arm("biased", "fixed_k3")
        k5 = arm("biased", "fixed_k5")
        good_staged = arm("good", "staged_audit")
        good_k5 = arm("good", "fixed_k5")
        vetoed = sum(
            1 for row in rows
            if row["quality"] == "biased" and row["policy"] == "staged_audit"
            and row.get("vetoed")
        )
        if None in (staged, k3, k5, good_staged, good_k5):
            ok, detail = False, "missing arms"
        else:
            ok = (
                staged > k3
                and staged > k5
                and good_staged >= good_k5 - 0.5
                and vetoed >= 1
            )
            detail = (
                f"biased staged={staged} k3={k3} k5={k5} vetoed={vetoed}; "
                f"good staged={good_staged} k5={good_k5}"
            )
    elif ok and claim.check.startswith("audit-blindness:"):
        rows = [
            row for row in _json_lines(output)
            if "quality" in row and "audit_mean" in row and "delta_vs_cold_mean" in row
        ]
        if len(rows) < 4:
            ok, detail = False, f"only {len(rows)} graded quality levels"
        else:
            audits = {round(float(row["audit_mean"]), 3) for row in rows}
            deltas = [float(row["delta_vs_cold_mean"]) for row in rows]
            positive = any(value > 0 for value in deltas)
            negative = any(value < 0 for value in deltas)
            ok = len(audits) == 1 and positive and negative
            detail = (
                f"distinct_audit={sorted(audits)} deltas="
                f"[{min(deltas):.3f}, {max(deltas):.3f}] signs={'+-' if positive and negative else 'single'}"
            )
    elif ok and claim.check.startswith("audit-calibration:"):
        rows = _json_lines(output)
        biased = [row for row in rows if str(row.get("quality")) == "biased"]
        if not biased:
            ok, detail = False, "missing the biased arm"
        else:
            audit = float(biased[0].get("audit_mean") or 0.0)
            delta = float(biased[0].get("delta_vs_cold_mean") or 0.0)
            ok = audit >= 0.6 and delta < 0
            detail = f"biased audit={audit} delta={delta}"
    elif ok and claim.check.startswith("reliability:"):
        rows = _json_lines(output)
        arms = {str(row.get("arm")): row for row in rows if "arm" in row}
        full, discounted = arms.get("r1.0"), arms.get("r0.4")
        if not full or not discounted:
            ok, detail = False, "missing reliability arms"
        else:
            ok = float(full.get("delta_vs_cold", 0)) > float(discounted.get("delta_vs_cold", 0))
            detail = (
                f"r1.0 delta={full.get('delta_vs_cold')} "
                f"r0.4 delta={discounted.get('delta_vs_cold')}"
            )
    elif ok and claim.check.startswith("top-k:"):
        rows = _json_lines(output)
        stats = {int(row.get("top_k", -1)): row for row in rows if "top_k" in row}
        cold, warm = stats.get(0), stats.get(3)
        if not cold or not warm:
            ok, detail = False, "missing top-k arms"
        else:
            ok = float(warm.get("delta_vs_cold", 0.0)) > 0
            detail = (
                f"cold={cold.get('best_mean')} k3={warm.get('best_mean')} "
                f"delta={warm.get('delta_vs_cold')}"
            )
    elif ok and claim.check.startswith("constrained:"):
        stats = {
            str(row.get("algorithm")): row
            for row in _json_lines(output)
            if "algorithm" in row and "best_feasible_mean" in row
        }
        constrained, bayes = stats.get("constrained"), stats.get("bayes")
        if not constrained or not bayes:
            ok, detail = False, "missing arm summaries"
        else:
            ok = (
                constrained["best_feasible_mean"] is not None
                and bayes["best_feasible_mean"] is not None
                and constrained["best_feasible_mean"] > bayes["best_feasible_mean"] + 1e-9
                and int(constrained.get("n_feasible", 0)) >= 2
            )
            detail = (
                f"constrained={constrained['best_feasible_mean']}"
                f"({constrained.get('n_feasible')} feasible) "
                f"bayes={bayes['best_feasible_mean']}({bayes.get('n_feasible')} feasible)"
            )
    elif ok and claim.check.startswith("noise-policy:"):
        by_noise: dict[float, dict[str, float]] = {}
        for row in _json_lines(output):
            if "noise_std" in row and isinstance(row.get("arms"), dict):
                by_noise[float(row["noise_std"])] = {
                    str(arm): float(value) for arm, value in row["arms"].items()
                }
        failures = [
            noise
            for noise, arms in by_noise.items()
            if "ei_default" in arms
            and "ucb_kappa8" in arms
            and arms["ucb_kappa8"] > arms["ei_default"] + 1e-9
        ]
        ok = bool(by_noise) and not failures
        detail = f"noise levels={sorted(by_noise)} ucb>ei at={failures}"
    elif ok and claim.check.startswith("tiny-budget:"):
        rows = _json_lines(output)
        stats = _by_key(rows, "algorithm", "metric_mean")
        parego, sobol = stats.get("parego"), stats.get("sobol")
        if parego is None or sobol is None:
            ok, detail = False, "missing arm summaries"
        else:
            ok = parego >= sobol
            detail = f"parego={parego} sobol={sobol}"
    elif ok and claim.check.startswith("confirm-budget:"):
        stats = {
            str(row.get("confirm_n")): row.get("best_mean")
            for row in _json_lines(output)
            if row.get("best_mean") is not None
        }
        three, two = stats.get("3"), stats.get("2")
        if three is None or two is None:
            ok, detail = False, "missing confirm-budget arms"
        else:
            ok = three >= two
            detail = f"confirm3={three} confirm2={two}"
    elif ok and claim.check.startswith("mid-budget:"):
        buckets: dict[str, list[float]] = {}
        for row in _json_lines(output):
            if row.get("best_mean") is None:
                continue
            buckets.setdefault(str(row.get("policy")), []).append(float(row["best_mean"]))
        values = {name: sum(items) / len(items) for name, items in buckets.items() if items}
        if len(values) < 2:
            ok, detail = False, "missing policies"
        else:
            spread = max(values.values()) - min(values.values())
            ok = spread <= 1e-6
            detail = f"policies={values} spread={round(spread, 6)}"
    elif ok and claim.check.startswith("transfer-grades:"):
        stats = {str(row.get("kind")): row.get("reliability_mean") for row in _json_lines(output)}
        consistent = stats.get("consistent")
        bad = [value for kind, value in stats.items() if kind != "consistent" and value is not None]
        if consistent is None or not bad:
            ok, detail = False, "missing arm summaries"
        else:
            ok = consistent >= 0.8 and max(bad) <= 0.6
            detail = f"consistent={consistent} worst_bad={max(bad)}"
    elif ok and claim.check.startswith("constrained"):
        summaries = _summaries(output)
        constrained = summaries.get("constrained")
        bayes = summaries.get("bayes")
        if not constrained or not bayes:
            ok, detail = False, "missing summary rows"
        else:
            ok = constrained["best_feasible_mean"] > bayes["best_feasible_mean"]
            detail = (
                f"constrained={constrained['best_feasible_mean']} "
                f"bayes={bayes['best_feasible_mean']}"
            )
    elif ok and claim.check.startswith("mfparego"):
        summaries = _summaries(output)
        mfparego = summaries.get("mfparego")
        momfbo = summaries.get("momfbo")
        if not mfparego or not momfbo:
            ok, detail = False, "missing summary rows"
        else:
            ok = mfparego["hv_feasible_full_mean"] >= 0.97 * momfbo["hv_feasible_full_mean"]
            detail = (
                f"mfparego={mfparego['hv_feasible_full_mean']} "
                f"momfbo={momfbo['hv_feasible_full_mean']}"
            )
    elif ok and claim.check.startswith("mfparego >= 0.97 * momfbo feasible HV (constrained)"):
        summaries = _summaries(output)
        mfparego = summaries.get("mfparego")
        momfbo = summaries.get("momfbo")
        if not mfparego or not momfbo:
            ok, detail = False, "missing summary rows"
        else:
            ok = mfparego["hv_feasible_full_mean"] >= 0.97 * momfbo["hv_feasible_full_mean"]
            detail = (
                f"mfparego={mfparego['hv_feasible_full_mean']} "
                f"momfbo={momfbo['hv_feasible_full_mean']}"
            )
    elif ok and claim.check.startswith("tune best_mean"):
        rows = [
            row for row in _json_lines(output)
            if row.get("problem") == "rosenbrock10" and "best_mean" in row
        ]
        tune = next((row for row in rows if row.get("arm") == "tune"), None)
        stop = next((row for row in rows if row.get("arm") == "stop"), None)
        if not tune or not stop:
            ok, detail = False, "missing stagnation summary rows"
        else:
            ok = tune["best_mean"] > stop["best_mean"]
            detail = f"tune={tune['best_mean']} stop={stop['best_mean']}"
    elif ok and claim.check.startswith("0 < feasible_hypervolume"):
        rows = [row for row in _json_lines(output) if "feasible_hypervolume" in row]
        if not rows:
            ok, detail = False, "missing example summary"
        else:
            feasible = rows[-1].get("feasible_hypervolume")
            raw = rows[-1].get("hypervolume")
            ok = bool(
                feasible is not None and raw is not None and 0 < feasible <= raw + 1e-9
            )
            detail = f"feasible_hv={feasible} raw_hv={raw}"
    elif ok and claim.check.startswith("has_feasible_full_rate"):
        rows = [
            row for row in _json_lines(output) if "has_feasible_full_rate" in row
            and row.get("feasibility_aware_full") is True
        ]
        if not rows:
            ok, detail = False, "missing aware-arm summary"
        else:
            rates = [row["has_feasible_full_rate"] for row in rows]
            ok = all(rate >= 0.99 for rate in rates)
            detail = f"aware has_feasible_full_rate={rates}"
    return {
        "claim": claim.name,
        "ok": bool(ok),
        "detail": detail,
        "description": claim.description,
        "seconds": elapsed,
    }


def main() -> int:
    _force_utf8_stdio()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    claims = [claim for claim in CLAIMS if claim.quick] if args.quick else list(CLAIMS)
    results = [evaluate_claim(claim) for claim in claims]
    failures = [row for row in results if not row["ok"]]
    if args.json:
        print(json.dumps({"results": results, "failures": len(failures)},
                         ensure_ascii=False, indent=2))
    else:
        total = 0.0
        for row in results:
            marker = "OK  " if row["ok"] else "FAIL"
            total += row["seconds"]
            print(
                f"[{marker}] {row['claim']} ({row['seconds']}s): "
                f"{row['detail'] or row['description']}"
            )
        print(
            f"verify_evidence: {len(results) - len(failures)}/{len(results)} claims hold "
            f"in {round(total, 1)}s"
        )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())