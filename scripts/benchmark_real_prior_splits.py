"""How robust is the real-data warm-start result? Vary seeds and cold-start size.

Re-runs the real ART prior-transfer flow (``run_real_prior_transfer.run_arm``)
over several seeds and ``cold_first`` values, so the documented conclusion rests
on a distribution rather than two runs.

Run:
    py -3 scripts/benchmark_real_prior_splits.py --seeds 0 1 2 3 --cold-first 2 3 4
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import shutil
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

EXAMPLE = ROOT / "examples" / "paper_campaigns" / "run_real_prior_transfer.py"
OUT = ROOT / "runs" / "benchmark_real_prior_splits"

_spec = importlib.util.spec_from_file_location("real_prior_transfer", EXAMPLE)
_module = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
sys.modules[_spec.name] = _module
_spec.loader.exec_module(_module)


def run_settings(seeds: list[int], cold_firsts: list[int], budget: int) -> list[dict]:
    rows = []
    for cold_first in cold_firsts:
        for seed in seeds:
            for use_transfer in (False, True):
                # keep every arm in its own directory so runs never overlap
                _module.OUT = OUT / f"cold{cold_first}_s{seed}"
                row = _module.run_arm(
                    use_transfer, seed=seed, total_budget=budget, cold_first=cold_first
                )
                row["cold_first"] = cold_first
                rows.append(row)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3])
    parser.add_argument("--cold-first", type=int, nargs="+", default=[2, 3, 4])
    parser.add_argument("--budget", type=int, default=16)
    parser.add_argument("--out", default=str(OUT / "results.csv"))
    args = parser.parse_args()

    shutil.rmtree(OUT, ignore_errors=True)
    rows = run_settings(args.seeds, args.cold_first, args.budget)
    summary = {}
    for arm in ("cold", "warm"):
        subset = [row for row in rows if row["arm"] == arm]
        hits = sum(row["rediscovered_real_best"] for row in subset)
        bests = [row["proxy_best"] for row in subset if row["proxy_best"] is not None]
        summary[arm] = {
            "n": len(subset),
            "hits": hits,
            "hit_rate": round(hits / max(1, len(subset)), 3),
            "best_mean": round(statistics.mean(bests), 2) if bests else None,
        }
        print(json.dumps({"arm": arm, **summary[arm]}, ensure_ascii=False))
    by_cold = {}
    for row in rows:
        by_cold.setdefault((row["cold_first"], row["arm"]), []).append(row)
    for (cold_first, arm), subset in sorted(by_cold.items()):
        hits = sum(row["rediscovered_real_best"] for row in subset)
        print(
            f"cold_first={cold_first} arm={arm}: hits={hits}/{len(subset)} "
            f"best_mean={round(statistics.mean(row['proxy_best'] for row in subset), 2)}"
        )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "cold_first", "arm", "seed", "suggested_reliability", "spearman", "transferred",
        "proxy_best", "rediscovered_real_best", "n_proxy_evals",
    ]
    with out.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                "cold_first": row["cold_first"],
                "arm": row["arm"],
                "seed": row["seed"],
                "suggested_reliability": row["audit"].get("suggested_reliability"),
                "spearman": row["audit"].get("spearman"),
                "transferred": row["transferred"],
                "proxy_best": row["proxy_best"],
                "rediscovered_real_best": row["rediscovered_real_best"],
                "n_proxy_evals": row["n_proxy_evals"],
            })
    print(f"CSV: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())