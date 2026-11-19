"""Real prior -> proxy campaign with automatic transfer credibility.

Uses the six REAL ART Br-campaign measurements (Table 2) as the source study and
a documented *proxy* oracle for the target campaign, so the flow "real history
informs a new campaign" can be exercised end to end:

1. target runs ``cold_first`` proxy experiments on its own;
2. ``assess_transfer`` estimates how well the real source agrees with them;
3. ``transfer_from`` imports the best real points with the suggested
   ``prior_reliability``;
4. the campaign continues on the proxy and we check whether it rediscovers the
   measured 99% condition.

Run:
    py -3 examples/paper_campaigns/run_real_prior_transfer.py
"""

from __future__ import annotations

import csv
import importlib.util
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from condopt.base import Objective  # noqa: E402
from condopt.space import SearchSpace  # noqa: E402
from condopt.study import Study  # noqa: E402

HERE = Path(__file__).resolve().parent
CSV_PATH = HERE / "art_table2_real.csv"
OUT = HERE / "runs" / "real_prior_transfer"

_spec = importlib.util.spec_from_file_location("run_paper_campaigns", HERE / "run_paper_campaigns.py")
_campaigns = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_campaigns)
art_proxy = _campaigns.art_proxy

REAL_BEST = {
    "solvent": "DMF",
    "photocatalyst": "4CzIPN",
    "ni_precursor": "Ni(PPh3)2Cl2",
    "ni_ligand": "4,4'-dimethoxy-2,2'-bpy",
    "amine": "morpholine",
}


def build_space() -> SearchSpace:
    return SearchSpace.from_dict([
        {"name": "solvent", "kind": "categorical", "choices": _campaigns.SOLVENTS},
        {"name": "photocatalyst", "kind": "categorical", "choices": _campaigns.PHOTOCATALYSTS},
        {"name": "ni_precursor", "kind": "categorical", "choices": _campaigns.NI_PRECURSORS},
        {"name": "ni_ligand", "kind": "categorical", "choices": _campaigns.NI_LIGANDS},
        {"name": "amine", "kind": "categorical", "choices": _campaigns.AMINES},
    ])


def make_source() -> Study:
    shutil.rmtree(OUT / "source_real", ignore_errors=True)
    study = Study.create(
        OUT / "source_real", build_space(), Objective(name="yield", direction="max"),
        algorithm="tpe", budget=8, batch_size=4, seed=0,
    )
    rows = list(csv.DictReader(CSV_PATH.read_text(encoding="utf-8").splitlines()))
    study.import_observations([
        {"x": {name: row[name] for name in study.space.names}, "y": float(row["yield"])}
        for row in rows
    ])
    return study


def drive(study: Study, budget: int, batch_size: int = 2) -> None:
    while study.remaining() > 0 and len(study.dataset.observations) < budget:
        n = min(batch_size, study.remaining(), budget - len(study.dataset.observations))
        pending = study.ask(n)
        study.tell([{"id": item["id"], "y": art_proxy(item["x"])} for item in pending])


def run_arm(use_transfer: bool, seed: int = 0, total_budget: int = 16, cold_first: int = 4) -> dict:
    run_dir = OUT / f"target_{'warm' if use_transfer else 'cold'}_{seed}"
    shutil.rmtree(run_dir, ignore_errors=True)
    source = make_source()
    target = Study.create(
        run_dir, build_space(), Objective(name="yield", direction="max"),
        algorithm="tpe", budget=total_budget, batch_size=2, seed=seed,
    )
    drive(target, cold_first)
    audit = target.assess_transfer(source.dir)
    reliability = audit.get("suggested_reliability") if audit.get("ready") else None
    transferred = 0
    if use_transfer:
        report = target.transfer_from(
            source.dir, top_k=3, reason="real ART priors",
            prior_reliability=reliability if reliability is not None else 0.6,
        )
        transferred = int(report["imported"])
    drive(target, total_budget + transferred if use_transfer else total_budget)
    best = target.best()
    hit = best is not None and all(best.x[name] == value for name, value in REAL_BEST.items())
    return {
        "arm": "warm" if use_transfer else "cold",
        "seed": seed,
        "audit": audit,
        "transferred": transferred,
        "proxy_best": None if best is None else round(float(best.y), 2),
        "rediscovered_real_best": hit,
        "n_proxy_evals": len(target.dataset.observations) - transferred,
    }


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = [run_arm(False, seed) for seed in (0, 1)]
    rows += [run_arm(True, seed) for seed in (0, 1)]
    report = {
        "note": "source = real ART Table 2 measurements; target oracle = documented proxy",
        "rows": rows,
        "cold_rediscovered": sum(row["rediscovered_real_best"] for row in rows if row["arm"] == "cold"),
        "warm_rediscovered": sum(row["rediscovered_real_best"] for row in rows if row["arm"] == "warm"),
    }
    (HERE / "real_prior_transfer_results.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8"
    )
    for row in rows:
        print(json.dumps({
            "arm": row["arm"], "seed": row["seed"],
            "suggested_reliability": row["audit"].get("suggested_reliability"),
            "spearman": row["audit"].get("spearman"),
            "transferred": row["transferred"],
            "proxy_best": row["proxy_best"],
            "rediscovered_real_best": row["rediscovered_real_best"],
        }, ensure_ascii=False))
    print(f"cold hits={report['cold_rediscovered']} warm hits={report['warm_rediscovered']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())