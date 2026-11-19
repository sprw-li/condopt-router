"""Replay two 2026 automated-lab campaigns with condopt's router.

Why this file exists
--------------------
The user asked for "complex real cases from the last month's automated-lab
papers".  No local campaign record was found on this machine, so this script
reconstructs two published 2026 campaigns from their papers:

1. ART metallaphotoredox HTE (Angew. Chem. Int. Ed. 2026, DOI
   10.1002/anie.3129364, PMC13360563).  The five categorical variables and
   their exact levels are REAL (paper Table 1, 30,000 combinations).  A subset
   of the measured top conditions with yields is REAL (Table 2, Br campaign).
   Because the full 30,000-point yield table is not available in the article
   text, the objective used for the *iterative loop* is a transparent proxy
   calibrated to the paper's reported trends.  It is NOT measured data.

2. Thermoset p-SDL formulation (Chem. Mater. 2026, DOI
   10.1021/acs.chemmater.6c01600, PMC13718556).  The component library,
   composition constraint (<=3 components, cross-linker <=50 wt%) and the
   process limits (curing time <=10 s, temperature rise <=70 C) are REAL
   (paper Section 2.1).  Target property windows and the property oracle are
   synthetic, because the article's target table was not accessible in the
   fetched text.

Both runs only *propose* conditions; no lab is involved.  The point is to
exercise the skill end-to-end (validate -> route -> ask -> tell -> report) on
realistic, high-dimensional, categorical/constrained spaces.

Run:
    py -3 examples/paper_campaigns/run_paper_campaigns.py
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402

from condopt import Objective, Observation, SearchSpace, Study  # noqa: E402
from condopt.render import render_study  # noqa: E402
from condopt.router import OptimizationContext  # noqa: E402

OUT = Path(__file__).resolve().parent / "runs"
OUT.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Case 1: ART metallaphotoredox HTE (Angew 2026)
# ---------------------------------------------------------------------------
SOLVENTS = [
    "DMF", "NBP", "Rhodiasolv Polarclean", "Acetonitrile", "Toluene",
    "Propylene carbonate", "Ethanol", "tert-amyl alcohol", "Anisole",
    "Ethyl acetate", "Isopropanol", "2-MeTHF",
]
PHOTOCATALYSTS = [
    "(Ir[dF(CF3)ppy]2(dtbpy))PF6", "(Ir[dF(CF3)ppy]2(bpy))PF6", "4CzIPN",
    "4DPAIPN", "3CzClIPN", "PheMesAcri.BF4", "PheMe2MesAcri.BF4",
    "Ru(bpy)3(PF6)2", "[Ir(dF(Me)ppy)2(4,4'-dCF3bpy)]PF6",
    "Ru(dtbbpy)3(PF6)2",
]
NI_PRECURSORS = ["Ni(PPh3)2Cl2", "Ni(DME)Cl2", "Ni(TMHD)2", "Ni(MeCN)Cl2", "NiOTf2"]
NI_LIGANDS = [
    "4,4'-dimethoxy-2,2'-bpy", "1,10-phenanthroline", "bathophenanthroline",
    "4,4'-di-tert-butyl-2,2'-bpy", "4,4'-bis(CF3)-2,2'-bpy", "DMEDA",
    "2-(2-pyridyl)benzoxazole", "2-methyl-1,10-phenanthroline",
    "1,10-phenanthroline-5,6-dione", "no ligand",
]
AMINES = ["pyrrolidine", "piperidine", "morpholine", "n-butylamine", "cyclohexylamine"]

# Table 2 (Br campaign), real measured yields.
BR_TOP = [
    {"solvent": "DMF", "photocatalyst": "4CzIPN", "ni_precursor": "Ni(PPh3)2Cl2",
     "ni_ligand": "4,4'-dimethoxy-2,2'-bpy", "amine": "morpholine", "yield": 99.0},
    {"solvent": "NBP", "photocatalyst": "(Ir[dF(CF3)ppy]2(bpy))PF6",
     "ni_precursor": "Ni(PPh3)2Cl2", "ni_ligand": "4,4'-dimethoxy-2,2'-bpy",
     "amine": "morpholine", "yield": 95.0},
    {"solvent": "NBP", "photocatalyst": "3CzClIPN", "ni_precursor": "Ni(PPh3)2Cl2",
     "ni_ligand": "1,10-phenanthroline", "amine": "piperidine", "yield": 91.0},
    {"solvent": "NBP", "photocatalyst": "4CzIPN", "ni_precursor": "Ni(PPh3)2Cl2",
     "ni_ligand": "4,4'-dimethoxy-2,2'-bpy", "amine": "morpholine", "yield": 78.0},
    {"solvent": "DMF", "photocatalyst": "(Ir[dF(CF3)ppy]2(bpy))PF6",
     "ni_precursor": "NiOTf2", "ni_ligand": "1,10-phenanthroline",
     "amine": "morpholine", "yield": 77.0},
    {"solvent": "Anisole", "photocatalyst": "4CzIPN", "ni_precursor": "NiOTf2",
     "ni_ligand": "4,4'-dimethoxy-2,2'-bpy", "amine": "morpholine", "yield": 71.0},
]

# Proxy main effects, hand-calibrated to the trends reported in the paper.
SOLVENT_EFFECT = {
    "DMF": 0.97, "NBP": 0.92, "Anisole": 0.75, "2-MeTHF": 0.70,
    "Rhodiasolv Polarclean": 0.62, "Acetonitrile": 0.55,
    "Propylene carbonate": 0.55, "Toluene": 0.50, "Ethyl acetate": 0.45,
    "Ethanol": 0.45, "tert-amyl alcohol": 0.40, "Isopropanol": 0.35,
}
PC_EFFECT = {
    "4CzIPN": 0.95, "(Ir[dF(CF3)ppy]2(bpy))PF6": 0.90,
    "(Ir[dF(CF3)ppy]2(dtbpy))PF6": 0.75, "3CzClIPN": 0.65, "4DPAIPN": 0.55,
    "PheMesAcri.BF4": 0.45, "PheMe2MesAcri.BF4": 0.40,
    "[Ir(dF(Me)ppy)2(4,4'-dCF3bpy)]PF6": 0.35, "Ru(bpy)3(PF6)2": 0.25,
    "Ru(dtbbpy)3(PF6)2": 0.20,
}
NI_EFFECT = {
    "Ni(PPh3)2Cl2": 0.90, "NiOTf2": 0.60, "Ni(DME)Cl2": 0.45,
    "Ni(MeCN)Cl2": 0.40, "Ni(TMHD)2": 0.35,
}
LIGAND_EFFECT = {
    "4,4'-dimethoxy-2,2'-bpy": 0.90, "1,10-phenanthroline": 0.60,
    "bathophenanthroline": 0.50, "4,4'-di-tert-butyl-2,2'-bpy": 0.50,
    "4,4'-bis(CF3)-2,2'-bpy": 0.40, "2-(2-pyridyl)benzoxazole": 0.35,
    "2-methyl-1,10-phenanthroline": 0.40, "DMEDA": 0.25,
    "1,10-phenanthroline-5,6-dione": 0.20, "no ligand": 0.15,
}
AMINE_EFFECT = {
    "morpholine": 0.95, "piperidine": 0.75, "pyrrolidine": 0.60,
    "n-butylamine": 0.40, "cyclohexylamine": 0.35,
}


def art_proxy(x: dict) -> float:
    raw = (
        SOLVENT_EFFECT[x["solvent"]]
        * PC_EFFECT[x["photocatalyst"]]
        * NI_EFFECT[x["ni_precursor"]]
        * LIGAND_EFFECT[x["ni_ligand"]]
        * AMINE_EFFECT[x["amine"]]
    ) ** (1.0 / 5.0)
    if x["photocatalyst"] == "4CzIPN" and x["amine"] == "morpholine":
        raw *= 1.08
    if x["solvent"] in ("DMF", "NBP") and x["ni_precursor"] == "Ni(PPh3)2Cl2":
        raw *= 1.03
    return float(min(99.0, max(1.0, 99.0 * raw / ART_RAW_MAX)))


def art_best_raw() -> float:
    best = 0.0
    for solvent in SOLVENTS:
        for pc in PHOTOCATALYSTS:
            for ni in NI_PRECURSORS:
                for ligand in NI_LIGANDS:
                    for amine in AMINES:
                        raw = (
                            SOLVENT_EFFECT[solvent] * PC_EFFECT[pc] * NI_EFFECT[ni]
                            * LIGAND_EFFECT[ligand] * AMINE_EFFECT[amine]
                        ) ** 0.2
                        if pc == "4CzIPN" and amine == "morpholine":
                            raw *= 1.08
                        if solvent in ("DMF", "NBP") and ni == "Ni(PPh3)2Cl2":
                            raw *= 1.03
                        best = max(best, raw)
    return best


ART_RAW_MAX = art_best_raw()


def run_art(algorithm: str = "auto") -> dict:
    space = SearchSpace.from_dict(
        [
            {"name": "solvent", "kind": "categorical", "choices": SOLVENTS},
            {"name": "photocatalyst", "kind": "categorical", "choices": PHOTOCATALYSTS},
            {"name": "ni_precursor", "kind": "categorical", "choices": NI_PRECURSORS},
            {"name": "ni_ligand", "kind": "categorical", "choices": NI_LIGANDS},
            {"name": "amine", "kind": "categorical", "choices": AMINES},
        ]
    )
    objective = Objective(name="yield", direction="max")
    context = OptimizationContext.from_space(
        space,
        budget_total=120,
        batch_size=4,
        n_objectives=1,
        domain="chemistry",
        smoothness="mixed",
        noise="medium",
    )
    suffix = "" if algorithm == "auto" else f"_{algorithm}"
    run_dir = OUT / f"art_hte_angew2026{suffix}"
    if run_dir.exists():
        shutil.rmtree(run_dir)
    study = Study.create(
        run_dir,
        space,
        objective,
        context=context,
        algorithm=algorithm,
        auto_route=(algorithm == "auto"),
        budget=120,
        batch_size=4,
        seed=0,
        meta={
            "paper": "Combining HTE and Automated Flow for BO of a Metallaphotoredox Reaction",
            "doi": "10.1002/anie.3129364",
            "pmcid": "PMC13360563",
            "space": "5 categorical variables, 12x10x5x10x5 = 30,000",
            "real_data": "Table 1 levels + Table 2 Br top-5 conditions/yields",
            "proxy_note": "iterative yields use a trend-calibrated proxy, NOT measurements",
        },
    )
    # Warm start from four of the real Br conditions (hold out the 99% optimum).
    seeds = BR_TOP[2:6]
    study.import_observations(
        [{"x": {k: row[k] for k in space.names}, "y": row["yield"], "source": "paper_table2"} for row in seeds]
    )
    known_optimum = {k: BR_TOP[0][k] for k in space.names}
    proposed = []
    for _ in range(25):
        pending = study.ask(4)
        results = [{"id": item["id"], "y": art_proxy(item["x"])} for item in pending]
        study.tell(results)
        proposed.extend(pending)
    dataset = study.dataset.observations
    best = max(dataset, key=lambda obs: obs.y)
    optimum_hit = any(
        all(item["x"][k] == known_optimum[k] for k in space.names) for item in proposed
    )
    render_study(run_dir)
    return {
        "case": f"art_hte_angew2026_{algorithm}",
        "requested_algorithm": algorithm,
        "paper": "Angew. Chem. Int. Ed. 2026, 65, e3129364",
        "url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC13360563/",
        "space_size": 30000,
        "router_algorithm": study.algorithm,
        "router_params": study.params,
        "real_seed_points": len(seeds),
        "measured_seed_yields": [row["yield"] for row in seeds],
        "initial_best_measured_yield": max(row["yield"] for row in seeds),
        "iterations": 25,
        "proposals": len(proposed),
        "best_proxy_yield": round(float(best.y), 2),
        "best_condition": best.x,
        "proposed_known_best_condition": optimum_hit,
        "unique_proposed": len({tuple(sorted(item["x"].items())) for item in proposed}),
        "run_dir": str(run_dir),
    }


# ---------------------------------------------------------------------------
# Case 2: Thermoset p-SDL formulation (Chem. Mater. 2026)
# ---------------------------------------------------------------------------
MONOMERS = ["IDA", "IBOA", "BCOE", "BMA", "2-HEA"]
CROSSLINKERS = ["none", "PEGDA-250", "PEGDA-700", "DEGDMA", "ETPTA"]


def thermoset_proxy(x: dict) -> tuple[dict, float, float]:
    """Synthetic property oracle (documented in README)."""
    m1, m2, xl = x["monomer1"], x["monomer2"], x["crosslinker"]
    m2_pct, xl_pct = float(x["m2_pct"]), float(x["xl_pct"])
    if m2 == "none":
        m2_pct = 0.0
    if xl == "none":
        xl_pct = 0.0
    m1_pct = max(0.0, 100.0 - m2_pct - xl_pct)
    cross = xl_pct / 50.0
    # Young's modulus and UTS rise with cross-linker; elongation falls.
    e_mod = 1.0 + 6.0 * cross + (0.8 if m1 == "IBOA" else 0.0) + 0.3 * (m2_pct / 60.0)
    uts = 3.0 + 6.0 * cross + (0.8 if m1 == "BMA" else 0.0)
    elong = 120.0 - 70.0 * cross + (25.0 if "BCOE" in (m1, m2) else 0.0)
    cure_time = 4.0 + 4.0 * cross + 0.8 * (m2_pct / 60.0)
    delta_t = 25.0 + 28.0 * cross + 0.15 * m2_pct
    # Non-smooth "islands" mentioned in the paper: small phase ripples.
    ripple = 0.08 * np.sin(6.0 * xl_pct / 50.0) + 0.05 * np.cos(5.0 * m2_pct / 60.0)
    # Composite target score in [0, 1]-ish.
    hit = (
        np.exp(-((e_mod - 4.5) ** 2) / (2 * 2.0**2))
        * np.exp(-((uts - 7.5) ** 2) / (2 * 2.5**2))
        * np.exp(-((elong - 75.0) ** 2) / (2 * 25.0**2))
    )
    score = float(np.clip(hit + ripple, 0.0, 1.5))
    return (
        {"E_MPa": e_mod, "UTS_MPa": uts, "elongation_pct": elong,
         "cure_time_s": cure_time, "delta_T_C": delta_t, "m1_pct": m1_pct},
        score,
        cure_time,
    )


def run_thermoset() -> dict:
    space = SearchSpace.from_dict(
        [
            {"name": "monomer1", "kind": "categorical", "choices": MONOMERS},
            {"name": "monomer2", "kind": "categorical", "choices": ["none", *MONOMERS]},
            {"name": "crosslinker", "kind": "categorical", "choices": CROSSLINKERS},
            {"name": "m2_pct", "kind": "continuous", "low": 0.0, "high": 60.0},
            {"name": "xl_pct", "kind": "continuous", "low": 0.0, "high": 50.0},
        ]
    )
    objective = Objective(
        name="formulation_score",
        direction="max",
        constraints=("cure_time_s", "delta_T_C"),
        constraint_limits={
            "cure_time_s": {"op": "<=", "value": 10.0},
            "delta_T_C": {"op": "<=", "value": 70.0},
        },
    )
    context = OptimizationContext.from_space(
        space,
        budget_total=40,
        batch_size=4,
        n_objectives=1,
        n_constraints=2,
        domain="materials",
        smoothness="rough",
        noise="high",
    )
    run_dir = OUT / "thermoset_psdl_chemmater2026"
    if run_dir.exists():
        shutil.rmtree(run_dir)
    study = Study.create(
        run_dir,
        space,
        objective,
        context=context,
        budget=40,
        batch_size=4,
        seed=0,
        auto_route=True,
        meta={
            "paper": "Scalable Thermoset Informatics Integrated with Autonomous Experimentation",
            "doi": "10.1021/acs.chemmater.6c01600",
            "pmcid": "PMC13718556",
            "space": "5 monomers + 4 cross-linkers, <=3 components, cross-linker <=50 wt%",
            "real_data": "component library, composition rules, cure time <=10 s, dT <=70 C",
            "proxy_note": "target windows and property oracle are synthetic (article target table not in fetched text)",
        },
    )
    # Feasible warm start (representative formulations; scores from the proxy).
    seeds = [
        {"monomer1": "BCOE", "monomer2": "none", "crosslinker": "DEGDMA", "m2_pct": 0.0, "xl_pct": 12.6},
        {"monomer1": "IDA", "monomer2": "none", "crosslinker": "PEGDA-250", "m2_pct": 0.0, "xl_pct": 25.0},
        {"monomer1": "2-HEA", "monomer2": "none", "crosslinker": "PEGDA-700", "m2_pct": 0.0, "xl_pct": 20.0},
        {"monomer1": "IBOA", "monomer2": "BCOE", "crosslinker": "PEGDA-250", "m2_pct": 30.0, "xl_pct": 15.0},
        {"monomer1": "BMA", "monomer2": "2-HEA", "crosslinker": "ETPTA", "m2_pct": 25.0, "xl_pct": 10.0},
        {"monomer1": "IDA", "monomer2": "IBOA", "crosslinker": "PEGDA-700", "m2_pct": 35.0, "xl_pct": 20.0},
    ]
    study.import_observations(
        [
            {"x": seed, "y": thermoset_proxy(seed)[1], "constraints": {
                "cure_time_s": thermoset_proxy(seed)[2],
                "delta_T_C": thermoset_proxy(seed)[0]["delta_T_C"],
            }, "source": "synthetic_seed"}
            for seed in seeds
        ]
    )
    proposed = []
    for _ in range(8):
        pending = study.ask(4)
        results = []
        for item in pending:
            props, score, cure = thermoset_proxy(item["x"])
            results.append({
                "id": item["id"], "y": score,
                "constraints": {"cure_time_s": cure, "delta_T_C": props["delta_T_C"]},
            })
        study.tell(results)
        proposed.extend(pending)
    dataset = study.dataset.observations
    feasible = [
        obs for obs in dataset
        if obs.constraints.get("cure_time_s", 99) <= 10.0
        and obs.constraints.get("delta_T_C", 999) <= 70.0
    ]
    best = max(feasible, key=lambda obs: obs.y) if feasible else None
    render_study(run_dir)
    return {
        "case": "thermoset_psdl_chemmater2026",
        "paper": "Chem. Mater. 2026, 38, 10562-10575",
        "url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC13718556/",
        "router_algorithm": study.algorithm,
        "router_params": study.params,
        "seeds": len(seeds),
        "iterations": 8,
        "proposals": len(proposed),
        "feasible_observations": len(feasible),
        "best_feasible_score": None if best is None else round(float(best.y), 4),
        "best_feasible_condition": None if best is None else best.x,
        "best_feasible_constraints": None if best is None else best.constraints,
        "run_dir": str(run_dir),
    }


def main() -> int:
    results = [run_art("auto"), run_art("bayes"), run_thermoset()]
    path = Path(__file__).resolve().parent / "results.json"
    path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    for item in results:
        print("\n==", item["case"], "==")
        for key, value in item.items():
            if key in ("run_dir", "url"):
                continue
            print(f"{key}: {value}")
    print("\nwritten:", path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
