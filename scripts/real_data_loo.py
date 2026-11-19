"""Leave-one-out validation of the surrogate on *real* measured data only.

Uses ``examples/paper_campaigns/art_table2_real.csv`` (six measured ART
metallaphotoredox conditions/yields from Angew. Chem. Int. Ed. 2026, Table 2).
For every held-out condition the GP is fitted on the other five and asked to
predict it; no proxy oracle is involved.

Run:
    py -3 scripts/real_data_loo.py
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from condopt.base import Objective  # noqa: E402
from condopt.space import SearchSpace  # noqa: E402
from condopt.study import Study  # noqa: E402

DEFAULT_CSV = ROOT / "examples" / "paper_campaigns" / "art_table2_real.csv"


def build_space(descriptor_file: str | None = None) -> SearchSpace:
    solvent = {"name": "solvent", "kind": "categorical",
               "choices": ["DMF", "NBP", "Rhodiasolv Polarclean", "Acetonitrile", "Toluene",
                           "Propylene carbonate", "Ethanol", "tert-amyl alcohol", "Anisole",
                           "Ethyl acetate", "Isopropanol", "2-MeTHF"]}
    if descriptor_file:
        payload = json.loads(Path(descriptor_file).read_text(encoding="utf-8"))
        solvent["descriptors"] = payload["descriptors"]
        solvent["descriptor_names"] = payload.get("_descriptor_names", [])
    return SearchSpace.from_dict([
        solvent,
        {"name": "photocatalyst", "kind": "categorical",
         "choices": ["(Ir[dF(CF3)ppy]2(dtbpy))PF6", "(Ir[dF(CF3)ppy]2(bpy))PF6", "4CzIPN",
                     "4DPAIPN", "3CzClIPN", "PheMesAcri.BF4", "PheMe2MesAcri.BF4",
                     "Ru(bpy)3(PF6)2", "[Ir(dF(Me)ppy)2(4,4'-dCF3bpy)]PF6",
                     "Ru(dtbbpy)3(PF6)2"]},
        {"name": "ni_precursor", "kind": "categorical",
         "choices": ["Ni(PPh3)2Cl2", "Ni(DME)Cl2", "Ni(TMHD)2", "Ni(MeCN)Cl2", "NiOTf2"]},
        {"name": "ni_ligand", "kind": "categorical",
         "choices": ["4,4'-dimethoxy-2,2'-bpy", "1,10-phenanthroline",
                     "bathophenanthroline", "4,4'-di-tert-butyl-2,2'-bpy",
                     "4,4'-bis(CF3)-2,2'-bpy", "DMEDA", "2-(2-pyridyl)benzoxazole",
                     "2-methyl-1,10-phenanthroline", "1,10-phenanthroline-5,6-dione",
                     "no ligand"]},
        {"name": "amine", "kind": "categorical",
         "choices": ["pyrrolidine", "piperidine", "morpholine", "n-butylamine",
                     "cyclohexylamine"]},
    ])


def _legacy_build_space() -> SearchSpace:
    return SearchSpace.from_dict([
        {"name": "solvent", "kind": "categorical",
         "choices": ["DMF", "NBP", "Rhodiasolv Polarclean", "Acetonitrile", "Toluene",
                     "Propylene carbonate", "Ethanol", "tert-amyl alcohol", "Anisole",
                     "Ethyl acetate", "Isopropanol", "2-MeTHF"]},
        {"name": "photocatalyst", "kind": "categorical",
         "choices": ["(Ir[dF(CF3)ppy]2(dtbpy))PF6", "(Ir[dF(CF3)ppy]2(bpy))PF6", "4CzIPN",
                     "4DPAIPN", "3CzClIPN", "PheMesAcri.BF4", "PheMe2MesAcri.BF4",
                     "Ru(bpy)3(PF6)2", "[Ir(dF(Me)ppy)2(4,4'-dCF3bpy)]PF6",
                     "Ru(dtbbpy)3(PF6)2"]},
        {"name": "ni_precursor", "kind": "categorical",
         "choices": ["Ni(PPh3)2Cl2", "Ni(DME)Cl2", "Ni(TMHD)2", "Ni(MeCN)Cl2", "NiOTf2"]},
        {"name": "ni_ligand", "kind": "categorical",
         "choices": ["4,4'-dimethoxy-2,2'-bpy", "1,10-phenanthroline",
                     "bathophenanthroline", "4,4'-di-tert-butyl-2,2'-bpy",
                     "4,4'-bis(CF3)-2,2'-bpy", "DMEDA", "2-(2-pyridyl)benzoxazole",
                     "2-methyl-1,10-phenanthroline", "1,10-phenanthroline-5,6-dione",
                     "no ligand"]},
        {"name": "amine", "kind": "categorical",
         "choices": ["pyrrolidine", "piperidine", "morpholine", "n-butylamine",
                     "cyclohexylamine"]},
    ])


def spearman(left: list[float], right: list[float]) -> float | None:
    import numpy as np

    if len(left) < 3:
        return None
    a = np.argsort(np.argsort(left))
    b = np.argsort(np.argsort(right))
    if float(np.std(a)) < 1e-12 or float(np.std(b)) < 1e-12:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", default=str(DEFAULT_CSV))
    parser.add_argument(
        "--descriptors",
        default=None,
        help="可选溶剂描述符 JSON（如 examples/paper_campaigns/art_solvent_descriptors.json）",
    )
    parser.add_argument("--out", default=str(ROOT / "runs" / "real_data_loo" / "results.json"))
    args = parser.parse_args()

    rows = list(csv.DictReader(Path(args.csv).read_text(encoding="utf-8").splitlines()))
    space = build_space(args.descriptors)
    actual, predicted = [], []
    records = []
    for index, row in enumerate(rows):
        study = Study.create(
            Path(args.out).parent / f"fold_{index}",
            space,
            Objective(name="yield", direction="max", constraints=()),
            algorithm="tpe",
            budget=8,
            batch_size=4,
            seed=0,
        )
        training = [
            {"x": {name: item[name] for name in space.names}, "y": float(item["yield"])}
            for position, item in enumerate(rows) if position != index
        ]
        study.import_observations(training)
        condition = {name: row[name] for name in space.names}
        prediction = study.predict(condition)["predictions"]["yield"]["mean"]
        truth = float(row["yield"])
        actual.append(truth)
        predicted.append(prediction)
        records.append({
            "held_out_yield": truth,
            "predicted_yield": round(prediction, 3),
            "condition": condition,
        })
    mae = sum(abs(a - p) for a, p in zip(actual, predicted)) / len(actual)
    rho = spearman(actual, predicted)
    best_actual = max(range(len(actual)), key=lambda i: actual[i])
    best_predicted = max(range(len(predicted)), key=lambda i: predicted[i])
    summary = {
        "n": len(actual),
        "descriptors": args.descriptors,
        "feature_dim": space.feature_dim,
        "mae": round(mae, 3),
        "spearman": None if rho is None else round(rho, 3),
        "true_best_is_predicted_best": best_actual == best_predicted,
        "actual": actual,
        "predicted": [round(value, 3) for value in predicted],
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps({"summary": summary, "folds": records}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False))
    print(f"written: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())