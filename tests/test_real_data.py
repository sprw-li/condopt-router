import os
import csv
import json
from pathlib import Path

def _utf8_env() -> dict:
    """Child environment that forces UTF-8 stdio regardless of the console codepage."""
    return {**os.environ, "PYTHONIOENCODING": "utf-8"}



ROOT = Path(__file__).resolve().parents[1]


def test_real_literature_data_round_trip(tmp_path, capsys):
    """Acceptance: measured data from a real paper drives the pipeline.

    Fixture ``examples/paper_campaigns/art_table2_real.csv`` holds the six
    measured Br-campaign conditions/yields transcribed from the ART
    metallaphotoredox HTE paper (Angew. Chem. Int. Ed. 2026, Table 2).  No
    proxy oracle is involved: the CSV is imported through the public CLI and
    the reported best condition must be the measured 99% entry.
    """
    from condopt.base import Objective
    from condopt.cli import main
    from condopt.space import SearchSpace
    from condopt.study import Study

    rows = list(csv.DictReader(
        (ROOT / "examples/paper_campaigns/art_table2_real.csv").read_text(
            encoding="utf-8"
        ).splitlines()
    ))
    assert len(rows) == 6

    space = SearchSpace.from_dict([
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
    study = Study.create(
        tmp_path / "art_real",
        space,
        Objective(name="yield", direction="max"),
        algorithm="tpe",
        budget=16,
        batch_size=4,
        seed=0,
    )
    csv_path = ROOT / "examples/paper_campaigns/art_table2_real.csv"
    assert main(["import", "--run", str(study.dir), "--results", str(csv_path), "--json"]) == 0
    capsys.readouterr()
    assert len(study.dataset.observations) == 6
    assert all(obs.source == "literature-angew2026-table2" for obs in study.dataset.observations)

    best = study.best()
    assert best is not None
    assert best.y == 99.0
    assert best.x["solvent"] == "DMF"
    assert best.x["photocatalyst"] == "4CzIPN"
    assert best.x["ni_precursor"] == "Ni(PPh3)2Cl2"
    assert best.x["ni_ligand"] == "4,4'-dimethoxy-2,2'-bpy"
    assert best.x["amine"] == "morpholine"

    summary = study.summary()
    assert summary["best"]["y"] == 99.0
    decision = study.decision()
    assert decision["selection_basis"] == "raw"
    assert study.recommend()["condition"] == best.x
    assert main(["digest", "--run", str(study.dir), "--format", "html", "--json"]) == 0
    capsys.readouterr()
    assert (study.dir / "digest.html").exists()

def test_real_data_loo_script_runs():
    import importlib.util
    import json
    from pathlib import Path

    script = Path(__file__).resolve().parents[1] / "scripts" / "real_data_loo.py"
    spec = importlib.util.spec_from_file_location("real_data_loo", script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    import subprocess
    import sys

    result = subprocess.run(
        [sys.executable, str(script)], capture_output=True, text=True, encoding="utf-8", errors="replace",
        env=_utf8_env()
    )
    assert result.returncode == 0, result.stderr
    summary = json.loads(result.stdout.splitlines()[0])
    assert summary["n"] == 6
    assert summary["mae"] >= 0.0

def test_real_data_loo_accepts_descriptor_table():
    import importlib.util
    import json
    import subprocess
    import sys
    from pathlib import Path

    script = Path(__file__).resolve().parents[1] / "scripts" / "real_data_loo.py"
    spec = importlib.util.spec_from_file_location("real_data_loo_desc", script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    descriptor_file = (
        Path(__file__).resolve().parents[1]
        / "examples/paper_campaigns/art_solvent_descriptors.json"
    )
    space = module.build_space(str(descriptor_file))
    assert space.variable("solvent").descriptors is not None
    assert space.feature_dim < module.build_space().feature_dim
    result = subprocess.run(
        [sys.executable, str(script), "--descriptors", str(descriptor_file),
         "--out", str(Path(__file__).resolve().parents[1] / "runs" / "loo_desc_test" / "r.json")],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
        env=_utf8_env(),
    )
    assert result.returncode == 0, result.stderr
    summary = json.loads(result.stdout.splitlines()[0])
    assert summary["descriptors"] is not None
    assert summary["feature_dim"] < 47

def test_real_prior_transfer_example_runs():
    import importlib.util
    from pathlib import Path

    script = (
        Path(__file__).resolve().parents[1]
        / "examples/paper_campaigns/run_real_prior_transfer.py"
    )
    spec = importlib.util.spec_from_file_location("real_prior_transfer", script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    row = module.run_arm(True, seed=0, total_budget=8, cold_first=3)
    assert row["arm"] == "warm"
    assert row["audit"]["ready"] is True
    assert row["transferred"] >= 1
    assert row["proxy_best"] is not None

def test_real_prior_split_script_separates_warm_from_cold(tmp_path):
    import importlib.util
    import sys
    from pathlib import Path

    script = Path(__file__).resolve().parents[1] / "scripts" / "benchmark_real_prior_splits.py"
    spec = importlib.util.spec_from_file_location("benchmark_real_prior_splits", script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module.OUT = tmp_path / "splits"
    rows = module.run_settings(seeds=[0], cold_firsts=[3], budget=12)
    assert len(rows) == 2
    cold = next(row for row in rows if row["arm"] == "cold")
    warm = next(row for row in rows if row["arm"] == "warm")
    assert cold["transferred"] == 0
    assert warm["transferred"] == 3
    assert warm["proxy_best"] > cold["proxy_best"]

def test_real_audit_splits_script_flags_low_confidence(tmp_path):
    import importlib.util
    import itertools
    import sys
    from pathlib import Path

    script = Path(__file__).resolve().parents[1] / "scripts" / "benchmark_real_audit_splits.py"
    spec = importlib.util.spec_from_file_location("benchmark_real_audit_splits", script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module.OUT = tmp_path / "audit_splits"

    rows = module.load_rows()
    space = module.build_space(rows)
    splits = list(itertools.combinations(range(len(rows)), 3))[:3]
    results = [
        module.audit_split(
            index,
            [rows[i] for i in combo],
            [rows[i] for i in range(len(rows)) if i not in combo],
            space,
        )
        for index, combo in enumerate(splits)
    ]
    assert len(results) == 3
    assert all(row["ready"] for row in results)
    assert all(row["confidence"] == "low" for row in results)  # n=3 must be flagged
    assert all(row["n_compared"] == 3 for row in results)

