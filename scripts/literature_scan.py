"""Reusable literature-scan pipeline for router experience.

Stages (all cacheable / resumable):

    fetch   Europe PMC (2024-2026, open access) -> research/literature_scan/raw/*.json
            -> corpus.jsonl (deduplicated)
    extract rule-based paper features          -> papers.jsonl + papers.csv
    chunk   split papers into agent-sized JSONL -> chunks/chunk_XX.jsonl
    report  aggregate statistics               -> AGGREGATE.md + experience.json
    all     fetch -> extract -> chunk -> report

Design goals
------------
* No third-party dependencies (stdlib urllib/json/csv only).
* Config-driven: queries, date window, page limits and output dir live in JSON.
* Resumable: raw/<query>.json + corpus.jsonl are reused unless --refresh.
* Auditable: every stage writes a manifest with parameters, counts and timestamps.
* Reusable: ``--config`` swaps campaigns; ``literature_scan.py all`` is the
  single entry point; downstream agents only read ``chunks/*.jsonl``.

Usage
-----
    py -3 scripts/literature_scan.py fetch  --config configs/literature_scan.json
    py -3 scripts/literature_scan.py extract --config configs/literature_scan.json
    py -3 scripts/literature_scan.py chunk   --config configs/literature_scan.json
    py -3 scripts/literature_scan.py report  --config configs/literature_scan.json
    py -3 scripts/literature_scan.py all     --config configs/literature_scan.json
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "literature_scan.json"
SCHEMA_VERSION = 1
API = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"

DEFAULT_QUERIES = [
    "self-driving laboratory", "self-driving lab", "autonomous laboratory",
    "autonomous experimentation", "closed-loop optimization chemistry",
    "self-optimizing flow chemistry", "high-throughput experimentation Bayesian optimization",
    "high-throughput experimentation reaction optimization", "Bayesian optimization reaction conditions",
    "Bayesian optimization cross-coupling", "Bayesian optimization photocatalysis",
    "Bayesian optimization electrocatalysis", "Bayesian optimization flow reactor",
    "Bayesian optimization crystallization", "Bayesian optimization materials",
    "multi-objective Bayesian optimization chemistry", "constrained Bayesian optimization chemistry",
    "multi-fidelity Bayesian optimization chemistry", "active learning reaction optimization",
    "design of experiments reaction optimization", "robotic chemistry optimization",
    "automated synthesis optimization", "self-driving laboratory nanoparticles",
    "self-driving laboratory formulation", "self-driving laboratory crystallization",
    "self-driving laboratory electrochemistry", "high-throughput experimentation photocatalysis",
    "high-throughput experimentation cross-coupling", "machine learning reaction yield optimization",
    "sequential model-based optimization chemistry", "Bayesian optimization process development",
    "Bayesian optimization polymer formulation", "Bayesian optimization electrolyte",
    "autonomous experimentation catalysis", "Bayesian optimization heterogeneous catalysis",
    "Bayesian optimization organic synthesis", "autonomous laboratory materials",
    "self-driving laboratory inorganic", "Bayesian optimization multi-fidelity",
    "Bayesian optimization constraints",
]

ALGORITHM_PATTERNS: list[tuple[str, str]] = [
    ("bayesian optimization", "Bayesian optimization"),
    ("bayes", "Bayesian optimization"),
    ("expected improvement", "EI / Bayesian optimization"),
    ("upper confidence bound", "UCB"),
    ("gaussian process", "Gaussian process"),
    ("tree-structured parzen", "TPE"),
    ("tpe", "TPE"),
    ("parzen", "TPE"),
    ("parego", "ParEGO"),
    ("qehvi", "qEHVI"),
    ("qnehvi", "qNEHVI"),
    ("design of experiments", "DoE"),
    ("response surface", "RSM"),
    ("central composite", "RSM"),
    ("box-behnken", "RSM"),
    ("factorial design", "Factorial design"),
    ("orthogonal", "Orthogonal design"),
    ("high-throughput experimentation", "HTE"),
    ("genetic algorithm", "Genetic algorithm"),
    ("particle swarm", "PSO"),
    ("simulated annealing", "Simulated annealing"),
    ("active learning", "Active learning"),
    ("reinforcement learning", "Reinforcement learning"),
    ("random forest", "Random forest"),
    ("gradient boosting", "Gradient boosting"),
    ("xgboost", "Gradient boosting"),
    ("neural network", "Neural network"),
    ("sobol", "Sobol"),
    ("latin hypercube", "LHS"),
    ("multi-objective", "Multi-objective"),
    ("multiobjective", "Multi-objective"),
    ("pareto", "Pareto"),
]

AUTOMATION_PATTERNS: list[tuple[str, str]] = [
    ("self-driving lab", "self-driving lab"),
    ("self-driving laboratory", "self-driving lab"),
    ("autonomous experimentation", "autonomous experimentation"),
    ("autonomous laboratory", "autonomous lab"),
    ("closed-loop", "closed loop"),
    ("self-optimizing", "self-optimizing"),
    ("automated flow", "flow"),
    ("flow reactor", "flow"),
    ("continuous flow", "flow"),
    ("robotic", "robotic"),
    ("high-throughput experimentation", "HTE"),
    ("active learning", "active learning"),
]

CONSTRAINT_PATTERNS = [
    "constraint", "constrained", "feasibility", "feasible", "safety", "upper limit",
    "lower limit", "must be", "should not exceed", "<= ", "≥", "temperature rise",
    "curing time", "within the range", "operating window",
]
NOISE_PATTERNS = [
    "noise", "noisy", "replicate", "replicates", "standard deviation", "error bars",
    "measurement error", "reproducibility", "uncertainty",
]
MOO_PATTERNS = ["multi-objective", "multiobjective", "pareto", "trade-off", "tradeoff"]
DOMAIN_PATTERNS = {
    "chemistry": ["reaction", "catalyst", "synthesis", "solvent", "yield", "coupling", "photoredox"],
    "materials": ["material", "polymer", "formulation", "alloy", "electrolyte", "nanoparticle", "resin"],
    "biology": ["protein", "cell", "enzyme", "biological", "biocatalysis"],
}

WORD_NUM = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
}

_NUM = r"(\d+|" + "|".join(WORD_NUM) + r")"
SPACE_PATTERNS = [
    (re.compile(rf"{_NUM}\s+(continuous|categorical|discrete|integer|ordinal)\s+(?:variables|parameters|factors|inputs|dimensions)", re.I), "typed"),
    (re.compile(rf"{_NUM}\s+(solvents?|ligands?|catalysts?|photocatalysts?|monomers?|cross-?linkers?|bases?|additives?|amines?)", re.I), "categories"),
    (re.compile(rf"{_NUM}\s+(?:reaction\s+)?conditions", re.I), "conditions"),
]


def _num(token: str) -> int | None:
    token = token.lower()
    if token.isdigit():
        return int(token)
    return WORD_NUM.get(token)


def load_config(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    config = {
        "output_dir": "research/literature_scan",
        "since": "2024-01-01",
        "until": "2026-12-31",
        "open_access": True,
        "page_size": 100,
        "max_per_query": 60,
        "sleep_seconds": 0.2,
        "chunk_size": 60,
        "queries": DEFAULT_QUERIES,
    }
    config.update(payload or {})
    config["output_dir"] = Path(config["output_dir"])
    if not config["output_dir"].is_absolute():
        config["output_dir"] = ROOT / config["output_dir"]
    return config


def _slug(query: str, limit: int = 80) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", query.lower()).strip("_")
    return slug[:limit] or "query"


def _build_query_url(query: str, config: dict, cursor: str = "*") -> str:
    parts = [f"({query})"]
    if config.get("open_access", True):
        parts.append("OPEN_ACCESS:y")
    since, until = config.get("since"), config.get("until")
    if since and until:
        parts.append(f"FIRST_PDATE:[{since} TO {until}]")
    params = {
        "query": " AND ".join(parts),
        "format": "json",
        "resultType": "core",
        "pageSize": str(int(config.get("page_size", 100))),
        "cursorMark": cursor,
    }
    return API + "?" + urllib.parse.urlencode(params)


def _http_json(url: str, retries: int = 3) -> dict:
    last_error = None
    for attempt in range(retries):
        try:
            request = urllib.request.Request(
                url, headers={"User-Agent": "condopt-literature-scan/1.0"}
            )
            with urllib.request.urlopen(request, timeout=120) as handle:
                return json.load(handle)
        except Exception as exc:  # noqa: BLE001 - network retries
            last_error = exc
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"request failed after {retries} attempts: {last_error}")


def _record_from_api(item: dict, query: str) -> dict:
    doi = (item.get("doi") or "").strip().lower()
    pmid = str(item.get("pmid") or "").strip()
    paper_id = doi or (f"pmid:{pmid}" if pmid else "")
    abstract = re.sub(r"\s+", " ", item.get("abstractText") or "").strip()[:3000]
    title = re.sub(r"\s+", " ", item.get("title") or "").strip()
    if not paper_id:
        paper_id = "title:" + re.sub(r"[^a-z0-9]+", "-", title.lower())[:120]
    return {
        "paper_id": paper_id,
        "title": title,
        "year": int(item["pubYear"]) if str(item.get("pubYear", "")).isdigit() else None,
        "journal": item.get("journalTitle") or "",
        "doi": doi,
        "pmid": pmid,
        "pmcid": item.get("pmcid") or "",
        "abstract": abstract,
        "authors": item.get("authorString") or "",
        "query_tags": [query],
    }


def fetch(config: dict, refresh: bool = False, queries: list[str] | None = None) -> dict:
    out_dir = config["output_dir"]
    raw_dir = out_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    corpus_path = out_dir / "corpus.jsonl"
    seen: set[str] = set()
    if corpus_path.exists():
        for line in corpus_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                seen.add(json.loads(line)["paper_id"])
    queries = queries or list(config["queries"])
    manifest = []
    fetched_new = 0
    for query in queries:
        slug = _slug(query)
        raw_path = raw_dir / f"{slug}.json"
        if raw_path.exists() and not refresh:
            payload = json.loads(raw_path.read_text(encoding="utf-8"))
            records = payload.get("records", [])
            status = "cached"
        else:
            records = []
            cursor = "*"
            pages = 0
            while len(records) < int(config["max_per_query"]) and pages < 5:
                data = _http_json(_build_query_url(query, config, cursor))
                items = data.get("resultList", {}).get("result", [])
                if not items:
                    break
                for item in items:
                    rec = _record_from_api(item, query)
                    if rec["title"]:
                        records.append(rec)
                fetched_new += len(items)
                pages += 1
                cursor = data.get("nextCursorMark") or ""
                if not cursor:
                    break
                time.sleep(float(config.get("sleep_seconds", 0.2)))
            records = records[: int(config["max_per_query"])]
            raw_path.write_text(
                json.dumps(
                    {"query": query, "fetched_at": _now(), "records": records},
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            status = "fetched"
            time.sleep(float(config.get("sleep_seconds", 0.2)))
        manifest.append({"query": query, "slug": slug, "status": status, "records": len(records)})
        with corpus_path.open("a", encoding="utf-8") as handle:
            for rec in records:
                if rec["paper_id"] in seen:
                    continue
                seen.add(rec["paper_id"])
                handle.write(json.dumps(rec, ensure_ascii=False) + "\n")
    summary = {
        "schema_version": SCHEMA_VERSION,
        "stage": "fetch",
        "finished_at": _now(),
        "queries": len(queries),
        "api_items_requested_new": fetched_new,
        "corpus_unique": len(seen),
        "queries_status": manifest,
    }
    _write_json(out_dir / "fetch_manifest.json", summary)
    return summary


def extract_features(record: dict) -> dict:
    text = f"{record.get('title', '')} {record.get('abstract', '')}".lower()
    algorithm_terms: list[str] = []
    for pattern, label in ALGORITHM_PATTERNS:
        if pattern in text and label not in algorithm_terms:
            algorithm_terms.append(label)
    automation = []
    for pattern, label in AUTOMATION_PATTERNS:
        if pattern in text and label not in automation:
            automation.append(label)
    categorical = None
    continuous = None
    for regex, kind in SPACE_PATTERNS:
        for match in regex.finditer(text):
            value = _num(match.group(1))
            if value is None:
                continue
            if kind == "typed":
                if match.group(2).lower() == "continuous":
                    continuous = (continuous or 0) + value
                else:
                    categorical = (categorical or 0) + value
            elif kind == "categories":
                categorical = (categorical or 0) + value
    n_objectives = None
    moo = any(pattern in text for pattern in MOO_PATTERNS)
    if moo:
        n_objectives = 2
    for token, count in (("two objectives", 2), ("three objectives", 3), ("four objectives", 4)):
        if token in text:
            n_objectives = count
    multi_objective = bool(moo or (n_objectives and n_objectives >= 2))
    constraints = any(pattern in text for pattern in CONSTRAINT_PATTERNS)
    noisy = any(pattern in text for pattern in NOISE_PATTERNS)
    domains = [name for name, keys in DOMAIN_PATTERNS.items() if any(key in text for key in keys)]
    domain = domains[0] if domains else None
    evidence = _evidence_sentences(record.get("abstract", ""), text)
    return {
        "paper_id": record["paper_id"],
        "title": record.get("title", ""),
        "year": record.get("year"),
        "journal": record.get("journal", ""),
        "doi": record.get("doi", ""),
        "algorithm_terms": algorithm_terms,
        "automation_tags": automation,
        "domain": domain,
        "space": {"categorical": categorical, "continuous": continuous},
        "multi_objective": multi_objective,
        "n_objectives": n_objectives,
        "constraints": constraints,
        "noisy": noisy,
        "abstract": record.get("abstract", "")[:1500],
        "abstract_chars": len(record.get("abstract", "")),
        "evidence": evidence,
        "query_tags": record.get("query_tags", []),
    }


def _evidence_sentences(abstract: str, lowered: str) -> list[str]:
    sentences = re.split(r"(?<=[.!?])\s+", abstract)
    hits = []
    keys = ["bayesian", "self-driving", "autonomous", "high-throughput", "multi-objective",
            "constraint", "design of experiments", "active learning", "flow"]
    for sentence in sentences:
        low = sentence.lower()
        if any(key in low for key in keys):
            hits.append(re.sub(r"\s+", " ", sentence).strip()[:220])
        if len(hits) >= 2:
            break
    return hits


def extract(config: dict) -> dict:
    out_dir = config["output_dir"]
    corpus = _read_jsonl(out_dir / "corpus.jsonl")
    papers = [extract_features(record) for record in corpus]
    _write_jsonl(out_dir / "papers.jsonl", papers)
    csv_fields = ["paper_id", "year", "journal", "doi", "title", "algorithm_terms",
                  "automation_tags", "domain", "space", "multi_objective", "n_objectives",
                  "constraints", "noisy"]
    with (out_dir / "papers.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=csv_fields)
        writer.writeheader()
        for paper in papers:
            row = {key: paper.get(key) for key in csv_fields}
            row["algorithm_terms"] = ";".join(paper["algorithm_terms"])
            row["automation_tags"] = ";".join(paper["automation_tags"])
            row["space"] = json.dumps(paper["space"], ensure_ascii=False)
            writer.writerow(row)
    summary = {
        "stage": "extract",
        "finished_at": _now(),
        "papers": len(papers),
        "with_algorithm_terms": sum(1 for p in papers if p["algorithm_terms"]),
        "with_automation_tags": sum(1 for p in papers if p["automation_tags"]),
        "with_categorical_space": sum(1 for p in papers if p["space"]["categorical"]),
        "multi_objective": sum(1 for p in papers if p["multi_objective"]),
        "constraints": sum(1 for p in papers if p["constraints"]),
        "noisy": sum(1 for p in papers if p["noisy"]),
    }
    _write_json(out_dir / "extract_manifest.json", summary)
    return summary


def chunk(config: dict) -> dict:
    out_dir = config["output_dir"]
    papers = _read_jsonl(out_dir / "papers.jsonl")
    chunks_dir = out_dir / "chunks"
    chunks_dir.mkdir(parents=True, exist_ok=True)
    for old in chunks_dir.glob("chunk_*.jsonl"):
        old.unlink()
    size = int(config.get("chunk_size", 60))
    files = []
    for index in range(0, len(papers), size):
        slice_ = papers[index:index + size]
        path = chunks_dir / f"chunk_{index // size:02d}.jsonl"
        _write_jsonl(path, slice_)
        try:
            rel_path = str(path.relative_to(ROOT))
        except ValueError:
            rel_path = str(path)
        files.append({"file": rel_path, "papers": len(slice_),
                      "first": slice_[0]["paper_id"], "last": slice_[-1]["paper_id"]})
    index_payload = {
        "stage": "chunk",
        "finished_at": _now(),
        "chunk_size": size,
        "n_chunks": len(files),
        "n_papers": len(papers),
        "chunks": files,
    }
    _write_json(out_dir / "chunks" / "index.json", index_payload)
    return index_payload


def aggregate(papers: list[dict]) -> dict:
    def count(key: str) -> dict:
        result: dict[str, int] = {}
        for paper in papers:
            for value in paper.get(key, []) or []:
                result[value] = result.get(value, 0) + 1
        return dict(sorted(result.items(), key=lambda item: item[1], reverse=True))

    years: dict[str, int] = {}
    for paper in papers:
        year = str(paper.get("year") or "unknown")
        years[year] = years.get(year, 0) + 1
    algorithms = count("algorithm_terms")
    automation = count("automation_tags")
    domains: dict[str, int] = {}
    for paper in papers:
        if paper.get("domain"):
            domains[paper["domain"]] = domains.get(paper["domain"], 0) + 1

    def subset(**criteria) -> list[dict]:
        selected = []
        for paper in papers:
            ok = True
            for key, value in criteria.items():
                if key == "categorical_min":
                    if not (paper.get("space", {}).get("categorical") or 0) >= value:
                        ok = False
                elif key == "continuous_min":
                    if not (paper.get("space", {}).get("continuous") or 0) >= value:
                        ok = False
                elif key == "algorithm":
                    if value not in paper.get("algorithm_terms", []):
                        ok = False
                elif paper.get(key) != value:
                    ok = False
            if ok:
                selected.append(paper)
        return selected

    categorical_heavy = subset(categorical_min=3)
    continuous_heavy = subset(continuous_min=3)
    constrained = subset(constraints=True)
    multi = subset(multi_objective=True)
    autonomous = [p for p in papers if p.get("automation_tags")]

    def algo_mix(selected: list[dict]) -> dict:
        return count_from(selected, "algorithm_terms") if selected else {}

    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": _now(),
        "n_papers": len(papers),
        "years": dict(sorted(years.items())),
        "algorithms": algorithms,
        "automation": automation,
        "domains": dict(sorted(domains.items(), key=lambda item: item[1], reverse=True)),
        "subsets": {
            "categorical_heavy": {"n": len(categorical_heavy), "algorithms": algo_mix(categorical_heavy)},
            "continuous_heavy": {"n": len(continuous_heavy), "algorithms": algo_mix(continuous_heavy)},
            "autonomous": {"n": len(autonomous), "algorithms": algo_mix(autonomous)},
            "multi_objective": {"n": len(multi), "algorithms": algo_mix(multi)},
            "constraints": {"n": len(constrained), "algorithms": algo_mix(constrained)},
        },
        "coverage": {
            "algorithm_terms": sum(1 for p in papers if p.get("algorithm_terms")) / max(len(papers), 1),
            "automation_tags": sum(1 for p in papers if p.get("automation_tags")) / max(len(papers), 1),
            "abstract": sum(1 for p in papers if p.get("abstract_chars", 0) > 0) / max(len(papers), 1),
        },
    }


def count_from(papers: list[dict], key: str) -> dict:
    result: dict[str, int] = {}
    for paper in papers:
        for value in paper.get(key, []) or []:
            result[value] = result.get(value, 0) + 1
    return dict(sorted(result.items(), key=lambda item: item[1], reverse=True))


def report(config: dict) -> dict:
    out_dir = config["output_dir"]
    papers = _read_jsonl(out_dir / "papers.jsonl")
    experience = aggregate(papers)
    _write_json(out_dir / "experience.json", experience)
    lines = [
        "# Literature scan — router experience",
        "",
        f"Generated: {experience['generated_at']}",
        f"Papers: {experience['n_papers']}",
        "",
        "## Coverage",
        "",
        *[f"- {key}: {value:.1%}" for key, value in experience["coverage"].items()],
        "",
        "## Algorithm mentions",
        "",
        *[f"- {key}: {value}" for key, value in experience["algorithms"].items()],
        "",
        "## Automation mentions",
        "",
        *[f"- {key}: {value}" for key, value in experience["automation"].items()],
        "",
        "## Subsets",
    ]
    for name, subset in experience["subsets"].items():
        lines.append(f"### {name} (n={subset['n']})")
        lines.append("")
        if subset["algorithms"]:
            for key, value in subset["algorithms"].items():
                lines.append(f"- {key}: {value}")
        else:
            lines.append("- (no algorithm terms extracted)")
        lines.append("")
    lines += [
        "## Reproduce",
        "",
        "```powershell",
        "py -3 scripts/literature_scan.py all --config configs/literature_scan.json",
        "```",
        "",
        "Caveat: rule-based extraction from titles/abstracts only; counts are",
        "mentions, not experimental superiority, and abstracts often omit details.",
    ]
    (out_dir / "AGGREGATE.md").write_text("\n".join(lines), encoding="utf-8")
    return experience


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["fetch", "extract", "chunk", "report", "all"])
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--refresh", action="store_true", help="refetch raw query files")
    parser.add_argument("--queries", nargs="*", help="override query list")
    args = parser.parse_args()
    config = load_config(Path(args.config))
    if args.command == "fetch":
        result = fetch(config, refresh=args.refresh, queries=args.queries)
    elif args.command == "extract":
        result = extract(config)
    elif args.command == "chunk":
        result = chunk(config)
    elif args.command == "report":
        result = report(config)
    else:
        fetch_summary = fetch(config, refresh=args.refresh, queries=args.queries)
        extract_summary = extract(config)
        chunk_summary = chunk(config)
        report_summary = report(config)
        result = {
            "fetch": fetch_summary,
            "extract": extract_summary,
            "chunk": chunk_summary,
            "report": {key: report_summary[key] for key in ("n_papers", "algorithms")},
        }
    print(json.dumps(result, ensure_ascii=False, indent=2)[:6000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
