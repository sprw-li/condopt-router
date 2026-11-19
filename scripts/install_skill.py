"""Install/refresh the ``condopt-router`` skill in the shared agent-library.

The project directory is the single source of truth for the skill body and its
references; this script syncs them into ``<agent-library>\\catalog``.

It is **self-contained**: the upstream ``create_skill.py`` helper has moved out of
``<agent-library>`` more than once, so the installer writes the catalog entry
itself (SKILL.md + references + manifest/intent patch) and only *optionally* uses
external tooling when it is actually present.

Usage:
    py -3 scripts/install_skill.py            # install/refresh
    py -3 scripts/install_skill.py --dry-run  # show what would change
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
def _resolve_shared_library(leaf: str) -> Path:
    """Locate the shared skill library shipped outside this repository.

    Order: ``CONDOPT_AGENT_LIBRARY`` override -> ``~/agent-library`` -> a sibling
    ``agent-library`` on the same drive as this checkout.  Nothing is hard-coded to a
    specific drive letter, so a clone on another machine simply falls back to the first
    candidate (and the caller reports "not found" instead of inventing a path).
    """
    override = os.environ.get("CONDOPT_AGENT_LIBRARY")
    if override:
        return (Path(override).expanduser() / leaf).expanduser()
    candidates = [Path.home() / "agent-library"]
    if PROJECT.drive:
        candidates.append(Path(f"{PROJECT.drive}/") / "agent-library")
    for candidate in candidates:
        # ``catalog/`` alone is not enough: a stale/partial library can have the
        # directory but no registry, and installing into it silently fails later.
        if (candidate / "catalog" / "intents.json").exists():
            return candidate / leaf
    return candidates[0] / leaf


SKILL_MD = PROJECT / "skill" / "SKILL.md"
REFERENCES = PROJECT / "skill" / "references"
#: 共享技能库根目录：CONDOPT_AGENT_LIBRARY 覆盖，否则自动发现
AGLIB = _resolve_shared_library("")
CATALOG = AGLIB / "catalog"
SKILL_ID = "condopt-router"

#: candidate locations for optional upstream helpers (may not exist)
STORE_CANDIDATES = [AGLIB / "scripts" / "store.py"]
LINT_CANDIDATES = [CATALOG / "skill-retro" / "scripts" / "check_skill.py"]

DESCRIPTION = (
    "条件优化算法路由器 + 实验闭环驱动：先问必要参数并给建议范围，再按问题特征选算法"
    "（bayes/tpe/parego/qehvi/qnehvi/cqehvi/constrained/mfbo/sobol/lhs/grid），生成待测条件表并回收实测结果。"
)
WHEN = "用户要优化实验/反应/工艺条件、设计下一批实验、或要选条件优化算法时使用"
PATTERNS = [
    "条件优化",
    "反应条件",
    "实验条件",
    "工艺参数",
    "贝叶斯优化",
    "下一批实验",
    "下一轮实验",
    "怎么取点",
    "设计.*实验",
    "优化.*收率",
    "提高.*收率",
    "不能超过",
    "温度上限",
    "约束.*条件",
]
KEYWORDS = [
    "条件优化",
    "反应条件",
    "实验条件",
    "贝叶斯优化",
    "取样",
    "收率",
    "工艺参数",
    "约束",
]
EXAMPLES = [
    "帮我用贝叶斯优化找反应的最优温度",
    "下一批实验条件该怎么取点？",
    "溶剂和配比怎么设计下一轮实验",
    "用更少的实验找到最高收率条件",
]
WHEN_NOT = [
    "只需一次性随机抽样、没有真实实验评价通道、或与实验条件优化无关的任务",
    "用户只问算法概念，不打算跑 ask/tell 实验闭环",
]
EXCLUDES = ["论文", "综述", "文献"]


def strip_frontmatter(text: str) -> str:
    if text.startswith("---"):
        _, _, rest = text.partition("---")
        _, _, body = rest.partition("---")
        return body.lstrip("\n")
    return text


def write_json_preserving_newlines(path: Path, data: dict) -> None:
    """Rewrite JSON without churning the file's existing line endings.

    agent-library's intents.json uses CRLF; a naive write_text() would turn a
    one-intent addition into a whole-file diff.
    """
    newline = "\r\n" if path.exists() and b"\r\n" in path.read_bytes() else "\n"
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline=newline,
    )


def _first_existing(candidates: list[Path]) -> Path | None:
    for path in candidates:
        if path.exists():
            return path
    return None


def write_skill_body(target: Path) -> None:
    body = strip_frontmatter(SKILL_MD.read_text(encoding="utf-8")).rstrip() + "\n"
    front = "---\nname: {name}\ndescription: >-\n  {description}\n---\n\n".format(
        name=SKILL_ID, description=DESCRIPTION
    )
    target.mkdir(parents=True, exist_ok=True)
    (target / "SKILL.md").write_text(front + body, encoding="utf-8")
    shutil.copytree(REFERENCES, target / "references", dirs_exist_ok=True)


def patch_catalog_entry() -> None:
    manifest_path = CATALOG / SKILL_ID / "manifest.json"
    intents_path = CATALOG / "intents.json"

    manifest: dict = {}
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.setdefault("schemaVersion", 1)
    manifest.setdefault("id", SKILL_ID)
    manifest.setdefault("kind", "skill")
    manifest.setdefault("mode", "auto")
    # the linter requires non-empty *arrays* for when/whenNot, and the required
    # fields requires/merge must exist even when nothing is merged in.
    manifest.setdefault("requires", [])
    manifest.setdefault("merge", {})
    manifest["description"] = DESCRIPTION
    manifest["when"] = [WHEN]
    manifest["whenNot"] = WHEN_NOT
    manifest["examples"] = EXAMPLES
    manifest["keywords"] = KEYWORDS
    manifest.setdefault("category", "general")
    write_json_preserving_newlines(manifest_path, manifest)

    intents = json.loads(intents_path.read_text(encoding="utf-8"))
    entry = {
        "id": SKILL_ID,
        "description": DESCRIPTION,
        "patterns": PATTERNS,
        "keywords": KEYWORDS,
        "examples": EXAMPLES,
        "excludes": EXCLUDES,
    }
    for index, item in enumerate(intents["intents"]):
        if item.get("id") == SKILL_ID:
            intents["intents"][index] = {**item, **entry}
            break
    else:
        intents["intents"].append(entry)
    write_json_preserving_newlines(intents_path, intents)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    target = CATALOG / SKILL_ID
    if args.dry_run:
        print(f"would write {target / 'SKILL.md'} (+ references) and patch manifest/intents")
        return 0

    write_skill_body(target)
    patch_catalog_entry()

    store = _first_existing(STORE_CANDIDATES)
    if store is not None:
        subprocess.run([sys.executable, str(store)], check=True)
    else:
        print("note: agent-library store.py not found; skipped the catalog index rebuild")

    lint = _first_existing(LINT_CANDIDATES)
    if lint is not None:
        subprocess.run([sys.executable, str(lint), str(target)], check=True)
    else:
        print("note: agent-library check_skill.py not found; skipped the lint step")

    print(f"installed {SKILL_ID} -> {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())