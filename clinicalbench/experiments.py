"""Expand a paper config into the runs it stands for.

Each file in ``configs/paper/`` describes one table or figure. This module turns
one into concrete commands, so the mapping from a number in the paper to the
command that produced it is mechanical rather than prose.

::

    python -m clinicalbench.experiments configs/paper/table_1.yaml            # print commands
    python -m clinicalbench.experiments configs/paper/table_1.yaml --check    # coverage report
    python -m clinicalbench.experiments configs/paper/table_1.yaml --run      # execute
"""

from __future__ import annotations

import argparse
import itertools
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional

import yaml

from clinicalbench.naming import result_path

CONFIG_ROOT = Path("configs")
MODELS_FILE = CONFIG_ROOT / "models.yaml"


@dataclass
class Run:
    """One (model, task, dataset, cohort, setting) cell."""

    model_name: str
    model_id: Optional[str]  # None for traditional baselines
    task: str
    dataset: str
    random_index: int
    mode: str = "ORI"
    scoring: str = "logits"
    temperature: Optional[float] = None
    ratio: Optional[float] = None

    @property
    def is_baseline(self) -> bool:
        return self.model_id is None

    def result_file(self, result_root: Path) -> Path:
        return result_path(
            result_root, self.task, self.dataset, self.model_name,
            self.random_index, self.mode, self.temperature, self.ratio,
        )

    def command(self, data_root: Path, result_root: Path) -> List[str]:
        if self.is_baseline:
            cmd = [
                sys.executable, "-m", "clinicalbench.baselines.traditional",
                "--task", self.task, "--dataset", self.dataset,
                "--random_index", str(self.random_index),
                "--models", self.model_name,
            ]
            if self.ratio is not None and self.ratio != 1:
                cmd += ["--ratio", str(self.ratio)]
        else:
            cmd = [
                sys.executable, "-m", "clinicalbench.inference.llm",
                "--base_model", self.model_id,
                "--task", self.task, "--dataset", self.dataset,
                "--mode", self.mode, "--scoring", self.scoring,
                "--random_index", str(self.random_index),
            ]
            if self.temperature is not None:
                cmd += ["--temperature", str(self.temperature)]
        return cmd + ["--data_root", str(data_root), "--result_root", str(result_root)]


def load_roster(path: Path = MODELS_FILE) -> Dict[str, List[Dict]]:
    """Resolve ``configs/models.yaml`` into ``group -> [{name, id}, ...]``."""
    with open(path) as f:
        spec = yaml.safe_load(f)
    catalogue = spec["catalogue"]
    roster: Dict[str, List[Dict]] = {}
    for group, names in spec["groups"].items():
        entries = []
        for name in names:
            if name not in catalogue:
                raise KeyError(f"group {group!r} references unknown model {name!r}")
            entries.append({"name": name, "id": catalogue[name]})
        roster[group] = entries
    return roster


def expand(config: Dict, roster: Dict[str, List[Dict]]) -> Iterator[Run]:
    models: List[Dict] = []
    seen = set()
    for group in config["models"]:
        if group not in roster:
            raise KeyError(
                f"unknown model group {group!r}; defined groups are {sorted(roster)}"
            )
        for entry in roster[group]:
            if entry["name"] not in seen:
                seen.add(entry["name"])
                models.append(entry)

    modes = config.get("modes") or [config.get("mode", "ORI")]
    temperatures = config.get("temperatures") or [None]
    ratios = config.get("ratios") or [None]
    scoring_by_mode = config.get("scoring_by_mode") or {}
    default_scoring = config.get("scoring", "logits")

    for model, task, dataset, index, mode, temp, ratio in itertools.product(
        models, config["tasks"], config["datasets"], config["random_indices"],
        modes, temperatures, ratios,
    ):
        is_baseline = model["id"] is None
        # prompt-engineering modes and temperature sweeps are LLM-only
        if is_baseline and (mode != "ORI" or temp is not None):
            continue
        # the baseline scaling tables are the only place ratios apply
        if not is_baseline and ratio is not None:
            continue
        yield Run(
            model_name=model["name"] if is_baseline else model["id"].split("/")[-1],
            model_id=model["id"],
            task=task, dataset=dataset, random_index=index, mode=mode,
            scoring=scoring_by_mode.get(mode, default_scoring),
            temperature=temp, ratio=ratio,
        )


def check(runs: List[Run], result_root: Path) -> int:
    """Report which cells already have a result file."""
    missing = [r for r in runs if not r.result_file(result_root).exists()]
    present = len(runs) - len(missing)
    print(f"{present}/{len(runs)} cells have a result file under {result_root}/")
    if present == 0 and not any(Path(result_root).glob("*/*/*.csv")):
        # the usual cause is a fresh clone: the results live on the Hub
        print(
            "\nNo result files found at all. The released results are not in git;\n"
            "fetch them with:\n"
            "    clinicalbench-fetch-results"
        )
        return len(missing)
    if missing:
        print(f"\nmissing ({len(missing)}):")
        by_model: Dict[str, int] = {}
        for r in missing:
            by_model[r.model_name] = by_model.get(r.model_name, 0) + 1
        for name, n in sorted(by_model.items(), key=lambda kv: -kv[1]):
            print(f"  {n:4d}  {name}")
    return len(missing)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("config", type=Path, help="a file under configs/paper/")
    parser.add_argument("--run", action="store_true", help="execute (default: print)")
    parser.add_argument("--check", action="store_true",
                        help="report which cells already have results")
    parser.add_argument("--skip-existing", action="store_true",
                        help="with --run, skip cells that already have a result file")
    parser.add_argument("--data_root", type=Path, default=Path("data"))
    parser.add_argument("--result_root", type=Path, default=Path("results"))
    parser.add_argument("--models_file", type=Path, default=MODELS_FILE)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)
    runs = list(expand(config, load_roster(args.models_file)))

    print(f"# {config.get('produces', args.config.name)}: {len(runs)} runs\n")

    if args.check:
        raise SystemExit(1 if check(runs, args.result_root) else 0)

    for run in runs:
        if args.skip_existing and run.result_file(args.result_root).exists():
            continue
        cmd = run.command(args.data_root, args.result_root)
        if args.run:
            print(f"$ {' '.join(cmd)}", flush=True)
            subprocess.run(cmd, check=True)
        else:
            print(" ".join(cmd))


if __name__ == "__main__":
    main()
