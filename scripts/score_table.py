#!/usr/bin/env python
"""Aggregate the released result files into a paper table.

Scores every cell a config stands for and prints mean (95% CI) across the
cohort reshuffles, which is the form the paper's tables use.

::

    python scripts/score_table.py configs/paper/table_1.yaml
    python scripts/score_table.py configs/paper/table_1.yaml --task mortality_pred --csv
"""

from __future__ import annotations

import argparse
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from clinicalbench.eval.metrics import score_file  # noqa: E402
from clinicalbench.experiments import expand, load_roster  # noqa: E402

T_95 = {2: 12.71, 3: 4.303, 4: 3.182, 5: 2.776}


def mean_ci95(values):
    """Mean and 95% CI half-width, using the t distribution for small n."""
    if len(values) < 2:
        return (values[0] if values else float("nan")), 0.0
    mean = statistics.mean(values)
    sem = statistics.stdev(values) / math.sqrt(len(values))
    return mean, T_95.get(len(values), 1.96) * sem


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("config", type=Path)
    parser.add_argument("--task", help="restrict to one task")
    parser.add_argument("--dataset", help="restrict to one database")
    parser.add_argument("--result_root", type=Path, default=Path("results"))
    parser.add_argument("--models_file", type=Path, default=Path("configs/models.yaml"))
    parser.add_argument("--csv", action="store_true", help="machine-readable output")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)
    runs = list(expand(config, load_roster(args.models_file)))
    if args.task:
        runs = [r for r in runs if r.task == args.task]
    if args.dataset:
        runs = [r for r in runs if r.dataset == args.dataset]

    want_auroc = "auroc" in config.get("metrics", [])
    cells = defaultdict(lambda: {"f1": [], "auroc": [], "missing": 0, "invalid": 0, "n": 0})

    for run in runs:
        key = (run.dataset, run.task, run.model_name, run.mode, run.temperature, run.ratio)
        cell = cells[key]
        path = run.existing_result_file(args.result_root)
        if path is None:
            cell["missing"] += 1
            continue
        try:
            score = score_file(path, run.task, with_auroc=want_auroc)
        except ValueError:
            score = score_file(path, run.task, with_auroc=False)
        cell["f1"].append(score.f1 * 100)
        if score.auroc is not None:
            cell["auroc"].append(score.auroc * 100)
        cell["invalid"] += score.n_invalid
        cell["n"] += score.n

    header = ["dataset", "task", "model", "mode", "temp", "ratio", "runs",
              "f1", "f1_ci", "auroc", "auroc_ci", "invalid_pct", "missing"]
    if args.csv:
        print(",".join(header))
    else:
        print(f"# {config.get('produces', args.config.name)}\n")
        print(f"{'dataset':8s} {'task':17s} {'model':28s} {'mode':5s} "
              f"{'runs':>4s} {'F1 (95% CI)':>22s} {'AUROC (95% CI)':>22s} {'inv%':>6s}")
        print("-" * 122)

    for key in sorted(cells):
        dataset, task, model, mode, temp, ratio = key
        cell = cells[key]
        if not cell["f1"]:
            continue
        f1, f1_ci = mean_ci95(cell["f1"])
        auroc, auroc_ci = mean_ci95(cell["auroc"]) if cell["auroc"] else (float("nan"), 0.0)
        inv = 100 * cell["invalid"] / cell["n"] if cell["n"] else 0.0
        if args.csv:
            print(",".join(str(v) for v in [
                dataset, task, model, mode, temp or "", ratio or "", len(cell["f1"]),
                f"{f1:.2f}", f"{f1_ci:.2f}",
                "" if math.isnan(auroc) else f"{auroc:.2f}",
                "" if math.isnan(auroc) else f"{auroc_ci:.2f}",
                f"{inv:.2f}", cell["missing"],
            ]))
        else:
            auroc_s = "-" if math.isnan(auroc) else f"{auroc:.2f} ({auroc-auroc_ci:.2f}, {auroc+auroc_ci:.2f})"
            print(f"{dataset:8s} {task:17s} {model:28s} {mode:5s} {len(cell['f1']):4d} "
                  f"{f'{f1:.2f} ({f1-f1_ci:.2f}, {f1+f1_ci:.2f})':>22s} {auroc_s:>22s} {inv:5.1f}%")

    total_missing = sum(c["missing"] for c in cells.values())
    if total_missing and not args.csv:
        print(f"\n{total_missing} runs had no result file and were skipped.")


if __name__ == "__main__":
    main()
