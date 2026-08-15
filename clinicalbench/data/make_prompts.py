"""Render per-task sample files into LLM prompts.

Step 2 of data preparation. Replaces the six near-identical ``convert.py``
scripts: the task wording, the database schema differences and the few-shot
exemplars are all data now (:mod:`clinicalbench.config` and
``templates/icl/``), so there is a single code path.

Writes ``data/{task}/{dataset}/{task}_data.csv`` (and ``..._data_ICL.csv``
with ``--icl``), with columns ID, VISIT_ID, SUBJECT_ID, QUESTION, ANSWER.

Usage::

    python -m clinicalbench.data.make_prompts \\
        --task mortality_pred --dataset mimic3 --mimic_path /path/to/mimic-iii
    python -m clinicalbench.data.make_prompts --all --mimic3_path ... --mimic4_path ...
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List

from clinicalbench.config import DATASET_SPECS, DATASETS, TASK_SPECS, TASKS
from clinicalbench.data.prompts import build_prompt, load_icl_prefix

UNKNOWN_DIAGNOSIS = "Unknown Diagnosis"
UNKNOWN_PROCEDURE = "Unknown Procedure"
UNKNOWN_DRUG = "Unknown Drug"

ATC_DICT_PATH = Path(__file__).parent / "atc_dic.json"


def load_code_titles(mimic_path: Path, table: str, code_col: str, title_col: str) -> Dict[str, str]:
    path = mimic_path / table
    if not path.exists():
        raise FileNotFoundError(f"{path} not found -- check the MIMIC path for this database")
    with open(path, "r", encoding="utf-8") as f:
        return {row[code_col]: row[title_col] for row in csv.DictReader(f)}


def load_atc_dict() -> Dict[str, str]:
    with open(ATC_DICT_PATH) as f:
        return json.load(f)


def _decode(codes: List[str], table: Dict[str, str], fallback: str, lower: bool = False) -> List[str]:
    """Map raw codes to human-readable names, falling back on a placeholder.

    Unmapped codes keep the placeholder verbatim -- the original lowercased only
    successful drug lookups, so "Unknown Drug" stays capitalised.
    """
    out = []
    for code in codes:
        if code in table:
            name = table[code]
            out.append(name.lower() if lower else name)
        else:
            out.append(fallback)
    return out


def run(task: str, dataset: str, mimic_path: Path, data_root: Path, icl: bool) -> None:
    task_spec = TASK_SPECS[task]
    dataset_spec = DATASET_SPECS[dataset]
    task_dir = data_root / task / dataset

    samples_path = task_dir / f"{task}_data.json"
    if not samples_path.exists():
        raise FileNotFoundError(
            f"{samples_path} not found -- run `python -m clinicalbench.data.build_cohort` first"
        )
    with open(samples_path) as f:
        samples = json.load(f)

    diagnoses = load_code_titles(
        mimic_path, dataset_spec.diagnoses_table,
        dataset_spec.code_column, dataset_spec.title_column,
    )
    procedures = load_code_titles(
        mimic_path, dataset_spec.procedures_table,
        dataset_spec.code_column, dataset_spec.title_column,
    )
    atc = load_atc_dict()

    icl_prefix = load_icl_prefix(task, dataset) if icl else ""
    suffix = "_ICL" if icl else ""
    out_path = task_dir / f"{task}_data{suffix}.csv"

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["ID", "VISIT_ID", "SUBJECT_ID", "QUESTION", "ANSWER"]
        )
        writer.writeheader()
        for count, sample in enumerate(samples):
            prompt = build_prompt(
                sample=sample,
                task_spec=task_spec,
                dataset_spec=dataset_spec,
                # only the first (index) visit is described, as in the original
                conditions=_decode(sample["conditions"][0], diagnoses, UNKNOWN_DIAGNOSIS),
                procedures=_decode(sample["procedures"][0], procedures, UNKNOWN_PROCEDURE),
                drugs=_decode(sample["drugs"][0], atc, UNKNOWN_DRUG, lower=True),
                icl_prefix=icl_prefix,
            )
            writer.writerow({
                "ID": count,
                "VISIT_ID": sample["visit_id"],
                "SUBJECT_ID": sample["patient_id"],
                "QUESTION": prompt,
                "ANSWER": sample["label"],
            })
    print(f"{task}/{dataset}{suffix}: {len(samples)} prompts -> {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--task", choices=TASKS)
    parser.add_argument("--dataset", choices=DATASETS)
    parser.add_argument("--mimic_path", type=Path,
                        help="MIMIC root for the chosen --dataset")
    parser.add_argument("--mimic3_path", type=Path, help="used with --all")
    parser.add_argument("--mimic4_path", type=Path, help="used with --all")
    parser.add_argument("--data_root", type=Path, default=Path("data"))
    parser.add_argument("--icl", action="store_true",
                        help="prepend few-shot exemplars and write *_ICL.csv")
    parser.add_argument("--both", action="store_true",
                        help="write the plain and the ICL variant in one pass")
    parser.add_argument("--all", action="store_true",
                        help="every task x database (needs --mimic3_path/--mimic4_path)")
    args = parser.parse_args()

    variants = [False, True] if args.both else [args.icl]

    if args.all:
        roots = {"mimic3": args.mimic3_path, "mimic4": args.mimic4_path}
        jobs = [(t, d, roots[d]) for t in TASKS for d in DATASETS if roots[d]]
        if not jobs:
            parser.error("--all needs --mimic3_path and/or --mimic4_path")
    elif args.task and args.dataset and args.mimic_path:
        jobs = [(args.task, args.dataset, args.mimic_path)]
    else:
        parser.error("pass --all, or all of --task --dataset --mimic_path")

    for task, dataset, root in jobs:
        for icl in variants:
            run(task, dataset, root, args.data_root, icl)


if __name__ == "__main__":
    main()
