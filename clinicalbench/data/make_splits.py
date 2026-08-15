"""Generate the train/val/test visit-id index files.

Replaces the three copies of ``get_index.py`` (one per task, each with the
per-database cohort sizes inlined). The sizes now live in
:mod:`clinicalbench.config`; the sampling procedure below is unchanged, so
re-running this reproduces the ``.npy`` files shipped in ``data/``.

Usage::

    python -m clinicalbench.data.make_splits --task mortality_pred --dataset mimic3
    python -m clinicalbench.data.make_splits --all
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List

import numpy as np

from clinicalbench.config import (
    DATASETS,
    RANDOM_INDICES,
    TASK_SPECS,
    TASKS,
    seed_for,
    split_spec,
)


def shuffle_with_seed(items: List, seed: int) -> List:
    shuffled = items.copy()
    random.seed(seed)
    random.shuffle(shuffled)
    return shuffled


def group_visits_by_label(samples: List[Dict], labels) -> Dict[int, List]:
    """Bucket visit ids by label, preserving the order they appear in the file."""
    pools: Dict[int, List] = {label: [] for label in labels}
    for sample in samples:
        label = sample["label"]
        if label in pools:
            pools[label].append(sample["visit_id"])
    return pools


def build_split(samples: List[Dict], task: str, dataset: str, random_index: int):
    """Return (train, val, test) visit-id lists for one cohort.

    Per label: shuffle, take ``train`` from the head so training is balanced,
    then take val and test starting at ``offset`` so that together they keep the
    database's natural class ratio.
    """
    spec = TASK_SPECS[task]
    sizes = split_spec(task, dataset, random_index)
    seed = seed_for(random_index)
    pools = group_visits_by_label(samples, spec.labels)

    train: List = []
    val: List = []
    test: List = []
    for i, label in enumerate(spec.labels):
        pool = shuffle_with_seed(pools[label], seed=seed)
        train += pool[: sizes.train[i]]
        rest = pool[sizes.offset[i]:]
        val += rest[: sizes.val[i]]
        rest = rest[sizes.val[i]:]
        test += rest[: sizes.test[i]]
    return train, val, test


def write_split(out_dir: Path, random_index: int, train, val, test) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, index in (("train", train), ("val", val), ("test", test)):
        with open(out_dir / f"{name}_index_{random_index}.npy", "wb") as f:
            np.save(f, index)


def run(task: str, dataset: str, data_root: Path) -> None:
    task_dir = data_root / task / dataset
    samples_path = task_dir / f"{task}_data.json"
    if not samples_path.exists():
        raise FileNotFoundError(
            f"{samples_path} not found -- run `python -m clinicalbench.data.build_cohort` first"
        )
    with open(samples_path) as f:
        samples = json.load(f)

    for random_index in RANDOM_INDICES:
        train, val, test = build_split(samples, task, dataset, random_index)
        write_split(task_dir, random_index, train, val, test)
        print(
            f"{task}/{dataset} index {random_index}: "
            f"train={len(train)} val={len(val)} test={len(test)}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--task", choices=TASKS)
    parser.add_argument("--dataset", choices=DATASETS)
    parser.add_argument("--data_root", type=Path, default=Path("data"))
    parser.add_argument("--all", action="store_true", help="every task x database")
    args = parser.parse_args()

    if args.all:
        pairs = [(t, d) for t in TASKS for d in DATASETS]
    elif args.task and args.dataset:
        pairs = [(args.task, args.dataset)]
    else:
        parser.error("pass --all, or both --task and --dataset")

    for task, dataset in pairs:
        run(task, dataset, args.data_root)


if __name__ == "__main__":
    main()
