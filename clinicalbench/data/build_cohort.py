"""Turn raw MIMIC tables into per-task sample files.

Step 1 of data preparation (was ``src/get_data.py``). Reads the credentialed
MIMIC-III / MIMIC-IV CSVs through the vendored PyHealth readers and writes
``data/{task}/{dataset}/{task}_data.json``.

Usage::

    python -m clinicalbench.data.build_cohort --mimic3_path /path/to/mimic-iii
    python -m clinicalbench.data.build_cohort --mimic4_path /path/to/mimic-iv/hosp
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

from clinicalbench._vendor.pyhealth.datasets import MIMIC3Dataset, MIMIC4Dataset
from clinicalbench._vendor.pyhealth.tasks import (
    length_of_stay_prediction_mimic3_fn,
    length_of_stay_prediction_mimic4_fn,
    mortality_prediction_mimic3_fn,
    mortality_prediction_mimic4_fn,
    readmission_prediction_mimic3_fn,
    readmission_prediction_mimic4_fn,
)
from clinicalbench.config import DATASET_SPECS

TASK_FNS = {
    "mimic3": {
        "length_pred": length_of_stay_prediction_mimic3_fn,
        "mortality_pred": mortality_prediction_mimic3_fn,
        "readmission_pred": readmission_prediction_mimic3_fn,
    },
    "mimic4": {
        "length_pred": length_of_stay_prediction_mimic4_fn,
        "mortality_pred": mortality_prediction_mimic4_fn,
        "readmission_pred": readmission_prediction_mimic4_fn,
    },
}

READERS = {"mimic3": MIMIC3Dataset, "mimic4": MIMIC4Dataset}

#: NDC drug codes are mapped to level-3 ATC classes so prompts name drug
#: classes ("beta blocking agents") rather than opaque product codes.
CODE_MAPPING = {"NDC": ("ATC", {"target_kwargs": {"level": 3}})}


def build(dataset: str, root: str, data_root: Path) -> None:
    spec = DATASET_SPECS[dataset]
    print(f"[{dataset}] reading tables from {root} ...")
    base = READERS[dataset](
        root=root,
        tables=list(spec.pyhealth_tables),
        code_mapping=CODE_MAPPING,
    )

    for task, task_fn in TASK_FNS[dataset].items():
        samples = base.set_task(task_fn=task_fn).samples
        out_dir = data_root / task / dataset
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{task}_data.json"
        with open(out_path, "w") as f:
            json.dump(samples, f, indent=2)
        print(f"[{dataset}] {task}: {len(samples)} samples -> {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--mimic3_path", type=str, default=None,
                        help="root of the MIMIC-III CSVs")
    parser.add_argument("--mimic4_path", type=str, default=None,
                        help="root of the MIMIC-IV hosp CSVs")
    parser.add_argument("--data_root", type=Path, default=Path("data"))
    args = parser.parse_args()

    if not args.mimic3_path and not args.mimic4_path:
        parser.error("pass at least one of --mimic3_path / --mimic4_path")

    if args.mimic3_path:
        build("mimic3", args.mimic3_path, args.data_root)
    if args.mimic4_path:
        build("mimic4", args.mimic4_path, args.data_root)


if __name__ == "__main__":
    main()
