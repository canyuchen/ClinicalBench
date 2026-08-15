"""Export a fine-tuning dataset in LLaMA-Factory (Alpaca) format.

Figure 4 fine-tunes LLMs with LLaMA-Factory, but the released code never
included the dataset export, so that half of the figure could not be
reproduced. This module writes it.

The instruction wording is transcribed from the paper's fine-tuning appendix,
which phrases the task slightly differently from the inference prompt
("Given the patient information, predict ..." rather than "Predict ..."). The
input is the same patient profile the inference prompts use, so the fine-tuned
model sees the same evidence.

Usage::

    python -m clinicalbench.data.export_finetune \\
        --task length_pred --dataset mimic3 --split train
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List

import numpy as np

from clinicalbench.config import DATASETS, SMALL_COHORT_INDEX, TASK_SPECS, TASKS

#: Transcribed from the paper's fine-tuning data-construction appendix.
INSTRUCTIONS: Dict[str, str] = {
    "length_pred": (
        "Given the patient information, predict the number of weeks of stay in hospital.\n"
        "Answer 1 if no more than one week,\n"
        "Answer 2 if more than one week but not more than two weeks,\n"
        "Answer 3 if more than two weeks.\n"
        "Answer with only the number"
    ),
    "mortality_pred": (
        "Given the patient information, predict the mortality of the patient.\n"
        "Answer 1 if the patient will die, answer 0 otherwise.\n"
        "Answer with only the number"
    ),
    "readmission_pred": (
        "Given the patient information, predict the readmission of the patient.\n"
        "Answer 1 if the patient will be readmitted to the hospital within two weeks, "
        "answer 0 otherwise.\n"
        "Answer with only the number"
    ),
}


def to_input(prompt: str, task: str) -> str:
    """Strip the question block from an inference prompt, keeping the profile.

    The instruction carries the question when fine-tuning, so the input is the
    patient profile followed by the bare answer cue.
    """
    spec = TASK_SPECS[task]
    body = prompt[: -len(spec.prompt_suffix())]
    return body + "Answer:"


def export(task: str, dataset: str, split: str, random_index: int,
           data_root: Path, out_path: Path) -> None:
    spec = TASK_SPECS[task]
    task_dir = data_root / task / dataset

    prompts_path = task_dir / f"{task}_data.csv"
    if not prompts_path.exists():
        raise FileNotFoundError(
            f"{prompts_path} not found -- run `python -m clinicalbench.data.make_prompts` first"
        )
    with open(task_dir / f"{split}_index_{random_index}.npy", "rb") as f:
        wanted = set(np.load(f).tolist())

    records: List[Dict[str, str]] = []
    with open(prompts_path) as f:
        for row in csv.DictReader(f):
            if row["VISIT_ID"] not in wanted:
                continue
            records.append({
                "instruction": INSTRUCTIONS[task],
                "input": to_input(row["QUESTION"], task),
                "output": row["ANSWER"],
            })

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(records, f, indent=2)

    counts: Dict[str, int] = {}
    for r in records:
        counts[r["output"]] = counts.get(r["output"], 0) + 1
    label_summary = "  ".join(f"label {k}: {v}" for k, v in sorted(counts.items()))
    print(f"{len(records)} examples -> {out_path}\n  {label_summary}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--task", required=True, choices=TASKS)
    parser.add_argument("--dataset", required=True, choices=DATASETS)
    parser.add_argument("--split", default="train", choices=["train", "val", "test"])
    parser.add_argument("--random_index", type=int, default=SMALL_COHORT_INDEX,
                        help="cohort to export; Figure 4 uses 6")
    parser.add_argument("--data_root", type=Path, default=Path("data"))
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    out = args.out or (
        args.data_root / "finetune" /
        f"{args.task}_{args.dataset}_{args.split}_{args.random_index}.json"
    )
    export(args.task, args.dataset, args.split, args.random_index, args.data_root, out)


if __name__ == "__main__":
    main()
