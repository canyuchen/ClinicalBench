"""Score a result file.

Replaces ``calculate.py``. Three things changed beyond the move:

* filenames come from :mod:`clinicalbench.naming`, so temperature results are
  actually found (the old evaluator looked for ``_temp0.2`` while the runner
  wrote ``_temp_0.2``, making every Figure 3 file unreadable);
* ``result_root`` is a parameter rather than a module global that only existed
  when the file was run as a script, so these functions can be imported;
* the number of rows that carried no valid answer is reported, since those rows
  are scored as wrong and materially affect a weak model's F1.

Usage::

    python -m clinicalbench.eval.metrics --base_model meta-llama/Meta-Llama-3-8B-Instruct \\
        --task mortality_pred --dataset mimic3 --random_index 0 --auroc
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Union

from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score

from clinicalbench.answers import penalise_invalid
from clinicalbench.config import DATASETS, MODES, TASK_SPECS, TASKS
from clinicalbench.naming import result_path


@dataclass
class Score:
    task: str
    path: Path
    n: int
    f1: float
    n_invalid: int
    auroc: Optional[float] = None
    confusion: Optional[List[List[int]]] = None

    @property
    def invalid_rate(self) -> float:
        return self.n_invalid / self.n if self.n else 0.0

    def report(self) -> str:
        lines = [
            f"{self.path}",
            f"  n         {self.n}",
            f"  F1        {self.f1 * 100:.2f}",
        ]
        if self.auroc is not None:
            lines.append(f"  AUROC     {self.auroc * 100:.2f}")
        lines.append(
            f"  invalid   {self.n_invalid} ({self.invalid_rate:.1%}) "
            f"rows had no parseable answer and are scored wrong"
        )
        if self.confusion is not None:
            lines.append("  confusion")
            for row in self.confusion:
                lines.append("    " + " ".join(f"{v:6d}" for v in row))
        return "\n".join(lines)


def score_file(path: Union[str, Path], task: str, with_auroc: bool = False) -> Score:
    """F1 (and optionally AUROC) for one result CSV."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found")
    spec = TASK_SPECS[task]
    valid = set(spec.labels)

    golds: List[int] = []
    preds: List[int] = []
    probs: List[float] = []
    n_invalid = 0

    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        has_prob = reader.fieldnames is not None and "PROB" in reader.fieldnames
        for row in reader:
            gold = int(row["ANSWER"])
            raw = row["PREDICTION"]
            try:
                pred = int(raw)
            except (TypeError, ValueError):
                pred = None
            if pred not in valid:
                # same penalty the runner applies: score an unparseable answer wrong
                pred = int(penalise_invalid(str(gold), spec))
                n_invalid += 1
            golds.append(gold)
            preds.append(pred)
            if has_prob and row.get("PROB") not in (None, ""):
                probs.append(float(row["PROB"]))

    f1 = f1_score(golds, preds, average=spec.average)
    score = Score(
        task=task, path=path, n=len(golds), f1=f1, n_invalid=n_invalid,
        confusion=confusion_matrix(golds, preds).tolist(),
    )

    if with_auroc:
        if len(probs) != len(golds):
            raise ValueError(
                f"{path} has no usable PROB column -- AUROC needs a run scored with "
                f"`--scoring logits` (traditional baselines always record it)"
            )
        if spec.is_multiclass:
            # length-of-stay AUROC is reported as "stays longer than two weeks"
            binary = [1 if g == spec.labels[-1] else 0 for g in golds]
        else:
            binary = golds
        score.auroc = roc_auc_score(binary, probs)
    return score


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--base_model", required=True,
                        help="HuggingFace id or baseline name, e.g. XGBoost")
    parser.add_argument("--task", required=True, choices=TASKS)
    parser.add_argument("--dataset", required=True, choices=DATASETS)
    parser.add_argument("--random_index", type=int, default=0)
    parser.add_argument("--mode", default="ORI", choices=MODES)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--ratio", type=float, default=None,
                        help="training-set fraction, for the baseline scaling tables")
    parser.add_argument("--auroc", action="store_true",
                        help="also report AUROC (needs a PROB column)")
    parser.add_argument("--result_root", type=Path, default=Path("results"))
    args = parser.parse_args()

    path = result_path(
        args.result_root, args.task, args.dataset, args.base_model.split("/")[-1],
        args.random_index, args.mode, args.temperature, args.ratio,
    )
    print(score_file(path, args.task, with_auroc=args.auroc).report())


if __name__ == "__main__":
    main()
