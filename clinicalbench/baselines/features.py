"""Bag-of-codes features for the traditional ML baselines.

Conditions, procedures and drugs are bagged separately (2000 features each),
age is bucketed into six bands and gender into two, and the five blocks are
concatenated. Only the index visit is used, matching the LLM prompts.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

from clinicalbench.config import TASK_SPECS

CODE_VOCAB_SIZE = 2000
AGE_VOCAB_SIZE = 10
GENDER_VOCAB_SIZE = 5


def convert_age(age: float) -> str:
    if age < 30:
        return "less than 30 years old"
    if age < 40:
        return "30 to 39 years old"
    if age < 50:
        return "40 to 49 years old"
    if age < 60:
        return "50 to 59 years old"
    if age < 70:
        return "60 to 69 years old"
    return "70 years old or older"


def convert_gender(gender: str) -> str:
    return "Male" if gender == "M" else "Female"


def _join_codes(values: Sequence) -> str:
    """Flatten one sample's index-visit codes into a space-separated string."""
    return "".join(f"{code} " for code in values)


def _subsample_quota(samples: List[Dict], train_index: set, labels, ratio: float) -> Dict[int, float]:
    """How many training rows to keep per label at this training-set fraction.

    Note this keeps the *first* ``ratio`` share of each label in file order
    rather than a random sample. That is what produced the published
    training-set scaling tables, so it is preserved; it means the fractions are
    nested (40% is a superset of 20%) rather than independent draws.
    """
    counts = {label: 0 for label in labels}
    for sample in samples:
        if sample["visit_id"] in train_index and sample["label"] in counts:
            counts[sample["label"]] += 1
    return {label: count * ratio for label, count in counts.items()}


def build_features(
    task: str,
    dataset: str,
    random_index: int,
    ratio: float = 1.0,
    data_root: Path = Path("data"),
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return train/val/test features and labels."""
    spec = TASK_SPECS[task]
    task_dir = data_root / task / dataset

    with open(task_dir / f"{task}_data.json") as f:
        samples = json.load(f)
    idx = {}
    for split in ("train", "val", "test"):
        with open(task_dir / f"{split}_index_{random_index}.npy", "rb") as f:
            idx[split] = set(np.load(f).tolist())

    quota = _subsample_quota(samples, idx["train"], spec.labels, ratio)
    blocks = {
        split: {k: [] for k in ("conditions", "procedures", "drugs", "age", "gender", "label")}
        for split in ("train", "val", "test")
    }

    def add(split: str, sample: Dict) -> None:
        b = blocks[split]
        b["conditions"].append(_join_codes(sample["conditions"][0]))
        b["procedures"].append(_join_codes(sample["procedures"][0]))
        b["drugs"].append(_join_codes(sample["drugs"][0]))
        b["age"].append(convert_age(sample["age"]) + " ")
        b["gender"].append(convert_gender(sample["gender"]) + " ")
        b["label"].append(sample["label"])

    for sample in samples:
        visit, label = sample["visit_id"], sample["label"]
        if visit in idx["train"]:
            if quota.get(label, 0) > 0:
                add("train", sample)
                quota[label] -= 1
        elif visit in idx["val"]:
            add("val", sample)
        elif visit in idx["test"]:
            add("test", sample)

    vectorizers = {
        "conditions": CountVectorizer(max_features=CODE_VOCAB_SIZE),
        "procedures": CountVectorizer(max_features=CODE_VOCAB_SIZE),
        "drugs": CountVectorizer(max_features=CODE_VOCAB_SIZE),
        "age": CountVectorizer(max_features=AGE_VOCAB_SIZE),
        "gender": CountVectorizer(max_features=GENDER_VOCAB_SIZE),
    }
    matrices = {"train": [], "val": [], "test": []}
    for field, vec in vectorizers.items():
        matrices["train"].append(vec.fit_transform(blocks["train"][field]).toarray())
        for split in ("val", "test"):
            matrices[split].append(vec.transform(blocks[split][field]).toarray())

    out = []
    for split in ("train", "val", "test"):
        labels = np.array(blocks[split]["label"])
        if spec.is_multiclass:
            labels = labels - 1  # sklearn wants 0-based classes
        out.extend([np.hstack(matrices[split]), labels])
    return tuple(out)
