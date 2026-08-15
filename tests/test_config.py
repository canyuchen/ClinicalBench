"""The transcription of the cohort spec and prompt wording must not drift.

These numbers and strings were lifted out of nine duplicated scripts. If any of
them changes, the released cohorts and prompts no longer describe the released
results, so each is pinned against an independent source of truth: the shipped
``.npy`` files for cohort sizes, and the literals from the original scripts for
the prompt spans.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from clinicalbench.config import (
    DATASETS,
    RANDOM_INDICES,
    TASK_SPECS,
    TASKS,
    split_spec,
)

DATA_ROOT = Path(__file__).parent.parent / "data"

# Verbatim from the original src/test.py, which sliced these by character count.
ORIGINAL_COT_STRIP = {
    "length_pred": (
        "\nAnswer 1 if no more than one week,\nAnswer 2 if more than one week but "
        "not more than two weeks,\nAnswer 3 if more than two weeks.\nAnswer with "
        "only the number. Answer: "
    ),
    "mortality_pred": "\nAnswer 1 if yes, 0 if no. Answer with only the number.\nAnswer: ",
    "readmission_pred": "\nAnswer 1 for yes, 0 for no. Answer with only the number.\nAnswer: ",
}
ORIGINAL_SR_STRIP = {
    "length_pred": "\nAnswer with only the number. Answer: ",
    "mortality_pred": "Answer with only the number.\nAnswer: ",
    "readmission_pred": "Answer with only the number.\nAnswer: ",
}


@pytest.mark.parametrize("task", TASKS)
def test_cot_strip_matches_original(task):
    assert TASK_SPECS[task].cot_strip == ORIGINAL_COT_STRIP[task]


@pytest.mark.parametrize("task", TASKS)
def test_sr_strip_matches_original(task):
    assert TASK_SPECS[task].sr_strip == ORIGINAL_SR_STRIP[task]


@pytest.mark.parametrize("task", TASKS)
def test_strips_are_real_suffixes(task):
    """A strip must be removable from a real prompt tail, or it would eat data."""
    spec = TASK_SPECS[task]
    suffix = spec.prompt_suffix()
    assert suffix.endswith(spec.cot_strip)
    assert suffix.endswith(spec.sr_strip)


@pytest.mark.skipif(not DATA_ROOT.exists(), reason="cohort indices not present")
@pytest.mark.parametrize("task", TASKS)
@pytest.mark.parametrize("dataset", DATASETS)
@pytest.mark.parametrize("random_index", RANDOM_INDICES)
def test_split_sizes_match_shipped_indices(task, dataset, random_index):
    """The transcribed cohort sizes must agree with the released .npy files."""
    spec = split_spec(task, dataset, random_index)
    expected = {"train": sum(spec.train), "val": sum(spec.val), "test": sum(spec.test)}
    for split, want in expected.items():
        path = DATA_ROOT / task / dataset / f"{split}_index_{random_index}.npy"
        assert len(np.load(path, allow_pickle=True)) == want, f"{path} size drifted"


@pytest.mark.parametrize("task", TASKS)
def test_label_space(task):
    spec = TASK_SPECS[task]
    assert spec.answer_tokens == [str(label) for label in spec.labels]
    assert spec.is_multiclass == (task == "length_pred")
    assert spec.average == ("macro" if task == "length_pred" else "binary")
