"""Result filenames must round-trip against the files actually released.

The released code built these names in two places and they disagreed: the
runner wrote ``_temp_0.2`` and the evaluator looked for ``_temp0.2``, so no
temperature result could be scored. These tests pin the writer and the reader
to the same function, and check it against the released filenames.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from _results import RESULTS_ROOT, requires_results
from clinicalbench.naming import mode_suffix, ratio_suffix, result_filename, temperature_suffix


def test_greedy_and_full_training_set_add_no_suffix():
    assert temperature_suffix(None) == ""
    assert temperature_suffix(0) == ""
    assert ratio_suffix(None) == ""
    assert ratio_suffix(1) == ""
    assert mode_suffix("ORI") == ""


def test_temperature_suffix_keeps_the_underscore():
    # the exact bug: the evaluator used to build "_temp0.2"
    assert temperature_suffix(0.2) == "_temp_0.2"
    assert result_filename("mortality_pred", "BioMistral-7B", 0, "ORI", 0.2) == (
        "mortality_pred_result_data_BioMistral-7B_0_temp_0.2.csv"
    )


def test_unknown_mode_is_rejected():
    with pytest.raises(ValueError, match="unknown mode"):
        mode_suffix("NOPE")


@requires_results
def test_every_temperature_file_is_reconstructible():
    """Regenerate each released temperature filename from its parsed parts."""
    checked = 0
    for path in RESULTS_ROOT.glob("*/*/*_temp_*.csv"):
        task = path.parts[-3]
        m = re.fullmatch(rf"{task}_result_data_(.+)_(\d+)_temp_([\d.]+)\.csv", path.name)
        assert m, f"unparseable released filename: {path.name}"
        model, index, temp = m.group(1), int(m.group(2)), float(m.group(3))
        assert result_filename(task, model, index, "ORI", temp) == path.name
        checked += 1
    assert checked > 0, "no temperature results found to check"


@requires_results
def test_every_ratio_file_is_reconstructible():
    checked = 0
    for path in RESULTS_ROOT.glob("*/*/*.csv"):
        task = path.parts[-3]
        m = re.fullmatch(rf"{task}_result_data_(.+)_(\d+)_(0\.\d+)\.csv", path.name)
        if not m:
            continue
        model, index, ratio = m.group(1), int(m.group(2)), float(m.group(3))
        assert result_filename(task, model, index, "ORI", None, ratio) == path.name
        checked += 1
    assert checked > 0, "no training-set-fraction results found to check"
