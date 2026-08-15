"""Answer extraction and the invalid-output penalty."""

from __future__ import annotations

import pytest

from clinicalbench.answers import extract_answer, penalise_invalid, resolve_prediction
from clinicalbench.config import TASK_SPECS

BINARY = TASK_SPECS["mortality_pred"]
TERNARY = TASK_SPECS["length_pred"]


@pytest.mark.parametrize(
    "text,expected",
    [
        ("1", "1"),
        ("0", "0"),
        ("The answer is 1.", "1"),          # backward scan past the full stop
        ("1 then 0", "0"),                   # last valid digit wins
        ("", None),
        ("no digits here", None),
        ("7", None),                          # outside the label space
    ],
)
def test_backward_scan(text, expected):
    assert extract_answer(text, BINARY, scan=True) == expected


@pytest.mark.parametrize(
    "text,expected",
    [("1", "1"), ("1.", None), ("x1", "1"), ("", None)],
)
def test_last_character_only(text, expected):
    """The logits path reads only the final character."""
    assert extract_answer(text, BINARY, scan=False) == expected


def test_the_two_paths_disagree_where_documented():
    assert extract_answer("1.", BINARY, scan=True) == "1"
    assert extract_answer("1.", BINARY, scan=False) is None


def test_penalty_is_always_wrong():
    for gold in ("0", "1"):
        assert penalise_invalid(gold, BINARY) != gold
    for gold in ("1", "2", "3"):
        assert penalise_invalid(gold, TERNARY) != gold


def test_penalty_stays_in_the_label_space():
    assert penalise_invalid("2", TERNARY) in {"1", "2", "3"}
    assert penalise_invalid("3", TERNARY) in {"1", "2", "3"}


def test_resolve_reports_invalidity():
    assert resolve_prediction("1", "1", BINARY) == ("1", False)
    assert resolve_prediction("banana", "1", BINARY) == ("0", True)
