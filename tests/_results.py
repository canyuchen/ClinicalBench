"""Shared guard for tests that score the released result files.

The results are not in git -- they are fetched from the Hub into ``results/``,
which otherwise holds only its README. So presence has to be decided by looking
for actual result files, not by the directory existing.
"""

from __future__ import annotations

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parent.parent
RESULTS_ROOT = REPO_ROOT / "results"
CONFIG_ROOT = REPO_ROOT / "configs"


def results_available() -> bool:
    return any(RESULTS_ROOT.glob("*/*/*.csv"))


requires_results = pytest.mark.skipif(
    not results_available(),
    reason="released results not fetched -- run `clinicalbench-fetch-results`",
)
