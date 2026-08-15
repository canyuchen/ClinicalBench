"""Table aggregation: the CI math and the end-to-end path over released files."""

from __future__ import annotations

from pathlib import Path

import pytest

from _results import RESULTS_ROOT, requires_results
from clinicalbench.eval.aggregate import mean_ci95


def test_mean_ci95_uses_the_t_distribution_for_five_runs():
    # five runs is the paper's setting: t(4) = 2.776, not the normal 1.96
    values = [10.0, 11.0, 12.0, 13.0, 14.0]
    mean, half = mean_ci95(values)
    assert mean == pytest.approx(12.0)
    sem = pytest.approx(2.776 * (2.5 ** 0.5 / 5 ** 0.5), rel=1e-3)
    assert half == sem


def test_mean_ci95_degenerate_inputs():
    assert mean_ci95([42.0]) == (42.0, 0.0)
    mean, half = mean_ci95([])
    assert half == 0.0 and mean != mean  # NaN mean, zero width


@requires_results
def test_aggregate_reproduces_table_1_xgboost():
    """The module-level path: config -> runs -> released files -> paper number."""
    import yaml

    from clinicalbench.eval.aggregate import score_file
    from clinicalbench.experiments import expand, load_roster

    root = Path(__file__).parent.parent
    config = yaml.safe_load(open(root / "configs/paper/table_1.yaml"))
    runs = [
        r for r in expand(config, load_roster(root / "configs/models.yaml"))
        if r.model_name == "XGBoost" and r.task == "length_pred"
    ]
    assert len(runs) == 5
    f1s = [score_file(r.result_file(RESULTS_ROOT), r.task).f1 * 100 for r in runs]
    mean, _ = mean_ci95(f1s)
    assert mean == pytest.approx(67.94, abs=0.01)
