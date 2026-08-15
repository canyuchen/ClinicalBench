"""End-to-end check that the evaluator still reproduces the paper.

Scores the released result files and compares against the numbers printed in
the paper. This is the test that fails if a refactor changes what a metric
means -- it caught ``average=None`` silently returning per-class F1 instead of
the positive-class F1 the paper reports.
"""

from __future__ import annotations

import statistics
from pathlib import Path

import pytest

from clinicalbench.eval.metrics import score_file
from clinicalbench.naming import result_path

RESULTS_ROOT = Path(__file__).parent.parent / "results"
pytestmark = pytest.mark.skipif(
    not RESULTS_ROOT.exists(), reason="released results not present"
)

# Table 1, MIMIC-III: (mean, 95% CI) over random_index 0-4.
TABLE_1 = [
    ("length_pred", "XGBoost", 67.94, (67.87, 68.01)),
    ("length_pred", "Meta-Llama-3-8B-Instruct", 25.78, (25.72, 25.84)),
]


@pytest.mark.parametrize("task,model,paper_mean,ci", TABLE_1)
def test_five_run_mean_matches_table_1(task, model, paper_mean, ci):
    f1s = [
        score_file(result_path(RESULTS_ROOT, task, "mimic3", model, i), task).f1 * 100
        for i in range(5)
    ]
    mean = statistics.mean(f1s)
    assert ci[0] <= mean <= ci[1], f"{model} {task}: {mean:.2f} outside published CI {ci}"
    assert mean == pytest.approx(paper_mean, abs=0.01)


def test_traditional_baseline_beats_the_llm():
    """The paper's headline claim, checked against the released files."""
    task = "mortality_pred"
    xgb = score_file(result_path(RESULTS_ROOT, task, "mimic3", "XGBoost", 0), task)
    llm = score_file(
        result_path(RESULTS_ROOT, task, "mimic3", "Meta-Llama-3-8B-Instruct", 0), task
    )
    assert xgb.f1 > llm.f1
    assert xgb.n == llm.n, "baseline and LLM must be scored on the same test set"


@pytest.mark.parametrize(
    "task,dataset,model,index,mode,temperature,ratio",
    [
        ("mortality_pred", "mimic3", "Meta-Llama-3-8B-Instruct", 0, "ORI", None, None),
        ("mortality_pred", "mimic3", "Meta-Llama-3-8B-Instruct", 6, "COT", None, None),
        ("mortality_pred", "mimic3", "Meta-Llama-3-8B-Instruct", 6, "SR", None, None),
        ("readmission_pred", "mimic4", "Meta-Llama-3-8B-Instruct", 6, "ICL", None, None),
        # the shapes the old evaluator could not open at all:
        ("mortality_pred", "mimic3", "BioMistral-7B", 0, "ORI", 0.2, None),
        ("mortality_pred", "mimic3", "BioMistral-7B", 0, "ORI", 1.0, None),
        ("mortality_pred", "mimic3", "XGBoost", 0, "ORI", None, 0.4),
    ],
)
def test_every_released_result_shape_is_scoreable(
    task, dataset, model, index, mode, temperature, ratio
):
    path = result_path(RESULTS_ROOT, task, dataset, model, index, mode, temperature, ratio)
    score = score_file(path, task)
    assert score.n > 0
    assert 0.0 <= score.f1 <= 1.0
    assert 0 <= score.n_invalid <= score.n


def test_auroc_needs_a_probability_column():
    path = result_path(RESULTS_ROOT, "length_pred", "mimic3", "Yi-1.5-34B-Chat", 0)
    with pytest.raises(ValueError, match="PROB"):
        score_file(path, "length_pred", with_auroc=True)


# Table 5, Meditron-70B base row on the 500-sample cohort. These files were
# originally named `Llama3-meditron-70b`; matching all six published numbers is
# what identified them as this model's runs, and they now carry its canonical
# name.
TABLE_5_MEDITRON_70B = [
    ("length_pred", "mimic3", 27.23),
    ("length_pred", "mimic4", 34.52),
    ("mortality_pred", "mimic3", 46.15),
    ("mortality_pred", "mimic4", 34.48),
    ("readmission_pred", "mimic3", 9.64),
    ("readmission_pred", "mimic4", 9.90),
]


@pytest.mark.parametrize("task,dataset,paper_f1", TABLE_5_MEDITRON_70B)
def test_meditron_70b_reproduces_table_5(task, dataset, paper_f1):
    path = result_path(RESULTS_ROOT, task, dataset, "meditron-70b", 6, "ORI")
    assert path.exists(), "Meditron-70B index-6 results missing"
    assert score_file(path, task).f1 * 100 == pytest.approx(paper_f1, abs=0.01)


def test_every_config_resolves_against_the_released_files():
    """Each paper config must find the results it stands for, bar Figure 4."""
    import yaml

    from clinicalbench.experiments import expand, load_roster

    root = Path(__file__).parent.parent
    roster = load_roster(root / "configs/models.yaml")
    for config_path in sorted((root / "configs/paper").glob("*.yaml")):
        runs = list(expand(yaml.safe_load(open(config_path)), roster))
        missing = [r for r in runs if not r.result_file(RESULTS_ROOT).exists()]
        if config_path.stem == "figure_4":
            # the fine-tuned adapters were never released, so none of its cells
            # can resolve; this pins that it is the *only* config in that state
            assert len(missing) == len(runs) > 0
            continue
        assert not missing, (
            f"{config_path.name}: {len(missing)}/{len(runs)} cells have no result file, "
            f"e.g. {missing[0].result_file(RESULTS_ROOT)}"
        )
