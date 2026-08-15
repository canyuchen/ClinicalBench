"""The prompt builder must reproduce the released prompts exactly.

The few-shot exemplar files under ``clinicalbench/data/templates/icl/`` are
themselves complete prompts followed by their gold answers, which makes them a
free regression fixture: rebuilding each shot from its parsed fields has to
return the same bytes.
"""

from __future__ import annotations

import re

import pytest

from clinicalbench.config import DATASET_SPECS, DATASETS, TASK_SPECS, TASKS
from clinicalbench.data.prompts import build_prompt, convert_gender, load_icl_prefix
from clinicalbench.inference.modes import apply_mode

SHOT_RE = re.compile(
    r"Age: (.+?)\nGender: (.+?)\nConditions: (.*?)\nProcedures: (.*?)\nUsing Drugs: (.*?)\n",
    re.S,
)
GENDER_BACK = {"male": "M", "female": "F"}


@pytest.mark.parametrize("task", TASKS)
@pytest.mark.parametrize("dataset", DATASETS)
def test_exemplars_rebuild_byte_identically(task, dataset):
    task_spec, dataset_spec = TASK_SPECS[task], DATASET_SPECS[dataset]
    text = load_icl_prefix(task, dataset)
    shots = [s for s in text.split("Patient information:\n") if s.strip()]
    assert shots, f"no exemplars found for {task}/{dataset}"

    for shot in shots:
        m = SHOT_RE.match(shot)
        assert m, f"unparseable exemplar in {task}/{dataset}"
        age, gender, conditions, procedures, drugs = m.groups()
        rebuilt = build_prompt(
            sample={"age": age, "gender": GENDER_BACK[gender]},
            task_spec=task_spec,
            dataset_spec=dataset_spec,
            conditions=conditions.split(", "),
            procedures=procedures.split(", "),
            drugs=drugs.split(", "),
        )
        # the stored shot is the prompt plus the gold answer
        assert shot.startswith(rebuilt.split("Patient information:\n", 1)[1])


def test_empty_code_list_keeps_the_released_quirk():
    """An empty list yields "Conditions" -- the trailing ", " was chopped blind."""
    prompt = build_prompt(
        sample={"age": 70, "gender": "F"},
        task_spec=TASK_SPECS["mortality_pred"],
        dataset_spec=DATASET_SPECS["mimic3"],
        conditions=[], procedures=["Ventilation"], drugs=["opioids"],
    )
    assert "\nConditions\n" in prompt


def test_mimic4_skips_blank_conditions():
    kwargs = dict(
        sample={"age": 70, "gender": "F"},
        task_spec=TASK_SPECS["mortality_pred"],
        conditions=["Pneumonia", "", "Sepsis"],
        procedures=["Ventilation"], drugs=["opioids"],
    )
    assert "Conditions: Pneumonia, Sepsis\n" in build_prompt(
        dataset_spec=DATASET_SPECS["mimic4"], **kwargs
    )
    assert "Conditions: Pneumonia, , Sepsis\n" in build_prompt(
        dataset_spec=DATASET_SPECS["mimic3"], **kwargs
    )


def test_gender_mapping():
    assert convert_gender("M") == "male"
    assert convert_gender("F") == "female"


@pytest.mark.parametrize("task", TASKS)
def test_modes_preserve_the_patient_profile(task):
    spec = TASK_SPECS[task]
    profile = "Patient information:\nAge: 70\nGender: female\nConditions: Sepsis\n"
    prompt = profile + spec.prompt_suffix()

    assert apply_mode(prompt, "ORI", spec) == prompt
    assert apply_mode(prompt, "RP", spec).endswith(prompt)
    for mode in ("COT", "SR"):
        out = apply_mode(prompt, mode, spec)
        assert profile in out, f"{mode} truncated the patient profile"
        assert not out.endswith(spec.answer_cue)


@pytest.mark.parametrize("task", TASKS)
def test_mode_rejects_a_prompt_that_lost_its_answer_block(task):
    """Guards the failure the original character-count slicing could not see."""
    with pytest.raises(ValueError, match="diverged"):
        apply_mode("Patient information:\nAge: 70\n", "COT", TASK_SPECS[task])
