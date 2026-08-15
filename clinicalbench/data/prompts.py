"""Patient-profile prompt construction.

This replaces the prompt-building half of the six near-identical ``convert.py``
scripts. The output is byte-for-byte what those scripts produced -- including
the quirk that an empty code list yields ``"Conditions"`` rather than
``"Conditions: "``, because the original trimmed a trailing ``", "`` blindly.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Sequence

from clinicalbench.config import DatasetSpec, TaskSpec

TEMPLATE_DIR = Path(__file__).parent / "templates"
ICL_DIR = TEMPLATE_DIR / "icl"

PROFILE_HEADER = "Patient information:\n"


def _code_line(prefix: str, codes: Iterable[str], skip_empty: bool = False) -> str:
    """Render ``"<prefix>: a, b, c\\n"``.

    Mirrors the original ``s += f'{x}, '`` loop followed by ``s[:-2] + '\\n'``,
    so a list that contributes nothing leaves the prefix with its ``": "``
    chopped off. That behaviour reached the released prompts, so it is kept.
    """
    line = f"{prefix}: "
    for code in codes:
        if skip_empty and code == "":
            continue
        line += f"{code}, "
    return line[:-2] + "\n"


def convert_gender(gender: str) -> str:
    if gender == "M":
        return "male"
    if gender == "F":
        return "female"
    return str(gender)


def load_icl_prefix(task: str, dataset: str) -> str:
    """Few-shot exemplars prepended in ``ICL`` mode.

    These are fixed exemplars drawn from the source database, held verbatim in
    ``templates/icl/`` so that all six task x database combinations share one
    code path instead of six hard-coded copies.
    """
    path = ICL_DIR / f"{task}.{dataset}.txt"
    if not path.exists():
        raise FileNotFoundError(f"missing ICL exemplar file: {path}")
    return path.read_text()


def build_patient_profile(
    sample: Dict,
    dataset_spec: DatasetSpec,
    conditions: Sequence[str],
    procedures: Sequence[str],
    drugs: Sequence[str],
) -> str:
    """The database-independent patient block shared by every task."""
    return (
        PROFILE_HEADER
        + f"Age: {sample['age']}\n"
        + f"Gender: {convert_gender(sample['gender'])}\n"
        + _code_line("Conditions", conditions, dataset_spec.skip_empty_conditions)
        + _code_line("Procedures", procedures)
        + _code_line("Using Drugs", drugs)
    )


def build_prompt(
    sample: Dict,
    task_spec: TaskSpec,
    dataset_spec: DatasetSpec,
    conditions: Sequence[str],
    procedures: Sequence[str],
    drugs: Sequence[str],
    icl_prefix: str = "",
) -> str:
    """Full prompt: optional few-shot prefix, patient profile, then the question."""
    return (
        icl_prefix
        + build_patient_profile(sample, dataset_spec, conditions, procedures, drugs)
        + task_spec.prompt_suffix()
    )
