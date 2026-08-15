"""Central specification of tasks, databases and cohort splits.

Everything that used to be duplicated across six ``convert.py`` and three
``get_index.py`` copies lives here as data. The numbers are transcribed
verbatim from the scripts that produced the ``.npy`` index files shipped in
``data/``; changing them changes the cohort and invalidates the released
results.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

TASKS: Tuple[str, ...] = ("length_pred", "mortality_pred", "readmission_pred")
DATASETS: Tuple[str, ...] = ("mimic3", "mimic4")

#: ``random_index`` values 0-4 are five reshuffles of the full cohort; index 6 is
#: the 500-sample test cohort used wherever inference cost made the full test set
#: impractical (prompt engineering, fine-tuning).
RANDOM_INDICES: Tuple[int, ...] = (0, 1, 2, 3, 4, 6)

#: Shuffle seeds for ``random_index`` 0-4, and for the 500-sample cohort.
SEEDS: Tuple[int, ...] = (3, 5, 7, 11, 13)
SMALL_COHORT_SEED: int = 19
SMALL_COHORT_INDEX: int = 6


@dataclass(frozen=True)
class TaskSpec:
    """Label space and prompt wording for one prediction task."""

    name: str
    labels: Tuple[int, ...]
    question: str
    instruction: str
    answer_cue: str
    #: ``f1_score`` averaging mode. Length-of-stay is 3-way so it uses macro-F1;
    #: the binary tasks report F1 of the positive class.
    average: str

    @property
    def is_multiclass(self) -> bool:
        return len(self.labels) > 2

    @property
    def answer_tokens(self) -> List[str]:
        return [str(label) for label in self.labels]

    def prompt_suffix(self) -> str:
        """The trailing block a prompt ends with, after the patient profile.

        Prompt-engineering modes replace part of this suffix. Deriving the
        replaced text here means the modes no longer hard-code character counts,
        which was how the original ``test.py`` trimmed prompts -- a silent
        mismatch there would have truncated the patient profile instead.
        """
        return self.question + self.instruction + self.answer_cue

    @property
    def cot_strip(self) -> str:
        """Text removed from the prompt tail before appending the CoT request.

        Chain-of-thought drops the whole answer-format block and the newline
        that ended the question, leaving the prompt at ``"...hospital."``.
        """
        return "\n" + self.instruction + self.answer_cue

    @property
    def sr_strip(self) -> str:
        """Text removed before appending the self-reflection request.

        Self-reflection keeps the label definitions but drops the
        "answer with only the number" constraint and everything after it, plus
        a directly preceding newline if there is one. For the 3-way task that
        phrase sits in ``answer_cue``; for the binary tasks it ends
        ``instruction``, which is why the removed spans differ in length.
        """
        marker = "Answer with only the number."
        suffix = self.prompt_suffix()
        cut = suffix.rindex(marker)
        if cut > 0 and suffix[cut - 1] == "\n":
            cut -= 1
        return suffix[cut:]


TASK_SPECS: Dict[str, TaskSpec] = {
    "length_pred": TaskSpec(
        name="length_pred",
        labels=(1, 2, 3),
        question="Predict the number of weeks of stay in hospital.\n",
        instruction=(
            "Answer 1 if no more than one week,\n"
            "Answer 2 if more than one week but not more than two weeks,\n"
            "Answer 3 if more than two weeks.\n"
        ),
        answer_cue="Answer with only the number. Answer: ",
        average="macro",
    ),
    "mortality_pred": TaskSpec(
        name="mortality_pred",
        labels=(0, 1),
        question="Will the patient die because of the above situation?\n",
        instruction="Answer 1 if yes, 0 if no. Answer with only the number.\n",
        answer_cue="Answer: ",
        average="binary",
    ),
    "readmission_pred": TaskSpec(
        name="readmission_pred",
        labels=(0, 1),
        question="Will the patient be readmitted to the hospital within two weeks?\n",
        instruction="Answer 1 for yes, 0 for no. Answer with only the number.\n",
        answer_cue="Answer: ",
        average="binary",
    ),
}


@dataclass(frozen=True)
class DatasetSpec:
    """Where the MIMIC dictionary tables live and how their columns are named."""

    name: str
    diagnoses_table: str
    procedures_table: str
    code_column: str
    title_column: str
    #: MIMIC-IV ships blank condition strings that MIMIC-III does not; the
    #: original ``convert.py`` for MIMIC-IV skipped them, so we keep doing that.
    skip_empty_conditions: bool
    #: Table names handed to the PyHealth dataset reader.
    pyhealth_tables: Tuple[str, ...]


DATASET_SPECS: Dict[str, DatasetSpec] = {
    "mimic3": DatasetSpec(
        name="mimic3",
        diagnoses_table="D_ICD_DIAGNOSES.csv",
        procedures_table="D_ICD_PROCEDURES.csv",
        code_column="ICD9_CODE",
        title_column="LONG_TITLE",
        skip_empty_conditions=False,
        pyhealth_tables=("DIAGNOSES_ICD", "PROCEDURES_ICD", "PRESCRIPTIONS"),
    ),
    "mimic4": DatasetSpec(
        name="mimic4",
        diagnoses_table="d_icd_diagnoses.csv",
        procedures_table="d_icd_procedures.csv",
        code_column="icd_code",
        title_column="long_title",
        skip_empty_conditions=True,
        pyhealth_tables=("diagnoses_icd", "procedures_icd", "prescriptions"),
    ),
}


@dataclass(frozen=True)
class SplitSpec:
    """Per-class cohort sizes for one (task, dataset, cohort) combination.

    For each label the visit ids are shuffled, then:

    * ``train`` visits are taken from the head of the shuffled list, so the
      training set is class-balanced;
    * validation and test are taken starting at ``offset``, which is chosen so
      that together they preserve the natural class ratio of the database.

    Visits between ``train`` and ``offset`` for the majority class are
    deliberately unused -- that gap is what keeps val/test at the real
    prevalence while train stays balanced.
    """

    train: Tuple[int, ...]
    offset: Tuple[int, ...]
    val: Tuple[int, ...]
    test: Tuple[int, ...]


# (task, dataset, is_small_cohort) -> SplitSpec
SPLIT_SPECS: Dict[Tuple[str, str, bool], SplitSpec] = {
    # ---- length-of-stay prediction: labels (1, 2, 3) ----
    ("length_pred", "mimic3", False): SplitSpec(
        train=(2980, 2980, 2980), offset=(8400, 4175, 2980),
        val=(1200, 596, 426), test=(2400, 1193, 852),
    ),
    ("length_pred", "mimic4", False): SplitSpec(
        train=(1292, 1292, 1292), offset=(14000, 2278, 1292),
        val=(2000, 325, 185), test=(4000, 651, 369),
    ),
    ("length_pred", "mimic3", True): SplitSpec(
        train=(335, 335, 335), offset=(945, 470, 335),
        val=(135, 67, 48), test=(270, 134, 96),
    ),
    ("length_pred", "mimic4", True): SplitSpec(
        train=(129, 129, 129), offset=(1394, 227, 129),
        val=(199, 32, 18), test=(398, 65, 37),
    ),
    # ---- mortality prediction: labels (0, 1) ----
    ("mortality_pred", "mimic3", False): SplitSpec(
        train=(2100, 2100), offset=(15911, 2100),
        val=(2273, 300), test=(4546, 600),
    ),
    ("mortality_pred", "mimic4", False): SplitSpec(
        train=(700, 700), offset=(19487, 700),
        val=(2784, 100), test=(5568, 200),
    ),
    ("mortality_pred", "mimic3", True): SplitSpec(
        train=(204, 204), offset=(1546, 204),
        val=(221, 29), test=(442, 58),
    ),
    ("mortality_pred", "mimic4", True): SplitSpec(
        train=(61, 61), offset=(1689, 61),
        val=(241, 9), test=(483, 17),
    ),
    # ---- readmission prediction: labels (0, 1) ----
    ("readmission_pred", "mimic3", False): SplitSpec(
        train=(277, 277), offset=(3500, 277),
        val=(500, 40), test=(1000, 79),
    ),
    ("readmission_pred", "mimic4", False): SplitSpec(
        train=(2323, 2323), offset=(14000, 2323),
        val=(2000, 332), test=(4000, 664),
    ),
    ("readmission_pred", "mimic3", True): SplitSpec(
        train=(128, 128), offset=(1622, 128),
        val=(232, 18), test=(463, 37),
    ),
    ("readmission_pred", "mimic4", True): SplitSpec(
        train=(249, 249), offset=(1501, 249),
        val=(214, 36), test=(429, 71),
    ),
}


def split_spec(task: str, dataset: str, random_index: int) -> SplitSpec:
    return SPLIT_SPECS[(task, dataset, random_index == SMALL_COHORT_INDEX)]


def seed_for(random_index: int) -> int:
    if random_index == SMALL_COHORT_INDEX:
        return SMALL_COHORT_SEED
    return SEEDS[random_index]


#: Prompt-engineering / fine-tuning modes. ``small_cohort`` modes are only ever
#: run against ``random_index=6`` because of inference cost.
MODES: Tuple[str, ...] = ("ORI", "ICL", "COT", "RP", "SR", "LORA")
SMALL_COHORT_MODES: Tuple[str, ...] = ("ICL", "COT", "RP", "SR")
