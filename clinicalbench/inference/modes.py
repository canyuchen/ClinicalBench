"""Prompt-engineering modes applied on top of a base prompt.

The original ``test.py`` and ``test_withprob.py`` each carried an inline copy of
these transforms, with the trimmed span expressed as a hard-coded character
count. Here the spans come from :class:`~clinicalbench.config.TaskSpec`, so a
change to the task wording can no longer silently eat the patient profile.
"""

from __future__ import annotations

from clinicalbench.config import TaskSpec

ROLE_PLAY_PREFIX = (
    "Imagine that you are a doctor. Today, you're seeing a patient with the "
    "following profile:\n"
)

SELF_REFLECTION_INSTRUCTION = (
    "First answer with a number. Then conduct a concise reflection. "
    "Finally output your answer again with a number."
)

COT_INSTRUCTIONS = {
    "length_pred": (
        "\nPlease provide your concise reasoning steps for the prediction"
        "(no more than 3 steps), and finally answer 1 if the patient will stay "
        "no more than one week, answer 2 if more than one week but not more "
        "than two weeks, answer 3 if more than two weeks."
    ),
    "mortality_pred": (
        "\nPlease provide your concise reasoning steps for the prediction"
        "(no more than 3 steps), and finally answer 1 if the patient will die "
        "and 0 otherwise."
    ),
    "readmission_pred": (
        "\nPlease provide your concise reasoning steps for the prediction"
        "(no more than 3 steps), and finally answer 1 if the patient will be "
        "readmitted and 0 otherwise."
    ),
}

#: Modes that need long generations rather than a single answer token.
GENERATIVE_MODES = ("COT", "SR")


def _strip_suffix(prompt: str, suffix: str) -> str:
    """Drop ``suffix`` from the end of ``prompt``.

    The original sliced by a hard-coded length without checking. We check, so a
    template drift raises instead of silently truncating the patient record.
    """
    if not prompt.endswith(suffix):
        raise ValueError(
            "prompt does not end with the expected answer-format block; the "
            "prompt CSV and clinicalbench.config have diverged.\n"
            f"  expected tail: {suffix!r}\n"
            f"  actual tail:   {prompt[-len(suffix):]!r}"
        )
    return prompt[: -len(suffix)]


def apply_mode(prompt: str, mode: str, task_spec: TaskSpec) -> str:
    """Rewrite a base prompt for one prompt-engineering mode.

    ``ORI`` and ``LORA`` use the prompt unchanged; ``ICL`` is handled at data
    generation time because the exemplars are baked into the prompt CSV.
    """
    if mode in ("ORI", "ICL", "LORA"):
        return prompt
    if mode == "RP":
        return ROLE_PLAY_PREFIX + prompt
    if mode == "COT":
        return _strip_suffix(prompt, task_spec.cot_strip) + COT_INSTRUCTIONS[task_spec.name]
    if mode == "SR":
        return _strip_suffix(prompt, task_spec.sr_strip) + "\n" + SELF_REFLECTION_INSTRUCTION
    raise ValueError(f"unknown mode: {mode}")
