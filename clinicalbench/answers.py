"""Turning raw model output into a scored prediction.

This is the single place where the benchmark's two scoring conventions live.
Both were previously reimplemented, slightly differently, in ``test.py``,
``test_withprob.py`` and ``calculate.py``.
"""

from __future__ import annotations

from typing import List, Optional

from clinicalbench.config import TaskSpec


def extract_answer(text: str, task_spec: TaskSpec, scan: bool = True) -> Optional[str]:
    """Return the model's answer digit in ``text``, or ``None`` if there is none.

    ``scan=True`` walks backwards to the last valid digit anywhere in the text,
    which is how the generative path reads a chain-of-thought answer: it picks
    up the final answer rather than a digit mentioned mid-reasoning.

    ``scan=False`` looks only at the final character. The logits path used this,
    and the two disagree on outputs such as ``"1."`` -- backward scan reads
    ``1``, last-character reads ``.`` and scores the row invalid. The
    distinction is preserved so each path reproduces its published numbers.
    """
    valid = task_spec.answer_tokens
    if not text:
        return None
    if not scan:
        return text[-1] if text[-1] in valid else None
    for char in reversed(text):
        if char in valid:
            return char
    return None


def penalise_invalid(gold: str, task_spec: TaskSpec) -> str:
    """The prediction recorded when a model fails to emit a valid answer.

    ClinicalBench scores a malformed answer as *wrong* rather than dropping the
    row, so that a model cannot raise its score by declining to answer. The
    recorded prediction is therefore a deliberately incorrect label chosen from
    the gold label:

    * binary tasks -> the opposite label;
    * length-of-stay -> ``2`` when the gold is ``1``, otherwise ``1``.

    Because the substituted value is derived from the gold label, these rows
    carry no information about the model beyond "it did not answer". The count
    of such rows is reported by :func:`clinicalbench.eval.metrics.score_file`
    and should be quoted alongside any headline number.
    """
    if task_spec.is_multiclass:
        return "2" if gold == "1" else "1"
    return "0" if gold == "1" else "1"


def resolve_prediction(
    raw: str, gold: str, task_spec: TaskSpec, scan: bool = True
) -> tuple[str, bool]:
    """Return ``(prediction, was_invalid)`` for one model output."""
    answer = extract_answer(raw, task_spec, scan=scan)
    if answer is None:
        return penalise_invalid(gold, task_spec), True
    return answer, False
