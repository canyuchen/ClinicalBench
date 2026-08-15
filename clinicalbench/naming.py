"""Result file naming -- the single source of truth for both writer and reader.

The released code built these names twice: the runner wrote ``..._temp_0.2.csv``
while the evaluator looked for ``..._temp0.2.csv``, so no temperature result
could be scored by the shipped scripts. Both sides now call in here.

Layout::

    results/{task}/{dataset}/{task}_result_data_{model}_{index}{mode}{temp}.csv
    results/{task}/{dataset}/{task}_result_data_{model}_{index}{ratio}.csv

``{mode}`` is empty for ORI, ``{temp}`` is empty for greedy decoding, and
``{ratio}`` is empty at the full training-set size.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

MODE_SUFFIXES = {
    "ORI": "",
    "ICL": "_ICL",
    "COT": "_COT",
    "RP": "_RP",
    "SR": "_SR",
    "LORA": "_LORA",
}


def mode_suffix(mode: str) -> str:
    try:
        return MODE_SUFFIXES[mode]
    except KeyError:
        raise ValueError(f"unknown mode {mode!r}; expected one of {sorted(MODE_SUFFIXES)}")


def temperature_suffix(temperature: Optional[float]) -> str:
    """``_temp_0.2``. Greedy decoding (``None`` or ``0``) adds nothing."""
    if not temperature:
        return ""
    return f"_temp_{temperature}"


def ratio_suffix(ratio: Optional[float]) -> str:
    """``_0.4`` for a training-set fraction. The full set adds nothing."""
    if ratio is None or ratio == 1:
        return ""
    return f"_{ratio}"


def result_filename(
    task: str,
    model_name: str,
    random_index: int,
    mode: str = "ORI",
    temperature: Optional[float] = None,
    ratio: Optional[float] = None,
) -> str:
    return (
        f"{task}_result_data_{model_name}_{random_index}"
        f"{mode_suffix(mode)}{temperature_suffix(temperature)}{ratio_suffix(ratio)}.csv"
    )


def result_path(
    result_root: Union[str, Path],
    task: str,
    dataset: str,
    model_name: str,
    random_index: int,
    mode: str = "ORI",
    temperature: Optional[float] = None,
    ratio: Optional[float] = None,
) -> Path:
    return Path(result_root) / task / dataset / result_filename(
        task, model_name, random_index, mode, temperature, ratio
    )
