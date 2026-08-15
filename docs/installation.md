# Installation

## Environments

ClinicalBench splits its dependencies so that verifying the released results
does not require a GPU stack.

| Install | Command | Gets you |
| --- | --- | --- |
| Base | `pip install -e .` | cohort building, the 11 traditional baselines, and scoring every released result file |
| With LLM inference | `pip install -e ".[llm]"` | the above plus `transformers`, `accelerate`, `peft` for running LLMs |
| Development | `pip install -e ".[llm,dev]"` | the above plus `pytest` |

```shell
conda create -n clinicalbench python=3.10
conda activate clinicalbench
pip install -e ".[llm]"
```

Python 3.8 or newer. The base install pins `pandas<2`, which the vendored
PyHealth readers require.

## Verify the install

```shell
pytest tests/ -q
```

101 tests, no GPU and no MIMIC access needed. They check that the cohort spec
still matches the shipped index files, that the prompt builder reproduces the
released prompts byte-for-byte, and that the evaluator still reproduces the
paper's Table 1 numbers from the released result files.

## Hardware

The published experiments ran on eight NVIDIA RTX A6000 GPUs (48 GB each).

| What you want to do | Needs |
| --- | --- |
| Score the released results | CPU only |
| Traditional baselines | CPU; a GPU speeds up the Transformer and RNN baselines |
| LLMs up to ~9B | one 48 GB GPU |
| 34B / 70B checkpoints | multiple GPUs; `--device_map auto` shards across them |

Model weights are pulled from HuggingFace on first use and cached under
`~/.cache/huggingface/`. Several checkpoints are gated and need you to accept
their terms first. Code-mapping resources (ATC, ICD) download on first cohort
build into `~/.cache/pyhealth/`.

## Note on the vendored PyHealth

`clinicalbench/_vendor/pyhealth/` is a reduced copy of PyHealth 1.1.4, kept so
that cohort construction stays pinned to the code that produced the published
results. Its internal imports are rewritten to `clinicalbench._vendor.pyhealth`,
so installing ClinicalBench alongside a real PyHealth is safe. See
[`NOTICE`](../NOTICE) for what was removed and changed.
