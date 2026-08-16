# Reproducing the paper

This page is for re-deriving the numbers printed in the paper. To evaluate your
own model or your own settings, see [running.md](running.md).

Every table and figure has a config under `configs/paper/` that expands into the
exact runs behind it, so the mapping from a published number to a command is
mechanical.

The released result files are not in git; fetch them first:

```shell
clinicalbench-fetch-results
```

They come from [`canyuchen/clinicalbench-results`](https://huggingface.co/datasets/canyuchen/clinicalbench-results) on the Hub, a gated
dataset (accept the terms once, approval is automatic), and land in `results/`,
so the default `--result_root` keeps working.

## The three verbs

```shell
# what runs does this table stand for?
python -m clinicalbench.experiments configs/paper/table_1.yaml

# which of them already have a result file?
python -m clinicalbench.experiments configs/paper/table_1.yaml --check

# run the missing ones
python -m clinicalbench.experiments configs/paper/table_1.yaml --run --skip-existing
```

And to turn released results back into the published table:

```shell
python -m clinicalbench.eval.aggregate configs/paper/table_1.yaml --task mortality_pred --dataset mimic3
```

```
dataset  task              model                mode  runs            F1 (95% CI)         AUROC (95% CI)   inv%
mimic3   mortality_pred    XGBoost              ORI      5   65.75 (63.85, 67.65)   95.97 (95.55, 96.39)   0.0%
mimic3   mortality_pred    SVM                  ORI      5   63.97 (62.37, 65.57)   95.69 (95.27, 96.11)   0.0%
mimic3   mortality_pred    gemma-2-9b-it        ORI      5   43.03 (42.35, 43.71)   86.46 (85.94, 86.99)   0.0%
mimic3   mortality_pred    Meta-Llama-3-8B-...  ORI      5   25.81 (25.55, 26.06)   85.40 (84.68, 86.12)   0.0%
```

## Config map

| Config | Paper | Cohort | Runs | Released |
| --- | --- | --- | ---: | ---: |
| `table_1.yaml` | Table 1: main results, MIMIC-III | index 0–4 | 360 | **360 (100%)** |
| `table_2.yaml` | Table 2: main results, MIMIC-IV (appendix) | index 0–4 | 360 | **360 (100%)** |
| `table_4.yaml` | Table 4: LLM scale vs baselines | index 0 | 96 | **96 (100%)** |
| `table_5.yaml` | Table 5: prompt engineering | index 6 | 144 | **144 (100%)** |
| `table_6.yaml` | Tables 6–8: training-set scaling (appendix) | index 0–4 | 1320 | **1320 (100%)** |
| `figure_3.yaml` | Figure 3: decoding temperature | index 0 | 225 | **225 (100%)** |

All 2,505 runs in the table above can be re-scored from what ships in `results/`.

### Figure 4 is not in that table

Fine-tuning is done with LLaMA-Factory, which is not vendored here, and the
trained adapters were not released, so none of its 24 cells can be re-scored
from `results/`. `configs/paper/figure_4.yaml` still describes them, and
[`fine_tuning.md`](fine_tuning.md) covers the dataset export and the training
setup. Each run needs its own `--lora_path`, so use that config as a checklist
rather than with `--run`.

### A note on Meditron-70B's filenames

Meditron-70B's `random_index 6` results (its Table 5 base row and its four
prompt-engineering rows) were originally written under the filename
`Llama3-meditron-70b`, which does not match its checkpoint id
(`epfl-llm/meditron-70b`). Scoring them reproduces the published Meditron-70B
row exactly on all six columns, which is what identified them; they have since
been renamed to the canonical `meditron-70b`, so one model now has one name
throughout. `tests/test_reproduction.py` pins those six numbers.

## Re-running a single published cell

The flags, the prompting modes and the two scoring paths are documented in
[running.md](running.md). To reproduce one cell, use the settings its config
lists. For example, the Table 1 Llama3-8B row on MIMIC-III, split 0:

```shell
clinicalbench-llm \
    --base_model meta-llama/Meta-Llama-3-8B-Instruct \
    --task mortality_pred --dataset mimic3 \
    --mode ORI --scoring logits --random_index 0
```

## Determinism

Everything except Figure 3 uses greedy decoding, so results are deterministic
given the same weights and library versions. Figure 3 samples and is not
bit-reproducible.

Exact library versions from the original runs were not recorded. If you
re-run anything, capture yours:

```shell
pip freeze > requirements.lock
```

## Compute

The published experiments used eight NVIDIA RTX A6000 GPUs. Rough guide: the
full `table_1.yaml` grid is 360 runs, of which 225 are LLM inference passes over
test sets of 4.4k–5.1k prompts. The 70B checkpoints in `table_4.yaml` dominate
the total. `table_6.yaml` is 1320 baseline fits and runs on CPU in hours, not
days.
