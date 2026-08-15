# Reproducing the paper

Every table and figure has a config under `configs/paper/` that expands into the
exact runs behind it, so the mapping from a published number to a command is
mechanical.

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
| `table_1.yaml` | Table 1 — main results, MIMIC-III | index 0–4 | 360 | **360 (100%)** |
| `table_2.yaml` | Table 2 — main results, MIMIC-IV (appendix) | index 0–4 | 360 | **360 (100%)** |
| `table_4.yaml` | Table 4 — LLM scale vs baselines | index 0 | 96 | **96 (100%)** |
| `table_5.yaml` | Table 5 — prompt engineering | index 6 | 144 | **144 (100%)** |
| `table_6.yaml` | Tables 6–8 — training-set scaling (appendix) | index 0–4 | 1320 | **1320 (100%)** |
| `figure_3.yaml` | Figure 3 — decoding temperature | index 0 | 225 | **225 (100%)** |
| `figure_4.yaml` | Figure 4 — fine-tuning | index 6 | 24 | 0 |

2505 of 2529 runs (99.1%) can be re-scored from what ships in `results/`.

### The remaining gap

**Figure 4 — all 24 cells.** Fine-tuning is done with LLaMA-Factory, which is
not vendored here, and the trained adapters are not released. See
[`fine_tuning.md`](fine_tuning.md), which does include the dataset export. Each
run needs its own `--lora_path`, so use this config as a checklist rather than
with `--run`.

### A note on Meditron-70B's filenames

Meditron-70B's `random_index 6` results — its Table 5 base row and its four
prompt-engineering rows — were originally written under the filename
`Llama3-meditron-70b`, which does not match its checkpoint id
(`epfl-llm/meditron-70b`). Scoring them reproduces the published Meditron-70B
row exactly on all six columns, which is what identified them; they have since
been renamed to the canonical `meditron-70b`, so one model now has one name
throughout. `tests/test_reproduction.py` pins those six numbers.

## Individual runs

An LLM on one task and cohort:

```shell
python -m clinicalbench.inference.llm \
    --base_model meta-llama/Meta-Llama-3-8B-Instruct \
    --task mortality_pred --dataset mimic3 \
    --mode ORI --scoring logits --random_index 0
```

`--scoring logits` records the `PROB` column that AUROC needs. Chain-of-thought
and self-reflection must use `--scoring generate`, because the answer is
embedded in prose:

```shell
python -m clinicalbench.inference.llm \
    --base_model meta-llama/Meta-Llama-3-8B-Instruct \
    --task mortality_pred --dataset mimic3 \
    --mode COT --scoring generate --random_index 6
```

All eleven traditional baselines on one cohort:

```shell
python -m clinicalbench.baselines.traditional \
    --task mortality_pred --dataset mimic3 --random_index 0
# or a subset, and a reduced training set:
python -m clinicalbench.baselines.traditional \
    --task mortality_pred --dataset mimic3 --random_index 0 --ratio 0.4 --models XGBoost SVM
```

Scoring one result file:

```shell
python -m clinicalbench.eval.metrics \
    --base_model meta-llama/Meta-Llama-3-8B-Instruct \
    --task mortality_pred --dataset mimic3 --random_index 0 --auroc
```

## Modes

| `--mode` | What it does | Cohort | `--scoring` |
| --- | --- | --- | --- |
| `ORI` | the prompt as-is | any | `logits` |
| `ICL` | few-shot exemplars prepended | 6 | `logits` |
| `RP` | "Imagine that you are a doctor…" prefix | 6 | `logits` |
| `COT` | asks for reasoning steps before the answer | 6 | `generate` |
| `SR` | answer, reflect, answer again | 6 | `generate` |
| `LORA` | merges an adapter before inference | 6 | `logits` |

`ICL`, `RP`, `COT` and `SR` force `random_index=6` regardless of what you pass,
because they are far slower per sample and were only ever run on the
500-sample cohort.

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
