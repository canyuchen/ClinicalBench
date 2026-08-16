# Running your own experiments

This page is for evaluating **your own** model, or ours on your own settings.
To re-derive the numbers printed in the paper instead, see
[reproduction.md](reproduction.md).

Everything below assumes the prompts exist under `data/`. If they do not, build
them first: [data_preparation.md](data_preparation.md).

## Evaluating an LLM

Any HuggingFace causal LM works; the paper's roster in
[`configs/models.yaml`](../configs/models.yaml) is a convenience list, not a
restriction.

```shell
clinicalbench-llm \
    --base_model meta-llama/Meta-Llama-3-8B-Instruct \
    --task mortality_pred --dataset mimic3 \
    --mode ORI --scoring logits --random_index 0
```

| Flag | Default | What it does |
| --- | --- | --- |
| `--base_model` | required | HuggingFace id, or a local path |
| `--task` | required | `length_pred`, `mortality_pred`, `readmission_pred` |
| `--dataset` | required | `mimic3` or `mimic4` |
| `--mode` | `ORI` | prompting strategy, see below |
| `--scoring` | `logits` | `logits` or `generate`, see below |
| `--random_index` | `0` | which cohort split, `0`–`4` or `6` |
| `--temperature` | greedy | enables sampling |
| `--lora_path` | none | adapter to merge before inference |
| `--token_id_mode` | `legacy` | how answer tokens are looked up, see [methodology.md](methodology.md#answer-token-lookup) |
| `--device_map` | `auto` | shard a large checkpoint across GPUs |
| `--data_root` / `--result_root` | `data` / `results` | where prompts are read and outputs written |

The output lands at
`results/{task}/{dataset}/{task}_result_data_{model}_{index}{mode}{temp}.csv`,
where `{model}` is the last path segment of the checkpoint id. Naming is
[`clinicalbench/naming.py`](../clinicalbench/naming.py), shared by the runner
and the evaluator, so a file written here is a file the scorer can find.

## Evaluating the traditional baselines

All eleven, on the same cohort, on CPU:

```shell
clinicalbench-baselines --task mortality_pred --dataset mimic3 --random_index 0
```

```shell
# a subset, and a reduced training set
clinicalbench-baselines --task mortality_pred --dataset mimic3 --random_index 0 \
    --ratio 0.4 --models XGBoost SVM
```

| Flag | Default | What it does |
| --- | --- | --- |
| `--models` | all 11 | subset by name |
| `--ratio` | `1.0` | train on the first share of each label, for scaling curves |
| `--random_index` | `0` | which cohort split |

Features are bag-of-codes over the same index visit the LLM prompt describes,
so the two sides are comparable; see
[methodology.md](methodology.md#features-for-the-traditional-baselines).

## Prompting modes

| `--mode` | What it does | Cohort | `--scoring` |
| --- | --- | --- | --- |
| `ORI` | the prompt as-is | any | `logits` |
| `ICL` | few-shot exemplars prepended | 6 | `logits` |
| `RP` | "Imagine that you are a doctor…" prefix | 6 | `logits` |
| `COT` | asks for reasoning steps before the answer | 6 | `generate` |
| `SR` | answer, reflect, answer again | 6 | `generate` |
| `LORA` | merges an adapter before inference | 6 | `logits` |

`ICL`, `RP`, `COT` and `SR` force `random_index=6` regardless of what you pass,
because they are far slower per sample and were only ever run on the 500-sample
cohort.

## The two scoring paths

`--scoring logits` runs a single forward pass and takes the argmax over the
answer tokens at the final position. It also records a `PROB` column, which is
what AUROC needs, so prefer it whenever the mode allows.

`--scoring generate` decodes up to 512 tokens and scans backwards for the last
valid digit. `COT` and `SR` require it, because the answer arrives after the
reasoning. These runs have no `PROB` column and therefore no AUROC.

The two paths extract answers differently; that difference is documented in
[methodology.md](methodology.md#two-ways-of-reading-an-answer).

## Scoring what you ran

One file:

```shell
clinicalbench-score --base_model meta-llama/Meta-Llama-3-8B-Instruct \
    --task mortality_pred --dataset mimic3 --random_index 0 --auroc
```

Unparseable answers are counted as wrong rather than dropped, and the invalid
rate is reported alongside F1. This is a deliberate choice that changes model
rankings; read
[methodology.md](methodology.md#invalid-answers-are-scored-wrong-not-dropped)
before comparing against numbers computed some other way.

To average several cohort splits into a table with confidence intervals, point
`clinicalbench-table` at a config:

```shell
clinicalbench-table configs/paper/table_1.yaml --task mortality_pred --dataset mimic3
clinicalbench-table configs/paper/table_1.yaml --csv > table_1.csv
```

## Running many cells

A config lists a set of (model, task, dataset, mode, index) cells. Add your own
model to the `groups` in [`configs/models.yaml`](../configs/models.yaml), copy a
config from `configs/paper/`, and the runner will expand and execute it:

```shell
python -m clinicalbench.experiments my_config.yaml            # list the runs
python -m clinicalbench.experiments my_config.yaml --check    # which already have results
python -m clinicalbench.experiments my_config.yaml --run --skip-existing
```

`--skip-existing` makes the command resumable: an interrupted sweep can be
re-issued and picks up where it stopped.

## Cost

Rough guide from the published runs, on eight NVIDIA RTX A6000 GPUs: one LLM
pass covers a test set of 4.4k–5.1k prompts, and `--scoring logits` is a single
forward pass per prompt. `COT` and `SR` decode up to 512 tokens per prompt on
the 500-sample cohort instead. Baseline fits are CPU-bound and take minutes.
