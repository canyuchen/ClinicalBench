# Released results

3,015 result files — every model output behind the paper's tables and figures.
They are here so the numbers can be re-scored without re-running inference:

```shell
python scripts/score_table.py configs/paper/table_1.yaml --task mortality_pred --dataset mimic3
```

## Layout

```
results/{task}/{dataset}/{task}_result_data_{model}_{index}{mode}{temp}.csv
results/{task}/{dataset}/{task}_result_data_{model}_{index}{ratio}.csv
```

| Field | Values |
| --- | --- |
| `{task}` | `length_pred`, `mortality_pred`, `readmission_pred` |
| `{dataset}` | `mimic3`, `mimic4` |
| `{model}` | last path segment of the HuggingFace id, or the baseline name |
| `{index}` | `0`–`4` (cohort reshuffles) or `6` (500-sample cohort) |
| `{mode}` | empty for `ORI`, else `_ICL`, `_COT`, `_RP`, `_SR`, `_LORA` |
| `{temp}` | empty for greedy, else `_temp_0.2` … `_temp_1.0` |
| `{ratio}` | empty for the full training set, else `_0.05` … `_0.4` |

Names are produced by [`clinicalbench/naming.py`](../clinicalbench/naming.py),
which both the runner and the evaluator import — they used to build these
strings separately and disagreed, which made every temperature file unreadable.

## Columns

| Column | In | Meaning |
| --- | --- | --- |
| `SUBJECT_ID` | LLM runs | patient id from the source database |
| `ANSWER` | all | gold label |
| `PREDICTION` | all | scored prediction; an unparseable answer is replaced by a deliberately wrong label (see [methodology](../docs/methodology.md)) |
| `ORIGINAL` | LLM runs | raw model output before parsing |
| `PROB` | `--scoring logits` runs, all baselines | softmax over the answer tokens only; what AUROC uses |

Traditional-baseline files have `ANSWER`, `PREDICTION`, `PROB` only — they are
scored positionally against the test split rather than by patient id.

## Coverage

2,505 of the 2,529 runs the paper's configs stand for (99.1%). Per config:

```shell
python -m clinicalbench.experiments configs/paper/table_1.yaml --check
```

Everything except Figure 4 is complete; that gap is documented in
[docs/reproduction.md](../docs/reproduction.md#the-remaining-gap).

## Models here that are not in the paper

These were run during development and kept; they are not part of any published
table, and the roster in `configs/models.yaml` does not reference them.

`GradientBoosting`, `Llama-2-7b-chat-hf`, `Llama-2-13b-chat-hf`,
`Llama-2-70b-chat-hf`, `Llama3-OpenBioLLM-8B`, `MedLLaMA_13B`, `medalpaca-7b`,
`vicuna-13b-v1.5`

Meditron-70B's `random_index 6` files were originally named
`Llama3-meditron-70b`, which did not match its checkpoint id. They have been
renamed to `meditron-70b`; the contents are unchanged and still reproduce the
published Table 5 row exactly.

Four stray `*_withprob.csv` files use a naming convention no current script
produces; they predate the `--scoring logits` flag.

## Provenance

These files contain per-patient predictions derived from MIMIC-III and MIMIC-IV,
including `SUBJECT_ID` and, for chain-of-thought and self-reflection runs,
generated text that restates parts of the patient record. They are published
here for verification of the paper's numbers. Access to the source databases
themselves requires PhysioNet credentialing — see
[docs/data_preparation.md](../docs/data_preparation.md).
