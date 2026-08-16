# Data preparation

## Getting access

MIMIC-III and MIMIC-IV are credentialed. You need a PhysioNet account, CITI
"Data or Specimens Only Research" training, and a signed data use agreement per
database before you can download anything.

- MIMIC-III v1.4: <https://physionet.org/content/mimiciii/1.4/>
- MIMIC-IV v3.0: <https://physionet.org/content/mimiciv/3.0/>

No patient data is downloaded by this repository. What ships here is the cohort
index files under `data/` and the model outputs under `results/`.

## Running the pipeline

```shell
scripts/prepare_data.sh \
    --mimic3 /path/to/mimic-iii/1.4 \
    --mimic4 /path/to/mimic-iv/3.0/hosp
```

Either database may be omitted. The script aborts on the first failure.

It runs three stages, which you can also invoke individually:

**1. Build cohorts.** Reads the raw tables through the vendored PyHealth
readers, derives the three tasks, and writes `data/{task}/{dataset}/{task}_data.json`.

```shell
python -m clinicalbench.data.build_cohort --mimic3_path /path/to/mimic-iii/1.4
```

Diagnoses, procedures and prescriptions are read; NDC drug codes are mapped to
level-3 ATC classes so prompts name drug classes rather than product codes.

**2. Render prompts.** Turns each sample into the text an LLM sees, writing
`{task}_data.csv` and, with `--both`, the few-shot `{task}_data_ICL.csv`.

```shell
python -m clinicalbench.data.make_prompts \
    --task mortality_pred --dataset mimic3 --mimic_path /path/to/mimic-iii/1.4 --both
```

A prompt looks like:

```
Patient information:
Age: 83
Gender: male
Conditions: Pneumonia, organism unspecified, Congestive heart failure, ...
Procedures: Continuous invasive mechanical ventilation for less than 96 ...
Using Drugs: beta blocking agents, opioids, high-ceiling diuretics, ...
Will the patient die because of the above situation?
Answer 1 if yes, 0 if no. Answer with only the number.
Answer:
```

The task wording lives in `clinicalbench/config.py` and the few-shot exemplars
in `clinicalbench/data/templates/icl/`. Both are data, not code, so the six
task-by-database combinations share one implementation.

**3. Generate splits.** Writes the train/val/test visit-id index files.

```shell
python -m clinicalbench.data.make_splits --all
```

These are already in the repository; regenerating overwrites them with
identical files. `pytest tests/test_config.py` checks the sizes against what
ships.

## How the cohorts are built

For each label the visit ids are shuffled with a fixed seed, then:

- **train** is taken from the head of the shuffled list, in equal counts per
  label, so training is class-balanced;
- **validation and test** are taken starting at a per-label offset chosen so
  that together they keep the database's natural class prevalence.

For the majority class the visits between the training block and that offset are
deliberately unused. That gap is what lets training be balanced while evaluation
stays at the real prevalence. That matters, because these tasks are heavily
imbalanced and a balanced test set would flatter every model.

`random_index` selects the cohort:

| Index | Seed | Cohort |
| --- | --- | --- |
| 0–4 | 3, 5, 7, 11, 13 | five reshuffles of the full data; the paper's 5-run means and CIs |
| 6 | 19 | a 500-sample test set at the same prevalence, for experiments too slow to run on the full set |

Exact per-label sizes are in `SPLIT_SPECS` in
[`clinicalbench/config.py`](../clinicalbench/config.py).

## Layout after preparation

```
data/{task}/{dataset}/
├── {task}_data.json          # samples (built, not shipped)
├── {task}_data.csv           # prompts (built, not shipped)
├── {task}_data_ICL.csv       # few-shot prompts (built, not shipped)
├── train_index_{0-4,6}.npy   # shipped
├── val_index_{0-4,6}.npy     # shipped
└── test_index_{0-4,6}.npy    # shipped
```

where `{task}` is `length_pred`, `mortality_pred` or `readmission_pred` and
`{dataset}` is `mimic3` or `mimic4`.
