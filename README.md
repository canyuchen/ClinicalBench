# ClinicalBench



[![Homepage](https://img.shields.io/badge/%F0%9F%8F%A0_HOMEPAGE-E8663C?style=for-the-badge&labelColor=E8663C)](https://clinicalbench.github.io)
[![Paper](https://img.shields.io/badge/%F0%9F%93%84_PAPER-C9314A?style=for-the-badge&labelColor=C9314A)](https://arxiv.org/abs/2411.06469)
[![Results](https://img.shields.io/badge/%F0%9F%A4%97_RESULTS-FFAE33?style=for-the-badge&labelColor=FFAE33)](https://huggingface.co/datasets/canyuchen/clinicalbench-results)
[![Venue](https://img.shields.io/badge/%F0%9F%8F%86_VENUE-KDD_2026-4C8BF5?style=for-the-badge&labelColor=4A4A4A)](https://kdd.org/kdd2026/)
[![License](https://img.shields.io/badge/%E2%9A%96%EF%B8%8F_LICENSE-MIT-4C8BF5?style=for-the-badge&labelColor=4A4A4A)](LICENSE)

> TLDR: **Can LLMs Beat Traditional ML Models in Clinical Prediction?** **Not yet.** We discover that both general-purpose and medical LLMs, even with different model scales and temperatures, diverse prompting or fine-tuning strategies, still cannot beat traditional ML models in clinical prediction yet, shedding light on their potential deficiency in clinical reasoning and decision-making.

<a href="https://canyuchen.com">Canyu Chen</a>\*,
<a href="https://openreview.net/profile?id=~Jian_Yu4">Jian Yu</a>\*,
<a href="https://shanchen.dev/">Shan Chen</a>,
<a href="https://scholar.google.com/citations?view_op=list_works&hl=zh-CN&user=HED_458AAAAJ&sortby=pubdate">Che Liu</a>,
<a href="https://scholar.google.com/citations?hl=zh-CN&user=EVj1cNoAAAAJ&view_op=list_works">Zhongwei Wan</a>,
<a href="https://scholar.google.com/citations?hl=en&user=yR_ZhV0AAAAJ">Shuang Zhou</a>,
<a href="https://www.feinberg.northwestern.edu/faculty-profiles/az/profile.html?xid=33821">Yuan Luo</a>,
<a href="https://med.umn.edu/bio/rui-zhang">Rui Zhang</a>,
<a href="https://www.bittermanlab.org/">Danielle S. Bitterman</a>,
<a href="https://wcm-wanglab.github.io/">Fei Wang</a>,
<a href="https://www.cs.emory.edu/~kshu5/">Kai Shu</a>†
<br><sub>\*equal contribution &nbsp;·&nbsp; †corresponding author</sub>

![framework](framework.png)

## Updates

- **2026-08.** `v1.0` restructures the repository into an installable
  `clinicalbench` package: one config per paper table and figure, 99.1% of the
  paper's runs re-scorable from the shipped results without a GPU, 104
  regression tests, and the reproducibility fixes listed in
  [docs/reproduction.md](docs/reproduction.md).
- **2026.** ClinicalBench is accepted at **KDD 2026**.
- **2024-11.** Paper on [arXiv](https://arxiv.org/abs/2411.06469), with the
  first release of code and results.

## Overview

ClinicalBench benchmarks **22 LLMs** (14 general-purpose, 8 medical) against
**11 traditional ML models** on three clinical prediction tasks across two
databases, under matched cohorts, features and evaluation.

| Task | Type | Question |
| --- | --- | --- |
| Length-of-Stay | 3-way | ≤ 1 week, 1–2 weeks, or > 2 weeks? |
| Mortality | binary | Will the patient die on this visit? |
| Readmission | binary | Readmitted within two weeks? |

Databases: [MIMIC-III](https://physionet.org/content/mimiciii/1.4/) and
[MIMIC-IV](https://physionet.org/content/mimiciv/3.0/) (both credentialed).

The gap is not small. On MIMIC-III mortality prediction, scored from the
released result files:

| Model | F1 (95% CI) | AUROC (95% CI) |
| --- | --- | --- |
| XGBoost | **65.75** (63.85, 67.65) | **95.97** (95.55, 96.39) |
| SVM | 63.97 (62.37, 65.57) | 95.69 (95.27, 96.11) |
| Gemma2-9B | 43.03 (42.35, 43.71) | 86.46 (85.94, 86.99) |
| Llama3-8B | 25.81 (25.55, 26.06) | 85.40 (84.68, 86.12) |

## Key features

- **Any HuggingFace causal LM on a clinical prediction task, in one command**:
  `clinicalbench-llm --base_model <hf-id> --task mortality_pred --dataset mimic3`.
  The paper's 22 checkpoints are a roster in
  [configs/models.yaml](configs/models.yaml), not a hard-coded list, so
  evaluating a model we never ran is one id away. 0.5B to 70B tested,
  `--device_map auto` shards across GPUs, and `--lora_path` merges an adapter
  before inference.
- **11 traditional ML baselines on matched inputs**: XGBoost, LogisticRegression,
  DecisionTree, RandomForest, AdaBoost, SVM, NaiveBayes, KNN, NeuralNetwork,
  Transformer and RNN, fit on bag-of-codes features (conditions, procedures and
  drugs bagged separately at **2,000** features each, plus age band and gender)
  built from *the same index visit the LLM prompt describes*. Each is refit under
  a **20-seed** sweep and the best validation-F1 checkpoint is scored. CPU only,
  minutes per cohort.
- **Six prompting strategies, two scoring paths**: `--mode ORI | ICL | COT | RP |
  SR | LORA`. `--scoring logits` takes a single forward pass and records a
  softmax over the answer tokens, which is what AUROC needs; `--scoring generate`
  decodes up to **512** tokens and backward-scans for the answer, which `COT` and
  `SR` require because the answer is buried in prose. `--temperature` sweeps
  decoding, `--lora_path` covers fine-tuning.
- **Raw MIMIC to prompts in one pass**: `scripts/prepare_data.sh --mimic3 …
  --mimic4 …` reads the credentialed PhysioNet tables and writes per-task
  samples, rendered prompts and cohort splits, for **3 tasks × 2 databases**.
  The MIMIC reader is a pruned, vendored PyHealth that cannot shadow an upstream
  install. Raw data is not redistributable; see
  [docs/data_preparation.md](docs/data_preparation.md).
- **Cohorts that don't flatter the model**: the train split is class-balanced,
  but val and test are drawn from an offset that **preserves natural prevalence**,
  so a mortality F1 is not inflated by a resampled test set. Splits are seeded
  (`--random_index 0-4`, seeds 3/5/7/11/13, plus a 500-sample cohort at index 6),
  and the index `.npy` files ship in `data/`, so your split *is* the published
  split. `tests/test_config.py` pins every size.
- **Invalid answers are scored, not dropped**: a model that refuses, hedges, or
  answers `3` on a binary task is counted as a deliberate miss instead of
  vanishing from the denominator, and every aggregated table carries an `inv%`
  column. This changes the ranking, so it is documented rather than buried; see
  [docs/methodology.md](docs/methodology.md).
- **Config-driven experiment matrix**: one YAML per paper table and figure
  expands into the exact runs behind it, with three verbs: list them, `--check`
  which ones you have already run, and `--run --skip-existing` to fill the gaps.
  **2,505 runs** across the six configs, plus **8 console scripts** for the
  individual steps.
- **Every published number re-scorable without a GPU**: all **3,015** result
  files live in a gated Hub dataset,
  [`canyuchen/clinicalbench-results`](https://huggingface.co/datasets/canyuchen/clinicalbench-results),
  fetched by one command; `summary.csv` has F1, AUROC and invalid-rate
  precomputed for browsing without downloading. **104 tests** pin the cohort
  sizes, the byte-exact prompts, and Table 1's headline numbers.

## Repository layout

```
clinicalbench/
├── config.py              task wording, database schemas, cohort split spec
├── naming.py              result-file naming (shared by runner and evaluator)
├── answers.py             answer extraction and the invalid-output penalty
├── experiments.py         expands a paper config into runs
├── data/                  cohort building, prompt rendering, splits, fine-tune export
│   └── templates/icl/     few-shot exemplars, one file per task x database
├── inference/             LLM runner and the prompt-engineering modes
├── baselines/             the 11 traditional ML models and their features
├── eval/                  scoring one result file; aggregating a whole table
└── _vendor/pyhealth/      reduced PyHealth, for reading MIMIC (see NOTICE)

configs/paper/             one config per table and figure
data/{task}/{dataset}/     cohort index files (.npy)
results/                   released model outputs, fetched from the Hub
docs/                      installation, data prep, reproduction, methodology
scripts/                   data-preparation shell entry point
tests/                     104 tests, no GPU or MIMIC access required
```

## Install

```shell
conda create -n clinicalbench python=3.10 && conda activate clinicalbench
pip install -e ".[llm]"     # omit [llm] to only score released results
pytest tests/ -q
```

Details in [docs/installation.md](docs/installation.md).

## Quick start

**1. Build the cohorts and prompts** (needs credentialed MIMIC):

```shell
scripts/prepare_data.sh --mimic3 /path/to/mimic-iii/1.4 --mimic4 /path/to/mimic-iv/3.0/hosp
```

**2. Run an LLM.** Any HuggingFace causal LM, not only the ones in the paper:

```shell
clinicalbench-llm \
    --base_model meta-llama/Meta-Llama-3-8B-Instruct \
    --task mortality_pred --dataset mimic3 \
    --mode ORI --scoring logits --random_index 0
```

**3. Run the traditional baselines** on the same cohort, on CPU:

```shell
clinicalbench-baselines --task mortality_pred --dataset mimic3 --random_index 0
# or a subset: --models XGBoost SVM
```

**4. Score them against each other:**

```shell
clinicalbench-score --base_model meta-llama/Meta-Llama-3-8B-Instruct \
    --task mortality_pred --dataset mimic3 --random_index 0 --auroc
```

**Comparing against the paper** needs no GPU and no MIMIC access:

```shell
clinicalbench-fetch-results     # 295 MB from the Hub into results/
python -m clinicalbench.eval.aggregate configs/paper/table_1.yaml --task mortality_pred --dataset mimic3
python -m clinicalbench.experiments configs/paper/table_1.yaml --check
# 360/360 cells have a result file under results/
```

Or read the precomputed metrics without downloading anything:

```python
import pandas as pd
df = pd.read_csv("hf://datasets/canyuchen/clinicalbench-results/summary.csv")
```

## Reproducing the paper

| Config | Paper | Runs | Released |
| --- | --- | ---: | ---: |
| `configs/paper/table_1.yaml` | Table 1: main results, MIMIC-III | 360 | **100%** |
| `configs/paper/table_2.yaml` | Table 2: main results, MIMIC-IV | 360 | **100%** |
| `configs/paper/table_4.yaml` | Table 4: LLM scale vs baselines | 96 | **100%** |
| `configs/paper/table_5.yaml` | Table 5: prompt engineering | 144 | **100%** |
| `configs/paper/table_6.yaml` | Tables 6–8: training-set scaling | 1320 | **100%** |
| `configs/paper/figure_3.yaml` | Figure 3: decoding temperature | 225 | **100%** |

Every run behind these tables is re-scorable from the released files. Figure 4
is the exception: it needs fine-tuned adapters, which were not released, so
[docs/fine_tuning.md](docs/fine_tuning.md) covers training them yourself.

Full commands in [docs/reproduction.md](docs/reproduction.md).

## Documentation

| Page | Covers |
| --- | --- |
| [installation.md](docs/installation.md) | environments, hardware, verifying the install |
| [data_preparation.md](docs/data_preparation.md) | PhysioNet access, the pipeline, how cohorts are built |
| [reproduction.md](docs/reproduction.md) | config-to-table map, individual runs, modes, determinism |
| [methodology.md](docs/methodology.md) | invalid-answer scoring, the two scoring paths, model selection, caveats |
| [fine_tuning.md](docs/fine_tuning.md) | LLaMA-Factory dataset export, LoRA settings, evaluation |
| [results/README.md](results/README.md) | fetching the results, file naming, columns, coverage |

**Read [methodology.md](docs/methodology.md) before quoting a number.** Two
things there change how results should be read: unparseable LLM answers are
scored as wrong rather than dropped, and the two scoring paths extract answers
differently.

## Models

14 general-purpose LLMs (Llama3 8B/70B, Mistral-v0.3-7B, Gemma2-9B, Qwen2
0.5B/1.5B/7B, Yi-v1.5 6B/9B/34B, Vicuna-v1.5-7B, Phi3.5-mini-3.8B,
InternLM2.5-7B, MiniCPM3-4B), 8 medical LLMs (Meditron 7B/70B, Medllama3-8B,
BioMistral-7B, Med42 8B/70B, BioMedGPT-7B, Internist-7B), and 11 traditional
models (XGBoost, LogisticRegression, DecisionTree, RandomForest, AdaBoost, SVM,
NaiveBayes, KNN, NeuralNetwork, Transformer, RNN).

Checkpoint ids and the roster each table uses are in
[configs/models.yaml](configs/models.yaml).

## Acknowledgments

Built on [PyHealth](https://sunlabuiuc.github.io/PyHealth/); a reduced copy is
vendored under `clinicalbench/_vendor/pyhealth/`. See [NOTICE](NOTICE) for
attribution and the list of modifications.

## License

MIT. See [LICENSE](LICENSE). We do not own any of the datasets used.

## Citation

```bibtex
@inproceedings{chen2026clinicalbench,
  title     = {ClinicalBench: Can LLMs Beat Traditional ML Models in Clinical Prediction?},
  author    = {Chen, Canyu and Yu, Jian and Chen, Shan and Liu, Che and Wan, Zhongwei
               and Zhou, Shuang and Luo, Yuan and Zhang, Rui and Bitterman, Danielle S.
               and Wang, Fei and Shu, Kai},
  booktitle = {Proceedings of the 32nd ACM SIGKDD Conference on Knowledge Discovery
               and Data Mining (KDD '26)},
  year      = {2026}
}
```
