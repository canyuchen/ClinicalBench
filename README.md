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

The gap is not small. On MIMIC-III mortality prediction, scored from the files
in this repository:

| Model | F1 (95% CI) | AUROC (95% CI) |
| --- | --- | --- |
| XGBoost | **65.75** (63.85, 67.65) | **95.97** (95.55, 96.39) |
| SVM | 63.97 (62.37, 65.57) | 95.69 (95.27, 96.11) |
| Gemma2-9B | 43.03 (42.35, 43.71) | 86.46 (85.94, 86.99) |
| Llama3-8B | 25.81 (25.55, 26.06) | 85.40 (84.68, 86.12) |

## What is here

- **All 3,015 result files** behind the paper, so every number can be re-scored
  without a GPU. That covers 99.1% of the runs the paper's configs stand for.
  They live in a gated dataset on the Hub,
  [`canyuchen/clinicalbench-results`](https://huggingface.co/datasets/canyuchen/clinicalbench-results),
  and one command pulls them in.
- **A config per table and figure**, expanding into the exact commands that
  produced it.
- **The cohort index files**, so splits are identical to the published ones.

Raw MIMIC data is not here and cannot be redistributed; see
[docs/data_preparation.md](docs/data_preparation.md).

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

**Re-score the paper without touching a GPU:**

```shell
clinicalbench-fetch-results     # 295 MB from the Hub into results/
python -m clinicalbench.eval.aggregate configs/paper/table_1.yaml --task mortality_pred --dataset mimic3
```

Or read the precomputed metrics without downloading anything:

```python
import pandas as pd
df = pd.read_csv("hf://datasets/canyuchen/clinicalbench-results/summary.csv")
```

**See what a table costs and what is already done:**

```shell
python -m clinicalbench.experiments configs/paper/table_1.yaml --check
# 360/360 cells have a result file under results/
```

**Prepare data** (needs credentialed MIMIC):

```shell
scripts/prepare_data.sh --mimic3 /path/to/mimic-iii/1.4 --mimic4 /path/to/mimic-iv/3.0/hosp
```

**Run one LLM:**

```shell
python -m clinicalbench.inference.llm \
    --base_model meta-llama/Meta-Llama-3-8B-Instruct \
    --task mortality_pred --dataset mimic3 \
    --mode ORI --scoring logits --random_index 0
```

**Run the traditional baselines:**

```shell
python -m clinicalbench.baselines.traditional \
    --task mortality_pred --dataset mimic3 --random_index 0
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
| `configs/paper/figure_4.yaml` | Figure 4: fine-tuning | 24 | 0% |

Full commands, and the one remaining gap, in
[docs/reproduction.md](docs/reproduction.md).

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
