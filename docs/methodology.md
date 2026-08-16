# Methodology

Decisions that affect how a number should be read. Several were implicit in the
original scripts; they are written down here because they change what the
metrics mean.

## Invalid answers are scored wrong, not dropped

An LLM asked for `0` or `1` sometimes returns prose, an empty string, or a
number outside the label space. ClinicalBench records such a row as a
**deliberately incorrect** prediction rather than discarding it: for a binary
task the opposite of the gold label, and for length-of-stay `2` when the gold is
`1` and `1` otherwise.

The alternative, dropping unparseable rows, lets a model raise its score by
refusing to answer the cases it finds hard, which would make weak and strong
models incomparable. The cost is that the substituted value is derived from the
gold label, so those rows carry no information about the model beyond "it did
not answer".

Every scoring path therefore reports the count:

```
invalid   37 (7.4%) rows had no parseable answer and are scored wrong
```

**Quote that number alongside any headline F1.** A model at 7% invalid is being
measured on a different thing than one at 0%. The implementation is in
[`clinicalbench/answers.py`](../clinicalbench/answers.py).

## Two ways of reading an answer

| `--scoring` | How | Used for | Records `PROB`? |
| --- | --- | --- | --- |
| `logits` | one forward pass; argmax at the final position | ORI, ICL, RP, LORA | yes, so AUROC is available |
| `generate` | decode up to 512 tokens, parse the answer out of the text | COT, SR, temperature sweeps | no, F1 only |

The two extract answers differently, and this is deliberate: `generate` scans
backwards for the last valid digit, so a chain-of-thought answer is read from
its conclusion rather than from a digit mentioned mid-reasoning; `logits` reads
only the final character. They disagree on output like `"1."`: the backward
scan reads `1`, the last-character rule scores it invalid. Each path keeps the
behaviour that produced its published numbers.

`PROB` is a softmax over **only the answer tokens**, not the full vocabulary, so
it is the model's relative preference between the permitted answers. For
length-of-stay, AUROC binarises the task as "stays longer than two weeks".

## Answer-token lookup

`--scoring logits` needs the token ids for `"0"`, `"1"`, `"2"`, `"3"`. The
published runs used `tokenizer.convert_tokens_to_ids`, which on SentencePiece
vocabularies often has no entry for a bare digit (the token is `"▁1"` with a
word-boundary marker) and silently returns the unknown-token id. When that
happens the `PROB` column is read off whatever `<unk>` scores, and AUROC for
that model is not meaningful.

The runner defaults to `--token_id_mode legacy` so it reproduces the published
numbers, and **prints a warning naming the affected tokens** whenever the lookup
falls back to `<unk>`. `--token_id_mode auto` encodes each answer string and
takes its final token id instead; use it for new work, and expect different
numbers for affected models.

## Model selection

Traditional baselines are refit under 20 seeds and the checkpoint with the best
**validation** F1 is scored on test. Six of the eleven (`XGBoost`,
`LogisticRegression`, `AdaBoost`, `SVM`, `NaiveBayes`, `KNN`) fit
deterministically here, so they are fit once. The two torch baselines
(Transformer, RNN) train 20 epochs under the same best-on-validation rule.

LLMs get no equivalent selection: they are evaluated zero-shot, one pass. The
comparison is therefore between a tuned baseline and an untuned LLM, which is
the intended reading: the question is whether an LLM can match a baseline that
a practitioner would actually have tuned.

## Decoding

Every experiment except the temperature sweep uses greedy decoding
(`do_sample=False`), so results are deterministic given the same weights and
library versions. Figure 3 samples, so its numbers are not bit-reproducible; the
trend is stable, the exact values are not.

## Training-set fractions

The scaling tables keep the **first** `ratio` share of each label in file order
rather than drawing a random subsample. The fractions are therefore nested (the
40% set contains the 20% set), so the curves show the effect of adding data to a
fixed core, not the variance across independent draws. This is what produced the
published tables and is preserved.

## Features for the traditional baselines

Conditions, procedures and drugs are bagged separately by `CountVectorizer`
(2000 features each), age is bucketed into six bands, gender into two, and the
five blocks are concatenated. Only the index visit is used, matching what the
LLM prompt shows. Neither side sees the patient's history beyond that visit.

## What the labels mean

| Task | Type | Label |
| --- | --- | --- |
| `length_pred` | 3-way | `1` at most one week, `2` one to two weeks, `3` more than two weeks |
| `mortality_pred` | binary | `1` if the patient dies on this visit |
| `readmission_pred` | binary | `1` if readmitted within two weeks |

Length-of-stay reports macro-F1; the binary tasks report positive-class F1.
Both are chosen because the tasks are strongly imbalanced and accuracy would be
dominated by the majority class.

## A caveat the paper states

Features are derived from ICD codes, which are **administrative** rather than
purely clinical data. Performance may differ with richer clinical
representations, and this bounds how far the LLM-versus-baseline comparison
generalises.
