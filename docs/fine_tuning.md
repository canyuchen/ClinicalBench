# Fine-tuning (Figure 4)

Figure 4 asks whether fine-tuning closes the gap. Fine-tuning itself is done
with [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory), which is not
vendored here; this page covers the parts that are.

> The released repository shipped neither the dataset export nor the training
> configuration, so this figure could not be reproduced from it. The export is
> now included (below) and the training settings are transcribed from the paper.
> The trained adapters are still not released, so `--check` on
> `configs/paper/figure_4.yaml` reports all 24 cells as missing.

## 1. Export the dataset

```shell
for split in train val test; do
    python -m clinicalbench.data.export_finetune \
        --task length_pred --dataset mimic3 --split $split --random_index 6
done
```

Writes `data/finetune/{task}_{dataset}_{split}_{index}.json` in Alpaca format:

```json
{
  "instruction": "Given the patient information, predict the number of weeks of stay in hospital.\nAnswer 1 if no more than one week,\n...",
  "input": "Patient information:\nAge: 78\nGender: male\nConditions: ...\nAnswer:",
  "output": "1"
}
```

Two things to note. The instruction wording is the paper's fine-tuning phrasing
("Given the patient information, predict …"), which differs slightly from the
inference prompt ("Predict …"); the input is the same patient profile the
inference prompts use, so both settings see the same evidence.

Figure 4 uses `--random_index 6`, the 500-sample cohort, so the fine-tuned
models are directly comparable to the prompt-engineering results in Table 5.

## 2. Register the dataset with LLaMA-Factory

Add to its `data/dataset_info.json`:

```json
"clinicalbench_length_mimic3": {
  "file_name": "length_pred_mimic3_train_6.json",
  "columns": { "prompt": "instruction", "query": "input", "response": "output" }
}
```

## 3. Train

The paper uses two variants of LoRA, both for 20 epochs, selecting the
checkpoint with the best **validation** performance — the same rule and the same
epoch budget as the traditional baselines, so neither side gets more tuning than
the other.

| Variant | What is adapted |
| --- | --- |
| LoRA (Full) | adapters on all target modules |
| LoRA (Last Layer) | adapters on the final layer only |

The smaller variant exists because the training set is small: with only a few
hundred balanced examples, adapting every layer risks overfitting, so the paper
reports both. LoRA (Full) generally helps more on length-of-stay.

```shell
llamafactory-cli train \
    --stage sft --do_train \
    --model_name_or_path meta-llama/Meta-Llama-3-8B-Instruct \
    --dataset clinicalbench_length_mimic3 \
    --template llama3 \
    --finetuning_type lora \
    --output_dir saves/llama3-8b-length-mimic3 \
    --num_train_epochs 20 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --lr_scheduler_type cosine \
    --logging_steps 10 \
    --save_steps 100 \
    --learning_rate 5e-5 \
    --fp16
```

For LoRA (Last Layer), restrict `--lora_target` to the final decoder block's
modules.

> Learning rate, batch size and scheduler are not stated in the paper; the
> values above are LLaMA-Factory defaults for this setup and are a starting
> point, not the published configuration. Record what you use.

## 4. Evaluate

```shell
python -m clinicalbench.inference.llm \
    --base_model meta-llama/Meta-Llama-3-8B-Instruct \
    --lora_path saves/llama3-8b-length-mimic3 \
    --task length_pred --dataset mimic3 \
    --mode LORA --scoring logits --random_index 6
```

The adapter is merged into the base weights before inference. Results land at
`results/{task}/{dataset}/{task}_result_data_{model}_6_LORA.csv` and score like
any other run:

```shell
python -m clinicalbench.eval.metrics \
    --base_model meta-llama/Meta-Llama-3-8B-Instruct \
    --task length_pred --dataset mimic3 --random_index 6 --mode LORA --auroc
```

## Which models

Figure 4 fine-tunes four general-purpose checkpoints — **Llama3-8B, Gemma2-9B,
Vicuna-v1.5-7B and Mistral-v0.3-7B** — on both databases and all three tasks,
which is the `finetune_llms` group in `configs/models.yaml`. The traditional
reference lines in the figure are the index-6 baseline runs already covered by
`configs/paper/table_5.yaml`.

One limitation to plan around: the result filename carries a single `_LORA`
suffix, so it cannot distinguish LoRA (Full) from LoRA (Last Layer). Give each
variant its own `--result_root`.
