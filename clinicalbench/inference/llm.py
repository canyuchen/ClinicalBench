"""Run an LLM over a task's test split.

Merges the former ``test.py`` and ``test_withprob.py``, which were the same
script with two different ways of reading an answer off the model:

``--scoring generate``
    Sample or greedily decode ``max_new_tokens`` and read the answer out of the
    text. Needed for COT and SR, which produce prose. No probability column, so
    these runs are scored by F1 only.

``--scoring logits``
    One forward pass; take the argmax at the final position for the prediction
    and a softmax over just the answer tokens for the probability. Produces the
    PROB column that AUROC needs.

Results go to ``results/{task}/{dataset}/{task}_result_data_{model}_{index}{mode}{temp}.csv``.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from torch.nn import functional as F
from tqdm.auto import tqdm

from clinicalbench.answers import resolve_prediction
from clinicalbench.config import (
    DATASETS,
    MODES,
    SMALL_COHORT_INDEX,
    SMALL_COHORT_MODES,
    TASK_SPECS,
    TASKS,
)
from clinicalbench.inference.modes import GENERATIVE_MODES, apply_mode
from clinicalbench.naming import result_path

#: Long generations for the modes that reason in prose; a single token otherwise.
GENERATIVE_MAX_NEW_TOKENS = 512


def resolve_answer_token_ids(tokenizer, answer_tokens: List[str], mode: str) -> List[int]:
    """Map answer strings such as ``"0"``/``"1"`` to token ids.

    ``legacy`` is ``convert_tokens_to_ids``, what the published runs used. For
    SentencePiece vocabularies the bare digit is often not a token (the token is
    ``"__1"`` with a word-boundary marker), and the call then returns the unknown
    id -- so the probability column is read off whatever ``<unk>`` scores. This
    is silent, which is why ``auto`` exists and why legacy warns.

    ``auto`` encodes each answer string and takes its final token id.
    """
    legacy = tokenizer.convert_tokens_to_ids(answer_tokens)
    unk = getattr(tokenizer, "unk_token_id", None)
    broken = [
        tok for tok, tid in zip(answer_tokens, legacy)
        if tid is None or (unk is not None and tid == unk)
    ]

    if mode == "legacy":
        if broken:
            print(
                f"WARNING: {tokenizer.__class__.__name__} has no single token for "
                f"{broken}; convert_tokens_to_ids returned the unknown id. The PROB "
                f"column (and therefore AUROC) for this run is not meaningful. "
                f"Re-run with --token_id_mode auto for a corrected probability.",
                file=sys.stderr,
            )
        return list(legacy)

    resolved = []
    for tok in answer_tokens:
        ids = tokenizer.encode(tok, add_special_tokens=False)
        if not ids:
            raise ValueError(f"tokenizer produced no ids for answer token {tok!r}")
        resolved.append(ids[-1])
    if broken:
        print(
            f"note: --token_id_mode auto remapped {broken} to ids {resolved}; "
            f"these differ from the published run for this model.",
            file=sys.stderr,
        )
    return resolved


def load_model(base_model: str, lora_path: Optional[str], device_map: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaTokenizer

    model = AutoModelForCausalLM.from_pretrained(
        base_model, device_map=device_map, torch_dtype=torch.float16, trust_remote_code=True,
    )
    if base_model == "chaoyi-wu/MedLLaMA_13B":
        # this checkpoint ships a config the auto tokenizer misreads
        tokenizer = LlamaTokenizer.from_pretrained(base_model, trust_remote_code=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)

    if lora_path:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, lora_path).merge_and_unload()
    return model, tokenizer


def _pad_token_id(model) -> int:
    eos = model.config.eos_token_id
    return eos[0] if isinstance(eos, list) else eos


def load_test_index(data_root: Path, task: str, dataset: str, random_index: int) -> set:
    path = data_root / task / dataset / f"test_index_{random_index}.npy"
    with open(path, "rb") as f:
        return set(np.load(f).tolist())


def iter_prompts(data_root: Path, task: str, dataset: str, icl: bool):
    suffix = "_ICL" if icl else ""
    path = data_root / task / dataset / f"{task}_data{suffix}.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found -- run `python -m clinicalbench.data.make_prompts` first"
        )
    with open(path, newline="") as f:
        total = sum(1 for _ in csv.reader(f)) - 1
    with open(path) as f:
        yield total
        yield from csv.DictReader(f)


@torch.no_grad()
def predict_generate(model, tokenizer, prompt, device, temperature, max_new_tokens):
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    outputs = model.generate(
        **inputs,
        do_sample=bool(temperature),
        temperature=temperature,
        max_new_tokens=max_new_tokens,
        pad_token_id=_pad_token_id(model),
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)[len(prompt):], None


@torch.no_grad()
def predict_logits(model, tokenizer, prompt, device, answer_token_ids):
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    logits = model(**inputs).logits[0, -1]
    text = tokenizer.decode(logits.argmax(dim=-1))
    # probability of the positive/highest class, renormalised over answer tokens only
    probs = F.softmax(logits[answer_token_ids], dim=0)
    return text, float(probs[-1])


def run(args) -> None:
    task_spec = TASK_SPECS[args.task]
    model_name = args.base_model.split("/")[-1]
    random_index = args.random_index
    if args.mode in SMALL_COHORT_MODES:
        # prompt-engineering runs are only ever scored on the 500-sample cohort
        random_index = SMALL_COHORT_INDEX

    out_path = result_path(
        args.result_root, args.task, args.dataset, model_name,
        random_index, args.mode, args.temperature,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    model, tokenizer = load_model(args.base_model, args.lora_path, args.device_map)
    answer_token_ids = (
        resolve_answer_token_ids(tokenizer, task_spec.answer_tokens, args.token_id_mode)
        if args.scoring == "logits" else None
    )

    test_index = load_test_index(args.data_root, args.task, args.dataset, random_index)
    rows = iter_prompts(args.data_root, args.task, args.dataset, icl=args.mode == "ICL")
    total = next(rows)

    max_new_tokens = GENERATIVE_MAX_NEW_TOKENS if args.mode in GENERATIVE_MODES else 1
    fields = ["SUBJECT_ID", "ANSWER", "PREDICTION", "ORIGINAL"]
    if args.scoring == "logits":
        fields.append("PROB")

    n_invalid = 0
    preds: List[int] = []
    golds: List[int] = []

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in tqdm(rows, total=total, desc=f"{model_name} {args.task}/{args.dataset}"):
            if row["VISIT_ID"] not in test_index:
                continue
            prompt = apply_mode(row["QUESTION"], args.mode, task_spec)
            gold = row["ANSWER"]

            if args.scoring == "logits":
                raw, prob = predict_logits(
                    model, tokenizer, prompt, args.device, answer_token_ids
                )
            else:
                raw, prob = predict_generate(
                    model, tokenizer, prompt, args.device, args.temperature, max_new_tokens
                )

            prediction, invalid = resolve_prediction(
                raw, gold, task_spec, scan=args.scoring == "generate"
            )
            n_invalid += invalid
            preds.append(int(prediction))
            golds.append(int(gold))

            record = {
                "SUBJECT_ID": row["SUBJECT_ID"], "ANSWER": gold,
                "PREDICTION": prediction, "ORIGINAL": raw,
            }
            if args.scoring == "logits":
                record["PROB"] = prob
            writer.writerow(record)

    from sklearn.metrics import f1_score

    f1 = f1_score(golds, preds, average=task_spec.average)
    print(f"\nwrote {out_path}")
    print(f"  scored {len(golds)} visits, F1 = {f1 * 100:.2f}")
    print(f"  {n_invalid} ({n_invalid / max(len(golds), 1):.1%}) produced no valid answer "
          f"and were scored wrong")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--base_model", required=True, help="HuggingFace model id")
    parser.add_argument("--task", required=True, choices=TASKS)
    parser.add_argument("--dataset", required=True, choices=DATASETS)
    parser.add_argument("--mode", default="ORI", choices=MODES)
    parser.add_argument("--scoring", default="logits", choices=["logits", "generate"],
                        help="logits also records PROB, which AUROC needs")
    parser.add_argument("--random_index", type=int, default=0,
                        help="0-4 are reshuffles; 6 is the 500-sample cohort")
    parser.add_argument("--temperature", type=float, default=None,
                        help="generate scoring only; unset means greedy decoding")
    parser.add_argument("--lora_path", default=None, help="adapter to merge before inference")
    parser.add_argument("--token_id_mode", default="legacy", choices=["legacy", "auto"],
                        help="legacy reproduces the published runs; auto fixes "
                             "answer-token lookup on SentencePiece tokenizers")
    parser.add_argument("--data_root", type=Path, default=Path("data"))
    parser.add_argument("--result_root", type=Path, default=Path("results"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--device_map", default="auto")
    args = parser.parse_args()

    if args.temperature and args.scoring == "logits":
        parser.error("--temperature applies to --scoring generate only")
    if args.mode == "LORA" and not args.lora_path:
        parser.error("--mode LORA needs --lora_path")
    run(args)


if __name__ == "__main__":
    main()
