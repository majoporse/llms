import argparse
import csv
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from nanochat.checkpoint_manager import load_model
from nanochat.common import autodetect_device_type, compute_cleanup, compute_init

@dataclass
class GrammarEvalSettings:
    source: str = "base"
    tsv_path: str = "data/eval-input.tsv"
    hf_path: str | None = None
    model_tag: str | None = "tuned"
    step: int | None = 500
    device_type: str = "cuda"
    batch_size: int = 16
    output_file: str = "eval.output"


def read_pairs(path):
    rows = []
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f, delimiter="\t")
        for line_no, row in enumerate(reader, start=1):
            if len(row) != 2:
                raise ValueError(
                    f"{path}:{line_no}: expected 2 columns, got {len(row)}"
                )
            rows.append((row[0], row[1]))
    return rows


def pairwise_preference_loss(good_nll, bad_nll):
    return torch.nn.functional.softplus(good_nll - bad_nll)


def load_model_and_tokenizer(settings, device):
    if settings.hf_path is not None:
        from scripts.base_eval import load_hf_model

        model, tokenizer = load_hf_model(settings.hf_path, device)
        model_name = settings.hf_path
        model_kind = "hf"
    else:
        model, tokenizer, meta = load_model(
            settings.source,
            device,
            phase="train",
            model_tag=settings.model_tag,
            step=settings.step,
        )
        model_name = f"{settings.source}_model (step {meta['step']})"
        model_kind = "nanochat"
    return model, tokenizer, model_name, model_kind


def batch_sentence_nll(model, tokenizer, sentences):
    nlls = []
    for sentence in sentences:
        ids = tokenizer(sentence, prepend=tokenizer.get_bos_token_id())
        if len(ids) < 2:
            nlls.append(
                torch.tensor(
                    float("inf"), dtype=torch.float32, device=model.get_device()
                )
            )
            continue
        x = torch.tensor(
            ids[:-1], dtype=torch.long, device=model.get_device()
        ).unsqueeze(0)
        y = torch.tensor(
            ids[1:], dtype=torch.long, device=model.get_device()
        ).unsqueeze(0)
        nlls.append(model(x, y))
    return torch.stack(nlls)


def score_pair_batch(model, tokenizer, pairs):
    good_sentences = [pair[0] for pair in pairs]
    bad_sentences = [pair[1] for pair in pairs]
    good_nll = batch_sentence_nll(model, tokenizer, good_sentences)
    bad_nll = batch_sentence_nll(model, tokenizer, bad_sentences)
    loss = pairwise_preference_loss(good_nll, bad_nll).mean()
    chosen_good = (good_nll < bad_nll).sum().item()
    preds = good_nll < bad_nll
    return loss, chosen_good, len(pairs), preds

def grammar_eval(settings=GrammarEvalSettings()):
    device_type = (
        autodetect_device_type() if settings.device_type == "" else settings.device_type
    )
    ddp, ddp_rank, _ddp_local_rank, ddp_world_size, device = compute_init(device_type)
    model, tokenizer, model_name, model_kind = load_model_and_tokenizer(
        settings, device
    )

    if ddp_world_size != 1:
        raise NotImplementedError(
            "grammar_preference_tune.py currently runs in single-process mode only"
        )

    pairs = read_pairs(settings.tsv_path)

    num_correct = 0
    total = 0
    pred_ids = []
    local_pairs = [(idx, pair) for idx, pair in enumerate(pairs)]
    for start in range(0, len(local_pairs), settings.batch_size):
        batch = local_pairs[start : start + settings.batch_size]
        batch_pairs = [pair for _, pair in batch]
        loss, batch_correct, batch_total, batch_preds = score_pair_batch(
            model, tokenizer, batch_pairs
        )
        num_correct += batch_correct
        total += batch_total
        for pred_first in batch_preds:
            pred_ids.append(pred_first)
        batch_accuracy = batch_correct / batch_total if batch_total else 0.0
        print(
            f"Batch {start // settings.batch_size + 1} accuracy: {batch_accuracy:.4f} ({batch_correct}/{batch_total})"
        )

    accuracy = num_correct / total if total else 0.0
    print(f"{model_name} accuracy: {accuracy:.4f} ({num_correct}/{total})")
    compute_cleanup()

    with open(settings.output_file, "w") as f:
        for idx, pred_first in enumerate(pred_ids):
            sentence = pairs[idx][0 if pred_first else 1]
            f.write(f"{sentence}\n")

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--tsv-path")
    args = args.parse_args()
    grammar_eval(settings=GrammarEvalSettings(tsv_path=args.tsv_path))
