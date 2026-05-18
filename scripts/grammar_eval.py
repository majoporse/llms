import argparse
import csv
from dataclasses import dataclass

import torch

from nanochat.checkpoint_manager import load_model
from nanochat.common import autodetect_device_type, compute_cleanup, compute_init, print0
from nanochat.loss_eval import evaluate_mean_nll


@dataclass
class GrammarEvalSettings:
    tsv_path: str = "data/devel.tsv"
    hf_path: str | None = None
    model_tag: str | None = "d8"
    step: int | None = None
    device_type: str = "cuda"
    batch_size: int = 32

args = GrammarEvalSettings()


def read_pairs(tsv_path):
    rows = []
    with open(tsv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f, delimiter="\t")
        for line_no, row in enumerate(reader, start=1):
            if len(row) != 2:
                raise ValueError(f"{tsv_path}:{line_no}: expected 2 columns, got {len(row)}")
            rows.append((row[0], row[1]))
    return rows


def load_model_and_tokenizer(settings, device):
    if settings.hf_path is not None:
        from scripts.base_eval import load_hf_model

        model, tokenizer = load_hf_model(settings.hf_path, device)
        model_name = settings.hf_path
    else:
        model, tokenizer, meta = load_model("base", device, phase="eval", model_tag=settings.model_tag, step=settings.step)
        model_name = f"base_model (step {meta['step']})"
    return model, tokenizer, model_name


def batch_sentence_nll(model, tokenizer, sentences):
    nlls = []
    for sentence in sentences:
        ids = tokenizer(sentence, prepend=tokenizer.get_bos_token_id())
        if len(ids) < 2:
            nlls.append(torch.tensor(float("inf"), dtype=torch.float32, device=model.get_device()))
            continue
        x = torch.tensor(ids[:-1], dtype=torch.long, device=model.get_device()).unsqueeze(0)
        y = torch.tensor(ids[1:], dtype=torch.long, device=model.get_device()).unsqueeze(0)
        nlls.append(model(x, y))
    return torch.stack(nlls)

def score_pair_batch(model, tokenizer, pairs):
    good_sentences = [pair[0] for pair in pairs]
    bad_sentences = [pair[1] for pair in pairs]
    good_nll = batch_sentence_nll(model, tokenizer, good_sentences)
    bad_nll = batch_sentence_nll(model, tokenizer, bad_sentences)
    chosen_good = (good_nll < bad_nll).sum().item()
    return chosen_good, len(pairs)

def grammar_eval(settings=args):
    device_type = autodetect_device_type() if settings.device_type == "" else settings.device_type
    ddp, ddp_rank, _ddp_local_rank, ddp_world_size, device = compute_init(device_type)
    model, tokenizer, model_name = load_model_and_tokenizer(settings, device)

    if ddp_world_size != 1:
        raise NotImplementedError("grammar_preference_tune.py currently runs in single-process mode only")

    pairs = read_pairs(settings.tsv_path)

    num_correct = 0
    total = 0
    correct_ids = []
    local_pairs = [(idx, pairs[idx]) for idx in range(ddp_rank, len(pairs), ddp_world_size)]
    for start in range(0, len(local_pairs), settings.batch_size):
        batch = local_pairs[start:start + settings.batch_size]
        batch_pairs = [pair for _, pair in batch]
        batch_correct, batch_total = score_pair_batch(model, tokenizer, batch_pairs)
        num_correct += batch_correct
        total += batch_total
        for idx, pred_first in batch:
            correct_ids.append((idx, pred_first))
        batch_accuracy = batch_correct / batch_total if batch_total else 0.0
        print0(f"Batch {start // settings.batch_size + 1} accuracy: {batch_accuracy:.4f} ({batch_correct}/{batch_total})")

    accuracy = num_correct / total if total else 0.0
    print0(f"{model_name} accuracy: {accuracy:.4f} ({num_correct}/{total})")
    compute_cleanup()

    with open("./data/correct_ids.txt", "w") as f:
        for idx, pred_first in correct_ids:
            sentence = pairs[idx][0 if pred_first else 1]
            f.write(f"{sentence}\n")


if __name__ == "__main__":
    grammar_eval(GrammarEvalSettings())
