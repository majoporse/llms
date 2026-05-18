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
class GrammarPreferenceTuneSettings:
    source: str = "base"
    hf_path: str | None = None
    model_tag: str | None = "d6"
    step: int | None = None
    pairs_path: str = "./data/devel.tsv"
    test_fraction: float = 0.05
    batch_size: int = 16
    lr: float = 5e-5
    weight_decay: float = 0.0
    max_steps: int = 1200
    eval_every: int = 100
    save_every: int = 100
    save_dir: str = "./data/base_checkpoints/tuned"
    seed: int = 1337
    device_type: str = "cuda"


@dataclass
class GrammarEvalSettings:
    source: str = "base"
    tsv_path: str = "data/devel-short.tsv"
    hf_path: str | None = None
    model_tag: str | None = "tuned"
    step: int | None = None
    device_type: str = "cuda"
    batch_size: int = 16


args = GrammarPreferenceTuneSettings()
user_config = asdict(args)


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


def split_pairs(pairs, test_fraction=0.05, seed=1337):
    if not 0.0 < test_fraction < 1.0:
        raise ValueError("test_fraction must be between 0 and 1")
    shuffled = list(pairs)
    rng = random.Random(seed)
    rng.shuffle(shuffled)
    test_count = (
        max(1, int(math.floor(len(shuffled) * test_fraction))) if shuffled else 0
    )
    test_pairs = shuffled[:test_count]
    train_pairs = shuffled[test_count:]
    return train_pairs, test_pairs


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


def get_trainable_model(model):
    return model.model if hasattr(model, "model") else model


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
    return loss, chosen_good, len(pairs)


def evaluate_pairs(model, tokenizer, pairs, batch_size):
    total_loss = 0.0
    total_correct = 0
    total = 0
    for start in range(0, len(pairs), batch_size):
        batch = pairs[start : start + batch_size]
        loss, correct, count = score_pair_batch(model, tokenizer, batch)
        total_loss += loss.item() * count
        total_correct += correct
        total += count
    avg_loss = total_loss / total if total else 0.0
    accuracy = total_correct / total if total else 0.0
    return avg_loss, accuracy


def save_model(model, tokenizer, settings, model_kind, output_dir, step):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    trainable_model = get_trainable_model(model)

    meta = {
        "step": step,
        "source": settings.source,
        "hf_path": settings.hf_path,
        "model_tag": settings.model_tag,
        "step_in": settings.step,
        "pairs_path": settings.pairs_path,
        "test_fraction": settings.test_fraction,
        "batch_size": settings.batch_size,
        "lr": settings.lr,
        "weight_decay": settings.weight_decay,
        "model_kind": model_kind,
        "user_config": user_config,
    }

    if model_kind != "hf":
        meta["model_config"] = dict(model.config.__dict__)

    if model_kind == "hf":
        trainable_model.save_pretrained(output_dir)
        if hasattr(tokenizer, "tokenizer") and hasattr(
            tokenizer.tokenizer, "save_pretrained"
        ):
            tokenizer.tokenizer.save_pretrained(output_dir)
        elif hasattr(tokenizer, "save_pretrained"):
            tokenizer.save_pretrained(output_dir)
        with open(output_dir / "meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
    else:
        torch.save(trainable_model.state_dict(), output_dir / f"model_{step:06d}.pt")
        with open(output_dir / f"meta_{step:06d}.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)


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
    correct_ids = []
    local_pairs = [
        (idx, pairs[idx]) for idx in range(ddp_rank, len(pairs), ddp_world_size)
    ]
    for start in range(0, len(local_pairs), settings.batch_size):
        batch = local_pairs[start : start + settings.batch_size]
        batch_pairs = [pair for _, pair in batch]
        loss, batch_correct, batch_total = score_pair_batch(
            model, tokenizer, batch_pairs
        )
        num_correct += batch_correct
        total += batch_total
        for idx, pred_first in batch:
            correct_ids.append((idx, pred_first))
        batch_accuracy = batch_correct / batch_total if batch_total else 0.0
        print(
            f"Batch {start // settings.batch_size + 1} accuracy: {batch_accuracy:.4f} ({batch_correct}/{batch_total})"
        )

    accuracy = num_correct / total if total else 0.0
    print(f"{model_name} accuracy: {accuracy:.4f} ({num_correct}/{total})")
    compute_cleanup()

    with open("./data/correct_ids.txt", "w") as f:
        for idx, pred_first in correct_ids:
            sentence = pairs[idx][0 if pred_first else 1]
            f.write(f"{sentence}\n")


def preference_tune(settings=GrammarPreferenceTuneSettings()):
    device_type = (
        autodetect_device_type() if settings.device_type == "" else settings.device_type
    )
    _ddp, _ddp_rank, _ddp_local_rank, ddp_world_size, device = compute_init(device_type)
    if ddp_world_size != 1:
        raise NotImplementedError(
            "grammar_preference_tune.py currently runs in single-process mode only"
        )

    rng = random.Random(settings.seed)
    torch.manual_seed(settings.seed)

    model, tokenizer, model_name, model_kind = load_model_and_tokenizer(
        settings, device
    )
    trainable_model = get_trainable_model(model)
    trainable_model.train()

    pairs = read_pairs(settings.pairs_path)
    train_pairs, test_pairs = split_pairs(
        pairs, test_fraction=settings.test_fraction, seed=settings.seed
    )

    if not train_pairs:
        raise ValueError("No training pairs available")

    print(f"Training model: {model_name}")
    print(f"Train pairs: {len(train_pairs)} | Test pairs: {len(test_pairs)}")

    optimizer = torch.optim.AdamW(
        trainable_model.parameters(), lr=settings.lr, weight_decay=settings.weight_decay
    )

    train_idx = 0
    best_test_acc = -1.0
    train_order = list(range(len(train_pairs)))
    rng.shuffle(train_order)

    for step in range(1, settings.max_steps + 1):
        batch_indices = []
        for _ in range(settings.batch_size):
            if train_idx >= len(train_order):
                rng.shuffle(train_order)
                train_idx = 0
            batch_indices.append(train_order[train_idx])
            train_idx += 1

        batch = [train_pairs[i] for i in batch_indices]
        optimizer.zero_grad(set_to_none=True)
        loss, correct, count = score_pair_batch(model, tokenizer, batch)
        loss.backward()
        optimizer.step()

        if step == 1 or step % 10 == 0:
            print(
                f"step {step:05d} | train_loss={loss.item():.4f} | train_acc={correct / count:.4f}"
            )

        if test_pairs and step % settings.eval_every == 0:
            test_loss, test_acc = evaluate_pairs(
                model, tokenizer, test_pairs, settings.batch_size
            )
            print(
                f"step {step:05d} | test_loss={test_loss:.4f} | test_acc={test_acc:.4f}"
            )
            if test_acc > best_test_acc:
                best_test_acc = test_acc
                save_model(
                    model, tokenizer, settings, model_kind, settings.save_dir, step
                )

        if step % settings.save_every == 0:
            save_model(model, tokenizer, settings, model_kind, settings.save_dir, step)

    save_model(
        model, tokenizer, settings, model_kind, settings.save_dir, settings.max_steps
    )
    compute_cleanup()


if __name__ == "__main__":
    preference_tune()
    grammar_eval()
