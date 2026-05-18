
from nanochat.tokenizer import RustBPETokenizer, get_tokenizer


with open("./data/eval-gold.tsv") as f:
    gold_lines = f.readlines()
    
with open("./data/eval-input.tsv") as f:
    pred_lines = f.readlines()

gold_lines_old = [line.strip() for line in gold_lines]
pred_lines_old = [line.strip() for line in pred_lines]

count = 0
wrong_lines = []

for idx, (gold, pred) in enumerate(zip(gold_lines_old, pred_lines_old)):
    pred_variants = pred.split("\t")
    
    if gold not in pred_variants:
        print(f"Gold: {gold}")
        print(f"Pred: {pred}")
        print()
        count += 1
        wrong_lines.append(idx)
        

print(f"Count: {count}")

# remove the lines
max_lines = 800
gold_lines_new = [gold_lines_old[idx] for idx in range(len(gold_lines_old)) if idx not in wrong_lines and idx < max_lines]
pred_lines_new = [pred_lines_old[idx] for idx in range(len(gold_lines_old)) if idx not in wrong_lines and idx < max_lines]
print(f"Gold lines: {len(gold_lines_new)}, Pred lines: {len(pred_lines_new)}")

with open("./data/eval-gold-clean.tsv", "w") as f:
    f.write("\n".join(gold_lines_new))

with open("./data/eval-input-clean.tsv", "w") as f:
    f.write("\n".join(pred_lines_new))

# get the max tokens per line
# tokenizer = RustBPETokenizer.from_pretrained("gpt2") # gpt-2 base model tokenizer
tokenizer = get_tokenizer()
max_tokens = max([len(tokenizer(line)) for line in gold_lines_new + pred_lines_new])
print(f"Max tokens: {max_tokens}")

# get the max tokens per line from train data
with open("./data/data.txt") as f:
    train_lines = f.readlines()
train_lines_clean = []
for line in train_lines[:80000]:
    line = line.strip()
    line_parts = line.split("\t")
    if len(line_parts) == 2:
        train_lines_clean.append(line_parts[1])

max_tokens_train = max([len(tokenizer(line)) for line in train_lines_clean])
print(f"Max tokens train: {max_tokens_train}")
import numpy as np
print(f"median: {np.median([len(tokenizer(line)) for line in train_lines_clean])}")
# print multiple quantiles
print(f"quantiles: {np.quantile([len(tokenizer(line)) for line in train_lines_clean], [0.25, 0.5, 0.75])}")

