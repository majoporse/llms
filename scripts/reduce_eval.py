
with open("./data/eval-input-base.tsv", "r") as f:
    lines = f.readlines()

modulo = 10
new_lines = [line for i, line in enumerate(lines) if i % modulo == 0]

with open("./data/eval-input.tsv", "w") as f:
    f.writelines(new_lines)
