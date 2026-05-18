#!/bin/bash

uv sync --extra gpu

source .venv/bin/activate

python -m scripts.tok_train

python -m scripts.base_train

python -m scripts.grammar_preference_tune
