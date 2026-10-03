#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

.venv/bin/python - <<'PY'
from datasets import get_dataset_config_names, load_dataset
from pathlib import Path

root = Path("datasets/MASK")
for config in get_dataset_config_names("cais/MASK"):
    load_dataset("cais/MASK", config).save_to_disk(str(root / config))

load_dataset(
    "mandarjoshi/trivia_qa", "rc.nocontext", split="validation"
).save_to_disk("datasets/trivia_qa_rc_nocontext_validation")
PY
