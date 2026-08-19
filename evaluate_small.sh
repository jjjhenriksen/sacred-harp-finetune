#!/bin/zsh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
PYTHON="$ROOT/../sacred_harp_finetune_venv/bin/python"
MODEL="$ROOT/models/llama-1b"
ADAPTER="$ROOT/adapters/sacred_harp_1b_lora_corrected"

if [[ $# -eq 0 ]]; then
  echo 'Usage: ./evaluate_small.sh "Your Sacred Harp question"' >&2
  exit 2
fi

exec "$PYTHON" -m mlx_lm.generate \
  --model "$MODEL" \
  --adapter-path "$ADAPTER" \
  --prompt "$*" \
  --max-tokens 512 \
  --temp 0.2
