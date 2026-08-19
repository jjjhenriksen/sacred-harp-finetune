#!/bin/zsh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
PYTHON="${SACRED_HARP_PYTHON:-$ROOT/../sacred_harp_finetune_venv/bin/python}"
[[ -x "$PYTHON" ]] || PYTHON="${SACRED_HARP_PYTHON:-$(command -v python3)}"
MODEL="mlx-community/Llama-3.2-3B-Instruct-4bit"
ADAPTER_PATH="$ROOT/adapters/sacred_harp_lora"
PROMPT="What are the lyrics to Idumea?"
if [[ $# -gt 0 ]]; then
  PROMPT="$1"
fi

exec "$PYTHON" -m mlx_lm.generate \
  --model "$MODEL" \
  --adapter-path "$ADAPTER_PATH" \
  --system-prompt "You are a careful Sacred Harp reference assistant. Answer from the trained corpus and do not invent lyrics or metadata." \
  --prompt "$PROMPT" \
  --max-tokens 700 \
  --temp 0.2 \
  --seed 42 \
  --verbose False
