#!/bin/zsh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
PYTHON="/Users/jacquelinehenriksen/CPSC298-LocalLLM/sacred_harp_finetune_venv/bin/python"
MODEL="mlx-community/Llama-3.2-3B-Instruct-4bit"
ITERS=4223
ADAPTER_PATH="$ROOT/adapters/sacred_harp_lora"

exec "$PYTHON" -m mlx_lm.lora \
  --model "$MODEL" \
  --train \
  --data "$ROOT/data" \
  --fine-tune-type lora \
  --mask-prompt \
  --num-layers 8 \
  --batch-size 1 \
  --iters "$ITERS" \
  --learning-rate 1e-5 \
  --steps-per-report 10 \
  --steps-per-eval 100 \
  --val-batches -1 \
  --adapter-path "$ADAPTER_PATH" \
  --save-every 250 \
  --max-seq-length 1024 \
  --grad-checkpoint \
  --seed 42
