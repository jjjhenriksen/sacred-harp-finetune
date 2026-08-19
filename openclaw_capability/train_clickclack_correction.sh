#!/bin/zsh
set -euo pipefail

HERE=${0:A:h}
ROOT=${HERE:h}
VENV=${SACRED_HARP_VENV:-${ROOT:h}/sacred_harp_finetune_venv}
MODEL=${ROOT}/models/llama-1b
SOURCE_ADAPTER=${ROOT}/adapters/sacred_harp_1b_lora_openclaw/adapters.safetensors
OUTPUT_ADAPTER=${ROOT}/adapters/sacred_harp_1b_lora_openclaw_v2
LOG_DIR=${HERE}/reports

"${VENV}/bin/python" "${HERE}/prepare_clickclack_correction_dataset.py"

"${VENV}/bin/python" -m mlx_lm lora \
  --model "$MODEL" \
  --train \
  --data "${HERE}/data_clickclack_correction" \
  --fine-tune-type lora \
  --mask-prompt \
  --num-layers 8 \
  --batch-size 2 \
  --iters 150 \
  --learning-rate 1e-6 \
  --steps-per-report 10 \
  --steps-per-eval 50 \
  --val-batches -1 \
  --resume-adapter-file "$SOURCE_ADAPTER" \
  --adapter-path "$OUTPUT_ADAPTER" \
  --save-every 50 \
  --max-seq-length 1152 \
  --grad-checkpoint \
  --seed 298131 \
  2>&1 | tee "${LOG_DIR}/clickclack_correction_training.log"

"${VENV}/bin/python" "${HERE}/select_best_checkpoint.py" \
  --log "${LOG_DIR}/clickclack_correction_training.log" \
  --adapter "$OUTPUT_ADAPTER"

echo "Corrected OpenClaw capability adapter saved to ${OUTPUT_ADAPTER}"
