#!/usr/bin/env python3
"""Promote the checkpoint with the lowest recorded mixed validation loss."""

from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path


HERE = Path(__file__).resolve().parent
DEFAULT_LOG = HERE / "reports" / "training.log"
DEFAULT_ADAPTER = HERE.parent / "adapters" / "sacred_harp_1b_lora_openclaw"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", type=Path, default=DEFAULT_LOG)
    parser.add_argument("--adapter", type=Path, default=DEFAULT_ADAPTER)
    args = parser.parse_args()
    matches = [
        (int(iteration), float(loss))
        for iteration, loss in re.findall(r"Iter (\d+): Val loss ([0-9.]+)", args.log.read_text())
    ]
    candidates = [(iteration, loss) for iteration, loss in matches if iteration > 1]
    if not candidates:
        raise RuntimeError("No saved validation checkpoint found in the training log")
    best_iteration, best_loss = min(candidates, key=lambda item: item[1])
    checkpoint = args.adapter / f"{best_iteration:07d}_adapters.safetensors"
    if not checkpoint.exists():
        raise FileNotFoundError(f"Best checkpoint is missing: {checkpoint}")
    destination = args.adapter / "adapters.safetensors"
    shutil.copy2(checkpoint, destination)
    receipt = {
        "selection_metric": "mixed validation loss",
        "best_iteration": best_iteration,
        "best_validation_loss": best_loss,
        "checkpoint": str(checkpoint),
        "promoted_to": str(destination),
        "all_validation_losses": [
            {"iteration": iteration, "loss": loss} for iteration, loss in matches
        ],
    }
    output = HERE / "reports" / "checkpoint_selection.json"
    output.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
