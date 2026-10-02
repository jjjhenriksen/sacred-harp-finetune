#!/usr/bin/env python3
"""Promote the checkpoint with the lowest recorded mixed validation loss."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import tempfile
from pathlib import Path


HERE = Path(__file__).resolve().parent
DEFAULT_LOG = HERE / "reports" / "training.log"
DEFAULT_ADAPTER = HERE.parent / "adapters" / "sacred_harp_1b_lora_openclaw"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verified_copy(checkpoint: Path, destination: Path) -> tuple[Path, str, int]:
    size = checkpoint.stat().st_size
    if not size:
        raise ValueError(f"Selected checkpoint is empty: {checkpoint}")
    digest = file_sha256(checkpoint)
    descriptor, name = tempfile.mkstemp(prefix=f".{destination.name}-", suffix=".tmp", dir=destination.parent)
    os.close(descriptor)
    staged = Path(name)
    verified = False
    try:
        shutil.copy2(checkpoint, staged)
        if staged.stat().st_size != size or file_sha256(staged) != digest:
            raise RuntimeError("Checkpoint copy verification failed; canonical adapter was preserved")
        with staged.open("rb") as handle:
            os.fsync(handle.fileno())
        verified = True
        return staged, digest, size
    finally:
        if not verified:
            staged.unlink(missing_ok=True)


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
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Best checkpoint is missing: {checkpoint}")
    destination = args.adapter / "adapters.safetensors"
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
    output.parent.mkdir(parents=True, exist_ok=True)
    staged_adapter = None
    staged_receipt = None
    try:
        staged_adapter, digest, size = verified_copy(checkpoint, destination)
        receipt.update(adapter_sha256=digest, adapter_bytes=size)
        serialized = json.dumps(receipt, indent=2) + "\n"
        descriptor, name = tempfile.mkstemp(prefix=f".{output.name}-", suffix=".tmp", dir=output.parent)
        staged_receipt = Path(name)
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(serialized)
            handle.flush()
            os.fsync(handle.fileno())
        # Each file becomes visible only when complete. Stage both before either promotion.
        os.replace(staged_adapter, destination)
        os.replace(staged_receipt, output)
        print(serialized.rstrip())
    finally:
        if staged_adapter is not None:
            staged_adapter.unlink(missing_ok=True)
        if staged_receipt is not None:
            staged_receipt.unlink(missing_ok=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
