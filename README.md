# Sacred Harp Llama 3.2 fine-tune

This directory contains a reproducible Apple Silicon LoRA training pipeline
for a small Sacred Harp reference model. The completed run uses the public MLX
conversion `mlx-community/Llama-3.2-1B-Instruct-4bit`, stored locally in
`models/llama-1b`. The 3B download was attempted first but stalled, so the 1B
model was used as the practical fallback.

This repository contains the source code, evaluation harnesses, and
documentation. Local model downloads, generated adapters, private corpus
exports, and machine-specific reports are intentionally excluded from Git.
Recreate those locally with the commands below.

The vault-generation scripts and their portable rebuild entry point live in
`vault_reproduction/README.md`.

The final presentation is the self-contained deck in
`presentation/sacred-harp-agent-final.html`; keep the HTML beside its two image
assets when opening it locally.

The presentation is also deployed at
<https://jjjhenriksen.github.io/sacred-harp-finetune/>.

## Corpus

prepare_dataset.py reads the canonical Obsidian shape-note text hubs and song
notes, removes Dataview/navigation material, and writes MLX-LM chat JSONL files.
The generated data includes lyric-answer examples from text hubs and metadata
examples from song notes. It preserves book families, edition labels, song
numbers, canonical text keys, and first lines.

Pass the generated vault root containing `texts/` and `songs/` directories.
Missing directories or an empty corpus are rejected before existing outputs
are touched. All splits and `manifest.json` are staged together; publication
preserves other files in the output directory and restores the previous
directory if promotion fails. The current generated manifest is in
`data/manifest.json`.

Split policy `sha1-160-80-10-10-v2` assigns the full 160-bit SHA-1 group hash
using integer thresholds at 80% and 90%. Every example in a group receives the
same deterministic split, independent of input order. The manifest records
the policy version and nominal 80/10/10 fractions. This changes membership
from the prior one-byte modulo policy: regenerate the dataset and rerun
training/evaluation before comparing results under the new policy. Prior
losses and held-out scores are not directly comparable across the policies.

The dataset publication regressions use fictional notes and the Python
standard library; they do not run training or load a model:

    python3 -m unittest discover -s tests -v

## Native environment setup

Training and the OpenClaw provider use `requirements-mlx.txt`, which pins
MLX 0.32.3 and MLX-LM 0.32.0 with its training dependencies. This runtime needs
Apple Silicon, native arm64 CPython 3.11 or later (3.12 recommended), and
macOS 14 or later. Python running under Rosetta and the system Python 3.9 do
not meet this contract. See the [MLX installation requirements](https://ml-explore.github.io/mlx/build/html/install.html)
and [MLX-LM package specification](https://github.com/ml-explore/mlx-lm/blob/main/pyproject.toml).

From a fresh clone, use a separately installed native Python 3.12:

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements-mlx.txt
.venv/bin/python -m pip check
.venv/bin/python scripts/check_mlx_environment.py
export SACRED_HARP_PYTHON="$PWD/.venv/bin/python"
export SACRED_HARP_VENV="$PWD/.venv"
```

The explicit runtime variables make the existing training, evaluation,
installation and verification scripts use this environment rather than a
preexisting sibling virtualenv. Activation is optional; the commands above
use the exact environment interpreter even when a shell aliases `python3`.
For a later terminal session, set both variables again from the repo root.

`check_mlx_environment.py` verifies platform and pinned versions, imports the
native training/provider APIs, and executes dataset, LoRA, generation and
provider CLI help with an empty temporary Hub cache and offline flags. It
loads no model, starts no training, and opens no provider port. Add
`--output reports/install-check.json` to retain its evidence. Import/help
checks do not prove model/adapter compatibility, training quality or the
external RAG corpus. The provider also requires a configured external backend
as described in `openclaw_capability/README.md`.

The pins version the native engine/API; pip resolves their transitive
dependencies. Save `.venv/bin/python -m pip freeze` with a training run to
record the full installed dependency set. Vault regeneration dependencies in
`vault_reproduction` and optional GGUF/Ollama export tools are separate scopes;
this file does not install their additional toolchains. Portable root tests
still need only the Python standard library and run without MLX on Linux.

## Train

After the environment setup above, run:

    SACRED_HARP_VAULT_ROOT=/path/to/generated-obsidian-vault \
      "$SACRED_HARP_PYTHON" prepare_dataset.py --output data
    ./train_small.sh

The training script performs one pass over the training split, with eight LoRA
layers, prompt masking, gradient checkpointing, and a conservative learning
rate.

The vault-linked retrain is stored at
`adapters/sacred_harp_1b_lora_vault/adapters.safetensors`. The existing
`adapters/sacred_harp_1b_lora_corrected` directory is preserved as a fallback.
The final OpenClaw presentation path uses the local base model at
`models/llama-1b` with
`adapters/sacred_harp_1b_lora_openclaw_v3_interpretation`.

## Evaluate

There are two different evaluation paths. The raw MLX command is useful for
checking the fine-tuned model, but it does not load the Sacred Harp RAG tool or
the OpenClaw response renderer. Do not use it as the presentation path: it can
answer from model memory, echo a prompt, or miss corpus-specific facts.

For a grounded Sacred Harp answer through the configured agent, use the
OpenClaw command in `openclaw_capability/README.md`.

From this directory, the raw model-only command is:

    ./evaluate_small.sh "What are the lyrics to Idumea?"
    ./evaluate_small.sh "Which song has the lyrics 'I can but perish if I go'?"

The raw command does not prove that OpenClaw, ClickClack, Docker, or
`sacred_harp_search` is working.

## Ollama export

The verified working artifact is the MLX adapter. To reproduce the fused
checkpoint:

    python3 -m mlx_lm.fuse --model models/llama-1b --adapter-path adapters/sacred_harp_1b_lora_corrected --save-path models/sacred-harp-1b-finetuned-corrected-dequantized --dequantize

`TRAINING_RECEIPT.md` records the completed losses and verification results.
The GGUF files are retained for investigation, but Ollama currently rejects
the MLX-produced tokenizer metadata during validation. Use MLX plus the
corrected adapter for the verified local inference path; keep RAG in use for
exact lyric and edition retrieval.
