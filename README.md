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

## Corpus

prepare_dataset.py reads the canonical Obsidian shape-note text hubs and song
notes, removes Dataview/navigation material, and writes MLX-LM chat JSONL files.
The generated data includes lyric-answer examples from text hubs and metadata
examples from song notes. It preserves book families, edition labels, song
numbers, canonical text keys, and first lines.

The current generated manifest is in data/manifest.json.

## Train

The environment is in the sibling virtualenv. Run:

    SACRED_HARP_VAULT_ROOT=/path/to/generated-obsidian-vault \
      python3 prepare_dataset.py --output data
    ./train_small.sh

The training script performs one pass over the training split, with eight LoRA
layers, prompt masking, gradient checkpointing, and a conservative learning
rate.

The vault-linked retrain is stored at
`adapters/sacred_harp_1b_lora_vault/adapters.safetensors`. The existing
`adapters/sacred_harp_1b_lora_corrected` directory is preserved as a fallback.

## Evaluate

There are two different evaluation paths. The raw MLX command is useful for
checking the fine-tuned model, but it does not load the Sacred Harp RAG tool or
the OpenClaw response renderer. Do not use it as the presentation path: it can
answer from model memory, echo a prompt, or miss corpus-specific facts.

For a grounded Sacred Harp answer through the configured agent, use the
OpenClaw command in `openclaw_capability/README.md`.

From this directory, the raw model-only command is:

    cd /Users/jacquelinehenriksen/CPSC298-LocalLLM/sacred_harp_finetune
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
