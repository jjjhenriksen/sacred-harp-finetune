# Submission package

Use the GitHub repository or the generated ZIP as the source submission. It
contains the implementation, tests, OpenClaw integration, vault-reproduction
scripts, evaluation report, training receipt, and exact run instructions.

The package intentionally does not contain:

- downloaded base models;
- generated LoRA adapters or GGUF files;
- private Obsidian vault exports;
- local ClickClack/OpenClaw configuration;
- generated caches and reports tied to one machine.

Those files are large, machine-specific, or private. A reviewer can inspect
the implementation and reproduce the pipeline with a local model and source
snapshot by following `README.md` and `vault_reproduction/README.md`.

Keep these local artifacts for the live demo:

- `models/llama-1b/`
- `adapters/sacred_harp_1b_lora_openclaw_v3_interpretation/`
- the generated local Sacred Harp RAG source/index used by the provider.

Before submitting, run:

```zsh
python3 -m compileall -q .
cd openclaw_capability
python3 -m unittest -q test_sacred_harp_openclaw_server.py
```
