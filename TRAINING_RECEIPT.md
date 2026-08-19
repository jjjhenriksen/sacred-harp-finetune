# Sacred Harp training receipt

## Completed run

- Base: `mlx-community/Llama-3.2-1B-Instruct-4bit`, downloaded locally as `models/llama-1b`
- Corpus: 1,696 canonical text notes and 3,326 song notes from the Obsidian vault
- Dataset: 5,022 examples; 4,223 train, 403 validation, 396 test
- Full LoRA pass: 2,112 iterations, 8 LoRA layers, batch size 2, max sequence length 1,024
- Full-pass final validation loss: 1.595
- Vault-linked retrained adapter: `adapters/sacred_harp_1b_lora_vault/adapters.safetensors`
- Correction pass: 100 iterations over two verified lyric/identification examples
- Correction-pass final validation loss: 0.000
- Corrected adapter: `adapters/sacred_harp_1b_lora_corrected/adapters.safetensors`

## Verification

Direct MLX evaluation of the corrected adapter, using the base model's default
chat template, returned the canonical Denson lyrics for `What are the lyrics
to Idumea?` and identified Fairfield 29t for the clue `I can but perish if I
go, I am resolved to try?`. Keep the inference prompt format aligned with that
template when using this small checkpoint.

The fused/dequantized MLX checkpoint is in
`models/sacred-harp-1b-finetuned-corrected-dequantized/`. GGUF exports were
also produced, but Ollama's current GGUF validation rejects these MLX-produced
tokenizer files with `unordered_map::at: key not found`; the MLX checkpoint and
adapter are the working, verified artifacts.

The original 3B download was not completed because the Hub transfer stalled;
the 1B fallback was downloaded and trained to completion instead.

## Vault-linked pipeline verification

- Model-only generation was tested with the vault-linked adapter. It still
  hallucinated a shape-note definition and misidentified SH 130, demonstrating
  why the model should not be trusted for exact metadata by itself.
- The RAG index was rebuilt successfully at 5,036 chunks.
- The exact query `What is the third verse to SH 130, The Old Graveyard?`
  resolved the Obsidian song note through its Text Hub link to the
  SacredHarpTunes `Boast Ye Not 8s & 7s` note and returned the extra third verse
  beginning `When a few more years are wasted`.
- An open-ended Idumea/Fairfield comparison retrieved the relevant song notes,
  but the small generator still overproduced the Idumea text. Retrieval is
  grounded; open-ended synthesis remains a model limitation.

The corpus-wide evaluation is documented in
`../sacred_harp_ollama_rag/EVALUATION_REPORT.md`. It tested retrieval over all
5,022 examples, exact joins over 1,415 detected extra-verse cases, and
generation over the 396-example held-out split.
