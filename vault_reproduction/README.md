# Sacred Harp vault reproduction

This directory contains the source-side scripts that generate the Sacred Harp
song notes and text hubs consumed by the model's RAG layer. The source snapshot
and generated vault are deliberately kept outside GitHub: they may contain
private Obsidian notes or locally cached source material.

## Install

From the repository root:

```zsh
python3 -m venv .venv
.venv/bin/python -m pip install -e vault_reproduction
```

## Rebuild from an existing source snapshot

The minimum input is a directory containing
`combined_first_lines_canonicalized.csv`, `bremen_lyrics_fixed4.csv`, and
`data/stanzas.yaml`. Run:

```zsh
.venv/bin/python vault_reproduction/rebuild_vault.py \
  --sources /path/to/sacred-harp-source-snapshot \
  --vault /path/to/generated-obsidian-vault
```

Optional lyric witnesses can be added without changing the script:

```zsh
.venv/bin/python vault_reproduction/rebuild_vault.py \
  --sources /path/to/sacred-harp-source-snapshot \
  --vault /path/to/generated-obsidian-vault \
  --extra-lyrics /path/to/southernharmony_lyrics.csv \
  --extra-lyrics /path/to/kentucky_lyrics.csv
```

Reports and caches go to `vault_reproduction/work` by default. Use
`--work-dir` to place them elsewhere. The generated notes are written under
`04 Music/shape-note/` inside the destination vault.

## Source acquisition scripts

The individual scripts are retained for the staged source workflow:

1. `texasfasola_scrape_cooper_firstlines.py`
2. `modern_shape_note_scrape_firstlines.py`
3. `modern_shape_note_scrape_lyrics.py`
4. `southern_harmony_scrape.py`
5. `kentucky_harmony_scrape.py`
6. `build_combined_first_lines.py`
7. `rebuild_vault.py`

Run scrapers only against sources you are permitted to access, and keep their
CSV outputs outside the Git repository. `rag_index.py` can then build the
offline SQLite projection from the generated Markdown without modifying the
source vault.
