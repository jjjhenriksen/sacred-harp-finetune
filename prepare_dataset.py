#!/usr/bin/env python3
"""Build MLX-LM chat data from the Sacred Harp Obsidian corpus."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path


DEFAULT_VAULT_ROOT = Path(
    "/Users/jacquelinehenriksen/Library/Mobile Documents/"
    "iCloud~md~obsidian/Documents/General/05 Music/shape-note"
)
SYSTEM_PROMPT = (
    "You are a careful Sacred Harp reference assistant. Answer from the "
    "provided corpus. Preserve book names, editions, song numbers, tune "
    "names, and attribution exactly when they are present. If the corpus "
    "does not establish an answer, say that clearly instead of guessing."
)


def split_frontmatter(text: str) -> tuple[dict[str, str], str]:
    if not text.startswith("---\n"):
        return {}, text
    end = text.find("\n---", 4)
    if end < 0:
        return {}, text
    header = text[4:end]
    body = text[end + 4:]
    metadata: dict[str, str] = {}
    for line in header.splitlines():
        match = re.match(r"^([A-Za-z0-9_]+):\s*(.*)$", line)
        if match:
            metadata[match.group(1)] = match.group(2).strip().strip('"')
    return metadata, body


def clean_body(body: str) -> str:
    body = body.split("<!--TEXT_HUB_QUERY_V1-->", 1)[0]
    body = re.sub(r"\[\[([^|\]]+)\|([^\]]+)\]\]", r"\2", body)
    body = re.sub(r"\[\[([^\]]+)\]\]", r"\1", body)
    body = re.sub(r"\n{3,}", "\n\n", body)
    return body.strip()


def title_from_body(body: str, fallback: str) -> str:
    match = re.search(r"^#\s+(.+?)\s*$", body, flags=re.MULTILINE)
    return match.group(1).strip().rstrip("…").strip() if match else fallback


def section(body: str, heading: str, next_heading: str | None = None) -> str:
    marker = f"## {heading}"
    if marker not in body:
        return ""
    value = body.split(marker, 1)[1]
    if next_heading:
        value = value.split(f"## {next_heading}", 1)[0]
    return value.strip()


def canonical_texts(root: Path) -> list[dict[str, str]]:
    records = []
    for path in sorted((root / "texts").glob("*.md")):
        raw = path.read_text(encoding="utf-8")
        metadata, raw_body = split_frontmatter(raw)
        body = clean_body(raw_body)
        full_texts = section(body, "Full Texts", "Appearances")
        if not full_texts:
            continue
        title = title_from_body(body, path.stem.replace("text--", "").replace("-", " "))
        records.append(
            {
                "kind": "text",
                "group": metadata.get("text_key", title.lower()),
                "title": title,
                "source": str(path),
                "content": full_texts,
            }
        )
    return records


def song_records(root: Path) -> list[dict[str, str]]:
    records = []
    for path in sorted((root / "songs").glob("*/*.md")):
        raw = path.read_text(encoding="utf-8")
        metadata, raw_body = split_frontmatter(raw)
        body = clean_body(raw_body)
        title = metadata.get("title") or title_from_body(body, path.stem)
        match = re.search(r"^- Raw First Line:\s*(.+)$", body, flags=re.MULTILINE)
        records.append(
            {
                "kind": "song",
                "group": f"{metadata.get('book_family', '')}:{metadata.get('song_no', '')}",
                "title": title,
                "book": metadata.get("book_family", ""),
                "song_no": metadata.get("song_no", ""),
                "text_key": metadata.get("text_key", ""),
                "first_line": match.group(1).strip() if match else "",
                "source": str(path),
            }
        )
    return records


def message_example(group: str, user: str, assistant: str, source: str, kind: str) -> dict:
    return {
        "group": group,
        "source": source,
        "kind": kind,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant.strip()},
        ],
    }


def make_examples(root: Path) -> list[dict]:
    examples = []
    texts = canonical_texts(root)
    text_by_key = {record["group"]: record for record in texts}

    for record in texts:
        user = f"What are the lyrics associated with the text “{record['title']}”?"
        assistant = (
            f"The corpus identifies this text as “{record['title']}”. "
            "Here is the canonical text, with book witnesses retained in "
            f"the source note:\n\n{record['content']}"
        )
        examples.append(
            message_example(record["group"], user, assistant, record["source"], "lyrics")
        )

    for record in song_records(root):
        fields = [
            f"Song: {record['title']}",
            f"Book family: {record['book'] or 'not specified'}",
            f"Song number: {record['song_no'] or 'not specified'}",
        ]
        if record["text_key"]:
            fields.append(f"Canonical text key: {record['text_key']}")
        if record["first_line"]:
            fields.append(f"Raw first line: {record['first_line']}")
        linked = text_by_key.get(record["text_key"])
        if linked:
            fields.append(f"Text hub title: {linked['title']}")
        user = f"What does the corpus say about {record['title']}?"
        examples.append(
            message_example(
                record["group"], user, "\n".join(fields), record["source"], "metadata"
            )
        )

    return examples


def split_examples(examples: list[dict]) -> dict[str, list[dict]]:
    splits = {"train": [], "valid": [], "test": []}
    for example in examples:
        digest = hashlib.sha1(example["group"].encode("utf-8")).digest()[0] % 100
        split = "train" if digest < 80 else "valid" if digest < 90 else "test"
        splits[split].append(
            {key: value for key, value in example.items() if key != "group"}
        )
    return splits


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vault-root", type=Path, default=DEFAULT_VAULT_ROOT)
    parser.add_argument("--output", type=Path, default=Path("data"))
    args = parser.parse_args()

    examples = make_examples(args.vault_root)
    splits = split_examples(examples)
    args.output.mkdir(parents=True, exist_ok=True)
    for name, records in splits.items():
        path = args.output / f"{name}.jsonl"
        with path.open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    manifest = {
        "vault_root": str(args.vault_root),
        "total_examples": len(examples),
        "splits": {name: len(records) for name, records in splits.items()},
        "source_text_notes": len(canonical_texts(args.vault_root)),
        "source_song_notes": len(song_records(args.vault_root)),
        "system_prompt": SYSTEM_PROMPT,
    }
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
