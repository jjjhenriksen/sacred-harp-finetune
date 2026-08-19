#!/usr/bin/env python3
"""Query the offline Sacred Harp RAG projection."""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
from pathlib import Path


def fts_query(query: str) -> str:
    terms = re.findall(r"[\w’'-]+", query, flags=re.UNICODE)
    if not terms:
        raise ValueError("Query must contain at least one searchable word")
    # Quoted tokens avoid FTS operator injection while OR keeps lyric fragments
    # useful when punctuation and OCR vary across editions.
    return " OR ".join(f'"{term.replace(chr(34), "")}"' for term in terms)


def search(database: Path, query: str, limit: int = 8) -> list[dict[str, object]]:
    connection = sqlite3.connect(database)
    connection.row_factory = sqlite3.Row
    try:
        rows = connection.execute(
            """
            SELECT
                c.chunk_id,
                c.document_id,
                d.kind,
                d.title,
                d.relative_path,
                d.source_path,
                c.ordinal,
                c.heading,
                c.text,
                c.metadata_json,
                sm.tune,
                sm.meter,
                sm.key_signature,
                sm.composer,
                sm.lyricist,
                sm.book_edition,
                sm.time_signature,
                sm.source_url,
                bm25(chunks_fts, 1.0, 0.7, 1.0, 0.4) AS score
            FROM chunks_fts
            JOIN chunks c ON c.chunk_id = chunks_fts.chunk_id
            JOIN documents d ON d.document_id = c.document_id
            LEFT JOIN (
                SELECT
                    document_id,
                    group_concat(DISTINCT tune) AS tune,
                    group_concat(DISTINCT meter) AS meter,
                    group_concat(DISTINCT key_signature) AS key_signature,
                    group_concat(DISTINCT composer) AS composer,
                    group_concat(DISTINCT lyricist) AS lyricist,
                    group_concat(DISTINCT book_edition) AS book_edition,
                    group_concat(DISTINCT time_signature) AS time_signature,
                    group_concat(DISTINCT source_url) AS source_url
                FROM song_metadata
                GROUP BY document_id
            ) sm ON sm.document_id = d.document_id
            WHERE chunks_fts MATCH ?
            ORDER BY score ASC
            LIMIT ?
            """,
            (fts_query(query), limit),
        ).fetchall()
    finally:
        connection.close()
    return [
        {
            "chunk_id": row["chunk_id"],
            "document_id": row["document_id"],
            "kind": row["kind"],
            "title": row["title"],
            "relative_path": row["relative_path"],
            "source_path": row["source_path"],
            "ordinal": row["ordinal"],
            "heading": row["heading"],
            "text": row["text"],
            "metadata": json.loads(row["metadata_json"]),
            "tune": row["tune"],
            "meter": row["meter"],
            "key_signature": row["key_signature"],
            "composer": row["composer"],
            "lyricist": row["lyricist"],
            "book_edition": row["book_edition"],
            "time_signature": row["time_signature"],
            "source_url": row["source_url"],
            "score": row["score"],
        }
        for row in rows
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("query", nargs="+")
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument("--json", action="store_true", dest="as_json")
    args = parser.parse_args()
    results = search(args.database.resolve(), " ".join(args.query), max(1, args.limit))
    if args.as_json:
        print(json.dumps(results, ensure_ascii=False, indent=2))
        return 0
    for index, result in enumerate(results, start=1):
        print(f"[{index}] {result['kind']}: {result['title']}  score={result['score']:.4f}")
        print(f"    {result['source_path']}  (chunk {result['ordinal']})")
        if result["heading"]:
            print(f"    {result['heading']}")
        print(f"    {result['text'][:700].replace(chr(10), ' ')}")
        print()
    if not results:
        print("No matches.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
