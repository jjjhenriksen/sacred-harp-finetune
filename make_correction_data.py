import json
from pathlib import Path

SYSTEM = "You are a careful Sacred Harp reference assistant."
EXAMPLES = [
    {
        "user": "What are the lyrics to Idumea?",
        "assistant": "Idumea is Denson song 47b in the 1991 and 2025 editions.\n\nAnd am I born to die?\nTo lay this body down!\nAnd must my trembling spirit fly\nInto a world unknown?\n\nA land of deepest shade,\nUnpierced by human thought;\nThe dreary regions of the dead,\nWhere all things are forgot!\n\nSoon as from earth I go,\nWhat will become of me?\nEternal happiness or woe\nMust then my portion be!\n\nWaked by the trumpet sound,\nI from my grave shall rise;\nAnd see the Judge with glory crowned,\nAnd see the flaming skies!",
    },
    {
        "user": "Which Sacred Harp song has the lyrics I can but perish if I go?",
        "assistant": "Fairfield, song 29t in the 1991 and 2025 Denson editions.",
    },
]

root = Path(__file__).parent / "data" / "correction"
root.mkdir(parents=True, exist_ok=True)
records = [{"messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": item["user"]}, {"role": "assistant", "content": item["assistant"]}]} for item in EXAMPLES]
for name in ("train", "valid", "test"):
    (root / (name + ".jsonl")).write_text("".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records), encoding="utf-8")
