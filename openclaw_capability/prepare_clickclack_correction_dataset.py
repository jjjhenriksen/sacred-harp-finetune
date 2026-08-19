#!/usr/bin/env python3
"""Build a correction mix for novel ClickClack discussion/message phrasing."""

from __future__ import annotations

import copy
import json
import random
from pathlib import Path

from prepare_openclaw_dataset import ALL_TOOLS, SYSTEM_PROMPT, tool_call


HERE = Path(__file__).resolve().parent
SOURCE = HERE / "data"
OUTPUT = HERE / "data_clickclack_correction"
SEED = 298131


def load(name: str) -> list[dict]:
    return [json.loads(line) for line in (SOURCE / f"{name}.jsonl").read_text().splitlines()]


def discussion_examples() -> list[dict]:
    templates = [
        "Read the most recent {limit} messages in the bound ClickClack discussion.",
        "Review up to {limit} messages from this ClickClack thread before replying.",
        "Check the latest {limit} discussion posts for mentions of a singing.",
        "Use the bound discussion's last {limit} messages as context.",
        "What are people saying in this thread? Read {limit} recent messages.",
    ]
    examples = []
    index = 0
    for limit in (8, 10, 18, 24, 30):
        for template in templates:
            prompt = template.format(limit=limit)
            examples.append({
                "source": "generated:correction:discussion",
                "kind": "discussion_route",
                "tools": ALL_TOOLS,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                    tool_call("discussion", {"limit": limit}, f"correction_discussion_{index}"),
                ],
            })
            index += 1
    return examples


def message_examples() -> list[dict]:
    current_messages = [
        ("Reply here with 'All ready.'", "All ready."),
        ("Post a quick 'Thank you' in this conversation.", "Thank you"),
        ("Send the current thread: Let's begin.", "Let's begin."),
        ("Say 'Dinner is served' in the current chat.", "Dinner is served"),
        ("Use the message tool to reply here with 'Fa sol la.'", "Fa sol la."),
        ("Put 'Page 159 next' into this conversation.", "Page 159 next"),
        ("Send a one-word hello to this thread.", "Hello"),
        ("Reply in place: Convention confirmed.", "Convention confirmed."),
    ]
    channel_messages = [
        ("general", "Let's sing Wondrous Love", "Send 'Let's sing Wondrous Love' to the ClickClack general channel."),
        ("ops", "Corpus refreshed.", "Message ClickClack channel ops: Corpus refreshed."),
        ("singing-school", "Practice starts at seven.", "Post to ClickClack channel singing-school that practice starts at seven."),
        ("general", "Dinner on the grounds!", "Tell the ClickClack general channel: Dinner on the grounds!"),
        ("requests", "Requested: Idumea", "Send a message to ClickClack channel requests saying Requested: Idumea"),
        ("ops", "Specialist check complete.", "Use message send for ClickClack ops: Specialist check complete."),
        ("announcements", "Singing begins at ten.", "Post 'Singing begins at ten.' to ClickClack channel announcements."),
        ("general", "Good singing today.", "Send ClickClack general a note: Good singing today."),
    ]
    examples = []
    index = 0
    for prompt, text in current_messages:
        examples.append({
            "source": "generated:correction:message",
            "kind": "message_route",
            "tools": ALL_TOOLS,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
                tool_call("message", {"action": "send", "message": text}, f"correction_message_{index}"),
            ],
        })
        index += 1
    for channel, text, prompt in channel_messages:
        examples.append({
            "source": "generated:correction:message",
            "kind": "message_route",
            "tools": ALL_TOOLS,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
                tool_call(
                    "message",
                    {"action": "send", "channel": "clickclack", "target": f"channel:{channel}", "message": text},
                    f"correction_message_{index}",
                ),
            ],
        })
        index += 1
    return examples


def plain_examples() -> list[dict]:
    pairs = [
        ("Hello.", "Hello. 🎼"),
        ("How's it going?", "Ready to talk Sacred Harp."),
        ("Thanks.", "You're welcome."),
        ("What is your scope?", "Sacred Harp corpus research and a tiny ClickClack interaction surface."),
        ("Run a shell command for me.", "I can't; I'm a narrowly scoped Sacred Harp specialist."),
        ("Manage my calendar.", "That's outside my Sacred Harp and ClickClack scope."),
        ("No research: say good night.", "Good night."),
        ("Are you a general agent?", "No. I'm deliberately specialized for Sacred Harp."),
    ]
    return [
        {
            "source": "generated:correction:plain",
            "kind": "plain_conversation",
            "tools": ALL_TOOLS,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": answer},
            ],
        }
        for prompt, answer in pairs
    ]


def state_examples() -> list[dict]:
    cases = [
        ("Current song: SH 47b, Idumea.", "What are the words?", "SH 47b Idumea lyrics"),
        ("We are on SH 159, Wondrous Love.", "What key?", "SH 159 Wondrous Love key"),
        ("Keep SH 130 as our subject.", "Find its differing verse.", "SH 130 differing verse linked text witness"),
        ("We're looking at Antioch.", "Who composed it?", "Antioch Sacred Harp composer"),
        ("Our tune is Mear, SH 49t.", "What's the meter?", "SH 49t Mear meter"),
        ("The text family is Boast Ye Not.", "Which songs use it?", "Boast Ye Not Sacred Harp text family songs"),
    ]
    return [
        {
            "source": "generated:correction:state",
            "kind": "stateful_rag_route",
            "tools": ALL_TOOLS,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prior},
                {"role": "assistant", "content": "I have that as the current Sacred Harp subject."},
                {"role": "user", "content": followup},
                tool_call("sacred_harp_search", {"query": query, "top_k": 5}, f"correction_state_{index}"),
            ],
        }
        for index, (prior, followup, query) in enumerate(cases)
    ]


def write(name: str, rows: list[dict]) -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / f"{name}.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8"
    )


def main() -> int:
    rng = random.Random(SEED)
    source_train = load("train")
    retention = [row for row in source_train if row.get("kind", "").startswith("retention_")][:300]
    rag = [row for row in source_train if row.get("kind") in {"rag_route", "rag_grounded_answer"}][:120]
    capability = discussion_examples() + message_examples() + plain_examples() + state_examples()
    train = retention + rag
    for _ in range(3):
        train.extend(copy.deepcopy(capability))
    rng.shuffle(train)
    valid = load("valid")
    test = load("test")
    write("train", train)
    write("valid", valid)
    write("test", test)
    manifest = {
        "seed": SEED,
        "counts": {"train": len(train), "valid": len(valid), "test": len(test)},
        "note": "Correction pass emphasizes novel ClickClack discussion/message wording; held-out test prompts are copied but never trained.",
    }
    (OUTPUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
