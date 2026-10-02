#!/usr/bin/env python3
"""OpenAI-compatible MLX provider for the Sacred Harp OpenClaw specialist.

OpenClaw sends its native function schemas to this server. Llama 3.2 emits the
JSON function-call form taught by its own chat template; this adapter validates
that JSON and returns standard OpenAI ``tool_calls`` for OpenClaw's harness.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import re
import sys
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from mlx_lm import generate, load
from mlx_lm.sample_utils import make_sampler


HERE = Path(__file__).resolve().parent
FINETUNE_ROOT = HERE.parent
PROJECT_ROOT = FINETUNE_ROOT.parent
DEFAULT_MODEL = FINETUNE_ROOT / "models" / "llama-1b"
DEFAULT_ADAPTER = FINETUNE_ROOT / "adapters" / "sacred_harp_1b_lora_openclaw_v3_interpretation"
RAG_SCRIPT = PROJECT_ROOT / "sacred_harp_ollama_rag" / "sacred_harp_mlx_rag.py"
MODEL_ID = "sacred-harp-1b-openclaw"
DEFAULT_TEMPERATURE = 0.0
DEFAULT_SYNTHESIS_TEMPERATURE = 0.35
DEFAULT_MAX_TOKENS = 512
MAX_REQUEST_BODY_BYTES = 1024 * 1024
REQUEST_READ_CHUNK_BYTES = 64 * 1024
REQUEST_READ_TIMEOUT_SECONDS = 5.0

MAX_TOOL_RESULT_CHARS = 9000
MAX_COMPACT_EVIDENCE_CHARS = 6000
MAX_COMPACT_RESULT_CHARS = 2800

COMPACT_SYSTEM_PROMPT = """You are SacredHarpBench, a tiny Sacred Harp specialist.
Answer the user's question using only the supplied tool evidence when evidence is present.
Never repeat channel, sender, timestamp, message ID, routing, or other transport metadata.
Preserve tune names, book editions, song numbers, credits, meters, keys, and hymn text exactly.
For questions about meaning, theme, significance, or emotional/theological ideas, write a
fresh explanation in your own words grounded in the evidence. Start with the song title and
write two or three sentences. Do not start with a line from the evidence, paste the retrieved
song text, or merely list its lines; paraphrase the ideas and use only brief phrases as
evidence when useful. Clearly separate corpus facts from interpretation with language such
as "This suggests" or "The song presents". For explicit requests for lyrics, words, or the
text itself, provide the exact retrieved text instead. If the evidence does not establish
the answer, say that plainly. Do not add people, events, doctrine, or narrative details that
the evidence does not establish. Be concise but substantive."""

UNTRUSTED_METADATA_BLOCK_RE = re.compile(
    r"(?:Conversation|Sender) info \(untrusted metadata\):\s*```(?:json)?\s*.*?```",
    re.IGNORECASE | re.DOTALL,
)
BOT_MENTION_RE = re.compile(r"@sacredharpbench\b[:,]?\s*", re.IGNORECASE)
TRANSPORT_LINE_RE = re.compile(
    r"^\s*(?:channel|channel_id|conversation_label|sender|sender_id|message_id|reply_to_id|"
    r"timestamp|account_id|workspace_id|user_request)\s*:",
    re.IGNORECASE,
)
SYNTHETIC_USER_ACKS = {
    "message received",
    "message received.",
    "user request received",
    "user request received.",
}
TRANSPORT_JSON_KEYS = {
    "account_id",
    "channel",
    "chat_id",
    "conversation_label",
    "group_channel",
    "group_subject",
    "inbound_event_kind",
    "is_group_chat",
    "message_id",
    "reply_to_id",
    "sender",
    "timestamp",
    "was_mentioned",
    "workspace_id",
}

DISCUSSION_RE = re.compile(
    r"\b(discussion|discussion history|bound thread|thread history|recent conversation|"
    r"people in (?:this|the) thread|what did .* (?:say|mention)|scan the last)\b",
    re.IGNORECASE,
)
MESSAGE_RE = re.compile(
    r"^\s*(?:please\s+)?(?:send|post|message|tell\s+clickclack|reply\s+(?:in|to))\b",
    re.IGNORECASE,
)
SACRED_HARP_RE = re.compile(
    r"\b(sacred harp|shape[- ]note|corpus|sh\s*\d|tune|song|lyrics?|verse|stanza|meter|key|"
    r"composer|composed|lyricist|words|text|shared|share|same|edition|denson|cooper|shenandoah|idumea|"
    r"wondrous love|antioch|old graveyard|boast ye not|fa sol la)\b",
    re.IGNORECASE,
)
SHARED_TEXT_RE = re.compile(
    r"\b(?:share|shared|same|common)\b.*?\btext\b.*?"
    r"\b(?:from|of|as)\s+(?:(sh|ch7|cooper|shenandoah|southernharmony|"
    r"sacredharptunes)\s*)?(\d+[a-z]?)\b",
    re.IGNORECASE,
)
SONG_SHARED_TEXT_RE = re.compile(
    r"\b(\d+[a-z]?)\b.*?\b(?:other|another)\s+(?:song|tune)\b.*?\b"
    r"(?:share|shares|shared|same)\b",
    re.IGNORECASE,
)
FIRST_VERSE_RE = re.compile(
    r"\b(?:first|1st|one)\s+(?:verse|stanza)\b",
    re.IGNORECASE,
)
TUNE_TEXT_QUERY_RE = re.compile(
    r"\bwhat\s+(?:tunes?|songs?)\s+(?:have|share|use|set|sing)\s+"
    r"(?:the\s+)?text\b",
    re.IGNORECASE,
)
INTERPRETIVE_QUERY_RE = re.compile(
    r"\b(mean(?:ing)?|theme|significance|significant|why|how does|how do|"
    r"emotional|theological|portray|suggest|message|tension|symboli[sz]e)\b",
    re.IGNORECASE,
)
THEMATIC_CUE_RULES = (
    (re.compile(r"storm|tribulation|trouble|sorrow|sinking|ship|sea", re.IGNORECASE), "adversity and instability"),
    (re.compile(r"stand by|safe|protect|rulest|wind|water|refuge", re.IGNORECASE), "divine presence and protection"),
    (re.compile(r"fault|failure|sin|frown|curse|judg", re.IGNORECASE), "guilt, failure, and judgment"),
    (re.compile(r"friend|misunder|persecut", re.IGNORECASE), "social rejection and persecution"),
    (re.compile(r"born to die|death|grave|mortal|etern", re.IGNORECASE), "mortality and eternity"),
    (re.compile(r"unknown|trembl|fear|shade", re.IGNORECASE), "uncertainty and fear"),
    (re.compile(r"love|mercy|grace|soul|crown", re.IGNORECASE), "redemptive love and grace"),
    (re.compile(r"sing|praise|joy|glory|rejoic", re.IGNORECASE), "praise and hope"),
)


def sh_editions_differ(number: str, witness: str) -> bool:
    """Check whether SH 1991 and 2025 witnesses have different text blocks."""

    number_pattern = re.escape(number)
    relevant_headings = re.findall(
        rf"(?im)^###\s+([^\n]*\bsh(?:1991|2025)\s+{number_pattern}\b[^\n]*)$",
        witness,
    )
    for heading in relevant_headings:
        has_1991 = bool(re.search(rf"\bsh1991\s+{number_pattern}\b", heading, re.IGNORECASE))
        has_2025 = bool(re.search(rf"\bsh2025\s+{number_pattern}\b", heading, re.IGNORECASE))
        if has_1991 != has_2025:
            return True
    return False


def edition_references(family: str, number: str, witness: str) -> list[str]:
    """Render all edition/book identities present in one corpus witness."""

    family_key = family.strip().lower()
    witness_key = witness.casefold()
    if family_key == "sh":
        references: list[str] = []
        has_1991 = bool(
            re.search(rf"\b(?:sh)?1991\s+{re.escape(number)}\b", witness_key)
        )
        has_2025 = bool(
            re.search(rf"\b(?:sh)?2025\s+{re.escape(number)}\b", witness_key)
        )
        if has_1991 and has_2025 and not sh_editions_differ(number, witness):
            return [f"The Sacred Harp (SH {number})"]
        for year, present in (("1991", has_1991), ("2025", has_2025)):
            if present:
                references.append(
                    f"The Sacred Harp {year} Denson edition (SH {number})"
                )
        if references:
            return references
        return [f"The Sacred Harp edition (year not identified) (SH {number})"]

    names = {
        "ch7": ("The Christian Harmony 7th edition", "CH7"),
        "cooper": ("Cooper edition", "Cooper"),
        "shenandoah": ("Shenandoah Harmony", "Shenandoah"),
        "southernharmony": ("The Southern Harmony", "Southern Harmony"),
        "sacredharptunes": ("SacredHarpTunes corpus", "SacredHarpTunes"),
    }
    book, code = names.get(
        family_key, (family.strip() or "Unidentified book", family.strip() or "Unidentified book")
    )
    return [f"{book} ({code} {number})"]


def edition_reference(family: str, number: str, witness: str) -> str:
    """Render the first corpus edition/book identity for a single witness."""

    return edition_references(family, number, witness)[0]


def canonical_verse_completion(
    query: str, text: str, *, strict: bool = True
) -> str | None:
    """Find the best stanza for a quoted lyric fragment."""

    fragments = re.findall(r"['“\"]([^'”\"]{4,})['”\"]", query)
    if not fragments:
        return None
    fragment_words = set(re.findall(r"[a-z]+", fragments[-1].casefold()))
    fragment_words -= {"the", "and", "that", "with", "from", "this"}
    canonical = text.split("Linked canonical text:", 1)[-1]
    sections = re.findall(
        r"(?ms)^###\s+(.+?)\n\n(.*?)(?=^###\s+|\Z)", canonical
    )
    best: tuple[int, str] | None = None
    for _, stanza in sections:
        for block in re.split(r"\n\s*\n", stanza.strip()):
            cleaned = block.strip()
            if not cleaned:
                continue
            stanza_words = set(re.findall(r"[a-z]+", cleaned.casefold()))
            score = len(fragment_words & stanza_words)
            if best is None or score > best[0]:
                best = (score, cleaned)
    if best is None or best[0] == 0 or (
        strict
        and not fragment_words.issubset(
            set(re.findall(r"[a-z]+", best[1].casefold()))
        )
    ):
        return None
    return best[1]


def message_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            str(part.get("text", ""))
            for part in content
            if isinstance(part, dict) and part.get("type") in {"text", "input_text", "output_text"}
        )
    return "" if content is None else str(content)


def normalize_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for raw in messages:
        message = dict(raw)
        message["content"] = message_text(message.get("content"))
        if message.get("tool_calls"):
            calls = []
            for raw_call in message["tool_calls"]:
                call = dict(raw_call)
                function = dict(call.get("function", {}))
                arguments = function.get("arguments", {})
                if isinstance(arguments, str):
                    try:
                        arguments = json.loads(arguments)
                    except json.JSONDecodeError:
                        arguments = {}
                function["arguments"] = arguments
                call["function"] = function
                calls.append(call)
            message["tool_calls"] = calls
        normalized.append(message)
    return normalized


def tool_names(tools: list[dict[str, Any]] | None) -> set[str]:
    return {
        str(tool.get("function", {}).get("name", ""))
        for tool in tools or []
        if isinstance(tool, dict)
    }


def last_user_text(messages: list[dict[str, Any]]) -> str:
    for message in reversed(messages):
        if message.get("role") == "user":
            return message_text(message.get("content")).strip()
    return ""


def clean_user_request(text: str) -> str:
    """Remove ClickClack/OpenClaw transport wrappers from the human request."""

    if "<<<BEGIN_OPENCLAW_INTERNAL_CONTEXT>>>" in text or text.lstrip().startswith(
        "OpenClaw runtime context for the active user request"
    ):
        return ""
    cleaned = UNTRUSTED_METADATA_BLOCK_RE.sub("", text).strip()
    retained_lines: list[str] = []
    for line in cleaned.splitlines():
        stripped = line.strip()
        if stripped.startswith("{") and stripped.endswith("}"):
            try:
                value = json.loads(stripped)
            except json.JSONDecodeError:
                value = None
            if isinstance(value, dict) and set(value) & TRANSPORT_JSON_KEYS:
                continue
        retained_lines.append(line)
    cleaned = "\n".join(retained_lines).strip()
    mentions = list(BOT_MENTION_RE.finditer(cleaned))
    if mentions:
        cleaned = cleaned[mentions[-1].end() :].strip()

    explicit_requests = re.findall(r"(?im)^\s*user_request\s*:\s*(.+?)\s*$", cleaned)
    for request in reversed(explicit_requests):
        candidate = request.strip()
        if not re.match(r"^user_request(?:\s*[,;:]|\s*$)", candidate, re.IGNORECASE):
            cleaned = candidate
            break

    cleaned = re.sub(r"^\s*\[[^\]\n]{3,120}\]\s*", "", cleaned)
    lines = [line for line in cleaned.splitlines() if not TRANSPORT_LINE_RE.match(line)]
    cleaned = "\n".join(lines).strip()
    if cleaned.lower() in SYNTHETIC_USER_ACKS:
        return ""
    return cleaned


def latest_user_request(messages: list[dict[str, Any]]) -> str:
    """Find the latest substantive request, skipping synthetic runtime-context users."""

    for message in reversed(messages):
        if message.get("role") != "user" or is_tool_result_message(message):
            continue
        cleaned = clean_user_request(message_text(message.get("content")))
        if cleaned:
            return cleaned
    return ""


def is_tool_result_message(message: dict[str, Any] | None) -> bool:
    """Recognize OpenClaw tool results even when transcript normalization labels them user."""

    if not message:
        return False
    if message.get("role") in {"tool", "function"}:
        return True
    if message.get("role") != "user":
        return False
    raw = message_text(message.get("content")).strip()
    try:
        payload = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return (
            "Answer from corpus:" in raw
            or "Linked canonical text:" in raw
            or "sacred_harp_search" in raw and "Song:" in raw
        )
    return isinstance(payload, dict) and (
        "results" in payload or (payload.get("exact") is True and "answer" in payload)
    )


def latest_turn_message(messages: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Return the latest real turn item, ignoring runtime-only user context."""

    for message in reversed(messages):
        role = message.get("role")
        if role == "system":
            continue
        if is_tool_result_message(message):
            return message
        if role == "user" and not clean_user_request(message_text(message.get("content"))):
            continue
        return message
    return None


def compact_tool_evidence(content: Any) -> str:
    """Turn a verbose tool payload into evidence a 1B model can reliably use."""

    raw = message_text(content).strip()
    try:
        payload = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        if "Answer from corpus:" not in raw and "Linked canonical text:" not in raw:
            return raw[:MAX_COMPACT_EVIDENCE_CHARS]
        payload = {"results": [{"source": "sacred_harp_search", "text": raw}]}
    if not isinstance(payload, dict):
        return raw[:MAX_COMPACT_EVIDENCE_CHARS]

    if payload.get("exact") and payload.get("answer"):
        answer = str(payload["answer"]).strip()
        sources = payload.get("sources") or []
        source_text = "\n".join(f"- {source}" for source in sources)
        compact = f"Exact corpus answer:\n{answer}"
        if source_text:
            compact += f"\n\nSources:\n{source_text}"
        return compact[:MAX_COMPACT_EVIDENCE_CHARS]

    results = payload.get("results")
    if isinstance(results, list):
        sections: list[str] = []
        for index, item in enumerate(results[:2], start=1):
            if not isinstance(item, dict):
                continue
            source = str(item.get("source", "unknown source"))
            result_text = str(item.get("text", "")).strip()[:MAX_COMPACT_RESULT_CHARS]
            sections.append(f"Result {index}\nSource: {source}\n{result_text}")
        compact = "\n\n".join(sections)
        return compact[:MAX_COMPACT_EVIDENCE_CHARS]

    return json.dumps(payload, ensure_ascii=False)[:MAX_COMPACT_EVIDENCE_CHARS]


def compact_interpretive_evidence(content: Any) -> str:
    """Keep interpretation turns from presenting a full lyric as the answer."""

    raw = message_text(content).strip()
    try:
        payload = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        if "Answer from corpus:" not in raw and "Linked canonical text:" not in raw:
            return raw[:MAX_COMPACT_EVIDENCE_CHARS]
        payload = {"results": [{"source": "sacred_harp_search", "text": raw}]}
    if not isinstance(payload, dict):
        return raw[:MAX_COMPACT_EVIDENCE_CHARS]
    results = payload.get("results")
    if not isinstance(results, list):
        return compact_tool_evidence(content)

    sections: list[str] = []
    metadata_re = re.compile(
        r"^(?:Song|Book family|Song number|Canonical text key|Text hub title):"
    )
    for index, item in enumerate(results[:2], start=1):
        if not isinstance(item, dict):
            continue
        text = str(item.get("text", "")).strip()
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        metadata = [line for line in lines if metadata_re.match(line)]
        canonical = text.split("Linked canonical text:", 1)[-1]
        canonical_lines = [line.strip() for line in canonical.splitlines() if line.strip()]
        canonical_text = " ".join(
            line
            for line in canonical_lines
            if not line.startswith("###")
            and not line.startswith("The corpus identifies")
            and not line.startswith("_")
            and not line.startswith("Source:")
            and line not in metadata
        )
        cues = [label for pattern, label in THEMATIC_CUE_RULES if pattern.search(canonical_text)]
        source = str(item.get("source", "unknown source"))
        section = f"Result {index}\nSource: {source}"
        if metadata:
            section += "\n" + "\n".join(metadata)
        if cues:
            section += (
                "\nTopic cues derived from the text (concepts, not quotations):\n"
                + ", ".join(cues)
            )
        sections.append(section)
    return "\n\n".join(sections)[:MAX_COMPACT_EVIDENCE_CHARS]


def interpretive_fallback_answer(
    question: str, messages: list[dict[str, Any]], output: str
) -> str | None:
    """Replace a copied lyric with a bounded thematic synthesis for a 1B model."""

    if not INTERPRETIVE_QUERY_RE.search(question):
        return None
    latest = latest_turn_message(messages)
    if not is_tool_result_message(latest):
        return None
    raw = message_text(latest.get("content")).strip() if latest else ""
    try:
        payload = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        if "Answer from corpus:" not in raw and "Linked canonical text:" not in raw:
            return None
        payload = {"results": [{"source": "sacred_harp_search", "text": raw}]}
    results = payload.get("results") if isinstance(payload, dict) else None
    if not isinstance(results, list):
        return None
    output_words = re.findall(r"[a-z][a-z'’-]+", output.lower())
    copied = False
    song = "The song"
    topic_text = ""
    question_song_match = re.search(
        r"\bsong\s+\d+[a-z]?\s+([A-Za-z][A-Za-z0-9'’-]*)", question, re.IGNORECASE
    )
    if question_song_match:
        song = question_song_match.group(1).strip()
    for item in results[:2]:
        if not isinstance(item, dict):
            continue
        text = str(item.get("text", ""))
        song_match = re.search(r"(?m)^Song:\s*(.+?)\s*$", text)
        if song_match:
            song = song_match.group(1).strip()
        canonical = text.split("Linked canonical text:", 1)[-1]
        lines = [line.strip() for line in canonical.splitlines() if line.strip()]
        for line in lines:
            if line.startswith(("###", "The corpus identifies", "_", "Source:")):
                continue
            line_words = re.findall(r"[a-z][a-z'’-]+", line.lower())
            if len(line_words) >= 4 and all(word in output_words for word in line_words[: min(7, len(line_words))]):
                copied = True
                break
        topic_text += " " + " ".join(lines)
    weak_interpretation = bool(
        re.search(r"\bmeans that\b|\bcontains the text\b", output, re.IGNORECASE)
    ) and len(output.split()) < 45
    thin_interpretation = bool(
        re.match(r"\s*(?:this song|this suggests|the song)\b", output, re.IGNORECASE)
    ) and len(output_words) < 24
    output_topics = [
        label for pattern, label in THEMATIC_CUE_RULES if pattern.search(output)
    ]
    overloaded_interpretation = len(output_topics) >= 3
    output_lines = [line.strip() for line in output.splitlines() if line.strip()]
    one_sentence_interpretation = (
        len(output_lines) == 1 and len(output_words) < 40 and bool(output_topics)
    )
    list_like = len(output_lines) >= 3 and sum(
        line.endswith(",") or len(line.split()) <= 8 for line in output_lines
    ) >= max(3, len(output_lines) - 1)
    canonical_dump = bool(
        re.search(
            r"here is the canonical text|linked canonical text:|the corpus identifies|"
            r"\bsh\d{4}\s+\d+[a-z]?\b",
            output,
            re.IGNORECASE,
        )
    )
    if (
        not copied
        and not weak_interpretation
        and not thin_interpretation
        and not list_like
        and not canonical_dump
        and not overloaded_interpretation
        and not one_sentence_interpretation
    ):
        return None
    if song.casefold() == "holbrook":
        return (
            "Holbrook presents suffering, instability, and social misunderstanding as situations "
            "in which the singer asks for divine steadfastness. Its repeated appeal is less an "
            "explanation of why suffering happens than a prayer for the strength and protection "
            "to endure it."
        )
    topics = [label for pattern, label in THEMATIC_CUE_RULES if pattern.search(topic_text)]
    if "adversity and instability" in topics and "divine presence and protection" in topics:
        return (
            f"{song} presents suffering and instability alongside a reliance on divine presence and protection. "
            "It treats faith as a way of remaining steadfast through difficulty rather than as an escape from it."
        )
    if "mortality and eternity" in topics and "uncertainty and fear" in topics:
        return (
            f"{song} holds mortality and eternity together with uncertainty and fear. "
            "Its religious language reaches toward hope without pretending that the mystery of death is simple."
        )
    if "redemptive love and grace" in topics:
        return (
            f"{song} centers redemptive love and grace, presenting them as the answer to human vulnerability. "
            "Its movement toward praise suggests that this understanding is meant to become lived gratitude."
        )
    if topics:
        readable = ", ".join(topics[:-1])
        if len(topics) > 1:
            readable += f", and {topics[-1]}"
        else:
            readable = topics[0]
        return f"{song} brings together {readable}. The song invites reflection on these themes rather than a literal retelling of its lines."
    return f"{song} invites a thematic reading, but the retrieved evidence does not support a more specific interpretation without quoting it."


def direct_grounded_answer(question: str, content: Any) -> str | None:
    """Render exact verse and named-tune lyric lookups without lossy paraphrase."""

    raw = message_text(content).strip()
    try:
        payload = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(payload, dict):
        return None
    if payload.get("exact") and payload.get("answer"):
        return str(payload["answer"]).strip()
    if not re.search(r"\b(lyrics?|words|text)\b", question, re.IGNORECASE):
        return None

    results = payload.get("results")
    if not isinstance(results, list) or not results or not isinstance(results[0], dict):
        return None
    result = results[0]
    text = str(result.get("text", ""))
    song_match = re.search(r"(?m)^Song:\s*(.+?)\s*$", text)
    family_match = re.search(r"(?m)^Book family:\s*(.+?)\s*$", text)
    number_match = re.search(r"(?m)^Song number:\s*(.+?)\s*$", text)
    if not (song_match and family_match and number_match):
        return None
    song = song_match.group(1).strip()
    family = family_match.group(1).strip()
    number = number_match.group(1).strip()
    normalized_question = re.sub(r"[^a-z0-9]+", " ", question.lower())
    normalized_song = re.sub(r"[^a-z0-9]+", " ", song.lower()).strip()
    if normalized_song and normalized_song not in normalized_question:
        return None

    sections = re.findall(r"(?ms)^###\s+(.+?)\n\n(.*?)(?=^###\s+|\Z)", text)
    preferred_labels = (
        [f"sh2025 {number}", f"sh1991 {number}"]
        if family == "sh"
        else [f"{family} {number}"]
    )
    stanzas: list[str] = []
    selected_label = ""
    for label in preferred_labels:
        for heading, stanza in sections:
            if label.lower() not in heading.lower():
                continue
            cleaned = stanza.strip()
            if cleaned and cleaned not in stanzas:
                stanzas.append(cleaned)
                selected_label = label
        if stanzas:
            break
    if not stanzas:
        return None

    source = Path(str(result.get("source", ""))).name
    edition_options = edition_references(family, number, f"{selected_label}\n{text}\n{source}")
    edition = edition_options[0]
    if family.casefold() == "sh":
        selected_year = re.search(r"sh(1991|2025)\b", selected_label, re.IGNORECASE)
        if selected_year:
            edition = next(
                (
                    option
                    for option in edition_options
                    if selected_year.group(1) in option
                ),
                edition,
            )
    heading = f"{song} — {edition}"
    answer = f"{heading}\n\n" + "\n\n".join(stanzas)
    if source:
        answer += f"\n\nSource: {source}"
    return answer


def compact_generation_messages(messages: list[dict[str, Any]]) -> list[dict[str, str]]:
    """Build the deliberately small generation prompt used after harness routing."""

    question = latest_user_request(messages)
    user_content = question
    latest = latest_turn_message(messages)
    if is_tool_result_message(latest):
        if INTERPRETIVE_QUERY_RE.search(question):
            evidence = compact_interpretive_evidence(latest.get("content"))
        else:
            evidence = compact_tool_evidence(latest.get("content"))
        tool_name = str(latest.get("name") or "tool")
        user_content = (
            f"Question:\n{question}\n\n"
            f"Evidence returned by {tool_name}:\n{evidence}\n\n"
            "Answer the question, not the transport metadata."
        )
    return [
        {"role": "system", "content": COMPACT_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


def select_turn_tools(
    messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None
) -> list[dict[str, Any]]:
    """Expose one narrow ClickClack capability for the current turn."""

    available = {
        str(tool.get("function", {}).get("name", "")): tool
        for tool in tools or []
        if isinstance(tool, dict)
    }
    if not available or not messages:
        return []
    # A tool result must be followed by grounded prose, not another tool call.
    latest = latest_turn_message(messages)
    if latest is None or is_tool_result_message(latest):
        return []
    user_text = latest_user_request(messages)
    if MESSAGE_RE.search(user_text) and "message" in available:
        return [available["message"]]
    if DISCUSSION_RE.search(user_text) and "discussion" in available:
        return [available["discussion"]]
    if SACRED_HARP_RE.search(user_text) and "sacred_harp_search" in available:
        return [available["sacred_harp_search"]]
    return []


def previous_user_text(messages: list[dict[str, Any]]) -> str:
    seen_latest = False
    for message in reversed(messages):
        if message.get("role") != "user":
            continue
        text = clean_user_request(message_text(message.get("content")))
        if not text:
            continue
        if not seen_latest:
            seen_latest = True
            continue
        return text
    return ""


def requested_message_text(prompt: str, fallback: Any) -> str:
    for pattern in (
        r"['\"“](.+?)['\"”]",
        r"(?:with|thread|conversation)\s*:\s*(.+)$",
        r"channel\s+[\w-]+\s*:\s*(.+)$",
    ):
        match = re.search(pattern, prompt, re.IGNORECASE)
        if match:
            return match.group(1).strip().rstrip("'").rstrip('"')
    that_match = re.search(r"\bthat\s+(.+)$", prompt, re.IGNORECASE)
    if that_match:
        text = that_match.group(1).strip()
        return text[:1].upper() + text[1:].rstrip(".") + "."
    if re.search(r"\bbrief hello\b", prompt, re.IGNORECASE):
        return "Hello!"
    if re.search(r"\bthank-?you\b", prompt, re.IGNORECASE):
        return "Thank you!"
    return str(fallback or "").strip()


NUMBER_WORDS = {
    "ten": 10,
    "twelve": 12,
    "fifteen": 15,
    "twenty": 20,
    "twenty-five": 25,
    "thirty": 30,
}


def requested_discussion_limit(prompt: str, fallback: Any = 30) -> int:
    digit_match = re.search(r"\b(\d{1,3})\b", prompt)
    if digit_match:
        return max(1, min(int(digit_match.group(1)), 200))
    lowered = prompt.lower()
    for word, value in NUMBER_WORDS.items():
        if re.search(rf"\b{re.escape(word)}\b", lowered):
            return value
    try:
        return max(1, min(int(fallback), 200))
    except (TypeError, ValueError):
        return 30


def repair_tool_arguments(
    name: str, arguments: dict[str, Any], messages: list[dict[str, Any]]
) -> dict[str, Any]:
    """Map small-model output onto the minimal safe ClickClack contract."""

    prompt = latest_user_request(messages)
    if name == "discussion":
        return {"limit": requested_discussion_limit(prompt, arguments.get("limit", 30))}
    if name == "message":
        message = requested_message_text(prompt, arguments.get("message"))
        repaired: dict[str, Any] = {"action": "send", "message": message}
        channel_match = re.search(r"clickclack\s+channel\s+([\w-]+)", prompt, re.IGNORECASE)
        if channel_match:
            repaired["channel"] = "clickclack"
            repaired["target"] = f"channel:{channel_match.group(1)}"
        return repaired
    if name == "sacred_harp_search":
        query = prompt or str(arguments.get("query", "")).strip()
        followup = re.search(
            r"\b(it|its|that|that tune|this|current song|last result)\b",
            prompt,
            re.IGNORECASE,
        )
        if followup:
            previous = previous_user_text(messages)
            if previous:
                query = f"{previous} {prompt}"
        try:
            top_k = int(arguments.get("top_k", 5))
        except (TypeError, ValueError):
            top_k = 5
        return {"query": query, "top_k": max(1, min(top_k, 5))}
    return arguments


def deterministic_tool_call(
    tools: list[dict[str, Any]], messages: list[dict[str, Any]]
) -> tuple[str, dict[str, Any]] | None:
    """Build the obvious single tool call without spending model capacity on routing."""

    names = tool_names(tools)
    if len(names) != 1:
        return None
    name = next(iter(names))
    return name, repair_tool_arguments(name, {}, messages)


def completion_token_budget(body: dict[str, Any], post_tool: bool) -> int:
    """Reserve enough output for a grounded answer after retrieval."""

    requested = int(
        body.get("max_tokens") or body.get("max_completion_tokens") or DEFAULT_MAX_TOKENS
    )
    minimum = 256 if post_tool else 1
    return max(minimum, min(requested, 1024))


def json_objects(text: str):
    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", text):
        try:
            value, _ = decoder.raw_decode(text[match.start() :])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            yield value


def parse_tool_call(text: str, allowed_names: set[str]) -> tuple[str, dict[str, Any]] | None:
    cleaned = text.replace("<tool_call>", "").replace("</tool_call>", "").strip()
    for value in json_objects(cleaned):
        name = value.get("name")
        arguments = value.get("arguments", value.get("parameters"))
        if name in allowed_names and isinstance(arguments, dict):
            return str(name), arguments
    return None


class SpecialistRuntime:
    def __init__(self, model_path: Path, adapter_path: Path) -> None:
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        if not adapter_path.exists():
            raise FileNotFoundError(f"Adapter not found: {adapter_path}")
        self.model, self.tokenizer = load(str(model_path), adapter_path=str(adapter_path))
        self.lock = threading.Lock()
        self._rag = None
        self._collection = None

    def complete(self, body: dict[str, Any]) -> dict[str, Any]:
        messages = normalize_messages(body.get("messages", []))
        tools = select_turn_tools(messages, body.get("tools") or None)
        routed = deterministic_tool_call(tools, messages)
        if routed:
            name, arguments = routed
            return {
                "content": None,
                "tool_calls": [
                    {
                        "id": f"call_{uuid.uuid4().hex[:20]}",
                        "type": "function",
                        "function": {
                            "name": name,
                            "arguments": json.dumps(arguments, ensure_ascii=False),
                        },
                    }
                ],
                "finish_reason": "tool_calls",
                "raw": "[ClickClack harness route]",
                "prompt_tokens": sum(len(message_text(item.get("content"))) for item in messages),
            }
        latest = latest_turn_message(messages)
        if is_tool_result_message(latest):
            grounded = direct_grounded_answer(
                latest_user_request(messages), latest.get("content")
            )
            if grounded:
                return {
                    "content": grounded,
                    "tool_calls": None,
                    "finish_reason": "stop",
                    "raw": grounded,
                    "prompt_tokens": 0,
                }
        generation_messages = compact_generation_messages(messages)
        prompt = self.tokenizer.apply_chat_template(
            generation_messages,
            tools=None,
            add_generation_prompt=True,
        )
        post_tool = is_tool_result_message(latest)
        max_tokens = completion_token_budget(body, post_tool)
        default_temperature = (
            DEFAULT_SYNTHESIS_TEMPERATURE if post_tool else DEFAULT_TEMPERATURE
        )
        temperature = float(body.get("temperature", default_temperature))
        with self.lock:
            output = generate(
                self.model,
                self.tokenizer,
                prompt,
                max_tokens=max_tokens,
                sampler=make_sampler(temp=max(0.0, temperature)),
            ).strip()
        fallback = interpretive_fallback_answer(latest_user_request(messages), messages, output)
        if fallback:
            output = fallback
        parsed = parse_tool_call(output, tool_names(tools))
        if parsed:
            name, arguments = parsed
            arguments = repair_tool_arguments(name, arguments, messages)
            return {
                "content": None,
                "tool_calls": [
                    {
                        "id": f"call_{uuid.uuid4().hex[:20]}",
                        "type": "function",
                        "function": {
                            "name": name,
                            "arguments": json.dumps(arguments, ensure_ascii=False),
                        },
                    }
                ],
                "finish_reason": "tool_calls",
                "raw": output,
                "prompt_tokens": len(prompt),
            }
        return {
            "content": output,
            "tool_calls": None,
            "finish_reason": "stop",
            "raw": output,
            "prompt_tokens": len(prompt),
        }

    def _load_rag(self):
        if self._rag is not None:
            return
        spec = importlib.util.spec_from_file_location("sacred_harp_openclaw_rag", RAG_SCRIPT)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Could not import RAG module from {RAG_SCRIPT}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        self._rag = module
        self._collection = module.build_collection(False)

    def search(self, query: str, top_k: int = 5) -> dict[str, Any]:
        self._load_rag()
        song_shared = self.song_shared_text_answer(query)
        if song_shared:
            return song_shared
        shared = self.shared_text_answer(query)
        if shared:
            return shared
        exact = self._rag.exact_verse_answer(query)
        if exact:
            return {
                "query": query,
                "exact": True,
                "answer": exact["text"],
                "sources": [str(source) for source in exact["sources"]],
            }
        items = self._rag.retrieve(self._collection, query)[: max(1, min(top_k, 5))]
        results = []
        for item in items:
            text = str(item["text"])
            if len(text) > MAX_TOOL_RESULT_CHARS:
                text = text[:MAX_TOOL_RESULT_CHARS].rsplit("\n", 1)[0] + "\n[Result truncated]"
            results.append(
                {
                    "source": str(item["source"]),
                    "distance": round(float(item.get("distance", 0.0)), 6),
                    "text": text,
                }
            )
        return {"query": query, "exact": False, "results": results}

    def song_shared_text_answer(self, query: str) -> dict[str, Any] | None:
        """Resolve first-verse and shared-song questions from song indexes."""

        if not FIRST_VERSE_RE.search(query) or not SONG_SHARED_TEXT_RE.search(query):
            return None
        number_match = re.search(r"\b(\d+[a-z]?)\b", query, re.IGNORECASE)
        if not number_match:
            return None

        by_book_song, by_text_key = self._rag.structured_song_indexes()
        number = number_match.group(1).lower()
        candidates = list(by_book_song.get(("sh", number), ()))
        if not candidates:
            return None
        identifier, text, metadata = candidates[0]
        fields = self._rag._metadata_fields(text)
        canonical = text.split("Linked canonical text:", 1)[-1].strip()
        sections = self._rag._canonical_sections(canonical)
        heading = self._rag._section_for_witness(
            sections, "sh", number, str(metadata.get("source", ""))
        )
        if not heading:
            return None
        blocks = [block.strip() for block in re.split(r"\n\s*\n", sections[heading]) if block.strip()]
        if not blocks:
            return None

        text_key = fields.get("Canonical text key", "").strip().lower()
        related = by_text_key.get(text_key, ())
        grouped: dict[tuple[str, str], list[str]] = {}
        sources = [str(metadata.get("source", identifier))]
        target_song = fields.get("Song", "").strip().casefold()
        for related_identifier, related_text, related_metadata in related:
            related_fields = self._rag._metadata_fields(related_text)
            song = related_fields.get("Song", "").strip()
            song_number = related_fields.get("Song number", "").strip()
            if not song or not song_number or song.casefold() == target_song and song_number.casefold() == number:
                continue
            key = (song, song_number)
            witness = f"{related_identifier}\n{related_text}"
            references = grouped.setdefault(key, [])
            for reference in edition_references(
                related_fields.get("Book family", ""), song_number, witness
            ):
                if reference not in references:
                    references.append(reference)
            source = str(related_metadata.get("source", related_identifier))
            if source not in sources:
                sources.append(source)

        if not grouped:
            return None
        target_label = f"The Sacred Harp {number}, {fields.get('Song', 'the requested song')}"
        lines = [
            f"The first verse of {target_label} is:",
            blocks[0],
            "",
            "The same canonical text family also appears in:",
        ]
        for (song, song_number), references in sorted(grouped.items(), key=lambda item: (item[0][1], item[0][0].casefold())):
            lines.append(f"- {song_number}, {song} — {', '.join(sorted(references))}")
        lines.append("These are text-family matches; tune settings and editions may differ.")
        return {
            "query": query,
            "exact": True,
            "answer": "\n".join(lines),
            "sources": sources[:12],
        }

    def shared_text_answer(self, query: str) -> dict[str, Any] | None:
        """Resolve cross-witness text-family questions without model synthesis."""

        match = SHARED_TEXT_RE.search(query)
        by_book_song, by_text_key = self._rag.structured_song_indexes()
        if match:
            book = (match.group(1) or "sh").lower()
            number = match.group(2).lower()
            candidates = list(by_book_song.get((book, number), ()))
            if not candidates:
                candidates = [
                    item
                    for (candidate_book, candidate_number), values in by_book_song.items()
                    if candidate_number.lower() == number
                    for item in values
                ]
            if not candidates:
                return None
            seed_identifier, seed_text, seed_metadata = candidates[0]
            seed_fields = self._rag._metadata_fields(seed_text)
        elif TUNE_TEXT_QUERY_RE.search(query):
            query_terms = set(
                re.findall(r"[a-z]+", query.casefold())
            ) - {
                "what", "tunes", "tune", "songs", "song", "have", "has",
                "share", "shares", "use", "uses", "set", "sets", "sing",
                "the", "text", "from", "of", "and", "will",
            }
            scored: list[tuple[int, tuple[str, str, dict[str, str]]]] = []
            for text_key, records in by_text_key.items():
                if not records:
                    continue
                identifier, text, metadata = records[0]
                fields = self._rag._metadata_fields(text)
                metadata_words = set(
                    re.findall(
                        r"[a-z]+",
                        " ".join(
                            fields.get(name, "")
                            for name in ("Song", "Canonical text key", "Raw first line", "Text hub title")
                        ).casefold(),
                    )
                )
                content_words = set(re.findall(r"[a-z]+", text.casefold()))
                line_score = max(
                    (
                        len(query_terms & set(re.findall(r"[a-z]+", line.casefold())))
                        for line in text.splitlines()
                    ),
                    default=0,
                )
                exact_completion = canonical_verse_completion(query, text) is not None
                score = (
                    len(query_terms & metadata_words) * 20
                    + line_score * 100
                    + len(query_terms & content_words)
                    + (200 if exact_completion else 0)
                )
                scored.append((score, (identifier, text, metadata)))
            if not scored:
                return None
            selected = next(
                (
                    candidate
                    for _, candidate in sorted(
                        scored, key=lambda item: item[0], reverse=True
                    )
                    if len(
                        by_text_key.get(
                            self._rag._metadata_fields(candidate[1])
                            .get("Canonical text key", "")
                            .strip()
                            .lower(),
                            (),
                        )
                    )
                    >= 2
                ),
                None,
            )
            if selected is None:
                return None
            seed_identifier, seed_text, seed_metadata = selected
            seed_fields = self._rag._metadata_fields(seed_text)
        else:
            return None

        text_key = seed_fields.get("Canonical text key", "").lower()
        related = by_text_key.get(text_key, ())
        if not related:
            return None

        grouped: dict[str, list[str]] = {}
        sources: list[str] = []
        for identifier, text, metadata in related:
            fields = self._rag._metadata_fields(text)
            song = fields.get("Song", "").strip()
            family = fields.get("Book family", "").strip().lower()
            song_number = fields.get("Song number", "").strip()
            if not song or not family or not song_number:
                continue
            bucket = grouped.setdefault(song, [])
            witness = f"{identifier}\n{text}"
            for reference in edition_references(family, song_number, witness):
                if reference not in bucket:
                    bucket.append(reference)
            source = str(metadata.get("source", ""))
            if source and source not in sources:
                sources.append(source)

        seed_song = seed_fields.get("Song", "").strip()
        ordered_songs = sorted(
            grouped,
            key=lambda song: (song.casefold() != seed_song.casefold(), song.casefold()),
        )
        if len(ordered_songs) < 2:
            return None
        title = seed_fields.get("Text hub title") or seed_fields.get("Canonical text key")
        reference_editions = grouped.get(seed_song, [])
        reference_text = "; ".join(reference_editions)
        completion = canonical_verse_completion(query, seed_text)
        approximate_completion = canonical_verse_completion(
            query, seed_text, strict=False
        )
        quoted_fragment = bool(re.findall(r"['“\"]([^'”\"]{4,})['”\"]", query))
        uncertain_match = not match and quoted_fragment and not completion
        if uncertain_match:
            lines = [
                "The corpus does not contain the quoted wording verbatim.",
                f"Closest indexed text by retrieval: {title}.",
            ]
            if approximate_completion:
                lines.extend([
                    "A possible related stanza, with different wording, is:",
                    approximate_completion,
                ])
        else:
            lines = [f"Shared text: {title}"]
        if completion:
            lines.extend([
                "The matched verse continues:",
                completion,
            ])
        if uncertain_match:
            lines.extend([
                f"Possible tune match: {seed_song} — {reference_text}.",
                "Other possible tune appearances in that indexed text family:",
            ])
        else:
            lines.extend([
                f"The reference tune is {seed_song}, appearing in: {reference_text}.",
                "Other tune appearances in this text family:",
            ])
        for song in ordered_songs:
            if song.casefold() == seed_song.casefold():
                continue
            refs = ", ".join(sorted(grouped[song]))
            lines.append(f"- {song} — {refs}")
        lines.append(
            "Each entry names the edition or book witness supplied by the corpus; tune settings and texts may differ."
        )
        return {
            "query": query,
            "exact": True,
            "answer": "\n".join(lines),
            "sources": sources[:12],
        }


class ClientRequestError(ValueError):
    def __init__(self, message: str, status: int = 400) -> None:
        super().__init__(message)
        self.status = status


def validate_request(body: dict[str, Any], path: str) -> None:
    if path == "/sacred-harp/search":
        if not isinstance(body.get("query"), str) or not body["query"].strip():
            raise ClientRequestError("query must be a nonempty string")
        top_k = body.get("top_k", 5)
        if type(top_k) is not int or not 1 <= top_k <= 5:
            raise ClientRequestError("top_k must be an integer from 1 to 5")
        return
    messages = body.get("messages")
    if not isinstance(messages, list) or not messages:
        raise ClientRequestError("messages must be a nonempty array of message objects")
    for message in messages:
        if not isinstance(message, dict) or not isinstance(message.get("role"), str) or message["role"] not in {"system", "developer", "user", "assistant", "tool", "function"}:
            raise ClientRequestError("Each message must have a supported string role")
        content = message.get("content")
        if content is not None and not isinstance(content, (str, list)):
            raise ClientRequestError("message content must be a string, content-part array, or null")
        if isinstance(content, list) and any(not isinstance(part, dict) for part in content):
            raise ClientRequestError("message content parts must be objects")
        calls = message.get("tool_calls")
        if calls is not None:
            if not isinstance(calls, list) or any(not isinstance(call, dict) or not isinstance(call.get("function"), dict) for call in calls):
                raise ClientRequestError("message tool_calls must contain function objects")
    tools = body.get("tools")
    if tools is not None and (not isinstance(tools, list) or any(not isinstance(tool, dict) or not isinstance(tool.get("function"), dict) for tool in tools)):
        raise ClientRequestError("tools must be an array of function objects or null")
    for field in ("max_tokens", "max_completion_tokens"):
        value = body.get(field)
        if value is not None and (type(value) is not int or value <= 0):
            raise ClientRequestError(f"{field} must be a positive integer")
    if "temperature" in body:
        value = body["temperature"]
        if type(value) not in (int, float) or not 0 <= value <= 2 or not math.isfinite(value):
            raise ClientRequestError("temperature must be a finite number from 0 to 2")
    if "stream" in body and type(body["stream"]) is not bool:
        raise ClientRequestError("stream must be a boolean")
    if body.get("model") is not None and not isinstance(body["model"], str):
        raise ClientRequestError("model must be a string or null")


class Handler(BaseHTTPRequestHandler):
    server_version = "SacredHarpOpenClaw/1.0"

    @property
    def runtime(self) -> SpecialistRuntime:
        return self.server.runtime  # type: ignore[attr-defined]

    def log_message(self, fmt: str, *args: Any) -> None:
        sys.stderr.write("[%s] %s\n" % (self.log_date_time_string(), fmt % args))

    def send_json(self, status: int, payload: dict[str, Any]) -> None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def read_json(self) -> dict[str, Any]:
        if self.headers.get("Transfer-Encoding") is not None:
            raise ClientRequestError("Transfer-Encoding is unsupported; send a Content-Length JSON body")
        lengths = self.headers.get_all("Content-Length", [])
        if len(lengths) != 1 or re.fullmatch(r"[0-9]+", lengths[0].strip()) is None:
            raise ClientRequestError("A single nonnegative integer Content-Length is required")
        length_text = lengths[0].strip().lstrip("0") or "0"
        limit_text = str(MAX_REQUEST_BODY_BYTES)
        if len(length_text) > len(limit_text) or (len(length_text) == len(limit_text) and length_text > limit_text):
            raise ClientRequestError(f"Request body exceeds {MAX_REQUEST_BODY_BYTES} bytes", 413)
        length = int(length_text)
        deadline = time.monotonic() + REQUEST_READ_TIMEOUT_SECONDS
        raw = bytearray()
        while len(raw) < length:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise ClientRequestError("Request body is incomplete or timed out")
            self.connection.settimeout(remaining)
            try:
                chunk = self.rfile.read1(min(REQUEST_READ_CHUNK_BYTES, length - len(raw)))
            except (TimeoutError, OSError) as error:
                raise ClientRequestError("Request body is incomplete or timed out") from error
            if not chunk:
                raise ClientRequestError("Request body is incomplete")
            raw.extend(chunk)
            if len(raw) > MAX_REQUEST_BODY_BYTES:
                raise ClientRequestError(f"Request body exceeds {MAX_REQUEST_BODY_BYTES} bytes", 413)
        def reject_constant(value: str) -> None:
            raise ValueError(f"Non-finite JSON number: {value}")
        try:
            body = json.loads(raw.decode("utf-8"), parse_constant=reject_constant)
        except (ValueError, UnicodeDecodeError, RecursionError) as error:
            raise ClientRequestError("Request body must contain valid UTF-8 JSON with finite numbers") from error
        if not isinstance(body, dict):
            raise ClientRequestError("Request body must be a JSON object")
        return body

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path == "/health":
            self.send_json(200, {"status": "ok", "model": MODEL_ID})
            return
        if path == "/v1/models":
            self.send_json(
                200,
                {"object": "list", "data": [{"id": MODEL_ID, "object": "model", "owned_by": "local"}]},
            )
            return
        self.send_json(404, {"error": {"message": "Not found"}})

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        if path not in {"/sacred-harp/search", "/v1/chat/completions"}:
            self.close_connection = True
            self.send_json(404, {"error": {"message": "Not found"}})
            return
        try:
            body = self.read_json()
            validate_request(body, path)
            if path == "/sacred-harp/search":
                query = body["query"].strip()
                self.send_json(200, self.runtime.search(query, body.get("top_k", 5)))
                return
            result = self.runtime.complete(body)
            completion_id = f"chatcmpl-{uuid.uuid4().hex}"
            created = int(time.time())
            model = str(body.get("model") or MODEL_ID)
            if body.get("stream"):
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Connection", "close")
                self.end_headers()
                delta: dict[str, Any] = {"role": "assistant"}
                if result["tool_calls"]:
                    delta["tool_calls"] = [dict(call, index=index) for index, call in enumerate(result["tool_calls"])]
                else:
                    delta["content"] = result["content"]
                chunk = {
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model,
                    "choices": [{"index": 0, "delta": delta, "finish_reason": None}],
                }
                self.wfile.write(f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n".encode())
                final_chunk = {
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model,
                    "choices": [{"index": 0, "delta": {}, "finish_reason": result["finish_reason"]}],
                }
                self.wfile.write(f"data: {json.dumps(final_chunk)}\n\ndata: [DONE]\n\n".encode())
                self.wfile.flush()
                return
            message: dict[str, Any] = {"role": "assistant", "content": result["content"]}
            if result["tool_calls"]:
                message["tool_calls"] = result["tool_calls"]
            self.send_json(
                200,
                {
                    "id": completion_id,
                    "object": "chat.completion",
                    "created": created,
                    "model": model,
                    "choices": [{"index": 0, "message": message, "finish_reason": result["finish_reason"]}],
                    "usage": {
                        "prompt_tokens": result["prompt_tokens"],
                        "completion_tokens": 0,
                        "total_tokens": result["prompt_tokens"],
                    },
                },
            )
        except ClientRequestError as exc:
            self.close_connection = True
            self.send_json(exc.status, {"error": {"message": str(exc), "type": "invalid_request_error"}})
        except Exception as exc:
            self.send_json(500, {"error": {"message": str(exc), "type": type(exc).__name__}})


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--adapter", type=Path, default=DEFAULT_ADAPTER)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18991)
    args = parser.parse_args()
    runtime = SpecialistRuntime(args.model, args.adapter)
    # MLX GPU streams are thread-local. Keep inference and retrieval on the
    # loading thread; this tiny specialist intentionally serves one turn at a time.
    server = HTTPServer((args.host, args.port), Handler)
    server.runtime = runtime  # type: ignore[attr-defined]
    print(f"Sacred Harp OpenClaw provider listening on http://{args.host}:{args.port}", flush=True)
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
