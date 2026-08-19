import json
import unittest

from sacred_harp_openclaw_server import (
    MAX_COMPACT_EVIDENCE_CHARS,
    clean_user_request,
    canonical_verse_completion,
    completion_token_budget,
    compact_generation_messages,
    compact_tool_evidence,
    direct_grounded_answer,
    edition_reference,
    latest_user_request,
    repair_tool_arguments,
    select_turn_tools,
)


class CompactClickClackPromptTests(unittest.TestCase):
    def test_removes_clickclack_metadata_and_bot_mention(self):
        wrapped = """Conversation info (untrusted metadata):
```json
{"channel": "chn_test", "sender": "usr_test"}
```

@sacredharpbench What are the lyrics to Idumea? Use the Sacred Harp corpus."""
        self.assertEqual(
            clean_user_request(wrapped),
            "What are the lyrics to Idumea? Use the Sacred Harp corpus.",
        )

    def test_rag_query_uses_clean_request(self):
        messages = [
            {
                "role": "user",
                "content": "channel: channel:chn_test\nsender: usr_test\nuser_request: What is SH 130?",
            }
        ]
        repaired = repair_tool_arguments("sacred_harp_search", {}, messages)
        self.assertEqual(repaired["query"], "What is SH 130?")

    def test_skips_openclaw_runtime_context_user_after_real_request(self):
        messages = [
            {
                "role": "user",
                "content": "@sacredharpbench What are the lyrics to Idumea?",
            },
            {
                "role": "user",
                "content": (
                    "user_request: user_request, channel: channel:chn_test, "
                    "sender: usr_test, inbound_event_kind: user_request"
                ),
            },
        ]
        self.assertEqual(latest_user_request(messages), "What are the lyrics to Idumea?")

    def test_skips_json_runtime_context_and_ack_after_real_request(self):
        messages = [
            {
                "role": "user",
                "content": "@sacredharpbench What are the lyrics to Idumea?",
            },
            {
                "role": "user",
                "content": (
                    '{"chat_id":"channel:chn_test","message_id":"msg_test",'
                    '"sender":{"id":"usr_test"},"inbound_event_kind":"user_request",'
                    '"was_mentioned":true}\nMessage received.'
                ),
            },
        ]
        self.assertEqual(latest_user_request(messages), "What are the lyrics to Idumea?")

    def test_skips_marked_openclaw_internal_context(self):
        messages = [
            {
                "role": "user",
                "content": "@sacredharpbench What are the lyrics to Idumea?",
            },
            {
                "role": "user",
                "content": (
                    "OpenClaw runtime context for the active user request in this turn. "
                    "Do not reply to or describe this context.\n"
                    "<<<BEGIN_OPENCLAW_INTERNAL_CONTEXT>>>\n"
                    "Conversation info: ⟦openclaw:ctx⟧\n```json\n{}\n```\n"
                    "<<<END_OPENCLAW_INTERNAL_CONTEXT>>>"
                ),
            },
        ]
        self.assertEqual(latest_user_request(messages), "What are the lyrics to Idumea?")

    def test_internal_context_after_tool_result_does_not_retrigger_rag(self):
        messages = [
            {"role": "user", "content": "What are the lyrics to Idumea?"},
            {
                "role": "tool",
                "name": "sacred_harp_search",
                "content": '{"exact": true, "answer": "And am I born to die?"}',
            },
            {
                "role": "user",
                "content": (
                    "OpenClaw runtime context for the active user request in this turn.\n"
                    "<<<BEGIN_OPENCLAW_INTERNAL_CONTEXT>>>\n{}\n"
                    "<<<END_OPENCLAW_INTERNAL_CONTEXT>>>"
                ),
            },
        ]
        tools = [
            {
                "type": "function",
                "function": {"name": "sacred_harp_search", "parameters": {}},
            }
        ]
        self.assertEqual(select_turn_tools(messages, tools), [])
        rendered = "\n".join(
            item["content"] for item in compact_generation_messages(messages)
        )
        self.assertIn("And am I born to die?", rendered)

    def test_short_named_query_is_not_mistaken_for_followup(self):
        messages = [{"role": "user", "content": "What are the lyrics to Idumea?"}]
        repaired = repair_tool_arguments("sacred_harp_search", {}, messages)
        self.assertEqual(repaired["query"], "What are the lyrics to Idumea?")

    def test_post_tool_answer_keeps_a_useful_completion_budget(self):
        self.assertEqual(completion_token_budget({"max_tokens": 1}, post_tool=True), 256)
        self.assertEqual(completion_token_budget({"max_tokens": 1}, post_tool=False), 1)

    def test_named_denson_lyrics_render_verbatim(self):
        payload = {
            "exact": False,
            "results": [
                {
                    "source": "/vault/sh 47b — Idumea.md",
                    "text": """Song: Idumea
Book family: sh
Song number: 47b

### sh1991 47b, sh2025 47b

And am I born to die?
To lay this body down!

### sh1991 47b, sh2025 47b

A land of deepest shade,
Unpierced by human thought;

### southernharmony 281

An unrelated extra witness stanza.""",
                }
            ],
        }
        answer = direct_grounded_answer(
            "Please give me the lyrics to Idumea.", json.dumps(payload)
        )
        self.assertEqual(
            answer,
            "Idumea — The Sacred Harp (SH 47b)\n\n"
            "And am I born to die?\nTo lay this body down!\n\n"
            "A land of deepest shade,\nUnpierced by human thought;\n\n"
            "Source: sh 47b — Idumea.md",
        )

    def test_edition_reference_does_not_collapse_denson_witnesses(self):
        self.assertEqual(
            edition_reference("sh", "47b", "/vault/sh1991 47b — Idumea.md"),
            "The Sacred Harp 1991 Denson edition (SH 47b)",
        )
        self.assertEqual(
            edition_reference("sh", "47b", "/vault/sh2025 47b — Idumea.md"),
            "The Sacred Harp 2025 Denson edition (SH 47b)",
        )
        identical = "### sh1991 47b, sh2025 47b\n\nAnd am I born to die?"
        self.assertEqual(
            edition_reference("sh", "47b", identical),
            "The Sacred Harp (SH 47b)",
        )
        differing = "### sh1991 47b\n\nVerse A\n\n### sh2025 47b\n\nVerse B"
        self.assertIn(
            "1991 Denson edition",
            edition_reference("sh", "47b", differing),
        )

    def test_does_not_complete_an_unsupported_quoted_fragment(self):
        text = (
            "Linked canonical text:\n"
            "### sh1991 404\n\n"
            "Youth, like the spring, will soon be gone,\n"
            "By fleeting time or conqu’ring death;\n"
            "Your morning sun may set at noon,\n"
            "And leave you ever in the dark."
        )
        completion = canonical_verse_completion(
            'What tunes have the text "Time and youth will steal..."?', text
        )
        self.assertIsNone(completion)

    def test_completes_a_supported_quoted_lyric_fragment(self):
        text = (
            "Linked canonical text:\n"
            "### sh1991 404\n\n"
            "Youth, like the spring, will soon be gone,\n"
            "By fleeting time or conqu’ring death;\n"
            "Your morning sun may set at noon,\n"
            "And leave you ever in the dark."
        )
        completion = canonical_verse_completion(
            'What tunes have the text "Youth, like the spring..."?', text
        )
        self.assertIn("By fleeting time or conqu’ring death;", completion or "")

    def test_compacts_rag_results_and_drops_extra_hits(self):
        payload = {
            "query": "Idumea",
            "exact": False,
            "results": [
                {"source": "idumea.md", "text": "And am I born to die?"},
                {"source": "text-hub.md", "text": "And am I born to die?"},
                {"source": "unrelated.md", "text": "New Jerusalem"},
            ],
        }
        evidence = compact_tool_evidence(json.dumps(payload))
        self.assertIn("idumea.md", evidence)
        self.assertIn("text-hub.md", evidence)
        self.assertNotIn("unrelated.md", evidence)
        self.assertLessEqual(len(evidence), MAX_COMPACT_EVIDENCE_CHARS)

    def test_post_tool_prompt_contains_only_question_and_evidence(self):
        messages = [
            {"role": "system", "content": "x" * 20_000},
            {
                "role": "user",
                "content": "@sacredharpbench What are the lyrics to Idumea?",
            },
            {"role": "assistant", "content": "", "tool_calls": []},
            {
                "role": "tool",
                "name": "sacred_harp_search",
                "content": json.dumps(
                    {
                        "exact": True,
                        "answer": "And am I born to die?",
                        "sources": ["idumea.md"],
                    }
                ),
            },
        ]
        compact = compact_generation_messages(messages)
        rendered = "\n".join(message["content"] for message in compact)
        self.assertIn("What are the lyrics to Idumea?", rendered)
        self.assertIn("And am I born to die?", rendered)
        self.assertNotIn("x" * 100, rendered)
        self.assertLess(len(rendered), 7_000)


if __name__ == "__main__":
    unittest.main()
