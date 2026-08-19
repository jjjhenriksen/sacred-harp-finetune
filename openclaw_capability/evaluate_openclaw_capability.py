#!/usr/bin/env python3
"""Evaluate native tool routing and grounded follow-up behavior for one adapter."""

from __future__ import annotations

import argparse
import json
import re
import time
from collections import Counter
from pathlib import Path
from typing import Any

from mlx_lm import generate, load
from mlx_lm.sample_utils import make_sampler

from sacred_harp_openclaw_server import (
    compact_generation_messages,
    deterministic_tool_call,
    parse_tool_call,
    repair_tool_arguments,
    select_turn_tools,
    tool_names,
)


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DEFAULT_MODEL = ROOT / "models" / "llama-1b"
DEFAULT_ADAPTER = ROOT / "adapters" / "sacred_harp_1b_lora_openclaw"
DEFAULT_DATA = HERE / "data" / "test.jsonl"
DEFAULT_REPORT = HERE / "reports" / "capability_eval.json"
ROUTE_KINDS = {
    "rag_route",
    "discussion_route",
    "message_route",
    "stateful_rag_route",
    "interpretation_route",
}


def normalized_tokens(value: Any) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", str(value).lower()))


def argument_score(expected: dict[str, Any], actual: dict[str, Any]) -> float:
    if not expected:
        return 1.0
    scores = []
    for key, expected_value in expected.items():
        if key == "top_k":
            continue
        actual_value = actual.get(key)
        if isinstance(expected_value, str):
            expected_tokens = normalized_tokens(expected_value)
            actual_tokens = normalized_tokens(actual_value)
            scores.append(
                len(expected_tokens & actual_tokens) / len(expected_tokens)
                if expected_tokens
                else float(expected_value == actual_value)
            )
        else:
            scores.append(float(expected_value == actual_value))
    return sum(scores) / len(scores) if scores else 1.0


def expected_call(row: dict[str, Any]) -> tuple[str, dict[str, Any]] | None:
    last = row["messages"][-1]
    calls = last.get("tool_calls") or []
    if not calls:
        return None
    function = calls[0]["function"]
    return function["name"], function["arguments"]


def content_overlap(expected: str, actual: str) -> float:
    expected_tokens = normalized_tokens(expected)
    actual_tokens = normalized_tokens(actual)
    common = expected_tokens & actual_tokens
    return len(common) / len(expected_tokens) if expected_tokens else 0.0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--adapter", type=Path, default=DEFAULT_ADAPTER)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--harness-gating",
        action="store_true",
        help="Evaluate the production ClickClack per-turn tool gate and argument repair.",
    )
    args = parser.parse_args()

    rows = [json.loads(line) for line in args.data.read_text(encoding="utf-8").splitlines()]
    rows = [
        row
        for row in rows
        if row.get("kind") in ROUTE_KINDS
        | {
            "rag_grounded_answer",
            "interpretation_grounded_answer",
            "grounding_failure",
            "plain_conversation",
        }
    ]
    if args.limit:
        rows = rows[: args.limit]

    model, tokenizer = load(str(args.model), adapter_path=str(args.adapter))
    sampler = make_sampler(temp=0.0)
    counts = Counter()
    details = []
    started = time.time()
    for index, row in enumerate(rows, start=1):
        final = row["messages"][-1]
        input_messages = row["messages"][:-1]
        tools = row.get("tools")
        if args.harness_gating:
            tools = select_turn_tools(input_messages, tools)
        routed = deterministic_tool_call(tools, input_messages) if args.harness_gating else None
        if routed:
            parsed = routed
            output = json.dumps(
                {"name": routed[0], "parameters": routed[1]}, ensure_ascii=False
            )
        else:
            if input_messages and input_messages[-1].get("role") in {"tool", "function"}:
                prompt_messages = compact_generation_messages(input_messages)
                prompt = tokenizer.apply_chat_template(
                    prompt_messages,
                    tools=None,
                    add_generation_prompt=True,
                )
            else:
                prompt = tokenizer.apply_chat_template(
                    input_messages,
                    tools=tools or None,
                    add_generation_prompt=True,
                )
            output = generate(model, tokenizer, prompt, max_tokens=220, sampler=sampler).strip()
            parsed = parse_tool_call(output, tool_names(tools))
            if parsed and args.harness_gating:
                parsed = (parsed[0], repair_tool_arguments(parsed[0], parsed[1], input_messages))
        expected = expected_call(row)
        passed = False
        score: dict[str, Any]
        if expected:
            expected_name, expected_args = expected
            actual_name, actual_args = parsed if parsed else (None, {})
            args_score = argument_score(expected_args, actual_args)
            passed = actual_name == expected_name and args_score >= 0.6
            score = {
                "expected_tool": expected_name,
                "actual_tool": actual_name,
                "argument_score": round(args_score, 4),
            }
            counts["route_total"] += 1
            counts["route_pass"] += passed
            counts[f"{expected_name}_total"] += 1
            counts[f"{expected_name}_pass"] += passed
        else:
            expected_text = str(final.get("content", ""))
            overlap = content_overlap(expected_text, output)
            passed = parsed is None and (overlap >= 0.18 or bool(output.strip()))
            score = {"unexpected_tool": parsed[0] if parsed else None, "content_overlap": round(overlap, 4)}
            counts["answer_total"] += 1
            counts["answer_pass"] += passed
        counts["total"] += 1
        counts["pass"] += passed
        details.append(
            {
                "index": index,
                "kind": row.get("kind"),
                "source": row.get("source"),
                "passed": passed,
                "score": score,
                "output": output,
            }
        )
        if index % 20 == 0:
            print(f"evaluated {index}/{len(rows)}", flush=True)

    def rate(hit: str, total: str) -> float:
        return counts[hit] / counts[total] if counts[total] else 0.0

    report = {
        "adapter": str(args.adapter),
        "harness_gating": args.harness_gating,
        "examples": counts["total"],
        "passed": counts["pass"],
        "pass_rate": rate("pass", "total"),
        "route": {
            "examples": counts["route_total"],
            "passed": counts["route_pass"],
            "pass_rate": rate("route_pass", "route_total"),
            "by_tool": {
                name: {
                    "examples": counts[f"{name}_total"],
                    "passed": counts[f"{name}_pass"],
                    "pass_rate": rate(f"{name}_pass", f"{name}_total"),
                }
                for name in ("sacred_harp_search", "discussion", "message")
            },
        },
        "answer": {
            "examples": counts["answer_total"],
            "passed": counts["answer_pass"],
            "pass_rate": rate("answer_pass", "answer_total"),
        },
        "seconds": round(time.time() - started, 3),
        "details": details,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "details"}, indent=2))
    print(f"Saved {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
