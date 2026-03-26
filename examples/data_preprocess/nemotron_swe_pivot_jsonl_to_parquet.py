# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Convert Nemotron SWE Pivot JSONL to VERL parquet schema.

Example:
    python examples/data_preprocess/nemotron_swe_pivot_jsonl_to_parquet.py \
        --input_jsonl /path/to/train.jsonl \
        --output_parquet /path/to/train.parquet \
        --split train
"""

import argparse
import json
from typing import Any

import pandas as pd


DATA_SOURCE = "nvidia/Nemotron-RL-Agentic-SWE-Pivot-v1"


def _normalize_expected_action(expected_action: Any) -> dict[str, Any]:
    """Normalize expected_action to {"expected_tool_calls": [...]} format whenever possible."""
    if isinstance(expected_action, dict):
        if "expected_tool_calls" in expected_action and isinstance(expected_action["expected_tool_calls"], list):
            return {"expected_tool_calls": expected_action["expected_tool_calls"]}

        if "tool_calls" in expected_action and isinstance(expected_action["tool_calls"], list):
            return {"expected_tool_calls": expected_action["tool_calls"]}

        if "name" in expected_action:
            return {"expected_tool_calls": [expected_action]}

        return {"expected_tool_calls": [expected_action]}

    if isinstance(expected_action, list):
        return {"expected_tool_calls": expected_action}

    # Keep primitive values wrapped to preserve information while matching schema.
    return {"expected_tool_calls": [{"value": expected_action}]}


def _to_text_content(content: Any) -> Any:
    """Collapse content parts when they are OpenAI Responses-style arrays."""
    if isinstance(content, list):
        texts: list[str] = []
        for part in content:
            if isinstance(part, dict):
                text = part.get("text")
                if text is not None:
                    texts.append(str(text))
            elif isinstance(part, str):
                texts.append(part)
        if texts:
            return "\n".join(texts)
    return content


def _infer_role_from_turn(turn: dict[str, Any]) -> str:
    """Infer role from various turn schemas used in agentic traces."""
    if isinstance(turn.get("role"), str) and turn["role"]:
        return turn["role"]

    turn_type = turn.get("type")
    if turn_type in {"message", "input_message"}:
        return "user"
    if turn_type in {"output_message", "reasoning"}:
        return "assistant"
    if turn_type in {"function_call", "tool_call"}:
        return "assistant"
    if turn_type in {"function_call_output", "tool_result", "tool_response"}:
        return "tool"

    if "tool_call_id" in turn:
        return "tool"
    if "name" in turn and ("arguments" in turn or "input" in turn):
        return "assistant"

    return "user"


def _extract_turn_content(turn: dict[str, Any]) -> Any:
    """Extract text-like content from a single turn regardless of its source schema."""
    for key in ("content", "text", "output_text"):
        if key in turn:
            return _to_text_content(turn.get(key))

    if "arguments" in turn:
        return json.dumps(turn["arguments"], ensure_ascii=False)
    if "output" in turn:
        return _to_text_content(turn["output"])
    if "input" in turn and not isinstance(turn["input"], (list, dict)):
        return str(turn["input"])

    return ""


def _normalize_prompt_input(raw_prompt: Any, idx: int) -> list[dict[str, Any]]:
    """Normalize prompt into a chat message list for both single/multi-turn input."""
    if isinstance(raw_prompt, str):
        return [{"role": "user", "content": raw_prompt}]

    if not isinstance(raw_prompt, list):
        raise ValueError(f"Unsupported prompt type: {type(raw_prompt)} at index={idx}")

    normalized_prompt: list[dict[str, Any]] = []
    for turn_id, turn in enumerate(raw_prompt):
        if isinstance(turn, str):
            normalized_prompt.append({"role": "user", "content": turn})
            continue

        if not isinstance(turn, dict):
            raise ValueError(f"Unsupported prompt turn type: {type(turn)} at index={idx}, turn={turn_id}")

        role = _infer_role_from_turn(turn)
        content = _extract_turn_content(turn)
        normalized_prompt.append({"role": role, "content": content})

    return normalized_prompt


def _build_row(record: dict[str, Any], idx: int, split: str | None) -> dict[str, Any]:
    responses_create_params = record.get("responses_create_params", {})
    raw_prompt = responses_create_params.get("input", record.get("input", []))
    prompt = _normalize_prompt_input(raw_prompt=raw_prompt, idx=idx)

    extra_info: dict[str, Any] = {"index": idx}
    if split:
        extra_info["split"] = split

    return {
        "data_source": DATA_SOURCE,
        "agent_name": "tool_agent",
        "prompt": prompt,
        "reward_model": {
            "style": "rule",
            "ground_truth": _normalize_expected_action(record.get("expected_action")),
        },
        "extra_info": extra_info,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert Nemotron SWE Pivot JSONL into parquet.")
    parser.add_argument("--input_jsonl", required=True, help="Input JSONL path (e.g., train.jsonl).")
    parser.add_argument("--output_parquet", required=True, help="Output parquet path.")
    parser.add_argument("--split", default=None, help="Optional split name (e.g., train/test).")
    args = parser.parse_args()

    rows: list[dict[str, Any]] = []
    with open(args.input_jsonl, encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            rows.append(_build_row(record=record, idx=idx, split=args.split))

    df = pd.DataFrame(rows)
    df.to_parquet(args.output_parquet, index=False)

    print(f"Converted {len(df)} rows -> {args.output_parquet}")


if __name__ == "__main__":
    main()
