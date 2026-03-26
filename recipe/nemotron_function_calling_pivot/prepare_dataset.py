# Copyright 2024 Bytedance Ltd. and/or its affiliates

from __future__ import annotations

import argparse
import json
import os
from typing import Any

import datasets

try:
    from .dataset_io import load_jsonl_rows, resolve_default_jsonl, train_val_split
except ImportError:
    from dataset_io import load_jsonl_rows, resolve_default_jsonl, train_val_split


def _safe_json_load(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    text = value.strip()
    if not text:
        return {}
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


def _normalize_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        texts = []
        for item in content:
            if isinstance(item, dict):
                if isinstance(item.get("text"), str):
                    texts.append(item["text"])
                elif isinstance(item.get("output"), str):
                    texts.append(item["output"])
            elif isinstance(item, str):
                texts.append(item)
        return "\n".join(texts) if texts else json.dumps(content, ensure_ascii=False)
    if content is None:
        return ""
    return str(content)


def _tool_call_to_text(name: str, arguments: Any) -> str:
    args = _safe_json_load(arguments)
    blocks = [f"<tool_call>{name}"]
    if isinstance(args, dict):
        for key, value in args.items():
            value_text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
            blocks.append(f"<arg_key>{key}</arg_key><arg_value>{value_text}</arg_value>")
    elif args not in ({}, None, ""):
        value_text = args if isinstance(args, str) else json.dumps(args, ensure_ascii=False)
        blocks.append(f"<arg_key>__raw__</arg_key><arg_value>{value_text}</arg_value>")
    blocks.append("</tool_call>")
    return "".join(blocks)


def normalize_expected_action(expected_action: dict[str, Any]) -> dict[str, Any]:
    """Use a fixed ground-truth schema for stable parquet typing.

    {
      "type": "message|function_call",
      "content": str,
      "tool_calls": [{"name": str, "arguments_json": str}]
    }
    """
    action_type = expected_action.get("type", "message")
    if action_type == "function_call":
        return {
            "type": "function_call",
            "content": "",
            "tool_calls": [
                {
                    "name": str(expected_action.get("name", "")),
                    "arguments_json": json.dumps(_safe_json_load(expected_action.get("arguments", "{}")), ensure_ascii=False),
                }
            ],
        }

    if action_type == "function_calls":
        calls = []
        for call in expected_action.get("tool_calls", []):
            calls.append(
                {
                    "name": str(call.get("name", "")),
                    "arguments_json": json.dumps(_safe_json_load(call.get("arguments", "{}")), ensure_ascii=False),
                }
            )
        return {"type": "function_call", "content": "", "tool_calls": calls}

    return {"type": "message", "content": _normalize_content(expected_action.get("content", "")), "tool_calls": []}


def convert_input_events_to_prompt(events: list[Any]) -> list[dict[str, str]]:
    prompt = []
    for event in events:
        if not isinstance(event, dict):
            continue
        if "role" in event:
            role = event.get("role")
            if role in {"system", "user", "assistant", "tool"}:
                prompt.append({"role": role, "content": _normalize_content(event.get("content", ""))})
            continue

        event_type = event.get("type")
        if event_type == "function_call":
            name = event.get("name", "")
            if name:
                prompt.append({"role": "assistant", "content": _tool_call_to_text(name, event.get("arguments", "{}"))})
        elif event_type == "function_call_output":
            prompt.append({"role": "tool", "content": _normalize_content(event.get("output", ""))})
        elif event_type == "reasoning":
            continue
    return prompt


def map_to_verl_row(row: dict[str, Any], split: str) -> dict[str, Any]:
    response_params = row["responses_create_params"]
    prompt = convert_input_events_to_prompt(response_params.get("input", []))
    gt = normalize_expected_action(row["expected_action"])

    return {
        "data_source": "nemotron_fc_pivot_v1",
        "prompt": prompt,
        "ability": "tool_calling",
        "agent_name": "single_turn_agent",
        # tools are auxiliary metadata; stringify for robust compatibility.
        "tools": json.dumps(response_params.get("tools", []), ensure_ascii=False),
        "reward_model": {"style": "rule", "ground_truth": gt},
        "extra_info": {
            "split": split,
            "trajectory_id": int(row.get("trajectory_id", -1)),
            "parallel_tool_calls": bool(response_params.get("parallel_tool_calls", False)),
        },
    }


def build_verl_dataset(rows: list[dict[str, Any]], split: str) -> datasets.Dataset:
    return datasets.Dataset.from_list([map_to_verl_row(row, split) for row in rows])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_jsonl", default=None)
    parser.add_argument("--dataset_name", default="nvidia/Nemotron-RL-Agentic-Function-Calling-Pivot-v1")
    parser.add_argument("--output_dir", default=os.path.expanduser("~/data/nemotron_fc_pivot"))
    parser.add_argument("--val_ratio", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    source = args.input_jsonl or resolve_default_jsonl(args.dataset_name)
    rows = load_jsonl_rows(source)
    train_rows, val_rows = train_val_split(rows, val_ratio=args.val_ratio, seed=args.seed)

    train_ds = build_verl_dataset(train_rows, split="train")
    val_ds = build_verl_dataset(val_rows, split="val")

    os.makedirs(args.output_dir, exist_ok=True)
    train_path = os.path.join(args.output_dir, "train.parquet")
    val_path = os.path.join(args.output_dir, "val.parquet")
    train_ds.to_parquet(train_path)
    val_ds.to_parquet(val_path)

    print(f"Loaded source: {source}")
    print(f"Saved train={len(train_ds)} -> {train_path}")
    print(f"Saved val={len(val_ds)} -> {val_path}")


if __name__ == "__main__":
    main()
