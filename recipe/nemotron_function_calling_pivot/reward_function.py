# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
"""Reward for Nemotron function-calling pivot data.

Supported tool-call format in model output:
<tool_call>{function_name}<arg_key>{k}</arg_key><arg_value>{v}</arg_value>...</tool_call>
"""

from __future__ import annotations

import json
import re
from typing import Any

TOOL_CALL_BLOCK_PATTERN = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)
ARG_PAIR_PATTERN = re.compile(r"<arg_key>(.*?)</arg_key>\s*<arg_value>(.*?)</arg_value>", re.DOTALL)
ARG_KEY_PATTERN = re.compile(r"<arg_key>", re.DOTALL)


def _normalize_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _normalize_value(value[k]) for k in sorted(value)}
    if isinstance(value, list):
        return [_normalize_value(v) for v in value]
    if isinstance(value, str):
        return value.strip()
    return value


def _safe_load_json(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    text = value.strip()
    if not text:
        return {}
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


def _parse_model_tool_calls(solution_str: str) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []
    for block in TOOL_CALL_BLOCK_PATTERN.findall(solution_str):
        block = block.strip()
        if not block:
            continue

        first_arg_key_match = ARG_KEY_PATTERN.search(block)
        if first_arg_key_match is None:
            function_name = block.strip()
            args = {}
        else:
            function_name = block[: first_arg_key_match.start()].strip()
            args = {
                key.strip(): _safe_load_json(value.strip())
                for key, value in ARG_PAIR_PATTERN.findall(block)
            }

        if function_name:
            calls.append({"name": function_name, "arguments": _normalize_value(args)})
    return calls


def _normalize_ground_truth(ground_truth: Any) -> tuple[str, list[dict[str, Any]]]:
    if isinstance(ground_truth, str):
        ground_truth = _safe_load_json(ground_truth)

    if not isinstance(ground_truth, dict):
        return "message", []

    gt_type = ground_truth.get("type", "message")
    if gt_type == "message":
        return "message", []

    calls = ground_truth.get("tool_calls")
    if calls is None:
        calls = [{"name": ground_truth.get("name"), "arguments": ground_truth.get("arguments", {})}]

    normalized_calls = []
    for item in calls:
        if not isinstance(item, dict):
            continue
        name = item.get("name")
        if not name:
            continue

        # preferred fixed schema from prepare_dataset.py
        if "arguments_json" in item:
            args = _safe_load_json(item.get("arguments_json", "{}"))
        else:
            args = _safe_load_json(item.get("arguments", {}))

        normalized_calls.append({"name": name, "arguments": _normalize_value(args)})
    return "function_call", normalized_calls


def nemotron_fc_reward_function(data_source, solution_str, ground_truth, extra_info=None):
    """Binary reward.

    Rule:
    - If ground-truth action is "message": reward=1 when model emits no <tool_call> blocks.
      Message content is intentionally ignored.
    - If ground-truth action is function-call: reward=1 only if all parsed model tool calls exactly
      match all expected tool calls (name, count, and normalized arguments).
    """
    predicted_calls = _parse_model_tool_calls(solution_str)
    gt_type, gt_calls = _normalize_ground_truth(ground_truth)

    if gt_type == "message":
        return 1.0 if len(predicted_calls) == 0 else 0.0

    if len(predicted_calls) != len(gt_calls):
        return 0.0

    for predicted_call, gt_call in zip(predicted_calls, gt_calls, strict=True):
        if predicted_call["name"] != gt_call["name"]:
            return 0.0
        if predicted_call["arguments"] != gt_call["arguments"]:
            return 0.0

    return 1.0
