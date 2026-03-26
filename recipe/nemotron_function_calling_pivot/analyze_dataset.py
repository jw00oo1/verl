"""Analysis for Nemotron function-calling pivot dataset."""

from __future__ import annotations

import argparse
import collections
import statistics
import tempfile

import datasets

try:
    from .dataset_io import load_jsonl_rows, resolve_default_jsonl
except ImportError:
    from dataset_io import load_jsonl_rows, resolve_default_jsonl


def _load_rows(dataset_name: str, input_jsonl: str | None):
    source = input_jsonl or resolve_default_jsonl(dataset_name)
    rows = load_jsonl_rows(source)
    return rows, source


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", default="nvidia/Nemotron-RL-Agentic-Function-Calling-Pivot-v1")
    parser.add_argument("--input_jsonl", default=None)
    parser.add_argument("--topk", type=int, default=20)
    parser.add_argument("--check_to_parquet", action="store_true")
    args = parser.parse_args()

    rows, source = _load_rows(args.dataset_name, args.input_jsonl)

    action_type_counter = collections.Counter()
    tool_name_counter = collections.Counter()
    input_event_type_counter = collections.Counter()
    role_counter = collections.Counter()
    prompt_turns = []
    tool_count_per_sample = []

    non_string_role_content = 0
    malformed_role_event = 0

    for row in rows:
        action = row.get("expected_action", {})
        action_type = action.get("type", "unknown")
        action_type_counter[action_type] += 1

        if action_type == "function_call":
            tool_name_counter[action.get("name", "")] += 1
            tool_count_per_sample.append(1)
        elif action_type == "function_calls":
            calls = action.get("tool_calls", [])
            tool_count_per_sample.append(len(calls))
            for call in calls:
                tool_name_counter[call.get("name", "")] += 1
        else:
            tool_count_per_sample.append(0)

        input_events = row.get("responses_create_params", {}).get("input", [])
        prompt_turns.append(len(input_events))

        for event in input_events:
            if not isinstance(event, dict):
                input_event_type_counter["non_dict"] += 1
                continue

            if "role" in event:
                input_event_type_counter["role_message"] += 1
                role = event.get("role", "")
                role_counter[role] += 1
                if "content" not in event:
                    malformed_role_event += 1
                elif not isinstance(event["content"], str):
                    non_string_role_content += 1
            else:
                input_event_type_counter[event.get("type", "unknown")] += 1

    n = len(rows)
    print(f"source: {source}")
    print(f"num_samples: {n}")

    print("expected_action.type distribution:")
    for name, cnt in action_type_counter.most_common():
        print(f"  - {name}: {cnt} ({cnt / n:.2%})")

    if tool_name_counter:
        print(f"top {args.topk} function names:")
        for name, cnt in tool_name_counter.most_common(args.topk):
            print(f"  - {name}: {cnt}")

    print("input event type distribution:")
    for name, cnt in input_event_type_counter.most_common():
        print(f"  - {name}: {cnt}")

    print("role distribution (for role_message events):")
    for name, cnt in role_counter.most_common():
        print(f"  - {name}: {cnt}")

    print("content quality checks:")
    print(f"  - role events with non-string content: {non_string_role_content}")
    print(f"  - malformed role events (no content key): {malformed_role_event}")

    print("prompt turn stats (raw responses_create_params.input length):")
    print(f"  - min: {min(prompt_turns)}")
    print(f"  - max: {max(prompt_turns)}")
    print(f"  - mean: {statistics.mean(prompt_turns):.2f}")

    print("tool calls per sample stats (from expected_action):")
    print(f"  - min: {min(tool_count_per_sample)}")
    print(f"  - max: {max(tool_count_per_sample)}")
    print(f"  - mean: {statistics.mean(tool_count_per_sample):.2f}")

    if args.check_to_parquet:
        with tempfile.TemporaryDirectory(prefix="nemotron_parquet_check_") as tmpdir:
            path = f"{tmpdir}/train.parquet"
            # stringify nested objects for robust parquet round-trip
            ds = datasets.Dataset.from_list(
                [
                    {
                        "trajectory_id": r.get("trajectory_id"),
                        "info": str(r.get("info", {})),
                        "responses_create_params": str(r.get("responses_create_params", {})),
                        "expected_action": str(r.get("expected_action", {})),
                        "agent_ref": str(r.get("agent_ref", {})),
                    }
                    for r in rows
                ]
            )
            ds.to_parquet(path)
            loaded = datasets.load_dataset("parquet", data_files={"train": path})["train"]
            print("to_parquet check:")
            print(f"  - parquet path: {path}")
            print(f"  - source rows: {len(rows)}")
            print(f"  - reloaded rows: {len(loaded)}")
            print(f"  - row_count_match: {len(rows) == len(loaded)}")


if __name__ == "__main__":
    main()
