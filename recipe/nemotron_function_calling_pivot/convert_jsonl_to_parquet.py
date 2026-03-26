"""Convert Nemotron JSONL to parquet.

Default mode writes verl-ready schema directly (recommended for training efficiency).
Optional raw mode writes JSON-stringified nested columns for archival/debug.
"""

from __future__ import annotations

import argparse
import json
import os

import datasets

try:
    from .dataset_io import load_jsonl_rows, resolve_default_jsonl
    from .prepare_dataset import map_to_verl_row
except ImportError:
    from dataset_io import load_jsonl_rows, resolve_default_jsonl
    from prepare_dataset import map_to_verl_row


def _load_rows(dataset_name: str, input_jsonl: str | None):
    source = input_jsonl or resolve_default_jsonl(dataset_name)
    return load_jsonl_rows(source), source


def _stringify_raw_row(row):
    return {
        "trajectory_id": row.get("trajectory_id"),
        "info": json.dumps(row.get("info", {}), ensure_ascii=False),
        "responses_create_params": json.dumps(row.get("responses_create_params", {}), ensure_ascii=False),
        "expected_action": json.dumps(row.get("expected_action", {}), ensure_ascii=False),
        "agent_ref": json.dumps(row.get("agent_ref", {}), ensure_ascii=False),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", default="nvidia/Nemotron-RL-Agentic-Function-Calling-Pivot-v1")
    parser.add_argument("--input_jsonl", default=None)
    parser.add_argument("--output_parquet", default=os.path.expanduser("~/data/nemotron_fc_pivot/train.parquet"))
    parser.add_argument("--format", choices=["verl", "raw"], default="verl")
    parser.add_argument("--verify_reload", action="store_true")
    args = parser.parse_args()

    rows, source = _load_rows(args.dataset_name, args.input_jsonl)

    if args.format == "verl":
        ds = datasets.Dataset.from_list([map_to_verl_row(r, split="train") for r in rows])
    else:
        ds = datasets.Dataset.from_list([_stringify_raw_row(r) for r in rows])

    os.makedirs(os.path.dirname(args.output_parquet), exist_ok=True)
    ds.to_parquet(args.output_parquet)
    print(f"Saved {len(ds)} rows from {source} as {args.format} -> {args.output_parquet}")

    if args.verify_reload:
        reloaded = datasets.load_dataset("parquet", data_files={"train": args.output_parquet})["train"]
        if len(reloaded) != len(ds):
            raise RuntimeError(f"Row mismatch: source={len(ds)}, parquet={len(reloaded)}")
        print(f"Reload verification passed: {len(reloaded)} rows")


if __name__ == "__main__":
    main()
