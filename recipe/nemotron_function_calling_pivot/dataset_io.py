"""Robust data loading utilities for Nemotron JSONL."""

from __future__ import annotations

import json
import random
import urllib.request
from pathlib import Path
from typing import Any


def resolve_default_jsonl(dataset_name: str) -> str:
    try:
        from huggingface_hub import hf_hub_download

        return hf_hub_download(repo_id=dataset_name, filename="train.jsonl", repo_type="dataset")
    except Exception:
        return f"https://huggingface.co/datasets/{dataset_name}/resolve/main/train.jsonl"


def _iter_jsonl_lines(source: str):
    if source.startswith("http://") or source.startswith("https://"):
        with urllib.request.urlopen(source) as f:
            for raw in f:
                line = raw.decode("utf-8").strip()
                if line:
                    yield line
        return

    with Path(source).open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield line


def load_jsonl_rows(source: str) -> list[dict[str, Any]]:
    return [json.loads(line) for line in _iter_jsonl_lines(source)]


def train_val_split(rows: list[dict[str, Any]], val_ratio: float, seed: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    idxs = list(range(len(rows)))
    random.Random(seed).shuffle(idxs)
    val_size = int(len(rows) * val_ratio)
    val_ids = set(idxs[:val_size])
    train_rows, val_rows = [], []
    for i, row in enumerate(rows):
        (val_rows if i in val_ids else train_rows).append(row)
    return train_rows, val_rows
