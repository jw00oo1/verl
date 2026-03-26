"""Tool-aware RLHFDataset for Nemotron function-calling data.

Injects row-level `tools` into tokenizer.apply_chat_template(..., tools=...).
"""

from __future__ import annotations

import copy
import json

import torch
from omegaconf import DictConfig, ListConfig, OmegaConf

import verl.utils.torch_functional as verl_F
from verl.utils.dataset.rl_dataset import RLHFDataset
from verl.utils.model import compute_position_id_with_mask


class NemotronToolAwareRLHFDataset(RLHFDataset):
    def _parse_tools(self, row_dict: dict) -> list[dict]:
        raw_tools = row_dict.get("tools", "[]")
        if isinstance(raw_tools, str):
            try:
                tools = json.loads(raw_tools)
            except json.JSONDecodeError:
                tools = []
        elif isinstance(raw_tools, list):
            tools = raw_tools
        else:
            tools = []
        return tools

    def _chat_template_with_tools(self, messages: list, tools: list[dict], tokenize: bool):
        kwargs = copy.deepcopy(self.apply_chat_template_kwargs) or {}
        if isinstance(kwargs, DictConfig | ListConfig):
            kwargs = OmegaConf.to_container(kwargs, resolve=True)
        if tools and "tools" not in kwargs:
            kwargs["tools"] = tools
        return self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=tokenize, **kwargs)

    def maybe_filter_out_long_prompts(self, dataframe=None):
        if not self.filter_overlong_prompts:
            return dataframe

        def doc2len(doc) -> int:
            try:
                messages = self._build_messages(doc)
                tools = self._parse_tools(doc)
                prompt_ids = self._chat_template_with_tools(messages, tools, tokenize=True)
                return len(prompt_ids)
            except Exception:
                return self.max_prompt_length + 1

        dataframe = dataframe.filter(
            lambda doc: doc2len(doc) <= self.max_prompt_length,
            num_proc=self.num_workers,
            desc=f"Filtering prompts longer than {self.max_prompt_length} tokens",
        )
        print(f"filter dataset len: {len(dataframe)}")
        return dataframe

    def __getitem__(self, item):
        row_dict: dict = self.dataframe[item]
        messages = self._build_messages(row_dict)
        tools = self._parse_tools(row_dict)

        raw_prompt = self._chat_template_with_tools(messages, tools, tokenize=False)
        model_inputs = self.tokenizer(raw_prompt, return_tensors="pt", add_special_tokens=False)
        input_ids = model_inputs.pop("input_ids")
        attention_mask = model_inputs.pop("attention_mask")

        input_ids, attention_mask = verl_F.postprocess_data(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=self.max_prompt_length,
            pad_token_id=self.tokenizer.pad_token_id,
            left_pad=True,
            truncation=self.truncation,
        )

        position_ids = compute_position_id_with_mask(attention_mask)
        row_dict["input_ids"] = input_ids[0]
        row_dict["attention_mask"] = attention_mask[0]
        row_dict["position_ids"] = position_ids[0]

        raw_prompt_ids = self.tokenizer.encode(raw_prompt, add_special_tokens=False)
        if len(raw_prompt_ids) > self.max_prompt_length:
            if self.truncation == "left":
                raw_prompt_ids = raw_prompt_ids[-self.max_prompt_length :]
            elif self.truncation == "right":
                raw_prompt_ids = raw_prompt_ids[: self.max_prompt_length]
            elif self.truncation == "middle":
                left_half = self.max_prompt_length // 2
                right_half = self.max_prompt_length - left_half
                raw_prompt_ids = raw_prompt_ids[:left_half] + raw_prompt_ids[-right_half:]
            elif self.truncation == "error":
                raise RuntimeError(f"Prompt length {len(raw_prompt_ids)} is longer than {self.max_prompt_length}.")

        row_dict["raw_prompt_ids"] = raw_prompt_ids
        if self.return_raw_chat:
            row_dict["raw_prompt"] = messages

        if "extra_info" not in row_dict or row_dict["extra_info"] is None:
            row_dict["extra_info"] = {}

        row_dict["index"] = row_dict.get("extra_info", {}).get("index", 0)
        row_dict["tools_kwargs"] = row_dict.get("extra_info", {}).get("tools_kwargs", {})
        row_dict["interaction_kwargs"] = row_dict.get("extra_info", {}).get("interaction_kwargs", {})
        return row_dict
