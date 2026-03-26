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
import json
import pathlib
import importlib.util
import asyncio

import pytest

_TOOL_PARSER_PATH = pathlib.Path(__file__).resolve().parents[3] / "verl/experimental/agent_loop/tool_parser.py"
_SPEC = importlib.util.spec_from_file_location("tool_parser_module", _TOOL_PARSER_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
GlmStyleToolParser = _MODULE.GlmStyleToolParser


class DummyTokenizer:
    def __init__(self, text: str):
        self._text = text

    def decode(self, _):
        return self._text


def test_glm_style_tool_parser_extracts_calls_and_overrides_duplicate_keys():
    text = """
prefix
<tool_call>
search
<arg_key>query</arg_key><arg_value>first</arg_value>
<arg_key>query</arg_key><arg_value>latest</arg_value>
<arg_key>lang</arg_key><arg_value>ko</arg_value>
</tool_call>
middle
<tool_call>
sum
<arg_key>a</arg_key><arg_value>1</arg_value>
<arg_key>b</arg_key><arg_value>2</arg_value>
</tool_call>
suffix
"""
    parser = GlmStyleToolParser(DummyTokenizer(text))
    content, calls = asyncio.run(parser.extract_tool_calls([1, 2, 3]))

    assert "<tool_call>" not in content
    assert len(calls) == 2
    assert calls[0].name == "search"
    assert json.loads(calls[0].arguments) == {"query": "latest", "lang": "ko"}
    assert calls[1].name == "sum"
    assert json.loads(calls[1].arguments) == {"a": "1", "b": "2"}


def test_glm_style_tool_parser_skips_malformed_block_only(caplog):
    text = """
<tool_call>
valid_tool
<arg_key>x</arg_key><arg_value>10</arg_value>
</tool_call>
<tool_call>
<arg_key>missing_name</arg_key><arg_value>oops</arg_value>
</tool_call>
"""
    parser = GlmStyleToolParser(DummyTokenizer(text))
    _, calls = asyncio.run(parser.extract_tool_calls([1]))

    assert len(calls) == 1
    assert calls[0].name == "valid_tool"
    assert json.loads(calls[0].arguments) == {"x": "10"}
    assert "Skipping malformed GLM-like tool call block" in caplog.text
