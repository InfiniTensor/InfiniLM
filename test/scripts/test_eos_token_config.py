"""EOS priority and nested text configs without loading checkpoint weights."""

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest

SOURCE = Path(__file__).resolve().parents[2] / "python/infinilm/infer_engine.py"
tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
engine = next(
    n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "InferEngine"
)
method = next(
    n
    for n in engine.body
    if isinstance(n, ast.FunctionDef) and n.name == "eos_token_id"
)
method.decorator_list = []
namespace = {}
exec(
    compile(ast.Module(body=[method], type_ignores=[]), str(SOURCE), "exec"), namespace
)
resolve = namespace["eos_token_id"]


@pytest.mark.parametrize(
    "generation,config,expected",
    [
        ({"eos_token_id": [2, 128131]}, {"eos_token_id": 2}, [2, 128131]),
        ({}, {"text_config": {"eos_token_id": 248044}}, [248044]),
        ({}, {"eos_token_id": 2, "text_config": {"eos_token_id": 3}}, [2]),
        ({"eos_token_id": None}, {"text_config": {"eos_token_id": [7, 8]}}, [7, 8]),
        ({"eos_token_id": 0}, {"eos_token_id": 2}, [0]),
        ({"eos_token_id": []}, {"eos_token_id": 2}, []),
        ({}, {}, []),
    ],
)
def test_eos_priority(generation, config, expected):
    instance = SimpleNamespace(hf_generation_config=generation, hf_config=config)
    assert resolve(instance) == expected
