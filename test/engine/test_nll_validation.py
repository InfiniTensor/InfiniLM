import importlib.util
import sys
import types
from pathlib import Path

import pytest


class FakeTensor:
    def __init__(self, shape, dtype):
        self.shape = tuple(shape)
        self.ndim = len(self.shape)
        self.dtype = dtype
        self._underlying = object()


@pytest.fixture
def validator(monkeypatch):
    """Load the pure validator without importing the hardware runtime."""
    package_root = Path(__file__).resolve().parents[2] / "python" / "infinilm"
    int64_dtype = object()

    infinilm_package = types.ModuleType("infinilm")
    infinilm_package.__path__ = [str(package_root)]
    monkeypatch.setitem(sys.modules, "infinilm", infinilm_package)

    fake_infinicore = types.ModuleType("infinicore")
    fake_infinicore.int64 = int64_dtype
    fake_infinicore.Tensor = type("Tensor", (), {})
    monkeypatch.setitem(sys.modules, "infinicore", fake_infinicore)

    cache_module = types.ModuleType("infinilm.cache")
    cache_module.PagedKVCacheConfig = type("PagedKVCacheConfig", (), {})
    monkeypatch.setitem(sys.modules, "infinilm.cache", cache_module)

    distributed_module = types.ModuleType("infinilm.distributed")
    distributed_module.DistConfig = type("DistConfig", (), {})
    monkeypatch.setitem(sys.modules, "infinilm.distributed", distributed_module)

    native_engine = type("InferEngine", (), {})
    lib_module = types.ModuleType("infinilm.lib")
    lib_module._infinilm = types.SimpleNamespace(InferEngine=native_engine)
    monkeypatch.setitem(sys.modules, "infinilm.lib", lib_module)

    exception_module = types.ModuleType("infinilm.exception_utils")
    exception_module.handle_oom_and_exit = lambda error: None
    monkeypatch.setitem(sys.modules, "infinilm.exception_utils", exception_module)

    modeling_module = types.ModuleType("infinilm.modeling_utils")
    modeling_module.parse_dtype = lambda dtype: dtype
    monkeypatch.setitem(sys.modules, "infinilm.modeling_utils", modeling_module)

    module_name = "infinilm.infer_engine"
    module_path = package_root / "infer_engine.py"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, module_name, module)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    return module._validate_nll_score_inputs, int64_dtype


def make_tensor(shape, int64_dtype, dtype=None):
    return FakeTensor(shape, int64_dtype if dtype is None else dtype)


def test_validate_nll_score_inputs_accepts_valid_window(validator):
    validate, int64_dtype = validator
    input_ids = make_tensor((1, 8), int64_dtype)
    labels = make_tensor((1, 8), int64_dtype)

    assert validate(input_ids, labels, 3) == (8, 3)


@pytest.mark.parametrize("name", ["input_ids", "labels"])
def test_validate_nll_score_inputs_requires_tensor_protocol(validator, name):
    validate, int64_dtype = validator
    tensors = {
        "input_ids": make_tensor((1, 8), int64_dtype),
        "labels": make_tensor((1, 8), int64_dtype),
    }
    tensors[name] = object()

    with pytest.raises(TypeError, match=name):
        validate(tensors["input_ids"], tensors["labels"], 0)


@pytest.mark.parametrize("name", ["input_ids", "labels"])
def test_validate_nll_score_inputs_requires_int64(validator, name):
    validate, int64_dtype = validator
    tensors = {
        "input_ids": make_tensor((1, 8), int64_dtype),
        "labels": make_tensor((1, 8), int64_dtype),
    }
    tensors[name] = make_tensor((1, 8), int64_dtype, dtype=object())

    with pytest.raises(ValueError, match=f"{name} must use infinicore.int64"):
        validate(tensors["input_ids"], tensors["labels"], 0)


@pytest.mark.parametrize(
    ("input_shape", "label_shape", "message"),
    [
        ((8,), (8,), "rank-2"),
        ((1, 8), (1, 7), "identical shapes"),
        ((2, 8), (2, 8), "batch_size=1"),
    ],
)
def test_validate_nll_score_inputs_rejects_invalid_shapes(
    validator, input_shape, label_shape, message
):
    validate, int64_dtype = validator
    with pytest.raises(ValueError, match=message):
        validate(
            make_tensor(input_shape, int64_dtype),
            make_tensor(label_shape, int64_dtype),
            0,
        )


@pytest.mark.parametrize("score_start", [-1, 8])
def test_validate_nll_score_inputs_rejects_empty_score_range(
    validator, score_start
):
    validate, int64_dtype = validator
    with pytest.raises(ValueError, match="select at least one token"):
        validate(
            make_tensor((1, 8), int64_dtype),
            make_tensor((1, 8), int64_dtype),
            score_start,
        )


@pytest.mark.parametrize("score_start", [True, 1.5, "1"])
def test_validate_nll_score_inputs_requires_integer_score_start(
    validator, score_start
):
    validate, int64_dtype = validator
    with pytest.raises(TypeError, match="score_start must be an integer"):
        validate(
            make_tensor((1, 8), int64_dtype),
            make_tensor((1, 8), int64_dtype),
            score_start,
        )
