import importlib.util
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import mock_open, patch


REPO_ROOT = Path(__file__).resolve().parents[2]


def _module(name, **attributes):
    module = types.ModuleType(name)
    for key, value in attributes.items():
        setattr(module, key, value)
    return module


def _load_bench_module():
    infinicore = _module("infinicore")
    infinicore.nn = types.SimpleNamespace(Module=object)
    infinicore.device = lambda *_args, **_kwargs: object()

    infinilm = _module("infinilm")
    infinilm.__path__ = []
    stub_modules = {
        "infinicore": infinicore,
        "infinilm": infinilm,
        "infinilm.modeling_utils": _module(
            "infinilm.modeling_utils", load_model_state_dict_by_file=lambda *_args: None
        ),
        "infinilm.distributed": _module("infinilm.distributed", DistConfig=object),
        "infinilm.infer_engine": _module(
            "infinilm.infer_engine", GenerationConfig=object, InferEngine=object
        ),
        "infinilm.base_config": _module("infinilm.base_config", BaseConfig=object),
        "infinilm.cache": _module(
            "infinilm.cache",
            StaticKVCacheConfig=object,
            PagedKVCacheConfig=object,
        ),
        "infinilm.moe_config": _module(
            "infinilm.moe_config", configure_moe_ep_backend=lambda *_args: (None, None)
        ),
        "infinilm.processors": _module(
            "infinilm.processors", AutoInfinilmProcessor=object
        ),
        "numpy": _module("numpy"),
        "tqdm": _module("tqdm", tqdm=lambda values, **_kwargs: values),
    }

    spec = importlib.util.spec_from_file_location(
        "infinilm_bench_case_test", REPO_ROOT / "examples" / "bench.py"
    )
    module = importlib.util.module_from_spec(spec)
    old_cwd = os.getcwd()
    try:
        os.chdir(REPO_ROOT)
        with patch.dict(sys.modules, stub_modules), patch(
            "builtins.open", mock_open(read_data="test prompt")
        ):
            spec.loader.exec_module(module)
    finally:
        os.chdir(old_cwd)
    return module


bench = _load_bench_module()


class PairSequenceLengthsTest(unittest.TestCase):
    def test_pairs_equal_length_lists_by_position(self):
        self.assertEqual(
            bench.pair_sequence_lengths([1024, 4096], [128, 256]),
            [(1024, 128), (4096, 256)],
        )

    def test_broadcasts_a_single_value_on_either_side(self):
        self.assertEqual(
            bench.pair_sequence_lengths([1024], [128, 256]),
            [(1024, 128), (1024, 256)],
        )
        self.assertEqual(
            bench.pair_sequence_lengths([1024, 4096], [128]),
            [(1024, 128), (4096, 128)],
        )

    def test_rejects_unpairable_or_nonpositive_lengths(self):
        with self.assertRaises(ValueError):
            bench.pair_sequence_lengths([1024, 2048], [64, 128, 256])
        with self.assertRaises(ValueError):
            bench.pair_sequence_lengths([0], [128])


class GetTestCasesTest(unittest.TestCase):
    @staticmethod
    def _model_dir():
        model_dir = tempfile.TemporaryDirectory()
        config_path = Path(model_dir.name) / "config.json"
        config_path.write_text(
            """{
                "model_type": "qwen3",
                "hidden_size": 128,
                "num_attention_heads": 8,
                "num_key_value_heads": 2,
                "num_hidden_layers": 4
            }""",
            encoding="utf-8",
        )
        return model_dir

    def test_preserves_single_case_behavior(self):
        with self._model_dir() as model_dir:
            cases = bench.get_test_cases(model_dir, [4], [2048], [512])

        self.assertEqual(len(cases), 1)
        case = next(iter(cases.values()))
        self.assertEqual(
            (case["batch_size"], case["input_len"], case["output_len"]),
            (4, 2048, 512),
        )

    def test_combines_batches_with_pairs_without_cartesian_lengths(self):
        with self._model_dir() as model_dir:
            cases = bench.get_test_cases(
                model_dir, [1, 2], [1024, 4096], [128, 256]
            )

        actual = {
            (case["batch_size"], case["input_len"], case["output_len"])
            for case in cases.values()
        }
        self.assertEqual(
            actual,
            {
                (1, 1024, 128),
                (1, 4096, 256),
                (2, 1024, 128),
                (2, 4096, 256),
            },
        )
        self.assertEqual(len(cases), 4)

    def test_cache_helpers_cover_each_shape_at_full_capacity(self):
        cases = [
            {"batch_size": 4, "input_len": 1024, "output_len": 128},
            {"batch_size": 1, "input_len": 4096, "output_len": 256},
            {"batch_size": 4, "input_len": 1024, "output_len": 512},
        ]

        self.assertEqual(bench.get_paged_kv_cache_num_blocks(cases, 256), 24)
        self.assertEqual(
            bench.get_warmup_shapes(cases),
            {(4, 1024): 512, (1, 4096): 256},
        )


if __name__ == "__main__":
    unittest.main()
