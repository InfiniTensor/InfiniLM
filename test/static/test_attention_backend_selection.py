import ast
import json
import runpy
import sys
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
ATTENTION_PATH = ROOT / "python/infinilm/config/attention.py"
resolve_attention_backend = runpy.run_path(str(ATTENTION_PATH))[
    "resolve_attention_backend"
]


def load_config(path):
    attention = ModuleType("infinilm.config.attention")
    attention.resolve_attention_backend = resolve_attention_backend
    kv_transfer = ModuleType("infinilm.config.kv_transfer")
    kv_transfer.KVTransferConfig = object
    moe = ModuleType("infinilm.moe_config")
    moe.MOE_EP_BACKEND_HELP = "MoE backend"
    with patch.dict(
        sys.modules,
        {
            attention.__name__: attention,
            kv_transfer.__name__: kv_transfer,
            moe.__name__: moe,
        },
    ):
        return runpy.run_path(str(ROOT / path))


class AttentionBackendSelectionTest(unittest.TestCase):
    def test_default_follows_cache_layout(self):
        self.assertEqual(resolve_attention_backend("default", "static"), "static-attn")
        self.assertEqual(resolve_attention_backend("default", "paged"), "flash-attn")

    def test_explicit_supported_backend_is_preserved(self):
        for backend in ("static-attn", "flash-attn"):
            for cache_type in ("static", "paged"):
                with self.subTest(backend=backend, cache_type=cache_type):
                    self.assertEqual(
                        resolve_attention_backend(backend, cache_type), backend
                    )

    def test_removed_backend_is_rejected(self):
        for cache_type in ("static", "paged"):
            with self.subTest(cache_type=cache_type):
                with self.assertRaisesRegex(ValueError, "removed.*flash-attn"):
                    resolve_attention_backend("paged-attn", cache_type)

    def test_engine_config_resolves_attention(self):
        engine_config = load_config("python/infinilm/config/engine_config.py")[
            "EngineConfig"
        ]
        self.assertEqual(engine_config("model").attn_backend, "flash-attn")
        self.assertEqual(
            engine_config("model", cache_type="static").attn_backend, "static-attn"
        )
        with self.assertRaisesRegex(ValueError, "removed"):
            engine_config("model", attn_backend="paged-attn")

    def test_cli_resolves_attention_for_the_cache_flag(self):
        base_config = load_config("python/infinilm/base_config.py")["BaseConfig"]
        for options, expected in (
            ([], "static-attn"),
            (["--enable-paged-attn"], "flash-attn"),
            (["--enable-paged-attn", "--attn=flash-attn"], "flash-attn"),
            (["--attn=static-attn"], "static-attn"),
        ):
            with self.subTest(options=options):
                with patch.object(
                    sys, "argv", ["infinilm", "--model=model", "--device=cpu", *options]
                ):
                    self.assertEqual(base_config().attn, expected)

    def test_low_level_engine_passes_resolved_backend_to_native(self):
        source = (ROOT / "python/infinilm/infer_engine.py").read_text(encoding="utf-8")
        engine = next(
            node
            for node in ast.parse(source).body
            if isinstance(node, ast.ClassDef) and node.name == "InferEngine"
        )
        engine.body = [
            node
            for node in engine.body
            if isinstance(node, ast.FunctionDef) and node.name == "__init__"
        ]
        engine.bases = [ast.Name(id="NativeEngine", ctx=ast.Load())]

        class NativeEngine:
            def __init__(self, *args):
                self.native_backend = args[5]

        class PagedCache:
            pass

        namespace = {
            "NativeEngine": NativeEngine,
            "PagedKVCacheConfig": PagedCache,
            "resolve_attention_backend": resolve_attention_backend,
            "json": json,
            "read_hf_config": lambda path: {},
            "read_hf_generation_config": lambda path: {},
            "_infer_position_id_axes": lambda config: 1,
            "model_uses_mamba_cache": lambda config: False,
        }
        exec(
            compile(
                ast.fix_missing_locations(ast.Module(body=[engine], type_ignores=[])),
                "infer_engine.py",
                "exec",
            ),
            namespace,
        )
        for cache, backend, expected in (
            (None, "default", "static-attn"),
            (PagedCache(), "default", "flash-attn"),
            (PagedCache(), "flash-attn", "flash-attn"),
        ):
            with self.subTest(cache=cache, backend=backend):
                instance = namespace["InferEngine"](
                    "model",
                    device=SimpleNamespace(_underlying=SimpleNamespace(type="cpu")),
                    distributed_config=SimpleNamespace(
                        _underlying=None, moe_ep_backend="disabled", moe_ep_size=1
                    ),
                    cache_config=cache,
                    attention_backend=backend,
                )
                self.assertEqual(instance.native_backend, expected)


if __name__ == "__main__":
    unittest.main()
