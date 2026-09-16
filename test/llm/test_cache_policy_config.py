"""CPU configuration coverage; native model construction is replaced at its boundary."""

import asyncio
import io
import os
import runpy
import sys
import unittest
from contextlib import redirect_stderr, redirect_stdout
from types import SimpleNamespace
from unittest.mock import patch

from config_test_support import (
    LLM_MODULE,
    SERVER_MODULE,
    SOURCE,
    BaseConfig,
    EngineConfig,
)


class CachePolicyConfigTests(unittest.TestCase):
    def test_engine_defaults_preserve_lru_for_both_cache_types(self):
        for cache_type in ("paged", "static"):
            config = EngineConfig("unused", cache_type=cache_type)
            self.assertEqual(config.prefix_cache_policy, "lru")
            self.assertEqual(config.prefix_cache_protected_ratio, 0.8)

    def test_engine_rejects_unknown_policy(self):
        with self.assertRaisesRegex(ValueError, "prefix_cache_policy"):
            EngineConfig("unused", prefix_cache_policy="fifo")

    def test_engine_rejects_nonfinite_and_boundary_ratios(self):
        for ratio in (-0.1, 0, 1, 1.1, float("nan"), float("inf"), -float("inf")):
            with self.subTest(ratio=ratio):
                with self.assertRaisesRegex(ValueError, "prefix_cache_protected_ratio"):
                    EngineConfig("unused", prefix_cache_protected_ratio=ratio)

    def test_static_cache_rejects_slru_even_when_prefix_caching_disabled(self):
        for enabled in (True, False):
            with self.assertRaisesRegex(ValueError, "paged"):
                EngineConfig(
                    "unused",
                    cache_type="static",
                    prefix_cache_policy="slru",
                    enable_prefix_caching=enabled,
                )

    def parse_cli(self, *args):
        with patch.object(
            sys, "argv", ["server", "--model", "unused", "--device", "cpu", *args]
        ):
            with patch.dict(os.environ):
                return BaseConfig()

    def test_cli_defaults_and_explicit_slru(self):
        default = self.parse_cli()
        self.assertEqual(default.prefix_cache_policy, "lru")
        self.assertEqual(default.prefix_cache_protected_ratio, 0.8)
        custom = self.parse_cli(
            "--enable-paged-attn",
            "--prefix-cache-policy",
            "slru",
            "--prefix-cache-protected-ratio",
            "0.6",
        )
        self.assertEqual(custom.prefix_cache_policy, "slru")
        self.assertEqual(custom.prefix_cache_protected_ratio, 0.6)

    def test_cli_rejects_unknown_policy_and_invalid_ratio(self):
        invalid = [("--prefix-cache-policy", "fifo")]
        invalid.extend(
            ("--prefix-cache-protected-ratio", value)
            for value in ("0", "1", "nan", "inf", "-inf")
        )
        for args in invalid:
            with self.subTest(args=args), redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    self.parse_cli(*args)

    def test_convenience_constructors_forward_policy_to_engine_config(self):
        # Engine construction normally loads model weights; retain real config validation.
        with patch.object(
            LLM_MODULE, "LLMEngine", lambda config: SimpleNamespace(config=config)
        ):
            for constructor in (LLM_MODULE.LLM, LLM_MODULE.AsyncLLMEngine):
                default = constructor("unused")
                custom = constructor(
                    "unused",
                    prefix_cache_policy="slru",
                    prefix_cache_protected_ratio=0.6,
                )
                self.assertEqual(default.engine.config.prefix_cache_policy, "lru")
                self.assertEqual(custom.engine.config.prefix_cache_policy, "slru")
                self.assertEqual(custom.engine.config.prefix_cache_protected_ratio, 0.6)
                with self.assertRaisesRegex(ValueError, "paged"):
                    constructor(
                        "unused", cache_type="static", prefix_cache_policy="slru"
                    )

    def test_paged_engine_forwards_policy_to_scheduler(self):
        runner = SimpleNamespace(
            device="cpu",
            dtype="float16",
            eos_token_id=[],
            processor=SimpleNamespace(get_tokenizer=lambda: None),
            model_engine=SimpleNamespace(hf_config={"max_position_embeddings": 4096}),
        )
        config = EngineConfig(
            "unused", prefix_cache_policy="slru", prefix_cache_protected_ratio=0.6
        )
        with patch.object(LLM_MODULE, "ModelRunner", lambda config: runner):
            with patch.object(LLM_MODULE, "Scheduler") as scheduler:
                LLM_MODULE.LLMEngine(config)
        self.assertEqual(scheduler.call_args.kwargs["prefix_cache_policy"], "slru")
        self.assertEqual(
            scheduler.call_args.kwargs["prefix_cache_protected_ratio"], 0.6
        )

    def run_offline_example(self, cli=False, **kwargs):
        configs = []

        def chat(model, messages):
            configs.append(model.config)
            return []

        modules = {
            "infinilm.base_config": SimpleNamespace(BaseConfig=BaseConfig),
            "infinilm.llm.llm": LLM_MODULE,
            "infinilm.moe_config": SimpleNamespace(
                configure_moe_ep_backend=SERVER_MODULE.configure_moe_ep_backend
            ),
            "infinilm.processors.videonsa_processor": SimpleNamespace(
                decode_video_frames=object
            ),
        }
        argv = [
            "test_infer",
            "--model",
            "unused",
            "--device",
            "cpu",
            "--enable-paged-attn",
            "--prefix-cache-policy",
            "slru",
            "--prefix-cache-protected-ratio",
            "0.6",
        ]
        with (
            patch.dict(sys.modules, modules),
            patch.dict(os.environ),
            patch.object(sys, "argv", argv),
            patch.object(
                LLM_MODULE,
                "LLMEngine",
                lambda config: SimpleNamespace(close=lambda: None),
            ),
            patch.object(LLM_MODULE.LLM, "chat", chat),
            patch.object(SERVER_MODULE.logging, "basicConfig"),
            redirect_stdout(io.StringIO()),
        ):
            example = runpy.run_path(
                str(SOURCE.parents[1] / "examples/test_infer.py"),
                run_name="__main__" if cli else "offline_example",
            )
            if not cli:
                example["test"](["hello"], "unused", enable_paged_attn=True, **kwargs)
        return configs

    def test_offline_example_forwards_policy_to_llm_config(self):
        default = self.run_offline_example()
        custom = self.run_offline_example(
            prefix_cache_policy="slru", prefix_cache_protected_ratio=0.6
        )
        self.assertEqual(default[0].prefix_cache_policy, "lru")
        self.assertEqual(default[0].prefix_cache_protected_ratio, 0.8)
        self.assertEqual(custom[0].prefix_cache_policy, "slru")
        self.assertEqual(custom[0].prefix_cache_protected_ratio, 0.6)

    def test_offline_cli_reaches_llm_config(self):
        configs = self.run_offline_example(cli=True)
        self.assertEqual(len(configs), 1)
        self.assertEqual(configs[0].prefix_cache_policy, "slru")
        self.assertEqual(configs[0].prefix_cache_protected_ratio, 0.6)

    def test_cli_reaches_server_lifespan_and_async_engine_config(self):
        configs = []

        async def start_without_listener(server):
            app = server._create_app()
            async with app.router.lifespan_context(app):
                configs.append(server.engine.config)

        argv = [
            "server",
            "--model",
            "unused",
            "--device",
            "cpu",
            "--enable-paged-attn",
            "--prefix-cache-policy",
            "slru",
            "--prefix-cache-protected-ratio",
            "0.6",
        ]
        with (
            patch.object(
                LLM_MODULE, "LLMEngine", lambda config: SimpleNamespace(config=config)
            ),
            patch.object(LLM_MODULE.AsyncLLMEngine, "start"),
            patch.object(LLM_MODULE.AsyncLLMEngine, "stop"),
            patch.object(
                SERVER_MODULE.InferenceServer,
                "start",
                lambda server: asyncio.run(start_without_listener(server)),
            ),
            patch.object(SERVER_MODULE, "setup_logging"),
            patch.object(sys, "argv", argv),
            patch.dict(os.environ),
        ):
            SERVER_MODULE.main()
        self.assertEqual(len(configs), 1)
        self.assertEqual(configs[0].prefix_cache_policy, "slru")
        self.assertEqual(configs[0].prefix_cache_protected_ratio, 0.6)


if __name__ == "__main__":
    unittest.main()
