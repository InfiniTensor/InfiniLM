"""Validate cache/chunk configuration once across public entrypoints."""

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
    load_module,
)

OPTIONS = dict(
    prefix_cache_policy="slru",
    prefix_cache_protected_ratio=0.6,
    prefill_chunk_size=128,
    tensor_parallel_size=2,
)
CLI = [
    "--enable-paged-attn",
    "--prefix-cache-policy",
    "slru",
    "--prefix-cache-protected-ratio",
    "0.6",
    "--prefill-chunk-size",
    "128",
    "--tp",
    "2",
]


class EngineConfigTests(unittest.TestCase):
    def parse_cli(self, *args):
        with (
            patch.object(
                sys, "argv", ["server", "--model", "unused", "--device", "cpu", *args]
            ),
            patch.dict(os.environ),
        ):
            return BaseConfig()

    def assert_options(self, config):
        for key, value in OPTIONS.items():
            self.assertEqual(getattr(config, key), value)

    def test_defaults_preserve_existing_behavior(self):
        for cache_type in ("paged", "static"):
            config = EngineConfig("unused", cache_type=cache_type)
            self.assertEqual(
                (
                    config.prefix_cache_policy,
                    config.prefix_cache_protected_ratio,
                    config.prefill_chunk_size,
                ),
                ("lru", 0.8, 0),
            )
        cli = self.parse_cli()
        self.assertEqual(
            (
                cli.prefix_cache_policy,
                cli.prefix_cache_protected_ratio,
                cli.prefill_chunk_size,
            ),
            ("lru", 0.8, 0),
        )

    def test_invalid_policy_ratio_and_chunk_values(self):
        invalid = [("prefix_cache_policy", "fifo")]
        invalid += [
            ("prefix_cache_protected_ratio", r)
            for r in (-0.1, 0, 1, 1.1, float("nan"), float("inf"), -float("inf"))
        ]
        invalid += [
            ("prefill_chunk_size", n) for n in (-1, 1.5, "128", None, True, False)
        ]
        for key, value in invalid:
            with (
                self.subTest(key=key, value=value),
                self.assertRaisesRegex(ValueError, key),
            ):
                EngineConfig("unused", **{key: value})
        for enabled in (True, False):
            with self.assertRaisesRegex(ValueError, "paged"):
                EngineConfig(
                    "unused",
                    cache_type="static",
                    prefix_cache_policy="slru",
                    enable_prefix_caching=enabled,
                )

    def test_chunk_execution_capabilities(self):
        inactive = SERVER_MODULE.KVTransferConfig()
        EngineConfig("unused", prefill_chunk_size=128, kv_transfer_config=inactive)
        for tp in (1, 2):
            for backend in ("default", "paged-attn", "flash-attn"):
                EngineConfig(
                    "unused",
                    prefill_chunk_size=128,
                    tensor_parallel_size=tp,
                    enable_graph=True,
                    attn_backend=backend,
                )
        for stage in (0, 1):
            EngineConfig(
                "unused",
                prefill_chunk_size=300,
                pipeline_parallel_size=2,
                pipeline_parallel_stage=stage,
                prefix_cache_policy="slru",
            )
        active = SERVER_MODULE.KVTransferConfig(
            kv_connector="MooncakeConnector", kv_role="kv_producer"
        )
        for overrides in (
            {"cache_type": "static"},
            {"tensor_parallel_size": 4},
            {"pipeline_parallel_size": 3},
            {"use_mla": True},
            {"draft_model_path": "draft"},
            {"kv_transfer_config": active},
            {"tensor_parallel_size": 2, "pipeline_parallel_size": 2},
        ):
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                EngineConfig("unused", prefill_chunk_size=128, **overrides)
            EngineConfig("unused", prefill_chunk_size=0, **overrides)
        for overrides in (
            {"pipeline_parallel_size": 2},
            {"device": "cpu"},
            {"attn_backend": "unsupported"},
        ):
            with (
                self.subTest(overrides=overrides),
                self.assertRaisesRegex(ValueError, "requires PP=1"),
            ):
                EngineConfig(
                    "unused", prefill_chunk_size=128, enable_graph=True, **overrides
                )

    def test_cli_validation_before_dispatch(self):
        config = self.parse_cli(*CLI)
        self.assertEqual(
            (
                config.prefix_cache_policy,
                config.prefix_cache_protected_ratio,
                config.prefill_chunk_size,
                config.tp,
            ),
            ("slru", 0.6, 128, 2),
        )
        invalid = [("--prefix-cache-policy", "fifo")]
        invalid += [
            ("--prefix-cache-protected-ratio", v)
            for v in ("0", "1", "nan", "inf", "-inf")
        ]
        invalid += [("--prefill-chunk-size", v) for v in ("-1", "1.5", "True")]
        invalid += [
            ("--prefill-chunk-size", "128", *extra)
            for extra in (
                ("--tp", "4"),
                ("--pp", "3"),
                ("--pp", "3", "--node-rank", "1"),
                ("--draft-model", "draft"),
            )
        ]
        for args in invalid:
            with (
                self.subTest(args=args),
                redirect_stderr(io.StringIO()),
                self.assertRaises(SystemExit) as error,
            ):
                self.parse_cli(*args)
            self.assertEqual(error.exception.code, 2)
        for value in ("0", "1", "128"):
            self.assertEqual(
                self.parse_cli("--prefill-chunk-size", value).prefill_chunk_size,
                int(value),
            )

    def test_convenience_apis_forward_and_validate_options(self):
        with patch.object(
            LLM_MODULE, "LLMEngine", lambda config: SimpleNamespace(config=config)
        ):
            for constructor in (LLM_MODULE.LLM, LLM_MODULE.AsyncLLMEngine):
                self.assert_options(constructor("unused", **OPTIONS).engine.config)
                default = constructor("unused").engine.config
                self.assertEqual(
                    (default.prefix_cache_policy, default.prefill_chunk_size),
                    ("lru", 0),
                )
                with self.assertRaisesRegex(ValueError, "paged"):
                    constructor(
                        "unused", cache_type="static", prefix_cache_policy="slru"
                    )
                with self.assertRaisesRegex(ValueError, "prefill_chunk_size"):
                    constructor("unused", cache_type="static", prefill_chunk_size=128)

    def test_engine_forwards_options_to_scheduler(self):
        runner = SimpleNamespace(
            device="cpu",
            dtype="float16",
            eos_token_id=[],
            processor=SimpleNamespace(get_tokenizer=lambda: None),
            model_engine=SimpleNamespace(hf_config={"max_position_embeddings": 4096}),
        )
        with (
            patch.object(LLM_MODULE, "ModelRunner", lambda config: runner),
            patch.object(LLM_MODULE, "Scheduler") as scheduler,
        ):
            LLM_MODULE.LLMEngine(EngineConfig("unused", **OPTIONS))
        for key in (
            "prefix_cache_policy",
            "prefix_cache_protected_ratio",
            "prefill_chunk_size",
        ):
            self.assertEqual(scheduler.call_args.kwargs[key], OPTIONS[key])

    def test_pipeline_worker_forwards_options(self):
        config = self.parse_cli(*CLI[:-2], "--pp", "2", "--node-rank", "1")
        captured, closed = [], []

        def make_runner(config, initialize_processor):
            self.assertFalse(initialize_processor)
            captured.append(config)
            return SimpleNamespace(close=lambda: closed.append(True))

        replacements = {
            "infinilm.base_config": SimpleNamespace(BaseConfig=BaseConfig),
            "infinilm.config.engine_config": SimpleNamespace(EngineConfig=EngineConfig),
            "infinilm.distributed.pipeline_transport": SimpleNamespace(
                PipelineWorkerClient=lambda *a, **kw: SimpleNamespace(
                    serve_forever=lambda: None
                )
            ),
            "infinilm.llm.model_runner.model_runner": SimpleNamespace(
                ModelRunner=make_runner
            ),
        }
        with patch.dict(sys.modules, replacements):
            worker = load_module(
                "infinilm.server.pipeline_worker", "server/pipeline_worker.py"
            )
            worker.run_worker(config)
        self.assertEqual(len(captured), 1)
        self.assertEqual(
            (
                captured[0].prefill_chunk_size,
                captured[0].prefix_cache_policy,
                captured[0].prefix_cache_protected_ratio,
                captured[0].pipeline_parallel_stage,
            ),
            (128, "slru", 0.6, 1),
        )
        self.assertEqual(closed, [True])

    def test_cli_reaches_server_lifespan(self):
        configs = []

        async def start_without_listener(server):
            app = server._create_app()
            async with app.router.lifespan_context(app):
                configs.append(server.engine.config)

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
            patch.dict(os.environ),
            patch.object(
                sys, "argv", ["server", "--model", "unused", "--device", "cpu", *CLI]
            ),
        ):
            SERVER_MODULE.main()
        self.assertEqual(len(configs), 1)
        self.assert_options(configs[0])

    def test_offline_cli_reaches_llm(self):
        configs = []
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

        def chat(model, messages):
            configs.append(model.config)
            return []

        with (
            patch.dict(sys.modules, modules),
            patch.dict(os.environ),
            patch.object(
                sys,
                "argv",
                ["test_infer", "--model", "unused", "--device", "cpu", *CLI],
            ),
            patch.object(
                LLM_MODULE,
                "LLMEngine",
                lambda config: SimpleNamespace(close=lambda: None),
            ),
            patch.object(LLM_MODULE.LLM, "chat", chat),
            patch.object(SERVER_MODULE.logging, "basicConfig"),
            redirect_stdout(io.StringIO()),
        ):
            runpy.run_path(
                str(SOURCE.parents[1] / "examples/test_infer.py"), run_name="__main__"
            )
        self.assertEqual(len(configs), 1)
        self.assert_options(configs[0])


if __name__ == "__main__":
    unittest.main()
