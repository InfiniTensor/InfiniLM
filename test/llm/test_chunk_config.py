"""CPU configuration coverage; native model construction is replaced at its boundary."""

import asyncio
import importlib.util
import io
import os
import runpy
import sys
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

from chunk_test_support import MODULES

SOURCE = Path(__file__).resolve().parents[2] / "python/infinilm"


def load_module(name, relative_path):
    spec = importlib.util.spec_from_file_location(name, SOURCE / relative_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_config_modules():
    with patch.dict(sys.modules):
        # Keep package __init__ files from importing the native model extension.
        for name in ("infinilm", "infinilm.config", "infinilm.llm"):
            sys.modules[name] = ModuleType(name)
        for name, module in MODULES.items():
            sys.modules[f"infinilm.llm.{name}"] = module
        kv = load_module("infinilm.config.kv_transfer", "config/kv_transfer.py")
        sys.modules["infinilm.config"].KVTransferConfig = kv.KVTransferConfig
        engine_config = load_module(
            "infinilm.config.engine_config", "config/engine_config.py"
        )
        load_module("infinilm.moe_config", "moe_config.py")
        base = load_module("infinilm.base_config", "base_config.py")
        load_module("infinilm.llm.static_scheduler", "llm/static_scheduler.py")
        sys.modules["infinilm.infer_engine"] = SimpleNamespace(
            read_hf_config=lambda path: {}, model_uses_mamba_cache=lambda config: False
        )
        sys.modules["infinilm.kv_connector"] = SimpleNamespace(
            KVConnectorFactory=object, KVConnectorRole=object
        )
        sys.modules["infinilm.llm.model_runner.model_runner"] = SimpleNamespace(
            ModelRunner=object
        )
        sys.modules["infinilm.multimodal.multimodal"] = SimpleNamespace(
            resolve_multimodal_inputs=object
        )
        llm = load_module("infinilm.llm.llm", "llm/llm.py")
        for name in ("AsyncLLMEngine", "FinishReason", "SamplingParams"):
            setattr(sys.modules["infinilm.llm"], name, getattr(llm, name))
        server = load_module(
            "infinilm.server.inference_server", "server/inference_server.py"
        )
    return engine_config.EngineConfig, base.BaseConfig, llm, server


EngineConfig, BaseConfig, LLM_MODULE, SERVER_MODULE = load_config_modules()


class ChunkConfigTests(unittest.TestCase):
    def test_default_preserves_unbounded_prefill_for_both_caches(self):
        for cache_type in ("paged", "static"):
            self.assertEqual(
                EngineConfig("unused", cache_type=cache_type).prefill_chunk_size, 0
            )

    def test_positive_chunk_size_and_inactive_transfer_are_supported(self):
        kv_config = SERVER_MODULE.KVTransferConfig()
        config = EngineConfig(
            "unused", prefill_chunk_size=128, kv_transfer_config=kv_config
        )
        self.assertEqual(config.prefill_chunk_size, 128)

    def test_tp2_chunking_is_supported_by_config_and_convenience_apis(self):
        config = EngineConfig("unused", prefill_chunk_size=128, tensor_parallel_size=2)
        self.assertEqual(config.tensor_parallel_size, 2)
        with patch.object(
            LLM_MODULE, "LLMEngine", lambda config: SimpleNamespace(config=config)
        ):
            for constructor in (LLM_MODULE.LLM, LLM_MODULE.AsyncLLMEngine):
                engine = constructor(
                    "unused", prefill_chunk_size=128, tensor_parallel_size=2
                )
                self.assertEqual(engine.config.tensor_parallel_size, 2)
                self.assertEqual(engine.config.prefill_chunk_size, 128)
        cli = self.parse_cli("--tp", "2", "--prefill-chunk-size", "128")
        self.assertEqual((cli.tp, cli.prefill_chunk_size), (2, 128))

    def test_pp2_eager_chunking_and_worker_cli(self):
        for stage in (0, 1):
            EngineConfig(
                "unused",
                prefill_chunk_size=300,
                pipeline_parallel_size=2,
                pipeline_parallel_stage=stage,
                prefix_cache_policy="slru",
            )
            cfg = self.parse_cli(
                "--pp", "2", "--node-rank", str(stage), "--prefill-chunk-size", "300"
            )
            self.assertEqual(cfg.pp, 2)
        with self.assertRaisesRegex(ValueError, "TP/PP"):
            EngineConfig(
                "unused",
                prefill_chunk_size=300,
                tensor_parallel_size=2,
                pipeline_parallel_size=2,
            )

    def test_pipeline_worker_forwards_chunk_and_cache_configuration(self):
        cfg = self.parse_cli(
            "--pp",
            "2",
            "--node-rank",
            "1",
            "--enable-paged-attn",
            "--prefill-chunk-size",
            "300",
            "--prefix-cache-policy",
            "slru",
        )
        captured = []
        closed = []

        def make_runner(config, initialize_processor):
            self.assertFalse(initialize_processor)
            captured.append(config)
            return SimpleNamespace(close=lambda: closed.append(True))

        replacements = {
            "infinilm.base_config": SimpleNamespace(BaseConfig=BaseConfig),
            "infinilm.config.engine_config": SimpleNamespace(EngineConfig=EngineConfig),
            "infinilm.distributed.pipeline_transport": SimpleNamespace(
                PipelineWorkerClient=lambda *args, **kwargs: SimpleNamespace(
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
            worker.run_worker(cfg)
        self.assertEqual(len(captured), 1)
        self.assertEqual(captured[0].prefill_chunk_size, 300)
        self.assertEqual(captured[0].prefix_cache_policy, "slru")
        self.assertEqual(captured[0].pipeline_parallel_stage, 1)
        self.assertEqual(closed, [True])

    def test_chunk_size_rejects_negative_noninteger_and_boolean(self):
        for value in (-1, 1.5, "128", None, True, False):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "prefill_chunk_size"):
                    EngineConfig("unused", prefill_chunk_size=value)

    def test_chunking_allows_tp_decode_graphs(self):
        config = EngineConfig(
            "unused",
            prefill_chunk_size=512,
            enable_graph=True,
            attn_backend="flash-attn",
            device="cuda",
            tensor_parallel_size=1,
        )
        self.assertTrue(config.enable_graph)
        self.assertEqual(config.prefill_chunk_size, 512)
        for tp in (1, 2):
            for backend in ("default", "paged-attn", "flash-attn"):
                EngineConfig(
                    "unused",
                    prefill_chunk_size=512,
                    enable_graph=True,
                    tensor_parallel_size=tp,
                    attn_backend=backend,
                )
        for overrides in (
            {"pipeline_parallel_size": 2},
            {"device": "cpu"},
            {"attn_backend": "unsupported"},
        ):
            values = dict(
                prefill_chunk_size=512,
                enable_graph=True,
                attn_backend="flash-attn",
                device="cuda",
            )
            values.update(overrides)
            with (
                self.subTest(overrides=overrides),
                self.assertRaisesRegex(ValueError, "requires PP=1"),
            ):
                EngineConfig("unused", **values)

    def test_enabled_chunking_rejects_unsupported_execution_modes(self):
        active_transfer = SERVER_MODULE.KVTransferConfig(
            kv_connector="MooncakeConnector", kv_role="kv_producer"
        )
        for overrides in (
            {"cache_type": "static"},
            {"enable_graph": True, "pipeline_parallel_size": 2},
            {"tensor_parallel_size": 4},
            {"pipeline_parallel_size": 3},
            {"use_mla": True},
            {"draft_model_path": "draft"},
            {"kv_transfer_config": active_transfer},
        ):
            with self.subTest(overrides=overrides):
                with self.assertRaisesRegex(ValueError, "prefill_chunk_size"):
                    EngineConfig("unused", prefill_chunk_size=128, **overrides)
                EngineConfig("unused", prefill_chunk_size=0, **overrides)

    def parse_cli(self, *args):
        with (
            patch.object(
                sys, "argv", ["server", "--model", "unused", "--device", "cpu", *args]
            ),
            patch.dict(os.environ),
        ):
            return BaseConfig()

    def test_cli_default_and_explicit_nonnegative_chunk_size(self):
        self.assertEqual(self.parse_cli().prefill_chunk_size, 0)
        for value in ("0", "1", "128"):
            self.assertEqual(
                self.parse_cli("--prefill-chunk-size", value).prefill_chunk_size,
                int(value),
            )

    def test_cli_rejects_chunked_parallelism_before_worker_dispatch(self):
        for parallel in (
            ("--tp", "4"),
            ("--pp", "3"),
            ("--pp", "3", "--node-rank", "1"),
        ):
            with self.subTest(parallel=parallel):
                with (
                    redirect_stderr(io.StringIO()),
                    self.assertRaises(SystemExit) as error,
                ):
                    self.parse_cli("--prefill-chunk-size", "128", *parallel)
                self.assertEqual(error.exception.code, 2)
                self.assertEqual(self.parse_cli(*parallel).prefill_chunk_size, 0)

    def test_cli_rejects_negative_and_noninteger_chunk_sizes(self):
        for value in ("-1", "1.5", "True"):
            with self.subTest(value=value), redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    self.parse_cli("--prefill-chunk-size", value)

    def test_cli_rejects_draft_model_with_chunking(self):
        with redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                self.parse_cli("--prefill-chunk-size", "128", "--draft-model", "draft")
        self.assertEqual(self.parse_cli("--draft-model", "draft").prefill_chunk_size, 0)

    def test_convenience_constructors_validate_chunking_before_model_load(self):
        # Only native model construction is replaced; EngineConfig stays real.
        with patch.object(
            LLM_MODULE, "LLMEngine", lambda config: SimpleNamespace(config=config)
        ):
            for constructor in (LLM_MODULE.LLM, LLM_MODULE.AsyncLLMEngine):
                self.assertEqual(
                    constructor("unused").engine.config.prefill_chunk_size, 0
                )
                config = constructor("unused", prefill_chunk_size=128).engine.config
                self.assertEqual(config.prefill_chunk_size, 128)
                with self.assertRaisesRegex(ValueError, "prefill_chunk_size"):
                    constructor("unused", cache_type="static", prefill_chunk_size=128)

    def test_cli_reaches_server_lifespan_and_async_engine_config(self):
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
            patch.object(
                sys,
                "argv",
                [
                    "server",
                    "--model",
                    "unused",
                    "--device",
                    "cpu",
                    "--enable-paged-attn",
                    "--tp",
                    "2",
                    "--prefill-chunk-size",
                    "128",
                ],
            ),
            patch.dict(os.environ),
        ):
            SERVER_MODULE.main()
        self.assertEqual(len(configs), 1)
        self.assertEqual(configs[0].prefill_chunk_size, 128)
        self.assertEqual(configs[0].tensor_parallel_size, 2)

    def test_offline_cli_reaches_llm_config(self):
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
        with (
            patch.dict(sys.modules, modules),
            patch.dict(os.environ),
            patch.object(
                sys,
                "argv",
                [
                    "test_infer",
                    "--model",
                    "unused",
                    "--device",
                    "cpu",
                    "--enable-paged-attn",
                    "--tp",
                    "2",
                    "--prefill-chunk-size",
                    "128",
                ],
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
        self.assertEqual(configs[0].prefill_chunk_size, 128)
        self.assertEqual(configs[0].tensor_parallel_size, 2)


if __name__ == "__main__":
    unittest.main()
