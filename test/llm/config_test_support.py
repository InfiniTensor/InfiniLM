"""Load configuration entrypoints without constructing native model extensions."""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

from cache_test_support import MODULES

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
