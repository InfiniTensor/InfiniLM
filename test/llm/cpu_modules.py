"""Load cache and scheduler modules without initializing the GPU engine."""

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace


def load_cpu_modules():
    directory = Path(__file__).resolve().parents[2] / "python/infinilm/llm"
    missing = object()
    originals = {}
    modules = {}
    try:
        for name in (
            "prefix_cache",
            "sampling_params",
            "request",
            "cache_manager",
            "scheduler",
        ):
            key = f"infinilm.llm.{name}"
            spec = importlib.util.spec_from_file_location(key, directory / f"{name}.py")
            module = importlib.util.module_from_spec(spec)
            originals[key] = sys.modules.get(key, missing)
            sys.modules[key] = module
            spec.loader.exec_module(module)
            modules[name] = module
    finally:
        # Retain imported dependencies, including non-reloadable native extensions.
        for key, original in originals.items():
            if original is missing:
                sys.modules.pop(key, None)
            else:
                sys.modules[key] = original
    return SimpleNamespace(**modules)
