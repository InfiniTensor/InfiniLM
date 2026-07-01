from importlib import import_module

from .plugins import (
    ModelSpec,
    load_plugin,
    load_plugins,
    register_model,
    registered_model_types,
)


_LAZY_ATTRS = {
    "AutoLlamaModel": ("infinilm.models", "AutoLlamaModel"),
    "LLM": ("infinilm.llm", "LLM"),
    "AsyncLLMEngine": ("infinilm.llm", "AsyncLLMEngine"),
    "SamplingParams": ("infinilm.llm", "SamplingParams"),
    "RequestOutput": ("infinilm.llm", "RequestOutput"),
    "TokenOutput": ("infinilm.llm", "TokenOutput"),
}

_LAZY_MODULES = {"distributed", "cache", "llm", "base_config"}


def __getattr__(name):
    if name in _LAZY_MODULES:
        module = import_module(f".{name}", __name__)
        globals()[name] = module
        return module

    target = _LAZY_ATTRS.get(name)
    if target is not None:
        module_name, attr_name = target
        value = getattr(import_module(module_name), attr_name)
        globals()[name] = value
        return value

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "AutoLlamaModel",
    "distributed",
    "cache",
    "llm",
    "base_config",
    # LLM classes
    "LLM",
    "AsyncLLMEngine",
    "SamplingParams",
    "RequestOutput",
    "TokenOutput",
    # Out-of-tree model plugins
    "ModelSpec",
    "load_plugin",
    "load_plugins",
    "register_model",
    "registered_model_types",
]
