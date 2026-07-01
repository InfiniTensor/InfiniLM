from __future__ import annotations

import importlib
import importlib.util
import inspect
import os
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Any


StateDict = dict[str, Any]
ConfigDict = dict[str, Any]
ConfigAdapter = Callable[[ConfigDict], ConfigDict] | Mapping[str, Any]
WeightRemapper = Callable[..., StateDict]


@dataclass(slots=True)
class ModelSpec:
    """Out-of-tree model registration contract.

    `model_type`/`model_types` identify HuggingFace config model_type values.
    `backend_model_type` is the InfiniLM C++ model implementation to reuse.
    Python callbacks run only while loading config or weights, not in the
    token-by-token inference hot path.
    """

    model_type: str | None = None
    model_types: Sequence[str] | None = None
    backend_model_type: str | None = None
    config_adapter: ConfigAdapter | None = None
    weight_remapper: WeightRemapper | None = None
    weight_rules: Sequence[WeightRemapper] = field(default_factory=tuple)
    processor_cls: type[Any] | None = None
    processor: str | None = None
    backend_plugin: str | os.PathLike[str] | None = None
    backend_plugins: Sequence[str | os.PathLike[str]] | None = None
    use_builtin_weight_remapper: bool = True
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def normalized_model_types(self) -> tuple[str, ...]:
        names: list[str] = []
        if self.model_type:
            names.append(self.model_type)
        if self.model_types:
            names.extend(self.model_types)

        normalized = tuple(dict.fromkeys(name.lower() for name in names if name))
        if not normalized:
            raise ValueError("ModelSpec requires model_type or model_types.")
        return normalized


_MODEL_SPECS: dict[str, ModelSpec] = {}
_LOADED_PLUGINS: dict[str, ModuleType] = {}


def register_model(spec: ModelSpec | None = None, **kwargs: Any) -> ModelSpec:
    """Register a ModelSpec and return it.

    Examples:
        register_model(ModelSpec(model_type="foo", backend_model_type="llama"))
        register_model(model_type="foo", backend_model_type="llama")
    """

    if spec is None:
        spec = ModelSpec(**kwargs)
    elif kwargs:
        raise TypeError("Pass either a ModelSpec or keyword arguments, not both.")

    for model_type in spec.normalized_model_types():
        previous = _MODEL_SPECS.get(model_type)
        if previous is not None and previous is not spec:
            raise ValueError(f"Duplicate ModelSpec registration for {model_type!r}.")
        _MODEL_SPECS[model_type] = spec
    return spec


def get_model_spec(model_type: str | None) -> ModelSpec | None:
    if not model_type:
        return None
    return _MODEL_SPECS.get(model_type.lower())


def registered_model_types() -> tuple[str, ...]:
    return tuple(sorted(_MODEL_SPECS))


def _split_plugin_list(value: str | None) -> list[str]:
    if not value:
        return []
    parts: list[str] = []
    for chunk in value.split(","):
        chunk = chunk.strip()
        if chunk:
            parts.append(chunk)
    return parts


def load_plugins(plugins: Sequence[str] | str | None = None) -> tuple[str, ...]:
    """Load plugin modules from INFINILM_PLUGINS and/or explicit names.

    INFINILM_PLUGINS accepts comma-separated Python module names or .py paths.
    Loading is idempotent for a process.
    """

    requested: list[str] = _split_plugin_list(os.environ.get("INFINILM_PLUGINS"))
    if isinstance(plugins, str):
        requested.extend(_split_plugin_list(plugins))
    elif plugins:
        requested.extend(str(plugin) for plugin in plugins)

    for plugin in requested:
        load_plugin(plugin)
    return tuple(_LOADED_PLUGINS)


def load_plugin(plugin: str | os.PathLike[str]) -> ModuleType:
    plugin_name = os.fspath(plugin)
    if plugin_name in _LOADED_PLUGINS:
        return _LOADED_PLUGINS[plugin_name]

    if plugin_name.endswith(".py") or Path(plugin_name).expanduser().exists():
        module = _load_plugin_file(plugin_name)
    else:
        try:
            module = importlib.import_module(plugin_name)
        except ModuleNotFoundError as exc:
            candidate = Path(*plugin_name.split(".")).with_suffix(".py")
            if exc.name == plugin_name.split(".")[0] and candidate.exists():
                module = _load_plugin_file(str(candidate))
            else:
                raise

    _LOADED_PLUGINS[plugin_name] = module
    return module


def _load_plugin_file(plugin_path: str) -> ModuleType:
    path = Path(plugin_path).expanduser().resolve()
    module_name = f"_infinilm_plugin_{abs(hash(str(path)))}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to import InfiniLM plugin from {path}.")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def adapt_config(config: Mapping[str, Any]) -> ConfigDict:
    """Return an InfiniLM-ready config for a registered model_type."""

    config_dict = dict(config)
    original_model_type = config_dict.get("model_type")
    spec = get_model_spec(str(original_model_type) if original_model_type else None)
    if spec is None:
        return config_dict

    adapted = _apply_config_adapter(config_dict, spec.config_adapter)
    adapted["_infinilm_original_model_type"] = original_model_type
    backend_plugins = _backend_plugins_for_spec(spec)
    if backend_plugins:
        adapted["_infinilm_backend_plugins"] = backend_plugins
    if spec.backend_model_type:
        backend_model_type = spec.backend_model_type.lower()
        adapted["_infinilm_backend_model_type"] = backend_model_type
        adapted["model_type"] = backend_model_type
    return adapted


def _backend_plugins_for_spec(spec: ModelSpec) -> list[str]:
    requested: list[str | os.PathLike[str]] = []
    if spec.backend_plugin:
        requested.append(spec.backend_plugin)
    if spec.backend_plugins:
        requested.extend(spec.backend_plugins)
    return [os.fspath(plugin) for plugin in requested]


def _apply_config_adapter(
    config: ConfigDict,
    adapter: ConfigAdapter | None,
) -> ConfigDict:
    if adapter is None:
        return dict(config)

    if callable(adapter):
        adapted = adapter(dict(config))
        if adapted is None:
            raise ValueError("ModelSpec.config_adapter must return a config dict.")
        return dict(adapted)

    adapted = dict(config)
    for key, value in adapter.items():
        if callable(value):
            adapted[key] = value(config)
        elif isinstance(value, str) and value.startswith("$"):
            adapted[key] = config[value[1:]]
        else:
            adapted[key] = value
    return adapted


def apply_weight_remapping(
    model_type: str | None,
    state_dict: StateDict,
    config: Mapping[str, Any] | None = None,
) -> StateDict:
    spec = get_model_spec(model_type)
    if spec is None:
        return state_dict

    result = state_dict
    if spec.weight_remapper is not None:
        result = _call_weight_remapper(spec.weight_remapper, result, config)
    for rule in spec.weight_rules:
        result = _call_weight_remapper(rule, result, config)
    return result


def _call_weight_remapper(
    remapper: WeightRemapper,
    state_dict: StateDict,
    config: Mapping[str, Any] | None,
) -> StateDict:
    try:
        signature = inspect.signature(remapper)
    except (TypeError, ValueError):
        return remapper(state_dict, config=config)

    params = signature.parameters.values()
    if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in params):
        return remapper(state_dict, config=config)

    params = signature.parameters.values()
    if any(param.name == "config" for param in params):
        return remapper(state_dict, config=config)

    required_positionals = [
        param
        for param in signature.parameters.values()
        if param.default is inspect.Parameter.empty
        and param.kind
        in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        )
    ]
    if len(required_positionals) >= 2:
        return remapper(state_dict, config)
    return remapper(state_dict)
