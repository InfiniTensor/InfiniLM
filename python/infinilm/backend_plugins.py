from __future__ import annotations

import os
from collections.abc import Sequence


def _split_plugin_list(value: str | None) -> list[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def _backend_module():
    from infinilm.lib import _infinilm

    return _infinilm


def load_backend_plugin(plugin: str | os.PathLike[str]) -> None:
    """Load one InfiniLM C++ backend plugin shared object."""

    _backend_module().load_backend_plugin(os.fspath(plugin))


def load_backend_plugins(plugins: Sequence[str | os.PathLike[str]] | str | None = None) -> tuple[str, ...]:
    """Load explicitly requested InfiniLM C++ backend plugins."""

    requested: list[str] = []
    if isinstance(plugins, (str, os.PathLike)):
        requested.extend(_split_plugin_list(os.fspath(plugins)))
    elif plugins:
        requested.extend(os.fspath(plugin) for plugin in plugins)

    for plugin in requested:
        load_backend_plugin(plugin)
    return loaded_backend_plugins()


def load_backend_plugins_from_env() -> tuple[str, ...]:
    """Load backend plugins from `INFINILM_BACKEND_PLUGINS`.

    This is an explicit compatibility helper for command-line or embedding
    workflows. Core config/model factories do not read environment variables
    implicitly.
    """

    return load_backend_plugins(os.environ.get("INFINILM_BACKEND_PLUGINS"))


def loaded_backend_plugins() -> tuple[str, ...]:
    """Return paths of C++ backend plugins already loaded in this process."""

    return tuple(_backend_module().loaded_backend_plugins())
