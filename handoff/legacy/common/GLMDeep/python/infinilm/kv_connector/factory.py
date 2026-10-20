# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright 2026 InfiniLM Contributors

"""
KVConnectorFactory — lazy-loading registry for KV connectors.
"""

import importlib
import logging
from collections.abc import Callable

from infinilm.kv_connector.base import KVConnectorBase, KVConnectorRole
from infinilm.config.kv_transfer import KVTransferConfig

logger = logging.getLogger(__name__)


class KVConnectorFactory:
    """Registry and factory for KV connectors (lazy-loaded)."""

    _registry: dict[str, Callable[..., KVConnectorBase]] = {}

    @classmethod
    def register_connector(
        cls,
        name: str,
        module_path: str,
        class_name: str,
    ) -> None:
        """Register a KV connector backend."""
        if name in cls._registry:
            logger.warning(f"KVConnector '{name}' already registered, overwriting")

        def _lazy_loader(**kwargs) -> KVConnectorBase:
            module = importlib.import_module(module_path)
            connector_cls = getattr(module, class_name)
            return connector_cls(**kwargs)

        cls._registry[name] = _lazy_loader
        logger.debug(f"Registered KVConnector: {name} -> {module_path}.{class_name}")

    @classmethod
    def create_connector(
        cls,
        connector_name: str,
        role: KVConnectorRole,
        kv_transfer_config: KVTransferConfig | None = None,
    ) -> KVConnectorBase:
        """Create a registered connector."""
        if connector_name not in cls._registry:
            raise ValueError(f"Unknown KVConnector: '{connector_name}'.")

        cfg = kv_transfer_config or KVTransferConfig()
        loader = cls._registry[connector_name]
        connector = loader(
            role=role,
            kv_transfer_config=cfg,
        )
        logger.info(
            f"Created KVConnector: {connector_name} "
            f"(role={role.name}, kv_role={cfg.kv_role})"
        )
        return connector

    @classmethod
    def get_available_connectors(cls) -> list:
        return list(cls._registry.keys())
