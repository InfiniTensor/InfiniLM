"""
KV connector package.

This module:
- Exposes core KV connector abstractions (base, role, metadata)
- Provides the KVConnectorFactory
- Registers built-in connectors (e.g. MooncakeConnector)

Note:
Importing this module will trigger connector registration.
"""

from infinilm.kv_connector.base import (
    KVConnectorBase,
    KVConnectorRole,
    KVConnectorMetadata,
    KVConnectorHandshakeMetadata,
    KVConnectorWorkerMetadata,
)
from infinilm.kv_connector.factory import KVConnectorFactory

KVConnectorFactory.register_connector(
    "MooncakeConnector",
    "infinilm.kv_connector.mooncake.mooncake_connector",
    "MooncakeConnector",
)


__all__ = [
    "KVConnectorBase",
    "KVConnectorRole",
    "KVConnectorMetadata",
    "KVConnectorHandshakeMetadata",
    "KVConnectorWorkerMetadata",
    "KVConnectorFactory",
]
