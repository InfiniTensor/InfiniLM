"""
KV Connector package for Prefill-Decode disaggregated inference.
"""

from infinilm.llm.kv_connector.base import (
    KVConnectorBase,
    KVConnectorMetadata,
    KVConnectorRole,
    NullKVConnector,
)

__all__ = [
    "KVConnectorBase",
    "KVConnectorMetadata",
    "KVConnectorRole",
    "NullKVConnector",
    "create_kv_connector",
]


def create_kv_connector(
    connector_type: str = "null",
    role: str = KVConnectorRole.NONE,
    **kwargs,
) -> KVConnectorBase:
    """Factory function to create KV connectors.

    Args:
        connector_type: Type of connector.
            - "null": No-op connector (standalone mode, default).
            - Future: "mooncake", "rdma", "tcp", etc.
        role: Role of the connector (none/sender/receiver/both).
        **kwargs: Additional connector-specific arguments.

    Returns:
        A KVConnectorBase instance.

    Raises:
        ValueError: If connector_type is not recognized.
    """
    if connector_type is None or connector_type == "null":
        return NullKVConnector(**kwargs)
    # ---- Future connector types can be registered here ----
    # elif connector_type == "mooncake":
    #     from infinilm.llm.kv_connector.mooncake_connector import MooncakeKVConnector
    #     return MooncakeKVConnector(role=role, **kwargs)
    else:
        raise ValueError(
            f"Unknown KV connector type: '{connector_type}'. "
            f"Supported types: ['null']"
        )
