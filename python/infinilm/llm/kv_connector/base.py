"""
KV Connector Base - Abstract interface for KV cache transfer in PD separation.

This module defines the base classes for KV cache connectors that enable
Prefill-Decode (PD) disaggregated inference.

In PD separation:
- Prefill workers compute KV caches and send them via the connector (Sender).
- Decode workers receive KV caches via the connector and run decoding (Receiver).

The connector is designed to be pluggable, allowing different transport
mechanisms (shared memory, RDMA, TCP, etc.) for KV cache transfer.

Lifecycle within a single forward pass:
    1. get_connector_metadata()   - Extract metadata from scheduler output
    2. bind_connector_metadata()  - Bind metadata for the current pass
    3. start_load_kv()            - (Receiver) Start loading KV caches
    4. [Model Forward Pass — layer by layer]
       - wait_for_layer_load()    - (Receiver) Wait for each layer's KV
       - save_kv_layer()          - (Sender)   Save each layer's KV
    5. wait_for_save()            - (Sender) Wait for all saves to complete
    6. finalize()                 - Clean up after the forward pass
"""

import logging
from abc import ABC
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from sympy import im
import enum

logger = logging.getLogger(__name__)
import infinicore

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


# @dataclass
# class KVConnectorMetadata:
#     """Metadata for KV connector operations during a forward pass.

#     This metadata is generated per scheduler step and carries information
#     needed by the KV connector to manage KV cache transfers.

#     Attributes:
#         request_ids: List of request IDs in the current batch.
#         is_prefill: Whether the current step is a prefill step.
#         kv_cache_spec: Optional specification of the KV cache layout
#             (num_layers, num_heads, head_dim, dtype, …).
#         block_tables: Optional per-request block tables for paged KV cache.
#         extra: Additional connector-specific metadata.
#     """

#     request_ids: List[str] = field(default_factory=list)
#     is_prefill: bool = False
#     kv_cache_spec: Optional[Dict[str, Any]] = None
#     block_tables: Optional[Any] = None
#     extra: Dict[str, Any] = field(default_factory=dict)


class KVConnectorMetadata(ABC):  # noqa: B024
    """
    Abstract Metadata used to communicate
    Scheduler KVConnector -> Worker KVConnector.
    """

    pass


# ---------------------------------------------------------------------------
# Role constants
# ---------------------------------------------------------------------------


class KVConnectorRole(enum.Enum):
    NONE = -1  # 测试使用，会删除

    # Connector running in the scheduler process
    SCHEDULER = 0

    # Connector running in the worker process
    WORKER = 1


# ---------------------------------------------------------------------------
# Abstract base class
# ---------------------------------------------------------------------------


class KVConnectorBase(ABC):
    """Abstract base class for KV cache connectors.

    Subclass this to implement a concrete KV transfer mechanism.
    """

    def __init__(self, role: str = KVConnectorRole.NONE, **kwargs):
        """Initialize the KV connector.

        Args:
            role: The role of this connector (none/sender/receiver/both).
        """
        self.role = role

    # ------ metadata -------------------------------------------------------

    def bind_connector_metadata(self, connector_metadata: KVConnectorMetadata) -> None:
        """Set the connector metadata from the scheduler.

        This function should be called by the model runner every time
        before the model execution. The metadata will be used for runtime
        KV cache loading and saving.

        Args:
            connector_metadata (dict): the connector metadata.
        """
        self._connector_metadata = connector_metadata

    def clear_connector_metadata(self) -> None:
        """Clear the connector metadata.

        This function should be called by the model runner every time
        after the model execution.
        """
        self._connector_metadata = None

    def _get_connector_metadata(self) -> Optional[KVConnectorMetadata]:
        """Extract connector metadata from scheduler output.

        Called before each forward pass.

        Args:
            scheduler_output: The scheduler output for the current step.

        Returns:
            KVConnectorMetadata, or None if no metadata is needed.
        """

        # Should only be called while set to valid metadata.
        assert self._connector_metadata is not None
        return self._connector_metadata

    def has_connector_metadata(self) -> bool:
        """Check whether the connector metadata is currently set.

        Returns:
            bool: True if connector metadata exists, False otherwise.
        """
        return self._connector_metadata is not None

    def register_kv_caches(self, kv_caches: dict[str, infinicore.Tensor]):
        """
        Initialize with the KV caches. Useful for pre-registering the
        KV Caches in the KVConnector (e.g. for NIXL).

        Args:
            kv_caches: dictionary of layer names, kv cache
        """
        return

    # ------ receiver (decode side) -----------------------------------------

    def start_load_kv(self, forward_context: "ForwardContext", **kwargs: Any) -> None:
        """Start loading KV caches for the current batch.

        Called *before* the model forward pass on the decode (receiver) side.
        May initiate asynchronous KV cache transfers.

        Args:
            scheduler_output: The scheduler output for the current step.
        """
        pass

    def wait_for_layer_load(self, layer_name: str) -> None:
        """Wait for a specific layer's KV cache to finish loading.

        Called during the model forward pass on the decode (receiver) side,
        *before* the attention layer at ``layer_idx`` processes its input.

        Args:
            layer_idx: The index of the transformer layer.
        """
        pass

    def save_kv_layer(
        self,
        layer_name: str,
        kv_layer: infinicore.Tensor,
        attn_metadata: "AttentionMetadata",
        **kwargs: Any,
    ) -> None:
        """Save KV cache for a specific layer.

        Called during the model forward pass on the prefill (sender) side,
        *after* the attention layer at ``layer_idx`` computes its KV cache.

        Args:
            layer_idx: The index of the transformer layer.
            kv_cache: The KV cache tensor(s) for this layer.
        """
        pass

    def wait_for_save(self) -> None:
        """Wait for all KV cache saves to complete.

        Called *after* the model forward pass on the prefill (sender) side.
        Ensures all asynchronous saves have finished before proceeding.
        """
        pass

    # ------ lifecycle -------------------------------------------------------

    def finalize(self, scheduler_output: Any = None) -> None:
        """Finalize connector operations after a forward pass.

        Called after the forward pass to clean up temporary resources or
        notify remote workers that the KV caches are ready.

        Args:
            scheduler_output: The scheduler output for the current step.
        """
        pass

    def close(self) -> None:
        """Close the connector and release all resources."""
        pass

    # ------ helpers ---------------------------------------------------------

    @property
    def is_sender(self) -> bool:
        return self.role in (KVConnectorRole.SENDER, KVConnectorRole.BOTH)

    @property
    def is_receiver(self) -> bool:
        return self.role in (KVConnectorRole.RECEIVER, KVConnectorRole.BOTH)


# ---------------------------------------------------------------------------
# Null (no-op) connector
# ---------------------------------------------------------------------------


class NullKVConnector(KVConnectorBase):
    """No-op KV connector that preserves existing standalone behaviour.

    All methods are no-ops, making the inference path exactly equivalent
    to the original non-PD code path.  This is the default connector.
    """

    def __init__(self, **kwargs):
        super().__init__(role=KVConnectorRole.NONE, **kwargs)
