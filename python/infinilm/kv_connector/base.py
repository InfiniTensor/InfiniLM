# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright 2026 InfiniLM Contributors

"""
KVConnector abstract base class.

All KV transfer backends (e.g. LMCache, Mooncake, NIXL) must subclass this.
The scheduler invokes connector hook points; each concrete implementation
determines the transfer behaviour.
"""

import enum
import logging

from abc import ABC, abstractmethod
from typing import Any, Optional

from infinilm.llm.request import InferenceRequest
from infinilm.config.kv_transfer import KVTransferConfig
import infinicore

logger = logging.getLogger(__name__)


class KVConnectorRole(enum.Enum):
    SCHEDULER = 0
    WORKER = 1


class KVConnectorHandshakeMetadata:
    """
    Metadata used for out of band connector handshake between
    P/D workers. This needs to serializable.
    """


class KVConnectorMetadata:
    """
    Abstract Metadata used to communicate
    Scheduler KVConnector -> Worker KVConnector.
    """


class KVConnectorWorkerMetadata(ABC):
    """
    Abstract Metadata used to communicate back
    Worker KVConnector -> Scheduler KVConnector.

    Each worker can output its own metadata.
    For a single engine step, all metadata objects returned by workers
    will be aggregated using the `aggregate` method below, before
    being passed to the Scheduler KVConnector.
    """

    @abstractmethod
    def aggregate(
        self, other: "KVConnectorWorkerMetadata"
    ) -> "KVConnectorWorkerMetadata":
        """
        Aggregate metadata with another `KVConnectorWorkerMetadata` object.
        """
        pass


class KVConnectorBase(ABC):
    """
    Base class for KV connectors.
    """

    @property
    def prefer_cross_layer_blocks(self) -> bool:
        """
        Indicates whether this connector prefers KV blocks that hold KV data for all
        layers, which can speed up KV data transfers. Defaults to False.
        """
        return False

    def __init__(
        self,
        role: KVConnectorRole,
        kv_transfer_config: KVTransferConfig | None = None,
    ):
        """
        Args:
            role: The role of the connector, either SCHEDULER or WORKER.
            kv_transfer_config: KV transfer configuration containing kv_role and connector_config.
        """
        self._connector_metadata: KVConnectorMetadata | None = None
        cfg = kv_transfer_config or KVTransferConfig()
        self._role = role
        self._kv_transfer_config = cfg

    @property
    def role(self) -> KVConnectorRole:
        return self._role

    @property
    def kv_transfer_config(self) -> KVTransferConfig:
        """PD/KV transfer options (not EngineConfig—use this name, not ``.config``)."""
        return self._kv_transfer_config

    # ==============================
    # Scheduler-side methods
    # ==============================

    @abstractmethod
    def get_num_new_matched_tokens(
        self,
        request: InferenceRequest,
        num_computed_tokens: int,
    ) -> tuple[int | None, bool]:
        """
        Get number of new tokens that can be loaded from the
        external KV cache beyond the num_computed_tokens.

        Args:
            request (InferenceRequest): the request object.
            num_computed_tokens (int): the number of locally
                computed tokens for this request

        Returns:
            A tuple with the following elements:
                - An optional number of tokens that can be loaded from the
                  external KV cache beyond what is already computed.
                  If None, it means that the connector needs more time to
                  determine the number of matched tokens, and the scheduler
                  should query for this request again later.
                - `True` if external KV cache tokens will be loaded
                  asynchronously (between scheduler steps). Must be
                  'False' if the first element is 0.

        Notes:
            The connector should only consider the largest prefix of prompt-
            tokens for which KV cache is actually available at the time of the
            call. If the cache cannot be loaded for some tokens (e.g., due to
            connectivity issues or eviction), those tokens must not be taken
            into account.
        """
        pass

    @abstractmethod
    def update_state_after_alloc(
        self,
        request: InferenceRequest,
        block_ids: list[int],
        num_external_tokens: int,
        block_size: Optional[int] = None,
    ) -> None:
        """
        Update KVConnector state after block allocation.

        If get_num_new_matched_tokens previously returned True for a
        request, this function may be called twice for that same request -
        first when blocks are allocated for the connector tokens to be
        asynchronously loaded into, and second when any additional blocks
        are allocated, after the load/transfer is complete.

        Args:
            request (InferenceRequest): the request object.
            block_ids (list[int]): the block IDs allocated for the request.
            num_external_tokens (int): the number of tokens that will be
                loaded from the external KV cache.
            block_size (Optional[int]): the size of each block. This is used
                to calculate the number of blocks needed for the external tokens.
        """
        pass

    @abstractmethod
    def build_connector_meta(self) -> KVConnectorMetadata:
        """Build the connector metadata for this step."""
        pass

    @abstractmethod
    def request_finished(
        self,
        request: InferenceRequest,
        block_ids: list[int],
        block_size: Optional[int] = None,
    ) -> tuple[bool, dict[str, Any] | None]:
        """
        Called exactly once when a request has finished, before its blocks are
        freed.

        The connector may assumes responsibility for freeing the blocks
        asynchronously by returning True.

        Returns:
            True if the request is being saved/sent asynchronously and blocks
            should not be freed until the request_id is returned from
            get_finished().
            Optional KVTransferParams to be included in the request outputs
            returned by the engine.
        """
        pass

    # ==============================
    # Worker-side methods
    # ==============================

    def bind_connector_metadata(self, connector_metadata: KVConnectorMetadata) -> None:
        """Set the connector metadata from the scheduler.

        This function should be called by the model runner every time
        before the model execution.

        Args:
            connector_metadata: the connector metadata.
        """
        self._connector_metadata = connector_metadata

    def clear_connector_metadata(self) -> None:
        """Clear the connector metadata.

        This function should be called by the model runner every time
        after the model execution.
        """
        self._connector_metadata = None

    @abstractmethod
    def register_kv_caches(self, kv_caches: dict[str, infinicore.Tensor]) -> None:
        """Register  KV cache tensors of the connector.

        Args:
            kv_caches: Mapping from layer name to KV cache tensor.
        """
        pass

    @abstractmethod
    def start_load_kv(self, **kwargs: Any) -> None:
        """
        Start loading KV cache from the connector to the KV buffer.
        This is called before the forward pass to enable async loading
        during model execution.

        Args:
            **kwargs: additional arguments for the load operation
        """
        pass

    def get_block_ids_with_load_errors(self) -> set[int]:
        """
        Get the set of block IDs that failed to load.

        """
        return set()

    def get_kv_connector_stats(self):
        """
        Get the KV connector stats collected during the last interval.
        """
        return None

    def get_finished(
        self, finished_req_ids: set[str]
    ) -> tuple[set[str] | None, set[str] | None, set[str] | None]:
        return None, None, None

    def shutdown(self):
        """
        Shutdown the connector.
        """
        return None
