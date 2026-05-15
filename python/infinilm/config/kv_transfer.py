# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright 2026 InfiniLM Contributors

import uuid
from dataclasses import dataclass, field
from typing import Optional
import os

KV_ROLE_CHOICES = frozenset({"kv_producer", "kv_consumer"})


@dataclass
class KVTransferConfig:
    """Configuration for KV cache transfer in prefill/decode (P/D) separation.

    Attributes:
        kv_connector: Name of the KV connector to use (e.g. 'MooncakeConnector').
                      None disables KV transfer.
        kv_role: Role of this node: 'kv_producer' (prefill) or 'kv_consumer' (decode).
        engine_id: Unique identifier for this engine instance used in KV transfers.
                   Auto-generated (UUID) if not provided.
        kv_connector_extra_config: Extra configuration dict passed to the connector backend.
    """

    kv_connector: Optional[str] = None
    kv_role: Optional[str] = None
    engine_id: Optional[str] = None
    kv_connector_extra_config: Optional[dict] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.kv_connector_extra_config is None:
            self.kv_connector_extra_config = {"mooncake_protocol": "rdma"}

        if self.kv_role is not None and self.kv_role not in KV_ROLE_CHOICES:
            raise ValueError(
                f"Unsupported kv_role: {self.kv_role!r}. "
                f"Supported roles are {sorted(KV_ROLE_CHOICES)}"
            )

        if self.kv_connector is not None and self.kv_role is None:
            raise ValueError("Please specify kv_role when kv_connector is set.")

        if self.engine_id is None:
            self.engine_id = f"{self.kv_role}_" + str(uuid.uuid4())

        # TODO : force use tcp for mooncake
        os.environ["MC_FORCE_TCP"] = "true"
