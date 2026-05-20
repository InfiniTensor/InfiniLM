# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright 2026 InfiniLM Contributors

import uuid
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class KVTransferConfig:
    """Configuration for KV cache transfer in prefill/decode (P/D) separation.

    Attributes:
        kv_connector: Name of the KV connector to use (e.g. 'MooncakeConnector').
                      None disables KV transfer.
        kv_role: Role of this node: 'kv_producer' (prefill) or 'kv_consumer' (decode).
        engine_id: Unique identifier for this engine instance used in KV transfers.
                   Auto-generated (UUID) if not provided.
        connector_config: Extra configuration dict passed to the connector backend.
    """

    kv_connector: Optional[str] = None
    kv_role: Optional[str] = None
    engine_id: Optional[str] = None
    connector_config: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.engine_id is None:
            self.engine_id = str(uuid.uuid4())
