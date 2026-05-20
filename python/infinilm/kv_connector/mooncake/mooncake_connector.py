import logging
from dataclasses import dataclass
from collections import defaultdict
from typing import Any, Optional

from infinilm.kv_connector import (
    KVConnectorBase,
    KVConnectorMetadata,
    KVConnectorRole,
)
from infinilm.config.kv_transfer import KVTransferConfig
from infinilm.llm import InferenceRequest

logger = logging.getLogger(__name__)

ReqId = str
TransferId = str
EngineId = str


@dataclass
class PullReqMeta:
    d_req_id: ReqId
    transfer_ids: TransferId
    local_block_ids: list[int]
    remote_engine_id: str
    remote_bootstrap_addr: str
    expire_time: float = float("inf")
    pull_tasks_count: int = 0


class MooncakeConnectorMetadata(KVConnectorMetadata):
    def __init__(self):
        self.reqs_to_recv: dict[EngineId, dict[ReqId, PullReqMeta]] = defaultdict(dict)
        self.reqs_to_send: dict[ReqId, tuple[TransferId, list[int]]] = {}
        self.reqs_not_processed: set[TransferId] = set()

    def add_new_req(
        self,
        request_id: str,
        local_block_ids: list[int],
        kv_transfer_params: dict[str, str],
        load_remote_cache: bool = True,
    ):
        transfer_id = kv_transfer_params["transfer_id"]
        if load_remote_cache:
            remote_engine_id = kv_transfer_params["remote_engine_id"]
            self.reqs_to_recv[remote_engine_id][request_id] = PullReqMeta(
                d_req_id=request_id,
                local_block_ids=local_block_ids,
                remote_engine_id=remote_engine_id,
                remote_bootstrap_addr=kv_transfer_params["remote_bootstrap_addr"],
                transfer_ids=transfer_id,
            )
        else:
            self.reqs_to_send[request_id] = (transfer_id, local_block_ids)


class MooncakeConnector(KVConnectorBase):
    def __init__(
        self,
        role: KVConnectorRole,
        kv_transfer_config: KVTransferConfig | None = None,
    ):
        cfg = kv_transfer_config or KVTransferConfig()
        super().__init__(
            role=role,
            kv_transfer_config=cfg,
        )

        self.engine_id: EngineId | None = cfg.engine_id

        if role == KVConnectorRole.SCHEDULER:
            from infinilm.kv_connector.mooncake.mooncake_connector_scheduler import (
                MooncakeConnectorScheduler,
            )

            self.connector_scheduler: MooncakeConnectorScheduler | None = (
                MooncakeConnectorScheduler(cfg, engine_id=self.engine_id)
            )
            self.connector_worker = None
        else:
            self.connector_scheduler = None
            self.connector_worker = None

    def get_num_new_matched_tokens(
        self, request: InferenceRequest, num_computed_tokens: int
    ) -> tuple[int, bool]:
        assert self.connector_scheduler is not None
        return self.connector_scheduler.get_num_new_matched_tokens(
            request, num_computed_tokens
        )

    def update_state_after_alloc(
        self,
        request: InferenceRequest,
        block_ids: list[int],
        num_external_tokens: int,
        block_size: Optional[int] = None,
    ) -> None:
        assert self.connector_scheduler is not None
        return self.connector_scheduler.update_state_after_alloc(
            request, block_ids, num_external_tokens, block_size
        )

    def build_connector_meta(
        self,
    ) -> KVConnectorMetadata | None:
        assert self.connector_scheduler is not None
        return self.connector_scheduler.build_connector_meta()

    def request_finished(
        self,
        request: InferenceRequest,
        block_ids: list[int],
        block_size: Optional[int] = None,
    ) -> tuple[bool, dict[str, Any] | None]:
        assert self.connector_scheduler is not None
        return self.connector_scheduler.request_finished(request, block_ids, block_size)
