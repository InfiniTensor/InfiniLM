from infinilm.llm.kv_connector.base import KVConnectorBase, KVConnectorRole

from typing import TYPE_CHECKING, Any, Optional
from mooncake.mooncake_connector_v1 import MooncakeConnectorMetadata
import infinicore


class MooncakeConnector(KVConnectorBase):
    def __init__(self, vllm_config: "VllmConfig", role: KVConnectorRole):
        super().__init__(role)
        # TODO: 一些初始化代码
        # 为了能够初始化，应该需要修infinilm中的代码调整一下config参数
        pass
        if role == KVConnectorRole.SCHEDULER:
            # TODO: MooncakeConnectorScheduler 类未实现
            self.connector_scheduler = "MooncakeConnectorScheduler"  # TODO: 后续修改为 class MooncakeConnectorScheduler的对象
            self.connector_worker = None
        elif role == KVConnectorRole.WORKER:
            # TODO: MooncakeConnectorWorker 类未实现
            self.connector_scheduler = None
            self.connector_worker = "MooncakeConnectorWorker"  # TODO: 后续修改为 class MooncakeConnectorScheduler的对象
        pass

    ############################################################
    # Scheduler Side Methods
    ############################################################
    def get_num_new_matched_tokens(
        self, request: "Request", num_computed_tokens: int
    ) -> tuple[int, bool]:
        # TODO：
        # request的数据类型为 class Request. # /vllm/v1/request.py
        # 相当于 InfiniLm中是InferRequest类。与服务侧再确认。
        assert self.connector_scheduler is not None
        return self.connector_scheduler.get_num_new_matched_tokens(
            request, num_computed_tokens
        )

    def update_state_after_alloc(
        self, request: "Request", blocks: "KVCacheBlocks", num_external_tokens: int
    ):
        # TODO：
        # blocks的数据类型为 class KVCacheBlocks. # vllm/v1/core/kv_cache_manager.py 中
        # 与服务侧再确认这个类
        assert self.connector_scheduler is not None
        return self.connector_scheduler.update_state_after_alloc(
            request, blocks, num_external_tokens
        )

    def build_connector_meta(
        self,
        scheduler_output: "SchedulerOutput",
    ) -> MooncakeConnectorMetadata:
        # TODO：
        # scheduler_output的数据类型为 class SchedulerOutput. # infinilm/llm/scheduler.py 中
        # scheduler_output虽然需要传递给了 build_connector_meta(), 但是build_connector_meta中没有使用
        assert self.connector_scheduler is not None
        return self.connector_scheduler.build_connector_meta(scheduler_output)

    def request_finished(
        self,
        request: "Request",
        block_ids: list[int],
    ) -> tuple[bool, Optional[dict[str, Any]]]:
        # TODO
        assert self.connector_scheduler is not None
        return self.connector_scheduler.request_finished(request, block_ids)

    ############################################################
    # Worker Side Methods
    ############################################################
    def register_kv_caches(self, kv_caches: dict[str, infinicore.Tensor]):
        # TODO:
        # kv_caches的创建在C++里面。通过函数拿到了python端，数据类型是 infinicore.Tensor
        assert self.connector_worker is not None
        self.connector_worker.register_kv_caches(kv_caches)

    def get_finished(
        self, finished_req_ids: set[str]
    ) -> tuple[Optional[set[str]], Optional[set[str]]]:
        """Get the finished recving and sending requests."""

        # TODO: finished_req_ids的参数没有被用到
        assert self.connector_worker is not None
        return self.connector_worker.get_finished()

    def start_load_kv(self, forward_context: "ForwardContext", **kwargs) -> None:
        # TODO:
        # forward_context虽然出现在 start_load_kv函数的参数中，但实现中未被使用。
        assert self.connector_worker is not None
        assert isinstance(self._connector_metadata, MooncakeConnectorMetadata)
        self.connector_worker.start_load_kv(self._connector_metadata)

    def wait_for_layer_load(self, layer_name: str) -> None:
        """MooncakeConnector does not do layerwise saving."""
        # TODO: 应该是无操作
        pass

    def save_kv_layer(
        self,
        layer_name: str,
        kv_layer: infinicore.Tensor,
        attn_metadata: "AttentionMetadata",
        **kwargs,
    ) -> None:
        """MooncakeConnector does not save explicitly."""
        # TODO: 应该是无操作
        pass

    def wait_for_save(self):
        # TODO: 应该是无操作

        pass
