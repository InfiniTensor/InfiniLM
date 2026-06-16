try:
    from mooncake.engine import TransferEngine
except ImportError as e:
    raise ImportError("Please pip install mooncake-transfer-engine") from e

import asyncio
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import httpx
import infinicore
import numpy as np
import msgspec
import zmq
import zmq.asyncio
from dataclasses import dataclass

from infinilm.kv_connector.mooncake.mooncake_connector import (
    MooncakeConnectorMetadata,
    PullReqMeta,
)

from infinilm.kv_connector.mooncake.mooncake_utils import (
    get_ip,
    get_mooncake_bootstrap_addr,
    make_zmq_path,
    make_zmq_socket,
    MooncakeBootstrapServer,
    RegisterWorkerPayload,
    should_launch_bootstrap_server,
)
from enum import IntEnum
from typing import Any

logger = logging.getLogger(__name__)


EngineId = str
ReqId = str
TransferId = str


class MooncakeXferResponseStatus(IntEnum):
    # Transfer finished
    FINISH = 0
    # Continue to receive
    CONTINUE = 1
    # Something wrong, see err_msg
    ERROR = 2


class MooncakeXferReqStatus(IntEnum):
    SUCCESS = 0  # normal
    TIMEOUT = 1  # P node task timeout
    ADDR_MISMATCH = 2  # address calculation failed before sending
    XFER_FAIL = 3  # mooncake write data failed


class MooncakeXferResponse(
    msgspec.Struct,
    omit_defaults=True,  # type: ignore[call-arg]
):
    status: MooncakeXferResponseStatus
    reqs_ids: list[ReqId] | None = None
    reqs_statues: list[MooncakeXferReqStatus] | None = None
    msg: str | None = None


class MooncakeXferMetadata(
    msgspec.Struct,
    omit_defaults=True,  # type: ignore[call-arg]
):
    remote_hostname: str
    remote_port: int
    remote_tp_size: int
    remote_tp_rank: int
    req_blocks: dict[ReqId, tuple[TransferId, list[int]]]
    kv_caches_base_addr: list[int]
    kv_flag_addr: list[int]


@dataclass
class SendBlockMeta:
    p_req_id: ReqId
    transfer_id: TransferId
    local_block_ids: list[int]
    ready: asyncio.Event
    expire_time: float = float("inf")
    need_send: int = 0
    sent: int = 0
    sending: int = 0


class MooncakeAgentMetadata(
    msgspec.Struct,
    omit_defaults=True,  # type: ignore[call-arg]
    # required for @cached_property.
    dict=True,
):
    remote_hostname: str
    remote_port: int
    request_ids: list[ReqId]
    kv_caches_base_addr: list[int]
    block_ids: list[list[int]]


@dataclass
class MooncakeParallelConfig:
    """Configuration for Mooncake distributed execution."""

    tensor_parallel_size: int = 1
    tensor_parallel_rank: int = 0
    world_size: int = 1
    rank: int = 0


class MooncakeConnectorWorker:
    def __init__(self, kv_transfer_config, engine_id: str) -> None:
        assert kv_transfer_config is not None
        assert engine_id is not None
        logger.info("Initializing MooncakeConnector worker %s", engine_id)

        self.parallel_config = MooncakeParallelConfig()
        self.is_kv_producer: bool = kv_transfer_config.kv_role == "kv_producer"
        self.is_kv_consumer: bool = kv_transfer_config.kv_role == "kv_consumer"

        self.num_sender_workers = kv_transfer_config.kv_connector_extra_config.get(
            "num_workers", 10
        )
        # Create more tasks than workers to keep the thread pool saturated.
        self.num_sender_tasks = self.num_sender_workers * 2
        protocol = kv_transfer_config.kv_connector_extra_config.get(
            "mooncake_protocol", "rdma"
        )
        logger.info(
            "The Mooncake Transfer Engine is using %s as its protocol.", protocol
        )

        self.engine = TransferEngine()
        self.hostname = get_ip()
        ret_value = self.engine.initialize(self.hostname, "P2PHANDSHAKE", protocol, "")
        if ret_value != 0:
            raise RuntimeError("Mooncake Transfer Engine initialization failed.")

        self.rpc_port = self.engine.get_rpc_port()

        logger.debug(
            "Mooncake Transfer Engine initialized at %s:%d",
            self.hostname,
            self.rpc_port,
        )

        self._remote_agents: dict[EngineId, dict[int, dict[int, str]]] = {}
        self._pending_bootstrap_queries: dict[str, asyncio.Event] = {}
        self.side_channel_port: int = 0  # we will bind it in register_kv_caches()
        self.engine_id: EngineId = engine_id

        self.tp_rank = self.parallel_config.tensor_parallel_rank
        self.tp_size = self.parallel_config.tensor_parallel_size

        self.num_blocks = 0
        self.kv_caches_base_addr: list[int] = []
        self.device_kv_caches: dict[str, infinicore.Tensor] = {}

        self.async_zmq_ctx = zmq.asyncio.Context()
        self._encoder = msgspec.msgpack.Encoder()
        self._xfer_meta_decoder = msgspec.msgpack.Decoder(MooncakeXferMetadata)
        self._xfer_resp_decoder = msgspec.msgpack.Decoder(MooncakeXferResponse)

        if not self.is_kv_consumer:
            # Background threads for sending kvcaches to D.
            self._sender_executor = ThreadPoolExecutor(
                max_workers=self.num_sender_workers,
                thread_name_prefix="infinilm-mooncake-sender",
            )
            logger.debug(
                "Mooncake Prefiller: use %d workers to send kvcaches",
                self.num_sender_workers,
            )
            # An asyncio queue to buffer incoming requests for the sender
            self.sender_worker_queue = asyncio.Queue[tuple[bytes, bytes]]()
            self.sender_loop = asyncio.new_event_loop()
            # Background thread for processing new sending requests.
            self._sender_listener_t = threading.Thread(
                target=self._async_loop, args=(self.sender_loop,), daemon=True
            )
            self._sender_listener_t.start()

            # Start bootstrap server on global rank 0.
            if should_launch_bootstrap_server(self.parallel_config):
                bootstrap_host, bootstrap_port = get_mooncake_bootstrap_addr(
                    self.parallel_config
                )
                self.bootstrap_server = MooncakeBootstrapServer(
                    bootstrap_host, bootstrap_port
                )
                self.bootstrap_server.start()

        if not self.is_kv_producer:
            self.receiver_loop = asyncio.new_event_loop()
            self._mooncake_receiver_t = threading.Thread(
                target=self._async_loop, args=(self.receiver_loop,), daemon=True
            )
            self._mooncake_receiver_t.start()
            logger.debug("Mooncake Decoder: start receiver thread")

        if self.is_kv_producer:
            self.reqs_need_send: dict[TransferId, SendBlockMeta] = {}
            self.finished_sending_reqs: set[ReqId] = set()

            self.reqs_need_send_timeout: list[TransferId] = []

        if self.is_kv_consumer:
            self.finished_recving_reqs: set[ReqId] = set()
            self._tp_size: dict[EngineId, int] = {self.engine_id: self.tp_size}

            # collect timeout reqs to recv
            self.timeout_reqs_to_recv: dict[EngineId, dict[ReqId, PullReqMeta]] = {}
            # collect xfer failed reqs id
            self.xfer_failed_recving_reqs_ids: set[ReqId] = set()

    def _async_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        asyncio.set_event_loop(loop)
        loop.run_forever()

    def __del__(self):
        self.shutdown()

    def shutdown(self) -> None:
        """Stop ZMQ / threads / bootstrap (idempotent from __del__)."""
        self.async_zmq_ctx.term()
        if not self.is_kv_consumer:
            self._sender_executor.shutdown(wait=False)
            if self.sender_loop.is_running():
                self.sender_loop.call_soon_threadsafe(self.sender_loop.stop)
                self._sender_listener_t.join()
            if (
                should_launch_bootstrap_server(self.parallel_config)
                and getattr(self, "bootstrap_server", None) is not None
            ):
                self.bootstrap_server.shutdown()
        if not self.is_kv_producer and self.receiver_loop.is_running():
            self.receiver_loop.call_soon_threadsafe(self.receiver_loop.stop)
            self._mooncake_receiver_t.join()

    async def _mooncake_sender_listener(self, ready_event: threading.Event):
        """
        Background thread that listens for Mooncake requests, dispatches them
        to a thread pool, and sends acknowledgments upon completion.
        """

        sock = self.async_zmq_ctx.socket(zmq.ROUTER)
        self.side_channel_port = sock.bind_to_random_port(f"tcp://{self.hostname}")
        logger.debug(
            "Mooncake sender starting listening on path: tcp://%s:%d",
            self.hostname,
            self.side_channel_port,
        )

        await self.register_worker_with_bootstrap()

        # Create async worker tasks that process items from the queue
        sender_tasks = [
            asyncio.create_task(self._sender_worker(sock))
            for _ in range(self.num_sender_tasks)
        ]

        ready_event.set()

        try:
            while True:
                identity, metadata_bytes = await sock.recv_multipart()
                logger.debug("ZMQ recv one msg, identity: %s.", identity)

                await self.sender_worker_queue.put((identity, metadata_bytes))
        except zmq.ContextTerminated:
            logger.debug("ZMQ context terminated, exiting Mooncake sender thread.")
        except Exception as e:
            logger.error("Error in Mooncake sender thread: %s. Exiting thread.", str(e))
        finally:
            # Clean up worker tasks
            for task in sender_tasks:
                task.cancel()
            await asyncio.gather(*sender_tasks, return_exceptions=True)
            sock.close()

    async def register_worker_with_bootstrap(self):
        host, port = get_mooncake_bootstrap_addr(self.parallel_config)
        logger.debug(
            "Mooncake sender register_worker_with_bootstrap start! host=%s port=%s",
            host,
            port,
        )

        url = make_zmq_path("http", host, port) + "/register"
        worker_addr = make_zmq_path("tcp", self.hostname, self.side_channel_port)
        payload = RegisterWorkerPayload(
            engine_id=self.engine_id,
            dp_rank=0,
            tp_rank=self.tp_rank,
            pp_rank=0,
            addr=worker_addr,
        )
        while True:
            try:
                async with httpx.AsyncClient() as client:
                    response = await client.post(url, json=payload.model_dump())
                    response.raise_for_status()
                logger.debug("Successfully registered with bootstrap server at %s", url)
                break
            except httpx.ConnectError:
                # Bootstrap server not ready, wait for a while and retry.
                logger.debug("Bootstrap server not ready, wait for a while and retry.")
                await asyncio.sleep(1)
            except Exception as e:
                err_msg = (
                    e.response.text if isinstance(e, httpx.HTTPStatusError) else str(e)
                )
                logger.error(
                    "Error registering %s with bootstrap server: %s", payload, err_msg
                )
                raise

    async def _sender_worker(self, sock: zmq.asyncio.Socket):
        while True:
            try:
                identity, metadata_bytes = await self.sender_worker_queue.get()
                try:
                    metadata = self._xfer_meta_decoder.decode(metadata_bytes)
                    await self.send_kv_to_decode(identity, sock, metadata)
                except Exception as e:
                    logger.error("Error processing Mooncake xfer request: %s", e)
                    error_response = MooncakeXferResponse(
                        status=MooncakeXferResponseStatus.ERROR, msg=str(e)
                    )
                    await sock.send_multipart(
                        (identity, self._encoder.encode(error_response))
                    )
                finally:
                    self.sender_worker_queue.task_done()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Error in _sender_worker: %s", e)

    async def send_kv_to_decode(
        self, identity: bytes, sock: zmq.asyncio.Socket, meta: MooncakeXferMetadata
    ):
        pending_reqs: dict[ReqId, SendBlockMeta] = {}
        remote_tp_ranks = [0]
        if self.tp_rank not in remote_tp_ranks:
            # This D worker does not pair with the P worker.
            raise RuntimeError(
                f"MooncakeConnectorWorker: This P tp_rank {self.tp_rank} not match with remote D target ranks {remote_tp_ranks}"
            )
        for d_req_id, (transfer_id, _) in meta.req_blocks.items():
            if transfer_id not in self.reqs_need_send:
                # This req is not enqueued in P side yet, create it here.
                self.reqs_need_send[transfer_id] = SendBlockMeta(
                    p_req_id="",
                    transfer_id=transfer_id,
                    local_block_ids=[],
                    ready=asyncio.Event(),
                )
            send_meta = self.reqs_need_send[transfer_id]
            pending_reqs[d_req_id] = send_meta

        async def wait_and_ret(
            d_req_id: ReqId, send_meta: SendBlockMeta
        ) -> tuple[ReqId, SendBlockMeta]:
            await send_meta.ready.wait()
            return d_req_id, send_meta

        wait_tasks = [
            asyncio.create_task(wait_and_ret(d_req_id, send_meta))
            for d_req_id, send_meta in pending_reqs.items()
        ]

        while wait_tasks:
            ABORT_REQUEST_TIMEOUT = 480
            done_tasks, pending_tasks = await asyncio.wait(
                wait_tasks,
                timeout=ABORT_REQUEST_TIMEOUT,
                return_when=asyncio.FIRST_COMPLETED,
            )

            if not done_tasks:
                # the tasks of wait_tasks are all timeout
                #  abort all pending requests.

                for task in pending_tasks:
                    task.cancel()

                pending_tasks_reqs_ids = list(pending_reqs.keys())
                # self.reqs_need_send_timeout.extend(pending_tasks_reqs_ids)
                logger.warning(
                    "Timeout waiting for P side ready: %s", list(pending_reqs)
                )

                response = MooncakeXferResponse(
                    status=MooncakeXferResponseStatus.FINISH,
                    reqs_ids=pending_tasks_reqs_ids,
                    reqs_statues=[MooncakeXferReqStatus.TIMEOUT]
                    * len(pending_tasks_reqs_ids),
                    msg="Timeout waiting for P side ready.",
                )

                await sock.send_multipart((identity, self._encoder.encode(response)))
                break

            wait_tasks = list(pending_tasks)
            response_status = (
                MooncakeXferResponseStatus.CONTINUE
                if wait_tasks
                else MooncakeXferResponseStatus.FINISH
            )
            ready_reqs: list[tuple[ReqId, SendBlockMeta]] = []
            for task in done_tasks:
                d_req_id, send_meta = task.result()
                del pending_reqs[d_req_id]

                if send_meta.transfer_id in self.reqs_need_send:
                    # Mark it sending to avoid expiration.
                    send_meta.sending += 1
                    if not send_meta.need_send:
                        self.resolve_need_send(send_meta, remote_tp_ranks)
                    ready_reqs.append((d_req_id, send_meta))
                else:
                    # Otherwise (expired, very unlikely), just forget it.
                    logger.warning(
                        "Request %s expired before sending on P side.", d_req_id
                    )

                    raise RuntimeError(
                        f"MooncakeConnectorWorker: Request {d_req_id} expired before sending on P side."
                    )

            (
                src_ptrs,
                dst_ptrs,
                lengths,
                mismatch_reqs_ids,
                xfer_reqs_ids,
                xfer_block_ids,
            ) = await self._build_transfer_params(ready_reqs, meta)

            if mismatch_reqs_ids:
                response = MooncakeXferResponse(
                    status=response_status,
                    reqs_ids=mismatch_reqs_ids,
                    reqs_statues=[MooncakeXferReqStatus.ADDR_MISMATCH]
                    * len(mismatch_reqs_ids),
                    msg="P num blocks less than D",
                )
                await sock.send_multipart((identity, self._encoder.encode(response)))

                raise RuntimeError(
                    f"MooncakeConnectorWorker: Address mismatch for requests {mismatch_reqs_ids}"
                )

            ret_value = 0
            if len(xfer_reqs_ids) > 0 and src_ptrs:
                remote_session = f"{meta.remote_hostname}:{meta.remote_port}"

                # wait until return value
                kv_flag_addr = meta.kv_flag_addr
                ret_value = await self.sender_loop.run_in_executor(
                    self._sender_executor,
                    self._send_blocks,
                    remote_session,
                    src_ptrs,
                    dst_ptrs,
                    lengths,
                    xfer_reqs_ids,
                    xfer_block_ids,
                    kv_flag_addr,
                )

            if ret_value != 0:
                # happen error during mooncake transfer
                xfer_failed_reqs_ids = []
                for d_req_id, send_meta in ready_reqs:
                    send_meta.sending -= 1
                    xfer_failed_reqs_ids.append(d_req_id)

                # not delete  send_meta object in self.reqs_need_send.
                # wait until D

                # Do best effort to transfer the remaining reqs.
                response = MooncakeXferResponse(
                    status=response_status,
                    reqs_ids=xfer_failed_reqs_ids,
                    reqs_statues=[MooncakeXferReqStatus.XFER_FAIL]
                    * len(xfer_failed_reqs_ids),
                    msg=f"Mooncake transfer engine returned {ret_value}",
                )
                await sock.send_multipart((identity, self._encoder.encode(response)))
            else:
                for d_req_id, send_meta in ready_reqs:
                    # TODO: for heterogeneous TP (one P pairs to multiple D),
                    # we need to check whether all headers are sent.
                    # If not, we should set expire_time to normal and skip the below.
                    send_meta.sending -= 1
                    send_meta.sent += 1
                    if send_meta.sent == send_meta.need_send:
                        del self.reqs_need_send[send_meta.transfer_id]
                        self.finished_sending_reqs.add(send_meta.p_req_id)

                response = MooncakeXferResponse(
                    status=response_status,
                    reqs_ids=[d_req_id for d_req_id, _ in ready_reqs],
                    reqs_statues=[MooncakeXferReqStatus.SUCCESS] * len(ready_reqs),
                    msg="successfully",
                )
                await sock.send_multipart((identity, self._encoder.encode(response)))

    def resolve_need_send(self, send_meta: SendBlockMeta, remote_tp_ranks: list[int]):
        # Prepare for heterogeneous TP (one P pairs to multiple D)
        send_meta.need_send = len(remote_tp_ranks)
        if send_meta.need_send != 1:
            logger.error("Mooncake: Heterogeneous TP is not supported yet.")
            raise NotImplementedError(
                "Mooncake: Heterogeneous TP is not supported yet."
            )

    async def _build_transfer_params(
        self,
        ready_reqs: list[tuple[ReqId, SendBlockMeta]],
        agent_meta: MooncakeXferMetadata,
    ) -> tuple[list[int], list[int], list[int], list[ReqId], list[ReqId], list[int]]:
        local_base_addr = self.kv_caches_base_addr
        remote_base_addr = agent_meta.kv_caches_base_addr
        block_len = self.block_len
        remote_session = f"{agent_meta.remote_hostname}:{agent_meta.remote_port}"

        src_ptrs = []
        dst_ptrs = []
        lengths = []

        mismatch_reqs_ids: list[ReqId] = []
        xfer_reqs_ids: list[ReqId] = []
        xfer_block_ids: list[int] = []

        for d_req_id, send_meta in ready_reqs:
            _, remote_block_ids = agent_meta.req_blocks[d_req_id]
            num_remote_blocks = len(remote_block_ids)
            if num_remote_blocks == 0:
                xfer_reqs_ids.append(d_req_id)
                continue

            local_block_ids = send_meta.local_block_ids
            # Partial prefix cache hit: just read uncomputed blocks.
            num_local_blocks = len(local_block_ids)
            if num_local_blocks < num_remote_blocks:
                logger.error(
                    "req %s: local blocks(%d) less than remote blocks(%d)!",
                    d_req_id,
                    num_local_blocks,
                    num_remote_blocks,
                )
                mismatch_reqs_ids.append(d_req_id)
                continue
            if num_local_blocks > num_remote_blocks:
                local_block_ids = local_block_ids[-num_remote_blocks:]

            # Group by indices
            group_local_block_ids, group_remote_block_ids = group_concurrent_contiguous(
                local_block_ids, remote_block_ids
            )

            for local_layer_addr, remote_layer_addr in zip(
                local_base_addr, remote_base_addr
            ):
                for group_local_block_id, group_remote_block_id in zip(
                    group_local_block_ids, group_remote_block_ids
                ):
                    src_ptrs.append(
                        local_layer_addr + group_local_block_id[0] * block_len
                    )
                    dst_ptrs.append(
                        remote_layer_addr + group_remote_block_id[0] * block_len
                    )
                    lengths.append(block_len * len(group_local_block_id))

            xfer_reqs_ids.append(d_req_id)
            xfer_block_ids.extend(remote_block_ids)
            logger.debug(
                "Calculate kv_caches ptrs for request %s (%d blocks) to %s",
                d_req_id,
                num_remote_blocks,
                remote_session,
            )

        return (
            src_ptrs,
            dst_ptrs,
            lengths,
            mismatch_reqs_ids,
            xfer_reqs_ids,
            xfer_block_ids,
        )

    def _send_blocks(
        self,
        remote_session: str,
        src_ptrs: list[int],
        dst_ptrs: list[int],
        lengths: list[int],
        xfer_reqs_ids: list[ReqId],
        xfer_block_ids: list[int],
        kv_flag_addr: list[int],
    ) -> int:
        logger.debug(
            "mooncake engine batch_transfer_sync_write to %s start,xfer_reqs_ids: %s",
            remote_session,
            ", ".join(xfer_reqs_ids),
        )

        start_time = time.perf_counter()
        ret_value = self.engine.batch_transfer_sync_write(
            remote_session, src_ptrs, dst_ptrs, lengths
        )

        if ret_value == 0 and len(kv_flag_addr) > 0:
            n = len(xfer_block_ids)
            flag_src_ptrs = [self.kv_flag_src_ptrs[b] for b in xfer_block_ids]
            flag_dst_ptrs = [kv_flag_addr[b] for b in xfer_block_ids]
            flag_lengths = [1] * n

            flag_ret_value = self.engine.batch_transfer_sync_write(
                remote_session, flag_src_ptrs, flag_dst_ptrs, flag_lengths
            )

            if flag_ret_value != 0:
                ret_value = flag_ret_value

            logger.debug(
                "mooncake engine sending flag to %s done, ret_value: %s",
                remote_session,
                flag_ret_value,
            )

        if ret_value == 0:
            logger.debug(
                "mooncake engine sending to %s done, took %s, xfer_reqs_ids: %s",
                remote_session,
                time.perf_counter() - start_time,
                ", ".join(xfer_reqs_ids),
            )
        else:
            logger.debug(
                "mooncake engine sending to %s failed, ret_value: %s, xfer_reqs_ids: %s",
                remote_session,
                ret_value,
                ", ".join(xfer_reqs_ids),
            )
        return ret_value

    def register_kv_caches(self, kv_caches: dict[str, infinicore.Tensor]):
        """Register the KV Cache data in mooncake."""

        logger.info("Registering KV_Caches.")

        kv_data_ptrs = []
        kv_data_lens = []
        seen_base_addresses = []

        split_k_and_v = True
        tensor_size_bytes = None
        for layer_name, cache_or_caches in kv_caches.items():
            logger.debug(
                "registering layer %s with shape %s", layer_name, cache_or_caches.shape
            )

            assert split_k_and_v, "split_k_and_v must be True"
            cache_list = [
                cache_or_caches.narrow(0, 0, 1).squeeze(0),  # k_cache
                cache_or_caches.narrow(0, 1, 1).squeeze(0),  # v_cache
            ]

            for cache in cache_list:
                base_addr = cache.data_ptr()
                if base_addr in seen_base_addresses:
                    continue

                seen_base_addresses.append(base_addr)

                if True:
                    if cache.dtype == infinicore.bfloat16:
                        dtype_size = 2
                    else:
                        raise ValueError(f"Unsupported dtype: {cache.dtype}")

                    numel = cache.numel()
                    curr_tensor_size_bytes = numel * dtype_size

                if tensor_size_bytes is None:
                    tensor_size_bytes = curr_tensor_size_bytes
                    self.num_blocks = cache.shape[0]

                assert tensor_size_bytes == curr_tensor_size_bytes, (
                    "All kv cache tensors must have the same size"
                )

                kv_data_ptrs.append(base_addr)
                kv_data_lens.append(tensor_size_bytes)

        self.kv_caches_base_addr = seen_base_addresses

        ret_value = self.engine.batch_register_memory(kv_data_ptrs, kv_data_lens)
        if ret_value != 0:
            raise RuntimeError("Mooncake batch memory registration failed.")

        assert tensor_size_bytes is not None
        assert self.num_blocks != 0
        assert tensor_size_bytes % self.num_blocks == 0
        self.block_len = tensor_size_bytes // self.num_blocks
        self.device_kv_caches = kv_caches
        logger.info(
            "registered num_blocks=%d block_len=%d", self.num_blocks, self.block_len
        )
        # logger.info("registered kv_caches_base_addr=%d", len(self.kv_caches_base_addr))

        if self.is_kv_consumer:
            self.kv_flag = np.zeros(self.num_blocks, dtype=np.uint8)
            flag_base_ptr = self.kv_flag.ctypes.data
            self.kv_flag_ptrs = [flag_base_ptr + i for i in range(self.num_blocks)]
            self.kv_flag_lens = [1] * self.num_blocks

            ret = self.engine.batch_register_memory(
                self.kv_flag_ptrs, self.kv_flag_lens
            )

            if ret != 0:
                raise RuntimeError(
                    "Mooncake batch memory registration failed for kv block flags."
                )

        if self.is_kv_producer:
            self.kv_flag_src = np.ones(self.num_blocks, dtype=np.uint8)
            flag_src_base_ptr = self.kv_flag_src.ctypes.data
            self.kv_flag_src_ptrs = [
                flag_src_base_ptr + i for i in range(self.num_blocks)
            ]
            self.kv_flag_src_lens = [1] * self.num_blocks

            ret = self.engine.batch_register_memory(
                self.kv_flag_src_ptrs, self.kv_flag_src_lens
            )
            if ret != 0:
                raise RuntimeError(
                    "Mooncake batch memory registration failed for kv block src flag."
                )

        # No need to launch server for D node.
        if self.is_kv_consumer:
            return

        ready_event = threading.Event()
        asyncio.run_coroutine_threadsafe(
            self._mooncake_sender_listener(ready_event), self.sender_loop
        )
        ready_event.wait()  # Wait for listener ZMQ socket to be ready.

    async def fetch_finished_recving_reqs(self) -> set[ReqId]:
        finished_recving_reqs = self.finished_recving_reqs
        self.finished_recving_reqs = set()
        return finished_recving_reqs

    async def fetch_xfer_failed_recving_reqs(self) -> set[ReqId]:
        xfer_failed_recving_reqs_ids = self.xfer_failed_recving_reqs_ids
        self.xfer_failed_recving_reqs_ids = set()
        return xfer_failed_recving_reqs_ids

    async def fetch_finished_sending_reqs(self) -> set[ReqId]:
        finished_sending_reqs = self.finished_sending_reqs
        self.finished_sending_reqs = set()

        # Handle timeout to avoid stranding blocks on remote.
        now = time.perf_counter()

        for transfer_id, send_meta in self.reqs_need_send.items():
            if (
                send_meta.p_req_id
                and send_meta.expire_time < now
                and send_meta.sending == 0
            ):
                logger.warning(
                    "Request %s timed out after %d seconds without "
                    "being sent. don't freeing its blocks on the producer side.",
                    send_meta.p_req_id,
                    480,
                )

                # reset time
                send_meta.expire_time = time.perf_counter() + 480

                # TODO: mv timeout reqs to finished_sending_reqs set
                finished_sending_reqs.add(send_meta.p_req_id)

        return finished_sending_reqs

    def get_finished(self) -> tuple[set[str] | None, set[str] | None, set[str] | None]:
        """
        Get requests that are done sending or recving on this specific worker.
        The scheduler process will use this output to track which workers are done.
        """
        recv_fut = None
        failed_recv_fut = None
        send_fut = None
        if not self.is_kv_producer:
            recv_fut = asyncio.run_coroutine_threadsafe(
                self.fetch_finished_recving_reqs(), self.receiver_loop
            )
            failed_recv_fut = asyncio.run_coroutine_threadsafe(
                self.fetch_xfer_failed_recving_reqs(), self.receiver_loop
            )

        if not self.is_kv_consumer:
            send_fut = asyncio.run_coroutine_threadsafe(
                self.fetch_finished_sending_reqs(), self.sender_loop
            )

        finished_recving_reqs = recv_fut.result() if recv_fut else set()
        failed_recving_reqs = failed_recv_fut.result() if failed_recv_fut else set()
        finished_sending_reqs = send_fut.result() if send_fut else set()

        if finished_sending_reqs or finished_recving_reqs:
            logger.debug(
                "Rank %s, get_finished: %s requests done sending "
                "and %s requests done recving",
                self.tp_rank,
                len(finished_sending_reqs),
                len(finished_recving_reqs),
            )

        return (
            finished_sending_reqs or None,
            failed_recving_reqs or None,
            finished_recving_reqs or None,
        )

    async def _wait_for_kv_flags_ready(self, block_ids: list[int]) -> None:
        """Wait until Mooncake sets all kv_flag entries for the given blocks to 1."""
        if not block_ids:
            return

        indices = np.asarray(block_ids, dtype=np.intp)
        while not np.all(self.kv_flag[indices] == 1):
            await asyncio.sleep(0.012)

        self.kv_flag[indices] = 0

    async def receive_kv_from_single_worker(
        self,
        remote_engine_id,
        worker_addr: str,
        pull_metas: dict[ReqId, PullReqMeta],
    ):
        req_ids = set(pull_metas)
        metadata = MooncakeXferMetadata(
            remote_hostname=self.hostname,
            remote_port=self.rpc_port,
            remote_tp_size=self.tp_size,
            remote_tp_rank=self.tp_rank,
            req_blocks={
                req_id: (pull_meta.transfer_id, pull_meta.local_block_ids)
                for req_id, pull_meta in pull_metas.items()
            },
            kv_caches_base_addr=self.kv_caches_base_addr,
            kv_flag_addr=self.kv_flag_ptrs,
        )

        encoded_data = self._encoder.encode(metadata)
        logger.debug(
            "Size of encoded MooncakeXferMetadata: %d bytes", len(encoded_data)
        )
        logger.debug(
            "Sending kv transfer request for %s on path: %s", req_ids, worker_addr
        )

        # Send query for the request.
        try:
            with make_zmq_socket(
                self.async_zmq_ctx, worker_addr, zmq.DEALER, bind=False, linger=0
            ) as sock:
                # If something goes wrong, let P wait timeout first (in asyncio.wait()).
                sock.setsockopt(zmq.RCVTIMEO, (480 + 60) * 1000)
                await sock.send(encoded_data)

                response_list = []
                while True:
                    ret_msg = await sock.recv()
                    response = self._xfer_resp_decoder.decode(ret_msg)
                    response_list.append(response)

                    # zmq exception happens
                    if response.status == MooncakeXferResponseStatus.ERROR:
                        logger.error(
                            "Error happens during transferring kvcache for %s: %s",
                            req_ids,
                            response.msg,
                        )
                        raise RuntimeError(
                            f"MooncakeConnectorWorker: recv response is Error happens during transferring kvcache for {req_ids}: {response.msg}"
                        )

                    if response.status == MooncakeXferResponseStatus.FINISH:
                        break

                # process response list
                processed_reqs_count = 0
                success_block_ids = []
                finished_recving_reqs = set()
                for response in response_list:
                    reqs_count, finished_reqs, block_ids = self.process_pulling_result(
                        remote_engine_id, response, pull_metas
                    )
                    processed_reqs_count += reqs_count
                    success_block_ids.extend(block_ids)
                    finished_recving_reqs.update(finished_reqs)

                # TODO:check if all reqs are processed
                assert processed_reqs_count == len(pull_metas), (
                    "processed_reqs_count must be equal to the number of pull_metas"
                )

                await self._wait_for_kv_flags_ready(success_block_ids)

                self.finished_recving_reqs.update(finished_recving_reqs)

        except zmq.ContextTerminated:
            logger.debug("ZMQ context terminated, exiting Mooncake receiver thread.")
            # TODO: handle this error
        except Exception as e:
            logger.error("MooncakeXferMetadata transfer failed for %s: %s", req_ids, e)
            return

    def process_pulling_result(
        self,
        remote_engine_id: EngineId,
        response: MooncakeXferResponse,
        pull_metas: dict[ReqId, PullReqMeta],
    ):
        response_reqs_ids = response.reqs_ids or []
        response_reqs_statues = response.reqs_statues or []

        reqs_count_of_response = len(response_reqs_ids)

        assert reqs_count_of_response == len(response_reqs_statues), (
            "response_reqs_ids and response_reqs_statues must have the same count"
        )

        success_reqs_ids = []
        timeout_reqs_ids = []
        addr_mismatch_reqs_ids = []
        xfer_failed_reqs_ids = []
        for req_id, status in zip(response_reqs_ids, response_reqs_statues):
            match status:
                case MooncakeXferReqStatus.SUCCESS:
                    success_reqs_ids.append(req_id)
                case MooncakeXferReqStatus.TIMEOUT:
                    timeout_reqs_ids.append(req_id)
                case MooncakeXferReqStatus.ADDR_MISMATCH:
                    addr_mismatch_reqs_ids.append(req_id)
                case MooncakeXferReqStatus.XFER_FAIL:
                    xfer_failed_reqs_ids.append(req_id)
                case _:
                    raise ValueError(
                        f"MooncakeConnectorWorker: Invalid status {status} for request {req_id}"
                    )
        success_block_ids = []
        if len(success_reqs_ids) > 0:
            for req_id in success_reqs_ids:
                pull_meta = pull_metas[req_id]

                success_block_ids.extend(pull_meta.local_block_ids)

                # No race because we are in async loop.
                pull_meta.pull_tasks_count -= 1
                if pull_meta.pull_tasks_count == 0:
                    assert req_id == pull_meta.d_req_id
                else:
                    raise RuntimeError(
                        f"MooncakeConnectorWorker: Pull tasks count is not 0 for request {req_id}"
                    )
            logger.debug("successfully pulling kv_caches for %s", success_reqs_ids)

        if timeout_reqs_ids:
            inner = self.timeout_reqs_to_recv.setdefault(remote_engine_id, {})
            for (
                req_id
            ) in timeout_reqs_ids:  # D 侧收集超时，下一拍在 _start_load_kv 合并重试
                pull_meta = pull_metas[req_id]
                pull_meta.pull_tasks_count = 0
                inner[req_id] = pull_meta

        if len(addr_mismatch_reqs_ids) > 0:
            raise RuntimeError(
                f"MooncakeConnectorWorker: Address mismatch for requests {addr_mismatch_reqs_ids}"
            )

        if len(xfer_failed_reqs_ids) > 0:
            logger.error(
                "MooncakeConnectorWorker: pulling kv_caches for %s  failed: %s",
                xfer_failed_reqs_ids,
                response.msg,
            )
            self.xfer_failed_recving_reqs_ids.update(xfer_failed_reqs_ids)

        finished_recving_reqs = set(success_reqs_ids) | set(xfer_failed_reqs_ids)

        return reqs_count_of_response, finished_recving_reqs, success_block_ids

    async def _connect_to_prefiller_bootstrap(self, remote_bootstrap_addr: str):
        url = remote_bootstrap_addr + "/query"
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(url)
                response.raise_for_status()
                data: dict = response.json()
                for _, dp_entry in data.items():
                    remote_engine_id = dp_entry["engine_id"]
                    self._remote_agents[remote_engine_id] = {
                        int(tp_rank): {
                            int(pp_rank): worker_addr
                            for pp_rank, worker_addr in tp_entry.items()
                        }
                        for tp_rank, tp_entry in dp_entry["worker_addr"].items()
                    }
                    self._tp_size[remote_engine_id] = len(dp_entry["worker_addr"])
        except Exception as e:
            logger.error(
                "Failed to connect to bootstrap server %s: %s",
                remote_bootstrap_addr,
                e,
            )

        # Always notify others regardless of connection success or failure.
        self._pending_bootstrap_queries[remote_bootstrap_addr].set()
        del self._pending_bootstrap_queries[remote_bootstrap_addr]

    def receive_kv(
        self,
        remote_engine_id: EngineId,
        pull_metas: dict[ReqId, PullReqMeta],
    ):
        remote_tp_ranks = [0]
        count = len(remote_tp_ranks)
        if count != 1:
            logger.error("Mooncake: Heterogeneous TP is not supported yet.")
            raise NotImplementedError(
                "Mooncake: Heterogeneous TP is not supported yet."
            )
        for pull_meta in pull_metas.values():
            pull_meta.pull_tasks_count = count
        for remote_tp_rank in remote_tp_ranks:
            worker_addr = self._remote_agents[remote_engine_id][remote_tp_rank][0]
            asyncio.create_task(
                self.receive_kv_from_single_worker(
                    remote_engine_id, worker_addr, pull_metas
                )
            )

    async def handle_new_engine_id(
        self,
        remote_engine_id: EngineId,
        pull_metas: dict[ReqId, PullReqMeta],
    ):
        remote_bootstrap_addr = next(iter(pull_metas.values())).remote_bootstrap_addr
        if remote_bootstrap_addr not in self._pending_bootstrap_queries:
            self._pending_bootstrap_queries[remote_bootstrap_addr] = asyncio.Event()
            await self._connect_to_prefiller_bootstrap(remote_bootstrap_addr)
        else:
            await self._pending_bootstrap_queries[remote_bootstrap_addr].wait()

        if remote_engine_id not in self._remote_agents:
            logger.error(
                "Failed to find remote engine_id %s from bootstrap server %s",
                remote_engine_id,
                remote_bootstrap_addr,
            )
            return

        self.receive_kv(remote_engine_id, pull_metas)

    async def _start_load_kv(
        self, reqs_to_recv: dict[EngineId, dict[ReqId, PullReqMeta]]
    ):
        # reprocess timeout reqs (merge per-engine pull maps, do not replace whole inner dicts)
        if self.timeout_reqs_to_recv:
            for engine_id, timed_out in self.timeout_reqs_to_recv.items():
                inner = reqs_to_recv.setdefault(engine_id, {})
                inner.update(timed_out)
            self.timeout_reqs_to_recv.clear()

        for remote_engine_id, pull_metas in reqs_to_recv.items():
            if remote_engine_id not in self._remote_agents:
                asyncio.create_task(
                    self.handle_new_engine_id(remote_engine_id, pull_metas)
                )
            else:
                self.receive_kv(remote_engine_id, pull_metas)

    async def record_send_reqs(self, metadata: MooncakeConnectorMetadata):
        for p_req_id, (transfer_id, block_ids) in metadata.reqs_to_send.items():
            if block_ids:
                # Already gone through request_finished()
                send_meta = self.reqs_need_send[transfer_id]
                send_meta.p_req_id = p_req_id
                send_meta.local_block_ids = block_ids
                send_meta.expire_time = time.perf_counter() + 480
                send_meta.ready.set()
            else:
                # From update_state_after_alloc(),
                # but not reach request_finished() yet
                # This may be already created by send_kv_to_decode()
                # when D is sending MooncakeXferMetadata.
                if transfer_id not in self.reqs_need_send:
                    self.reqs_need_send[transfer_id] = SendBlockMeta(
                        p_req_id=p_req_id,
                        transfer_id=transfer_id,
                        local_block_ids=[],
                        ready=asyncio.Event(),
                    )
        for transfer_id in metadata.reqs_not_processed:
            send_meta = self.reqs_need_send.pop(transfer_id)
            if send_meta:
                assert not send_meta.ready.is_set()

    def start_load_kv(self, metadata: MooncakeConnectorMetadata):
        if not self.is_kv_producer and metadata.reqs_to_recv:
            asyncio.run_coroutine_threadsafe(
                self._start_load_kv(metadata.reqs_to_recv), self.receiver_loop
            )

        if not self.is_kv_consumer and (
            metadata.reqs_to_send or metadata.reqs_not_processed
        ):
            asyncio.run_coroutine_threadsafe(
                self.record_send_reqs(metadata), self.sender_loop
            )


def group_concurrent_contiguous(
    src_indices: list[int], dst_indices: list[int]
) -> tuple[list[list[int]], list[list[int]]]:
    """Group parallel src/dst index lists into contiguous runs (NumPy)."""
    if len(src_indices) == 0:
        return [], []

    brk = np.where((np.diff(src_indices) != 1) | (np.diff(dst_indices) != 1))[0] + 1
    src_groups = np.split(src_indices, brk)
    dst_groups = np.split(dst_indices, brk)
    return [g.tolist() for g in src_groups], [g.tolist() for g in dst_groups]
