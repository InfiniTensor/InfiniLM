"""Persistent host-to-worker control transport for pipeline parallelism.

PP startup has two network layers which intentionally reuse ``master_port`` in
sequence. During ``InferEngine`` construction, the C++ TCP rendezvous briefly
binds the port and distributes the InfiniCCL unique ID. Once every process has
joined the global communicator, that socket closes. After model loading, stage
0 binds the same port as the persistent control server implemented here.

For each scheduler step, stage 0 sends identical model metadata to all worker
stages before entering its own forward. Tensor activations do not travel over
this socket: matching TP ranks send them through the global InfiniCCL
communicator, and each receiving node all-gathers its hidden-size shards.
"""

import ctypes
import logging
import pickle
import socket
import struct
import time
import traceback
from typing import Any

from .. import _core as infinicore
import numpy as np
from .._core.utils import infinicore_to_numpy_dtype

_FRAME_HEADER = struct.Struct("!Q")
_MAX_CONTROL_FRAME_BYTES = 512 * 1024 * 1024
_STARTUP_TIMEOUT_S = 180.0
_FORWARD_TIMEOUT_S = 300.0
_TENSOR_MARKER = "__infinicore_tensor__"
_TUPLE_MARKER = "__tuple__"

logger = logging.getLogger(__name__)


def _recv_exact(
    sock: socket.socket, size: int, deadline: float | None = None
) -> bytearray:
    payload = bytearray(size)
    view = memoryview(payload)
    received = 0
    while received < size:
        if deadline is not None:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("pipeline control receive timed out")
            sock.settimeout(remaining)
        count = sock.recv_into(view[received:])
        if count == 0:
            raise ConnectionError("pipeline control connection closed")
        received += count
    return payload


def _send_payload(
    sock: socket.socket, payload: bytes, deadline: float | None = None
) -> None:
    if len(payload) > _MAX_CONTROL_FRAME_BYTES:
        raise ValueError("pipeline control message exceeds the maximum frame size")
    segments = [memoryview(_FRAME_HEADER.pack(len(payload))), memoryview(payload)]
    if not hasattr(sock, "sendmsg"):
        for segment in segments:
            if deadline is not None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("pipeline control send timed out")
                sock.settimeout(remaining)
            sock.sendall(segment)
        return

    # Keep the length prefix and payload in one syscall without concatenating
    # and copying the complete control message into another bytes object.
    while segments:
        if deadline is not None:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("pipeline control send timed out")
            sock.settimeout(remaining)
        sent = sock.sendmsg(segments)
        if sent == 0:
            raise ConnectionError("pipeline control connection closed")
        while segments and sent >= len(segments[0]):
            sent -= len(segments[0])
            segments.pop(0)
        if segments and sent:
            segments[0] = segments[0][sent:]


def _recv_payload(
    sock: socket.socket, deadline: float | None = None
) -> bytearray:
    (size,) = _FRAME_HEADER.unpack(_recv_exact(sock, _FRAME_HEADER.size, deadline))
    if size > _MAX_CONTROL_FRAME_BYTES:
        raise ValueError("pipeline control message exceeds the maximum frame size")
    return _recv_exact(sock, size, deadline)


def _send_message(
    sock: socket.socket, message: Any, deadline: float | None = None
) -> None:
    _send_payload(sock, pickle.dumps(message, protocol=pickle.HIGHEST_PROTOCOL), deadline)


def _recv_message(sock: socket.socket, deadline: float | None = None) -> Any:
    return pickle.loads(_recv_payload(sock, deadline))


_FORWARD_OK_PAYLOAD = pickle.dumps({"ok": True}, protocol=pickle.HIGHEST_PROTOCOL)


def _tensor_to_numpy(tensor: infinicore.Tensor) -> np.ndarray:
    owns_temporary_storage = False
    if tensor.device.type != "cpu":
        tensor = tensor.to(infinicore.device("cpu", 0))
        infinicore.sync_device()
        owns_temporary_storage = True
    if not tensor.is_contiguous():
        tensor = tensor.contiguous()
        owns_temporary_storage = True

    shape = tuple(tensor.shape)
    dtype = np.dtype(infinicore_to_numpy_dtype(tensor.dtype))
    if tensor.numel() == 0:
        return np.empty(shape, dtype=dtype)

    num_bytes = tensor.numel() * dtype.itemsize
    buffer_type = ctypes.c_ubyte * num_bytes
    buffer = buffer_type.from_address(tensor.data_ptr())
    array = np.frombuffer(buffer, dtype=dtype, count=tensor.numel()).reshape(shape)
    # Normal scheduler inputs are already contiguous CPU tensors and remain
    # alive until dispatch completes, so pickle can serialize their storage
    # directly. Preserve a copy only when this function created temporary
    # tensor storage whose lifetime would otherwise end on return.
    return array.copy() if owns_temporary_storage else array


def _encode(value: Any) -> Any:
    if isinstance(value, infinicore.Tensor):
        return {_TENSOR_MARKER: _tensor_to_numpy(value)}
    if isinstance(value, dict):
        return {key: _encode(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_encode(item) for item in value]
    if isinstance(value, tuple):
        return {_TUPLE_MARKER: [_encode(item) for item in value]}
    return value


def _decode(value: Any) -> Any:
    if isinstance(value, dict) and _TENSOR_MARKER in value:
        return infinicore.from_numpy(value[_TENSOR_MARKER])
    if isinstance(value, dict) and _TUPLE_MARKER in value:
        return tuple(_decode(item) for item in value[_TUPLE_MARKER])
    if isinstance(value, dict):
        return {key: _decode(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_decode(item) for item in value]
    return value


def _parse_endpoint(endpoint: str) -> tuple[str, int]:
    host, separator, port = endpoint.rpartition(":")
    if not separator or not host or not port:
        raise ValueError(
            f"invalid pipeline worker endpoint {endpoint!r}; expected HOST:PORT"
        )
    return host, int(port)


def _connect_with_retry(
    endpoint: str, timeout_s: float = _STARTUP_TIMEOUT_S
) -> socket.socket:
    host, port = _parse_endpoint(endpoint)
    if timeout_s <= 0:
        raise ValueError("pipeline startup timeout must be positive")
    deadline = time.monotonic() + timeout_s
    last_error = None
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError(
                f"pipeline control connection to {endpoint} timed out"
            ) from last_error
        try:
            sock = socket.create_connection((host, port), timeout=min(5.0, remaining))
            try:
                sock.settimeout(None)
                sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            except BaseException:
                sock.close()
                raise
            return sock
        except OSError as error:
            last_error = error
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(
                    f"pipeline control connection to {endpoint} timed out"
                ) from last_error
            time.sleep(min(0.1, remaining))


class PipelineControlServer:
    """Stage-0 control plane for persistent PP worker processes."""

    def __init__(
        self, pp_size: int, master_port: int, *,
        bind_addr: str = "0.0.0.0",
        startup_timeout_s: float = _STARTUP_TIMEOUT_S,
        forward_timeout_s: float = _FORWARD_TIMEOUT_S,
    ):
        if startup_timeout_s <= 0 or forward_timeout_s <= 0:
            raise ValueError("pipeline control timeouts must be positive")
        self._forward_timeout_s = forward_timeout_s
        self._forward_deadline: float | None = None
        connections_by_stage: dict[int, socket.socket] = {}
        self._connections: list[socket.socket] = []
        deadline = time.monotonic() + startup_timeout_s
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
                server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                server.bind((bind_addr, master_port))
                server.listen(pp_size - 1)
                logger.info(
                    "Pipeline control server listening: role=coordinator, "
                    "stage=0, endpoint=%s:%s, workers=%s",
                    bind_addr,
                    master_port,
                    pp_size - 1,
                )

                while len(connections_by_stage) < pp_size - 1:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise TimeoutError("pipeline worker registration timed out")
                    server.settimeout(remaining)
                    connection, peer = server.accept()
                    try:
                        connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                        ready = _recv_message(connection, deadline)
                        stage = ready.get("pp_stage") if isinstance(ready, dict) else None
                        if (
                            not isinstance(ready, dict)
                            or ready.get("status") != "ready"
                            or not isinstance(stage, int)
                            or not 1 <= stage < pp_size
                            or stage in connections_by_stage
                        ):
                            raise RuntimeError(
                                f"invalid pipeline worker registration: {ready!r}"
                            )
                        _send_message(
                            connection, {"status": "registered", "pp_stage": stage},
                            deadline,
                        )
                        connection.settimeout(None)
                    except BaseException:
                        connection.close()
                        raise
                    connections_by_stage[stage] = connection
                    logger.info(
                        "Pipeline control connection established: "
                        "role=coordinator, stage=0, peer_stage=%s, peer=%s:%s",
                        stage,
                        peer[0],
                        peer[1],
                    )

            self._connections = [
                connections_by_stage[stage] for stage in range(1, pp_size)
            ]
            logger.info(
                "Pipeline control plane ready: role=coordinator, stage=0, stages=%s",
                pp_size,
            )
        except BaseException:
            for connection in connections_by_stage.values():
                connection.close()
            raise

    def dispatch_forward(self, model_input: dict[str, Any]) -> None:
        if not self._connections:
            raise RuntimeError("pipeline control plane is closed")
        if self._forward_deadline is not None:
            raise RuntimeError("previous pipeline forward has not completed")
        payload = pickle.dumps(
            {"command": "forward", "model_input": _encode(model_input)},
            protocol=pickle.HIGHEST_PROTOCOL,
        )
        deadline = time.monotonic() + self._forward_timeout_s
        try:
            for connection in self._connections:
                _send_payload(connection, payload, deadline)
        except BaseException:
            self.abort()
            raise
        finally:
            for connection in self._connections:
                connection.settimeout(None)
        self._forward_deadline = deadline

    def wait_forward(self) -> None:
        if self._forward_deadline is None:
            raise RuntimeError("no pipeline forward is in progress")
        try:
            for connection in self._connections:
                response = _recv_message(connection, self._forward_deadline)
                if not isinstance(response, dict) or not response.get("ok", False):
                    detail = (
                        response.get("error", "unknown error")
                        if isinstance(response, dict) else repr(response)
                    )
                    raise RuntimeError("pipeline worker forward failed:\n" + str(detail))
        except BaseException:
            self.abort()
            raise
        finally:
            self._forward_deadline = None
            for connection in self._connections:
                connection.settimeout(None)

    def abort(self) -> None:
        """Close the control plane immediately after an unrecoverable failure."""
        for connection in self._connections:
            try:
                connection.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            finally:
                connection.close()
        self._connections.clear()
        self._forward_deadline = None

    def close(self) -> None:
        deadline = time.monotonic() + self._forward_timeout_s
        for connection in self._connections:
            try:
                _send_message(connection, {"command": "shutdown"}, deadline)
                _recv_message(connection, deadline)
            except (ConnectionError, OSError):
                pass
            finally:
                connection.close()
        self._connections.clear()
        self._forward_deadline = None


class PipelineWorkerClient:
    """Persistent control client run by a nonzero PP stage."""

    def __init__(self, model_runner, master_addr: str, master_port: int, pp_stage: int):
        self._model_runner = model_runner
        self._pp_stage = pp_stage
        endpoint = f"{master_addr}:{master_port}"
        logger.info(
            "Connecting pipeline control plane: role=worker, stage=%s, coordinator=%s",
            self._pp_stage,
            endpoint,
        )
        deadline = time.monotonic() + _STARTUP_TIMEOUT_S
        self._connection = _connect_with_retry(endpoint)
        try:
            _send_message(
                self._connection, {"status": "ready", "pp_stage": self._pp_stage},
                deadline,
            )
            response = _recv_message(self._connection, deadline)
        except BaseException:
            self._connection.close()
            raise
        self._connection.settimeout(None)
        if response != {"status": "registered", "pp_stage": self._pp_stage}:
            self._connection.close()
            raise RuntimeError(
                f"invalid pipeline coordinator registration response: {response!r}"
            )
        logger.info(
            "Pipeline control connection established: "
            "role=worker, stage=%s, coordinator=%s",
            self._pp_stage,
            endpoint,
        )

    def serve_forever(self) -> None:
        with self._connection as connection:
            while True:
                message = _recv_message(connection)
                if not isinstance(message, dict):
                    _send_message(
                        connection, {"ok": False, "error": "invalid pipeline command"}
                    )
                    return
                command = message.get("command")
                if command == "shutdown":
                    _send_message(connection, {"ok": True})
                    return
                if command != "forward":
                    _send_message(
                        connection,
                        {"ok": False, "error": f"unknown command: {command}"},
                    )
                    return
                try:
                    model_input = _decode(message["model_input"])
                    self._model_runner.model_engine.forward(**model_input)
                    _send_payload(connection, _FORWARD_OK_PAYLOAD)
                except BaseException:
                    error = traceback.format_exc()
                    try:
                        _send_message(connection, {"ok": False, "error": error})
                    except (ConnectionError, OSError):
                        pass
                    return
