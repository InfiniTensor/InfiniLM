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

import infinicore
import numpy as np
from infinicore.utils import infinicore_to_numpy_dtype

_FRAME_HEADER = struct.Struct("!Q")
_TENSOR_MARKER = "__infinicore_tensor__"
_TUPLE_MARKER = "__tuple__"

logger = logging.getLogger(__name__)


def _recv_exact(sock: socket.socket, size: int) -> bytes:
    chunks = bytearray()
    while len(chunks) < size:
        chunk = sock.recv(size - len(chunks))
        if not chunk:
            raise ConnectionError("pipeline control connection closed")
        chunks.extend(chunk)
    return bytes(chunks)


def _send_payload(sock: socket.socket, payload: bytes) -> None:
    sock.sendall(_FRAME_HEADER.pack(len(payload)))
    sock.sendall(payload)


def _recv_payload(sock: socket.socket) -> bytes:
    (size,) = _FRAME_HEADER.unpack(_recv_exact(sock, _FRAME_HEADER.size))
    return _recv_exact(sock, size)


def _send_message(sock: socket.socket, message: Any) -> None:
    _send_payload(sock, pickle.dumps(message, protocol=pickle.HIGHEST_PROTOCOL))


def _recv_message(sock: socket.socket) -> Any:
    return pickle.loads(_recv_payload(sock))


def _tensor_to_numpy(tensor: infinicore.Tensor) -> np.ndarray:
    if tensor.device.type != "cpu":
        tensor = tensor.to(infinicore.device("cpu", 0))
        infinicore.sync_device()
    if not tensor.is_contiguous():
        tensor = tensor.contiguous()

    shape = tuple(tensor.shape)
    dtype = np.dtype(infinicore_to_numpy_dtype(tensor.dtype))
    if tensor.numel() == 0:
        return np.empty(shape, dtype=dtype)

    num_bytes = tensor.numel() * dtype.itemsize
    buffer_type = ctypes.c_ubyte * num_bytes
    buffer = buffer_type.from_address(tensor.data_ptr())
    return (
        np.frombuffer(buffer, dtype=dtype, count=tensor.numel()).reshape(shape).copy()
    )


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


def _connect_with_retry(endpoint: str, timeout: float = 300.0) -> socket.socket:
    host, port = _parse_endpoint(endpoint)
    deadline = time.monotonic() + timeout
    last_error: OSError | None = None
    while time.monotonic() < deadline:
        try:
            sock = socket.create_connection((host, port), timeout=5.0)
            sock.settimeout(None)
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            return sock
        except OSError as error:
            last_error = error
            time.sleep(0.1)
    raise ConnectionError(
        f"failed to connect to pipeline host {endpoint}"
    ) from last_error


class PipelineControlServer:
    """Stage-0 control plane for persistent PP worker processes."""

    def __init__(self, pp_size: int, master_port: int):
        connections_by_stage: dict[int, socket.socket] = {}
        self._connections: list[socket.socket] = []
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
                server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                server.bind(("0.0.0.0", master_port))
                server.listen(pp_size - 1)
                logger.info(
                    "Pipeline control server listening: role=coordinator, "
                    "stage=0, endpoint=0.0.0.0:%s, workers=%s",
                    master_port,
                    pp_size - 1,
                )

                while len(connections_by_stage) < pp_size - 1:
                    connection, peer = server.accept()
                    connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                    ready = _recv_message(connection)
                    stage = ready.get("pp_stage")
                    if (
                        ready.get("status") != "ready"
                        or not isinstance(stage, int)
                        or not 1 <= stage < pp_size
                        or stage in connections_by_stage
                    ):
                        connection.close()
                        raise RuntimeError(
                            f"invalid pipeline worker registration: {ready!r}"
                        )
                    connections_by_stage[stage] = connection
                    _send_message(
                        connection,
                        {"status": "registered", "pp_stage": stage},
                    )
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
        payload = pickle.dumps(
            {"command": "forward", "model_input": _encode(model_input)},
            protocol=pickle.HIGHEST_PROTOCOL,
        )
        for connection in self._connections:
            _send_payload(connection, payload)

    def wait_forward(self) -> None:
        for connection in self._connections:
            response = _recv_message(connection)
            if not response.get("ok", False):
                raise RuntimeError(
                    "pipeline worker forward failed:\n"
                    + response.get("error", "unknown error")
                )

    def close(self) -> None:
        for connection in self._connections:
            try:
                _send_message(connection, {"command": "shutdown"})
                _recv_message(connection)
            except (ConnectionError, OSError):
                pass
            finally:
                connection.close()
        self._connections.clear()


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
        self._connection = _connect_with_retry(endpoint)
        _send_message(self._connection, {"status": "ready", "pp_stage": self._pp_stage})
        response = _recv_message(self._connection)
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
                command = message.get("command")
                if command == "shutdown":
                    _send_message(connection, {"ok": True})
                    return
                if command != "forward":
                    _send_message(
                        connection,
                        {"ok": False, "error": f"unknown command: {command}"},
                    )
                    continue
                try:
                    model_input = _decode(message["model_input"])
                    self._model_runner.model_engine.forward(**model_input)
                    _send_message(connection, {"ok": True})
                except BaseException:
                    error = traceback.format_exc()
                    try:
                        _send_message(connection, {"ok": False, "error": error})
                    except (ConnectionError, OSError):
                        pass
                    return
