from __future__ import annotations

import json
import socket
from pathlib import Path
from typing import Any


class LoaderConnectionError(RuntimeError):
    pass


def request(socket_path: str | Path, payload: dict[str, Any], timeout: float = 5.0) -> dict[str, Any]:
    sock_path = str(socket_path)
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
            client.settimeout(timeout)
            client.connect(sock_path)
            client.sendall(json.dumps(payload).encode("utf-8") + b"\n")
            data = bytearray()
            while True:
                chunk = client.recv(65536)
                if not chunk:
                    break
                data.extend(chunk)
                if len(data) > 64 * 1024 * 1024:
                    raise LoaderConnectionError("daemon response exceeds 64 MiB")
                if b"\n" in chunk:
                    break
    except OSError as exc:
        raise LoaderConnectionError(f"cannot connect to {sock_path}: {exc}") from exc

    if not data:
        raise LoaderConnectionError(f"empty response from {sock_path}")
    try:
        response = json.loads(data.split(b"\n", 1)[0])
        if not isinstance(response, dict):
            raise ValueError("response must be a JSON object")
    except ValueError as exc:
        raise LoaderConnectionError(f"invalid response from {sock_path}: {exc}") from exc
    if not response.get("ok", False):
        raise RuntimeError(response.get("error", "InfiniLoader daemon request failed"))
    return response
