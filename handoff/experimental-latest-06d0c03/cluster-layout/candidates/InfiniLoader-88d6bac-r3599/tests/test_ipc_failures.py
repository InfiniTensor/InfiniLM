import os
import socket
import time

import pytest

from conftest import make_model_dir
from infiniloader.distributed import simulate_distributed_read
from infiniloader.ipc import request
from test_lifecycle import running_daemon

pytestmark = pytest.mark.skipif(not hasattr(os, "memfd_create"), reason="Linux IPC tests")


def test_disconnected_malformed_and_oversized_clients_do_not_kill_daemon(tmp_path):
    model = make_model_dir(tmp_path / "model")
    with running_daemon(tmp_path, model) as (proc, state):
        sock = state / "run/unit.sock"
        for payload in (b"", b"[]\n", b"not-json\n", b"x" * 65536):
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
                client.connect(str(sock))
                client.sendall(payload)
                client.shutdown(socket.SHUT_RDWR)
            assert request(sock, {"command": "ping"})["pong"]
        assert proc.poll() is None
        assert sock.stat().st_mode & 0o777 == 0o600


def test_idle_client_does_not_block_status_or_stop(tmp_path):
    model = make_model_dir(tmp_path / "model")
    with running_daemon(tmp_path, model) as (proc, state):
        sock = state / "run/unit.sock"
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
            client.connect(str(sock))
            assert request(sock, {"command": "status"}, timeout=.5)["ok"]
            assert request(sock, {"command": "stop"}, timeout=.5)["stopping"]
            proc.wait(timeout=4)
            assert proc.returncode == 0


def test_rank_failure_cancels_barrier_and_reaps_children(tmp_path):
    import multiprocessing as mp

    make_model_dir(tmp_path / "rank0")
    before = {p.pid for p in mp.active_children()}
    start = time.monotonic()
    result = simulate_distributed_read(tmp_path, ranks=2, rank_view_root=tmp_path, timeout=2)
    assert result["ok"] is False
    assert len(result["rank_results"]) == 2
    assert time.monotonic() - start < 5
    assert {p.pid for p in mp.active_children()} == before
