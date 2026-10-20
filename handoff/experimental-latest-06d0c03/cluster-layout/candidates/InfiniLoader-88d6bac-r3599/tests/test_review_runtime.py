"""Runtime and CLI regressions reproduced during independent expert review."""
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

import pytest

from conftest import make_model_dir
from infiniloader.cli import main
from infiniloader.ipc import request
from infiniloader.runtime import process_token
from test_lifecycle import running_daemon

pytestmark = pytest.mark.skipif(not hasattr(os, "memfd_create"), reason="Linux review regressions")


def test_ipc_timeout_cannot_signal_a_ready_leased_daemon(tmp_path):
    with running_daemon(tmp_path, make_model_dir(tmp_path / "model")) as (proc, state):
        sock = state / "run/unit.sock"
        request(sock, {"command": "lease-acquire", "pid": os.getpid(), "process_token": process_token(os.getpid())})
        clients = []
        try:
            for _ in range(4):
                client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                client.connect(str(sock))
                clients.append(client)
            time.sleep(.05)
            assert main(["stop", "unit", "--base-dir", str(state), "--timeout", ".05"]) == 1
            assert proc.poll() is None
        finally:
            for client in clients:
                client.close()
        assert request(sock, {"command": "status"})["status"]["active_leases"] == 1


@pytest.mark.parametrize("option", ["--socket", "--pid-file", "--state-file"])
@pytest.mark.parametrize("symlink", [False, True])
def test_runtime_collisions_never_overwrite_ordinary_files(tmp_path, option, symlink):
    model = make_model_dir(tmp_path / "model")
    valuable = tmp_path / "important.txt"
    valuable.write_text("preserve this data")
    destination = tmp_path / "alias" if symlink else valuable
    if symlink:
        destination.symlink_to(valuable)
    env = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[1] / "src"))
    result = subprocess.run([sys.executable, "-m", "infiniloader", "_serve", str(model), "--name", "unit",
                             "--base-dir", str(tmp_path / "state"), option, str(destination)],
                            env=env, capture_output=True, timeout=15)
    assert result.returncode == 1, result.stderr.decode()
    assert valuable.read_text() == "preserve this data"
    if symlink:
        assert destination.is_symlink()


def test_stop_all_continues_after_a_leased_instance(tmp_path, monkeypatch):
    first_root, second_root = tmp_path / "first", tmp_path / "second"
    first_root.mkdir()
    second_root.mkdir()
    with running_daemon(first_root, make_model_dir(first_root / "model")) as (a, first):
        with running_daemon(second_root, make_model_dir(second_root / "model")) as (b, second):
            request(first / "run/unit.sock", {"command": "lease-acquire", "pid": os.getpid(), "process_token": process_token(os.getpid())})
            rows = [dict(name="a", state="ready"), dict(name="b", state="ready")]
            monkeypatch.setattr("infiniloader.cli._active_daemon_rows", lambda _: rows)
            monkeypatch.setattr("infiniloader.cli._socket", lambda args: (first if args.name == "a" else second) / "run/unit.sock")
            assert main(["stop", "--all", "--base-dir", str(tmp_path)]) == 1
            b.wait(timeout=4)
            assert a.poll() is None


def test_existing_unrelated_unix_listener_is_preserved(tmp_path):
    model = make_model_dir(tmp_path / "model")
    path = tmp_path / "service.sock"
    env = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[1] / "src"))
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as server:
        server.bind(str(path))
        server.listen(1)
        inode = path.stat().st_ino
        result = subprocess.run([sys.executable, "-m", "infiniloader", "_serve", str(model),
                                 "--name", "unit", "--base-dir", str(tmp_path / "state"), "--socket", str(path)],
                                capture_output=True, env=env, timeout=15)
        assert result.returncode == 1
        assert path.stat().st_ino == inode
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
            client.connect(str(path))
        connection, _ = server.accept()
        connection.close()


def test_shutdown_preserves_replaced_pid_file(tmp_path):
    with running_daemon(tmp_path, make_model_dir(tmp_path / "model")) as (proc, state):
        replacement = tmp_path / "replacement"
        replacement.write_text("unrelated replacement data")
        replacement.replace(state / "run/unit.pid")
        request(state / "run/unit.sock", {"command": "stop"})
        proc.wait(timeout=4)
        assert (state / "run/unit.pid").read_text() == "unrelated replacement data"


def test_dist_simulate_failure_text_has_all_paths(tmp_path, capsys):
    make_model_dir(tmp_path / "rank0")
    assert main(["dist-simulate", str(tmp_path), "--ranks", "2", "--rank-view-root", str(tmp_path), "--timeout", "2"]) == 1
    output = capsys.readouterr()
    assert "rank 0: failed" in output.out
    assert "rank 1: failed" in output.out
    assert "error: 'path'" not in output.err
