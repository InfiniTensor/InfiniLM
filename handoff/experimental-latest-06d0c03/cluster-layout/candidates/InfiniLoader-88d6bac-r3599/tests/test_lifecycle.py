from __future__ import annotations

from contextlib import contextmanager
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import pytest

from conftest import make_model_dir
from infiniloader.cli import main
from infiniloader.ipc import request
from infiniloader.runtime import process_token, signal_owned_process

pytestmark = pytest.mark.skipif(not hasattr(os, "memfd_create"), reason="Linux integration tests")


def wait_for(check, timeout=8):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if check():
            return
        time.sleep(.02)
    raise AssertionError("daemon did not reach expected state")


@contextmanager
def running_daemon(tmp_path, model, *, slow=False, extra=()):
    state = tmp_path / "state"
    env = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[1] / "src"))
    code = (
        "import time; from infiniloader.daemon import WeightLoaderDaemon; "
        "from infiniloader.cli import main; "
        "original=WeightLoaderDaemon.load; "
        "WeightLoaderDaemon.load=lambda self: (time.sleep(1), original(self))[1]; "
        "raise SystemExit(main())"
    )
    command = [sys.executable, "-c", code] if slow else [sys.executable, "-m", "infiniloader"]
    command += ["_serve", str(model), "--name", "unit", "--base-dir", str(state), "--replace", "--quiet"]
    command += list(extra)
    with (tmp_path / "daemon.log").open("w+") as log:
        proc = subprocess.Popen(command, env=env, stdout=log, stderr=log)
        try:
            def registered():
                if proc.poll() is not None:
                    raise AssertionError(f"daemon exited {proc.returncode}: {(tmp_path / 'daemon.log').read_text()}")
                return (state / "run/unit.state.json").exists()
            wait_for(registered, timeout=15)
            if not slow:
                wait_for(lambda: (state / "run/unit.sock").exists())
            yield proc, state
        finally:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=3)


def test_different_model_and_revision_cannot_reuse_name(tmp_path, capsys):
    a = make_model_dir(tmp_path / "a")
    b = make_model_dir(tmp_path / "b")
    with running_daemon(tmp_path, a) as (_, state):
        assert main(["start", str(b), "--name", "unit", "--base-dir", str(state)]) == 1
        (a / "config.json").write_text('{"changed":true}')
        assert main(["start", str(a), "--name", "unit", "--base-dir", str(state)]) == 1
        assert "different model" in capsys.readouterr().err


def test_start_joins_in_progress_daemon(tmp_path, capsys):
    model = make_model_dir(tmp_path / "model")
    with running_daemon(tmp_path, model, slow=True) as (proc, state):
        assert main(["start", str(model), "--name", "unit", "--base-dir", str(state)]) == 0
        status = request(state / "run/unit.sock", {"command": "status"})["status"]
        assert status["pid"] == proc.pid
        assert (Path(status["view"]) / "model.safetensors").read_bytes() == b"0123456789abcdef"


def test_stop_loading_preserves_unmanaged_directory(tmp_path):
    model = make_model_dir(tmp_path / "model")
    ordinary = tmp_path / "state/views/unit"
    ordinary.mkdir(parents=True)
    (ordinary / "important").write_text("keep")
    with running_daemon(tmp_path, model, slow=True) as (proc, state):
        assert main(["stop", "unit", "--base-dir", str(state), "--remove-view"]) == 1
        proc.wait(timeout=3)
        assert (ordinary / "important").read_text() == "keep"


def test_reused_pid_is_never_signalled():
    with pytest.raises(RuntimeError, match="unverified PID"):
        signal_owned_process(os.getpid(), "wrong-start-time")
    assert process_token(os.getpid()) is not None


def test_two_simultaneous_cli_starts_share_one_daemon(tmp_path):
    model = make_model_dir(tmp_path / "model", payload=b"x" * 1024 * 1024)
    state = tmp_path / "state"
    env = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[1] / "src"))
    command = [sys.executable, "-m", "infiniloader", "start", str(model), "--name", "unit",
               "--base-dir", str(state), "--no-wait"]
    clients = [subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env) for _ in range(2)]
    daemon = None
    try:
        outputs = [client.communicate(timeout=20) for client in clients]
        assert all(client.returncode == 0 for client in clients), outputs
        assert outputs[0][0] == outputs[1][0]
        wait_for(lambda: (state / "run/unit.sock").exists())
        daemon = request(state / "run/unit.sock", {"command": "status"})["status"]
        entries = json.loads((state / "run/resident.json").read_text())["entries"]
        assert len(entries) == 1
        assert next(iter(entries.values()))["pid"] == daemon["pid"]
    finally:
        for client in clients:
            if client.poll() is None:
                client.kill()
                client.wait(timeout=3)
        if daemon:
            signal_owned_process(daemon["pid"], daemon["process_token"])
            wait_for(lambda: process_token(daemon["pid"]) is None)


@pytest.mark.parametrize("name", ["../model", "/tmp/model", "a/b", "a..b", "", "x" * 97])
def test_invalid_names_are_rejected(name, tmp_path):
    assert main(["start", str(tmp_path), "--name", name, "--base-dir", str(tmp_path / "state")]) == 1
