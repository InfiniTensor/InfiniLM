import os
import sys
import time

import pytest

from conftest import make_model_dir
from infiniloader.cli import main
from infiniloader.diagnostics import diagnose
from infiniloader.ipc import request
from infiniloader.local_cache import materialize_model_cache
from infiniloader.runtime import process_token
from test_lifecycle import running_daemon

pytestmark = pytest.mark.skipif(not hasattr(os, "memfd_create"), reason="Linux management tests")


def test_leases_protect_stop_gc_and_idle_ttl(tmp_path):
    model = make_model_dir(tmp_path / "model")
    with running_daemon(tmp_path, model, extra=["--idle-ttl", "2"]) as (proc, state):
        sock = state / "run/unit.sock"
        lease = request(sock, {"command": "lease-acquire", "pid": os.getpid(), "process_token": process_token(os.getpid())})
        time.sleep(2.1)
        assert request(sock, {"command": "status"})["status"]["active_leases"] == 1
        assert main(["stop", "unit", "--base-dir", str(state)]) == 1
        assert main(["gc", "--resident", "--base-dir", str(state), "--max-bytes", "0", "--apply"]) == 1
        assert proc.poll() is None
        request(sock, {"command": "lease-release", "lease": lease["lease"]})
        proc.wait(timeout=4)
        assert not (state / "views/unit").exists()


def test_run_substitutes_model_path_and_releases_lease(tmp_path):
    model = make_model_dir(tmp_path / "model")
    out = tmp_path / "result"
    program = [sys.executable, "-c", "import pathlib,sys; pathlib.Path(sys.argv[2]).write_bytes((pathlib.Path(sys.argv[1])/'model.safetensors').read_bytes())", "{model}", str(out)]
    with running_daemon(tmp_path, model) as (_, state):
        assert main(["run", "--base-dir", str(state), "unit", "--", *program]) == 0
        assert out.read_bytes() == b"0123456789abcdef"
        assert request(state / "run/unit.sock", {"command": "status"})["status"]["active_leases"] == 0


def test_run_disk_cache_preserves_command_exit_code(tmp_path):
    local = materialize_model_cache(make_model_dir(tmp_path / "model"), base_dir=tmp_path / "state")
    assert main(["run", "--cache-path", str(local.cache_path), "--", sys.executable, "-c", "raise SystemExit(7)"]) == 7


def test_doctor_checks_actual_linux_capabilities(tmp_path):
    result = diagnose(tmp_path)
    assert result["ok"], result
    assert result["memory"]["available_bytes"] is not None
    assert "locked_memory" in result["limits"]


def test_resident_gc_removes_unleased_cache(tmp_path):
    with running_daemon(tmp_path, make_model_dir(tmp_path / "model")) as (proc, state):
        assert main(["gc", "--resident", "--base-dir", str(state), "--ttl", "0"]) == 0
        assert proc.poll() is None
        assert main(["gc", "--resident", "--base-dir", str(state), "--ttl", "0", "--apply"]) == 0
        proc.wait(timeout=4)
        assert not (state / "views/unit").exists()
