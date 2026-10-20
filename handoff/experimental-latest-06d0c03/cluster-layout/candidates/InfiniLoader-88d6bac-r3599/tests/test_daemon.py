from __future__ import annotations

import signal

import pytest

from infiniloader.daemon import WeightLoaderDaemon


def test_signal_during_load_exits_immediately_and_cleans_runtime_files(tmp_path, monkeypatch):
    pid_path = tmp_path / "run" / "unit.pid"
    state_path = tmp_path / "run" / "unit.state.json"
    socket_path = tmp_path / "run" / "unit.sock"
    pid_path.parent.mkdir(parents=True)
    pid_path.write_text("12345", encoding="utf-8")
    state_path.write_text("{}", encoding="utf-8")

    exit_codes = []

    def fake_exit(code: int) -> None:
        exit_codes.append(code)
        raise SystemExit(code)

    monkeypatch.setattr("infiniloader.daemon.os._exit", fake_exit)
    daemon = WeightLoaderDaemon(
        name="unit",
        source=tmp_path / "model",
        view=tmp_path / "view",
        socket_path=socket_path,
        pid_path=pid_path,
        state_path=state_path,
        upstream_source=None,
        local_cache_path=None,
        include_patterns=(),
        exclude_patterns=(),
        replace=True,
        force=False,
        max_memory_fraction=0.9,
        chunk_size=1024,
        workers=1,
    )
    daemon._owns_runtime = True
    daemon._claim_runtime_file(pid_path)
    daemon._claim_runtime_file(state_path)

    with pytest.raises(SystemExit):
        daemon._handle_signal(signal.SIGTERM, None)

    assert exit_codes == [0]
    assert not pid_path.exists()
    assert not state_path.exists()
