from __future__ import annotations

from pathlib import Path

import pytest

from infiniloader import paths


def test_default_base_dir_uses_iloader_base_dir(tmp_path, monkeypatch):
    base = tmp_path / "custom-state"
    monkeypatch.setenv("ILOADER_BASE_DIR", str(base))
    monkeypatch.delenv("XDG_CACHE_HOME", raising=False)

    assert paths.default_base_dir() == base
    assert base.exists()


def test_default_base_dir_uses_xdg_cache_home(tmp_path, monkeypatch):
    xdg = tmp_path / "xdg-cache"
    monkeypatch.delenv("ILOADER_BASE_DIR", raising=False)
    monkeypatch.setenv("XDG_CACHE_HOME", str(xdg))

    expected = xdg / "infiniloader"
    assert paths.default_base_dir() == expected
    assert expected.exists()


def test_default_base_dir_errors_when_configured_path_is_file(tmp_path, monkeypatch):
    target = tmp_path / "not-a-dir"
    target.write_text("nope\n", encoding="utf-8")
    monkeypatch.setenv("ILOADER_BASE_DIR", str(target))

    with pytest.raises(RuntimeError):
        paths.default_base_dir()


def test_named_runtime_paths_are_under_base_dir(tmp_path):
    base = tmp_path / "state"

    assert paths.default_socket_path("unit", base) == base / "run" / "unit.sock"
    assert paths.default_pid_path("unit", base) == base / "run" / "unit.pid"
    assert paths.default_state_path("unit", base) == base / "run" / "unit.state.json"
    assert paths.default_view_path("unit", base) == base / "views" / "unit"
    assert paths.default_log_path("unit", base) == base / "logs" / "unit.log"
