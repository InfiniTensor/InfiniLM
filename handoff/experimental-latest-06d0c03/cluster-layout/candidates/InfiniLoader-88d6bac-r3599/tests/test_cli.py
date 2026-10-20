from __future__ import annotations

import json
import os

from conftest import make_model_dir
from infiniloader.cli import build_parser, main
from infiniloader.runtime import process_token


def test_main_inspect_json_outputs_model_summary(tmp_path, capsys):
    model = make_model_dir(tmp_path / "model")

    code = main(["inspect", str(model), "--json"])
    captured = capsys.readouterr()
    data = json.loads(captured.out)

    assert code == 0
    assert data["file_count"] == 4
    assert data["by_suffix"][".safetensors"]["count"] == 1


def test_main_returns_error_for_missing_model(tmp_path, capsys):
    code = main(["inspect", str(tmp_path / "missing")])
    captured = capsys.readouterr()

    assert code == 1
    assert "error:" in captured.err


def test_status_reports_starting_when_pid_exists_without_socket(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr("infiniloader.cli._pid_is_running", lambda _pid: True)
    run_dir = tmp_path / "state" / "run"
    run_dir.mkdir(parents=True)
    (run_dir / "unit.pid").write_text(str(os.getpid()), encoding="utf-8")
    (run_dir / "unit.state.json").write_text(
        json.dumps(
            {
                "state": "loading",
                "process_token": process_token(os.getpid()),
                "copied_bytes": 5,
                "total_bytes": 10,
                "rate_bytes_per_second": 2,
                "eta_seconds": 3,
            }
        ),
        encoding="utf-8",
    )

    code = main(["status", "--name", "unit", "--base-dir", str(tmp_path / "state")])
    captured = capsys.readouterr()

    assert code == 0
    assert "state:  loading" in captured.out
    assert "progress:" in captured.out
    assert "50.00%" in captured.out
    assert "daemon is still loading" in captured.out


def test_list_reports_active_starting_cache_name(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr("infiniloader.cli._pid_is_running", lambda _pid: True)
    run_dir = tmp_path / "state" / "run"
    run_dir.mkdir(parents=True)
    (run_dir / "unit.pid").write_text(str(os.getpid()), encoding="utf-8")
    (run_dir / "unit.state.json").write_text(json.dumps({"process_token": process_token(os.getpid())}))

    code = main(["list", "--base-dir", str(tmp_path / "state"), "--names-only"])
    captured = capsys.readouterr()

    assert code == 0
    assert captured.out.splitlines() == ["unit"]


def test_stop_all_can_stop_starting_daemon(tmp_path, monkeypatch, capsys):
    killed = []
    monkeypatch.setattr("infiniloader.cli._pid_is_running", lambda _pid: True)
    monkeypatch.setattr("infiniloader.cli.os.kill", lambda pid, sig: killed.append((pid, sig)))
    monkeypatch.setattr("infiniloader.cli.process_matches", lambda pid, token: True)
    monkeypatch.setattr("infiniloader.cli.signal_owned_process", lambda pid, token: killed.append((pid, token)))
    run_dir = tmp_path / "state" / "run"
    run_dir.mkdir(parents=True)
    (run_dir / "unit.pid").write_text("12345", encoding="utf-8")
    (run_dir / "unit.state.json").write_text(json.dumps({"process_token": "test"}))

    code = main(["stop", "--all", "--base-dir", str(tmp_path / "state")])
    captured = capsys.readouterr()

    assert code == 0
    assert killed
    assert "unit: stopping pid=12345" in captured.out


def test_completion_bash_outputs_completion_function(capsys):
    code = main(["completion", "bash"])
    captured = capsys.readouterr()

    assert code == 0
    assert "_iloader_complete" in captured.out
    assert "complete -F _iloader_complete iloader" in captured.out
    assert "torch-cache" not in captured.out
    assert "torch-run" not in captured.out
    assert "start-all" not in captured.out
    assert " serve " not in captured.out


def test_help_shows_only_public_commands(capsys):
    parser = build_parser()
    parser.print_help()
    captured = capsys.readouterr()

    assert "start" in captured.out
    assert "list" in captured.out
    assert "_serve" not in captured.out
    assert "==SUPPRESS==" not in captured.out
    assert "torch-cache" not in captured.out
    assert "start-all" not in captured.out
