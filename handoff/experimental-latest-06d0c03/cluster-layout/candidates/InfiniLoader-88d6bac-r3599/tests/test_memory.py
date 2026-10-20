import json
import os
from pathlib import Path

import pytest

from infiniloader import memory
from infiniloader.memory import ResidentReservation, check_memory, memory_info, parse_bytes
from infiniloader.runtime import atomic_json


def fake_cgroup(tmp_path, *, v1=False):
    proc = tmp_path / "proc"
    (proc / "self").mkdir(parents=True)
    (proc / "meminfo").write_text("MemAvailable: 1000000 kB\n")
    mount = tmp_path / "cgroup"
    leaf = mount / "job"
    leaf.mkdir(parents=True)
    if v1:
        (proc / "self/cgroup").write_text("3:memory:/job\n")
        (proc / "self/mountinfo").write_text(f"1 0 0:0 / {mount} rw - cgroup cgroup rw,memory\n")
        limit, used = "memory.limit_in_bytes", "memory.usage_in_bytes"
    else:
        (proc / "self/cgroup").write_text("0::/job\n")
        (proc / "self/mountinfo").write_text(f"1 0 0:0 / {mount} rw - cgroup2 cgroup rw\n")
        limit, used = "memory.max", "memory.current"
    (mount / limit).write_text("3000")
    (mount / used).write_text("2800")
    (leaf / limit).write_text("2000")
    (leaf / used).write_text("500")
    return proc


@pytest.mark.parametrize("v1", [False, True])
def test_effective_headroom_honors_all_cgroup_ancestors(tmp_path, v1):
    info = memory_info(fake_cgroup(tmp_path, v1=v1))
    assert info["host_available_bytes"] == 1024000000
    assert info["available_bytes"] == 200
    assert len(info["cgroups"]) == 2


def test_force_does_not_bypass_absolute_cap(monkeypatch):
    monkeypatch.setattr(memory, "memory_info", lambda: dict(available_bytes=100, cgroups=[], host_available_bytes=100))
    with pytest.raises(MemoryError):
        check_memory(90, reserve_bytes=20)
    assert check_memory(90, reserve_bytes=20, force=True)["available_check_bypassed"]
    with pytest.raises(MemoryError, match="max-memory-bytes"):
        check_memory(90, max_bytes=80, force=True)


@pytest.mark.skipif(os.name != "posix", reason="POSIX file locks")
def test_reservations_enforce_shared_quota_and_remove_stale_pids(tmp_path, monkeypatch):
    monkeypatch.setattr(memory, "memory_info", lambda: dict(available_bytes=10000, cgroups=[], host_available_bytes=10000))
    atomic_json(tmp_path / "resident.json", {"entries": {"dead": {"pid": 99999999, "process_token": "old", "reserved_bytes": 99999}}})
    with ResidentReservation(tmp_path, "a", 60, 10, max_resident_bytes=100, reserve_bytes=0) as a:
        with pytest.raises(MemoryError, match="quota"):
            with ResidentReservation(tmp_path, "b", 40, 0, reserve_bytes=0):
                pass
        a.ready()
        with ResidentReservation(tmp_path, "b", 40, 0, reserve_bytes=0):
            data = json.loads((tmp_path / "resident.json").read_text())
            assert len(data["entries"]) == 2
    assert not json.loads((tmp_path / "resident.json").read_text())["entries"]


@pytest.mark.parametrize("value, expected", [("0", 0), ("2GiB", 2 * 1024**3), ("256M", 256 * 1024**2)])
def test_human_readable_sizes(value, expected):
    assert parse_bytes(value) == expected
