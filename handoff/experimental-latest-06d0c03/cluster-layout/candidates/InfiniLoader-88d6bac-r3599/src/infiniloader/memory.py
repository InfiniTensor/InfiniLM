"""Container-aware headroom and per-base-directory resident reservations."""
from __future__ import annotations

import os
from pathlib import Path
import re

from .runtime import FileLock, atomic_json, process_matches, process_token, read_json


def parse_bytes(value: str) -> int:
    match = re.fullmatch(r"(\d+)([KMGT](?:I?B)?|B)?", value.upper())
    if not match:
        raise ValueError("size must be an integer optionally followed by B, KiB, MiB, GiB or TiB")
    unit = (match[2] or "B")[0]
    return int(match[1]) * (1024 ** ("BKMGT".index(unit)))


def _number(path: Path) -> int | None:
    try:
        value = int(path.read_text().strip())
        return value if 0 <= value < 1 << 60 else None
    except (OSError, ValueError):
        return None


def memory_info(proc_root: Path = Path("/proc")) -> dict:
    host = None
    try:
        for line in (proc_root / "meminfo").read_text().splitlines():
            if line.startswith("MemAvailable:"):
                host = int(line.split()[1]) * 1024
    except (OSError, ValueError):
        pass
    groups = []
    try:
        memberships = [line.split(":", 2) for line in (proc_root / "self/cgroup").read_text().splitlines()]
        for line in (proc_root / "self/mountinfo").read_text().splitlines():
            before, after = line.split(" - ", 1)
            fields, fs = before.split(), after.split()
            if fs[0] not in ("cgroup", "cgroup2"):
                continue
            v2 = fs[0] == "cgroup2"
            if not v2 and "memory" not in fs[2].split(","):
                continue
            membership = next((item[2] for item in memberships if (item[0] == "0" if v2 else "memory" in item[1].split(","))), None)
            if membership is None:
                continue
            mount = Path(fields[4].replace("\\040", " "))
            root = Path(fields[3])
            member = Path(membership)
            relative = member.relative_to(root) if member.is_relative_to(root) else Path(".")
            if ".." in relative.parts:
                relative = Path(".")
            leaf = mount / relative
            if not leaf.is_dir():
                leaf = mount  # Namespaced cgroup mounts may already represent the leaf.
            while leaf.is_relative_to(mount):
                limit = _number(leaf / ("memory.max" if v2 else "memory.limit_in_bytes"))
                used = _number(leaf / ("memory.current" if v2 else "memory.usage_in_bytes"))
                if limit is not None and used is not None:
                    groups.append(dict(path=str(leaf), limit_bytes=limit, used_bytes=used,
                                       available_bytes=max(0, limit - used)))
                if leaf == mount:
                    break
                leaf = leaf.parent
    except (OSError, ValueError, IndexError):
        pass
    candidates = [group["available_bytes"] for group in groups]
    if host is not None:
        candidates.append(host)
    return dict(host_available_bytes=host, cgroups=groups,
                available_bytes=min(candidates) if candidates else None)


def check_memory(required: int, *, fraction=.9, max_bytes=None, reserve_bytes=256 * 1024 * 1024,
                  force=False, pending_bytes=0) -> dict:
    if not 0 < fraction <= 1 or reserve_bytes < 0 or max_bytes is not None and max_bytes <= 0:
        raise ValueError("invalid memory budget")
    info = memory_info()
    available = info["available_bytes"]
    allowed = max(0, int(available * fraction) - reserve_bytes - pending_bytes) if available is not None else None
    if max_bytes is not None and required > max_bytes:
        raise MemoryError(f"preload requires {required} bytes, exceeding --max-memory-bytes={max_bytes}")
    if not force and allowed is not None and required > allowed:
        raise MemoryError(f"preload requires {required} bytes including worker buffers; effective budget is {allowed} bytes")
    return dict(**info, budget_bytes=allowed, required_bytes=required, reserve_bytes=reserve_bytes,
                pending_bytes=pending_bytes, max_memory_bytes=max_bytes, available_check_bypassed=force)


class ResidentReservation:
    def __init__(self, run_dir: Path, name: str, total_bytes: int, buffer_bytes: int, *,
                 max_resident_bytes=None, fraction=.9, max_bytes=None, reserve_bytes=256 * 1024 * 1024, force=False):
        self.run_dir, self.name, self.total, self.buffers = run_dir, name, total_bytes, buffer_bytes
        self.cap = max_resident_bytes
        self.options = dict(fraction=fraction, max_bytes=max_bytes, reserve_bytes=reserve_bytes, force=force)
        self.token, self.pid = process_token(os.getpid()), os.getpid()
        self.key = f"{self.name}:{self.pid}:{self.token}"
        self.info = {}

    def _read(self):
        data = read_json(self.run_dir / "resident.json") or {"entries": {}}
        data["entries"] = {key: row for key, row in data.get("entries", {}).items()
                           if process_matches(row["pid"], row.get("process_token"))}
        return data

    def __enter__(self):
        with FileLock(self.run_dir / "resident.lock"):
            data = self._read()
            cap = self.cap if self.cap is not None else data.get("max_resident_bytes")
            reserved = sum(row["reserved_bytes"] for row in data["entries"].values())
            required = self.total + self.buffers
            if cap is not None and reserved + required > cap:
                raise MemoryError(f"resident quota exceeded: {reserved} reserved + {required} requested > {cap}")
            pending = sum(row["reserved_bytes"] for row in data["entries"].values() if row["pending"])
            self.info = check_memory(required, pending_bytes=pending, **self.options)
            self.info["max_resident_bytes"] = cap
            data["max_resident_bytes"] = cap
            data["entries"][self.key] = dict(name=self.name, pid=self.pid, process_token=self.token,
                                             model_bytes=self.total, reserved_bytes=required, pending=True)
            atomic_json(self.run_dir / "resident.json", data)
        return self

    def ready(self):
        with FileLock(self.run_dir / "resident.lock"):
            data = self._read()
            data["entries"][self.key].update(reserved_bytes=self.total, pending=False)
            atomic_json(self.run_dir / "resident.json", data)

    def __exit__(self, *_):
        with FileLock(self.run_dir / "resident.lock"):
            data = self._read()
            data["entries"].pop(self.key, None)
            atomic_json(self.run_dir / "resident.json", data)
