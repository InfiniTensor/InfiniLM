#!/usr/bin/env python3
"""Fail closed unless all eight Hygon devices are idle."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


EXPECTED_DEVICES = set(range(8))
SMI_TIMEOUT_SECONDS = 60


def _local_gpu_processes() -> list[str]:
    users: list[str] = []
    own_pid = os.getpid()
    for process_dir in Path("/proc").glob("[0-9]*"):
        try:
            pid = int(process_dir.name)
        except ValueError:
            continue
        if pid == own_pid:
            continue
        try:
            targets = [entry.resolve() for entry in (process_dir / "fd").iterdir()]
        except OSError:
            continue
        if not any(
            str(target) == "/dev/kfd" or str(target).startswith("/dev/dri/renderD")
            for target in targets
        ):
            continue
        try:
            command = (process_dir / "cmdline").read_bytes().replace(b"\0", b" ").decode(
                "utf-8", errors="replace"
            ).strip()
        except OSError:
            command = ""
        users.append(f"pid={pid} command={command or '[unknown]'}")
    return sorted(users)


def require_idle_gpu() -> None:
    try:
        result = subprocess.run(
            ["hy-smi"],
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=SMI_TIMEOUT_SECONDS,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        print(f"拒绝启动：hy-smi 空闲检查失败：{error}", file=sys.stderr)
        raise SystemExit(90) from error
    if result.returncode != 0:
        print(
            f"拒绝启动：hy-smi 退出码为 {result.returncode}。\n{result.stdout}",
            file=sys.stderr,
        )
        raise SystemExit(90)

    utilization: dict[int, tuple[float, float]] = {}
    for line in result.stdout.splitlines():
        fields = line.split()
        if (
            len(fields) >= 7
            and fields[0].isdigit()
            and fields[5].endswith("%")
            and fields[6].endswith("%")
        ):
            try:
                utilization[int(fields[0])] = (
                    float(fields[5][:-1]),
                    float(fields[6][:-1]),
                )
            except ValueError:
                continue
    if set(utilization) != EXPECTED_DEVICES:
        print(
            "拒绝启动：hy-smi 未完整报告 0-7 号设备。\n" + result.stdout,
            file=sys.stderr,
        )
        raise SystemExit(90)

    busy_devices = {
        device: values
        for device, values in utilization.items()
        if values[0] > 0.0 or values[1] > 0.0
    }
    local_users = _local_gpu_processes()
    if busy_devices or local_users:
        print(
            "拒绝启动：GPU 未完全空闲；"
            f"设备占用={busy_devices}，容器内进程={local_users}。\n{result.stdout}",
            file=sys.stderr,
        )
        raise SystemExit(90)
