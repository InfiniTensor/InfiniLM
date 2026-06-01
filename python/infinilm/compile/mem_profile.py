# Copyright (c) 2025, InfiniCore
"""Optional mx-smi GPU memory snapshots for CUDAGraph init/replay profiling."""

from __future__ import annotations

import json
import logging
import os
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

_PROFILED: set[str] = set()


def mem_profile_enabled() -> bool:
    return os.environ.get("INFINI_PREFILL_MEM_PROFILE", "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def mem_profile_log_path() -> Optional[Path]:
    raw = os.environ.get("INFINI_PREFILL_MEM_PROFILE_LOG", "").strip()
    if not raw:
        return None
    return Path(raw)


def _gpu_index() -> int:
    raw = os.environ.get("MACA_VISIBLE_DEVICES", "0").strip()
    first = raw.split(",")[0].strip()
    return int(first) if first else 0


def _parse_mx_smi_gpu_line(text: str, gpu_index: int) -> Optional[Dict[str, Any]]:
    """Parse Memory-Usage from mx-smi table for the given logical GPU index."""
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if f"| {gpu_index}     MetaX" in line or f"| {gpu_index}    MetaX" in line:
            mem_line = lines[i + 1] if i + 1 < len(lines) else ""
            if "MiB" not in mem_line:
                return None
            parts = mem_line.split("|")
            if len(parts) < 4:
                return None
            mem_field = parts[3].strip()
            if "/" not in mem_field:
                return None
            used_s, total_s = mem_field.split("/", 1)
            used_mib = int(used_s.strip().split()[0])
            total_mib = int(total_s.strip().split()[0])
            return {
                "gpu_index": gpu_index,
                "used_mib": used_mib,
                "total_mib": total_mib,
                "free_mib": total_mib - used_mib,
                "used_gib": round(used_mib / 1024, 2),
                "free_gib": round((total_mib - used_mib) / 1024, 2),
            }
    return None


def snapshot_gpu_mem(checkpoint: str, *, once: bool = False) -> Optional[Dict[str, Any]]:
    """Log mx-smi memory for checkpoint T0–T5 when ``INFINI_PREFILL_MEM_PROFILE=1``."""
    if not mem_profile_enabled():
        return None
    if once and checkpoint in _PROFILED:
        return None

    gpu_index = _gpu_index()
    try:
        text = subprocess.check_output(
            ["mx-smi"],
            stderr=subprocess.STDOUT,
            text=True,
            timeout=30,
        )
    except (subprocess.SubprocessError, FileNotFoundError) as exc:
        logger.warning("mem_profile %s: mx-smi failed: %s", checkpoint, exc)
        return None

    parsed = _parse_mx_smi_gpu_line(text, gpu_index)
    record: Dict[str, Any] = {
        "checkpoint": checkpoint,
        "ts": time.time(),
        "gpu_index": gpu_index,
    }
    if parsed:
        record.update(parsed)

    msg = (
        f"mem_profile {checkpoint}: GPU{gpu_index} "
        f"used={record.get('used_mib', '?')}/{record.get('total_mib', '?')} MiB "
        f"(free={record.get('free_mib', '?')} MiB)"
    )
    logger.info(msg)

    log_path = mem_profile_log_path()
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

    _PROFILED.add(checkpoint)
    return record
