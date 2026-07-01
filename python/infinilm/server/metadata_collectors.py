"""Collect build, host, and config metadata for GET /metadata."""

from __future__ import annotations

import os
import platform
import re
import subprocess
from typing import Any

_PROBE_TIMEOUT_S = 2.0

_CONFIG_ENV_EXACT = (
    "HPCC_VISIBLE_DEVICES",
    "CUDA_VISIBLE_DEVICES",
    "MACA_VISIBLE_DEVICES",
)

_IMAGE_TAG_SHA_RE = re.compile(
    r"([0-9a-f]{7,40})-([0-9a-f]{7,40})(?:-(\d{8}))?\s*$",
    re.I,
)


def _run_cmd(args: list[str]) -> str:
    try:
        return subprocess.check_output(
            args,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=_PROBE_TIMEOUT_S,
        )
    except (subprocess.SubprocessError, FileNotFoundError, OSError):
        return ""


def _read_os_release() -> dict[str, str]:
    out: dict[str, str] = {}
    try:
        with open("/etc/os-release", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line or "=" not in line:
                    continue
                key, _, val = line.partition("=")
                out[key] = val.strip().strip('"')
    except OSError:
        pass
    return out


def _probe_os() -> dict[str, str]:
    out: dict[str, str] = {}
    release = _read_os_release()
    if release.get("ID"):
        out["os_id"] = release["ID"]
    if release.get("VERSION_ID"):
        out["os_version"] = release["VERSION_ID"]
    uname = platform.uname()
    if uname.system:
        out.setdefault("os_id", uname.system.lower())
    if uname.release:
        out["kernel"] = uname.release
    if uname.machine:
        out["arch"] = uname.machine
    return out


def _probe_cpu() -> dict[str, str]:
    out: dict[str, str] = {}
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as fh:
            text = fh.read()
    except OSError:
        return out
    models: list[str] = []
    count = 0
    for line in text.splitlines():
        if line.startswith("processor"):
            count += 1
        elif line.lower().startswith("model name") or line.lower().startswith("cpu implementer"):
            _, _, val = line.partition(":")
            val = val.strip()
            if val and val not in models:
                models.append(val)
    if models:
        out["cpu_model"] = models[0]
    if count:
        out["cpu_count"] = str(count)
    return out


def _parse_smi_output(text: str) -> dict[str, str]:
    out: dict[str, str] = {}
    if not text:
        return out
    driver_m = re.search(r"Driver Version\s*:\s*(\S+)", text, re.I)
    if driver_m:
        out["gpu_driver"] = driver_m.group(1)
    models: list[str] = []
    for line in text.splitlines():
        if "MetaX" in line or "NVIDIA" in line or "GPU" in line:
            m = re.search(r"\|\s*\d+\s+(\S+(?:\s+\S+)?)\s+\|", line)
            if m:
                name = m.group(1).strip()
                if name and name not in models:
                    models.append(name)
    if models:
        out["gpu_model"] = models[0]
    gpu_lines = [ln for ln in text.splitlines() if re.search(r"\|\s*\d+\s+", ln)]
    if gpu_lines:
        out["gpu_count"] = str(len(gpu_lines))
    visible = os.environ.get("HPCC_VISIBLE_DEVICES") or os.environ.get(
        "MACA_VISIBLE_DEVICES"
    ) or os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible:
        out["gpu_visible_devices"] = visible.strip()
    return out


def _probe_gpu() -> dict[str, str]:
    for cmd in (["mx-smi"], ["ht-smi"], ["nvidia-smi"]):
        text = _run_cmd(cmd)
        parsed = _parse_smi_output(text)
        if parsed:
            return parsed
    return {}


def collect_runtime_env() -> dict[str, str]:
    """Host/runtime probe via commands and files (no config env keys)."""
    out: dict[str, str] = {}
    for probe in (_probe_os, _probe_cpu, _probe_gpu):
        for key, val in probe().items():
            if val:
                out[key] = str(val)
    return out


def collect_config_env() -> dict[str, str]:
    """Snapshot whitelisted INFINI_* and visible-device env vars."""
    out: dict[str, str] = {}
    for key, val in sorted(os.environ.items()):
        if key.startswith("INFINI_") and str(val).strip() != "":
            out[key] = val
    for key in _CONFIG_ENV_EXACT:
        val = os.environ.get(key)
        if val is not None and str(val).strip() != "":
            out[key] = val
    return out


def collect_config(startup_args: dict[str, Any]) -> dict[str, Any]:
    """Group server CLI args and INFINI env flags."""
    return {
        "startup": startup_args,
        "env": collect_config_env(),
    }


def _parse_shas_from_image_tag(image_tag: str) -> dict[str, str]:
    out: dict[str, str] = {}
    m = _IMAGE_TAG_SHA_RE.search(image_tag)
    if m:
        out["il_sha"] = m.group(1)
        out["ic_sha"] = m.group(2)
        if m.group(3):
            out["build_ts"] = m.group(3)
    return out


def collect_build_info() -> dict[str, str]:
    """Build provenance from env and IMAGE_TAG fallback."""
    out: dict[str, str] = {}
    for key in ("IL_SHA", "IC_SHA", "IO_SHA", "BUILD_TS", "IMAGE_TAG"):
        val = os.environ.get(key)
        if val is not None and str(val).strip() != "":
            out[key.lower()] = val.strip()
    build_sha = os.environ.get("INFINI_BUILD_SHA")
    if build_sha and str(build_sha).strip():
        out.setdefault("infinilm_build_sha", build_sha.strip())
    image_tag = out.get("image_tag") or os.environ.get("IMAGE_TAG", "")
    if image_tag:
        out.setdefault("image_tag", image_tag)
        parsed = _parse_shas_from_image_tag(image_tag)
        for k, v in parsed.items():
            out.setdefault(k, v)
    return out
