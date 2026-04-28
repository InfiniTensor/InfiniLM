import ctypes
import os
from typing import Iterable, List


def _candidate_prefixes(path: str) -> List[str]:
    """
    Return HPCC install prefixes to search for libs.
    Prefer HPCC_PATH; if absent and explicitly opted-in, fall back to /opt/hpcc.
    """
    prefixes: List[str] = []
    if path:
        prefixes.append(path)

    seen = set()
    unique: List[str] = []
    for p in prefixes:
        if p and p not in seen:
            seen.add(p)
            unique.append(p)
    return unique


def _try_load(paths: Iterable[str], name: str) -> bool:
    """Try to load a shared library from given paths or system search path."""
    for path in paths:
        full = os.path.join(path, "lib", name)
        if os.path.exists(full):
            try:
                ctypes.CDLL(full, mode=ctypes.RTLD_GLOBAL)
                return True
            except OSError:
                # Try next candidate
                continue
    # Last resort: rely on loader search path
    try:
        ctypes.CDLL(name, mode=ctypes.RTLD_GLOBAL)
        return True
    except OSError:
        return False


def preload_hpcc() -> None:
    """
    Best-effort preload of key HPCC runtime libs with RTLD_GLOBAL.

    This mirrors the behavior of torch's HPCC build that loads libtorch_global_deps.so,
    but avoids introducing a hard torch dependency. All failures are swallowed.
    """
    hpcc_path = os.getenv("HPCC_PATH")
    if not hpcc_path:
        return

    prefixes = _candidate_prefixes(hpcc_path)
    libs = [
        "libhcruntime.so",
        "libhcToolsExt.so",
        "libruntime_cu.so",
        "libhccompiler.so",
    ]

    for lib in libs:
        _try_load(prefixes, lib)


def _should_preload_device(device_type: str) -> bool:
    """
    Check if preload is needed for a specific device type.
    """
    device_env_map = {
        "METAX": ["HPCC_PATH", "INFINICORE_PRELOAD_HPCC"],  # HPCC/METAX
        # Add other device types here as needed:
        # "ASCEND": ["ASCEND_PATH"],
        # "CAMBRICON": ["NEUWARE_HOME"],
    }

    env_vars = device_env_map.get(device_type, [])
    for env_var in env_vars:
        if os.getenv(env_var):
            return True
    return False


def preload_device(device_type: str) -> None:
    """
    Preload runtime libraries for a specific device type if needed.

    Args:
        device_type: Device type name (e.g., "METAX", "ASCEND", etc.)
    """
    if device_type == "METAX":
        preload_hpcc()
    # Add other device preload functions here as needed:
    # elif device_type == "ASCEND":
    #     preload_ascend()
    # etc.


def preload() -> None:
    """
    Universal preload function that loops through device types and preloads when required.

    This function detects available device types and preloads their runtime libraries
    if the environment indicates they are needed.
    """
    # Device types that may require preload
    device_types = [
        "METAX",  # HPCC/METAX
        # Add other device types here as they are implemented:
        # "ASCEND",
        # "CAMBRICON",
        # etc.
    ]

    for device_type in device_types:
        if _should_preload_device(device_type):
            try:
                preload_device(device_type)
            except Exception:
                # Swallow all errors - preload is best-effort
                pass
