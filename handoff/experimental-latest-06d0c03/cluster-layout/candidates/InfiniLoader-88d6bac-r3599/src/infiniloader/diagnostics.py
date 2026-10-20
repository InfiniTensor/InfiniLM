from __future__ import annotations

import os
from pathlib import Path
import platform
import resource
import shutil
import socket
import subprocess
import sys
import tempfile

from .memory import memory_info
from .paths import default_base_dir
from .runtime import FileLock


def diagnose(base_dir: Path | None = None) -> dict:
    checks = []
    def record(name, ok, detail):
        checks.append(dict(name=name, ok=ok, detail=str(detail)))

    record("linux", sys.platform == "linux", platform.platform())
    base = (base_dir or default_base_dir()).expanduser().resolve()
    try:
        base.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="doctor-", dir=base) as tmp:
            root = Path(tmp)
            with FileLock(root / "probe.lock"):
                try:
                    with FileLock(root / "probe.lock", timeout=0):
                        record("file_lock", False, "exclusive locks did not exclude another opener")
                except TimeoutError:
                    record("file_lock", True, "exclusive locks work")
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
                sock.bind(str(root / "probe.sock"))
                record("unix_socket", True, "socket paths under base directory are usable")
            fd = os.memfd_create("iloader-doctor", os.MFD_CLOEXEC | os.MFD_ALLOW_SEALING)
            try:
                import fcntl

                os.write(fd, b"iloader")
                fcntl.fcntl(fd, fcntl.F_ADD_SEALS,
                            fcntl.F_SEAL_WRITE | fcntl.F_SEAL_GROW | fcntl.F_SEAL_SHRINK | fcntl.F_SEAL_SEAL)
                result = subprocess.run([sys.executable, "-c", "import sys; assert open(sys.argv[1],'rb').read()==b'iloader'",
                                         f"/proc/{os.getpid()}/fd/{fd}"], capture_output=True, timeout=5)
                record("memfd_seals_and_proc_reopen", result.returncode == 0,
                       "cross-process reopening works" if result.returncode == 0 else result.stderr.decode(errors="replace"))
            finally:
                os.close(fd)
    except (OSError, RuntimeError, AttributeError, subprocess.TimeoutExpired) as exc:
        record("runtime_probe", False, exc)
    limits = dict(file_descriptors=list(resource.getrlimit(resource.RLIMIT_NOFILE)),
                  locked_memory=list(resource.getrlimit(resource.RLIMIT_MEMLOCK)))
    usage = shutil.disk_usage(base) if base.exists() else None
    warnings = ["memfd pages are not mlocked; swap and memory pressure can affect latency",
                "use a node-local base directory; containers must access the daemon PID and /proc fd paths"]
    soft = limits["file_descriptors"][0]
    if soft != resource.RLIM_INFINITY and soft < 4096:
        warnings.append(f"RLIMIT_NOFILE={soft}; checkpoints with many files may require a higher limit")
    return dict(ok=all(row["ok"] for row in checks), base_dir=str(base), checks=checks,
                memory=memory_info(), limits=limits, disk_free_bytes=usage.free if usage else None, warnings=warnings)
