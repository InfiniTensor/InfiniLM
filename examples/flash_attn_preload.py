"""
Resolve ``flash_attn_2_cuda*.so`` for the active interpreter and preload with ``RTLD_GLOBAL``.

Use the same PyTorch ABI as ``import torch``. Preloading a system image flash-attn ``.so``
while running a **venv** PyTorch often breaks decode (e.g. tensors without accessible storage).

**Installing flash-attn into the venv:** pip's default build isolation hides ``torch``, so use::

    pip install flash-attn --no-build-isolation

Resolution order:

1. ``INFINILM_FLASH_ATTN_CUDA_SO`` if set, the file exists, and it lives in the **same**
   ``site-packages`` as ``torch`` (unless ``INFINILM_ALLOW_SYSTEM_FLASH_WITH_VENV_TORCH=1``).
2. ``<site-packages>/flash_attn_2_cuda*.so`` next to the ``torch`` package.
3. Next to the importable ``flash_attn`` package **only if** that ``.so`` is under the same
   ``site-packages`` as ``torch`` (avoids ``--system-site-packages`` picking image flash).
4. If ``torch`` is under ``/usr/local/lib/python*`` or ``/usr/lib/python*``, fallback glob under
   those distro trees (image ``python3``). **Skipped** for venv / conda ``torch`` so we do not
   preload a mismatched extension.

Disable with ``INFINILM_DISABLE_FLASH_ATTN_RTLD_GLOBAL=1``.

If the chosen ``.so`` is not in the same ``site-packages`` as ``torch`` (e.g. venv torch but
``/usr/local/...`` flash), a warning is printed unless ``INFINILM_SILENCE_FLASH_ATTN_SITE_MISMATCH=1``.
Prefer ``pip install flash-attn --no-build-isolation`` into the venv; see
``FLASH_ATTN_AND_VLLM_FUSED_MOE.md``. For debug-only forcing of a system path with venv torch, set
``INFINILM_ALLOW_SYSTEM_FLASH_WITH_VENV_TORCH=1``.
"""

from __future__ import annotations

import ctypes
import glob
import os
import sys

_warned_site_mismatch = False


def strip_flash_attn_cuda_from_ld_preload() -> None:
    """Remove ``flash_attn_2_cuda*.so`` entries from ``LD_PRELOAD``.

    Shell-based subprocesses (Triton/torch helpers) inherit ``LD_PRELOAD``. Preloading the
    FlashAttention CUDA extension there forces the dynamic linker to resolve ``libtorch_python``
    in processes that are not the embedding Python interpreter, which triggers::

        undefined symbol: PyInstanceMethod_Type

    This repo resolves FlashAttention in-process via ``ctypes.CDLL(..., RTLD_GLOBAL)`` in
    :func:`maybe_load_flash_attn_global`; dropping the flash ``.so`` from ``LD_PRELOAD`` keeps
    child processes healthy while the parent still loads the extension explicitly.
    """
    raw = os.environ.get("LD_PRELOAD")
    if not raw:
        return
    chunks = [c.strip() for c in raw.split(":") if c.strip()]
    kept: list[str] = []
    for c in chunks:
        base = os.path.basename(c)
        if base.startswith("flash_attn_2_cuda") and ".so" in base:
            continue
        kept.append(c)
    if len(kept) == len(chunks):
        return
    if kept:
        os.environ["LD_PRELOAD"] = ":".join(kept)
    else:
        os.environ.pop("LD_PRELOAD", None)


def _torch_site_packages() -> str | None:
    try:
        import torch

        return os.path.normpath(os.path.dirname(os.path.dirname(torch.__file__)))
    except ImportError:
        return None


def _flash_so_in_torch_site_packages(fa_path: str, torch_sp: str) -> bool:
    fa_dir = os.path.normpath(os.path.dirname(os.path.realpath(fa_path)))
    return fa_dir == torch_sp or fa_path.startswith(torch_sp + os.sep)


def _is_system_style_torch_site_packages(torch_sp: str) -> bool:
    """True if torch looks like image/distro site-packages (safe to try /usr fallbacks)."""
    sp = os.path.normpath(torch_sp)
    return sp.startswith("/usr/local/lib/python") or sp.startswith("/usr/lib/python")


def _warn_flash_torch_site_mismatch(fa_path: str) -> None:
    global _warned_site_mismatch
    if _warned_site_mismatch:
        return
    if os.environ.get("INFINILM_SILENCE_FLASH_ATTN_SITE_MISMATCH") == "1":
        return
    if os.environ.get("INFINILM_ALLOW_SYSTEM_FLASH_WITH_VENV_TORCH") == "1":
        return
    tsp = _torch_site_packages()
    if tsp is None or _flash_so_in_torch_site_packages(fa_path, tsp):
        return
    _warned_site_mismatch = True
    print(
        "[flash_attn_preload] WARNING: flash_attn CUDA extension and torch come from different "
        "site-packages. This often breaks FlashAttention decode (tensor storage errors). "
        f"torch: {tsp!r}, flash_attn .so: {fa_path!r}. "
        "Fix: pip install flash-attn --no-build-isolation in the same venv, or use image "
        "python3 + LD_PRELOAD for flash (see InfiniLM/examples/FLASH_ATTN_AND_VLLM_FUSED_MOE.md). "
        "Silence: INFINILM_SILENCE_FLASH_ATTN_SITE_MISMATCH=1",
        file=sys.stderr,
        flush=True,
    )


def resolve_flash_attn_cuda_so() -> str | None:
    if os.environ.get("INFINILM_DISABLE_FLASH_ATTN_RTLD_GLOBAL") == "1":
        return None
    allow_mismatch = os.environ.get("INFINILM_ALLOW_SYSTEM_FLASH_WITH_VENV_TORCH") == "1"
    torch_sp: str | None = None
    try:
        import torch

        torch_sp = os.path.normpath(os.path.dirname(os.path.dirname(torch.__file__)))
    except ImportError:
        pass

    def _same_site_ok(fa_path: str) -> bool:
        if allow_mismatch or torch_sp is None:
            return True
        return _flash_so_in_torch_site_packages(fa_path, torch_sp)

    override = os.environ.get("INFINILM_FLASH_ATTN_CUDA_SO", "").strip()
    if override and os.path.isfile(override):
        if _same_site_ok(override):
            return override
        print(
            "[flash_attn_preload] ignoring INFINILM_FLASH_ATTN_CUDA_SO (not under same site-packages "
            f"as torch {torch_sp!r}); install flash-attn in the venv or set "
            "INFINILM_ALLOW_SYSTEM_FLASH_WITH_VENV_TORCH=1",
            file=sys.stderr,
            flush=True,
        )

    if torch_sp is not None:
        candidates = sorted(glob.glob(os.path.join(torch_sp, "flash_attn_2_cuda*.so")))
        if candidates:
            return candidates[0]

    if torch_sp is not None:
        try:
            import importlib.util

            spec = importlib.util.find_spec("flash_attn")
            if spec and spec.origin:
                pkg_dir = os.path.dirname(spec.origin)
                for base in (pkg_dir, os.path.dirname(pkg_dir)):
                    for cand in sorted(
                        glob.glob(os.path.join(base, "flash_attn_2_cuda*.so"))
                    ):
                        if _same_site_ok(cand):
                            return cand
        except Exception:
            pass

    if torch_sp is not None and (
        _is_system_style_torch_site_packages(torch_sp) or allow_mismatch
    ):
        for pattern in (
            "/usr/local/lib/python3.*/dist-packages/flash_attn_2_cuda*.so",
            "/usr/lib/python3.*/dist-packages/flash_attn_2_cuda*.so",
        ):
            candidates = sorted(glob.glob(pattern))
            if candidates:
                return candidates[0]
    return None


def maybe_load_flash_attn_global() -> None:
    strip_flash_attn_cuda_from_ld_preload()
    if os.environ.get("INFINILM_DISABLE_FLASH_ATTN_RTLD_GLOBAL") == "1":
        return
    fa = resolve_flash_attn_cuda_so()
    if not fa:
        return
    _warn_flash_torch_site_mismatch(fa)
    try:
        ctypes.CDLL(fa, mode=ctypes.RTLD_GLOBAL)
    except OSError:
        pass
