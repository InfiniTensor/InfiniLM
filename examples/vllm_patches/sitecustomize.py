"""
Auto-patch HF dynamic modules for vLLM workers.

This file is imported automatically by Python if its directory is on PYTHONPATH.
We use it to monkeypatch `transformers.dynamic_module_utils` so that when
Transformers copies remote-code modules into `HF_MODULES_CACHE`, we can apply a
small compatibility patch for MiniCPM5 MoE running under vLLM's
TransformersMoEForCausalLM backend.

The same ``modeling_minicpm.py`` may live under the model directory (e.g.
``$MODEL/modeling_minicpm.py``); if ``MODEL`` is set, we patch that copy on disk
as well so workers do not depend only on the HF cache path.

Usage (must be in the environment inherited by vLLM EngineCore workers):

  export PYTHONPATH="$REPO/InfiniLM/examples/vllm_patches:${PYTHONPATH:-}"
"""

from __future__ import annotations

import importlib
import importlib.abc
import os
import sys
from pathlib import Path

# Child interpreters (vLLM EngineCore) must disable Dynamo before torch imports remote HF code.
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")


def _invalidate_module_pycache(py_file: Path) -> None:
    pycache = py_file.parent / "__pycache__"
    if not pycache.is_dir():
        return
    try:
        import shutil

        shutil.rmtree(pycache, ignore_errors=True)
    except Exception:
        pass


_PATCH_SENTINEL_LINE = "        # vllm_autopatch_minicpm5_moe"
_MOE_NEEDLE = (
    "        final_hidden_states = torch.zeros_like(hidden_states, dtype=topk_weights.dtype)"
)
# vLLM replaces ``self.experts`` with ``TransformersFusedMoE`` (defined under ``vllm.*``). The HF scatter path
# uses ``len(self.experts)`` / indexing; that breaks on fused MoE. Route by expert type's module, not ``len``.
_VLLM_MOE_GUARD_MARKER = "_vllm_moe_ex_t = type(self.experts)"
_MOE_GUARD_PREFIX = (
    "        # vllm_autopatch_minicpm5_moe\n"
    f"        {_VLLM_MOE_GUARD_MARKER}\n"
    "        if \"vllm\" in getattr(_vllm_moe_ex_t, \"__module__\", \"\"):\n"
    "            return self.experts(hidden_states, topk_indices, topk_weights).type(hidden_states.dtype)\n"
    "\n"
)
_ONE_HOT_LEN = "        expert_mask = torch.nn.functional.one_hot(topk_indices, num_classes=len(self.experts))"
_ONE_HOT_N = "        expert_mask = torch.nn.functional.one_hot(topk_indices, num_classes=_n_routed_experts)"
_FOR_LEN = "        for expert_idx in range(len(self.experts)):"
_FOR_N = "        for expert_idx in range(_n_routed_experts):"


def _strip_minicpm5_vllm_autopatch(text: str) -> str:
    """Remove every legacy autopatch block: from our sentinel through the line before ``_MOE_NEEDLE``."""
    lines = text.splitlines(keepends=True)
    out: list[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.rstrip("\r\n") == _PATCH_SENTINEL_LINE:
            j = i + 1
            found = False
            while j < len(lines) and (j - i) <= 80:
                if lines[j].startswith(_MOE_NEEDLE):
                    found = True
                    break
                j += 1
            if found:
                i = j
                continue
        out.append(line)
        i += 1
    s = "".join(out)
    s = s.replace(_ONE_HOT_N, _ONE_HOT_LEN)
    s = s.replace(_FOR_N, _FOR_LEN)
    return s


def _patch_minicpm5_moe_modeling(path: str | os.PathLike) -> None:
    p = Path(path)
    if p.name != "modeling_minicpm.py":
        return

    try:
        text = p.read_text(encoding="utf-8")
    except Exception:
        return

    if "MiniCPM5MoEMoE" not in text or _MOE_NEEDLE not in text:
        return

    if _PATCH_SENTINEL_LINE in text and _VLLM_MOE_GUARD_MARKER in text:
        _invalidate_module_pycache(p)
        return

    cleaned = _strip_minicpm5_vllm_autopatch(text)
    want = _MOE_GUARD_PREFIX + _MOE_NEEDLE
    patched = cleaned.replace(_MOE_NEEDLE, want, 1)
    if patched == cleaned:
        return

    bak = p.with_suffix(p.suffix + ".bak_autopatch")
    try:
        if not bak.exists():
            bak.write_text(text, encoding="utf-8")
    except Exception:
        pass

    try:
        p.write_text(patched, encoding="utf-8")
        _invalidate_module_pycache(p)
    except Exception:
        return


def _install_transformers_copy_hook() -> None:
    try:
        import transformers.dynamic_module_utils as dmu
    except Exception:
        return

    # Only patch once.
    if getattr(dmu, "_vllm_autopatch_installed", False):
        return

    orig_copy = dmu.shutil.copy

    def copy(src, dst, *args, **kwargs):  # type: ignore[no-untyped-def]
        out = orig_copy(src, dst, *args, **kwargs)
        try:
            _patch_minicpm5_moe_modeling(dst)
        finally:
            return out

    dmu.shutil.copy = copy  # type: ignore[assignment]
    dmu._vllm_autopatch_installed = True  # type: ignore[attr-defined]


_install_transformers_copy_hook()


# --- Import-time patch (more reliable than file rewrite) ---

_TARGET_SUBSTR = "transformers_modules.minicpm5_dot_16a3_dot_v0314.modeling_minicpm"


def _patch_minicpm5_moe_in_memory(module) -> None:  # type: ignore[no-untyped-def]
    cls = getattr(module, "MiniCPM5MoEMoE", None)
    if cls is None:
        return
    if getattr(cls, "_vllm_autopatch_applied", False):
        return

    orig = getattr(cls, "moe", None)
    if orig is None:
        return

    def moe(self, hidden_states, topk_indices, topk_weights):  # type: ignore[no-untyped-def]
        t = type(self.experts)
        if "vllm" in getattr(t, "__module__", ""):
            return self.experts(hidden_states, topk_indices, topk_weights).type(
                hidden_states.dtype
            )
        return orig(self, hidden_states, topk_indices, topk_weights)

    setattr(cls, "moe", moe)
    setattr(cls, "_vllm_autopatch_applied", True)


class _PatchedLoader(importlib.abc.Loader):
    def __init__(self, base_loader):
        self._base = base_loader

    def create_module(self, spec):  # type: ignore[no-untyped-def]
        if hasattr(self._base, "create_module"):
            return self._base.create_module(spec)
        return None

    def exec_module(self, module):  # type: ignore[no-untyped-def]
        self._base.exec_module(module)
        try:
            _patch_minicpm5_moe_in_memory(module)
        except Exception:
            pass


class _PatchFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):  # type: ignore[no-untyped-def]
        if _TARGET_SUBSTR not in fullname:
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path, target)
        if spec is None or spec.loader is None:
            return spec
        spec.loader = _PatchedLoader(spec.loader)
        return spec


if not any(isinstance(f, _PatchFinder) for f in sys.meta_path):
    sys.meta_path.insert(0, _PatchFinder())


def _install_get_class_in_module_hook() -> None:
    """Patch at import time (spec_from_file_location path), which bypasses meta_path."""
    try:
        import transformers.dynamic_module_utils as dmu
    except Exception:
        return

    if getattr(dmu, "_vllm_autopatch_get_class_hook", False):
        return

    orig = dmu.get_class_in_module

    def get_class_in_module(class_name, module_path, *, force_reload=False):  # type: ignore[no-untyped-def]
        cls = orig(class_name, module_path, force_reload=force_reload)
        try:
            norm = os.path.normpath(module_path)
            name = norm.removesuffix(".py").replace(os.path.sep, ".")
            mod = sys.modules.get(name)
            if mod is not None and _TARGET_SUBSTR in name:
                _patch_minicpm5_moe_in_memory(mod)
            else:
                # vLLM / HF often load remote-code modules by filesystem path; `name` is then
                # e.g. `.root.cache.huggingface...` and never contains `transformers_modules...`.
                if "minicpm5" in norm.lower() and norm.endswith("modeling_minicpm.py"):
                    for cand in sys.modules.values():
                        if cand is None:
                            continue
                        mf = getattr(cand, "__file__", None) or ""
                        if mf and os.path.normpath(mf) == norm and hasattr(
                            cand, "MiniCPM5MoEMoE"
                        ):
                            _patch_minicpm5_moe_in_memory(cand)
                            break
        except Exception:
            pass
        return cls

    dmu.get_class_in_module = get_class_in_module  # type: ignore[assignment]
    dmu._vllm_autopatch_get_class_hook = True  # type: ignore[attr-defined]


_install_get_class_in_module_hook()


def _scan_and_patch_minicpm5_modeling_on_disk() -> None:
    """vLLM EngineCore workers often start via ``spawn``; hooks must not depend on import order."""
    seen: set[str] = set()
    model_root = os.environ.get("MODEL", "").strip()
    if model_root:
        mp = os.path.join(os.path.expanduser(model_root), "modeling_minicpm.py")
        if os.path.isfile(mp):
            seen.add(os.path.realpath(mp))
            _patch_minicpm5_moe_modeling(mp)

    hf_home = os.environ.get(
        "HF_HOME", os.path.join(os.path.expanduser("~"), ".cache", "huggingface")
    )
    roots = [
        os.path.join(hf_home, "modules", "transformers_modules"),
    ]
    for base in roots:
        if not os.path.isdir(base):
            continue
        for dirpath, _dirnames, filenames in os.walk(base):
            if "modeling_minicpm.py" not in filenames:
                continue
            if "minicpm5" not in dirpath.lower():
                continue
            p = os.path.join(dirpath, "modeling_minicpm.py")
            rp = os.path.realpath(p)
            if rp in seen:
                continue
            seen.add(rp)
            _patch_minicpm5_moe_modeling(p)


_scan_and_patch_minicpm5_modeling_on_disk()
