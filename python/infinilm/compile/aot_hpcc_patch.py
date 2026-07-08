# Copyright (c) 2025, InfiniCore
"""HPCC/MetaX workarounds for PyTorch AOTInductor host wrapper compilation."""

from __future__ import annotations

import contextlib
import glob
import logging
import os
import re
from typing import Dict, Iterator, List, Optional

logger = logging.getLogger(__name__)

_HPCC_MARKERS = (
    "/opt/hpcc",
    "USE_HPCC",
    "cu-bridge",
    "htcc",
    "cucc",
    "mars3",
)


def is_hpcc_aot_environment() -> bool:
    """True when running under MetaX HPCC cu-bridge / htcc toolchain."""
    if os.environ.get("INFINI_PIECEWISE_AOT_HPCC", "").strip().lower() in (
        "1",
        "true",
        "yes",
    ):
        return True
    if os.environ.get("INFINI_PIECEWISE_AOT_HPCC", "").strip().lower() in (
        "0",
        "false",
        "no",
    ):
        return False
    hpcc_path = os.environ.get("HPCC_PATH", "/opt/hpcc")
    if os.path.isdir(hpcc_path):
        return True
    cxx = os.environ.get("CXX", "")
    return any(marker in cxx for marker in ("cucc", "htcc"))


def hpcc_aot_host_cxx() -> str:
    """Host compiler for AOTInductor wrapper objects (not GPU cubin)."""
    return os.environ.get("INFINI_PIECEWISE_AOT_HOST_CXX", "g++")


def _is_wrapper_build(name: str, sources: object) -> bool:
    srcs = sources if isinstance(sources, list) else [sources]
    text = " ".join(str(s) for s in srcs) + " " + str(name)
    return "wrapper" in text.lower()


def _append_unique(flags: List[str], item: str) -> None:
    if item and item not in flags:
        flags.append(item)


def write_filesystem_shim_header(artifact_dir: str) -> str:
    """Emit a header mapping std::filesystem to experimental::filesystem (GCC 10)."""
    os.makedirs(artifact_dir, exist_ok=True)
    path = os.path.join(artifact_dir, "aot_filesystem_shim.h")
    content = """#pragma once
#include <experimental/filesystem>
namespace std {
namespace filesystem = experimental::filesystem;
}
"""
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    return path


def patch_wrapper_sources(inductor_cache_dir: str) -> int:
    """
    Post-process generated AOT wrapper sources for libstdc++ without std::filesystem.

    Returns the number of files patched.
    """
    if not os.path.isdir(inductor_cache_dir):
        return 0

    patched = 0
    for path in glob.glob(
        os.path.join(inductor_cache_dir, "**", "*.wrapper.cpp"), recursive=True
    ):
        if _patch_wrapper_source_file(path):
            patched += 1

    # PyTorch 2.8 writes plain wrapper.cpp under the inductor cache tree.
    for path in glob.glob(
        os.path.join(inductor_cache_dir, "**", "wrapper.cpp"), recursive=True
    ):
        if _patch_wrapper_source_file(path):
            patched += 1
    return patched


def _patch_wrapper_source(src: str) -> str:
    if "std::filesystem::" not in src:
        return src
    return re.sub(r"\bstd::filesystem::", "std::experimental::filesystem::", src)


def _patch_wrapper_source_file(path: str) -> bool:
    if not os.path.isfile(path):
        return False
    with open(path, "r", encoding="utf-8") as f:
        src = f.read()
    new_src = _patch_wrapper_source(src)
    if new_src == src:
        return False
    with open(path, "w", encoding="utf-8") as f:
        f.write(new_src)
    logger.info("patched AOT wrapper for HPCC: %s", path)
    return True


def hpcc_aot_inductor_configs(*, artifact_dir: Optional[str] = None) -> Dict[str, object]:
    """Extra Inductor config for HPCC AOT packaging (``config.patch`` dot keys)."""
    _ = artifact_dir  # shim header is injected via CXXFLAGS in the compile profile
    return {"cpp.cxx": (None, hpcc_aot_host_cxx())}


@contextlib.contextmanager
def hpcc_aot_compile_profile(
    *, artifact_dir: Optional[str] = None
) -> Iterator[Dict[str, str]]:
    """
    Apply HPCC AOT host-wrapper compile profile for ``aoti_compile_and_package``.

    PyTorch MetaX wheels set ``USE_HPCC=True``, which forces ``cucc`` for all C++
    builds including the CPU-only AOT wrapper.  We route wrapper.cpp to ``g++`` via
    a narrow ``CppBuilder`` hook and add ``-lstdc++fs`` on the final link step.
    """
    if not is_hpcc_aot_environment():
        yield {}
        return

    host_cxx = hpcc_aot_host_cxx()
    saved_env = {
        key: os.environ.get(key)
        for key in ("CXX", "TORCHINDUCTOR_CXX", "LDFLAGS", "CXXFLAGS")
    }

    os.environ["CXX"] = host_cxx
    os.environ.setdefault("TORCHINDUCTOR_CXX", host_cxx)

    ldflags = os.environ.get("LDFLAGS", "")
    if "-lstdc++fs" not in ldflags:
        os.environ["LDFLAGS"] = f"{ldflags} -lstdc++fs".strip()

    cxxflags = os.environ.get("CXXFLAGS", "")
    if "-std=c++17" not in cxxflags and "-std=gnu++17" not in cxxflags:
        os.environ["CXXFLAGS"] = f"{cxxflags} -std=c++17".strip()

    import torch._inductor.config as inductor_config
    import torch._inductor.cpp_builder as cpp_builder

    prior_cxx = inductor_config.cpp.cxx
    inductor_config.cpp.cxx = (None, host_cxx)

    orig_builder_init = cpp_builder.CppBuilder.__init__
    orig_builder_build = cpp_builder.CppBuilder.build
    orig_device_options_init = cpp_builder.CppTorchDeviceOptions.__init__

    def _wrapper_source_paths(name, sources) -> List[str]:
        srcs = sources if isinstance(sources, list) else str(sources).split()
        return [str(s) for s in srcs if str(s).endswith(".cpp")]

    def _patch_wrapper_sources_for_build(name, sources) -> None:
        if not _is_wrapper_build(name, sources):
            return
        for path in _wrapper_source_paths(name, sources):
            _patch_wrapper_source_file(path)

    def _patched_device_options_init(self, *args, **kwargs):
        orig_device_options_init(self, *args, **kwargs)
        compile_only = kwargs.get("compile_only", False)
        precompiling = kwargs.get("precompiling", False)
        preprocessing = kwargs.get("preprocessing", False)
        if not compile_only and not precompiling and not preprocessing:
            ld = list(self.get_ldflags())
            _append_unique(ld, "lstdc++fs")
            self._ldflags = ld

    def _patched_builder_init(self, name, sources, BuildOption, output_dir=""):
        if _is_wrapper_build(name, sources):
            BuildOption._compiler = host_cxx
            _patch_wrapper_sources_for_build(name, sources)
            logger.info(
                "HPCC AOT wrapper build: host CXX=%s sources=%s",
                host_cxx,
                sources,
            )
        orig_builder_init(self, name, sources, BuildOption, output_dir)
        if _is_wrapper_build(name, sources):
            self._compiler = host_cxx

    def _patched_builder_build(self):
        paths = getattr(self, "_orig_source_paths", None) or self._sources_args
        _patch_wrapper_sources_for_build(self._name, paths)
        return orig_builder_build(self)

    cpp_builder.CppBuilder.__init__ = _patched_builder_init
    cpp_builder.CppBuilder.build = _patched_builder_build
    cpp_builder.CppTorchDeviceOptions.__init__ = _patched_device_options_init

    logger.info(
        "HPCC AOT compile profile: host CXX=%s CXXFLAGS=%s LDFLAGS=%s",
        host_cxx,
        os.environ.get("CXXFLAGS", ""),
        os.environ.get("LDFLAGS", ""),
    )
    try:
        yield {"CXX": host_cxx, "TORCHINDUCTOR_CXX": host_cxx}
    finally:
        cpp_builder.CppBuilder.__init__ = orig_builder_init
        cpp_builder.CppBuilder.build = orig_builder_build
        cpp_builder.CppTorchDeviceOptions.__init__ = orig_device_options_init
        inductor_config.cpp.cxx = prior_cxx
        for key, value in saved_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def build_aot_inductor_configs(
    *,
    artifact_dir: Optional[str] = None,
    inductor_cache_dir: Optional[str] = None,
) -> Dict[str, object]:
    """Merge HPCC host-compiler settings for ``aoti_compile_and_package``."""
    configs = hpcc_aot_inductor_configs(artifact_dir=artifact_dir)
    if inductor_cache_dir:
        patch_wrapper_sources(inductor_cache_dir)
    return configs


def apply_hpcc_aot_compile_env(*, artifact_dir: Optional[str] = None) -> Dict[str, str]:
    """Legacy helper; prefer ``hpcc_aot_compile_profile`` context manager."""
    host_cxx = hpcc_aot_host_cxx()
    os.environ["CXX"] = host_cxx
    os.environ.setdefault("TORCHINDUCTOR_CXX", host_cxx)
    if artifact_dir:
        write_filesystem_shim_header(artifact_dir)
    return {"CXX": host_cxx, "TORCHINDUCTOR_CXX": host_cxx}
