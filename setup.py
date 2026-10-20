import hashlib
import os
import re
import shutil
import subprocess
import sysconfig
from pathlib import Path

from setuptools import Distribution, find_packages, setup
from setuptools.command.build_ext import build_ext
from setuptools.command.build_py import build_py

PROJECT_ROOT = Path(__file__).resolve().parent
INFINI_LIBRARY_NAMES = ("infiniops", "infiniccl", "infinirt")
NATIVE_AUXILIARY_NAMES = ("sampling.so", "libtvm_ffi.so")
INFINILM_NATIVE_ARTIFACTS = [
    "libinfinilm_runtime.so",
    "libinfinilm_runtime.so.*",
    "libinfinilm_runtime.dylib",
    "infinilm_runtime.dll",
    "libinfinilm_runtime.dll",
    "libinfiniops.so",
    "libinfiniops.so.*",
    "libinfiniops.dylib",
    "infiniops.dll",
    "libinfiniops.dll",
    "libinfiniccl.so",
    "libinfiniccl.so.*",
    "libinfiniccl.dylib",
    "infiniccl.dll",
    "libinfiniccl.dll",
    "libinfinirt.so",
    "libinfinirt.so.*",
    "libinfinirt.dylib",
    "infinirt.dll",
    "libinfinirt.dll",
]
INFINILM_EXTENSION_ARTIFACTS = [
    "_infinilm*.so",
    "_infinilm*.dylib",
    "_infinilm*.dll",
    "_infinilm*.pyd",
]
SHARED_LIBRARY_PATTERNS = (
    "lib{name}.so",
    "lib{name}.so.*",
    "lib{name}.dylib",
    "{name}.dll",
    "lib{name}.dll",
)


class BinaryDistribution(Distribution):
    def has_ext_modules(self):
        return True


def source_digest(source):
    digest = hashlib.sha256()
    with source.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.digest()


def add_runtime_source(sources, source):
    previous = sources.get(source.name)
    if previous is not None and source_digest(previous) != source_digest(source):
        raise RuntimeError(
            f"Conflicting runtime dependency `{source.name}`: {previous} and {source}"
        )
    sources.setdefault(source.name, source)


def read_elf_dependencies(source):
    with source.open("rb") as stream:
        if stream.read(4) != b"\x7fELF":
            return None
    try:
        result = subprocess.run(
            ["readelf", "-W", "-h", "-d", str(source)],
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            env={**os.environ, "LC_ALL": "C"},
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise RuntimeError(
            f"readelf could not inspect ELF dependency {source}"
        ) from error
    elf_class = re.search(
        r"^[ \t]*Class:[ \t]*(ELF32|ELF64)[ \t\r]*$", result.stdout, re.MULTILINE
    )
    machine = re.search(r"^[ \t]*Machine:[ \t]*([^\r\n]+)", result.stdout, re.MULTILINE)
    if elf_class is None or machine is None or not machine.group(1).strip():
        raise RuntimeError(f"readelf did not report ELF class/machine for {source}")
    needed = set()
    for line in result.stdout.splitlines():
        if "(NEEDED)" in line:
            match = re.search(r"\(NEEDED\)\s+Shared library:\s*\[([^\]]+)\]", line)
            if match is None:
                raise RuntimeError(
                    f"readelf reported an unrecognized DT_NEEDED for {source}"
                )
            needed.add(match.group(1))
    return (elf_class.group(1), machine.group(1).strip()), needed


def source_stack_prefix():
    explicit = os.environ.get("INFINI_ROOT")
    if explicit:
        return Path(explicit).resolve()
    raise RuntimeError(
        "INFINI_ROOT is required for native source builds. First build the pinned stack "
        "using third_party/InfiniCore/README.md, then set INFINI_ROOT to its "
        "installed prefix and configure xmake with the matching ABI/profile."
    )


def stage_runtime_dependencies(extra_staging_directory=None):
    """Stage the modern Infini libraries next to the shared runtime."""
    infini_root = source_stack_prefix()
    search_directories = (infini_root / "lib", infini_root / "lib64")
    destination = PROJECT_ROOT / "python/infinilm/_lib"
    all_sources = {}
    ops_sources = []

    for library in INFINI_LIBRARY_NAMES:
        sources = {}
        for directory in search_directories:
            for pattern in SHARED_LIBRARY_PATTERNS:
                for source in sorted(directory.glob(pattern.format(name=library))):
                    if source.is_file():
                        add_runtime_source(sources, source)

        if not sources:
            searched = ", ".join(str(directory) for directory in search_directories)
            raise FileNotFoundError(
                f"Could not find the `{library}` shared library in {searched}. "
                "Set `INFINI_ROOT` to an installed Infini stack."
            )

        all_sources.update(sources)
        if library == "infiniops":
            ops_sources.extend(sources.values())

    required_aux = {}
    for source in ops_sources:
        elf = read_elf_dependencies(source)
        if elf is not None:
            identity, needed = elf
            for name in NATIVE_AUXILIARY_NAMES:
                if name in needed:
                    required_aux.setdefault(name, set()).add(identity)

    for name, identities in required_aux.items():
        sources = {}
        for directory in search_directories:
            source = directory / name
            if source.is_file():
                add_runtime_source(sources, source)
        if not sources:
            raise FileNotFoundError(
                f"Ops requires `{name}` in {infini_root}/lib or lib64; "
                "old staging cannot supply a missing prefix dependency."
            )
        source = sources[name]
        elf = read_elf_dependencies(source)
        if elf is None or any(elf[0] != identity for identity in identities):
            raise RuntimeError(
                f"ELF class/machine mismatch for Ops dependency {source}"
            )
        all_sources.update(sources)

    staging_directories = [destination]
    if extra_staging_directory is not None:
        staging_directories.append(Path(extra_staging_directory))
    for directory in staging_directories:
        if directory.is_symlink() or (directory.exists() and not directory.is_dir()):
            raise RuntimeError(
                f"Use a clean, non-symlink runtime staging directory: {directory}"
            )
        for name in NATIVE_AUXILIARY_NAMES:
            staged = directory / name
            if name not in required_aux and (staged.exists() or staged.is_symlink()):
                raise RuntimeError(
                    f"Unneeded native dependency {staged}; use clean staging before packaging."
                )
        for name in all_sources:
            staged = directory / name
            if staged.is_symlink() or (staged.exists() and not staged.is_file()):
                raise RuntimeError(f"Use a clean runtime staging file: {staged}")

    # No dependency is copied until the complete prefix and staging preflight passes.
    license_source = infini_root / "share/licenses/InfiniOps"
    if (
        not (license_source / "LICENSE").is_file()
        or not (license_source / "THIRD_PARTY.txt").is_file()
    ):
        raise FileNotFoundError(
            f"Missing Ops redistribution notices in {license_source}; "
            "install the pinned InfiniOps SDK before packaging native libraries."
        )
    shutil.copytree(
        license_source,
        PROJECT_ROOT / "python/infinilm/licenses/InfiniOps",
        dirs_exist_ok=True,
    )
    destination.mkdir(parents=True, exist_ok=True)
    for source in all_sources.values():
        shutil.copy2(source, destination / source.name)


def reject_legacy_native_artifacts():
    for suffix in ("so", "dylib", "dll", "pyd"):
        if any((PROJECT_ROOT / "python").rglob("_infinicore*." + suffix)):
            raise RuntimeError("stale _infinicore native artifact must not be packaged")


def root_extension():
    reject_legacy_native_artifacts()
    suffix = sysconfig.get_config_var("EXT_SUFFIX")
    if not isinstance(suffix, str) or not suffix:
        raise RuntimeError("Python EXT_SUFFIX is required")
    source = PROJECT_ROOT / "python" / ("_infinilm" + suffix)
    candidates = {
        p
        for pattern in INFINILM_EXTENSION_ARTIFACTS
        for p in (PROJECT_ROOT / "python").glob(pattern)
    }
    if candidates != {source} or not source.is_file() or source.is_symlink():
        raise RuntimeError(
            "exactly one regular root _infinilm EXT_SUFFIX artifact is required"
        )
    return source


def build_cpp_module(extra_staging_directory=None):
    """Build and install the C++ extension modules."""
    reject_legacy_native_artifacts()
    stage_runtime_dependencies(extra_staging_directory)
    for target in ("infinilm_runtime", "_infinilm"):
        subprocess.run(["xmake", "build", target], check=True)
        subprocess.run(["xmake", "install", target], check=True)
    root_extension()


class BuildPy(build_py):
    def run(self):
        if not self.editable_mode:
            build_cpp_module(Path(self.build_lib) / "infinilm/_lib")
        super().run()
        if not self.editable_mode:
            destination = Path(self.build_lib) / root_extension().name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(root_extension(), destination)

    def get_outputs(self, include_bytecode=1):
        outputs = super().get_outputs(include_bytecode)
        native = str(
            Path(self.build_lib)
            / ("_infinilm" + sysconfig.get_config_var("EXT_SUFFIX"))
        )
        if native not in outputs:
            outputs.append(native)
        return outputs

    def get_output_mapping(self):
        mapping = super().get_output_mapping()
        name = "_infinilm" + sysconfig.get_config_var("EXT_SUFFIX")
        mapping[str(Path(self.build_lib) / name)] = str(PROJECT_ROOT / "python" / name)
        return mapping


class BuildExt(build_ext):
    def run(self):
        if self.editable_mode:
            build_cpp_module()
        super().run()


setup(
    package_dir={"": "python"},
    packages=find_packages(where="python"),
    package_data={
        "infinilm": [
            "licenses/*.txt",
            "licenses/**/*",
            *[
                "_lib/" + pattern
                for pattern in (*INFINILM_NATIVE_ARTIFACTS, *NATIVE_AUXILIARY_NAMES)
            ],
        ],
    },
    include_package_data=True,
    distclass=BinaryDistribution,
    zip_safe=False,
    cmdclass={
        "build_ext": BuildExt,
        "build_py": BuildPy,
    },
)
