import os
import shutil
import subprocess
from pathlib import Path

from setuptools import Distribution, find_packages, setup
from setuptools.command.build import build
from setuptools.command.develop import develop
from setuptools.command.egg_info import egg_info

PROJECT_ROOT = Path(__file__).resolve().parent
INFINI_LIBRARY_NAMES = ("infiniops", "infiniccl", "infinirt")
INFINICORE_NATIVE_ARTIFACTS = [
    "_infinicore*.so",
    "_infinicore*.dylib",
    "_infinicore*.dll",
    "_infinicore*.pyd",
    "libinfinicore_runtime.so",
    "libinfinicore_runtime.so.*",
    "libinfinicore_runtime.dylib",
    "infinicore_runtime.dll",
    "libinfinicore_runtime.dll",
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


def stage_runtime_dependencies():
    """Stage the modern Infini libraries next to the shared runtime."""
    infini_root = Path(os.environ.get("INFINI_ROOT", Path.home() / ".infini"))
    search_directories = (infini_root / "lib", infini_root / "lib64")
    destination = PROJECT_ROOT / "python/infinicore/lib"
    destination.mkdir(parents=True, exist_ok=True)

    for library in INFINI_LIBRARY_NAMES:
        sources = {}
        for directory in search_directories:
            for pattern in SHARED_LIBRARY_PATTERNS:
                for source in sorted(directory.glob(pattern.format(name=library))):
                    if source.is_file():
                        sources.setdefault(source.name, source)

        if not sources:
            searched = ", ".join(str(directory) for directory in search_directories)
            raise FileNotFoundError(
                f"Could not find the `{library}` shared library in {searched}. "
                "Set `INFINI_ROOT` to an installed Infini stack."
            )

        for source in sources.values():
            shutil.copy2(source, destination / source.name)


def build_cpp_module():
    """Build and install the C++ extension modules."""
    stage_runtime_dependencies()
    for target in ("infinicore_runtime", "_infinicore", "_infinilm"):
        subprocess.run(["xmake", "build", target], check=True)
        subprocess.run(["xmake", "install", target], check=True)


class Build(build):
    def run(self):
        build_cpp_module()
        super().run()


class Develop(develop):
    def run(self):
        build_cpp_module()
        super().run()


class EggInfo(egg_info):
    def run(self):
        # Ensure C++ module is built before creating egg-info
        build_cpp_module()
        super().run()


setup(
    name="InfiniLM",
    version="0.1.0",
    description="InfiniLM model implementations",
    package_dir={"": "python"},
    packages=find_packages(where="python"),
    package_data={
        "infinicore.lib": INFINICORE_NATIVE_ARTIFACTS,
        "infinilm.lib": INFINILM_EXTENSION_ARTIFACTS,
    },
    include_package_data=True,
    distclass=BinaryDistribution,
    zip_safe=False,
    cmdclass={
        "build": Build,
        "develop": Develop,
        "egg_info": EggInfo,
    },
    python_requires=">=3.10",
)
