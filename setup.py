import subprocess
from pathlib import Path
import os
import sys
import shutil
from setuptools import setup, find_packages
from setuptools.command.build_py import build_py
from setuptools.command.develop import develop
from setuptools.command.egg_info import egg_info
from setuptools.command.editable_wheel import editable_wheel


def find_real_cmake():
    """Find the real cmake executable, skipping Python module wrappers"""
    # First, try to find cmake that's not /usr/local/bin/cmake (which is a Python module)
    for cmake_path in ["/usr/bin/cmake", "/snap/bin/cmake", "/opt/cmake/bin/cmake"]:
        if os.path.isfile(cmake_path) and os.access(cmake_path, os.X_OK):
            try:
                result = subprocess.run(
                    [cmake_path, "--version"], capture_output=True, text=True, timeout=5
                )
                if "cmake version" in result.stdout and "Python" not in result.stdout:
                    return cmake_path
            except:
                continue

    # Try shutil.which but verify it's real cmake
    cmake_exe = shutil.which("cmake")
    if cmake_exe and cmake_exe != "/usr/local/bin/cmake":
        try:
            result = subprocess.run(
                [cmake_exe, "--version"], capture_output=True, text=True, timeout=5
            )
            if "cmake version" in result.stdout:
                return cmake_exe
        except:
            pass

    # Last resort: install cmake if not found
    print("CMake not found, attempting to install...")
    try:
        subprocess.run(["apt-get", "update"], check=True)
        subprocess.run(["apt-get", "install", "-y", "cmake"], check=True)
        cmake_exe = shutil.which("cmake")
        if cmake_exe:
            return cmake_exe
    except:
        pass

    raise RuntimeError(
        "Could not find cmake. Please install it:\n"
        "  sudo apt-get install cmake\n"
        "Or if using pip's cmake package, uninstall it:\n"
        "  pip uninstall cmake\n"
        "  sudo apt-get install cmake"
    )


def build_extension():
    """Build the C++ extension and copy to correct location"""
    project_root = Path(__file__).parent.absolute()
    build_dir = project_root / "build"
    target_dir = project_root / "python" / "infinilm" / "lib"

    # Get CMake options from environment
    cmake_options = []
    if os.environ.get("USE_KV_CACHING", "").lower() in ("1", "true", "yes", "on"):
        cmake_options.append("-DUSE_KV_CACHING=ON")
        print("Enabling KV CACHING")
    if os.environ.get("USE_CLASSIC_LLAMA", "").lower() in ("1", "true", "yes", "on"):
        cmake_options.append("-DUSE_CLASSIC_LLAMA=ON")
        print("Enabling CLASSIC LLAMA")

    # Create directories
    build_dir.mkdir(exist_ok=True)
    target_dir.mkdir(parents=True, exist_ok=True)

    # Find real cmake
    cmake_exe = find_real_cmake()
    print(f"Using cmake: {cmake_exe}")

    # Configure and build
    print(f"Configuring CMake with: {cmake_options}")
    subprocess.run(
        [cmake_exe, str(project_root)] + cmake_options, cwd=build_dir, check=True
    )

    print("Building _infinilm target...")
    subprocess.run(
        [
            cmake_exe,
            "--build",
            ".",
            "--target",
            "_infinilm",
            "-j",
            str(os.cpu_count() or 4),
        ],
        cwd=build_dir,
        check=True,
    )

    # Find and copy the .so file
    print(f"Searching for .so file in {build_dir}...")
    so_files = list(build_dir.rglob("_infinilm*.so"))

    if not so_files:
        raise RuntimeError(f"Could not find _infinilm.so in {build_dir}")

    for so_file in so_files:
        dest = target_dir / so_file.name
        print(
            f"Copying {so_file.relative_to(project_root)} -> {dest.relative_to(project_root)}"
        )
        shutil.copy2(so_file, dest)

    print("C++ extension built and installed successfully!")
    return True


class BuildPyCommand(build_py):
    def run(self):
        print("Building C++ extension...")
        build_extension()
        super().run()


class DevelopCommand(develop):
    def run(self):
        print("Building C++ extension for development install...")
        build_extension()
        super().run()


class EditableWheelCommand(editable_wheel):
    def run(self):
        print("Building C++ extension for editable wheel...")
        build_extension()
        super().run()


class EggInfoCommand(egg_info):
    def run(self):
        # Don't build extension during egg_info - it's just for metadata
        super().run()


setup(
    name="InfiniLM",
    version="0.1.0",
    description="InfiniLM model implementations",
    package_dir={"": "python"},
    packages=find_packages(where="python"),
    cmdclass={
        "build_py": BuildPyCommand,
        "develop": DevelopCommand,
        "editable_wheel": EditableWheelCommand,
        "egg_info": EggInfoCommand,
    },
    python_requires=">=3.10",
)
