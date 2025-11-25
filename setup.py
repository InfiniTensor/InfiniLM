import subprocess
from pathlib import Path

from setuptools import setup
from setuptools.command.build import build
from setuptools.command.develop import develop
from setuptools.command.egg_info import egg_info


def build_cpp_module():
    """Build and install the C++ extension module"""
    subprocess.run(["xmake", "build", "_infinilm_llama"], check=True)
    subprocess.run(["xmake", "install", "_infinilm_llama"], check=True)


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
    packages=["infinilm", "infinilm.models", "infinilm.lib"],
    cmdclass={
        "build": Build,
        "develop": Develop,
        "egg_info": EggInfo,
    },
    python_requires=">=3.10",
)
