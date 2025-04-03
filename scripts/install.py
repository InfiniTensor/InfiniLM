import os
import subprocess
import platform
import sys

PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
os.chdir(PROJECT_DIR)


def set_env():
    if os.environ.get("INFINI_ROOT", "") == "":
        os.environ["INFINI_ROOT"] = os.path.expanduser("~/.infini")

    if platform.system() == "Windows":
        new_path = os.path.expanduser(os.environ.get("INFINI_ROOT") + "/bin")
        if new_path not in os.environ.get("PATH", ""):
            os.environ["PATH"] = f"{new_path};{os.environ.get('PATH', '')}"

    elif platform.system() == "Linux":
        new_path = os.path.expanduser(os.environ.get("INFINI_ROOT") + "/bin")
        if new_path not in os.environ.get("PATH", ""):
            os.environ["PATH"] = f"{new_path}:{os.environ.get('PATH', '')}"

        new_lib_path = os.path.expanduser(os.environ.get("INFINI_ROOT") + "/lib")
        if new_lib_path not in os.environ.get("LD_LIBRARY_PATH", ""):
            os.environ["LD_LIBRARY_PATH"] = (
                f"{new_lib_path}:{os.environ.get('LD_LIBRARY_PATH', '')}"
            )
    else:
        raise RuntimeError("Unsupported platform.")


def run_cmd(cmd):
    subprocess.run(cmd, text=True, encoding="utf-8", check=True, shell=True)


def install(xmake_config_flags=""):
    run_cmd(f"xmake f {xmake_config_flags} -cv")
    run_cmd("xmake")
    run_cmd("xmake install")
    run_cmd("xmake build infiniop-test")
    run_cmd("xmake install infiniop-test")


if __name__ == "__main__":
    set_env()
    install(" ".join(sys.argv[1:]))
