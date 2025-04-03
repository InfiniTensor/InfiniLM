import os
import subprocess
import platform
import sys

PROJECT_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "test", "infiniop")
)
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


def run_tests(args):
    failed = []
    for test in [
        "gemm.py",
        "rms_norm.py",
        "causal_softmax.py",
        "swiglu.py",
        "random_sample.py",
    ]:
        result = subprocess.run(
            f"python {test} {args}", text=True, encoding="utf-8", shell=True
        )
        if result.returncode != 0:
            failed.append(test)

    return failed


if __name__ == "__main__":
    set_env()
    failed = run_tests(" ".join(sys.argv[1:]))
    if len(failed) == 0:
        print("\033[92mAll tests passed!\033[0m")
    else:
        print("\033[91mThe following tests failed:\033[0m")
        for test in failed:
            print(f"\033[91m - {test}\033[0m")
    exit(len(failed))
