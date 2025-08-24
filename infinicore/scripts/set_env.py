import os
import platform


def set_env():
    if os.environ.get("INFINI_ROOT") == None:
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

    elif platform.system() == "Darwin":
        new_path = os.path.expanduser(os.environ.get("INFINI_ROOT") + "/bin")
        if new_path not in os.environ.get("PATH", ""):
            os.environ["PATH"] = f"{new_path}:{os.environ.get('PATH', '')}"

        new_lib_path = os.path.expanduser(os.environ.get("INFINI_ROOT") + "/lib")
        if new_lib_path not in os.environ.get("DYLD_LIBRARY_PATH", ""):
            os.environ["DYLD_LIBRARY_PATH"] = (
                f"{new_lib_path}:{os.environ.get('DYLD_LIBRARY_PATH', '')}"
            )
    else:
        raise RuntimeError("Unsupported platform.")
