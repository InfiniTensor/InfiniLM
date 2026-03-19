import os
import ctypes


def preload_infllmv2_if_available() -> None:
    """
    Best-effort preload of the InfLLM-v2 CUDA extension into the current
    process with RTLD_GLOBAL so that libinfinicore_cpp_api.so can resolve
    mha_varlen_fwd / mha_fwd_kvcache symbols without using LD_PRELOAD.

    Safe to call multiple times and safe when infllm_v2 is not installed.
    """
    guard = os.getenv("INFINI_PRELOAD_INFLLMV2", "1")
    if guard in ("0", "", "false", "False"):
        return

    try:
        import infllm_v2  # type: ignore[import]

        so_path = getattr(infllm_v2, "__file__", None)
        base_dir = os.path.dirname(so_path or "")

        if not so_path or not so_path.endswith(".so"):
            for name in os.listdir(base_dir):
                if name.endswith(".so"):
                    so_path = os.path.join(base_dir, name)
                    break

        if not so_path or not os.path.exists(so_path):
            return

        ctypes.CDLL(so_path, mode=ctypes.RTLD_GLOBAL)
    except Exception:
        return

