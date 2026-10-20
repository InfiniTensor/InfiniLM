import os

if os.getenv("INFINILM_VLLM_FORCE_ROCM") == "1":
    try:
        import vllm.third_party.pynvml as pynvml

        class HygonNVMLError(RuntimeError):
            pass

        def disabled_nvml_init():
            raise HygonNVMLError("NVML detection disabled for a HIP-native build")

        pynvml.nvmlInit = disabled_nvml_init
    except ImportError:
        pass
