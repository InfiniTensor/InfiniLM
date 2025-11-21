import os
from typing import Dict, Optional, Union

import torch
from safetensors import safe_open

# from safetensors.torch import load_file as safe_load_file
# from safetensors.torch import save_file as safe_save_file
import infinicore

str_to_torch_dtype = {
    "BOOL": torch.bool,
    "U8": torch.uint8,
    "I8": torch.int8,
    "I16": torch.int16,
    "F16": torch.float16,
    "BF16": torch.bfloat16,
    "I32": torch.int32,
    "F32": torch.float32,
    "F64": torch.float64,
    "I64": torch.int64,
    "F8_E4M3": torch.float8_e4m3fn,
    "F8_E5M2": torch.float8_e5m2,
}


def load_state_dict(
    checkpoint_file: Union[str, os.PathLike],
    map_location: Optional[Union[str, torch.device]] = "cpu",
    weights_only: bool = True,
) -> Dict[str, torch.Tensor]:
    """
    Reads a `safetensor` checkpoint file. We load the checkpoint on "cpu" by default.
    """
    # Use safetensors if possible
    if not checkpoint_file.endswith(".safetensors"):
        return {}

    state_dict = {}
    with safe_open(checkpoint_file, framework="pt") as f:
        metadata = f.metadata()
        if metadata is not None and metadata.get("format") not in [
            "pt",
            "tf",
            "flax",
            "mlx",
        ]:
            raise OSError(
                f"The safetensors archive passed at {checkpoint_file} does not contain the valid metadata."
            )

        for k in f.keys():
            if map_location == "meta":
                _slice = f.get_slice(k)
                k_dtype = _slice.get_dtype()
                if k_dtype in str_to_torch_dtype:
                    dtype = str_to_torch_dtype[k_dtype]
                else:
                    raise ValueError(
                        f"Cannot load safetensors of unknown dtype {k_dtype}"
                    )
                state_dict[k] = torch.empty(
                    size=_slice.get_shape(), dtype=dtype, device="meta"
                )
            else:
                state_dict[k] = f.get_tensor(k)

    return state_dict


def get_model_state_dict(
    model_path: str,
    device: infinicore.device,
    dtype=infinicore.dtype,
) -> Dict[str, infinicore.Tensor]:
    """
    Load the model weights.
    """
    path = os.path.join(model_path, "model.safetensors")
    model_param = load_state_dict(path)

    torch_device = device.type
    torch_dtype = infinicore.utils.to_torch_dtype(dtype)

    model_param_infini = {}
    for key, value in model_param.items():
        model_param[key] = value.to(device=torch_device, dtype=torch_dtype)

    for key, value in model_param.items():
        model_param_infini[key] = infinicore.from_torch(model_param[key])

    return model_param_infini
