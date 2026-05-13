import os
import json
from typing import Dict, Union, Optional, List
import time
import torch
from safetensors import safe_open
import glob
from tqdm import tqdm
import infinicore


def _get_scale_emb(model_path: str) -> float:
    config_path = os.path.join(model_path, "config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"config.json not found at {config_path}")
    with open(config_path, "r") as f:
        config = json.load(f)
    if config.get("model_type") != "fm9g":
        return 1.0
    return config.get("scale_emb", 1.0)


def parse_dtype(dtype_str: str):
    if dtype_str == "float32":
        return infinicore.float32
    elif dtype_str == "float16":
        return infinicore.float16
    elif dtype_str == "bfloat16":
        return infinicore.bfloat16
    elif dtype_str == "int8":
        return infinicore.int8
    elif dtype_str == "int32":
        return infinicore.int32
    elif dtype_str == "int64":
        return infinicore.int64
    else:
        raise ValueError(f"Unknown dtype string: {dtype_str}")


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


def check_parameters(model_keys: list, already_loaded_keys: list):
    model_keys = set(model_keys)
    already_loaded_keys = set(already_loaded_keys)
    intersection = model_keys & already_loaded_keys

    missing_keys = model_keys - intersection
    unexpected_keys = already_loaded_keys - intersection
    error_msgs: list[str] = []

    if len(unexpected_keys) > 0:
        error_msgs.append(
            "Unexpected key(s) in state_dict: {}. ".format(
                ", ".join('"{}"'.format(k) for k in unexpected_keys)
            )
        )
    if len(missing_keys) > 0:
        error_msgs.append(
            "Missing key(s) in state_dict: {}. ".format(
                ", ".join('"{}"'.format(k) for k in missing_keys)
            )
        )

    if len(error_msgs) > 0:
        raise RuntimeError(
            "Error(s) in loading state_dict\n\t{}".format("\n\t".join(error_msgs))
        )


def load_state_dict(
    checkpoint_file: Union[str, os.PathLike], device="cpu", dtype=torch.bfloat16
) -> Dict[str, torch.Tensor]:
    """
    Reads a `safetensor` checkpoint file. We load the checkpoint on "cpu" by default.
    """

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
            state_dict[k] = f.get_tensor(k).to(device=device)

    return state_dict


def get_model_state_dict(
    model_path: str,
    device: infinicore.device,
    dtype=infinicore.dtype,
) -> Dict[str, infinicore.Tensor]:
    """
    Load the model weights.
    """

    print(" read weights ......")
    t1 = time.time()

    torch_device = device.type
    torch_dtype = infinicore.utils.to_torch_dtype(dtype)

    # --------------------------------------------------------- #
    #          Load weights from  all *.safetensors files
    # --------------------------------------------------------- #
    model_param = {}
    for file_path in glob.glob(os.path.join(model_path, "*.safetensors")):
        model_param.update(
            load_state_dict(file_path, device=torch_device, dtype=torch_dtype)
        )

    # Apply scale_emb for fm9g models (embed_tokens uses lookup, not GEMM)
    scale_emb = _get_scale_emb(model_path)
    embed_tokens_unscaled = None
    if "model.embed_tokens.weight" in model_param:
        embed_tokens_unscaled = model_param["model.embed_tokens.weight"]
        if scale_emb != 1.0:
            model_param["model.embed_tokens.weight"] = embed_tokens_unscaled * float(
                scale_emb
            )

    if model_param.get("lm_head.weight", None) is None:
        # Use unscaled weight for lm_head (C++ alpha handles dim_model_base scaling)
        if embed_tokens_unscaled is not None:
            model_param["lm_head.weight"] = embed_tokens_unscaled

    # --------------------------------------------------------- #
    #         model_param_infini references torch.Tensor
    # --------------------------------------------------------- #
    model_param_infini = {}
    for key in model_param.keys():
        model_param_infini[key] = infinicore.from_torch(model_param[key])

    t2 = time.time()
    print(f" read weights over! {(t2 - t1) * 1000} ms \n")
    return model_param_infini


def load_model_state_dict_by_file(
    model: infinicore.nn.Module,
    model_path: str,
    dtype=infinicore.dtype,
) -> Dict[str, infinicore.Tensor]:
    """
    Load the model weights from file.
    """
    print(" load weights ......")
    t1 = time.time()

    model_type = model.hf_config.get("model_type", "")

    torch_device = "cpu"
    torch_dtype = infinicore.utils.to_torch_dtype(dtype)
    model_keys = model.state_dict_keyname()
    scale_emb = _get_scale_emb(model_path)

    already_loaded_keys = []
    embed_tokens_torch_unscaled = None

    file_list = glob.glob(os.path.join(model_path, "*.safetensors"))
    if len(file_list) > 0:
        for file_path in tqdm(file_list, desc="Processing files"):
            tqdm.write(f"Processing: {os.path.basename(file_path)}")

            # --------------------------------------------------------- #
            #          Load weights from *.safetensors file
            # --------------------------------------------------------- #
            model_param = load_state_dict(
                file_path, device=torch_device, dtype=torch_dtype
            )

            # Apply model-specific weight remapping
            remapper = _WEIGHT_REMAPPER.get(model_type)
            if remapper is not None:
                model_param = remapper(model_param)

            already_loaded_keys.extend(model_param.keys())

            # --------------------------------------------------------- #
            #         Scale embed_tokens on torch side before converting
            # --------------------------------------------------------- #
            if "model.embed_tokens.weight" in model_param:
                embed_tokens_torch_unscaled = model_param["model.embed_tokens.weight"]
                if scale_emb != 1.0:
                    model_param["model.embed_tokens.weight"] = (
                        embed_tokens_torch_unscaled * float(scale_emb)
                    )

            # --------------------------------------------------------- #
            #         model_param_infini references torch.Tensor
            # --------------------------------------------------------- #
            model_param_infini = {}
            for key in model_param.keys():
                model_param_infini[key] = infinicore.from_torch(model_param[key])
            model.load_state_dict(model_param_infini, strict=False)
            infinicore.sync_device()
        model.process_weights_after_loading()

    elif os.path.exists(os.path.join(model_path, "pytorch_model.bin")):
        file_path = os.path.join(model_path, "pytorch_model.bin")
        model_params = torch.load(file_path, weights_only=True, map_location="cpu")

        # Apply model-specific weight remapping
        remapper = _WEIGHT_REMAPPER.get(model_type)
        if remapper is not None:
            model_params = remapper(model_params)

        # Scale embed_tokens on torch side before converting
        if "model.embed_tokens.weight" in model_params:
            embed_tokens_torch_unscaled = model_params["model.embed_tokens.weight"].to(
                dtype=torch_dtype
            )
            if scale_emb != 1.0:
                model_params["model.embed_tokens.weight"] = (
                    embed_tokens_torch_unscaled * float(scale_emb)
                )

        model_param_infini = {}
        for key in model_params.keys():
            model_param_infini[key] = infinicore.from_torch(
                model_params[key].to(dtype=torch_dtype)
            )
            already_loaded_keys.append(key)

        model.load_state_dict(model_param_infini, strict=True)
        infinicore.sync_device()
    else:
        raise KeyError("Weight file not found.")

    # Handle tied weights: if lm_head.weight is missing, share embed_tokens.weight
    # Use unscaled weight for lm_head (C++ alpha handles dim_model_base scaling)
    if "lm_head.weight" in model_keys and "lm_head.weight" not in already_loaded_keys:
        if embed_tokens_torch_unscaled is not None:
            lm_head_tensor = infinicore.from_torch(embed_tokens_torch_unscaled)
            model.load_state_dict({"lm_head.weight": lm_head_tensor}, strict=False)
            already_loaded_keys.append("lm_head.weight")

    check_parameters(model_keys, already_loaded_keys)

    t2 = time.time()
    print(f" load weights over! {(t2 - t1) * 1000} ms \n")


def load_model_state_dict_by_tensor(
    model: infinicore.nn.Module,
    model_path: str,
    dtype=infinicore.dtype,
):
    """
    Load the model weights by tensor.
    """

    print(" load weights ......")
    t1 = time.time()

    torch_dtype = infinicore.utils.to_torch_dtype(dtype)
    model_keys = model.state_dict_keyname()
    scale_emb = _get_scale_emb(model_path)
    already_loaded_keys = []
    embed_tokens_torch_unscaled = None

    file_list = glob.glob(os.path.join(model_path, "*.safetensors"))
    if len(file_list) > 0:
        for file_path in tqdm(file_list, desc="Processing files"):
            tqdm.write(f"Processing: {os.path.basename(file_path)}")

            with safe_open(file_path, "pt", "cpu") as f:
                for name in f.keys():
                    tensor = f.get_tensor(name).to(dtype=torch_dtype)

                    if name == "model.embed_tokens.weight":
                        embed_tokens_torch_unscaled = tensor
                        if scale_emb != 1.0:
                            tensor = tensor * float(scale_emb)

                    weight_infini = infinicore.from_torch(tensor)
                    model.load_param(name, weight_infini)
                    already_loaded_keys.append(name)
                    infinicore.sync_stream()

    elif os.path.exists(os.path.join(model_path, "pytorch_model.bin")):
        file_path = os.path.join(model_path, "pytorch_model.bin")
        model_params = torch.load(file_path, weights_only=True, map_location="cpu")

        for key in model_params.keys():
            tensor = model_params[key].to(dtype=torch_dtype)
            if key == "model.embed_tokens.weight":
                embed_tokens_torch_unscaled = tensor
                if scale_emb != 1.0:
                    tensor = tensor * float(scale_emb)
            weight_infini = infinicore.from_torch(tensor)
            model.load_param(key, weight_infini)
            already_loaded_keys.append(key)
    else:
        raise KeyError("Weight file not found.")

    # Handle tied weights: if lm_head.weight is missing, share embed_tokens.weight
    # Use unscaled weight for lm_head (C++ alpha handles dim_model_base scaling)
    if "lm_head.weight" in model_keys and "lm_head.weight" not in already_loaded_keys:
        if embed_tokens_torch_unscaled is not None:
            lm_head_tensor = infinicore.from_torch(embed_tokens_torch_unscaled)
            model.load_param("lm_head.weight", lm_head_tensor)
            already_loaded_keys.append("lm_head.weight")

    check_parameters(model_keys, already_loaded_keys)

    t2 = time.time()
    print(f" load weights over! {(t2 - t1) * 1000} ms \n")


# ============================================================================
# Common weight transformation utilities
# ============================================================================


def split_fused_weight(
    state_dict: Dict[str, torch.Tensor],
    fused_key: str,
    output_names: List[str],
    split_dim: int = 0,
    split_ratios: Optional[List[float]] = None,
) -> Dict[str, torch.Tensor]:
    """Split fused weight tensors into separate weights.

    Args:
        state_dict: Original state dict from HuggingFace safetensors.
        fused_key: Substring to match in key names (e.g. "gate_up_proj").
        output_names: Names of the split outputs (e.g. ["gate_proj", "up_proj"]).
        split_dim: Dimension along which to split. Default 0.
        split_ratios: Optional ratios. If None, split equally.

    Returns:
        New state dict with fused keys replaced by split keys.
    """
    result = {}
    for key, tensor in state_dict.items():
        if fused_key not in key:
            result[key] = tensor
            continue

        base_key = key.replace(f".{fused_key}.weight", "")
        dim_size = tensor.shape[split_dim]
        num_splits = len(output_names)

        if split_ratios is not None:
            total_ratio = sum(split_ratios)
            sizes = [int(dim_size * r / total_ratio) for r in split_ratios[:-1]]
            sizes.append(dim_size - sum(sizes))
        else:
            chunk = dim_size // num_splits
            sizes = [chunk] * (num_splits - 1)
            sizes.append(dim_size - chunk * (num_splits - 1))

        splits = torch.split(tensor, sizes, dim=split_dim)
        for name, split_tensor in zip(output_names, splits):
            result[f"{base_key}.{name}.weight"] = split_tensor

    return result


def rename_keys(
    state_dict: Dict[str, torch.Tensor],
    mapping: Dict[str, str],
) -> Dict[str, torch.Tensor]:
    """Rename weight keys according to a substring mapping."""
    result = {}
    for key, tensor in state_dict.items():
        new_key = key
        for old_str, new_str in mapping.items():
            new_key = new_key.replace(old_str, new_str)
        result[new_key] = tensor
    return result


# ============================================================================
# Model-specific remap functions
# ============================================================================


def _remap_glm4(state_dict):
    """Split GLM-4 fused gate_up_proj into gate_proj + up_proj."""
    return split_fused_weight(
        state_dict,
        fused_key="gate_up_proj",
        output_names=["gate_proj", "up_proj"],
    )


# Add more model remap functions here as needed:
#
# def _remap_qwen3(state_dict):
#     state_dict = split_fused_weight(state_dict, "gate_up_proj", ["gate_proj", "up_proj"])
#     state_dict = rename_keys(state_dict, {"model.layers": "decoder.layers"})
#     return state_dict

# Model type → remap function mapping
_WEIGHT_REMAPPER = {
    "glm4": _remap_glm4,
    # "qwen3": _remap_qwen3,
}
