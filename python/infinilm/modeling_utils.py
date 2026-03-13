import os
import json
import math
from typing import Dict, Union
import time
import torch
from safetensors import safe_open
import glob
from tqdm import tqdm
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
            # Explicitly cast dtype: some ops (e.g. embedding) may not support BF16 on all backends.
            state_dict[k] = f.get_tensor(k).to(device=device, dtype=dtype)

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

    if model_param.get("lm_head.weight", None) is None:
        model_param["lm_head.weight"] = model_param["model.embed_tokens.weight"]

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

    torch_device = "cpu"
    torch_dtype = infinicore.utils.to_torch_dtype(dtype)
    model_keys = model.state_dict_keyname()

    # MiniCPM-style scaling (used by MiniCPM / FM9G; also applies to MiniCPM-SALA checkpoints).
    # This matches `InfiniLM/scripts/jiuge.py` weight scaling behavior.
    scale_input = 1.0
    scale_output = 1.0
    scale_o = 1.0
    scale_down = 1.0
    scale_lm_head = 1.0
    try:
        with open(os.path.join(model_path, "config.json")) as f:
            cfg = json.load(f)
        if (
            cfg.get("model_type") in ["fm9g", "minicpm", "minicpm_sala"]
            and "scale_emb" in cfg
            and "scale_depth" in cfg
        ):
            scale_input = float(cfg["scale_emb"])
            scale_o = float(cfg["scale_depth"]) / math.sqrt(float(cfg["num_hidden_layers"]))
            scale_down = float(cfg["scale_depth"]) / math.sqrt(float(cfg["num_hidden_layers"]))
            if cfg.get("model_type") in ["fm9g", "minicpm"] and "dim_model_base" in cfg:
                scale_output = float(int(cfg["hidden_size"]) // int(cfg["dim_model_base"]))
            if cfg.get("model_type") == "minicpm_sala" and "dim_model_base" in cfg and "hidden_size" in cfg:
                scale_lm_head = float(cfg["dim_model_base"]) / float(cfg["hidden_size"])
            # minicpm_sala: only bake embed and lm_head; residual scaling done at forward in C++
            if cfg.get("model_type") == "minicpm_sala":
                scale_o = 1.0
                scale_down = 1.0
    except Exception:
        pass

    already_loaded_keys = []

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
            already_loaded_keys.extend(model_param.keys())

            # Apply MiniCPM scaling to loaded tensors (in torch space).
            if scale_input != 1.0 and "model.embed_tokens.weight" in model_param:
                model_param["model.embed_tokens.weight"] = (
                    model_param["model.embed_tokens.weight"] * scale_input
                )
            if scale_output != 1.0 and "model.norm.weight" in model_param:
                model_param["model.norm.weight"] = (
                    model_param["model.norm.weight"] * scale_output
                )
            if scale_o != 1.0 or scale_down != 1.0:
                for k, v in list(model_param.items()):
                    if scale_o != 1.0 and k.endswith(".self_attn.o_proj.weight"):
                        model_param[k] = v * scale_o
                    elif scale_down != 1.0 and k.endswith(".mlp.down_proj.weight"):
                        model_param[k] = v * scale_down
            if scale_lm_head != 1.0 and "lm_head.weight" in model_param:
                model_param["lm_head.weight"] = model_param["lm_head.weight"] * scale_lm_head

            # --------------------------------------------------------- #
            #         model_param_infini references torch.Tensor
            # --------------------------------------------------------- #
            model_param_infini = {}
            for key in model_param.keys():
                model_param_infini[key] = infinicore.from_torch(model_param[key])
            model.load_state_dict(model_param_infini, strict=False)
            infinicore.sync_device()

    elif os.path.exists(os.path.join(model_path, "pytorch_model.bin")):
        file_path = os.path.join(model_path, "pytorch_model.bin")
        model_params = torch.load(file_path, weights_only=True, map_location="cpu")

        if scale_input != 1.0 and "model.embed_tokens.weight" in model_params:
            model_params["model.embed_tokens.weight"] = model_params["model.embed_tokens.weight"] * scale_input
        if scale_output != 1.0 and "model.norm.weight" in model_params:
            model_params["model.norm.weight"] = model_params["model.norm.weight"] * scale_output
        if scale_o != 1.0 or scale_down != 1.0:
            for k, v in list(model_params.items()):
                if scale_o != 1.0 and k.endswith(".self_attn.o_proj.weight"):
                    model_params[k] = v * scale_o
                elif scale_down != 1.0 and k.endswith(".mlp.down_proj.weight"):
                    model_params[k] = v * scale_down
        if scale_lm_head != 1.0 and "lm_head.weight" in model_params:
            model_params["lm_head.weight"] = model_params["lm_head.weight"] * scale_lm_head

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
    already_loaded_keys = []

    file_list = glob.glob(os.path.join(model_path, "*.safetensors"))
    if len(file_list) > 0:
        for file_path in tqdm(file_list, desc="Processing files"):
            tqdm.write(f"Processing: {os.path.basename(file_path)}")

            with safe_open(file_path, "pt", "cpu") as f:
                for name in f.keys():
                    weight_infini = infinicore.from_torch(
                        f.get_tensor(name).to(dtype=torch_dtype)
                    )
                    model.load_param(name, weight_infini)
                    already_loaded_keys.append(name)
                    infinicore.sync_stream()

    elif os.path.exists(os.path.join(model_path, "pytorch_model.bin")):
        file_path = os.path.join(model_path, "pytorch_model.bin")
        model_params = torch.load(file_path, weights_only=True, map_location="cpu")

        for key in model_params.keys():
            weight_infini = infinicore.from_torch(
                model_params[key].to(dtype=torch_dtype)
            )
            model.load_param(key, weight_infini)
            already_loaded_keys.append(key)
    else:
        raise KeyError("Weight file not found.")

    check_parameters(model_keys, already_loaded_keys)

    t2 = time.time()
    print(f" load weights over! {(t2 - t1) * 1000} ms \n")
