import os
import re
from typing import Dict, Union
import time
import torch
from safetensors import safe_open
import glob
from tqdm import tqdm
import infinicore


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


def _split_first_dim(tensor, sizes, name):
    if tensor.dim() not in (1, 2):
        raise ValueError(f"Cannot split {name} with shape {tensor.shape}")
    return torch.split(tensor, sizes, dim=0)


def _remap_glm4_weights(state_dict):
    new_sd = {}

    for key, tensor in state_dict.items():
        if "gate_up_proj" not in key:
            new_sd[key] = tensor
            continue

        base_key = key.replace(".gate_up_proj.weight", "")
        intermediate = tensor.shape[0] // 2
        gate, up = _split_first_dim(
            tensor,
            [intermediate, tensor.shape[0] - intermediate],
            "gate_up_proj",
        )
        new_sd[f"{base_key}.gate_proj.weight"] = gate
        new_sd[f"{base_key}.up_proj.weight"] = up
    return new_sd


def _rename_chatglm_key(key):
    new_key = key.replace("transformer.encoder.layers.", "model.layers.")
    new_key = new_key.replace("transformer.embedding.word_embeddings", "model.embed_tokens")
    new_key = new_key.replace("transformer.encoder.final_layernorm", "model.norm")
    new_key = new_key.replace("transformer.output_layer", "lm_head")
    new_key = new_key.replace("self_attention.", "self_attn.")
    new_key = new_key.replace("self_attn.dense", "self_attn.o_proj")
    return new_key.replace("mlp.dense_4h_to_h", "mlp.down_proj")


def _remap_chatglm_weights(state_dict, hf_config):
    num_heads = hf_config.get("num_attention_heads", 32)
    num_kv = hf_config.get("multi_query_group_num", 2)
    head_dim = hf_config.get("kv_channels", 128)
    ffn_hidden = hf_config.get("ffn_hidden_size", 13696)
    q_dim = num_heads * head_dim
    k_dim = num_kv * head_dim
    new_sd = {}

    for key, tensor in state_dict.items():
        if "rotary_pos_emb" in key:
            continue

        new_key = _rename_chatglm_key(key)
        if "query_key_value" in new_key:
            suffix = "weight" if tensor.dim() == 2 else "bias"
            q, k, v = _split_first_dim(
                tensor,
                [q_dim, k_dim, tensor.shape[0] - q_dim - k_dim],
                "query_key_value",
            )
            base_key = new_key.rsplit(".query_key_value", 1)[0]
            new_sd[f"{base_key}.q_proj.{suffix}"] = q
            new_sd[f"{base_key}.k_proj.{suffix}"] = k
            new_sd[f"{base_key}.v_proj.{suffix}"] = v
            continue

        if "dense_h_to_4h" in new_key:
            base_key = new_key.replace(".dense_h_to_4h.weight", "")
            gate, up = _split_first_dim(
                tensor,
                [ffn_hidden, tensor.shape[0] - ffn_hidden],
                "dense_h_to_4h",
            )
            new_sd[f"{base_key}.gate_proj.weight"] = gate
            new_sd[f"{base_key}.up_proj.weight"] = up
            continue

        new_sd[new_key] = tensor
    return new_sd


def maybe_remap_weights(state_dict, model):
    if not hasattr(model, "hf_config"):
        return state_dict

    hf_config = model.hf_config
    model_type = hf_config.get("model_type", "")
    if model_type == "glm4":
        return _remap_glm4_weights(state_dict)
    if model_type == "chatglm":
        return _remap_chatglm_weights(state_dict, hf_config)
    return state_dict


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
            model_param = maybe_remap_weights(model_param, model)
            already_loaded_keys.extend(model_param.keys())

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

        model_params = maybe_remap_weights(model_params, model)

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
