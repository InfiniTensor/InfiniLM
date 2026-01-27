import os
from typing import Dict, Union, List, Optional
import time
import warnings
import torch
from safetensors import safe_open
import glob
from tqdm import tqdm
import infinicore

try:
    from infinicore.gguf import GGUFReader, find_split_files, load_gguf_tensors
    GGUF_AVAILABLE = True
except ImportError:
    GGUF_AVAILABLE = False

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


def is_gguf_path(model_path: str) -> bool:
    """
    Check if the model path contains GGUF files.

    Args:
        model_path: Path to model directory or GGUF file.

    Returns:
        True if GGUF files are found, False otherwise.
    """
    if not GGUF_AVAILABLE:
        return False

    if os.path.isfile(model_path) and model_path.endswith(".gguf"):
        return True

    gguf_files = glob.glob(os.path.join(model_path, "*.gguf"))
    return len(gguf_files) > 0


def map_gguf_to_infinilm_name(gguf_name: str) -> str:
    """
    Map GGUF tensor names to InfiniLM parameter names.

    GGUF naming convention:
    - token_embd.weight -> model.embed_tokens.weight
    - output.weight -> lm_head.weight
    - output_norm.weight -> model.norm.weight
    - blk.N.attn_norm.weight -> model.layers.N.input_layernorm.weight
    - blk.N.attn_q.weight -> model.layers.N.self_attn.q_proj.weight
    - blk.N.attn_k.weight -> model.layers.N.self_attn.k_proj.weight
    - blk.N.attn_v.weight -> model.layers.N.self_attn.v_proj.weight
    - blk.N.attn_output.weight -> model.layers.N.self_attn.o_proj.weight
    - blk.N.ffn_norm.weight -> model.layers.N.post_attention_layernorm.weight
    - blk.N.ffn_gate.weight -> model.layers.N.mlp.gate_proj.weight
    - blk.N.ffn_up.weight -> model.layers.N.mlp.up_proj.weight
    - blk.N.ffn_down.weight -> model.layers.N.mlp.down_proj.weight

    Args:
        gguf_name: Tensor name from GGUF file

    Returns:
        Mapped parameter name for InfiniLM model
    """
    # Handle special cases first
    if gguf_name == "token_embd.weight":
        return "model.embed_tokens.weight"
    elif gguf_name == "output.weight":
        return "lm_head.weight"
    elif gguf_name == "output_norm.weight":
        return "model.norm.weight"

    # Handle rope factors (skip these as they're not model parameters)
    if "rope_factors" in gguf_name:
        return None  # Signal to skip

    # Handle block layers: blk.N.* -> model.layers.N.*
    if gguf_name.startswith("blk."):
        parts = gguf_name.split(".")
        if len(parts) < 3:
            return None

        layer_idx = parts[1]
        param_type = parts[2]
        suffix = ".".join(parts[3:]) if len(parts) > 3 else "weight"

        # Map parameter types
        if param_type == "attn_norm":
            return f"model.layers.{layer_idx}.input_layernorm.{suffix}"
        elif param_type == "attn_q":
            return f"model.layers.{layer_idx}.self_attn.q_proj.{suffix}"
        elif param_type == "attn_k":
            return f"model.layers.{layer_idx}.self_attn.k_proj.{suffix}"
        elif param_type == "attn_v":
            return f"model.layers.{layer_idx}.self_attn.v_proj.{suffix}"
        elif param_type == "attn_output":
            return f"model.layers.{layer_idx}.self_attn.o_proj.{suffix}"
        elif param_type == "ffn_norm":
            return f"model.layers.{layer_idx}.post_attention_layernorm.{suffix}"
        elif param_type == "ffn_gate":
            return f"model.layers.{layer_idx}.mlp.gate_proj.{suffix}"
        elif param_type == "ffn_up":
            return f"model.layers.{layer_idx}.mlp.up_proj.{suffix}"
        elif param_type == "ffn_down":
            return f"model.layers.{layer_idx}.mlp.down_proj.{suffix}"

    # If no mapping found, return original name
    return gguf_name


def check_parameters(model_keys: list, already_loaded_keys: list, allow_missing_bias: bool = False):
    model_keys = set(model_keys)
    already_loaded_keys = set(already_loaded_keys)
    intersection = model_keys & already_loaded_keys

    missing_keys = model_keys - intersection
    unexpected_keys = already_loaded_keys - intersection

    # Filter out bias parameters from missing keys if allow_missing_bias is True
    if allow_missing_bias:
        missing_keys = {k for k in missing_keys if not k.endswith(".bias")}

    error_msgs: list[str] = []

    if len(unexpected_keys) > 0:
        error_msgs.append(
            "Unexpected key(s) in state_dict: {}. ".format(
                ", ".join('"{}"'.format(k) for k in sorted(unexpected_keys)[:20])  # Limit output
            )
        )
    if len(missing_keys) > 0:
        error_msgs.append(
            "Missing key(s) in state_dict: {}. ".format(
                ", ".join('"{}"'.format(k) for k in sorted(missing_keys)[:20])  # Limit output
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
    Supports safetensors, pytorch_model.bin, and GGUF formats.
    """
    print(" load weights ......")
    t1 = time.time()

    torch_device = "cpu"
    torch_dtype = infinicore.utils.to_torch_dtype(dtype)
    model_keys = model.state_dict_keyname()

    already_loaded_keys = []

    # Check if model_path is a GGUF file
    if os.path.isfile(model_path) and model_path.endswith(".gguf") and GGUF_AVAILABLE:
        # Load directly from GGUF file
        split_files = find_split_files(model_path)

        for file_path in tqdm(split_files, desc="Processing GGUF files"):
            tqdm.write(f"Processing: {os.path.basename(file_path)}")
            reader = GGUFReader(file_path)

            model_param = {}
            # Suppress non-writable array warning - safe for read-only operations
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message=".*not writable.*")
                for tensor_name in reader.get_tensor_names():
                    # Map to InfiniLM name first to check if we've already loaded it
                    mapped_name = map_gguf_to_infinilm_name(tensor_name)
                    if mapped_name is None or mapped_name in already_loaded_keys:
                        continue
                    try:
                        np_array = reader.get_tensor_data(tensor_name)
                        torch_tensor = torch.from_numpy(np_array).to(dtype=torch_dtype)
                        model_param[tensor_name] = torch_tensor
                    except Exception as e:
                        print(f"Warning: Failed to load tensor {tensor_name}: {e}")
                        continue

            # Get expected parameter names from model
            try:
                expected_keys = set(model.state_dict_keyname())
            except Exception:
                expected_keys = None

            # Convert to InfiniCore tensors and map GGUF names to InfiniLM names
            model_param_infini = {}
            for gguf_key in model_param.keys():
                # Map GGUF tensor name to InfiniLM parameter name
                mapped_key = map_gguf_to_infinilm_name(gguf_key)

                # Skip if mapping returns None (e.g., rope_factors)
                if mapped_key is None:
                    continue

                # Check if mapped key exists in model (if we can check)
                if expected_keys is not None:
                    if mapped_key not in expected_keys:
                        # Try original key as fallback
                        if gguf_key not in expected_keys:
                            continue  # Skip if neither mapped nor original key exists
                        else:
                            mapped_key = gguf_key  # Use original key if it exists
                    # Only add to already_loaded_keys if key exists in model
                    if mapped_key in expected_keys:
                        try:
                            model_param_infini[mapped_key] = infinicore.from_torch(model_param[gguf_key])
                            already_loaded_keys.append(mapped_key)
                        except Exception as e:
                            print(f"Warning: Failed to convert tensor {gguf_key} -> {mapped_key} to InfiniCore: {e}")
                            continue
                else:
                    # If we can't check expected_keys, add it anyway (will be validated later)
                    try:
                        model_param_infini[mapped_key] = infinicore.from_torch(model_param[gguf_key])
                        already_loaded_keys.append(mapped_key)
                    except Exception as e:
                        print(f"Warning: Failed to convert tensor {gguf_key} -> {mapped_key} to InfiniCore: {e}")
                        continue

            # Load with error handling for missing parameters
            model.load_state_dict(model_param_infini, strict=False)
            infinicore.sync_device()

    # Check for HuggingFace format files in directory (prioritize over GGUF)
    elif os.path.isdir(model_path):
        # Check for config.json first (HuggingFace format takes priority)
        config_path = os.path.join(model_path, "config.json")
        if os.path.exists(config_path):
            # HuggingFace format detected, load from safetensors or pytorch_model.bin
            if len(glob.glob(os.path.join(model_path, "*.safetensors"))) > 0:
                file_list = glob.glob(os.path.join(model_path, "*.safetensors"))
                for file_path in tqdm(file_list, desc="Processing files"):
                    tqdm.write(f"Processing: {os.path.basename(file_path)}")

                    # --------------------------------------------------------- #
                    #          Load weights from *.safetensors file
                    # --------------------------------------------------------- #
                    model_param = load_state_dict(
                        file_path, device=torch_device, dtype=torch_dtype
                    )
                    already_loaded_keys.extend(model_param.keys())

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

                model_param_infini = {}
                for key in model_params.keys():
                    model_param_infini[key] = infinicore.from_torch(
                        model_params[key].to(dtype=torch_dtype)
                    )
                    already_loaded_keys.append(key)

                model.load_state_dict(model_param_infini, strict=True)
                infinicore.sync_device()
            else:
                raise KeyError("Weight file not found in HuggingFace format directory.")
        else:
            # No config.json, check if directory contains GGUF files
            gguf_files = glob.glob(os.path.join(model_path, "*.gguf"))
            if len(gguf_files) > 0 and GGUF_AVAILABLE:
                # Load from GGUF file(s) (only if no config.json exists)
                gguf_file = gguf_files[0]
                split_files = find_split_files(gguf_file)

                for file_path in tqdm(split_files, desc="Processing GGUF files"):
                    tqdm.write(f"Processing: {os.path.basename(file_path)}")
                    reader = GGUFReader(file_path)

                    model_param = {}
                    # Suppress non-writable array warning - safe for read-only operations
                    with warnings.catch_warnings():
                        warnings.filterwarnings("ignore", message=".*not writable.*")
                        for tensor_name in reader.get_tensor_names():
                            # Map to InfiniLM name first to check if we've already loaded it
                            mapped_name = map_gguf_to_infinilm_name(tensor_name)
                            if mapped_name is None or mapped_name in already_loaded_keys:
                                continue
                            try:
                                np_array = reader.get_tensor_data(tensor_name)
                                torch_tensor = torch.from_numpy(np_array).to(dtype=torch_dtype)
                                model_param[tensor_name] = torch_tensor
                            except Exception as e:
                                print(f"Warning: Failed to load tensor {tensor_name}: {e}")
                                continue

                    # Get expected parameter names from model
                    try:
                        expected_keys = set(model.state_dict_keyname())
                    except Exception:
                        expected_keys = None

                    # Convert to InfiniCore tensors and map GGUF names to InfiniLM names
                    model_param_infini = {}
                    for gguf_key in model_param.keys():
                        # Map GGUF tensor name to InfiniLM parameter name
                        mapped_key = map_gguf_to_infinilm_name(gguf_key)

                        # Skip if mapping returns None (e.g., rope_factors)
                        if mapped_key is None:
                            continue

                        # Check if mapped key exists in model (if we can check)
                        if expected_keys is not None:
                            if mapped_key not in expected_keys:
                                # Try original key as fallback
                                if gguf_key not in expected_keys:
                                    continue  # Skip if neither mapped nor original key exists
                                else:
                                    mapped_key = gguf_key  # Use original key if it exists
                            # Only add to already_loaded_keys if key exists in model
                            if mapped_key in expected_keys:
                                try:
                                    model_param_infini[mapped_key] = infinicore.from_torch(model_param[gguf_key])
                                    already_loaded_keys.append(mapped_key)
                                except Exception as e:
                                    print(f"Warning: Failed to convert tensor {gguf_key} -> {mapped_key} to InfiniCore: {e}")
                                    continue
                        else:
                            # If we can't check expected_keys, add it anyway (will be validated later)
                            try:
                                model_param_infini[mapped_key] = infinicore.from_torch(model_param[gguf_key])
                                already_loaded_keys.append(mapped_key)
                            except Exception as e:
                                print(f"Warning: Failed to convert tensor {gguf_key} -> {mapped_key} to InfiniCore: {e}")
                                continue

                    # Load with error handling for missing parameters
                    model.load_state_dict(model_param_infini, strict=False)
                    infinicore.sync_device()
            else:
                raise KeyError("Weight file not found. No config.json, safetensors, pytorch_model.bin, or GGUF files found.")
    else:
        raise KeyError(f"Model path not found or unsupported: {model_path}")

    # For GGUF files, allow missing bias parameters (GGUF typically doesn't include biases)
    # Only consider it GGUF if:
    # 1. It's explicitly a .gguf file, OR
    # 2. It's a directory with .gguf files but NO config.json (GGUF-only directory)
    is_gguf = False
    if os.path.isfile(model_path) and model_path.endswith(".gguf"):
        is_gguf = True
    elif os.path.isdir(model_path):
        config_path = os.path.join(model_path, "config.json")
        # Only consider GGUF if no config.json exists (meaning it's a GGUF-only directory)
        if not os.path.exists(config_path) and len(glob.glob(os.path.join(model_path, "*.gguf"))) > 0:
            is_gguf = True
    check_parameters(model_keys, already_loaded_keys, allow_missing_bias=is_gguf)

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

    # Check if model_path is a GGUF file
    if os.path.isfile(model_path) and model_path.endswith(".gguf") and GGUF_AVAILABLE:
        # Load directly from GGUF file
        split_files = find_split_files(model_path)

        for file_path in tqdm(split_files, desc="Processing GGUF files"):
            tqdm.write(f"Processing: {os.path.basename(file_path)}")
            reader = GGUFReader(file_path)

            for tensor_name in reader.get_tensor_names():
                # Map to InfiniLM name first to check if we've already loaded it
                mapped_name = map_gguf_to_infinilm_name(tensor_name)
                if mapped_name is None or mapped_name in already_loaded_keys:
                    continue

                try:
                    # Get tensor data as numpy array
                    np_array = reader.get_tensor_data(tensor_name)

                    # Get tensor info for debugging (note: shape may be reversed in info)
                    try:
                        tensor_info = reader.get_tensor_info(tensor_name)
                        info_shape = tensor_info.get("shape", None)
                    except Exception:
                        info_shape = None

                    # Suppress non-writable warning and convert to torch tensor
                    with warnings.catch_warnings():
                        warnings.filterwarnings("ignore", message=".*not writable.*")
                        torch_tensor = torch.from_numpy(np_array).to(dtype=torch_dtype)

                    # Convert to infinicore tensor
                    weight_infini = infinicore.from_torch(torch_tensor)

                    # Verify shape before loading (for better error messages)
                    try:
                        expected_keys = set(model.state_dict_keyname())
                        if mapped_name in expected_keys:
                            # Try to get expected shape from model (if available)
                            # This helps diagnose shape mismatches
                            model.load_param(mapped_name, weight_infini._underlying)
                            already_loaded_keys.append(mapped_name)
                            infinicore.sync_stream()
                        else:
                            # Skip if parameter doesn't exist in model
                            continue
                    except RuntimeError as e:
                        # Enhanced error message for shape mismatches
                        error_msg = str(e)
                        if "Shape mismatch" in error_msg or "shape" in error_msg.lower():
                            print(f"Warning: Shape mismatch loading {tensor_name} -> {mapped_name}")
                            print(f"  Tensor shape: {np_array.shape}")
                            if info_shape:
                                print(f"  GGUF info shape: {info_shape} (may be reversed)")
                            print(f"  Error: {error_msg}")
                        else:
                            print(f"Warning: Failed to load tensor {tensor_name} -> {mapped_name}: {e}")
                        continue
                except Exception as e:
                    print(f"Warning: Failed to load tensor {tensor_name}: {e}")
                    continue

    # Check for HuggingFace format files in directory (prioritize over GGUF)
    elif os.path.isdir(model_path):
        # Check for config.json first (HuggingFace format takes priority)
        config_path = os.path.join(model_path, "config.json")
        if os.path.exists(config_path):
            # HuggingFace format detected, load from safetensors or pytorch_model.bin
            if len(glob.glob(os.path.join(model_path, "*.safetensors"))) > 0:
                file_list = glob.glob(os.path.join(model_path, "*.safetensors"))
                for file_path in tqdm(file_list, desc="Processing files"):
                    tqdm.write(f"Processing: {os.path.basename(file_path)}")

                    with safe_open(file_path, "pt", "cpu") as f:
                        for name in f.keys():
                            weight_infini = infinicore.from_torch(
                                f.get_tensor(name).to(dtype=torch_dtype)
                            )
                            model.load_param(name, weight_infini._underlying)
                            already_loaded_keys.append(name)
                            infinicore.sync_stream()
            elif os.path.exists(os.path.join(model_path, "pytorch_model.bin")):
                file_path = os.path.join(model_path, "pytorch_model.bin")
                model_params = torch.load(file_path, weights_only=True, map_location="cpu")

                for key in model_params.keys():
                    weight_infini = infinicore.from_torch(
                        model_params[key].to(dtype=torch_dtype)
                    )
                    model.load_param(key, weight_infini._underlying)
                    already_loaded_keys.append(key)
            else:
                raise KeyError("Weight file not found in HuggingFace format directory.")
        else:
            # No config.json, check if directory contains GGUF files
            gguf_files = glob.glob(os.path.join(model_path, "*.gguf"))
            if len(gguf_files) > 0 and GGUF_AVAILABLE:
                # Load from GGUF file(s) (only if no config.json exists)
                gguf_file = gguf_files[0]  # Use first GGUF file found
                split_files = find_split_files(gguf_file)

                for file_path in tqdm(split_files, desc="Processing GGUF files"):
                    tqdm.write(f"Processing: {os.path.basename(file_path)}")
                    reader = GGUFReader(file_path)

                    for tensor_name in reader.get_tensor_names():
                        # Map to InfiniLM name first to check if we've already loaded it
                        mapped_name = map_gguf_to_infinilm_name(tensor_name)
                        if mapped_name is None or mapped_name in already_loaded_keys:
                            continue

                        try:
                            # Get tensor data as numpy array
                            np_array = reader.get_tensor_data(tensor_name)

                            # Get tensor info for debugging (note: shape may be reversed in info)
                            try:
                                tensor_info = reader.get_tensor_info(tensor_name)
                                info_shape = tensor_info.get("shape", None)
                            except Exception:
                                info_shape = None

                            # Suppress non-writable warning and convert to torch tensor
                            with warnings.catch_warnings():
                                warnings.filterwarnings("ignore", message=".*not writable.*")
                                torch_tensor = torch.from_numpy(np_array).to(dtype=torch_dtype)

                            # Convert to infinicore tensor
                            weight_infini = infinicore.from_torch(torch_tensor)

                            # Verify shape before loading (for better error messages)
                            try:
                                expected_keys = set(model.state_dict_keyname())
                                if mapped_name in expected_keys:
                                    # Try to get expected shape from model (if available)
                                    # This helps diagnose shape mismatches
                                    model.load_param(mapped_name, weight_infini._underlying)
                                    already_loaded_keys.append(mapped_name)
                                    infinicore.sync_stream()
                                else:
                                    # Skip if parameter doesn't exist in model
                                    continue
                            except RuntimeError as e:
                                # Enhanced error message for shape mismatches
                                error_msg = str(e)
                                if "Shape mismatch" in error_msg or "shape" in error_msg.lower():
                                    print(f"Warning: Shape mismatch loading {tensor_name} -> {mapped_name}")
                                    print(f"  Tensor shape: {np_array.shape}")
                                    if info_shape:
                                        print(f"  GGUF info shape: {info_shape} (may be reversed)")
                                    print(f"  Error: {error_msg}")
                                else:
                                    print(f"Warning: Failed to load tensor {tensor_name} -> {mapped_name}: {e}")
                                continue
                        except Exception as e:
                            print(f"Warning: Failed to load tensor {tensor_name}: {e}")
                            continue
            else:
                raise KeyError("Weight file not found. No config.json, safetensors, pytorch_model.bin, or GGUF files found.")

    # For GGUF files, allow missing bias parameters (GGUF typically doesn't include biases)
    # Only consider it GGUF if:
    # 1. It's explicitly a .gguf file, OR
    # 2. It's a directory with .gguf files but NO config.json (GGUF-only directory)
    is_gguf = False
    if os.path.isfile(model_path) and model_path.endswith(".gguf"):
        is_gguf = True
    elif os.path.isdir(model_path):
        config_path = os.path.join(model_path, "config.json")
        # Only consider GGUF if no config.json exists (meaning it's a GGUF-only directory)
        if not os.path.exists(config_path) and len(glob.glob(os.path.join(model_path, "*.gguf"))) > 0:
            is_gguf = True
    check_parameters(model_keys, already_loaded_keys, allow_missing_bias=is_gguf)

    t2 = time.time()
    print(f" load weights over! {(t2 - t1) * 1000} ms \n")


def load_gguf_state_dict(
    gguf_file: Union[str, List[str]],
    device: str = "cpu",
    dtype=torch.bfloat16,
) -> Dict[str, torch.Tensor]:
    """
    Load state dict from GGUF file(s).

    Args:
        gguf_file: Path to GGUF file or list of split file paths.
        device: Target device (currently only "cpu" supported).
        dtype: Target data type.

    Returns:
        Dictionary mapping tensor names to torch tensors.
    """
    if not GGUF_AVAILABLE:
        raise ImportError("GGUF support not available. Install gguf-py: pip install gguf")

    tensors = load_gguf_tensors(gguf_file, device=device)

    # Convert numpy arrays to torch tensors
    state_dict = {}
    for name, np_array in tensors.items():
        state_dict[name] = torch.from_numpy(np_array).to(device=device, dtype=dtype)

    return state_dict


def load_model_state_dict_from_gguf(
    model: infinicore.nn.Module,
    model_path: str,
    dtype=infinicore.dtype,
) -> Dict[str, infinicore.Tensor]:
    """
    Load model weights from GGUF file(s).

    Args:
        model: InfiniCore model instance.
        model_path: Path to model directory or GGUF file.
        dtype: Target data type.

    Returns:
        Dictionary mapping parameter names to InfiniCore tensors.
    """
    if not GGUF_AVAILABLE:
        raise ImportError("GGUF support not available. Install gguf-py: pip install gguf")

    print(" load weights from GGUF ......")
    t1 = time.time()

    torch_dtype = infinicore.utils.to_torch_dtype(dtype)
    model_keys = model.state_dict_keyname()

    # Find GGUF files
    if os.path.isfile(model_path) and model_path.endswith(".gguf"):
        gguf_file = model_path
    else:
        gguf_files = glob.glob(os.path.join(model_path, "*.gguf"))
        if len(gguf_files) == 0:
            raise FileNotFoundError(f"No GGUF files found in {model_path}")
        gguf_file = gguf_files[0]

    # Find all split files if this is a split GGUF
    split_files = find_split_files(gguf_file)

    # Load all tensors
    model_param = {}
    already_loaded_keys = []

    for file_path in tqdm(split_files, desc="Loading GGUF files"):
        tqdm.write(f"Processing: {os.path.basename(file_path)}")
        reader = GGUFReader(file_path)

        for tensor_name in reader.get_tensor_names():
            if tensor_name not in already_loaded_keys:
                try:
                    np_array = reader.get_tensor_data(tensor_name)
                    torch_tensor = torch.from_numpy(np_array).to(dtype=torch_dtype)
                    model_param[tensor_name] = torch_tensor
                    already_loaded_keys.append(tensor_name)
                except Exception as e:
                    print(f"Warning: Failed to load tensor {tensor_name}: {e}")
                    continue

    # Handle lm_head.weight if missing
    if model_param.get("lm_head.weight", None) is None:
        if "model.embed_tokens.weight" in model_param:
            model_param["lm_head.weight"] = model_param["model.embed_tokens.weight"]

    # Convert to InfiniCore tensors
    model_param_infini = {}
    for key in model_param.keys():
        model_param_infini[key] = infinicore.from_torch(model_param[key])

    t2 = time.time()
    print(f" load weights over! {(t2 - t1) * 1000} ms \n")

    return model_param_infini
