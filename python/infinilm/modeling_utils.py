import os
from typing import Dict, Union, List, Optional
import time
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

            # Convert to InfiniCore tensors and load
            model_param_infini = {}
            for key in model_param.keys():
                model_param_infini[key] = infinicore.from_torch(model_param[key])

            model.load_state_dict(model_param_infini, strict=False)
            infinicore.sync_device()

    # Check for GGUF files in directory
    elif os.path.isdir(model_path):
        gguf_files = glob.glob(os.path.join(model_path, "*.gguf"))
        if len(gguf_files) > 0 and GGUF_AVAILABLE:
            # Load from GGUF file(s)
            gguf_file = gguf_files[0]
            split_files = find_split_files(gguf_file)

            for file_path in tqdm(split_files, desc="Processing GGUF files"):
                tqdm.write(f"Processing: {os.path.basename(file_path)}")
                reader = GGUFReader(file_path)

                model_param = {}
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

                # Convert to InfiniCore tensors and load
                model_param_infini = {}
                for key in model_param.keys():
                    model_param_infini[key] = infinicore.from_torch(model_param[key])

                model.load_state_dict(model_param_infini, strict=False)
                infinicore.sync_device()
        elif len(glob.glob(os.path.join(model_path, "*.safetensors"))) > 0:
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
            raise KeyError("Weight file not found.")
    else:
        raise KeyError(f"Model path not found or unsupported: {model_path}")

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

    # Check for GGUF files first
    gguf_files = glob.glob(os.path.join(model_path, "*.gguf"))
    if len(gguf_files) > 0 and GGUF_AVAILABLE:
        # Load from GGUF file(s)
        gguf_file = gguf_files[0]  # Use first GGUF file found
        split_files = find_split_files(gguf_file)

        for file_path in tqdm(split_files, desc="Processing GGUF files"):
            tqdm.write(f"Processing: {os.path.basename(file_path)}")
            reader = GGUFReader(file_path)

            for tensor_name in reader.get_tensor_names():
                if tensor_name not in already_loaded_keys:
                    try:
                        # Get tensor data as numpy array
                        np_array = reader.get_tensor_data(tensor_name)
                        # Convert to torch tensor
                        torch_tensor = torch.from_numpy(np_array).to(dtype=torch_dtype)
                        # Convert to infinicore tensor
                        weight_infini = infinicore.from_torch(torch_tensor)
                        model.load_param(tensor_name, weight_infini)
                        already_loaded_keys.append(tensor_name)
                        infinicore.sync_stream()
                    except Exception as e:
                        print(f"Warning: Failed to load tensor {tensor_name}: {e}")
                        continue

    elif len(glob.glob(os.path.join(model_path, "*.safetensors"))) > 0:
        file_list = glob.glob(os.path.join(model_path, "*.safetensors"))
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
