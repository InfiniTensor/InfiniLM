import os
import json
from typing import Dict, Union, Optional, List
import time
import torch
from safetensors import safe_open
import glob
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from queue import Queue
import threading
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


def _weight_load_worker_count(file_count: int) -> int:
    if file_count <= 1:
        return 1

    configured = os.environ.get("INFINILM_WEIGHT_LOAD_WORKERS")
    if configured is not None:
        try:
            return max(1, min(file_count, int(configured)))
        except ValueError:
            raise ValueError("INFINILM_WEIGHT_LOAD_WORKERS must be an integer.")

    return min(file_count, 4)


def _weight_load_batch_bytes() -> int:
    configured = os.environ.get("INFINILM_WEIGHT_LOAD_BATCH_MB")
    if configured is None:
        return 512 * 1024 * 1024

    try:
        batch_mb = int(configured)
    except ValueError:
        raise ValueError("INFINILM_WEIGHT_LOAD_BATCH_MB must be an integer.")

    return max(1, batch_mb) * 1024 * 1024


def _load_safetensors_shard(file_path: str, torch_device: str, torch_dtype):
    return file_path, load_state_dict(file_path, device=torch_device, dtype=torch_dtype)


def _iter_safetensors_shards(file_list: List[str], torch_device: str, torch_dtype):
    worker_count = _weight_load_worker_count(len(file_list))
    if worker_count == 1:
        for file_path in file_list:
            yield _load_safetensors_shard(file_path, torch_device, torch_dtype)
        return

    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        pending = set()
        file_iter = iter(file_list)

        for _ in range(worker_count):
            try:
                file_path = next(file_iter)
            except StopIteration:
                break
            pending.add(
                executor.submit(
                    _load_safetensors_shard, file_path, torch_device, torch_dtype
                )
            )

        while pending:
            done, pending = wait(pending, return_when=FIRST_COMPLETED)
            for future in done:
                yield future.result()
                try:
                    file_path = next(file_iter)
                except StopIteration:
                    continue
                pending.add(
                    executor.submit(
                        _load_safetensors_shard, file_path, torch_device, torch_dtype
                    )
                )


def _tensor_nbytes(tensor: torch.Tensor) -> int:
    return tensor.numel() * tensor.element_size()


def _read_safetensors_batches(
    file_path: str,
    torch_device: str,
    torch_dtype,
    max_batch_bytes: int,
    output_queue: Queue,
):
    batch = {}
    batch_bytes = 0
    with safe_open(file_path, framework="pt") as f:
        metadata = f.metadata()
        if metadata is not None and metadata.get("format") not in [
            "pt",
            "tf",
            "flax",
            "mlx",
        ]:
            raise OSError(
                f"The safetensors archive passed at {file_path} does not contain the valid metadata."
            )

        for name in f.keys():
            tensor = f.get_tensor(name).to(device=torch_device)
            tensor_bytes = _tensor_nbytes(tensor)
            if batch and batch_bytes + tensor_bytes > max_batch_bytes:
                output_queue.put((file_path, batch))
                batch = {}
                batch_bytes = 0
            batch[name] = tensor
            batch_bytes += tensor_bytes

    if batch:
        output_queue.put((file_path, batch))


def _iter_safetensors_batches(file_list: List[str], torch_device: str, torch_dtype):
    if len(file_list) == 0:
        return

    worker_count = _weight_load_worker_count(len(file_list))
    max_batch_bytes = _weight_load_batch_bytes()
    output_queue = Queue(maxsize=worker_count * 2)
    file_queue = Queue()
    stop_event = threading.Event()
    sentinel = object()

    for file_path in file_list:
        file_queue.put(file_path)
    for _ in range(worker_count):
        file_queue.put(None)

    def worker():
        try:
            while not stop_event.is_set():
                file_path = file_queue.get()
                if file_path is None:
                    return
                _read_safetensors_batches(
                    file_path, torch_device, torch_dtype, max_batch_bytes, output_queue
                )
        except BaseException as exc:
            output_queue.put(exc)
        finally:
            output_queue.put(sentinel)

    threads = [
        threading.Thread(target=worker, name=f"weight-loader-{idx}", daemon=True)
        for idx in range(worker_count)
    ]
    for thread in threads:
        thread.start()

    finished = 0
    try:
        while finished < worker_count:
            item = output_queue.get()
            if item is sentinel:
                finished += 1
                continue
            if isinstance(item, BaseException):
                stop_event.set()
                raise item
            yield item
    finally:
        stop_event.set()
        for thread in threads:
            thread.join()


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
    file_list = glob.glob(os.path.join(model_path, "*.safetensors"))
    for _, shard_param in _iter_safetensors_shards(
        file_list, torch_device, torch_dtype
    ):
        model_param.update(shard_param)

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
        batch_iter = _iter_safetensors_batches(file_list, torch_device, torch_dtype)
        for file_path, model_param in tqdm(batch_iter, desc="Processing weights"):
            tqdm.write(f"Processing: {os.path.basename(file_path)}")

            # Apply model-specific weight remapping
            remapper = _WEIGHT_REMAPPER.get(model_type)
            if remapper is not None:
                model_param = remapper(model_param, config=model.hf_config)

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
            model_params = remapper(model_params, config=model.hf_config)

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

def drop_keys(
    state_dict: Dict[str, torch.Tensor],
    substrings: List[str],
) -> Dict[str, torch.Tensor]:
    """Drop keys containing any of the given substrings."""
    return {
        k: v for k, v in state_dict.items()
        if not any(sub in k for sub in substrings)
    }


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


def split_fused_weight(
    state_dict: Dict[str, torch.Tensor],
    fused_key: str,
    output_names: List[str],
    split_dim: int = 0,
    split_sizes: Optional[List[int]] = None,
) -> Dict[str, torch.Tensor]:
    """Split fused weight tensors into separate weights.

    Args:
        state_dict: Original state dict.
        fused_key: Substring to match in key names (e.g. "query_key_value").
        output_names: Names of the split outputs (e.g. ["q_proj", "k_proj", "v_proj"]).
        split_dim: Dimension along which to split. Default 0.
        split_sizes: Optional explicit sizes for each split. Supports -1 to mean
            "the remaining size". If None, split equally.

    Returns:
        New state dict with fused keys replaced by split keys.

    Examples:
        # Equal 2-way split (e.g. gate_up_proj.weight)
        split_fused_weight(sd, "gate_up_proj", ["gate_proj", "up_proj"])

        # Dynamic 3-way split with bias (e.g. query_key_value.weight + bias)
        split_fused_weight(sd, "query_key_value", ["q_proj", "k_proj", "v_proj"],
                           split_sizes=[q_dim, k_dim, -1])
    """
    result = {}
    marker = f".{fused_key}."

    for key, tensor in state_dict.items():
        if marker not in key:
            result[key] = tensor
            continue

        # Extract base_key and suffix (handles both .weight and .bias)
        base_key, suffix = key.split(marker, 1)
        dim_size = tensor.shape[split_dim]

        # Calculate split sizes
        if split_sizes is not None:
            sizes = []
            remainder = dim_size
            for s in split_sizes:
                if s == -1:
                    sizes.append(0)  # placeholder
                else:
                    sizes.append(s)
                    remainder -= s
            # Fill -1 placeholders with remainder
            sizes = [remainder if s == 0 else s for s in sizes]
        else:
            num_splits = len(output_names)
            chunk = dim_size // num_splits
            sizes = [chunk] * (num_splits - 1)
            sizes.append(dim_size - chunk * (num_splits - 1))

        splits = torch.split(tensor, sizes, dim=split_dim)
        for name, split_tensor in zip(output_names, splits):
            result[f"{base_key}.{name}.{suffix}"] = split_tensor

    return result

def split_fused_weight_with_sizes(
    state_dict: Dict[str, torch.Tensor],
    fused_key: str,
    output_names: List[str],
    split_sizes: List[int],
    split_dim: int = 0,
) -> Dict[str, torch.Tensor]:
    """Split fused weight tensors into separate weights with explicit sizes.
    Supports -1 in split_sizes to mean "the remaining size".
    Handles both .weight and .bias suffixes (unlike split_fused_weight
    which only handles .weight).
    """
    result = {}
    marker = f".{fused_key}."

    for key, tensor in state_dict.items():
        if marker not in key:
            result[key] = tensor
            continue

        base_key, suffix = key.split(marker, 1)
        dim_size = tensor.shape[split_dim]

        # Resolve -1 (remainder)
        sizes = []
        remainder = dim_size
        for s in split_sizes:
            if s == -1:
                sizes.append(0)  # placeholder
            else:
                sizes.append(s)
                remainder -= s
        sizes = [remainder if s == 0 else s for s in sizes]

        splits = torch.split(tensor, sizes, dim=split_dim)
        for name, split_tensor in zip(output_names, splits):
            result[f"{base_key}.{name}.{suffix}"] = split_tensor

    return result

# ============================================================================
# Model-specific remap functions
# ============================================================================
def _remap_glm4(state_dict, config=None):
    """Split GLM-4 fused gate_up_proj into gate_proj + up_proj."""
    return split_fused_weight(
        state_dict,
        fused_key="gate_up_proj",
        output_names=["gate_proj", "up_proj"],
    )


def _remap_chatglm(state_dict, config=None):
    """Remap ChatGLM weights to InfiniLM format.

    Faithfully ported from the original working _remap_chatglm_weights.
    """
    hf_config = config or {}
    num_heads = hf_config.get("num_attention_heads", 32)
    num_kv = hf_config.get("multi_query_group_num", 2)
    head_dim = hf_config.get("kv_channels", 128)
    ffn_hidden = hf_config.get("ffn_hidden_size", 13696)

    q_dim = num_heads * head_dim
    k_dim = num_kv * head_dim

    # 1. Drop unused keys
    state_dict = drop_keys(state_dict, ["rotary_pos_emb"])

    # 2. Split QKV
    state_dict = split_fused_weight_with_sizes(
        state_dict,
        fused_key="query_key_value",
        output_names=["q_proj", "k_proj", "v_proj"],
        split_sizes=[q_dim, k_dim, -1],
    )

    # 3. Split gate_up
    state_dict = split_fused_weight_with_sizes(
        state_dict,
        fused_key="dense_h_to_4h",
        output_names=["gate_proj", "up_proj"],
        split_sizes=[ffn_hidden, -1],
    )

    # 4. Rename keys
    state_dict = rename_keys(state_dict, {
        "transformer.encoder.layers.": "model.layers.",
        "transformer.embedding.word_embeddings": "model.embed_tokens",
        "transformer.encoder.final_layernorm": "model.norm",
        "transformer.output_layer": "lm_head",
        "self_attention.": "self_attn.",
        "self_attn.dense": "self_attn.o_proj",
        "mlp.dense_4h_to_h": "mlp.down_proj",
    })

    return state_dict

def _is_baichuan2(config):
    """
    Baichuan1 and Baichuan2 share the same model_type "baichuan" in official HuggingFace configs,
    making them indistinguishable by model_type alone. However, their inference logic differs
    critically: Baichuan2 requires normalized lm_head while Baichuan1 does not.

    The most reliable automatic way to distinguish them is by vocab_size:
      - Baichuan1: vocab_size = 64000
      - Baichuan2: vocab_size = 125696
    """
    return config.get("vocab_size") == 125696

def _remap_baichuan(state_dict, config=None):
    """Split Baichuan fused W_pack into q_proj, k_proj, v_proj
    and apply Baichuan2-specific fixes."""
    import torch.nn.functional as F

    hf_config = config or {}
    hidden_size = hf_config.get("hidden_size", 4096)
    num_heads = hf_config.get("num_attention_heads", 32)
    vocab_size = hf_config.get("vocab_size", 125696)
    per_head_dim = num_heads * (hidden_size // num_heads)

    # 1. Split W_pack → q_proj, k_proj, v_proj
    state_dict = split_fused_weight(
        state_dict,
        fused_key="W_pack",
        output_names=["q_proj", "k_proj", "v_proj"],
        split_sizes=[per_head_dim, per_head_dim, -1],
    )

    # 2. Baichuan2: normalize lm_head.weight
    #    Baichuan2 trains with normalized lm_head. Inference must match this,
    #    otherwise the logits distribution will be distorted, causing severe
    #    repetitive output especially under greedy decoding.
    #    (See _is_baichuan2 for how we distinguish Baichuan1 vs Baichuan2)
    if _is_baichuan2(hf_config) and "lm_head.weight" in state_dict:
        state_dict["lm_head.weight"] = F.normalize(
            state_dict["lm_head.weight"], p=2, dim=-1
        )

    return state_dict


# Model type → remap function mapping
_WEIGHT_REMAPPER = {
    "glm4": _remap_glm4,
    "chatglm": _remap_chatglm,
    "baichuan": _remap_baichuan,
}
