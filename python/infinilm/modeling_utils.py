from __future__ import annotations

import glob
import json
import math
import os
import re
import time
import zipfile
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Union

from tqdm import tqdm

from . import _core as infinicore
from ._safetensors_mmap import SafeTensorsMmap

if TYPE_CHECKING:
    import torch

    from .infer_engine import InferEngine


def _get_scale_emb(model_path: str) -> float:
    config_path = os.path.join(model_path, "config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"config.json not found at {config_path}")
    with open(config_path, "r") as f:
        config = json.load(f)
    if config.get("model_type") not in ("fm9g", "minicpm"):
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


def _require_torch_checkpoint(operation):
    """Describe the optional bridge without hiding a broken Torch installation."""
    import importlib

    try:
        return importlib.import_module("torch")
    except ModuleNotFoundError as exc:
        if exc.name != "torch":
            raise
        raise ImportError(
            f"{operation} requires the optional Torch checkpoint compatibility path; "
            "install InfiniLM[torch-compat]. This path is not Torch-free."
        ) from exc


def _load_pytorch_bin(file_path: str) -> Dict[str, torch.Tensor]:
    torch = _require_torch_checkpoint(".bin checkpoint loading")

    load_kwargs = {
        "weights_only": True,
        "map_location": "cpu",
    }
    if zipfile.is_zipfile(file_path):
        load_kwargs["mmap"] = True
    return torch.load(file_path, **load_kwargs)


def _declares_tied_embeddings(config: dict | None) -> bool:
    if not config:
        return False
    if config.get("model_type") == "gpt2":
        tied = config.get("tie_word_embeddings", True)
        if not isinstance(tied, bool):
            raise ValueError("GPT2 tie_word_embeddings must be a boolean")
        return tied
    if "tie_word_embeddings" in config:
        return bool(config["tie_word_embeddings"])
    text_config = config.get("text_config")
    return isinstance(text_config, dict) and bool(
        text_config.get("tie_word_embeddings", False)
    )


def _borrow_checkpoint_tensor(tensor) -> infinicore.Tensor:
    """Borrow a CPU checkpoint tensor until the rank-local copy has completed."""
    if isinstance(tensor, infinicore.Tensor):
        if tensor.device.type != "cpu":
            raise ValueError("checkpoint tensor borrowing requires CPU storage")
        return tensor
    if tensor.device.type != "cpu":
        raise ValueError("checkpoint tensor borrowing requires CPU storage")
    owner = tensor.detach().contiguous()
    return infinicore.from_blob(
        owner.data_ptr(),
        list(owner.shape),
        dtype=infinicore.utils.to_infinicore_dtype(owner.dtype),
        device="cpu",
        owner=owner,
    )


def load_state_dict(
    checkpoint_file: Union[str, os.PathLike],
    device="cpu",
    dtype=None,
    preserve_fp32_suffixes: Tuple[str, ...] = (".e_score_correction_bias",),
    keys=None,
) -> Dict[str, torch.Tensor]:
    """
    Reads a `safetensor` checkpoint file. We load the checkpoint on "cpu" by default.
    """
    _require_torch_checkpoint("legacy safetensors framework=pt loading")
    from safetensors import safe_open

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

        names = list(f.keys()) if keys is None else list(keys)
        if not set(names).issubset(f.keys()):
            raise KeyError("Selected checkpoint keys are absent from the shard")
        for k in names:
            tensor = f.get_tensor(k)
            preserve_fp32 = k.endswith(preserve_fp32_suffixes)
            if dtype is not None and tensor.is_floating_point() and not preserve_fp32:
                tensor = tensor.to(device=device, dtype=dtype)
            else:
                tensor = tensor.to(device=device)
            state_dict[k] = tensor

    return state_dict


def _cast_checkpoint_tensor(tensor, dtype):
    """Keep the optional Torch bridge; native CPU storage uses the existing Cast."""
    if not isinstance(tensor, infinicore.Tensor):
        return tensor.to(dtype=infinicore.utils.to_torch_dtype(dtype))
    if tensor.device.type != "cpu":
        raise ValueError("checkpoint conversion requires CPU storage")
    if tensor.dtype == dtype:
        return tensor
    from .lib import _infinilm

    converted = infinicore.empty(tensor.shape, dtype=dtype, device="cpu")
    _infinilm._cast_checkpoint_tensor_(tensor._underlying, converted._underlying)
    return converted


def _prepare_checkpoint_tensor(tensor, dtype):
    source = _borrow_checkpoint_tensor(tensor)
    floating = (
        infinicore.float16,
        infinicore.bfloat16,
        infinicore.float32,
        infinicore.float64,
    )
    if dtype in floating and source.dtype in floating and source.dtype != dtype:
        return _borrow_checkpoint_tensor(_cast_checkpoint_tensor(tensor, dtype))
    return source


def _scale_checkpoint_tensor(tensor, alpha):
    """Preserve the Torch bridge; native CPU scaling is owned by Ops."""
    if not isinstance(tensor, infinicore.Tensor):
        return tensor * alpha
    from .lib import _infinilm

    scaled = infinicore.empty(tensor.shape, dtype=tensor.dtype, device="cpu")
    _infinilm._scale_checkpoint_tensor_(tensor._underlying, alpha, scaled._underlying)
    return scaled


def _normalize_checkpoint_tensor(tensor, eps=1e-12):
    """Keep the optional Torch bridge; native row L2 arithmetic belongs to Ops."""
    if not isinstance(tensor, infinicore.Tensor):
        import torch.nn.functional as F

        return F.normalize(tensor, p=2, dim=-1, eps=eps)
    from .lib import _infinilm

    normalized = infinicore.empty(tensor.shape, dtype=tensor.dtype, device="cpu")
    _infinilm._normalize_checkpoint_tensor_(
        tensor._underlying, eps, normalized._underlying
    )
    return normalized


def _load_checkpoint_shard(file_path, dtype, preserve_fp32_suffixes, *, raw, keys=None):
    """Borrow checkpoint storage; an explicit dtype requests CPU float conversion."""
    if raw:
        types = {
            "U8": infinicore.uint8,
            "I8": infinicore.int8,
            "I16": infinicore.int16,
            "I32": infinicore.int32,
            "I64": infinicore.int64,
            "F16": infinicore.float16,
            "BF16": infinicore.bfloat16,
            "F32": infinicore.float32,
            "F64": infinicore.float64,
        }
        cast_types = (infinicore.float32, infinicore.float16, infinicore.bfloat16)
        with SafeTensorsMmap(file_path) as shard:
            if shard.metadata is not None and shard.metadata.get("format") not in (
                "pt",
                "tf",
                "flax",
                "mlx",
            ):
                raise OSError(f"Invalid safetensors metadata at {file_path}")
            names = list(shard.tensors) if keys is None else list(keys)
            if not set(names).issubset(shard.tensors):
                raise KeyError("Selected checkpoint keys are absent from the shard")
            # Default None retains the whole-shard contract; selected reads only
            # expose requested keys and keep unrelated tensors out of host memory.
            compatible = all(
                info.dtype in types
                and (
                    (dtype is None and types[info.dtype] in cast_types)
                    or not info.dtype.startswith(("F", "BF"))
                    or name.endswith(preserve_fp32_suffixes)
                    or types[info.dtype] == dtype
                    or (types[info.dtype] in cast_types and dtype in cast_types)
                )
                for name, info in ((name, shard.tensors[name]) for name in names)
            )
            if compatible:
                tensors = {}
                for name in names:
                    info = shard.tensors[name]
                    owner = shard.tensor_bytes(name)
                    tensor = infinicore.from_blob(
                        owner.ctypes.data,
                        list(info.shape),
                        dtype=types[info.dtype],
                        device="cpu",
                        owner=owner,
                    )
                    if (
                        dtype is not None
                        and types[info.dtype] in cast_types
                        and types[info.dtype] != dtype
                        and not name.endswith(preserve_fp32_suffixes)
                    ):
                        tensor = _cast_checkpoint_tensor(tensor, dtype)
                    tensors[name] = tensor
                    del tensor, owner
                return tensors
    _require_torch_checkpoint("unqualified safetensors checkpoint fallback")
    selection = {} if keys is None else {"keys": keys}
    return load_state_dict(
        file_path,
        device="cpu",
        dtype=infinicore.utils.to_torch_dtype(dtype) if dtype is not None else None,
        preserve_fp32_suffixes=preserve_fp32_suffixes,
        **selection,
    )


def load_model_state_dict_by_file(
    model: InferEngine,
    model_path: str,
    dtype=infinicore.dtype,
) -> Dict[str, infinicore.Tensor]:
    """Load through one publication/tie/finalization transaction."""
    print(" load weights ......")
    t1 = time.time()

    model_type = model.hf_config.get("model_type", "")
    is_ernie = model_type == "ernie4_5_moe_vl"
    tie_word_embeddings = _declares_tied_embeddings(model.hf_config)
    model_keys = model.state_dict_keyname()
    model_key_set = set(model_keys)
    parameter_dtypes = {
        name: infinicore.Tensor(parameter).dtype
        for parameters in model.state_dict()
        for name, parameter in parameters.items()
    }
    dist_config = getattr(model, "distributed_config", None)
    is_pipeline_parallel = dist_config is not None and dist_config.pp_size > 1
    scale_emb = _get_scale_emb(model_path)
    already_loaded_keys = []
    embed_tokens_torch_unscaled = None
    remapper = _WEIGHT_REMAPPER.get(model_type)
    is_qwen35 = remapper in (_remap_qwen3_5, _remap_qwen3_5_moe)
    is_baichuan2 = remapper is _remap_baichuan and _is_baichuan2(model.hf_config)
    tie_embedding_key = (
        "backbone.embeddings.weight"
        if remapper is _remap_mamba
        else "model.language_model.embed_tokens.weight"
        if is_qwen35
        else "model.embed_tokens.weight"
    )
    # Ernie uses selected native reads through its single group iterator below.
    raw_checkpoint = (
        not is_ernie
        and (
            remapper
            in (
                None,
                _remap_mamba,
                _remap_glm4,
                _remap_chatglm,
                _remap_videonsa,
                _remap_gpt2,
                _remap_qwen3_next,
                _remap_qwen3_5,
                _remap_qwen3_5_moe,
                _remap_kimi_k3,
            )
            or (
                remapper is _remap_baichuan
                and (
                    not is_baichuan2
                    or dtype
                    in (infinicore.float32, infinicore.float16, infinicore.bfloat16)
                )
            )
        )
        and (
            scale_emb == 1.0
            or dtype in (infinicore.float32, infinicore.float16, infinicore.bfloat16)
        )
    )

    index_file_path = os.path.join(model_path, "model.safetensors.index.json")
    weight_map = None
    if os.path.exists(index_file_path):
        print(f"Found index file: {index_file_path}. Loading shards by index.")
        with open(index_file_path, "r") as f:
            index_data = (
                json.load(f, object_pairs_hook=_ernie_unique_index)
                if is_ernie
                else json.load(f)
            )
        weight_map = index_data.get("weight_map", {})
        if is_ernie:
            if (
                not isinstance(weight_map, dict)
                or not weight_map
                or any(
                    not isinstance(name, str) or not isinstance(filename, str)
                    for name, filename in weight_map.items()
                )
            ):
                raise ValueError(
                    "ERNIE checkpoint index requires a nonempty string weight_map"
                )
            root = os.path.realpath(model_path)
            resolved = {}
            for name, filename in weight_map.items():
                file_path = os.path.realpath(os.path.join(root, filename))
                if (
                    os.path.isabs(filename)
                    or os.path.commonpath((root, file_path)) != root
                ):
                    raise ValueError(
                        "ERNIE checkpoint shard must stay within the model directory"
                    )
                resolved[name] = file_path
            weight_map = resolved
            file_list = sorted(set(weight_map.values()))
        else:
            unique_filenames = sorted(set(weight_map.values()))
            file_list = [os.path.join(model_path, fname) for fname in unique_filenames]
    else:
        print("No index file found. Scanning all safetensors files...")
        file_list = sorted(glob.glob(os.path.join(model_path, "*.safetensors")))
    is_bin = not file_list

    def checkpoint_batches():
        if is_ernie:
            if file_list:
                catalog = _ernie_checkpoint_catalog(file_list, weight_map)

                def read_keys(file_path, names):
                    raw = True
                    if names and all(
                        re.fullmatch(
                            r"model\.layers\.\d+\.mlp\.experts\.\d+\."
                            r"(?:gate_proj|up_proj|down_proj)\.(?:weight|bias)",
                            name,
                        )
                        for name in names
                    ):
                        with SafeTensorsMmap(file_path) as shard:
                            raw = all(
                                shard.tensors[name].dtype in ("F16", "BF16", "F32")
                                for name in names
                            )
                    return _load_checkpoint_shard(
                        file_path, None, (), raw=raw, keys=names
                    )
            else:
                file_path = os.path.join(model_path, "pytorch_model.bin")
                if not os.path.exists(file_path):
                    raise KeyError("Weight file not found.")
                source = _load_pytorch_bin(file_path)
                catalog = {
                    name: (None, tuple(tensor.shape)) for name, tensor in source.items()
                }

                def read_keys(_file_path, names):
                    return {name: source.pop(name) for name in names}

            for label, params in _iter_ernie_checkpoint_batches(
                catalog, read_keys, model.hf_config, dtype
            ):
                try:
                    yield label, params, False
                finally:
                    del params
            return
        if file_list:
            for file_path in file_list:
                shard_raw = raw_checkpoint
                selection = {}
                if shard_raw and (
                    scale_emb != 1.0
                    or is_baichuan2
                    or remapper
                    in (_remap_qwen3_next, _remap_qwen3_5, _remap_qwen3_5_moe)
                ):
                    with SafeTensorsMmap(file_path) as shard:
                        if is_qwen35:
                            selection["keys"] = sorted(shard.tensors)
                        if is_baichuan2:
                            head = shard.tensors.get("lm_head.weight")
                            if head is not None and (
                                head.dtype not in ("F16", "BF16", "F32")
                                or len(head.shape) != 2
                            ):
                                shard_raw = False
                        if scale_emb != 1.0:
                            embedding = shard.tensors.get("model.embed_tokens.weight")
                            if embedding is not None and embedding.dtype not in (
                                "F16",
                                "BF16",
                                "F32",
                            ):
                                shard_raw = False
                        shard_raw = shard_raw and all(
                            info.dtype in ("F16", "BF16", "F32")
                            for name, info in shard.tensors.items()
                            if _is_qwen_offset_norm(name)
                        )
                params = _load_checkpoint_shard(
                    file_path, None, (), raw=shard_raw, **selection
                )
                if remapper is not None:
                    if remapper in (
                        _remap_mamba,
                        _remap_gpt2,
                        _remap_qwen3_5,
                        _remap_qwen3_5_moe,
                    ):
                        params = remapper(
                            params, config=model.hf_config, defer_tie=True
                        )
                    else:
                        params = remapper(params, config=model.hf_config)
                try:
                    yield os.path.basename(file_path), params, False
                finally:
                    params.clear()
                    del params
        elif os.path.exists(os.path.join(model_path, "pytorch_model.bin")):
            params = _load_pytorch_bin(os.path.join(model_path, "pytorch_model.bin"))
            if remapper is not None:
                if is_qwen35:
                    params = remapper(params, config=model.hf_config, defer_tie=True)
                else:
                    params = remapper(params, config=model.hf_config)
            try:
                yield "pytorch_model.bin", params, True
            finally:
                params.clear()
                del params
        else:
            raise KeyError("Weight file not found.")

    for label, model_param, strict in tqdm(
        checkpoint_batches(), desc="Processing files"
    ):
        tqdm.write(f"Processing: {label}")
        embedding_key = (
            "model.embed_tokens.weight"
            if is_bin and not is_qwen35
            else tie_embedding_key
        )
        if embedding_key in model_param:
            embed_tokens_torch_unscaled = model_param[embedding_key]
            if (
                scale_emb != 1.0
                and embedding_key == "model.embed_tokens.weight"
                and embedding_key in parameter_dtypes
            ):
                model_param[embedding_key] = _scale_checkpoint_tensor(
                    _prepare_checkpoint_tensor(
                        embed_tokens_torch_unscaled, parameter_dtypes[embedding_key]
                    ),
                    float(scale_emb),
                )
        if is_pipeline_parallel:
            model_param = {
                key: tensor
                for key, tensor in model_param.items()
                if key in model_key_set
            }
        already_loaded_keys.extend(model_param.keys())
        model_param_infini = {}
        for key in model_param:
            tensor = model_param[key]
            model_param_infini[key] = _prepare_checkpoint_tensor(
                tensor, parameter_dtypes.get(key)
            )
        if model_param:
            del tensor
        model.load_state_dict(model_param_infini, strict=strict)
        # Native load joins each rank's copy before these acyclic owners are released.
        del model_param_infini
        del model_param
        if not (
            tie_word_embeddings
            and "lm_head.weight" in model_keys
            and "lm_head.weight" not in already_loaded_keys
        ):
            embed_tokens_torch_unscaled = None

    if (
        tie_word_embeddings
        and "lm_head.weight" in model_keys
        and "lm_head.weight" not in already_loaded_keys
    ):
        if embed_tokens_torch_unscaled is not None:
            lm_head_tensor = _prepare_checkpoint_tensor(
                embed_tokens_torch_unscaled, parameter_dtypes["lm_head.weight"]
            )
            model.load_state_dict({"lm_head.weight": lm_head_tensor}, strict=False)
            already_loaded_keys.append("lm_head.weight")
            del lm_head_tensor
            embed_tokens_torch_unscaled = None

    # Native finalization remains the sole required-name/Ready boundary.
    model.process_weights_after_loading()
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
        k: v for k, v in state_dict.items() if not any(sub in k for sub in substrings)
    }


def rename_keys(
    state_dict: Dict[str, torch.Tensor],
    mapping: Dict[str, str],
    *,
    reject_collision: bool = False,
) -> Dict[str, torch.Tensor]:
    """Rename weight keys according to a substring mapping."""
    result = {}
    for key, tensor in state_dict.items():
        new_key = key
        for old_str, new_str in mapping.items():
            new_key = new_key.replace(old_str, new_str)
        if reject_collision and new_key in result:
            raise ValueError(f"weight remap collision: {new_key}")
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
        ndim = len(tensor.shape)
        dim = split_dim if split_dim >= 0 else split_dim + ndim
        if dim < 0 or dim >= ndim:
            raise IndexError(f"split dimension {split_dim} is out of range")
        dim_size = tensor.shape[dim]

        # Calculate split sizes
        if not output_names:
            raise ValueError("fused split requires output names")
        if split_sizes is not None:
            if len(split_sizes) != len(output_names) or split_sizes.count(-1) > 1:
                raise ValueError(
                    "split sizes must match output names with at most one remainder"
                )
            remainder = dim_size - sum(size for size in split_sizes if size != -1)
            sizes = [remainder if size == -1 else size for size in split_sizes]
        else:
            num_splits = len(output_names)
            chunk = dim_size // num_splits
            sizes = [chunk] * (num_splits - 1)
            sizes.append(dim_size - chunk * (num_splits - 1))

        if any(size < 0 for size in sizes) or sum(sizes) != dim_size:
            raise ValueError(f"invalid fused split sizes for {key}: {sizes}")
        if isinstance(tensor, infinicore.Tensor):
            start = 0
            splits = []
            for size in sizes:
                splits.append(tensor.narrow(dim, start, size))
                start += size
        else:
            splits = tensor.split(sizes, dim=split_dim)
        for name, split_tensor in zip(output_names, splits):
            output_key = f"{base_key}.{name}.{suffix}"
            if output_key in result or output_key in state_dict:
                raise ValueError(f"fused weight remap collision: {output_key}")
            result[output_key] = split_tensor

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
    state_dict = split_fused_weight(
        state_dict,
        fused_key="query_key_value",
        output_names=["q_proj", "k_proj", "v_proj"],
        split_sizes=[q_dim, k_dim, -1],
    )

    # 3. Split gate_up
    state_dict = split_fused_weight(
        state_dict,
        fused_key="dense_h_to_4h",
        output_names=["gate_proj", "up_proj"],
        split_sizes=[ffn_hidden, -1],
    )

    # 4. Rename keys
    state_dict = rename_keys(
        state_dict,
        {
            "transformer.encoder.layers.": "model.layers.",
            "transformer.embedding.word_embeddings": "model.embed_tokens",
            "transformer.encoder.final_layernorm": "model.norm",
            "transformer.output_layer": "lm_head",
            "self_attention.": "self_attn.",
            "self_attn.dense": "self_attn.o_proj",
            "mlp.dense_4h_to_h": "mlp.down_proj",
        },
        reject_collision=True,
    )

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
    hf_config = config or {}
    hidden_size = hf_config.get("hidden_size", 4096)
    num_heads = hf_config.get("num_attention_heads", 32)
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
        state_dict["lm_head.weight"] = _normalize_checkpoint_tensor(
            state_dict["lm_head.weight"]
        )

    return state_dict


def _remap_gpt2(state_dict, config=None, *, defer_tie=False):
    """Remap HuggingFace GPT-2 weights to InfiniLM GPT-2 module names.

    HuggingFace GPT-2 uses Conv1D modules whose weights are stored as
    [in_features, out_features]. InfiniLM Linear expects [out_features,
    in_features], so projection weights must be transposed.
    """
    remapped = {}

    def publish(name, tensor):
        if name in remapped:
            raise ValueError(f"GPT2 weight remap collision: {name}")
        remapped[name] = tensor

    for key, tensor in state_dict.items():
        key = key.removeprefix("transformer.")
        if key.endswith((".attn.bias", ".attn.masked_bias")):
            continue

        new_key = key

        if new_key.startswith("wte."):
            new_key = new_key.replace("wte.", "model.embed_tokens.", 1)
        elif new_key.startswith("wpe."):
            new_key = new_key.replace("wpe.", "model.embed_positions.", 1)
        elif new_key.startswith("h."):
            new_key = new_key.replace("h.", "model.layers.", 1)
        elif new_key.startswith("ln_f."):
            new_key = new_key.replace("ln_f.", "model.norm.", 1)

        if key.endswith(
            (
                "attn.c_proj.weight",
                "mlp.c_fc.weight",
                "mlp.c_proj.weight",
                "attn.c_attn.weight",
            )
        ):
            if len(tensor.shape) != 2 or any(size <= 0 for size in tensor.shape):
                raise ValueError(
                    f"GPT2 Conv1D requires a positive 2D weight: {new_key}"
                )
            tensor = tensor.permute([1, 0])

        new_key = new_key.replace(".attn.c_proj.", ".attn.o_proj.")
        if new_key.endswith(".attn.o_proj.bias"):
            new_key = new_key.removesuffix(".attn.o_proj.bias") + ".attn.o_proj_bias"
        elif new_key.endswith(".mlp.c_proj.bias"):
            new_key = new_key.removesuffix(".mlp.c_proj.bias") + ".mlp.c_proj_bias"

        if new_key.endswith((".attn.c_attn.weight", ".attn.c_attn.bias")):
            is_weight = new_key.endswith(".weight")
            shape = tensor.shape
            if (
                len(shape) != (2 if is_weight else 1)
                or shape[0] <= 0
                or shape[0] % 3
                or (is_weight and shape[0] != 3 * shape[1])
            ):
                raise ValueError(f"GPT2 fused QKV requires 3H storage: {new_key}")
            splits = split_fused_weight(
                {new_key: tensor}, "c_attn", ["q_proj", "k_proj", "v_proj"]
            )
            for name, part in splits.items():
                publish(name, part)
        else:
            publish(new_key, tensor)

    if (
        not defer_tie
        and _declares_tied_embeddings(config or {"model_type": "gpt2"})
        and "lm_head.weight" not in remapped
        and "model.embed_tokens.weight" in remapped
    ):
        remapped["lm_head.weight"] = remapped["model.embed_tokens.weight"]

    return remapped


def _remap_mamba(state_dict, config=None, *, defer_tie=False):
    """Remap HuggingFace Mamba weights to InfiniLM native names."""
    remapped = {}
    for key, tensor in state_dict.items():
        new_key = key
        new_key = new_key.replace(".mixer.conv1d.weight", ".mixer.conv1d_weight")
        new_key = new_key.replace(".mixer.conv1d.bias", ".mixer.conv1d_bias")
        if defer_tie and new_key in remapped:
            raise ValueError(f"Mamba weight remap collision: {new_key}")
        remapped[new_key] = tensor
    if (
        not defer_tie
        and _declares_tied_embeddings(config)
        and "lm_head.weight" not in remapped
        and "backbone.embeddings.weight" in remapped
    ):
        remapped["lm_head.weight"] = remapped["backbone.embeddings.weight"]
    return remapped


def _remap_videonsa(state_dict, config=None):
    """Adapt VideoNSA/Qwen2.5-VL weights to the InfiniLM C++ module layout."""
    key = "visual.patch_embed.proj.weight"
    if key in state_dict:
        state_dict = dict(state_dict)
        if state_dict[key].ndim == 5:
            tensor = state_dict[key]
            if tensor.shape[0] == 0:
                raise ValueError(
                    "VideoNSA patch weight cannot infer a zero output dimension"
                )
            state_dict[key] = tensor.contiguous().view(
                [tensor.shape[0], math.prod(tensor.shape[1:])]
            )
        state_dict["visual.patch_embed.proj_weight"] = state_dict[key]
    return state_dict


# Model type → remap function mapping
def _remap_qwen3_5(state_dict, config, *, defer_tie=False):
    """Apply Qwen3.5-specific load-time weight fixes."""
    state_dict = drop_keys(state_dict, ["mtp."])
    llm_config = config["text_config"]
    key_dim = llm_config["linear_key_head_dim"] * llm_config["linear_num_key_heads"]

    to_drop = []
    to_add = {}
    for key, tensor in state_dict.items():
        if _is_qwen_offset_norm(key):
            state_dict[key] = _offset_qwen_norm_weight(tensor)
        elif key.endswith("linear_attn.in_proj_qkv.weight"):
            prefix = key[: -len("in_proj_qkv.weight")]
            if tensor.ndim < 2:
                raise IndexError(
                    f"too many indices for tensor of dimension {tensor.ndim}"
                )
            for name, bounds in (
                ("q", (None, key_dim)),
                ("k", (key_dim, key_dim * 2)),
                ("v", (key_dim * 2, None)),
            ):
                start, stop, _ = slice(*bounds).indices(tensor.shape[0])
                to_add[prefix + f"in_proj_{name}.weight"] = tensor.narrow(
                    0, start, max(stop - start, 0)
                ).contiguous()
            to_drop.append(key)

    state_dict = drop_keys(state_dict, to_drop)
    state_dict.update(to_add)

    embed_tokens_key = "model.language_model.embed_tokens.weight"
    if (
        not defer_tie
        and _declares_tied_embeddings(config)
        and embed_tokens_key in state_dict
        and "lm_head.weight" not in state_dict
    ):
        state_dict["lm_head.weight"] = state_dict[embed_tokens_key]

    return state_dict


def _ernie_unique_index(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate ERNIE checkpoint index key: {key}")
        result[key] = value
    return result


def _ernie_checkpoint_catalog(file_list, weight_map=None):
    """Keep only source names and shapes, never shard tensor owners."""
    catalog = {}
    for file_path in file_list:
        with SafeTensorsMmap(file_path) as shard:
            if shard.metadata is not None and shard.metadata.get("format") not in (
                "pt",
                "tf",
                "flax",
                "mlx",
            ):
                raise OSError(f"Invalid safetensors metadata at {file_path}")
            for name, info in shard.tensors.items():
                if name in catalog:
                    raise ValueError(f"Duplicate ERNIE checkpoint source: {name}")
                catalog[name] = (file_path, info.shape)
    if weight_map is not None:
        if set(catalog) != set(weight_map):
            raise ValueError("ERNIE checkpoint index and shard keys differ")
        for name, file_path in weight_map.items():
            if catalog[name][0] != file_path:
                raise ValueError(
                    f"ERNIE checkpoint index maps {name} to the wrong shard"
                )
    return catalog


def _iter_ernie_checkpoint_batches(catalog, read_keys, config, dtype):
    """Assemble one owning expert group, then release it after publication."""
    config = config or {}
    target_dtype = infinicore.bfloat16
    for key in ("torch_dtype", "dtype"):
        if config.get(key) in ("float16", "float32", "bfloat16"):
            target_dtype = parse_dtype(config[key])
            break
    counts = config.get("moe_num_experts", config.get("num_experts"))
    counts = counts if isinstance(counts, list) else [counts]
    if not 1 <= len(counts) <= 2 or any(type(e) is not int or e <= 0 for e in counts):
        raise ValueError("ERNIE checkpoint packing requires one or two expert counts")
    total_experts = sum(counts)
    expert_re = re.compile(
        r"^(?P<prefix>model\.layers\.\d+\.mlp\.experts)\."
        r"(?P<expert>\d+)\.(?P<proj>gate_proj|up_proj|down_proj)\."
        r"(?P<kind>weight|bias)$"
    )
    groups, ordinary = {}, {}
    for name, (file_path, shape) in catalog.items():
        match = expert_re.match(name)
        if match is None:
            ordinary.setdefault(file_path, []).append(name)
            continue
        expert = int(match["expert"])
        if expert >= total_experts:
            raise ValueError(f"ERNIE checkpoint expert ID is out of range: {name}")
        group = int(expert >= counts[0])
        slot = expert - (counts[0] if group else 0)
        parts = groups.setdefault((match["prefix"], group), {})
        part = (slot, match["proj"], match["kind"])
        if part in parts:
            raise ValueError(f"Duplicate ERNIE checkpoint expert part: {name}")
        parts[part] = name

    # Preflight every group before publishing any tensor; no partial-bias dropping.
    plans = []
    use_bias = config.get("mlp_bias", config.get("use_bias", False))
    for (prefix, group), parts in sorted(groups.items()):
        suffix = "_1" if group else ""
        for kind in ("w1", "w2", "b1", "b2"):
            if f"{prefix}.{kind}{suffix}" in catalog:
                raise ValueError(
                    f"ERNIE individual and packed weights collide: {prefix}"
                )
        expected = {
            (i, proj, "weight")
            for i in range(counts[group])
            for proj in ("gate_proj", "up_proj", "down_proj")
        }
        if not expected.issubset(parts):
            raise ValueError(
                f"Missing ERNIE expert projections: {prefix}, group {group}"
            )
        gate_shape = catalog[parts[(0, "gate_proj", "weight")]][1]
        if len(gate_shape) != 2 or not all(gate_shape):
            raise ValueError(f"Invalid ERNIE expert weight shape: {prefix}")
        intermediate, hidden = gate_shape
        if config.get("hidden_size", hidden) != hidden:
            raise ValueError(
                f"ERNIE expert hidden size differs from configuration: {prefix}"
            )
        configured = config.get("moe_intermediate_size")
        if configured is not None:
            configured = configured if isinstance(configured, list) else [configured]
            if configured[group if len(configured) > 1 else 0] != intermediate:
                raise ValueError(
                    f"ERNIE expert intermediate size differs from configuration: {prefix}"
                )
        biases = {
            (i, proj, "bias")
            for i in range(counts[group])
            for proj in ("gate_proj", "up_proj", "down_proj")
        }
        present = set(parts) & biases
        if present and present != biases:
            raise ValueError(f"Partial ERNIE expert biases: {prefix}, group {group}")
        if bool(present) != bool(use_bias):
            raise ValueError(
                f"ERNIE expert bias presence differs from configuration: {prefix}"
            )
        for (slot, proj, kind), name in parts.items():
            expected_shape = (
                (hidden, intermediate)
                if proj == "down_proj"
                else (intermediate, hidden)
            )
            if kind == "bias":
                expected_shape = (expected_shape[0],)
            if tuple(catalog[name][1]) != expected_shape:
                raise ValueError(f"Invalid ERNIE expert projection shape: {name}")
        plans.append((prefix, group, parts, intermediate, hidden, bool(present)))
    # A standard individual-expert prefix must contain all configured groups.
    for prefix in {prefix for prefix, _ in groups}:
        if any((prefix, group) not in groups for group in range(len(counts))):
            raise ValueError(f"Missing ERNIE expert group: {prefix}")

    for file_path, names in ordinary.items():
        batch = read_keys(file_path, names)
        try:
            yield str(file_path), batch
        finally:
            batch.clear()
            del batch

    for prefix, group, parts, intermediate, hidden, has_bias in plans:
        suffix = "_1" if group else ""
        packed = {
            f"{prefix}.w1{suffix}": infinicore.empty(
                (counts[group], 2 * intermediate, hidden),
                dtype=target_dtype,
                device="cpu",
            ),
            f"{prefix}.w2{suffix}": infinicore.empty(
                (counts[group], hidden, intermediate), dtype=target_dtype, device="cpu"
            ),
        }
        if has_bias:
            packed[f"{prefix}.b1{suffix}"] = infinicore.empty(
                (counts[group], 2 * intermediate), dtype=target_dtype, device="cpu"
            )
            packed[f"{prefix}.b2{suffix}"] = infinicore.empty(
                (counts[group], hidden), dtype=target_dtype, device="cpu"
            )
        try:
            files = dict.fromkeys(catalog[name][0] for name in parts.values())
            for file_path in files:
                names = [
                    name for name in parts.values() if catalog[name][0] == file_path
                ]
                selected = read_keys(file_path, names)
                try:
                    for (slot, proj, kind), name in parts.items():
                        if name not in selected:
                            continue
                        source = selected[name]
                        converted = borrowed = target_view = None
                        try:
                            converted = _cast_checkpoint_tensor(source, target_dtype)
                            borrowed = _borrow_checkpoint_tensor(converted)
                            if proj == "down_proj":
                                target = f"{prefix}.{'w2' if kind == 'weight' else 'b2'}{suffix}"
                                target_view = (
                                    packed[target].narrow(0, slot, 1).squeeze(0)
                                )
                            else:
                                target = f"{prefix}.{'w1' if kind == 'weight' else 'b1'}{suffix}"
                                start = intermediate if proj == "up_proj" else 0
                                target_view = (
                                    packed[target]
                                    .narrow(0, slot, 1)
                                    .squeeze(0)
                                    .narrow(0, start, intermediate)
                                )
                            target_view.copy_(borrowed)
                        finally:
                            del target_view, borrowed, converted, source
                finally:
                    selected.clear()
                    del selected
            yield f"{prefix}, group {group}", packed
        finally:
            packed.clear()
            del packed


def _is_qwen_offset_norm(key):
    return "mtp." not in key and (
        key in ("model.norm.weight", "model.language_model.norm.weight")
        or key.endswith(
            (
                "input_layernorm.weight",
                "post_attention_layernorm.weight",
                "self_attn.q_norm.weight",
                "self_attn.k_norm.weight",
            )
        )
    )


def _offset_qwen_norm_weight(tensor):
    one = (
        infinicore.ones(tensor.shape, dtype=tensor.dtype, device="cpu")
        if isinstance(tensor, infinicore.Tensor)
        else tensor.new_ones(tensor.shape)
    )
    return tensor + one


def _remap_qwen3_next(state_dict, config):
    """Adapt Qwen3Next fused linear-attention weights to InfiniLM module names."""
    state_dict = drop_keys(state_dict, ["mtp."])
    num_key_heads = config["linear_num_key_heads"]
    num_value_heads = config["linear_num_value_heads"]
    key_head_dim = config["linear_key_head_dim"]
    value_head_dim = config["linear_value_head_dim"]
    if any(
        type(value) is not int or value <= 0
        for value in (num_key_heads, num_value_heads, key_head_dim, value_head_dim)
    ):
        raise ValueError(
            "Qwen3Next linear-attention geometry must contain positive integers"
        )
    if num_value_heads % num_key_heads:
        raise ValueError("Qwen3Next value heads must be divisible by key heads")
    values_per_key = num_value_heads // num_key_heads
    value_group_dim = values_per_key * value_head_dim

    def pack_grouped_rows(tensor, widths, key):
        group_rows = sum(widths)
        shape = tensor.shape
        if len(shape) != 2 or shape[0] != num_key_heads * group_rows or shape[1] <= 0:
            raise ValueError(
                f"Qwen3Next {key} requires {num_key_heads * group_rows} rows "
                f"and positive input width, got shape {tuple(shape)}"
            )
        outputs = []
        for width in widths:
            output_shape = [num_key_heads * width, shape[1]]
            output = (
                infinicore.empty(output_shape, dtype=tensor.dtype, device="cpu")
                if isinstance(tensor, infinicore.Tensor)
                else tensor.new_empty(output_shape)
            )
            outputs.append(output)
        # Each full row span is contiguous for mmap input; no strided gather.
        for head in range(num_key_heads):
            offset = 0
            for width, output in zip(widths, outputs):
                source_rows = tensor.narrow(0, head * group_rows + offset, width)
                output_rows = output.narrow(0, head * width, width)
                output_rows.copy_(source_rows)
                del source_rows, output_rows
                offset += width
        return outputs

    to_drop = []
    to_add = {}
    for key, tensor in state_dict.items():
        if _is_qwen_offset_norm(key):
            state_dict[key] = _offset_qwen_norm_weight(tensor)
        elif key.endswith("linear_attn.in_proj_qkvz.weight"):
            prefix = key[: -len("in_proj_qkvz.weight")]
            q, k, v, z = pack_grouped_rows(
                tensor,
                (key_head_dim, key_head_dim, value_group_dim, value_group_dim),
                key,
            )
            to_add[prefix + "in_proj_q.weight"] = q
            to_add[prefix + "in_proj_k.weight"] = k
            to_add[prefix + "in_proj_v.weight"] = v
            to_add[prefix + "in_proj_z.weight"] = z
            to_drop.append(key)
        elif key.endswith("linear_attn.in_proj_ba.weight"):
            prefix = key[: -len("in_proj_ba.weight")]
            b, a = pack_grouped_rows(tensor, (values_per_key, values_per_key), key)
            to_add[prefix + "in_proj_b.weight"] = b
            to_add[prefix + "in_proj_a.weight"] = a
            to_drop.append(key)

    state_dict = drop_keys(state_dict, to_drop)
    state_dict.update(to_add)
    return state_dict


def _remap_qwen3_5_moe(state_dict, config, *, defer_tie=False):
    """Adapt packed Qwen3.5-MoE experts to InfiniLM expert parameter names."""
    state_dict = _remap_qwen3_5(state_dict, config, defer_tie=defer_tie)
    text_config = config.get("text_config", config)
    expected_num_experts = text_config["num_experts"]
    expected_intermediate_size = text_config["moe_intermediate_size"]

    remapped = {}
    for key, tensor in state_dict.items():
        if key.endswith(".mlp.experts.gate_up_proj"):
            if tensor.ndim != 3:
                raise ValueError(
                    f"Expected packed gate_up_proj to be 3D, got {tensor.shape} for {key}"
                )
            if tensor.shape[0] != expected_num_experts:
                raise ValueError(
                    f"Expected {expected_num_experts} experts, got {tensor.shape[0]} for {key}"
                )
            if tensor.shape[1] != expected_intermediate_size * 2:
                raise ValueError(
                    f"Expected packed gate/up size {expected_intermediate_size * 2}, "
                    f"got {tensor.shape[1]} for {key}"
                )

            prefix = key[: -len("gate_up_proj")]
            for expert_idx in range(tensor.shape[0]):
                expert_gate_up = tensor.narrow(0, expert_idx, 1).squeeze(0)
                if tensor.shape[1] == 1:
                    raise ValueError("not enough values to unpack (expected 2, got 1)")
                width = (tensor.shape[1] + 1) // 2
                gate = expert_gate_up.narrow(0, 0, width)
                up = expert_gate_up.narrow(0, width, tensor.shape[1] - width)
                expert_prefix = f"{prefix}{expert_idx}."
                remapped[f"{expert_prefix}gate_proj.weight"] = gate
                remapped[f"{expert_prefix}up_proj.weight"] = up
        elif key.endswith(".mlp.experts.down_proj"):
            if tensor.ndim != 3:
                raise ValueError(
                    f"Expected packed down_proj to be 3D, got {tensor.shape} for {key}"
                )
            if tensor.shape[0] != expected_num_experts:
                raise ValueError(
                    f"Expected {expected_num_experts} experts, got {tensor.shape[0]} for {key}"
                )
            if tensor.shape[2] != expected_intermediate_size:
                raise ValueError(
                    f"Expected down projection input size {expected_intermediate_size}, "
                    f"got {tensor.shape[2]} for {key}"
                )

            prefix = key[: -len("down_proj")]
            for expert_idx in range(tensor.shape[0]):
                expert_down = tensor.narrow(0, expert_idx, 1).squeeze(0)
                remapped[f"{prefix}{expert_idx}.down_proj.weight"] = expert_down
        else:
            remapped[key] = tensor

    return remapped


def _remap_kimi_k3(state_dict, config):
    """Adapt released Kimi-K3 KDA weights to the reference module layout."""
    text_config = config.get("text_config", config)
    num_heads = text_config["linear_attn_config"]["num_heads"]

    for key, tensor in state_dict.items():
        if key.endswith(".self_attn.A_log") and tuple(tensor.shape) != (num_heads,):
            if tensor.ndim != 1 or tensor.shape[0] < num_heads:
                raise ValueError(
                    f"Kimi-K3 A_log must contain at least {num_heads} values, "
                    f"but {key} has shape {tuple(tensor.shape)}"
                )
            # Released Kimi-K3 checkpoints contain head_dim entries although
            # the bundled reference module and KDA ABI consume one per head.
            stop = slice(None, num_heads).indices(tensor.shape[0])[1]
            state_dict[key] = tensor.narrow(0, 0, stop).contiguous()

    return state_dict


_WEIGHT_REMAPPER = {
    "glm4": _remap_glm4,
    "chatglm": _remap_chatglm,
    "baichuan": _remap_baichuan,
    "gpt2": _remap_gpt2,
    "mamba": _remap_mamba,
    "videonsa": _remap_videonsa,
    "qwen3_5": _remap_qwen3_5,
    "qwen3_5_moe": _remap_qwen3_5_moe,
    "qwen3_next": _remap_qwen3_next,
    "kimi_k3": _remap_kimi_k3,
}
