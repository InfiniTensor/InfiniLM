import json
import os
import numpy as np

from infinilm.models.llama.configuration_llama import LlamaConfig

try:
    from infinicore.gguf import GGUFReader, find_split_files
    GGUF_AVAILABLE = True
except ImportError:
    GGUF_AVAILABLE = False


def _to_python_value(value):
    """Convert numpy array/scalar to Python native type."""
    if value is None:
        return None
    if isinstance(value, np.ndarray):
        if value.size == 0:
            return None
        if value.size == 1:
            return value.item()
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, bytes):
        return value.decode('utf-8')
    return value


class AutoConfig:
    def from_pretrained(model_path):
        # Check if model_path is a GGUF file (explicit .gguf file)
        if os.path.isfile(model_path) and model_path.endswith(".gguf"):
            if not GGUF_AVAILABLE:
                raise ImportError("GGUF support not available. Install gguf-py: pip install gguf")
            return AutoConfig._from_gguf(model_path)

        # Determine the directory to check
        # If model_path is a file (but not .gguf), use its directory
        if os.path.isfile(model_path):
            model_dir = os.path.dirname(model_path)
        else:
            model_dir = model_path

        # Check for config.json first (HuggingFace format takes priority)
        config_path = os.path.join(model_dir, "config.json")
        if os.path.exists(config_path):
            # HuggingFace format detected, use config.json
            pass  # Continue to config.json loading below
        elif os.path.isdir(model_dir):
            # No config.json, check if directory contains GGUF files
            import glob
            gguf_files = glob.glob(os.path.join(model_dir, "*.gguf"))
            if len(gguf_files) > 0 and GGUF_AVAILABLE:
                # Use first GGUF file found (only if no config.json exists)
                return AutoConfig._from_gguf(gguf_files[0])

        # Load from config.json (HuggingFace format)

        if not os.path.exists(config_path):
            raise FileNotFoundError(f"`{config_path}` not found")

        with open(config_path) as f:
            config_dict = json.load(f)

        if "model_type" not in config_dict:
            raise ValueError(
                f"`model_type` is not specified in the config file `{config_path}`."
            )

        if config_dict["model_type"] == "llama":
            return LlamaConfig(**config_dict)
        elif config_dict["model_type"] == "qwen2":
            return LlamaConfig(**config_dict)

        raise ValueError(f"Unsupported model type `{config_dict['model_type']}`.")

    @staticmethod
    def _from_gguf(gguf_file: str):
        """Load config from GGUF file metadata."""
        reader = GGUFReader(gguf_file)

        # Get architecture and convert to Python string
        architecture_raw = reader.get_metadata("general.architecture", "llama")
        architecture = _to_python_value(architecture_raw)
        if isinstance(architecture, bytes):
            architecture = architecture.decode('utf-8')
        elif not isinstance(architecture, str):
            architecture = str(architecture)

        # Default to "llama" if architecture is None or empty
        if not architecture:
            architecture = "llama"

        # Map architecture to model_type
        if architecture in ["llama", "llama3"]:
            model_type = "llama"
        elif architecture in ["qwen2", "qwen2_5"]:
            model_type = "qwen2"
        else:
            # Default to llama for unknown architectures
            model_type = "llama"

        # Build config dict from GGUF metadata
        config_dict = {
            "model_type": model_type,
        }

        # Map GGUF keys to config keys
        # Block count -> num_hidden_layers
        block_count = _to_python_value(reader.get_metadata(f"{architecture}.block_count"))
        if block_count is not None:
            config_dict["num_hidden_layers"] = int(block_count)

        # Embedding length -> hidden_size
        embedding_length = _to_python_value(reader.get_metadata(f"{architecture}.embedding_length"))
        hidden_size = None
        if embedding_length is not None:
            hidden_size = int(embedding_length)
            print(f"Detected hidden_size from metadata: {hidden_size}")

        # If metadata not available, infer from tensor shapes
        if hidden_size is None:
            try:
                tensor_names = reader.get_tensor_names()
                for tensor_name in tensor_names:
                    # Try token_embd.weight (shape: [vocab_size, hidden_size])
                    if tensor_name == "token_embd.weight":
                        try:
                            np_array = reader.get_tensor_data(tensor_name)
                            if hasattr(np_array, 'shape') and len(np_array.shape) >= 2:
                                # token_embd.weight is [vocab_size, hidden_size]
                                hidden_size = int(np_array.shape[1])
                                print(f"Detected hidden_size from {tensor_name} shape: {np_array.shape} -> {hidden_size}")
                                break
                        except Exception as e:
                            pass
                    # Try any layer norm weight (shape: [hidden_size])
                    elif "norm" in tensor_name and "weight" in tensor_name:
                        try:
                            np_array = reader.get_tensor_data(tensor_name)
                            if hasattr(np_array, 'shape') and len(np_array.shape) == 1:
                                hidden_size = int(np_array.shape[0])
                                print(f"Detected hidden_size from {tensor_name} shape: {np_array.shape} -> {hidden_size}")
                                break
                        except Exception as e:
                            pass
            except Exception as e:
                print(f"Warning: Failed to detect hidden_size from tensor shape: {e}")

        if hidden_size is not None:
            config_dict["hidden_size"] = int(hidden_size)
        else:
            print(f"Warning: Could not detect hidden_size from GGUF, using default")

        # Feed forward length -> intermediate_size
        feed_forward_length = _to_python_value(reader.get_metadata(f"{architecture}.feed_forward_length"))
        intermediate_size = None
        if feed_forward_length is not None:
            intermediate_size = int(feed_forward_length)
            print(f"Detected intermediate_size from metadata: {intermediate_size}")

        # If metadata not available, infer from tensor shapes
        if intermediate_size is None:
            try:
                tensor_names = reader.get_tensor_names()
                for tensor_name in tensor_names:
                    # Try ffn_down.weight or blk.0.ffn_down.weight (shape: [hidden_size, intermediate_size])
                    if "ffn_down" in tensor_name or "mlp.down_proj" in tensor_name:
                        try:
                            np_array = reader.get_tensor_data(tensor_name)
                            if hasattr(np_array, 'shape') and len(np_array.shape) >= 2:
                                # ffn_down.weight is [hidden_size, intermediate_size]
                                intermediate_size = int(np_array.shape[1])
                                print(f"Detected intermediate_size from {tensor_name} shape: {np_array.shape} -> {intermediate_size}")
                                break
                        except Exception as e:
                            pass
                    # Try ffn_gate.weight or blk.0.ffn_gate.weight (shape: [intermediate_size, hidden_size])
                    elif "ffn_gate" in tensor_name or "mlp.gate_proj" in tensor_name:
                        try:
                            np_array = reader.get_tensor_data(tensor_name)
                            if hasattr(np_array, 'shape') and len(np_array.shape) >= 2:
                                # ffn_gate.weight is [intermediate_size, hidden_size]
                                intermediate_size = int(np_array.shape[0])
                                print(f"Detected intermediate_size from {tensor_name} shape: {np_array.shape} -> {intermediate_size}")
                                break
                        except Exception as e:
                            pass
            except Exception as e:
                print(f"Warning: Failed to detect intermediate_size from tensor shape: {e}")

        if intermediate_size is not None:
            config_dict["intermediate_size"] = int(intermediate_size)
        else:
            print(f"Warning: Could not detect intermediate_size from GGUF, using default")

        # Attention head count -> num_attention_heads
        head_count = _to_python_value(reader.get_metadata(f"{architecture}.attention.head_count"))
        num_attention_heads = None
        if head_count is not None:
            # head_count might be an array, take first element if so
            if isinstance(head_count, list):
                head_count = head_count[0]
            num_attention_heads = int(head_count)
            print(f"Detected num_attention_heads from metadata: {num_attention_heads}")

        # Attention head count KV -> num_key_value_heads
        head_count_kv = _to_python_value(reader.get_metadata(f"{architecture}.attention.head_count_kv"))
        num_key_value_heads = None
        if head_count_kv is not None:
            if isinstance(head_count_kv, list):
                head_count_kv = head_count_kv[0]
            num_key_value_heads = int(head_count_kv)
            print(f"Detected num_key_value_heads from metadata: {num_key_value_heads}")

        # If metadata not available, infer from tensor shapes
        # Attention weight shapes:
        # - q_proj.weight: [num_attention_heads * head_dim, hidden_size]
        # - k_proj.weight: [num_key_value_heads * head_dim, hidden_size]
        # - v_proj.weight: [num_key_value_heads * head_dim, hidden_size]
        # - o_proj.weight: [hidden_size, num_attention_heads * head_dim]
        # head_dim = hidden_size / num_attention_heads (typically)
        if num_attention_heads is None or num_key_value_heads is None:
            try:
                tensor_names = reader.get_tensor_names()
                hidden_size_val = config_dict.get("hidden_size")

                if hidden_size_val is not None:
                    # First, try to infer num_attention_heads from q_proj or o_proj
                    if num_attention_heads is None:
                        for tensor_name in tensor_names:
                            if "attn_q" in tensor_name:
                                try:
                                    np_array = reader.get_tensor_data(tensor_name)
                                    if hasattr(np_array, 'shape') and len(np_array.shape) >= 2:
                                        q_dim = int(np_array.shape[0])
                                        # Try common head_dim values: 64, 128, 256
                                        for head_dim in [64, 128, 256]:
                                            if q_dim % head_dim == 0:
                                                candidate = q_dim // head_dim
                                                if hidden_size_val % candidate == 0:
                                                    num_attention_heads = candidate
                                                    print(f"Detected num_attention_heads from {tensor_name} shape {np_array.shape}: {num_attention_heads} (head_dim={head_dim})")
                                                    break
                                        if num_attention_heads is not None:
                                            break
                                except Exception:
                                    pass

                    # Then infer num_key_value_heads from k_proj or v_proj
                    if num_key_value_heads is None and hidden_size_val is not None:
                        # Calculate head_dim if we have num_attention_heads
                        head_dim = None
                        if num_attention_heads is not None:
                            head_dim = hidden_size_val // num_attention_heads

                        for tensor_name in tensor_names:
                            if "attn_k" in tensor_name or "attn_v" in tensor_name:
                                try:
                                    np_array = reader.get_tensor_data(tensor_name)
                                    if hasattr(np_array, 'shape') and len(np_array.shape) >= 2:
                                        kv_dim = int(np_array.shape[0])
                                        if head_dim is not None:
                                            # Use calculated head_dim
                                            if kv_dim % head_dim == 0:
                                                num_key_value_heads = kv_dim // head_dim
                                                print(f"Detected num_key_value_heads from {tensor_name} shape {np_array.shape}: {num_key_value_heads} (head_dim={head_dim})")
                                                break
                                        else:
                                            # Try common head_dim values
                                            for head_dim_candidate in [64, 128, 256]:
                                                if kv_dim % head_dim_candidate == 0:
                                                    candidate = kv_dim // head_dim_candidate
                                                    # Verify it makes sense
                                                    if num_attention_heads is None or candidate <= num_attention_heads:
                                                        num_key_value_heads = candidate
                                                        print(f"Detected num_key_value_heads from {tensor_name} shape {np_array.shape}: {num_key_value_heads} (head_dim={head_dim_candidate})")
                                                        break
                                            if num_key_value_heads is not None:
                                                break
                                except Exception:
                                    pass
            except Exception as e:
                print(f"Warning: Failed to detect attention heads from tensor shape: {e}")

        if num_attention_heads is not None:
            config_dict["num_attention_heads"] = int(num_attention_heads)
        else:
            print(f"Warning: Could not detect num_attention_heads from GGUF, using default")

        if num_key_value_heads is not None:
            config_dict["num_key_value_heads"] = int(num_key_value_heads)
            print(f"Using num_key_value_heads: {num_key_value_heads}")
        else:
            print(f"Warning: Could not detect num_key_value_heads from GGUF, will default to num_attention_heads")

        # Rope freq base -> rope_theta
        rope_freq_base = _to_python_value(reader.get_metadata(f"{architecture}.rope.freq_base"))
        if rope_freq_base is not None:
            config_dict["rope_theta"] = float(rope_freq_base)

        # Layer norm RMS eps -> rms_norm_eps
        layer_norm_rms_eps = _to_python_value(reader.get_metadata(f"{architecture}.attention.layer_norm_rms_epsilon"))
        if layer_norm_rms_eps is not None:
            config_dict["rms_norm_eps"] = float(layer_norm_rms_eps)

        # Context length -> max_position_embeddings
        context_length = _to_python_value(reader.get_metadata(f"{architecture}.context_length"))
        if context_length is not None:
            config_dict["max_position_embeddings"] = int(context_length)

        # Vocab size - prioritize tensor shape over metadata (metadata may be incorrect)
        vocab_size = None
        # First, try to get vocab size from token_embd.weight tensor shape (most reliable)
        try:
            tensor_names = reader.get_tensor_names()
            for tensor_name in tensor_names:
                if tensor_name == "token_embd.weight":
                    # Get actual numpy array shape (most reliable)
                    try:
                        np_array = reader.get_tensor_data(tensor_name)
                        if hasattr(np_array, 'shape') and len(np_array.shape) >= 2:
                            # token_embd.weight is [vocab_size, hidden_size]
                            # Use the first dimension as vocab_size
                            vocab_size = int(np_array.shape[0])
                            print(f"Detected vocab_size from token_embd.weight shape: {np_array.shape} -> {vocab_size}")
                            break
                    except Exception as e:
                        print(f"Warning: Failed to get numpy array shape: {e}")
                    # Fallback: try tensor_info (may be reversed)
                    try:
                        tensor_info = reader.get_tensor_info(tensor_name)
                        shape = tensor_info.get("shape", [])
                        if len(shape) >= 2:
                            # GGUF may store shapes in reverse order, try both
                            vocab_size_candidate1 = int(shape[0])
                            vocab_size_candidate2 = int(shape[-1])
                            # Use the larger one as vocab_size (vocab_size > hidden_size typically)
                            vocab_size = max(vocab_size_candidate1, vocab_size_candidate2)
                            print(f"Detected vocab_size from tensor_info shape {shape}: {vocab_size}")
                            break
                    except Exception as e:
                        print(f"Warning: Failed to get tensor_info shape: {e}")
        except Exception as e:
            print(f"Warning: Failed to detect vocab_size from tensor shape: {e}")

        # Fallback to metadata if tensor shape detection failed
        if vocab_size is None:
            vocab_size = _to_python_value(reader.get_metadata("tokenizer.ggml.vocab_size"))
            if vocab_size is None:
                vocab_size = _to_python_value(reader.get_metadata("tokenizer.ggml.token_count"))
            if vocab_size is not None:
                print(f"Detected vocab_size from metadata: {vocab_size}")

        if vocab_size is not None:
            config_dict["vocab_size"] = int(vocab_size)
            print(f"Using vocab_size: {vocab_size}")
        else:
            print(f"Warning: Could not detect vocab_size from GGUF, using default")

        # Determine dtype from tensor types in GGUF file
        # Check a few weight tensors to infer dtype
        torch_dtype = "float16"  # Default
        try:
            tensor_names = reader.get_tensor_names()
            # Look for weight tensors to determine dtype
            for tensor_name in tensor_names[:10]:  # Check first 10 tensors
                tensor_info = reader.get_tensor_info(tensor_name)
                dtype_str = tensor_info.get("dtype", "")
                # Map GGML types to torch dtypes
                if dtype_str in ["F32", "GGML_TYPE_F32"]:
                    torch_dtype = "float32"
                    break
                elif dtype_str in ["F16", "GGML_TYPE_F16"]:
                    torch_dtype = "float16"
                    break
                elif dtype_str in ["BF16", "GGML_TYPE_BF16"]:
                    torch_dtype = "bfloat16"
                    break
                # For quantized types, default to float16
                elif any(q in dtype_str for q in ["Q4", "Q8", "Q2", "Q3", "Q5", "Q6"]):
                    torch_dtype = "float16"
                    break
        except Exception:
            # If we can't determine dtype, use default
            pass

        config_dict["torch_dtype"] = torch_dtype

        # Set defaults for missing values
        defaults = {
            "vocab_size": 32000,
            "hidden_size": 4096,
            "intermediate_size": 11008,
            "num_hidden_layers": 32,
            "num_attention_heads": 32,
            "num_key_value_heads": None,  # Will default to num_attention_heads in LlamaConfig
            "max_position_embeddings": 2048,
            "rms_norm_eps": 1e-6,
            "rope_theta": 10000.0,
            "hidden_act": "silu",
        }

        for key, default_value in defaults.items():
            if key not in config_dict:
                config_dict[key] = default_value

        return LlamaConfig(**config_dict)
