import json
import os

from infinilm.models.llama.configuration_llama import LlamaConfig

try:
    from infinicore.gguf import GGUFReader, find_split_files
    GGUF_AVAILABLE = True
except ImportError:
    GGUF_AVAILABLE = False


class AutoConfig:
    def from_pretrained(model_path):
        # Check if model_path is a GGUF file
        if os.path.isfile(model_path) and model_path.endswith(".gguf"):
            if not GGUF_AVAILABLE:
                raise ImportError("GGUF support not available. Install gguf-py: pip install gguf")
            return AutoConfig._from_gguf(model_path)

        # Check if model_path directory contains GGUF files
        if os.path.isdir(model_path):
            import glob
            gguf_files = glob.glob(os.path.join(model_path, "*.gguf"))
            if len(gguf_files) > 0 and GGUF_AVAILABLE:
                # Use first GGUF file found
                return AutoConfig._from_gguf(gguf_files[0])

        # Fallback to config.json
        # If model_path is a file (but not .gguf), use its directory
        if os.path.isfile(model_path):
            model_path = os.path.dirname(model_path)

        config_path = os.path.join(model_path, "config.json")

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

        # Get architecture
        architecture = reader.get_metadata("general.architecture", "llama")

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
        block_count = reader.get_metadata(f"{architecture}.block_count")
        if block_count is not None:
            config_dict["num_hidden_layers"] = int(block_count)

        # Embedding length -> hidden_size
        embedding_length = reader.get_metadata(f"{architecture}.embedding_length")
        if embedding_length is not None:
            config_dict["hidden_size"] = int(embedding_length)

        # Feed forward length -> intermediate_size
        feed_forward_length = reader.get_metadata(f"{architecture}.feed_forward_length")
        if feed_forward_length is not None:
            config_dict["intermediate_size"] = int(feed_forward_length)

        # Attention head count -> num_attention_heads
        head_count = reader.get_metadata(f"{architecture}.attention.head_count")
        if head_count is not None:
            # head_count might be an array, take first element if so
            if isinstance(head_count, list):
                head_count = head_count[0]
            config_dict["num_attention_heads"] = int(head_count)

        # Attention head count KV -> num_key_value_heads
        head_count_kv = reader.get_metadata(f"{architecture}.attention.head_count_kv")
        if head_count_kv is not None:
            if isinstance(head_count_kv, list):
                head_count_kv = head_count_kv[0]
            config_dict["num_key_value_heads"] = int(head_count_kv)

        # Rope freq base -> rope_theta
        rope_freq_base = reader.get_metadata(f"{architecture}.rope.freq_base")
        if rope_freq_base is not None:
            config_dict["rope_theta"] = float(rope_freq_base)

        # Layer norm RMS eps -> rms_norm_eps
        layer_norm_rms_eps = reader.get_metadata(f"{architecture}.attention.layer_norm_rms_epsilon")
        if layer_norm_rms_eps is not None:
            config_dict["rms_norm_eps"] = float(layer_norm_rms_eps)

        # Context length -> max_position_embeddings
        context_length = reader.get_metadata(f"{architecture}.context_length")
        if context_length is not None:
            config_dict["max_position_embeddings"] = int(context_length)

        # Vocab size
        vocab_size = reader.get_metadata("tokenizer.ggml.vocab_size")
        if vocab_size is not None:
            config_dict["vocab_size"] = int(vocab_size)
        else:
            # Try alternative key
            vocab_size = reader.get_metadata("tokenizer.ggml.token_count")
            if vocab_size is not None:
                config_dict["vocab_size"] = int(vocab_size)

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
