from ....generation.utils import GenerationMixin
import infinicore
from infinilm.models.llama.configuration_llama import LlamaConfig as _LlamaConfig
from infinilm.lib import _infinilm
from infinilm.distributed import DistConfig
import json
import os
from typing import Optional, Union


class LlamaConfig:
    """Llama model configuration adapter for C++ bindings.

    This class wraps configuration_llama.LlamaConfig and provides
    a _underlying property that creates the C++ config object.

    Automatically detects and handles both regular Llama models and Jiuge models
    (fm9g7b, fm9g, minicpm) with appropriate defaults and validation.
    """

    def __init__(self, config_dict=None, **kwargs):
        """Create LlamaConfig from dictionary or keyword arguments"""
        # Use the Python config from configuration_llama
        if isinstance(config_dict, _LlamaConfig):
            self._python_config = config_dict
        else:
            if config_dict is not None and isinstance(config_dict, dict):
                merged = {**config_dict, **kwargs}
            else:
                merged = kwargs
            self._python_config = _LlamaConfig(**merged)

        # Lazy initialization of C++ config
        self._cpp_config = None

        # Detect if this is a jiuge model
        self._is_jiuge_model = self._detect_jiuge_model()

    def _detect_jiuge_model(self):
        """Detect if this is a jiuge-specific model type"""
        model_type = getattr(self._python_config, "model_type", "")
        return model_type in ["fm9g7b", "fm9g", "minicpm"]

    def __getattr__(self, name):
        """Delegate attribute access to Python config"""
        return getattr(self._python_config, name)

    def __setattr__(self, name, value):
        """Delegate attribute setting to Python config"""
        if name.startswith("_"):
            super().__setattr__(name, value)
        else:
            if hasattr(self, "_python_config"):
                setattr(self._python_config, name, value)
                # Invalidate C++ config cache when Python config changes
                self._cpp_config = None
                # Re-detect model type if model_type changes
                if name == "model_type":
                    self._is_jiuge_model = self._detect_jiuge_model()
            else:
                super().__setattr__(name, value)

    @property
    def _underlying(self):
        """Get underlying C++ config object, creating it if needed"""
        if self._cpp_config is None:
            self._cpp_config = _infinilm.LlamaConfig()

            # Copy attributes from Python config to C++ config
            for key in dir(self._python_config):
                if key.startswith("_"):
                    continue
                try:
                    value = getattr(self._python_config, key)
                    if hasattr(self._cpp_config, key) and not callable(value):
                        setattr(self._cpp_config, key, value)
                except (AttributeError, TypeError):
                    pass

            # Handle num_key_value_heads with validation
            python_num_kv_heads = getattr(self._python_config, "num_key_value_heads", None)
            if python_num_kv_heads is None or python_num_kv_heads == 0:
                self._cpp_config.num_key_value_heads = (
                    self._cpp_config.num_attention_heads
                )
            else:
                self._cpp_config.num_key_value_heads = python_num_kv_heads

            # Handle head_dim with validation (critical for GEMM operations)
            python_head_dim = getattr(self._python_config, "head_dim", None)
            if python_head_dim is None or python_head_dim == 0:
                # Compute from hidden_size and num_attention_heads
                if self._cpp_config.hidden_size > 0 and self._cpp_config.num_attention_heads > 0:
                    computed_head_dim = self._cpp_config.hidden_size // self._cpp_config.num_attention_heads
                    self._cpp_config.head_dim = computed_head_dim
                else:
                    raise ValueError(
                        f"Cannot compute head_dim: hidden_size={self._cpp_config.hidden_size}, "
                        f"num_attention_heads={self._cpp_config.num_attention_heads}"
                    )
            else:
                # Use from Python config
                self._cpp_config.head_dim = python_head_dim
                # Validate it matches expected value (warn but allow for flexibility)
                if self._cpp_config.hidden_size > 0 and self._cpp_config.num_attention_heads > 0:
                    expected_head_dim = self._cpp_config.hidden_size // self._cpp_config.num_attention_heads
                    if self._cpp_config.head_dim != expected_head_dim:
                        import warnings
                        warnings.warn(
                            f"head_dim ({self._cpp_config.head_dim}) != hidden_size/num_attention_heads ({expected_head_dim}). "
                            f"Using head_dim from config."
                        )

            # Ensure vocab_size is set (explicit handling)
            if hasattr(self._python_config, "vocab_size"):
                self._cpp_config.vocab_size = self._python_config.vocab_size

            # Handle attention_bias and attention_output_bias based on model type
            if self._is_jiuge_model:
                # For jiuge models: q/k/v have bias, but o_proj does NOT have bias
                # Align with Python backend which hardcodes attention_bias=True
                # and o_proj bias=False
                self._cpp_config.attention_bias = True  # Always True for jiuge models
                self._cpp_config.attention_output_bias = False  # Always False for jiuge models
            else:
                # For regular Llama models: use config values or defaults
                # Handle attention_bias: if not in Python config, use C++ default (true)
                # This supports models that don't have attention_bias in config
                if not hasattr(self._python_config, "attention_bias"):
                    # Keep C++ default (true) - no need to set explicitly
                    pass
                # If attention_bias is in Python config, it was already copied above

                # Default attention_output_bias to attention_bias if not explicitly set
                # (for backward compatibility with models that don't have this field)
                if not hasattr(self._python_config, "attention_output_bias"):
                    self._cpp_config.attention_output_bias = self._cpp_config.attention_bias

            # Validate config after setting all values (especially important for jiuge models)
            if not self._cpp_config.validate():
                raise ValueError("C++ LlamaConfig validation failed. Check config values.")

            # Log key config values for debugging (especially useful for jiuge models)
            import logging
            logger = logging.getLogger(__name__)
            model_type_str = "jiuge" if self._is_jiuge_model else "llama"
            logger.info(
                f"LlamaConfig ({model_type_str}) C++ LlamaConfig created: vocab_size={self._cpp_config.vocab_size}, "
                f"hidden_size={self._cpp_config.hidden_size}, "
                f"num_attention_heads={self._cpp_config.num_attention_heads}, "
                f"num_key_value_heads={self._cpp_config.num_key_value_heads}, "
                f"head_dim={self._cpp_config.head_dim}, "
                f"kv_dim={self._cpp_config.kv_dim()}, "
                f"attention_bias={self._cpp_config.attention_bias}, "
                f"attention_output_bias={self._cpp_config.attention_output_bias}"
            )

        return self._cpp_config


class LlamaForCausalLM(GenerationMixin):
    """Llama model for causal language modeling"""

    def __init__(
        self,
        config,
        device=None,
        dtype=None,
        distributed_config=DistConfig(1),
    ):
        """
        Create LlamaForCausalLM

        Args:
            config: LlamaConfig instance or dict
            device: Device instance (defaults to CPU)
            dtype: Optional dtype for model parameters (defaults to None)
        """
        super().__init__()

        # Convert config to LlamaConfig (handles both regular Llama and Jiuge models)
        if isinstance(config, dict):
            config = LlamaConfig(**config)
        elif not isinstance(config, LlamaConfig):
            # Not a dict or LlamaConfig, try to convert
            config = LlamaConfig(config)
        # If already LlamaConfig, use as-is (it will auto-detect jiuge models)

        if device is None:
            device = infinicore.device()

        self.use_cache = False

        # Store the Python wrapper config so it can be accessed later
        # This is needed for DynamicCache which calls config.get_text_config()
        self._config = config

        self._device = device
        # self._model = _infinilm.LlamaForCausalLM(
        #     config._underlying, device._underlying, dtype
        # )
        self._model = _infinilm.InferEngine(
            config._underlying, distributed_config._underlying, device._underlying.type
        )

    def state_dict(self):
        """Get model state dictionary with parameter shapes"""
        return self._model.state_dict()

    def load_state_dict(self, state_dict):
        """
        Load state dictionary into the model

        Args:
            state_dict: Dictionary mapping parameter names to InfiniCore tensors, numpy arrays, or torch tensors
        """
        # self._model.load_state_dict(state_dict, self._device._underlying)
        for name, param in state_dict.items():
            self._model.load_param(name, param._underlying)

    def get_parameter(self, name):
        """
        Get a parameter tensor by name

        Args:
            name: Parameter name

        Returns:
            InfiniCore tensor
        """
        return self._model.get_parameter(name)

    @property
    def config(self):
        """Get model configuration"""
        # Return the Python wrapper config instead of C++ config
        # This ensures compatibility with code that expects PretrainedConfig methods
        # like get_text_config() used by DynamicCache
        return self._config

    def forward(self, input_ids, position_ids, *args, **kwargs):
        kv_caches = None
        # return infinicore.Tensor(
        #     self._model.forward(input_ids, position_ids, kv_caches)
        # )
        return infinicore.Tensor(
            self._model.generate(
                input_ids._underlying,
                position_ids._underlying,
            )
        )

    def __call__(self, input_ids, position_ids, *args, **kwargs):
        return self.forward(input_ids=input_ids, position_ids=position_ids)

    @classmethod
    def from_pretrained(
        cls,
        model_path: Union[str, os.PathLike],
        device: Optional[infinicore.device] = None,
        dtype: Optional[infinicore.dtype] = None,
        **kwargs,
    ):
        """
        Load a pretrained LlamaForCausalLM model from a directory.

        Args:
            model_path: Path to the model directory containing config.json
            device: Device instance (defaults to CPU)
            dtype: Optional dtype for model parameters (defaults to None)

        Returns:
            LlamaForCausalLM instance
        """
        config_path = os.path.join(model_path, "config.json")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_path, "r") as f:
            config_dict = json.load(f)

        # LlamaConfig automatically detects and handles jiuge models
        config = LlamaConfig(config_dict)
        return cls(config, device=device, dtype=dtype, **kwargs)
