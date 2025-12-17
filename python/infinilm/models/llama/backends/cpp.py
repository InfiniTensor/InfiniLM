from ....generation.utils import GenerationMixin
import infinicore
from infinilm.models.llama.configuration_llama import LlamaConfig as _LlamaConfig
from infinilm.lib import _infinilm
from infinilm.distributed import DistConfig
import json
import os
from typing import Optional, Union
from collections import OrderedDict


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
            else:
                super().__setattr__(name, value)


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
            config, distributed_config._underlying, device._underlying.type
        )

    def reset_cache(self, batch_size: int, pos: int = 0, initial_capacity: int = 1024):
        """Reset the cache for the model"""
        infinicore.sync_device()

        cache_config = self._model.get_cache_config()
        cache_config.initial_batch_size = batch_size
        cache_config.initial_capacity = initial_capacity

        self._model.reset_cache(cache_config, pos)

    def state_dict_keyname(self):
        """Get model key name."""
        return self._model.state_dict()[0].keys()

    def load_state_dict(self, state_dict, strict=None):
        """
        Load state dictionary into the model

        Args:
            state_dict: Dictionary mapping parameter names to InfiniCore tensors, numpy arrays, or torch tensors
        """
        # self._model.load_state_dict(state_dict, self._device._underlying)
        for name, param in state_dict.items():
            self._model.load_param(name, param._underlying)

    def load_param(self, name: str, weight: infinicore.Tensor):
        self._model.load_param(name, weight._underlying)

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
