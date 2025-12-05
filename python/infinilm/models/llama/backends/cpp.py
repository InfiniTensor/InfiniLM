from ....generation.utils import GenerationMixin
import infinicore
from infinilm.models.llama.configuration_llama import LlamaConfig as _LlamaConfig
from infinilm.lib import _infinilm
import json
import os
from typing import Optional, Union
from collections import OrderedDict


class LlamaConfig:
    """Llama model configuration adapter for C++ bindings.

    This class wraps configuration_llama.LlamaConfig and provides
    a _underlying property that creates the C++ config object.
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

            # Handle defaults
            if (
                not hasattr(self._cpp_config, "num_key_value_heads")
                or self._cpp_config.num_key_value_heads == 0
            ):
                self._cpp_config.num_key_value_heads = (
                    self._cpp_config.num_attention_heads
                )

            if (
                not hasattr(self._cpp_config, "head_dim")
                or self._cpp_config.head_dim == 0
            ):
                self._cpp_config.head_dim = (
                    self._cpp_config.hidden_size // self._cpp_config.num_attention_heads
                )

        return self._cpp_config


class LlamaForCausalLM(GenerationMixin):
    """Llama model for causal language modeling"""

    def __init__(self, config, device=None, dtype=None):
        """
        Create LlamaForCausalLM

        Args:
            config: LlamaConfig instance or dict
            device: Device instance (defaults to CPU)
            dtype: Optional dtype for model parameters (defaults to None)
        """
        super().__init__()

        if isinstance(config, dict):
            config = LlamaConfig(**config)
        elif not isinstance(config, LlamaConfig):
            config = LlamaConfig(**config)

        if device is None:
            device = infinicore.device()

        self.use_cache = False

        self._device = device
        self._model = _infinilm.LlamaForCausalLM(
            config._underlying, device._underlying, dtype
        )

    def state_dict(self):
        """Get model state dictionary with parameter shapes"""
        destination = OrderedDict()
        for name, param in self._model.state_dict().items():
            destination[name] = infinicore.Tensor(param)
        return destination

    def load_state_dict(self, state_dict, strict=None):
        """
        Load state dictionary into the model

        Args:
            state_dict: Dictionary mapping parameter names to InfiniCore tensors, numpy arrays, or torch tensors
        """
        strict = None
        self._model.load_state_dict(state_dict, self._device._underlying)

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
        return self._model.config()

    def forward(self, input_ids, position_ids, *args, **kwargs):
        kv_caches = None
        return infinicore.Tensor(
            self._model.forward(input_ids, position_ids, kv_caches)
        )

    def __call__(self, input_ids, position_ids, *args, **kwargs):
        return self.forward(input_ids=input_ids, position_ids=position_ids)

    @classmethod
    def from_pretrained(
        cls,
        model_path: Union[str, os.PathLike],
        device: Optional[infinicore.device] = None,
        dtype: Optional[infinicore.dtype] = None,
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

        config = LlamaConfig(config_dict)
        return cls(config, device=device, dtype=dtype)
