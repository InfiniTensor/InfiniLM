from ....generation.utils import GenerationMixin
import infinicore
from infinilm.models.llama.configuration_llama import LlamaConfig as _LlamaConfig
import json
import os
from typing import Optional, Union


class LlamaForCausalLMWrapper(GenerationMixin):
    """
    Wrapper for LlamaForCausalLM that provides the same Python interface.

    This class can be used as a drop-in replacement for LlamaForCausalLM
    in Python code, while internally using the C++ wrapper implementation.
    """

    def __init__(self, config, device=None, dtype=None):
        """
        Create LlamaForCausalLMWrapper

        Args:
            config: LlamaConfig instance or dict
            device: Device instance (defaults to CPU)
            dtype: Optional dtype for model parameters (defaults to None)
        """
        super().__init__()

        # Import the wrapper module
        # Note: This assumes the wrapper is compiled into the same module
        # or a similarly named module
        try:
            # Try to import from _infinilm_llama module
            from infinilm.lib import _infinilm_llama as wrapper_module

            self._wrapper_class = wrapper_module.LlamaForCausalLMWrapper
        except ImportError:
            # Fall back to regular implementation if wrapper not available
            from infinilm.lib import _infinilm_llama

            self._wrapper_class = _infinilm_llama.LlamaForCausalLM

        if isinstance(config, dict):
            config = self._create_config(config)
        elif not isinstance(config, LlamaConfigWrapper):
            config = self._create_config(config.__dict__)

        if device is None:
            device = infinicore.device()

        self.use_cache = False
        self._device = device

        # Create the underlying wrapper/model
        self._model = self._wrapper_class(config._underlying, device._underlying, dtype)

    def _create_config(self, config_dict):
        """Helper to create LlamaConfigWrapper from dictionary"""
        return LlamaConfigWrapper(config_dict)

    def state_dict(self):
        """Get model state dictionary with parameter shapes"""
        return self._model.state_dict()

    def load_state_dict(self, state_dict):
        """
        Load state dictionary into the model

        Args:
            state_dict: Dictionary mapping parameter names to InfiniCore tensors,
                       numpy arrays, or torch tensors
        """
        self._model.load_state_dict(state_dict, self._device._underlying)

    def get_parameter(self, name):
        """
        Get a parameter tensor by name

        Args:
            name: Parameter name

        Returns:
            InfiniCore tensor
        """
        # The C++ wrapper returns a tensor directly
        return self._model.get_parameter(name)

    @property
    def config(self):
        """Get model configuration"""
        return self._model.config()

    def forward(self, input_ids, position_ids, *args, **kwargs):
        """
        Forward pass through the model

        Args:
            input_ids: Input token IDs
            position_ids: Position IDs
            *args: Additional positional arguments
            **kwargs: Additional keyword arguments

        Returns:
            Logits tensor
        """
        kv_caches = None
        return infinicore.Tensor(
            self._model.forward(input_ids, position_ids, kv_caches)
        )

    def __call__(self, input_ids, position_ids, *args, **kwargs):
        """Make the model callable"""
        return self.forward(input_ids=input_ids, position_ids=position_ids)

    @classmethod
    def from_pretrained(
        cls,
        model_path: Union[str, os.PathLike],
        device: Optional[infinicore.device] = None,
        dtype: Optional[infinicore.dtype] = None,
    ):
        """
        Load a pretrained model from a directory.

        Args:
            model_path: Path to the model directory containing config.json
            device: Device instance (defaults to CPU)
            dtype: Optional dtype for model parameters (defaults to None)

        Returns:
            LlamaForCausalLMWrapper instance
        """
        config_path = os.path.join(model_path, "config.json")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_path, "r") as f:
            config_dict = json.load(f)

        config = LlamaConfigWrapper(config_dict)
        return cls(config, device=device, dtype=dtype)


class LlamaConfigWrapper:
    """
    Configuration wrapper that maintains compatibility with the original interface.

    This class adapts the Python configuration to work with the C++ wrapper.
    """

    def __init__(self, config_dict=None, **kwargs):
        """Create LlamaConfigWrapper from dictionary or keyword arguments"""
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
            # Import the C++ config class
            from infinilm.lib import _infinilm_llama

            self._cpp_config = _infinilm_llama.LlamaConfig()

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
