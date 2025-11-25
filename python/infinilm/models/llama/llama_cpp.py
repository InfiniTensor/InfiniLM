"""
Llama model C++ bindings implementation
"""

from infinilm.lib import _infinilm_llama

# Import Device from InfiniCore (it's already bound there)
try:
    from infinicore.lib import _infinicore
    _Device = _infinicore.Device
except ImportError:
    # Fallback: try to get Device from _infinilm_llama if available
    try:
        _Device = _infinilm_llama.Device
    except AttributeError:
        raise ImportError(
            "Device not found. Please ensure InfiniCore is properly installed.")


class LlamaConfig:
    """Llama model configuration"""

    def __init__(self, config_dict=None, **kwargs):
        """Create LlamaConfig from dictionary or keyword arguments"""
        # If config_dict is provided, merge it with kwargs (kwargs take precedence)
        if config_dict is not None and isinstance(config_dict, dict):
            merged = {**config_dict, **kwargs}
        else:
            merged = kwargs

        self._config = _infinilm_llama.LlamaConfig()

        # Set attributes from merged dict
        for key, value in merged.items():
            if hasattr(self._config, key):
                setattr(self._config, key, value)

        # Handle defaults if not explicitly set
        if 'num_key_value_heads' not in merged:
            self._config.num_key_value_heads = self._config.num_attention_heads

        if 'head_dim' not in merged:
            self._config.head_dim = self._config.hidden_size // self._config.num_attention_heads

    def __getattr__(self, name):
        """Delegate attribute access to underlying config"""
        return getattr(self._config, name)

    def __setattr__(self, name, value):
        """Delegate attribute setting to underlying config"""
        if name.startswith('_'):
            super().__setattr__(name, value)
        else:
            if hasattr(self, '_config'):
                setattr(self._config, name, value)
            else:
                super().__setattr__(name, value)

    @property
    def _underlying(self):
        """Get underlying C++ config object"""
        return self._config


class LlamaModel:
    """Llama base model (without language modeling head)"""

    def __init__(self, config, device=None):
        """
        Create LlamaModel

        Args:
            config: LlamaConfig instance or dict
            device: Device instance (defaults to CPU)
        """
        if isinstance(config, dict):
            config = LlamaConfig(**config)
        elif not isinstance(config, LlamaConfig):
            config = LlamaConfig(**config)

        if device is None:
            device = Device()

        self._device = device
        self._model = _infinilm_llama.LlamaModel(
            config._underlying, device._underlying)

    def state_dict(self):
        """Get model state dictionary with parameter shapes"""
        return self._model.state_dict()

    def load_state_dict(self, state_dict):
        """
        Load state dictionary into the model

        Args:
            state_dict: Dictionary mapping parameter names to InfiniCore tensors, numpy arrays, or torch tensors
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
        return self._model.get_parameter(name)

    @property
    def config(self):
        """Get model configuration"""
        return self._model.config()

    @property
    def num_layers(self):
        """Get number of layers"""
        return self._model.num_layers()


class LlamaForCausalLM:
    """Llama model for causal language modeling"""

    def __init__(self, config, device=None):
        """
        Create LlamaForCausalLM

        Args:
            config: LlamaConfig instance or dict
            device: Device instance (defaults to CPU)
        """
        if isinstance(config, dict):
            config = LlamaConfig(**config)
        elif not isinstance(config, LlamaConfig):
            config = LlamaConfig(**config)

        if device is None:
            device = Device()

        print(f"[LOG] Python: Device created: {device}", flush=True)
        self._device = device
        print(
            f"[LOG] Python: About to create LlamaForCausalLM C++ object...", flush=True)
        self._model = _infinilm_llama.LlamaForCausalLM(
            config._underlying, device._underlying)
        print(
            f"[LOG] Python: LlamaForCausalLM C++ object created successfully", flush=True)

    def state_dict(self):
        """Get model state dictionary with parameter shapes"""
        return self._model.state_dict()

    def load_state_dict(self, state_dict):
        """
        Load state dictionary into the model

        Args:
            state_dict: Dictionary mapping parameter names to InfiniCore tensors, numpy arrays, or torch tensors
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
        return self._model.get_parameter(name)

    @property
    def config(self):
        """Get model configuration"""
        return self._model.config()


class Device:
    """Device for tensor operations"""

    def __init__(self, device_type=None, device_index=0):
        """
        Create Device

        Args:
            device_type: Device type (defaults to CPU)
            device_index: Device index (defaults to 0)
        """
        if device_type is None:
            self._device = _Device()
        else:
            if isinstance(device_type, str):
                # Convert string to enum
                device_type = getattr(_Device.Type, device_type.upper())
            self._device = _Device(device_type, device_index)

    @property
    def _underlying(self):
        """Get underlying C++ device object"""
        return self._device
