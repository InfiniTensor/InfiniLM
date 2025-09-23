import importlib
from collections import OrderedDict
# from icinfer.models.libinfinicore_infer.base import DeviceType
import os
import json

# 1. Define the mapping for model classes
# Using OrderedDict to maintain order for documentation purposes
MODEL_MAPPING_FILES = OrderedDict([
    ("fm9g7b", "jiuge"),
    ("jiuge_awq", "jiuge_awq"),
    ("deepseek_v3", "deepseek_v3"),
])
MODEL_MAPPING_NAMES = OrderedDict([
    ("jiuge", "JiugeForCausalLM"),
    ("jiuge_awq", "JiugeAWQForCausalLM"),
    ("deepseek_v3", "DeepSeekV3ForCauslLM"),
])

# Dynamically import the module to avoid loading all model code at startup (lazy loading)
def get_model_class(config):
    """Gets the corresponding model class based on model_type in the config."""
    # model_type = config.get("model_type")
    model_type = config.hf_config.model_type
    model_file = MODEL_MAPPING_FILES[model_type]
    print(f"model_file: {model_file}")
    if model_file in MODEL_MAPPING_NAMES:
        model_name = MODEL_MAPPING_NAMES[model_file]
        
        # Dynamically construct the module path and import it.
        # For example, if model_type="llama", it will import icinfer.models.llama
        module_path = f"icinfer.models.{model_file}"
        module = importlib.import_module(module_path)
        
        # Get the model class from the imported module
        model_class = getattr(module, model_name, None)
        if model_class is None:
            raise AttributeError(f"Module '{module_path}' does not have a class named '{model_name}'")
        return model_class
    
    raise KeyError(
        f"Model type '{model_type}' not found in MODEL_MAPPING_NAMES. "
        f"Available model types: {list(MODEL_MAPPING_NAMES.keys())}"
    )


class AutoModelForCausalLM:
    """
    This is a generic model factory class. It will automatically instantiate the
    correct model architecture based on the provided config.
    """
    def __init__(self):
        # Direct instantiation of AutoModel is not allowed
        raise EnvironmentError(
            "AutoModel is designed to be instantiated using the `AutoModel.from_config(config)` class method."
        )

    @classmethod
    def from_config(cls, config, device):
        """
        Instantiates one of the base model classes from a configuration.
        
        Args:
            config (dict): A configuration dictionary containing a 'model_type' field.
            
        Returns:
            An instance of a model class (e.g., LlamaModel, BertModel).
        """
        model_class = get_model_class(config)
        max_tokens = config.max_model_len
        model_dir_path = config.model_path
        ndev = config.tensor_parallel_size
        # hf_config = config.hf_config
        hf_config = None
        with open(os.path.join(model_dir_path, "config.json"), "r") as f:
            hf_config = json.load(f)

        model = model_class(model_dir_path=model_dir_path, 
                                  config=hf_config,
                                  device=device, 
                                  ndev=ndev, 
                                  max_tokens=max_tokens)
        # Call the from_config method of the specific model class
        return model