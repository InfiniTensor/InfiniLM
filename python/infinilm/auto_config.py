import json
import os

from infinilm.models.llama.configuration_llama import LlamaConfig
from infinilm.models.llava.configuration_llava import LlavaConfig
from infinilm.models.minicpmv.configuration_minicpmv import MiniCPMVConfig


class AutoConfig:
    def from_pretrained(model_path):
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
        elif config_dict["model_type"] == "llava":
            return LlavaConfig(**config_dict)
        elif config_dict["model_type"] == "minicpmv":
            return MiniCPMVConfig(**config_dict)

        raise ValueError(f"Unsupported model type `{config_dict['model_type']}`.")
