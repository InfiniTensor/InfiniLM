import json
import os

from infinilm.models.llama.configuration_llama import LlamaConfig


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

        raise ValueError(f"Unsupported model type `{config_dict['model_type']}`.")
