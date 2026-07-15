import json
import os
import logging
from infinilm.models.llama.configuration_llama import LlamaConfig

logger = logging.getLogger(__name__)


class AutoConfig:
    def from_pretrained(model_path):
        logger.warning(f"The AutoConfig will be deprecated, please don't use it !")

        config_path = os.path.join(model_path, "config.json")

        if not os.path.exists(config_path):
            raise FileNotFoundError(f"`{config_path}` not found")

        with open(config_path) as f:
            config_dict = json.load(f)

        if "model_type" not in config_dict:
            raise ValueError(
                f"`model_type` is not specified in the config file `{config_path}`."
            )

        mt = config_dict["model_type"]
        if mt in (
            "llama",
            "qwen2",
            "qwen3",
            "minicpm",
            "fm9g",
            "fm9g7b",
            "qwen3_next",
            "minicpm_sala",
            "qwen3_vl",
            "qwen3_moe",
            "minicpm5_moe",
        ):
            return LlamaConfig(**config_dict)

        raise ValueError(f"Unsupported model type `{mt}`.")
