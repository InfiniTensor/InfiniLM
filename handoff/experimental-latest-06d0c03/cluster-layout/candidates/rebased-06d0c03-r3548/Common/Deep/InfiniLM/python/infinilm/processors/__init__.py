import json
from pathlib import Path

from .processor import InfinilmProcessor, get_processor_class


class AutoInfinilmProcessor:
    """Factory class to instantiate the appropriate model processor."""

    @classmethod
    def from_pretrained(cls, model_dir_path: str, **kwargs) -> InfinilmProcessor:
        """Instantiate a processor based on the model's configuration.

        Reads the model_type from the config and looks up the corresponding
        registered Processor. Falls back to the registered default processor
        for unregistered or standard architectures.
        """
        raw_config_path = Path(model_dir_path) / "config.json"
        raw_config = {}
        if raw_config_path.exists():
            with raw_config_path.open("r") as f:
                raw_config = json.load(f)

        raw_model_type = str(raw_config.get("model_type", "")).lower()
        if raw_model_type in {"qwen3_5", "qwen3_5_moe"}:
            model_type = raw_model_type
        else:
            from transformers import AutoConfig

            config = AutoConfig.from_pretrained(model_dir_path, trust_remote_code=True)
            model_type = config.model_type.lower()

        architectures = raw_config.get("architectures", []) or []
        if (
            model_type == "qwen2_5_vl"
            and "VideoNSAForConditionalGeneration" in architectures
        ):
            model_type = "videonsa"

        processor_cls = get_processor_class(model_type)
        return processor_cls(model_dir_path)
