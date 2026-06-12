import importlib
import pkgutil
from pathlib import Path
import json
from .processor import get_processor_class, InfinilmProcessor

# ---------------------------------------------------------------------------
# Auto-discovery mechanism
# ---------------------------------------------------------------------------
# Automatically discover and import all modules matching *_processor.py in
# this directory. This ensures that any @register_processor decorators are
# executed at load time, populating the processor registry dynamically
# without requiring manual imports for each new model.
# ---------------------------------------------------------------------------
_current_dir = Path(__file__).resolve().parent

for _module_info in pkgutil.iter_modules([str(_current_dir)]):
    if _module_info.name.endswith("_processor"):
        importlib.import_module(f".{_module_info.name}", __package__)


class AutoInfinilmProcessor:
    """Factory class to instantiate the appropriate model processor."""

    @classmethod
    def from_pretrained(cls, model_dir_path: str, **kwargs) -> InfinilmProcessor:
        """Instantiate a processor based on the model's configuration.

        Reads the model_type from the config and looks up the corresponding
        registered Processor. Falls back to the registered default processor
        for unregistered or standard architectures.
        """
        config_path = Path(model_dir_path) / "config.json"
        with open(config_path, "r") as f:
            config = json.load(f)
        model_type = config.get("model_type", "default").lower()

        processor_cls = get_processor_class(model_type)
        return processor_cls(model_dir_path)
