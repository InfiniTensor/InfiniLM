import importlib
import pkgutil
from pathlib import Path
from transformers import AutoConfig
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
    def get_processor_class(cls, model_dir_path: str):
        config = AutoConfig.from_pretrained(model_dir_path, trust_remote_code=True)
        model_type = config.model_type.lower()
        return get_processor_class(model_type)

    @classmethod
    def from_pretrained(cls, model_dir_path: str, **kwargs) -> InfinilmProcessor:
        """Instantiate a processor based on the model's configuration.

        Reads the model_type from the config and looks up the corresponding
        registered Processor. Falls back to the registered default processor
        for unregistered or standard architectures.
        """
        processor_cls = cls.get_processor_class(model_dir_path)
        return processor_cls(model_dir_path)
